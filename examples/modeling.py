# %%
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import lightning as L
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sonarlight import Sonar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage import measure
import segmentation_models_pytorch as smp

# Paths
train_data_paths = [
    ("emma_data/raw/Sonar_2020-09-14 BROMME 1.sl3", "emma_data/masks/Sonar_2020-09-14 BROMME 1.jsonl"),
    ("emma_data/raw/Sonar_2020-09-14 BROMME 2.sl3", "emma_data/masks/Sonar_2020-09-14 BROMME 2.jsonl"),
]
valid_data_paths = [
    ("emma_data/raw/Sonar_2025-08-08 Bromme 1.sl3", "emma_data/masks/Sonar_2025-08-08 Bromme 1.jsonl"),
]

# Data / dataset
patch_size  = 512   # model input resolution (pixels)
base_height = 512   # nominal window height in raw sonar pixels
n_samples   = 256   # random train samples per epoch
batch_size  = 16

# Training
n_epochs = 200
lr       = 1e-4

# Inference / postprocessing
threshold = 0.5     # probability threshold for binary mask
min_area  = 50      # minimum connected-component area (pixels) to keep

# %%
# Helper functions
def polygons_to_mask(polygons, height, width, row_offset=0):
    """Rasterize polygons into a binary mask (0=background, 1=foreground)."""
    mask_pil = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_pil)
    for poly in polygons:
        verts = [(float(p[1]), float(p[0] - row_offset)) for p in poly]
        if len(verts) >= 3:
            draw.polygon(verts, outline=1, fill=1)
    return np.array(mask_pil, dtype=np.uint8)


def apply_egn(image):
    """Empirical Gain Normalization (EGN) for sidescan imagery.

    Input: 2D uint8 image (H, W)
    Output: 2D uint8 image (H, W), gain-flattened
    """
    img_f = image.astype(np.float32)
    col_means = np.mean(img_f, axis=0, keepdims=True)
    col_means[col_means == 0] = 1e-6

    normalized = img_f / col_means
    low, high = np.percentile(normalized, (1, 99))
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = np.clip((normalized - low) / (high - low) * 255.0, 0, 255)
    return normalized.astype(np.uint8)


class NadirAugmentation(A.ImageOnlyTransform):
    """Perturb nadir width/intensity around the center across-track column."""

    def __init__(self, width_range=(2, 14), intensity_range=(0.7, 1.7), p=0.5):
        super().__init__(p=p)
        self.width_range = width_range
        self.intensity_range = intensity_range

    def apply(self, img, **params):
        img = img.copy()
        h, w = img.shape[:2]
        center = w // 2
        width = np.random.randint(self.width_range[0], self.width_range[1] + 1)
        intensity = np.random.uniform(*self.intensity_range)

        start, end = max(0, center - width), min(w, center + width)
        nadir_zone = img[:, start:end].astype(np.float32) * intensity
        img[:, start:end] = np.clip(nadir_zone, 0, 255).astype(img.dtype)
        return img


def apply_ping_dropout(image, **kwargs):
    """Randomly zero full ping rows to simulate sonar line dropout."""
    img = image.copy()
    for _ in range(np.random.randint(1, 4)):
        y = np.random.randint(0, img.shape[0])
        thickness = np.random.randint(1, 3)
        img[y : y + thickness, :] = 0
    return img


# Sonar-aware training augmentation pipeline (no rotations)
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.04,
        scale_limit=0.05,
        rotate_limit=0,
        border_mode=0,
        p=0.5,
    ),
    NadirAugmentation(p=0.4),
    A.Lambda(image=apply_ping_dropout, p=0.2),
    A.CoarseDropout(
        num_holes_range=(1, 6),
        hole_height_range=(4, 20),
        hole_width_range=(4, 20),
        fill=0,
        p=0.3,
    ),
    A.PixelDropout(dropout_prob=0.01, p=0.2),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
])

# Validation / normalization pipeline: map uint8 [0,255] -> float32 [-1,1]
val_transform = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
])


def preprocess_window(crop, mask=None, patch_size=512, transform=None, normalize=None):
    """Shared preprocessing for train/val/inference.

    Pipeline:
      1) EGN
      2) optional augmentation (train only)
      3) resize to patch_size x patch_size
      4) normalize to [-1, 1]
    """
    if normalize is None:
        normalize = val_transform

    crop = apply_egn(crop.astype(np.uint8))

    if transform is not None:
        if mask is None:
            out = transform(image=crop)
            crop = out["image"]
        else:
            out = transform(image=crop, mask=mask)
            crop, mask = out["image"], out["mask"]

    crop_r = np.array(
        Image.fromarray(crop).resize((patch_size, patch_size), Image.BILINEAR),
        dtype=np.uint8,
    )
    crop_r = normalize(image=crop_r)["image"].astype(np.float32)

    if mask is None:
        return crop_r

    mask_r = np.array(
        Image.fromarray(mask.astype(np.uint8)).resize((patch_size, patch_size), Image.NEAREST),
        dtype=np.uint8,
    )
    mask_r = (mask_r > 0).astype(np.uint8)
    return crop_r, mask_r


def load_sonar_and_masks(sonar_path, labels_path, positive_labels=None):
    sonar = Sonar(sonar_path, clean=False)
    df = sonar.df.query("survey == 'sidescan'").reset_index(drop=True)
    img = np.stack(df["frames"])

    annotations = pd.read_json(labels_path, lines=True)
    
    if positive_labels is not None:
        annotations = annotations[annotations["label"].isin(positive_labels)]
        
    mask = polygons_to_mask(
        annotations["polygon"].tolist(),
        height=img.shape[0],
        width=img.shape[1],
    )
    return img, mask

# Function for listing the set of labels in mask files
def get_all_labels(label_paths):
    labels = set()
    for path in label_paths:
        annotations = pd.read_json(path, lines=True)
        labels.update(annotations["label"].unique().tolist())
    return labels

# %%
labels = get_all_labels([labels_path for _, labels_path in train_data_paths + valid_data_paths])
print(f"Unique labels in dataset: {labels}")

# Labels that should be 1 in the binary mask (everything else is background/0)
plant_labels = ['Kransnålsalger', 
                'Glinsende vandaks, no doubt', 
                'Kransnålalger', 
                'Chara', 
                'Glinsende vandaks', 
                'plant']

# Assert all plant labels are present in the dataset
assert all(label in labels for label in plant_labels), "Some plant labels not found in dataset"

# %%
train_data = [load_sonar_and_masks(sonar_path, labels_path, plant_labels) for sonar_path, labels_path in train_data_paths]
valid_data = [load_sonar_and_masks(sonar_path, labels_path, plant_labels) for sonar_path, labels_path in valid_data_paths]

# %%
# Dataset
class SidescanDataset(Dataset):
    """Sliding-window dataset over multiple sidescan files.

    Training:
      - each sample picks a random sonar file/mask pair
      - then picks a random vertical window within that file

    Validation:
      - deterministic sliding windows over every validation file
      - no randomization, so metrics are stable across epochs

    Each sample is a 1-channel tensor from EGN-normalized sidescan intensity.
    Target: binary mask resized with nearest-neighbour interpolation.
    """

    def __init__(self, data, patch_size=512, base_height=512, n_samples=100, train=True, transform=None, train_height_scale_range=(0.5, 1.5)):
        self.data = data        # list[(img, mask)], each (H, W) uint8
        self.patch_size = patch_size
        self.base_height = base_height
        self.train = train
        self.n_samples = n_samples
        self.transform = transform
        self.normalize = val_transform
        self.train_height_scale_range = train_height_scale_range

        if not train:
            # Deterministic sliding windows with configurable stride.
            # Use stride = base_height // 2 to match inference-time windowing
            # (only the center patch_size // 2 pixels are kept per window at inference).
            stride = base_height // 2
            self.val_windows = []
            for file_idx, (img, _mask) in enumerate(self.data):
                H, _W = img.shape
                starts = list(range(0, max(1, H - base_height + 1), stride))
                if not starts or starts[-1] + base_height < H:
                    starts.append(max(0, H - base_height))
                for r0 in starts:
                    self.val_windows.append((file_idx, r0))

    def __len__(self):
        return self.n_samples if self.train else len(self.val_windows)

    def __getitem__(self, idx):
        if self.train:
            file_idx = np.random.randint(0, len(self.data))
            img, mask = self.data[file_idx]
            H, _W = img.shape
            h = int(self.base_height * np.random.uniform(*self.train_height_scale_range))
            h = np.clip(h, 1, H)
            r0 = np.random.randint(0, max(1, H - h + 1))
        else:
            file_idx, r0 = self.val_windows[idx]
            img, mask = self.data[file_idx]
            H, _W = img.shape
            h = min(self.base_height, H - r0)

        crop = img[r0 : r0 + h, :].astype(np.uint8)   # (h, W) uint8
        mask = mask[r0 : r0 + h, :].astype(np.uint8)  # (h, W) uint8

        crop_r, mask_r = preprocess_window(
            crop,
            mask=mask,
            patch_size=self.patch_size,
            transform=self.transform,
            normalize=self.normalize,
        )

        x = torch.from_numpy(crop_r).unsqueeze(0).float()            # (1, P, P)
        y = torch.from_numpy(mask_r).long()                          # (P, P)

        return x, y


train_ds = SidescanDataset(
    train_data,
    patch_size=patch_size,
    base_height=base_height,
    n_samples=n_samples,
    train=True,
    transform=train_transform,
)
val_ds = SidescanDataset(
    valid_data,
    patch_size=patch_size,
    base_height=base_height,
    train=False,
    transform=None,
)

print(f"Train dataset: {len(train_ds)} samples")
print(f"Val   dataset: {len(val_ds)} windows")

# %%
# Visualize a random training sample (sonar imagery with mask overlay)
x, y = train_ds[0]

print(f"Sample x: {x.shape} {x.dtype}, range [{x.min():.2f}, {x.max():.2f}]")
print(f"Sample y: {y.shape} {y.dtype}, unique values: {y.unique().tolist()}")

# Convert tensors to numpy for plotting
img_np = x.squeeze(0).cpu().numpy()            # normalized to [-1, 1]
img_np = ((img_np + 1.0) * 0.5).clip(0, 1)     # back to [0, 1] for display
mask_np = y.cpu().numpy().astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_np, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Train sample (image)")
axes[0].axis("off")

axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Train sample (mask)")
axes[1].axis("off")

axes[2].imshow(img_np, cmap="gray", vmin=0, vmax=1)
axes[2].imshow(mask_np, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
axes[2].set_title("Train sample (overlay)")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# %%
# DataLoaders
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

# %%
# Model, loss, optimizer
class DoubleConv(nn.Module):
    """Two 3x3 convolutions with padding to preserve HxW."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpSampleConv(nn.Module):
    """Nearest-neighbor 2x upsampling followed by a 3x3 conv block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    """Classic U-Net-style encoder/decoder with skip connections.

    Uses padded convolutions so output spatial size matches input spatial size.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c4, c5)

        self.up4 = UpSampleConv(c5, c4)
        self.dec4 = DoubleConv(c4 + c4, c4)

        self.up3 = UpSampleConv(c4, c3)
        self.dec3 = DoubleConv(c3 + c3, c3)

        self.up2 = UpSampleConv(c3, c2)
        self.dec2 = DoubleConv(c2 + c2, c2)

        self.up1 = UpSampleConv(c2, c1)
        self.dec1 = DoubleConv(c1 + c1, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleUNet(
    in_channels=1,
    out_channels=1,
    base_channels=16,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")

_dice_loss = smp.losses.DiceLoss(mode="binary")
_bce_loss  = smp.losses.SoftBCEWithLogitsLoss()

def loss_fn(logits, targets):
    return _dice_loss(logits, targets) + _bce_loss(logits, targets)

# %%
# Lightning module
class SidescanSegmentation(L.LightningModule):
    def __init__(self, model, loss_fn, lr):
        super().__init__()
        self.model   = model
        self.loss_fn = loss_fn
        self.lr      = lr
        self._reset_stats("train")
        self._reset_stats("val")

    def _reset_stats(self, prefix):
        setattr(self, f"_{prefix}_tp",   [])
        setattr(self, f"_{prefix}_fp",   [])
        setattr(self, f"_{prefix}_fn",   [])
        setattr(self, f"_{prefix}_tn",   [])
        setattr(self, f"_{prefix}_loss", [])

    def _step(self, batch):
        x, y   = batch
        logits = self.model(x)                              # (B, 1, H, W)
        loss   = self.loss_fn(logits, y.unsqueeze(1).float())
        probs  = torch.sigmoid(logits)
        tp, fp, fn, tn = smp.metrics.get_stats(
            probs, y.unsqueeze(1), mode="binary", threshold=0.5
        )
        return loss, tp, fp, fn, tn

    def _accumulate(self, prefix, loss, tp, fp, fn, tn):
        getattr(self, f"_{prefix}_loss").append(loss.detach())
        getattr(self, f"_{prefix}_tp").append(tp)
        getattr(self, f"_{prefix}_fp").append(fp)
        getattr(self, f"_{prefix}_fn").append(fn)
        getattr(self, f"_{prefix}_tn").append(tn)

    def _log_epoch(self, prefix):
        iou = smp.metrics.iou_score(
            torch.cat(getattr(self, f"_{prefix}_tp")),
            torch.cat(getattr(self, f"_{prefix}_fp")),
            torch.cat(getattr(self, f"_{prefix}_fn")),
            torch.cat(getattr(self, f"_{prefix}_tn")),
            reduction="micro",
        )
        avg_loss = torch.stack(getattr(self, f"_{prefix}_loss")).mean()
        self.log(f"{prefix}/loss", avg_loss, prog_bar=True)
        self.log(f"{prefix}/iou",  iou,      prog_bar=True)
        self._reset_stats(prefix)

    def training_step(self, batch, _batch_idx):
        loss, tp, fp, fn, tn = self._step(batch)
        self._accumulate("train", loss, tp, fp, fn, tn)
        return loss

    def on_train_epoch_end(self):
        self._log_epoch("train")

    def validation_step(self, batch, batch_idx):
        loss, tp, fp, fn, tn = self._step(batch)
        self._accumulate("val", loss, tp, fp, fn, tn)

    def on_validation_epoch_end(self):
        self._log_epoch("val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# %%
# Train
checkpoint_cb = ModelCheckpoint(
    monitor="val/iou",
    mode="max",
    save_top_k=1,
    verbose=True,
)
logger = TensorBoardLogger(save_dir="logs")

lit_model = SidescanSegmentation(model, loss_fn, lr=lr)

trainer = L.Trainer(
    max_epochs=n_epochs,
    logger=logger,
    callbacks=[checkpoint_cb],
    log_every_n_steps=1,
    enable_progress_bar=True,
    precision="bf16",
)

trainer.fit(lit_model, train_dl, val_dl)

# %%
# Load best model
best_path = checkpoint_cb.best_model_path
print(f"Best model path: {best_path}")
lit_model = SidescanSegmentation(model, loss_fn, lr=lr)
lit_model.load_state_dict(torch.load(best_path)["state_dict"])
lit_model.eval()

# %%
# Windowed inference over the full sidescan image

def windowed_inference(model, img, base_height, patch_size, threshold=0.5, device=None):
    """Slide a window over img, average overlapping probability maps, threshold.

    Stride = base_height // 2 (50% overlap).  Returns binary uint8 mask same
    shape as img.
    """
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    H, W = img.shape
    stride = base_height // 2

    starts = list(range(0, H - base_height + 1, stride))
    if not starts or starts[-1] + base_height < H:
        starts.append(max(0, H - base_height))

    normalize = val_transform

    prob_sum = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H,),   dtype=np.float32)

    with torch.no_grad():
        for r0 in tqdm(starts, desc="Inference", unit="window"):
            h    = min(base_height, H - r0)
            crop = img[r0 : r0 + h, :].astype(np.uint8)

            crop_r = preprocess_window(
                crop,
                mask=None,
                patch_size=patch_size,
                transform=None,
                normalize=normalize,
            )
            x   = torch.from_numpy(crop_r).unsqueeze(0).unsqueeze(0).float().to(device)

            prob = model(x).sigmoid().squeeze().cpu().numpy()   # (patch_size, patch_size)

            # Resize probability map back to raw pixel dimensions
            prob_raw = np.array(
                Image.fromarray(prob).resize((W, h), Image.BILINEAR),
                dtype=np.float32,
            )
            prob_sum[r0 : r0 + h, :] += prob_raw
            count[r0 : r0 + h]       += 1

    avg_prob  = prob_sum / np.maximum(count[:, None], 1)
    pred_mask = (avg_prob >= threshold).astype(np.uint8)
    return pred_mask, avg_prob


# Pick one full sidescan image for demo inference.
inference_img = valid_data[0][0] if len(valid_data) > 0 else train_data[0][0]
pred_mask, avg_prob = windowed_inference(
    lit_model.model,
    inference_img,
    base_height,
    patch_size,
    threshold=threshold,
    device=device,
)
print(f"Prediction mask shape: {pred_mask.shape}, foreground: {pred_mask.mean():.3%}")

# %%
# Convert predicted binary mask to annotation polygons (same format as annotations JSONL)

def mask_to_polygons(mask, label="plant", min_area=50):
    """Extract connected-component contours from a binary mask.

    Returns a list of dicts matching the annotations JSONL schema:
      {"id": int, "label": str, "polygon": [[row, col], ...], "timestamp": str}
    """
    labeled = measure.label(mask, connectivity=2)
    timestamp = pd.Timestamp.now("UTC").isoformat(timespec="milliseconds").replace("+00:00", "Z")
    records = []
    ann_id = 1
    for region in measure.regionprops(labeled):
        if region.area < min_area:
            continue
        # Crop to bounding box with 1px padding so find_contours always sees a
        # ring of background — without it, components touching the bbox edge produce clipped contours
        r0, c0, r1, c1 = region.bbox
        H_img, W_img = labeled.shape
        rp0 = max(0, r0 - 1); cp0 = max(0, c0 - 1)
        rp1 = min(H_img, r1 + 1); cp1 = min(W_img, c1 + 1)
        crop     = (labeled[rp0:rp1, cp0:cp1] == region.label)
        contours = measure.find_contours(crop, level=0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        # Offset back to full-image space using the padded origin
        polygon = [[int(round(r)) + rp0, int(round(c)) + cp0] for r, c in contour]
        records.append({"id": ann_id, "label": label, "polygon": polygon, "timestamp": timestamp})
        ann_id += 1
    return records


pred_polygons = mask_to_polygons(pred_mask, min_area=min_area)
print(f"Predicted polygons: {len(pred_polygons)}")

output_path = "predictions.jsonl"
with open(output_path, "w") as f:
    for rec in pred_polygons:
        f.write(json.dumps(rec) + "\n")
print(f"Saved to {output_path}")

# %%
