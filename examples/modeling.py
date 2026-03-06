# %%
import json
import torch
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
labels_path = "annotations_many.jsonl"
sonar_path  = "/media/kenneth/d6c13395-8492-49ee-9c0f-6a165e34c95c/sonar_project/data/Bromme 01.sl3"

# Data / dataset
patch_size  = 512   # model input resolution (pixels)
base_height = 512   # nominal window height in raw sonar pixels
n_samples   = 200   # random train samples per epoch
batch_size  = 4

# Training
n_epochs = 10
lr       = 3e-4

# Inference / postprocessing
threshold = 0.5     # probability threshold for binary mask
min_area  = 50      # minimum connected-component area (pixels) to keep

# %%
# Helper functions
def annotation_row_bounds(annotations):
    """Return (min_row, max_row) bounding all annotation polygons."""
    all_vertices = [pt for poly in annotations["polygon"] for pt in poly]
    min_row = int(np.floor(min(pt[0] for pt in all_vertices)))
    max_row = int(np.ceil(max(pt[0] for pt in all_vertices)))
    return min_row, max_row


def polygons_to_mask(polygons, height, width, row_offset=0):
    """Rasterize polygons into a binary mask (0=background, 1=foreground)."""
    mask_pil = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_pil)
    for poly in polygons:
        verts = [(float(p[1]), float(p[0] - row_offset)) for p in poly]
        if len(verts) >= 3:
            draw.polygon(verts, outline=1, fill=1)
    return np.array(mask_pil, dtype=np.uint8)


# %%
sonar = Sonar(sonar_path, clean=False)

sidescan_df = sonar.df.query("survey == 'sidescan'").reset_index(drop=True)
sidescan_img = np.stack(sidescan_df["frames"])
n_pings, frame_width = sidescan_img.shape

# %%
annotations = pd.read_json(labels_path, lines=True)

min_row, max_row = annotation_row_bounds(annotations)
crop_start = max(0, min_row)
crop_end = min(n_pings, max_row + 1)

cropped_img = sidescan_img[crop_start:crop_end, :]
cropped_df = sidescan_df.iloc[crop_start:crop_end].reset_index(drop=True)

# %%
# Train/val split along the row axis at a gap between annotation polygons (~80/20)

def merge_intervals(intervals):
    """Merge overlapping or adjacent [start, end] intervals."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


annotations["min_row"] = annotations["polygon"].apply(lambda p: min(pt[0] for pt in p))
annotations["max_row"] = annotations["polygon"].apply(lambda p: max(pt[0] for pt in p))

merged = merge_intervals(zip(annotations["min_row"], annotations["max_row"]))

global_min = merged[0][0]
global_max = merged[-1][1]
target_split_row = global_min + int(0.8 * (global_max - global_min))

# Find gaps between merged annotation clusters
gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]

# Pick the gap midpoint closest to the 80% target
best_gap = min(gaps, key=lambda g: abs((g[0] + g[1]) // 2 - target_split_row))
split_row = (best_gap[0] + best_gap[1]) // 2

train_annotations = annotations[annotations["max_row"] <= split_row].reset_index(drop=True)
val_annotations = annotations[annotations["min_row"] > split_row].reset_index(drop=True)

print(f"Annotated row range: {global_min} – {global_max}")
print(f"Split row: {split_row} (gap: {best_gap[0]}–{best_gap[1]})")
print(f"Train annotations: {len(train_annotations)}, Val annotations: {len(val_annotations)}")
print(f"Val fraction: {len(val_annotations) / len(annotations):.1%}")

# %%
# Build sidescan image crops and binary masks for train and val

def make_split(split_annotations, sidescan_img, n_pings):
    r0 = max(0, split_annotations["min_row"].min())
    r1 = min(n_pings, split_annotations["max_row"].max() + 1)
    img = sidescan_img[r0:r1, :]
    mask = polygons_to_mask(split_annotations["polygon"], *img.shape, row_offset=r0)
    return img, mask, r0, r1


train_img, train_mask, train_r0, train_r1 = make_split(train_annotations, sidescan_img, n_pings)
val_img, val_mask, val_r0, val_r1 = make_split(val_annotations, sidescan_img, n_pings)

print(f"Train image: {train_img.shape}, mask foreground: {train_mask.mean():.3%}")
print(f"Val   image: {val_img.shape},  mask foreground: {val_mask.mean():.3%}")

# %%
# Dataset

class SidescanDataset(Dataset):
    """Sliding-window dataset over sidescan imagery.

    Training: randomly samples windows with height ~ base_height * Uniform(0.5, 1.5).
    Validation: deterministic non-overlapping windows with stride = base_height.

    Each sample is a 2-channel tensor:
      ch0: img / 255                        (absolute intensity)
      ch1: (img - col_means) / 255          (column-normalized, removes range-gain gradient)
    Target: binary mask resized with nearest-neighbour interpolation.
    """

    def __init__(self, img, mask, patch_size=512, base_height=512, n_samples=200, train=True, augment=False):
        self.img = img          # (H, W) uint8
        self.mask = mask        # (H, W) uint8
        self.H, self.W = img.shape
        self.patch_size = patch_size
        self.base_height = base_height
        self.train = train
        self.n_samples = n_samples
        self.augment = augment

        if not train:
            # Deterministic sliding windows with configurable stride.
            # Use stride = base_height // 2 to match inference-time windowing
            # (only the center patch_size // 2 pixels are kept per window at inference).
            stride = base_height // 2
            starts = list(range(0, self.H - base_height + 1, stride))
            if not starts or starts[-1] + base_height < self.H:
                starts.append(max(0, self.H - base_height))
            self.val_starts = starts

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(std_range=(0.02, 0.08), p=0.5),   # std as fraction of 255
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),      # simulate varying sonar focus
            ])

    def __len__(self):
        return self.n_samples if self.train else len(self.val_starts)

    def __getitem__(self, idx):
        if self.train:
            h = int(self.base_height * np.random.uniform(0.5, 1.5))
            h = np.clip(h, 1, self.H)
            r0 = np.random.randint(0, max(1, self.H - h + 1))
        else:
            r0 = self.val_starts[idx]
            h = min(self.base_height, self.H - r0)

        crop = self.img[r0 : r0 + h, :]        # (h, W) uint8
        mask = self.mask[r0 : r0 + h, :]       # (h, W) uint8

        if self.augment and self.train:
            out = self.transform(image=crop, mask=mask)
            crop, mask = out["image"], out["mask"]

        # Resize to square patch_size
        crop_r = np.array(
            Image.fromarray(crop).resize((self.patch_size, self.patch_size), Image.BILINEAR),
            dtype=np.float32,
        )
        mask_r = np.array(
            Image.fromarray(mask).resize((self.patch_size, self.patch_size), Image.NEAREST),
            dtype=np.uint8,
        )

        ch0 = crop_r / 255.0
        ch1 = (crop_r - crop_r.mean(axis=0)) / 255.0   # subtract per-column mean

        x = torch.from_numpy(np.stack([ch0, ch1], axis=0)).float()  # (2, P, P)
        y = torch.from_numpy(mask_r).long()                          # (P, P)

        return x, y


train_ds = SidescanDataset(train_img, train_mask, patch_size=patch_size, base_height=base_height, n_samples=n_samples, train=True, augment=True)
val_ds   = SidescanDataset(val_img,   val_mask,   patch_size=patch_size, base_height=base_height, train=False)

print(f"Train dataset: {len(train_ds)} samples")
print(f"Val   dataset: {len(val_ds)} windows")

x, y = train_ds[0]
print(f"Sample x: {x.shape} {x.dtype}, range [{x.min():.2f}, {x.max():.2f}]")
print(f"Sample y: {y.shape} {y.dtype}, unique values: {y.unique().tolist()}")

# %%
# DataLoaders
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

# %%
# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,   # 2-channel input — no ImageNet pretraining
    in_channels=2,
    classes=1,
).to(device)

_dice_loss = smp.losses.DiceLoss(mode="binary")
_bce_loss  = smp.losses.SoftBCEWithLogitsLoss()

def loss_fn(logits, targets):
    return _dice_loss(logits, targets) + _bce_loss(logits, targets)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")

# %%
# Lightning module

class SidescanSegmentation(L.LightningModule):
    def __init__(self, model, loss_fn, lr=1e-3):
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
        tp, fp, fn, tn = smp.metrics.get_stats(
            logits, y.unsqueeze(1), mode="binary", threshold=0.5
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
    filename="best-{epoch:02d}-{val/iou:.4f}",
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
    precision="bf16-mixed",
)

trainer.fit(lit_model, train_dl, val_dl)

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

    prob_sum = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H,),   dtype=np.float32)

    with torch.no_grad():
        for r0 in tqdm(starts, desc="Inference", unit="window"):
            h    = min(base_height, H - r0)
            crop = img[r0 : r0 + h, :]

            crop_r = np.array(
                Image.fromarray(crop).resize((patch_size, patch_size), Image.BILINEAR),
                dtype=np.float32,
            )
            ch0 = crop_r / 255.0
            ch1 = (crop_r - crop_r.mean(axis=0)) / 255.0
            x   = torch.from_numpy(np.stack([ch0, ch1])).unsqueeze(0).float().to(device)

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


pred_mask, avg_prob = windowed_inference(lit_model.model, sidescan_img, base_height, patch_size, threshold=threshold, device=device)
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
