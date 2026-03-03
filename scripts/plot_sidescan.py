#!/usr/bin/env python3
"""Dump sidescan sonar data as a raw PNG with optional polygon annotation overlays.

Output is a lossless grayscale PNG (or RGBA when annotations are present):
  - Width : 2800 px (sonar native frame width is used as-is; columns are not rescaled)
  - Height: one pixel row per sidescan ping frame

Expected annotation format (from SonarLabel): JSONL, one JSON object per line, each with
at least: {"id": <int>, "label": <str>, "polygon": [[row, col], ...]}
where row/col are in raw sidescan data coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sonarlight import Sonar

OUTPUT_WIDTH = 2800

# Distinct colours per label (cycling if more labels than entries)
LABEL_COLOURS = [
    (255, 80,  80,  200),   # red
    (80,  220, 160, 200),   # green
    (80,  160, 255, 200),   # blue
    (255, 200,  60, 200),   # yellow
    (200,  80, 255, 200),   # purple
    (80,  230, 230, 200),   # cyan
    (255, 140,  40, 200),   # orange
]


def load_annotations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "polygon" in obj:
                records.append(obj)
        return records

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [obj for obj in parsed if isinstance(obj, dict) and "polygon" in obj]
    if isinstance(parsed, dict) and "annotations" in parsed and isinstance(parsed["annotations"], list):
        return [obj for obj in parsed["annotations"] if isinstance(obj, dict) and "polygon" in obj]
    if isinstance(parsed, dict) and "polygon" in parsed:
        return [parsed]
    return []


def label_colour_map(annotations: list[dict]) -> dict[str, tuple[int, int, int, int]]:
    seen: dict[str, tuple[int, int, int, int]] = {}
    idx = 0
    for ann in annotations:
        lbl = str(ann.get("label", "unknown"))
        if lbl not in seen:
            seen[lbl] = LABEL_COLOURS[idx % len(LABEL_COLOURS)]
            idx += 1
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump sidescan sonar as a raw PNG with optional annotation overlays.")
    parser.add_argument("--sonar", required=True, help="Path to .sl2 or .sl3 file")
    parser.add_argument("--annotations", required=False, default=None, help="Path to annotation file (.jsonl or .json)")
    parser.add_argument("--plot-output", default="sidescan.png", help="Output PNG path")
    args = parser.parse_args()

    sonar = Sonar(args.sonar, clean=False)
    sidescan_df = sonar.df.query("survey == 'sidescan'").reset_index(drop=True)
    if sidescan_df.empty:
        raise ValueError("No sidescan data found in sonar file")

    # All frames are the same length — stack directly
    raw = np.stack(sidescan_df["frames"].to_numpy())  # shape: (n_rows, frame_width)
    n_rows, frame_width = raw.shape

    # Rescale columns to OUTPUT_WIDTH
    if frame_width != OUTPUT_WIDTH:
        img_gray = Image.fromarray(raw.astype(np.uint8), mode="L").resize(
            (OUTPUT_WIDTH, n_rows), resample=Image.NEAREST
        )
        col_scale = OUTPUT_WIDTH / frame_width
    else:
        img_gray = Image.fromarray(raw.astype(np.uint8), mode="L")
        col_scale = 1.0

    annotations = load_annotations(Path(args.annotations)) if args.annotations else []

    if not annotations:
        img_gray.save(args.plot_output, format="PNG", optimize=False)
        print(f"Saved raw PNG: {args.plot_output}  ({OUTPUT_WIDTH}x{n_rows} px, {len(sidescan_df)} frames)")
        return

    # Convert to RGBA so we can draw semi-transparent overlays
    img = img_gray.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colour_map = label_colour_map(annotations)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for ann in annotations:
        polygon = ann.get("polygon", [])
        if len(polygon) < 3:
            continue
        label = str(ann.get("label", "unknown"))
        colour = colour_map[label]
        outline = colour[:3] + (230,)
        fill    = colour[:3] + (60,)

        # Scale col coordinates; rows are 1:1
        pts = [(col * col_scale, row) for row, col in polygon]

        draw.polygon(pts, fill=fill, outline=outline)
        # Label near centroid
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        draw.text((cx + 2, cy + 2), label, fill=(0, 0, 0, 200), font=font)
        draw.text((cx, cy),         label, fill=colour[:3] + (255,), font=font)

    img = Image.alpha_composite(img, overlay).convert("RGB")
    img.save(args.plot_output, format="PNG", optimize=False)

    print(f"Saved annotated PNG: {args.plot_output}  ({OUTPUT_WIDTH}x{n_rows} px, {len(sidescan_df)} frames, {len(annotations)} annotations)")


if __name__ == "__main__":
    main()
