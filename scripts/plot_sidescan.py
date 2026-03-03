#!/usr/bin/env python3
"""Plot SonarLight sidescan imagery with optional polygon annotations.

Expected annotation format (from SonarLabel): one JSON object per line (JSONL), each with
at least: {"id": <int>, "label": <str>, "polygon": [[row, col], ...]}
where row/col are in raw sidescan data coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sonarlight import Sonar


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sidescan image with optional annotation overlays.")
    parser.add_argument("--sonar", required=True, help="Path to .sl2 or .sl3 file")
    parser.add_argument("--annotations", required=False, default=None, help="Path to annotation file (.jsonl or .json)")
    parser.add_argument("--plot-output", default="sidescan.png", help="Output PNG for plotted image and polygons")
    args = parser.parse_args()

    sonar = Sonar(args.sonar, clean=False)
    sidescan_df = sonar.df.query("survey == 'sidescan'").reset_index(drop=True)
    if sidescan_df.empty:
        raise ValueError("No sidescan data found in sonar file")

    image = np.stack(sidescan_df["frames"].to_numpy())
    annotations = load_annotations(Path(args.annotations)) if args.annotations else []

    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
    ax.imshow(image, cmap="gray", aspect="auto", origin="upper")
    ax.set_title("Sidescan with Annotations")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    for ann in annotations:
        polygon = np.asarray(ann.get("polygon", []), dtype=float)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            continue

        label = str(ann.get("label", "unknown"))

        rows = polygon[:, 0]
        cols = polygon[:, 1]

        ax.plot(np.r_[cols, cols[0]], np.r_[rows, rows[0]], linewidth=1.5)
        if len(rows) > 0:
            center_row = float(np.mean(rows))
            center_col = float(np.mean(cols))
            ax.text(center_col, center_row, f" {label}", fontsize=8, va="center")

    fig.tight_layout()
    fig.savefig(args.plot_output)

    print(f"Saved overlay plot: {args.plot_output}")
    print(f"Annotations processed: {len(annotations)}")


if __name__ == "__main__":
    main()
