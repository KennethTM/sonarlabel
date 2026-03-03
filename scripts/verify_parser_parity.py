#!/usr/bin/env python3
"""Verify SL3 sidescan parsing parity between a JS-equivalent parser and sonarlight.Sonar.

This parser mirrors the browser logic in index.html:
- Starts at file byte offset 8
- Reads frame_size at header offset 8
- Advances position by frame_size
- Uses only frames where new position < buffer length
- Keeps survey_type == 5 as sidescan
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sonarlight import Sonar

FILE_HEADER_SIZE = 8
SL3_HEADER_SIZE = 168
FEET_TO_METERS = 1 / 3.2808399


@dataclass
class SidescanFrame:
    x: int
    y: int
    min_range: float
    max_range: float
    water_depth: float
    gps_heading: float
    echo: np.ndarray


def parse_sl3_js_equivalent(path: str) -> list[SidescanFrame]:
    blob = Path(path).read_bytes()
    buf = memoryview(blob)

    position = FILE_HEADER_SIZE
    out: list[SidescanFrame] = []

    while position < len(buf):
        if position + SL3_HEADER_SIZE > len(buf):
            break

        frame_size = int.from_bytes(buf[position + 8 : position + 10], "little", signed=False)
        if frame_size == 0:
            break

        frame_start = position
        position += frame_size

        if position > len(buf):
            break

        if position < len(buf):
            survey_type = int.from_bytes(buf[frame_start + 12 : frame_start + 14], "little", signed=False)
            if survey_type != 5:
                continue

            x = int.from_bytes(buf[frame_start + 92 : frame_start + 96], "little", signed=True)
            y = int.from_bytes(buf[frame_start + 96 : frame_start + 100], "little", signed=True)
            min_range = np.frombuffer(buf[frame_start + 20 : frame_start + 24], dtype="<f4")[0] * FEET_TO_METERS
            max_range = np.frombuffer(buf[frame_start + 24 : frame_start + 28], dtype="<f4")[0] * FEET_TO_METERS
            water_depth = np.frombuffer(buf[frame_start + 48 : frame_start + 52], dtype="<f4")[0] * FEET_TO_METERS
            gps_heading = np.frombuffer(buf[frame_start + 104 : frame_start + 108], dtype="<f4")[0]
            echo_size = int.from_bytes(buf[frame_start + 44 : frame_start + 48], "little", signed=False)
            echo = np.frombuffer(buf[frame_start + SL3_HEADER_SIZE : frame_start + SL3_HEADER_SIZE + echo_size], dtype="uint8")

            out.append(
                SidescanFrame(
                    x=x,
                    y=y,
                    min_range=float(min_range),
                    max_range=float(max_range),
                    water_depth=float(water_depth),
                    gps_heading=float(gps_heading),
                    echo=echo,
                )
            )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify parser parity between JS-equivalent parser and sonarlight.Sonar")
    parser.add_argument("--sonar", required=True, help="Path to .sl3 file")
    args = parser.parse_args()

    js_frames = parse_sl3_js_equivalent(args.sonar)

    sonar = Sonar(args.sonar, clean=False)
    py_df = sonar.df.query("survey == 'sidescan'").reset_index(drop=True)

    if len(js_frames) != len(py_df):
        raise AssertionError(f"Frame count mismatch: js={len(js_frames)} py={len(py_df)}")

    for i, (jf, (_, row)) in enumerate(zip(js_frames, py_df.iterrows())):
        if jf.x != int(row["x"]):
            raise AssertionError(f"x mismatch at row {i}: js={jf.x} py={int(row['x'])}")
        if jf.y != int(row["y"]):
            raise AssertionError(f"y mismatch at row {i}: js={jf.y} py={int(row['y'])}")
        if not np.isclose(jf.min_range, float(row["min_range"])):
            raise AssertionError(f"min_range mismatch at row {i}")
        if not np.isclose(jf.max_range, float(row["max_range"])):
            raise AssertionError(f"max_range mismatch at row {i}")
        if not np.isclose(jf.water_depth, float(row["water_depth"])):
            raise AssertionError(f"water_depth mismatch at row {i}")
        if not np.isclose(jf.gps_heading, float(row["gps_heading"])):
            raise AssertionError(f"gps_heading mismatch at row {i}")

        py_echo = row["frames"]
        if len(jf.echo) != len(py_echo):
            raise AssertionError(f"echo length mismatch at row {i}: js={len(jf.echo)} py={len(py_echo)}")
        if not np.array_equal(jf.echo, py_echo):
            raise AssertionError(f"echo bytes mismatch at row {i}")

    print(f"Parity OK for {len(js_frames)} sidescan frames")


if __name__ == "__main__":
    main()
