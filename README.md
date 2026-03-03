# SonarLabel

Minimal repo with:

- `index.html` (web annotator)
- `scripts/verify_parser_parity.py` (JS parser parity check vs `sonarlight`)
- `scripts/plot_sidescan.py` (simple sidescan plot with optional polygon overlay)

## Commands

Setup:

```bash
uv sync
```

Parity check:

```bash
uv run python scripts/verify_parser_parity.py --sonar data/example_sl3_file.sl3
```

Plot with annotations:

```bash
uv run python scripts/plot_sidescan.py \
  --sonar data/example_sl3_file.sl3 \
  --annotations data/annotations.jsonl \
  --plot-output sidescan_annotated.png
```

Plot without annotations:

```bash
uv run python scripts/plot_sidescan.py \
  --sonar data/example_sl3_file.sl3 \
  --plot-output sidescan.png
```
