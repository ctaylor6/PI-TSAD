# Contributing

Thanks for helping improve PI-TSAD.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

```bash
pytest
ruff check src tests examples
```

## Data Policy

Keep the repository small and reproducible:

- Commit only compact example data needed for tests or tutorials.
- Do not commit full-part builds, model artifacts, local notebooks, or generated outputs.
- Put large local files outside the repo or under ignored output folders.

## Method Changes

When changing the detection method, run the APS 4-fold validation:

```bash
pi-tsad cross-validate-aps
```

Update the README if command behavior, expected CSV columns, or model defaults change.
