# Manufacturing Quality Control

Adds production records, quality rules, and shop-floor screen composition.

## What This Example Demonstrates

- Production run table
- Quality-control rules
- Line dashboard screen

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/15_manufacturing_quality_control/main.apg --output examples/15_manufacturing_quality_control/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/15_manufacturing_quality_control/main.apg --output examples/15_manufacturing_quality_control/output
```

## Run Generated App

```bash
cd examples/15_manufacturing_quality_control/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
