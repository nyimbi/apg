# Inventory Typed Records

Shows generated validation and form coercion for integers and booleans.

## What This Example Demonstrates

- Integer fields
- Boolean fields
- Generated typed forms

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/03_inventory_typed_records/main.apg --output examples/03_inventory_typed_records/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/03_inventory_typed_records/main.apg --output examples/03_inventory_typed_records/output
```

## Run Generated App

```bash
cd examples/03_inventory_typed_records/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
