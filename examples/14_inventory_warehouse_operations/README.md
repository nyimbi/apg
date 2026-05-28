# Inventory Warehouse Operations

Combines executable inventory records with warehouse capability metadata.

## What This Example Demonstrates

- Stock item records
- Stock movement records
- Warehouse rules and streaming

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/14_inventory_warehouse_operations/main.apg --output examples/14_inventory_warehouse_operations/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/14_inventory_warehouse_operations/main.apg --output examples/14_inventory_warehouse_operations/output
```

## Run Generated App

```bash
cd examples/14_inventory_warehouse_operations/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
