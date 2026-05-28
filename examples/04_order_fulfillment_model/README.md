# Order Fulfillment Model

Models a small operational workflow with customers, orders, shipments, and events.

## What This Example Demonstrates

- Multi-table CRUD app
- Shipment tracking records
- Event records with ISO timestamp strings

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/04_order_fulfillment_model/main.apg --output examples/04_order_fulfillment_model/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/04_order_fulfillment_model/main.apg --output examples/04_order_fulfillment_model/output
```

## Run Generated App

```bash
cd examples/04_order_fulfillment_model/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
