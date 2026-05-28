# Customer Orders Relationship

Adds a second table and a conventional customer_id relationship.

## What This Example Demonstrates

- Two related tables
- Relationship graph by *_id convention
- Order totals as numeric fields

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/02_customer_orders_relationship/main.apg --output examples/02_customer_orders_relationship/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/02_customer_orders_relationship/main.apg --output examples/02_customer_orders_relationship/output
```

## Run Generated App

```bash
cd examples/02_customer_orders_relationship/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
