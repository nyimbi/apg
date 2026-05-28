# Minimal Customer Records

Introduces the smallest useful APG app: module metadata plus one table.

## What This Example Demonstrates

- Module declaration
- Single typed record entity
- Generated CRUD API and browser UI

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/01_minimal_customer_records/main.apg --output examples/01_minimal_customer_records/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/01_minimal_customer_records/main.apg --output examples/01_minimal_customer_records/output
```

## Run Generated App

```bash
cd examples/01_minimal_customer_records/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
