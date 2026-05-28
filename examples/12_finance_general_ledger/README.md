# Finance General Ledger

Builds a finance ERP capability with rules, approvals, and localization.

## What This Example Demonstrates

- ERP module mapping
- Business rules
- Ledger screen route

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/12_finance_general_ledger/main.apg --output examples/12_finance_general_ledger/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/12_finance_general_ledger/main.apg --output examples/12_finance_general_ledger/output
```

## Run Generated App

```bash
cd examples/12_finance_general_ledger/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
