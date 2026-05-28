# Procurement Approval Workbench

Models procurement approval workflows and a split approval queue screen.

## What This Example Demonstrates

- Purchase request capability
- Three-level approvals
- Approval queue screen

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/13_procurement_approval_workbench/main.apg --output examples/13_procurement_approval_workbench/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/13_procurement_approval_workbench/main.apg --output examples/13_procurement_approval_workbench/output
```

## Run Generated App

```bash
cd examples/13_procurement_approval_workbench/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
