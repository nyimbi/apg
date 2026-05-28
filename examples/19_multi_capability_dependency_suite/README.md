# Multi-Capability Dependency Suite

Shows provides/requires planning across multiple ERP capabilities.

## What This Example Demonstrates

- Audit events dependency
- Customer master capability
- Order and billing dependencies

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/19_multi_capability_dependency_suite/main.apg --output examples/19_multi_capability_dependency_suite/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/19_multi_capability_dependency_suite/main.apg --output examples/19_multi_capability_dependency_suite/output
```

## Run Generated App

```bash
cd examples/19_multi_capability_dependency_suite/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
