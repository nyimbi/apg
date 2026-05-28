# Capability Rules Configuration

Adds rules, configuration, approvals, and required dependencies.

## What This Example Demonstrates

- Capability rules
- Configuration values
- Approval contract

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/09_capability_rules_configuration/main.apg --output examples/09_capability_rules_configuration/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/09_capability_rules_configuration/main.apg --output examples/09_capability_rules_configuration/output
```

## Run Generated App

```bash
cd examples/09_capability_rules_configuration/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
