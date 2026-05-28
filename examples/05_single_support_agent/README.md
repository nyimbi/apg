# Single Support Agent

Introduces a first-class AI agent declaration.

## What This Example Demonstrates

- Agent role/model/runtime
- Tools and vector memory
- Declarative agent rule

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/05_single_support_agent/main.apg --output examples/05_single_support_agent/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/05_single_support_agent/main.apg --output examples/05_single_support_agent/output
```

## Run Generated App

```bash
cd examples/05_single_support_agent/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
