# Basic Capability Contract

Introduces a first-class capability with provides/configuration metadata.

## What This Example Demonstrates

- Capability id
- Provides contract
- Master data declaration

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/08_basic_capability_contract/main.apg --output examples/08_basic_capability_contract/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/08_basic_capability_contract/main.apg --output examples/08_basic_capability_contract/output
```

## Run Generated App

```bash
cd examples/08_basic_capability_contract/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
