# Screen Composition Relationships

Shows how screens contain, compose, bind, and relate elements.

## What This Example Demonstrates

- Dashboard screen
- Contains/composes/binds
- Screen relationship edges

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/11_screen_composition_relationships/main.apg --output examples/11_screen_composition_relationships/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/11_screen_composition_relationships/main.apg --output examples/11_screen_composition_relationships/output
```

## Run Generated App

```bash
cd examples/11_screen_composition_relationships/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
