# Themed I18n Streaming Capability

Combines UI route metadata, visual theme tokens, African language codes, and ByteWax streaming.

## What This Example Demonstrates

- Theme tokens
- i18n language list
- ByteWax stream topology

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/10_themed_i18n_streaming_capability/main.apg --output examples/10_themed_i18n_streaming_capability/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/10_themed_i18n_streaming_capability/main.apg --output examples/10_themed_i18n_streaming_capability/output
```

## Run Generated App

```bash
cd examples/10_themed_i18n_streaming_capability/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
