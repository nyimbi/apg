# Multi-Runtime Agent Team

Declares a team that can route work across rapidly changing AI runtimes.

## What This Example Demonstrates

- Codex runtime
- Claude Code runtime
- OpenCode/OpenAI/Ollama/Pi alternatives

## Files

- `main.apg` - annotated APG source for this example.
- `output/` - generated dependency-free Python application artifacts compiled from `main.apg`.

## Compile

```bash
apg compile examples/07_multi_runtime_agent_team/main.apg --output examples/07_multi_runtime_agent_team/output
```

Equivalent direct Python invocation from this repository:

```bash
.venv/bin/python -m cli.main compile examples/07_multi_runtime_agent_team/main.apg --output examples/07_multi_runtime_agent_team/output
```

## Run Generated App

```bash
cd examples/07_multi_runtime_agent_team/output
python app.py --self-test
python smoke_test.py
python app.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/ui` for the generated browser interface, or inspect `component.json`, `openapi.json`, and `README.md` in `output/` for composable runtime details.
