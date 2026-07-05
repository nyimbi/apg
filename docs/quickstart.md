# APG Quick Start

This guide follows the current executable compiler path: `.apg` source in,
self-contained Flask application out.

## 1. Install

From the repository root:

```bash
uv venv .venv
uv pip install -e ".[dev]"
```

Confirm the current CLI:

```bash
apg --help
apg doctor --json
```

## 2. Write A Small APG Spec

Create `customer_ops.apg`:

```apg
module customer_ops version 1.0.0 {
    description: "Customer operations";
}

table Customer {
    name: str;
    email: str;
    status: str;
}

workflow FollowUp {
    steps: ["new", "contacted", "resolved"];
}

agent SupportPlanner {
    role: "support planner";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Plan customer support follow-up.";
}
```

## 3. Compile

```bash
apg compile customer_ops.apg --output generated/customer_ops --verify
```

The generated directory contains:

```text
app.py
__init__.py
README.md
requirements.txt
semantic_model.json
smoke_test.py
Dockerfile
.dockerignore
.env.example
static/
```

Optional sidecars appear when the APG source declares matching constructs:

```text
ai_agents.py
apg_capabilities.py
apg_application.py
```

## 4. Verify The Generated App

```bash
python generated/customer_ops/app.py --self-test
python generated/customer_ops/smoke_test.py
python generated/customer_ops/app.py --describe
python generated/customer_ops/app.py --semantic-model
python generated/customer_ops/app.py --validate
```

## 5. Run

```bash
python generated/customer_ops/app.py --host 127.0.0.1 --port 8080
```

Open:

- `http://127.0.0.1:8080/ui`
- `http://127.0.0.1:8080/openapi.json`
- `http://127.0.0.1:8080/component.json`
- `http://127.0.0.1:8080/semantic-model.json`

Create a record:

```bash
curl -s \
  -H "Content-Type: application/json" \
  -d '{"record":{"name":"Asha","email":"asha@example.com","status":"new"}}' \
  http://127.0.0.1:8080/entities/Customer/records
```

Use the browser UI:

- `/ui/entities/Customer`
- `/ui/entities/Customer?view=kanban`
- `/ui/entities/Customer?view=analytics`
- `/ui/workflows`
- `/ui/agents/SupportPlanner`

## 6. Compile A Checked Example

```bash
apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-minimal --verify
python /tmp/apg-minimal/app.py --self-test
python /tmp/apg-minimal/smoke_test.py
```

## 7. Refresh The Baseline

When compiler output intentionally changes:

```bash
apg baseline examples --refresh
```

Then run the full test target:

```bash
uv run pytest tests/ -q
```

Current documented evidence: 1486 passed, 1 skipped, and 3 warnings.
