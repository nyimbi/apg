# APG Quick Start

This guide uses the current executable compiler path: APG source in, dependency-light Python application artifacts out.

## 1. Prepare The Workspace

From the repository root:

```bash
uv venv .venv
uv pip install -e ".[dev]"
```

If the virtual environment already exists, refresh it with:

```bash
uv pip install -e ".[dev]"
```

The compiler target is `python`. Older framework-style targets such as `flask-appbuilder` or `django` are intentionally not accepted compiler targets.

## 2. Compile An Existing Example

Compile the smallest checked-in example:

```bash
apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-customer-records --verify
```

The generated directory contains:

- `app.py` - standard-library Python HTTP application
- `__init__.py` - importable package surface
- `README.md` - generated app runbook
- `requirements.txt` - generated app dependency note
- `Dockerfile`, `.dockerignore`, `.env.example` - deployment scaffold
- `smoke_test.py` - standalone generated app smoke test
- Optional composition modules such as `ai_agents.py`, `apg_capabilities.py`, or `apg_application.py`

## 3. Verify The Generated Application

Run the generated self-test:

```bash
python /tmp/apg-customer-records/app.py --self-test
```

Run the standalone smoke test:

```bash
cd /tmp/apg-customer-records
python smoke_test.py
```

A passing generated self-test means the app validated its OpenAPI contract, component manifest, route dispatch table, database schemas, and any generated agent/capability/application contracts.

## 4. Run The Application

Start the generated HTTP app:

```bash
python /tmp/apg-customer-records/app.py --host 127.0.0.1 --port 8080
```

Useful endpoints:

- `GET /health` - runtime health and validation summary
- `GET /self-test` - full generated self-test report
- `GET /manifest` - application metadata
- `GET /component.json` - composable component manifest
- `GET /openapi.json` - OpenAPI 3.1 contract
- `GET /ui` - generated browser UI
- `GET /records` - records grouped by entity

## 5. Create A Small APG File

Create `customer_ops.apg`:

```apg
module customer_ops version 1.0.0 {
    description: "Customer operations";
}

table Customer {
    name: str;
    email: str;
}

agent SupportPlanner {
    role: "support planner";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Plan customer support follow-up.";
}
```

Compile and verify it:

```bash
apg compile customer_ops.apg --output generated/customer_ops --verify
python generated/customer_ops/app.py --self-test
python generated/customer_ops/smoke_test.py
```

Run it:

```bash
python generated/customer_ops/app.py --port 8080
```

Create a record:

```bash
curl -s \
  -H "Content-Type: application/json" \
  -d '{"record":{"name":"Asha","email":"asha@example.com"}}' \
  http://127.0.0.1:8080/entities/Customer/records
```

## 6. Use The Compiler From Python

For scripts and tests:

```python
from compiler.compiler import compile_apg_file

result = compile_apg_file("customer_ops.apg", "generated/customer_ops")
assert result.success, result.errors
print(sorted(result.generated_files))
```

## 7. Work With The Numbered Examples

The `examples/` directory contains 20 parseable APG programs of increasing complexity. Each example has:

- `main.apg` - annotated source
- `README.md` - example-specific explanation
- `output/` - checked-in generated Python application artifacts

The largest executable baseline is:

```bash
python examples/20_enterprise_erp_platform/output/smoke_test.py
```

That smoke test exercises the generated enterprise ERP app self-test and proves the documented OpenAPI routes, composable manifest, UI routes, capability contracts, and route dispatch table remain internally consistent.
