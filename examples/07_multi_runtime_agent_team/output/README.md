# multi_runtime_team

Dependency-free APG generated Python application.

## Run

```bash
python app.py
```

## Verify

```bash
python app.py --self-test
python smoke_test.py
python app.py --describe
python app.py --semantic-model
python app.py --validate
```

## Core HTTP endpoints

- `GET /health` - runtime health and validation summary
- `GET /component.json` - composable application component manifest
- `GET /semantic-model.json` - normalized APG semantic model
- `GET /self-test` - generated app smoke contract
- `GET /manifest` - application manifest
- `GET /openapi.json` - OpenAPI 3.1 contract
- `GET /metrics` - runtime metrics snapshot
- `GET /ui` - generated HTML application index

## Browser UI

- Open the generated browser interface at `/ui` after starting `python app.py`.
- Entity screens include dependency-free create, edit, delete, and validation-error flows.
- Typed APG fields render as matching HTML controls and are coerced before validation.
- Record edits and deletes use `_revision` checks to avoid overwriting stale browser forms.

## Data records

- `GET /records` - all records grouped by entity
- `GET /entities/{Entity}/records` - query records for an entity
- `POST /entities/{Entity}/records` - create a record
- `PUT /entities/{Entity}/records/{id}` - update a record
- `DELETE /entities/{Entity}/records/{id}` - delete a record
- `GET /entities/{Entity}/records/export` - export records
- `POST /entities/{Entity}/records/import` - import records

Python package helpers: `create_record()`, `get_record()`, `query_records()`, `update_record()`, and `delete_record()` expose the same executable record behavior for composition.

Set `APG_DATA_FILE=/path/to/data.json` to persist records to JSON.
Set `APG_API_KEY=<key>` to require an API key for mutations.

## Deployment

```bash
docker build -t apg-generated-app .
docker run --rm -p 8080:8080 --env-file .env.example apg-generated-app
```

Generated deployment artifacts:

- `Dockerfile` - Flask 3.x container entrypoint
- `.dockerignore` - container build exclusions
- `.env.example` - documented runtime environment variables
- `semantic_model.json` - normalized APG semantic model for IDEs, agents, and release checks
- `smoke_test.py` - standalone generated app smoke test

## Entities

- `AnalyticsCrew`
- `MultiRuntimeApp`

## AI agents

- `CloudAnalyst` - runtime `local`, invoke with `POST /agents/CloudAnalyst/invoke`
- `LocalPrivacyAgent` - runtime `local`, invoke with `POST /agents/LocalPrivacyAgent/invoke`
- `ConversationalAgent` - runtime `local`, invoke with `POST /agents/ConversationalAgent/invoke`
- `OpenSourceCoder` - runtime `local`, invoke with `POST /agents/OpenSourceCoder/invoke`

Typed agent stub classes live in `agent_stubs.py`. Wire up a runtime adapter by setting the environment variable:

```
export APG_AGENT_CODEX_PROVIDER_COMMAND='python my_provider.py'
```

The provider receives JSON `{"agent": {...}, "input": "...", "context": {...}}` on stdin and writes `{"output": "..."}` to stdout.
