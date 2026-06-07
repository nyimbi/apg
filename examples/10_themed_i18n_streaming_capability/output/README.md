# localised_inventory

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

- Open `http://127.0.0.1:8080/ui` after starting `python app.py`.
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

- `Dockerfile` - standard-library Python container entrypoint
- `.dockerignore` - container build exclusions
- `.env.example` - documented runtime environment variables
- `semantic_model.json` - normalized APG semantic model for IDEs, agents, and release checks
- `smoke_test.py` - standalone generated app smoke test

## Entities

- `LocalisedInventoryApp`

## Capabilities

- `LocalisedInventory` - provides stock_balances, inventory_events, reorder_alerts

Capability operations:

- `GET /capabilities` - capability catalog and dependency graph
- `GET /streaming` - ByteWax streaming topology
- `GET /capabilities/{Capability}/streaming` - capability streaming contract
- `POST /capabilities/{Capability}/rules/evaluate` - evaluate capability rules
- `POST /capabilities/{Capability}/configuration/resolve` - resolve configuration
- `POST /capabilities/{Capability}/configuration/validate` - validate configuration
- `POST /capabilities/{Capability}/approval/plan` - plan approvals

Capability screens:

- `GET /inventory`
- `GET /inventory/movements`
- `GET /inventory/reorder`
