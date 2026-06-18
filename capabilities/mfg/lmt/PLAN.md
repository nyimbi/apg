# Lot and Batch Management Development Plan

## Phases

### Phase 1 — Models
Define SQLAlchemy ORM models and Pydantic schemas in `models.py` / `views.py`.

### Phase 2 — Service Layer
Implement business logic in `service.py` with async methods.

### Phase 3 — API
Wire Flask Blueprint routes in `api.py`.

### Phase 4 — Tests
Write unit and integration tests under `tests/`.

### Phase 5 — Release
Generate `release_report.json` after all tests pass.
