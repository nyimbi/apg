# Workflow Orchestration Specification

## Objective

Build `composition_orchestration` as an executable APG capability that generated applications can compose for business process definition, validation, release, execution, human task coordination, and AI-assisted workflow governance.

## Functional Surface

- Workflow definition lifecycle with owner, version, start event, terminal state, and task graph.
- Task validation for handlers, human assignment, approval policies, SLA escalation, retry limits, and cross-capability contracts.
- Graph validation with unknown-dependency and cycle detection.
- Release governance with validation evidence, dry-run result, rollback plan, and approval metadata.
- Execution lifecycle with idempotency key, active task set, dependency-based task advancement, and completion state.
- Human task assignment records.
- Bytewax lifecycle stream metadata for all orchestration state changes.
- Deterministic rule engine for operational guardrails.
- UI routes, view models, and theme tokens for generated applications.
- AI agent registration and privileged-action validation for `codex`, `claude_code`, `opencode`, and `pi`.

## Non-Goals

- Embedding a live workflow scheduler or worker pool inside the package surface.
- Owning auth, audit, notification, registry, or event infrastructure.
- Requiring FastAPI, Flask-AppBuilder, Airflow, Prefect, Celery, or other optional orchestration engines for package loading.

## Acceptance Criteria

- `get_capability_contract()` validates through `capabilities.capability_contract_registry.validate_contract_shape`.
- `WorkflowOrchestrationService` can define, release, start, advance, and summarize workflows without optional web or scheduler dependencies.
- Rules deny unsafe definitions, unsafe releases, unsafe execution starts, unsupported agents, and non-Bytewax lifecycle coordination.
- `app.semantic_model()` exposes provides/requires, configuration, rules, screens, theme, streaming, and agent team metadata.
- Focused package tests compile and pass.
- `apg capabilities inspect`, `publish-plan`, and `implementation-audit` succeed for the package.
