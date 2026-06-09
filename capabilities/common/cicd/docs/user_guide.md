# Continuous Integration and Delivery

**Capability ID**: `cicd` | **Domain**: `common` | **Version**: `1.0.0`

## Description

CICD is the APG capability for governed build, test, package, scan, promotion, and release-delivery workflows. It gives generated APG applications a tenant-aware CI/CD lifecycle that can be composed with deployment, environment,

## Installation

```bash
pip install apg-common-cicd
```

## Provides

- `pipeline_management`
- `build_orchestration`
- `quality_gates`
- `artifact_promotion`
- `release_automation`

## Requires

- `depl`
- `envm`
- `logt`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/cicd/dashboard` | `cicd:view` | Overview |
| `/cicd/pipelines` | `cicd:manage_pipelines` | Pipelines |
| `/cicd/builds` | `cicd:run_builds` | Builds |
| `/cicd/artifacts` | `cicd:view` | Artifacts |
| `/cicd/gates` | `cicd:promote` | Release |
| `/cicd/promotions` | `cicd:promote` | Release |
| `/cicd/agents` | `cicd:promote` | Agents |
| `/cicd/audit` | `cicd:audit` | Governance |

## Key Service Methods

- `uuid7str()`
- `_audit()`
- `pipeline_create()`
- `trigger_build()`
- `build_complete()`
- `store_artifact()`
- `quality_gate_add()`
- `deployment_promote()`
- `rollback_release()`
- `feature_flag_release()`

_(See `service.py` for complete API.)_

## Interoperability

`cicd` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use cicd;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `CICD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
