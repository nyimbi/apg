# Sandbox/Testing Environment

**Capability ID**: `sbox` | **Domain**: `common` | **Version**: `1.0.0`

## Description

SBOX gives APG applications a tenant-scoped safe execution runtime: isolation profiles, sandbox templates, controlled datasets, sandbox environments, test runs, run completion evidence, sandbox governance agents, UI metadata, theme

## Installation

```bash
pip install apg-common-sbox
```

## Provides

- `sandbox_registry`
- `isolation_profiles`
- `test_runs`
- `synthetic_datasets`
- `safety_policy`

## Requires

- `plgn`
- `secu`
- `envm`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/sbox/dashboard` | `sbox:view` | Overview |
| `/sbox/sandboxes` | `sbox:create` | Sandboxes |
| `/sbox/templates` | `sbox:create` | Templates |
| `/sbox/datasets` | `sbox:manage_policy` | Data |
| `/sbox/runs` | `sbox:run_tests` | Runs |
| `/sbox/agents` | `sbox:admin` | Operations |
| `/sbox/policies` | `sbox:manage_policy` | Governance |
| `/sbox/audit` | `sbox:admin` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_sandbox()`
- `reset_sandbox()`
- `destroy_sandbox()`
- `sandbox_status()`
- `load_test_data()`
- `mock_service_register()`
- `simulate_event()`
- `run_test_scenario()`

_(See `service.py` for complete API.)_

## Interoperability

`sbox` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use sbox;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `SBOX_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
