# Custom Scripting Engine

**Capability ID**: `scpt` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`scpt` is the APG common capability for governed custom scripting. It gives generated applications a dependency-light runtime for registering tenant-owned scripts, constraining them with package and sandbox policy, approving risky

## Installation

```bash
pip install apg-common-scpt
```

## Provides

- `script_registry`
- `secure_sandbox`
- `workflow_extensions`
- `package_policy`
- `script_execution`

## Requires

- `wflo`
- `secu`
- `auth`
- `audl`
- `aicr`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/scpt/dashboard` | `scpt:view` | Overview |
| `/scpt/workbench` | `scpt:write` | Scripts |
| `/scpt/scripts` | `scpt:view` | Scripts |
| `/scpt/executions` | `scpt:execute` | Runtime |
| `/scpt/sandboxes` | `scpt:admin` | Runtime |
| `/scpt/packages` | `scpt:approve` | Governance |
| `/scpt/approvals` | `scpt:approve` | Governance |
| `/scpt/agents` | `scpt:write` | Scripts |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_package_policy()`
- `create_sandbox()`
- `create_script()`
- `request_script_review()`
- `approve_script()`
- `publish_script()`
- `bind_workflow()`
- `execute_script()`

_(See `service.py` for complete API.)_

## Interoperability

`scpt` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use scpt;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `SCPT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
