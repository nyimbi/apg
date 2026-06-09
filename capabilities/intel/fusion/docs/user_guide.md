# Intelligence Fusion

**Capability ID**: `intel_fusion` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_fusion` is an executable APG capability for lawful, evidence-led intelligence fusion. It can be composed into generated APG applications that need cross-source operational pictures, threat fusion, fraud fusion,

## Installation

```bash
pip install apg-intel-fusion
```

## Provides

- `fusion_authority_workflow`
- `fusion_workspace_workflow`
- `fusion_source_workflow`
- `fusion_artifact_workflow`
- `fusion_correlation_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-fusion/dashboard` | `intel_fusion:view` | Overview |
| `/intel-fusion/authorities` | `intel_fusion:authorities` | Governance |
| `/intel-fusion/workspaces` | `intel_fusion:workspaces` | Planning |
| `/intel-fusion/sources` | `intel_fusion:sources` | Sources |
| `/intel-fusion/artifacts` | `intel_fusion:artifacts` | Evidence |
| `/intel-fusion/correlations` | `intel_fusion:correlations` | Analysis |
| `/intel-fusion/hypotheses` | `intel_fusion:hypotheses` | Analysis |
| `/intel-fusion/assessments` | `intel_fusion:assessments` | Analysis |

## Key Service Methods

- `create_intel_item()`
- `get_intel_item()`
- `list_intel_items()`
- `update_intel_item()`
- `delete_intel_item()`
- `validate_intel_item()`
- `reject_intel_item()`
- `_set_item_status()`
- `create_workspace()`
- `get_workspace()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_fusion` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_fusion;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_FUSION_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
