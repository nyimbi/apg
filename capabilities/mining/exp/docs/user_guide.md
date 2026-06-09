# Exploration Data Management

**Capability ID**: `mining_exp` | **Domain**: `mining` | **Version**: `1.0.0`

## Description

Manages the full lifecycle of mineral exploration data from drill-hole collar logging through downhole surveys, geological interval logging, geochemical assay management, QAQC monitoring, resource estimation workflows, and JORC/NI 43-101/SAMREC compliance reporting. Enforces data integrity rules including interval non-overlap, competent person requirements, and QAQC insertion obligations before any resource can be published.

## Installation

```bash
pip install apg-mining-exp
```

## Provides

- `drillhole_collar_management`
- `downhole_survey_management`
- `lithology_logging`
- `assay_data_management`
- `qaqc_monitoring`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mining-exp/dashboard` | `mining_exp:view` | Overview |
| `/mining-exp/drillholes` | `mining_exp:view` | Field Data |
| `/mining-exp/drillholes/create` | `mining_exp:write` | Field Data |
| `/mining-exp/drillholes/:id` | `mining_exp:view` | Field Data |
| `/mining-exp/assays` | `mining_exp:view` | Geochemistry |
| `/mining-exp/assays/import` | `mining_exp:write` | Geochemistry |
| `/mining-exp/geology` | `mining_exp:view` | Geology |
| `/mining-exp/qaqc` | `mining_exp:view` | Quality |

## Key Service Methods

- `create_drillhole_collar()`
- `get_drillhole_collar()`
- `get_drillhole_collar_by_hole_id()`
- `list_drillhole_collars()`
- `update_drillhole_actual_depth()`
- `import_assay_results()`
- `_check_assay_interval_overlap()`
- `get_assay_results_for_hole()`
- `flag_qaqc_result()`
- `list_assays()`

_(See `service.py` for complete API.)_

## Interoperability

`mining_exp` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mining_exp;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MINING_EXP_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
