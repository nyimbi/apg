# Prescriptive Analytics

**Capability ID**: `bia_psa` | **Domain**: `bia` | **Version**: `1.0.0`

## Description

The Prescriptive Analytics capability (bia_psa) provides optimisation engines (LP, IP, GA, RL), decision support with explainability, recommendation action management with approval workflows, and what-if analysis — all tenant-scoped with mandatory governance and full audit.

## Installation

```bash
pip install apg-bia-psa
```

## Provides

- `optimisation_engine`
- `decision_support_system`
- `recommendation_actions`
- `whatif_analysis`
- `constraint_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `mqeb`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bia/psa/dashboard` | `bia_psa:view` | Overview |
| `/bia/psa/optimisations` | `bia_psa:optimise` | Optimisation |
| `/bia/psa/optimisations/<id>` | `bia_psa:optimise` | Optimisation |
| `/bia/psa/decisions` | `bia_psa:decisions` | Decisions |
| `/bia/psa/decisions/<id>` | `bia_psa:decisions` | Decisions |
| `/bia/psa/recommendations` | `bia_psa:recommendations` | Recommendations |
| `/bia/psa/recommendations/<id>` | `bia_psa:recommendations` | Recommendations |
| `/bia/psa/whatif` | `bia_psa:whatif` | Simulation |

## Key Service Methods

- `describe()`
- `create_optimisation()`
- `get_optimisation()`
- `list_optimisations()`
- `run_optimisation()`
- `archive_optimisation()`
- `delete_optimisation()`
- `optimisation_problem()`
- `linear_programme()`
- `simulation_run()`

_(See `service.py` for complete API.)_

## Interoperability

`bia_psa` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bia_psa;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BIA_PSA_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
