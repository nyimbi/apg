# Property Valuation

**Capability ID**: `realestate_val` | **Domain**: `realestate` | **Version**: `1.0.0`

## Description

Full-cycle property valuation: comparable sales database, DCF model builder with range-validated discount rates, mass appraisal engine (regression, spatial, hedonic, AI AVM), valuation roll with automatic supersession, revaluation cycle management, Red Book sign-off enforcement with independent valuer validation, and structured challenge workflow requiring counter-evidence.

## Installation

```bash
pip install apg-realestate-val
```

## Provides

- `comparable_sales_analysis`
- `dcf_valuation_engine`
- `mass_appraisal_engine`
- `valuation_roll_management`
- `revaluation_cycle_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/val/dashboard` | `realestate_val:view` | Overview |
| `/realestate/val/valuations` | `realestate_val:valuations` | Valuations |
| `/realestate/val/valuations/<id>` | `realestate_val:valuations` | Valuations |
| `/realestate/val/comparables` | `realestate_val:comparables` | Analysis |
| `/realestate/val/dcf` | `realestate_val:dcf` | Models |
| `/realestate/val/mass-appraisal` | `realestate_val:mass_appraisal` | Models |
| `/realestate/val/roll` | `realestate_val:roll` | Roll |
| `/realestate/val/cycles` | `realestate_val:cycles` | Planning |

## Key Service Methods

- `register_valuer()`
- `get_valuer()`
- `list_valuers()`
- `add_comparable()`
- `list_comparables()`
- `verify_comparable()`
- `instruct_valuation()`
- `get_valuation()`
- `list_valuations()`
- `update_valuation()`

_(See `service.py` for complete API.)_

## Interoperability

`realestate_val` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use realestate_val;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `REALESTATE_VAL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
