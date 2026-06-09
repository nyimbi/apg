# Recommender Systems

**Capability ID**: `recs` | **Domain**: `common` | **Version**: `1.0.0`

## Description

RECS is APG's governed recommendation and personalization capability. It provides tenant-scoped recommendation datasets, interaction events, catalog items, user/profile features, ranking policies, model training, model approval,

## Installation

```bash
pip install apg-common-recs
```

## Provides

- `personalized_recommendations`
- `ranking_policies`
- `catalog_matching`
- `interaction_datasets`
- `model_training`

## Requires

- `pred`
- `aicr`
- `nlpc`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/recs/dashboard` | `recs:view` | Overview |
| `/recs/recommendations` | `recs:recommend` | Runtime |
| `/recs/datasets` | `recs:manage_data` | Data |
| `/recs/models` | `recs:manage_models` | Models |
| `/recs/deployments` | `recs:deploy` | Models |
| `/recs/catalogs` | `recs:view` | Data |
| `/recs/profiles` | `recs:view` | Data |
| `/recs/feedback` | `recs:recommend` | Runtime |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_catalog_item()`
- `register_dataset()`
- `record_interaction()`
- `record_profile()`
- `attach_ranking_policy()`
- `train_model()`
- `approve_model()`
- `deploy_model()`

_(See `service.py` for complete API.)_

## Interoperability

`recs` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use recs;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `RECS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
