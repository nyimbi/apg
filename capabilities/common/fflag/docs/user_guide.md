# Feature Flags User Guide

## Overview

The Feature Flags capability (`fflag`) provides runtime feature toggles with percentage rollout, A/B experiment assignment, per-user targeting rules, and a complete audit trail. Evaluation is deterministic — the same user always gets the same flag result for a given flag configuration.

## Key Concepts

- **Flag**: a named boolean toggle with optional variants
- **Rollout**: percentage of users to receive the flag (0–100)
- **Targeting Rule**: conditions on user attributes that override rollout (e.g., beta users)
- **Override**: hard-coded per-user flag assignment for testing or support
- **Experiment**: A/B test tied to a flag with weighted variant assignment

## Quickstart

### Create a flag

```http
POST /api/fflag/flags
{
  "tenant_id": "acme",
  "key": "new_checkout_flow",
  "name": "New Checkout Flow",
  "enabled": true,
  "rollout_percentage": 25.0,
  "owner": "payments-team"
}
```

### Evaluate for a user

```http
GET /api/fflag/evaluate/new_checkout_flow?tenant_id=acme&user_id=user-123
```

Response:
```json
{
  "flag_key": "new_checkout_flow",
  "enabled": true,
  "variant": null,
  "reason": "rollout",
  "targeting_matched": false
}
```

### Add a targeting rule (beta users)

```http
PUT /api/fflag/flags/new_checkout_flow
{
  "tenant_id": "acme",
  "targeting_rules": [{
    "conditions": [{"attribute": "user_tier", "operator": "eq", "value": "beta"}],
    "enabled": true,
    "variant": "beta_variant"
  }]
}
```

### Run an A/B experiment

```http
POST /api/fflag/experiments
{
  "tenant_id": "acme",
  "flag_key": "new_checkout_flow",
  "name": "Checkout A/B Test",
  "variants": [
    {"key": "control", "weight": 50},
    {"key": "treatment", "weight": 50}
  ]
}

POST /api/fflag/experiments/{id}/start
POST /api/fflag/experiments/{id}/assign  {"user_id": "user-456"}
```

## Evaluation Priority

1. Per-user override (highest priority)
2. Targeting rules (first match wins)
3. Percentage rollout (consistent hash)
4. Flag disabled (returns false)
