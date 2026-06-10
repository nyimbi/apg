# Underwriting Engine (ins_und) — User Guide

## Overview

The Underwriting Engine provides automated and manual risk assessment, premium rating, capacity management, and reinsurance treaty administration.

## Use Cases

- **Risk Submissions**: Accept proposals for all supported product lines
- **Automated Scoring**: Heuristic risk scoring with band classification (preferred/standard/substandard/declined)
- **Premium Rating**: Base rate + adjustment factor computation
- **Capacity Checks**: Verify available underwriting capacity before binding
- **Reinsurance Treaties**: Quota share, surplus, excess of loss, facultative management
- **Underwriting Rules**: Configurable rule engine for accept/refer/decline decisions

## Quick Start

```python
from capabilities.insurance.und.service import UnderwritingEngineService
from decimal import Decimal

svc = UnderwritingEngineService(tenant_id="acme_insurance")

# Submit risk
submission = await svc.submit_risk(
    tenant_id="acme_insurance",
    proposer_name="ABC Logistics Ltd",
    proposer_id="BUS-001",
    product_code="motor_comprehensive",
    risk_class="commercial",
    sum_insured=Decimal("5000000"),
    submitted_by="agent_001",
    risk_attributes={"vehicle_age_years": 3, "claim_history_count": 1},
)

# Assess
assessment = await svc.assess_risk("acme_insurance", submission["id"])
# Returns: risk_score, risk_band, recommended_premium, decision
```

## Risk Scoring

Scores are computed on a 0-1 scale. Band mapping:
- 0.0-0.3: preferred (discount applied)
- 0.3-0.6: standard
- 0.6-0.8: substandard (25% loading)
- 0.8-1.0: declined
