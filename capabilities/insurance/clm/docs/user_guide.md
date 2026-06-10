# Claims Management (ins_clm) — User Guide

## Overview

The Claims Management capability provides end-to-end claims handling from FNOL registration through payment and subrogation recovery.

## Use Cases

- **FNOL Registration**: Capture first notification of loss with incident details and estimated loss
- **Loss Assessment**: Assign assessors, capture findings, and submit assessment reports
- **Reserve Management**: Set and adjust outstanding claims reserves (OCR, IBNR, ALAE)
- **Payment Processing**: Authorise and disburse partial, full, or advance claim payments
- **Fraud Detection**: Score claims against fraud indicators and flag high-risk cases
- **Repudiation**: Deny claims with documented reason codes
- **Subrogation**: Initiate and track third-party recovery actions

## Quick Start

```python
from capabilities.insurance.clm.service import ClaimsManagementService
from decimal import Decimal

svc = ClaimsManagementService(tenant_id="acme_insurance")

# Register FNOL
claim = await svc.register_fnol(
    tenant_id="acme_insurance",
    policy_id="pol-001",
    policy_number="POL-2025-001",
    claimant_name="Jane Smith",
    claimant_id="ID-99999",
    incident_date="2025-03-15",
    incident_description="Vehicle rear-ended at intersection",
    estimated_loss=Decimal("350000"),
    reported_by="agent_002",
)

# Set reserve
reserve = await svc.set_reserve(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    reserve_amount=Decimal("300000"),
    reserve_type="outstanding",
    set_by="claims_manager",
    justification="Assessment complete",
)
```

## Fraud Thresholds

Claims with a fraud score >= 0.75 are automatically flagged. Manual flagging is also supported.
