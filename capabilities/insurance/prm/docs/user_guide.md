# Premium & Billing (ins_prm) — User Guide

## Overview

Manages all premium-related financial flows: schedule creation, instalment tracking, payment collection, refunds, and period reconciliation.

## Supported Frequencies

- annual (1 instalment)
- semi_annual (2 instalments)
- quarterly (4 instalments)
- monthly (12 instalments)

## Quick Start

```python
from capabilities.insurance.prm.service import PremiumBillingService
from decimal import Decimal

svc = PremiumBillingService(tenant_id="acme_insurance")

# Create schedule
schedule = await svc.create_schedule(
    tenant_id="acme_insurance",
    policy_id="pol-001",
    policy_number="POL-2025-001",
    total_premium=Decimal("45000"),
    frequency="quarterly",
    inception_date="2025-01-01",
    expiry_date="2025-12-31",
)

# Collect first instalment
instalments = await svc.list_instalments("acme_insurance", schedule["id"])
collection = await svc.collect_payment(
    tenant_id="acme_insurance",
    instalment_id=instalments[0]["id"],
    payment_method="mpesa",
    payment_reference="QW123456",
    amount=Decimal("11250"),
    collected_by="cashier_01",
)
```
