# Policy Administration (ins_pol) — User Guide

## Overview

The Policy Administration capability manages the complete lifecycle of insurance policies across all product lines supported by the platform.

## Use Cases

- **Policy Issuance**: Issue new policies for motor, fire, marine, life, health, and liability lines
- **Endorsements**: Process mid-term changes (sum insured adjustments, beneficiary changes, vehicle swaps)
- **Renewals**: Manage annual and multi-year renewal cycles with premium recalculation
- **Cancellations**: Process voluntary and involuntary cancellations with pro-rata refund calculation
- **Reinstatements**: Reinstate lapsed or cancelled policies subject to outstanding premium settlement
- **Document Generation**: Produce policy schedules, certificates, renewal notices, and endorsement schedules

## Quick Start

```python
from capabilities.insurance.pol.service import PolicyAdministrationService
from decimal import Decimal

svc = PolicyAdministrationService(tenant_id="acme_insurance")

# Issue a policy
policy = await svc.create_policy(
    tenant_id="acme_insurance",
    policy_number="POL-2025-001",
    product_code="motor_comprehensive",
    insured_name="John Doe",
    insured_id="ID-12345",
    sum_insured=Decimal("2000000"),
    inception_date="2025-01-01",
    expiry_date="2025-12-31",
    premium=Decimal("45000"),
    underwriter_id="UW-001",
)

# Endorse it
endorsement = await svc.create_endorsement(
    tenant_id="acme_insurance",
    policy_id=policy["id"],
    endorsement_type="sum_insured_change",
    effective_date="2025-06-01",
    description="Increase sum insured",
    change_in_sum_insured=Decimal("500000"),
    requested_by="agent_001",
)
```

## API Reference

See README.md for full endpoint listing.

## Supported Products

motor_comprehensive, motor_third_party, fire_industrial, fire_domestic, marine_cargo, marine_hull, life_whole, life_term, health_individual, health_group, travel, engineering, liability_public, liability_employers
