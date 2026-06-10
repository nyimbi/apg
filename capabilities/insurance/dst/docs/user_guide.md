# Distribution & Agency Management (ins_dst) — User Guide

## Overview

Manages the full agent lifecycle: registration, commission computation, performance reporting, compliance tracking, and bancassurance partnerships.

## Agent Types

tied, independent, bancassurance, digital, broker, corporate

## Commission Structure

Default commission rates by product are built-in but can be overridden per transaction. Commission workflow: pending → approved → paid.

## Quick Start

```python
from capabilities.insurance.dst.service import DistributionAgencyService
from decimal import Decimal

svc = DistributionAgencyService(tenant_id="acme_insurance")

agent = await svc.register_agent(
    tenant_id="acme_insurance",
    agent_code="AGT-001",
    agent_name="John Kamau",
    agent_type="tied",
    id_number="12345678",
    ira_licence_number="IRA/AGT/2024/001",
    phone="+254700000001",
    email="jkamau@example.com",
)

commission = await svc.compute_commission(
    tenant_id="acme_insurance",
    agent_id=agent["id"],
    policy_id="pol-001",
    policy_number="POL-2025-001",
    product_code="motor_comprehensive",
    premium_amount=Decimal("45000"),
)
# commission_amount = 45000 * 12.5% = 5625
```
