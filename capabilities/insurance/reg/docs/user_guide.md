# Insurance Regulatory Reporting (ins_reg) — User Guide

## Overview

Manages the end-to-end regulatory return lifecycle: preparation, review, submission, and acceptance across IRA (Kenya), NAICOM (Nigeria), FSA, and FCA.

## Workflow

draft → prepared → reviewed → submitted → accepted | rejected

## Supported Regulators

IRA, NAICOM, FSA, FCA, IAIS, AKI, PRA

## Solvency Reporting

The SCR ratio = Eligible Own Funds / SCR. Triggers warnings if SCR ratio < 1.0 or MCR ratio < 0.25.

## Quick Start

```python
from capabilities.insurance.reg.service import InsuranceRegulatoryReportingService
from decimal import Decimal

svc = InsuranceRegulatoryReportingService(tenant_id="acme_insurance")

# Create quarterly IRA return
ret = await svc.create_return(
    tenant_id="acme_insurance",
    return_type="quarterly_statistical",
    regulator="IRA",
    period_start="2025-01-01",
    period_end="2025-03-31",
    prepared_by="actuarial_dept",
)

# Submit through workflow
await svc.review_return("acme_insurance", ret["id"], "compliance_officer")
await svc.submit_return("acme_insurance", ret["id"], "compliance_officer")
```
