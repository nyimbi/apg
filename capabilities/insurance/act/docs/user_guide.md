# Actuarial Tools (ins_act) — User Guide

## Overview

Provides actuarial work-product management: mortality/morbidity tables, loss ratio analysis, technical reserves, IBNR estimation, pricing models, and experience studies.

## Use Cases

- **Mortality Tables**: Load and query CSO 1958/1980/2001/2017 or population tables
- **Loss Ratios**: Compute loss ratio, expense ratio, and combined ratio by product/period
- **Technical Reserves**: Calculate OCR and UPR using chain-ladder, BF, or cape-cod methods
- **IBNR**: Estimate incurred-but-not-reported liabilities from claims triangles
- **Pricing Models**: Parameterise GLM-style rating models with risk factors and adjustment coefficients
- **Experience Analysis**: A/E frequency and severity studies for assumption validation

## Quick Start

```python
from capabilities.insurance.act.service import ActuarialToolsService
from decimal import Decimal

svc = ActuarialToolsService(tenant_id="acme_insurance")

# Loss ratio
lr = await svc.calculate_loss_ratio(
    tenant_id="acme_insurance",
    product_code="motor_comprehensive",
    period_start="2024-01-01",
    period_end="2024-12-31",
    earned_premium=Decimal("50000000"),
    incurred_losses=Decimal("32000000"),
    expenses=Decimal("8000000"),
)
# loss_ratio_pct: 64.00, combined_ratio_pct: 80.00
```
