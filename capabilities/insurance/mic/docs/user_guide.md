# Micro-Insurance Platform (ins_mic) — User Guide

## Overview

Enables insurers to distribute low-cost, short-duration products to mobile subscribers via USSD, SMS, WhatsApp, and STK Push. Premiums are collected via airtime deduction or mobile money wallets; claims are paid out instantly via M-Pesa, Airtel Money, or T-Kash.

## Auto-Pay for Small Claims

Claims at or below KES 10,000 are auto-approved and immediately queued for mobile money disbursement — no manual assessment required.

## Supported Operators

safaricom, airtel, telkom, faiba

## Quick Start

```python
from capabilities.insurance.mic.service import MicroInsurancePlatformService
from decimal import Decimal

svc = MicroInsurancePlatformService(tenant_id="acme_insurance")

# Create product
product = await svc.create_product(
    tenant_id="acme_insurance",
    product_code="HOSP30",
    product_name="Hospital Cash 30 Days",
    product_type="hospital",
    sum_insured=Decimal("30000"),
    premium=Decimal("50"),
    coverage_days=30,
    ussd_menu_code="*384#",
    airtime_deduction=True,
    mobile_money_payout=True,
)

# Enrol via USSD
enrolment = await svc.enrol_subscriber(
    tenant_id="acme_insurance",
    msisdn="0712345678",
    product_code="HOSP30",
    name="Mary Wanjiku",
    enrolment_channel="ussd",
    payment_method="airtime",
)
```
