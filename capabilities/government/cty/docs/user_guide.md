# County / Devolved Services — User Guide

## Overview

The County Services capability (gov_cty) provides a unified platform for Kenya county governments to manage devolved services including revenue collection, permit administration, social welfare disbursement, health facility management, and public works maintenance.

## Core Modules

### Revenue Collection
Collect and track county revenues: land rates, business permits, parking fees, market fees, advertisement levies, and building permits. Supports M-Pesa, bank transfer, and cash payment methods.

### Permit Issuance
Manage the full permit lifecycle from application through approval/rejection to issuance. Covers business permits, building permits, health certificates, liquor licences, food hygiene certificates, and fire clearance certificates.

### Social Welfare
Process applications for devolved social protection programmes: cash transfers, food subsidies, elderly grants, disability grants, orphan support, and school bursaries. Includes needs assessment and case worker assignment.

### Devolved Health
Register and manage county health facilities (dispensaries, health centres, hospitals). Register patients and maintain basic health records for the county health information system.

### Public Works Ticketing
Citizens and county staff report public infrastructure issues: road potholes, drainage blockages, broken streetlights, water supply failures, and waste collection complaints. Full ticket lifecycle with priority levels and contractor assignment.

## Use Cases

### Collect Business Permit Fee

```
POST /api/government/cty/revenues
{
  "payer_id": "biz-001",
  "payer_name": "Mama Mboga Traders",
  "revenue_type": "business_permit",
  "amount_kes": 5000,
  "period": "2025",
  "payment_method": "mpesa",
  "tenant_id": "nairobi_county"
}
```

### Apply for Business Permit

```
POST /api/government/cty/permits
{
  "applicant_id": "biz-001",
  "applicant_name": "Jane Mwangi",
  "business_name": "Mama Mboga Traders",
  "permit_type": "business_permit",
  "location": "Gikomba Market, Stall 45",
  "sub_county": "Makadara",
  "fee_paid_kes": 5000,
  "tenant_id": "nairobi_county"
}
```

### Apply for Cash Transfer Programme

```
POST /api/government/cty/welfare
{
  "applicant_id": "id-12345678",
  "applicant_name": "Mary Wanjiku",
  "id_number": "12345678",
  "programme_type": "elderly_grant",
  "sub_county": "Ruiru",
  "ward": "Gitothua",
  "household_size": 3,
  "monthly_income_kes": 0,
  "tenant_id": "kiambu_county"
}
```

### Report Pothole

```
POST /api/government/cty/tickets
{
  "reporter_id": "citizen-001",
  "reporter_name": "Peter Kamau",
  "reporter_phone": "254700000000",
  "ticket_type": "road_repair",
  "description": "Large pothole on Ngong Road near Junction Mall, 2m diameter",
  "location": "Ngong Road, near Junction Mall",
  "sub_county": "Langata",
  "ward": "Karen",
  "priority": "high",
  "tenant_id": "nairobi_county"
}
```

## Supported Revenue Types

land_rates, business_permit, parking, market_fee, advertisement, building_permit, health_certificate, liquor_licence, billboard, estate_charges

## Supported Welfare Programmes

cash_transfer, food_subsidy, elderly_grant, disability_grant, orphan_support, school_bursary

## Supported Ticket Types

road_repair, drainage, streetlight, water_supply, waste_collection, park_maintenance, sewer_blockage, bridge_repair, bus_shelter

## Error Codes

| Code | Meaning |
|------|---------|
| 422 | Validation error — business rule violated or invalid input |
| 404 | Resource not found |
| 500 | Internal service error |
