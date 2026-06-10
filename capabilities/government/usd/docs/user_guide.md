# USSD Government Services — User Guide

## Overview

The USSD Government Services capability (gov_usd) enables citizens to access government services from any mobile phone by dialling a USSD code. No internet connection or smartphone required.

## Supported Services

- **Permit Status**: Check business permit, building permit, health certificate validity
- **Tax Balance**: Query outstanding KRA tax balances by PIN
- **ID Verification**: Verify national ID, passport, alien card against IPRS
- **Certificate Requests**: Apply for good conduct, tax compliance, birth/death/marriage certificates

## USSD Service Codes

| Code | Service |
|------|---------|
| *384# | Permit Status |
| *385# | Tax Balance |
| *386# | ID Verification |
| *400# | Certificate Requests |

## Use Cases

### Permit Status Enquiry
1. Dial `*384#`
2. Select `1. Check Permit`
3. Enter permit number
4. Receive status SMS

### Tax Balance Enquiry
1. Dial `*385#`
2. Enter KRA PIN
3. Select tax type
4. View outstanding balance

### ID Verification
1. Dial `*386#`
2. Enter ID number
3. Confirm with OTP
4. Receive verification result

### Certificate Request
1. Dial `*400#`
2. Select certificate type
3. Enter applicant details
4. Receive reference number via SMS
5. Track request status by reference number

## API Reference

### Sessions

```
POST /api/government/usd/sessions
{
  "msisdn": "254700000000",
  "service_code": "*384#",
  "tenant_id": "nairobi_county"
}
```

### Permit Enquiry

```
POST /api/government/usd/permits/enquiries
{
  "msisdn": "254700000000",
  "permit_number": "BP-2024-001234",
  "permit_type": "business_permit",
  "tenant_id": "nairobi_county"
}
```

### Tax Balance

```
POST /api/government/usd/tax/enquiries
{
  "msisdn": "254700000000",
  "tax_pin": "A000000000A",
  "tax_type": "income_tax",
  "tenant_id": "kra"
}
```

### ID Verification

```
POST /api/government/usd/id-verifications
{
  "msisdn": "254700000000",
  "id_number": "12345678",
  "id_type": "national_id",
  "full_name": "John Doe"
}
```

### Certificate Request

```
POST /api/government/usd/certificates
{
  "msisdn": "254700000000",
  "certificate_type": "good_conduct",
  "applicant_id": "12345678",
  "applicant_name": "John Doe",
  "tenant_id": "dci_kenya"
}
```

## Error Codes

| Code | Meaning |
|------|---------|
| 422 | Validation error — invalid input data |
| 404 | Resource not found |
| 500 | Internal service error |
