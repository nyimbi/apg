# USSD Government Services (gov_usd)

USSD-based government services: permit status, tax balance inquiry, ID verification, certificate requests.

## Overview

Provides a complete USSD gateway for citizen-facing government services, accessible from any mobile phone without internet. Supports session management, OTP authentication, payment references, and SMS confirmations.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/government/usd/health | Service health check |
| GET | /api/government/usd/dashboard | Dashboard metrics |
| GET | /api/government/usd/sessions | List USSD sessions |
| GET | /api/government/usd/sessions/{id} | Get session |
| POST | /api/government/usd/sessions | Create session |
| PUT | /api/government/usd/sessions/{id} | Advance session |
| DELETE | /api/government/usd/sessions/{id} | Delete session |
| POST | /api/government/usd/permits/enquiries | Enquire permit status |
| GET | /api/government/usd/permits/enquiries | List permit enquiries |
| POST | /api/government/usd/tax/enquiries | Enquire tax balance |
| GET | /api/government/usd/tax/enquiries | List tax enquiries |
| POST | /api/government/usd/id-verifications | Verify ID |
| GET | /api/government/usd/id-verifications | List verifications |
| GET | /api/government/usd/certificates | List certificate requests |
| GET | /api/government/usd/certificates/{id} | Get certificate request |
| POST | /api/government/usd/certificates | Submit certificate request |
| PUT | /api/government/usd/certificates/{id} | Update certificate request |
| DELETE | /api/government/usd/certificates/{id} | Delete certificate request |
| GET | /api/government/usd/audit-events | List audit events |
