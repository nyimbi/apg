# Micro-Insurance Platform (ins_mic)

Mobile-first product design, USSD enrolment, airtime premium deduction, instant payout via mobile money.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/mic/health | Health check |
| GET | /api/insurance/mic/describe | Capability description |
| GET | /api/insurance/mic/products | List products |
| POST | /api/insurance/mic/products | Create product |
| GET | /api/insurance/mic/products/{id} | Get product |
| PUT | /api/insurance/mic/products/{id} | Update product |
| DELETE | /api/insurance/mic/products/{id} | Deactivate product |
| POST | /api/insurance/mic/ussd | Process USSD session |
| POST | /api/insurance/mic/enrolments | Enrol subscriber |
| GET | /api/insurance/mic/enrolments | List enrolments |
| GET | /api/insurance/mic/enrolments/{id} | Get enrolment |
| PUT | /api/insurance/mic/enrolments/{id} | Update enrolment |
| DELETE | /api/insurance/mic/enrolments/{id} | Cancel enrolment |
| POST | /api/insurance/mic/enrolments/{id}/renew | Renew enrolment |
| POST | /api/insurance/mic/airtime/deduct | Deduct airtime premium |
| POST | /api/insurance/mic/claims | Register claim |
| GET | /api/insurance/mic/claims | List claims |
| POST | /api/insurance/mic/claims/{id}/payout | Mobile money payout |
| GET | /api/insurance/mic/summary | Platform summary |
| GET | /api/insurance/mic/audit | Audit trail |
