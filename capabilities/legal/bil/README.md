# leg_bil — Legal Billing & Time Tracking

Time capture, matter billing, disbursements, invoice approval, client trust accounting.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/bil/health | Health check |
| GET | /api/legal/bil/time-entries | List time entries |
| GET | /api/legal/bil/time-entries/{id} | Get time entry |
| POST | /api/legal/bil/time-entries | Create time entry |
| PUT | /api/legal/bil/time-entries/{id} | Update time entry |
| DELETE | /api/legal/bil/time-entries/{id} | Write off time entry |
| POST | /api/legal/bil/time-entries/{id}/submit | Submit time entry |
| POST | /api/legal/bil/time-entries/{id}/approve | Approve time entry |
| GET | /api/legal/bil/disbursements | List disbursements |
| POST | /api/legal/bil/disbursements | Record disbursement |
| PUT | /api/legal/bil/disbursements/{id} | Update disbursement |
| DELETE | /api/legal/bil/disbursements/{id} | Cancel disbursement |
| GET | /api/legal/bil/invoices | List invoices |
| GET | /api/legal/bil/invoices/{id} | Get invoice |
| POST | /api/legal/bil/invoices | Create invoice |
| PUT | /api/legal/bil/invoices/{id} | Update invoice |
| DELETE | /api/legal/bil/invoices/{id} | Write off invoice |
| POST | /api/legal/bil/invoices/{id}/approve | Approve invoice |
| POST | /api/legal/bil/invoices/{id}/send | Send invoice |
| POST | /api/legal/bil/invoices/{id}/pay | Record payment |
| GET | /api/legal/bil/trust-accounts | List trust accounts |
| POST | /api/legal/bil/trust-accounts | Open trust account |
| POST | /api/legal/bil/trust-accounts/{id}/transactions | Trust transaction |
| GET | /api/legal/bil/trust-accounts/{id}/transactions | List transactions |
| GET | /api/legal/bil/dashboard | Billing dashboard |
| GET | /api/legal/bil/audit | Audit events |

## Service Class

`LegalBillingService` — ABA activity codes, time entry approval workflow, auto-invoice calculation with 16% Kenya VAT, trust account ledger with running balance, attorney rate cards.
