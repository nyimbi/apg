# Premium & Billing (ins_prm)

Premium calculation, instalment management, collections, reconciliation, and refunds.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/prm/health | Health check |
| GET | /api/insurance/prm/describe | Capability description |
| GET | /api/insurance/prm/schedules | List schedules |
| POST | /api/insurance/prm/schedules | Create schedule |
| GET | /api/insurance/prm/schedules/{id} | Get schedule |
| DELETE | /api/insurance/prm/schedules/{id} | Cancel schedule |
| GET | /api/insurance/prm/instalments | List instalments |
| GET | /api/insurance/prm/instalments/overdue | Overdue instalments |
| POST | /api/insurance/prm/instalments/{id}/collect | Record collection |
| POST | /api/insurance/prm/refunds | Process refund |
| POST | /api/insurance/prm/reconcile | Period reconciliation |
| POST | /api/insurance/prm/calculate | Premium calculation |
| GET | /api/insurance/prm/summary | Billing summary |
| GET | /api/insurance/prm/audit | Audit trail |
