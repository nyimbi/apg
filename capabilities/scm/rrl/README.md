# Returns & Reverse Logistics (scm_rrl)

RMA processing, refurbishment workflow, disposal management, credit notes, reverse shipment tracking.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/rrl/health | Health check |
| GET | /api/scm/rrl/describe | Capability contract |
| GET | /api/scm/rrl/rmas | List RMAs |
| POST | /api/scm/rrl/rmas | Create RMA |
| GET | /api/scm/rrl/rmas/{id} | Get RMA |
| PUT | /api/scm/rrl/rmas/{id} | Update RMA |
| DELETE | /api/scm/rrl/rmas/{id} | Close RMA |
| POST | /api/scm/rrl/rmas/{id}/approve | Approve RMA |
| POST | /api/scm/rrl/rmas/{id}/reject | Reject RMA |
| POST | /api/scm/rrl/rmas/{id}/receive | Receive returned goods |
| POST | /api/scm/rrl/rmas/{id}/resolve | Resolve RMA |
| GET | /api/scm/rrl/refurbishments | List refurbishments |
| POST | /api/scm/rrl/refurbishments | Create refurbishment |
| POST | /api/scm/rrl/refurbishments/{id}/complete | Complete refurbishment |
| GET | /api/scm/rrl/disposals | List disposals |
| POST | /api/scm/rrl/disposals | Create disposal |
| GET | /api/scm/rrl/credit-notes | List credit notes |
| POST | /api/scm/rrl/credit-notes | Issue credit note |
| GET | /api/scm/rrl/analytics | Returns analytics |
| GET | /api/scm/rrl/audit-events | Audit events |
