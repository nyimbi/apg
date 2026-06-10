# Supplier Relationship Management (scm_srm)

Supplier scorecard, risk assessment, collaboration portal, performance reviews, preferred supplier status.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/srm/health | Health check |
| GET | /api/scm/srm/describe | Capability contract |
| GET | /api/scm/srm/suppliers | List suppliers |
| POST | /api/scm/srm/suppliers | Create supplier |
| GET | /api/scm/srm/suppliers/{id} | Get supplier |
| PUT | /api/scm/srm/suppliers/{id} | Update supplier |
| DELETE | /api/scm/srm/suppliers/{id} | Deactivate supplier |
| POST | /api/scm/srm/suppliers/{id}/approve | Approve supplier |
| POST | /api/scm/srm/suppliers/{id}/suspend | Suspend supplier |
| POST | /api/scm/srm/suppliers/{id}/preferred | Set preferred status |
| GET | /api/scm/srm/scorecards | List scorecards |
| POST | /api/scm/srm/scorecards | Create scorecard |
| GET | /api/scm/srm/risk-assessments | List risks |
| POST | /api/scm/srm/risk-assessments | Create risk assessment |
| POST | /api/scm/srm/risk-assessments/{id}/review | Review risk |
| GET | /api/scm/srm/messages | List collaboration messages |
| POST | /api/scm/srm/messages | Send message |
| GET | /api/scm/srm/performance-reviews | List reviews |
| POST | /api/scm/srm/performance-reviews | Create review |
| GET | /api/scm/srm/analytics | Supplier analytics |
| GET | /api/scm/srm/audit-events | Audit events |
