# Supplier Relationship Management (scm_srm)

Supplier scorecard, risk assessment, collaboration portal, performance reviews, preferred supplier status, ESG scoring, contract lifecycle, development plans, segmentation, and benchmarking.

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
| POST | /api/scm/srm/suppliers/bulk | Bulk-create suppliers |
| GET | /api/scm/srm/suppliers/segment | Segment portfolio |
| GET | /api/scm/srm/suppliers/concentration-risk | Concentration risk report |
| GET | /api/scm/srm/suppliers/{id}/benchmark | Benchmark vs peers |
| GET | /api/scm/srm/scorecards | List scorecards |
| POST | /api/scm/srm/scorecards | Create scorecard |
| GET | /api/scm/srm/scorecards/{supplier_id}/trend | Score trend series |
| GET | /api/scm/srm/risk-assessments | List risks |
| POST | /api/scm/srm/risk-assessments | Create risk assessment |
| POST | /api/scm/srm/risk-assessments/{id}/review | Review risk |
| GET | /api/scm/srm/risk-heatmap | Portfolio risk heatmap |
| GET | /api/scm/srm/messages | List collaboration messages |
| POST | /api/scm/srm/messages | Send message |
| GET | /api/scm/srm/performance-reviews | List reviews |
| POST | /api/scm/srm/performance-reviews | Create review |
| GET | /api/scm/srm/certifications | List certifications |
| POST | /api/scm/srm/certifications | Add certification |
| GET | /api/scm/srm/esg-scores | List ESG scores |
| POST | /api/scm/srm/esg-scores | Record ESG score |
| GET | /api/scm/srm/contracts | List contracts |
| POST | /api/scm/srm/contracts | Register contract |
| GET | /api/scm/srm/development-plans | List development plans |
| POST | /api/scm/srm/development-plans | Create development plan |
| PUT | /api/scm/srm/development-plans/{id}/progress | Update plan progress |
| GET | /api/scm/srm/escalations | List escalations |
| POST | /api/scm/srm/escalations | Raise escalation |
| POST | /api/scm/srm/escalations/{id}/resolve | Resolve escalation |
| POST | /api/scm/srm/onboarding | Start onboarding workflow |
| PUT | /api/scm/srm/onboarding/{id}/items/{idx} | Complete onboarding item |
| GET | /api/scm/srm/analytics | Supplier analytics |
| GET | /api/scm/srm/audit-events | Audit events |

## Key Constants

| Constant | Values |
|----------|--------|
| SUPPLIER_CATEGORIES | raw_material, packaging, services, technology, logistics, equipment, consumables |
| RISK_LEVELS | low, medium, high, critical |
| RISK_CATEGORIES | financial, geopolitical, operational, compliance, esg, concentration |
| SUPPLIER_STATUSES | active, pending_approval, probation, suspended, blacklisted, inactive |
| MESSAGE_TYPES | general, forecast_share, po_update, complaint, escalation, nda, performance_review |
| SEGMENT_STRATEGIES | risk_score, spend_category, geography |
