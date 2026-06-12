# Agricultural Supply Chain (agr_sup)

Farm-to-buyer traceability, input procurement, cold chain management, export documentation.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/sup/health | Health check |
| GET | /api/agriculture/sup/batches | List batches |
| POST | /api/agriculture/sup/batches | Create batch |
| GET | /api/agriculture/sup/batches/{id} | Get batch |
| PUT | /api/agriculture/sup/batches/{id} | Update batch |
| DELETE | /api/agriculture/sup/batches/{id} | Delete batch |
| GET | /api/agriculture/sup/batches/{id}/trace | Full trace |
| GET | /api/agriculture/sup/batches/{id}/export-readiness | Export check |
| GET | /api/agriculture/sup/procurement | List orders |
| POST | /api/agriculture/sup/procurement | Create order |
| PUT | /api/agriculture/sup/procurement/{id} | Update order |
| DELETE | /api/agriculture/sup/procurement/{id} | Delete order |
| GET | /api/agriculture/sup/cold-chain | Cold chain logs |
| POST | /api/agriculture/sup/cold-chain | Log temperature |
| GET | /api/agriculture/sup/cold-chain/{batch_id}/summary | Integrity summary |
| GET | /api/agriculture/sup/export-docs | Export documents |
| POST | /api/agriculture/sup/export-docs | Add document |
| DELETE | /api/agriculture/sup/export-docs/{id} | Delete document |
| GET | /api/agriculture/sup/audit | Audit log |

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Blockchain-Anchored Immutable Provenance Hashes** [Security]
- **I2. Dynamic Cold Chain SLA Alerting with Escalation Chains** [Feature]
- **I3. Supplier Risk Scoring with Weighted KPIs** [AI/ML]
- **I4. Demand Forecasting Integration for Input Procurement** [AI/ML]
- **I5. GS1 EPCIS Event Stream Export** [Integration]
- **I6. Weighted Quality Score with Reject Reason Tracking** [Feature]
- **I7. Multi-Modal Transport Leg Tracking** [Feature]
- **I8. Carbon Footprint Calculation per Batch** [Compliance]
- **I9. Buyer Portal Token-Scoped Trace Access** [Security]
- **I10. Input Recall Management** [Compliance]
- **I11. Dynamic Pricing Engine with Market Feed Integration** [Feature]
- **I12. Batch Splitting and Merging for Aggregation Points** [Feature]
- **I13. Compliance Checklist Engine with Regulatory Ruleset Versioning** [Compliance]
- **I14. Offline-First IoT Sensor Ingestion with Deduplication** [Integration]
- **I15. Predictive Shelf-Life Estimation** [AI/ML]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
