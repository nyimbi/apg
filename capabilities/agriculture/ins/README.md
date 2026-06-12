# Crop Insurance (agr_ins)

Parametric index products, satellite verification, weather trigger claims, premium calculation.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/ins/health | Health check |
| GET | /api/agriculture/ins/products | List products |
| POST | /api/agriculture/ins/products | Create product |
| GET | /api/agriculture/ins/products/{id} | Get product |
| PUT | /api/agriculture/ins/products/{id} | Update product |
| DELETE | /api/agriculture/ins/products/{id} | Delete product |
| GET | /api/agriculture/ins/premium-calc | Calculate premium |
| GET | /api/agriculture/ins/policies | List policies |
| POST | /api/agriculture/ins/policies | Issue policy |
| GET | /api/agriculture/ins/policies/{id} | Get policy |
| PUT | /api/agriculture/ins/policies/{id} | Update policy |
| POST | /api/agriculture/ins/policies/{id}/activate | Activate policy |
| DELETE | /api/agriculture/ins/policies/{id} | Delete policy |
| GET | /api/agriculture/ins/claims | List claims |
| POST | /api/agriculture/ins/claims | Submit claim |
| GET | /api/agriculture/ins/claims/{id} | Get claim |
| PUT | /api/agriculture/ins/claims/{id} | Update claim |
| POST | /api/agriculture/ins/claims/{id}/verify | Verify trigger |
| GET | /api/agriculture/ins/portfolio | Portfolio stats |
| GET | /api/agriculture/ins/coverage/{farmer_id} | Farmer coverage |
| GET | /api/agriculture/ins/audit | Audit log |

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Multi-Index Basket Triggers** [Feature]
- **I2. Actuarial Premium Pricing with Historical Loss Distribution** [AI/ML]
- **I3. Satellite NDVI Zonal Statistics Ingestion** [Integration]
- **I4. Mobile Money Payout Disbursement (M-Pesa / Airtel Money)** [Integration]
- **I5. Reinsurance Treaty Cession Tracking** [Compliance]
- **I6. Season-Level Portfolio Loss Alerting** [Feature]
- **I7. Basis Risk Score per Policy** [AI/ML]
- **I8. Regulatory Compliance Certificate Generation** [Compliance]
- **I9. Group / Cooperative Policy Bundling** [Feature]
- **I10. Churn Prediction and Renewal Propensity Scoring** [AI/ML]
- **I11. Drought Early Warning Integration** [Integration]
- **I12. Audit-Grade Immutable Event Log with Sequence Numbers** [Security]
- **I13. Multi-Currency Premium and Payout Support** [Feature]
- **I14. Fraud Detection via Claim Velocity Analysis** [Security]
- **I15. Carbon Credit Co-issuance for Climate-Smart Practices** [Feature]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
