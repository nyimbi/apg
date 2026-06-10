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
