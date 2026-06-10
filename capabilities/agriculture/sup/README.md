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
