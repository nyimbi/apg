# Cooperative Management (agr_coo)

Member registry, share management, pooled inputs, dividend allocation, annual returns.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/coo/health | Health check |
| GET | /api/agriculture/coo/coops | List cooperatives |
| POST | /api/agriculture/coo/coops | Create cooperative |
| GET | /api/agriculture/coo/coops/{id} | Get cooperative |
| PUT | /api/agriculture/coo/coops/{id} | Update cooperative |
| GET | /api/agriculture/coo/coops/{id}/summary | Summary |
| GET | /api/agriculture/coo/members | List members |
| POST | /api/agriculture/coo/members | Add member |
| GET | /api/agriculture/coo/members/{id} | Get member |
| PUT | /api/agriculture/coo/members/{id} | Update member |
| GET | /api/agriculture/coo/members/{id}/statement | Statement |
| POST | /api/agriculture/coo/members/transfer-shares | Transfer shares |
| GET | /api/agriculture/coo/input-pools | List input pools |
| POST | /api/agriculture/coo/input-pools | Create pool |
| POST | /api/agriculture/coo/input-pools/{id}/allocate | Allocate |
| GET | /api/agriculture/coo/dividends | Dividend history |
| POST | /api/agriculture/coo/dividends | Allocate dividends |
| GET | /api/agriculture/coo/annual-returns | Annual returns |
| POST | /api/agriculture/coo/annual-returns | File return |
| GET | /api/agriculture/coo/audit | Audit log |
