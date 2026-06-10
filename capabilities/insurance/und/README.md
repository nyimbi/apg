# Underwriting Engine (ins_und)

Risk assessment, rating engine, capacity management, reinsurance treaties, underwriting rules.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/und/health | Service health check |
| GET | /api/insurance/und/describe | Capability description |
| GET | /api/insurance/und/submissions | List submissions |
| POST | /api/insurance/und/submissions | Submit risk |
| GET | /api/insurance/und/submissions/{id} | Get submission |
| POST | /api/insurance/und/submissions/{id}/assess | Run risk assessment |
| POST | /api/insurance/und/submissions/{id}/rate | Rate risk |
| POST | /api/insurance/und/capacity/check | Check capacity |
| GET | /api/insurance/und/treaties | List treaties |
| POST | /api/insurance/und/treaties | Create treaty |
| GET | /api/insurance/und/rules | List rules |
| POST | /api/insurance/und/rules | Create rule |
| DELETE | /api/insurance/und/rules/{id} | Delete rule |
| GET | /api/insurance/und/summary | Underwriting summary |
| GET | /api/insurance/und/audit | Audit trail |
