# Feature Flags (fflag)

Runtime feature toggles, percentage rollout, A/B experiment assignment, per-tenant targeting, and full audit trail.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fflag/health | Service health |
| GET | /api/fflag/flags | List flags |
| POST | /api/fflag/flags | Create flag |
| GET | /api/fflag/flags/{key} | Get flag |
| PUT | /api/fflag/flags/{key} | Update flag |
| DELETE | /api/fflag/flags/{key} | Delete flag |
| POST | /api/fflag/flags/{key}/enable | Enable flag |
| POST | /api/fflag/flags/{key}/disable | Disable flag |
| POST | /api/fflag/flags/{key}/rollout | Set rollout % |
| GET | /api/fflag/evaluate/{key} | Evaluate flag for user |
| POST | /api/fflag/evaluate/batch | Batch evaluate flags |
| POST | /api/fflag/evaluate/all | Evaluate all flags for user |
| POST | /api/fflag/overrides | Set user override |
| DELETE | /api/fflag/overrides | Clear user override |
| GET | /api/fflag/overrides | List overrides |
| POST | /api/fflag/experiments | Create experiment |
| GET | /api/fflag/experiments | List experiments |
| GET | /api/fflag/experiments/{id} | Get experiment |
| POST | /api/fflag/experiments/{id}/start | Start experiment |
| POST | /api/fflag/experiments/{id}/stop | Stop experiment |
| GET | /api/fflag/experiments/{id}/results | Experiment results |
| POST | /api/fflag/experiments/{id}/assign | Assign variant |
| GET | /api/fflag/statistics | Flag statistics |
| GET | /api/fflag/audit | Audit trail |
| GET | /api/fflag/flags/{key}/history | Flag change history |
