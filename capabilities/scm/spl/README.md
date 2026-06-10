# Supply Planning (scm_spl)

MRP-II, safety stock optimisation, replenishment rules, capacity planning, supply/demand balancing.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/spl/health | Health check |
| GET | /api/scm/spl/describe | Capability contract |
| GET | /api/scm/spl/demand-forecasts | List forecasts |
| POST | /api/scm/spl/demand-forecasts | Create forecast |
| GET | /api/scm/spl/demand-forecasts/{id} | Get forecast |
| DELETE | /api/scm/spl/demand-forecasts/{id} | Deactivate forecast |
| GET | /api/scm/spl/mrp-runs | List MRP runs |
| POST | /api/scm/spl/mrp-runs | Run MRP-II |
| GET | /api/scm/spl/mrp-runs/{id} | Get MRP run |
| GET | /api/scm/spl/safety-stocks | List safety stocks |
| POST | /api/scm/spl/safety-stocks | Calculate safety stock |
| GET | /api/scm/spl/replenishment-rules | List rules |
| POST | /api/scm/spl/replenishment-rules | Create rule |
| GET | /api/scm/spl/replenishment-rules/{id} | Get rule |
| PUT | /api/scm/spl/replenishment-rules/{id} | Update rule |
| DELETE | /api/scm/spl/replenishment-rules/{id} | Deactivate rule |
| POST | /api/scm/spl/replenishment-rules/evaluate | Evaluate triggers |
| GET | /api/scm/spl/capacity-plans | List capacity plans |
| POST | /api/scm/spl/capacity-plans | Create capacity plan |
| GET | /api/scm/spl/supply-demand-balances | List balances |
| POST | /api/scm/spl/supply-demand-balances | Create balance |
| GET | /api/scm/spl/analytics/dashboard | Planning dashboard |
| GET | /api/scm/spl/audit-events | Audit events |
