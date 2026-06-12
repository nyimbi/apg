# Supply Planning (scm_spl)

MRP-II, safety stock optimisation, replenishment rules, capacity planning, supply/demand balancing, EOQ analysis, ABC-XYZ segmentation, scenario planning, forecast bias detection, supplier performance tracking, and inventory turnover analytics.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/spl/health | Health check |
| GET | /api/scm/spl/describe | Capability contract |
| GET | /api/scm/spl/audit-events | Audit events |
| **Demand Forecasting** | | |
| GET | /api/scm/spl/demand-forecasts | List forecasts |
| POST | /api/scm/spl/demand-forecasts | Create forecast |
| POST | /api/scm/spl/demand-forecasts/bulk | Bulk-create forecasts |
| GET | /api/scm/spl/demand-forecasts/{id} | Get forecast |
| PATCH | /api/scm/spl/demand-forecasts/{id}/actual | Record actual demand |
| DELETE | /api/scm/spl/demand-forecasts/{id} | Deactivate forecast |
| **MRP-II** | | |
| GET | /api/scm/spl/mrp-runs | List MRP runs |
| POST | /api/scm/spl/mrp-runs | Run MRP-II |
| GET | /api/scm/spl/mrp-runs/{id} | Get MRP run |
| **Planned Orders** | | |
| POST | /api/scm/spl/planned-orders/{id}/firm | Firm a planned order |
| POST | /api/scm/spl/planned-orders/{id}/release | Release a planned order |
| **Safety Stock** | | |
| GET | /api/scm/spl/safety-stocks | List safety stocks |
| POST | /api/scm/spl/safety-stocks | Calculate safety stock |
| **Replenishment Rules** | | |
| GET | /api/scm/spl/replenishment-rules | List rules |
| POST | /api/scm/spl/replenishment-rules | Create rule |
| GET | /api/scm/spl/replenishment-rules/{id} | Get rule |
| PUT | /api/scm/spl/replenishment-rules/{id} | Update rule |
| DELETE | /api/scm/spl/replenishment-rules/{id} | Deactivate rule |
| POST | /api/scm/spl/replenishment-rules/evaluate | Evaluate triggers |
| **Capacity Planning** | | |
| GET | /api/scm/spl/capacity-plans | List capacity plans |
| POST | /api/scm/spl/capacity-plans | Create capacity plan |
| **Supply/Demand Balance** | | |
| GET | /api/scm/spl/supply-demand-balances | List balances |
| POST | /api/scm/spl/supply-demand-balances | Create balance |
| **Supply Exceptions** | | |
| POST | /api/scm/spl/supply-exceptions | Raise exception |
| POST | /api/scm/spl/supply-exceptions/{id}/resolve | Resolve exception |
| **EOQ Analysis** | | |
| POST | /api/scm/spl/eoq | Calculate EOQ and total cost |
| **ABC-XYZ Segmentation** | | |
| POST | /api/scm/spl/segment-skus | Segment SKU portfolio |
| **Scenario Planning** | | |
| POST | /api/scm/spl/scenarios | Create what-if scenario |
| POST | /api/scm/spl/scenarios/{id}/run-mrp | Run MRP over scenario |
| **Supplier Performance** | | |
| POST | /api/scm/spl/supplier-performance | Record delivery performance |
| GET | /api/scm/spl/supplier-performance/{supplier_id}/{sku} | Get supplier stats |
| **Forecast Bias** | | |
| GET | /api/scm/spl/forecast-bias/{sku} | Detect forecast bias |
| **Inventory Analytics** | | |
| POST | /api/scm/spl/analytics/inventory-turnover | Turnover and DOH analysis |
| GET | /api/scm/spl/analytics/dashboard | Planning KPI dashboard |

---

## World-Class Enhancements (v2.0)

- **I1.** Supply Planning (scm_spl) — World-Class Improvement Roadmap
- **I2.** Probabilistic Demand Forecasting with Prediction Intervals
- **I3.** Dynamic Safety Stock with Demand Sensing
- **I4.** Multi-Echelon Inventory Optimisation (MEIO)
- **I5.** Supplier Lead-Time Variability Tracking
- **I6.** Economic Order Quantity (EOQ) and Total Cost Optimisation
- **I7.** Constrained MRP with Capacity Feedback Loop
- **I8.** Vendor-Managed Inventory (VMI) and Collaborative Replenishment
- **I9.** ABC-XYZ Segmentation Engine
- **I10.** Perishability and Shelf-Life Constraints
- **I11.** Scenario Planner and What-If Simulation
- **I12.** Rolling Wave Planning with Frozen Zone Enforcement
- **I13.** Forecast Bias Detection and Auto-Correction
- **I14.** Supply Chain Risk Scoring
- **I15.** Planned Order Firm-and-Release Workflow

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
