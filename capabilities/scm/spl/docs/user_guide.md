# Supply Planning User Guide

© 2025 Datacraft | Author: Nyimbi Odero

---

## Overview

`scm_spl` is the MRP-II–driven supply planning capability of APG. It covers the full planning cycle from demand forecasting through planned order release, with advanced analytics for EOQ optimisation, ABC-XYZ segmentation, scenario simulation, forecast bias detection, and supplier performance management.

---

## Key Use Cases

| Capability | Description |
|---|---|
| **Demand forecasting** | Statistical, ML, or manual forecasts per SKU/period; reconcile against actuals to track accuracy. |
| **MRP-II runs** | Execute planning runs over a configurable horizon; auto-generate planned orders incorporating safety stock. |
| **Safety stock** | Z-score method: `SS = Z(SL) × σ_demand × √(lead_time_days)`. Supports manual override. |
| **Replenishment rules** | Min-max, reorder-point, or periodic-review rules; evaluate triggers against live stock levels. |
| **Capacity planning** | Model warehouse/line/supplier capacity vs planned demand; flag overloaded resources. |
| **Supply/demand balance** | Compute closing stock and flag surplus or shortage by SKU/period. |
| **EOQ analysis** | Wilson EOQ formula with quantity-discount break-point evaluation. |
| **ABC-XYZ segmentation** | Classify SKU portfolio by value (ABC) and demand volatility (XYZ). |
| **Scenario planning** | Fork the baseline plan with demand shocks or supply disruptions; run isolated MRP. |
| **Supplier performance** | Track OTD, fill rate, and lead-time statistics per supplier/SKU; auto-raise exceptions on σ breaches. |
| **Forecast bias** | Tracking signal (CFE/MAD) per SKU; auto-flag and suggest correction factor. |
| **Planned order lifecycle** | planned → firmed → released → confirmed → received. |
| **Inventory turnover** | Turnover ratio and days-on-hand per SKU; breach alerts vs replenishment rules. |

---

## Workflow: Full Planning Cycle

```
1. create_demand_forecast(sku, period, qty, method)
2. calculate_safety_stock(sku, lead_time_days, service_level_pct, demand_std_dev)
3. create_replenishment_rule(sku, rule_type, reorder_point, order_quantity)
4. run_mrp(run_name, horizon_weeks)          # generates planned orders
5. evaluate_replenishment_triggers(current_stocks)
6. firm_planned_order(order_id, firmed_by)
7. release_planned_order(order_id, released_by, target_system="procurement")
```

---

## API Reference

### Run MRP-II

```
POST /api/scm/spl/mrp-runs
{
  "tenant_id": "acme",
  "run_name": "June 2026 Plan",
  "horizon_weeks": 8,
  "include_safety_stock": true
}
```

### Calculate Safety Stock

```
POST /api/scm/spl/safety-stocks
{
  "tenant_id": "acme",
  "sku": "PROD-A",
  "lead_time_days": 14,
  "target_service_level_pct": 95,
  "demand_std_dev": 12.5
}
```

### Evaluate Replenishment Triggers

```
POST /api/scm/spl/replenishment-rules/evaluate
{
  "tenant_id": "acme",
  "current_stocks": {"PROD-A": 45, "PROD-B": 8}
}
```

### Calculate EOQ

```
POST /api/scm/spl/eoq
{
  "tenant_id": "acme",
  "sku": "PROD-A",
  "annual_demand": 5200,
  "ordering_cost": 75.00,
  "holding_cost_rate": 0.20,
  "unit_cost": 12.50,
  "quantity_breaks": [
    {"min_qty": 500, "unit_cost": 11.80},
    {"min_qty": 1000, "unit_cost": 11.20}
  ]
}
```

**Response excerpt:**
```json
{
  "classic_eoq": 394.5,
  "best_policy": {
    "policy": "quantity_break",
    "order_quantity": 500,
    "unit_cost": 11.80,
    "total_annual_cost": 63284.50
  }
}
```

### Segment SKUs (ABC-XYZ)

```
POST /api/scm/spl/segment-skus
{
  "tenant_id": "acme",
  "sku_data": [
    {"sku": "PROD-A", "annual_value": 120000, "demand_history": [450, 480, 420, 510, 460]},
    {"sku": "PROD-B", "annual_value": 8000,  "demand_history": [20, 80, 5, 150, 10]}
  ],
  "abc_thresholds": [0.80, 0.95],
  "cv_thresholds": [0.5, 1.0]
}
```

**Response excerpt:**
```json
{
  "by_segment": {"AX": 1, "CZ": 1},
  "segments": [
    {"sku": "PROD-A", "segment": "AX", "demand_cv": 0.072},
    {"sku": "PROD-B", "segment": "CZ", "demand_cv": 1.243}
  ]
}
```

### Scenario Planning

```
# Create scenario
POST /api/scm/spl/scenarios
{
  "tenant_id": "acme",
  "scenario_name": "Demand +20% Q3",
  "demand_adjustment_pct": 20.0,
  "lead_time_adjustment_days": 0
}

# Run MRP against it (isolated from baseline)
POST /api/scm/spl/scenarios/{scenario_id}/run-mrp
{"tenant_id": "acme", "horizon_weeks": 8}
```

**Response includes delta_pct** showing how much total required quantity changes vs baseline.

### Record Supplier Performance

```
POST /api/scm/spl/supplier-performance
{
  "tenant_id": "acme",
  "supplier_id": "SUP-001",
  "sku": "PROD-A",
  "promised_lead_time_days": 14,
  "actual_lead_time_days": 19,
  "promised_quantity": 500,
  "delivered_quantity": 490,
  "period": "2026-06"
}
```

Auto-raises a `lead_time_breach` supply exception if `actual_lead_time_days > mean + 2σ`.

### Detect Forecast Bias

```
GET /api/scm/spl/forecast-bias/PROD-A?tenant_id=acme&tracking_signal_threshold=4.0
```

Returns `biased: true/false`, tracking signal value, and `suggested_correction_factor`.

### Firm and Release a Planned Order

```
POST /api/scm/spl/planned-orders/{order_id}/firm
{"tenant_id": "acme", "firmed_by": "planner@acme.com"}

POST /api/scm/spl/planned-orders/{order_id}/release
{"tenant_id": "acme", "released_by": "planner@acme.com", "target_system": "procurement"}
```

### Inventory Turnover Analytics

```
POST /api/scm/spl/analytics/inventory-turnover
{
  "tenant_id": "acme",
  "sku_cogs": {"PROD-A": 62400, "PROD-B": 9600},
  "sku_avg_inventory": {"PROD-A": 5200, "PROD-B": 3200}
}
```

Returns per-SKU `turnover_ratio`, `days_on_hand`, `doh_breach` flag, and aggregate portfolio metrics.

---

## Safety Stock Formula

```
SS = Z(SL) × σ_demand × √(lead_time_days)
```

| Service Level | Z-Score |
|---|---|
| 80% | 0.84 |
| 90% | 1.28 |
| 95% | 1.645 |
| 98% | 2.05 |
| 99% | 2.33 |

---

## EOQ Formula

```
Q* = √(2 × D × S / (h × C))
```

Where:
- `D` = annual demand (units)
- `S` = ordering cost per order
- `h` = holding cost rate (fraction of unit cost)
- `C` = unit cost

Total annual cost = ordering cost + holding cost + purchase cost. When quantity discounts exist, evaluate each break-point quantity and select the policy with the lowest total cost.

---

## Forecast Bias (Tracking Signal)

```
Tracking Signal = CFE / MAD
```

Where:
- `CFE` = Cumulative Forecast Error = Σ(Forecast − Actual)
- `MAD` = Mean Absolute Deviation = mean(|Forecast − Actual|)

|TS| > 4 typically indicates statistically significant bias. The service auto-raises a `forecast_bias` exception and returns a Trigg's correction factor.

---

## Planned Order Lifecycle

```
planned → firmed → released → confirmed → received
```

- **planned**: Generated by MRP run; not yet committed.
- **firmed**: Procurement intent confirmed; visible in VMI/supplier feeds.
- **released**: Sent to downstream procurement (scm_po) or production system.
- **confirmed**: Supplier/production has acknowledged.
- **received**: Goods or work order completed.

---

## Composability

`scm_spl` composes with:

| Capability | Trigger |
|---|---|
| `scm_po` | Released planned orders → purchase orders |
| `scm_inv` | Safety stock levels → inventory policy enforcement |
| `scm_wms` | Capacity plans → warehouse slotting and receiving |
| `intel_alerts` | Supply exceptions → alert routing |
| `fin_cost` | EOQ and holding cost → cost accounting |
