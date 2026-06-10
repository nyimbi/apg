# Supply Planning User Guide

## Overview

`scm_spl` provides MRP-II driven supply planning: demand forecasting, safety stock calculation, replenishment rule evaluation, capacity planning and supply/demand balance tracking.

## Key Use Cases

- **Demand forecasting**: Create statistical, ML, or manual forecasts per SKU/period; reconcile against actuals to track accuracy.
- **MRP-II runs**: Execute planning runs over a configurable horizon; auto-generate planned orders incorporating safety stock.
- **Safety stock**: Calculate optimal safety stock using the Z-score method (service-level × demand std-dev × √lead-time).
- **Replenishment rules**: Define min-max, reorder-point, or periodic-review rules; evaluate triggers against live stock levels.
- **Capacity planning**: Model warehouse/line/supplier capacity vs planned demand; flag overloaded resources.
- **Supply/demand balance**: Compute closing stock and flag surplus or shortage by SKU/period.

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

## Safety Stock Formula

`SS = Z(SL) × σ_demand × √(lead_time_days)`

Where Z is the service-level Z-score (e.g. 1.645 for 95%).
