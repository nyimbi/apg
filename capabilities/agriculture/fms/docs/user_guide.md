# Farm Management System — User Guide

## Overview

agr_fms provides a complete farm management backbone: parcel registry, input tracking,
labour scheduling, cost aggregation, and a farm diary for operational notes.

## Key Use Cases

- **Parcel Registry**: Register farm parcels with GPS coordinates, soil type, area, and ownership.
- **Input Recording**: Track seeds, fertilizers, pesticides, and fuel applied per parcel/crop with
  costs automatically computed (quantity × unit cost).
- **Labour Scheduling**: Plan and track field work tasks with worker counts, daily rates, and
  completion status. Recalculates cost when actual worker count differs from planned.
- **Cost Tracking**: Aggregate input and labour costs per parcel with per-hectare breakdown.
- **Farm Diary**: Log observations, events, and decisions with tags and image attachments.

## Example Workflows

### Register a Parcel
```
POST /api/agriculture/fms/parcels
{"name": "Block A", "area_ha": 5.0, "soil_type": "loam", "location_lat": -0.4, "location_lng": 36.9}
```

### Record Fertiliser Application
```
POST /api/agriculture/fms/inputs
{
  "farm_parcel_id": "par-abc",
  "category": "fertilizer",
  "product_name": "CAN 26%N",
  "quantity": 200,
  "unit": "kg",
  "unit_cost": 85,
  "applied_date": "2025-04-02"
}
```

### Schedule Weeding Labour
```
POST /api/agriculture/fms/labour
{
  "farm_parcel_id": "par-abc",
  "task_type": "weeding",
  "scheduled_date": "2025-04-10",
  "worker_count": 8,
  "daily_rate": 600,
  "duration_days": 2
}
```
