# Crop Management — User Guide

## Overview

The Crop Management capability (agr_crp) provides end-to-end tracking of crop lifecycles from
planting planning through yield recording. It manages variety registries, planting calendars,
phenology observations, rotation plans, and yield statistics.

## Key Use Cases

- **Variety Selection**: Maintain a registry of crop varieties with agronomic attributes
  (maturity days, yield potential, disease resistance, climate tolerances).
- **Planting Calendar**: Define region-specific planting windows and get recommendations
  for optimal planting dates.
- **Crop Lifecycle Tracking**: Create crop records for each planting event, track status
  from planned → planted → growing → harvested.
- **Phenology Monitoring**: Record growth stage observations (germination through maturity)
  with measurements and images.
- **Rotation Planning**: Define multi-season crop rotation sequences to maintain soil health
  and break pest cycles.
- **Yield Recording**: Record gross and net yields post-harvest; compute seasonal statistics.

## API Reference

### Varieties

```
POST /api/agriculture/crp/varieties
{
  "name": "H614D",
  "crop_type": "maize",
  "maturity_days": 120,
  "yield_potential_kg_ha": 8000,
  "drought_tolerance": "moderate",
  "disease_resistance": ["MLN", "GLS"]
}
```

### Planting Calendars

```
POST /api/agriculture/crp/calendars
{
  "crop_type": "maize",
  "region": "Rift Valley",
  "planting_window_start": "03-01",
  "planting_window_end": "04-15",
  "harvest_window_start": "07-01",
  "harvest_window_end": "08-30"
}
```

### Crop Records

```
POST /api/agriculture/crp/crops
{
  "farm_parcel_id": "parcel-001",
  "crop_type": "maize",
  "variety_id": "var-abc123",
  "season": "2025A",
  "planting_date": "2025-03-15",
  "area_ha": 2.5,
  "target_yield_kg": 18000
}
```

### Yield Recording

```
POST /api/agriculture/crp/yields
{
  "crop_id": "crp-xyz789",
  "harvest_date": "2025-08-10",
  "gross_yield_kg": 19500,
  "net_yield_kg": 18200,
  "moisture_pct": 13.5,
  "grade": "A"
}
```

## Tenant Isolation

Pass `X-Tenant-ID` header on all requests. Each tenant's data is isolated.
