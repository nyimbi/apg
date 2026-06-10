# Weather & Climate Analytics — User Guide

## Overview

agr_wth provides agricultural weather intelligence: ingesting forecasts from external sources,
evaluating them against configurable alert thresholds, maintaining historical climate records,
and generating per-crop climate risk assessments.

## Key Use Cases

- **Forecast Integration**: Push forecast data from any provider (KMD, ENACTS, custom APIs).
  Thresholds are evaluated automatically on ingestion.
- **Alert Thresholds**: Configure parameter/operator/value rules per region
  (e.g., rainfall_mm > 80 → "flood_watch"). Alerts auto-fire when forecasts breach thresholds.
- **Historical Patterns**: Build a multi-year archive of monthly climate normals to support
  risk scoring and seasonal recommendations.
- **Climate Risk Assessment**: Run an on-demand risk computation that scores drought, flood,
  frost, and heat stress for a region/crop/season combination.

## Example Workflows

### Set a Threshold
```
POST /api/agriculture/wth/thresholds
{
  "region": "Rift Valley",
  "parameter": "rainfall_mm",
  "operator": "gt",
  "threshold_value": 80,
  "severity": "watch"
}
```

### Ingest a Forecast (auto-evaluates thresholds)
```
POST /api/agriculture/wth/forecasts
{
  "region": "Rift Valley",
  "source": "KMD",
  "forecast_date": "2025-04-01",
  "valid_from": "2025-04-02",
  "valid_to": "2025-04-03",
  "rainfall_mm": 95,
  "temperature_max_c": 28
}
```

### Assess Climate Risk
```
POST /api/agriculture/wth/risk-assessments
{"region": "Rift Valley", "crop_type": "maize", "season": "2025A"}
```
