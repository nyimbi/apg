# Data Quality User Guide

## Overview

The Data Quality capability (`dcat_dq`) provides automated dataset profiling, rule-based quality scoring, statistical anomaly detection, and time-series DQ reports.

## Use Cases

- Define completeness, uniqueness, accuracy, and range rules per dataset column
- Profile datasets to compute column-level statistics (null rates, distinct counts, min/max)
- Run quality checks against live data samples and get a 0–1 quality score
- Detect statistical anomalies in quality scores over time (2-sigma rule)
- Generate DQ trend reports for governance review
- Build tenant-level quality dashboards

## Supported Rule Types

| Type | Description |
|------|-------------|
| `completeness` | Fraction of non-null values in a column |
| `uniqueness` | Fraction of distinct values in a column |
| `accuracy` | Fraction of values passing an expression (e.g. `> 0`) |
| `range` | Fraction of values within min:max bounds |
| `regex` | Fraction of values matching a regular expression |
| `referential` | Values that exist in a reference set |
| `freshness` | Recency of data based on timestamp column |
| `custom` | User-defined expression |

## Quickstart

### 1. Create a completeness rule

```http
POST /api/dcat/dq/rules
{
  "tenant_id": "acme",
  "dataset_id": "ds-abc123",
  "rule_type": "completeness",
  "column": "customer_id",
  "threshold": 0.99,
  "severity": "error"
}
```

### 2. Profile the dataset

```http
POST /api/dcat/dq/profiles
{
  "tenant_id": "acme",
  "dataset_id": "ds-abc123",
  "row_count": 10000,
  "column_profiles": [
    {"column": "customer_id", "null_count": 45, "distinct_count": 9800}
  ]
}
```

### 3. Run quality checks

```http
POST /api/dcat/dq/runs
{
  "tenant_id": "acme",
  "dataset_id": "ds-abc123",
  "data_sample": [{"customer_id": "c1"}, {"customer_id": null}]
}
```

### 4. Generate a report

```http
GET /api/dcat/dq/reports/ds-abc123?tenant_id=acme&period_start=2026-01-01&period_end=2026-06-30
```
