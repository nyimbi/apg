# Time Series Analytics — User Guide

**Capability ID**: `bia_tsa` | **Domain**: `bia` | **Version**: `1.1.0`
**© 2025 Datacraft** | **Author**: Nyimbi Odero

---

## Overview

`bia_tsa` provides a tenant-scoped time-series analytics platform covering ingestion,
anomaly detection, forecasting, decomposition, signal processing, and data quality.
All methods are `async`, use `Decimal` for financial values, and enforce tenant isolation
via `guard_tenant_id`.

---

## Installation

```bash
pip install apg-bia-tsa
```

---

## Quick Start

```python
import asyncio
from capabilities.bia.tsa.service import TimeSeriesService

svc = TimeSeriesService(tenant_id="acme", actor_id="analyst")

async def main():
    # 1. Register a stream
    stream = await svc.register_stream(
        tenant_id="acme",
        name="server_cpu",
        protocol="batch",
        frequency="1min",
        owner_id="analyst",
        source_identifier="host-01/cpu",
    )
    stream_id = stream["id"]

    # 2. Ingest data
    points = [{"ts": str(1000 + i * 60), "value": 40.0 + i * 0.3} for i in range(120)]
    await svc.ingest_time_series("acme", stream_id, points, timestamp_col="ts")

    # 3. Score data quality before running analytics
    quality = await svc.score_data_quality("acme", stream_id)
    print(f"Quality score: {quality['score']}/100")

    # 4. Run anomaly detection
    anomalies = await svc.anomaly_detect_ts("acme", stream_id, method="zscore", sensitivity=0.95)
    print(f"Anomalies: {anomalies['anomaly_count']}")

asyncio.run(main())
```

---

## Core Concepts

### Tenant Isolation

Every method accepts `tenant_id` as the first positional argument.  The service
enforces isolation — a tenant can only access streams, forecasts, and anomaly events
belonging to their own `tenant_id`.

### Data Points

A data point is a plain `dict` with at minimum two keys:

```python
{"ts": "1717200000", "value": 123.45}
```

The column names are configurable per call (`timestamp_col`, `value_col`).

### Stream Lifecycle

```
registered → active → paused → active → archived
```

Paused and archived streams reject ingestion.  Use `resume_stream()` to reactivate
a paused stream.

---

## Ingestion

### Batch Ingest with Deduplication

```python
result = await svc.ingest_time_series(
    tenant_id="acme",
    series_id=stream_id,
    data_points=[
        {"ts": "1000", "value": 99.5},
        {"ts": "1060", "value": 100.1},
        {"ts": "1120", "value": 98.7},
    ],
    timestamp_col="ts",
    value_col="value",
)
# result["duplicates_skipped"] counts points with existing timestamps
```

### Backfill Late-Arriving Data

```python
backfill = await svc.backfill_stream(
    tenant_id="acme",
    stream_id=stream_id,
    source_data=[{"ts": "500", "value": 88.0}, {"ts": "560", "value": 89.1}],
    timestamp_col="ts",
    strategy="merge_newer_wins",   # existing data takes precedence
)
print(backfill["points_added"], backfill["points_skipped"])
```

Available strategies: `merge_newer_wins`, `merge_older_wins`, `replace`.

### Resample to Uniform Frequency

```python
resampled = await svc.resample_series(
    tenant_id="acme",
    series_id=stream_id,
    target_frequency="5m",         # 5-minute bars
    aggregation="mean",
    fill_method="forward_fill",
)
# Output stored as a new stream: "{stream_id}_resampled_5m"
print(resampled["output_series_id"], resampled["output_points"])
```

---

## Data Quality

```python
quality = await svc.score_data_quality(
    tenant_id="acme",
    series_id=stream_id,
    expected_frequency_seconds=60,
)
# Returns:
# {
#   "score": 94.5,            # composite 0-100
#   "completeness": 100.0,    # % non-null
#   "uniqueness": 98.3,       # % non-duplicate timestamps
#   "consistency": 97.1,      # % within 4-sigma
#   "timeliness": 91.7,       # % gaps ≤ 2× expected_frequency
#   "issues": [...],
#   "recommendations": [...]
# }
```

**Recommendation**: call `score_data_quality` immediately after ingestion and before
running any forecasting or anomaly detection.

---

## Anomaly Detection

### Single Series

```python
result = await svc.anomaly_detect_ts(
    tenant_id="acme",
    series_id=stream_id,
    method="zscore",     # zscore | iqr | isolation_forest | moving_average | stl
    sensitivity=0.95,
)
for anomaly in result["anomalies"]:
    print(anomaly["timestamp"], anomaly["value"], anomaly["severity"])
```

### Multi-Series Batch (Concurrent)

```python
batch = await svc.anomaly_detect_batch(
    tenant_id="acme",
    series_ids=[stream_id, stream_id_2, stream_id_3],
    method="zscore",
    sensitivity=0.95,
)
print(f"Total anomalies across {batch['series_count']} series: {batch['total_anomalies_detected']}")
for sid, res in batch["per_series"].items():
    print(f"  {sid}: {res.get('anomaly_count', 'error')} anomalies")
```

---

## Seasonal Decomposition

```python
decomp = await svc.seasonal_decompose(
    tenant_id="acme",
    series_id=stream_id,
    period=12,                  # 12 steps = annual seasonality on monthly data
    model_type="additive",      # additive | multiplicative
    extrapolate_trend=3,
)
print(f"Seasonal strength: {decomp['seasonal_strength']:.2%}")
```

---

## Spectral Analysis (FFT)

Use spectral analysis to auto-discover the dominant period before decomposing:

```python
spectrum = await svc.spectral_analysis(
    tenant_id="acme",
    series_id=stream_id,
    n_top_frequencies=5,
    window_function="hanning",  # rectangular | hanning | hamming
)
print(f"Suggested period: {spectrum['suggested_period']}")
print(f"Dominant frequency: {spectrum['dominant_frequency']:.4f} cycles/sample")

# Feed suggested_period into seasonal_decompose
decomp = await svc.seasonal_decompose(
    tenant_id="acme",
    series_id=stream_id,
    period=spectrum["suggested_period"],
)
```

---

## Forecasting

### ARIMA / SARIMA

```python
arima = await svc.forecast_arima(
    tenant_id="acme",
    series_id=stream_id,
    periods_ahead=24,
    confidence=0.95,
    order=(2, 1, 2),
    seasonal_order=(1, 0, 1, 12),   # SARIMA seasonal component
)
for pt in arima["forecast_points"][:5]:
    print(f"h={pt['h']}: {pt['forecast']} [{pt['lower']}, {pt['upper']}]")
```

### Prophet-Style Decomposable Model

```python
prophet = await svc.forecast_prophet(
    tenant_id="acme",
    series_id=stream_id,
    periods_ahead=52,
    growth="linear",
    seasonality={"yearly": True, "weekly": True, "daily": False},
    changepoint_prior_scale=0.05,
)
```

### Holt-Winters ETS

```python
from decimal import Decimal

ets = await svc.forecast_ets(
    tenant_id="acme",
    series_id=stream_id,
    periods_ahead=12,
    seasonal_periods=12,
    alpha=Decimal("0.30"),
    beta=Decimal("0.10"),
    gamma=Decimal("0.20"),
    trend_type="additive",
    seasonal_type="additive",
)
print(f"RMSE: {ets['rmse']}, AIC: {ets['aic']}")
```

### Conformal Prediction Interval Calibration

Apply distribution-free calibration to any existing forecast:

```python
cal = await svc.calibrate_forecast_intervals(
    tenant_id="acme",
    series_id=stream_id,
    forecast_id=ets["id"],
    calibration_fraction=0.2,   # use last 20% of training data as calibration set
    alpha=0.05,                 # target 95% coverage
)
print(f"Conformal q: {cal['conformal_q']}")
# Each forecast point now has conformal_lower / conformal_upper fields
```

### Walk-Forward Backtesting

```python
bt = await svc.backtest_forecast(
    tenant_id="acme",
    series_id=stream_id,
    model="arima",             # arima | holt_winters | linear
    n_splits=5,
    horizon=12,
    metric="mae",              # mae | rmse | mape | smape
    order=(1, 1, 1),
)
print(f"Mean MAE: {bt['mean_mae']}, Quality: {bt['quality_assessment']}")
for fold in bt["fold_results"]:
    print(f"  Fold {fold['fold']}: train={fold['train_size']} mae={fold['mae']}")
```

---

## Feature Extraction (ML Pipelines)

```python
features = await svc.extract_features(
    tenant_id="acme",
    series_id=stream_id,
    feature_set="comprehensive",   # minimal | basic | comprehensive
    window_size=60,                # use last 60 points only
)
# features["features"] is a flat dict[str, float] ready for XGBoost/LightGBM
print(features["features"].keys())
# dict_keys(['count', 'mean', 'std', 'min', 'max', 'cv', 'range', 'iqr', 'skew',
#            'kurtosis', 'autocorr_lag1', 'autocorr_lag2', 'autocorr_lag7',
#            'zero_crossing_rate', 'energy', 'trend_slope', 'peak_count', 'approx_entropy'])
```

---

## Financial Time Series (OHLCV)

Use `Decimal`-precision OHLCV aggregation for tick data:

```python
# Ingest tick data
ticks = [
    {"ts": str(1000 + i), "value": 100.0 + i * 0.01, "volume": 500}
    for i in range(300)
]
await svc.ingest_time_series("acme", tick_stream_id, ticks)

# Aggregate into 60-second OHLCV bars with VWAP
ohlcv = await svc.aggregate_ohlcv(
    tenant_id="acme",
    series_id=tick_stream_id,
    bar_seconds=60,
    price_col="value",
    volume_col="volume",
)
for bar in ohlcv["bars"][:3]:
    print(f"ts={bar['bar_ts']} O={bar['open']} H={bar['high']} L={bar['low']} C={bar['close']} V={bar['volume']} VWAP={bar['vwap']}")
```

All price and volume values are stored and returned as `str(Decimal)` to preserve
6-decimal-place precision without floating-point error.

---

## Rolling Statistics

```python
rolling = await svc.rolling_statistics(
    tenant_id="acme",
    series_id=stream_id,
    window=20,
    metrics=["mean", "std", "cv", "skew"],
    min_periods=5,
)
```

Supported metrics: `mean`, `std`, `variance`, `min`, `max`, `sum`, `median`, `cv`, `skew`.

---

## Cross-Series Correlation

```python
corr = await svc.correlation_ts(
    tenant_id="acme",
    series1_id=cpu_stream_id,
    series2_id=mem_stream_id,
    lag_range=(-10, 10),
    method="pearson",       # pearson | spearman | kendall
)
print(f"Optimal lag: {corr['optimal_lag']}, Correlation: {corr['max_correlation']:.3f}")
print(f"Significant: {corr['significant']}")
```

---

## Changepoint Detection

```python
cp = await svc.changepoint_detection(
    tenant_id="acme",
    series_id=stream_id,
    method="pelt",            # pelt | binary_segmentation | dynamic_programming | prophet | cusum
    penalty=1.0,
    min_segment_length=5,
)
print(f"Changepoints at indices: {cp['changepoint_indices']}")
for seg in cp["segments"]:
    print(f"  Segment {seg['segment_index']}: [{seg['start_index']}, {seg['end_index']}) mean={seg['mean']}")
```

---

## Interpolation / Gap Filling

```python
filled = await svc.interpolate_missing(
    tenant_id="acme",
    series_id=stream_id,
    method="linear",          # linear | forward_fill | backward_fill | cubic | spline | seasonal | mean
    max_gap=5,                # skip runs of >5 consecutive NaN values
)
print(f"Gaps filled: {filled['gaps_filled']}, skipped: {filled['gaps_skipped']}")
```

---

## Comprehensive TS Report

```python
report = await svc.ts_report(
    tenant_id="acme",
    series_id=stream_id,
    period="last_30_days",     # last_24_hours | last_7_days | last_30_days | last_90_days | all_time
    include_forecast=True,
    include_anomalies=True,
)
print(report["descriptive_statistics"])
print(report["anomaly_summary"])
print(report["recommendations"])
```

---

## Service Statistics

```python
stats = await svc.get_stats(tenant_id="acme")
# {stream_count, anomaly_config_count, anomaly_event_count, forecast_count,
#  window_count, decomposition_count, correlation_count, changepoint_count, ...}
```

---

## Provides

- `high_frequency_time_series_ingestion`
- `anomaly_detection`
- `seasonality_decomposition`
- `spectral_analysis`
- `time_series_forecasting`
- `ets_forecasting`
- `forecast_backtesting`
- `conformal_prediction_intervals`
- `stream_windowing`
- `multi_stream_correlation`
- `changepoint_detection`
- `gap_filling_interpolation`
- `feature_extraction`
- `data_quality_scoring`
- `ohlcv_aggregation`
- `series_resampling`
- `stream_backfill`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `mqeb`
- `moni`
- `ntfy`
- `schd`

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bia/tsa/dashboard` | `bia_tsa:view` | Overview |
| `/bia/tsa/streams` | `bia_tsa:streams` | Streams |
| `/bia/tsa/streams/<id>` | `bia_tsa:streams` | Streams |
| `/bia/tsa/streams/<id>/explore` | `bia_tsa:streams` | Streams |
| `/bia/tsa/streams/<id>/quality` | `bia_tsa:streams` | Streams |
| `/bia/tsa/anomalies` | `bia_tsa:anomalies` | Analysis |
| `/bia/tsa/anomalies/<id>` | `bia_tsa:anomalies` | Analysis |
| `/bia/tsa/decomposition` | `bia_tsa:decompose` | Analysis |
| `/bia/tsa/spectral` | `bia_tsa:streams` | Analysis |
| `/bia/tsa/forecasts` | `bia_tsa:forecast` | Forecasting |
| `/bia/tsa/forecasts/<id>/backtest` | `bia_tsa:forecast` | Forecasting |

---

## Further Reading

- `service.py` — Business logic; all public async methods
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints (Flask-AppBuilder)
- `views.py` — FAB views and Pydantic schemas
- `capability_contract.py` — Business rules and feature flags
- `README.md` — Quick reference card
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 production-grade enhancements
