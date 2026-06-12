# Time Series Analytics

## Overview
The Time Series Analytics capability (bia_tsa) provides high-frequency time-series stream ingestion via 7 protocols, configurable anomaly detection with 8 methods, seasonality decomposition (trend/seasonality/residual/cyclical), time-series forecasting with 7 models, stream windowing, gap-filling interpolation, and real-time alerting — all tenant-scoped and bytewax-streamed.

## Capability ID
`bia_tsa`

## Provides
- high_frequency_time_series_ingestion: 7 protocols from tick to daily frequency
- anomaly_detection: 8 methods including LSTM autoencoder and prophet residual
- seasonality_decomposition: Additive and multiplicative decomposition
- time_series_forecasting: ARIMA, SARIMA, Prophet, LSTM, Transformer, Ensemble
- stream_windowing: Tumbling, sliding, session, hopping windows
- multi_stream_correlation: Cross-stream correlation analysis
- gap_filling_interpolation: 6 interpolation methods for sparse streams
- real_time_alerting: Rate-gated anomaly alerts via ntfy

## Requires
| Capability | Reason |
|------------|--------|
| auth | User identity and permission checks |
| audl | Audit stream registrations and anomalies |
| mten | Tenant context enforcement |
| conf | Runtime configuration |
| mqeb | High-throughput event streaming via bytewax |
| moni | Stream health and ingestion rate monitoring |
| ntfy | Anomaly alert delivery |
| schd | Scheduled decomposition and forecast jobs |

## Configuration
| Option | Default | Description |
|--------|---------|-------------|
| max_streams_per_tenant | 200 | Stream registration limit |
| max_window_size_seconds | 86,400 | 24-hour maximum window size |
| max_horizon_periods | 365 | Maximum forecast horizon |
| default_anomaly_method | zscore | Default detection algorithm |
| default_forecast_model | prophet | Default forecasting model |
| sensitivity_default | 0.95 | Default anomaly sensitivity |

## New Features (v1.1)

| Feature | Method | Description |
|---------|--------|-------------|
| Spectral Analysis | `spectral_analysis()` | FFT-based dominant frequency discovery; auto-suggests `period` for `seasonal_decompose` |
| Data Quality Scoring | `score_data_quality()` | Composite 0-100 score across completeness, uniqueness, consistency, timeliness dimensions |
| Feature Extraction | `extract_features()` | tsfresh-style: 'minimal' (5), 'basic' (12), 'comprehensive' (25+) statistical features for ML pipelines |
| ETS Forecast | `forecast_ets()` | Holt-Winters Triple Exponential Smoothing with additive/multiplicative error, trend, seasonality; Decimal-precision parameters |
| Batch Anomaly Detection | `anomaly_detect_batch()` | Concurrent multi-series anomaly scoring via `asyncio.gather`; aggregate summary across all series |
| Stream Backfill | `backfill_stream()` | Idempotent late-data backfill with `merge_newer_wins`, `merge_older_wins`, and `replace` strategies; audit-tagged records |
| Walk-Forward Backtesting | `backtest_forecast()` | n-fold expanding-window backtesting for ARIMA, Holt-Winters, and linear models; MAE/RMSE/MAPE/sMAPE |
| OHLCV Aggregation | `aggregate_ohlcv()` | Tick-to-bar aggregation using `Decimal` arithmetic; includes VWAP when volume column present |
| Series Resampling | `resample_series()` | Uniform frequency resampling (mean/sum/last/first/min/max) with forward/backward fill; stores result as new stream |
| Conformal Calibration | `calibrate_forecast_intervals()` | Distribution-free interval calibration using held-out residual quantiles; updates existing forecast records in-place |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/tsa/streams | GET/POST | List/register streams | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/ingest | POST | Ingest data points | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/pause | POST | Pause stream | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/backfill | POST | Backfill late-arriving data | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/resample | POST | Resample to uniform frequency | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/quality | GET | Data quality score | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/features | GET | Extract ML features | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/spectral | GET | Spectral / FFT analysis | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/ohlcv | GET | OHLCV bar aggregation | bia_tsa:streams |
| /api/bia/tsa/anomaly-configs | GET/POST | Anomaly configs | bia_tsa:anomalies |
| /api/bia/tsa/anomaly-events | GET | Anomaly events | bia_tsa:anomalies |
| /api/bia/tsa/anomaly-batch | POST | Multi-series batch detection | bia_tsa:anomalies |
| /api/bia/tsa/forecasts | GET/POST | Forecasts | bia_tsa:forecast |
| /api/bia/tsa/forecasts/<id>/calibrate | POST | Conformal interval calibration | bia_tsa:forecast |
| /api/bia/tsa/forecasts/backtest | POST | Walk-forward backtesting | bia_tsa:forecast |
| /api/bia/tsa/decompositions | GET/POST | Decompositions | bia_tsa:decompose |
| /api/bia/tsa/windows | GET/POST | Stream windows | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/fill-gaps | POST | Fill gaps | bia_tsa:streams |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| max_streams_enforced | >200 streams | deny |
| paused_stream_cannot_ingest | state=paused | deny |
| archived_stream_read_only | state=archived on ingest | deny |
| forecast_requires_sufficient_history | Insufficient data | deny |
| horizon_limit_enforced | >365 periods | deny |
| anomaly_alert_gated | Alert rate exceeded | deny |
| window_size_limit_enforced | >86400 seconds | deny |

## Data Models
- StreamResponse: id, name, protocol, frequency, state, source_identifier, point_count, last_ingested_at
- AnomalyConfigResponse: id, stream_id, method, sensitivity, active
- AnomalyEvent: id, stream_id, detected_at, value, score, severity, confirmed
- ForecastResponse: id, stream_id, model, horizon_periods, forecast_data (t, forecast, lower, upper)
- WindowResponse: id, stream_id, window_type, size_seconds, aggregation_function
- DecompositionResult: id, stream_id, components, trend_data, seasonality_data, residual_data

## Streaming Events
- stream_registered, stream_data_ingested, anomaly_detected, anomaly_confirmed
- decomposition_completed, forecast_generated, window_opened, window_closed
- alert_triggered, gap_filled, stream_paused, stream_resumed

## Edge Cases Handled
- Paused streams reject ingestion — prevents silent data loss during maintenance
- Anomaly alert rate-limiting prevents alert storms from noisy sensors
- Archived streams are read-only — no ingestion or modification after archival
- Forecast horizon capped at 365 periods to prevent computationally unbounded jobs
- Gap filling uses configurable interpolation — defaults to forward fill for IoT streams
- Streams in error state require explicit remediation before resuming ingestion

## Composability Notes
- Feeds anomaly events and forecasts into bia_pda for ML model training data
- bia_dsh can bind stream widgets to real-time data via refresh intervals
- bia_rpt generates scheduled stream health and anomaly summary reports
- mqeb (bytewax) handles high-throughput streaming ingestion pipeline
- moni tracks ingestion rates, latency, and anomaly detection accuracy

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements: Time Series Analytics (bia_tsa)
- **I2.** Improvement 1: Adaptive Threshold Anomaly Detection
- **I3.** Improvement 2: Hierarchical Time Series Forecasting
- **I4.** Improvement 3: Online Learning / Incremental Model Updates
- **I5.** Improvement 4: Causal Impact Analysis
- **I6.** Improvement 5: Vector Autoregression (VAR) Multi-Series Forecasting
- **I7.** Improvement 6: Time Series Feature Engineering (tsfresh-style)
- **I8.** Improvement 7: Spectral Analysis (FFT / Periodogram)
- **I9.** Improvement 8: Conformal Prediction Intervals
- **I10.** Improvement 9: Automated Data Quality Scoring
- **I11.** Improvement 10: Stream Backfill and Replay
- **I12.** Improvement 11: Multi-Condition Alert Rules Engine
- **I13.** Improvement 12: Exponential Smoothing (ETS) Models
- **I14.** Improvement 13: Time Series Clustering (DTW)
- **I15.** Improvement 14: Persistence Layer Abstraction (TimescaleDB Backend)

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
