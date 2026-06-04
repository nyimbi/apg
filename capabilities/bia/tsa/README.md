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

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/tsa/streams | GET/POST | List/register streams | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/ingest | POST | Ingest data points | bia_tsa:streams |
| /api/bia/tsa/streams/<id>/pause | POST | Pause stream | bia_tsa:streams |
| /api/bia/tsa/anomaly-configs | GET/POST | Anomaly configs | bia_tsa:anomalies |
| /api/bia/tsa/anomaly-events | GET | Anomaly events | bia_tsa:anomalies |
| /api/bia/tsa/forecasts | GET/POST | Forecasts | bia_tsa:forecast |
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
