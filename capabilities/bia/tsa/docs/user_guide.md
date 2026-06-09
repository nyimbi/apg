# Time Series Analytics

**Capability ID**: `bia_tsa` | **Domain**: `bia` | **Version**: `1.0.0`

## Description

The Time Series Analytics capability (bia_tsa) provides high-frequency time-series stream ingestion via 7 protocols, configurable anomaly detection with 8 methods, seasonality decomposition (trend/seasonality/residual/cyclical), time-series forecasting with 7 models, stream windowing, gap-filling interpolation, and real-time alerting — all tenant-scoped and bytewax-streamed.

## Installation

```bash
pip install apg-bia-tsa
```

## Provides

- `high_frequency_time_series_ingestion`
- `anomaly_detection`
- `seasonality_decomposition`
- `time_series_forecasting`
- `stream_windowing`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `mqeb`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bia/tsa/dashboard` | `bia_tsa:view` | Overview |
| `/bia/tsa/streams` | `bia_tsa:streams` | Streams |
| `/bia/tsa/streams/<id>` | `bia_tsa:streams` | Streams |
| `/bia/tsa/streams/<id>/explore` | `bia_tsa:streams` | Streams |
| `/bia/tsa/anomalies` | `bia_tsa:anomalies` | Analysis |
| `/bia/tsa/anomalies/<id>` | `bia_tsa:anomalies` | Analysis |
| `/bia/tsa/decomposition` | `bia_tsa:decompose` | Analysis |
| `/bia/tsa/forecasts` | `bia_tsa:forecast` | Forecasting |

## Key Service Methods

- `describe()`
- `register_stream()`
- `get_stream()`
- `list_streams()`
- `pause_stream()`
- `resume_stream()`
- `archive_stream()`
- `ingest_data()`
- `ingest_time_series()`
- `configure_anomaly_detection()`

_(See `service.py` for complete API.)_

## Interoperability

`bia_tsa` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bia_tsa;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BIA_TSA_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
