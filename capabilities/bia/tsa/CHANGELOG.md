# Changelog — apg-bia-tsa

All notable changes to this package are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-06-02

### Added
- Initial production release of the **Time Series Analytics** capability.
- Capability contract with 8 provided services: high_frequency_time_series_ingestion, anomaly_detection, seasonality_decomposition, time_series_forecasting, stream_windowing....
- Deterministic rule engine with governance rules for all operations.
- Bytewax streaming support for real-time event processing.
- Flask Blueprint API with full CRUD routes and structured error responses.
- Pydantic v2 data models with tenant isolation and UUID7 primary keys.
- Comprehensive test suite covering happy path and edge cases.
- Standalone PyPI packaging (`pip install apg-bia-tsa`).

### Dependencies
- pydantic >= 2.0
- uuid6 >= 0.4
- sqlalchemy >= 2.0
- flask >= 3.0
