# World-Class Improvements: Time Series Analytics (bia_tsa)

**Capability**: `bia_tsa` | **Domain**: Business Intelligence & Analytics
**Author**: Nyimbi Odero | **Date**: 2026-06-11 | **© 2025 Datacraft**

---

## Improvement 1: Adaptive Threshold Anomaly Detection

**Category**: Algorithms / Machine Learning

**Justification**: Static Z-score thresholds fail on non-stationary series (sensor drift, seasonal ramp-ups). Production telemetry systems at Datadog and InfluxDB dynamically update thresholds using exponentially weighted moving statistics, cutting false-positive alert rates by 40-60%.

**Implementation**:
- Maintain per-series EWMA (exponential weighted moving average) of mean and variance using decay factor α configurable per stream.
- Replace static `mean_val / std` in `anomaly_detect_ts` with EWMA-tracked values updated on each ingestion batch.
- Store EWMA state in `_stream_ewma: dict[tuple[str,str], dict]` keyed by `(tenant_id, series_id)`.
- Add `ewma_alpha: float = 0.05` parameter to `configure_anomaly_detection` and `anomaly_detect_ts`.

**Competitor Reference**: Datadog Adaptive Anomaly Detection — "uses machine learning to automatically learn and adapt to the typical behavior of a metric" (docs.datadoghq.com/monitors/types/anomaly).

---

## Improvement 2: Hierarchical Time Series Forecasting

**Category**: Forecasting / Aggregation

**Justification**: Organisations rarely forecast a single stream in isolation. Sales, IoT sensor networks, and financial aggregates all have tree-shaped hierarchies (store → region → national). Reconciling bottom-up forecasts to be coherent with top-down aggregates is critical for planning. Nixtla's StatsForecast and Meta's Kats both implement MinT (minimum trace) reconciliation.

**Implementation**:
- Add `register_hierarchy(tenant_id, parent_id, child_ids, weights)` to define series parent-child relationships stored in `_hierarchy: dict`.
- Add `forecast_hierarchical(tenant_id, root_id, method, periods)` generating leaf forecasts and reconciling upward using proportional allocation.
- Support bottom-up, top-down, and MinT reconciliation strategies via a `strategy` parameter.

**Competitor Reference**: Nixtla `HierarchicalForecast` — hierarchical reconciliation including MinT, BottomUp, TopDown (nixtla.mintlify.app/hierarchicalforecast).

---

## Improvement 3: Online Learning / Incremental Model Updates

**Category**: Real-Time ML

**Justification**: Batch ARIMA and Prophet models trained once become stale as data distribution shifts. High-frequency streams (tick data, IoT) need models that update in O(1) per new observation. River and online-statsmodels implement Kalman filter-based online regression and ARIMA variants that eliminate batch retraining overhead.

**Implementation**:
- Add `update_model_online(tenant_id, series_id, new_point)` applying a one-step Kalman gain update to stored forecast model parameters.
- Track model state in `_online_models: dict[tuple[str,str], dict]` including Kalman state vector, covariance matrix, and observation noise.
- Return updated model parameters and a one-step-ahead prediction with confidence interval.

**Competitor Reference**: River (riverml.xyz) — `time_series.SNARIMAX` with `learn_one` / `predict_one` interface for fully online time series.

---

## Improvement 4: Causal Impact Analysis

**Category**: Causal Inference

**Justification**: Detecting whether an intervention (product launch, policy change, marketing campaign) caused a measurable shift is a top-tier analytics requirement. Google's CausalImpact is widely cited in finance, retail, and public health. Current TSA capabilities detect changepoints but cannot attribute causality or estimate counterfactuals.

**Implementation**:
- Add `causal_impact(tenant_id, series_id, intervention_date, control_series_ids, periods_post)` building a Bayesian structural time series (BSTS) model on pre-intervention data.
- Estimate a synthetic counterfactual from control series using linear regression during the pre-intervention period.
- Return point effect estimate, cumulative effect, posterior credible interval, and a p-value for the intervention effect.

**Competitor Reference**: Google CausalImpact (R/Python) — structural time series counterfactual estimation for marketing and policy analytics.

---

## Improvement 5: Vector Autoregression (VAR) Multi-Series Forecasting

**Category**: Multivariate Forecasting

**Justification**: Univariate forecasting ignores information in correlated companion series. VAR models are the standard approach for macroeconomic and multi-sensor forecasting. Cross-correlation detection is already implemented; VAR is its forecasting counterpart and the logical next step.

**Implementation**:
- Add `forecast_var(tenant_id, series_ids, lags, periods_ahead, confidence)` fitting a VAR(p) model across all specified series.
- Estimate the coefficient matrix A via OLS on the lagged data matrix.
- Return a joint forecast dict per series with point estimates and confidence ellipsoids.
- Store result in `_forecasts` under a compound key linking all series.

**Competitor Reference**: statsmodels `VAR` — standard multivariate forecasting with AIC/BIC lag selection and Granger causality tests (statsmodels.org).

---

## Improvement 6: Time Series Feature Engineering (tsfresh-style)

**Category**: Feature Extraction / ML Pipeline

**Justification**: Time series data fed into tabular ML models (XGBoost, LightGBM) requires structured feature extraction. tsfresh extracts 794 statistical features per series. Without systematic feature engineering, ML models trained on raw windows underperform by 15-30% on UCR archive benchmarks.

**Implementation**:
- Add `extract_features(tenant_id, series_id, window_size, feature_set)` supporting feature sets: `basic` (8 stats), `comprehensive` (40+ features), `minimal` (5 features for low-latency pipelines).
- Compute: zero-crossing rate, energy, spectral entropy, autocorrelation at lags 1/2/7, Hurst exponent, permutation entropy, approximate entropy.
- Return a flat `dict[str, float]` suitable for direct use as ML model input features.

**Competitor Reference**: tsfresh (tsfresh.readthedocs.io) — automated time series feature extraction with relevance filtering for ML pipelines.

---

## Improvement 7: Spectral Analysis (FFT / Periodogram)

**Category**: Frequency Domain Analysis

**Justification**: Identifying dominant periodicities through Fourier analysis is foundational for seasonality decomposition, filter design, and bandwidth estimation. Currently `seasonal_decompose` requires a known `period` parameter; spectral analysis would auto-discover it, removing a major analyst friction point.

**Implementation**:
- Add `spectral_analysis(tenant_id, series_id, window_function)` computing the DFT using the Cooley-Tukey algorithm (stdlib `cmath` only).
- Return power spectral density, dominant frequency, implied period, and top-5 harmonics.
- Auto-suggest `period` parameter for `seasonal_decompose` based on dominant frequency.
- Support window functions: `rectangular`, `hanning`, `hamming` to reduce spectral leakage.

**Competitor Reference**: SciPy `signal.periodogram` and `signal.welch` — standard PSD estimation used in signal processing, econometrics, and climate science.

---

## Improvement 8: Conformal Prediction Intervals

**Category**: Uncertainty Quantification

**Justification**: Symmetric Gaussian confidence intervals undercover during regime changes and fat-tailed distributions. Conformal prediction provides distribution-free coverage guarantees regardless of the underlying data distribution. Nixtla's ConformalCalibration shows 95% coverage with 30% tighter intervals than Gaussian CI on financial time series.

**Implementation**:
- Add `calibrate_forecast_intervals(tenant_id, series_id, forecast_id, calibration_fraction, alpha)` using held-out residuals to compute non-conformity scores.
- Sort residuals to find the (1-α) quantile and use it to expand/contract existing forecast intervals.
- Replace symmetric `± z * sigma` intervals with asymmetric conformal intervals `[yhat - q_lower, yhat + q_upper]`.
- Store calibrated intervals back into the forecast record.

**Competitor Reference**: MAPIE (conformal-prediction.readthedocs.io) — model-agnostic conformal prediction intervals for regression and time series.

---

## Improvement 9: Automated Data Quality Scoring

**Category**: Data Quality / Observability

**Justification**: Production time series pipelines regularly receive corrupted, delayed, or duplicated data. Without automated quality scoring, analysts manually identify issues only after downstream models fail. Evidently AI and Great Expectations provide automated quality profiling; no equivalent exists in bia_tsa today.

**Implementation**:
- Add `score_data_quality(tenant_id, series_id)` computing a composite quality score (0-100) from:
  - Completeness (% non-null values)
  - Timeliness (% points adhering to expected frequency ± 10%)
  - Consistency (% values within 4-sigma of rolling mean)
  - Uniqueness (% non-duplicate timestamps)
- Return per-dimension scores, issue inventory, and automated remediation suggestions.
- Emit `data_quality_scored` audit event and store result in `_quality_scores: list`.

**Competitor Reference**: Evidently AI DataQualityReport — automated profiling and drift detection for production ML pipelines (evidentlyai.com).

---

## Improvement 10: Stream Backfill and Replay

**Category**: Data Management / Reliability

**Justification**: Sensor outages, ETL failures, and late-arriving data are endemic to production systems. Without a backfill primitive, ops teams manually patch gaps by re-running entire pipelines. Apache Kafka and Flink provide native backfill/replay semantics; bia_tsa should offer a first-class idempotent backfill API.

**Implementation**:
- Add `backfill_stream(tenant_id, stream_id, start_ts, end_ts, source_data, strategy)` where `strategy` is `merge_newer_wins`, `merge_older_wins`, or `replace`.
- Backfill records are tagged `{"backfill": True, "backfill_run_id": run_id}` for auditability.
- Return diff statistics: added, replaced, skipped counts.
- Rate-limit via a semaphore to prevent backfill jobs from starving real-time ingestion.

**Competitor Reference**: Flink `BoundedOutOfOrdernessWatermarks` and Kafka `--from-beginning` replay — production-grade late-data and backfill handling (flink.apache.org).

---

## Improvement 11: Multi-Condition Alert Rules Engine

**Category**: Alerting / Operations

**Justification**: Single-metric threshold alerts generate high false-positive rates. Production observability platforms (PagerDuty, Grafana Alerting) support composite rules: `anomaly_score > 0.9 AND rolling_mean_7d > baseline * 1.2`. bia_tsa only supports single-event anomaly alerts today.

**Implementation**:
- Add `create_alert_rule(tenant_id, name, conditions, severity, notification_channels, cooldown_seconds)` where `conditions` is a list of `{metric, operator, threshold, series_id}` dicts evaluated with AND/OR logic.
- Store rules in `_alert_rules: dict` keyed by `(tenant_id, rule_id)`.
- Add `evaluate_alert_rules(tenant_id)` that evaluates all active rules against current series state and fires notifications for triggered rules.
- Enforce `cooldown_seconds` to prevent repeated firing within the cooldown window.

**Competitor Reference**: Grafana Alerting — multi-condition alert rules with label-based routing, silencing, and inhibition rules (grafana.com/docs/grafana/latest/alerting).

---

## Improvement 12: Exponential Smoothing (ETS) Models

**Category**: Forecasting Models

**Justification**: ETS (Error-Trend-Seasonality) models are the industry standard for inventory forecasting and demand planning. Holt-Winters triple exponential smoothing consistently outperforms ARIMA on M4 and M5 competition benchmarks for business time series. Currently absent from the bia_tsa forecast model catalogue.

**Implementation**:
- Add `forecast_ets(tenant_id, series_id, periods_ahead, error, trend, seasonality, seasonal_periods, damped)` implementing Holt-Winters triple exponential smoothing.
- Parameters: `error` ∈ {additive, multiplicative}, `trend` ∈ {None, additive, multiplicative, damped_additive}, `seasonality` ∈ {None, additive, multiplicative}.
- Estimate smoothing parameters α, β, γ via gradient descent minimizing SSE.
- Return point forecasts, 95% CI from analytical formulas, and AIC/BIC for model selection.

**Competitor Reference**: statsmodels `ETSModel` — full ETS state space with MLE parameter estimation (statsmodels.org/dev/examples/notebooks/generated/ets.html).

---

## Improvement 13: Time Series Clustering (DTW)

**Category**: Pattern Discovery / Segmentation

**Justification**: Grouping similar streams (sensors with identical failure modes, customer segments with identical usage patterns) enables targeted operations. DTW-based clustering is standard for time series similarity — used in industrial IoT (Siemens MindSphere) and financial portfolio analysis. Currently bia_tsa only supports pairwise correlation, not cluster-level segmentation.

**Implementation**:
- Add `cluster_streams(tenant_id, series_ids, n_clusters, distance_metric)` where `distance_metric` ∈ {`euclidean`, `dtw`, `correlation`}.
- Implement DTW with Sakoe-Chiba band constraint (O(n·w) vs O(n²)) for scalability.
- Use k-means with DTW barycenter averaging (DBA) for cluster centroid computation.
- Return cluster assignments, centroids, silhouette score, and within-cluster inertia.

**Competitor Reference**: tslearn (tslearn.readthedocs.io) — time series clustering with DTW, soft-DTW, k-Shape algorithms used in production IoT and financial analytics.

---

## Improvement 14: Persistence Layer Abstraction (TimescaleDB Backend)

**Category**: Storage / Performance

**Justification**: In-memory dicts hit Python GC pressure at ~10M data points and offer zero durability. TimescaleDB delivers 10-100x better compression and query performance for time series via hypertables and continuous aggregates. The existing `database/store.py` and alembic migrations indicate intent but the service layer bypasses the store entirely.

**Implementation**:
- Refactor `TimeSeriesService.__init__` to accept a `store: AbstractTSStore` interface with `write_points`, `read_points`, `delete_points`.
- Add `TimescaleDBStore(db_url)` implementing `AbstractTSStore` using asyncpg with INSERT ON CONFLICT DO NOTHING for idempotent ingestion.
- Implement continuous aggregate views for rolling statistics, eliminating Python-side rolling computation for large series.
- Maintain `MemoryStore` in-memory fallback for testing and embedded deployments.

**Competitor Reference**: TimescaleDB (timescale.com) — PostgreSQL-based time series DB with hypertables, columnar compression, and continuous aggregates. Used at Walmart, Volvo, and Siemens for production telemetry.

---

## Improvement 15: Forecast Accuracy Backtesting (Walk-Forward Validation)

**Category**: Model Evaluation / MLOps

**Justification**: Generating forecasts without measuring historical accuracy is operationally blind. Walk-forward (expanding window) backtesting is the gold standard for time series model evaluation — it replicates real deployment conditions and prevents lookahead bias. Without it, model selection defaults to assumptions rather than empirical evidence.

**Implementation**:
- Add `backtest_forecast(tenant_id, series_id, model, order, n_splits, horizon, metric)` partitioning historical data into n_splits train/test folds using expanding or sliding windows.
- For each fold: fit model on training set, predict `horizon` steps, compute error vs held-out test.
- Aggregate fold-level metrics: MAE, MAPE, RMSE, sMAPE, CRPS (for probabilistic forecasts).
- Return per-fold breakdown, aggregate statistics, and model ranking if multiple models supplied.
- Store backtest results in `_backtests: list` and emit `forecast_backtested` audit event.

**Competitor Reference**: darts `backtesting` module (unit8co.github.io/darts) — walk-forward backtesting with residual diagnostics, coverage metrics, and multi-model comparison.
