# Data Quality (dcat_dq)

Dataset profiling, quality scoring, anomaly detection, completeness/uniqueness/accuracy rules, DQ reports.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/dcat/dq/health | Service health |
| GET | /api/dcat/dq/rules | List rules |
| POST | /api/dcat/dq/rules | Create rule |
| GET | /api/dcat/dq/rules/{id} | Get rule |
| PUT | /api/dcat/dq/rules/{id} | Update rule |
| DELETE | /api/dcat/dq/rules/{id} | Delete rule |
| POST | /api/dcat/dq/profiles | Profile dataset |
| GET | /api/dcat/dq/profiles/{dataset_id} | Get latest profile |
| POST | /api/dcat/dq/runs | Run quality checks |
| GET | /api/dcat/dq/runs | List runs |
| GET | /api/dcat/dq/runs/{id} | Get run |
| GET | /api/dcat/dq/anomalies | List anomalies |
| POST | /api/dcat/dq/anomalies/{id}/acknowledge | Acknowledge anomaly |
| GET | /api/dcat/dq/scorecard/{dataset_id} | Quality scorecard |
| GET | /api/dcat/dq/reports/{dataset_id} | DQ report |
| GET | /api/dcat/dq/dashboard | Tenant dashboard |
| GET | /api/dcat/dq/audit | Audit trail |

---

## World-Class Enhancements (v2.0)

- **I1.** Data Quality — World-Class Improvement Proposals
- **I2.** Column-Level Distribution Fingerprinting
- **I3.** Regex Rule Evaluation Against Real Data
- **I4.** Referential Integrity Rule Evaluation
- **I5.** Freshness Rule with Timestamp Column Evaluation
- **I6.** Multi-Dimensional DQ Scorecard (6 ISO 25012 Dimensions)
- **I7.** Statistical Outlier Detection Per Column (IQR + Z-Score)
- **I8.** Rule Template Library
- **I9.** Incremental Data Profiling (Partition-Aware)
- **I10.** Data Lineage Impact Score
- **I11.** Expectations Catalog (Declarative YAML/JSON Import)
- **I12.** SLA Breach Tracking and Escalation
- **I13.** Cross-Dataset Consistency Validation
- **I14.** Automated Rule Suggestion via Column Profiling
- **I15.** Quality Score Trend Forecasting (EWMA)

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
