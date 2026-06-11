# Process Mining (pmin)

Infer BPMN process models from NATS event streams, conformance checking, bottleneck analysis, and variant discovery.

## New Features (v1.1)

| Feature | Method | Description |
|---------|--------|-------------|
| SLA Rule Configuration | `configure_sla_rules` | Attach per-log SLA rules (transition or case scope) with a max_duration_s limit |
| SLA Breach Scanning | `check_sla_breaches` | Scan all cases against configured SLA rules; returns per-rule breach rates and case IDs |
| Predictive Completion Time | `predict_completion_time` | Estimate remaining case duration (p50/p75/p95) using empirical prefix matching |
| Happy-Path Alignment | `compute_happy_path_alignment` | Levenshtein-distance alignment score per case against the most frequent variant |
| Deviation Root-Cause Analysis | `analyze_deviation_root_causes` | Fisher-test ranked attribute drivers that explain why cases deviate from the model |
| Process Cost Analysis | `analyze_process_costs` | Resource-rate × duration cost calculation per activity and variant using Decimal precision |
| Multi-Log Comparison | `compare_event_logs` | Structural diff of two event logs: Jaccard similarity, edge divergences, duration deltas |
| Streaming Conformance | `update_streaming_conformance` | Incremental per-event conformance with `conformance_deviation` NATS emit on first breach |
| Case Attribute Enrichment | `enrich_case_attributes` | Attach arbitrary business attributes (region, tier, amount) to cases for segmentation |
| Segmented Analysis | `segment_analysis` | Re-run variant or bottleneck analysis scoped to an attribute-filtered case subset |

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/pmin/health | Service health |
| GET | /api/pmin/logs | List event logs |
| POST | /api/pmin/logs | Create event log |
| GET | /api/pmin/logs/{id} | Get event log |
| PUT | /api/pmin/logs/{id} | Update event log |
| DELETE | /api/pmin/logs/{id} | Delete event log |
| POST | /api/pmin/logs/{id}/events | Ingest events |
| POST | /api/pmin/logs/{id}/events/nats | Ingest from NATS |
| GET | /api/pmin/logs/{id}/events | Query events |
| GET | /api/pmin/logs/{id}/cases/{case_id} | Case trace |
| POST | /api/pmin/logs/{id}/discover | Discover BPMN model |
| POST | /api/pmin/logs/{id}/bottlenecks | Bottleneck analysis |
| POST | /api/pmin/logs/{id}/variants | Variant discovery |
| GET | /api/pmin/logs/{id}/performance | Performance metrics |
| GET | /api/pmin/models | List BPMN models |
| GET | /api/pmin/models/{id} | Get model |
| DELETE | /api/pmin/models/{id} | Delete model |
| GET | /api/pmin/models/{id}/xml | Export BPMN 2.0 XML |
| POST | /api/pmin/models/{id}/simulate | Process simulation |
| POST | /api/pmin/conformance | Check conformance |
| GET | /api/pmin/conformance | List results |
| POST | /api/pmin/conformance/deviating-cases | Deviating cases |
| GET | /api/pmin/bottlenecks | List bottleneck reports |
| GET | /api/pmin/bottlenecks/{id} | Get report |
| GET | /api/pmin/variants | List variant analyses |
| GET | /api/pmin/variants/{id} | Get analysis |
| GET | /api/pmin/dashboard | Dashboard |
| GET | /api/pmin/audit | Audit trail |
| PUT | /api/pmin/logs/{id}/sla | Configure SLA rules |
| GET | /api/pmin/logs/{id}/sla/breaches | Check SLA breaches |
| POST | /api/pmin/logs/{id}/predict-completion | Predict completion time for in-flight cases |
| GET | /api/pmin/logs/{id}/alignment | Happy-path alignment scores |
| POST | /api/pmin/logs/{id}/root-cause | Deviation root-cause analysis |
| POST | /api/pmin/logs/{id}/costs | Process cost analysis |
| POST | /api/pmin/logs/compare | Compare two event logs |
| POST | /api/pmin/streaming-conformance | Streaming conformance update |
| POST | /api/pmin/logs/{id}/cases/enrich | Enrich case attributes |
| POST | /api/pmin/logs/{id}/segment | Segmented variant/bottleneck analysis |
