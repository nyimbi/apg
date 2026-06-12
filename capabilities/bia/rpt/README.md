# Report Builder

## Overview
The Report Builder capability (bia_rpt) provides parameterised report authoring, multi-format export (PDF/Excel/CSV/HTML/DOCX), report scheduling with 7 frequency options, governed distribution across 7 channels with external-distribution approval, run history, and a complete audit trail.

## Capability ID
`bia_rpt`

## Provides
- parameterised_report_authoring: 8 report types with sections and typed parameters
- report_scheduling: Daily/weekly/monthly/cron schedules with notification targets
- report_distribution: 7 channels with external-distribution approval gate
- multi_format_export: PDF, XLSX, CSV, HTML, DOCX, JSON, XML
- report_audit_trail: Full run history with output references and page counts
- report_template_library: Reusable report templates across teams
- report_versioning: Semantic versioning with state lifecycle
- report_bursting: Parameter-driven distribution to multiple recipients

## Requires
| Capability | Reason |
|------------|--------|
| auth | User identity and permission checks |
| audl | Audit all report runs and distributions |
| mten | Tenant context enforcement |
| conf | Runtime configuration |
| schd | Scheduled report execution |
| mqeb | Streaming report lifecycle events |
| ntfy | Distribution delivery notifications |
| bia_anl | Query and metric datasources for report data |

## Configuration
| Option | Default | Description |
|--------|---------|-------------|
| max_pages | 500 | Hard page limit per report run |
| max_parameters | 30 | Parameter limit per report |
| max_schedules_per_report | 5 | Schedule limit per report |
| require_approval_for_external | true | External distributions need approval |
| watermark_enabled | true | Add tenant watermark to PDF/HTML outputs |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/rpt/reports | GET/POST | List/create reports | bia_rpt:view/create |
| /api/bia/rpt/reports/<id>/publish | POST | Publish report | bia_rpt:edit |
| /api/bia/rpt/reports/<id>/run | POST | Run report | bia_rpt:run |
| /api/bia/rpt/schedules | GET/POST | List/create schedules | bia_rpt:schedule |
| /api/bia/rpt/distributions | GET/POST | List/create distributions | bia_rpt:distribute |
| /api/bia/rpt/distributions/<id>/approve | POST | Approve distribution | bia_rpt:distribute |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| run_requires_published | state=draft | deny |
| external_distribution_requires_approval | is_external + not approved | deny |
| max_pages_enforced | Pages exceed limit | deny |
| schedule_requires_published_report | state=draft | deny |
| archived_report_read_only | state=archived | deny |
| delete_published_requires_archive | state=published on delete | deny |

## Data Models
- ReportResponse: id, tenant_id, name, report_type, state, version, datasource_id, parameters
- ScheduleResponse: id, report_id, frequency, cron_expression, output_format, active
- DistributionResponse: id, report_id, channel, recipient, is_external, approved
- RunRecord: id, report_id, output_format, status, output_ref, run_duration_ms, page_count

## Streaming Events
- report_created, report_published, report_run_started, report_run_completed
- report_distributed, report_scheduled, report_archived
- distribution_approved, distribution_rejected

## Edge Cases Handled
- Published reports must be archived before deletion — prevents accidental loss
- External distributions (SFTP, S3, SharePoint) require explicit approval regardless of internal trust
- Draft reports cannot be run, scheduled, or distributed
- Deprecated reports reject run requests — users redirected to current version
- Page limit enforcement prevents runaway report generation from bloated queries

## Composability Notes
- Uses bia_anl queries and metrics as report datasources
- Distributes via ntfy for email/webhook delivery confirmation
- Report scheduling driven by schd capability
- bia_dsh published dashboards can be embedded as report attachments
- wflo can gate report publication with multi-step review

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Parameterised Report Templates with Semantic Variables** [Authoring]
- **I2. Incremental / Streaming Report Generation** [Performance]
- **I3. Monetary Column Precision with Decimal Arithmetic** [Data Integrity]
- **I4. Report Bursting — Per-Recipient Parameter Injection** [Distribution]
- **I5. Semantic Caching of Report Runs** [Performance]
- **I6. Column-Level Data Masking and PII Redaction** [Governance]
- **I7. Automated Anomaly Flagging in Report Outputs** [Intelligence]
- **I8. Cross-Report Diff — Version-to-Version Comparison** [Auditability]
- **I9. Natural Language Report Builder (Ollama-backed)** [Intelligence]
- **I10. Subscription Self-Service Portal with Preference Centre** [Distribution]
- **I11. Report Output Watermarking and Digital Signature** [Governance]
- **I12. Report Data Lineage Graph (Column-to-Source Tracing)** [Governance]
- **I13. Multi-Tenant Report Marketplace** [Composability]
- **I14. Adaptive Report Caching with Staleness Budget** [Performance]
- **I15. Scheduled Report Health Monitoring with SLA Alerting** [Reliability]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
