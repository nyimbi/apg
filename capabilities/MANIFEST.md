# APG Capability Manifest

**259 capabilities** across **28 domains**.

Each capability is independently installable: `pip install apg-<domain>-<code>`

## Quick Reference

| Domain | Capabilities |
|--------|-------------|
| [bia](#bia) | 8 |
| [ckm](#ckm) | 3 |
| [common](#common) | 81 |
| [composition](#composition) | 6 |
| [crm](#crm) | 1 |
| [eam](#eam) | 1 |
| [ecd](#ecd) | 1 |
| [education](#education) | 3 |
| [energy](#energy) | 6 |
| [fin](#fin) | 6 |
| [fintech](#fintech) | 30 |
| [government](#government) | 10 |
| [grc](#grc) | 6 |
| [hcm](#hcm) | 3 |
| [healthcare](#healthcare) | 9 |
| [int](#int) | 1 |
| [intel](#intel) | 20 |
| [loc](#loc) | 3 |
| [mining](#mining) | 6 |
| [mob](#mob) | 3 |
| [pde](#pde) | 1 |
| [pharma](#pharma) | 9 |
| [ppm](#ppm) | 6 |
| [realestate](#realestate) | 10 |
| [retail](#retail) | 5 |
| [scm](#scm) | 1 |
| [telecom](#telecom) | 10 |
| [transport](#transport) | 10 |

---

## BIA

### Analytics Engine `bia_anl`

> The Analytics Engine (bia_anl) provides the core analytical computation runtime for the BIA domain. It delivers ad-hoc SQL query execution, OLAP cube management, metric definition and calculation, multi-datasource connectivity, result caching, query scheduling, and governed analytical data access — all scoped to a tenant.

**Package**: `apg-bia-anl`  
**Path**: `capabilities/bia/anl`  
**Version**: 1.0.0  

**Provides:**
- `ad_hoc_query_execution`
- `olap_cube_management`
- `metric_definition_registry`
- `analytical_data_access`
- `query_result_cache`
- `datasource_connectivity`
- `saved_query_library`
- `query_scheduling`
- `result_export`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `schd`
- `mqeb`
- `moni`
- `nlpc`

**Service methods** (51 total):
`describe`, `evaluate`, `register_datasource`, `test_datasource`, `list_datasources`, `get_datasource`, `delete_datasource`, `save_query`, `get_query`, `list_queries`, `update_query`, `delete_query`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `query_type_supported`, `query_owner_required`, `cross_tenant_query_denied`, `query_timeout_enforced`, `max_rows_enforced`, `cube_dimension_supported`, ...

**UI Routes** (14):
- `/bia/anl/dashboard` — dashboard (bia_anl:view)
- `/bia/anl/query-builder` — query_builder (bia_anl:query)
- `/bia/anl/saved-queries` — saved_queries (bia_anl:query)
- `/bia/anl/saved-queries/<id>` — query_detail (bia_anl:query)
- `/bia/anl/cubes` — cube_explorer (bia_anl:cubes)
- `/bia/anl/cubes/<id>` — cube_detail (bia_anl:cubes)
- _8 more..._

**Streaming events** via `bytewax`:
`query_executed`, `query_saved`, `query_scheduled`, `cube_created`, `cube_refreshed`, ...

**Standalone usage:**
```bash
pip install apg-bia-anl
apg-bia-anl --port 8080
```

---

### Dashboard Management `bia_dsh`

> Dashboard Management (bia_dsh) provides dynamic dashboard creation, a widget library with 15 chart types, real-time data binding, responsive layout engines, cross-widget filtering, scheduled snapshot capture, and governed sharing — all tenant-scoped and audit-logged.

**Package**: `apg-bia-dsh`  
**Path**: `capabilities/bia/dsh`  
**Version**: 1.0.0  

**Provides:**
- `dashboard_creation`
- `widget_library`
- `real_time_data_binding`
- `responsive_layout_engine`
- `scheduled_snapshots`
- `cross_widget_filtering`
- `dashboard_sharing`
- `dashboard_export`
- `dashboard_embedding`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `schd`
- `mqeb`
- `ntfy`
- `bia_anl`

**Service methods** (42 total):
`describe`, `create_dashboard`, `get_dashboard`, `list_dashboards`, `update_dashboard`, `publish_dashboard`, `archive_dashboard`, `delete_dashboard`, `refresh_dashboard`, `share_dashboard`, `embed_dashboard`, `filter_context`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_dashboard_denied`, `public_access_requires_approval`, `widget_limit_enforced`, `widget_type_supported`, `widget_requires_datasource`, `layout_type_supported`, ...

**UI Routes** (14):
- `/bia/dsh/` — dashboard_home (bia_dsh:view)
- `/bia/dsh/gallery` — dashboard_gallery (bia_dsh:view)
- `/bia/dsh/<id>/view` — dashboard_view (bia_dsh:view)
- `/bia/dsh/<id>/build` — dashboard_builder (bia_dsh:edit)
- `/bia/dsh/new` — dashboard_new (bia_dsh:create)
- `/bia/dsh/widgets` — widget_library (bia_dsh:view)
- _8 more..._

**Streaming events** via `bytewax`:
`dashboard_created`, `dashboard_published`, `dashboard_archived`, `widget_added`, `widget_updated`, ...

**Standalone usage:**
```bash
pip install apg-bia-dsh
apg-bia-dsh --port 8080
```

---

### Data Warehouse `bia_dwh`

> The Data Warehouse capability (bia_dwh) provides dimensional modelling with star/snowflake/data-vault schema management, table registration, ETL job orchestration with multiple load strategies (full refresh, SCD type 1/2/3, incremental, merge), partition management, data quality rule enforcement with quarantine, and full lineage tracking.

**Package**: `apg-bia-dwh`  
**Path**: `capabilities/bia/dwh`  
**Version**: 1.0.0  

**Provides:**
- `dimensional_schema_management`
- `star_snowflake_schema_design`
- `etl_orchestration`
- `data_partitioning`
- `data_quality_enforcement`
- `lineage_tracking`
- `storage_tier_management`
- `warehouse_catalogue`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `schd`
- `mqeb`
- `moni`
- `comp`

**Service methods** (42 total):
`describe`, `create_schema`, `get_schema`, `list_schemas`, `update_schema`, `delete_schema`, `register_table`, `get_table`, `list_tables`, `update_table`, `delete_table`, `load_dimension`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_access_denied`, `schema_type_supported`, `schema_owner_required`, `schema_grain_required`, `table_type_supported`, `table_owner_required`, ...

**UI Routes** (13):
- `/bia/dwh/dashboard` — dashboard (bia_dwh:view)
- `/bia/dwh/schemas` — schemas (bia_dwh:schemas)
- `/bia/dwh/schemas/<id>` — schema_detail (bia_dwh:schemas)
- `/bia/dwh/tables` — tables (bia_dwh:tables)
- `/bia/dwh/tables/<id>` — table_detail (bia_dwh:tables)
- `/bia/dwh/etl` — etl_jobs (bia_dwh:etl)
- _7 more..._

**Streaming events** via `bytewax`:
`schema_created`, `schema_updated`, `table_registered`, `table_updated`, `etl_job_started`, ...

**Standalone usage:**
```bash
pip install apg-bia-dwh
apg-bia-dwh --port 8080
```

---

### Predictive Analytics `bia_pda`

> The Predictive Analytics capability (bia_pda) provides ML-based model training and deployment, demand and time-series forecasting, trend analysis, regression modelling, scenario simulation, and prediction serving — all tenant-scoped with full versioning, governance, and audit trails.

**Package**: `apg-bia-pda`  
**Path**: `capabilities/bia/pda`  
**Version**: 1.0.0  

**Provides:**
- `ml_model_training`
- `demand_forecasting`
- `trend_analysis`
- `regression_modelling`
- `scenario_simulation`
- `anomaly_prediction`
- `model_versioning`
- `prediction_serving`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `schd`
- `mqeb`
- `moni`
- `bia_anl`

**Service methods** (42 total):
`describe`, `create_model`, `train_model`, `evaluate_model`, `get_model`, `list_models`, `deploy_model`, `deprecate_model`, `delete_model`, `run_prediction`, `serve_prediction`, `batch_predict`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_model_access_denied`, `model_type_supported`, `model_owner_required`, `training_data_required`, `min_samples_enforced`, `forecast_horizon_supported`, ...

**UI Routes** (13):
- `/bia/pda/dashboard` — dashboard (bia_pda:view)
- `/bia/pda/models` — models (bia_pda:models)
- `/bia/pda/models/<id>` — model_detail (bia_pda:models)
- `/bia/pda/models/train` — model_train (bia_pda:train)
- `/bia/pda/forecasts` — forecasts (bia_pda:forecasts)
- `/bia/pda/forecasts/<id>` — forecast_detail (bia_pda:forecasts)
- _7 more..._

**Streaming events** via `bytewax`:
`model_training_started`, `model_trained`, `model_deployed`, `model_deprecated`, `forecast_generated`, ...

**Standalone usage:**
```bash
pip install apg-bia-pda
apg-bia-pda --port 8080
```

---

### Prescriptive Analytics `bia_psa`

> The Prescriptive Analytics capability (bia_psa) provides optimisation engines (LP, IP, GA, RL), decision support with explainability, recommendation action management with approval workflows, and what-if analysis — all tenant-scoped with mandatory governance and full audit.

**Package**: `apg-bia-psa`  
**Path**: `capabilities/bia/psa`  
**Version**: 1.0.0  

**Provides:**
- `optimisation_engine`
- `decision_support_system`
- `recommendation_actions`
- `whatif_analysis`
- `constraint_management`
- `multi_objective_analysis`
- `allocation_optimisation`
- `process_improvement_recommendations`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `mqeb`
- `moni`
- `wflo`
- `bia_pda`

**Service methods** (41 total):
`describe`, `create_optimisation`, `get_optimisation`, `list_optimisations`, `run_optimisation`, `archive_optimisation`, `delete_optimisation`, `optimisation_problem`, `linear_programme`, `simulation_run`, `decision_tree_analysis`, `sensitivity_analysis`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_analysis_denied`, `optimisation_type_supported`, `optimisation_owner_required`, `decision_type_supported`, `recommendation_type_supported`, `unapproved_recommendation_action_denied`, ...

**UI Routes** (13):
- `/bia/psa/dashboard` — dashboard (bia_psa:view)
- `/bia/psa/optimisations` — optimisations (bia_psa:optimise)
- `/bia/psa/optimisations/<id>` — optimisation_detail (bia_psa:optimise)
- `/bia/psa/decisions` — decisions (bia_psa:decisions)
- `/bia/psa/decisions/<id>` — decision_detail (bia_psa:decisions)
- `/bia/psa/recommendations` — recommendations (bia_psa:recommendations)
- _7 more..._

**Streaming events** via `bytewax`:
`optimisation_started`, `optimisation_completed`, `decision_recorded`, `recommendation_generated`, `recommendation_approved`, ...

**Standalone usage:**
```bash
pip install apg-bia-psa
apg-bia-psa --port 8080
```

---

### Report Builder `bia_rpt`

> The Report Builder capability (bia_rpt) provides parameterised report authoring, multi-format export (PDF/Excel/CSV/HTML/DOCX), report scheduling with 7 frequency options, governed distribution across 7 channels with external-distribution approval, run history, and a complete audit trail.

**Package**: `apg-bia-rpt`  
**Path**: `capabilities/bia/rpt`  
**Version**: 1.0.0  

**Provides:**
- `parameterised_report_authoring`
- `report_scheduling`
- `report_distribution`
- `multi_format_export`
- `report_audit_trail`
- `report_template_library`
- `report_versioning`
- `report_bursting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `schd`
- `mqeb`
- `ntfy`
- `bia_anl`

**Service methods** (42 total):
`describe`, `create_report`, `get_report`, `list_reports`, `update_report`, `publish_report`, `archive_report`, `delete_report`, `add_column`, `list_columns`, `remove_column`, `apply_filter`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_report_denied`, `report_type_supported`, `report_owner_required`, `output_format_supported`, `run_requires_published`, `schedule_frequency_supported`, ...

**UI Routes** (13):
- `/bia/rpt/dashboard` — dashboard (bia_rpt:view)
- `/bia/rpt/reports` — report_library (bia_rpt:view)
- `/bia/rpt/reports/<id>` — report_detail (bia_rpt:view)
- `/bia/rpt/reports/<id>/build` — report_builder (bia_rpt:edit)
- `/bia/rpt/reports/new` — report_new (bia_rpt:create)
- `/bia/rpt/reports/<id>/run` — report_run (bia_rpt:run)
- _7 more..._

**Streaming events** via `bytewax`:
`report_created`, `report_published`, `report_run_started`, `report_run_completed`, `report_distributed`, ...

**Standalone usage:**
```bash
pip install apg-bia-rpt
apg-bia-rpt --port 8080
```

---

### Self-Service BI `bia_sbi`

> The Self-Service BI capability (bia_sbi) provides a drag-and-drop visual chart builder, natural-language query (NLQ) processing, a governed data catalogue with tiered access control, user sandboxes with row limits and auto-expiry, and a template gallery — giving business users governed self-service analytics without requiring SQL expertise.

**Package**: `apg-bia-sbi`  
**Path**: `capabilities/bia/sbi`  
**Version**: 1.0.0  

**Provides:**
- `drag_drop_visual_builder`
- `natural_language_queries`
- `governed_data_catalogue`
- `user_sandboxes`
- `template_gallery`
- `self_service_chart_creation`
- `catalogue_governance`
- `embedded_analytics`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `nlpc`
- `mqeb`
- `ntfy`
- `bia_anl`

**Service methods** (42 total):
`describe`, `natural_language_query`, `submit_nlq`, `list_nlq_history`, `suggested_insights`, `drag_and_drop_report_create`, `data_catalogue_search`, `dataset_preview`, `bookmark_report`, `list_bookmarks`, `remove_bookmark`, `personalised_feed`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_access_denied`, `chart_type_supported`, `datasource_mode_supported`, `catalogue_approval_required`, `sandbox_row_limit_enforced`, `sandbox_limit_per_user_enforced`, ...

**UI Routes** (13):
- `/bia/sbi/` — home (bia_sbi:view)
- `/bia/sbi/builder` — builder (bia_sbi:build)
- `/bia/sbi/workspaces/<id>` — workspace (bia_sbi:build)
- `/bia/sbi/ask` — nlq (bia_sbi:query)
- `/bia/sbi/catalogue` — catalogue (bia_sbi:catalogue)
- `/bia/sbi/catalogue/<id>` — catalogue_detail (bia_sbi:catalogue)
- _7 more..._

**Streaming events** via `bytewax`:
`workspace_created`, `chart_created`, `nlq_submitted`, `nlq_answered`, `catalogue_entry_created`, ...

**Standalone usage:**
```bash
pip install apg-bia-sbi
apg-bia-sbi --port 8080
```

---

### Time Series Analytics `bia_tsa`

> The Time Series Analytics capability (bia_tsa) provides high-frequency time-series stream ingestion via 7 protocols, configurable anomaly detection with 8 methods, seasonality decomposition (trend/seasonality/residual/cyclical), time-series forecasting with 7 models, stream windowing, gap-filling interpolation, and real-time alerting — all tenant-scoped and bytewax-streamed.

**Package**: `apg-bia-tsa`  
**Path**: `capabilities/bia/tsa`  
**Version**: 1.0.0  

**Provides:**
- `high_frequency_time_series_ingestion`
- `anomaly_detection`
- `seasonality_decomposition`
- `time_series_forecasting`
- `stream_windowing`
- `multi_stream_correlation`
- `gap_filling_interpolation`
- `real_time_alerting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `mqeb`
- `moni`
- `ntfy`
- `schd`

**Service methods** (42 total):
`describe`, `register_stream`, `get_stream`, `list_streams`, `pause_stream`, `resume_stream`, `archive_stream`, `ingest_data`, `ingest_time_series`, `configure_anomaly_detection`, `list_anomaly_configs`, `detect_anomaly`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_stream_access_denied`, `ingestion_protocol_supported`, `stream_frequency_supported`, `stream_owner_required`, `max_streams_enforced`, `anomaly_method_supported`, ...

**UI Routes** (13):
- `/bia/tsa/dashboard` — dashboard (bia_tsa:view)
- `/bia/tsa/streams` — streams (bia_tsa:streams)
- `/bia/tsa/streams/<id>` — stream_detail (bia_tsa:streams)
- `/bia/tsa/streams/<id>/explore` — stream_explorer (bia_tsa:streams)
- `/bia/tsa/anomalies` — anomaly_detection (bia_tsa:anomalies)
- `/bia/tsa/anomalies/<id>` — anomaly_detail (bia_tsa:anomalies)
- _7 more..._

**Streaming events** via `bytewax`:
`stream_registered`, `stream_data_ingested`, `anomaly_detected`, `anomaly_confirmed`, `decomposition_completed`, ...

**Standalone usage:**
```bash
pip install apg-bia-tsa
apg-bia-tsa --port 8080
```

---

## CKM

### Notification System `ckm_not`

> The Notification System (`ckm_not`) is a multi-channel notification engine that manages the full lifecycle of notifications across email, SMS, push, in-app, voice, webhook, WhatsApp, Slack, Teams, and web push channels. It provides template-driven content authoring, campaign orchestration, recipient preference enforcement, and delivery governance — all within a tenant-scoped, consent-enforced, audit-trailed runtime.

**Package**: `apg-ckm-not`  
**Path**: `capabilities/ckm/not`  
**Version**: 1.0.0  

**Provides:**
- `notification_delivery`
- `template_management`
- `campaign_orchestration`
- `preference_center`
- `channel_provider_registry`
- `engagement_analytics`
- `notification_agents`

**Requires:**
- `auth`
- `conf`
- `encr`
- `audl`

**Service methods** (53 total):
`send_notification_request`, `send_bulk_notifications`, `process_with_semaphore`, `execute_campaign`, `get_delivery_analytics`, `track_engagement_event`, `get_user_preferences`, `update_user_preferences`, `get_service_health`, `_get_user_preferences`, `_optimize_channel_selection`, `_execute_multi_channel_delivery`, ...

**Governance rules** (17 total):
`tenant_context_required`, `template_requires_channel_content`, `template_requires_variable_schema`, `external_delivery_requires_consent`, `delivery_channel_must_be_allowed`, `suppressed_recipient_blocks_delivery`, `quiet_hours_requires_deferral`, `campaign_requires_audience_policy`, ...

**UI Routes** (11):
- `/ckm-not/dashboard` — dashboard (ckm_not:view)
- `/ckm-not/templates` — templates (ckm_not:manage_templates)
- `/ckm-not/campaigns` — campaigns (ckm_not:manage_campaigns)
- `/ckm-not/deliveries` — deliveries (ckm_not:send)
- `/ckm-not/preferences` — preferences (ckm_not:view_preferences)
- `/ckm-not/providers` — providers (ckm_not:admin)
- _5 more..._

**Streaming events** via `bytewax`:
`notification_template_created`, `notification_template_approved`, `notification_campaign_requested`, `notification_campaign_approved`, `notification_delivery_requested`, ...

**Standalone usage:**
```bash
pip install apg-ckm-not
apg-ckm-not --port 8080
```

---

### Real-Time Collaboration `ckm_rtc`

> The Real-Time Collaboration capability (`ckm_rtc`) provides synchronous and asynchronous collaboration infrastructure across chat, presence, voice, video, screen sharing, co-editing, and whiteboarding modes. It manages collaboration sessions with participant policies, real-time messaging with audit retention, and structured decision capture with voting and consensus building — all scoped per tenant and wired into the APG audit trail.

**Package**: `apg-ckm-rtc`  
**Path**: `capabilities/ckm/rtc`  
**Version**: 1.0.0  

**Provides:**
- `collaboration_sessions`
- `presence_awareness`
- `real_time_messaging`
- `media_collaboration`
- `decision_capture`
- `page_collaboration`
- `rtc_agents`

**Requires:**
- `auth`
- `conf`
- `audl`
- `ckm_not`

**Service methods** (40 total):
`_validate_permissions`, `create_session`, `join_session`, `end_session`, `enable_page_collaboration`, `delegate_form_field`, `request_assistance`, `start_video_call`, `start_screen_share`, `start_recording`, `setup_teams_integration`, `setup_zoom_integration`, ...

**Governance rules** (17 total):
`tenant_context_required`, `session_requires_owner`, `session_requires_participant_policy`, `join_requires_allowed_participant`, `presence_requires_heartbeat`, `message_requires_active_session`, `sensitive_message_requires_review`, `screen_share_requires_permission`, ...

**UI Routes** (11):
- `/ckm-rtc/dashboard` — dashboard (ckm_rtc:view)
- `/ckm-rtc/rooms` — rooms (ckm_rtc:manage_rooms)
- `/ckm-rtc/presence` — presence (ckm_rtc:view)
- `/ckm-rtc/messages` — messages (ckm_rtc:participate)
- `/ckm-rtc/media` — media (ckm_rtc:participate)
- `/ckm-rtc/decisions` — decisions (ckm_rtc:participate)
- _5 more..._

**Streaming events** via `bytewax`:
`rtc_session_created`, `rtc_participant_joined`, `rtc_presence_updated`, `rtc_message_posted`, `rtc_screen_share_started`, ...

**Standalone usage:**
```bash
pip install apg-ckm-rtc
apg-ckm-rtc --port 8080
```

---

### Workflow Automation `ckm_wfa`

> The Workflow Automation capability (`ckm_wfa`) provides a BPMN 2.0-compliant workflow engine for defining, deploying, and operating business processes across the APG platform. It covers the full lifecycle from process design (drag-and-drop visual designer, BPMN XML/JSON definitions, version control) through execution (instance management, task queues, SLA tracking) to governance (approval chains with independent-reviewer requirements, exception escalation, and AI-powered optimization recommendations).

**Package**: `apg-ckm-wfa`  
**Path**: `capabilities/ckm/wfa`  
**Version**: 1.0.0  

**Provides:**
- `workflow_definitions`
- `workflow_instances`
- `task_orchestration`
- `approval_governance`
- `exception_management`
- `workflow_analytics`
- `wfa_agents`

**Requires:**
- `auth`
- `conf`
- `audl`
- `ckm_not`
- `ckm_rtc`

**Service methods** (46 total):
`validate_user_permissions`, `_validate_permissions_with_auth_service`, `_call_auth_method`, `_coerce_auth_result`, `_validate_permissions_over_http`, `log_audit_event`, `send_notification`, `create_process_definition`, `get_process_definition`, `list_process_definitions`, `start_process_instance`, `create_task`, ...

**Governance rules** (21 total):
`tenant_context_required`, `definition_requires_owner`, `definition_requires_version`, `activation_requires_approval`, `instance_requires_active_definition`, `instance_requires_initiator`, `human_task_requires_assignee`, `sla_task_requires_due_at`, ...

**UI Routes** (12):
- `/ckm-wfa/dashboard` — dashboard (ckm_wfa:view)
- `/ckm-wfa/designer` — designer (ckm_wfa:design)
- `/ckm-wfa/definitions` — definitions (ckm_wfa:design)
- `/ckm-wfa/instances` — instances (ckm_wfa:operate)
- `/ckm-wfa/tasks` — tasks (ckm_wfa:participate)
- `/ckm-wfa/approvals` — approvals (ckm_wfa:approve)
- _6 more..._

**Streaming events** via `bytewax`:
`workflow_definition_created`, `workflow_definition_activated`, `workflow_instance_started`, `workflow_task_created`, `workflow_task_completed`, ...

**Standalone usage:**
```bash
pip install apg-ckm-wfa
apg-ckm-wfa --port 8080
```

---

## COMMON

### Accessibility Services `accs`

> ACCS makes accessibility governance an executable APG capability. It gives generated applications a tenant-scoped way to register accessibility standards, register UI/content/media targets, run deterministic audits, record findings,

**Package**: `apg-common-accs`  
**Path**: `capabilities/common/accs`  
**Version**: 1.0.0  

**Provides:**
- `accessibility_audits`
- `remediation_workflows`
- `accessibility_exceptions`
- `assistive_metadata`
- `media_accessibility`
- `standards_governance`
- `accessibility_agents`

**Requires:**
- `them`
- `i18n`
- `nlpc`

**Service methods** (40 total):
`describe`, `evaluate`, `register_standard`, `list_standards`, `register_target`, `list_targets`, `list_records`, `create_record`, `run_audit`, `list_audits`, `record_finding`, `list_findings`, ...

**Governance rules** (20 total):
`tenant_context_required`, `audit_requires_standard`, `violation_requires_remediation_owner`, `published_ui_requires_contrast`, `media_requires_captions`, `critical_issue_requires_review`, `finding_closure_requires_resolution`, `accessibility_exception_requires_expiry`, ...

**UI Routes** (12):
- `/accs/dashboard` — dashboard (accs:view)
- `/accs/audits` — audits (accs:audit)
- `/accs/findings` — findings (accs:view)
- `/accs/remediation` — remediation (accs:remediate)
- `/accs/exceptions` — exceptions (accs:review)
- `/accs/assistive` — assistive (accs:audit)
- _6 more..._

**Streaming events** via `bytewax`:
`standard_registered`, `target_registered`, `audit_completed`, `finding_recorded`, `remediation_updated`, ...

**Standalone usage:**
```bash
pip install apg-common-accs
apg-common-accs --port 8080
```

---

### AI Agent Composition `agnt`

> AGNT makes AI agents first-class APG citizens. It gives generated applications a provider-neutral way to register agent runtimes, request approval for external providers, declare agents with models and contracts, compose teams

**Package**: `apg-common-agnt`  
**Path**: `capabilities/common/agnt`  
**Version**: 1.0.0  

**Provides:**
- `agent_registry`
- `runtime_registry`
- `agent_teams`
- `handoff_graphs`
- `execution_plans`
- `execution_runs`
- `runtime_approval_governance`

**Requires:**
- `aicr`
- `sbox`
- `audl`

**Service methods** (52 total):
`describe`, `evaluate`, `register_runtime`, `list_runtimes`, `request_runtime_approval`, `decide_runtime_approval`, `list_runtime_approvals`, `list_audit_events`, `register_agent`, `list_agents`, `register_team`, `list_teams`, ...

**Governance rules** (22 total):
`tenant_context_required`, `agent_requires_model`, `agent_requires_system_prompt`, `agent_requires_tool_allowlist`, `agent_requires_io_contract`, `agent_requires_memory_policy`, `agent_runtime_must_be_registered`, `runtime_requires_cost_limit`, ...

**UI Routes** (12):
- `/agnt/dashboard` — dashboard (agnt:view)
- `/agnt/agents` — agents (agnt:compose)
- `/agnt/teams` — teams (agnt:compose)
- `/agnt/handoffs` — handoffs (agnt:compose)
- `/agnt/runtimes` — runtimes (agnt:manage_runtimes)
- `/agnt/executions` — executions (agnt:run)
- _6 more..._

**Streaming events** via `bytewax`:
`runtime_approval_requested`, `runtime_approval_decided`, `runtime_registered`, `agent_registered`, `team_registered`, ...

**Standalone usage:**
```bash
pip install apg-common-agnt
apg-common-agnt --port 8080
```

---

### AI Core Framework `aicr`

> AICR is the APG AI control plane. It lets generated applications register AI services, providers, models, workflows, and agent runtimes while enforcing tenant, policy, approval, audit, and observability guardrails.

**Package**: `apg-common-aicr`  
**Path**: `capabilities/common/aicr`  
**Version**: 1.0.0  

**Provides:**
- `ai_core`
- `model_inference`
- `model_metrics`
- `ai_agent_composition`

**Requires:**
- `conf`
- `auth`
- `mqeb`
- `moni`

**Service methods** (113 total):
`nonblocking_cpu_percent`, `can_allocate`, `allocate`, `deallocate`, `get_utilization`, `duration_ms`, `add_intermediate_result`, `complete_session`, `register_model`, `get_model`, `list_models`, `update_performance`, ...

**Governance rules** (41 total):
`tenant_context_required`, `service_registration_requires_owner`, `service_registration_requires_endpoint`, `provider_type_must_be_supported`, `provider_requires_owner`, `provider_requires_credential_vault`, `provider_requires_egress_policy`, `model_requires_owner`, ...

**UI Routes** (15):
- `/aicr/dashboard` — dashboard (aicr:view)
- `/aicr/services` — services (aicr:manage_services)
- `/aicr/providers` — providers (aicr:manage_services)
- `/aicr/models` — models (aicr:view_models)
- `/aicr/model-metrics` — model_metrics (aicr:govern)
- `/aicr/inference` — inference (aicr:run_inference)
- _9 more..._

**Streaming events** via `bytewax`:
`service_registered`, `service_updated`, `service_retired`, `provider_registered`, `provider_updated`, ...

**Standalone usage:**
```bash
pip install apg-common-aicr
apg-common-aicr --port 8080
```

---

### Anomaly Detection `anom`

> ANOM is the APG capability for governed anomaly detection across monitored metrics, events, traces, forecast residuals, and security signals. It lets generated applications register monitoring sources, build statistical

**Package**: `apg-common-anom`  
**Path**: `capabilities/common/anom`  
**Version**: 1.0.0  

**Provides:**
- `anomaly_detection`
- `signal_intelligence`
- `anomaly_agent_composition`

**Requires:**
- `pred`
- `aicr`
- `moni`
- `conf`

**Service methods** (42 total):
`describe`, `evaluate`, `register_source`, `list_sources`, `create_baseline`, `list_baselines`, `reset_baseline`, `detect`, `list_signals`, `list_records`, `create_record`, `open_investigation`, ...

**Governance rules** (39 total):
`tenant_context_required`, `source_requires_name`, `source_requires_owner`, `source_requires_kind`, `source_kind_requires_review`, `baseline_requires_source`, `baseline_requires_metric`, `baseline_requires_history`, ...

**UI Routes** (14):
- `/anom/dashboard` — dashboard (anom:view)
- `/anom/sources` — sources (anom:tune)
- `/anom/baselines` — baselines (anom:tune)
- `/anom/detector` — detector (anom:detect)
- `/anom/signals` — signals (anom:detect)
- `/anom/investigations` — investigations (anom:investigate)
- _8 more..._

**Streaming events** via `bytewax`:
`source_registered`, `source_updated`, `baseline_created`, `baseline_reset`, `anomaly_detected`, ...

**Standalone usage:**
```bash
pip install apg-common-anom
apg-common-anom --port 8080
```

---

### API Gateway & Management `apig`

> APIG is APG's governed API gateway control plane. It lets generated applications register upstream services and API consumers, request routes, enforce security and traffic guardrails, review high quotas and canary traffic

**Package**: `apg-common-apig`  
**Path**: `capabilities/common/apig`  
**Version**: 1.0.0  

**Provides:**
- `api_gateway`
- `traffic_management`
- `gateway_agent_composition`
- `review_evidence`

**Requires:**
- `auth`
- `moni`
- `mqeb`
- `conf`

**Service methods** (55 total):
`initialize`, `_initialize_apg_connections`, `_initialize_core_components`, `_initialize_monitoring`, `_load_initial_configuration`, `create_gateway`, `process_request`, `create_policy_from_natural_language`, `get_service_status`, `shutdown`, `_find_gateway_for_request`, `_authenticate_request`, ...

**Governance rules** (33 total):
`tenant_context_required`, `upstream_requires_owner`, `upstream_requires_https`, `upstream_requires_health_check`, `consumer_requires_owner`, `consumer_requires_credential_rotation`, `restricted_consumer_requires_rbac`, `route_requires_owner`, ...

**UI Routes** (15):
- `/apig/dashboard` — dashboard (apig:view)
- `/apig/routes` — routes (apig:manage_routes)
- `/apig/upstreams` — upstreams (apig:manage_routes)
- `/apig/consumers` — consumers (apig:manage_security)
- `/apig/traffic` — traffic (apig:manage_traffic)
- `/apig/security` — security (apig:manage_security)
- _9 more..._

**Streaming events** via `bytewax`:
`upstream_registered`, `upstream_updated`, `upstream_retired`, `consumer_registered`, `consumer_updated`, ...

**Standalone usage:**
```bash
pip install apg-common-apig
apg-common-apig --port 8080
```

---

### Audit Logging `audl`

> **Comprehensive audit logging system providing secure, scalable, and queryable audit trail capabilities for the APG platform.**

**Package**: `apg-common-audl`  
**Path**: `capabilities/common/audl`  
**Version**: 1.0.0  

**Provides:**
- `audl_operations`
- `audit_agents`
- `review_evidence`

**Service methods** (58 total):
`subscribe_domain_events`, `log_event`, `immutable_log_write`, `audit_trail_search`, `matches`, `tamper_detection`, `compliance_report`, `gdpr_data_subject_access`, `right_to_erasure_audit_impact`, `evidence_package_export`, `retention_enforcement`, `cross_tenant_audit_correlation`, ...

**Governance rules** (21 total):
`require_tenant_context`, `immutable_events_require_checksum`, `legal_hold_blocks_purge`, `regulated_exports_require_masking`, `critical_events_require_escalation`, `high_volume_ingestion_requires_stream_processing`, `bytewax_event_stream_required`, `audit_agent_runtime_supported`, ...

**UI Routes** (12):
- `/audit/dashboard` — dashboard (audl:view)
- `/audit/events` — events (audl:view)
- `/audit/timeline` — timeline (audl:view)
- `/audit/investigations` — investigations (audl:investigate)
- `/audit/legal-holds` — legal_holds (audl:hold)
- `/audit/exports` — exports (audl:export)
- _6 more..._

**Streaming events** via `bytewax`:
`audit_event_recorded`, `audit_batch_ingested`, `investigation_created`, `investigation_closed`, `legal_hold_placed`, ...

**Standalone usage:**
```bash
pip install apg-common-audl
apg-common-audl --port 8080
```

---

### Audio Processing `audp`

> AUDP is APG's governed audio-processing capability. It gives generated applications a dependency-light way to compose transcription, synthesis, voice-cloning consent, audio analysis, model policy, human review,

**Package**: `apg-common-audp`  
**Path**: `capabilities/common/audp`  
**Version**: 1.0.0  

**Provides:**
- `audio_transcription`
- `voice_synthesis`
- `audio_analysis`
- `speaker_diarization`
- `audio_enhancement`
- `audio_consent_governance`
- `audio_review_governance`
- `audio_agents`

**Requires:**
- `aicr`
- `nlpc`
- `mlcm`

**Service methods** (115 total):
`create_transcription_job`, `start_transcription_job`, `_process_transcription_job`, `_get_transcription_model`, `_transcribe_with_whisper`, `_transcribe_with_google`, `_transcribe_with_azure`, `_transcribe_with_deepgram`, `_transcribe_with_custom_model`, `_process_transcription_results`, `_generate_sentence_segments`, `_calculate_processing_cost`, ...

**Governance rules** (20 total):
`tenant_context_required`, `recording_consent_required`, `voice_cloning_requires_consent`, `synthetic_audio_requires_watermark`, `synthetic_audio_requires_release_review`, `audio_model_requires_policy`, `low_transcription_confidence_requires_review`, `audio_retention_policy_required`, ...

**UI Routes** (13):
- `/audp/dashboard` — dashboard (audp:view)
- `/audp/transcription` — transcription (audp:transcribe)
- `/audp/synthesis` — synthesis (audp:synthesize)
- `/audp/analysis` — analysis (audp:analyze)
- `/audp/sessions` — sessions (audp:view)
- `/audp/models` — models (audp:manage_models)
- _7 more..._

**Streaming events** via `bytewax`:
`audio_consent_recorded`, `audio_model_policy_attached`, `transcription_requested`, `transcript_review_requested`, `transcript_review_decided`, ...

**Standalone usage:**
```bash
pip install apg-common-audp
apg-common-audp --port 8080
```

---

### Authentication & RBAC `auth`

> AUTH is the APG identity, session, role, access-decision, privacy-budget, and security-agent governance capability. It gives generated applications a dependency-light control plane for registering tenant identities, defining

**Package**: `apg-common-auth`  
**Path**: `capabilities/common/auth`  
**Version**: 1.0.0  

**Provides:**
- `identity_registry`
- `role_governance`
- `session_control`
- `access_decisions`
- `privacy_budget_governance`
- `security_agents`
- `review_evidence`

**Requires:**
- `audl`
- `mten`
- `keym`

**Service methods** (45 total):
`describe`, `evaluate`, `register_identity`, `list_identities`, `define_role`, `list_roles`, `request_role_assignment_approval`, `decide_role_assignment_approval`, `list_role_assignment_approvals`, `assign_role`, `list_role_assignments`, `start_session`, ...

**Governance rules** (23 total):
`locked_accounts_denied`, `privileged_access_requires_mfa`, `high_risk_sessions_require_step_up`, `elevated_role_assignment_requires_approval`, `role_assignment_approval_requires_independent_reviewer`, `untrusted_federation_denied`, `cross_tenant_access_requires_membership`, `privacy_queries_require_budget`, ...

**UI Routes** (21):
- `/auth/access/login` — login (public)
- `/auth/access/dashboard` — dashboard (auth:view)
- `/auth/roles/workbench` — role_workbench (auth:manage_roles)
- `/auth/roles/approvals` — role_approvals (auth:approve_roles)
- `/auth/sessions` — sessions (auth:manage_sessions)
- `/auth/access/decisions` — access_decisions (auth:view)
- _15 more..._

**Streaming events** via `bytewax`:
`identity_registered`, `role_defined`, `role_assignment_approval_requested`, `role_assignment_approval_decided`, `role_assigned`, ...

**Standalone usage:**
```bash
pip install apg-common-auth
apg-common-auth --port 8080
```

---

### Blockchain Ledger Services `bclg`

> BCLG provides governed distributed-ledger services for APG applications. It covers tenant ledger registration, key-custody binding, signed transaction submission, high-value transaction review, smart contract deployment approval,

**Package**: `apg-common-bclg`  
**Path**: `capabilities/common/bclg`  
**Version**: 1.0.0  

**Provides:**
- `ledger_registry`
- `transaction_governance`
- `smart_contract_governance`
- `key_custody_governance`
- `ledger_audit`
- `ledger_agents`

**Requires:**
- `encr`
- `keym`
- `comp`

**Service methods** (43 total):
`describe`, `evaluate`, `register_ledger`, `list_ledgers`, `bind_key_custody`, `list_key_custody`, `submit_transaction`, `request_transaction_review`, `decide_transaction_review`, `approve_transaction`, `list_transaction_reviews`, `list_transactions`, ...

**Governance rules** (20 total):
`tenant_context_required`, `ledger_requires_owner`, `transaction_requires_signature`, `key_custody_required`, `contract_requires_review`, `high_value_transaction_requires_review`, `transaction_review_requires_independent_reviewer`, `contract_deployment_review_requires_independent_reviewer`, ...

**UI Routes** (12):
- `/bclg/dashboard` — dashboard (bclg:view)
- `/bclg/ledgers` — ledgers (bclg:manage_ledgers)
- `/bclg/transactions` — transactions (bclg:transact)
- `/bclg/transactions/reviews` — transaction_reviews (bclg:review_transactions)
- `/bclg/contracts` — contracts (bclg:manage_contracts)
- `/bclg/contracts/reviews` — contract_reviews (bclg:review_contracts)
- _6 more..._

**Streaming events** via `bytewax`:
`ledger_registered`, `key_custody_bound`, `transaction_submitted`, `transaction_review_requested`, `transaction_review_decided`, ...

**Standalone usage:**
```bash
pip install apg-common-bclg
apg-common-bclg --port 8080
```

---

### Biometric Processing `biop`

> BIOP provides governed biometric processing for APG applications. It covers consent, enrollment, encrypted template metadata, liveness-backed verification, match confidence review, cross-border privacy review, first-class biometric governance agents, Bytewax lifecycle batch validation, revocation, retirement, audit evidence, and route-ready UI models.

**Package**: `apg-common-biop`  
**Path**: `capabilities/common/biop`  
**Version**: 1.0.0  

**Provides:**
- `biometric_processing`
- `biometric_verification`
- `biometric_agent_composition`

**Requires:**
- `mfau`
- `cvsn`
- `aicr`
- `encr`
- `audl`
- `conf`

**Service methods** (45 total):
`uuid7str`, `_audit`, `register_user`, `enrol_template`, `verify`, `issue_liveness_challenge`, `complete_liveness_challenge`, `multimodal_fusion`, `quality_assess`, `template_age_check`, `biometric_update`, `duplicate_detect`, ...

**Governance rules** (48 total):
`tenant_context_required`, `consent_requires_subject`, `consent_requires_purpose`, `consent_requires_modalities`, `consent_requires_jurisdictions`, `consent_requires_evidence`, `biometric_processing_requires_consent`, `enrollment_requires_active_consent`, ...

**UI Routes** (14):
- `/biop/dashboard` — dashboard (biop:view)
- `/biop/users` — users (biop:view)
- `/biop/consents` — consents (biop:manage_consent)
- `/biop/enrollments` — enrollments (biop:enroll)
- `/biop/templates` — templates (biop:manage_templates)
- `/biop/verification` — verification (biop:verify)
- _8 more..._

**Streaming events** via `bytewax`:
`consent_recorded`, `consent_revoked`, `template_enrolled`, `template_retired`, `verification_completed`, ...

**Standalone usage:**
```bash
pip install apg-common-biop
apg-common-biop --port 8080
```

---

### Backup and Restore `bkup`

> BKUP provides governed backup, restore, retention, and continuity operations for APG applications. It covers tenant backup plans, encrypted snapshots, restore approval, stale restore-test review, retention disposition, legal-hold

**Package**: `apg-common-bkup`  
**Path**: `capabilities/common/bkup`  
**Version**: 1.0.0  

**Provides:**
- `backup_plan_governance`
- `snapshot_vault`
- `restore_governance`
- `retention_governance`
- `continuity_reporting`
- `backup_agents`

**Requires:**
- `encr`
- `conf`
- `audl`

**Service methods** (44 total):
`uuid7str`, `_audit`, `create_backup_plan`, `backup_schedule`, `backup_run`, `incremental_backup`, `differential_backup`, `restore_from`, `verify_backup`, `test_restore`, `retention_policy`, `encryption_at_rest`, ...

**Governance rules** (21 total):
`tenant_context_required`, `backup_plan_requires_owner`, `snapshot_requires_encryption`, `snapshot_requires_integrity`, `restore_requires_integrity_check`, `production_restore_requires_approval`, `stale_restore_test_requires_review`, `restore_review_requires_independent_reviewer`, ...

**UI Routes** (13):
- `/bkup/dashboard` — dashboard (bkup:view)
- `/bkup/plans` — plans (bkup:manage_plans)
- `/bkup/snapshots` — snapshots (bkup:view)
- `/bkup/backup` — backup (bkup:run_backup)
- `/bkup/restore` — restore (bkup:restore)
- `/bkup/restore/approvals` — restore_approvals (bkup:approve_restore)
- _7 more..._

**Streaming events** via `bytewax`:
`backup_plan_created`, `snapshot_created`, `restore_approval_requested`, `restore_approval_decided`, `restore_requested`, ...

**Standalone usage:**
```bash
pip install apg-common-bkup
apg-common-bkup --port 8080
```

---

### Cache Management `cach`

> CACH is APG's cache governance and runtime-adapter capability. It gives generated applications a tenant-aware way to register cache namespaces, enforce entry admission rules, manage warming and eviction reviews, publish UI metadata,

**Package**: `apg-common-cach`  
**Path**: `capabilities/common/cach`  
**Version**: 1.0.0  

**Provides:**
- `cache_governance`
- `cache_runtime_adapters`
- `cache_agent_composition`
- `review_evidence`

**Requires:**
- `conf`
- `auth`
- `audl`

**Service methods** (46 total):
`uuid7str`, `_audit`, `cache_set`, `cache_get`, `cache_delete`, `cache_exists`, `bulk_set`, `bulk_get`, `cache_flush`, `ttl_update`, `cache_stats`, `eviction_policy`, ...

**Governance rules** (24 total):
`tenant_context_required`, `write_requires_namespace`, `sensitive_entry_requires_encryption`, `regulated_entry_requires_encryption`, `restricted_entry_requires_encryption`, `cross_tenant_cache_access_denied`, `disabled_namespace_blocks_cache_writes`, `disabled_namespace_blocks_cache_warming`, ...

**UI Routes** (14):
- `/cach/dashboard` — dashboard (cach:view)
- `/cach/namespaces` — namespaces (cach:manage_namespaces)
- `/cach/entries` — entries (cach:read)
- `/cach/policies` — policies (cach:manage_policies)
- `/cach/warming` — warming (cach:warm)
- `/cach/evictions` — evictions (cach:review_eviction)
- _8 more..._

**Streaming events** via `bytewax`:
`namespace_created`, `namespace_updated`, `namespace_evicted`, `cache_warmed`, `cache_invalidated`, ...

**Standalone usage:**
```bash
pip install apg-common-cach
apg-common-cach --port 8080
```

---

### Chat and Messaging `chat`

> `chat` provides the APG common capability for tenant-scoped team messaging. It is a dependency-light generated-application packet that can be composed into larger APG applications while keeping live WebSocket servers, durable brokers, identity providers, and notification providers behind adapter boundaries.

**Package**: `apg-common-chat`  
**Path**: `capabilities/common/chat`  
**Version**: 1.0.0  

**Service methods** (43 total):
`describe`, `evaluate`, `create_room`, `approve_room`, `join_room`, `leave_room`, `list_rooms`, `room_members`, `room_permissions`, `send_message`, `edit_message`, `delete_message`, ...

**Governance rules** (42 total):
`tenant_context_required`, `room_requires_owner`, `room_requires_name`, `room_requires_member`, `retention_policy_required`, `external_guest_requires_policy`, `external_guest_requires_expiry`, `large_room_requires_review`, ...

**UI Routes** (12):
- `/chat/dashboard` — dashboard (chat:view)
- `/chat/rooms` — rooms (chat:manage_rooms)
- `/chat/direct` — direct (chat:send)
- `/chat/messages` — messages (chat:send)
- `/chat/presence` — presence (chat:view)
- `/chat/agents` — agents (chat:manage_rooms)
- _6 more..._

**Streaming events** via `bytewax`:
`room_created`, `room_updated`, `room_archived`, `message_sent`, `message_edited`, ...

**Standalone usage:**
```bash
pip install apg-common-chat
apg-common-chat --port 8080
```

---

### Continuous Integration and Delivery `cicd`

> CICD is the APG capability for governed build, test, package, scan, promotion, and release-delivery workflows. It gives generated APG applications a tenant-aware CI/CD lifecycle that can be composed with deployment, environment,

**Package**: `apg-common-cicd`  
**Path**: `capabilities/common/cicd`  
**Version**: 1.0.0  

**Provides:**
- `pipeline_management`
- `build_orchestration`
- `quality_gates`
- `artifact_promotion`
- `release_automation`
- `delivery_agents`

**Requires:**
- `depl`
- `envm`
- `logt`

**Service methods** (46 total):
`uuid7str`, `_audit`, `pipeline_create`, `trigger_build`, `build_complete`, `store_artifact`, `quality_gate_add`, `deployment_promote`, `rollback_release`, `feature_flag_release`, `canary_deploy`, `blue_green_switch`, ...

**Governance rules** (26 total):
`tenant_context_required`, `pipeline_requires_owner`, `build_requires_secret_scope`, `artifact_requires_signature`, `promotion_requires_quality_gate`, `high_parallelism_requires_review`, `pipeline_requires_source_policy`, `pipeline_requires_worker_pool`, ...

**UI Routes** (10):
- `/cicd/dashboard` — dashboard (cicd:view)
- `/cicd/pipelines` — pipelines (cicd:manage_pipelines)
- `/cicd/builds` — builds (cicd:run_builds)
- `/cicd/artifacts` — artifacts (cicd:view)
- `/cicd/gates` — gates (cicd:promote)
- `/cicd/promotions` — promotions (cicd:promote)
- _4 more..._

**Streaming events** via `bytewax`:
`pipeline_created`, `pipeline_review_approved`, `pipeline_state_changed`, `build_run_completed`, `artifact_published`, ...

**Standalone usage:**
```bash
pip install apg-common-cicd
apg-common-cicd --port 8080
```

---

### Collaboration Tools `colb`

> `colb` provides APG's common capability for tenant-scoped collaborative workspaces. It composes chat, notifications, authentication, realtime protocols, governed shared artifacts, annotations, decision records, presence, and AI collaborators into a generated-application packet that can run without live collaboration infrastructure.

**Package**: `apg-common-colb`  
**Path**: `capabilities/common/colb`  
**Version**: 1.0.0  

**Service methods** (47 total):
`uuid7str`, `_audit`, `_notify`, `workspace_create`, `workspace_invite`, `workspace_remove_member`, `list_workspace_members`, `document_create`, `document_share`, `document_update`, `co_edit_session`, `co_edit_apply_op`, ...

**Governance rules** (45 total):
`tenant_context_required`, `workspace_requires_owner`, `workspace_requires_name`, `workspace_requires_participant`, `workspace_requires_retention`, `external_collaboration_requires_policy`, `external_collaboration_requires_expiry`, `large_workspace_requires_review`, ...

**UI Routes** (13):
- `/colb/dashboard` — dashboard (colb:view)
- `/colb/workspaces` — workspaces (colb:create_workspace)
- `/colb/sessions` — sessions (colb:manage_sessions)
- `/colb/presence` — presence (colb:view)
- `/colb/artifacts` — artifacts (colb:collaborate)
- `/colb/annotations` — annotations (colb:collaborate)
- _7 more..._

**Streaming events** via `bytewax`:
`workspace_created`, `workspace_updated`, `workspace_archived`, `session_started`, `session_ended`, ...

**Standalone usage:**
```bash
pip install apg-common-colb
apg-common-colb --port 8080
```

---

### Compliance Management `comp`

> `comp` is APG's package-backed Compliance Management capability. It gives generated applications a tenant-scoped compliance runtime for frameworks, obligations, controls, encrypted evidence, assessments, findings, remediation,

**Package**: `apg-common-comp`  
**Path**: `capabilities/common/comp`  
**Version**: 1.0.0  

**Service methods** (41 total):
`describe`, `evaluate`, `register_framework`, `create_control`, `record_evidence`, `assess_control`, `open_finding`, `resolve_finding`, `escalate_overdue_findings`, `prepare_report`, `approve_report`, `attest_report`, ...

**Governance rules** (45 total):
`tenant_context_required`, `framework_requires_owner`, `framework_requires_obligations`, `framework_requires_policy_version`, `duplicate_framework_blocked`, `control_requires_framework`, `control_requires_name`, `control_requires_owner`, ...

**UI Routes** (14):
- `/comp/dashboard` — dashboard (comp:view)
- `/comp/frameworks` — frameworks (comp:manage_controls)
- `/comp/controls` — controls (comp:manage_controls)
- `/comp/evidence` — evidence (comp:collect_evidence)
- `/comp/assessments` — assessments (comp:manage_controls)
- `/comp/findings` — findings (comp:remediate)
- _8 more..._

**Streaming events** via `bytewax`:
`framework_registered`, `framework_updated`, `control_created`, `control_tested`, `evidence_collected`, ...

**Standalone usage:**
```bash
pip install apg-common-comp
apg-common-comp --port 8080
```

---

### Configuration Management `conf`

> **System-wide configuration store providing centralized, hierarchical configuration management with environment-specific overrides, validation, and real-time updates.**

**Package**: `apg-common-conf`  
**Path**: `capabilities/common/conf`  
**Version**: 1.0.0  

**Provides:**
- `conf_operations`
- `conf_agents`
- `review_evidence`

**Service methods** (63 total):
`set_config_manager`, `set_gitops_manager`, `set_nlp_service`, `describe_runtime`, `_maybe_await`, `_maybe_initialize`, `_maybe_shutdown`, `initialize`, `create_configuration`, `deploy_configuration`, `detect_and_remediate_drift`, `create_intelligent_template`, ...

**Governance rules** (22 total):
`tenant_context_required`, `configuration_record_requires_owner`, `validate_before_apply`, `production_changes_require_approval`, `encrypted_secrets_required`, `drift_requires_remediation_plan`, `production_deployment_requires_rollback`, `production_change_requires_review`, ...

**UI Routes** (13):
- `/config/dashboard` — dashboard (conf:view)
- `/config/resources` — resources (conf:view)
- `/config/templates` — templates (conf:create)
- `/config/changes` — changes (conf:edit)
- `/config/approvals` — approvals (conf:approve)
- `/config/policies` — policies (conf:admin)
- _7 more..._

**Streaming events** via `bytewax`:
`configuration_record_created`, `configuration_change_requested`, `configuration_change_decided`, `configuration_change_deployed`, `configuration_drift_detected`, ...

**Standalone usage:**
```bash
pip install apg-common-conf
apg-common-conf --port 8080
```

---

### Connection Management `conn`

> CONN is APG's governed connector and data-flow control plane. It lets generated applications register local Singer taps and other connectors, create secured connections, test and activate those connections, compose data flows with

**Package**: `apg-common-conn`  
**Path**: `capabilities/common/conn`  
**Version**: 1.0.0  

**Provides:**
- `connector_management`
- `connection_orchestration`
- `connector_agent_composition`
- `review_evidence`

**Requires:**
- `apig`
- `auth`
- `encr`
- `audl`

**Service methods** (57 total):
`get`, `copy`, `start_monitoring`, `get_system_metrics`, `add_job`, `suggest_mappings`, `initialize`, `create_connection`, `get_connection`, `list_connections`, `update_connection`, `delete_connection`, ...

**Governance rules** (39 total):
`tenant_context_required`, `connector_requires_owner`, `connector_requires_runtime`, `connector_runtime_must_be_supported`, `connector_requires_source`, `connector_requires_checksum`, `unverified_connector_requires_review`, `connection_requires_owner`, ...

**UI Routes** (14):
- `/conn/dashboard` — dashboard (conn:view)
- `/conn/connectors` — connectors (conn:view)
- `/conn/connections` — connections (conn:create)
- `/conn/designer` — designer (conn:create)
- `/conn/sync-runs` — sync_runs (conn:view)
- `/conn/quality` — quality (conn:view)
- _8 more..._

**Streaming events** via `bytewax`:
`connector_registered`, `connector_updated`, `connector_retired`, `connection_created`, `connection_activated`, ...

**Standalone usage:**
```bash
pip install apg-common-conn
apg-common-conn --port 8080
```

---

### Consent and Privacy Management `cons`

> CONS is the APG capability for governed consent, privacy preferences, privacy requests, consent-gated processing, and auditable privacy operations. It lets generated APG applications publish notices, register lawful purposes, capture

**Package**: `apg-common-cons`  
**Path**: `capabilities/common/cons`  
**Version**: 1.0.0  

**Provides:**
- `purpose_registry`
- `consent_capture`
- `privacy_requests`
- `preference_center`
- `privacy_audit`
- `privacy_agents`

**Requires:**
- `comp`
- `auth`
- `dlpd`

**Service methods** (45 total):
`describe`, `evaluate`, `publish_notice`, `create_purpose`, `capture_consent`, `withdraw_consent`, `update_preferences`, `process_consent_gated_data`, `submit_privacy_request`, `complete_privacy_request`, `register_privacy_agent`, `change_purpose_state`, ...

**Governance rules** (20 total):
`tenant_context_required`, `purpose_requires_legal_basis`, `consent_capture_requires_notice`, `processing_requires_active_consent`, `privacy_request_requires_identity_verification`, `stale_consent_requires_review`, `purpose_requires_owner`, `purpose_requires_retention_policy`, ...

**UI Routes** (10):
- `/cons/dashboard` — dashboard (cons:view)
- `/cons/purposes` — purposes (cons:manage_purposes)
- `/cons/notices` — notices (cons:manage_purposes)
- `/cons/consents` — consents (cons:view)
- `/cons/requests` — requests (cons:process_requests)
- `/cons/preferences` — preferences (cons:capture)
- _4 more..._

**Streaming events** via `bytewax`:
`notice_published`, `purpose_created`, `purpose_state_changed`, `consent_captured`, `consent_withdrawn`, ...

**Standalone usage:**
```bash
pip install apg-common-cons
apg-common-cons --port 8080
```

---

### Computer Vision `cvsn`

> CVSN is the APG capability for governed visual intelligence. It lets generated applications ingest tenant-scoped image, document, and video assets; run configured vision tasks; manage model and pipeline lifecycle; expose visual UI

**Package**: `apg-common-cvsn`  
**Path**: `capabilities/common/cvsn`  
**Version**: 1.0.0  

**Provides:**
- `computer_vision`
- `visual_intelligence`
- `vision_agent_composition`

**Requires:**
- `aicr`
- `mlcm`
- `conf`
- `auth`

**Service methods** (68 total):
`pipeline`, `is_available`, `_log_processing_operation`, `_log_processing_success`, `_log_processing_error`, `create_processing_job`, `get_job_status`, `list_jobs`, `cancel_job`, `process_job`, `_process_ocr_job`, `_process_object_detection_job`, ...

**Governance rules** (38 total):
`tenant_context_required`, `asset_requires_source`, `asset_requires_supported_mime_type`, `asset_size_within_limit`, `asset_hash_required`, `task_must_be_enabled`, `operator_required`, `document_task_requires_document_asset`, ...

**UI Routes** (15):
- `/cvsn/dashboard` — dashboard (cv:read)
- `/cvsn/assets` — assets (cv:write)
- `/cvsn/documents` — documents (cv:ocr)
- `/cvsn/images` — images (cv:object_detection)
- `/cvsn/video` — video (cv:video_analysis)
- `/cvsn/quality` — quality (cv:quality_control)
- _9 more..._

**Streaming events** via `bytewax`:
`model_registered`, `model_updated`, `model_retired`, `inference_completed`, `object_detected`, ...

**Standalone usage:**
```bash
pip install apg-common-cvsn
apg-common-cvsn --port 8080
```

---

### Deployment Management `depl`

> DEPL is the APG capability for governed release, deployment, health-gate, rollback, deployment-agent, audit, and deployment-evidence workflows. It gives generated APG applications a tenant-aware deployment lifecycle that can be

**Package**: `apg-common-depl`  
**Path**: `capabilities/common/depl`  
**Version**: 1.0.0  

**Provides:**
- `release_management`
- `deployment_rollouts`
- `health_gates`
- `rollback_control`
- `deployment_audit`
- `deployment_agents`

**Requires:**
- `logt`
- `moni`
- `hlth`

**Service methods** (47 total):
`describe`, `evaluate`, `register_environment`, `create_release`, `attach_rollback_plan`, `record_health_gate`, `create_deployment_plan`, `approve_deployment_plan`, `execute_deployment`, `execute_rollback`, `register_deployment_agent`, `change_deployment_plan_state`, ...

**Governance rules** (20 total):
`tenant_context_required`, `release_requires_owner`, `release_requires_manifest`, `release_requires_signature`, `release_requires_change_ticket`, `health_gate_requires_checks`, `deployment_requires_health_gate`, `production_requires_approval`, ...

**UI Routes** (11):
- `/depl/dashboard` — dashboard (depl:view)
- `/depl/releases` — releases (depl:plan)
- `/depl/deployments` — deployments (depl:deploy)
- `/depl/rollouts` — rollouts (depl:deploy)
- `/depl/health` — health (depl:view)
- `/depl/rollback` — rollback (depl:rollback)
- _5 more..._

**Streaming events** via `bytewax`:
`environment_registered`, `release_created`, `rollback_plan_attached`, `health_gate_recorded`, `deployment_plan_created`, ...

**Standalone usage:**
```bash
pip install apg-common-depl
apg-common-depl --port 8080
```

---

### Distributed Computing `dist`

> DIST is the APG capability for governed worker pools, partitioned jobs, distributed execution, result aggregation, scaling decisions, compute-agent governance, audit, and lifecycle stream metadata. It gives generated APG

**Package**: `apg-common-dist`  
**Path**: `capabilities/common/dist`  
**Version**: 1.0.0  

**Provides:**
- `distributed_jobs`
- `worker_pools`
- `partitioned_execution`
- `coordination`
- `distributed_scaling`
- `compute_agents`

**Requires:**
- `mqeb`
- `moni`
- `conf`

**Service methods** (46 total):
`describe`, `evaluate`, `create_worker_pool`, `register_worker`, `submit_job`, `approve_job`, `dispatch_partitions`, `complete_partition`, `aggregate_results`, `record_scaling_decision`, `register_compute_agent`, `change_job_state`, ...

**Governance rules** (20 total):
`tenant_context_required`, `job_requires_owner`, `idempotency_key_required`, `retry_policy_required`, `event_stream_required`, `result_aggregation_required`, `worker_pool_requires_health`, `partition_count_required`, ...

**UI Routes** (10):
- `/dist/dashboard` — dashboard (dist:view)
- `/dist/jobs` — jobs (dist:submit_jobs)
- `/dist/workers` — workers (dist:manage_workers)
- `/dist/partitions` — partitions (dist:view)
- `/dist/queues` — queues (dist:view)
- `/dist/scaling` — scaling (dist:scale)
- _4 more..._

**Streaming events** via `bytewax`:
`worker_pool_created`, `worker_registered`, `job_submitted`, `partition_review_approved`, `job_state_changed`, ...

**Standalone usage:**
```bash
pip install apg-common-dist
apg-common-dist --port 8080
```

---

### Data Loss Prevention `dlpd`

> DLPD is APG's generated-application capability for tenant-scoped data loss prevention. It gives composed applications a deterministic, dependency-light surface for data classification, policy enforcement, egress inspection,

**Package**: `apg-common-dlpd`  
**Path**: `capabilities/common/dlpd`  
**Version**: 1.0.0  

**Service methods** (40 total):
`describe`, `evaluate`, `register_policy`, `create_policy`, `update_policy`, `policy_effectiveness`, `register_classifier`, `regex_pattern_library`, `ml_classifier_train`, `classify_content`, `evaluate_content`, `scan_file`, ...

**Governance rules** (45 total):
`tenant_context_required`, `policy_requires_owner`, `policy_requires_channels`, `policy_requires_classifiers`, `policy_requires_egress_binding`, `inspection_source_requires_policy`, `inspection_requires_active_policy`, `inspection_requires_covered_channel`, ...

**UI Routes** (14):
- `/dlpd/dashboard` — dashboard (dlpd:view)
- `/dlpd/policies` — policies (dlpd:manage_policies)
- `/dlpd/classifiers` — classifiers (dlpd:manage_policies)
- `/dlpd/channels` — channels (dlpd:inspect)
- `/dlpd/inspections` — inspections (dlpd:inspect)
- `/dlpd/incidents` — incidents (dlpd:respond)
- _8 more..._

**Streaming events** via `bytewax`:
`policy_created`, `policy_updated`, `policy_activated`, `policy_deactivated`, `scan_completed`, ...

**Standalone usage:**
```bash
pip install apg-common-dlpd
apg-common-dlpd --port 8080
```

---

### Digital Twin Framework `dtwn`

> DTWN is the APG capability for governed digital twins, simulation models, authenticated telemetry fusion, topology mapping, prediction review, AI twin-agent governance, audit, and lifecycle stream metadata. It gives generated

**Package**: `apg-common-dtwn`  
**Path**: `capabilities/common/dtwn`  
**Version**: 1.0.0  

**Provides:**
- `twin_registry`
- `simulation_models`
- `telemetry_fusion`
- `prediction_workflows`
- `asset_topology`
- `twin_agents`

**Requires:**
- `pred`
- `iotd`
- `geos`
- `cvsn`

**Service methods** (48 total):
`describe`, `evaluate`, `create_twin`, `register_simulation_model`, `ingest_telemetry`, `link_topology`, `run_simulation`, `record_prediction`, `review_prediction`, `register_twin_agent`, `change_twin_status`, `validate_batch_twin_mutation`, ...

**Governance rules** (20 total):
`tenant_context_required`, `twin_requires_owner`, `twin_requires_asset_identity`, `simulation_model_requires_calibration`, `simulation_model_requires_confidence`, `simulation_requires_model`, `telemetry_requires_authenticated_source`, `telemetry_requires_measurements`, ...

**UI Routes** (11):
- `/dtwn/dashboard` — dashboard (dtwn:view)
- `/dtwn/twins` — twins (dtwn:manage_twins)
- `/dtwn/models` — models (dtwn:model)
- `/dtwn/telemetry` — telemetry (dtwn:view)
- `/dtwn/simulations` — simulations (dtwn:simulate)
- `/dtwn/predictions` — predictions (dtwn:view)
- _5 more..._

**Streaming events** via `bytewax`:
`twin_created`, `model_registered`, `telemetry_ingested`, `topology_linked`, `simulation_completed`, ...

**Standalone usage:**
```bash
pip install apg-common-dtwn
apg-common-dtwn --port 8080
```

---

### Data Virtualization `dvrl`

> DVRL is APG's governed data virtualization capability. It gives composed APG applications a first-class way to register virtual data sources, review schemas, publish virtual tables, evaluate federated read-query requests, manage cache

**Package**: `apg-common-dvrl`  
**Path**: `capabilities/common/dvrl`  
**Version**: 1.0.0  

**Provides:**
- `data_virtualization`
- `federated_query_lifecycle`
- `virtualization_agent_composition`
- `review_evidence`

**Requires:**
- `mdm`
- `etlp`
- `meta`

**Service methods** (47 total):
`uuid7str`, `describe`, `register_source`, `activate_source`, `refresh_schema`, `publish_virtual_table`, `execute_query`, `cache_result`, `change_policy`, `retire_source`, `register_virtualization_agent`, `validate_dvrl_lifecycle_batch`, ...

**Governance rules** (28 total):
`tenant_context_required`, `source_registration_requires_owner`, `source_type_must_be_supported`, `source_registration_requires_credentials`, `source_connection_requires_encryption`, `source_activation_requires_approval`, `schema_refresh_requires_review`, `virtual_table_requires_owner`, ...

**UI Routes** (14):
- `/dvrl/dashboard` — dashboard (dvrl:view)
- `/dvrl/query` — query (dvrl:query)
- `/dvrl/sources` — sources (dvrl:manage_sources)
- `/dvrl/schemas` — schemas (dvrl:view)
- `/dvrl/virtual-tables` — virtual_tables (dvrl:manage_sources)
- `/dvrl/federation` — federation (dvrl:view_lineage)
- _8 more..._

**Streaming events** via `bytewax`:
`virtual_source_registered`, `virtual_source_updated`, `virtual_source_retired`, `federated_query_executed`, `query_plan_generated`, ...

**Standalone usage:**
```bash
pip install apg-common-dvrl
apg-common-dvrl --port 8080
```

---

### Edge Computing `edge`

> `edge` is the APG common edge computing capability. It lets generated applications compose tenant-scoped edge nodes, fleets, signed workloads, deployments, offline execution, state synchronization, resource pressure,

**Package**: `apg-common-edge`  
**Path**: `capabilities/common/edge`  
**Version**: 1.0.0  

**Provides:**
- `edge_nodes`
- `edge_fleets`
- `edge_workloads`
- `edge_deployments`
- `offline_execution`
- `edge_sync`
- `edge_agents`

**Requires:**
- `auth`
- `conf`
- `audl`
- `dist`
- `cach`
- `moni`

**Service methods** (49 total):
`describe`, `evaluate`, `register_edge_node`, `node_health_monitor`, `deploy_workload`, `workload_status`, `offload_computation`, `edge_to_cloud_sync`, `auto_scaling`, `failover`, `edge_analytics`, `bandwidth_optimisation`, ...

**Governance rules** (20 total):
`tenant_context_required`, `node_requires_owner`, `node_requires_attestation`, `node_requires_location_policy`, `fleet_requires_owner`, `fleet_requires_policy_version`, `workload_requires_owner`, `workload_requires_signed_artifact`, ...

**UI Routes** (11):
- `/edge/dashboard` — dashboard (edge:view)
- `/edge/nodes` — nodes (edge:manage_nodes)
- `/edge/fleets` — fleets (edge:manage_nodes)
- `/edge/workloads` — workloads (edge:deploy_workloads)
- `/edge/deployments` — deployments (edge:deploy_workloads)
- `/edge/sync` — sync (edge:sync)
- _5 more..._

**Streaming events** via `bytewax`:
`edge_node_registered`, `edge_fleet_created`, `edge_workload_registered`, `edge_workload_deployed`, `edge_sync_completed`, ...

**Standalone usage:**
```bash
pip install apg-common-edge
apg-common-edge --port 8080
```

---

### Encryption Services `encr`

> Encryption Services (`encr`) is APG's cryptographic governance capability for generated applications. It gives application builders a dependency-light runtime for key-domain posture, crypto operation decisions, legacy algorithm

**Package**: `apg-common-encr`  
**Path**: `capabilities/common/encr`  
**Version**: 1.0.0  

**Provides:**
- `encr_operations`
- `crypto_governance`
- `crypto_agent_composition`
- `review_evidence`

**Requires:**
- `conf`
- `auth`
- `secu`
- `audl`

**Service methods** (58 total):
`uuid7str`, `uuid7str`, `put`, `get`, `list`, `delete`, `log_event`, `send`, `encrypt_data`, `decrypt_data`, `key_generate`, `key_rotate`, ...

**Governance rules** (20 total):
`tenant_context_required`, `restricted_data_requires_quantum_safe_algorithm`, `plaintext_export_blocked`, `low_entropy_blocks_key_generation`, `legacy_algorithm_requires_review`, `active_threat_requires_key_rotation`, `crypto_exception_requires_independent_reviewer`, `crypto_exception_requires_notes`, ...

**UI Routes** (12):
- `/encr/dashboard` — dashboard (encr:view)
- `/encr/operations` — operations (encr:operate)
- `/encr/keys` — keys (encr:view_keys)
- `/encr/policies` — policies (encr:manage_policies)
- `/encr/entropy` — entropy (encr:view_entropy)
- `/encr/exceptions` — exceptions (encr:review)
- _6 more..._

**Streaming events** via ``:
`key_domain_registered`, `crypto_operation_allowed`, `crypto_operation_denied`, `crypto_operation_review_required`, `crypto_exception_decided`, ...

**Standalone usage:**
```bash
pip install apg-common-encr
apg-common-encr --port 8080
```

---

### Environment Management `envm`

> `envm` is the APG common environment management capability. It lets generated applications compose tenant-scoped environment inventory, stage and region policy, governed promotion paths, promotion runs, configuration drift reports,

**Package**: `apg-common-envm`  
**Path**: `capabilities/common/envm`  
**Version**: 1.0.0  

**Provides:**
- `environment_inventory`
- `environment_promotion`
- `configuration_drift`
- `secret_scopes`
- `environment_policy`
- `envm_agents`

**Requires:**
- `auth`
- `conf`
- `audl`
- `depl`
- `keym`
- `moni`

**Service methods** (53 total):
`uuid7str`, `uuid7str`, `put`, `get`, `list`, `delete`, `log_event`, `send`, `env_create`, `env_clone`, `env_compare`, `env_promote`, ...

**Governance rules** (20 total):
`tenant_context_required`, `environment_requires_owner`, `environment_requires_region_policy`, `environment_requires_configuration_source`, `environment_requires_rbac_policy`, `production_change_requires_approval`, `promotion_requires_path`, `promotion_requires_artifact_reference`, ...

**UI Routes** (11):
- `/envm/dashboard` — dashboard (envm:view)
- `/envm/environments` — environments (envm:manage_environments)
- `/envm/promotion` — promotion (envm:promote)
- `/envm/drift` — drift (envm:view)
- `/envm/secrets` — secrets (envm:manage_secrets)
- `/envm/agents` — agents (envm:govern)
- _5 more..._

**Streaming events** via `bytewax`:
`environment_registered`, `promotion_path_created`, `environment_promoted`, `drift_recorded`, `secret_scope_registered`, ...

**Standalone usage:**
```bash
pip install apg-common-envm
apg-common-envm --port 8080
```

---

### ESG and Carbon Tracking `esgc`

> `esgc` is the APG common ESG and carbon tracking capability. It lets generated applications compose tenant-scoped emissions inventories, factor libraries, activity emissions, sustainability reports, reduction targets, compliance

**Package**: `apg-common-esgc`  
**Path**: `capabilities/common/esgc`  
**Version**: 1.0.0  

**Provides:**
- `emissions_inventory`
- `factor_library`
- `activity_emissions`
- `sustainability_reporting`
- `target_tracking`
- `esg_evidence`
- `esgc_agents`

**Requires:**
- `auth`
- `conf`
- `audl`
- `geos`
- `pred`
- `comp`

**Service methods** (55 total):
`uuid7str`, `uuid7str`, `put`, `get`, `list`, `delete`, `log_event`, `send`, `create_inventory`, `register_factor`, `record_activity`, `scope1_record`, ...

**Governance rules** (20 total):
`tenant_context_required`, `inventory_requires_owner`, `inventory_requires_boundary`, `factor_requires_approved_source`, `factor_requires_source_evidence`, `factor_requires_version`, `emission_requires_boundary`, `activity_requires_evidence`, ...

**UI Routes** (10):
- `/esgc/dashboard` — dashboard (esgc:view)
- `/esgc/emissions` — emissions (esgc:manage_data)
- `/esgc/factors` — factors (esgc:manage_data)
- `/esgc/data-sources` — data_sources (esgc:manage_data)
- `/esgc/reports` — reports (esgc:report)
- `/esgc/targets` — targets (esgc:view)
- _4 more..._

**Streaming events** via `bytewax`:
`esgc_inventory_created`, `esgc_factor_registered`, `esgc_activity_recorded`, `esgc_report_published`, `esgc_target_created`, ...

**Standalone usage:**
```bash
pip install apg-common-esgc
apg-common-esgc --port 8080
```

---

### Digital Forms and eSign `esgn`

> `esgn` provides APG's common capability for governed digital forms and electronic signatures. It composes form-template authoring, schema validation, publication approval, submissions, signature envelopes, ordered signing ceremonies, cancellation/rejection, tamper sealing, encrypted evidence packages, first-class provider-neutral signing agents, UI route metadata, visual theming, and Bytewax lifecycle guardrails.

**Package**: `apg-common-esgn`  
**Path**: `capabilities/common/esgn`  
**Version**: 1.0.0  

**Provides:**
- `digital_forms`
- `signature_envelopes`
- `signing_ceremonies`
- `evidence_packages`
- `signing_agent_composition`
- `form_workflows`

**Requires:**
- `auth`
- `encr`
- `audl`
- `comp`
- `aicr`

**Service methods** (56 total):
`uuid7str`, `uuid7str`, `put`, `get`, `list`, `delete`, `log_event`, `send`, `form_create`, `form_publish`, `form_submit`, `signature_request`, ...

**Governance rules** (45 total):
`tenant_context_required`, `form_template_requires_owner`, `form_template_requires_name`, `form_template_requires_schema`, `form_template_requires_compliance_framework`, `regulated_form_requires_dlp`, `form_publication_requires_approval`, `regulated_form_requires_compliance_review`, ...

**UI Routes** (12):
- `/esgn/dashboard` — dashboard (esgn:view)
- `/esgn/forms` — forms (esgn:create_forms)
- `/esgn/builder` — builder (esgn:create_forms)
- `/esgn/submissions` — submissions (esgn:view)
- `/esgn/envelopes` — envelopes (esgn:send_envelopes)
- `/esgn/signing` — signing (esgn:sign)
- _6 more..._

**Streaming events** via `bytewax`:
`template_created`, `template_published`, `form_submitted`, `envelope_sent`, `envelope_signed`, ...

**Standalone usage:**
```bash
pip install apg-common-esgn
apg-common-esgn --port 8080
```

---

### ETL/ELT Processing `etlp`

> ETLP is APG's tenant-scoped data pipeline capability. It gives generated APG applications a composable control plane for pipeline design, datasource registration, field mapping, execution, quality gates, lineage emission,

**Package**: `apg-common-etlp`  
**Path**: `capabilities/common/etlp`  
**Version**: 1.0.0  

**Provides:**
- `pipeline_lifecycle`
- `data_integration_governance`
- `pipeline_agent_composition`
- `review_evidence`

**Requires:**
- `mdm`
- `meta`
- `mqeb`
- `moni`

**Service methods** (57 total):
`uuid7str`, `uuid7str`, `put`, `get`, `list`, `delete`, `log_event`, `send`, `pipeline_design`, `source_connect`, `target_connect`, `transform_rule`, ...

**Governance rules** (31 total):
`tenant_context_required`, `pipeline_registration_requires_owner`, `pipeline_mode_must_be_supported`, `pipeline_execution_requires_owner`, `production_execution_requires_approval`, `execution_requires_idempotency_key`, `datasource_registration_requires_owner`, `datasource_requires_secret_reference`, ...

**UI Routes** (16):
- `/etlp/dashboard` — dashboard (etlp:pipeline:read)
- `/etlp/pipelines` — pipelines (etlp:pipeline:read)
- `/etlp/designer` — designer (etlp:pipeline:write)
- `/etlp/field-mapper` — field_mapper (etlp:transformation:write)
- `/etlp/executions` — executions (etlp:pipeline:execute)
- `/etlp/quality` — quality (etlp:quality:read)
- _10 more..._

**Streaming events** via `bytewax`:
`pipeline_created`, `pipeline_updated`, `pipeline_activated`, `pipeline_deactivated`, `pipeline_run_started`, ...

**Standalone usage:**
```bash
pip install apg-common-etlp
apg-common-etlp --port 8080
```

---

### Federated Learning `fedl`

> FEDL is the APG capability for privacy-preserving collaborative model training. It lets generated applications create governed federations, attest participants, run approved training rounds, collect participant updates, apply poisoning

**Package**: `apg-common-fedl`  
**Path**: `capabilities/common/fedl`  
**Version**: 1.0.0  

**Provides:**
- `federated_learning`
- `privacy_preserving_training`
- `federation_agent_composition`

**Requires:**
- `aicr`
- `mlcm`
- `encr`
- `mten`

**Service methods** (43 total):
`describe`, `evaluate`, `create_federation`, `register_participant`, `start_round`, `submit_update`, `aggregate_updates`, `release_model`, `retire_federation`, `register_federation_agent`, `validate_fedl_lifecycle_batch`, `list_federations`, ...

**Governance rules** (38 total):
`tenant_context_required`, `federation_requires_coordinator`, `federation_requires_model_family`, `federation_requires_objective_metric`, `federation_requires_data_residency`, `privacy_budget_must_be_positive`, `participant_requires_attestation`, `participant_requires_contract`, ...

**UI Routes** (15):
- `/fedl/dashboard` — dashboard (fedl:view)
- `/fedl/federations` — federations (fedl:manage_federations)
- `/fedl/participants` — participants (fedl:view_participants)
- `/fedl/attestation` — attestation (fedl:manage_federations)
- `/fedl/rounds` — rounds (fedl:run_rounds)
- `/fedl/updates` — updates (fedl:run_rounds)
- _9 more..._

**Streaming events** via `bytewax`:
`federation_created`, `federation_updated`, `participant_added`, `participant_removed`, `training_round_started`, ...

**Standalone usage:**
```bash
pip install apg-common-fedl
apg-common-fedl --port 8080
```

---

### Facial Recognition `frec`

> FREC provides governed facial recognition for APG applications. It covers face consent, face-template enrollment, liveness evidence, one-to-one verification, one-to-many identification, watchlist policy, emotion-analysis governance, review queues, first-class facial-recognition governance agents, Bytewax lifecycle batch validation, audit evidence, UI metadata, and visual theming.

**Package**: `apg-common-frec`  
**Path**: `capabilities/common/frec`  
**Version**: 1.0.0  

**Provides:**
- `facial_recognition`
- `face_identification`
- `facial_recognition_agent_composition`

**Requires:**
- `biop`
- `cvsn`
- `aicr`
- `encr`
- `audl`
- `conf`
- `mfau`

**Service methods** (53 total):
`_create_audit_log`, `_simple_liveness_check`, `_extract_probe_features`, `initialize`, `close`, `create_user`, `get_user`, `get_user_by_external_id`, `update_verification_threshold`, `get_service_statistics`, `cleanup_expired_data`, `enroll_face`, ...

**Governance rules** (44 total):
`tenant_context_required`, `face_consent_requires_subject`, `face_consent_requires_purpose`, `face_consent_requires_evidence`, `face_enrollment_requires_consent`, `face_enrollment_requires_active_consent`, `face_template_requires_hash`, `face_template_requires_encryption`, ...

**UI Routes** (15):
- `/frec/dashboard` — dashboard (frec:view)
- `/frec/subjects` — subjects (frec:view)
- `/frec/consents` — consents (frec:enroll)
- `/frec/enrollment` — enrollment (frec:enroll)
- `/frec/templates` — templates (frec:enroll)
- `/frec/verification` — verification (frec:verify)
- _9 more..._

**Streaming events** via `bytewax`:
`face_enrolled`, `face_retired`, `face_identified`, `face_verified`, `liveness_checked`, ...

**Standalone usage:**
```bash
pip install apg-common-frec
apg-common-frec --port 8080
```

---

### Geo-Spatial Services `geos`

> GEOS is APG's governed location-intelligence capability. It gives generated applications a dependency-light way to compose event-source registration, geofence creation, location-event processing, territory planning, spatial

**Package**: `apg-common-geos`  
**Path**: `capabilities/common/geos`  
**Version**: 1.0.0  

**Provides:**
- `geofencing`
- `location_events`
- `spatial_analytics`
- `territory_management`
- `location_prediction`
- `location_agents`

**Requires:**
- `pred`
- `aicr`
- `mdm`

**Service methods** (110 total):
`generate_h3_indices`, `calculate_geohash`, `fuzzy_string_match`, `detect_trajectory_patterns`, `calculate_spatial_autocorrelation`, `calculate_distance`, `calculate_bearing`, `point_in_polygon`, `point_in_circle`, `geocode_address`, `batch_geocode`, `reverse_geocode`, ...

**Governance rules** (21 total):
`tenant_context_required`, `location_consent_required`, `geofence_requires_owner`, `event_source_must_be_registered`, `sensitive_location_requires_review`, `large_polygon_requires_review`, `data_residency_policy_required`, `active_geofence_rule_required`, ...

**UI Routes** (10):
- `/geos/dashboard` — dashboard (geos:view)
- `/geos/maps` — maps (geos:view)
- `/geos/geofences` — geofences (geos:manage_geofences)
- `/geos/events` — events (geos:process_events)
- `/geos/territories` — territories (geos:manage_geofences)
- `/geos/analytics` — analytics (geos:analyze)
- _4 more..._

**Streaming events** via `bytewax`:
`event_source_registered`, `geofence_created`, `location_event_processed`, `territory_created`, `spatial_analysis_completed`, ...

**Standalone usage:**
```bash
pip install apg-common-geos
apg-common-geos --port 8080
```

---

### Graph-based RAG `grag`

> GRAG is the APG capability for graph-grounded retrieval augmented generation. It composes the document and answer workflow from RAGN with the graph management surfaces from KNGR and GRPH so generated applications can retrieve

**Package**: `apg-common-grag`  
**Path**: `capabilities/common/grag`  
**Version**: 1.0.0  

**Provides:**
- `graph_based_rag`
- `hybrid_graph_vector_retrieval`
- `graphrag_agent_composition`

**Requires:**
- `ragn`
- `kngr`
- `grph`
- `srch`
- `nlpc`
- `aicr`
- `conf`
- `audl`

**Service methods** (66 total):
`initialize`, `cleanup`, `create_knowledge_graph`, `get_knowledge_graph`, `add_documents_to_graph`, `process_query`, `process_batch_queries`, `process_single_query`, `bounded_process`, `explore_graph`, `detect_communities`, `get_graph_statistics`, ...

**Governance rules** (51 total):
`tenant_context_required`, `graph_source_requires_id`, `graph_source_requires_name`, `graph_source_requires_owner`, `graph_source_requires_registered_graph`, `graph_source_requires_provenance`, `graph_source_retire_requires_review`, `vector_source_requires_id`, ...

**UI Routes** (14):
- `/grag/dashboard` — dashboard (grag:view)
- `/grag/query` — query (grag:query)
- `/grag/graph-sources` — graph_sources (grag:manage_graphs)
- `/grag/vector-sources` — vector_sources (grag:manage_sources)
- `/grag/hybrid-retrieval` — hybrid_retrieval (grag:query)
- `/grag/reasoning` — reasoning (grag:reason)
- _8 more..._

**Streaming events** via `bytewax`:
`graph_ingested`, `graph_updated`, `entity_extracted`, `relationship_mapped`, `query_executed`, ...

**Standalone usage:**
```bash
pip install apg-common-grag
apg-common-grag --port 8080
```

---

### Graph Data Management `grph`

> GRPH provides the APG graph foundation: tenant-scoped schemas, nodes, edges, lineage graphs, relationship governance, bounded traversal, graph quality inspection, first-class graph-agent composition, Bytewax lifecycle batch

**Package**: `apg-common-grph`  
**Path**: `capabilities/common/grph`  
**Version**: 1.0.0  

**Provides:**
- `graph_data_management`
- `relationship_intelligence`
- `graph_agent_composition`

**Requires:**
- `mdm`
- `meta`
- `etlp`
- `srch`
- `aicr`
- `conf`

**Service methods** (47 total):
`describe`, `evaluate`, `create_schema`, `create_node`, `create_edge`, `traverse`, `lineage_path`, `impact_analysis`, `neighborhood`, `quality_report`, `retire_schema`, `create_record`, ...

**Governance rules** (43 total):
`tenant_context_required`, `schema_requires_id`, `schema_requires_name`, `schema_requires_kind`, `schema_kind_requires_review`, `schema_requires_node_types`, `schema_requires_edge_types`, `lineage_schema_requires_source_asset`, ...

**UI Routes** (14):
- `/grph/dashboard` — dashboard (grph:view)
- `/grph/explorer` — explorer (grph:query)
- `/grph/schemas` — schemas (grph:manage_schema)
- `/grph/nodes` — nodes (grph:write)
- `/grph/edges` — edges (grph:write)
- `/grph/traversal` — traversal (grph:query)
- _8 more..._

**Streaming events** via `bytewax`:
`graph_created`, `graph_updated`, `graph_archived`, `node_created`, `node_updated`, ...

**Standalone usage:**
```bash
pip install apg-common-grph
apg-common-grph --port 8080
```

---

### Help and Knowledge Base `help`

> `help` provides APG's common capability for tenant-scoped help centers and governed support knowledge. It composes source registration, article authoring, publication approval, localization, cited answer generation, feedback curation, first-class provider-neutral help agents, audit events, UI routes, visual theming, and Bytewax lifecycle guardrails into a generated-application packet that runs without live search or RAG infrastructure.

**Package**: `apg-common-help`  
**Path**: `capabilities/common/help`  
**Version**: 1.0.0  

**Service methods** (40 total):
`describe`, `evaluate`, `register_source`, `approve_source`, `create_article`, `publish_article`, `search_articles`, `generate_answer`, `localize_article`, `record_feedback`, `close_curation_item`, `freshness_queue`, ...

**Governance rules** (41 total):
`tenant_context_required`, `source_requires_owner`, `source_requires_uri`, `source_requires_approval`, `article_requires_owner`, `article_requires_title`, `article_requires_body`, `publication_requires_approval`, ...

**UI Routes** (13):
- `/help/dashboard` — dashboard (help:view)
- `/help/home` — home (help:view)
- `/help/articles` — articles (help:view)
- `/help/editor` — editor (help:edit_articles)
- `/help/sources` — sources (help:publish)
- `/help/answers` — answers (help:ask)
- _7 more..._

**Streaming events** via `bytewax`:
`article_created`, `article_updated`, `article_published`, `article_archived`, `category_created`, ...

**Standalone usage:**
```bash
pip install apg-common-help
apg-common-help --port 8080
```

---

### Health Checks and Diagnostics `hlth`

> HLTH is APG's tenant-scoped health checks and diagnostics capability. It gives generated applications a dependency-light control plane for registering components, recording health checks, maintaining baselines, opening alerts and

**Package**: `apg-common-hlth`  
**Path**: `capabilities/common/hlth`  
**Version**: 1.0.0  

**Provides:**
- `health_governance`
- `diagnostic_lifecycle`
- `health_agent_composition`
- `review_evidence`

**Requires:**
- `moni`
- `mqeb`
- `conf`

**Service methods** (184 total):
`initialize`, `process_health_metric`, `get_component_health_status`, `calculate_component_health_score`, `perform_comprehensive_health_assessment`, `process_health_alert`, `generate_health_report`, `predict_component_health`, `get_service_health`, `_initialize_apg_integrations`, `_discover_system_components`, `_establish_health_baselines`, ...

**Governance rules** (28 total):
`tenant_context_required`, `component_health_requires_component_id`, `component_must_be_registered`, `disabled_component_blocks_health_check`, `health_score_below_range_denied`, `health_score_above_range_denied`, `critical_health_score_creates_alert`, `critical_alert_requires_owner`, ...

**UI Routes** (15):
- `/hlth/dashboard` — dashboard (health.view)
- `/hlth/components` — components (health.view)
- `/hlth/checks` — checks (health.view)
- `/hlth/baselines` — baselines (health.manage)
- `/hlth/alerts` — alerts (health.alerts.acknowledge)
- `/hlth/incidents` — incidents (health.incidents.manage)
- _9 more..._

**Streaming events** via `bytewax`:
`health_check_registered`, `health_check_executed`, `health_check_passed`, `health_check_failed`, `diagnostic_run_started`, ...

**Standalone usage:**
```bash
pip install apg-common-hlth
apg-common-hlth --port 8080
```

---

### Internationalization `i18n`

> I18N provides APG applications with tenant-scoped localization services: locale registration, fallback policy, regional formats, glossary terms, translation memory, reviewed translation publication, coverage reporting,

**Package**: `apg-common-i18n`  
**Path**: `capabilities/common/i18n`  
**Version**: 1.0.0  

**Provides:**
- `locale_management`
- `translation_memory`
- `content_localization`
- `language_fallbacks`
- `regional_formatting`
- `language_policy`
- `i18n_agents`

**Requires:**
- `conf`
- `nlpc`
- `auth`
- `audl`

**Service methods** (41 total):
`describe`, `evaluate`, `create_locale`, `add_glossary_term`, `upsert_translation`, `reuse_translation_memory`, `publish_translations`, `resolve_text`, `coverage_report`, `register_i18n_agent`, `validate_batch_i18n_mutation`, `create_record`, ...

**Governance rules** (21 total):
`tenant_context_required`, `locale_requires_owner`, `locale_language_supported`, `locale_requires_fallback`, `locale_requires_regional_format`, `glossary_requires_owner`, `translation_requires_key`, `translation_requires_text`, ...

**UI Routes** (10):
- `/i18n/dashboard` — dashboard (i18n:view)
- `/i18n/locales` — locales (i18n:manage_locales)
- `/i18n/translations` — translations (i18n:translate)
- `/i18n/glossaries` — glossaries (i18n:translate)
- `/i18n/coverage` — coverage (i18n:view)
- `/i18n/publishing` — publishing (i18n:publish)
- _4 more..._

**Streaming events** via `bytewax`:
`i18n_locale_created`, `i18n_glossary_term_added`, `i18n_translation_upserted`, `i18n_translation_published`, `i18n_coverage_reported`, ...

**Standalone usage:**
```bash
pip install apg-common-i18n
apg-common-i18n --port 8080
```

---

### Identity Federation `idfd`

> IDFD is APG's generated-application capability for tenant-scoped identity federation. It gives composed applications a deterministic, dependency-light surface for SAML, OIDC, LDAP, SCIM, claim mapping, federated sessions,

**Package**: `apg-common-idfd`  
**Path**: `capabilities/common/idfd`  
**Version**: 1.0.0  

**Provides:**
- `identity_federation`
- `federated_sso`
- `federation_agent_composition`

**Requires:**
- `auth`
- `mfau`
- `encr`
- `audl`
- `secu`
- `keym`
- `moni`
- `cach`

**Service methods** (41 total):
`describe`, `evaluate`, `register_provider`, `refresh_provider_metadata`, `add_claim_mapping`, `issue_session`, `revoke_session`, `register_certificate`, `health_report`, `register_federation_agent`, `validate_idfd_lifecycle_batch`, `create_record`, ...

**Governance rules** (44 total):
`tenant_context_required`, `provider_requires_owner`, `provider_requires_signing_key`, `provider_protocol_must_be_enabled`, `provider_metadata_url_required`, `provider_metadata_signature_required`, `saml_assertion_requires_encryption`, `saml_requires_signed_response`, ...

**UI Routes** (13):
- `/idfd/dashboard` — dashboard (idfd:view)
- `/idfd/providers` — providers (idfd:manage_providers)
- `/idfd/protocols` — protocols (idfd:manage_providers)
- `/idfd/mappings` — mappings (idfd:manage_mappings)
- `/idfd/sessions` — sessions (idfd:view)
- `/idfd/certificates` — certificates (idfd:rotate_keys)
- _7 more..._

**Streaming events** via `bytewax`:
`idp_registered`, `idp_updated`, `idp_deactivated`, `sso_session_started`, `sso_session_ended`, ...

**Standalone usage:**
```bash
pip install apg-common-idfd
apg-common-idfd --port 8080
```

---

### Import/Export `imex`

> IMEX is the APG capability for governed import, export, and migration workflows. It gives generated applications a dependency-light runtime for building transfer jobs while preserving integration points for ETLP, CONN,

**Package**: `apg-common-imex`  
**Path**: `capabilities/common/imex`  
**Version**: 1.0.0  

**Provides:**
- `import_export`
- `bulk_transfer`
- `transfer_agent_composition`
- `review_evidence`

**Requires:**
- `etlp`
- `conn`
- `auth`
- `audl`

**Service methods** (68 total):
`initialize`, `_initialize_apg_clients`, `create_job`, `execute_job`, `get_job_metrics`, `detect_schema_automatically`, `suggest_field_mappings`, `validate_data_quality`, `create_workflow`, `execute_workflow`, `format_detect_auto`, `large_file_stream`, ...

**Governance rules** (38 total):
`tenant_context_required`, `job_requires_owner`, `job_requires_direction`, `job_requires_source`, `job_requires_destination`, `format_must_be_supported`, `source_profile_required`, `checksum_required`, ...

**UI Routes** (14):
- `/imex/dashboard` — dashboard (imex:view)
- `/imex/jobs` — jobs (imex:view)
- `/imex/designer` — designer (imex:create)
- `/imex/mappings` — mappings (imex:manage)
- `/imex/monitor` — monitor (imex:execute)
- `/imex/validation` — validation (imex:manage)
- _8 more..._

**Streaming events** via `bytewax`:
`import_job_created`, `import_job_started`, `import_job_completed`, `import_job_failed`, `export_job_created`, ...

**Standalone usage:**
```bash
pip install apg-common-imex
apg-common-imex --port 8080
```

---

### IoT Device Integration `iotd`

> IOTD provides APG applications with a tenant-scoped device-operations runtime: device identity, certificate ownership, fleet grouping, encrypted telemetry ingestion, governed command dispatch, command acknowledgement, signed firmware

**Package**: `apg-common-iotd`  
**Path**: `capabilities/common/iotd`  
**Version**: 1.0.0  

**Provides:**
- `device_registry`
- `telemetry_ingestion`
- `command_dispatch`
- `firmware_lifecycle`
- `device_security`
- `device_health`
- `iotd_agents`

**Requires:**
- `auth`
- `encr`
- `audl`
- `conf`

**Service methods** (44 total):
`describe`, `evaluate`, `register_device`, `ingest_telemetry`, `dispatch_command`, `acknowledge_command`, `register_firmware`, `deploy_firmware`, `health_report`, `register_iotd_agent`, `validate_batch_iot_mutation`, `stale_device_queue`, ...

**Governance rules** (20 total):
`tenant_context_required`, `device_requires_identity`, `device_requires_owner`, `device_requires_certificate`, `telemetry_requires_bytewax_stream`, `telemetry_requires_encryption`, `telemetry_requires_schema`, `command_requires_name`, ...

**UI Routes** (11):
- `/iotd/dashboard` — dashboard (iotd:view)
- `/iotd/devices` — devices (iotd:register)
- `/iotd/telemetry` — telemetry (iotd:view)
- `/iotd/commands` — commands (iotd:command)
- `/iotd/firmware` — firmware (iotd:manage_firmware)
- `/iotd/agents` — agents (iotd:admin)
- _5 more..._

**Streaming events** via `bytewax`:
`iotd_device_registered`, `iotd_telemetry_ingested`, `iotd_command_dispatched`, `iotd_command_acknowledged`, `iotd_firmware_registered`, ...

**Standalone usage:**
```bash
pip install apg-common-iotd
apg-common-iotd --port 8080
```

---

### Key Management `keym`

> Key Management (`keym`) is APG's cryptographic key lifecycle control plane. It provides generated applications with dependency-light key governance while keeping live HSM, KMS, vault, blockchain audit, AI lifecycle, and security

**Package**: `apg-common-keym`  
**Path**: `capabilities/common/keym`  
**Version**: 1.0.0  

**Provides:**
- `keym_operations`
- `key_lifecycle_governance`
- `key_agent_composition`
- `review_evidence`

**Requires:**
- `conf`
- `audl`
- `mten`

**Service methods** (71 total):
`to_dict`, `to_dict`, `to_dict`, `to_dict`, `to_dict`, `to_dict`, `to_dict`, `to_dict`, `describe`, `evaluate`, `create_managed_key`, `evaluate_key_operation`, ...

**Governance rules** (22 total):
`tenant_context_required`, `key_creation_requires_policy`, `root_key_requires_hsm_attestation`, `export_requires_dual_control`, `overdue_rotation_requires_review`, `compromised_key_blocks_use`, `disabled_key_blocks_use`, `destroyed_key_blocks_use`, ...

**UI Routes** (12):
- `/keym/dashboard` — dashboard (keym.read_key)
- `/keym/keys` — inventory (keym.read_key)
- `/keym/lifecycle` — lifecycle (keym.rotate_key)
- `/keym/export-approvals` — export_approvals (keym.export_key)
- `/keym/rotation-exceptions` — rotation_exceptions (keym.rotate_key)
- `/keym/compromise` — compromise (keym.admin)
- _6 more..._

**Streaming events** via ``:
`managed_key_created`, `key_operation_allowed`, `key_operation_denied`, `export_approval_decided`, `rotation_exception_decided`, ...

**Standalone usage:**
```bash
pip install apg-common-keym
apg-common-keym --port 8080
```

---

### Knowledge Graph `kngr`

> KNGR provides APG's executable knowledge-graph capability: tenant-scoped source registration, entity resolution, evidence-backed relationship linking, semantic enrichment, bounded reasoning paths, curation, publication, first-class

**Package**: `apg-common-kngr`  
**Path**: `capabilities/common/kngr`  
**Version**: 1.0.0  

**Provides:**
- `knowledge_graph`
- `semantic_context`
- `knowledge_agent_composition`

**Requires:**
- `grph`
- `nlpc`
- `meta`
- `srch`
- `onto`
- `aicr`
- `conf`

**Service methods** (40 total):
`describe`, `evaluate`, `register_source`, `resolve_entity`, `link_relationship`, `enrich_entity`, `build_reasoning_path`, `curate_entity`, `publish_graph`, `context_neighborhood`, `create_record`, `register_knowledge_agent`, ...

**Governance rules** (46 total):
`tenant_context_required`, `source_requires_id`, `source_requires_name`, `source_requires_uri`, `source_requires_owner`, `source_requires_evidence`, `source_requires_confidence`, `source_confidence_requires_review`, ...

**UI Routes** (14):
- `/kngr/dashboard` — dashboard (kngr:view)
- `/kngr/sources` — sources (kngr:source)
- `/kngr/entities` — entities (kngr:query)
- `/kngr/relationships` — relationships (kngr:query)
- `/kngr/enrichment` — enrichment (kngr:enrich)
- `/kngr/reasoning` — reasoning (kngr:reason)
- _8 more..._

**Streaming events** via `bytewax`:
`knowledge_item_created`, `knowledge_item_updated`, `knowledge_item_published`, `knowledge_item_archived`, `graph_link_created`, ...

**Standalone usage:**
```bash
pip install apg-common-kngr
apg-common-kngr --port 8080
```

---

### Logging and Tracing `logt`

> LOGT provides APG applications with a tenant-scoped observability runtime: structured log ingestion, distributed trace roots, span recording, diagnostic search, approved diagnostic exports, retention policy, audit evidence,

**Package**: `apg-common-logt`  
**Path**: `capabilities/common/logt`  
**Version**: 1.0.0  

**Provides:**
- `structured_logging`
- `distributed_tracing`
- `trace_correlation`
- `log_search`
- `diagnostic_retention`
- `diagnostic_exports`
- `logt_agents`

**Requires:**
- `moni`
- `conf`
- `audl`

**Service methods** (40 total):
`describe`, `evaluate`, `create_retention_policy`, `create_pipeline`, `ingest_log`, `ingest_trace`, `record_span`, `search_logs`, `export_logs`, `create_record`, `list_records`, `list_pipelines`, ...

**Governance rules** (22 total):
`tenant_context_required`, `pipeline_requires_owner`, `pipeline_requires_schema`, `pipeline_requires_bytewax_stream`, `pipeline_requires_sampling_policy`, `trace_context_required`, `trace_requires_identifier`, `span_requires_service`, ...

**UI Routes** (10):
- `/logt/dashboard` — dashboard (logt:view)
- `/logt/logs` — logs (logt:query)
- `/logt/traces` — traces (logt:query)
- `/logt/spans` — spans (logt:query)
- `/logt/pipelines` — pipelines (logt:manage_pipelines)
- `/logt/retention` — retention (logt:manage_retention)
- _4 more..._

**Streaming events** via `bytewax`:
`logt_pipeline_created`, `logt_log_ingested`, `logt_trace_ingested`, `logt_span_recorded`, `logt_query_executed`, ...

**Standalone usage:**
```bash
pip install apg-common-logt
apg-common-logt --port 8080
```

---

### Multi-Channel Output `mchn`

> MCHN provides APG applications with a tenant-scoped output runtime: output channels, approved templates, delivery policies, delivery routes, rendered messages and documents, delivery batches, provider receipts, output agents, UI

**Package**: `apg-common-mchn`  
**Path**: `capabilities/common/mchn`  
**Version**: 1.0.0  

**Provides:**
- `channel_routing`
- `format_rendering`
- `output_templates`
- `delivery_policy`
- `delivery_receipts`
- `omnichannel_analytics`
- `mchn_agents`

**Requires:**
- `ntfy`
- `auth`
- `conf`
- `audl`

**Service methods** (41 total):
`describe`, `evaluate`, `create_channel`, `publish_template`, `create_delivery_policy`, `create_route`, `render_output`, `deliver_batch`, `record_receipt`, `create_record`, `list_records`, `list_channels`, ...

**Governance rules** (26 total):
`tenant_context_required`, `channel_requires_owner`, `channel_requires_provider`, `template_requires_approval`, `template_requires_approver`, `template_requires_content`, `template_requires_channel`, `policy_requires_recipient_limit`, ...

**UI Routes** (10):
- `/mchn/dashboard` — dashboard (mchn:view)
- `/mchn/render` — render (mchn:render)
- `/mchn/templates` — templates (mchn:manage_templates)
- `/mchn/routes` — routes (mchn:route)
- `/mchn/channels` — channels (mchn:admin)
- `/mchn/agents` — agents (mchn:admin)
- _4 more..._

**Streaming events** via `bytewax`:
`mchn_channel_created`, `mchn_template_published`, `mchn_policy_created`, `mchn_route_created`, `mchn_output_rendered`, ...

**Standalone usage:**
```bash
pip install apg-common-mchn
apg-common-mchn --port 8080
```

---

### Master Data Management `mdm`

> `common/mdm` provides the master-data governance layer for APG applications. It lets generated applications register tenant-scoped entities, score data quality, review duplicate candidates, compose golden records, manage

**Package**: `apg-common-mdm`  
**Path**: `capabilities/common/mdm`  
**Version**: 1.0.0  

**Provides:**
- `master_data_governance`
- `golden_record_lifecycle`
- `data_agent_composition`
- `review_evidence`

**Requires:**
- `auth`
- `audl`
- `conf`
- `mten`

**Service methods** (50 total):
`create_entity`, `update_entity`, `get_entity`, `search_entities`, `delete_entity`, `_create_entity_version`, `assess_quality`, `_fallback_assessment`, `_assess_completeness`, `_assess_accuracy`, `_assess_consistency`, `_assess_validity`, ...

**Governance rules** (25 total):
`tenant_context_required`, `entity_type_must_be_supported`, `business_key_required_for_entity`, `restricted_entity_requires_data_owner`, `entity_publish_requires_data_owner`, `publish_requires_latest_quality_assessment`, `low_quality_blocks_publish`, `invalid_quality_score_blocks_assessment`, ...

**UI Routes** (15):
- `/mdm/dashboard` — dashboard (mdm:view)
- `/mdm/entities` — entities (mdm:manage_entities)
- `/mdm/golden-records` — golden_records (mdm:manage_golden_records)
- `/mdm/quality` — quality (mdm:view_quality)
- `/mdm/duplicates` — duplicates (mdm:review_duplicates)
- `/mdm/stewardship` — stewardship (mdm:steward)
- _9 more..._

**Streaming events** via `bytewax`:
`entity_created`, `entity_updated`, `entity_merged`, `entity_deactivated`, `golden_record_created`, ...

**Standalone usage:**
```bash
pip install apg-common-mdm
apg-common-mdm --port 8080
```

---

### Metadata Management `meta`

> `common/meta` provides the metadata catalog and governance layer for APG applications. It lets generated applications register metadata assets, schedule approved discovery, classify sensitive assets, capture lineage, assess metadata

**Package**: `apg-common-meta`  
**Path**: `capabilities/common/meta`  
**Version**: 1.0.0  

**Provides:**
- `metadata_catalog_governance`
- `metadata_lifecycle`
- `catalog_agent_composition`
- `review_evidence`

**Requires:**
- `mdm`
- `auth`
- `audl`

**Service methods** (47 total):
`to_dict`, `describe`, `create_record`, `register_asset`, `schedule_discovery`, `record_discovery_result`, `classify_asset`, `review_classification`, `capture_lineage`, `assess_quality`, `request_certification`, `publish_asset`, ...

**Governance rules** (27 total):
`tenant_context_required`, `asset_type_must_be_supported`, `asset_registration_requires_business_key`, `asset_registration_requires_source_system`, `published_asset_requires_owner`, `publish_requires_quality_assessment`, `restricted_asset_requires_classification`, `sensitive_asset_requires_steward`, ...

**UI Routes** (15):
- `/meta/dashboard` — dashboard (meta:view)
- `/meta/catalog` — catalog (meta:view_assets)
- `/meta/discovery` — discovery (meta:run_discovery)
- `/meta/lineage` — lineage (meta:view_lineage)
- `/meta/classification` — classification (meta:classify)
- `/meta/quality` — quality (meta:view_quality)
- _9 more..._

**Streaming events** via `bytewax`:
`asset_created`, `asset_updated`, `asset_deprecated`, `asset_deleted`, `tag_applied`, ...

**Standalone usage:**
```bash
pip install apg-common-meta
apg-common-meta --port 8080
```

---

### Multi-Factor Authentication `mfau`

> MFAU provides adaptive multi-factor authentication for APG applications. It is a composable security capability for enrolling factors, assessing risk, issuing challenges, binding devices, governing account recovery, managing backup codes, composing first-class MFA security agents, validating Bytewax lifecycle batches, and exposing UI surfaces that generated applications can assemble into complete authentication flows.

**Package**: `apg-common-mfau`  
**Path**: `capabilities/common/mfau`  
**Version**: 1.0.0  

**Provides:**
- `multi_factor_authentication`
- `adaptive_authentication`
- `mfa_agent_composition`

**Requires:**
- `auth`
- `secu`
- `encr`
- `aicr`
- `conf`
- `audl`

**Service methods** (51 total):
`authenticate_user`, `enroll_mfa_method`, `start_biometric_enrollment`, `remove_mfa_method`, `initiate_account_recovery`, `get_user_mfa_status`, `generate_backup_codes`, `verify_step_up_authentication`, `get_service_metrics`, `_authentication_successful`, `_authentication_failed`, `_handle_step_up_auth`, ...

**Governance rules** (48 total):
`tenant_context_required`, `profile_requires_user`, `profile_requires_policy`, `profile_status_requires_allowed_value`, `enrollment_requires_profile`, `enrollment_requires_method_type`, `method_type_requires_allowed_value`, `enrollment_requires_verified_channel`, ...

**UI Routes** (16):
- `/mfau/dashboard` — dashboard (mfau:view)
- `/mfau/profiles` — profiles (mfau:view)
- `/mfau/methods` — methods (mfau:manage_methods)
- `/mfau/enrollment` — enrollment (mfau:enroll)
- `/mfau/challenges` — challenges (mfau:challenge)
- `/mfau/risk` — risk (mfau:challenge)
- _10 more..._

**Streaming events** via `bytewax`:
`factor_enrolled`, `factor_retired`, `challenge_issued`, `challenge_passed`, `challenge_failed`, ...

**Standalone usage:**
```bash
pip install apg-common-mfau
apg-common-mfau --port 8080
```

---

### AI Model Lifecycle Management `mlcm`

> MLCM is the APG capability for governed AI model operations. It gives generated applications a tenant-scoped model registry, version lineage, evaluation gates, promotion approvals, deployment controls, drift response, rollback, retirement,

**Package**: `apg-common-mlcm`  
**Path**: `capabilities/common/mlcm`  
**Version**: 1.0.0  

**Provides:**
- `model_lifecycle`
- `model_governance`
- `model_lifecycle_agent_composition`

**Requires:**
- `aicr`
- `moni`
- `audl`

**Service methods** (45 total):
`describe`, `evaluate`, `register_model`, `create_version`, `record_evaluation`, `request_promotion`, `create_target`, `deploy_model`, `record_drift`, `record_drift_review`, `rollback_deployment`, `retire_model`, ...

**Governance rules** (43 total):
`tenant_context_required`, `model_registration_requires_owner`, `model_registration_requires_name`, `model_registration_requires_problem_type`, `model_registration_requires_risk_level`, `version_creation_requires_registered_model`, `version_creation_requires_artifact_uri`, `version_creation_requires_training_data`, ...

**UI Routes** (15):
- `/mlcm/dashboard` — dashboard (mlcm:view)
- `/mlcm/models` — registry (mlcm:view_models)
- `/mlcm/versions` — versions (mlcm:manage_models)
- `/mlcm/model-cards` — model_cards (mlcm:view_models)
- `/mlcm/evaluation` — evaluation (mlcm:evaluate)
- `/mlcm/baselines` — baselines (mlcm:evaluate)
- _9 more..._

**Streaming events** via `bytewax`:
`model_registered`, `model_versioned`, `model_promoted`, `model_deprecated`, `model_retired`, ...

**Standalone usage:**
```bash
pip install apg-common-mlcm
apg-common-mlcm --port 8080
```

---

### Monitoring and Observability `moni`

> MONI is APG's tenant-scoped monitoring and observability capability. It gives generated applications a dependency-light control plane for registering signal sources, governing metrics/logs/traces, managing SLOs, routing alerts,

**Package**: `apg-common-moni`  
**Path**: `capabilities/common/moni`  
**Version**: 1.0.0  

**Provides:**
- `observability_governance`
- `metrics_lifecycle`
- `monitoring_agent_composition`
- `review_evidence`

**Requires:**
- `conf`
- `audl`
- `mqeb`

**Service methods** (68 total):
`initialize`, `shutdown`, `track_metric`, `query_metrics`, `create_alert_rule`, `get_health_status`, `detect_anomalies`, `predict_resource_usage`, `analyze_performance`, `get_active_alerts`, `acknowledge_alert`, `resolve_alert`, ...

**Governance rules** (24 total):
`tenant_context_required`, `metric_ingestion_requires_source`, `signal_requires_registered_source`, `disabled_source_blocks_ingestion`, `trace_requires_trace_id`, `trace_requires_service_name`, `critical_alert_requires_route`, `critical_alert_requires_owner`, ...

**UI Routes** (16):
- `/moni/dashboard` — dashboard (moni:view)
- `/moni/sources` — sources (moni:manage_sources)
- `/moni/metrics` — metrics (moni:view_metrics)
- `/moni/logs` — logs (moni:view_logs)
- `/moni/alerts` — alerts (moni:manage_alerts)
- `/moni/traces` — traces (moni:view_traces)
- _10 more..._

**Streaming events** via `bytewax`:
`metric_recorded`, `metric_threshold_breached`, `alert_triggered`, `alert_resolved`, `health_check_failed`, ...

**Standalone usage:**
```bash
pip install apg-common-moni
apg-common-moni --port 8080
```

---

### Message Queue Event Bus `mqeb`

> MQEB is APG's package-backed event fabric. It provides tenant-scoped topic management, governed message publishing, subscription lifecycle state, delivery/dead-letter evidence, replay review, priority quota review, rule

**Package**: `apg-common-mqeb`  
**Path**: `capabilities/common/mqeb`  
**Version**: 1.0.0  

**Provides:**
- `mqeb_event_fabric`
- `message_governance`
- `event_agent_composition`
- `review_evidence`

**Requires:**
- `conf`
- `auth`
- `audl`
- `secu`

**Service methods** (62 total):
`to_dict`, `to_dict`, `to_dict`, `to_dict`, `to_dict`, `to_dict`, `to_dict`, `to_dict`, `to_dict`, `describe`, `evaluate`, `create_topic`, ...

**Governance rules** (23 total):
`tenant_context_required`, `publish_requires_topic`, `restricted_topic_requires_encryption`, `regulated_topic_requires_schema`, `regulated_topic_requires_encryption`, `cross_tenant_publish_denied`, `guaranteed_delivery_requires_dead_letter_queue`, `exactly_once_requires_idempotency_key`, ...

**UI Routes** (14):
- `/mqeb/dashboard` — dashboard (mqeb:view)
- `/mqeb/topics` — topics (mqeb:manage_topics)
- `/mqeb/publish` — publish (mqeb:publish)
- `/mqeb/subscriptions` — subscriptions (mqeb:subscribe)
- `/mqeb/delivery` — delivery (mqeb:view_metrics)
- `/mqeb/dead-letters` — dead_letters (mqeb:manage_routing)
- _8 more..._

**Streaming events** via `bytewax`:
`topic_created`, `topic_updated`, `topic_retired`, `subscription_created`, `subscription_deleted`, ...

**Standalone usage:**
```bash
pip install apg-common-mqeb
apg-common-mqeb --port 8080
```

---

### Multi-Tenant Management `mten`

> **Enterprise multi-tenancy framework providing tenant isolation, management, and context switching for the APG platform.**

**Package**: `apg-common-mten`  
**Path**: `capabilities/common/mten`  
**Version**: 1.0.0  

**Provides:**
- `mten_operations`
- `tenant_agents`
- `review_evidence`

**Service methods** (59 total):
`model_dump`, `get_tenant_permissions`, `log_event`, `status`, `initialize`, `_initialize_apg_integrations`, `_load_default_templates`, `create_tenant`, `_provision_tenant_async`, `_allocate_compute_resources`, `_allocate_storage_resources`, `_allocate_network_resources`, ...

**Governance rules** (20 total):
`tenant_context_required`, `cross_tenant_access_requires_membership`, `suspended_tenants_block_mutations`, `custom_domain_requires_dns_validation`, `capacity_overcommit_requires_review`, `capacity_review_requires_independent_reviewer`, `isolation_boundary_requires_encryption`, `isolation_breach_requires_suspension`, ...

**UI Routes** (12):
- `/mten/dashboard` — dashboard (mten:view)
- `/mten/tenants` — tenants (mten:view)
- `/mten/provisioning` — provisioning (mten:provision)
- `/mten/capacity/approvals` — capacity_approvals (mten:approve_capacity)
- `/mten/isolation` — isolation (mten:admin)
- `/mten/migrations` — live_migrations (mten:migrate)
- _6 more..._

**Streaming events** via `bytewax`:
`tenant_created`, `tenant_updated`, `tenant_activated`, `tenant_suspended`, `tenant_deactivated`, ...

**Standalone usage:**
```bash
pip install apg-common-mten
apg-common-mten --port 8080
```

---

### No-Code/Low-Code Builder `ncod`

> NCOD is APG's governed no-code and low-code application composition capability. It gives tenants a deterministic app library, screen composer, component catalog, data modeler, workflow binding surface, script and connector extension

**Package**: `apg-common-ncod`  
**Path**: `capabilities/common/ncod`  
**Version**: 1.0.0  

**Provides:**
- `app_builder`
- `page_composer`
- `data_modeler`
- `workflow_binding`
- `script_extensions`
- `connector_bindings`
- `ai_builder_agents`
- `app_publishing`
- `app_deployment`

**Requires:**
- `wflo`
- `scpt`
- `auth`

**Service methods** (42 total):
`describe`, `evaluate`, `create_app`, `add_page`, `add_component`, `define_data_model`, `bind_data_source`, `attach_workflow`, `create_theme_variant`, `add_script_extension`, `add_connector_binding`, `register_builder_agent`, ...

**Governance rules** (33 total):
`tenant_context_required`, `app_requires_owner`, `app_requires_name`, `app_requires_theme`, `app_requires_rbac_policy`, `app_requires_data_residency_policy`, `page_requires_route`, `page_requires_relationship_policy`, ...

**UI Routes** (14):
- `/ncod/dashboard` — dashboard (ncod:view)
- `/ncod/apps` — apps (ncod:manage_apps)
- `/ncod/builder` — builder (ncod:build)
- `/ncod/pages` — pages (ncod:build)
- `/ncod/data-models` — data_models (ncod:build)
- `/ncod/components` — components (ncod:build)
- _8 more..._

**Streaming events** via `bytewax`:
`app_created`, `page_added`, `component_added`, `data_model_defined`, `data_binding_added`, ...

**Standalone usage:**
```bash
pip install apg-common-ncod
apg-common-ncod --port 8080
```

---

### NLP Core `nlpc`

> NLPC is the APG capability for governed text intelligence. It lets generated applications ingest tenant-scoped documents, detect or declare language, execute configured NLP tasks, manage pipelines and model releases, coordinate

**Package**: `apg-common-nlpc`  
**Path**: `capabilities/common/nlpc`  
**Version**: 1.0.0  

**Provides:**
- `text_intelligence`
- `multilingual_processing`
- `nlp_agent_composition`

**Requires:**
- `aicr`
- `mlcm`
- `conf`

**Service methods** (40 total):
`uuid7str`, `create_document`, `get_document`, `list_documents`, `delete_document`, `detect_language`, `extract_entities`, `sentiment_analysis`, `intent_classification`, `text_summarisation`, `_ollama_summarise`, `translate`, ...

**Governance rules** (38 total):
`tenant_context_required`, `document_requires_content`, `document_size_within_limit`, `document_requires_language_or_detection`, `language_required_or_detected`, `language_must_be_supported`, `language_detection_low_confidence_requires_review`, `task_must_be_enabled`, ...

**UI Routes** (16):
- `/nlpc/dashboard` — dashboard (nlpc:view)
- `/nlpc/process` — process (nlpc:process)
- `/nlpc/documents` — documents (nlpc:process)
- `/nlpc/pipelines` — pipelines (nlpc:manage_models)
- `/nlpc/batches` — batches (nlpc:process)
- `/nlpc/annotations` — annotations (nlpc:annotate)
- _10 more..._

**Streaming events** via `bytewax`:
`document_processed`, `entity_extracted`, `sentiment_analyzed`, `classification_completed`, `translation_completed`, ...

**Standalone usage:**
```bash
pip install apg-common-nlpc
apg-common-nlpc --port 8080
```

---

### Notifications and Alerts `ntfy`

> `ntfy` is APG's package-backed Notifications and Alerts capability. It gives generated applications a tenant-scoped notification runtime for recipient preferences, channel providers, template governance, delivery decisions,

**Package**: `apg-common-ntfy`  
**Path**: `capabilities/common/ntfy`  
**Version**: 1.0.0  

**Service methods** (53 total):
`send_notification_request`, `send_bulk_notifications`, `process_with_semaphore`, `execute_campaign`, `get_delivery_analytics`, `track_engagement_event`, `get_user_preferences`, `update_user_preferences`, `get_service_health`, `_get_user_preferences`, `_optimize_channel_selection`, `_execute_multi_channel_delivery`, ...

**Governance rules** (40 total):
`tenant_context_required`, `recipient_requires_address`, `recipient_requires_channel_preferences`, `marketing_requires_opt_in`, `unsubscribe_blocks_marketing`, `quiet_hours_require_urgent_priority`, `template_requires_owner`, `template_requires_name`, ...

**UI Routes** (12):
- `/ntfy/dashboard` — dashboard (ntfy:view)
- `/ntfy/messages` — messages (ntfy:send)
- `/ntfy/templates` — templates (ntfy:manage_templates)
- `/ntfy/campaigns` — campaigns (ntfy:manage_campaigns)
- `/ntfy/preferences` — preferences (ntfy:view)
- `/ntfy/suppression` — suppression (ntfy:manage_campaigns)
- _6 more..._

**Streaming events** via `bytewax`:
`notification_sent`, `notification_delivered`, `notification_failed`, `notification_read`, `template_created`, ...

**Standalone usage:**
```bash
pip install apg-common-ntfy
apg-common-ntfy --port 8080
```

---

### Ontology Management `onto`

> ONTO is the APG capability for governed ontologies, taxonomies, controlled vocabularies, semantic mappings, validation, publication, ontology exchange, first-class ontology agents, and Bytewax lifecycle batches. It gives generated applications an executable vocabulary workbench that can be composed with Knowledge Graph, Metadata, NLP, Search, Auth, Audit, AICR, Cache, Metrics, and Bytewax-backed event processing.

**Package**: `apg-common-onto`  
**Path**: `capabilities/common/onto`  
**Version**: 1.0.0  

**Provides:**
- `ontology_management`
- `semantic_vocabulary_governance`
- `ontology_agent_composition`

**Requires:**
- `meta`
- `nlpc`
- `grph`
- `srch`
- `aicr`
- `conf`
- `auth`
- `audl`

**Service methods** (43 total):
`describe`, `evaluate`, `register_ontology`, `register_namespace`, `create_term`, `curate_term`, `add_synonym`, `add_taxonomy_edge`, `create_mapping`, `deprecate_term`, `validate_ontology`, `review_mapping`, ...

**Governance rules** (55 total):
`tenant_context_required`, `ontology_requires_id`, `ontology_requires_name`, `ontology_requires_owner`, `ontology_requires_domain`, `ontology_retire_requires_review`, `namespace_requires_ontology`, `namespace_requires_prefix`, ...

**UI Routes** (15):
- `/onto/dashboard` — dashboard (onto:view)
- `/onto/ontologies` — ontologies (onto:view)
- `/onto/namespaces` — namespaces (onto:edit)
- `/onto/terms` — terms (onto:edit)
- `/onto/taxonomy` — taxonomy (onto:edit)
- `/onto/mappings` — mappings (onto:map)
- _9 more..._

**Streaming events** via `bytewax`:
`ontology_created`, `ontology_updated`, `ontology_published`, `ontology_deprecated`, `concept_created`, ...

**Standalone usage:**
```bash
pip install apg-common-onto
apg-common-onto --port 8080
```

---

### Platform Foundation `plfd`

> PLFD provides APG applications with a tenant-scoped foundation governance runtime: platform service registry, dependency posture, required baselines, readiness gates, platform change approval, foundation agents, UI metadata,

**Package**: `apg-common-plfd`  
**Path**: `capabilities/common/plfd`  
**Version**: 1.0.0  

**Provides:**
- `foundation_registry`
- `dependency_posture`
- `configuration_baselines`
- `readiness_gates`
- `platform_governance`
- `plfd_agents`

**Requires:**
- `conf`
- `mten`
- `auth`
- `audl`

**Service methods** (45 total):
`describe`, `evaluate`, `health_check_all_services`, `platform_configuration`, `feature_flag_set`, `feature_flag_check`, `circuit_breaker_status`, `circuit_breaker_reset`, `dependency_graph`, `service_discovery_register`, `rate_limiter_configure`, `platform_metrics_dashboard`, ...

**Governance rules** (24 total):
`tenant_context_required`, `foundation_service_requires_owner`, `foundation_service_requires_tier`, `foundation_service_requires_readiness_score`, `dependency_requires_evidence`, `baseline_requires_evidence`, `baseline_requires_approver`, `dependency_health_required`, ...

**UI Routes** (10):
- `/plfd/dashboard` — dashboard (plfd:view)
- `/plfd/services` — services (plfd:manage_services)
- `/plfd/dependencies` — dependencies (plfd:view)
- `/plfd/baselines` — baselines (plfd:manage_baselines)
- `/plfd/readiness` — readiness (plfd:view)
- `/plfd/changes` — changes (plfd:approve_changes)
- _4 more..._

**Streaming events** via `bytewax`:
`plfd_service_registered`, `plfd_dependency_recorded`, `plfd_baseline_attached`, `plfd_readiness_assessed`, `plfd_change_proposed`, ...

**Standalone usage:**
```bash
pip install apg-common-plfd
apg-common-plfd --port 8080
```

---

### Plugin/Extension Framework `plgn`

> PLGN gives APG applications a tenant-scoped extension system: plugin manifests, curated marketplace listings, permission review, sandbox policy, release gates, installation, activation, plugin-governance agents, UI metadata, theme

**Package**: `apg-common-plgn`  
**Path**: `capabilities/common/plgn`  
**Version**: 1.0.0  

**Provides:**
- `plugin_registry`
- `extension_marketplace`
- `permission_review`
- `sandbox_policy`
- `plugin_release_lifecycle`
- `plgn_agents`

**Requires:**
- `auth`
- `secu`
- `conf`
- `audl`

**Service methods** (45 total):
`describe`, `evaluate`, `register_plugin`, `install_plugin`, `uninstall_plugin`, `plugin_health_check`, `plugin_event_hook`, `plugin_sandboxed_execution`, `plugin_permission_check`, `plugin_marketplace_listing`, `plugin_analytics`, `plugin_dependency_resolution`, ...

**Governance rules** (21 total):
`tenant_context_required`, `plugin_requires_owner`, `plugin_requires_signature`, `plugin_requires_manifest_schema`, `plugin_requires_dependency_validation`, `plugin_requires_supply_chain_scan`, `permissions_require_review`, `plugin_requires_sandbox`, ...

**UI Routes** (10):
- `/plgn/dashboard` — dashboard (plgn:view)
- `/plgn/marketplace` — marketplace (plgn:install)
- `/plgn/plugins` — plugins (plgn:view)
- `/plgn/manifests` — manifests (plgn:publish)
- `/plgn/permissions` — permissions (plgn:review)
- `/plgn/sandbox` — sandbox (plgn:review)
- _4 more..._

**Streaming events** via `bytewax`:
`plgn_plugin_registered`, `plgn_permission_review_recorded`, `plgn_sandbox_policy_attached`, `plgn_marketplace_listing_published`, `plgn_plugin_released`, ...

**Standalone usage:**
```bash
pip install apg-common-plgn
apg-common-plgn --port 8080
```

---

### Pose Estimation `pose`

> POSE is APG's governed human pose-estimation capability. It provides tenant-scoped model registration, tracking sessions, frame capture, pose estimates, biomechanical analysis, 3D reconstruction records, AI pose-agent

**Package**: `apg-common-pose`  
**Path**: `capabilities/common/pose`  
**Version**: 1.0.0  

**Provides:**
- `pose_estimation`
- `multi_person_tracking`
- `biomechanical_analysis`
- `pose_3d_reconstruction`
- `edge_pose_inference`
- `pose_agents`
- `pose_quality_governance`

**Requires:**
- `cvsn`
- `aicr`
- `mlcm`

**Service methods** (40 total):
`describe`, `evaluate`, `register_model`, `start_session`, `record_frame`, `estimate_pose`, `analyze_pose`, `reconstruct_3d`, `register_pose_agent`, `change_session_state`, `create_record`, `list_records`, ...

**Governance rules** (22 total):
`tenant_context_required`, `pose_model_requires_owner`, `pose_model_requires_policy`, `tracking_session_requires_owner`, `subject_consent_required`, `tracking_source_required`, `secure_stream_required`, `sensitive_use_requires_approval`, ...

**UI Routes** (12):
- `/pose/dashboard` — dashboard (pose:view)
- `/pose/estimate` — estimate (pose:estimate)
- `/pose/tracking` — tracking (pose:track)
- `/pose/analysis` — analysis (pose:analyze)
- `/pose/reconstruction` — reconstruction (pose:analyze)
- `/pose/sessions` — sessions (pose:view)
- _6 more..._

**Streaming events** via `bytewax`:
`pose_model_registered`, `pose_session_started`, `pose_frame_recorded`, `pose_estimated`, `pose_analysis_completed`, ...

**Standalone usage:**
```bash
pip install apg-common-pose
apg-common-pose --port 8080
```

---

### Predictive Analytics `pred`

> PRED is the APG capability for governed forecasting, scoring, scenario simulation, and predictive model operations. It lets generated applications register predictive models and feature sets, create forecasts, score entities,

**Package**: `apg-common-pred`  
**Path**: `capabilities/common/pred`  
**Version**: 1.0.0  

**Provides:**
- `predictive_analytics`
- `forecasting`
- `prediction_agent_composition`

**Requires:**
- `aicr`
- `mlcm`
- `etlp`
- `conf`

**Service methods** (40 total):
`describe`, `evaluate`, `register_model`, `approve_model`, `register_feature_set`, `create_forecast`, `score_entity`, `simulate_scenario`, `record_drift`, `register_prediction_agent`, `validate_pred_lifecycle_batch`, `create_record`, ...

**Governance rules** (39 total):
`tenant_context_required`, `model_requires_owner`, `model_requires_algorithm`, `model_requires_target`, `model_requires_training_history`, `model_requires_feature_names`, `model_approval_requires_explainability`, `feature_set_requires_owner`, ...

**UI Routes** (14):
- `/pred/dashboard` — dashboard (pred:view)
- `/pred/forecasts` — forecasts (pred:forecast)
- `/pred/scores` — scores (pred:score)
- `/pred/features` — features (pred:manage_models)
- `/pred/scenarios` — scenarios (pred:simulate)
- `/pred/models` — models (pred:manage_models)
- _8 more..._

**Streaming events** via `bytewax`:
`model_registered`, `model_trained`, `model_evaluated`, `model_deployed`, `prediction_completed`, ...

**Standalone usage:**
```bash
pip install apg-common-pred
apg-common-pred --port 8080
```

---

### Quantum Computing `quan`

> QUAN gives APG applications a tenant-scoped quantum lab runtime: backend registry, provider credentials posture, quota policies, circuit library, job submission, deterministic result capture, experiment workbench, quantum agents,

**Package**: `apg-common-quan`  
**Path**: `capabilities/common/quan`  
**Version**: 1.0.0  

**Provides:**
- `quantum_backend_registry`
- `circuit_management`
- `quantum_job_orchestration`
- `result_analysis`
- `post_quantum_governance`
- `quan_agents`

**Requires:**
- `aicr`
- `encr`
- `keym`
- `audl`

**Service methods** (46 total):
`describe`, `evaluate`, `submit_quantum_job`, `job_status`, `job_result`, `quantum_error_mitigation`, `variational_quantum_eigensolver`, `quantum_approximate_optimisation`, `quantum_key_distribution`, `post_quantum_encryption`, `quantum_simulation`, `quantum_analytics`, ...

**Governance rules** (25 total):
`tenant_context_required`, `backend_requires_approval`, `backend_requires_credentials_reference`, `backend_requires_qubit_capacity`, `circuit_requires_owner`, `circuit_requires_version`, `circuit_requires_qubits`, `circuit_requires_gates`, ...

**UI Routes** (10):
- `/quan/dashboard` — dashboard (quan:view)
- `/quan/backends` — backends (quan:manage_backends)
- `/quan/circuits` — circuits (quan:experiment)
- `/quan/jobs` — jobs (quan:run_jobs)
- `/quan/experiments` — experiments (quan:experiment)
- `/quan/results` — results (quan:view)
- _4 more..._

**Streaming events** via `bytewax`:
`quan_backend_registered`, `quan_quota_policy_attached`, `quan_circuit_created`, `quan_job_submitted`, `quan_result_recorded`, ...

**Standalone usage:**
```bash
pip install apg-common-quan
apg-common-quan --port 8080
```

---

### Retrieval-Augmented Generation `ragn`

> RAGN provides APG's executable retrieval-augmented generation capability: tenant-scoped knowledge bases, document ingestion, governed context retrieval, cited answer generation, conversation memory, answer curation, audit evidence,

**Package**: `apg-common-ragn`  
**Path**: `capabilities/common/ragn`  
**Version**: 1.0.0  

**Provides:**
- `retrieval_augmented_generation`
- `grounded_answering`
- `rag_agent_composition`

**Requires:**
- `srch`
- `nlpc`
- `aicr`
- `conf`
- `audl`

**Service methods** (47 total):
`uuid7str`, `start`, `stop`, `create_knowledge_base`, `get_knowledge_base`, `list_knowledge_bases`, `add_document`, `get_document`, `delete_document`, `query_knowledge_base`, `generate_response`, `create_conversation`, ...

**Governance rules** (46 total):
`tenant_context_required`, `knowledge_base_requires_id`, `knowledge_base_requires_name`, `knowledge_base_requires_owner`, `knowledge_base_requires_source_attribution`, `document_requires_knowledge_base`, `document_requires_title`, `document_requires_content_hash`, ...

**UI Routes** (14):
- `/ragn/dashboard` — dashboard (ragn:view)
- `/ragn/studio` — studio (ragn:query)
- `/ragn/knowledge-bases` — knowledge_bases (ragn:manage_kb)
- `/ragn/documents` — documents (ragn:manage_kb)
- `/ragn/retrieval` — retrieval (ragn:query)
- `/ragn/generation` — generation (ragn:query)
- _8 more..._

**Streaming events** via `bytewax`:
`document_ingested`, `document_chunked`, `embedding_indexed`, `index_refreshed`, `query_executed`, ...

**Standalone usage:**
```bash
pip install apg-common-ragn
apg-common-ragn --port 8080
```

---

### Recommender Systems `recs`

> RECS is APG's governed recommendation and personalization capability. It provides tenant-scoped recommendation datasets, interaction events, catalog items, user/profile features, ranking policies, model training, model approval,

**Package**: `apg-common-recs`  
**Path**: `capabilities/common/recs`  
**Version**: 1.0.0  

**Provides:**
- `personalized_recommendations`
- `ranking_policies`
- `catalog_matching`
- `interaction_datasets`
- `model_training`
- `model_deployments`
- `feedback_loops`
- `experiment_optimization`
- `recommender_agents`

**Requires:**
- `pred`
- `aicr`
- `nlpc`

**Service methods** (42 total):
`describe`, `evaluate`, `register_catalog_item`, `register_dataset`, `record_interaction`, `record_profile`, `attach_ranking_policy`, `train_model`, `approve_model`, `deploy_model`, `generate_recommendations`, `record_feedback`, ...

**Governance rules** (32 total):
`tenant_context_required`, `dataset_requires_owner`, `dataset_requires_source`, `dataset_requires_schema`, `dataset_requires_policy`, `interaction_event_requires_actor`, `interaction_event_requires_item`, `interaction_event_requires_timestamp`, ...

**UI Routes** (14):
- `/recs/dashboard` — dashboard (recs:view)
- `/recs/recommendations` — recommendations (recs:recommend)
- `/recs/datasets` — datasets (recs:manage_data)
- `/recs/models` — models (recs:manage_models)
- `/recs/deployments` — deployments (recs:deploy)
- `/recs/catalogs` — catalogs (recs:view)
- _8 more..._

**Streaming events** via `bytewax`:
`dataset_registered`, `interaction_recorded`, `catalog_item_registered`, `profile_recorded`, `ranking_policy_attached`, ...

**Standalone usage:**
```bash
pip install apg-common-recs
apg-common-recs --port 8080
```

---

### API/Service Registry `regy`

> REGY is APG's governed API and service registry. It lets generated applications register services and instances, discover healthy endpoints, govern API versions, publish registry evidence to gateway adapters, retire

**Package**: `apg-common-regy`  
**Path**: `capabilities/common/regy`  
**Version**: 1.0.0  

**Provides:**
- `service_registry`
- `service_discovery`
- `registry_agent_composition`
- `review_evidence`

**Requires:**
- `apig`
- `auth`
- `conf`

**Service methods** (44 total):
`append`, `initialize`, `_initialize_apg_integrations`, `_load_ml_models`, `_setup_health_monitoring`, `_initialize_circuit_breakers`, `register_service`, `deregister_service`, `discover_services`, `get_service_health`, `update_service_health`, `get_service_metrics`, ...

**Governance rules** (33 total):
`tenant_context_required`, `service_registration_requires_owner`, `service_registration_requires_health_endpoint`, `service_registration_requires_api_version`, `service_registration_requires_contract_schema`, `duplicate_service_name_blocked`, `production_registration_requires_review`, `instance_requires_endpoint`, ...

**UI Routes** (14):
- `/regy/dashboard` — dashboard (registry:view_statistics)
- `/regy/services` — services (registry:list_services)
- `/regy/register` — register (registry:register_service)
- `/regy/instances` — instances (registry:update_service)
- `/regy/discovery` — discovery (registry:discover_services)
- `/regy/health` — health (registry:view_health)
- _8 more..._

**Streaming events** via `bytewax`:
`service_registered`, `service_updated`, `service_activated`, `service_deactivated`, `service_retired`, ...

**Standalone usage:**
```bash
pip install apg-common-regy
apg-common-regy --port 8080
```

---

### Sandbox/Testing Environment `sbox`

> SBOX gives APG applications a tenant-scoped safe execution runtime: isolation profiles, sandbox templates, controlled datasets, sandbox environments, test runs, run completion evidence, sandbox governance agents, UI metadata, theme

**Package**: `apg-common-sbox`  
**Path**: `capabilities/common/sbox`  
**Version**: 1.0.0  

**Provides:**
- `sandbox_registry`
- `isolation_profiles`
- `test_runs`
- `synthetic_datasets`
- `safety_policy`
- `sbox_agents`

**Requires:**
- `plgn`
- `secu`
- `envm`
- `audl`

**Service methods** (41 total):
`describe`, `evaluate`, `create_sandbox`, `reset_sandbox`, `destroy_sandbox`, `sandbox_status`, `load_test_data`, `mock_service_register`, `simulate_event`, `run_test_scenario`, `sandbox_analytics`, `sandbox_cost_tracking`, ...

**Governance rules** (24 total):
`tenant_context_required`, `sandbox_requires_owner`, `sandbox_requires_template`, `sandbox_requires_isolation_profile`, `sandbox_requires_positive_ttl`, `secrets_require_redaction`, `outbound_network_requires_approval`, `long_lived_sandbox_requires_review`, ...

**UI Routes** (10):
- `/sbox/dashboard` — dashboard (sbox:view)
- `/sbox/sandboxes` — sandboxes (sbox:create)
- `/sbox/templates` — templates (sbox:create)
- `/sbox/datasets` — datasets (sbox:manage_policy)
- `/sbox/runs` — runs (sbox:run_tests)
- `/sbox/agents` — agents (sbox:admin)
- _4 more..._

**Streaming events** via `bytewax`:
`sbox_isolation_profile_created`, `sbox_template_created`, `sbox_dataset_registered`, `sbox_sandbox_created`, `sbox_run_started`, ...

**Standalone usage:**
```bash
pip install apg-common-sbox
apg-common-sbox --port 8080
```

---

### Scheduling and Job Orchestration `schd`

> `schd` is the APG common capability for governed schedules, calendar triggers, job definitions, worker pools, run recovery, and scheduler operations. It gives generated applications a dependency-light runtime that can define jobs, attach

**Package**: `apg-common-schd`  
**Path**: `capabilities/common/schd`  
**Version**: 1.0.0  

**Provides:**
- `job_scheduling`
- `calendar_triggers`
- `worker_orchestration`
- `retry_policies`
- `job_monitoring`
- `scheduler_agent_composition`
- `run_recovery`
- `bytewax_scheduler_lifecycle`

**Requires:**
- `wflo`
- `mqeb`
- `moni`
- `audl`
- `aicr`

**Service methods** (44 total):
`describe`, `evaluate`, `create_calendar_policy`, `register_worker_pool`, `change_worker_state`, `define_job`, `create_schedule`, `trigger_run`, `complete_run`, `retry_run`, `dead_letter_run`, `cancel_run`, ...

**Governance rules** (44 total):
`tenant_context_required`, `schedule_requires_owner`, `timezone_required`, `calendar_policy_required`, `worker_pool_required`, `manual_schedule_requires_reason`, `event_schedule_requires_policy`, `job_requires_owner`, ...

**UI Routes** (11):
- `/schd/dashboard` — dashboard (schd:view)
- `/schd/schedules` — schedules (schd:schedule)
- `/schd/jobs` — jobs (schd:run_jobs)
- `/schd/runs` — runs (schd:view)
- `/schd/workers` — workers (schd:manage_workers)
- `/schd/calendars` — calendars (schd:schedule)
- _5 more..._

**Streaming events** via `bytewax`:
`calendar_policy_created`, `worker_pool_registered`, `worker_pool_state_changed`, `job_defined`, `schedule_created`, ...

**Standalone usage:**
```bash
pip install apg-common-schd
apg-common-schd --port 8080
```

---

### Custom Scripting Engine `scpt`

> `scpt` is the APG common capability for governed custom scripting. It gives generated applications a dependency-light runtime for registering tenant-owned scripts, constraining them with package and sandbox policy, approving risky

**Package**: `apg-common-scpt`  
**Path**: `capabilities/common/scpt`  
**Version**: 1.0.0  

**Provides:**
- `script_registry`
- `secure_sandbox`
- `workflow_extensions`
- `package_policy`
- `script_execution`
- `scripting_agent_composition`
- `script_governance`
- `bytewax_script_lifecycle`

**Requires:**
- `wflo`
- `secu`
- `auth`
- `audl`
- `aicr`

**Service methods** (43 total):
`describe`, `evaluate`, `create_package_policy`, `create_sandbox`, `create_script`, `request_script_review`, `approve_script`, `publish_script`, `bind_workflow`, `execute_script`, `complete_execution`, `cancel_execution`, ...

**Governance rules** (46 total):
`tenant_context_required`, `script_requires_owner`, `script_requires_name`, `script_requires_source`, `script_requires_package_policy`, `script_requires_sandbox_policy`, `script_blocked_import_denied`, `sandbox_required`, ...

**UI Routes** (12):
- `/scpt/dashboard` — dashboard (scpt:view)
- `/scpt/workbench` — workbench (scpt:write)
- `/scpt/scripts` — scripts (scpt:view)
- `/scpt/executions` — executions (scpt:execute)
- `/scpt/sandboxes` — sandboxes (scpt:admin)
- `/scpt/packages` — packages (scpt:approve)
- _6 more..._

**Streaming events** via `bytewax`:
`package_policy_created`, `sandbox_created`, `sandbox_state_changed`, `script_created`, `script_reviewed`, ...

**Standalone usage:**
```bash
pip install apg-common-scpt
apg-common-scpt --port 8080
```

---

### Scraper/Data Harvesting `scrp`

> SCRP is the APG capability for governed data-source harvesting. It lets an APG application register tenant-owned sources, define extractors, schedule harvest jobs, run guarded harvest lifecycles, record result batches, hand results to

**Package**: `apg-common-scrp`  
**Path**: `capabilities/common/scrp`  
**Version**: 1.0.0  

**Provides:**
- `source_registry`
- `harvest_jobs`
- `extractor_profiles`
- `compliance_controls`
- `pipeline_handoff`
- `harvest_agents`

**Requires:**
- `conn`
- `etlp`
- `auth`

**Service methods** (45 total):
`describe`, `evaluate`, `schedule_scrape`, `run_scrape`, `scrape_result`, `extract_structured_data`, `javascript_rendered_scrape`, `rate_limit_management`, `proxy_rotation`, `captcha_handling`, `data_deduplication`, `scraping_analytics`, ...

**Governance rules** (21 total):
`tenant_context_required`, `source_requires_owner`, `source_terms_required`, `pii_requires_handling_policy`, `harvest_requires_schedule_policy`, `sensitive_source_requires_review`, `credential_vault_required`, `robots_policy_required`, ...

**UI Routes** (11):
- `/scrp/dashboard` — dashboard (scrp:view)
- `/scrp/sources` — sources (scrp:configure_sources)
- `/scrp/jobs` — jobs (scrp:run_jobs)
- `/scrp/extractors` — extractors (scrp:configure_sources)
- `/scrp/pipelines` — pipelines (scrp:view)
- `/scrp/compliance` — compliance (scrp:approve_harvests)
- _5 more..._

**Streaming events** via `bytewax`:
`source_registered`, `extractor_created`, `harvest_job_created`, `harvest_run_started`, `harvest_run_completed`, ...

**Standalone usage:**
```bash
pip install apg-common-scrp
apg-common-scrp --port 8080
```

---

### Security Framework `secu`

> The Security Framework capability (`secu`) is the executable security control plane for generated APG applications. It provides tenant-scoped security policy, device posture, threat indicator, access assessment, compliance

**Package**: `apg-common-secu`  
**Path**: `capabilities/common/secu`  
**Version**: 1.0.0  

**Provides:**
- `risk_assessment`
- `threat_detection`
- `security_policies`
- `compliance_automation`
- `incident_response_governance`
- `security_agents`
- `review_evidence`

**Requires:**
- `auth`
- `conf`
- `audl`

**Service methods** (79 total):
`initialize`, `_load_default_configurations`, `_load_security_policies`, `_set_config`, `get_config`, `update_policy`, `get_policies_for_context`, `_evaluate_policy_conditions`, `_evaluate_condition`, `_get_context_value`, `_log_security_event`, `calculate_risk_score`, ...

**Governance rules** (21 total):
`known_malicious_network_denied`, `compromised_device_quarantined`, `critical_risk_denied`, `high_risk_requires_challenge`, `compliance_violation_alert`, `policy_exception_requires_independent_reviewer`, `expired_policy_exception_denied`, `critical_incident_requires_containment`, ...

**UI Routes** (12):
- `/secu/dashboard` — dashboard (secu:view)
- `/secu/risk` — risk (secu:view_risk)
- `/secu/threats` — threats (secu:view_threats)
- `/secu/policies` — policies (secu:manage_policies)
- `/secu/exceptions` — exceptions (secu:approve_exception)
- `/secu/incidents` — incidents (secu:respond)
- _6 more..._

**Streaming events** via `bytewax`:
`policy_created`, `device_posture_recorded`, `device_quarantined`, `threat_indicator_registered`, `access_challenge`, ...

**Standalone usage:**
```bash
pip install apg-common-secu
apg-common-secu --port 8080
```

---

### Security Operations `seop`

> SEOP is the APG security-operations capability. It gives generated applications a composable runtime for detections, incident response, response playbooks, posture controls, audit evidence, governed AI agents, UI view models, visual theming, and Bytewax lifecycle events.

**Package**: `apg-common-seop`  
**Path**: `capabilities/common/seop`  
**Version**: 1.0.0  

**Provides:**
- `detection_pipeline`
- `incident_response`
- `threat_triage`
- `response_playbooks`
- `security_posture`
- `seop_agents`

**Requires:**
- `secu`
- `anom`
- `moni`
- `logt`
- `audl`

**Service methods** (45 total):
`describe`, `evaluate`, `create_detection`, `open_incident`, `approve_playbook`, `execute_response`, `record_posture_control`, `close_incident`, `create_soc_alert`, `triage_alert`, `escalate_to_incident`, `incident_response`, ...

**Governance rules** (20 total):
`tenant_context_required`, `detection_requires_alert_source`, `detection_requires_bytewax_stream`, `incident_requires_owner`, `incident_requires_evidence`, `critical_incident_requires_escalation`, `response_requires_playbook_approval`, `response_requires_actor`, ...

**UI Routes** (10):
- `/seop/dashboard` — dashboard (seop:view)
- `/seop/detections` — detections (seop:triage)
- `/seop/incidents` — incidents (seop:respond)
- `/seop/triage` — triage (seop:triage)
- `/seop/playbooks` — playbooks (seop:manage_playbooks)
- `/seop/responses` — responses (seop:respond)
- _4 more..._

**Streaming events** via `bytewax`:
`detection_created`, `incident_opened`, `playbook_approved`, `response_executed`, `incident_closed`, ...

**Standalone usage:**
```bash
pip install apg-common-seop
apg-common-seop --port 8080
```

---

### Shutdown and Lifecycle Control `shdn`

> SHDN is the APG capability for governed service lifecycle control. It gives generated applications a composable runtime for registering lifecycle targets, building shutdown plans, draining services, enforcing backup and health gates, executing shutdowns, recording recovery evidence, composing AI-assisted review, and emitting Bytewax lifecycle events.

**Package**: `apg-common-shdn`  
**Path**: `capabilities/common/shdn`  
**Version**: 1.0.0  

**Provides:**
- `service_lifecycle`
- `shutdown_orchestration`
- `restart_plans`
- `backup_gates`
- `operational_safety`
- `shdn_agents`

**Requires:**
- `moni`
- `hlth`
- `bkup`
- `audl`
- `envm`

**Service methods** (40 total):
`describe`, `evaluate`, `register_service`, `create_shutdown_plan`, `start_drain`, `record_backup_snapshot`, `execute_shutdown`, `record_recovery`, `create_record`, `list_records`, `list_targets`, `list_plans`, ...

**Governance rules** (20 total):
`tenant_context_required`, `service_requires_owner`, `service_requires_dependency_map`, `shutdown_requires_health_gate`, `shutdown_requires_backup_snapshot`, `shutdown_requires_actor`, `shutdown_requires_bytewax_stream`, `production_shutdown_requires_approval`, ...

**UI Routes** (10):
- `/shdn/dashboard` — dashboard (shdn:view)
- `/shdn/services` — services (shdn:view)
- `/shdn/plans` — plans (shdn:plan)
- `/shdn/executions` — executions (shdn:execute)
- `/shdn/approvals` — approvals (shdn:approve)
- `/shdn/recovery` — recovery (shdn:execute)
- _4 more..._

**Streaming events** via `bytewax`:
`target_registered`, `plan_created`, `drain_started`, `snapshot_recorded`, `shutdown_executed`, ...

**Standalone usage:**
```bash
pip install apg-common-shdn
apg-common-shdn --port 8080
```

---

### Search Engine `srch`

> SRCH is the APG capability for governed enterprise search across tenant-scoped indices, documents, facets, keyword retrieval, semantic retrieval, hybrid retrieval, and query analytics. It lets generated applications create indices,

**Package**: `apg-common-srch`  
**Path**: `capabilities/common/srch`  
**Version**: 1.0.0  

**Provides:**
- `enterprise_search`
- `semantic_retrieval`
- `search_agent_composition`

**Requires:**
- `etlp`
- `meta`
- `nlpc`
- `aicr`
- `conf`

**Service methods** (44 total):
`describe`, `evaluate`, `create_index`, `mark_embedding_index_ready`, `index_document`, `bulk_index_documents`, `query`, `facets`, `create_record`, `list_records`, `list_indices`, `list_documents`, ...

**Governance rules** (39 total):
`tenant_context_required`, `index_requires_name`, `indexing_requires_owner`, `index_requires_content_type`, `index_content_type_requires_review`, `index_requires_classification`, `index_classification_requires_review`, `restricted_index_requires_lineage`, ...

**UI Routes** (14):
- `/srch/dashboard` — dashboard (srch:view)
- `/srch/search` — search (srch:query)
- `/srch/indices` — indices (srch:manage_indices)
- `/srch/documents` — documents (srch:index)
- `/srch/bulk` — bulk (srch:index)
- `/srch/facets` — facets (srch:view)
- _8 more..._

**Streaming events** via `bytewax`:
`index_created`, `index_updated`, `index_rebuilt`, `document_indexed`, `document_removed`, ...

**Standalone usage:**
```bash
pip install apg-common-srch
apg-common-srch --port 8080
```

---

### Tenants Legacy `tens`

> TENS is the APG capability for legacy tenant compatibility and migration governance. It gives generated applications a composable runtime for legacy tenant registration, APG tenant mapping, access-boundary validation, migration approval, migration completion, deprecation planning, AI-assisted review, and Bytewax lifecycle events.

**Package**: `apg-common-tens`  
**Path**: `capabilities/common/tens`  
**Version**: 1.0.0  

**Provides:**
- `legacy_tenant_registry`
- `tenant_mapping`
- `migration_controls`
- `access_boundaries`
- `deprecation_governance`
- `tens_agents`

**Requires:**
- `mten`
- `auth`
- `audl`
- `idfd`
- `usrm`

**Service methods** (40 total):
`describe`, `evaluate`, `register_legacy_tenant`, `map_tenant`, `validate_access_boundary`, `create_migration_plan`, `complete_migration`, `record_deprecation_plan`, `create_record`, `list_records`, `list_legacy_tenants`, `list_mappings`, ...

**Governance rules** (21 total):
`tenant_context_required`, `legacy_tenant_requires_owner`, `legacy_tenant_requires_source_system`, `legacy_tenant_requires_compatibility_scope`, `mapping_requires_validation`, `mapping_requires_bytewax_stream`, `migration_requires_approval`, `migration_requires_rollback_plan`, ...

**UI Routes** (10):
- `/tens/dashboard` — dashboard (tens:view)
- `/tens/tenants` — tenants (tens:view)
- `/tens/mappings` — mappings (tens:map)
- `/tens/migrations` — migrations (tens:migrate)
- `/tens/boundaries` — boundaries (tens:approve)
- `/tens/deprecation` — deprecation (tens:approve)
- _4 more..._

**Streaming events** via `bytewax`:
`legacy_tenant_registered`, `tenant_mapped`, `boundary_validated`, `migration_plan_created`, `migration_completed`, ...

**Standalone usage:**
```bash
pip install apg-common-tens
apg-common-tens --port 8080
```

---

### UI/UX Theming and Branding `them`

> THEM is the APG capability for governed visual systems. It gives generated applications a composable runtime for tenant theme records, design tokens, brand assets, preview evidence, accessibility contrast gates, publication approvals,

**Package**: `apg-common-them`  
**Path**: `capabilities/common/them`  
**Version**: 1.0.0  

**Provides:**
- `theme_tokens`
- `brand_governance`
- `asset_libraries`
- `preview_workflows`
- `theme_publication_governance`
- `visual_theming`
- `them_agents`

**Requires:**
- `conf`
- `auth`
- `i18n`
- `audl`

**Service methods** (41 total):
`describe`, `evaluate`, `create_theme`, `update_tokens`, `add_brand_asset`, `create_preview`, `publish_theme`, `register_them_agent`, `validate_agent_theme_action`, `validate_batch_theme_rollout`, `apply_tenant_theme`, `get_theme_tokens`, ...

**Governance rules** (20 total):
`tenant_context_required`, `theme_requires_owner`, `theme_requires_guidelines`, `token_update_requires_reviewer`, `brand_asset_requires_license`, `brand_asset_requires_approval`, `preview_requires_artifact`, `publish_requires_approval`, ...

**UI Routes** (9):
- `/them/dashboard` — dashboard (them:view)
- `/them/themes` — themes (them:design)
- `/them/tokens` — tokens (them:design)
- `/them/branding` — branding (them:manage_brand)
- `/them/assets` — assets (them:manage_brand)
- `/them/preview` — preview (them:view)
- _3 more..._

**Streaming events** via `bytewax`:
`theme_created`, `tokens_updated`, `brand_asset_added`, `theme_preview_created`, `theme_published`, ...

**Standalone usage:**
```bash
pip install apg-common-them
apg-common-them --port 8080
```

---

### User Management `usrm`

> USRM is the APG capability for governed user lifecycle management. It gives generated applications a composable runtime for user identity, profiles, consented invitations, role assignment, privileged MFA, access reviews, privacy

**Package**: `apg-common-usrm`  
**Path**: `capabilities/common/usrm`  
**Version**: 1.0.0  

**Provides:**
- `user_directory`
- `profile_management`
- `consented_invitations`
- `role_assignment_governance`
- `access_review_workflows`
- `deprovisioning_governance`
- `user_audit_events`
- `usrm_agents`

**Requires:**
- `auth`
- `mfau`
- `cons`
- `audl`
- `idfd`

**Service methods** (40 total):
`describe`, `evaluate`, `create_user`, `update_profile`, `invite_user`, `assign_role`, `record_access_review`, `deprovision_user`, `bulk_suspend_users`, `register_usrm_agent`, `validate_agent_user_action`, `validate_batch_user_lifecycle`, ...

**Governance rules** (20 total):
`tenant_context_required`, `user_requires_identity`, `user_requires_owner`, `user_requires_profile_validation`, `invite_requires_consent_notice`, `invite_requires_bytewax_stream`, `profile_requires_privacy_sync`, `privileged_user_requires_mfa`, ...

**UI Routes** (10):
- `/usrm/dashboard` — dashboard (usrm:view)
- `/usrm/users` — users (usrm:manage_users)
- `/usrm/profiles` — profiles (usrm:manage_users)
- `/usrm/lifecycle` — lifecycle (usrm:manage_users)
- `/usrm/access` — access (usrm:review_access)
- `/usrm/privacy` — privacy (usrm:view)
- _4 more..._

**Streaming events** via `bytewax`:
`user_created`, `profile_updated`, `user_invited`, `role_assigned`, `access_review_recorded`, ...

**Standalone usage:**
```bash
pip install apg-common-usrm
apg-common-usrm --port 8080
```

---

### Video Conferencing `vidc`

> `vidc` provides APG's common capability for tenant-scoped video meetings. It composes meeting rooms, accountable hosts, waiting-room controls, participants, encrypted recordings, caption artifacts, AI meeting agents, first-class provider-neutral video agents, audit events, UI routes, visual theming, and Bytewax lifecycle guardrails into a generated-application packet that runs without live media infrastructure.

**Package**: `apg-common-vidc`  
**Path**: `capabilities/common/vidc`  
**Version**: 1.0.0  

**Service methods** (40 total):
`describe`, `evaluate`, `create_room`, `start_meeting`, `add_participant`, `create_recording`, `generate_captions`, `register_meeting_agent`, `register_video_agent`, `validate_vidc_lifecycle_batch`, `end_meeting`, `create_record`, ...

**Governance rules** (37 total):
`tenant_context_required`, `room_requires_name`, `room_requires_owner`, `room_requires_moderation_policy`, `meeting_requires_room`, `meeting_requires_host`, `meeting_requires_secure_transport`, `meeting_requires_screen_share_policy`, ...

**UI Routes** (11):
- `/vidc/dashboard` — dashboard (vidc:view)
- `/vidc/meetings` — meetings (vidc:schedule)
- `/vidc/rooms` — rooms (vidc:moderate)
- `/vidc/participants` — participants (vidc:moderate)
- `/vidc/recordings` — recordings (vidc:manage_recordings)
- `/vidc/captions` — captions (vidc:view)
- _5 more..._

**Streaming events** via `bytewax`:
`meeting_created`, `meeting_started`, `meeting_ended`, `participant_joined`, `participant_left`, ...

**Standalone usage:**
```bash
pip install apg-common-vidc
apg-common-vidc --port 8080
```

---

### Wallet and Payment Core `walt`

> WALT is the APG capability for governed wallet and payment operations. It gives generated applications a composable runtime for tenant wallets, payment instruments, transaction authorization, MFA checks, risk review, capture,

**Package**: `apg-common-walt`  
**Path**: `capabilities/common/walt`  
**Version**: 1.0.0  

**Provides:**
- `wallet_ledger`
- `payment_instruments`
- `transaction_authorization`
- `settlement`
- `reconciliation`
- `payment_risk_governance`
- `walt_agents`

**Requires:**
- `encr`
- `auth`
- `comp`
- `audl`
- `wflo`

**Service methods** (40 total):
`describe`, `evaluate`, `create_wallet`, `register_instrument`, `authorize_transaction`, `capture_transaction`, `create_settlement_batch`, `record_reconciliation`, `register_walt_agent`, `validate_agent_payment_action`, `validate_batch_settlement`, `create_record`, ...

**Governance rules** (20 total):
`tenant_context_required`, `wallet_requires_owner`, `wallet_requires_ledger`, `wallet_requires_compliance_policy`, `instrument_requires_encryption`, `instrument_requires_token`, `instrument_requires_verification`, `high_value_requires_mfa`, ...

**UI Routes** (10):
- `/walt/dashboard` — dashboard (walt:view)
- `/walt/wallets` — wallets (walt:manage_wallets)
- `/walt/transactions` — transactions (walt:authorize)
- `/walt/instruments` — instruments (walt:manage_wallets)
- `/walt/settlement` — settlement (walt:settle)
- `/walt/reconciliation` — reconciliation (walt:settle)
- _4 more..._

**Streaming events** via `bytewax`:
`wallet_created`, `instrument_registered`, `transaction_authorized`, `transaction_captured`, `settlement_batch_created`, ...

**Standalone usage:**
```bash
pip install apg-common-walt
apg-common-walt --port 8080
```

---

### Workflow Orchestration `wflo`

> `wflo` provides APG's common capability for governed workflow and process automation. It composes workflow definition, versioning, publication approval, trigger policy, retry policy, task routing, approval gates, execution state, event streams, compensation, first-class provider-neutral AI workflow agents, UI route metadata, visual theming, and Bytewax lifecycle guardrails.

**Package**: `apg-common-wflo`  
**Path**: `capabilities/common/wflo`  
**Version**: 1.0.0  

**Provides:**
- `workflow_definitions`
- `event_orchestration`
- `task_routing`
- `approval_flows`
- `execution_monitoring`
- `workflow_agent_composition`
- `review_evidence`
- `compensation_controls`
- `bytewax_workflow_lifecycle`

**Requires:**
- `mqeb`
- `auth`
- `audl`
- `aicr`

**Service methods** (42 total):
`describe`, `evaluate`, `create_workflow_definition`, `publish_workflow`, `retire_workflow`, `start_execution`, `create_task`, `claim_task`, `complete_task`, `escalate_task`, `request_approval`, `record_approval`, ...

**Governance rules** (43 total):
`tenant_context_required`, `workflow_requires_owner`, `workflow_requires_name`, `workflow_requires_steps`, `workflow_step_limit_review`, `workflow_duplicate_step_ids_blocked`, `workflow_requires_retry_policy`, `publish_requires_approval`, ...

**UI Routes** (11):
- `/wflo/dashboard` — dashboard (wflo:view)
- `/wflo/designer` — designer (wflo:design)
- `/wflo/definitions` — definitions (wflo:design)
- `/wflo/executions` — executions (wflo:view)
- `/wflo/tasks` — tasks (wflo:execute)
- `/wflo/approvals` — approvals (wflo:approve)
- _5 more..._

**Streaming events** via `bytewax`:
`workflow_created`, `workflow_published`, `workflow_retired`, `workflow_started`, `task_created`, ...

**Standalone usage:**
```bash
pip install apg-common-wflo
apg-common-wflo --port 8080
```

---

### Website Builder `wsbl`

> WSBL is the APG capability for governed website and page composition. It gives generated applications a composable runtime for tenant sites, domains, pages, components, public-site controls, publishing, rollback, accessibility,

**Package**: `apg-common-wsbl`  
**Path**: `capabilities/common/wsbl`  
**Version**: 1.0.0  

**Provides:**
- `site_management`
- `page_composition`
- `component_library`
- `publishing_workflows`
- `site_theming`
- `website_governance`
- `wsbl_agents`
- `review_evidence`

**Requires:**
- `them`
- `auth`
- `ncod`
- `accs`
- `cons`

**Service methods** (41 total):
`describe`, `evaluate`, `create_site`, `register_domain`, `validate_domain`, `create_component`, `review_component`, `create_page`, `add_page_section`, `create_publish_request`, `publish_site`, `rollback_site`, ...

**Governance rules** (20 total):
`tenant_context_required`, `site_requires_owner`, `domain_requires_validation_before_publish`, `page_requires_structured_sections`, `preview_requires_evidence`, `publish_requires_approval`, `publish_requires_bytewax_stream`, `custom_component_registration_requires_review`, ...

**UI Routes** (10):
- `/wsbl/dashboard` — dashboard (wsbl:view)
- `/wsbl/sites` — sites (wsbl:manage_sites)
- `/wsbl/pages` — pages (wsbl:build)
- `/wsbl/editor` — editor (wsbl:build)
- `/wsbl/components` — components (wsbl:build)
- `/wsbl/publishing` — publishing (wsbl:publish)
- _4 more..._

**Streaming events** via `bytewax`:
`site_created`, `domain_registered`, `domain_validated`, `component_created`, `component_reviewed`, ...

**Standalone usage:**
```bash
pip install apg-common-wsbl
apg-common-wsbl --port 8080
```

---

### Zero Trust Network Access `ztna`

> `ztna` is APG's package-backed Zero Trust Network Access capability. It gives generated applications a tenant-scoped access broker for identity, device posture, protected resources, access requests, access reviews, governed

**Package**: `apg-common-ztna`  
**Path**: `capabilities/common/ztna`  
**Version**: 1.0.0  

**Service methods** (40 total):
`describe`, `evaluate`, `register_identity`, `verify_identity`, `register_device`, `update_device_posture`, `register_resource`, `attach_resource_policy`, `request_access`, `approve_access_request`, `start_session`, `reevaluate_session`, ...

**Governance rules** (42 total):
`tenant_context_required`, `identity_subject_required`, `identity_display_name_required`, `identity_must_be_verified`, `suspended_identity_denied`, `federated_identity_requires_provider`, `device_requires_identity`, `device_posture_required`, ...

**UI Routes** (13):
- `/ztna/dashboard` — dashboard (ztna:view)
- `/ztna/policies` — policies (ztna:manage_policies)
- `/ztna/identities` — identities (ztna:manage_policies)
- `/ztna/devices` — devices (ztna:manage_devices)
- `/ztna/resources` — resources (ztna:manage_policies)
- `/ztna/access` — access (ztna:approve_access)
- _7 more..._

**Streaming events** via `bytewax`:
`policy_created`, `policy_updated`, `policy_activated`, `policy_deactivated`, `access_granted`, ...

**Standalone usage:**
```bash
pip install apg-common-ztna
apg-common-ztna --port 8080
```

---

## COMPOSITION

### Access Control Integration Hub `composition_access`

> The Access Control Integration Hub provides unified identity, policy, grant, and session management for the APG composition layer. It federates multiple identity providers (local, OIDC, SAML, LDAP, API key, JWT) behind a single policy engine, enforcing fine-grained resource access across all tenant composition boundaries.

**Package**: `apg-composition-access`  
**Path**: `capabilities/composition/access`  
**Version**: 1.2.0  

**Provides:**
- `identity_provider_composition`
- `resource_access_registry`
- `policy_orchestration`
- `grant_lifecycle`
- `session_risk_control`
- `access_decision_audit`
- `access_agents`
- `cross_tenant_isolation`
- `privilege_escalation_prevention`
- `circuit_breaker_enforcement`
- `cascading_failure_containment`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `conf`
- `composition_events`
- `moni`

**Service methods** (42 total):
`describe`, `evaluate`, `register_provider`, `activate_provider`, `register_resource`, `create_policy`, `activate_policy`, `create_grant`, `revoke_grant`, `evaluate_session`, `record_decision`, `register_access_agent`, ...

**Governance rules** (35 total):
`tenant_context_required`, `access_write_requires_policy`, `cross_tenant_access_blocked`, `cross_tenant_policy_blocked`, `privilege_escalation_blocked`, `grant_scope_exceeds_grantor_scope`, `circuit_breaker_open_blocks_requests`, `circuit_breaker_half_open_limits_throughput`, ...

**UI Routes** (12):
- `/composition-access/dashboard` — dashboard (composition_access:view)
- `/composition-access/providers` — providers (composition_access:admin)
- `/composition-access/resources` — resources (composition_access:govern)
- `/composition-access/policies` — policies (composition_access:govern)
- `/composition-access/grants` — grants (composition_access:grant)
- `/composition-access/decisions` — decisions (composition_access:view)
- _6 more..._

**Streaming events** via `bytewax`:
`provider_registered`, `provider_activated`, `provider_deactivated`, `resource_registered`, `policy_created`, ...

**Standalone usage:**
```bash
pip install apg-composition-access
apg-composition-access --port 8080
```

---

### Central Configuration Management `composition_config`

> Central Configuration Management is APG's shared configuration plane for the composition layer. It provides tenant-aware namespaces, versioned configuration values with schema validation, production deployment approval workflows, reusable template libraries, and continuous drift detection across all environments.

**Package**: `apg-composition-config`  
**Path**: `capabilities/composition/config`  
**Version**: 1.2.0  

**Provides:**
- `configuration_namespace_registry`
- `configuration_value_lifecycle`
- `configuration_schema_validation`
- `configuration_release_workflows`
- `configuration_template_library`
- `configuration_drift_monitoring`
- `config_agents`
- `cross_tenant_config_isolation`
- `circuit_breaker_config_gate`
- `cascading_config_failure_containment`
- `config_change_audit_trail`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_access`
- `composition_events`
- `moni`
- `conf`

**Service methods** (42 total):
`describe`, `evaluate`, `get_config`, `set_config`, `delete_config`, `list_configs`, `validate_schema`, `config_version_history`, `rollback_config`, `config_diff`, `bulk_config_import`, `config_analytics`, ...

**Governance rules** (28 total):
`tenant_context_required`, `configuration_requires_policy`, `cross_tenant_config_write_blocked`, `cross_tenant_template_reference_blocked`, `config_privilege_escalation_blocked`, `circuit_breaker_open_blocks_deployments`, `circuit_breaker_half_open_limits_deployments`, `circuit_breaker_trip_requires_event`, ...

**UI Routes** (11):
- `/composition-config/dashboard` — dashboard (composition_config:view)
- `/composition-config/namespaces` — namespaces (composition_config:admin)
- `/composition-config/configurations` — configurations (composition_config:edit)
- `/composition-config/releases` — releases (composition_config:release)
- `/composition-config/templates` — templates (composition_config:edit)
- `/composition-config/drift` — drift (composition_config:operate)
- _5 more..._

**Streaming events** via `bytewax`:
`namespace_registered`, `namespace_quarantined`, `configuration_created`, `configuration_validated`, `configuration_activated`, ...

**Standalone usage:**
```bash
pip install apg-composition-config
apg-composition-config --port 8080
```

---

### Event Streaming Bus `composition_events`

> The Event Streaming Bus is the foundational messaging backbone for the APG composition layer. It provides Bytewax-powered event streams with schema validation, producer attribution, consumer group management, stateful stream processors, dead-letter handling, and approved event replay. Every other composition capability routes its lifecycle events through this bus.

**Package**: `apg-composition-events`  
**Path**: `capabilities/composition/events`  
**Version**: 1.2.0  

**Provides:**
- `event_stream_registry`
- `bytewax_event_publishing`
- `event_schema_registry`
- `subscription_lifecycle`
- `stream_processor_topology`
- `dead_letter_operations`
- `event_agents`
- `cross_tenant_event_isolation`
- `circuit_breaker_event_gate`
- `cascading_failure_stream_containment`
- `event_replay_governance`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `moni`
- `conf`

**Service methods** (169 total):
`get`, `set`, `delete`, `close`, `from_url`, `_maybe_await`, `_commit`, `_append_to_bytewax`, `_resolve`, `start`, `stop`, `append`, ...

**Governance rules** (32 total):
`tenant_context_required`, `event_write_requires_policy`, `cross_tenant_publish_blocked`, `cross_tenant_subscription_blocked`, `stream_privilege_escalation_blocked`, `circuit_breaker_open_blocks_publish`, `circuit_breaker_half_open_limits_publish`, `circuit_breaker_trip_requires_event`, ...

**UI Routes** (11):
- `/composition-events/dashboard` — dashboard (composition_events:view)
- `/composition-events/streams` — streams (composition_events:manage_streams)
- `/composition-events/schemas` — schemas (composition_events:govern)
- `/composition-events/subscriptions` — subscriptions (composition_events:operate)
- `/composition-events/processors` — processors (composition_events:operate)
- `/composition-events/dead-letters` — dead_letters (composition_events:operate)
- _5 more..._

**Streaming events** via `bytewax`:
`stream_created`, `stream_quarantined`, `schema_registered`, `schema_breaking_change_reviewed`, `event_published`, ...

**Standalone usage:**
```bash
pip install apg-composition-events
apg-composition-events --port 8080
```

---

### API Service Mesh `composition_gateway`

> The API Service Mesh provides service discovery, intelligent routing, traffic management, TLS certificate lifecycle, and policy enforcement for all services exposed within the APG composition layer. It acts as the single ingress and inter-service control plane, ensuring that every public route is protected by a policy, a rate limiter, a circuit breaker, and a valid TLS certificate before traffic is allowed.

**Package**: `apg-composition-gateway`  
**Path**: `capabilities/composition/gateway`  
**Version**: 1.2.0  

**Provides:**
- `service_mesh_registry`
- `gateway_route_lifecycle`
- `traffic_management`
- `gateway_policy_enforcement`
- `certificate_lifecycle`
- `mesh_health_observability`
- `gateway_agents`
- `cross_tenant_mesh_isolation`
- `circuit_breaker_mesh_gate`
- `cascading_failure_mesh_containment`
- `mtls_identity_enforcement`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_registry`
- `composition_access`
- `composition_events`
- `moni`
- `conf`

**Service methods** (84 total):
`register_service`, `discover_services`, `route_request`, `select_endpoint`, `record_request_metrics`, `get_service_topology`, `get_mesh_status`, `create_natural_language_policy`, `generate_intelligent_topology`, `start_collaborative_session`, `execute_autonomous_remediation`, `apply_federated_learning_insights`, ...

**Governance rules** (30 total):
`tenant_context_required`, `gateway_write_requires_policy`, `cross_tenant_route_blocked`, `cross_tenant_policy_application_blocked`, `gateway_privilege_escalation_blocked`, `circuit_breaker_open_blocks_route`, `circuit_breaker_half_open_limits_traffic`, `circuit_breaker_trip_requires_event`, ...

**UI Routes** (11):
- `/composition-gateway/dashboard` — dashboard (composition_gateway:view)
- `/composition-gateway/services` — services (composition_gateway:manage_services)
- `/composition-gateway/routes` — routes (composition_gateway:manage_routes)
- `/composition-gateway/policies` — policies (composition_gateway:govern)
- `/composition-gateway/traffic` — traffic (composition_gateway:operate)
- `/composition-gateway/certificates` — certificates (composition_gateway:admin)
- _5 more..._

**Streaming events** via `bytewax`:
`service_registered`, `service_quarantined`, `route_created`, `route_deleted`, `policy_attached`, ...

**Standalone usage:**
```bash
pip install apg-composition-gateway
apg-composition-gateway --port 8080
```

---

### Workflow Orchestration `composition_orchestration`

> Workflow Orchestration provides the runtime engine for defining, validating, releasing, and executing multi-step business processes within the APG composition layer. It supports automated tasks, human task assignments, approval workflows, cross-capability integration tasks, transactional compensation, SLA escalation, and event-triggered execution — all coordinated through Bytewax.

**Package**: `apg-composition-orchestration`  
**Path**: `capabilities/composition/orchestration`  
**Version**: 1.2.0  

**Provides:**
- `workflow_definition_lifecycle`
- `workflow_graph_validation`
- `workflow_execution_lifecycle`
- `human_task_coordination`
- `workflow_release_governance`
- `workflow_rule_enforcement`
- `workflow_agents`
- `cross_tenant_workflow_isolation`
- `circuit_breaker_workflow_gate`
- `cascading_failure_workflow_containment`
- `saga_compensation_coordination`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_registry`
- `composition_events`
- `composition_config`
- `composition_access`
- `moni`

**Service methods** (41 total):
`define_workflow`, `release_workflow`, `start_execution`, `complete_task`, `assign_human_task`, `register_workflow_agent`, `validate_agent_workflow_action`, `validate_batch_schedule`, `create_workflow`, `start_instance`, `get_instance`, `advance_step`, ...

**Governance rules** (36 total):
`tenant_context_required`, `workflow_write_requires_policy`, `cross_tenant_execution_blocked`, `cross_tenant_task_delegation_blocked`, `workflow_privilege_escalation_blocked`, `circuit_breaker_open_blocks_execution`, `circuit_breaker_half_open_limits_executions`, `circuit_breaker_trip_requires_event`, ...

**UI Routes** (12):
- `/composition-orchestration/dashboard` — dashboard (composition_orchestration:view)
- `/composition-orchestration/definitions` — definitions (composition_orchestration:manage_definitions)
- `/composition-orchestration/designer` — designer (composition_orchestration:design)
- `/composition-orchestration/executions` — executions (composition_orchestration:operate)
- `/composition-orchestration/tasks` — tasks (composition_orchestration:manage_tasks)
- `/composition-orchestration/releases` — releases (composition_orchestration:release)
- _6 more..._

**Streaming events** via `bytewax`:
`workflow_defined`, `workflow_validated`, `workflow_released`, `workflow_execution_started`, `workflow_execution_advanced`, ...

**Standalone usage:**
```bash
pip install apg-composition-orchestration
apg-composition-orchestration --port 8080
```

---

### Capability Registry `composition_registry`

> The Capability Registry is the authoritative catalog and governance service for all APG capabilities. It stores capability metadata, manages dependency graphs with cycle detection, validates composition blueprints, governs version releases with compatibility evidence, and coordinates marketplace publication — all within the multi-tenant APG composition layer.

**Package**: `apg-composition-registry`  
**Path**: `capabilities/composition/registry`  
**Version**: 1.2.0  

**Provides:**
- `capability_catalog_lifecycle`
- `dependency_graph_management`
- `composition_blueprint_validation`
- `version_compatibility_governance`
- `marketplace_publication_governance`
- `registry_discovery`
- `registry_agents`
- `cross_tenant_registry_isolation`
- `circuit_breaker_registry_gate`
- `cascading_dependency_failure_containment`
- `capability_health_propagation`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `composition_access`
- `moni`
- `srch`

**Service methods** (51 total):
`register_capability`, `add_dependency`, `create_composition`, `publish_composition`, `validate_composition`, `release_version`, `deprecate_capability`, `publish_to_marketplace`, `register_registry_agent`, `validate_agent_registry_action`, `validate_import_batch`, `discover_capabilities`, ...

**Governance rules** (34 total):
`tenant_context_required`, `registry_write_requires_policy`, `cross_tenant_catalog_write_blocked`, `cross_tenant_composition_reference_blocked`, `registry_privilege_escalation_blocked`, `circuit_breaker_open_blocks_resolution`, `circuit_breaker_half_open_limits_resolution`, `circuit_breaker_trip_requires_event`, ...

**UI Routes** (12):
- `/composition-registry/dashboard` — dashboard (composition_registry:view)
- `/composition-registry/catalog` — catalog (composition_registry:manage_catalog)
- `/composition-registry/dependencies` — dependencies (composition_registry:manage_dependencies)
- `/composition-registry/compositions` — compositions (composition_registry:compose)
- `/composition-registry/versions` — versions (composition_registry:release)
- `/composition-registry/marketplace` — marketplace (composition_registry:publish)
- _6 more..._

**Streaming events** via `bytewax`:
`capability_registered`, `capability_quarantined`, `dependency_added`, `dependency_health_updated`, `health_state_propagated`, ...

**Standalone usage:**
```bash
pip install apg-composition-registry
apg-composition-registry --port 8080
```

---

## CRM

### Advanced CRM Analytics `crm_adv`

> Advanced CRM Analytics (`crm.adv`) is the full-lifecycle customer relationship management capability for the APG platform. It provides a governed, multi-tenant surface covering account management, contact relationship mapping, lead scoring and assignment, sales pipeline tracking, activity timelines, campaign governance, and forecast analytics — all wired to the APG event bus via Bytewax for real-time state propagation.

**Package**: `apg-crm-adv`  
**Path**: `capabilities/crm/adv`  
**Version**: 1.0.0  

**Provides:**
- `account_lifecycle`
- `contact_relationship_management`
- `lead_scoring_and_assignment`
- `sales_pipeline_management`
- `activity_timeline`
- `campaign_governance`
- `forecast_analytics`
- `crm_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `mdm`

**Service methods** (40 total):
`lead_capture`, `lead_scoring`, `lead_assignment`, `opportunity_create`, `opportunity_stage_advance`, `pipeline_report`, `customer_segmentation`, `campaign_analytics`, `win_loss_analysis`, `crm_dashboard`, `create_account`, `create_contact`, ...

**Governance rules** (28 total):
`tenant_context_required`, `crm_write_requires_policy`, `account_requires_owner`, `account_requires_segment`, `contact_outreach_requires_consent`, `lead_requires_source`, `lead_assignment_requires_score`, `lead_assignment_requires_policy`, ...

**UI Routes** (10):
- `/crm-adv/dashboard` — dashboard (crm_adv:view)
- `/crm-adv/accounts` — accounts (crm_adv:manage_accounts)
- `/crm-adv/contacts` — contacts (crm_adv:manage_contacts)
- `/crm-adv/leads` — leads (crm_adv:manage_leads)
- `/crm-adv/pipeline` — pipeline (crm_adv:manage_pipeline)
- `/crm-adv/activities` — activities (crm_adv:manage_activities)
- _4 more..._

**Streaming events** via `bytewax`:
`account_created`, `contact_created`, `lead_created`, `lead_assigned`, `opportunity_created`, ...

**Standalone usage:**
```bash
pip install apg-crm-adv
apg-crm-adv --port 8080
```

---

## EAM

### Enterprise Asset Management `eam_ast`

> Enterprise Asset Management (EAM) is the APG capability for the full lifecycle of physical capital assets: facilities, fleet, rotating equipment, tooling, production plant, and infrastructure. It provides a multi-tenant, policy-governed Python service surface that covers location hierarchy, asset master data, maintenance plans, work orders, inspections, condition readings, inventory reservations, and reliability analytics — all underpinned by deterministic guardrails and a Bytewax event stream.

**Package**: `apg-eam-ast`  
**Path**: `capabilities/eam/ast`  
**Version**: 1.0.0  

**Provides:**
- `asset_registry_lifecycle`
- `asset_location_hierarchy`
- `criticality_and_condition_management`
- `maintenance_plan_lifecycle`
- `work_order_lifecycle`
- `inspection_and_condition_readings`
- `asset_reliability_analytics`
- `eam_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`

**Service methods** (40 total):
`register_asset`, `asset_transfer`, `asset_disposal`, `depreciation_run`, `condition_assessment`, `maintenance_record`, `asset_insurance`, `warranty_tracking`, `asset_lifecycle_report`, `asset_register`, `register_location`, `create_maintenance_plan`, ...

**Governance rules** (32 total):
`tenant_context_required`, `eam_write_requires_policy`, `location_requires_type`, `asset_requires_owner`, `asset_requires_category`, `asset_requires_location`, `asset_requires_criticality`, `capital_asset_requires_fixed_asset_reference`, ...

**UI Routes** (10):
- `/eam-ast/dashboard` — dashboard (eam_ast:view)
- `/eam-ast/assets` — assets (eam_ast:manage_assets)
- `/eam-ast/locations` — locations (eam_ast:manage_locations)
- `/eam-ast/maintenance-plans` — maintenance_plans (eam_ast:manage_maintenance)
- `/eam-ast/work-orders` — work_orders (eam_ast:manage_work_orders)
- `/eam-ast/inspections` — inspections (eam_ast:inspect)
- _4 more..._

**Streaming events** via `bytewax`:
`location_registered`, `asset_registered`, `maintenance_plan_created`, `work_order_opened`, `work_order_completed`, ...

**Standalone usage:**
```bash
pip install apg-eam-ast
apg-eam-ast --port 8080
```

---

## ECD

### Sustainability and ESG Management `ecd_esg`

> The Sustainability and ESG Management capability provides end-to-end lifecycle management for Environmental, Social, and Governance programs within the APG platform. It covers the full data chain from tenant profile setup through framework selection, metric definition, measurement recording, target tracking, supplier assessment, initiative management, risk governance, report generation, and stakeholder engagement. Every write operation is gated by deterministic business rules enforced at the capability boundary, with all state transitions emitted to a Bytewax event stream for real-time observability.

**Package**: `apg-ecd-esg`  
**Path**: `capabilities/ecd/esg`  
**Version**: 2.1.0  

**Provides:**
- `esg_profile_lifecycle`
- `esg_framework_lifecycle`
- `esg_metric_lifecycle`
- `esg_measurement_lifecycle`
- `esg_target_lifecycle`
- `esg_supplier_assessment_lifecycle`
- `esg_initiative_lifecycle`
- `esg_risk_lifecycle`
- `esg_report_workflow`
- `esg_stakeholder_lifecycle`
- `esg_engagement_lifecycle`
- `esg_dashboard_service`
- `esg_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `wflo`
- `grc_doc`
- `mdm`
- `grc_rsa`

**Service methods** (40 total):
`esg_materiality_assessment`, `environmental_kpi_record`, `social_kpi_record`, `governance_score`, `esg_report_generation`, `sdg_alignment_mapping`, `supply_chain_esg_audit`, `biodiversity_impact`, `esg_rating_submission`, `esg_analytics`, `count`, `describe`, ...

**Governance rules** (59 total):
`tenant_context_required`, `operation_policy_required`, `profile_name_required`, `profile_industry_required`, `profile_country_required`, `profile_year_required`, `profile_owner_required`, `framework_profile_required`, ...

**UI Routes** (14):
- `/ecd/esg/dashboard` — dashboard (ecd_esg:view)
- `/ecd/esg/profiles` — profiles (ecd_esg:manage_profiles)
- `/ecd/esg/frameworks` — frameworks (ecd_esg:manage_frameworks)
- `/ecd/esg/metrics` — metrics (ecd_esg:manage_metrics)
- `/ecd/esg/measurements` — measurements (ecd_esg:record_data)
- `/ecd/esg/targets` — targets (ecd_esg:manage_targets)
- _8 more..._

**Streaming events** via `bytewax`:
`esg_profile_created`, `esg_framework_added`, `esg_metric_defined`, `esg_measurement_recorded`, `esg_target_set`, ...

**Standalone usage:**
```bash
pip install apg-ecd-esg
apg-ecd-esg --port 8080
```

---

## EDUCATION

### Learning Management System `education_lms`

> The LMS capability provides full lifecycle management for online and blended learning: course authoring, content delivery (including SCORM 1.2/2004 and xAPI), learner enrolment, assessment creation and grading, certificate issuance, learning path orchestration, and per-learner analytics. It enforces governance rules around grade overrides, certificate eligibility, analytics consent, and cross-tenant isolation.

**Package**: `apg-education-lms`  
**Path**: `capabilities/education/lms`  
**Version**: 1.0.0  

**Provides:**
- `course_lifecycle_workflow`
- `content_delivery_workflow`
- `enrolment_workflow`
- `assessment_workflow`
- `grading_workflow`
- `certificate_issuance_workflow`
- `learner_analytics_workflow`
- `scorm_xapi_compliance_workflow`
- `learning_path_workflow`
- `cohort_management_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `nlpc`
- `moni`
- `comp`
- `mqeb`
- `schd`

**Service methods** (40 total):
`describe`, `evaluate`, `create_course`, `get_course`, `list_courses`, `update_course`, `publish_course`, `archive_course`, `add_content_item`, `list_course_content`, `enrol_learner`, `list_enrolments`, ...

**Governance rules** (22 total):
`tenant_context_required`, `lms_write_requires_policy`, `course_type_supported`, `course_publish_requires_review`, `content_type_supported`, `scorm_version_supported`, `enrolment_type_supported`, `paid_enrolment_requires_payment_reference`, ...

**UI Routes** (14):
- `/lms/dashboard` — dashboard (education_lms:view)
- `/lms/courses` — courses (education_lms:view)
- `/lms/courses/create` — course_create (education_lms:manage_courses)
- `/lms/courses/<course_id>` — course_detail (education_lms:view)
- `/lms/courses/<course_id>/content` — content_builder (education_lms:manage_content)
- `/lms/enrolments` — enrolments (education_lms:manage_enrolments)
- _8 more..._

**Streaming events** via `bytewax`:
`course_created`, `course_published`, `course_archived`, `content_item_added`, `enrolment_recorded`, ...

**Standalone usage:**
```bash
pip install apg-education-lms
apg-education-lms --port 8080
```

---

### School Management `education_sch_mgmt`

> The School Management capability provides end-to-end administration for educational institutions: student records and lifecycle management, structured admissions workflows with capacity control, fee generation and payment tracking, staff administration, academic calendar management, document vault with consent-gated sharing, multi-channel communications, and reporting. All operations are tenant-scoped with strict governance on sensitive actions (expulsion, fee waivers, student data exports).

**Package**: `apg-education-sch_mgmt`  
**Path**: `capabilities/education/sch_mgmt`  
**Version**: 1.0.0  

**Provides:**
- `student_records_workflow`
- `admissions_workflow`
- `fee_management_workflow`
- `parent_portal_workflow`
- `staff_administration_workflow`
- `academic_calendar_workflow`
- `document_management_workflow`
- `communications_workflow`
- `reporting_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `mqeb`
- `schd`

**Service methods** (40 total):
`describe`, `evaluate`, `create_student`, `get_student`, `list_students`, `update_student_status`, `submit_application`, `update_admission_status`, `list_admissions`, `generate_fee_invoice`, `record_fee_payment`, `waive_fee`, ...

**Governance rules** (21 total):
`tenant_context_required`, `sch_mgmt_write_requires_policy`, `student_status_supported`, `expulsion_requires_approval`, `admission_status_supported`, `admission_offer_requires_capacity_check`, `fee_type_supported`, `fee_waiver_requires_approval`, ...

**UI Routes** (14):
- `/sch-mgmt/dashboard` — dashboard (education_sch_mgmt:view)
- `/sch-mgmt/students` — students (education_sch_mgmt:view_students)
- `/sch-mgmt/students/<student_id>` — student_detail (education_sch_mgmt:view_students)
- `/sch-mgmt/admissions` — admissions (education_sch_mgmt:manage_admissions)
- `/sch-mgmt/fees` — fees (education_sch_mgmt:manage_fees)
- `/sch-mgmt/fees/invoices` — fee_invoices (education_sch_mgmt:manage_fees)
- _8 more..._

**Streaming events** via `bytewax`:
`student_enrolled`, `student_status_changed`, `admission_submitted`, `admission_decision_recorded`, `fee_invoice_generated`, ...

**Standalone usage:**
```bash
pip install apg-education-sch_mgmt
apg-education-sch_mgmt --port 8080
```

---

### Timetabling & Scheduling `education_ttbl`

> The Timetabling capability provides constraint-based timetable generation and management for educational institutions. It supports master, class, teacher, room, and exam timetables; hard and soft constraint modelling; automated conflict detection (teacher double-booking, room double-booking, student group overlaps); conflict resolution workflows; room inventory management; teacher-consent-gated substitution management; multi-format export (iCal, CSV, PDF, JSON, HTML, Excel); and approval-gated publication. Publication is hard-blocked when any unresolved conflict remains.

**Package**: `apg-education-ttbl`  
**Path**: `capabilities/education/ttbl`  
**Version**: 1.0.0  

**Provides:**
- `timetable_generation_workflow`
- `constraint_management_workflow`
- `room_allocation_workflow`
- `teacher_assignment_workflow`
- `conflict_detection_workflow`
- `conflict_resolution_workflow`
- `substitution_management_workflow`
- `timetable_publication_workflow`
- `exam_scheduling_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `mqeb`
- `schd`
- `comp`

**Service methods** (40 total):
`describe`, `evaluate`, `create_timetable`, `get_timetable`, `list_timetables`, `publish_timetable`, `add_constraint`, `remove_constraint`, `list_constraints`, `create_room`, `list_rooms`, `create_time_slot`, ...

**Governance rules** (21 total):
`tenant_context_required`, `ttbl_write_requires_policy`, `timetable_type_supported`, `timetable_publish_requires_zero_conflicts`, `timetable_publish_requires_approval`, `constraint_type_supported`, `constraint_removal_requires_approval`, `slot_duration_supported`, ...

**UI Routes** (14):
- `/ttbl/dashboard` — dashboard (education_ttbl:view)
- `/ttbl/timetables` — timetables (education_ttbl:view)
- `/ttbl/timetables/<timetable_id>/build` — timetable_builder (education_ttbl:manage_timetables)
- `/ttbl/timetables/<timetable_id>/view` — timetable_viewer (education_ttbl:view)
- `/ttbl/constraints` — constraints (education_ttbl:manage_constraints)
- `/ttbl/rooms` — rooms (education_ttbl:manage_rooms)
- _8 more..._

**Streaming events** via `bytewax`:
`timetable_created`, `timetable_generation_started`, `timetable_generation_completed`, `conflict_detected`, `conflict_resolved`, ...

**Standalone usage:**
```bash
pip install apg-education-ttbl
apg-education-ttbl --port 8080
```

---

## ENERGY

### Energy Billing & Tariffs `energy_bil`

> Energy Billing & Tariffs manages the complete revenue cycle from tariff configuration through bill generation, payment processing, credit issuance, dispute resolution, and revenue assurance. It supports 13 tariff structures including time-of-use, demand charges, and net metering. Collection rates, write-off approvals, and revenue assurance flagging ensure financial governance across all customer classes.

**Package**: `apg-energy-bil`  
**Path**: `capabilities/energy/bil`  
**Version**: 1.0.0  

**Provides:**
- `tariff_management`
- `consumption_billing`
- `demand_charge_calculation`
- `renewable_credits_management`
- `revenue_assurance`
- `payment_processing`
- `dispute_management`
- `billing_analytics`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `mqeb`
- `schd`

**Service methods** (40 total):
`describe`, `evaluate`, `create_tariff`, `approve_tariff`, `activate_tariff`, `list_tariffs`, `get_active_tariff`, `generate_bill`, `issue_bill`, `write_off_bill`, `list_bills`, `record_payment`, ...

**Governance rules** (25 total):
`tenant_context_required`, `write_requires_policy`, `tariff_type_supported`, `tariff_customer_class_supported`, `tariff_effective_date_required`, `tariff_approval_required`, `tariff_rate_positive`, `bill_cycle_supported`, ...

**UI Routes** (12):
- `/energy-bil/dashboard` — dashboard (energy_bil:view)
- `/energy-bil/tariffs` — tariffs (energy_bil:tariffs)
- `/energy-bil/tariffs/<id>` — tariff_detail (energy_bil:tariffs)
- `/energy-bil/bills` — bills (energy_bil:billing)
- `/energy-bil/bills/<id>` — bill_detail (energy_bil:billing)
- `/energy-bil/payments` — payments (energy_bil:payments)
- _6 more..._

**Streaming events** via `bytewax`:
`tariff_created`, `tariff_approved`, `tariff_activated`, `bill_generated`, `bill_issued`, ...

**Standalone usage:**
```bash
pip install apg-energy-bil
apg-energy-bil --port 8080
```

---

### Distribution Network `energy_dis`

> Distribution Network manages the complete operational lifecycle of electricity distribution infrastructure. It provides network topology management for feeders and equipment, real-time fault detection and isolation, switching order workflows with live-network safety controls, outage recording with SAIDI/SAIFI reliability tracking, SCADA telemetry ingestion across multiple protocols, and automated load balancing with voltage constraint enforcement.

**Package**: `apg-energy-dis`  
**Path**: `capabilities/energy/dis`  
**Version**: 1.0.0  

**Provides:**
- `network_topology_management`
- `fault_detection_and_isolation`
- `outage_restoration`
- `switching_order_management`
- `scada_integration`
- `load_balancing`
- `reliability_kpis`
- `distribution_reporting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `schd`
- `mqeb`
- `geos`

**Service methods** (42 total):
`describe`, `evaluate`, `register_feeder`, `list_feeders`, `register_element`, `list_elements`, `report_fault`, `isolate_fault`, `restore_fault`, `dispatch_crew`, `list_faults`, `create_switching_order`, ...

**Governance rules** (24 total):
`tenant_context_required`, `write_requires_policy`, `element_type_supported`, `voltage_level_supported`, `element_feeder_required`, `fault_type_supported`, `fault_element_exists`, `fault_location_required`, ...

**UI Routes** (13):
- `/energy-dis/dashboard` — dashboard (energy_dis:view)
- `/energy-dis/topology` — topology (energy_dis:topology)
- `/energy-dis/elements` — elements (energy_dis:topology)
- `/energy-dis/faults` — faults (energy_dis:faults)
- `/energy-dis/faults/<id>` — fault_detail (energy_dis:faults)
- `/energy-dis/switching` — switching (energy_dis:switching)
- _7 more..._

**Streaming events** via `bytewax`:
`network_element_registered`, `topology_updated`, `fault_detected`, `fault_isolated`, `switching_order_created`, ...

**Standalone usage:**
```bash
pip install apg-energy-dis
apg-energy-dis --port 8080
```

---

### Generation Management `energy_gen`

> Generation Management provides end-to-end lifecycle management of power generation assets including thermal, hydro, and renewable plants. It covers plant registration, economic dispatch scheduling, outage management with approval workflows, KPI calculation (availability, capacity factor, heat rate), capacity planning, and fuel stock monitoring with low-supply alerting.

**Package**: `apg-energy-gen`  
**Path**: `capabilities/energy/gen`  
**Version**: 1.0.0  

**Provides:**
- `plant_registry`
- `dispatch_scheduling`
- `outage_management`
- `capacity_planning`
- `generation_kpis`
- `fuel_management`
- `performance_reporting`
- `dispatch_optimization`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `schd`
- `mqeb`
- `comp`

**Service methods** (40 total):
`describe`, `evaluate`, `register_plant`, `update_plant_status`, `list_plants`, `get_plant`, `decommission_plant`, `create_dispatch_schedule`, `approve_dispatch_schedule`, `list_dispatch_schedules`, `schedule_outage`, `approve_outage`, ...

**Governance rules** (28 total):
`tenant_context_required`, `write_requires_policy`, `plant_type_supported`, `plant_capacity_positive`, `plant_owner_required`, `plant_commissioning_date_required`, `fuel_type_supported`, `dispatch_mode_supported`, ...

**UI Routes** (13):
- `/energy-gen/dashboard` — dashboard (energy_gen:view)
- `/energy-gen/plants` — plants (energy_gen:plants)
- `/energy-gen/plants/<id>` — plant_detail (energy_gen:plants)
- `/energy-gen/dispatch` — dispatch (energy_gen:dispatch)
- `/energy-gen/schedules` — schedules (energy_gen:dispatch)
- `/energy-gen/outages` — outages (energy_gen:outages)
- _7 more..._

**Streaming events** via `bytewax`:
`plant_registered`, `plant_status_changed`, `dispatch_schedule_created`, `dispatch_schedule_approved`, `outage_scheduled`, ...

**Standalone usage:**
```bash
pip install apg-energy-gen
apg-energy-gen --port 8080
```

---

### Grid Operations `energy_grd`

> Grid Operations provides the real-time operational intelligence layer for power system management. It covers state estimation with convergence tracking, N-1/N-2 contingency analysis with automatic system status classification, voltage control via multiple methods (tap changers, SVCs, STATCOMs), frequency control including AGC and UFLS, market interval settlement with imbalance calculation, a full grid alarm management system with severity-gated acknowledgement, and EMS function execution in real-time and study modes.

**Package**: `apg-energy-grd`  
**Path**: `capabilities/energy/grd`  
**Version**: 1.0.0  

**Provides:**
- `real_time_state_estimation`
- `contingency_analysis`
- `voltage_control`
- `frequency_control`
- `market_settlement`
- `grid_alarm_management`
- `ems_function_management`
- `grid_operational_reporting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `comp`
- `mqeb`
- `schd`

**Service methods** (42 total):
`describe`, `evaluate`, `run_state_estimation`, `get_latest_se_run`, `list_se_runs`, `run_contingency`, `list_contingency_cases`, `apply_voltage_control`, `list_voltage_control_actions`, `apply_frequency_control`, `configure_ufls`, `list_frequency_control_actions`, ...

**Governance rules** (25 total):
`tenant_context_required`, `write_requires_policy`, `se_type_supported`, `se_network_model_required`, `se_measurements_required`, `contingency_type_supported`, `contingency_base_case_required`, `voltage_control_method_supported`, ...

**UI Routes** (13):
- `/energy-grd/dashboard` — dashboard (energy_grd:view)
- `/energy-grd/state-estimation` — state_estimation (energy_grd:state_estimation)
- `/energy-grd/contingency` — contingency (energy_grd:contingency)
- `/energy-grd/contingency/<id>` — contingency_detail (energy_grd:contingency)
- `/energy-grd/voltage-control` — voltage_control (energy_grd:voltage_control)
- `/energy-grd/frequency-control` — frequency_control (energy_grd:frequency_control)
- _7 more..._

**Streaming events** via `bytewax`:
`state_estimation_completed`, `contingency_violation_detected`, `contingency_cleared`, `voltage_control_action_taken`, `frequency_control_action_taken`, ...

**Standalone usage:**
```bash
pip install apg-energy-grd
apg-energy-grd --port 8080
```

---

### Smart Metering & AMI `energy_met`

> Smart Metering & AMI manages the full lifecycle of advanced metering infrastructure from meter registration through interval data collection, tamper detection with evidence workflows, remote connect/disconnect with approval controls, demand response event coordination with customer opt-out, and data quality flagging. It also monitors AMI head-end connectivity ratios across communication technologies.

**Package**: `apg-energy-met`  
**Path**: `capabilities/energy/met`  
**Version**: 1.0.0  

**Provides:**
- `meter_registry`
- `ami_head_end_management`
- `interval_data_collection`
- `tamper_detection`
- `remote_connect_disconnect`
- `demand_response_coordination`
- `data_quality_management`
- `meter_data_export`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `mqeb`
- `schd`

**Service methods** (41 total):
`describe`, `evaluate`, `register_meter`, `update_meter_status`, `list_meters`, `get_meter`, `submit_reading`, `list_readings`, `report_tamper`, `resolve_tamper`, `list_tamper_events`, `issue_command`, ...

**Governance rules** (24 total):
`tenant_context_required`, `write_requires_policy`, `meter_type_supported`, `meter_serial_required`, `meter_comm_tech_supported`, `meter_location_required`, `reading_type_supported`, `reading_interval_supported`, ...

**UI Routes** (12):
- `/energy-met/dashboard` — dashboard (energy_met:view)
- `/energy-met/meters` — meters (energy_met:meters)
- `/energy-met/meters/<id>` — meter_detail (energy_met:meters)
- `/energy-met/readings` — readings (energy_met:readings)
- `/energy-met/tamper` — tamper (energy_met:tamper)
- `/energy-met/commands` — commands (energy_met:commands)
- _6 more..._

**Streaming events** via `bytewax`:
`meter_registered`, `meter_status_changed`, `interval_reading_received`, `tamper_event_detected`, `remote_command_sent`, ...

**Standalone usage:**
```bash
pip install apg-energy-met
apg-energy-met --port 8080
```

---

### Renewable Energy `energy_ren`

> Renewable Energy manages the full lifecycle of renewable generation assets — solar PV, wind, hydro, biomass, geothermal and others. It tracks curtailment events with revenue loss accounting, issues and retires Renewable Energy Certificates (RECs) with double-issuance prevention, manages carbon credits requiring third-party verification, administers feed-in tariffs, publishes multi-horizon generation forecasts, and computes performance metrics against benchmarks.

**Package**: `apg-energy-ren`  
**Path**: `capabilities/energy/ren`  
**Version**: 1.0.0  

**Provides:**
- `renewable_asset_registry`
- `curtailment_tracking`
- `rec_certificate_management`
- `carbon_credit_management`
- `feed_in_tariff_management`
- `generation_forecasting`
- `renewable_performance_analytics`
- `green_energy_reporting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `comp`
- `mqeb`
- `schd`

**Service methods** (42 total):
`describe`, `evaluate`, `register_asset`, `update_asset_status`, `list_assets`, `get_asset`, `record_curtailment`, `approve_curtailment`, `list_curtailments`, `get_curtailment_summary`, `issue_rec`, `transfer_rec`, ...

**Governance rules** (24 total):
`tenant_context_required`, `write_requires_policy`, `renewable_type_supported`, `asset_capacity_positive`, `asset_commissioning_date_required`, `asset_location_required`, `curtailment_reason_supported`, `curtailment_mwh_positive`, ...

**UI Routes** (12):
- `/energy-ren/dashboard` — dashboard (energy_ren:view)
- `/energy-ren/assets` — assets (energy_ren:assets)
- `/energy-ren/assets/<id>` — asset_detail (energy_ren:assets)
- `/energy-ren/curtailment` — curtailment (energy_ren:curtailment)
- `/energy-ren/recs` — recs (energy_ren:recs)
- `/energy-ren/carbon-credits` — carbon_credits (energy_ren:carbon_credits)
- _6 more..._

**Streaming events** via `bytewax`:
`renewable_asset_registered`, `asset_status_changed`, `curtailment_event_created`, `curtailment_event_approved`, `rec_issued`, ...

**Standalone usage:**
```bash
pip install apg-energy-ren
apg-energy-ren --port 8080
```

---

## FIN

### Accounts Payable `apy_accounts_payable`

> `apy_accounts_payable` is the APG capability for composing vendor liability, invoice, matching, approval, payment, reimbursement, close, and AP-agent workflows into generated Python applications. It provides an executable service surface, deterministic guardrails, UI metadata, theme metadata, and Bytewax lifecycle-stream declarations.

**Package**: `apg-fin-accounts_payable`  
**Path**: `capabilities/fin/apy/accounts_payable`  
**Version**: 1.0.0  

**Provides:**
- `vendor_payables_lifecycle`
- `invoice_capture_and_matching`
- `approval_workflow`
- `payment_run_lifecycle`
- `expense_reimbursement_lifecycle`
- `ap_aging_and_close`
- `ap_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `glr_general_ledger`
- `cbm_cash_management`
- `grc_doc`

**Service methods** (49 total):
`register_vendor`, `record_invoice`, `match_invoice`, `approve_invoice`, `place_invoice_hold`, `release_invoice_hold`, `schedule_payment`, `release_payment_batch`, `record_expense_report`, `close_period`, `register_ap_agent`, `validate_agent_ap_action`, ...

**Governance rules** (35 total):
`tenant_context_required`, `ap_write_requires_policy`, `vendor_requires_owner`, `vendor_requires_tax_profile`, `vendor_requires_payment_method`, `vendor_bank_change_requires_review`, `invoice_requires_vendor`, `invoice_requires_number`, ...

**UI Routes** (11):
- `/apy-accounts-payable/dashboard` — dashboard (apy_accounts_payable:view)
- `/apy-accounts-payable/vendors` — vendors (apy_accounts_payable:manage_vendors)
- `/apy-accounts-payable/invoices` — invoices (apy_accounts_payable:manage_invoices)
- `/apy-accounts-payable/matching` — matching (apy_accounts_payable:match)
- `/apy-accounts-payable/approvals` — approvals (apy_accounts_payable:approve)
- `/apy-accounts-payable/payments` — payments (apy_accounts_payable:pay)
- _5 more..._

**Streaming events** via `bytewax`:
`vendor_registered`, `invoice_recorded`, `invoice_matched`, `invoice_approved`, `invoice_hold_placed`, ...

**Standalone usage:**
```bash
pip install apg-fin-accounts_payable
apg-fin-accounts_payable --port 8080
```

---

### Accounts Receivable `arc_accounts_receivable`

> `arc_accounts_receivable` is the APG financial capability for customer receivables. It provides a composable lifecycle for customers, credit assessment, invoices, payment receipts, cash application, collections, disputes, aging, and receivables-focused AI agent review.

**Package**: `apg-fin-accounts_receivable`  
**Path**: `capabilities/fin/arc/accounts_receivable`  
**Version**: 2.1.0  

**Provides:**
- `customer_receivable_lifecycle`
- `credit_assessment_workflow`
- `invoice_lifecycle`
- `invoice_line_management`
- `payment_receipt_lifecycle`
- `cash_application_workflow`
- `collections_workflow`
- `dispute_resolution_workflow`
- `receivables_aging_service`
- `arc_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `mqeb`
- `glr_general_ledger`
- `cbm_cash_management`

**Service methods** (72 total):
`uuid7str`, `_log_audit`, `_log_notify`, `_put`, `_get`, `_require`, `_query`, `create_customer`, `update_customer`, `get_customer`, `list_customers`, `apply_credit_hold`, ...

**Governance rules** (39 total):
`tenant_context_required`, `arc_write_requires_policy`, `customer_requires_code`, `customer_requires_legal_name`, `customer_type_supported`, `credit_requires_customer`, `credit_limit_required`, `low_credit_score_requires_review`, ...

**UI Routes** (11):
- `/arc-accounts-receivable/dashboard` — dashboard (arc_accounts_receivable:view)
- `/arc-accounts-receivable/customers` — customers (arc_accounts_receivable:manage_customers)
- `/arc-accounts-receivable/credit` — credit (arc_accounts_receivable:credit)
- `/arc-accounts-receivable/invoices` — invoices (arc_accounts_receivable:invoice)
- `/arc-accounts-receivable/payments` — payments (arc_accounts_receivable:receive_payments)
- `/arc-accounts-receivable/cash-application` — cash_application (arc_accounts_receivable:apply_cash)
- _5 more..._

**Streaming events** via `bytewax`:
`customer_created`, `credit_assessed`, `invoice_created`, `invoice_issued`, `payment_recorded`, ...

**Standalone usage:**
```bash
pip install apg-fin-accounts_receivable
apg-fin-accounts_receivable --port 8080
```

---

### Budgeting and Forecasting `bfc_budgeting_forecasting`

> `bfc_budgeting_forecasting` is the APG capability for composing budget planning, financial forecasting, scenario planning, variance analysis, planning collaboration, and budget approval workflows into generated Python applications. It provides an executable service surface, deterministic guardrails, UI metadata, theme metadata, and Bytewax lifecycle-stream declarations.

**Package**: `apg-fin-budgeting_forecasting`  
**Path**: `capabilities/fin/bfc/budgeting_forecasting`  
**Version**: 1.0.0  

**Provides:**
- `budget_planning_lifecycle`
- `budget_line_management`
- `budget_approval_workflow`
- `forecast_lifecycle`
- `scenario_planning`
- `variance_analysis_lifecycle`
- `planning_collaboration`
- `bfc_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `glr_general_ledger`
- `apy_accounts_payable`
- `arc_accounts_receivable`
- `cbm_cash_management`
- `bia_anl`

**Service methods** (43 total):
`as_actor`, `create_budget_cycle`, `update_budget`, `add_budget_line`, `submit_budget`, `approve_budget`, `reject_budget`, `lock_budget`, `close_budget`, `cancel_budget`, `get_budget`, `list_budgets`, ...

**Governance rules** (35 total):
`tenant_context_required`, `bfc_write_requires_policy`, `budget_requires_owner`, `budget_requires_fiscal_year`, `budget_requires_currency`, `budget_requires_period_dates`, `budget_period_end_after_start`, `budget_line_requires_budget`, ...

**UI Routes** (10):
- `/bfc-budgeting-forecasting/dashboard` — dashboard (bfc_budgeting_forecasting:view)
- `/bfc-budgeting-forecasting/budgets` — budgets (bfc_budgeting_forecasting:manage_budgets)
- `/bfc-budgeting-forecasting/budget-lines` — budget_lines (bfc_budgeting_forecasting:manage_budgets)
- `/bfc-budgeting-forecasting/forecasts` — forecasts (bfc_budgeting_forecasting:forecast)
- `/bfc-budgeting-forecasting/scenarios` — scenarios (bfc_budgeting_forecasting:scenario)
- `/bfc-budgeting-forecasting/variances` — variances (bfc_budgeting_forecasting:analyze)
- _4 more..._

**Streaming events** via `bytewax`:
`budget_created`, `budget_line_added`, `budget_submitted`, `budget_approved`, `forecast_created`, ...

**Standalone usage:**
```bash
pip install apg-fin-budgeting_forecasting
apg-fin-budgeting_forecasting --port 8080
```

---

### Cash Management `cbm_cash_management`

> `cbm_cash_management` is the APG treasury liquidity packet. It owns bank relationships, cash accounts, cash positions, cash flows, forecasts, liquidity reviews, bank reconciliation, treasury investments, payment-run funding checks,

**Package**: `apg-fin-cash_management`  
**Path**: `capabilities/fin/cbm/cash_management`  
**Version**: 2.1.0  

**Provides:**
- `bank_relationship_lifecycle`
- `cash_account_lifecycle`
- `cash_position_service`
- `cash_flow_lifecycle`
- `cash_forecasting_workflow`
- `liquidity_control_workflow`
- `bank_reconciliation_workflow`
- `treasury_investment_workflow`
- `payment_run_funding_control`
- `cbm_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `mqeb`
- `wflo`
- `glr_general_ledger`

**Service methods** (49 total):
`bank_account_balance`, `import_bank_statement`, `auto_reconcile_statement`, `manual_match`, `reconciliation_report`, `cash_position_report`, `liquidity_forecast`, `fx_position`, `cash_pooling_sweep`, `intercompany_settlement`, `bank_covenant_compliance`, `mobile_money_reconciliation`, ...

**Governance rules** (36 total):
`tenant_context_required`, `cbm_write_requires_policy`, `bank_requires_code`, `bank_requires_name`, `cash_account_requires_bank`, `cash_account_requires_number`, `cash_account_requires_name`, `cash_account_type_supported`, ...

**UI Routes** (12):
- `/cbm-cash-management/dashboard` — dashboard (cbm_cash_management:view)
- `/cbm-cash-management/banks` — banks (cbm_cash_management:manage_banks)
- `/cbm-cash-management/accounts` — accounts (cbm_cash_management:manage_accounts)
- `/cbm-cash-management/positions` — positions (cbm_cash_management:view_positions)
- `/cbm-cash-management/flows` — flows (cbm_cash_management:manage_flows)
- `/cbm-cash-management/forecasts` — forecasts (cbm_cash_management:forecast)
- _6 more..._

**Streaming events** via `bytewax`:
`bank_created`, `cash_account_created`, `cash_position_recorded`, `cash_flow_recorded`, `cash_forecast_created`, ...

**Standalone usage:**
```bash
pip install apg-fin-cash_management
apg-fin-cash_management --port 8080
```

---

### Financial Reporting `fin_rpt`

> `fin_rpt` is the APG capability for composing financial report templates, report lines, reporting periods, statement generation, statement publication, consolidation, disclosures, and report distribution into generated Python applications. It provides an executable service surface, deterministic guardrails, UI metadata, theme metadata, and Bytewax lifecycle-stream declarations.

**Package**: `apg-fin-rpt`  
**Path**: `capabilities/fin/rpt`  
**Version**: 1.0.0  

**Provides:**
- `financial_report_template_lifecycle`
- `report_line_mapping`
- `reporting_period_lifecycle`
- `financial_statement_generation`
- `statement_publication_workflow`
- `financial_consolidation`
- `disclosure_management`
- `report_distribution`
- `rpt_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `glr_general_ledger`
- `apy_accounts_payable`
- `arc_accounts_receivable`
- `cbm_cash_management`
- `grc_doc`
- `bia_anl`

**Service methods** (40 total):
`create_template`, `add_report_line`, `open_period`, `generate_report`, `publish_statement`, `create_consolidation`, `record_disclosure`, `distribute_statement`, `register_rpt_agent`, `validate_agent_rpt_action`, `validate_batch`, `generate_ifrs_income_statement`, ...

**Governance rules** (35 total):
`tenant_context_required`, `rpt_write_requires_policy`, `template_requires_name`, `template_statement_type_supported`, `report_line_requires_template`, `report_line_requires_account_mapping`, `report_line_requires_sort_order`, `period_requires_name`, ...

**UI Routes** (11):
- `/fin-rpt/dashboard` — dashboard (fin_rpt:view)
- `/fin-rpt/templates` — templates (fin_rpt:manage_templates)
- `/fin-rpt/lines` — lines (fin_rpt:manage_templates)
- `/fin-rpt/periods` — periods (fin_rpt:manage_periods)
- `/fin-rpt/generation` — generation (fin_rpt:generate)
- `/fin-rpt/statements` — statements (fin_rpt:publish)
- _5 more..._

**Streaming events** via `bytewax`:
`template_created`, `report_line_added`, `period_opened`, `report_generated`, `statement_published`, ...

**Standalone usage:**
```bash
pip install apg-fin-rpt
apg-fin-rpt --port 8080
```

---

### Financial Management General Ledger `glr_general_ledger`

> `glr_general_ledger` is the APG financial system of record. It owns chart of accounts, ledger dimensions, accounting periods, journal batches, balanced journal entries, postings, reversals, allocations, trial-balance production,

**Package**: `apg-fin-general_ledger`  
**Path**: `capabilities/fin/glr/general_ledger`  
**Version**: 2.1.0  

**Provides:**
- `chart_of_accounts_lifecycle`
- `ledger_dimension_management`
- `accounting_period_lifecycle`
- `journal_batch_lifecycle`
- `journal_entry_lifecycle`
- `journal_posting_workflow`
- `ledger_balance_service`
- `trial_balance_reporting`
- `allocation_and_reversal_workflow`
- `glr_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `mqeb`
- `wflo`
- `srch`

**Service methods** (57 total):
`create_account`, `record_dimension`, `open_period`, `create_journal_batch`, `create_journal_entry`, `approve_journal`, `post_journal`, `reverse_journal`, `record_currency_rate`, `create_allocation`, `generate_trial_balance`, `register_glr_agent`, ...

**Governance rules** (36 total):
`tenant_context_required`, `glr_write_requires_policy`, `account_requires_code`, `account_requires_name`, `account_type_supported`, `account_parent_cycle_blocked`, `period_requires_name`, `period_requires_fiscal_year`, ...

**UI Routes** (12):
- `/glr-general-ledger/dashboard` — dashboard (glr_general_ledger:view)
- `/glr-general-ledger/accounts` — accounts (glr_general_ledger:manage_accounts)
- `/glr-general-ledger/dimensions` — dimensions (glr_general_ledger:manage_dimensions)
- `/glr-general-ledger/periods` — periods (glr_general_ledger:manage_periods)
- `/glr-general-ledger/batches` — journal_batches (glr_general_ledger:enter_journals)
- `/glr-general-ledger/journals` — journals (glr_general_ledger:enter_journals)
- _6 more..._

**Streaming events** via `bytewax`:
`account_created`, `dimension_recorded`, `period_opened`, `journal_batch_created`, `journal_entry_created`, ...

**Standalone usage:**
```bash
pip install apg-fin-general_ledger
apg-fin-general_ledger --port 8080
```

---

## FINTECH

### Agency Banking `fintech_agency`

> Agency Banking extends financial services reach through a network of accredited third-party outlets — retail shops, pharmacies, petrol stations, mobile agents, cooperatives, and community banks — operating under a governed program structure. Each outlet holds a float account, serves KYC/AML-verified customers, and processes transactions across services including cash-in/out, bill payment, airtime, loan disbursement, card services, and government payments.

**Package**: `apg-fintech-agency`  
**Path**: `capabilities/fintech/agency`  
**Version**: 1.1.0  

**Provides:**
- `agency_program_governance`
- `agency_outlet_lifecycle`
- `agency_agent_accreditation`
- `agency_float_management`
- `agency_customer_workflow`
- `agency_transaction_workflow`
- `agency_cash_movement_workflow`
- `agency_commission_settlement_workflow`
- `agency_dispute_workflow`
- `agency_supervision_workflow`
- `agency_ai_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_cards`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_remittance`
- `fintech_neobanking`
- `fintech_lending`

**Service methods** (40 total):
`describe`, `evaluate`, `register_program`, `onboard_outlet`, `accredit_agent`, `open_float_account`, `onboard_customer`, `record_transaction`, `record_cash_movement`, `settle_commission`, `open_dispute`, `record_supervision_visit`, ...

**Governance rules** (77 total):
`tenant_context_required`, `agency_write_requires_policy`, `program_owner_required`, `program_country_supported`, `program_currency_supported`, `program_settlement_model_supported`, `program_services_required`, `outlet_program_required`, ...

**UI Routes** (13):
- `/fintech-agency/dashboard` — dashboard (fintech_agency:view)
- `/fintech-agency/programs` — programs (fintech_agency:manage_programs)
- `/fintech-agency/outlets` — outlets (fintech_agency:manage_outlets)
- `/fintech-agency/agents` — agents (fintech_agency:manage_agents)
- `/fintech-agency/float-accounts` — float_accounts (fintech_agency:float)
- `/fintech-agency/customers` — customers (fintech_agency:customers)
- _7 more..._

**Streaming events** via `bytewax`:
`agency_program_registered`, `agency_outlet_onboarded`, `agency_agent_accredited`, `float_account_opened`, `agency_customer_onboarded`, ...

**Standalone usage:**
```bash
pip install apg-fintech-agency
apg-fintech-agency --port 8080
```

---

### Anti Money Laundering `fintech_aml`

> Anti Money Laundering provides real-time transaction monitoring, typology-driven alert generation, sanctions and PEP screening escalation, AML case investigation, and Suspicious Activity Report (SAR) drafting workflows. It acts as the AML control layer across all payment-generating capabilities, receiving transaction signals, applying velocity/structuring/sanctions rules, and routing findings to human analysts or AI-assisted reviewers.

**Package**: `apg-fintech-aml`  
**Path**: `capabilities/fintech/aml`  
**Version**: 1.1.0  

**Provides:**
- `transaction_monitoring`
- `aml_alert_triage`
- `sanctions_pep_escalation`
- `suspicious_activity_case_management`
- `sar_workflow`
- `typology_rule_engine`
- `aml_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`

**Service methods** (71 total):
`_emit_event`, `create_rule`, `get_rule`, `update_rule`, `delete_rule`, `list_rules`, `monitor_transaction`, `evaluate_rules`, `_evaluate_single_rule`, `generate_alert`, `create_alert`, `get_alert`, ...

**Governance rules** (42 total):
`tenant_context_required`, `aml_write_requires_policy`, `transaction_subject_required`, `transaction_amount_required`, `transaction_currency_required`, `transaction_source_required`, `transaction_requires_kyc_link`, `large_transaction_requires_review`, ...

**UI Routes** (8):
- `/fintech-aml/dashboard` — dashboard (fintech_aml:view)
- `/fintech-aml/alerts` — alerts (fintech_aml:triage)
- `/fintech-aml/monitoring` — monitoring (fintech_aml:monitor)
- `/fintech-aml/cases` — cases (fintech_aml:investigate)
- `/fintech-aml/sar` — sar (fintech_aml:file_sar)
- `/fintech-aml/typologies` — typologies (fintech_aml:admin)
- _2 more..._

**Streaming events** via `bytewax`:
`aml_transaction_monitored`, `aml_alert_created`, `aml_alert_triaged`, `aml_case_opened`, `aml_sar_drafted`, ...

**Standalone usage:**
```bash
pip install apg-fintech-aml
apg-fintech-aml --port 8080
```

---

### Banking APIs `fintech_apis`

> Banking APIs is the Open Banking and API-as-a-product layer for the APG fintech platform. It governs the full lifecycle of API products, developer onboarding, application registration, customer consent grants, API client credential issuance, endpoint policy publishing, webhook subscriptions, call auditing, rate limiting, and SLA incident management. It implements Open Banking-style consent flows where scopes must be explicitly granted before client credentials can be issued.

**Package**: `apg-fintech-apis`  
**Path**: `capabilities/fintech/apis`  
**Version**: 1.1.0  

**Provides:**
- `banking_api_product_governance`
- `developer_onboarding_workflow`
- `developer_application_workflow`
- `banking_consent_workflow`
- `api_client_credential_workflow`
- `api_endpoint_policy_workflow`
- `webhook_subscription_workflow`
- `api_call_audit_workflow`
- `api_rate_limit_workflow`
- `api_sla_incident_workflow`
- `banking_api_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_cards`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_neobanking`
- `fintech_lending`
- `fintech_bnpl`
- `fintech_agency`
- `fintech_mobile`

**Service methods** (40 total):
`describe`, `evaluate`, `register_api_product`, `onboard_developer`, `register_application`, `create_consent_grant`, `issue_api_client`, `publish_endpoint_policy`, `subscribe_webhook`, `record_api_call`, `update_rate_limit`, `open_sla_incident`, ...

**Governance rules** (56 total):
`tenant_context_required`, `apis_write_requires_policy`, `product_owner_required`, `product_type_supported`, `product_environment_supported`, `product_scopes_required`, `developer_kyb_required`, `developer_security_required`, ...

**UI Routes** (13):
- `/fintech-apis/dashboard` — dashboard (fintech_apis:view)
- `/fintech-apis/products` — products (fintech_apis:products)
- `/fintech-apis/developers` — developers (fintech_apis:developers)
- `/fintech-apis/applications` — applications (fintech_apis:applications)
- `/fintech-apis/consents` — consents (fintech_apis:consents)
- `/fintech-apis/clients` — clients (fintech_apis:clients)
- _7 more..._

**Streaming events** via `bytewax`:
`api_product_registered`, `developer_onboarded`, `developer_application_registered`, `consent_grant_created`, `api_client_issued`, ...

**Standalone usage:**
```bash
pip install apg-fintech-apis
apg-fintech-apis --port 8080
```

---

### Blockchain Services `fintech_blockchain`

> Blockchain Services provides governed, multi-network blockchain infrastructure for fintech applications: network registration, wallet and custody management, smart contract deployment, on-chain transaction recording, evidence anchoring, oracle feed management, node health monitoring, and review workflows. It is deliberately provider-neutral — live chain RPC calls, signing keys, custody providers, and oracle connectivity remain adapter boundaries.

**Package**: `apg-fintech-blockchain`  
**Path**: `capabilities/fintech/blockchain`  
**Version**: 1.1.0  

**Provides:**
- `blockchain_network_workflow`
- `blockchain_wallet_workflow`
- `smart_contract_workflow`
- `chain_transaction_workflow`
- `evidence_anchor_workflow`
- `oracle_feed_workflow`
- `node_health_workflow`
- `blockchain_review_workflow`
- `blockchain_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_risk`
- `fintech_compliance`
- `fintech_regtech`
- `fintech_wallets`

**Service methods** (41 total):
`describe`, `evaluate`, `create_private_blockchain`, `deploy_smart_contract`, `invoke_smart_contract`, `record_transaction`, `verify_transaction`, `get_block`, `audit_trail_on_chain`, `verify_anchor`, `supply_chain_tracking`, `get_product_journey`, ...

**Governance rules** (59 total):
`tenant_context_required`, `blockchain_write_requires_policy`, `network_type_supported`, `network_environment_supported`, `network_chain_id_required`, `network_rpc_required`, `network_owner_required`, `network_evidence_required`, ...

**UI Routes** (11):
- `/fintech-blockchain/dashboard` — dashboard (fintech_blockchain:view)
- `/fintech-blockchain/networks` — networks (fintech_blockchain:networks)
- `/fintech-blockchain/wallets` — wallets (fintech_blockchain:wallets)
- `/fintech-blockchain/contracts` — contracts (fintech_blockchain:contracts)
- `/fintech-blockchain/transactions` — transactions (fintech_blockchain:transactions)
- `/fintech-blockchain/anchors` — anchors (fintech_blockchain:anchors)
- _5 more..._

**Streaming events** via `bytewax`:
`blockchain_network_registered`, `blockchain_wallet_registered`, `smart_contract_deployed`, `chain_transaction_recorded`, `evidence_anchor_recorded`, ...

**Standalone usage:**
```bash
pip install apg-fintech-blockchain
apg-fintech-blockchain --port 8080
```

---

### Buy Now Pay Later `fintech_bnpl`

> Buy Now Pay Later manages the lifecycle of deferred payment products for consumers and merchants: BNPL program governance, consumer and merchant onboarding, checkout session capture, affordability decisioning, repayment plan creation, installment scheduling, merchant settlement, and dispute handling. It enforces consumer protection through mandatory KYC, AML, fraud evidence, and explicit fee disclosure at every stage where a consumer commits to debt.

**Package**: `apg-fintech-bnpl`  
**Path**: `capabilities/fintech/bnpl`  
**Version**: 1.1.0  

**Provides:**
- `bnpl_merchant_program_governance`
- `consumer_bnpl_lifecycle`
- `merchant_checkout_workflow`
- `affordability_decisioning`
- `bnpl_plan_workflow`
- `installment_schedule_workflow`
- `merchant_settlement_workflow`
- `bnpl_dispute_workflow`
- `bnpl_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_cards`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_lending`
- `fintech_neobanking`

**Service methods** (40 total):
`describe`, `evaluate`, `register_merchant_program`, `onboard_consumer`, `register_merchant`, `create_checkout_session`, `record_affordability_decision`, `create_bnpl_plan`, `schedule_installment`, `record_merchant_settlement`, `open_bnpl_dispute`, `register_bnpl_agent`, ...

**Governance rules** (73 total):
`tenant_context_required`, `bnpl_write_requires_policy`, `program_owner_required`, `program_country_supported`, `program_currency_supported`, `program_settlement_policy_required`, `program_fee_disclosure_required`, `program_installment_count_valid`, ...

**UI Routes** (12):
- `/fintech-bnpl/dashboard` — dashboard (fintech_bnpl:view)
- `/fintech-bnpl/programs` — programs (fintech_bnpl:manage_programs)
- `/fintech-bnpl/consumers` — consumers (fintech_bnpl:manage_consumers)
- `/fintech-bnpl/merchants` — merchants (fintech_bnpl:manage_merchants)
- `/fintech-bnpl/checkouts` — checkouts (fintech_bnpl:manage_checkouts)
- `/fintech-bnpl/affordability` — affordability (fintech_bnpl:decisioning)
- _6 more..._

**Streaming events** via `bytewax`:
`bnpl_program_registered`, `bnpl_consumer_onboarded`, `bnpl_merchant_registered`, `checkout_session_created`, `affordability_decision_recorded`, ...

**Standalone usage:**
```bash
pip install apg-fintech-bnpl
apg-fintech-bnpl --port 8080
```

---

### Digital Cards `fintech_cards`

> Digital Cards provides executable card issuing and operations workflows: program governance, cardholder onboarding, virtual and physical card issuance, token provisioning (wallet, device, merchant, network tokens), authorization decisions with fraud and AML controls, and dispute intake. It is the issuing layer that sits between a payment wallet and the card network, enforcing per-authorization fraud scoring and AML result checks before any card transaction is approved.

**Package**: `apg-fintech-cards`  
**Path**: `capabilities/fintech/cards`  
**Version**: 1.1.0  

**Provides:**
- `card_program_governance`
- `cardholder_card_lifecycle`
- `tokenized_card_credentialing`
- `card_authorization_control`
- `card_dispute_workflow`
- `card_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `encr`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`

**Service methods** (40 total):
`describe`, `evaluate`, `register_program`, `onboard_cardholder`, `issue_card`, `provision_token`, `authorize_transaction`, `file_dispute`, `register_card_agent`, `validate_batch`, `dashboard_summary`, `list_cards`, ...

**Governance rules** (47 total):
`tenant_context_required`, `card_write_requires_policy`, `program_owner_required`, `program_bin_range_required`, `program_currency_supported`, `program_settlement_required`, `cardholder_customer_required`, `cardholder_kyc_required`, ...

**UI Routes** (9):
- `/fintech-cards/dashboard` — dashboard (fintech_cards:view)
- `/fintech-cards/programs` — programs (fintech_cards:manage_programs)
- `/fintech-cards/cardholders` — cardholders (fintech_cards:manage_cardholders)
- `/fintech-cards/cards` — cards (fintech_cards:issue)
- `/fintech-cards/tokens` — tokens (fintech_cards:tokenize)
- `/fintech-cards/authorizations` — authorizations (fintech_cards:authorize)
- _3 more..._

**Streaming events** via `bytewax`:
`card_program_registered`, `cardholder_onboarded`, `card_issued`, `card_token_provisioned`, `card_authorization_decided`, ...

**Standalone usage:**
```bash
pip install apg-fintech-cards
apg-fintech-cards --port 8080
```

---

### FinTech Compliance Automation `fintech_compliance`

> FinTech Compliance Automation provides a structured framework for managing regulatory obligations, control mappings, compliance checks, evidence collection, attestations, issues, remediation plans, reports, and governance reviews across all supported regulatory frameworks. It acts as the internal compliance layer that links every operational capability to its governing regulatory requirements.

**Package**: `apg-fintech-compliance`  
**Path**: `capabilities/fintech/compliance`  
**Version**: 1.1.0  

**Provides:**
- `compliance_obligation_workflow`
- `compliance_control_workflow`
- `compliance_check_workflow`
- `compliance_evidence_workflow`
- `compliance_attestation_workflow`
- `compliance_issue_workflow`
- `compliance_remediation_workflow`
- `compliance_report_workflow`
- `compliance_review_workflow`
- `compliance_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_risk`
- `fin_rpt`

**Service methods** (40 total):
`describe`, `evaluate`, `register_obligation`, `map_control`, `record_check`, `attach_evidence`, `record_attestation`, `open_issue`, `record_remediation`, `publish_report`, `record_review`, `register_compliance_agent`, ...

**Governance rules** (55 total):
`tenant_context_required`, `compliance_write_requires_policy`, `obligation_framework_supported`, `obligation_type_supported`, `obligation_owner_required`, `obligation_evidence_required`, `obligation_effective_date_required`, `control_obligation_required`, ...

**UI Routes** (11):
- `/fintech-compliance/dashboard` — dashboard (fintech_compliance:view)
- `/fintech-compliance/obligations` — obligations (fintech_compliance:obligations)
- `/fintech-compliance/controls` — controls (fintech_compliance:controls)
- `/fintech-compliance/checks` — checks (fintech_compliance:checks)
- `/fintech-compliance/evidence` — evidence (fintech_compliance:evidence)
- `/fintech-compliance/attestations` — attestations (fintech_compliance:attestations)
- _5 more..._

**Streaming events** via `bytewax`:
`compliance_obligation_registered`, `compliance_control_mapped`, `compliance_check_recorded`, `compliance_evidence_attached`, `compliance_attestation_recorded`, ...

**Standalone usage:**
```bash
pip install apg-fintech-compliance
apg-fintech-compliance --port 8080
```

---

### Crowdfunding Platform `fintech_crowdfunding`

> Crowdfunding Platform manages the lifecycle of alternative finance campaigns: issuer due diligence, campaign publishing across equity, debt, reward, donation, and revenue-share structures, investor disclosure management, commitment recording, escrow funding, milestone tracking, payout authorization, investor updates, compliance alerts, and review workflows. It is designed for regulated crowdfunding operations where every campaign requires disclosure review before investors can commit.

**Package**: `apg-fintech-crowdfunding`  
**Path**: `capabilities/fintech/crowdfunding`  
**Version**: 1.1.0  

**Provides:**
- `crowdfunding_issuer_workflow`
- `crowdfunding_campaign_workflow`
- `crowdfunding_disclosure_workflow`
- `crowdfunding_commitment_workflow`
- `crowdfunding_escrow_workflow`
- `crowdfunding_milestone_workflow`
- `crowdfunding_payout_workflow`
- `crowdfunding_investor_update_workflow`
- `crowdfunding_compliance_workflow`
- `crowdfunding_review_workflow`
- `crowdfunding_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_portfolio`
- `fintech_wealth`
- `bia_anl`
- `fin_rpt`

**Service methods** (42 total):
`describe`, `evaluate`, `onboard_issuer`, `get_issuer`, `launch_campaign`, `campaign_status`, `campaign_analytics`, `contribute`, `refund_failed_campaign`, `disburse_funds`, `equity_share_allocation`, `investor_returns_report`, ...

**Governance rules** (44 total):
`tenant_context_required`, `crowdfunding_write_requires_policy`, `issuer_kyc_required`, `issuer_owner_required`, `issuer_risk_rating_required`, `campaign_issuer_required`, `campaign_type_supported`, `campaign_currency_supported`, ...

**UI Routes** (13):
- `/fintech-crowdfunding/dashboard` — dashboard (fintech_crowdfunding:view)
- `/fintech-crowdfunding/issuers` — issuers (fintech_crowdfunding:issuers)
- `/fintech-crowdfunding/campaigns` — campaigns (fintech_crowdfunding:campaigns)
- `/fintech-crowdfunding/disclosures` — disclosures (fintech_crowdfunding:disclosures)
- `/fintech-crowdfunding/commitments` — commitments (fintech_crowdfunding:commitments)
- `/fintech-crowdfunding/escrow` — escrow (fintech_crowdfunding:escrow)
- _7 more..._

**Streaming events** via `bytewax`:
`issuer_onboarded`, `campaign_published`, `disclosure_recorded`, `investor_commitment_recorded`, `escrow_funding_recorded`, ...

**Standalone usage:**
```bash
pip install apg-fintech-crowdfunding
apg-fintech-crowdfunding --port 8080
```

---

### Cryptocurrency Services `fintech_crypto`

> Cryptocurrency Services provides governed digital asset operations: asset registry, custody account management, balance snapshots, order management, trade execution recording, transfer requests with approval gates, compliance screening (wallet, transaction, sanctions, travel rule), market price snapshots, and governance reviews. It is the regulated operational layer over blockchain infrastructure, providing the audit trail and compliance controls that raw chain operations lack.

**Package**: `apg-fintech-crypto`  
**Path**: `capabilities/fintech/crypto`  
**Version**: 1.1.0  

**Provides:**
- `crypto_asset_workflow`
- `crypto_custody_workflow`
- `crypto_balance_workflow`
- `crypto_order_workflow`
- `crypto_trade_workflow`
- `crypto_transfer_workflow`
- `crypto_screening_workflow`
- `crypto_price_workflow`
- `crypto_review_workflow`
- `crypto_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_blockchain`
- `fintech_wallets`
- `fintech_risk`
- `fintech_compliance`
- `fintech_regtech`
- `fintech_aml`
- `fintech_kyc`

**Service methods** (40 total):
`describe`, `evaluate`, `create_crypto_wallet`, `get_wallet_balance`, `buy_crypto`, `sell_crypto`, `crypto_to_crypto_swap`, `send_crypto`, `receive_crypto`, `crypto_price_feed`, `transaction_history`, `tax_report`, ...

**Governance rules** (69 total):
`tenant_context_required`, `crypto_write_requires_policy`, `asset_symbol_required`, `asset_type_supported`, `asset_network_required`, `asset_precision_valid`, `asset_owner_required`, `asset_evidence_required`, ...

**UI Routes** (12):
- `/fintech-crypto/dashboard` — dashboard (fintech_crypto:view)
- `/fintech-crypto/assets` — assets (fintech_crypto:assets)
- `/fintech-crypto/custody` — custody (fintech_crypto:custody)
- `/fintech-crypto/balances` — balances (fintech_crypto:balances)
- `/fintech-crypto/orders` — orders (fintech_crypto:orders)
- `/fintech-crypto/trades` — trades (fintech_crypto:trades)
- _6 more..._

**Streaming events** via `bytewax`:
`crypto_asset_registered`, `crypto_custody_account_opened`, `crypto_balance_recorded`, `crypto_order_created`, `crypto_trade_recorded`, ...

**Standalone usage:**
```bash
pip install apg-fintech-crypto
apg-fintech-crypto --port 8080
```

---

### Decentralized Finance `fintech_defi`

> Decentralized Finance provides governed operations over DeFi protocols: protocol registry, position management (supply, borrow, liquidity, stake, vault share), action execution workflow (deposit, withdraw, borrow, repay, swap, stake, unstake, claim, rebalance), yield strategy management, reward accruals, governance voting, risk tier assessments, and reviews. Every action against a DeFi protocol requires an approval reference before it is recorded, enforcing human oversight over autonomous on-chain interactions.

**Package**: `apg-fintech-defi`  
**Path**: `capabilities/fintech/defi`  
**Version**: 1.1.0  

**Provides:**
- `defi_protocol_workflow`
- `defi_position_workflow`
- `defi_action_workflow`
- `defi_yield_strategy_workflow`
- `defi_reward_workflow`
- `defi_governance_workflow`
- `defi_risk_workflow`
- `defi_review_workflow`
- `defi_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_blockchain`
- `fintech_crypto`
- `fintech_wallets`
- `fintech_risk`
- `fintech_compliance`
- `fintech_regtech`
- `fintech_aml`
- `fintech_kyc`

**Service methods** (40 total):
`describe`, `evaluate`, `liquidity_pool_deposit`, `liquidity_pool_withdraw`, `yield_farming_enrol`, `claim_farming_rewards`, `lending_deposit`, `borrow_against_collateral`, `repay_loan`, `collateral_health_factor`, `liquidation_risk_alert`, `amm_swap`, ...

**Governance rules** (60 total):
`tenant_context_required`, `defi_write_requires_policy`, `protocol_type_supported`, `protocol_network_required`, `protocol_reference_required`, `protocol_owner_required`, `protocol_evidence_required`, `protocol_risk_supported`, ...

**UI Routes** (11):
- `/fintech-defi/dashboard` — dashboard (fintech_defi:view)
- `/fintech-defi/protocols` — protocols (fintech_defi:protocols)
- `/fintech-defi/positions` — positions (fintech_defi:positions)
- `/fintech-defi/actions` — actions (fintech_defi:actions)
- `/fintech-defi/yield-strategies` — yield_strategies (fintech_defi:yield)
- `/fintech-defi/rewards` — rewards (fintech_defi:rewards)
- _5 more..._

**Streaming events** via `bytewax`:
`defi_protocol_registered`, `defi_position_opened`, `defi_action_recorded`, `defi_yield_strategy_registered`, `defi_reward_recorded`, ...

**Standalone usage:**
```bash
pip install apg-fintech-defi
apg-fintech-defi --port 8080
```

---

### Embedded Finance `fintech_embedded`

> Embedded Finance enables non-financial businesses to offer financial products inside their own applications without owning banking infrastructure. It manages partner program onboarding, host application registration, product placement publishing, customer consent capture, and the end-to-end lifecycle of embedded accounts, payments, card offers, lending offers, settlement batches, and revenue share — all within a consent-scoped access model.

**Package**: `apg-fintech-embedded`  
**Path**: `capabilities/fintech/embedded`  
**Version**: 1.1.0  

**Provides:**
- `partner_program_workflow`
- `host_application_workflow`
- `embedded_product_placement_workflow`
- `embedded_customer_consent_workflow`
- `embedded_account_workflow`
- `embedded_payment_workflow`
- `embedded_card_workflow`
- `embedded_lending_workflow`
- `embedded_settlement_workflow`
- `embedded_revenue_share_workflow`
- `embedded_finance_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_apis`
- `fintech_payments`
- `fintech_wallets`
- `fintech_cards`
- `fintech_lending`
- `fintech_bnpl`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_mobile`

**Service methods** (41 total):
`describe`, `evaluate`, `register_partner_program`, `register_host_application`, `publish_product_placement`, `capture_customer_consent`, `open_embedded_account`, `initiate_embedded_payment`, `offer_embedded_card`, `create_lending_offer`, `close_settlement_batch`, `record_revenue_share`, ...

**Governance rules** (50 total):
`tenant_context_required`, `embedded_write_requires_policy`, `program_kyb_required`, `program_contract_required`, `program_risk_required`, `application_program_required`, `application_environment_supported`, `application_domain_required`, ...

**UI Routes** (13):
- `/fintech-embedded/dashboard` — dashboard (fintech_embedded:view)
- `/fintech-embedded/programs` — programs (fintech_embedded:programs)
- `/fintech-embedded/applications` — applications (fintech_embedded:applications)
- `/fintech-embedded/placements` — placements (fintech_embedded:placements)
- `/fintech-embedded/consents` — consents (fintech_embedded:consents)
- `/fintech-embedded/accounts` — accounts (fintech_embedded:accounts)
- _7 more..._

**Streaming events** via `bytewax`:
`partner_program_registered`, `host_application_registered`, `product_placement_published`, `customer_consent_captured`, `embedded_account_opened`, ...

**Standalone usage:**
```bash
pip install apg-fintech-embedded
apg-fintech-embedded --port 8080
```

---

### Fraud Detection `fintech_fraud`

> Fraud Detection provides real-time transaction risk scoring, multi-factor decision making (approve, step-up, hold, block, review), account takeover detection, device risk assessment, chargeback evidence management, and fraud case investigation. It acts as the cross-cutting fraud control layer across all payment-generating capabilities — every financial operation that carries a monetary amount requires a fraud signal before authorization can proceed.

**Package**: `apg-fintech-fraud`  
**Path**: `capabilities/fintech/fraud`  
**Version**: 1.1.0  

**Provides:**
- `fraud_signal_scoring`
- `transaction_risk_decisioning`
- `account_takeover_detection`
- `device_risk_detection`
- `chargeback_evidence_workflow`
- `fraud_case_management`
- `fraud_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`

**Service methods** (40 total):
`describe`, `evaluate`, `score_signal`, `record_decision`, `open_case`, `resolve_case`, `register_fraud_agent`, `validate_batch`, `dashboard_summary`, `list_signals`, `list_cases`, `detect_transaction_fraud`, ...

**Governance rules** (42 total):
`tenant_context_required`, `fraud_write_requires_policy`, `signal_subject_required`, `signal_type_supported`, `signal_channel_supported`, `signal_source_required`, `signal_requires_kyc_link`, `money_amount_positive`, ...

**UI Routes** (8):
- `/fintech-fraud/dashboard` — dashboard (fintech_fraud:view)
- `/fintech-fraud/signals` — signals (fintech_fraud:score)
- `/fintech-fraud/decisions` — decisions (fintech_fraud:decide)
- `/fintech-fraud/cases` — cases (fintech_fraud:investigate)
- `/fintech-fraud/chargebacks` — chargebacks (fintech_fraud:chargebacks)
- `/fintech-fraud/devices` — devices (fintech_fraud:devices)
- _2 more..._

**Streaming events** via `bytewax`:
`fraud_signal_scored`, `fraud_decision_recorded`, `fraud_case_opened`, `fraud_case_resolved`, `fraud_agent_registered`

**Standalone usage:**
```bash
pip install apg-fintech-fraud
apg-fintech-fraud --port 8080
```

---

### Fintech Gateway `fintech_gateway`

> Fintech Gateway is the payment orchestration capability responsible for merchant onboarding, payment provider connections, payment method tokenization, payment intent lifecycle, routing decisions, fraud risk review, authorization and capture, refunds, webhook ingestion, settlement reconciliation, and dispute management. It is the operational hub that connects the APG payment layer to external payment processors (Stripe, Adyen, MPESA, Flutterwave, Pesapal, DPO, PayPal, and others) while enforcing routing, risk, and governance rules on every payment.

**Package**: `apg-fintech-gateway`  
**Path**: `capabilities/fintech/gateway`  
**Version**: 2.1.0  

**Provides:**
- `merchant_onboarding_lifecycle`
- `provider_connection_lifecycle`
- `payment_method_tokenization_workflow`
- `payment_intent_lifecycle`
- `payment_routing_workflow`
- `fraud_risk_review_workflow`
- `authorization_capture_workflow`
- `refund_lifecycle`
- `webhook_ingestion_workflow`
- `settlement_reconciliation_workflow`
- `payment_dispute_workflow`
- `gateway_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `keym`
- `encr`
- `cbm_cash_management`
- `arc_accounts_receivable`
- `crm_adv`
- `bia_anl`

**Service methods** (40 total):
`describe`, `onboard_merchant`, `connect_provider`, `tokenize_payment_method`, `create_payment_intent`, `assess_payment_risk`, `authorize_payment`, `capture_payment`, `refund_payment`, `ingest_webhook`, `record_settlement`, `open_dispute`, ...

**Governance rules** (57 total):
`tenant_context_required`, `gateway_write_requires_policy`, `merchant_requires_code`, `merchant_requires_legal_name`, `merchant_requires_country`, `high_risk_merchant_requires_review`, `provider_name_supported`, `provider_type_supported`, ...

**UI Routes** (12):
- `/fintech-gateway/dashboard` — dashboard (fintech_gateway:view)
- `/fintech-gateway/merchants` — merchants (fintech_gateway:manage_merchants)
- `/fintech-gateway/providers` — providers (fintech_gateway:manage_providers)
- `/fintech-gateway/payment-methods` — payment_methods (fintech_gateway:manage_payment_methods)
- `/fintech-gateway/payments` — payments (fintech_gateway:process)
- `/fintech-gateway/routing` — routing (fintech_gateway:route)
- _6 more..._

**Streaming events** via `bytewax`:
`merchant_onboarded`, `provider_connected`, `payment_method_tokenized`, `payment_intent_created`, `payment_risk_assessed`, ...

**Standalone usage:**
```bash
pip install apg-fintech-gateway
apg-fintech-gateway --port 8080
```

---

### InsurTech `fintech_insurance`

> InsurTech manages the end-to-end lifecycle of insurance operations: policyholder onboarding, product publishing across life, health, property, motor, travel, crop, and microinsurance lines, quote generation with underwriting evidence, policy binding, premium recording, claim intake, document management, risk assessment, reinsurance attachment, compliance alerts, and governance reviews. It is designed for regulated insurance operations where every quote must have an underwriting reference and every claim must have supporting evidence.

**Package**: `apg-fintech-insurance`  
**Path**: `capabilities/fintech/insurance`  
**Version**: 1.1.0  

**Provides:**
- `insurance_policyholder_workflow`
- `insurance_product_workflow`
- `insurance_quote_workflow`
- `insurance_policy_workflow`
- `insurance_premium_workflow`
- `insurance_claim_workflow`
- `insurance_document_workflow`
- `insurance_risk_workflow`
- `insurance_reinsurance_workflow`
- `insurance_compliance_workflow`
- `insurance_review_workflow`
- `insurance_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `bia_anl`
- `fin_rpt`

**Service methods** (42 total):
`describe`, `evaluate`, `onboard_policyholder`, `get_policyholder`, `list_policyholders`, `publish_product`, `generate_quote`, `create_policy`, `bind_policy`, `underwrite_policy`, `process_premium`, `record_premium`, ...

**Governance rules** (44 total):
`tenant_context_required`, `insurance_write_requires_policy`, `policyholder_kyc_required`, `policyholder_contact_required`, `product_line_supported`, `product_coverage_required`, `quote_policyholder_required`, `quote_product_required`, ...

**UI Routes** (14):
- `/fintech-insurance/dashboard` — dashboard (fintech_insurance:view)
- `/fintech-insurance/policyholders` — policyholders (fintech_insurance:policyholders)
- `/fintech-insurance/products` — products (fintech_insurance:products)
- `/fintech-insurance/quotes` — quotes (fintech_insurance:quotes)
- `/fintech-insurance/policies` — policies (fintech_insurance:policies)
- `/fintech-insurance/premiums` — premiums (fintech_insurance:premiums)
- _8 more..._

**Streaming events** via `bytewax`:
`policyholder_onboarded`, `insurance_product_published`, `quote_generated`, `policy_bound`, `premium_recorded`, ...

**Standalone usage:**
```bash
pip install apg-fintech-insurance
apg-fintech-insurance --port 8080
```

---

### Know Your Customer `fintech_kyc`

> Know Your Customer provides the customer identity foundation for the entire APG fintech platform: tenant-scoped identity profiles, consent-backed onboarding, document verification with minimum confidence thresholds, sanctions/PEP/adverse-media/watchlist screening, KYC risk scoring, customer due diligence, enhanced due diligence for high-risk profiles, and AI-assisted review workflows. It is a hard dependency for every capability that onboards customers.

**Package**: `apg-fintech-kyc`  
**Path**: `capabilities/fintech/kyc`  
**Version**: 1.1.0  

**Provides:**
- `customer_identity_lifecycle`
- `document_verification_workflow`
- `sanctions_pep_screening`
- `kyc_risk_scoring`
- `customer_due_diligence`
- `enhanced_due_diligence`
- `kyc_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `cons`
- `ntfy`
- `biop`
- `cvsn`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`

**Service methods** (72 total):
`_emit`, `_get_app`, `_require_app`, `_get_document`, `_open_reviews_count`, `_has_doc_type`, `_screening_done`, `_risk_assessed`, `start_kyc_application`, `update_application`, `get_application`, `list_applications`, ...

**Governance rules** (38 total):
`tenant_context_required`, `kyc_write_requires_policy`, `profile_subject_required`, `profile_legal_name_required`, `profile_customer_type_supported`, `profile_country_required`, `profile_consent_required`, `document_profile_required`, ...

**UI Routes** (8):
- `/fintech-kyc/dashboard` — dashboard (fintech_kyc:view)
- `/fintech-kyc/profiles` — profiles (fintech_kyc:manage_profiles)
- `/fintech-kyc/documents` — documents (fintech_kyc:manage_documents)
- `/fintech-kyc/screening` — screening (fintech_kyc:screen)
- `/fintech-kyc/risk` — risk (fintech_kyc:review_risk)
- `/fintech-kyc/reviews` — reviews (fintech_kyc:review)
- _2 more..._

**Streaming events** via `bytewax`:
`kyc_profile_opened`, `kyc_document_registered`, `kyc_screening_recorded`, `kyc_risk_scored`, `kyc_decision_recorded`, ...

**Standalone usage:**
```bash
pip install apg-fintech-kyc
apg-fintech-kyc --port 8080
```

---

### Digital Lending `fintech_lending`

> Digital Lending manages the complete credit lifecycle: loan product governance, borrower onboarding, credit application submission with affordability and bank statement evidence, underwriting decisioning with adverse-action tracking, loan offer issuance and acceptance, disbursement control with mandatory human approval, repayment scheduling, and collections case management. It enforces consumer protection at every stage — declines require adverse-action reasons, accepted offers require borrower acceptance evidence, and every disbursement requires human approval regardless of amount.

**Package**: `apg-fintech-lending`  
**Path**: `capabilities/fintech/lending`  
**Version**: 1.1.0  

**Provides:**
- `loan_product_governance`
- `borrower_lifecycle`
- `credit_application_workflow`
- `underwriting_decisioning`
- `loan_offer_workflow`
- `disbursement_control`
- `repayment_schedule_workflow`
- `collections_workflow`
- `lending_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_cards`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_remittance`

**Service methods** (65 total):
`to_dict`, `forced_sale_value`, `to_dict`, `to_dict`, `to_dict`, `describe`, `evaluate`, `register_product`, `onboard_borrower`, `submit_application`, `record_underwriting`, `issue_offer`, ...

**Governance rules** (63 total):
`tenant_context_required`, `lending_write_requires_policy`, `loan_product_owner_required`, `loan_product_currency_supported`, `loan_product_type_supported`, `loan_product_term_valid`, `loan_product_rate_valid`, `loan_product_amount_limits_valid`, ...

**UI Routes** (11):
- `/fintech-lending/dashboard` — dashboard (fintech_lending:view)
- `/fintech-lending/products` — products (fintech_lending:manage_products)
- `/fintech-lending/borrowers` — borrowers (fintech_lending:manage_borrowers)
- `/fintech-lending/applications` — applications (fintech_lending:submit)
- `/fintech-lending/underwriting` — underwriting (fintech_lending:underwrite)
- `/fintech-lending/offers` — offers (fintech_lending:offer)
- _5 more..._

**Streaming events** via `bytewax`:
`loan_product_registered`, `borrower_onboarded`, `loan_application_submitted`, `underwriting_recorded`, `loan_offer_issued`, ...

**Standalone usage:**
```bash
pip install apg-fintech-lending
apg-fintech-lending --port 8080
```

---

### Mobile Banking `fintech_mobile`

> Mobile Banking provides the customer-facing mobile channel layer: banking program governance, customer enrollment, trusted device binding with attestation, authentication factor registration (passcode, biometric, OTP, device binding, hardware key), account and wallet linking, mobile payment initiation, bill payment, airtime purchase, service request intake, notification preference management, and mobile fraud event recording. It is the channel capability that surfaces neobanking, payments, cards, lending, BNPL, and agency services through iOS, Android, web, USSD, and SMS interfaces.

**Package**: `apg-fintech-mobile`  
**Path**: `capabilities/fintech/mobile`  
**Version**: 1.1.0  

**Provides:**
- `mobile_banking_program_governance`
- `mobile_customer_enrollment`
- `trusted_device_lifecycle`
- `mobile_authentication_factor_workflow`
- `mobile_account_linking`
- `mobile_payment_workflow`
- `mobile_bill_payment_workflow`
- `mobile_airtime_workflow`
- `mobile_service_request_workflow`
- `mobile_notification_workflow`
- `mobile_fraud_event_workflow`
- `mobile_banking_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_cards`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_neobanking`
- `fintech_lending`
- `fintech_bnpl`
- `fintech_agency`

**Service methods** (40 total):
`describe`, `evaluate`, `register_program`, `enroll_customer`, `bind_device`, `register_auth_factor`, `link_account`, `initiate_payment`, `record_bill_payment`, `purchase_airtime`, `open_service_request`, `set_notification_preference`, ...

**Governance rules** (66 total):
`tenant_context_required`, `mobile_write_requires_policy`, `program_owner_required`, `program_country_supported`, `program_currency_supported`, `program_platforms_valid`, `customer_reference_required`, `customer_country_supported`, ...

**UI Routes** (14):
- `/fintech-mobile/dashboard` — dashboard (fintech_mobile:view)
- `/fintech-mobile/programs` — programs (fintech_mobile:manage_programs)
- `/fintech-mobile/customers` — customers (fintech_mobile:customers)
- `/fintech-mobile/devices` — devices (fintech_mobile:devices)
- `/fintech-mobile/auth-factors` — auth_factors (fintech_mobile:auth)
- `/fintech-mobile/account-links` — account_links (fintech_mobile:accounts)
- _8 more..._

**Streaming events** via `bytewax`:
`mobile_program_registered`, `mobile_customer_enrolled`, `trusted_device_bound`, `auth_factor_registered`, `account_linked`, ...

**Standalone usage:**
```bash
pip install apg-fintech-mobile
apg-fintech-mobile --port 8080
```

---

### Digital Neobanking `fintech_neobanking`

> Digital Neobanking provides the core banking layer for digital-first banks: program governance, customer onboarding with full AML/KYC/fraud evidence chain, deposit account opening (current, savings, joint, business, youth, merchant), payment rail linking, transaction posting with risk reference, savings pot management, account statement generation, and customer service case handling. It is the account ledger that other capabilities — mobile, cards, lending, remittance — use as their underlying account infrastructure.

**Package**: `apg-fintech-neobanking`  
**Path**: `capabilities/fintech/neobanking`  
**Version**: 1.1.0  

**Provides:**
- `neobank_program_governance`
- `digital_customer_onboarding`
- `deposit_account_lifecycle`
- `payment_rail_linking`
- `account_transaction_posting`
- `savings_pot_workflow`
- `statement_workflow`
- `customer_service_case_workflow`
- `neobanking_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_cards`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_lending`
- `fintech_remittance`

**Service methods** (42 total):
`describe`, `evaluate`, `open_account`, `close_account`, `account_features_bundle`, `virtual_card_issue`, `virtual_card_freeze`, `peer_transfer`, `split_bill`, `savings_pot_create`, `savings_pot_deposit`, `savings_round_up`, ...

**Governance rules** (48 total):
`tenant_context_required`, `neobanking_write_requires_policy`, `program_owner_required`, `program_country_supported`, `program_currency_supported`, `program_settlement_required`, `customer_reference_required`, `customer_kyc_required`, ...

**UI Routes** (11):
- `/fintech-neobanking/dashboard` — dashboard (fintech_neobanking:view)
- `/fintech-neobanking/programs` — programs (fintech_neobanking:manage_programs)
- `/fintech-neobanking/customers` — customers (fintech_neobanking:manage_customers)
- `/fintech-neobanking/accounts` — accounts (fintech_neobanking:manage_accounts)
- `/fintech-neobanking/rails` — rails (fintech_neobanking:manage_rails)
- `/fintech-neobanking/transactions` — transactions (fintech_neobanking:post_transactions)
- _5 more..._

**Streaming events** via `bytewax`:
`bank_program_registered`, `digital_customer_onboarded`, `deposit_account_opened`, `payment_rail_linked`, `account_transaction_posted`, ...

**Standalone usage:**
```bash
pip install apg-fintech-neobanking
apg-fintech-neobanking --port 8080
```

---

### Digital Payments `fintech_payments`

> Digital Payments is the application-facing payment lifecycle capability: account creation, payment instrument registration with vault token references, payment order creation, risk screening, authorization with provider routing, capture, refunds, payouts, settlement reconciliation, and dispute management. It sits between application capabilities (neobanking, mobile, lending) and the gateway layer, owning the structured payment state machine while delegating actual provider communication to `fintech_gateway`.

**Package**: `apg-fintech-payments`  
**Path**: `capabilities/fintech/payments`  
**Version**: 1.1.0  

**Provides:**
- `payment_account_lifecycle`
- `payment_instrument_vault`
- `payment_order_lifecycle`
- `risk_screening_workflow`
- `authorization_capture_refund_workflow`
- `payout_workflow`
- `settlement_reconciliation_workflow`
- `payment_dispute_workflow`
- `payment_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `keym`
- `encr`
- `fintech_gateway`
- `cbm_cash_management`
- `arc_accounts_receivable`

**Service methods** (89 total):
`_save`, `_get`, `_query`, `_emit`, `initiate_payment`, `mpesa_stk_push`, `mpesa_b2c`, `mpesa_b2b`, `mtn_momo_request_to_pay`, `airtel_money_push`, `tigo_pesa_collect`, `bank_eft_transfer`, ...

**Governance rules** (40 total):
`tenant_context_required`, `payment_write_requires_policy`, `account_owner_required`, `account_currency_supported`, `instrument_account_required`, `instrument_type_supported`, `instrument_token_required`, `payment_amount_positive`, ...

**UI Routes** (9):
- `/fintech-payments/dashboard` — dashboard (fintech_payments:view)
- `/fintech-payments/accounts` — accounts (fintech_payments:manage_accounts)
- `/fintech-payments/instruments` — instruments (fintech_payments:manage_instruments)
- `/fintech-payments/orders` — orders (fintech_payments:operate)
- `/fintech-payments/risk` — risk (fintech_payments:risk)
- `/fintech-payments/settlement` — settlement (fintech_payments:settle)
- _3 more..._

**Streaming events** via `bytewax`:
`payment_account_opened`, `payment_instrument_registered`, `payment_order_created`, `payment_risk_screened`, `payment_authorized`, ...

**Standalone usage:**
```bash
pip install apg-fintech-payments
apg-fintech-payments --port 8080
```

---

### Portfolio Management `fintech_portfolio`

> Portfolio Management provides regulated investment book operations: portfolio book creation, holding ledger recording, allocation policy activation (totals must equal exactly 100%), valuation capture, benchmark assignment, risk exposure tracking, performance attribution, cash movement recording, corporate action processing, compliance breach recording, and governance reviews. It is the investment operations layer for discretionary, advisory, model, and execution-only portfolios.

**Package**: `apg-fintech-portfolio`  
**Path**: `capabilities/fintech/portfolio`  
**Version**: 1.1.0  

**Provides:**
- `portfolio_book_workflow`
- `portfolio_holding_workflow`
- `portfolio_allocation_policy_workflow`
- `portfolio_valuation_workflow`
- `portfolio_benchmark_workflow`
- `portfolio_risk_workflow`
- `portfolio_attribution_workflow`
- `portfolio_cash_workflow`
- `portfolio_corporate_action_workflow`
- `portfolio_compliance_workflow`
- `portfolio_review_workflow`
- `portfolio_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_wealth`
- `fintech_robo`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `bia_anl`
- `fin_rpt`

**Service methods** (42 total):
`describe`, `evaluate`, `create_portfolio`, `get_portfolio`, `list_portfolios`, `close_portfolio`, `add_holding`, `remove_holding`, `get_holding`, `list_holdings`, `portfolio_valuation`, `activate_allocation_policy`, ...

**Governance rules** (46 total):
`tenant_context_required`, `portfolio_write_requires_policy`, `portfolio_owner_required`, `portfolio_type_supported`, `portfolio_currency_supported`, `holding_portfolio_required`, `holding_instrument_required`, `holding_positive_quantity`, ...

**UI Routes** (14):
- `/fintech-portfolio/dashboard` — dashboard (fintech_portfolio:view)
- `/fintech-portfolio/portfolios` — portfolios (fintech_portfolio:portfolios)
- `/fintech-portfolio/holdings` — holdings (fintech_portfolio:holdings)
- `/fintech-portfolio/allocations` — allocations (fintech_portfolio:allocations)
- `/fintech-portfolio/valuations` — valuations (fintech_portfolio:valuations)
- `/fintech-portfolio/benchmarks` — benchmarks (fintech_portfolio:benchmarks)
- _8 more..._

**Streaming events** via `bytewax`:
`portfolio_book_created`, `portfolio_holding_recorded`, `allocation_policy_activated`, `portfolio_valuation_recorded`, `benchmark_assigned`, ...

**Standalone usage:**
```bash
pip install apg-fintech-portfolio
apg-fintech-portfolio --port 8080
```

---

### Regulatory Technology `fintech_regtech`

> Regulatory Technology provides automated tracking and management of regulatory obligations: regulatory source registration, change intake (new rules, updates, guidance, enforcement actions, consultations), obligation mapping with policy references, impact assessment across APG capabilities, regulatory filing preparation and submission, regulatory inquiry management, and approved response recording. It is the regulatory horizon scanning and filing layer that feeds obligation evidence into `fintech_compliance`.

**Package**: `apg-fintech-regtech`  
**Path**: `capabilities/fintech/regtech`  
**Version**: 1.1.0  

**Provides:**
- `regulatory_source_workflow`
- `regulatory_change_workflow`
- `regulatory_obligation_mapping_workflow`
- `regulatory_policy_mapping_workflow`
- `regulatory_impact_workflow`
- `regulatory_filing_workflow`
- `regulatory_submission_workflow`
- `regulatory_inquiry_workflow`
- `regulatory_response_workflow`
- `regulatory_review_workflow`
- `regulatory_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_compliance`
- `fintech_risk`
- `fintech_aml`
- `fintech_kyc`
- `fin_rpt`

**Service methods** (42 total):
`describe`, `evaluate`, `regulatory_calendar`, `compliance_obligation_check`, `auto_report_generation`, `regulatory_change_monitoring`, `compliance_gap_analysis`, `prepare_filing`, `regulatory_filing`, `record_submission`, `cbk_returns`, `prudential_ratios`, ...

**Governance rules** (57 total):
`tenant_context_required`, `regtech_write_requires_policy`, `source_regulator_supported`, `source_jurisdiction_supported`, `source_reference_required`, `source_owner_required`, `source_evidence_required`, `change_source_required`, ...

**UI Routes** (12):
- `/fintech-regtech/dashboard` — dashboard (fintech_regtech:view)
- `/fintech-regtech/sources` — sources (fintech_regtech:sources)
- `/fintech-regtech/changes` — changes (fintech_regtech:changes)
- `/fintech-regtech/obligations` — obligations (fintech_regtech:obligations)
- `/fintech-regtech/impact` — impact (fintech_regtech:impact)
- `/fintech-regtech/filings` — filings (fintech_regtech:filings)
- _6 more..._

**Streaming events** via `bytewax`:
`regulatory_source_registered`, `regulatory_change_recorded`, `regulatory_obligation_mapped`, `regulatory_impact_assessed`, `regulatory_filing_prepared`, ...

**Standalone usage:**
```bash
pip install apg-fintech-regtech
apg-fintech-regtech --port 8080
```

---

### Cross-Border Remittance `fintech_remittance`

> Cross-Border Remittance manages the lifecycle of international money transfers: corridor and currency eligibility checks, FX quote creation with rate and fee locking, transfer creation with dual-side KYC and source-of-funds evidence, AML screening with sanctions blocking, fraud decisioning, payout release with provider receipt, and refund handling. Same-country transfers are architecturally blocked — the capability is strictly cross-border.

**Package**: `apg-fintech-remittance`  
**Path**: `capabilities/fintech/remittance`  
**Version**: 1.1.0  

**Provides:**
- `remittance_corridor_governance`
- `remittance_quote_lifecycle`
- `cross_border_transfer_workflow`
- `remittance_payout_workflow`
- `remittance_refund_workflow`
- `remittance_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`

**Service methods** (40 total):
`describe`, `evaluate`, `get_fx_quote`, `initiate_remittance`, `compliance_check`, `partner_routing`, `track_remittance`, `recipient_notification`, `payout_methods`, `deliver_to_mobile_money`, `bank_payout`, `cash_pickup`, ...

**Governance rules** (47 total):
`tenant_context_required`, `remittance_write_requires_policy`, `corridor_supported`, `same_country_blocked`, `source_currency_supported`, `destination_currency_supported`, `send_amount_positive`, `fx_rate_positive`, ...

**UI Routes** (8):
- `/fintech-remittance/dashboard` — dashboard (fintech_remittance:view)
- `/fintech-remittance/corridors` — corridors (fintech_remittance:govern_corridors)
- `/fintech-remittance/quotes` — quotes (fintech_remittance:quote)
- `/fintech-remittance/transfers` — transfers (fintech_remittance:transfer)
- `/fintech-remittance/payouts` — payouts (fintech_remittance:payout)
- `/fintech-remittance/refunds` — refunds (fintech_remittance:refund)
- _2 more..._

**Streaming events** via `bytewax`:
`remittance_quote_created`, `remittance_transfer_created`, `remittance_payout_released`, `remittance_refund_filed`, `remittance_agent_registered`

**Standalone usage:**
```bash
pip install apg-fintech-remittance
apg-fintech-remittance --port 8080
```

---

### FinTech Risk Management `fintech_risk`

> FinTech Risk Management provides the enterprise risk framework for the APG platform: risk appetite registration across credit, market, liquidity, operational, fraud, compliance, model, and third-party domains; tenant-scoped risk profiles for customers, merchants, wallets, accounts, portfolios, loans, agents, and counterparties; exposure tracking with limit enforcement and human-approval-gated overrides; control assurance with effectiveness scoring; stress scenario modeling; limit breach recording; risk event management; and governance reviews.

**Package**: `apg-fintech-risk`  
**Path**: `capabilities/fintech/risk`  
**Version**: 1.1.0  

**Provides:**
- `risk_appetite_workflow`
- `risk_profile_workflow`
- `risk_exposure_workflow`
- `risk_control_workflow`
- `risk_stress_testing_workflow`
- `risk_limit_breach_workflow`
- `risk_event_workflow`
- `risk_review_workflow`
- `risk_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `bia_anl`
- `fin_rpt`

**Service methods** (40 total):
`describe`, `evaluate`, `register_appetite`, `create_profile`, `record_exposure`, `evaluate_control`, `run_stress_scenario`, `record_limit_breach`, `open_risk_event`, `record_review`, `register_risk_agent`, `validate_agent_action`, ...

**Governance rules** (52 total):
`tenant_context_required`, `risk_write_requires_policy`, `appetite_domain_supported`, `appetite_threshold_required`, `appetite_owner_required`, `appetite_evidence_required`, `profile_subject_required`, `profile_subject_type_supported`, ...

**UI Routes** (11):
- `/fintech-risk/dashboard` — dashboard (fintech_risk:view)
- `/fintech-risk/appetite` — appetite (fintech_risk:appetite)
- `/fintech-risk/profiles` — profiles (fintech_risk:profiles)
- `/fintech-risk/exposures` — exposures (fintech_risk:exposures)
- `/fintech-risk/controls` — controls (fintech_risk:controls)
- `/fintech-risk/stress-tests` — stress_tests (fintech_risk:stress)
- _5 more..._

**Streaming events** via `bytewax`:
`risk_appetite_registered`, `risk_profile_created`, `risk_exposure_recorded`, `risk_control_evaluated`, `risk_stress_scenario_recorded`, ...

**Standalone usage:**
```bash
pip install apg-fintech-risk
apg-fintech-risk --port 8080
```

---

### Robo Advisory `fintech_robo`

> Robo Advisory provides algorithm-guided investment advice under governance: investor profile creation with KYC and suitability evidence, goal planning, model portfolio publication with exact 100% allocation totals, recommendation generation and approval workflows, automated investment plan configuration, portfolio drift monitoring, tax-loss harvesting candidate recording, and governance reviews. It builds on Wealth Management by making model-driven recommendations, automated rebalancing, and tax optimization first-class governed operations.

**Package**: `apg-fintech-robo`  
**Path**: `capabilities/fintech/robo`  
**Version**: 1.1.0  

**Provides:**
- `robo_investor_profile_workflow`
- `robo_goal_plan_workflow`
- `robo_model_portfolio_workflow`
- `robo_recommendation_workflow`
- `robo_automation_workflow`
- `robo_drift_workflow`
- `robo_tax_loss_workflow`
- `robo_review_workflow`
- `robo_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_wealth`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `bia_anl`
- `fin_rpt`

**Service methods** (40 total):
`describe`, `evaluate`, `risk_questionnaire`, `determine_risk_profile`, `recommended_portfolio`, `auto_invest`, `auto_rebalance`, `goal_tracking`, `onboard_client`, `drift_monitoring`, `tax_optimisation`, `robo_performance_report`, ...

**Governance rules** (42 total):
`tenant_context_required`, `robo_write_requires_policy`, `profile_client_required`, `profile_kyc_required`, `profile_suitability_required`, `profile_risk_supported`, `goal_profile_required`, `goal_type_supported`, ...

**UI Routes** (11):
- `/fintech-robo/dashboard` — dashboard (fintech_robo:view)
- `/fintech-robo/profiles` — profiles (fintech_robo:profiles)
- `/fintech-robo/goals` — goals (fintech_robo:goals)
- `/fintech-robo/models` — models (fintech_robo:models)
- `/fintech-robo/recommendations` — recommendations (fintech_robo:recommendations)
- `/fintech-robo/automation` — automation (fintech_robo:automation)
- _5 more..._

**Streaming events** via `bytewax`:
`investor_profile_created`, `goal_plan_defined`, `model_portfolio_published`, `recommendation_generated`, `recommendation_approved`, ...

**Standalone usage:**
```bash
pip install apg-fintech-robo
apg-fintech-robo --port 8080
```

---

### Payment Switch `fintech_switch`

> Payment Switch provides a world-class, standalone-deployable implementation of payment switch capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

**Package**: `apg-fintech-switch`  
**Path**: `capabilities/fintech/switch`  
**Version**: 1.1.0  

**Provides:**
- `iso8583_message_switching`
- `payment_routing_engine`
- `channel_key_management`
- `pin_block_translation`
- `mac_generation_verification`
- `mobile_money_switching`
- `ussd_session_management`
- `switch_settlement_reconciliation`
- `network_interface_management`
- `switch_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `keym`
- `encr`
- `keym`
- `fintech_payments`
- `fintech_gateway`
- `fintech_aml`

**Service methods** (42 total):
`_audit_event`, `route_transaction`, `switch_authorisation`, `_velocity_check_internal`, `settlement_routing`, `interchange_fee_calculation`, `scheme_compliance_check`, `switch_analytics`, `downtime_failover`, `transaction_replay`, `switch_health_check`, `load_balancing_status`, ...

**Governance rules** (38 total):
`tenant_context_required`, `switch_write_requires_policy`, `cross_tenant_access_denied`, `privilege_escalation_denied`, `transaction_message_type_supported`, `transaction_type_supported`, `transaction_stan_required`, `transaction_stan_unique`, ...

**UI Routes** (11):
- `/fintech-switch/dashboard` — dashboard (fintech_switch:view)
- `/fintech-switch/routing` — routing (fintech_switch:manage_routing)
- `/fintech-switch/transactions` — transactions (fintech_switch:monitor)
- `/fintech-switch/channels` — channels (fintech_switch:manage_channels)
- `/fintech-switch/security` — security (fintech_switch:manage_keys)
- `/fintech-switch/mobile-money` — mobile_money (fintech_switch:mobile_money)
- _5 more..._

**Streaming events** via `bytewax`:
`switch_transaction_received`, `switch_transaction_routed`, `switch_transaction_authorized`, `switch_transaction_reversed`, `switch_channel_registered`, ...

**Standalone usage:**
```bash
pip install apg-fintech-switch
apg-fintech-switch --port 8080
```

---

### Terminal Management System `fintech_terminal`

> Terminal Management System provides a world-class, standalone-deployable implementation of terminal management system capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

**Package**: `apg-fintech-terminal`  
**Path**: `capabilities/fintech/terminal`  
**Version**: 1.1.0  

**Provides:**
- `terminal_lifecycle_management`
- `terminal_key_injection_workflow`
- `terminal_parameter_deployment`
- `terminal_certificate_management`
- `terminal_health_monitoring`
- `pci_dss_compliance_tracking`
- `mobile_money_sdk_deployment`
- `terminal_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `keym`
- `encr`
- `keym`
- `fintech_switch`
- `fintech_payments`

**Service methods** (43 total):
`_audit_event`, `_get_terminal`, `_assert_active`, `register_terminal`, `activate_terminal`, `terminal_transaction`, `cash_deposit`, `cash_withdrawal`, `fund_transfer_terminal`, `bill_payment_terminal`, `balance_inquiry`, `mini_statement_terminal`, ...

**Governance rules** (37 total):
`tenant_context_required`, `terminal_write_requires_policy`, `cross_tenant_access_denied`, `privilege_escalation_denied`, `terminal_type_supported`, `terminal_serial_required`, `terminal_merchant_required`, `terminal_location_required`, ...

**UI Routes** (11):
- `/fintech-terminal/dashboard` — dashboard (fintech_terminal:view)
- `/fintech-terminal/terminals` — terminals (fintech_terminal:manage)
- `/fintech-terminal/keys` — key_management (fintech_terminal:manage_keys)
- `/fintech-terminal/parameters` — parameters (fintech_terminal:deploy_parameters)
- `/fintech-terminal/certificates` — certificates (fintech_terminal:manage_certificates)
- `/fintech-terminal/compliance` — compliance (fintech_terminal:compliance)
- _5 more..._

**Streaming events** via `bytewax`:
`terminal_registered`, `terminal_deployed`, `terminal_suspended`, `terminal_decommissioned`, `terminal_key_injected`, ...

**Standalone usage:**
```bash
pip install apg-fintech-terminal
apg-fintech-terminal --port 8080
```

---

### Algorithmic Trading `fintech_trading`

> Algorithmic Trading provides governed strategy-driven trading operations: strategy registration with asset class and policy controls, signal source attachment with freshness SLAs and lineage, backtesting with trade count and data source evidence, risk limit activation with approval, order intent staging with instrument and approval gates, execution recording, position snapshots, trading surveillance, and governance reviews. Every order intent requires both a risk limit reference and an explicit approval before it can be staged — preventing unsanctioned automated order flow.

**Package**: `apg-fintech-trading`  
**Path**: `capabilities/fintech/trading`  
**Version**: 1.1.0  

**Provides:**
- `trading_strategy_workflow`
- `trading_signal_workflow`
- `trading_backtest_workflow`
- `trading_risk_limit_workflow`
- `trading_order_intent_workflow`
- `trading_execution_workflow`
- `trading_position_workflow`
- `trading_surveillance_workflow`
- `trading_review_workflow`
- `trading_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_portfolio`
- `fintech_wealth`
- `fintech_robo`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `bia_anl`
- `fin_rpt`

**Service methods** (43 total):
`describe`, `evaluate`, `register_strategy`, `get_strategy`, `list_strategies`, `deactivate_strategy`, `attach_signal_source`, `place_order`, `cancel_order`, `order_status`, `order_book_snapshot`, `execute_algo_strategy`, ...

**Governance rules** (46 total):
`tenant_context_required`, `trading_write_requires_policy`, `strategy_owner_required`, `strategy_type_supported`, `strategy_asset_class_supported`, `strategy_policy_reference_required`, `signal_strategy_required`, `signal_source_required`, ...

**UI Routes** (12):
- `/fintech-trading/dashboard` — dashboard (fintech_trading:view)
- `/fintech-trading/strategies` — strategies (fintech_trading:strategies)
- `/fintech-trading/signals` — signals (fintech_trading:signals)
- `/fintech-trading/backtests` — backtests (fintech_trading:backtests)
- `/fintech-trading/risk` — risk (fintech_trading:risk)
- `/fintech-trading/orders` — orders (fintech_trading:orders)
- _6 more..._

**Streaming events** via `bytewax`:
`trading_strategy_registered`, `signal_source_attached`, `backtest_recorded`, `risk_limit_set`, `order_intent_staged`, ...

**Standalone usage:**
```bash
pip install apg-fintech-trading
apg-fintech-trading --port 8080
```

---

### Treasury Management System `fintech_treasury`

> Treasury Management System provides a world-class, standalone-deployable implementation of treasury management system capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

**Package**: `apg-fintech-treasury`  
**Path**: `capabilities/fintech/treasury`  
**Version**: 1.1.0  

**Provides:**
- `cash_position_management`
- `treasury_dealing_workflow`
- `counterparty_limit_governance`
- `settlement_instruction_workflow`
- `fx_rate_management`
- `liquidity_forecasting`
- `nostro_reconciliation`
- `cbk_regulatory_reporting`
- `treasury_risk_monitoring`
- `treasury_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `keym`
- `fintech_payments`
- `fintech_kyc`
- `fintech_aml`
- `fintech_risk`

**Service methods** (41 total):
`_audit_event`, `cash_position`, `liquidity_forecast`, `fx_exposure_report`, `hedge_instrument_create`, `hedge_effectiveness_test`, `bank_relationship_management`, `intercompany_loan`, `money_market_placement`, `fx_forward_booking`, `swap_valuation`, `payment_factory`, ...

**Governance rules** (42 total):
`tenant_context_required`, `treasury_write_requires_policy`, `cross_tenant_access_denied`, `privilege_escalation_denied`, `segregation_of_duties_required`, `cash_account_type_supported`, `cash_currency_supported`, `cash_posting_requires_double_entry`, ...

**UI Routes** (12):
- `/fintech-treasury/dashboard` — dashboard (fintech_treasury:view)
- `/fintech-treasury/cash` — cash_management (fintech_treasury:manage_cash)
- `/fintech-treasury/dealing` — dealing (fintech_treasury:deal)
- `/fintech-treasury/limits` — limits (fintech_treasury:manage_limits)
- `/fintech-treasury/settlement` — settlement (fintech_treasury:settle)
- `/fintech-treasury/fx` — fx (fintech_treasury:manage_fx)
- _6 more..._

**Streaming events** via `bytewax`:
`treasury_cash_position_updated`, `treasury_deal_booked`, `treasury_deal_confirmed`, `treasury_deal_settled`, `treasury_deal_cancelled`, ...

**Standalone usage:**
```bash
pip install apg-fintech-treasury
apg-fintech-treasury --port 8080
```

---

### Digital Wallets `fintech_wallets`

> Digital Wallets provides the stored-value ledger layer: wallet lifecycle (consumer, merchant, agent, escrow, treasury), instrument registration with verified token references, double-entry ledger operations (credit, debit, transfer), hold management for reserved funds, and limit governance. It is the balance-holding layer that other capabilities — payments, mobile, agency, neobanking — use to maintain available and held balances for their customers and operational accounts.

**Package**: `apg-fintech-wallets`  
**Path**: `capabilities/fintech/wallets`  
**Version**: 1.1.0  

**Provides:**
- `wallet_lifecycle`
- `stored_value_ledger`
- `wallet_instrument_registry`
- `wallet_transfer_workflow`
- `wallet_hold_workflow`
- `wallet_limit_governance`
- `wallet_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `walt`
- `fintech_payments`
- `fintech_gateway`
- `keym`

**Service methods** (40 total):
`describe`, `evaluate`, `open_wallet`, `register_instrument`, `credit_wallet`, `debit_wallet`, `transfer`, `place_hold`, `release_hold`, `register_wallet_agent`, `validate_batch`, `dashboard_summary`, ...

**Governance rules** (36 total):
`tenant_context_required`, `wallet_write_requires_policy`, `wallet_owner_required`, `wallet_type_supported`, `wallet_currency_supported`, `instrument_wallet_required`, `instrument_type_supported`, `instrument_token_required`, ...

**UI Routes** (8):
- `/fintech-wallets/dashboard` — dashboard (fintech_wallets:view)
- `/fintech-wallets/wallets` — wallets (fintech_wallets:manage_wallets)
- `/fintech-wallets/instruments` — instruments (fintech_wallets:manage_instruments)
- `/fintech-wallets/ledger` — ledger (fintech_wallets:view_ledger)
- `/fintech-wallets/limits` — limits (fintech_wallets:govern_limits)
- `/fintech-wallets/holds` — holds (fintech_wallets:operate)
- _2 more..._

**Streaming events** via `bytewax`:
`wallet_opened`, `wallet_instrument_registered`, `wallet_credited`, `wallet_debited`, `wallet_transfer_posted`, ...

**Standalone usage:**
```bash
pip install apg-fintech-wallets
apg-fintech-wallets --port 8080
```

---

### Wealth Management `fintech_wealth`

> Wealth Management provides regulated advisory and portfolio services: client profile onboarding with KYC, tax, and risk evidence; suitability assessment across risk tolerance, investment horizon, and goals; portfolio creation with advisor assignment and investment policy statement; advisory mandate setup (advisory, discretionary, model, execution-only); portfolio rebalance proposals with exact 100% allocation totals and analysis evidence; trade order staging with approval gates for large orders; performance recording; and fee schedule management. It is the client-facing wealth services layer that backs Robo Advisory and Portfolio Management.

**Package**: `apg-fintech-wealth`  
**Path**: `capabilities/fintech/wealth`  
**Version**: 1.1.0  

**Provides:**
- `wealth_client_profile_workflow`
- `suitability_profile_workflow`
- `portfolio_management_workflow`
- `advisory_mandate_workflow`
- `portfolio_rebalance_workflow`
- `wealth_order_workflow`
- `performance_reporting_workflow`
- `wealth_fee_workflow`
- `wealth_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_payments`
- `fintech_wallets`
- `bia_anl`
- `fin_rpt`

**Service methods** (40 total):
`describe`, `evaluate`, `client_suitability_assessment`, `create_portfolio`, `portfolio_rebalance`, `asset_allocation_review`, `performance_report`, `tax_loss_harvesting`, `dividend_reinvestment`, `financial_plan`, `portfolio_stress_test`, `wealth_dashboard`, ...

**Governance rules** (46 total):
`tenant_context_required`, `wealth_write_requires_policy`, `client_kyc_required`, `client_tax_required`, `client_risk_required`, `suitability_client_required`, `suitability_risk_supported`, `suitability_tolerance_supported`, ...

**UI Routes** (11):
- `/fintech-wealth/dashboard` — dashboard (fintech_wealth:view)
- `/fintech-wealth/clients` — clients (fintech_wealth:clients)
- `/fintech-wealth/suitability` — suitability (fintech_wealth:suitability)
- `/fintech-wealth/portfolios` — portfolios (fintech_wealth:portfolios)
- `/fintech-wealth/mandates` — mandates (fintech_wealth:mandates)
- `/fintech-wealth/rebalances` — rebalances (fintech_wealth:rebalances)
- _5 more..._

**Streaming events** via `bytewax`:
`client_profile_registered`, `suitability_profile_captured`, `portfolio_created`, `advisory_mandate_created`, `rebalance_proposed`, ...

**Standalone usage:**
```bash
pip install apg-fintech-wealth
apg-fintech-wealth --port 8080
```

---

## GOVERNMENT

### Budget Management `government_bud`

> Programme budgeting, vote accounting, commitment control, budget revisions, fiscal reporting, and Treasury submission for government entities. Enforces appropriation limits, prevents over-commitment, and ensures every budget revision carries a treasury notification reference.

**Package**: `apg-government-bud`  
**Path**: `capabilities/government/bud`  
**Version**: 1.0.0  

**Provides:**
- `budget_programme_workflow`
- `vote_accounting_workflow`
- `budget_revision_workflow`
- `commitment_control_workflow`
- `expenditure_recording_workflow`
- `fiscal_reporting_workflow`
- `budget_approval_workflow`
- `budget_review_workflow`
- `budget_agent_workflow`
- `treasury_submission_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `moni`
- `mqeb`

**Service methods** (41 total):
`describe`, `evaluate`, `record_budget`, `create_budget_ceiling`, `requisition`, `commitment_check`, `payment_approval`, `budget_revision`, `expenditure_report`, `budget_vs_actual`, `supplementary_budget`, `treasury_single_account`, ...

**Governance rules** (39 total):
`tenant_context_required`, `budget_write_requires_policy`, `budget_type_supported`, `budget_vote_required`, `budget_fund_source_supported`, `budget_approver_required`, `budget_evidence_required`, `vote_type_supported`, ...

**UI Routes** (12):
- `/government-bud/dashboard` — dashboard (government_bud:view)
- `/government-bud/budgets` — budgets (government_bud:budgets)
- `/government-bud/votes` — votes (government_bud:votes)
- `/government-bud/revisions` — revisions (government_bud:revisions)
- `/government-bud/commitments` — commitments (government_bud:commitments)
- `/government-bud/expenditures` — expenditures (government_bud:expenditures)
- _6 more..._

**Streaming events** via `bytewax`:
`budget_recorded`, `vote_recorded`, `budget_revision_recorded`, `commitment_recorded`, `expenditure_recorded`, ...

**Standalone usage:**
```bash
pip install apg-government-bud
apg-government-bud --port 8080
```

---

### Case Management `government_cas`

> Citizen case intake, assignment, workflow routing, SLA tracking, escalation, and outcome recording for government service delivery. Handles complaints, enquiries, applications, and regulatory referrals across all intake channels with full audit trail.

**Package**: `apg-government-cas`  
**Path**: `capabilities/government/cas`  
**Version**: 1.0.0  

**Provides:**
- `case_intake_workflow`
- `case_assignment_workflow`
- `case_routing_workflow`
- `sla_tracking_workflow`
- `case_escalation_workflow`
- `case_outcome_workflow`
- `case_notification_workflow`
- `case_review_workflow`
- `case_agent_workflow`
- `citizen_case_portal_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `nlpc`
- `srch`
- `moni`
- `mqeb`

**Service methods** (57 total):
`describe`, `evaluate`, `open_case`, `create_case`, `assign_officer`, `case_update`, `schedule_hearing`, `record_decision`, `close_case`, `appeal_management`, `sla_monitoring`, `workload_report`, ...

**Governance rules** (32 total):
`tenant_context_required`, `case_write_requires_policy`, `case_type_supported`, `case_intake_channel_supported`, `case_citizen_id_required`, `case_priority_supported`, `case_evidence_required`, `assignment_case_required`, ...

**UI Routes** (12):
- `/government-cas/dashboard` — dashboard (government_cas:view)
- `/government-cas/intake` — intake (government_cas:create)
- `/government-cas/cases` — cases (government_cas:cases)
- `/government-cas/assignments` — assignments (government_cas:assign)
- `/government-cas/escalations` — escalations (government_cas:escalate)
- `/government-cas/sla` — sla (government_cas:sla)
- _6 more..._

**Streaming events** via `bytewax`:
`case_opened`, `case_assigned`, `case_escalated`, `case_sla_breached`, `case_outcome_recorded`, ...

**Standalone usage:**
```bash
pip install apg-government-cas
apg-government-cas --port 8080
```

---

### Government Contracts and Procurement `government_con`

> End-to-end public procurement process covering tender management, bid evaluation, contract award, contract lifecycle management, variation control, performance monitoring, and PPDA compliance. Enforces the Public Procurement and Disposal Act requirements including debarment register and mandatory notifications.

**Package**: `apg-government-con`  
**Path**: `capabilities/government/con`  
**Version**: 1.0.0  

**Provides:**
- `tender_management_workflow`
- `evaluation_workflow`
- `contract_award_workflow`
- `contract_lifecycle_workflow`
- `contract_variation_workflow`
- `contract_performance_workflow`
- `ppda_compliance_workflow`
- `procurement_review_workflow`
- `procurement_agent_workflow`
- `debarment_register_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `moni`
- `mqeb`

**Service methods** (57 total):
`describe`, `evaluate`, `publish_tender`, `tender_publish`, `bid_submission`, `evaluation_committee`, `evaluate_bid`, `award_contract`, `contract_performance`, `variation_order`, `contract_close`, `procurement_analytics`, ...

**Governance rules** (31 total):
`tenant_context_required`, `con_write_requires_policy`, `procurement_method_supported`, `tender_ppda_threshold_required`, `tender_approver_required`, `tender_evidence_required`, `single_source_requires_justification`, `evaluation_tender_required`, ...

**UI Routes** (12):
- `/government-con/dashboard` — dashboard (government_con:view)
- `/government-con/tenders` — tenders (government_con:tenders)
- `/government-con/evaluations` — evaluations (government_con:evaluate)
- `/government-con/awards` — awards (government_con:award)
- `/government-con/contracts` — contracts (government_con:contracts)
- `/government-con/variations` — variations (government_con:vary)
- _6 more..._

**Streaming events** via `bytewax`:
`tender_published`, `tender_awarded`, `contract_signed`, `contract_varied`, `contract_performance_recorded`, ...

**Standalone usage:**
```bash
pip install apg-government-con
apg-government-con --port 8080
```

---

### Citizen Services Portal `government_csr`

> Self-service citizen portal supporting application submission, status tracking, e-payment, document verification, and service delivery analytics. Provides a unified interface for all government-to-citizen service transactions across web, mobile, USSD, and kiosk channels.

**Package**: `apg-government-csr`  
**Path**: `capabilities/government/csr`  
**Version**: 1.0.0  

**Provides:**
- `citizen_self_service_workflow`
- `service_application_workflow`
- `application_status_tracking_workflow`
- `epayment_workflow`
- `document_verification_workflow`
- `service_notification_workflow`
- `service_delivery_analytics_workflow`
- `citizen_review_workflow`
- `citizen_services_agent_workflow`
- `service_catalogue_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `srch`
- `moni`
- `mqeb`

**Service methods** (40 total):
`describe`, `evaluate`, `register_service`, `submit_application`, `submit_service_request`, `track_application`, `schedule_appointment`, `citizen_portal_login`, `document_verification_request`, `payment_for_service`, `feedback_submission`, `case_escalation`, ...

**Governance rules** (25 total):
`tenant_context_required`, `csr_write_requires_policy`, `service_type_supported`, `submission_citizen_id_required`, `submission_channel_supported`, `unauthenticated_submission_denied`, `cross_tenant_service_denied`, `payment_method_supported`, ...

**UI Routes** (11):
- `/government-csr/dashboard` — dashboard (government_csr:view)
- `/government-csr/services` — services (government_csr:services)
- `/government-csr/apply` — apply (government_csr:apply)
- `/government-csr/applications` — applications (government_csr:applications)
- `/government-csr/payments` — payments (government_csr:payments)
- `/government-csr/verifications` — verifications (government_csr:verify)
- _5 more..._

**Streaming events** via `bytewax`:
`service_application_submitted`, `application_status_updated`, `payment_completed`, `payment_failed`, `document_verified`, ...

**Standalone usage:**
```bash
pip install apg-government-csr
apg-government-csr --port 8080
```

---

### Electoral and Civil Registration `government_ele`

> Voter registration with biometric deduplication, polling station management, election results collation, and civil registry for births, deaths, marriages, and other vital events. Enforces integrity rules that prevent duplicate voter registration, underage registration, and result manipulation.

**Package**: `apg-government-ele`  
**Path**: `capabilities/government/ele`  
**Version**: 1.0.0  

**Provides:**
- `voter_registration_workflow`
- `biometric_deduplication_workflow`
- `polling_station_management_workflow`
- `election_management_workflow`
- `results_collation_workflow`
- `civil_registration_workflow`
- `electoral_verification_workflow`
- `electoral_review_workflow`
- `electoral_agent_workflow`
- `civil_registry_amendment_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `geos`
- `comp`
- `moni`
- `mqeb`

**Service methods** (40 total):
`describe`, `evaluate`, `register_voter`, `voter_registration`, `biometric_capture`, `polling_station_setup`, `voter_list_verification`, `ballot_management`, `vote_counting`, `result_collation`, `result_transmission`, `election_analytics`, ...

**Governance rules** (28 total):
`tenant_context_required`, `ele_write_requires_policy`, `registration_type_supported`, `voter_biometric_required`, `voter_national_id_required`, `voter_deduplication_required`, `duplicate_voter_denied`, `underage_voter_denied`, ...

**UI Routes** (12):
- `/government-ele/dashboard` — dashboard (government_ele:view)
- `/government-ele/registrations` — registrations (government_ele:register)
- `/government-ele/deduplication` — deduplication (government_ele:deduplicate)
- `/government-ele/polling-stations` — polling_stations (government_ele:stations)
- `/government-ele/elections` — elections (government_ele:elections)
- `/government-ele/results` — results (government_ele:results)
- _6 more..._

**Streaming events** via `bytewax`:
`voter_registered`, `duplicate_detected`, `duplicate_resolved`, `polling_station_assigned`, `election_results_collated`, ...

**Standalone usage:**
```bash
pip install apg-government-ele
apg-government-ele --port 8080
```

---

### Emergency Management `government_eme`

> Incident command, resource mobilisation, multi-agency coordination, EOC management, situation reporting, and after-action reviews. Implements the Incident Command System (ICS) framework with mandatory after-action reviews and strict EOC activation authority controls.

**Package**: `apg-government-eme`  
**Path**: `capabilities/government/eme`  
**Version**: 1.0.0  

**Provides:**
- `incident_command_workflow`
- `resource_mobilisation_workflow`
- `multi_agency_coordination_workflow`
- `eoc_management_workflow`
- `situation_reporting_workflow`
- `after_action_review_workflow`
- `emergency_review_workflow`
- `emergency_agent_workflow`
- `incident_phase_transition_workflow`
- `resource_demobilisation_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `geos`
- `moni`
- `schd`
- `mqeb`

**Service methods** (40 total):
`describe`, `evaluate`, `declare_incident`, `declare_emergency`, `activate_eoc`, `resource_mobilisation`, `multi_agency_coordination`, `situation_report`, `evacuation_management`, `relief_distribution`, `casualty_tracking`, `after_action_review`, ...

**Governance rules** (29 total):
`tenant_context_required`, `eme_write_requires_policy`, `incident_type_supported`, `incident_severity_supported`, `incident_location_required`, `incident_commander_required`, `incident_evidence_required`, `resource_type_supported`, ...

**UI Routes** (11):
- `/government-eme/dashboard` — dashboard (government_eme:view)
- `/government-eme/incidents` — incidents (government_eme:incidents)
- `/government-eme/resources` — resources (government_eme:resources)
- `/government-eme/agencies` — agencies (government_eme:agencies)
- `/government-eme/eoc` — eoc (government_eme:eoc)
- `/government-eme/situation-reports` — situation_reports (government_eme:reports)
- _5 more..._

**Streaming events** via `bytewax`:
`incident_declared`, `incident_phase_transitioned`, `resource_mobilised`, `resource_demobilised`, `agency_activated`, ...

**Standalone usage:**
```bash
pip install apg-government-eme
apg-government-eme --port 8080
```

---

### Law Enforcement and Justice `government_law`

> Incident reporting with OB number generation, case docket management, evidence chain of custody, court scheduling, and prosecution tracking from arrest to conviction. Enforces strict chain-of-custody rules and requires DPP reference numbers before prosecution can commence.

**Package**: `apg-government-law`  
**Path**: `capabilities/government/law`  
**Version**: 1.0.0  

**Provides:**
- `incident_reporting_workflow`
- `docket_management_workflow`
- `evidence_chain_of_custody_workflow`
- `court_scheduling_workflow`
- `prosecution_tracking_workflow`
- `law_enforcement_review_workflow`
- `law_enforcement_agent_workflow`
- `ob_number_generation_workflow`
- `witness_management_workflow`
- `inter_agency_referral_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `geos`
- `schd`
- `moni`
- `mqeb`

**Service methods** (40 total):
`describe`, `evaluate`, `report_incident`, `incident_report`, `assign_case`, `evidence_intake`, `suspect_record`, `arrest_record`, `court_scheduling`, `prosecution_handover`, `case_analytics`, `crime_statistics`, ...

**Governance rules** (28 total):
`tenant_context_required`, `law_write_requires_policy`, `incident_type_supported`, `incident_ob_number_required`, `incident_reporting_officer_required`, `incident_location_required`, `incident_evidence_required`, `docket_incident_required`, ...

**UI Routes** (11):
- `/government-law/dashboard` — dashboard (government_law:view)
- `/government-law/incidents` — incidents (government_law:incidents)
- `/government-law/dockets` — dockets (government_law:dockets)
- `/government-law/evidence` — evidence (government_law:evidence)
- `/government-law/custody` — custody (government_law:custody)
- `/government-law/court-scheduling` — court_scheduling (government_law:court)
- _5 more..._

**Streaming events** via `bytewax`:
`incident_reported`, `docket_opened`, `docket_status_changed`, `evidence_logged`, `evidence_custody_action_recorded`, ...

**Standalone usage:**
```bash
pip install apg-government-law
apg-government-law --port 8080
```

---

### Licensing and Permits `government_lic`

> Business and professional licence applications, renewals, inspections, revocations, and fee collection with full compliance monitoring. Enforces that licences cannot be renewed if the last inspection failed, prevents duplicate licences, and requires formal notice before revocation.

**Package**: `apg-government-lic`  
**Path**: `capabilities/government/lic`  
**Version**: 1.0.0  

**Provides:**
- `licence_application_workflow`
- `licence_issuance_workflow`
- `inspection_scheduling_workflow`
- `licence_renewal_workflow`
- `fee_collection_workflow`
- `licence_revocation_workflow`
- `licensing_review_workflow`
- `licensing_agent_workflow`
- `licence_status_tracking_workflow`
- `compliance_monitoring_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `schd`
- `comp`
- `moni`
- `mqeb`

**Service methods** (40 total):
`describe`, `evaluate`, `submit_application`, `apply_licence`, `background_check`, `premises_inspection`, `issue_licence`, `renew_licence`, `licence_renewal`, `suspend_licence`, `revoke_licence`, `licence_revoke`, ...

**Governance rules** (28 total):
`tenant_context_required`, `lic_write_requires_policy`, `licence_type_supported`, `application_applicant_required`, `application_fee_required`, `application_evidence_required`, `licence_approved_application_required`, `licence_number_required`, ...

**UI Routes** (11):
- `/government-lic/dashboard` — dashboard (government_lic:view)
- `/government-lic/applications` — applications (government_lic:apply)
- `/government-lic/licences` — licences (government_lic:licences)
- `/government-lic/inspections` — inspections (government_lic:inspect)
- `/government-lic/renewals` — renewals (government_lic:renew)
- `/government-lic/fees` — fees (government_lic:fees)
- _5 more..._

**Streaming events** via `bytewax`:
`licence_application_submitted`, `licence_issued`, `inspection_scheduled`, `inspection_outcome_recorded`, `licence_renewed`, ...

**Standalone usage:**
```bash
pip install apg-government-lic
apg-government-lic --port 8080
```

---

### Permits Management `government_per`

> Building permits, environmental permits, conditional approvals, inspection scheduling, and compliance monitoring. Prevents construction before permit issuance, enforces occupation certificate requirements, and triggers enforcement actions on condition breaches.

**Package**: `apg-government-per`  
**Path**: `capabilities/government/per`  
**Version**: 1.0.0  

**Provides:**
- `permit_application_workflow`
- `permit_issuance_workflow`
- `conditional_approval_workflow`
- `inspection_scheduling_workflow`
- `permit_compliance_monitoring_workflow`
- `permit_revocation_workflow`
- `permits_review_workflow`
- `permits_agent_workflow`
- `permit_transfer_workflow`
- `enforcement_action_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `geos`
- `schd`
- `comp`
- `moni`
- `mqeb`

**Service methods** (40 total):
`describe`, `evaluate`, `submit_application`, `apply_permit`, `technical_review`, `schedule_inspection`, `record_inspection`, `issue_permit`, `reject_permit`, `permit_renewal`, `revoke_permit`, `permit_register`, ...

**Governance rules** (27 total):
`tenant_context_required`, `per_write_requires_policy`, `permit_type_supported`, `application_applicant_required`, `application_site_required`, `application_fee_required`, `application_evidence_required`, `permit_approved_application_required`, ...

**UI Routes** (11):
- `/government-per/dashboard` — dashboard (government_per:view)
- `/government-per/applications` — applications (government_per:apply)
- `/government-per/permits` — permits (government_per:permits)
- `/government-per/conditions` — conditions (government_per:conditions)
- `/government-per/inspections` — inspections (government_per:inspect)
- `/government-per/compliance` — compliance (government_per:compliance)
- _5 more..._

**Streaming events** via `bytewax`:
`permit_application_submitted`, `permit_issued`, `permit_condition_recorded`, `inspection_scheduled`, `inspection_outcome_recorded`, ...

**Standalone usage:**
```bash
pip install apg-government-per
apg-government-per --port 8080
```

---

### Tax Administration `government_tax`

> Taxpayer registration, return filing, assessment, objections, debt collection, and audit case management. Implements the full tax administration lifecycle from TIN issuance through audit closure, with strict controls on duplicate PINs, objection deadlines, and debt collection procedures.

**Package**: `apg-government-tax`  
**Path**: `capabilities/government/tax`  
**Version**: 1.0.0  

**Provides:**
- `taxpayer_registration_workflow`
- `return_filing_workflow`
- `tax_assessment_workflow`
- `objection_management_workflow`
- `debt_collection_workflow`
- `audit_case_management_workflow`
- `tax_review_workflow`
- `tax_agent_workflow`
- `tax_refund_workflow`
- `compliance_risk_scoring_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `schd`
- `moni`
- `mqeb`

**Service methods** (49 total):
`put`, `get_item`, `tenant_values`, `count`, `describe`, `evaluate`, `register_taxpayer`, `update_taxpayer`, `deregister_taxpayer`, `taxpayer_search`, `verify_tin`, `submit_return`, ...

**Governance rules** (30 total):
`tenant_context_required`, `tax_write_requires_policy`, `tax_type_supported`, `registration_pin_required`, `registration_national_id_required`, `registration_evidence_required`, `duplicate_pin_denied`, `return_type_supported`, ...

**UI Routes** (12):
- `/government-tax/dashboard` — dashboard (government_tax:view)
- `/government-tax/registrations` — registrations (government_tax:register)
- `/government-tax/returns` — returns (government_tax:returns)
- `/government-tax/assessments` — assessments (government_tax:assess)
- `/government-tax/objections` — objections (government_tax:object)
- `/government-tax/debt-collection` — debt_collection (government_tax:collect)
- _6 more..._

**Streaming events** via `bytewax`:
`taxpayer_registered`, `tax_return_filed`, `tax_assessed`, `objection_filed`, `objection_determined`, ...

**Standalone usage:**
```bash
pip install apg-government-tax
apg-government-tax --port 8080
```

---

## GRC

### Audit Management `grc_aud`

> Audit Management provides a world-class, standalone-deployable implementation of audit management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

**Package**: `apg-grc-aud`  
**Path**: `capabilities/grc/aud`  
**Version**: 1.0.0  

**Provides:**
- `audit_program_lifecycle`
- `audit_finding_lifecycle`
- `audit_evidence_workflow`
- `audit_report_workflow`
- `audit_dashboard_service`
- `audit_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `grc_doc`
- `wflo`
- `grc_pol`

**Service methods** (40 total):
`_audit_event`, `_get_engagement`, `_get_finding`, `create_audit_plan`, `create_audit_engagement`, `fieldwork_record`, `draft_audit_report`, `management_response`, `finalise_report`, `issue_tracking`, `follow_up_audit`, `close_finding`, ...

**Governance rules** (47 total):
`tenant_context_required`, `cross_tenant_access_denied`, `aud_write_requires_policy`, `privilege_escalation_denied`, `admin_operation_requires_mfa`, `audit_requires_title`, `audit_requires_auditor`, `audit_type_supported`, ...

**UI Routes** (10):
- `/grc-aud/dashboard` — dashboard (grc_aud:view)
- `/grc-aud/audits` — audits (grc_aud:manage_audits)
- `/grc-aud/audits/:id` — audit_detail (grc_aud:view)
- `/grc-aud/findings` — findings (grc_aud:manage_findings)
- `/grc-aud/findings/:id` — finding_detail (grc_aud:view)
- `/grc-aud/evidence` — evidence (grc_aud:manage_evidence)
- _4 more..._

**Streaming events** via `bytewax`:
`audit_planned`, `audit_started`, `audit_fieldwork_completed`, `audit_finding_raised`, `audit_finding_updated`, ...

**Standalone usage:**
```bash
pip install apg-grc-aud
apg-grc-aud --port 8080
```

---

### Document Management `grc_doc`

> `grc_doc` is the APG capability packet for governed document repositories, templates, revisions, approvals, publication, retention, access, processing, and AI-agent review. It keeps the package boundary dependency-light so generated APG

**Package**: `apg-grc-doc`  
**Path**: `capabilities/grc/doc`  
**Version**: 2.2.0  

**Provides:**
- `document_repository_lifecycle`
- `document_template_lifecycle`
- `document_revision_workflow`
- `document_approval_workflow`
- `document_publication_workflow`
- `document_retention_workflow`
- `document_access_workflow`
- `document_processing_workflow`
- `document_dashboard_service`
- `document_audit_trail`
- `doc_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `grc_pol`
- `wflo`
- `srch`

**Service methods** (40 total):
`describe`, `evaluate`, `create_document`, `register_template`, `create_revision`, `approve_document`, `publish_document`, `assign_retention_policy`, `archive_document`, `grant_access`, `register_processing_job`, `complete_processing_job`, ...

**Governance rules** (62 total):
`tenant_context_required`, `cross_tenant_access_denied`, `doc_write_requires_policy`, `privilege_escalation_denied`, `admin_operation_requires_mfa`, `document_requires_title`, `document_requires_owner`, `document_type_supported`, ...

**UI Routes** (11):
- `/grc-doc/dashboard` — dashboard (grc_doc:view)
- `/grc-doc/documents` — documents (grc_doc:manage_documents)
- `/grc-doc/documents/:id` — document_detail (grc_doc:view)
- `/grc-doc/templates` — templates (grc_doc:manage_templates)
- `/grc-doc/reviews` — reviews (grc_doc:review)
- `/grc-doc/retention` — retention (grc_doc:manage_retention)
- _5 more..._

**Streaming events** via `bytewax`:
`document_created`, `document_updated`, `document_deleted`, `template_registered`, `template_updated`, ...

**Standalone usage:**
```bash
pip install apg-grc-doc
apg-grc-doc --port 8080
```

---

### Incident and Case Management `grc_icm`

> Incident and Case Management provides a world-class, standalone-deployable implementation of incident and case management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

**Package**: `apg-grc-icm`  
**Path**: `capabilities/grc/icm`  
**Version**: 1.0.0  

**Provides:**
- `incident_lifecycle_management`
- `case_management_workflow`
- `incident_evidence_workflow`
- `regulatory_notification_workflow`
- `post_incident_review_workflow`
- `icm_dashboard_service`
- `icm_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `grc_doc`
- `wflo`
- `grc_pol`

**Service methods** (40 total):
`_audit_event`, `_get_incident`, `report_incident`, `incident_triage`, `incident_investigation`, `root_cause_analysis`, `corrective_action`, `corrective_action_update`, `close_incident`, `regulatory_notification`, `compliance_test`, `compliance_deficiency`, ...

**Governance rules** (41 total):
`tenant_context_required`, `cross_tenant_access_denied`, `icm_write_requires_policy`, `privilege_escalation_denied`, `admin_operation_requires_mfa`, `incident_requires_title`, `incident_type_supported`, `incident_severity_supported`, ...

**UI Routes** (10):
- `/grc-icm/dashboard` — dashboard (grc_icm:view)
- `/grc-icm/incidents` — incidents (grc_icm:manage_incidents)
- `/grc-icm/incidents/:id` — incident_detail (grc_icm:view)
- `/grc-icm/cases` — cases (grc_icm:manage_cases)
- `/grc-icm/cases/:id` — case_detail (grc_icm:view)
- `/grc-icm/evidence` — evidence (grc_icm:manage_evidence)
- _4 more..._

**Streaming events** via `bytewax`:
`incident_reported`, `incident_triaged`, `incident_severity_upgraded`, `incident_severity_downgraded`, `incident_contained`, ...

**Standalone usage:**
```bash
pip install apg-grc-icm
apg-grc-icm --port 8080
```

---

### Policy Management `grc_pol`

> Policy Management provides a world-class, standalone-deployable implementation of policy management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

**Package**: `apg-grc-pol`  
**Path**: `capabilities/grc/pol`  
**Version**: 1.0.0  

**Provides:**
- `policy_lifecycle_management`
- `policy_acknowledgement_workflow`
- `policy_exception_workflow`
- `policy_review_workflow`
- `policy_publication_workflow`
- `policy_dashboard_service`
- `pol_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`

**Service methods** (40 total):
`_audit_event`, `_get_policy`, `create_policy`, `draft_policy_content`, `policy_review`, `approve_policy`, `publish_policy`, `acknowledge_policy`, `policy_exception_request`, `approve_exception`, `policy_revision`, `retire_policy`, ...

**Governance rules** (42 total):
`tenant_context_required`, `cross_tenant_access_denied`, `pol_write_requires_policy`, `privilege_escalation_denied`, `admin_operation_requires_mfa`, `policy_requires_title`, `policy_type_supported`, `policy_requires_owner`, ...

**UI Routes** (10):
- `/grc-pol/dashboard` — dashboard (grc_pol:view)
- `/grc-pol/policies` — policies (grc_pol:manage_policies)
- `/grc-pol/policies/:id` — policy_detail (grc_pol:view)
- `/grc-pol/acknowledgements` — acknowledgements (grc_pol:manage_acknowledgements)
- `/grc-pol/exceptions` — exceptions (grc_pol:manage_exceptions)
- `/grc-pol/reviews` — reviews (grc_pol:review)
- _4 more..._

**Streaming events** via `bytewax`:
`policy_drafted`, `policy_submitted_for_review`, `policy_review_completed`, `policy_approved`, `policy_rejected`, ...

**Standalone usage:**
```bash
pip install apg-grc-pol
apg-grc-pol --port 8080
```

---

### Risk and Compliance Management `grc_rcm`

> `grc_rcm` is the APG capability packet for governed risk, control, compliance, evidence, issue, exception, governance-decision, and AI-agent review lifecycles. It is intentionally dependency-light at the package boundary so APG applications

**Package**: `apg-grc-rcm`  
**Path**: `capabilities/grc/rcm`  
**Version**: 2.2.0  

**Provides:**
- `risk_register_lifecycle`
- `control_library_lifecycle`
- `compliance_obligation_lifecycle`
- `control_assessment_workflow`
- `evidence_management_workflow`
- `issue_remediation_workflow`
- `governance_decision_workflow`
- `exception_management_workflow`
- `risk_heatmap_service`
- `rcm_dashboard_service`
- `rcm_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `grc_doc`
- `wflo`
- `grc_pol`

**Service methods** (46 total):
`describe`, `evaluate`, `register_risk`, `register_control`, `register_obligation`, `assess_control`, `collect_evidence`, `open_issue`, `remediate_issue`, `record_governance_decision`, `register_exception`, `register_rcm_agent`, ...

**Governance rules** (66 total):
`tenant_context_required`, `cross_tenant_access_denied`, `rcm_write_requires_policy`, `privilege_escalation_denied`, `admin_operation_requires_mfa`, `risk_requires_title`, `risk_requires_owner`, `risk_category_supported`, ...

**UI Routes** (13):
- `/grc-rcm/dashboard` — dashboard (grc_rcm:view)
- `/grc-rcm/heatmap` — heatmap (grc_rcm:view)
- `/grc-rcm/risks` — risks (grc_rcm:manage_risks)
- `/grc-rcm/risks/:id` — risk_detail (grc_rcm:view)
- `/grc-rcm/controls` — controls (grc_rcm:manage_controls)
- `/grc-rcm/obligations` — obligations (grc_rcm:manage_obligations)
- _7 more..._

**Streaming events** via `bytewax`:
`risk_registered`, `risk_updated`, `risk_accepted`, `risk_mitigated`, `risk_closed`, ...

**Standalone usage:**
```bash
pip install apg-grc-rcm
apg-grc-rcm --port 8080
```

---

### Risk and Security Assessment `grc_rsa`

> Risk and Security Assessment provides a world-class, standalone-deployable implementation of risk and security assessment capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

**Package**: `apg-grc-rsa`  
**Path**: `capabilities/grc/rsa`  
**Version**: 1.0.0  

**Provides:**
- `security_assessment_lifecycle`
- `vulnerability_finding_workflow`
- `remediation_tracking_workflow`
- `vendor_risk_assessment_workflow`
- `threat_modelling_workflow`
- `rsa_dashboard_service`
- `rsa_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `grc_rcm`
- `grc_doc`
- `wflo`

**Service methods** (40 total):
`_audit_event`, `_get_risk`, `risk_register_entry`, `risk_assessment`, `inherent_risk_score`, `residual_risk_score`, `update_residual_score`, `risk_heat_map`, `control_assessment`, `control_gap`, `risk_treatment_plan`, `risk_treatment_update`, ...

**Governance rules** (43 total):
`tenant_context_required`, `cross_tenant_access_denied`, `rsa_write_requires_policy`, `privilege_escalation_denied`, `admin_operation_requires_mfa`, `assessment_requires_title`, `assessment_type_supported`, `assessment_requires_lead_assessor`, ...

**UI Routes** (10):
- `/grc-rsa/dashboard` — dashboard (grc_rsa:view)
- `/grc-rsa/assessments` — assessments (grc_rsa:manage_assessments)
- `/grc-rsa/assessments/:id` — assessment_detail (grc_rsa:view)
- `/grc-rsa/findings` — findings (grc_rsa:manage_findings)
- `/grc-rsa/findings/:id` — finding_detail (grc_rsa:view)
- `/grc-rsa/remediation` — remediation (grc_rsa:manage_remediation)
- _4 more..._

**Streaming events** via `bytewax`:
`assessment_scoped`, `assessment_started`, `assessment_finding_raised`, `assessment_finding_severity_upgraded`, `assessment_finding_accepted`, ...

**Standalone usage:**
```bash
pip install apg-grc-rsa
apg-grc-rsa --port 8080
```

---

## HCM

### Employee Data Management `chr_employee_data_management`

> `chr_employee_data_management` is the APG capability packet for governed employee profiles, organization structure, personal information, emergency contacts, employment history, skills, certifications, data-quality issues, and employee

**Package**: `apg-hcm-employee_data_management`  
**Path**: `capabilities/hcm/chr/employee_data_management`  
**Version**: 2.2.0  

**Provides:**
- `employee_profile_lifecycle`
- `employee_identity_registry`
- `department_lifecycle`
- `position_lifecycle`
- `employment_history_lifecycle`
- `employee_skill_lifecycle`
- `employee_certification_lifecycle`
- `employee_contact_lifecycle`
- `employee_data_quality_workflow`
- `employee_dashboard_service`
- `employee_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `srch`
- `mdm`

**Service methods** (40 total):
`describe`, `evaluate`, `create_department`, `create_position`, `create_employee`, `change_employee_status`, `record_personal_info`, `record_emergency_contact`, `record_employment_history`, `assign_skill`, `assign_certification`, `record_data_quality_issue`, ...

**Governance rules** (83 total):
`tenant_context_required`, `employee_write_requires_policy`, `cross_tenant_employee_access_denied`, `cross_tenant_department_access_denied`, `cross_tenant_position_access_denied`, `self_role_promotion_denied`, `non_admin_delete_employee_denied`, `mass_update_requires_dual_control`, ...

**UI Routes** (12):
- `/hcm/employees/dashboard` — dashboard (chr_employee_data_management:view)
- `/hcm/employees` — employees (chr_employee_data_management:manage_employees)
- `/hcm/employees/departments` — departments (chr_employee_data_management:manage_structure)
- `/hcm/employees/positions` — positions (chr_employee_data_management:manage_structure)
- `/hcm/employees/personal-info` — personal_info (chr_employee_data_management:manage_sensitive)
- `/hcm/employees/contacts` — contacts (chr_employee_data_management:manage_employees)
- _6 more..._

**Streaming events** via `bytewax`:
`department_created`, `department_updated`, `position_created`, `position_updated`, `employee_created`, ...

**Standalone usage:**
```bash
pip install apg-hcm-employee_data_management
apg-hcm-employee_data_management --port 8080
```

---

### Payroll Management `pay_payroll`

> `pay_payroll` is the APG capability packet for governed payroll periods, pay groups, employee pay profiles, pay components, time imports, payroll runs, line items, tax calculations, adjustments, payment batches, payslips, tax filings, and payroll-agent review. It keeps the package boundary dependency-light so generated APG applications can compose it immediately while production deployments attach durable employee data, time, benefits, general ledger, banking, tax authority, workflow, audit, notification, and Bytewax topology through adapters.

**Package**: `apg-hcm-payroll`  
**Path**: `capabilities/hcm/pay/payroll`  
**Version**: 2.2.0  

**Provides:**
- `payroll_period_lifecycle`
- `pay_group_lifecycle`
- `employee_pay_profile_lifecycle`
- `pay_component_lifecycle`
- `payroll_tax_rule_lifecycle`
- `time_import_lifecycle`
- `payroll_run_lifecycle`
- `payroll_line_item_lifecycle`
- `payroll_tax_lifecycle`
- `payroll_adjustment_lifecycle`
- `payroll_payment_workflow`
- `payslip_lifecycle`
- `payroll_tax_filing_lifecycle`
- `payroll_dashboard_service`
- `payroll_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `mqeb`
- `comp`
- `schd`

**Service methods** (44 total):
`describe`, `evaluate`, `create_payroll_period`, `create_pay_group`, `create_employee_pay_profile`, `create_pay_component`, `record_time_import`, `start_payroll_run`, `add_line_item`, `record_tax`, `record_adjustment`, `approve_payroll_run`, ...

**Governance rules** (100 total):
`tenant_context_required`, `payroll_write_requires_policy`, `cross_tenant_employee_access_denied`, `cross_tenant_period_access_denied`, `cross_tenant_pay_group_access_denied`, `initiator_cannot_approve_own_run`, `non_admin_void_run_denied`, `self_payment_creation_denied`, ...

**UI Routes** (16):
- `/hcm/payroll/dashboard` — dashboard (pay_payroll:view)
- `/hcm/payroll/periods` — periods (pay_payroll:manage_periods)
- `/hcm/payroll/pay-groups` — pay_groups (pay_payroll:manage_setup)
- `/hcm/payroll/profiles` — profiles (pay_payroll:manage_profiles)
- `/hcm/payroll/components` — components (pay_payroll:manage_setup)
- `/hcm/payroll/tax-rules` — tax_rules (pay_payroll:manage_tax_rules)
- _10 more..._

**Streaming events** via `bytewax`:
`payroll_period_created`, `payroll_period_locked`, `pay_group_created`, `employee_pay_profile_created`, `pay_component_created`, ...

**Standalone usage:**
```bash
pip install apg-hcm-payroll
apg-hcm-payroll --port 8080
```

---

### Time and Attendance Tracking `tat_time_attendance`

> Time and Attendance is the APG capability packet for work policies, schedules, shifts, time entries, breaks, timesheets, leave requests, attendance exceptions, payroll exports, and attendance-focused AI agents.

**Package**: `apg-hcm-time_attendance`  
**Path**: `capabilities/hcm/tat/time_attendance`  
**Version**: 2.2.0  

**Provides:**
- `time_policy_lifecycle`
- `work_schedule_lifecycle`
- `shift_lifecycle`
- `time_entry_lifecycle`
- `break_lifecycle`
- `timesheet_lifecycle`
- `overtime_calculation_service`
- `comp_time_accrual_service`
- `leave_request_lifecycle`
- `attendance_approval_workflow`
- `attendance_exception_workflow`
- `attendance_payroll_export`
- `attendance_dashboard_service`
- `attendance_agents`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `schd`
- `mqeb`

**Service methods** (66 total):
`_fetch_one`, `_fetch_many`, `_soft_delete`, `create_time_policy`, `get_time_policy`, `list_time_policies`, `update_time_policy`, `delete_time_policy`, `create_shift_schedule`, `get_shift_schedule`, `list_shift_schedules`, `create_shift`, ...

**Governance rules** (81 total):
`tenant_context_required`, `operation_policy_required`, `cross_tenant_employee_access_denied`, `cross_tenant_schedule_access_denied`, `cross_tenant_policy_access_denied`, `employee_cannot_approve_own_timesheet`, `employee_cannot_approve_own_overtime`, `supervisor_entry_requires_employee_acknowledgement`, ...

**UI Routes** (14):
- `/hcm/time-attendance/dashboard` — dashboard (tat_time_attendance:view)
- `/hcm/time-attendance/policies` — policies (tat_time_attendance:manage_policies)
- `/hcm/time-attendance/overtime-rules` — overtime_rules (tat_time_attendance:manage_policies)
- `/hcm/time-attendance/schedules` — schedules (tat_time_attendance:manage_schedules)
- `/hcm/time-attendance/shifts` — shifts (tat_time_attendance:manage_schedules)
- `/hcm/time-attendance/time-entries` — time_entries (tat_time_attendance:record_time)
- _8 more..._

**Streaming events** via `bytewax`:
`attendance_policy_created`, `attendance_policy_updated`, `overtime_rule_created`, `overtime_rule_updated`, `attendance_schedule_created`, ...

**Standalone usage:**
```bash
pip install apg-hcm-time_attendance
apg-hcm-time_attendance --port 8080
```

---

## HEALTHCARE

### Clinical Analytics `healthcare_ana`

> Provides population health analytics, clinical outcomes measurement, readmission prediction, quality indicator tracking, and care gap identification for healthcare tenants. Supports cohort management, predictive model deployment, and structured report generation aligned with CMS Star, Joint Commission, and peer-group benchmarks.

**Package**: `apg-healthcare-ana`  
**Path**: `capabilities/healthcare/ana`  
**Version**: 1.0.0  

**Provides:**
- `population_health_analytics`
- `clinical_outcomes_measurement`
- `readmission_prediction`
- `quality_indicator_tracking`
- `cohort_management`
- `clinical_benchmarking`
- `analytics_report_generation`
- `care_gap_identification`
- `predictive_model_management`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `nlpc`
- `moni`
- `mqeb`
- `schd`

**Service methods** (40 total):
`describe`, `evaluate`, `create_cohort`, `get_cohort`, `list_cohorts`, `update_cohort`, `activate_cohort`, `delete_cohort`, `population_health_report`, `readmission_analysis`, `length_of_stay_analytics`, `disease_surveillance`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `analysis_type_supported`, `cohort_requires_segment`, `metric_type_supported`, `prediction_model_supported`, `model_deployment_requires_approval`, `phi_export_requires_deidentification`, ...

**UI Routes** (13):
- `/healthcare-ana/dashboard` — dashboard (healthcare_ana:view)
- `/healthcare-ana/population` — population (healthcare_ana:population)
- `/healthcare-ana/cohorts` — cohorts (healthcare_ana:cohorts)
- `/healthcare-ana/cohorts/<id>` — cohort_detail (healthcare_ana:cohorts)
- `/healthcare-ana/metrics` — metrics (healthcare_ana:metrics)
- `/healthcare-ana/predictions` — predictions (healthcare_ana:predictions)
- _7 more..._

**Streaming events** via `bytewax`:
`cohort_created`, `cohort_updated`, `metric_recorded`, `prediction_generated`, `benchmark_updated`, ...

**Standalone usage:**
```bash
pip install apg-healthcare-ana
apg-healthcare-ana --port 8080
```

---

### Clinical Management `healthcare_cli`

> Clinical workflow orchestration capability providing care plan management, clinical protocol activation, workflow task tracking, clinical decision support (CDS) alerts, structured handoff management, and care team coordination. Enforces structured SBAR handoff format and requires team assignment before care plan activation.

**Package**: `apg-healthcare-cli`  
**Path**: `capabilities/healthcare/cli`  
**Version**: 1.0.0  

**Provides:**
- `care_plan_management`
- `clinical_workflow_orchestration`
- `protocol_adherence_tracking`
- `clinical_decision_support`
- `care_team_management`
- `clinical_handoff_management`
- `intervention_tracking`
- `deterioration_alerting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `nlpc`
- `moni`
- `mqeb`

**Service methods** (40 total):
`describe`, `create_care_plan`, `activate_care_plan`, `complete_care_plan`, `get_care_plan`, `list_care_plans`, `add_intervention`, `create_care_pathway`, `enrol_patient_pathway`, `pathway_progress`, `clinical_audit`, `quality_indicator_report`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_care_plan_access_denied`, `care_plan_status_supported`, `care_plan_requires_team_member`, `protocol_type_supported`, `protocol_activation_requires_criteria`, `workflow_state_supported`, ...

**UI Routes** (13):
- `/healthcare-cli/dashboard` — dashboard (healthcare_cli:view)
- `/healthcare-cli/care-plans` — care_plans (healthcare_cli:care_plans)
- `/healthcare-cli/care-plans/new` — care_plan_new (healthcare_cli:care_plans_write)
- `/healthcare-cli/care-plans/<id>` — care_plan_detail (healthcare_cli:care_plans)
- `/healthcare-cli/protocols` — protocols (healthcare_cli:protocols)
- `/healthcare-cli/protocols/<id>` — protocol_detail (healthcare_cli:protocols)
- _7 more..._

**Streaming events** via `bytewax`:
`care_plan_created`, `care_plan_activated`, `care_plan_completed`, `protocol_activated`, `workflow_state_changed`, ...

**Standalone usage:**
```bash
pip install apg-healthcare-cli
apg-healthcare-cli --port 8080
```

---

### Medical Device Management `healthcare_dev`

> Medical device lifecycle management covering device inventory with FDA UDI tracking, preventive and corrective maintenance scheduling with work orders, calibration record management, and adverse event reporting. Enforces UDI requirements for Class II/III devices, blocks use of recalled or calibration-overdue devices, and automatically escalates serious adverse events.

**Package**: `apg-healthcare-dev`  
**Path**: `capabilities/healthcare/dev`  
**Version**: 1.0.0  

**Provides:**
- `device_inventory_management`
- `maintenance_schedule_management`
- `calibration_record_tracking`
- `fda_udi_tracking`
- `adverse_event_reporting`
- `work_order_management`
- `device_lifecycle_management`
- `regulatory_submission_support`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `schd`
- `moni`
- `mqeb`

**Service methods** (40 total):
`describe`, `register_device`, `update_device_status`, `get_device`, `list_devices`, `device_inventory`, `udi_lookup`, `schedule_maintenance`, `maintenance_schedule`, `log_maintenance`, `complete_maintenance`, `list_maintenance`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_device_access_denied`, `recalled_device_use_denied`, `calibration_overdue_blocks_use`, `udi_required_for_class_ii_iii`, `device_type_supported`, `device_class_supported`, ...

**UI Routes** (13):
- `/healthcare-dev/dashboard` — dashboard (healthcare_dev:view)
- `/healthcare-dev/inventory` — inventory (healthcare_dev:inventory)
- `/healthcare-dev/inventory/register` — device_register (healthcare_dev:inventory_write)
- `/healthcare-dev/inventory/<id>` — device_detail (healthcare_dev:inventory)
- `/healthcare-dev/maintenance` — maintenance (healthcare_dev:maintenance)
- `/healthcare-dev/work-orders` — work_orders (healthcare_dev:maintenance)
- _7 more..._

**Streaming events** via `bytewax`:
`device_registered`, `device_status_changed`, `maintenance_scheduled`, `work_order_completed`, `calibration_recorded`, ...

**Standalone usage:**
```bash
pip install apg-healthcare-dev
apg-healthcare-dev --port 8080
```

---

### Electronic Medical Records `healthcare_emr`

> Full-featured EMR capability providing patient chart management, SOAP and structured clinical note authoring, problem list maintenance with ICD-10 coding, medication reconciliation with allergy-check enforcement, vital signs recording, and HL7 FHIR R4 export. Designed for HIPAA compliance with cross-tenant PHI isolation enforced at the rule layer.

**Package**: `apg-healthcare-emr`  
**Path**: `capabilities/healthcare/emr`  
**Version**: 1.0.0  

**Provides:**
- `patient_chart_management`
- `clinical_note_authoring`
- `problem_list_management`
- `medication_reconciliation`
- `allergy_tracking`
- `vital_signs_recording`
- `fhir_r4_export`
- `icd10_coding`
- `encounter_management`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `nlpc`
- `wflo`
- `mqeb`

**Service methods** (112 total):
`check_permission`, `record`, `send`, `get`, `put`, `list`, `delete`, `get_auth_adapter`, `get_audit_adapter`, `get_notify_adapter`, `get_store`, `describe`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_record_access_denied`, `note_type_supported`, `note_amendment_requires_original`, `problem_requires_icd10`, `problem_status_supported`, `medication_allergy_check_required`, ...

**UI Routes** (14):
- `/healthcare-emr/dashboard` — dashboard (healthcare_emr:view)
- `/healthcare-emr/chart/<patient_id>` — chart (healthcare_emr:chart)
- `/healthcare-emr/notes` — notes (healthcare_emr:notes)
- `/healthcare-emr/notes/new` — note_new (healthcare_emr:notes_write)
- `/healthcare-emr/notes/<id>` — note_detail (healthcare_emr:notes)
- `/healthcare-emr/problems/<patient_id>` — problems (healthcare_emr:problems)
- _8 more..._

**Streaming events** via `bytewax`:
`note_created`, `note_amended`, `problem_added`, `problem_resolved`, `medication_prescribed`, ...

**Standalone usage:**
```bash
pip install apg-healthcare-emr
apg-healthcare-emr --port 8080
```

---

### Laboratory Information System `healthcare_lab`

> Full-featured LIS capability providing lab order management, specimen tracking with chain of custody, result entry and verification, critical value alerting with mandatory acknowledgement, QC management with Westgard rule evaluation, and instrument status tracking. Critical value workflow blocks result release until notification is confirmed.

**Package**: `apg-healthcare-lab`  
**Path**: `capabilities/healthcare/lab`  
**Version**: 1.0.0  

**Provides:**
- `lab_order_management`
- `specimen_tracking`
- `result_entry_verification`
- `critical_value_alerting`
- `qc_management`
- `instrument_management`
- `lis_integration`
- `reference_range_evaluation`
- `lab_reporting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `mqeb`

**Service methods** (77 total):
`describe`, `evaluate`, `create_order`, `receive_lab_order`, `cancel_order`, `get_order`, `list_orders`, `collect_specimen`, `label_specimen`, `track_specimen_chain_of_custody`, `reject_specimen`, `receive_specimen`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_result_access_denied`, `order_status_supported`, `specimen_type_supported`, `specimen_rejection_reason_required`, `rejection_reason_supported`, `result_status_supported`, ...

**UI Routes** (13):
- `/healthcare-lab/dashboard` — dashboard (healthcare_lab:view)
- `/healthcare-lab/orders` — orders (healthcare_lab:orders)
- `/healthcare-lab/orders/new` — order_new (healthcare_lab:orders_write)
- `/healthcare-lab/orders/<id>` — order_detail (healthcare_lab:orders)
- `/healthcare-lab/specimens` — specimens (healthcare_lab:specimens)
- `/healthcare-lab/specimens/<id>` — specimen_detail (healthcare_lab:specimens)
- _7 more..._

**Streaming events** via `bytewax`:
`order_created`, `order_cancelled`, `specimen_collected`, `specimen_rejected`, `result_entered`, ...

**Standalone usage:**
```bash
pip install apg-healthcare-lab
apg-healthcare-lab --port 8080
```

---

### Pharmacy Management `healthcare_pha`

> Full-featured pharmacy management capability covering drug formulary management, prescription dispensing with pharmacist verification, LASA (look-alike/sound-alike) alert tracking, controlled substance logging with dual-witness enforcement, drug-drug interaction checking, inventory management with expiry tracking, and prior authorization workflows.

**Package**: `apg-healthcare-pha`  
**Path**: `capabilities/healthcare/pha`  
**Version**: 1.0.0  

**Provides:**
- `drug_formulary_management`
- `prescription_dispensing`
- `lasa_alert_management`
- `controlled_substance_tracking`
- `drug_interaction_checking`
- `pharmacy_inventory_management`
- `prior_authorization_workflow`
- `medication_adherence_tracking`
- `pharmacist_verification_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `moni`
- `mqeb`

**Service methods** (88 total):
`describe`, `evaluate`, `add_drug_to_formulary`, `get_drug`, `mark_drug_lasa`, `update_formulary_status`, `formulary_review`, `verify_prescription`, `check_drug_interactions_at_dispense`, `dispense_medication`, `create_dispense_order`, `verify_dispense`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_dispense_denied`, `contraindicated_dispense_denied`, `pharmacist_verification_required`, `recalled_drug_dispense_denied`, `expired_drug_dispense_denied`, `out_of_stock_dispense_denied`, ...

**UI Routes** (13):
- `/healthcare-pha/dashboard` — dashboard (healthcare_pha:view)
- `/healthcare-pha/formulary` — formulary (healthcare_pha:formulary)
- `/healthcare-pha/formulary/<id>` — drug_detail (healthcare_pha:formulary)
- `/healthcare-pha/dispense` — dispense_queue (healthcare_pha:dispense)
- `/healthcare-pha/dispense/<id>/verify` — dispense_verify (healthcare_pha:dispense_verify)
- `/healthcare-pha/interactions` — interactions (healthcare_pha:interactions)
- _7 more..._

**Streaming events** via `bytewax`:
`drug_added_to_formulary`, `drug_dispensed`, `dispense_verified`, `drug_interaction_detected`, `lasa_alert_triggered`, ...

**Standalone usage:**
```bash
pip install apg-healthcare-pha
apg-healthcare-pha --port 8080
```

---

### Patient Management `healthcare_pmt`

> Core patient lifecycle management covering registration with MRN generation, ADT (Admit/Discharge/Transfer) workflow enforcement, real-time bed board management, appointment scheduling, and insurance tracking. Enforces physician discharge orders, prevents admission of inactive patients, requires approval for patient merges, and enforces cancellation reason documentation.

**Package**: `apg-healthcare-pmt`  
**Path**: `capabilities/healthcare/pmt`  
**Version**: 1.0.0  

**Provides:**
- `patient_registration`
- `adt_workflow`
- `bed_management`
- `appointment_scheduling`
- `patient_billing`
- `mrn_generation`
- `insurance_verification`
- `patient_search`
- `visit_management`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `schd`
- `mqeb`

**Service methods** (54 total):
`describe`, `register_patient`, `get_patient`, `search_patient`, `search_patients`, `update_patient_status`, `merge_patients`, `admit_patient`, `transfer_patient`, `discharge_patient`, `list_admissions`, `register_bed`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `duplicate_mrn_denied`, `admission_type_supported`, `discharge_requires_physician_order`, `discharge_disposition_supported`, `transfer_requires_receiving_unit`, `bed_status_supported`, ...

**UI Routes** (14):
- `/healthcare-pmt/dashboard` — dashboard (healthcare_pmt:view)
- `/healthcare-pmt/patients` — patients (healthcare_pmt:patients)
- `/healthcare-pmt/patients/register` — patient_register (healthcare_pmt:patients_write)
- `/healthcare-pmt/patients/<id>` — patient_detail (healthcare_pmt:patients)
- `/healthcare-pmt/admissions` — admissions (healthcare_pmt:adt)
- `/healthcare-pmt/discharges` — discharges (healthcare_pmt:adt)
- _8 more..._

**Streaming events** via `bytewax`:
`patient_registered`, `patient_updated`, `patient_merged`, `patient_admitted`, `patient_discharged`, ...

**Standalone usage:**
```bash
pip install apg-healthcare-pmt
apg-healthcare-pmt --port 8080
```

---

### Healthcare Regulatory `healthcare_reg`

> Regulatory compliance management covering facility and professional licensing with expiry tracking, accreditation management (Joint Commission, DNV, CAP, etc.), incident reporting with sentinel event workflow enforcement, regulatory submission management (CMS IQR/OQR, HIPAA breach, FDA MDR), and corrective action tracking. Sentinel event closure requires a completed root cause analysis reference.

**Package**: `apg-healthcare-reg`  
**Path**: `capabilities/healthcare/reg`  
**Version**: 1.0.0  

**Provides:**
- `facility_licensing_management`
- `accreditation_management`
- `incident_reporting`
- `hipaa_compliance_tracking`
- `regulatory_submission_management`
- `audit_management`
- `corrective_action_tracking`
- `compliance_dashboard`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `moni`
- `mqeb`

**Service methods** (40 total):
`describe`, `add_license`, `facility_licence_apply`, `licence_renewal`, `get_license`, `list_licenses`, `get_expiring_licenses`, `add_accreditation`, `accreditation_application`, `update_accreditation_status`, `list_accreditations`, `inspection_schedule`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_regulatory_access_denied`, `license_type_supported`, `accreditation_body_supported`, `accreditation_status_supported`, `incident_type_supported`, `incident_severity_supported`, ...

**UI Routes** (13):
- `/healthcare-reg/dashboard` — dashboard (healthcare_reg:view)
- `/healthcare-reg/licenses` — licenses (healthcare_reg:licenses)
- `/healthcare-reg/licenses/<id>` — license_detail (healthcare_reg:licenses)
- `/healthcare-reg/accreditation` — accreditation (healthcare_reg:accreditation)
- `/healthcare-reg/incidents` — incidents (healthcare_reg:incidents)
- `/healthcare-reg/incidents/new` — incident_new (healthcare_reg:incidents_write)
- _7 more..._

**Streaming events** via `bytewax`:
`license_added`, `license_expiring`, `accreditation_status_changed`, `incident_reported`, `sentinel_event_reported`, ...

**Standalone usage:**
```bash
pip install apg-healthcare-reg
apg-healthcare-reg --port 8080
```

---

### Telemedicine `healthcare_tel`

> Full-featured telemedicine capability covering virtual consultation booking, video session management with consent and E-911 disclosure enforcement, remote patient monitoring enrollment, electronic prescription transmission, and telehealth-specific billing code management. Schedule II/III prescription transmission is blocked without a prior in-person visit.

**Package**: `apg-healthcare-tel`  
**Path**: `capabilities/healthcare/tel`  
**Version**: 1.0.0  

**Provides:**
- `virtual_consultation_booking`
- `video_session_management`
- `remote_patient_monitoring`
- `prescription_transmission`
- `telehealth_billing`
- `patient_consent_management`
- `technical_readiness_check`
- `asynchronous_consultation`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `schd`
- `comp`
- `moni`
- `mqeb`

**Service methods** (40 total):
`describe`, `book_consultation`, `book_teleconsult`, `cancel_consultation`, `get_consultation`, `list_consultations`, `create_session`, `video_session_start`, `video_session_end`, `complete_session`, `get_session`, `list_sessions`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_session_access_denied`, `patient_consent_required`, `e911_disclosure_required`, `consultation_type_supported`, `session_status_supported`, `platform_type_supported`, ...

**UI Routes** (13):
- `/healthcare-tel/dashboard` — dashboard (healthcare_tel:view)
- `/healthcare-tel/schedule` — schedule (healthcare_tel:schedule)
- `/healthcare-tel/schedule/new` — consultation_new (healthcare_tel:schedule_write)
- `/healthcare-tel/schedule/<id>` — consultation_detail (healthcare_tel:schedule)
- `/healthcare-tel/sessions` — sessions (healthcare_tel:sessions)
- `/healthcare-tel/sessions/<id>/room` — session_room (healthcare_tel:sessions)
- _7 more..._

**Streaming events** via `bytewax`:
`consultation_booked`, `consultation_cancelled`, `session_started`, `session_completed`, `session_failed`, ...

**Standalone usage:**
```bash
pip install apg-healthcare-tel
apg-healthcare-tel --port 8080
```

---

## INT

### Integration API Management `int_api`

> Integration API Management (`int_api`) is the central governance layer for the APG platform's API lifecycle. It provides a unified registry and control plane for defining, securing, versioning, and deploying APIs across all integration domains — tracking every API from initial draft through active production deployment and eventual retirement.

**Package**: `apg-int-api`  
**Path**: `capabilities/int/api`  
**Version**: 2.1.0  

**Provides:**
- `api_registry_lifecycle`
- `api_endpoint_lifecycle`
- `api_policy_lifecycle`
- `api_consumer_lifecycle`
- `api_key_lifecycle`
- `api_subscription_lifecycle`
- `api_deployment_workflow`
- `api_gateway_route_catalog`
- `api_analytics_workflow`
- `api_dashboard_service`
- `api_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `grc_pol`
- `service_discovery`

**Service methods** (46 total):
`describe`, `evaluate`, `register_api`, `register_endpoint`, `attach_policy`, `register_consumer`, `issue_api_key`, `create_subscription`, `approve_api`, `deploy_api`, `record_usage`, `register_api_agent`, ...

**Governance rules** (49 total):
`tenant_context_required`, `api_write_requires_policy`, `api_requires_name`, `api_requires_title`, `api_requires_base_path`, `api_base_path_format`, `api_requires_upstream`, `api_requires_owner`, ...

**UI Routes** (11):
- `/int-api/dashboard` — dashboard (int_api:view)
- `/int-api/apis` — apis (int_api:manage_apis)
- `/int-api/endpoints` — endpoints (int_api:manage_endpoints)
- `/int-api/policies` — policies (int_api:manage_policies)
- `/int-api/consumers` — consumers (int_api:manage_consumers)
- `/int-api/keys` — keys (int_api:manage_keys)
- _5 more..._

**Streaming events** via `bytewax`:
`api_registered`, `endpoint_registered`, `policy_attached`, `consumer_registered`, `api_key_issued`, ...

**Standalone usage:**
```bash
pip install apg-int-api
apg-int-api --port 8080
```

---

## INTEL

### Alert Management `intel_alerts`

> `intel_alerts` is an executable APG capability package for building governed alert-management applications. It gives generated APG apps a concrete runtime for lawful authority, alert workspaces, rules, signals, alerts, escalations,

**Package**: `apg-intel-alerts`  
**Path**: `capabilities/intel/alerts`  
**Version**: 1.1.0  

**Provides:**
- `alert_authority_workflow`
- `alert_workspace_workflow`
- `alert_rule_workflow`
- `alert_signal_workflow`
- `alert_record_workflow`
- `alert_escalation_workflow`
- `alert_notification_workflow`
- `alert_assignment_workflow`
- `alert_resolution_workflow`
- `alert_review_workflow`
- `alert_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `geos`

**Service methods** (50 total):
`describe`, `evaluate`, `record_authority`, `record_workspace`, `record_rule`, `record_signal`, `record_alert`, `record_escalation`, `record_notification`, `record_assignment`, `record_resolution`, `record_review`, ...

**Governance rules** (64 total):
`tenant_context_required`, `alert_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-alerts/dashboard` — dashboard (intel_alerts:view)
- `/intel-alerts/authorities` — authorities (intel_alerts:authorities)
- `/intel-alerts/workspaces` — workspaces (intel_alerts:workspaces)
- `/intel-alerts/rules` — rules (intel_alerts:rules)
- `/intel-alerts/signals` — signals (intel_alerts:signals)
- `/intel-alerts/alerts` — alerts (intel_alerts:alerts)
- _7 more..._

**Streaming events** via `bytewax`:
`alert_authority_recorded`, `alert_workspace_recorded`, `alert_rule_recorded`, `alert_signal_recorded`, `alert_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-alerts
apg-intel-alerts --port 8080
```

---

### Intelligence Analytics `intel_analytics`

> `intel_analytics` is an executable APG capability for governed, evidence-backed intelligence analytics. It can be composed into generated APG applications that need threat analytics, fraud analytics, public-safety

**Package**: `apg-intel-analytics`  
**Path**: `capabilities/intel/analytics`  
**Version**: 1.1.0  

**Provides:**
- `analytics_authority_workflow`
- `analytics_workspace_workflow`
- `analytics_dataset_workflow`
- `analytics_feature_workflow`
- `analytics_model_workflow`
- `analytics_run_workflow`
- `analytics_insight_workflow`
- `analytics_dashboard_workflow`
- `analytics_narrative_workflow`
- `analytics_recommendation_workflow`
- `analytics_review_workflow`
- `analytics_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `geos`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_workspace`, `register_dataset`, `record_feature_set`, `record_model`, `record_run`, `record_insight`, `record_dashboard`, `record_narrative`, `record_recommendation`, ...

**Governance rules** (73 total):
`tenant_context_required`, `analytics_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (14):
- `/intel-analytics/dashboard` — dashboard (intel_analytics:view)
- `/intel-analytics/authorities` — authorities (intel_analytics:authorities)
- `/intel-analytics/workspaces` — workspaces (intel_analytics:workspaces)
- `/intel-analytics/datasets` — datasets (intel_analytics:datasets)
- `/intel-analytics/features` — features (intel_analytics:features)
- `/intel-analytics/models` — models (intel_analytics:models)
- _8 more..._

**Streaming events** via `bytewax`:
`analytics_authority_recorded`, `analytics_workspace_recorded`, `analytics_dataset_registered`, `analytics_feature_set_recorded`, `analytics_model_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-analytics
apg-intel-analytics --port 8080
```

---

### Data Correlation `intel_correlation`

> `intel_correlation` is an executable APG capability for governed, evidence-backed cross-source data correlation. It can be composed into generated APG applications that need entity resolution, link analysis, fraud

**Package**: `apg-intel-correlation`  
**Path**: `capabilities/intel/correlation`  
**Version**: 1.1.0  

**Provides:**
- `correlation_authority_workflow`
- `correlation_workspace_workflow`
- `correlation_source_workflow`
- `correlation_entity_workflow`
- `correlation_observation_workflow`
- `correlation_rule_workflow`
- `correlation_run_workflow`
- `correlation_cluster_workflow`
- `correlation_decision_workflow`
- `correlation_referral_workflow`
- `correlation_review_workflow`
- `correlation_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `geos`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_workspace`, `register_source`, `record_entity`, `record_observation`, `record_rule`, `record_run`, `record_cluster`, `record_decision`, `record_referral`, ...

**Governance rules** (71 total):
`tenant_context_required`, `correlation_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (14):
- `/intel-correlation/dashboard` — dashboard (intel_correlation:view)
- `/intel-correlation/authorities` — authorities (intel_correlation:authorities)
- `/intel-correlation/workspaces` — workspaces (intel_correlation:workspaces)
- `/intel-correlation/sources` — sources (intel_correlation:sources)
- `/intel-correlation/entities` — entities (intel_correlation:entities)
- `/intel-correlation/observations` — observations (intel_correlation:observations)
- _8 more..._

**Streaming events** via `bytewax`:
`correlation_authority_recorded`, `correlation_workspace_recorded`, `correlation_source_registered`, `correlation_entity_recorded`, `correlation_observation_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-correlation
apg-intel-correlation --port 8080
```

---

### Intelligence Crawler `intel_crawler`

> `intel_crawler` is the APG capability for composing governed source-collection applications. It wraps source registration, crawl jobs, extraction quality, dataset publication, validation, RAG preparation, graph projection, and crawler-agent review in an executable, dependency-light package surface.

**Package**: `apg-intel-crawler`  
**Path**: `capabilities/intel/crawler`  
**Version**: 1.1.0  

**Provides:**
- `source_intelligence_registry`
- `crawl_job_lifecycle`
- `extraction_pipeline`
- `dataset_quality_control`
- `validation_workflow`
- `rag_graphrag_preparation`
- `crawler_authority_workflow`
- `crawler_governance_workflow`
- `crawler_review_workflow`
- `crawler_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `mten`
- `conf`

**Service methods** (41 total):
`register_source`, `create_crawl_job`, `complete_crawl_job`, `record_extraction`, `open_validation_session`, `complete_validation_session`, `publish_dataset`, `record_rag_plan`, `record_graph_projection`, `register_crawler_agent`, `validate_agent_crawler_action`, `validate_batch_ingest`, ...

**Governance rules** (55 total):
`tenant_context_required`, `crawler_write_requires_policy`, `cross_tenant_crawl_denied`, `privilege_escalation_denied`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, ...

**UI Routes** (12):
- `/intel-crawler/dashboard` — dashboard (intel_crawler:view)
- `/intel-crawler/authorities` — authorities (intel_crawler:authorities)
- `/intel-crawler/sources` — sources (intel_crawler:manage_sources)
- `/intel-crawler/crawl-jobs` — crawl_jobs (intel_crawler:operate)
- `/intel-crawler/extractions` — extractions (intel_crawler:extract)
- `/intel-crawler/datasets` — datasets (intel_crawler:publish)
- _6 more..._

**Streaming events** via `bytewax`:
`crawler_authority_recorded`, `source_registered`, `crawl_job_created`, `crawl_job_completed`, `extraction_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-crawler
apg-intel-crawler --port 8080
```

---

### Cyber Intelligence `intel_cybint`

> `intel_cybint` is the APG package-backed capability for governed defensive cyber-intelligence applications. It composes authorities, indicators, sightings, enrichment, threat profiles, risk assessments, incident links, dissemination,

**Package**: `apg-intel-cybint`  
**Path**: `capabilities/intel/cybint`  
**Version**: 1.1.0  

**Provides:**
- `cybint_authority_workflow`
- `cybint_indicator_workflow`
- `cybint_sighting_workflow`
- `cybint_enrichment_workflow`
- `cybint_threat_profile_workflow`
- `cybint_risk_workflow`
- `cybint_incident_link_workflow`
- `cybint_dissemination_workflow`
- `cybint_review_workflow`
- `cybint_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_indicator`, `record_sighting`, `record_enrichment`, `record_profile`, `record_risk`, `record_incident_link`, `record_dissemination`, `record_review`, `register_cybint_agent`, ...

**Governance rules** (61 total):
`tenant_context_required`, `cybint_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (12):
- `/intel-cybint/dashboard` — dashboard (intel_cybint:view)
- `/intel-cybint/authorities` — authorities (intel_cybint:authorities)
- `/intel-cybint/indicators` — indicators (intel_cybint:indicators)
- `/intel-cybint/sightings` — sightings (intel_cybint:sightings)
- `/intel-cybint/enrichment` — enrichment (intel_cybint:enrichment)
- `/intel-cybint/profiles` — profiles (intel_cybint:profiles)
- _6 more..._

**Streaming events** via `bytewax`:
`cybint_authority_recorded`, `cybint_indicator_recorded`, `cybint_sighting_recorded`, `cybint_enrichment_recorded`, `cybint_profile_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-cybint
apg-intel-cybint --port 8080
```

---

### Dark Web Monitoring `intel_darkweb`

> `intel_darkweb` is an executable APG capability for lawful, defensive dark-web-monitoring workflows. It can be composed into generated APG applications that need exposure monitoring, fraud-market intelligence,

**Package**: `apg-intel-darkweb`  
**Path**: `capabilities/intel/darkweb`  
**Version**: 1.1.0  

**Provides:**
- `darkweb_authority_workflow`
- `darkweb_program_workflow`
- `darkweb_source_workflow`
- `darkweb_observation_workflow`
- `darkweb_indicator_workflow`
- `darkweb_marketplace_risk_workflow`
- `darkweb_threat_actor_workflow`
- `darkweb_referral_workflow`
- `darkweb_dissemination_workflow`
- `darkweb_review_workflow`
- `darkweb_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_program`, `register_source`, `record_observation`, `record_indicator`, `record_marketplace_risk`, `record_threat_actor`, `record_referral`, `record_dissemination`, `record_review`, ...

**Governance rules** (69 total):
`tenant_context_required`, `darkweb_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-darkweb/dashboard` — dashboard (intel_darkweb:view)
- `/intel-darkweb/authorities` — authorities (intel_darkweb:authorities)
- `/intel-darkweb/programs` — programs (intel_darkweb:programs)
- `/intel-darkweb/sources` — sources (intel_darkweb:sources)
- `/intel-darkweb/observations` — observations (intel_darkweb:observations)
- `/intel-darkweb/indicators` — indicators (intel_darkweb:indicators)
- _7 more..._

**Streaming events** via `bytewax`:
`darkweb_authority_recorded`, `darkweb_program_recorded`, `darkweb_source_registered`, `darkweb_observation_recorded`, `darkweb_indicator_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-darkweb
apg-intel-darkweb --port 8080
```

---

### Intelligence Dashboard `intel_dashboard`

> `intel_dashboard` is an executable APG capability package for building governed intelligence-dashboard applications. It gives generated APG apps a concrete runtime for lawful authority, dashboard workspaces, dashboards, data sources,

**Package**: `apg-intel-dashboard`  
**Path**: `capabilities/intel/dashboard`  
**Version**: 1.1.0  

**Provides:**
- `dashboard_authority_workflow`
- `dashboard_workspace_workflow`
- `dashboard_composition_workflow`
- `dashboard_source_workflow`
- `dashboard_metric_workflow`
- `dashboard_widget_workflow`
- `dashboard_filter_workflow`
- `dashboard_view_workflow`
- `dashboard_share_workflow`
- `dashboard_review_workflow`
- `dashboard_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `geos`

**Service methods** (44 total):
`describe`, `evaluate`, `record_authority`, `record_workspace`, `record_dashboard`, `record_source`, `record_metric`, `record_widget`, `record_filter`, `record_view`, `record_share`, `record_review`, ...

**Governance rules** (63 total):
`tenant_context_required`, `dashboard_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-dashboard/dashboard` — dashboard (intel_dashboard:view)
- `/intel-dashboard/authorities` — authorities (intel_dashboard:authorities)
- `/intel-dashboard/workspaces` — workspaces (intel_dashboard:workspaces)
- `/intel-dashboard/dashboards` — dashboards (intel_dashboard:dashboards)
- `/intel-dashboard/sources` — sources (intel_dashboard:sources)
- `/intel-dashboard/metrics` — metrics (intel_dashboard:metrics)
- _7 more..._

**Streaming events** via `bytewax`:
`dashboard_authority_recorded`, `dashboard_workspace_recorded`, `dashboard_recorded`, `dashboard_source_recorded`, `dashboard_metric_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-dashboard
apg-intel-dashboard --port 8080
```

---

### Financial Intelligence `intel_finint`

> `intel_finint` is the APG package-backed capability for governed financial-intelligence applications. It composes authorities, financial sources, subjects, transactions, patterns, risk assessments, referrals, dissemination,

**Package**: `apg-intel-finint`  
**Path**: `capabilities/intel/finint`  
**Version**: 1.1.0  

**Provides:**
- `finint_authority_workflow`
- `finint_source_workflow`
- `finint_subject_workflow`
- `finint_transaction_workflow`
- `finint_pattern_workflow`
- `finint_risk_workflow`
- `finint_referral_workflow`
- `finint_dissemination_workflow`
- `finint_review_workflow`
- `finint_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `fintech_kyc`
- `fintech_aml`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `register_source`, `record_subject`, `record_transaction`, `record_pattern`, `record_risk`, `record_referral`, `record_dissemination`, `record_review`, `register_finint_agent`, ...

**Governance rules** (62 total):
`tenant_context_required`, `finint_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (12):
- `/intel-finint/dashboard` — dashboard (intel_finint:view)
- `/intel-finint/authorities` — authorities (intel_finint:authorities)
- `/intel-finint/sources` — sources (intel_finint:sources)
- `/intel-finint/subjects` — subjects (intel_finint:subjects)
- `/intel-finint/transactions` — transactions (intel_finint:transactions)
- `/intel-finint/patterns` — patterns (intel_finint:patterns)
- _6 more..._

**Streaming events** via `bytewax`:
`finint_authority_recorded`, `finint_source_registered`, `finint_subject_recorded`, `finint_transaction_recorded`, `finint_pattern_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-finint
apg-intel-finint --port 8080
```

---

### Intelligence Fusion `intel_fusion`

> `intel_fusion` is an executable APG capability for lawful, evidence-led intelligence fusion. It can be composed into generated APG applications that need cross-source operational pictures, threat fusion, fraud fusion,

**Package**: `apg-intel-fusion`  
**Path**: `capabilities/intel/fusion`  
**Version**: 1.1.0  

**Provides:**
- `fusion_authority_workflow`
- `fusion_workspace_workflow`
- `fusion_source_workflow`
- `fusion_artifact_workflow`
- `fusion_correlation_workflow`
- `fusion_hypothesis_workflow`
- `fusion_assessment_workflow`
- `fusion_referral_workflow`
- `fusion_dissemination_workflow`
- `fusion_review_workflow`
- `fusion_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `geos`

**Service methods** (96 total):
`create_intel_item`, `get_intel_item`, `list_intel_items`, `update_intel_item`, `delete_intel_item`, `validate_intel_item`, `reject_intel_item`, `_set_item_status`, `create_workspace`, `get_workspace`, `list_workspaces`, `update_workspace`, ...

**Governance rules** (67 total):
`tenant_context_required`, `fusion_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-fusion/dashboard` — dashboard (intel_fusion:view)
- `/intel-fusion/authorities` — authorities (intel_fusion:authorities)
- `/intel-fusion/workspaces` — workspaces (intel_fusion:workspaces)
- `/intel-fusion/sources` — sources (intel_fusion:sources)
- `/intel-fusion/artifacts` — artifacts (intel_fusion:artifacts)
- `/intel-fusion/correlations` — correlations (intel_fusion:correlations)
- _7 more..._

**Streaming events** via `bytewax`:
`fusion_authority_recorded`, `fusion_workspace_recorded`, `fusion_source_registered`, `fusion_artifact_recorded`, `fusion_correlation_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-fusion
apg-intel-fusion --port 8080
```

---

### Geospatial Intelligence `intel_geoint`

> `intel_geoint` is the APG package-backed capability for governed geospatial intelligence applications. It composes authorities, areas of interest, imagery/geospatial sources, collection plans, observations, features, change

**Package**: `apg-intel-geoint`  
**Path**: `capabilities/intel/geoint`  
**Version**: 1.1.0  

**Provides:**
- `geoint_authority_workflow`
- `geoint_area_workflow`
- `geoint_source_workflow`
- `geoint_collection_workflow`
- `geoint_observation_workflow`
- `geoint_feature_workflow`
- `geoint_change_workflow`
- `geoint_assessment_workflow`
- `geoint_dissemination_workflow`
- `geoint_review_workflow`
- `geoint_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `geos`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_area`, `register_source`, `record_collection_plan`, `record_observation`, `record_feature`, `record_change`, `record_assessment`, `record_dissemination`, `record_review`, ...

**Governance rules** (70 total):
`tenant_context_required`, `geoint_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-geoint/dashboard` — dashboard (intel_geoint:view)
- `/intel-geoint/authorities` — authorities (intel_geoint:authorities)
- `/intel-geoint/areas` — areas (intel_geoint:areas)
- `/intel-geoint/sources` — sources (intel_geoint:sources)
- `/intel-geoint/collection-plans` — collection_plans (intel_geoint:collection)
- `/intel-geoint/observations` — observations (intel_geoint:observations)
- _7 more..._

**Streaming events** via `bytewax`:
`geoint_authority_recorded`, `geoint_area_recorded`, `geoint_source_registered`, `geoint_collection_plan_recorded`, `geoint_observation_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-geoint
apg-intel-geoint --port 8080
```

---

### Human Intelligence `intel_humint`

> `intel_humint` is the APG package-backed capability for governed human-intelligence applications. It composes authorities, human sources, contact plans, contact reports, debriefings, reliability assessments, leads,

**Package**: `apg-intel-humint`  
**Path**: `capabilities/intel/humint`  
**Version**: 1.1.0  

**Provides:**
- `humint_authority_workflow`
- `humint_source_workflow`
- `humint_contact_plan_workflow`
- `humint_contact_report_workflow`
- `humint_debriefing_workflow`
- `humint_reliability_workflow`
- `humint_lead_workflow`
- `humint_dissemination_workflow`
- `humint_review_workflow`
- `humint_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `register_source`, `record_contact_plan`, `record_contact_report`, `record_debriefing`, `record_reliability`, `record_lead`, `record_dissemination`, `record_review`, `register_humint_agent`, ...

**Governance rules** (63 total):
`tenant_context_required`, `humint_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (12):
- `/intel-humint/dashboard` — dashboard (intel_humint:view)
- `/intel-humint/authorities` — authorities (intel_humint:authorities)
- `/intel-humint/sources` — sources (intel_humint:sources)
- `/intel-humint/contact-plans` — contact_plans (intel_humint:contacts)
- `/intel-humint/contact-reports` — contact_reports (intel_humint:reports)
- `/intel-humint/debriefings` — debriefings (intel_humint:analysis)
- _6 more..._

**Streaming events** via `bytewax`:
`humint_authority_recorded`, `humint_source_registered`, `humint_contact_plan_recorded`, `humint_contact_report_recorded`, `humint_debriefing_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-humint
apg-intel-humint --port 8080
```

---

### Real-Time Monitoring `intel_monitoring`

> `intel_monitoring` is an executable APG capability for lawful, defensive real-time monitoring workflows. It can be composed into generated APG applications that need security monitoring, fraud monitoring, public-safety

**Package**: `apg-intel-monitoring`  
**Path**: `capabilities/intel/monitoring`  
**Version**: 1.1.0  

**Provides:**
- `monitoring_authority_workflow`
- `monitoring_policy_workflow`
- `monitoring_source_workflow`
- `monitoring_watch_workflow`
- `monitoring_event_workflow`
- `monitoring_signal_workflow`
- `monitoring_incident_workflow`
- `monitoring_referral_workflow`
- `monitoring_dissemination_workflow`
- `monitoring_review_workflow`
- `monitoring_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_policy`, `register_source`, `record_watch`, `record_event`, `record_signal`, `record_incident`, `record_referral`, `record_dissemination`, `record_review`, ...

**Governance rules** (69 total):
`tenant_context_required`, `monitoring_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-monitoring/dashboard` — dashboard (intel_monitoring:view)
- `/intel-monitoring/authorities` — authorities (intel_monitoring:authorities)
- `/intel-monitoring/policies` — policies (intel_monitoring:policies)
- `/intel-monitoring/sources` — sources (intel_monitoring:sources)
- `/intel-monitoring/watches` — watches (intel_monitoring:watches)
- `/intel-monitoring/events` — events (intel_monitoring:events)
- _7 more..._

**Streaming events** via `bytewax`:
`monitoring_authority_recorded`, `monitoring_policy_recorded`, `monitoring_source_registered`, `monitoring_watch_recorded`, `monitoring_event_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-monitoring
apg-intel-monitoring --port 8080
```

---

### Open Source Intelligence `intel_osint`

> `intel_osint` is the APG package-backed capability for governed open-source intelligence applications. It composes requirements, sources, collection plans, evidence, triage, assessments, dissemination, reviews, Bytewax lifecycle

**Package**: `apg-intel-osint`  
**Path**: `capabilities/intel/osint`  
**Version**: 2.0.0  

**Provides:**
- `osint_source_workflow`
- `osint_collection_task_workflow`
- `osint_raw_intel_workflow`
- `osint_processed_intel_workflow`
- `osint_entity_workflow`
- `osint_relationship_workflow`
- `osint_social_profile_workflow`
- `osint_web_content_workflow`
- `osint_domain_intel_workflow`
- `osint_ip_intel_workflow`
- `osint_document_analysis_workflow`
- `osint_dissemination_workflow`
- `osint_review_workflow`
- `osint_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `intel_crawler`
- `srch`
- `grph`
- `ragn`
- `geoi`

**Service methods** (73 total):
`describe`, `evaluate_rules`, `register_source`, `update_source`, `get_source`, `list_sources`, `delete_source`, `create_task`, `start_task`, `complete_task`, `fail_task`, `cancel_task`, ...

**Governance rules** (49 total):
`tenant_context_required`, `osint_write_requires_policy`, `cross_tenant_osint_write_denied`, `source_type_supported`, `source_name_required`, `source_owner_required`, `source_terms_review_required`, `source_risk_tier_supported`, ...

**UI Routes** (17):
- `/intel-osint/dashboard` — dashboard (intel_osint:view)
- `/intel-osint/sources` — sources (intel_osint:sources)
- `/intel-osint/tasks` — tasks (intel_osint:tasks)
- `/intel-osint/raw-intel` — raw_intel (intel_osint:raw_intel)
- `/intel-osint/triage` — triage (intel_osint:triage)
- `/intel-osint/processed-intel` — processed_intel (intel_osint:processed_intel)
- _11 more..._

**Streaming events** via `bytewax`:
`osint_source_registered`, `osint_source_updated`, `osint_task_created`, `osint_task_status_changed`, `osint_raw_intel_ingested`, ...

**Standalone usage:**
```bash
pip install apg-intel-osint
apg-intel-osint --port 8080
```

---

### Predictive Intelligence `intel_prediction`

> `intel_prediction` is an executable APG capability package for building governed predictive-intelligence applications. It gives generated APG apps a concrete runtime for lawful authority, analytical workspaces, scenarios,

**Package**: `apg-intel-prediction`  
**Path**: `capabilities/intel/prediction`  
**Version**: 1.1.0  

**Provides:**
- `prediction_authority_workflow`
- `prediction_workspace_workflow`
- `prediction_scenario_workflow`
- `prediction_indicator_workflow`
- `prediction_model_workflow`
- `prediction_forecast_workflow`
- `prediction_projection_workflow`
- `prediction_warning_workflow`
- `prediction_recommendation_workflow`
- `prediction_review_workflow`
- `prediction_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `geos`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_workspace`, `record_scenario`, `record_indicator`, `record_model`, `record_forecast`, `record_projection`, `record_warning`, `record_recommendation`, `record_review`, ...

**Governance rules** (68 total):
`tenant_context_required`, `prediction_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-prediction/dashboard` — dashboard (intel_prediction:view)
- `/intel-prediction/authorities` — authorities (intel_prediction:authorities)
- `/intel-prediction/workspaces` — workspaces (intel_prediction:workspaces)
- `/intel-prediction/scenarios` — scenarios (intel_prediction:scenarios)
- `/intel-prediction/indicators` — indicators (intel_prediction:indicators)
- `/intel-prediction/models` — models (intel_prediction:models)
- _7 more..._

**Streaming events** via `bytewax`:
`prediction_authority_recorded`, `prediction_workspace_recorded`, `prediction_scenario_recorded`, `prediction_indicator_recorded`, `prediction_model_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-prediction
apg-intel-prediction --port 8080
```

---

### Radio Intelligence Listener `intel_radio`

> `intel_radio` is an executable APG capability for lawful, passive radio-monitoring workflows. It can be composed into generated APG applications that need public-safety monitoring, spectrum management, interference review,

**Package**: `apg-intel-radio`  
**Path**: `capabilities/intel/radio`  
**Version**: 1.1.0  

**Provides:**
- `radio_authority_workflow`
- `radio_band_plan_workflow`
- `radio_receiver_workflow`
- `radio_collection_session_workflow`
- `radio_observation_workflow`
- `radio_classification_workflow`
- `radio_event_workflow`
- `radio_referral_workflow`
- `radio_dissemination_workflow`
- `radio_review_workflow`
- `radio_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `geos`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_band_plan`, `register_receiver`, `record_session`, `record_observation`, `record_classification`, `record_event`, `record_referral`, `record_dissemination`, `record_review`, ...

**Governance rules** (71 total):
`tenant_context_required`, `radio_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-radio/dashboard` — dashboard (intel_radio:view)
- `/intel-radio/authorities` — authorities (intel_radio:authorities)
- `/intel-radio/band-plans` — band-plans (intel_radio:band_plans)
- `/intel-radio/receivers` — receivers (intel_radio:receivers)
- `/intel-radio/sessions` — sessions (intel_radio:sessions)
- `/intel-radio/observations` — observations (intel_radio:observations)
- _7 more..._

**Streaming events** via `bytewax`:
`radio_authority_recorded`, `radio_band_plan_recorded`, `radio_receiver_registered`, `radio_session_recorded`, `radio_observation_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-radio
apg-intel-radio --port 8080
```

---

### Intelligence Reporting `intel_reporting`

> `intel_reporting` is an executable APG capability package for building governed intelligence-reporting applications. It gives generated APG apps a concrete runtime for lawful authority, reporting workspaces, templates, products,

**Package**: `apg-intel-reporting`  
**Path**: `capabilities/intel/reporting`  
**Version**: 1.1.0  

**Provides:**
- `reporting_authority_workflow`
- `reporting_workspace_workflow`
- `reporting_template_workflow`
- `reporting_product_workflow`
- `reporting_section_workflow`
- `reporting_citation_workflow`
- `reporting_approval_workflow`
- `reporting_distribution_workflow`
- `reporting_publication_workflow`
- `reporting_review_workflow`
- `reporting_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `geos`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_workspace`, `record_template`, `record_product`, `record_section`, `record_citation`, `record_approval`, `record_distribution`, `record_publication`, `record_review`, ...

**Governance rules** (63 total):
`tenant_context_required`, `reporting_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-reporting/dashboard` — dashboard (intel_reporting:view)
- `/intel-reporting/authorities` — authorities (intel_reporting:authorities)
- `/intel-reporting/workspaces` — workspaces (intel_reporting:workspaces)
- `/intel-reporting/templates` — templates (intel_reporting:templates)
- `/intel-reporting/products` — products (intel_reporting:products)
- `/intel-reporting/sections` — sections (intel_reporting:sections)
- _7 more..._

**Streaming events** via `bytewax`:
`reporting_authority_recorded`, `reporting_workspace_recorded`, `reporting_template_recorded`, `reporting_product_recorded`, `reporting_section_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-reporting
apg-intel-reporting --port 8080
```

---

### Signals Intelligence `intel_sigint`

> `intel_sigint` is the APG package-backed capability for governed signals-intelligence applications. It composes authorities, sources, collection tasks, observations, processing batches, patterns, assessments, reviews, Bytewax

**Package**: `apg-intel-sigint`  
**Path**: `capabilities/intel/sigint`  
**Version**: 1.1.0  

**Provides:**
- `sigint_authority_workflow`
- `sigint_source_workflow`
- `sigint_collection_workflow`
- `sigint_observation_workflow`
- `sigint_processing_workflow`
- `sigint_pattern_workflow`
- `sigint_assessment_workflow`
- `sigint_review_workflow`
- `sigint_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `intel_radio`
- `intel_crawler`
- `grph`
- `ragn`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `register_source`, `record_collection_task`, `record_observation`, `record_processing_batch`, `record_pattern`, `record_assessment`, `record_review`, `register_sigint_agent`, `validate_agent_action`, ...

**Governance rules** (55 total):
`tenant_context_required`, `sigint_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (11):
- `/intel-sigint/dashboard` — dashboard (intel_sigint:view)
- `/intel-sigint/authorities` — authorities (intel_sigint:authorities)
- `/intel-sigint/sources` — sources (intel_sigint:sources)
- `/intel-sigint/collection-tasks` — collection_tasks (intel_sigint:collection)
- `/intel-sigint/observations` — observations (intel_sigint:observations)
- `/intel-sigint/processing` — processing (intel_sigint:processing)
- _5 more..._

**Streaming events** via `bytewax`:
`sigint_authority_recorded`, `sigint_source_registered`, `sigint_collection_task_recorded`, `sigint_observation_recorded`, `sigint_processing_batch_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-sigint
apg-intel-sigint --port 8080
```

---

### Social Media Intelligence `intel_socint`

> `intel_socint` is an executable APG capability for lawful public or authorized social-source intelligence. It can be composed into generated APG applications that need social monitoring, public-safety alerting, fraud and disinformation

**Package**: `apg-intel-socint`  
**Path**: `capabilities/intel/socint`  
**Version**: 1.1.0  

**Provides:**
- `socint_authority_workflow`
- `socint_topic_workflow`
- `socint_source_workflow`
- `socint_post_workflow`
- `socint_signal_workflow`
- `socint_influence_workflow`
- `socint_network_workflow`
- `socint_referral_workflow`
- `socint_dissemination_workflow`
- `socint_review_workflow`
- `socint_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_topic`, `register_source`, `record_post`, `record_signal`, `record_influence`, `record_network`, `record_referral`, `record_dissemination`, `record_review`, ...

**Governance rules** (67 total):
`tenant_context_required`, `socint_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-socint/dashboard` — dashboard (intel_socint:view)
- `/intel-socint/authorities` — authorities (intel_socint:authorities)
- `/intel-socint/topics` — topics (intel_socint:topics)
- `/intel-socint/sources` — sources (intel_socint:sources)
- `/intel-socint/posts` — posts (intel_socint:posts)
- `/intel-socint/signals` — signals (intel_socint:signals)
- _7 more..._

**Streaming events** via `bytewax`:
`socint_authority_recorded`, `socint_topic_recorded`, `socint_source_registered`, `socint_post_recorded`, `socint_signal_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-socint
apg-intel-socint --port 8080
```

---

### Digital Surveillance `intel_surveillance`

> `intel_surveillance` is an executable APG capability for lawful, defensive digital-surveillance workflows. It can be composed into generated APG applications that need facility monitoring, endpoint telemetry review,

**Package**: `apg-intel-surveillance`  
**Path**: `capabilities/intel/surveillance`  
**Version**: 1.1.0  

**Provides:**
- `surveillance_authority_workflow`
- `surveillance_program_workflow`
- `surveillance_asset_workflow`
- `surveillance_sensor_workflow`
- `surveillance_observation_workflow`
- `surveillance_alert_workflow`
- `surveillance_risk_workflow`
- `surveillance_referral_workflow`
- `surveillance_dissemination_workflow`
- `surveillance_review_workflow`
- `surveillance_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `cvsn`
- `grph`
- `ragn`
- `geos`

**Service methods** (40 total):
`describe`, `evaluate`, `record_authority`, `record_program`, `record_asset`, `register_sensor`, `record_observation`, `record_alert`, `record_risk`, `record_referral`, `record_dissemination`, `record_review`, ...

**Governance rules** (70 total):
`tenant_context_required`, `surveillance_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-surveillance/dashboard` — dashboard (intel_surveillance:view)
- `/intel-surveillance/authorities` — authorities (intel_surveillance:authorities)
- `/intel-surveillance/programs` — programs (intel_surveillance:programs)
- `/intel-surveillance/assets` — assets (intel_surveillance:assets)
- `/intel-surveillance/sensors` — sensors (intel_surveillance:sensors)
- `/intel-surveillance/observations` — observations (intel_surveillance:observations)
- _7 more..._

**Streaming events** via `bytewax`:
`surveillance_authority_recorded`, `surveillance_program_recorded`, `surveillance_asset_recorded`, `surveillance_sensor_registered`, `surveillance_observation_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-surveillance
apg-intel-surveillance --port 8080
```

---

### Threat Intelligence `intel_threats`

> `intel_threats` is an executable APG capability package for building governed threat-intelligence applications. It gives generated APG apps a concrete runtime for lawful authority, threat workspaces, source lineage, indicators,

**Package**: `apg-intel-threats`  
**Path**: `capabilities/intel/threats`  
**Version**: 1.1.0  

**Provides:**
- `threat_authority_workflow`
- `threat_workspace_workflow`
- `threat_source_workflow`
- `threat_indicator_workflow`
- `threat_actor_workflow`
- `threat_campaign_workflow`
- `threat_assessment_workflow`
- `threat_report_workflow`
- `threat_mitigation_workflow`
- `threat_review_workflow`
- `threat_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`
- `ragn`
- `geos`

**Service methods** (53 total):
`describe`, `evaluate`, `record_authority`, `record_workspace`, `register_source`, `record_indicator`, `record_actor`, `record_campaign`, `record_assessment`, `record_report`, `record_mitigation`, `record_review`, ...

**Governance rules** (65 total):
`tenant_context_required`, `threat_write_requires_policy`, `authority_type_supported`, `authority_scope_required`, `authority_classification_supported`, `authority_approver_required`, `authority_expiry_required`, `authority_evidence_required`, ...

**UI Routes** (13):
- `/intel-threats/dashboard` — dashboard (intel_threats:view)
- `/intel-threats/authorities` — authorities (intel_threats:authorities)
- `/intel-threats/workspaces` — workspaces (intel_threats:workspaces)
- `/intel-threats/sources` — sources (intel_threats:sources)
- `/intel-threats/indicators` — indicators (intel_threats:indicators)
- `/intel-threats/actors` — actors (intel_threats:actors)
- _7 more..._

**Streaming events** via `bytewax`:
`threat_authority_recorded`, `threat_workspace_recorded`, `threat_source_registered`, `threat_indicator_recorded`, `threat_actor_recorded`, ...

**Standalone usage:**
```bash
pip install apg-intel-threats
apg-intel-threats --port 8080
```

---

## LOC

### Multi-Country Operations `loc_mco`

> Multi-Country Operations (MCO) provides country entity management, local regulatory compliance mapping, cross-border intercompany transaction governance, and statutory reporting for organisations operating across multiple jurisdictions. It enforces arms-length transfer pricing, tenant-scoped entity isolation, and audit-trailed compliance workflows across any combination of supported jurisdictions.

**Package**: `apg-loc-mco`  
**Path**: `capabilities/loc/mco`  
**Version**: 1.0.0  

**Provides:**
- `country_entity_management`
- `regulatory_compliance_mapping`
- `intercompany_transaction_workflow`
- `statutory_reporting_workflow`
- `transfer_pricing_validation`
- `cross_border_governance`
- `multi_entity_consolidation_data`
- `jurisdiction_registry`
- `compliance_monitoring`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `moni`
- `mqeb`

**Service methods** (40 total):
`uuid7str`, `uuid7str`, `describe`, `evaluate`, `register_country`, `get_country`, `list_countries`, `update_country`, `register_entity`, `get_entity`, `list_entities`, `update_entity`, ...

**Governance rules** (32 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_entity_denied`, `country_jurisdiction_supported`, `country_currency_supported`, `country_regulatory_framework_required`, `country_name_required`, `entity_type_supported`, ...

**UI Routes** (14):
- `/loc-mco/dashboard` — dashboard (loc_mco:view)
- `/loc-mco/countries` — countries (loc_mco:countries)
- `/loc-mco/countries/create` — countries_create (loc_mco:countries_write)
- `/loc-mco/entities` — entities (loc_mco:entities)
- `/loc-mco/entities/create` — entities_create (loc_mco:entities_write)
- `/loc-mco/compliance` — compliance (loc_mco:compliance)
- _8 more..._

**Streaming events** via `bytewax`:
`country_registered`, `country_updated`, `entity_registered`, `entity_updated`, `compliance_mapping_recorded`, ...

**Standalone usage:**
```bash
pip install apg-loc-mco
apg-loc-mco --port 8080
```

---

### Multi-Currency Management `loc_mcy`

> Multi-Currency Management (MCY) provides full lifecycle management of currencies, exchange rates, FX revaluation, currency translation, and FX gain/loss reporting for organisations operating across multiple currencies. It enforces positive exchange rates, arms-length approval for manual rates, approval-gated revaluation posting, and tenant-scoped isolation of all currency data.

**Package**: `apg-loc-mcy`  
**Path**: `capabilities/loc/mcy`  
**Version**: 1.0.0  

**Provides:**
- `currency_configuration`
- `exchange_rate_management`
- `fx_revaluation_workflow`
- `currency_translation_workflow`
- `fx_gain_loss_reporting`
- `multi_currency_rounding`
- `rate_feed_integration`
- `currency_exposure_dashboard`
- `fx_account_registry`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `schd`
- `mqeb`

**Service methods** (40 total):
`uuid7str`, `uuid7str`, `describe`, `evaluate`, `configure_currency`, `get_currency`, `get_currency_by_code`, `list_currencies`, `update_currency`, `record_exchange_rate`, `get_exchange_rate`, `list_exchange_rates`, ...

**Governance rules** (31 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_rate_denied`, `currency_code_supported`, `currency_name_required`, `currency_precision_valid`, `rounding_mode_supported`, `rate_from_currency_supported`, ...

**UI Routes** (14):
- `/loc-mcy/dashboard` — dashboard (loc_mcy:view)
- `/loc-mcy/currencies` — currencies (loc_mcy:currencies)
- `/loc-mcy/currencies/create` — currencies_create (loc_mcy:currencies_write)
- `/loc-mcy/exchange-rates` — exchange_rates (loc_mcy:exchange_rates)
- `/loc-mcy/exchange-rates/create` — exchange_rates_create (loc_mcy:exchange_rates_write)
- `/loc-mcy/exchange-rates/upload` — exchange_rates_upload (loc_mcy:exchange_rates_write)
- _8 more..._

**Streaming events** via `bytewax`:
`currency_configured`, `currency_updated`, `exchange_rate_recorded`, `exchange_rate_bulk_loaded`, `revaluation_created`, ...

**Standalone usage:**
```bash
pip install apg-loc-mcy
apg-loc-mcy --port 8080
```

---

### Multi-Language & Localisation `loc_mlg`

> Multi-Language & Localisation (MLG) manages translation workflows, locale configuration, RTL language support, date/number formatting rules, content localisation, and terminology management. It enforces reviewer independence, approval-gated publishing, RTL direction consistency, and tenant-scoped translation memory across all supported languages and locales.

**Package**: `apg-loc-mlg`  
**Path**: `capabilities/loc/mlg`  
**Version**: 1.0.0  

**Provides:**
- `locale_configuration`
- `translation_management`
- `rtl_support`
- `date_number_formatting`
- `content_localisation_workflow`
- `locale_registry`
- `terminology_management`
- `translation_memory`
- `locale_aware_rendering`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `nlpc`
- `moni`
- `mqeb`

**Service methods** (40 total):
`uuid7str`, `uuid7str`, `describe`, `evaluate`, `configure_locale`, `get_locale`, `get_locale_by_code`, `list_locales`, `update_locale`, `get_default_locale`, `create_translation`, `get_translation`, ...

**Governance rules** (26 total):
`tenant_context_required`, `write_requires_policy`, `cross_tenant_translation_denied`, `locale_code_supported`, `locale_language_supported`, `locale_script_supported`, `locale_direction_supported`, `locale_date_format_supported`, ...

**UI Routes** (13):
- `/loc-mlg/dashboard` — dashboard (loc_mlg:view)
- `/loc-mlg/locales` — locales (loc_mlg:locales)
- `/loc-mlg/locales/create` — locales_create (loc_mlg:locales_write)
- `/loc-mlg/translations` — translations (loc_mlg:translations)
- `/loc-mlg/translations/create` — translations_create (loc_mlg:translations_write)
- `/loc-mlg/translations/review` — translation_review (loc_mlg:translations_review)
- _7 more..._

**Streaming events** via `bytewax`:
`locale_configured`, `locale_updated`, `translation_created`, `translation_submitted_for_review`, `translation_approved`, ...

**Standalone usage:**
```bash
pip install apg-loc-mlg
apg-loc-mlg --port 8080
```

---

## MINING

### Environmental & Rehabilitation `mining_env`

> Manages mine environmental obligations including environmental monitoring data collection and exceedance detection, tailings storage facility management, progressive rehabilitation tracking, closure planning, rehabilitation bond management, environmental permit registers, ESG data reporting across GRI/SASB/TCFD/ICMM frameworks, and waste stream management. Enforces regulatory requirements including automatic exceedance notification triggers, regulatory body notification for significant exceedances, annual tailings reviews, and approval gating for bond reductions and closure plans.

**Package**: `apg-mining-env`  
**Path**: `capabilities/mining/env`  
**Version**: 1.0.0  

**Provides:**
- `environmental_monitoring_management`
- `tailings_facility_management`
- `rehabilitation_bond_management`
- `closure_plan_workflow`
- `progressive_rehabilitation_tracking`
- `esg_reporting_workflow`
- `environmental_permit_management`
- `exceedance_notification_workflow`
- `waste_stream_management`
- `environmental_compliance_register`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `moni`
- `geos`
- `mqeb`

**Service methods** (42 total):
`record_monitoring_data`, `get_monitoring_data`, `list_monitoring_data`, `record_exceedance`, `send_regulatory_notification`, `close_exceedance`, `list_exceedances`, `register_tailings_facility`, `get_tailings_facility`, `record_tailings_annual_review`, `record_stability_assessment`, `list_tailings_facilities`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `monitoring_type_supported`, `exceedance_notification_required`, `exceedance_regulatory_notification`, `tailings_type_supported`, `tailings_responsible_engineer_required`, `tailings_annual_review_required`, ...

**UI Routes** (14):
- `/mining-env/dashboard` — dashboard (mining_env:view)
- `/mining-env/monitoring` — monitoring (mining_env:view)
- `/mining-env/monitoring/record` — monitoring_record (mining_env:write)
- `/mining-env/exceedances` — exceedances (mining_env:view)
- `/mining-env/tailings` — tailings (mining_env:tailings)
- `/mining-env/tailings/:id` — tailings_detail (mining_env:tailings)
- _8 more..._

**Streaming events** via `bytewax`:
`monitoring_data_recorded`, `exceedance_detected`, `exceedance_regulatory_notification_sent`, `tailings_facility_inspection_completed`, `tailings_stability_assessment_completed`, ...

**Standalone usage:**
```bash
pip install apg-mining-env
apg-mining-env --port 8080
```

---

### Equipment & Plant Management `mining_eqp`

> Manages the full lifecycle of mining fleet and processing plant equipment including registration, dispatch, maintenance work orders, preventive maintenance scheduling, pre-shift inspections, fuel consumption tracking, fault reporting, and fleet KPI reporting. Enforces equipment availability guardrails: breakdown equipment cannot be dispatched, operators must hold valid licences, and pre-shift inspections must pass before daily dispatch.

**Package**: `apg-mining-eqp`  
**Path**: `capabilities/mining/eqp`  
**Version**: 1.0.0  

**Provides:**
- `fleet_register_management`
- `equipment_lifecycle_tracking`
- `maintenance_work_order_workflow`
- `preventive_maintenance_scheduling`
- `equipment_dispatch_management`
- `fuel_consumption_tracking`
- `equipment_kpi_reporting`
- `pre_shift_inspection_workflow`
- `fault_and_defect_management`
- `tyre_management`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `schd`
- `mqeb`

**Service methods** (42 total):
`register_equipment`, `get_equipment`, `get_equipment_by_asset_number`, `update_equipment`, `decommission_equipment`, `list_equipment`, `dispatch_equipment`, `create_work_order`, `approve_work_order`, `complete_work_order`, `list_work_orders`, `submit_inspection`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `equipment_class_supported`, `asset_number_unique`, `ownership_type_supported`, `maintenance_type_supported`, `work_order_approval_required`, `breakdown_dispatch_denied`, ...

**UI Routes** (14):
- `/mining-eqp/dashboard` — dashboard (mining_eqp:view)
- `/mining-eqp/fleet` — fleet (mining_eqp:view)
- `/mining-eqp/fleet/create` — equipment_create (mining_eqp:write)
- `/mining-eqp/fleet/:id` — equipment_detail (mining_eqp:view)
- `/mining-eqp/maintenance` — maintenance (mining_eqp:view)
- `/mining-eqp/maintenance/create` — maintenance_create (mining_eqp:maintenance)
- _8 more..._

**Streaming events** via `bytewax`:
`equipment_commissioned`, `equipment_decommissioned`, `work_order_created`, `work_order_completed`, `equipment_breakdown_recorded`, ...

**Standalone usage:**
```bash
pip install apg-mining-eqp
apg-mining-eqp --port 8080
```

---

### Exploration Data Management `mining_exp`

> Manages the full lifecycle of mineral exploration data from drill-hole collar logging through downhole surveys, geological interval logging, geochemical assay management, QAQC monitoring, resource estimation workflows, and JORC/NI 43-101/SAMREC compliance reporting. Enforces data integrity rules including interval non-overlap, competent person requirements, and QAQC insertion obligations before any resource can be published.

**Package**: `apg-mining-exp`  
**Path**: `capabilities/mining/exp`  
**Version**: 1.0.0  

**Provides:**
- `drillhole_collar_management`
- `downhole_survey_management`
- `lithology_logging`
- `assay_data_management`
- `qaqc_monitoring`
- `resource_estimation_workflow`
- `jorc_reporting_workflow`
- `ni_43_101_reporting_workflow`
- `geological_map_management`
- `exploration_target_delineation`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `nlpc`
- `geos`
- `srch`
- `mqeb`

**Service methods** (41 total):
`create_drillhole_collar`, `get_drillhole_collar`, `get_drillhole_collar_by_hole_id`, `list_drillhole_collars`, `update_drillhole_actual_depth`, `import_assay_results`, `_check_assay_interval_overlap`, `get_assay_results_for_hole`, `flag_qaqc_result`, `list_assays`, `log_geology_interval`, `get_geology_for_hole`, ...

**Governance rules** (23 total):
`tenant_context_required`, `write_requires_policy`, `hole_type_supported`, `collar_coordinates_required`, `collar_coordinate_system_required`, `drillhole_id_unique`, `assay_requires_collar`, `assay_from_to_required`, ...

**UI Routes** (14):
- `/mining-exp/dashboard` — dashboard (mining_exp:view)
- `/mining-exp/drillholes` — drillholes (mining_exp:view)
- `/mining-exp/drillholes/create` — drillhole_create (mining_exp:write)
- `/mining-exp/drillholes/:id` — drillhole_detail (mining_exp:view)
- `/mining-exp/assays` — assays (mining_exp:view)
- `/mining-exp/assays/import` — assay_import (mining_exp:write)
- _8 more..._

**Streaming events** via `bytewax`:
`drillhole_collar_recorded`, `downhole_survey_recorded`, `lithology_interval_logged`, `assay_result_imported`, `qaqc_flag_raised`, ...

**Standalone usage:**
```bash
pip install apg-mining-exp
apg-mining-exp --port 8080
```

---

### Ore Processing & Metallurgy `mining_ore`

> Manages ore processing plant operations including plant feed tracking, process circuit status monitoring, reagent inventory management, metallurgical mass balance preparation and approval, product quality assurance, ore reconciliation, and process deviation alert management. Enforces metallurgical integrity constraints including recovery bounds [0, 100%], cyanide code compliance, approval gating before balance publication, and off-specification product dispatch controls.

**Package**: `apg-mining-ore`  
**Path**: `capabilities/mining/ore`  
**Version**: 1.0.0  

**Provides:**
- `plant_feed_tracking`
- `metallurgical_balance_workflow`
- `reagent_management`
- `recovery_optimisation_tracking`
- `product_quality_management`
- `process_circuit_monitoring`
- `ore_reconciliation_workflow`
- `deviation_alert_management`
- `assay_database_management`
- `process_kpi_reporting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `mqeb`

**Service methods** (42 total):
`record_plant_feed`, `get_plant_feed`, `list_plant_feeds`, `get_feed_summary`, `update_circuit_status`, `get_current_circuit_statuses`, `record_reagent_usage`, `add_reagent_stock`, `get_reagent_inventory`, `list_reagent_usage`, `submit_metallurgical_balance`, `approve_metallurgical_balance`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `feed_source_supported`, `feed_grade_required`, `feed_tonnage_required`, `circuit_type_supported`, `reagent_type_supported`, `reagent_dosage_required`, ...

**UI Routes** (14):
- `/mining-ore/dashboard` — dashboard (mining_ore:view)
- `/mining-ore/plant-feed` — plant_feed (mining_ore:view)
- `/mining-ore/plant-feed/record` — plant_feed_record (mining_ore:write)
- `/mining-ore/circuits` — circuits (mining_ore:view)
- `/mining-ore/circuits/:id` — circuit_detail (mining_ore:view)
- `/mining-ore/reagents` — reagents (mining_ore:view)
- _8 more..._

**Streaming events** via `bytewax`:
`plant_feed_recorded`, `circuit_status_changed`, `reagent_usage_recorded`, `reagent_reorder_triggered`, `metallurgical_balance_submitted`, ...

**Standalone usage:**
```bash
pip install apg-mining-ore
apg-mining-ore --port 8080
```

---

### Mine Production Operations `mining_pro`

> Manages daily mine production operations including shift reporting, ore and waste movement tracking, blast design and firing authorisation, grade control boundary management, stockpile inventory, and production scheduling. Enforces a strict blast status state machine, requires fire authority before detonation, and gates grade boundary changes behind approval workflows to prevent unauthorised ore/waste misclassification.

**Package**: `apg-mining-pro`  
**Path**: `capabilities/mining/pro`  
**Version**: 1.0.0  

**Provides:**
- `shift_report_workflow`
- `production_ledger_management`
- `blast_design_workflow`
- `blast_firing_authorization`
- `ore_tracking_management`
- `grade_control_workflow`
- `production_scheduling`
- `stockpile_inventory_management`
- `delay_recording`
- `production_kpi_reporting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `schd`
- `mqeb`

**Service methods** (42 total):
`create_shift_report`, `get_shift_report`, `update_shift_report`, `submit_shift_report`, `approve_shift_report`, `list_shift_reports`, `create_blast`, `get_blast`, `update_blast`, `approve_blast_design`, `fire_blast`, `list_blasts`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `shift_type_supported`, `shift_supervisor_required`, `shift_dates_required`, `production_area_supported`, `material_type_supported`, `ore_tracking_method_required`, ...

**UI Routes** (14):
- `/mining-pro/dashboard` — dashboard (mining_pro:view)
- `/mining-pro/shifts` — shift_reports (mining_pro:view)
- `/mining-pro/shifts/create` — shift_create (mining_pro:write)
- `/mining-pro/shifts/:id` — shift_detail (mining_pro:view)
- `/mining-pro/production` — production_ledger (mining_pro:view)
- `/mining-pro/ore-tracking` — ore_tracking (mining_pro:write)
- _8 more..._

**Streaming events** via `bytewax`:
`shift_report_submitted`, `shift_report_approved`, `production_tonnes_recorded`, `ore_movement_recorded`, `blast_designed`, ...

**Standalone usage:**
```bash
pip install apg-mining-pro
apg-mining-pro --port 8080
```

---

### Mine Safety & Compliance `mining_saf`

> Manages mine safety operations including incident reporting and investigation, hazard identification and risk assessment, risk register maintenance, permit-to-work issuance, corrective action tracking, compliance obligation registers, safety audits, and safety statistics reporting. Enforces statutory requirements including mandatory investigation before closing LTI and above incidents, stop-work authority for extreme risks, and issuer qualification checks for high-risk permits.

**Package**: `apg-mining-saf`  
**Path**: `capabilities/mining/saf`  
**Version**: 1.0.0  

**Provides:**
- `incident_reporting_workflow`
- `hazard_identification_workflow`
- `risk_register_management`
- `permit_to_work_workflow`
- `corrective_action_tracking`
- `compliance_register_management`
- `safety_audit_workflow`
- `emergency_drill_management`
- `safety_statistics_reporting`
- `stop_work_authority_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `moni`
- `mqeb`

**Service methods** (42 total):
`report_incident`, `get_incident`, `send_regulatory_notification`, `open_investigation`, `close_incident`, `list_incidents`, `identify_hazard`, `get_hazard`, `close_hazard`, `list_hazards`, `add_risk_register_entry`, `get_risk_register_entry`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `incident_type_supported`, `incident_location_required`, `incident_immediate_notification`, `lti_investigation_required`, `hazard_category_supported`, `hazard_risk_assessment_required`, ...

**UI Routes** (14):
- `/mining-saf/dashboard` — dashboard (mining_saf:view)
- `/mining-saf/incidents` — incidents (mining_saf:view)
- `/mining-saf/incidents/create` — incident_create (mining_saf:write)
- `/mining-saf/incidents/:id` — incident_detail (mining_saf:view)
- `/mining-saf/hazards` — hazards (mining_saf:view)
- `/mining-saf/hazards/create` — hazard_create (mining_saf:write)
- _8 more..._

**Streaming events** via `bytewax`:
`incident_reported`, `incident_escalated`, `incident_investigation_opened`, `incident_closed`, `hazard_identified`, ...

**Standalone usage:**
```bash
pip install apg-mining-saf
apg-mining-saf --port 8080
```

---

## MOB

### Mobile App Platform `mob_map`

> The Mobile App Platform (MAP) capability provides a complete cross-platform mobile application lifecycle management runtime. It covers app registration across iOS, Android, PWA and desktop targets; offline data sync with configurable conflict resolution; push notification dispatch via APNS/FCM/Web; biometric authentication enrollment and revocation; app version publishing with phased rollouts and rollbacks; granular permission scope governance; and an analytics event pipeline — all governed by tenant-scoped deterministic policy rules.

**Package**: `apg-mob-map`  
**Path**: `capabilities/mob/map`  
**Version**: 1.0.0  

**Provides:**
- `mobile_app_registry`
- `cross_platform_build_workflow`
- `offline_sync_workflow`
- `push_notification_dispatch`
- `biometric_auth_enrollment`
- `app_version_management`
- `phased_rollout_workflow`
- `permission_scope_governance`
- `app_analytics_pipeline`
- `sync_conflict_resolution`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `moni`
- `mqeb`
- `mob_mdm`

**Service methods** (40 total):
`uuid7str`, `describe`, `evaluate`, `register_app`, `get_app`, `list_apps`, `update_app`, `retire_app`, `publish_version`, `deploy_version`, `rollback_version`, `list_versions`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `platform_must_be_supported`, `app_category_must_be_supported`, `deployment_requires_approval`, `sync_encryption_mandatory`, `sync_strategy_must_be_supported`, `offline_mode_must_be_supported`, ...

**UI Routes** (13):
- `/mob-map/dashboard` — dashboard (mob_map:view)
- `/mob-map/apps` — apps (mob_map:apps:list)
- `/mob-map/apps/<app_id>` — app_detail (mob_map:apps:view)
- `/mob-map/versions` — versions (mob_map:versions:list)
- `/mob-map/versions/<version_id>/deploy` — version_deploy (mob_map:versions:deploy)
- `/mob-map/sync` — sync_sessions (mob_map:sync:list)
- _7 more..._

**Streaming events** via `bytewax`:
`app_registered`, `app_state_changed`, `app_version_published`, `app_version_deployed`, `sync_session_started`, ...

**Standalone usage:**
```bash
pip install apg-mob-map
apg-mob-map --port 8080
```

---

### Mobile Device Management `mob_mdm`

> The Mobile Device Management (MDM) capability provides an enterprise-grade device lifecycle management runtime. It covers device enrolment across multiple platforms and methods; deterministic policy creation, activation, and assignment; continuous compliance evaluation with automatic alert generation; silent app distribution; remote wipe with mandatory dual approval; MDM configuration profile deployment; and a device inventory registry — all tenant-scoped with full audit trails.

**Package**: `apg-mob-mdm`  
**Path**: `capabilities/mob/mdm`  
**Version**: 1.0.0  

**Provides:**
- `device_enrolment_workflow`
- `mdm_policy_enforcement`
- `compliance_monitoring`
- `remote_wipe_workflow`
- `app_distribution_workflow`
- `mdm_profile_deployment`
- `device_lock_workflow`
- `enrolment_state_machine`
- `corporate_wipe_workflow`
- `device_inventory_registry`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `comp`
- `moni`
- `wflo`
- `mqeb`

**Service methods** (40 total):
`uuid7str`, `describe`, `evaluate`, `enrol_device`, `get_device`, `list_devices`, `update_device`, `unenrol_device`, `suspend_device`, `create_policy`, `activate_policy`, `update_policy`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `device_type_must_be_supported`, `os_platform_must_be_supported`, `enrolment_method_must_be_supported`, `enrolment_requires_approval`, `policy_type_must_be_supported`, `policy_activation_requires_approval`, ...

**UI Routes** (14):
- `/mob-mdm/dashboard` — dashboard (mob_mdm:view)
- `/mob-mdm/devices` — devices (mob_mdm:devices:list)
- `/mob-mdm/devices/<device_id>` — device_detail (mob_mdm:devices:view)
- `/mob-mdm/enrolment` — enrolment (mob_mdm:enrolment:manage)
- `/mob-mdm/policies` — policies (mob_mdm:policies:list)
- `/mob-mdm/policies/<policy_id>` — policy_detail (mob_mdm:policies:view)
- _8 more..._

**Streaming events** via `bytewax`:
`device_enrolled`, `device_unenrolled`, `device_suspended`, `device_wiped`, `policy_created`, ...

**Standalone usage:**
```bash
pip install apg-mob-mdm
apg-mob-mdm --port 8080
```

---

### Remote Workforce `mob_rwf`

> The Remote Workforce (RWF) capability provides a complete remote and hybrid work governance runtime. It manages remote work policy authoring, activation, and employee acknowledgment; VPN access provisioning with MFA enforcement and split-tunneling prevention; consent-based productivity tracking; equipment requisition with per-employee limits; digital onboarding orchestration with step tracking; remote compliance checks; and remote incident management — all governed by tenant-scoped deterministic rules with full audit trails.

**Package**: `apg-mob-rwf`  
**Path**: `capabilities/mob/rwf`  
**Version**: 1.0.0  

**Provides:**
- `remote_work_policy_management`
- `vpn_access_governance`
- `productivity_tracking_workflow`
- `equipment_requisition_workflow`
- `digital_onboarding_workflow`
- `remote_compliance_monitoring`
- `remote_incident_management`
- `onboarding_step_orchestration`
- `policy_acknowledgment_workflow`
- `remote_workforce_analytics`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `nlpc`
- `moni`
- `wflo`
- `schd`
- `mqeb`

**Service methods** (40 total):
`uuid7str`, `describe`, `evaluate`, `create_work_policy`, `activate_work_policy`, `update_work_policy`, `list_work_policies`, `get_work_policy`, `acknowledge_policy`, `list_acknowledgments`, `provision_vpn`, `revoke_vpn`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `work_policy_type_must_be_supported`, `work_policy_activation_requires_approval`, `policy_acknowledgment_requires_active_policy`, `vpn_protocol_must_be_supported`, `vpn_requires_approval`, `vpn_requires_mfa`, ...

**UI Routes** (15):
- `/mob-rwf/dashboard` — dashboard (mob_rwf:view)
- `/mob-rwf/policies` — work_policies (mob_rwf:policies:list)
- `/mob-rwf/policies/<policy_id>` — policy_detail (mob_rwf:policies:view)
- `/mob-rwf/policies/<policy_id>/acknowledge` — policy_acknowledge (mob_rwf:policies:acknowledge)
- `/mob-rwf/vpn` — vpn_access (mob_rwf:vpn:list)
- `/mob-rwf/vpn/provision` — vpn_provision (mob_rwf:vpn:provision)
- _9 more..._

**Streaming events** via `bytewax`:
`work_policy_created`, `work_policy_activated`, `work_policy_acknowledged`, `vpn_access_provisioned`, `vpn_access_revoked`, ...

**Standalone usage:**
```bash
pip install apg-mob-rwf
apg-mob-rwf --port 8080
```

---

## PDE

### Product Information Management `pde_pim`

> Product Information Management (PIM) is the APG capability packet that owns the authoritative record for every product in a tenant's catalog. It governs the full product data lifecycle — from catalog and SKU creation through attribute enrichment, variant modelling, content localisation, digital asset attachment, compliance documentation, channel listing, and final publication — enforcing a rule-based governance layer at every transition.

**Package**: `apg-pde-pim`  
**Path**: `capabilities/pde/pim`  
**Version**: 2.1.0  

**Provides:**
- `product_catalog_lifecycle`
- `product_record_lifecycle`
- `product_attribute_lifecycle`
- `product_variant_lifecycle`
- `product_content_lifecycle`
- `product_asset_lifecycle`
- `product_compliance_lifecycle`
- `product_channel_listing_lifecycle`
- `product_publish_workflow`
- `product_data_quality_workflow`
- `product_change_workflow`
- `pim_dashboard_service`
- `pim_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `wflo`
- `mdm`
- `onto`

**Service methods** (40 total):
`create_product`, `update_attributes`, `add_media`, `product_categorisation`, `data_quality_score`, `publish_to_channel`, `unpublish`, `bulk_import`, `product_search`, `pim_analytics`, `count`, `describe`, ...

**Governance rules** (48 total):
`tenant_context_required`, `operation_policy_required`, `catalog_code_required`, `catalog_name_required`, `catalog_owner_required`, `product_catalog_required`, `product_sku_required`, `product_name_required`, ...

**UI Routes** (13):
- `/pde/pim/dashboard` — dashboard (pde_pim:view)
- `/pde/pim/catalogs` — catalogs (pde_pim:manage_catalogs)
- `/pde/pim/products` — products (pde_pim:manage_products)
- `/pde/pim/attributes` — attributes (pde_pim:manage_attributes)
- `/pde/pim/content` — content (pde_pim:manage_content)
- `/pde/pim/assets` — assets (pde_pim:manage_assets)
- _7 more..._

**Streaming events** via `bytewax`:
`catalog_created`, `product_created`, `attribute_defined`, `attribute_value_set`, `variant_created`, ...

**Standalone usage:**
```bash
pip install apg-pde-pim
apg-pde-pim --port 8080
```

---

## PHARMA

### Commercial Operations `pharma_com`

> Manages pharmaceutical field force activities including territory management, sales rep assignments, physician call recording, PDMA-compliant sample dispensing, HCP interaction tracking, aggregate spend management, and commercial planning. Enforces Sunshine Act reporting and PDMA compliance rules at every transactional boundary.

**Package**: `apg-pharma-com`  
**Path**: `capabilities/pharma/com`  
**Version**: 1.0.0  

**Provides:**
- `territory_management_workflow`
- `sales_rep_management_workflow`
- `call_activity_workflow`
- `sample_management_workflow`
- `hcp_interaction_workflow`
- `commercial_plan_workflow`
- `target_segmentation_workflow`
- `aggregate_spend_workflow`
- `pdma_compliance_workflow`
- `commercial_dashboard_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `schd`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `create_territory`, `get_territory`, `list_territories`, `update_territory`, `assign_rep`, `get_rep`, `list_reps`, `list_reps_by_territory`, `record_call`, `list_calls`, ...

**Governance rules** (25 total):
`tenant_context_required`, `write_requires_policy`, `territory_type_supported`, `territory_owner_required`, `territory_approval_required`, `rep_type_supported`, `rep_territory_required`, `rep_certification_required`, ...

**UI Routes** (14):
- `/pharma-com/dashboard` — dashboard (pharma_com:view)
- `/pharma-com/territories` — territories (pharma_com:territories)
- `/pharma-com/territories/<id>` — territory_detail (pharma_com:territories)
- `/pharma-com/reps` — reps (pharma_com:reps)
- `/pharma-com/calls` — calls (pharma_com:calls)
- `/pharma-com/samples` — samples (pharma_com:samples)
- _8 more..._

**Streaming events** via `bytewax`:
`territory_created`, `territory_updated`, `rep_assigned`, `call_recorded`, `sample_dispensed`, ...

**Standalone usage:**
```bash
pip install apg-pharma-com
apg-pharma-com --port 8080
```

---

### Clinical Trials Management `pharma_ctr`

> Manages the complete clinical trial lifecycle from protocol development through site initiation, patient enrolment, randomisation, adverse event reporting, data management, and regulatory submissions. Enforces GCP compliance, informed consent requirements, IRB approvals, and ICH E6 expedited reporting timelines at every boundary.

**Package**: `apg-pharma-ctr`  
**Path**: `capabilities/pharma/ctr`  
**Version**: 1.0.0  

**Provides:**
- `trial_protocol_workflow`
- `site_selection_workflow`
- `patient_randomisation_workflow`
- `adverse_event_workflow`
- `clinical_data_management_workflow`
- `regulatory_submission_workflow`
- `informed_consent_workflow`
- `monitoring_visit_workflow`
- `safety_reporting_workflow`
- `trial_closure_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `nlpc`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `create_trial`, `register_trial`, `activate_trial`, `get_trial`, `list_trials`, `create_protocol`, `approve_protocol`, `list_protocols`, `select_site`, `initiate_site`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `trial_phase_supported`, `trial_irb_approval_required`, `trial_sponsor_required`, `protocol_version_required`, `protocol_irb_review_required`, `site_qualification_required`, ...

**UI Routes** (14):
- `/pharma-ctr/dashboard` — dashboard (pharma_ctr:view)
- `/pharma-ctr/trials` — trials (pharma_ctr:trials)
- `/pharma-ctr/trials/<id>` — trial_detail (pharma_ctr:trials)
- `/pharma-ctr/protocols` — protocols (pharma_ctr:protocols)
- `/pharma-ctr/sites` — sites (pharma_ctr:sites)
- `/pharma-ctr/patients` — patients (pharma_ctr:patients)
- _8 more..._

**Streaming events** via `bytewax`:
`trial_created`, `protocol_approved`, `site_initiated`, `patient_enrolled`, `patient_randomised`, ...

**Standalone usage:**
```bash
pip install apg-pharma-ctr
apg-pharma-ctr --port 8080
```

---

### Pharmaceutical Distribution `pharma_dis`

> Manages pharmaceutical distribution operations including cold chain monitoring, product serialisation and verification, wholesale distribution authorisations, product recalls, GDP compliance, and import/export shipment tracking. Enforces WDA requirements, temperature monitoring, serialisation verification, and recall timeline obligations at every distribution boundary.

**Package**: `apg-pharma-dis`  
**Path**: `capabilities/pharma/dis`  
**Version**: 1.0.0  

**Provides:**
- `wholesale_distribution_workflow`
- `cold_chain_management_workflow`
- `serialisation_verification_workflow`
- `recall_management_workflow`
- `gdp_compliance_workflow`
- `wda_management_workflow`
- `shipment_tracking_workflow`
- `temperature_excursion_workflow`
- `import_export_workflow`
- `distribution_audit_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `moni`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `create_shipment`, `dispatch_shipment`, `deliver_shipment`, `get_shipment`, `list_shipments`, `create_cold_chain_record`, `report_excursion`, `list_excursions`, `serialise_product`, `verify_serialisation`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `wda_required_for_wholesale`, `cold_chain_classification_supported`, `cold_chain_monitoring_required`, `excursion_reporting_required`, `serialisation_standard_supported`, `serialisation_verification_required`, ...

**UI Routes** (14):
- `/pharma-dis/dashboard` — dashboard (pharma_dis:view)
- `/pharma-dis/shipments` — shipments (pharma_dis:shipments)
- `/pharma-dis/shipments/<id>` — shipment_detail (pharma_dis:shipments)
- `/pharma-dis/cold-chain` — cold_chain (pharma_dis:cold_chain)
- `/pharma-dis/cold-chain/excursions` — excursions (pharma_dis:cold_chain)
- `/pharma-dis/serialisation` — serialisation (pharma_dis:serialisation)
- _8 more..._

**Streaming events** via `bytewax`:
`shipment_dispatched`, `shipment_delivered`, `shipment_exception`, `cold_chain_excursion_detected`, `temperature_breach_escalated`, ...

**Standalone usage:**
```bash
pip install apg-pharma-dis
apg-pharma-dis --port 8080
```

---

### Pharmaceutical Manufacturing `pharma_mfg`

> Manages pharmaceutical manufacturing operations from batch record creation through equipment qualification, yield management, deviation handling, line clearance, raw material management, and QP batch release. Enforces GMP compliance, electronic batch records, QP release signatures, and equipment qualification requirements at every production step.

**Package**: `apg-pharma-mfg`  
**Path**: `capabilities/pharma/mfg`  
**Version**: 1.0.0  

**Provides:**
- `batch_record_management_workflow`
- `manufacturing_execution_workflow`
- `equipment_qualification_workflow`
- `yield_management_workflow`
- `deviation_management_workflow`
- `gmp_compliance_workflow`
- `material_management_workflow`
- `line_clearance_workflow`
- `cleaning_validation_workflow`
- `qp_release_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `moni`
- `schd`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `create_batch`, `start_batch`, `release_batch`, `reject_batch`, `get_batch`, `list_batches`, `register_equipment`, `qualify_equipment`, `use_equipment`, `list_equipment`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `batch_master_formula_required`, `batch_number_required`, `batch_status_supported`, `qp_release_required`, `electronic_signature_required`, `equipment_qualification_required`, ...

**UI Routes** (14):
- `/pharma-mfg/dashboard` — dashboard (pharma_mfg:view)
- `/pharma-mfg/batches` — batches (pharma_mfg:batches)
- `/pharma-mfg/batches/<id>` — batch_detail (pharma_mfg:batches)
- `/pharma-mfg/batches/<id>/ebr` — batch_record (pharma_mfg:ebr)
- `/pharma-mfg/lines` — lines (pharma_mfg:lines)
- `/pharma-mfg/equipment` — equipment (pharma_mfg:equipment)
- _8 more..._

**Streaming events** via `bytewax`:
`batch_started`, `batch_completed`, `batch_released`, `batch_rejected`, `equipment_qualified`, ...

**Standalone usage:**
```bash
pip install apg-pharma-mfg
apg-pharma-mfg --port 8080
```

---

### Pharmacovigilance `pharma_pvi`

> Manages the complete pharmacovigilance lifecycle from adverse event intake through ICSR submission, signal detection, PSUR/PBRER generation, and regulatory database reporting. Enforces ICH E2B(R3) formatting, 7-day/15-day expedited reporting timelines, MedDRA coding, duplicate detection, and benefit-risk assessment requirements.

**Package**: `apg-pharma-pvi`  
**Path**: `capabilities/pharma/pvi`  
**Version**: 1.0.0  

**Provides:**
- `adverse_event_collection_workflow`
- `case_processing_workflow`
- `signal_detection_workflow`
- `psur_generation_workflow`
- `regulatory_reporting_workflow`
- `literature_screening_workflow`
- `benefit_risk_assessment_workflow`
- `follow_up_management_workflow`
- `duplicate_detection_workflow`
- `meddra_coding_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `nlpc`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `create_case`, `process_case`, `close_case`, `mark_duplicate`, `get_case`, `list_cases`, `submit_icsr`, `list_icsr_submissions`, `create_signal`, `evaluate_signal`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `ae_source_supported`, `case_type_supported`, `meddra_coding_required`, `narrative_required`, `causality_required`, `duplicate_check_required`, ...

**UI Routes** (14):
- `/pharma-pvi/dashboard` — dashboard (pharma_pvi:view)
- `/pharma-pvi/cases/intake` — case_intake (pharma_pvi:cases)
- `/pharma-pvi/cases` — cases (pharma_pvi:cases)
- `/pharma-pvi/cases/<id>` — case_detail (pharma_pvi:cases)
- `/pharma-pvi/cases/follow-up` — follow_up (pharma_pvi:follow_up)
- `/pharma-pvi/signals` — signals (pharma_pvi:signals)
- _8 more..._

**Streaming events** via `bytewax`:
`ae_received`, `case_created`, `case_processed`, `case_closed`, `duplicate_detected`, ...

**Standalone usage:**
```bash
pip install apg-pharma-pvi
apg-pharma-pvi --port 8080
```

---

### Quality Management System `pharma_qms`

> End-to-end pharmaceutical QMS covering change control, CAPA management, deviation handling, controlled document management, audit management, validation lifecycle, and risk assessment. All workflows enforce GMP compliance, electronic signature requirements, and effectiveness check obligations before closure.

**Package**: `apg-pharma-qms`  
**Path**: `capabilities/pharma/qms`  
**Version**: 1.0.0  

**Provides:**
- `change_control_workflow`
- `capa_management_workflow`
- `deviation_management_workflow`
- `document_control_workflow`
- `audit_management_workflow`
- `validation_lifecycle_workflow`
- `risk_management_workflow`
- `quality_metrics_workflow`
- `supplier_quality_workflow`
- `qms_review_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `schd`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `initiate_change`, `approve_change`, `implement_change`, `close_change`, `list_changes`, `create_capa`, `close_capa`, `check_overdue_capas`, `list_capas`, `raise_deviation`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `change_type_supported`, `change_impact_assessment_required`, `change_risk_assessment_required`, `change_approval_required`, `change_effectiveness_check_required`, `capa_type_supported`, ...

**UI Routes** (14):
- `/pharma-qms/dashboard` — dashboard (pharma_qms:view)
- `/pharma-qms/change-control` — change_control (pharma_qms:change_control)
- `/pharma-qms/change-control/<id>` — change_detail (pharma_qms:change_control)
- `/pharma-qms/capa` — capa (pharma_qms:capa)
- `/pharma-qms/capa/<id>` — capa_detail (pharma_qms:capa)
- `/pharma-qms/deviations` — deviations (pharma_qms:deviations)
- _8 more..._

**Streaming events** via `bytewax`:
`change_initiated`, `change_approved`, `change_implemented`, `capa_raised`, `capa_closed`, ...

**Standalone usage:**
```bash
pip install apg-pharma-qms
apg-pharma-qms --port 8080
```

---

### Regulatory Compliance `pharma_rec`

> Manages pharmaceutical regulatory compliance obligations across multiple frameworks (FDA, EMA, GMP, ICH), including gap assessments, inspection readiness, label change management, post-market surveillance, regulatory intelligence dissemination, and regulatory commitment tracking. Enforces inspection response timelines, label QP approval, and overdue commitment escalation.

**Package**: `apg-pharma-rec`  
**Path**: `capabilities/pharma/rec`  
**Version**: 1.0.0  

**Provides:**
- `regulatory_compliance_monitoring_workflow`
- `inspection_readiness_workflow`
- `label_management_workflow`
- `post_market_surveillance_workflow`
- `regulatory_intelligence_workflow`
- `commitment_tracking_workflow`
- `compliance_gap_assessment_workflow`
- `inspection_response_workflow`
- `regulatory_change_impact_workflow`
- `compliance_audit_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `nlpc`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `register_compliance`, `list_frameworks`, `create_gap_assessment`, `close_gap_assessment`, `list_gap_assessments`, `record_inspection`, `record_inspection_outcome`, `respond_to_inspection`, `list_inspections`, `create_label`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `compliance_framework_supported`, `compliance_gap_implementation_plan_required`, `inspection_type_supported`, `inspection_outcome_supported`, `warning_letter_30d_response`, `inspection_capa_required`, ...

**UI Routes** (13):
- `/pharma-rec/dashboard` — dashboard (pharma_rec:view)
- `/pharma-rec/compliance` — compliance_register (pharma_rec:compliance)
- `/pharma-rec/compliance/gap` — gap_assessment (pharma_rec:gap_assessment)
- `/pharma-rec/inspections` — inspections (pharma_rec:inspections)
- `/pharma-rec/inspections/<id>` — inspection_detail (pharma_rec:inspections)
- `/pharma-rec/labeling` — labeling (pharma_rec:labeling)
- _7 more..._

**Streaming events** via `bytewax`:
`compliance_gap_identified`, `inspection_announced`, `inspection_completed`, `warning_letter_received`, `inspection_response_submitted`, ...

**Standalone usage:**
```bash
pip install apg-pharma-rec
apg-pharma-rec --port 8080
```

---

### Product Registration `pharma_reg`

> Manages pharmaceutical product registration across global regulatory regions including dossier compilation, eCTD validation, authority interactions, approval tracking, variation management, renewal lifecycle, certificate storage, and multi-regional procedure coordination. Enforces QP sign-off, eCTD validation, and 180-day renewal alert requirements.

**Package**: `apg-pharma-reg`  
**Path**: `capabilities/pharma/reg`  
**Version**: 1.0.0  

**Provides:**
- `registration_application_workflow`
- `dossier_compilation_workflow`
- `authority_interaction_workflow`
- `approval_tracking_workflow`
- `lifecycle_maintenance_workflow`
- `variation_management_workflow`
- `renewal_management_workflow`
- `procedure_management_workflow`
- `registration_certificate_workflow`
- `global_dossier_alignment_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `schd`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `prepare_dossier`, `dossier_completeness_check`, `submit_registration`, `track_review_status`, `respond_to_query`, `registration_approval`, `variation_application`, `annual_renewal`, `registration_withdrawal`, `registration_analytics`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `registration_type_supported`, `dossier_required_for_submission`, `dossier_format_supported`, `ectd_validation_required`, `qp_sign_off_required`, `approval_before_distribution`, ...

**UI Routes** (14):
- `/pharma-reg/dashboard` — dashboard (pharma_reg:view)
- `/pharma-reg/registrations` — registrations (pharma_reg:registrations)
- `/pharma-reg/registrations/<id>` — registration_detail (pharma_reg:registrations)
- `/pharma-reg/dossiers` — dossiers (pharma_reg:dossiers)
- `/pharma-reg/dossiers/<id>` — dossier_detail (pharma_reg:dossiers)
- `/pharma-reg/approvals` — approvals (pharma_reg:approvals)
- _8 more..._

**Streaming events** via `bytewax`:
`registration_submitted`, `registration_approved`, `registration_refused`, `dossier_compiled`, `dossier_updated`, ...

**Standalone usage:**
```bash
pip install apg-pharma-reg
apg-pharma-reg --port 8080
```

---

### Pharmaceutical Supply Chain `pharma_sup`

> Manages the pharmaceutical supply chain from active ingredient sourcing through CMO management, demand planning, import licensing, supply security monitoring, purchase order management, and supply contract lifecycle. Enforces approved supplier list requirements, quality agreement obligations, import license verification, and dual sourcing requirements for high-risk products.

**Package**: `apg-pharma-sup`  
**Path**: `capabilities/pharma/sup`  
**Version**: 1.0.0  

**Provides:**
- `active_ingredient_sourcing_workflow`
- `cmo_management_workflow`
- `demand_planning_workflow`
- `import_licensing_workflow`
- `supply_security_monitoring_workflow`
- `supplier_qualification_workflow`
- `purchase_order_workflow`
- `supply_contract_workflow`
- `approved_supplier_list_workflow`
- `supply_risk_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `moni`
- `schd`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `create_supplier`, `qualify_supplier`, `suspend_supplier`, `get_supplier`, `list_suppliers`, `activate_cmo`, `list_cmos`, `create_forecast`, `approve_sop`, `list_forecasts`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `supplier_type_supported`, `approved_supplier_list_required`, `quality_agreement_required`, `supplier_qualification_required`, `cmo_type_supported`, `cmo_technical_agreement_required`, ...

**UI Routes** (14):
- `/pharma-sup/dashboard` — dashboard (pharma_sup:view)
- `/pharma-sup/suppliers` — suppliers (pharma_sup:suppliers)
- `/pharma-sup/suppliers/<id>` — supplier_detail (pharma_sup:suppliers)
- `/pharma-sup/asl` — approved_supplier_list (pharma_sup:asl)
- `/pharma-sup/cmo` — cmo (pharma_sup:cmo)
- `/pharma-sup/cmo/<id>` — cmo_detail (pharma_sup:cmo)
- _8 more..._

**Streaming events** via `bytewax`:
`supplier_qualified`, `supplier_suspended`, `supplier_audit_completed`, `cmo_activated`, `cmo_agreement_signed`, ...

**Standalone usage:**
```bash
pip install apg-pharma-sup
apg-pharma-sup --port 8080
```

---

## PPM

### Project Accounting `ppm_pac`

> Project Accounting (pac) provides complete financial tracking for projects: cost capture, revenue recognition under multiple WIP methods, milestone billing, budget control, and profitability reporting. Every transaction is tenant-scoped, approval-gated, and streamed via Bytewax for real-time financial visibility.

**Package**: `apg-ppm-pac`  
**Path**: `capabilities/ppm/pac`  
**Version**: 1.0.0  

**Provides:**
- `project_cost_tracking`
- `revenue_recognition_workflow`
- `wip_accounting_workflow`
- `milestone_billing_workflow`
- `project_profitability_reporting`
- `budget_vs_actual_analysis`
- `cost_variance_alerts`
- `cash_flow_forecasting`
- `multi_currency_project_accounting`
- `audit_trail_maintenance`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `comp`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `create_account`, `get_account`, `list_accounts`, `project_budget_setup`, `cost_code_create`, `record_timesheet_cost`, `record_expense`, `purchase_order_project`, `invoice_project_cost`, `earned_value_analysis`, ...

**Governance rules** (35 total):
`tenant_context_required`, `write_requires_policy`, `account_status_supported`, `account_owner_required`, `account_budget_required`, `account_currency_supported`, `account_evidence_required`, `cost_type_supported`, ...

**UI Routes** (14):
- `/ppm-pac/dashboard` — dashboard (ppm_pac:view)
- `/ppm-pac/accounts` — project_accounts (ppm_pac:accounts)
- `/ppm-pac/accounts/<id>` — account_detail (ppm_pac:accounts)
- `/ppm-pac/costs` — cost_transactions (ppm_pac:costs)
- `/ppm-pac/revenue` — revenue_recognition (ppm_pac:revenue)
- `/ppm-pac/wip` — wip_accounting (ppm_pac:wip)
- _8 more..._

**Streaming events** via `bytewax`:
`project_account_created`, `cost_transaction_recorded`, `revenue_recognised`, `wip_adjustment_posted`, `milestone_invoice_raised`, ...

**Standalone usage:**
```bash
pip install apg-ppm-pac
apg-ppm-pac --port 8080
```

---

### Portfolio Analytics `ppm_pan`

> Portfolio Analytics (pan) delivers executive-grade visibility across the project portfolio: strategic alignment scoring, risk-return matrices, capacity heat maps, performance scorecards, benchmark comparisons, and scenario analysis. All analytics are tenant-scoped, approval-gated for writes, and emitted as events for downstream consumption.

**Package**: `apg-ppm-pan`  
**Path**: `capabilities/ppm/pan`  
**Version**: 1.0.0  

**Provides:**
- `portfolio_performance_dashboard`
- `strategic_alignment_scoring`
- `risk_return_analysis`
- `capacity_heat_map`
- `portfolio_investment_analysis`
- `project_pipeline_reporting`
- `benchmark_comparison`
- `portfolio_optimisation_recommendations`
- `executive_portfolio_briefings`
- `scenario_analysis`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `nlpc`
- `moni`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `create_portfolio`, `get_portfolio`, `list_portfolios`, `portfolio_overview`, `strategic_alignment_score`, `score_alignment`, `list_alignment_scores`, `risk_return_analysis`, `analyse_risk_return`, `capacity_demand_chart`, ...

**Governance rules** (28 total):
`tenant_context_required`, `write_requires_policy`, `portfolio_status_supported`, `portfolio_owner_required`, `portfolio_classification_supported`, `portfolio_evidence_required`, `portfolio_write_requires_approval`, `alignment_dimension_supported`, ...

**UI Routes** (14):
- `/ppm-pan/dashboard` — dashboard (ppm_pan:view)
- `/ppm-pan/portfolios` — portfolios (ppm_pan:portfolios)
- `/ppm-pan/portfolios/<id>` — portfolio_detail (ppm_pan:portfolios)
- `/ppm-pan/alignment` — strategic_alignment (ppm_pan:alignment)
- `/ppm-pan/risk-return` — risk_return (ppm_pan:risk)
- `/ppm-pan/capacity` — capacity_heat_map (ppm_pan:capacity)
- _8 more..._

**Streaming events** via `bytewax`:
`portfolio_created`, `portfolio_updated`, `alignment_score_calculated`, `risk_return_analysed`, `capacity_heat_map_generated`, ...

**Standalone usage:**
```bash
pip install apg-ppm-pan
apg-ppm-pan --port 8080
```

---

### Project Baseline Management `ppm_pbl`

> Project Baseline Management (pbl) establishes and protects the scope, schedule, and cost baselines for projects. It enforces formal change control, calculates earned value metrics, detects variance threshold breaches, and prevents retroactive baseline manipulation — providing the performance measurement baseline required for EVM compliance.

**Package**: `apg-ppm-pbl`  
**Path**: `capabilities/ppm/pbl`  
**Version**: 1.0.0  

**Provides:**
- `scope_baseline_management`
- `schedule_baseline_management`
- `cost_baseline_management`
- `change_control_workflow`
- `earned_value_analysis`
- `baseline_variance_tracking`
- `change_impact_assessment`
- `baseline_approval_workflow`
- `integrated_baseline_review`
- `performance_measurement_baseline`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `set_scope_baseline`, `set_schedule_baseline`, `set_cost_baseline`, `change_request`, `approve_change`, `baseline_comparison`, `variance_analysis`, `change_log`, `baseline_restore`, `baseline_analytics`, ...

**Governance rules** (26 total):
`tenant_context_required`, `write_requires_policy`, `baseline_type_supported`, `baseline_owner_required`, `baseline_approval_required`, `baseline_evidence_required`, `baseline_approval_requires_designated_approver`, `retroactive_baseline_edit_denied`, ...

**UI Routes** (14):
- `/ppm-pbl/dashboard` — dashboard (ppm_pbl:view)
- `/ppm-pbl/baselines` — baselines (ppm_pbl:baselines)
- `/ppm-pbl/baselines/<id>` — baseline_detail (ppm_pbl:baselines)
- `/ppm-pbl/scope` — scope_baseline (ppm_pbl:scope)
- `/ppm-pbl/schedule` — schedule_baseline (ppm_pbl:schedule)
- `/ppm-pbl/cost` — cost_baseline (ppm_pbl:cost)
- _8 more..._

**Streaming events** via `bytewax`:
`baseline_created`, `baseline_approved`, `baseline_superseded`, `change_request_submitted`, `change_impact_assessed`, ...

**Standalone usage:**
```bash
pip install apg-ppm-pbl
apg-ppm-pbl --port 8080
```

---

### Project Planning & Scheduling `ppm_pps`

> Project Planning & Scheduling (pps) manages the full project schedule lifecycle: WBS decomposition, task definition, dependency linking with circular-dependency prevention, critical path calculation (CPM/PERT/CCPM/Monte Carlo), resource levelling, calendar management, and milestone tracking. Retroactive edits are blocked to maintain schedule integrity.

**Package**: `apg-ppm-pps`  
**Path**: `capabilities/ppm/pps`  
**Version**: 1.0.0  

**Provides:**
- `wbs_creation_and_management`
- `critical_path_analysis`
- `resource_levelling`
- `dependency_management`
- `timeline_management`
- `schedule_optimisation`
- `project_calendar_management`
- `milestone_tracking`
- `schedule_risk_analysis`
- `gantt_chart_generation`
- `schedule_baseline_export`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `schd`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `create_project`, `get_project`, `list_projects`, `add_wbs_element`, `list_wbs_elements`, `add_task`, `update_task_status`, `list_tasks`, `link_dependency`, `create_wbs`, ...

**Governance rules** (30 total):
`tenant_context_required`, `write_requires_policy`, `project_status_supported`, `project_owner_required`, `project_start_date_required`, `project_methodology_supported`, `project_evidence_required`, `task_type_supported`, ...

**UI Routes** (14):
- `/ppm-pps/dashboard` — dashboard (ppm_pps:view)
- `/ppm-pps/projects` — projects (ppm_pps:projects)
- `/ppm-pps/projects/<id>` — project_detail (ppm_pps:projects)
- `/ppm-pps/projects/<id>/wbs` — wbs (ppm_pps:wbs)
- `/ppm-pps/projects/<id>/gantt` — gantt (ppm_pps:gantt)
- `/ppm-pps/projects/<id>/critical-path` — critical_path (ppm_pps:critical_path)
- _8 more..._

**Streaming events** via `bytewax`:
`project_created`, `project_updated`, `wbs_element_added`, `task_status_changed`, `dependency_linked`, ...

**Standalone usage:**
```bash
pip install apg-ppm-pps
apg-ppm-pps --port 8080
```

---

### Resource Management `ppm_res`

> Resource Management (res) manages the full resource lifecycle: pool registration, skill cataloguing with evidence-backed proficiency, allocation to projects with over-allocation controls, capacity planning, utilisation band tracking, demand forecasting, leave management, and cost rate governance with finance-approval gates.

**Package**: `apg-ppm-res`  
**Path**: `capabilities/ppm/res`  
**Version**: 1.0.0  

**Provides:**
- `resource_pool_management`
- `skill_matching_engine`
- `capacity_planning`
- `utilisation_tracking`
- `demand_forecasting`
- `resource_allocation_workflow`
- `leave_and_availability_management`
- `cost_rate_management`
- `resource_demand_vs_supply_analysis`
- `hiring_and_contractor_planning`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `schd`
- `nlpc`
- `mqeb`

**Service methods** (42 total):
`describe`, `evaluate`, `create_resource`, `get_resource`, `list_resources`, `register_resource`, `skill_search`, `assign_resource`, `resource_utilisation`, `team_capacity`, `skills_gap_analysis`, `resource_forecasting`, ...

**Governance rules** (30 total):
`tenant_context_required`, `write_requires_policy`, `resource_type_supported`, `resource_status_supported`, `resource_owner_required`, `resource_cost_rate_required`, `resource_evidence_required`, `skill_proficiency_supported`, ...

**UI Routes** (14):
- `/ppm-res/dashboard` — dashboard (ppm_res:view)
- `/ppm-res/resources` — resource_pool (ppm_res:resources)
- `/ppm-res/resources/<id>` — resource_detail (ppm_res:resources)
- `/ppm-res/skills` — skills (ppm_res:skills)
- `/ppm-res/skill-match` — skill_matching (ppm_res:skill_match)
- `/ppm-res/allocations` — allocations (ppm_res:allocations)
- _8 more..._

**Streaming events** via `bytewax`:
`resource_created`, `resource_updated`, `skill_added`, `allocation_confirmed`, `allocation_cancelled`, ...

**Standalone usage:**
```bash
pip install apg-ppm-res
apg-ppm-res --port 8080
```

---

### Time & Expense Management `ppm_tex`

> Time & Expense Management (tex) handles the complete employee time and expense lifecycle: weekly/bi-weekly/monthly timesheet entry with project and task linkage, expense claim submission with receipt enforcement above configurable thresholds, multi-step approval workflows, reimbursement processing via payroll or bank transfer, billing rate management per resource/project, and billable hour export to project accounting.

**Package**: `apg-ppm-tex`  
**Path**: `capabilities/ppm/tex`  
**Version**: 1.0.0  

**Provides:**
- `timesheet_entry_and_management`
- `expense_claim_workflow`
- `approval_workflow_engine`
- `billable_hour_tracking`
- `reimbursement_processing`
- `project_time_reporting`
- `billing_rate_management`
- `compliance_and_policy_enforcement`
- `multi_currency_expense_management`
- `audit_trail_for_time_and_expenses`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `mqeb`

**Service methods** (45 total):
`describe`, `evaluate`, `submit_timesheet`, `approve_timesheet`, `reject_timesheet`, `submit_expense`, `approve_expense`, `reject_expense`, `reimburse_expense`, `per_diem_calculation`, `timesheet_analytics`, `expense_analytics`, ...

**Governance rules** (30 total):
`tenant_context_required`, `write_requires_policy`, `timesheet_status_supported`, `timesheet_project_required`, `timesheet_period_supported`, `timesheet_approval_required`, `time_entry_type_supported`, `time_entry_billable_status_supported`, ...

**UI Routes** (14):
- `/ppm-tex/dashboard` — dashboard (ppm_tex:view)
- `/ppm-tex/timesheets/my` — my_timesheets (ppm_tex:timesheets)
- `/ppm-tex/timesheets/entry` — timesheet_entry (ppm_tex:timesheets)
- `/ppm-tex/timesheets/approvals` — timesheet_approvals (ppm_tex:approve_timesheets)
- `/ppm-tex/expenses/my` — my_expenses (ppm_tex:expenses)
- `/ppm-tex/expenses/claim` — expense_claim (ppm_tex:expenses)
- _8 more..._

**Streaming events** via `bytewax`:
`timesheet_submitted`, `timesheet_approved`, `timesheet_rejected`, `time_entry_recorded`, `expense_claim_submitted`, ...

**Standalone usage:**
```bash
pip install apg-ppm-tex
apg-ppm-tex --port 8080
```

---

## REALESTATE

### Real Estate Accounting `realestate_acc`

> Provides the full property accounting stack: chart-of-accounts management, journal entry posting with period controls, service charge raising and approval, CAM (Common Area Maintenance) reconciliation, IFRS 16 lease liability and right-of-use asset schedules, revenue recognition under multiple methods, dual-control period close, and tenant account statements.

**Package**: `apg-realestate-acc`  
**Path**: `capabilities/realestate/acc`  
**Version**: 1.0.0  

**Provides:**
- `property_ledger_management`
- `service_charge_accounting`
- `cam_reconciliation_workflow`
- `ifrs16_lease_accounting`
- `revenue_recognition_engine`
- `journal_entry_management`
- `period_close_workflow`
- `tenant_statement_generation`
- `tax_calculation_engine`
- `financial_report_generation`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `mqeb`
- `schd`

**Service methods** (43 total):
`create_account`, `get_account`, `list_accounts`, `update_account`, `create_journal_entry`, `approve_journal_entry`, `post_journal_entry`, `reverse_journal_entry`, `list_journals`, `raise_service_charge`, `approve_service_charge`, `list_service_charges`, ...

**Governance rules** (25 total):
`tenant_context_required`, `write_requires_policy`, `journal_requires_balanced_entries`, `journal_requires_period_open`, `journal_above_threshold_requires_approval`, `journal_reversal_requires_original`, `service_charge_requires_property`, `service_charge_type_supported`, ...

**UI Routes** (14):
- `/realestate/acc/dashboard` — dashboard (realestate_acc:view)
- `/realestate/acc/ledger` — ledger (realestate_acc:ledger)
- `/realestate/acc/journals` — journal-entries (realestate_acc:journals)
- `/realestate/acc/service-charges` — service-charges (realestate_acc:service_charges)
- `/realestate/acc/cam` — cam-reconciliation (realestate_acc:cam)
- `/realestate/acc/ifrs16` — ifrs16 (realestate_acc:ifrs16)
- _8 more..._

**Streaming events** via `bytewax`:
`journal_entry_created`, `journal_entry_posted`, `journal_entry_reversed`, `service_charge_raised`, `service_charge_approved`, ...

**Standalone usage:**
```bash
pip install apg-realestate-acc
apg-realestate-acc --port 8080
```

---

### Property Contracts `realestate_con`

> Full contract lifecycle management for all real estate agreements: sale/purchase, management contracts, construction contracts, service agreements, joint ventures, and development agreements. Covers party management, digital signatures, milestone tracking, variation orders (with board-approval thresholds), dispute resolution, retention management, and a searchable clause library.

**Package**: `apg-realestate-con`  
**Path**: `capabilities/realestate/con`  
**Version**: 1.0.0  

**Provides:**
- `contract_lifecycle_management`
- `contractor_registry_management`
- `milestone_tracking_workflow`
- `variation_order_management`
- `dispute_resolution_workflow`
- `contract_clause_library`
- `retention_management`
- `contract_expiry_alerts`
- `digital_signature_workflow`
- `contract_performance_reporting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `nlpc`
- `comp`
- `mqeb`

**Service methods** (42 total):
`create_contract`, `get_contract`, `list_contracts`, `update_contract`, `execute_contract`, `terminate_contract`, `sign_contract_party`, `get_expiry_pipeline`, `register_contractor`, `get_contractor`, `list_contractors`, `grade_contractor`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `contract_type_supported`, `contract_requires_parties`, `contract_requires_governing_law`, `execution_requires_all_signatures`, `execution_requires_legal_review`, `contractor_blacklisted_engagement_denied`, ...

**UI Routes** (13):
- `/realestate/con/dashboard` — dashboard (realestate_con:view)
- `/realestate/con/contracts` — contracts (realestate_con:contracts)
- `/realestate/con/contracts/<id>` — contract-detail (realestate_con:contracts)
- `/realestate/con/contractors` — contractors (realestate_con:contractors)
- `/realestate/con/milestones` — milestones (realestate_con:milestones)
- `/realestate/con/variations` — variations (realestate_con:variations)
- _7 more..._

**Streaming events** via `bytewax`:
`contract_created`, `contract_executed`, `contract_suspended`, `contract_terminated`, `contractor_registered`, ...

**Standalone usage:**
```bash
pip install apg-realestate-con
apg-realestate-con --port 8080
```

---

### Property Insurance `realestate_ins`

> End-to-end property insurance portfolio management: policy creation and binding with asset schedules, claims lodgement through settlement with large-claim senior-approval gates, endorsement issuance, premium allocation across properties, automated coverage gap detection, insurer/broker registry, and renewal pipeline tracking.

**Package**: `apg-realestate-ins`  
**Path**: `capabilities/realestate/ins`  
**Version**: 1.0.0  

**Provides:**
- `policy_lifecycle_management`
- `asset_schedule_management`
- `claims_processing_workflow`
- `premium_allocation_engine`
- `coverage_gap_analysis`
- `endorsement_management`
- `insurer_broker_registry`
- `renewal_pipeline_tracking`
- `insurance_reporting`
- `compliance_certificate_management`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `mqeb`
- `schd`

**Service methods** (42 total):
`register_insurer`, `get_insurer`, `list_insurers`, `create_policy`, `get_policy`, `list_policies`, `bind_policy`, `update_policy`, `get_renewal_pipeline`, `add_asset_to_schedule`, `list_policy_assets`, `remove_asset_from_schedule`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `policy_type_supported`, `policy_requires_insurer`, `suspended_insurer_cannot_bind`, `policy_requires_asset_schedule`, `claim_requires_active_policy`, `claim_peril_must_be_covered`, ...

**UI Routes** (13):
- `/realestate/ins/dashboard` — dashboard (realestate_ins:view)
- `/realestate/ins/policies` — policies (realestate_ins:policies)
- `/realestate/ins/assets` — asset-schedule (realestate_ins:assets)
- `/realestate/ins/claims` — claims (realestate_ins:claims)
- `/realestate/ins/premiums` — premium-allocation (realestate_ins:premiums)
- `/realestate/ins/gaps` — coverage-gaps (realestate_ins:gaps)
- _7 more..._

**Streaming events** via `bytewax`:
`policy_created`, `policy_bound`, `policy_lapsed`, `policy_expired`, `policy_cancelled`, ...

**Standalone usage:**
```bash
pip install apg-realestate-ins
apg-realestate-ins --port 8080
```

---

### Lease Management `realestate_lea`

> Full lease lifecycle from heads of terms through abstraction, activation, rent escalation, option tracking, IFRS 16/ASC 842 schedule generation, rent reviews, assignments, and expiry pipeline management. AI-assisted abstraction with mandatory human verification before activation.

**Package**: `apg-realestate-lea`  
**Path**: `capabilities/realestate/lea`  
**Version**: 1.0.0  

**Provides:**
- `lease_abstraction_engine`
- `rent_escalation_scheduler`
- `lease_option_tracker`
- `ifrs16_asc842_compliance`
- `lease_expiry_pipeline`
- `rent_review_workflow`
- `lease_assignment_management`
- `dilapidation_management`
- `lease_renewal_workflow`
- `lease_performance_reporting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `nlpc`
- `comp`
- `mqeb`
- `schd`

**Service methods** (72 total):
`create_lease`, `review_lease_terms`, `execute_lease`, `amend_lease`, `renew_lease`, `surrender_lease`, `terminate_lease`, `get_lease_expiry_pipeline`, `classify_lease_ifrs16`, `calculate_rou_asset`, `calculate_lease_liability`, `amortise_rou_asset`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `lease_type_supported`, `lease_requires_property`, `lease_requires_tenant`, `lease_requires_commencement_date`, `lease_requires_expiry_date`, `escalation_type_supported`, ...

**UI Routes** (14):
- `/realestate/lea/dashboard` — dashboard (realestate_lea:view)
- `/realestate/lea/leases` — leases (realestate_lea:leases)
- `/realestate/lea/leases/<id>` — lease-detail (realestate_lea:leases)
- `/realestate/lea/abstraction` — abstraction (realestate_lea:abstraction)
- `/realestate/lea/escalations` — escalations (realestate_lea:escalations)
- `/realestate/lea/rent-reviews` — rent-reviews (realestate_lea:rent_reviews)
- _8 more..._

**Streaming events** via `bytewax`:
`lease_created`, `lease_signed`, `lease_activated`, `lease_expired`, `lease_surrendered`, ...

**Standalone usage:**
```bash
pip install apg-realestate-lea
apg-realestate-lea --port 8080
```

---

### Facilities Maintenance `realestate_mai`

> Full CAFM-grade maintenance management: asset register with lifecycle tracking, preventive maintenance (PPM) schedules with automatic next-due calculation, corrective and emergency work orders with SLA deadline enforcement, contractor management with insurance validation, statutory inspection tracking, defect management, and SLA compliance dashboards.

**Package**: `apg-realestate-mai`  
**Path**: `capabilities/realestate/mai`  
**Version**: 1.0.0  

**Provides:**
- `preventive_maintenance_scheduling`
- `work_order_management`
- `contractor_management`
- `asset_lifecycle_tracking`
- `cafm_integration_bridge`
- `sla_monitoring`
- `inspection_management`
- `defect_tracking`
- `maintenance_cost_management`
- `compliance_maintenance_reporting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `schd`
- `comp`
- `mqeb`
- `moni`

**Service methods** (42 total):
`register_asset`, `get_asset`, `list_assets`, `update_asset`, `get_end_of_life_assets`, `create_ppm_schedule`, `list_ppm_schedules`, `complete_ppm`, `get_overdue_ppms`, `raise_work_order`, `assign_work_order`, `update_work_order`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `work_order_type_supported`, `work_order_requires_asset`, `decommissioned_asset_work_order_denied`, `p1_work_order_requires_immediate_assignment`, `work_order_priority_supported`, `ppm_frequency_supported`, ...

**UI Routes** (14):
- `/realestate/mai/dashboard` — dashboard (realestate_mai:view)
- `/realestate/mai/work-orders` — work-orders (realestate_mai:work_orders)
- `/realestate/mai/ppm` — ppm-schedules (realestate_mai:ppm)
- `/realestate/mai/assets` — assets (realestate_mai:assets)
- `/realestate/mai/assets/<id>` — asset-detail (realestate_mai:assets)
- `/realestate/mai/contractors` — contractors (realestate_mai:contractors)
- _8 more..._

**Streaming events** via `bytewax`:
`work_order_raised`, `work_order_assigned`, `work_order_completed`, `work_order_overdue`, `ppm_schedule_generated`, ...

**Standalone usage:**
```bash
pip install apg-realestate-mai
apg-realestate-mai --port 8080
```

---

### Property Management `realestate_prm`

> Central portfolio management for all real estate assets. Registers properties and units, manages owner entities and their distributions, tracks performance KPIs (occupancy, WAULT, yield), coordinates handovers, and provides an owner portal and searchable data room for each property.

**Package**: `apg-realestate-prm`  
**Path**: `capabilities/realestate/prm`  
**Version**: 1.0.0  

**Provides:**
- `property_portfolio_management`
- `unit_management`
- `owner_portal_service`
- `property_performance_reporting`
- `portfolio_analytics`
- `handover_management`
- `owner_distribution_management`
- `property_data_room`
- `performance_kpi_engine`
- `property_benchmarking`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `mqeb`
- `srch`

**Service methods** (43 total):
`register_owner`, `get_owner`, `list_owners`, `update_owner`, `register_property`, `get_property`, `list_properties`, `update_property`, `delete_property`, `create_unit`, `get_unit`, `list_units`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `property_type_supported`, `property_requires_owner`, `property_requires_address`, `property_deletion_requires_board_approval`, `unit_type_supported`, `unit_requires_property`, ...

**UI Routes** (14):
- `/realestate/prm/dashboard` — dashboard (realestate_prm:view)
- `/realestate/prm/portfolio` — portfolio (realestate_prm:portfolio)
- `/realestate/prm/properties` — properties (realestate_prm:properties)
- `/realestate/prm/properties/<id>` — property-detail (realestate_prm:properties)
- `/realestate/prm/units` — units (realestate_prm:units)
- `/realestate/prm/owners` — owners (realestate_prm:owners)
- _8 more..._

**Streaming events** via `bytewax`:
`property_registered`, `property_status_changed`, `property_sold`, `unit_status_changed`, `unit_let`, ...

**Standalone usage:**
```bash
pip install apg-realestate-prm
apg-realestate-prm --port 8080
```

---

### Rental Operations `realestate_ren`

> End-to-end tenancy lifecycle: application, referencing, right-to-rent checks, deposit registration and accounting, rent collection with shortfall detection, arrears management and legal escalation, notice serving, and renewal pipeline management. Produces a live rent roll for any property.

**Package**: `apg-realestate-ren`  
**Path**: `capabilities/realestate/ren`  
**Version**: 1.0.0  

**Provides:**
- `tenancy_lifecycle_management`
- `rent_collection_engine`
- `arrears_management_workflow`
- `deposit_accounting`
- `tenancy_renewal_pipeline`
- `referencing_workflow`
- `notice_management`
- `legal_action_tracking`
- `rent_roll_management`
- `tenancy_performance_reporting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `comp`
- `mqeb`
- `schd`

**Service methods** (44 total):
`create_tenancy`, `get_tenancy`, `list_tenancies`, `activate_tenancy`, `update_tenancy`, `record_rent_payment`, `list_payments`, `_update_arrears`, `_clear_arrears`, `record_arrears`, `get_arrears_report`, `escalate_arrears_to_legal`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `tenancy_type_supported`, `tenancy_requires_unit`, `tenancy_requires_tenant`, `activation_requires_deposit_registered`, `activation_requires_referencing_complete`, `right_to_rent_required_for_residential`, ...

**UI Routes** (13):
- `/realestate/ren/dashboard` — dashboard (realestate_ren:view)
- `/realestate/ren/tenancies` — tenancies (realestate_ren:tenancies)
- `/realestate/ren/tenancies/<id>` — tenancy-detail (realestate_ren:tenancies)
- `/realestate/ren/referencing` — referencing (realestate_ren:referencing)
- `/realestate/ren/rent-collection` — rent-collection (realestate_ren:rent_collection)
- `/realestate/ren/arrears` — arrears (realestate_ren:arrears)
- _7 more..._

**Streaming events** via `bytewax`:
`tenancy_created`, `tenancy_activated`, `tenancy_vacated`, `rent_received`, `rent_overdue`, ...

**Standalone usage:**
```bash
pip install apg-realestate-ren
apg-realestate-ren --port 8080
```

---

### Space Planning & Management `realestate_spa`

> Comprehensive workplace and space management: versioned floor plans, space allocation and deallocation, move management with headcount-threshold approvals, conflict-checked space bookings, anonymised sensor-data ingestion for occupancy analytics, workplace density planning, and space chargeback calculation.

**Package**: `apg-realestate-spa`  
**Path**: `capabilities/realestate/spa`  
**Version**: 1.0.0  

**Provides:**
- `floor_plan_management`
- `space_allocation_engine`
- `move_management_workflow`
- `occupancy_analytics`
- `workplace_density_planning`
- `space_booking_engine`
- `sensor_integration_bridge`
- `department_space_reporting`
- `space_optimisation_advisor`
- `chargeback_space_accounting`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `mqeb`
- `schd`

**Service methods** (42 total):
`upload_floor_plan`, `get_floor_plan`, `list_floor_plans`, `create_space`, `get_space`, `list_spaces`, `update_space`, `get_available_spaces`, `allocate_space`, `deallocate_space`, `list_allocations`, `create_move`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `space_type_supported`, `space_requires_floor_plan`, `space_double_booking_denied`, `decommissioned_space_booking_denied`, `allocation_type_supported`, `move_type_supported`, ...

**UI Routes** (13):
- `/realestate/spa/dashboard` — dashboard (realestate_spa:view)
- `/realestate/spa/floor-plans` — floor-plans (realestate_spa:floor_plans)
- `/realestate/spa/spaces` — spaces (realestate_spa:spaces)
- `/realestate/spa/allocations` — allocations (realestate_spa:allocations)
- `/realestate/spa/moves` — moves (realestate_spa:moves)
- `/realestate/spa/bookings` — bookings (realestate_spa:bookings)
- _7 more..._

**Streaming events** via `bytewax`:
`space_registered`, `space_status_changed`, `space_allocated`, `space_deallocated`, `move_created`, ...

**Standalone usage:**
```bash
pip install apg-realestate-spa
apg-realestate-spa --port 8080
```

---

### Tenant Management `realestate_ten`

> Full tenant lifecycle from prospect registration through onboarding (10-step workflow with mandatory-step gating), service request management with SLA enforcement, multi-channel communication portal, satisfaction surveying with automatic review triggers, tenant scoring and credit grading, escalation management, and retention risk analytics.

**Package**: `apg-realestate-ten`  
**Path**: `capabilities/realestate/ten`  
**Version**: 1.0.0  

**Provides:**
- `tenant_onboarding_workflow`
- `tenant_communication_portal`
- `service_request_management`
- `tenant_scoring_engine`
- `satisfaction_tracking`
- `tenant_document_management`
- `tenant_event_timeline`
- `escalation_management`
- `tenant_performance_reporting`
- `tenant_retention_analytics`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `nlpc`
- `mqeb`
- `schd`

**Service methods** (42 total):
`register_tenant`, `get_tenant`, `list_tenants`, `update_tenant`, `activate_tenant`, `blacklist_tenant`, `complete_onboarding_step`, `get_onboarding_progress`, `raise_service_request`, `get_service_request`, `list_service_requests`, `update_service_request`, ...

**Governance rules** (20 total):
`tenant_context_required`, `write_requires_policy`, `tenant_type_supported`, `blacklisted_tenant_activation_denied`, `activation_requires_completed_onboarding`, `service_request_type_supported`, `service_request_requires_tenant`, `sla_breach_triggers_escalation`, ...

**UI Routes** (14):
- `/realestate/ten/dashboard` — dashboard (realestate_ten:view)
- `/realestate/ten/tenants` — tenants (realestate_ten:tenants)
- `/realestate/ten/tenants/<id>` — tenant-detail (realestate_ten:tenants)
- `/realestate/ten/onboarding` — onboarding (realestate_ten:onboarding)
- `/realestate/ten/service-requests` — service-requests (realestate_ten:service_requests)
- `/realestate/ten/communications` — communications (realestate_ten:communications)
- _8 more..._

**Streaming events** via `bytewax`:
`tenant_registered`, `tenant_onboarded`, `tenant_activated`, `tenant_vacated`, `tenant_blacklisted`, ...

**Standalone usage:**
```bash
pip install apg-realestate-ten
apg-realestate-ten --port 8080
```

---

### Property Valuation `realestate_val`

> Full-cycle property valuation: comparable sales database, DCF model builder with range-validated discount rates, mass appraisal engine (regression, spatial, hedonic, AI AVM), valuation roll with automatic supersession, revaluation cycle management, Red Book sign-off enforcement with independent valuer validation, and structured challenge workflow requiring counter-evidence.

**Package**: `apg-realestate-val`  
**Path**: `capabilities/realestate/val`  
**Version**: 1.0.0  

**Provides:**
- `comparable_sales_analysis`
- `dcf_valuation_engine`
- `mass_appraisal_engine`
- `valuation_roll_management`
- `revaluation_cycle_management`
- `valuation_report_generation`
- `yield_analysis`
- `valuer_panel_management`
- `valuation_challenge_workflow`
- `valuation_benchmarking`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `nlpc`
- `comp`
- `mqeb`
- `schd`

**Service methods** (42 total):
`register_valuer`, `get_valuer`, `list_valuers`, `add_comparable`, `list_comparables`, `verify_comparable`, `instruct_valuation`, `get_valuation`, `list_valuations`, `update_valuation`, `sign_off_valuation`, `publish_valuation`, ...

**Governance rules** (21 total):
`tenant_context_required`, `write_requires_policy`, `valuation_method_supported`, `valuation_purpose_supported`, `valuation_requires_property`, `valuation_requires_qualified_valuer`, `red_book_requires_independent_valuer`, `sign_off_requires_approved_valuer_grade`, ...

**UI Routes** (14):
- `/realestate/val/dashboard` — dashboard (realestate_val:view)
- `/realestate/val/valuations` — valuations (realestate_val:valuations)
- `/realestate/val/valuations/<id>` — valuation-detail (realestate_val:valuations)
- `/realestate/val/comparables` — comparables (realestate_val:comparables)
- `/realestate/val/dcf` — dcf-builder (realestate_val:dcf)
- `/realestate/val/mass-appraisal` — mass-appraisal (realestate_val:mass_appraisal)
- _8 more..._

**Streaming events** via `bytewax`:
`valuation_instructed`, `valuation_completed`, `valuation_approved`, `valuation_published`, `comparable_added`, ...

**Standalone usage:**
```bash
pip install apg-realestate-val
apg-realestate-val --port 8080
```

---

## RETAIL

### Loyalty & Rewards `retail_loy`

> Provides end-to-end loyalty programme management for retail tenants: member enrolment with consent and identity verification, points earn/redeem/adjust transactions, tier qualification and downgrade management, coalition partner integration, targeted campaign authoring with approval workflows, a reward catalogue, customer lifetime value (CLV) segmentation, and configurable points-expiry policies. All operations are tenant-isolated, streamed to Bytewax, and governed by 28 deterministic rules.

**Package**: `apg-retail-loy`  
**Path**: `capabilities/retail/loy`  
**Version**: 1.0.0  

**Provides:**
- `loyalty_member_enrolment`
- `loyalty_points_earn`
- `loyalty_points_redeem`
- `loyalty_tier_management`
- `loyalty_campaign_management`
- `loyalty_partner_coalition`
- `loyalty_clv_analytics`
- `loyalty_expiry_management`
- `loyalty_reward_catalogue`
- `loyalty_transaction_ledger`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `mqeb`
- `moni`
- `schd`

**Service methods** (44 total):
`create_programme`, `get_programme`, `list_programmes`, `enrol_member`, `get_member`, `get_member_by_number`, `update_member`, `list_members`, `freeze_member`, `reactivate_member`, `create_tier`, `list_tiers`, ...

**Governance rules** (28 total):
`tenant_context_required`, `write_requires_policy`, `enrolment_requires_consent`, `enrolment_requires_identity`, `programme_type_supported`, `earn_requires_receipt`, `earn_requires_valid_amount`, `earn_exceeds_max_denied`, ...

**UI Routes** (14):
- `/retail-loy/dashboard` — dashboard (retail_loy:view)
- `/retail-loy/members` — members (retail_loy:view)
- `/retail-loy/members/<id>` — member_detail (retail_loy:view)
- `/retail-loy/members/enrol` — enrolment (retail_loy:write)
- `/retail-loy/transactions` — transactions (retail_loy:view)
- `/retail-loy/earn` — earn (retail_loy:write)
- _8 more..._

**Streaming events** via `bytewax`:
`member_enrolled`, `points_earned`, `points_redeemed`, `points_expired`, `points_adjusted`, ...

**Standalone usage:**
```bash
pip install apg-retail-loy
apg-retail-loy --port 8080
```

---

### Omnichannel Commerce `retail_omc`

> Provides unified commerce orchestration across all retail touchpoints: channel registry, cross-channel inventory visibility with reservation TTL, unified cart and order management, buy-online-pickup-in-store (BOPIS/C&C), ship-from-store fulfilment, multi-channel returns, customer journey event tracking with attribution, cross-channel pricing rules, and fraud screening integration. All operations are tenant-isolated and streamed to Bytewax.

**Package**: `apg-retail-omc`  
**Path**: `capabilities/retail/omc`  
**Version**: 1.0.0  

**Provides:**
- `omnichannel_order_management`
- `inventory_visibility`
- `click_and_collect`
- `customer_journey_orchestration`
- `unified_cart`
- `cross_channel_fulfilment`
- `omnichannel_search`
- `return_management`
- `channel_pricing_engine`
- `session_attribution`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `mqeb`
- `moni`
- `nlpc`
- `schd`

**Service methods** (42 total):
`create_channel`, `get_channel`, `list_channels`, `create_catalogue_item`, `get_catalogue_item`, `get_catalogue_item_by_sku`, `list_catalogue_items`, `set_channel_price`, `unified_inventory_check`, `upsert_inventory`, `get_inventory`, `reserve_inventory`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `channel_type_supported`, `order_requires_channel`, `order_fulfilment_mode_supported`, `oversell_denied`, `payment_method_supported`, `payment_requires_fraud_check`, ...

**UI Routes** (14):
- `/retail-omc/dashboard` — dashboard (retail_omc:view)
- `/retail-omc/orders` — orders (retail_omc:view)
- `/retail-omc/orders/<id>` — order_detail (retail_omc:view)
- `/retail-omc/orders/create` — order_create (retail_omc:write)
- `/retail-omc/inventory` — inventory (retail_omc:view)
- `/retail-omc/channels` — channels (retail_omc:admin)
- _8 more..._

**Streaming events** via `bytewax`:
`order_created`, `order_paid`, `order_shipped`, `order_delivered`, `order_collected`, ...

**Standalone usage:**
```bash
pip install apg-retail-omc
apg-retail-omc --port 8080
```

---

### Point of Sale `retail_pos`

> Provides complete POS transaction processing for physical retail: terminal registration and heartbeat monitoring, session lifecycle management with opening float and reconciliation enforcement, sale/refund/void/exchange transaction posting with automated total calculation and digital signing, cash event recording with safe-drop thresholds, till reconciliation with variance reporting and approval, multi-format receipt issuance, and full offline resilience with configurable floor limits and store-and-forward queuing.

**Package**: `apg-retail-pos`  
**Path**: `capabilities/retail/pos`  
**Version**: 1.0.0  

**Provides:**
- `pos_transaction_processing`
- `pos_session_management`
- `pos_cash_management`
- `pos_till_reconciliation`
- `pos_receipt_management`
- `pos_discount_management`
- `pos_offline_resilience`
- `pos_payment_processing`
- `pos_void_management`
- `pos_audit_trail`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `mqeb`
- `moni`
- `comp`

**Service methods** (87 total):
`model_dump`, `model_dump`, `model_dump`, `model_dump`, `model_dump`, `put`, `get_item`, `tenant_values`, `all_values`, `set_stock`, `get_stock`, `adjust_stock`, ...

**Governance rules** (24 total):
`tenant_context_required`, `write_requires_policy`, `transaction_requires_open_session`, `terminal_type_supported`, `transaction_type_supported`, `payment_method_supported`, `unsigned_transaction_denied`, `void_requires_reason`, ...

**UI Routes** (14):
- `/retail-pos/dashboard` — dashboard (retail_pos:view)
- `/retail-pos/terminal` — terminal (retail_pos:transact)
- `/retail-pos/sessions` — sessions (retail_pos:view)
- `/retail-pos/sessions/<id>` — session_detail (retail_pos:view)
- `/retail-pos/transactions` — transactions (retail_pos:view)
- `/retail-pos/transactions/<id>` — transaction_detail (retail_pos:view)
- _8 more..._

**Streaming events** via `bytewax`:
`session_opened`, `session_closed`, `transaction_posted`, `refund_posted`, `void_posted`, ...

**Standalone usage:**
```bash
pip install apg-retail-pos
apg-retail-pos --port 8080
```

---

### Promotions Management `retail_prm`

> Provides complete promotion lifecycle management: authoring 12 promotion types with multi-trigger conditions, an approval workflow, stack-policy enforcement, budget and margin-floor governance, coupon issuance and redemption with expiry validation, channel and audience targeting, clearance/markdown optimisation with cascade support, real-time budget tracking, and promotion effectiveness analytics. All operations are tenant-isolated and governed by 24 deterministic rules.

**Package**: `apg-retail-prm`  
**Path**: `capabilities/retail/prm`  
**Version**: 1.0.0  

**Provides:**
- `promotion_authoring`
- `promotion_activation`
- `pricing_rules_engine`
- `coupon_management`
- `coupon_redemption`
- `markdown_optimisation`
- `promotion_effectiveness_analytics`
- `audience_targeting`
- `promotion_budget_management`
- `promotion_stacking_engine`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `mqeb`
- `moni`
- `nlpc`
- `schd`

**Service methods** (42 total):
`create_promotion`, `get_promotion`, `update_promotion`, `activate_promotion`, `check_promotion_eligibility`, `apply_promotion_to_cart`, `promotion_stacking_rules`, `submit_for_approval`, `approve_promotion`, `reject_promotion`, `pause_promotion`, `list_promotions`, ...

**Governance rules** (24 total):
`tenant_context_required`, `write_requires_policy`, `promotion_type_supported`, `promotion_requires_end_date`, `promotion_requires_budget_cap`, `unapproved_activation_denied`, `budget_exceeded_denied`, `margin_floor_breach_denied`, ...

**UI Routes** (14):
- `/retail-prm/dashboard` — dashboard (retail_prm:view)
- `/retail-prm/promotions` — promotions (retail_prm:view)
- `/retail-prm/promotions/<id>` — promotion_detail (retail_prm:view)
- `/retail-prm/promotions/create` — promotion_create (retail_prm:write)
- `/retail-prm/coupons` — coupons (retail_prm:view)
- `/retail-prm/coupons/create` — coupon_create (retail_prm:write)
- _8 more..._

**Streaming events** via `bytewax`:
`promotion_created`, `promotion_approved`, `promotion_activated`, `promotion_paused`, `promotion_expired`, ...

**Standalone usage:**
```bash
pip install apg-retail-prm
apg-retail-prm --port 8080
```

---

### Store Intelligence `retail_sin`

> Provides anonymised in-store analytics: foot traffic counting with multi-sensor support, zone-level dwell time and heatmap generation, AI-assisted planogram compliance auditing, real-time shelf availability alerting with automatic replenishment triggering, shopper conversion funnel tracking, store KPI scorecards with peer-group benchmarking, and a store performance dashboard. All personal data is anonymised at ingest; raw video storage and biometric identification are denied by rule engine.

**Package**: `apg-retail-sin`  
**Path**: `capabilities/retail/sin`  
**Version**: 1.0.0  

**Provides:**
- `store_foot_traffic_analytics`
- `planogram_compliance_monitoring`
- `shelf_availability_alerting`
- `store_conversion_optimisation`
- `store_performance_benchmarking`
- `zone_heatmap_analytics`
- `store_kpi_reporting`
- `replenishment_triggering`
- `shopper_journey_analytics`
- `store_format_benchmarking`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `mqeb`
- `moni`
- `nlpc`
- `schd`
- `geos`

**Service methods** (42 total):
`create_store`, `get_store`, `get_store_by_code`, `list_stores`, `create_zone`, `list_zones`, `register_sensor`, `sensor_heartbeat`, `list_sensors`, `foot_traffic_record`, `conversion_rate`, `record_traffic_count`, ...

**Governance rules** (22 total):
`tenant_context_required`, `write_requires_policy`, `pii_anonymisation_required`, `raw_video_storage_denied`, `biometric_id_denied`, `sensor_type_supported`, `zone_type_supported`, `store_location_required`, ...

**UI Routes** (14):
- `/retail-sin/dashboard` — dashboard (retail_sin:view)
- `/retail-sin/traffic` — traffic (retail_sin:view)
- `/retail-sin/heatmaps` — heatmaps (retail_sin:view)
- `/retail-sin/planogram` — planogram (retail_sin:view)
- `/retail-sin/planogram/<id>` — planogram_detail (retail_sin:view)
- `/retail-sin/shelf-alerts` — shelf_alerts (retail_sin:view)
- _8 more..._

**Streaming events** via `bytewax`:
`traffic_count_recorded`, `zone_dwell_recorded`, `planogram_audit_completed`, `planogram_deviation_detected`, `shelf_alert_raised`, ...

**Standalone usage:**
```bash
pip install apg-retail-sin
apg-retail-sin --port 8080
```

---

## SCM

### Vendor Management `scm_ven`

> Vendor Management is the APG capability for the full supplier lifecycle — from initial prospecting through qualification, onboarding, active relationship management, and eventual offboarding. It consolidates vendor master records, performance tracking, risk governance, compliance evidence, contract management, communication logging, self-service portal access, and AI-generated scorecards into a single coherent service boundary.

**Package**: `apg-scm-ven`  
**Path**: `capabilities/scm/ven`  
**Version**: 2.1.0  

**Provides:**
- `vendor_profile_lifecycle`
- `vendor_onboarding_workflow`
- `vendor_qualification_lifecycle`
- `vendor_performance_lifecycle`
- `vendor_risk_lifecycle`
- `vendor_contract_lifecycle`
- `vendor_compliance_lifecycle`
- `vendor_communication_lifecycle`
- `vendor_portal_lifecycle`
- `vendor_scorecard_service`
- `vendor_sourcing_integration`
- `vendor_agents`

**Requires:**
- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `wflo`
- `grc_doc`
- `grc_doc`
- `grc_rsa`
- `mdm`

**Service methods** (40 total):
`onboard_vendor`, `vendor_qualification`, `vendor_performance_score`, `approved_vendor_list`, `vendor_suspension`, `contract_management`, `preferred_vendor_designation`, `spend_analysis`, `vendor_risk_assessment`, `vendor_portal_access`, `create_vendor`, `qualify_vendor`, ...

**Governance rules** (52 total):
`tenant_context_required`, `operation_policy_required`, `vendor_code_required`, `vendor_name_required`, `vendor_type_supported`, `vendor_category_required`, `vendor_country_required`, `vendor_owner_required`, ...

**UI Routes** (14):
- `/scm/vendors/dashboard` — dashboard (scm_ven:view)
- `/scm/vendors` — vendors (scm_ven:manage_vendors)
- `/scm/vendors/qualification` — qualification (scm_ven:qualify)
- `/scm/vendors/onboarding` — onboarding (scm_ven:onboard)
- `/scm/vendors/performance` — performance (scm_ven:score)
- `/scm/vendors/risk` — risk (scm_ven:govern)
- _8 more..._

**Streaming events** via `bytewax`:
`vendor_created`, `vendor_qualified`, `vendor_onboarded`, `vendor_performance_recorded`, `vendor_risk_recorded`, ...

**Standalone usage:**
```bash
pip install apg-scm-ven
apg-scm-ven --port 8080
```

---

## TELECOM

### Telecom Analytics `telecom_ana`

> Provides network performance analytics, churn prediction, ARPU analysis, usage pattern analytics, and revenue assurance for telecom operators. Integrates with ML model management to surface predictive insights, anomaly detection, and customer segmentation across all network layers.

**Package**: `apg-telecom-ana`  
**Path**: `capabilities/telecom/ana`  
**Version**: 1.0.0  

**Provides:**
- `analytics_pipeline`
- `churn_prediction_workflow`
- `arpu_analysis_workflow`
- `usage_pattern_workflow`
- `revenue_assurance_workflow`
- `network_performance_analytics`
- `customer_segmentation_workflow`
- `anomaly_detection_workflow`
- `model_management_workflow`
- `analytics_reporting_workflow`
- `analytics_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `nlpc`
- `moni`
- `mqeb`
- `schd`

**Service methods** (42 total):
`describe`, `evaluate`, `record_analysis_run`, `record_metric`, `record_churn_prediction`, `record_revenue_event`, `record_segment`, `record_network_analytics`, `record_anomaly`, `register_model`, `generate_report`, `register_agent`, ...

**Governance rules** (30 total):
`tenant_context_required`, `ana_write_requires_policy`, `analysis_type_supported`, `analysis_owner_required`, `analysis_evidence_required`, `metric_type_supported`, `metric_baseline_required`, `churn_risk_level_supported`, ...

**UI Routes** (12):
- `/telecom-ana/dashboard` — dashboard (telecom_ana:view)
- `/telecom-ana/analysis` — analysis (telecom_ana:analysis)
- `/telecom-ana/metrics` — metrics (telecom_ana:metrics)
- `/telecom-ana/churn` — churn (telecom_ana:churn)
- `/telecom-ana/revenue` — revenue (telecom_ana:revenue)
- `/telecom-ana/segments` — segments (telecom_ana:segments)
- _6 more..._

**Streaming events** via `bytewax`:
`analysis_run_recorded`, `metric_recorded`, `churn_prediction_recorded`, `revenue_assurance_event_recorded`, `segment_recorded`, ...

**Standalone usage:**
```bash
pip install apg-telecom-ana
apg-telecom-ana --port 8080
```

---

### Telecom Billing `telecom_bil`

> Convergent billing capability covering the full billing stack: CDR mediation and normalisation, real-time rating and charging, bill cycle management, invoice generation and approval, dunning workflow, payment reconciliation, discount management, and convergent billing for households and corporate groups.

**Package**: `apg-telecom-bil`  
**Path**: `capabilities/telecom/bil`  
**Version**: 1.0.0  

**Provides:**
- `mediation_workflow`
- `rating_workflow`
- `charging_workflow`
- `invoice_workflow`
- `bill_cycle_management`
- `dunning_workflow`
- `payment_reconciliation_workflow`
- `discount_workflow`
- `convergent_billing_workflow`
- `billing_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `mqeb`
- `schd`
- `comp`

**Service methods** (73 total):
`get`, `put`, `delete`, `scan`, `emit`, `send`, `check`, `get`, `put`, `delete`, `scan`, `emit`, ...

**Governance rules** (27 total):
`tenant_context_required`, `bil_write_requires_policy`, `mediation_status_supported`, `cdr_source_required`, `rating_type_supported`, `charge_type_supported`, `charge_amount_positive`, `bill_cycle_type_supported`, ...

**UI Routes** (12):
- `/telecom-bil/dashboard` — dashboard (telecom_bil:view)
- `/telecom-bil/mediation` — mediation (telecom_bil:mediation)
- `/telecom-bil/rating` — rating (telecom_bil:rating)
- `/telecom-bil/bill-cycles` — bill_cycles (telecom_bil:bill_cycles)
- `/telecom-bil/invoices` — invoices (telecom_bil:invoices)
- `/telecom-bil/dunning` — dunning (telecom_bil:dunning)
- _6 more..._

**Streaming events** via `bytewax`:
`cdr_mediated`, `charge_rated`, `invoice_generated`, `invoice_approved`, `invoice_sent`, ...

**Standalone usage:**
```bash
pip install apg-telecom-bil
apg-telecom-bil --port 8080
```

---

### Customer Management `telecom_cus`

> End-to-end customer lifecycle management covering onboarding, KYC verification, plan activation, SIM and device management, and customer service case tracking. Enforces KYC requirements, credit checks for postpaid plans, IMEI blacklist checks, and tenant-scoped PII access controls.

**Package**: `apg-telecom-cus`  
**Path**: `capabilities/telecom/cus`  
**Version**: 1.0.0  

**Provides:**
- `customer_lifecycle_workflow`
- `kyc_workflow`
- `plan_management_workflow`
- `sim_management_workflow`
- `device_management_workflow`
- `case_tracking_workflow`
- `customer_360_view`
- `churn_management_workflow`
- `cus_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `nlpc`
- `mqeb`
- `comp`

**Service methods** (42 total):
`describe`, `evaluate`, `create_customer`, `update_customer_status`, `submit_kyc_document`, `verify_kyc`, `reject_kyc`, `activate_plan`, `provision_sim`, `update_sim_status`, `register_device`, `open_case`, ...

**Governance rules** (25 total):
`tenant_context_required`, `cus_write_requires_policy`, `customer_type_supported`, `customer_kyc_required`, `customer_msisdn_required`, `kyc_document_type_supported`, `kyc_bypass_denied`, `plan_type_supported`, ...

**UI Routes** (12):
- `/telecom-cus/dashboard` — dashboard (telecom_cus:view)
- `/telecom-cus/customers` — customers (telecom_cus:customers)
- `/telecom-cus/customers/<id>` — customer_detail (telecom_cus:customers)
- `/telecom-cus/kyc` — kyc (telecom_cus:kyc)
- `/telecom-cus/plans` — plans (telecom_cus:plans)
- `/telecom-cus/sims` — sims (telecom_cus:sims)
- _6 more..._

**Streaming events** via `bytewax`:
`customer_onboarded`, `kyc_verified`, `kyc_rejected`, `plan_activated`, `plan_changed`, ...

**Standalone usage:**
```bash
pip install apg-telecom-cus
apg-telecom-cus --port 8080
```

---

### Network Inventory `telecom_inv`

> Physical and logical network inventory management covering asset commissioning and decommissioning, circuit provisioning, IP address management (IPAM), network topology documentation, and automated reconciliation with field audit results. Provides the single source of truth for all network resources.

**Package**: `apg-telecom-inv`  
**Path**: `capabilities/telecom/inv`  
**Version**: 1.0.0  

**Provides:**
- `asset_inventory_workflow`
- `circuit_management_workflow`
- `ipam_workflow`
- `topology_documentation_workflow`
- `inventory_reconciliation_workflow`
- `network_resource_query`
- `inv_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `nlpc`
- `moni`
- `mqeb`
- `geos`

**Service methods** (42 total):
`describe`, `evaluate`, `commission_asset`, `update_asset_status`, `decommission_asset`, `provision_circuit`, `update_circuit_status`, `allocate_ip_block`, `release_ip_block`, `record_topology`, `register_site`, `record_discrepancy`, ...

**Governance rules** (23 total):
`tenant_context_required`, `inv_write_requires_policy`, `asset_type_supported`, `asset_serial_number_required`, `asset_location_required`, `asset_status_supported`, `decommission_requires_approval`, `circuit_type_supported`, ...

**UI Routes** (11):
- `/telecom-inv/dashboard` — dashboard (telecom_inv:view)
- `/telecom-inv/assets` — assets (telecom_inv:assets)
- `/telecom-inv/assets/<id>` — asset_detail (telecom_inv:assets)
- `/telecom-inv/circuits` — circuits (telecom_inv:circuits)
- `/telecom-inv/ipam` — ipam (telecom_inv:ipam)
- `/telecom-inv/topology` — topology (telecom_inv:topology)
- _5 more..._

**Streaming events** via `bytewax`:
`asset_commissioned`, `asset_decommissioned`, `circuit_provisioned`, `circuit_decommissioned`, `ip_block_allocated`, ...

**Standalone usage:**
```bash
pip install apg-telecom-inv
apg-telecom-inv --port 8080
```

---

### Network Management `telecom_net`

> Network operations centre capability providing fault management with alarm correlation, performance monitoring with threshold alerting, configuration change management with freeze period enforcement, SLA monitoring, and NOC shift handover management. Designed for 24×7 NOC operations with a dark-themed UI.

**Package**: `apg-telecom-net`  
**Path**: `capabilities/telecom/net`  
**Version**: 1.0.0  

**Provides:**
- `fault_management_workflow`
- `performance_management_workflow`
- `configuration_management_workflow`
- `sla_monitoring_workflow`
- `noc_operations_workflow`
- `alarm_correlation_workflow`
- `change_management_workflow`
- `net_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `mqeb`
- `schd`

**Service methods** (42 total):
`describe`, `evaluate`, `raise_alarm`, `update_alarm_status`, `suppress_alarm`, `open_fault_ticket`, `resolve_fault_ticket`, `escalate_fault`, `record_performance`, `submit_config_change`, `complete_config_change`, `record_sla`, ...

**Governance rules** (24 total):
`tenant_context_required`, `net_write_requires_policy`, `fault_severity_supported`, `fault_category_supported`, `alarm_ne_required`, `alarm_status_supported`, `alarm_suppression_requires_approval`, `performance_metric_supported`, ...

**UI Routes** (12):
- `/telecom-net/dashboard` — dashboard (telecom_net:view)
- `/telecom-net/alarms` — alarms (telecom_net:faults)
- `/telecom-net/faults` — fault_tickets (telecom_net:faults)
- `/telecom-net/performance` — performance (telecom_net:performance)
- `/telecom-net/config-changes` — config_changes (telecom_net:config)
- `/telecom-net/sla` — sla (telecom_net:sla)
- _6 more..._

**Streaming events** via `bytewax`:
`alarm_raised`, `alarm_cleared`, `fault_ticket_opened`, `fault_ticket_resolved`, `performance_threshold_breached`, ...

**Standalone usage:**
```bash
pip install apg-telecom-net
apg-telecom-net --port 8080
```

---

### Order Management `telecom_ord`

> End-to-end service order management covering order capture, validation, decomposition into provisioning tasks, orchestration, fallout management, number portability, bulk order processing, and real-time order tracking. Enforces duplicate detection and requires explicit approval for bulk operations.

**Package**: `apg-telecom-ord`  
**Path**: `capabilities/telecom/ord`  
**Version**: 1.0.0  

**Provides:**
- `order_capture_workflow`
- `order_validation_workflow`
- `order_decomposition_workflow`
- `provisioning_orchestration_workflow`
- `fallout_management_workflow`
- `order_tracking_workflow`
- `number_portability_workflow`
- `ord_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `mqeb`
- `schd`
- `comp`

**Service methods** (40 total):
`describe`, `evaluate`, `submit_order`, `validate_order`, `decompose_order`, `create_task`, `complete_task`, `record_fallout`, `retry_fallout`, `resolve_fallout`, `complete_order`, `submit_portability_request`, ...

**Governance rules** (23 total):
`tenant_context_required`, `ord_write_requires_policy`, `order_type_supported`, `order_channel_supported`, `order_priority_supported`, `duplicate_order_denied`, `order_customer_required`, `order_status_supported`, ...

**UI Routes** (12):
- `/telecom-ord/dashboard` — dashboard (telecom_ord:view)
- `/telecom-ord/orders` — orders (telecom_ord:orders)
- `/telecom-ord/orders/<id>` — order_detail (telecom_ord:orders)
- `/telecom-ord/decomposition` — decomposition (telecom_ord:decomposition)
- `/telecom-ord/tasks` — tasks (telecom_ord:tasks)
- `/telecom-ord/fallout` — fallout (telecom_ord:fallout)
- _6 more..._

**Streaming events** via `bytewax`:
`order_submitted`, `order_validated`, `order_decomposed`, `task_completed`, `order_fallout`, ...

**Standalone usage:**
```bash
pip install apg-telecom-ord
apg-telecom-ord --port 8080
```

---

### Performance Management `telecom_per`

> Telecom network performance management covering KPI monitoring across all network layers, SLA compliance tracking with breach alerting, capacity utilisation forecasting, trend analysis with ML-based predictions, configurable threshold management, benchmark gap analysis, and scheduled performance reporting.

**Package**: `apg-telecom-per`  
**Path**: `capabilities/telecom/per`  
**Version**: 1.0.0  

**Provides:**
- `kpi_monitoring_workflow`
- `sla_compliance_workflow`
- `capacity_utilisation_workflow`
- `trend_reporting_workflow`
- `performance_reporting_workflow`
- `threshold_management_workflow`
- `benchmark_analysis_workflow`
- `per_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `moni`
- `mqeb`
- `schd`
- `nlpc`

**Service methods** (42 total):
`describe`, `evaluate`, `record_kpi`, `update_kpi_status`, `record_sla_compliance`, `record_capacity`, `record_trend`, `set_threshold`, `record_benchmark`, `generate_report`, `register_agent`, `kpi_collection`, ...

**Governance rules** (22 total):
`tenant_context_required`, `per_write_requires_policy`, `kpi_category_supported`, `kpi_baseline_required`, `kpi_status_supported`, `sla_compliance_status_supported`, `sla_breach_notification_required`, `capacity_state_supported`, ...

**UI Routes** (12):
- `/telecom-per/dashboard` — dashboard (telecom_per:view)
- `/telecom-per/kpis` — kpis (telecom_per:kpis)
- `/telecom-per/kpis/<id>` — kpi_detail (telecom_per:kpis)
- `/telecom-per/sla` — sla_compliance (telecom_per:sla)
- `/telecom-per/capacity` — capacity (telecom_per:capacity)
- `/telecom-per/trends` — trends (telecom_per:trends)
- _6 more..._

**Streaming events** via `bytewax`:
`kpi_threshold_breached`, `sla_breach_detected`, `capacity_congestion_alert`, `trend_degradation_detected`, `report_generated`, ...

**Standalone usage:**
```bash
pip install apg-telecom-per
apg-telecom-per --port 8080
```

---

### Service Provisioning `telecom_pro`

> Service activation and provisioning engine covering workflow orchestration, network resource reservation, configuration push to network elements via multiple protocols (NETCONF, RESTCONF, CLI, REST API), end-to-end activation verification, automated rollback on failure, and bulk provisioning with pre-approval gating.

**Package**: `apg-telecom-pro`  
**Path**: `capabilities/telecom/pro`  
**Version**: 1.0.0  

**Provides:**
- `service_activation_workflow`
- `network_resource_allocation`
- `configuration_push_workflow`
- `activation_confirmation_workflow`
- `rollback_workflow`
- `bulk_provisioning_workflow`
- `pro_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `mqeb`
- `moni`
- `schd`

**Service methods** (49 total):
`describe`, `evaluate`, `start_workflow`, `update_workflow_status`, `reserve_resource`, `release_resource`, `push_config`, `confirm_activation`, `trigger_rollback`, `complete_rollback`, `start_bulk_provisioning`, `register_agent`, ...

**Governance rules** (21 total):
`tenant_context_required`, `pro_write_requires_policy`, `workflow_type_supported`, `workflow_order_required`, `workflow_status_supported`, `resource_type_supported`, `resource_conflict_check_required`, `config_push_method_supported`, ...

**UI Routes** (12):
- `/telecom-pro/dashboard` — dashboard (telecom_pro:view)
- `/telecom-pro/workflows` — workflows (telecom_pro:workflows)
- `/telecom-pro/workflows/<id>` — workflow_detail (telecom_pro:workflows)
- `/telecom-pro/resources` — resources (telecom_pro:resources)
- `/telecom-pro/config-push` — config_push (telecom_pro:config_push)
- `/telecom-pro/network-elements` — network_elements (telecom_pro:network_elements)
- _6 more..._

**Streaming events** via `bytewax`:
`workflow_queued`, `resource_reserved`, `config_push_dispatched`, `config_push_completed`, `service_activated`, ...

**Standalone usage:**
```bash
pip install apg-telecom-pro
apg-telecom-pro --port 8080
```

---

### Quality of Service `telecom_qos`

> QoS policy management and enforcement covering bearer QoS, traffic shaping and policing, SLA parameter measurement, real-time degradation detection with root cause analysis, automated and manual remediation workflows, and PCRF/PCEF integration for policy enforcement on network elements.

**Package**: `apg-telecom-qos`  
**Path**: `capabilities/telecom/qos`  
**Version**: 1.0.0  

**Provides:**
- `qos_policy_management_workflow`
- `traffic_prioritisation_workflow`
- `sla_enforcement_workflow`
- `degradation_detection_workflow`
- `root_cause_analysis_workflow`
- `auto_remediation_workflow`
- `qos_reporting_workflow`
- `qos_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `moni`
- `mqeb`
- `wflo`

**Service methods** (42 total):
`describe`, `evaluate`, `create_qos_policy`, `change_qos_policy`, `classify_traffic`, `update_enforcement_status`, `record_sla_measurement`, `record_degradation`, `record_root_cause`, `trigger_remediation`, `complete_remediation`, `register_agent`, ...

**Governance rules** (23 total):
`tenant_context_required`, `qos_write_requires_policy`, `qos_policy_type_supported`, `qos_class_supported`, `qos_policy_approval_required`, `qos_conflict_check_required`, `traffic_type_supported`, `traffic_classification_required`, ...

**UI Routes** (12):
- `/telecom-qos/dashboard` — dashboard (telecom_qos:view)
- `/telecom-qos/policies` — policies (telecom_qos:policies)
- `/telecom-qos/policies/<id>` — policy_detail (telecom_qos:policies)
- `/telecom-qos/traffic` — traffic (telecom_qos:traffic)
- `/telecom-qos/enforcement` — enforcement (telecom_qos:enforcement)
- `/telecom-qos/sla` — sla_monitoring (telecom_qos:sla)
- _6 more..._

**Streaming events** via `bytewax`:
`qos_policy_activated`, `qos_policy_changed`, `sla_breach_detected`, `degradation_detected`, `root_cause_identified`, ...

**Standalone usage:**
```bash
pip install apg-telecom-qos
apg-telecom-qos --port 8080
```

---

### Telecom Security `telecom_sec`

> Provides comprehensive telecom security management covering fraud detection (WANGIRI, IRSF, SIM swap), SS7/Diameter signalling security, roaming security, VoIP fraud detection, lawful intercept management, security incident response, and threat intelligence sharing. Enforces strict warrant and evidence requirements throughout.

**Package**: `apg-telecom-sec`  
**Path**: `capabilities/telecom/sec`  
**Version**: 1.0.0  

**Provides:**
- `fraud_management_workflow`
- `ss7_security_workflow`
- `diameter_security_workflow`
- `lawful_intercept_workflow`
- `security_incident_workflow`
- `threat_intel_workflow`
- `voip_fraud_detection_workflow`
- `roaming_security_workflow`
- `sec_agent_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `mqeb`
- `comp`

**Service methods** (42 total):
`describe`, `evaluate`, `raise_fraud_case`, `apply_fraud_block`, `record_ss7_attack`, `record_diameter_attack`, `activate_intercept`, `update_intercept_status`, `open_incident`, `update_incident_status`, `record_threat_intel`, `register_agent`, ...

**Governance rules** (25 total):
`tenant_context_required`, `sec_write_requires_policy`, `fraud_type_supported`, `fraud_block_requires_evidence`, `fraud_confidence_required`, `ss7_attack_type_supported`, `ss7_evidence_required`, `diameter_attack_type_supported`, ...

**UI Routes** (12):
- `/telecom-sec/dashboard` — dashboard (telecom_sec:view)
- `/telecom-sec/fraud` — fraud_queue (telecom_sec:fraud)
- `/telecom-sec/fraud-rules` — fraud_rules (telecom_sec:fraud_rules)
- `/telecom-sec/ss7` — ss7_security (telecom_sec:ss7)
- `/telecom-sec/diameter` — diameter_security (telecom_sec:diameter)
- `/telecom-sec/intercept` — lawful_intercept (telecom_sec:intercept)
- _6 more..._

**Streaming events** via `bytewax`:
`fraud_case_raised`, `fraud_block_applied`, `ss7_attack_detected`, `diameter_attack_detected`, `intercept_activated`, ...

**Standalone usage:**
```bash
pip install apg-telecom-sec
apg-telecom-sec --port 8080
```

---

## TRANSPORT

### Cargo Management `transport_car`

> The Cargo Management capability provides end-to-end cargo lifecycle management including booking creation, manifest generation, dangerous goods compliance, real-time cargo tracking, and revenue management. It enforces IATA, IMDG, ADR, and C-TPAT compliance standards and integrates with bytewax for streaming cargo lifecycle events.

**Package**: `apg-transport-car`  
**Path**: `capabilities/transport/car`  
**Version**: 1.0.0  

**Provides:**
- `cargo_booking_workflow`
- `cargo_manifest_workflow`
- `dangerous_goods_compliance_workflow`
- `cargo_tracking_workflow`
- `cargo_revenue_workflow`
- `cargo_compliance_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `comp`
- `mqeb`
- `schd`

**Service methods** (45 total):
`describe`, `evaluate`, `create_booking`, `create_manifest`, `declare_dangerous_goods`, `update_tracking`, `record_revenue`, `record_compliance`, `register_cargo_agent`, `validate_batch`, `cancel_booking`, `get_booking`, ...

**Governance rules** (26 total):
`tenant_context_required`, `cargo_write_requires_policy`, `booking_shipper_required`, `booking_consignee_required`, `booking_origin_required`, `booking_destination_required`, `booking_weight_required`, `booking_cargo_type_supported`, ...

**UI Routes** (13):
- `/transport-cargo/dashboard` — dashboard (transport_car:view)
- `/transport-cargo/bookings` — bookings (transport_car:bookings)
- `/transport-cargo/bookings/create` — booking_create (transport_car:bookings_write)
- `/transport-cargo/manifests` — manifests (transport_car:manifests)
- `/transport-cargo/dangerous-goods` — dangerous_goods (transport_car:dg_compliance)
- `/transport-cargo/tracking` — tracking (transport_car:tracking)
- _7 more..._

**Streaming events** via `bytewax`:
`cargo_booked`, `cargo_manifest_submitted`, `cargo_dg_declared`, `cargo_tracking_updated`, `cargo_delivered`, ...

**Standalone usage:**
```bash
pip install apg-transport-car
apg-transport-car --port 8080
```

---

### Delivery Management `transport_del`

> The Delivery Management capability handles last-mile delivery planning, proof-of-delivery capture, customer notifications, failed delivery handling, rescheduling workflows, SLA tracking, and return management. It enforces geo-stamped POD capture and protects against POD falsification.

**Package**: `apg-transport-del`  
**Path**: `capabilities/transport/del`  
**Version**: 1.0.0  

**Provides:**
- `delivery_planning_workflow`
- `proof_of_delivery_workflow`
- `customer_notification_workflow`
- `failed_delivery_workflow`
- `sla_tracking_workflow`
- `delivery_return_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `comp`
- `mqeb`
- `schd`

**Service methods** (44 total):
`describe`, `evaluate`, `create_delivery`, `record_pod`, `record_failed_delivery`, `reschedule_delivery`, `set_sla`, `send_notification`, `create_return`, `register_delivery_agent`, `validate_batch`, `get_delivery`, ...

**Governance rules** (24 total):
`tenant_context_required`, `delivery_write_requires_policy`, `delivery_type_supported`, `delivery_address_required`, `delivery_time_window_required`, `delivery_recipient_required`, `pod_type_supported`, `pod_delivery_required`, ...

**UI Routes** (12):
- `/transport-delivery/dashboard` — dashboard (transport_del:view)
- `/transport-delivery/deliveries` — deliveries (transport_del:deliveries)
- `/transport-delivery/deliveries/create` — delivery_create (transport_del:deliveries_write)
- `/transport-delivery/pod` — proof_of_delivery (transport_del:pod)
- `/transport-delivery/failed` — failed_deliveries (transport_del:failed)
- `/transport-delivery/rescheduling` — rescheduling (transport_del:rescheduling)
- _6 more..._

**Streaming events** via `bytewax`:
`delivery_created`, `delivery_assigned`, `delivery_out_for_delivery`, `delivery_completed`, `delivery_failed`, ...

**Standalone usage:**
```bash
pip install apg-transport-del
apg-transport-del --port 8080
```

---

### Dispatch Operations `transport_dis`

> The Dispatch Operations capability manages load planning, driver assignment with hours-of-service compliance, dispatch optimisation, real-time GPS tracking updates, and exception management. It enforces vehicle capacity limits, driver hours regulations, and provides multi-channel driver communication.

**Package**: `apg-transport-dis`  
**Path**: `capabilities/transport/dis`  
**Version**: 1.0.0  

**Provides:**
- `load_planning_workflow`
- `driver_assignment_workflow`
- `dispatch_optimisation_workflow`
- `real_time_tracking_workflow`
- `exception_management_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `schd`
- `mqeb`
- `nlpc`

**Service methods** (43 total):
`describe`, `evaluate`, `plan_load`, `assign_driver`, `create_dispatch`, `update_dispatch_status`, `update_tracking`, `raise_exception`, `resolve_exception`, `send_communication`, `register_dispatch_agent`, `validate_batch`, ...

**Governance rules** (22 total):
`tenant_context_required`, `dispatch_write_requires_policy`, `load_type_supported`, `load_vehicle_capacity_required`, `overload_dispatch_denied`, `driver_assignment_type_supported`, `driver_hours_of_service_check`, `driver_licence_required`, ...

**UI Routes** (12):
- `/transport-dispatch/dashboard` — dashboard (transport_dis:view)
- `/transport-dispatch/loads` — loads (transport_dis:loads)
- `/transport-dispatch/loads/create` — load_create (transport_dis:loads_write)
- `/transport-dispatch/board` — dispatch_board (transport_dis:dispatch)
- `/transport-dispatch/drivers` — driver_assignment (transport_dis:drivers)
- `/transport-dispatch/tracking` — tracking (transport_dis:tracking)
- _6 more..._

**Streaming events** via `bytewax`:
`load_planned`, `driver_assigned`, `dispatch_created`, `dispatch_started`, `tracking_updated`, ...

**Standalone usage:**
```bash
pip install apg-transport-dis
apg-transport-dis --port 8080
```

---

### Fleet Management `transport_fle`

> The Fleet Management capability handles the complete vehicle lifecycle from registration to disposal, telematics integration with major providers, driver management with CPC and tachograph tracking, utilisation analytics, and compliance enforcement including DVLA, C-TPAT, and Euro emissions standards.

**Package**: `apg-transport-fle`  
**Path**: `capabilities/transport/fle`  
**Version**: 1.0.0  

**Provides:**
- `vehicle_lifecycle_workflow`
- `telematics_integration_workflow`
- `driver_management_workflow`
- `fleet_utilisation_analytics_workflow`
- `fleet_compliance_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `comp`
- `mqeb`
- `schd`

**Service methods** (51 total):
`register_vehicle`, `get_vehicle`, `list_vehicles`, `update_vehicle`, `delete_vehicle`, `set_vehicle_status`, `register_driver`, `get_driver`, `list_drivers`, `update_driver`, `delete_driver`, `assign_driver`, ...

**Governance rules** (21 total):
`tenant_context_required`, `fleet_write_requires_policy`, `vehicle_type_supported`, `vehicle_registration_required`, `vehicle_vin_required`, `vehicle_ownership_type_supported`, `vehicle_status_supported`, `non_compliant_vehicle_dispatch_denied`, ...

**UI Routes** (12):
- `/transport-fleet/dashboard` — dashboard (transport_fle:view)
- `/transport-fleet/vehicles` — vehicles (transport_fle:vehicles)
- `/transport-fleet/vehicles/create` — vehicle_create (transport_fle:vehicles_write)
- `/transport-fleet/vehicles/<vehicle_id>` — vehicle_detail (transport_fle:vehicles)
- `/transport-fleet/drivers` — drivers (transport_fle:drivers)
- `/transport-fleet/drivers/<driver_id>` — driver_detail (transport_fle:drivers)
- _6 more..._

**Streaming events** via `bytewax`:
`vehicle_registered`, `vehicle_status_changed`, `driver_registered`, `driver_status_changed`, `telematics_event`, ...

**Standalone usage:**
```bash
pip install apg-transport-fle
apg-transport-fle --port 8080
```

---

### Fuel Management `transport_fue`

> The Fuel Management capability covers fuel procurement, transaction recording with odometer capture, fuel card management and reconciliation, bunker management, carbon footprint calculation across GHG Protocol and ISO 14064 standards, and storage tank monitoring. Built-in phantom fill and theft pattern detection protect against fraud.

**Package**: `apg-transport-fue`  
**Path**: `capabilities/transport/fue`  
**Version**: 1.0.0  

**Provides:**
- `fuel_procurement_workflow`
- `fuel_consumption_tracking_workflow`
- `bunker_management_workflow`
- `fuel_card_reconciliation_workflow`
- `carbon_footprint_reporting_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `comp`
- `mqeb`
- `schd`

**Service methods** (44 total):
`describe`, `evaluate`, `create_procurement`, `record_transaction`, `register_fuel_card`, `reconcile_fuel_card`, `record_carbon_emission`, `register_storage_tank`, `register_fuel_agent`, `validate_batch`, `list_transactions`, `list_fuel_cards`, ...

**Governance rules** (21 total):
`tenant_context_required`, `fuel_write_requires_policy`, `fuel_type_supported`, `transaction_vehicle_required`, `transaction_driver_required`, `transaction_odometer_required`, `transaction_type_supported`, `transaction_quantity_positive`, ...

**UI Routes** (12):
- `/transport-fuel/dashboard` — dashboard (transport_fue:view)
- `/transport-fuel/procurement` — procurement (transport_fue:procurement)
- `/transport-fuel/transactions` — transactions (transport_fue:transactions)
- `/transport-fuel/cards` — fuel_cards (transport_fue:cards)
- `/transport-fuel/cards/reconciliation` — card_reconciliation (transport_fue:cards)
- `/transport-fuel/storage` — storage (transport_fue:storage)
- _6 more..._

**Streaming events** via `bytewax`:
`fuel_procurement_recorded`, `fuel_transaction_recorded`, `fuel_card_reconciled`, `carbon_emission_calculated`, `fuel_storage_updated`, ...

**Standalone usage:**
```bash
pip install apg-transport-fue
apg-transport-fue --port 8080
```

---

### Vehicle Maintenance `transport_mai`

> The Vehicle Maintenance capability manages preventive and corrective maintenance job scheduling, workshop bay allocation, parts inventory and ordering, warranty tracking, vehicle inspections with digital signature capture, and roadworthiness certificate management. It enforces pre-dispatch safety checks and blocks operation of expired-MOT or unsafe vehicles.

**Package**: `apg-transport-mai`  
**Path**: `capabilities/transport/mai`  
**Version**: 1.0.0  

**Provides:**
- `preventive_maintenance_schedule_workflow`
- `workshop_management_workflow`
- `parts_inventory_workflow`
- `warranty_tracking_workflow`
- `roadworthiness_compliance_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `comp`
- `mqeb`
- `schd`

**Service methods** (43 total):
`describe`, `evaluate`, `create_job`, `update_job_status`, `dispatch_vehicle_check`, `allocate_workshop`, `order_parts`, `record_warranty`, `conduct_inspection`, `issue_roadworthiness`, `create_maintenance_schedule`, `register_maintenance_agent`, ...

**Governance rules** (21 total):
`tenant_context_required`, `maintenance_write_requires_policy`, `maintenance_type_supported`, `job_vehicle_required`, `job_technician_required`, `job_status_supported`, `job_priority_supported`, `expired_mot_dispatch_denied`, ...

**UI Routes** (12):
- `/transport-maintenance/dashboard` — dashboard (transport_mai:view)
- `/transport-maintenance/jobs` — jobs (transport_mai:jobs)
- `/transport-maintenance/jobs/create` — job_create (transport_mai:jobs_write)
- `/transport-maintenance/workshop` — workshop (transport_mai:workshop)
- `/transport-maintenance/parts` — parts (transport_mai:parts)
- `/transport-maintenance/warranty` — warranty (transport_mai:warranty)
- _6 more..._

**Streaming events** via `bytewax`:
`maintenance_job_created`, `maintenance_job_completed`, `parts_ordered`, `warranty_claimed`, `inspection_completed`, ...

**Standalone usage:**
```bash
pip install apg-transport-mai
apg-transport-mai --port 8080
```

---

### Route Optimisation `transport_rou`

> The Route Optimisation capability provides multi-stop route planning with time-window enforcement, 8 optimisation objectives, dynamic traffic-triggered rerouting, multi-modal segment planning (road, rail, sea, air), and geospatial address validation. It integrates with HERE Maps, Google Maps, TomTom, and other traffic providers for real-time incident awareness.

**Package**: `apg-transport-rou`  
**Path**: `capabilities/transport/rou`  
**Version**: 1.0.0  

**Provides:**
- `multi_stop_route_planning_workflow`
- `dynamic_rerouting_workflow`
- `traffic_integration_workflow`
- `time_window_constraint_workflow`
- `multimodal_routing_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `nlpc`
- `mqeb`
- `schd`

**Service methods** (40 total):
`describe`, `evaluate`, `plan_route`, `add_route_stop`, `add_constraint`, `record_traffic_event`, `trigger_reroute`, `plan_multimodal_segment`, `register_route_agent`, `validate_batch`, `list_routes`, `get_route`, ...

**Governance rules** (21 total):
`tenant_context_required`, `route_write_requires_policy`, `route_type_supported`, `route_origin_required`, `route_destination_required`, `route_vehicle_required`, `unvalidated_address_dispatch_denied`, `optimisation_objective_supported`, ...

**UI Routes** (13):
- `/transport-route/dashboard` — dashboard (transport_rou:view)
- `/transport-route/routes` — routes (transport_rou:routes)
- `/transport-route/routes/create` — route_create (transport_rou:routes_write)
- `/transport-route/routes/<route_id>/map` — route_map (transport_rou:routes)
- `/transport-route/optimisation` — optimisation (transport_rou:optimisation)
- `/transport-route/constraints` — constraints (transport_rou:constraints)
- _7 more..._

**Streaming events** via `bytewax`:
`route_planned`, `route_optimised`, `route_dispatched`, `traffic_incident_detected`, `reroute_triggered`, ...

**Standalone usage:**
```bash
pip install apg-transport-rou
apg-transport-rou --port 8080
```

---

### Transport Scheduling `transport_sch`

> The Transport Scheduling capability manages load scheduling, driver shift planning with tachograph and HOS compliance, vehicle assignment, charter management (school, corporate, tourist, medical), schedule optimisation, and conflict detection. It blocks schedule publication when unresolved conflicts exist and enforces tacho compliance on all shifts.

**Package**: `apg-transport-sch`  
**Path**: `capabilities/transport/sch`  
**Version**: 1.0.0  

**Provides:**
- `load_scheduling_workflow`
- `driver_shift_planning_workflow`
- `vehicle_assignment_workflow`
- `charter_management_workflow`
- `schedule_optimisation_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `schd`
- `mqeb`
- `comp`

**Service methods** (40 total):
`describe`, `evaluate`, `create_schedule`, `publish_schedule`, `create_shift`, `assign_vehicle`, `create_charter`, `record_conflict`, `resolve_conflict`, `send_notification`, `register_scheduling_agent`, `validate_batch`, ...

**Governance rules** (21 total):
`tenant_context_required`, `scheduling_write_requires_policy`, `schedule_type_supported`, `schedule_status_supported`, `shift_type_supported`, `driver_hours_breach_denied`, `double_booking_denied`, `charter_type_supported`, ...

**UI Routes** (12):
- `/transport-scheduling/dashboard` — dashboard (transport_sch:view)
- `/transport-scheduling/schedules` — schedules (transport_sch:schedules)
- `/transport-scheduling/schedules/create` — schedule_create (transport_sch:schedules_write)
- `/transport-scheduling/calendar` — calendar (transport_sch:view)
- `/transport-scheduling/shifts` — shifts (transport_sch:shifts)
- `/transport-scheduling/vehicles` — vehicle_assignment (transport_sch:vehicles)
- _6 more..._

**Streaming events** via `bytewax`:
`schedule_created`, `schedule_published`, `shift_assigned`, `vehicle_assigned`, `charter_confirmed`, ...

**Standalone usage:**
```bash
pip install apg-transport-sch
apg-transport-sch --port 8080
```

---

### Asset Tracking `transport_tra`

> The Asset Tracking capability provides real-time GPS tracking for vehicles, trailers, containers, pallets, and equipment. It supports geofence creation (circle, polygon, corridor, exclusion zone), cold-chain temperature monitoring with breach detection, container tracking with ISO number and seal management, and utilisation analytics. Tamper detection requires immediate escalation.

**Package**: `apg-transport-tra`  
**Path**: `capabilities/transport/tra`  
**Version**: 1.0.0  

**Provides:**
- `realtime_gps_tracking_workflow`
- `geofencing_workflow`
- `cold_chain_monitoring_workflow`
- `container_tracking_workflow`
- `asset_utilisation_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `comp`
- `mqeb`
- `nlpc`

**Service methods** (40 total):
`describe`, `evaluate`, `register_asset`, `update_asset_location`, `create_geofence`, `raise_alert`, `acknowledge_alert`, `record_cold_chain`, `register_container`, `update_container_status`, `record_utilisation`, `register_tracking_agent`, ...

**Governance rules** (21 total):
`tenant_context_required`, `tracking_write_requires_policy`, `asset_type_supported`, `asset_unique_id_required`, `asset_owner_required`, `tracking_technology_supported`, `monitoring_type_supported`, `geofence_type_supported`, ...

**UI Routes** (12):
- `/transport-tracking/dashboard` — dashboard (transport_tra:view)
- `/transport-tracking/map` — live_map (transport_tra:view)
- `/transport-tracking/assets` — assets (transport_tra:assets)
- `/transport-tracking/assets/<asset_id>` — asset_detail (transport_tra:assets)
- `/transport-tracking/geofencing` — geofencing (transport_tra:geofencing)
- `/transport-tracking/alerts` — alerts (transport_tra:alerts)
- _6 more..._

**Streaming events** via `bytewax`:
`asset_registered`, `asset_location_updated`, `geofence_entered`, `geofence_exited`, `tracking_alert_raised`, ...

**Standalone usage:**
```bash
pip install apg-transport-tra
apg-transport-tra --port 8080
```

---

### Warehouse Operations `transport_war`

> The Warehouse Operations capability handles all inbound and outbound warehouse processes: goods receiving (ASN, PO, blind), directed putaway with 7 strategies, multi-method picking, packing with weight verification, cross-docking, cycle counting with approval workflows, dock door management, and inventory adjustment control. Cold-chain temperature checks are enforced at receiving. Unapproved inventory adjustments are blocked.

**Package**: `apg-transport-war`  
**Path**: `capabilities/transport/war`  
**Version**: 1.0.0  

**Provides:**
- `warehouse_receiving_workflow`
- `putaway_workflow`
- `picking_workflow`
- `packing_workflow`
- `cross_docking_workflow`
- `cycle_counting_workflow`
- `wms_integration_workflow`

**Requires:**
- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `comp`
- `mqeb`
- `schd`

**Service methods** (40 total):
`describe`, `evaluate`, `register_warehouse`, `receive_goods`, `execute_putaway`, `create_pick_task`, `complete_pick_task`, `create_pack_task`, `complete_packing`, `initiate_cycle_count`, `complete_cycle_count`, `adjust_inventory`, ...

**Governance rules** (22 total):
`tenant_context_required`, `warehouse_write_requires_policy`, `warehouse_type_supported`, `receipt_method_supported`, `receipt_barcode_required`, `receipt_damage_inspection_required`, `putaway_strategy_supported`, `putaway_slot_verification_required`, ...

**UI Routes** (13):
- `/transport-warehouse/dashboard` — dashboard (transport_war:view)
- `/transport-warehouse/receiving` — receiving (transport_war:receiving)
- `/transport-warehouse/putaway` — putaway (transport_war:putaway)
- `/transport-warehouse/inventory` — inventory (transport_war:inventory)
- `/transport-warehouse/picking` — picking (transport_war:picking)
- `/transport-warehouse/packing` — packing (transport_war:packing)
- _7 more..._

**Streaming events** via `bytewax`:
`goods_received`, `putaway_completed`, `pick_task_created`, `pick_completed`, `packing_completed`, ...

**Standalone usage:**
```bash
pip install apg-transport-war
apg-transport-war --port 8080
```

---
