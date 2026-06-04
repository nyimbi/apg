# Self-Service BI

## Overview
The Self-Service BI capability (bia_sbi) provides a drag-and-drop visual chart builder, natural-language query (NLQ) processing, a governed data catalogue with tiered access control, user sandboxes with row limits and auto-expiry, and a template gallery — giving business users governed self-service analytics without requiring SQL expertise.

## Capability ID
`bia_sbi`

## Provides
- drag_drop_visual_builder: 12 chart types via drag-and-drop or guided wizard
- natural_language_queries: Hybrid rule/LLM NLQ with SQL generation
- governed_data_catalogue: 4 governance tiers with approval workflows
- user_sandboxes: Per-user sandboxes with 500K row limit and 30-day TTL
- template_gallery: Shared chart and dashboard templates
- self_service_chart_creation: Workspace-based chart authoring
- catalogue_governance: Draft → pending → published lifecycle
- embedded_analytics: Workspace publishing with embedding tokens

## Requires
| Capability | Reason |
|------------|--------|
| auth | User identity and permission checks |
| audl | Audit NLQ queries and catalogue changes |
| mten | Tenant context enforcement |
| conf | Runtime configuration |
| nlpc | NLQ text processing and SQL generation |
| mqeb | Streaming SBI lifecycle events |
| ntfy | Catalogue approval notifications |
| bia_anl | Governed query execution backing NLQ results |

## Configuration
| Option | Default | Description |
|--------|---------|-------------|
| sandbox_row_limit | 500,000 | Hard row limit per sandbox query |
| max_sandboxes_per_user | 5 | Active sandbox limit per user |
| sandbox_ttl_days | 30 | Auto-expiry for sandboxes |
| max_datasets_per_workspace | 20 | Dataset limit per workspace |
| require_catalogue_approval | true | Governed catalogue entries need approval |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/sbi/workspaces | GET/POST | List/create workspaces | bia_sbi:build |
| /api/bia/sbi/workspaces/<id>/charts | GET/POST | List/create charts | bia_sbi:build |
| /api/bia/sbi/catalogue | GET/POST | Data catalogue | bia_sbi:catalogue |
| /api/bia/sbi/catalogue/<id>/approve | POST | Approve entry | bia_sbi:admin |
| /api/bia/sbi/sandboxes | GET/POST | List/create sandboxes | bia_sbi:sandbox |
| /api/bia/sbi/ask | POST | Submit NLQ | bia_sbi:query |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| classified_data_restricted | classified tier + no clearance | deny |
| sandbox_row_limit_enforced | Row limit exceeded | deny |
| sandbox_limit_per_user_enforced | >5 active sandboxes | deny |
| catalogue_approval_required | governed catalogue + not approved | deny |
| expired_sandbox_cannot_query | state=expired | deny |
| deprecated_catalogue_cannot_be_queried | state=deprecated | deny |

## Data Models
- WorkspaceResponse: id, name, owner_id, access_level, charts, datasource_ids
- ChartResponse: id, workspace_id, name, chart_type, datasource_id, config
- CatalogueEntryResponse: id, name, datasource_id, state, governance_tier, description
- SandboxResponse: id, name, owner_id, state, datasource_ids, row_count, expires_at
- NLQResponse: id, query_text, generated_sql, chart_type_suggestion, confidence

## Streaming Events
- workspace_created, chart_created, nlq_submitted, nlq_answered
- catalogue_entry_created, catalogue_entry_approved
- sandbox_created, sandbox_expired, template_used, analytics_published

## Edge Cases Handled
- Classified data requires explicit clearance check — not just governance tier membership
- Expired sandboxes reject queries — users must create new sandbox
- Sandbox TTL cannot be extended beyond maximum — forces data freshness
- Deprecated catalogue entries cannot be queried — users guided to current entry
- NLQ confidence score is returned so UX can warn on low-confidence SQL generation

## Composability Notes
- nlpc provides the LLM/rule engine for NLQ SQL generation
- bia_anl executes the generated SQL against governed datasources
- bia_dsh can consume SBI workspace charts as embedded widgets
- Catalogue approvals integrate with wflo for multi-step governance
- bia_rpt can use published SBI analytics as report data sources
