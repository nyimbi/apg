# Product Information Management

## Overview

Product Information Management (PIM) is the APG capability packet that owns the authoritative record for every product in a tenant's catalog. It governs the full product data lifecycle — from catalog and SKU creation through attribute enrichment, variant modelling, content localisation, digital asset attachment, compliance documentation, channel listing, and final publication — enforcing a rule-based governance layer at every transition.

The capability is designed for multi-channel, multi-locale product operations. It integrates with commerce channels, ERP item masters, media asset systems, translation memory, and taxonomy services through APG composition adapters, keeping the core import-time dependency surface minimal. AI-assisted review agents (Codex, Claude Code, OpenCode, Pi) operate within a bounded, human-in-the-loop governance model and can inspect, prepare, and recommend but cannot autonomously commit privileged state changes.

Version 2.2.0 adds a full async method layer enabling concurrent enrichment pipelines, parallel marketplace syndication, and non-blocking quality scoring — all composable with `asyncio.gather` in async web handlers and agent runtimes.

## Capability ID

`pde_pim`  Version: 2.2.0

## Provides

| Service | Description |
|---------|-------------|
| product_catalog_lifecycle | Create, update, and retire product catalogs with code/name/owner enforcement |
| product_record_lifecycle | Full SKU lifecycle from concept through active to archived, across supported product types |
| product_attribute_lifecycle | Define typed attributes (text, number, boolean, date, enum, money, media, rich_text) with tenant ownership |
| product_variant_lifecycle | Parent-child variant relationships with mandatory option-value sets and per-variant SKUs |
| product_content_lifecycle | Locale-scoped content enrichment (title, description, generated copy) with mandatory review gate for AI-generated content |
| product_asset_lifecycle | Attach and manage digital assets (images, video, documents) with rights-basis enforcement |
| product_compliance_lifecycle | Record regulatory compliance per framework with evidence attachment and review gate for high-risk items |
| product_channel_listing_lifecycle | Create and approve listings across web, marketplace, ERP, POS, print, and API channels |
| product_publish_workflow | Orchestrate the approval chain — approved content + approved channel + publish approval — before a product goes live |
| product_data_quality_workflow | Record, score, and resolve data quality issues; escalate high/critical severity to assigned owners |
| product_change_workflow | Change request lifecycle with mandatory reason, approver recording, and full audit trail |
| pim_dashboard_service | Aggregate health, coverage, and readiness metrics across catalogs and channels |
| pim_agents | Managed AI review agents for catalog, data quality, enrichment, channel, compliance, and product query roles |
| async_enrichment_pipeline | Concurrent attribute enrichment across multiple SKUs via asyncio.gather with semaphore-bounded concurrency |
| async_syndication | Parallel marketplace publication with pre-flight channel quality gate per channel |
| async_localisation | Multi-locale content application for a single SKU in one non-blocking call |
| quality_remediation_plan | Ordered, actionable remediation plan with per-dimension effort estimates and score-gain priorities |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authentication and RBAC for all PIM operations |
| audl | Audit trail for every state-changing write |
| ntfy | Notification dispatch for approvals, reviews, and escalations |
| composition_events | APG event bus used to publish and consume cross-capability lifecycle events |
| composition_config | Tenant configuration resolution and override merging |
| workflow | Approval workflow engine for publish, compliance review, and change request flows |
| media_asset_lifecycle | Upstream provider of media asset records referenced by product assets |
| commerce_channel_lifecycle | Upstream provider of channel definitions consumed by channel listing creation |
| erp_item_master | Bidirectional sync of SKU and product master data with ERP systems |
| translation_memory | Source of locale-aware translations used during content enrichment |
| taxonomy_management | Provides category and attribute taxonomy referenced when defining product attributes |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant isolation key injected at runtime |
| catalogs.code_required | bool | true | Reject catalog creation without a code |
| catalogs.name_required | bool | true | Reject catalog creation without a name |
| catalogs.owner_required | bool | true | Reject catalog creation without an owner |
| products.sku_required | bool | true | Reject product creation without a SKU |
| products.supported_types | list | physical, digital, service, bundle, component, raw_material | Allowlist of valid product type values |
| products.catalog_required | bool | true | Reject product creation without a catalog |
| attributes.supported_types | list | text, number, boolean, date, enum, money, media, rich_text | Allowlist of valid attribute types |
| attribute_values.locale_required_for_rich_text | bool | true | Enforce locale on rich_text attribute values |
| content.review_required_for_generated_content | bool | true | Block generated content publication until a reviewer signs off |
| assets.rights_basis_required | bool | true | Reject asset attachment without a rights basis |
| compliance.evidence_required | bool | true | Reject compliance records without evidence documents |
| compliance.review_required_for_high_risk | bool | true | Require explicit review recording for high-risk compliance records |
| channels.supported_channels | list | web, marketplace, erp, pos, print, api | Allowlist of valid channel values |
| channels.approval_required | bool | true | Channel listings require approval before use in publishing |
| publishing.approval_required | bool | true | Publication requires an explicit publish approver |
| quality.severity_owner_required | list | high, critical | Severities that mandate an assigned owner |
| changes.reason_required | bool | true | Change requests require a stated reason |
| changes.approval_required | bool | true | Change approval requires a recorded approver |
| pim_agents.max_autonomous_scope | string | "inspect_prepare_and_recommend" | Hard ceiling on what agents may do without human approval |
| pim_agents.human_approval_required | bool | true | All privileged agent actions require a human approval record |
| governance.require_tenant_context | bool | true | All operations require a resolved tenant context |
| governance.audit_state_changes | bool | true | Every write emits an audit event |
| observability.stream_processor | string | "bytewax" | Event stream processor; Bytewax is the only supported value |
| theme.default_theme | string | "pim_control" | Default UI theme |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /pde/pim/dashboard | GET | pde_pim:view | Overview |
| catalogs | /pde/pim/catalogs | GET/POST | pde_pim:manage_catalogs | Setup |
| products | /pde/pim/products | GET/POST | pde_pim:manage_products | Products |
| attributes | /pde/pim/attributes | GET/POST | pde_pim:manage_attributes | Setup |
| content | /pde/pim/content | GET/POST | pde_pim:manage_content | Content |
| assets | /pde/pim/assets | GET/POST | pde_pim:manage_assets | Content |
| compliance | /pde/pim/compliance | GET/POST | pde_pim:govern | Governance |
| channels | /pde/pim/channels | GET/POST | pde_pim:publish | Publishing |
| quality | /pde/pim/quality | GET/POST | pde_pim:quality | Governance |
| changes | /pde/pim/changes | GET/POST | pde_pim:approve_changes | Governance |
| agents | /pde/pim/agents | GET/POST | pde_pim:agent_manage | Automation |
| rules | /pde/pim/rules | GET | pde_pim:govern | Governance |
| settings | /pde/pim/settings | GET/POST | pde_pim:admin | Administration |

REST API prefix: `/pde/pim/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present = false | deny — attach_tenant_context |
| operation_policy_required | write operation without policy | deny — attach_operation_policy |
| catalog_code_required | create_catalog, code absent | deny — provide_catalog_code |
| catalog_owner_required | create_catalog, owner absent | deny — assign_owner |
| product_catalog_required | create_product, catalog absent | deny — select_catalog |
| product_sku_required | create_product, sku absent | deny — provide_sku |
| product_type_supported | create_product, unsupported type | deny — choose_supported_product_type |
| product_owner_required | create_product, owner absent | deny — assign_owner |
| attribute_type_supported | define_attribute, unsupported type | deny — choose_supported_attribute_type |
| rich_text_locale_required | set_attribute_value rich_text, locale absent | deny — provide_locale |
| variant_options_required | create_variant, options absent | deny — provide_options |
| generated_content_review_required | enrich_content with generated flag, no review | require_review — record_content_review |
| asset_rights_required | attach_asset, rights_basis absent | deny — provide_rights_basis |
| compliance_evidence_required | record_compliance, evidence absent | deny — attach_evidence |
| compliance_high_risk_review | record_compliance, high_risk, no review | require_review — record_compliance_review |
| channel_approval_required | create_channel_listing, no approval | deny — record_channel_approval |
| channel_supported | create_channel_listing, unsupported channel | deny — choose_supported_channel |
| publish_content_required | publish_product, no approved content | deny — approve_content |
| publish_channel_required | publish_product, no approved channel | deny — approve_channel |
| publish_approval_required | publish_product, no approval | deny — record_publish_approval |
| quality_owner_required | record_quality_issue high/critical, no owner | deny — assign_owner |
| change_reason_required | create_change_request, reason absent | deny — provide_reason |
| change_approval_required | approve_change, no approval record | deny — record_approver |
| bytewax_event_stream_required | pim_batch routed to queue (not bytewax) | deny — route_to_bytewax_stream |
| agent_scope_limited | agent privileged action, no human approval | require_review — record_human_approval |
| audit_required_for_state_change | write with audit disabled | deny — enable_audit |

## Data Models

| Model | Key Fields |
|-------|-----------|
| PLProduct | product_id, tenant_id, product_number (SKU), product_name, product_type, lifecycle_phase, parent_product_id, product_family, target_cost, current_price, launch_date, end_of_life_date, manufacturing_bom_id, digital_twin_id |
| PLProductStructure | structure_id, tenant_id, parent_product_id, child_product_id, quantity, unit_of_measure, position_number, assembly_sequence, critical_component, effective_date, obsolete_date |
| PLEngineeringChange | change_id, tenant_id, change_number, change_type, change_priority, status, affected_products, reason_for_change, cost_impact, schedule_impact, approvers, approved_by, workflow_instance_id |
| PLProductConfiguration | configuration_id, tenant_id, base_product_id, configuration_code, configuration_type, selected_options, feature_codes, base_price, option_price_delta, generates_bom, orderable, quotable |
| PLCollaborationSession | session_id, tenant_id, session_type, session_status, host_user_id, participants, products_discussed, decisions_made, action_items, recording_url |
| PLComplianceRecord | compliance_id, tenant_id, product_id, standard, status, certificate_number, issued_by, expiry_date, evidence_documents, test_reports, next_review_date |
| PLManufacturingIntegration | integration_id, tenant_id, product_id, manufacturing_bom_id, sync_status, auto_sync_enabled, sync_direction, error_count |
| PLDigitalTwinBinding | binding_id, tenant_id, product_id, digital_twin_id, binding_status, sync_properties, sync_frequency, performance_data |
| PLProductionSystem | system_id, tenant_id, system_type, synergy_score, value_multiplier, performance_metrics, autonomous_decisions_made, decision_accuracy_rate |
| PLGenerativeAISession | session_id, tenant_id, design_brief, constraints, concepts_generated, evolution_iterations, innovation_score, feasibility_score |
| PLXRCollaborationSession | session_id, tenant_id, xr_environment_type, participants, presence_quality, collaboration_effectiveness, spatial_manipulations |
| PLSustainabilityProfile | profile_id, tenant_id, product_id, carbon_footprint_reduction, circularity_score, autonomous_optimizations, cost_savings_achieved |
| PLQuantumOptimization | optimization_id, tenant_id, quantum_system_id, optimization_problem, quantum_speedup_achieved, quantum_advantage_achieved, cost_optimization_savings |
| PLDigitalProductPassport | passport_id, tenant_id, product_id, lifecycle_events, supply_chain_provenance, iot_sensor_data, blockchain_hash, end_of_life_planning |

Product type enum: `manufactured, purchased, configured, service, software, digital`
Lifecycle phase enum: `concept, design, development, testing, production, launch, growth, maturity, decline, retirement`
Compliance standards: `iso_9001, iso_13485, fda_510k, fda_pma, ce_marking, itar, rohs, reach, ul, fcc`

## Streaming Events

Events emitted to the `apg.pde.pim.lifecycle` event stream via Bytewax. Delivery: at-least-once. Ordering key: `tenant_id`.

| Event | Trigger |
|-------|---------|
| catalog_created | A new product catalog is successfully persisted |
| product_created | A new product record passes all creation rules |
| attribute_defined | A new attribute definition is committed to the tenant schema |
| attribute_value_set | An attribute value is written against a product |
| variant_created | A product variant with option values is created under a parent |
| content_enriched | A locale-scoped content record (title + body) is saved for a product |
| asset_attached | A digital asset with rights basis is attached to a product |
| compliance_recorded | A compliance record with evidence is saved for a product |
| channel_listing_created | An approved channel listing is created for a product |
| product_published | A product clears the full publish approval chain and goes live |
| quality_issue_recorded | A data quality issue is logged against a product |
| change_request_created | A product change request with stated reason is submitted |
| change_request_approved | All required approvers have signed off on a change request |
| pim_agent_registered | A PIM AI agent is registered with a supported runtime and role |

## Edge Cases Handled

- **Generated content publication block**: Any content record flagged as AI-generated is held at `require_review` regardless of other approval state until an explicit human review is recorded. The rule fires even if the content is otherwise complete and all other publish prerequisites are met.
- **Rich-text attribute values without locale**: Setting a `rich_text` typed attribute value without a locale is denied at the rule engine level before persistence, preventing locale-ambiguous text from entering the catalog.
- **High/critical quality issues without owner**: Quality issues with severity `high` or `critical` cannot be persisted without an assigned owner, ensuring every actionable defect has a responsible party.
- **Channel listings bypassing approval**: A channel listing cannot be referenced in a publish workflow unless it carries an approval record. Approval and listing creation are separate operations to prevent the publish step from silently accepting an unapproved listing.
- **Bytewax stream routing enforcement**: Batch PIM operations that attempt to route events through a queue rather than the Bytewax stream are denied. This keeps the observability layer coherent — all PIM events flow through a single processor.
- **Agent privilege escalation**: PIM agents that attempt to execute a privileged state change (anything beyond inspect/prepare/recommend) are intercepted at the rule engine with a `require_review` decision, not a silent pass-through. The scope ceiling is enforced in the contract, not in the agent runtime.
- **Compliance expiration proximity**: `PLComplianceRecord.check_expiration()` fires a warning 90 days before expiry and an alert on or after expiry, giving teams a managed runway for renewal without relying on manual calendar tracking.
- **Multi-approver change requests**: `PLEngineeringChange.approve_change()` accumulates approvals in `approved_by` and only transitions to `APPROVED` once the full `approvers` set is covered. Partial approval does not change the record status.
- **Tenant context on every operation**: The rule engine denies any operation — read or write — that arrives without a resolved tenant context. There is no fallback to a shared or default tenant in production configuration.

## Composability

- **Upstream**: `media_asset_lifecycle` provides the digital asset records that PIM attaches to products. `commerce_channel_lifecycle` provides channel definitions. `taxonomy_management` provides the attribute and category schema. `erp_item_master` is the source of truth for SKU master data in ERP-led deployments.
- **Downstream**: Channel publishing events consumed by commerce adapters to sync live listings. `product_publish_workflow` completion events can trigger downstream pricing, promotions, and fulfilment capabilities. `product_compliance_lifecycle` events feed regulatory reporting and audit dashboards.
- **Peer**: `pde_cfm` (Catalogue and Feed Management) consumes published product records to produce channel-specific feeds. `pde_spd` (Supplier Product Data) feeds raw supplier content into PIM enrichment flows. `pde_pmd` (Product Master Data) governs the golden record that PIM enriches.

## Development Notes

- All models use `pl_` table prefix for PLM domain isolation. UUID7 string IDs via `uuid6.uuid7` (the `uuid_extensions` package referenced in older code is not on PyPI; use the `uuid6` package and a local `uuid7str` shim).
- The `models.py` file contains PLM-oriented models (PLProduct, PLProductStructure, PLEngineeringChange, etc.) that reflect a product lifecycle management heritage; the service layer maps these onto the PIM contract's concepts (catalog, attribute, variant, content, channel).
- `DEFAULT_CONFIGURATION` is deep-merged with tenant overrides at runtime via `get_capability_contract(tenant_id, overrides)`. Overrides are additive — they cannot remove required fields.
- The rule engine in `evaluate_capability_rules` is deterministic and pure: it iterates all rules, accumulates effects, and returns the strictest decision (`deny` > `require_review` > `allow`). It does not short-circuit on first denial, so a single context evaluation returns the full set of matched rules.
- PIM agents are registered with one of four runtimes (`codex`, `claude_code`, `opencode`, `pi`) and one of six roles. The `max_autonomous_scope` cap is enforced in the contract's rule engine, not in the agent SDK, so it applies regardless of which runtime is in use.
- The UI shell is `apg_python` with Flask-Appbuilder blueprints. Each nav group maps to a logical workbench component; the theme (`pim_control`) uses a compact density with a teal/amber palette.
- Run focused verification with: `uv run pytest -q capabilities/pde/pim/tests/test_package_contract.py`

## Quick Start

### Synchronous API

```python
from capabilities.pde.pim.service import ProductInformationLifecycleService

svc = ProductInformationLifecycleService(tenant_id="tenant-a")
svc.create_catalog("cat-1", "tenant-a", "MAIN", "Main Catalog", "owner-1")
product = svc.create_product("SKU-001", "Solar Charger 20W", "Electronics", {"weight_g": 180}, tenant_id="tenant-a")
svc.update_attributes("SKU-001", {"colour": "black", "warranty_years": 2}, tenant_id="tenant-a")
svc.add_media("SKU-001", "image", "https://cdn.example.com/sku001-hero.jpg", "Hero shot", tenant_id="tenant-a")
svc.product_categorisation("SKU-001", ["Electronics", "Solar", "Chargers"], tenant_id="tenant-a")
score = svc.data_quality_score("SKU-001", tenant_id="tenant-a")
svc.publish_to_channel("SKU-001", "web", tenant_id="tenant-a", approved_by="catalog_manager")
```

### Async API (v2.2.0+)

```python
import asyncio
from capabilities.pde.pim.service import ProductInformationLifecycleService

async def main():
    svc = ProductInformationLifecycleService(tenant_id="tenant-a")
    # Create product in async context
    product = await svc.async_create_product(
        "SKU-001", "Solar Charger 20W", "Electronics",
        {"weight_g": 180}, tenant_id="tenant-a",
    )
    # Enrich 100 products concurrently (semaphore-bounded to 10 at a time)
    tasks = [{"sku": f"SKU-{i:03d}", "attributes": {"batch": "2026-Q2"}} for i in range(100)]
    result = await svc.async_bulk_enrich(tasks, tenant_id="tenant-a", concurrency=10)
    # Quality score + remediation plan
    plan = await svc.async_quality_remediation_plan("SKU-001", tenant_id="tenant-a")
    # Localise in three markets simultaneously
    await svc.async_localise_product("SKU-001", [
        {"locale": "en", "title": "Solar Charger 20W", "description": "Portable solar charging."},
        {"locale": "fr", "title": "Chargeur Solaire 20W", "description": "Chargeur solaire portable."},
        {"locale": "sw", "title": "Chaja ya Jua 20W", "description": "Chaja ya jua inayobebeka."},
    ], tenant_id="tenant-a")
    # Syndicate to marketplaces in parallel
    syndication = await svc.async_syndicate_marketplaces(
        "SKU-001", ["marketplace_amazon", "web", "mobile"],
        tenant_id="tenant-a", approved_by="catalog_manager",
    )
    print(syndication)

asyncio.run(main())
```
