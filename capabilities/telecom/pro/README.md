# Service Provisioning & Product Catalogue

## Overview
Service activation and provisioning engine with an integrated product catalogue covering TMF620-aligned service/product definitions, bundle management, effective-dated price lists, promotional campaign management, and lifecycle state machine governance. Orchestration layer handles workflow management, network resource reservation, multi-protocol configuration push (NETCONF, RESTCONF, CLI, REST API), end-to-end activation verification, automated rollback, and bulk provisioning with pre-approval gating.

## Capability ID
`telecom_pro`

## Provides
- product_catalogue: TMF620-aligned service/product catalogue with versioning and lifecycle
- bundle_management: Bundle composition, eligibility evaluation, and order decomposition
- price_list_management: Effective-dated price records with tiered rate tables
- promotional_campaigns: Time-limited discount campaign management
- catalogue_health_analytics: Product completeness scoring and KPI dashboard
- service_activation_workflow: End-to-end service activation orchestration
- network_resource_allocation: Conflict-checked resource reservation and release
- configuration_push_workflow: Multi-protocol config push with dry-run
- activation_confirmation_workflow: E2E test and confirmation recording
- rollback_workflow: Automated and manual rollback on failure
- bulk_provisioning_workflow: Pre-approved bulk service activation
- pro_agent_workflow: Provisioning automation agent management

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Provisioning and catalogue event audit trail |
| mten | Tenant isolation |
| conf | Configuration |
| ntfy | Activation and failure notifications |
| wflo | Workflow state management |
| mqeb | Event streaming |
| moni | NE health monitoring |
| schd | Scheduled bulk job execution |
| telecom_bil | Billing charge activation on service completion |
| telecom_ord | Order decomposition handoff |
| telecom_inv | IPAM and circuit inventory |

## Configuration
| Key | Description |
|-----|-------------|
| workflows.timeout_minutes | 60-minute workflow timeout |
| workflows.max_retries | Maximum 3 retries |
| resources.reservation_ttl_minutes | 30-minute reservation TTL |
| network_elements.health_check_before_push | Mandatory NE health check |
| config_push.dry_run_enabled | Dry run before live push |
| catalogue.default_currency | Default currency for price records (default: KES) |
| catalogue.lifecycle_approval_required | Require justification on all status transitions |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-pro/catalogue | GET/POST | Product catalogue browser | telecom_pro:catalogue |
| /telecom-pro/catalogue/\<id\> | GET/PUT | Product detail and lifecycle | telecom_pro:catalogue |
| /telecom-pro/catalogue/search | GET | Faceted catalogue search | telecom_pro:catalogue |
| /telecom-pro/bundles | GET/POST | Bundle management | telecom_pro:bundles |
| /telecom-pro/bundles/\<id\>/decompose | POST | Decompose bundle to orders | telecom_pro:bundles |
| /telecom-pro/prices | GET/POST | Price list management | telecom_pro:prices |
| /telecom-pro/prices/effective | GET | Effective price query | telecom_pro:prices |
| /telecom-pro/prices/bulk-update | POST | Bulk price update | telecom_pro:prices |
| /telecom-pro/promotions | GET/POST | Promotion campaign management | telecom_pro:promotions |
| /telecom-pro/catalogue/health | GET | Catalogue health dashboard | telecom_pro:catalogue |
| /telecom-pro/workflows | GET/POST | Workflow console | telecom_pro:workflows |
| /telecom-pro/resources | GET/POST | Resource management | telecom_pro:resources |
| /telecom-pro/config-push | GET/POST | Config push console | telecom_pro:config_push |
| /telecom-pro/activation | GET/POST | Activation management | telecom_pro:activation |
| /telecom-pro/rollback | GET/POST | Rollback console | telecom_pro:rollback |
| /telecom-pro/bulk | GET/POST | Bulk provisioning | telecom_pro:bulk |
| /telecom-pro/network-elements | GET/POST | NE health console | telecom_pro:network_elements |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| product_status_machine | invalid state transition | deny with allowed-transitions map |
| bundle_component_must_be_active | component not active/approved | deny |
| price_requires_existing_product | product not in catalogue | deny |
| promotion_window_enforced | order date outside valid_from/valid_to | deny |
| promotion_usage_limit | usage_count >= usage_limit | deny |
| workflow_type_not_supported | unknown type | deny |
| order_reference_required | no order reference | deny |
| resource_conflict_check_required | no conflict check | deny |
| dry_run_bypass_denied | dry_run_bypassed=True | deny |
| activation_verification_required | verification not completed | deny |
| bulk_provisioning_approval_required | no approval reference | deny |
| cross_tenant_provisioning_denied | cross-tenant scope | deny |

## Data Models

### Product Catalogue Models
- ProProduct (dict): product_id, name, category, characteristics, status, version, tenant_id, created_at, updated_at
- ProBundle (dict): bundle_id, name, components[{product_id, quantity, mandatory}], pricing_tier, eligibility_rules, incompatible_with, status
- ProPrice (dict): price_id, product_id, amount, currency, charge_type, effective_from, effective_to, rate_table[{threshold, unit_price}]
- ProPromotion (dict): campaign_id, discount_type, discount_value, applies_to, valid_from, valid_to, usage_limit, usage_count, status

### Provisioning Models
- ProWorkflow: id, tenant_id, workflow_type, order_reference, status, retry_count, started_at
- ProResourceReservation: id, tenant_id, workflow_id, resource_type, resource_value, reserved_at, expires_at, released
- ProConfigPush: id, tenant_id, workflow_id, ne_reference, push_method, template_reference, dry_run_completed, status
- ProActivation: id, tenant_id, workflow_id, service_reference, status, verification_completed, e2e_test_passed
- ProRollback: id, tenant_id, workflow_id, trigger, description, status, triggered_at
- ProBulkJob: id, tenant_id, workflow_type, item_count, approval_reference, status
- ProAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events

### Catalogue Events
- product_created, product_status_review, product_status_approved, product_status_active, product_status_deprecated, product_status_retired
- bundle_created, bundle_decomposed
- price_created, bulk_prices_updated
- promotion_created, promotion_applied
- catalogue_searched

### Provisioning Events
- workflow_queued, resource_reserved, config_push_dispatched, config_push_completed
- service_activated, activation_confirmed, workflow_failed, rollback_triggered, rollback_completed, pro_agent_registered

## Product Lifecycle State Machine
```
draft -> review -> approved -> active -> deprecated -> retired
              \-> draft         \-> draft
```
Every transition requires a justification string and is audit-logged.

## Edge Cases Handled
- Product lifecycle transitions are strictly enforced — terminal states (retired) reject all transitions
- Bundle decomposition is idempotent at the order level — duplicate component orders are detected via service_order_receive conflict check
- Promotion application is idempotent — applying the same campaign to the same order twice is a no-op
- Price effective-date queries return the most recently effective price, not the most recently created
- Dry run cannot be bypassed even by privileged agents — hard rule, not configurable
- Resource reservations expire after TTL; expired reservations are auto-released
- Rollback preserves the original workflow record with status=rolled_back for audit
- Bulk price updates are atomic — validation of all items runs before any update is applied

## Composability Notes
Product catalogue feeds telecom_ord (order validation against active products). Bundle decomposition generates multiple service orders routed through the provisioning workflow engine. Price records feed telecom_bil (charge code generation). Activation confirmation triggers telecom_cus (lifecycle event). Receives provisioning tasks from telecom_ord. Reserves resources from telecom_inv (IPAM, circuit). Pushes configuration to NEs tracked in telecom_inv.

## World-Class Enhancements (v2.0)

1. **TMF620 Product Catalogue** — `ProProduct` model with category, status, characteristics, and version; governed release pipelines replace magic strings.
2. **Bundle Management Engine** — `ProBundle` with ordered components, pricing tiers, eligibility rules, and incompatibility guards; bridges catalogue to provisioning.
3. **Effective-Dated Price List** — `ProPrice` with charge type, tiered rate tables, and temporal queries ("price of X on date Y").
4. **Product Lifecycle State Machine** — enforced `draft→review→approved→active→deprecated→retired` with audited, justified transitions.
5. **Faceted Catalogue Search** — `search_catalogue` with category/status/price/keyword filters, pagination, and facet counts in a single call.
6. **Offer Eligibility Engine** — `evaluate_offer_eligibility` checks segment, geography, existing services, and credit class; returns disqualifiers and alternatives.
7. **Promotional Campaign Management** — `ProPromotion` with idempotent, audit-logged discount application; supports percentage, fixed, and free-month types.
8. **SLA Tier Catalogue** — `ProSlaTier` capturing availability, MTTR, provisioning SLA, and support tier; stamps SLA on each workflow at order time.
9. **Catalogue Versioning and Audit** — immutable version records per `(product_id, version)`; `diff_product_versions` for change history and rollback.
10. **Product Dependency Graph** — `build_product_dependency_graph` emits a JSON-LD DAG of capabilities, resources, and sub-products for impact analysis.
11. **TMF620 Import/Export** — `export_catalogue` / `import_catalogue` support `tmf620_json`, CSV, and XLSX with atomic transactional apply and conflict detection.
12. **Bulk Price Update with Approval Gate** — `bulk_update_prices` validates all items before applying; atomic rollback on any failure.
13. **Rules-Based Product Recommendation Engine** — `recommend_products` ranks eligible products via affinity rules without ML infrastructure dependency.
14. **Regulatory Compliance Tagging** — `regulatory_tags` per product; `get_compliance_report` flags missing CAK/jurisdiction tags across portfolio.
15. **Catalogue Health Dashboard** — `catalogue_health_dashboard` returns active/draft/deprecated counts, price coverage, SLA distribution, expiring promotions, and a 0–100 completeness score in O(n).

## New Methods

### `create_product` — register a TMF620-aligned product

```python
product = await svc.create_product(
    product_id="FTTX-100",
    name="Fibre 100 Mbps Home",
    category="broadband",
    characteristics={"speed_mbps": 100, "technology": "FTTH"},
    tenant_id="ke-nairobi",
    status="draft",
)
# Advance through lifecycle (each step requires a justification)
await svc.update_product_status("FTTX-100", "review", "QA sign-off received", tenant_id="ke-nairobi")
await svc.update_product_status("FTTX-100", "active", "Approved by product board", tenant_id="ke-nairobi")
```

### `search_catalogue` — faceted product discovery

```python
results = await svc.search_catalogue(
    tenant_id="ke-nairobi",
    category="broadband",
    status="active",
    keyword="fibre",
    min_price=500.0,
    max_price=5000.0,
    offset=0,
    limit=20,
)
# results["facets"] carries {"category": {...}, "status": {...}}
# for rendering filter chips without extra roundtrips
active_products = results["results"]
facets = results["facets"]
```

### `catalogue_health_dashboard` — at-a-glance catalogue KPIs

```python
health = await svc.catalogue_health_dashboard(tenant_id="ke-nairobi")
# {
#   "active_products": 42,
#   "draft_products": 7,
#   "deprecated_products": 3,
#   "price_coverage_pct": 88.1,
#   "promotions_expiring_soon": 2,
#   "completeness_score": 79,   # 0-100
#   ...
# }
if health["completeness_score"] < 80:
    alert("Catalogue completeness below threshold")
```
