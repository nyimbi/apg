# World-Class Improvements — telecom_pro Product Catalogue

Capability: Product Catalogue (telecom_pro)
Domain: telecom | Path: capabilities/telecom/pro
Focus: Service catalogue, bundle management, price list

---

## Improvement 1 — Service Catalogue with TMF620 Alignment

**Current state**: No structured service/product catalogue. Product codes are bare strings with no schema, hierarchy, or lifecycle state.

**Improvement**: Introduce a `ProProduct` model aligned with TM Forum TMF620 Product Catalogue API. Products carry `category`, `status` (active/deprecated/draft), `characteristics`, and version. The service exposes `create_product`, `get_product`, `list_products`, `deprecate_product` as first-class async methods instead of ad-hoc product_code strings.

**Impact**: Enables governed product release pipelines, downstream order validation against catalogue, and eliminates magic strings in provisioning flows.

---

## Improvement 2 — Bundle Management Engine

**Current state**: No concept of bundles. A "bulk_service" product_code is a stub with no composition semantics.

**Improvement**: Add `ProBundle` model with ordered constituent products (`components: list[BundleComponent]`), pricing tiers, eligibility rules, and incompatibility guards. Methods `create_bundle`, `validate_bundle_eligibility`, `decompose_bundle_to_orders` bridge the catalogue to the provisioning workflow engine.

**Impact**: Enables triple-play and convergent bundles (voice + data + TV), automated bundle decomposition into provisioning sub-orders, and commercial eligibility checks before order acceptance.

---

## Improvement 3 — Hierarchical Price List with Effective Dating

**Current state**: No price list. Pricing is invisible to the provisioning layer.

**Improvement**: `ProPrice` model with `effective_from`, `effective_to`, currency, charge type (one-time/recurring/usage), and tiered rate tables. `create_price`, `get_effective_price`, `list_prices_for_product` support temporal queries — "what was the price of product X on date Y".

**Impact**: Enables accurate billing integration (telecom_bil), quote generation, promotional pricing windows, and regulatory price transparency.

---

## Improvement 4 — Product Lifecycle State Machine

**Current state**: Product status is an uncontrolled string field.

**Improvement**: Enforce valid state transitions: `draft → review → approved → active → deprecated → retired`. Each transition is audited and requires a justification string. Invalid transitions raise `ValueError` with the full allowed-transitions map in the error message.

**Impact**: Prevents accidental activation of draft products, ensures deprecated products cannot be ordered, and provides a complete lifecycle audit trail for regulatory review.

---

## Improvement 5 — Real-Time Catalogue Search with Faceting

**Current state**: No search capability; consumers must iterate all products.

**Improvement**: `search_catalogue` method with support for `category`, `status`, `price_range`, `keyword` filters and pagination (`offset`/`limit`). Returns facet counts alongside results so UIs can render filter chips without additional roundtrips.

**Impact**: Reduces catalogue browsing latency, enables self-service product discovery APIs, and supports customer-facing e-commerce flows.

---

## Improvement 6 — Offer Composition and Eligibility Engine

**Current state**: No mechanism to determine which products a customer is eligible for.

**Improvement**: `evaluate_offer_eligibility(customer_profile, product_id)` checks customer segment, geography, existing services, and credit class against product eligibility rules. Returns `eligible: bool`, `disqualifiers: list[str]`, and `recommended_alternatives: list[str]`.

**Impact**: Enables agent-assist tools, reduces order fallout from ineligible orders, and supports personalised upsell recommendations.

---

## Improvement 7 — Promotional Campaign Management

**Current state**: No promotion or discount layer.

**Improvement**: `ProPromotion` model with `campaign_id`, `discount_type` (percentage/fixed/free-month), `applies_to` (product list or category), `valid_from/valid_to`, and `usage_limit`. Methods: `create_promotion`, `apply_promotion_to_order`, `list_active_promotions`. Promotion application is idempotent and audit-logged.

**Impact**: Enables time-limited offers, win-back campaigns, and partner bundle discounts without touching core pricing records.

---

## Improvement 8 — SLA Tier Catalogue

**Current state**: No SLA parameters attached to products.

**Improvement**: `ProSlaTier` model capturing `availability_pct`, `mttr_hours`, `provisioning_sla_hours`, `support_tier` (standard/priority/premium). Products reference a `sla_tier_id`. `get_sla_for_order` resolves the applicable SLA at order time and stamps it on the workflow.

**Impact**: Drives automated jeopardy detection thresholds, informs NOC prioritisation, and fulfils customer contract obligations automatically.

---

## Improvement 9 — Catalogue Versioning and Change-Set Audit

**Current state**: Product records are mutated in place with no history.

**Improvement**: Immutable product versions — each update creates a new version record keyed by `(product_id, version)`. `get_product_version`, `list_product_versions`, `diff_product_versions` surface change history. The active version pointer is updated atomically.

**Impact**: Enables rollback to previous product definitions, satisfies regulatory change-management requirements, and provides traceability for pricing disputes.

---

## Improvement 10 — Cross-Capability Product Dependency Graph

**Current state**: Product composability with other capabilities (telecom_inv, telecom_bil, telecom_ord) is implicit and undocumented at runtime.

**Improvement**: `build_product_dependency_graph(product_id)` returns a DAG of capabilities, resources, and sub-products required to fulfil the product. Nodes include required inventory items (telecom_inv), billing charge codes (telecom_bil), and provisioning templates (telecom_pro). Graph serialises to JSON-LD for machine consumption.

**Impact**: Enables impact analysis before product retirement, automated feasibility checks, and integration with telecom_ord for end-to-end order validation.

---

## Improvement 11 — Catalogue Import/Export with TM Forum Open API Format

**Current state**: Export is a raw JSON dump with no schema compliance.

**Improvement**: `export_catalogue(format)` supports `tmf620_json`, `csv`, and `xlsx`. `import_catalogue(data, format)` validates against TMF620 schema, detects conflicts with existing products, and applies changes in a single atomic transaction with rollback on validation failure.

**Impact**: Enables catalogue migration between environments, integration with third-party BSS/OSS systems, and regulatory submissions.

---

## Improvement 12 — Async Bulk Price Update with Approval Gate

**Current state**: No bulk price management.

**Improvement**: `bulk_update_prices(updates: list[PriceUpdate], approval_reference)` processes a list of price changes, validates effective dates and currency consistency, and applies only after approval gate check. Returns per-item success/failure and a summary change-set record.

**Impact**: Enables annual price reviews, currency revaluation, and promotional period rollouts across hundreds of products atomically.

---

## Improvement 13 — Product Recommendation Engine (Rules-Based)

**Current state**: No recommendation capability.

**Improvement**: `recommend_products(customer_id, context)` applies a configurable rules engine (eligibility + affinity rules loaded from catalogue metadata) to return a ranked list of products. Affinity rules encode "customers who have X also buy Y" logic from aggregate usage data injected at service init.

**Impact**: Enables agent-assist upsell, automated cross-sell in self-service portals, and revenue uplift without ML infrastructure dependency.

---

## Improvement 14 — Regulatory Compliance Tagging

**Current state**: No regulatory metadata on products.

**Improvement**: Products carry `regulatory_tags: list[str]` (e.g. `CAK_licensed`, `universal_service`, `roaming_regulated`). `get_compliance_report(tenant_id)` aggregates product portfolio by tag, flags products missing required tags for the tenant's jurisdiction, and emits structured compliance findings.

**Impact**: Supports Communications Authority of Kenya (CAK) licensing obligations, prevents sale of regulated services without correct tagging, and automates compliance reporting.

---

## Improvement 15 — Real-Time Catalogue Health Dashboard

**Current state**: `dashboard_summary` is workflow-centric with no catalogue KPIs.

**Improvement**: `catalogue_health_dashboard(tenant_id)` returns: active product count, draft/deprecated ratios, price coverage (products with no current price), SLA tier distribution, promotions expiring within 7 days, and catalogue completeness score (0–100). All metrics computed in a single pass for O(n) performance.

**Impact**: Gives product managers an at-a-glance view of catalogue health, surfaces incomplete products before they cause order fallout, and tracks catalogue maturity over time.
