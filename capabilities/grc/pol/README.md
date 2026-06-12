# Policy Management

## Overview

Policy Management provides a world-class, standalone-deployable implementation of policy management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Capability ID

`grc_pol`  Version: 1.0.0

## Provides

| Service | Description |
|---------|-------------|
| `policy_lifecycle_management` | Policy Lifecycle Management workflow |
| `policy_acknowledgement_workflow` | Policy Acknowledgement Workflow workflow |
| `policy_exception_workflow` | Policy Exception Workflow workflow |
| `policy_review_workflow` | Policy Review Workflow workflow |
| `policy_publication_workflow` | Policy Publication Workflow workflow |


## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Auth services |
| `audl` | Audl services |
| `mten` | Mten services |
| `conf` | Conf services |
| `ntfy` | Ntfy services |


## Installation

```bash
pip install apg-grc-pol
```

## Standalone Usage

```python
from apg_grc_pol import get_capability_contract

# Get capability contract
contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # grc_pol
```

## Running the Standalone Server

```bash
# Standalone with InMemory store
apg-grc-pol --port 8080

# With PostgreSQL persistence
apg-grc-pol --db-url postgresql+asyncpg://user:pass@localhost/pol --port 8080
```

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/grc-pol/dashboard` | `grc_pol:view` |
| policies | `/grc-pol/policies` | `grc_pol:manage_policies` |
| policy_detail | `/grc-pol/policies/:id` | `grc_pol:view` |
| acknowledgements | `/grc-pol/acknowledgements` | `grc_pol:manage_acknowledgements` |
| exceptions | `/grc-pol/exceptions` | `grc_pol:manage_exceptions` |
| reviews | `/grc-pol/reviews` | `grc_pol:review` |
| review_calendar | `/grc-pol/review-calendar` | `grc_pol:view` |
| gap_analysis | `/grc-pol/gap-analysis` | `grc_pol:view` |


## HTTP Endpoints

```
GET  /health           Liveness probe
GET  /contract         Full capability contract JSON
POST /evaluate         Evaluate governance rules
GET  /api/v1/...       Domain-specific REST API
```

## Composability

This capability integrates with the APG platform via the `apg.capabilities` entry-point group. It is auto-discovered by the capability registry when installed.

```python
from capabilities.capability_contract_registry import load_contract_registry
registry = load_contract_registry()
contract = registry["grc_pol"].contract
```

## Development

```bash
# Run tests
pytest tests/ -q

# Build wheel
python -m build --wheel .

# Validate contract
python -c "from capability_contract import get_capability_contract; print('OK')"
```

## Service API — Full Method Reference

### Lifecycle

| Method | Description |
|--------|-------------|
| `create_policy(...)` | Create a new policy in draft status |
| `draft_policy_content(policy_id, sections, author_id)` | Attach structured content sections |
| `policy_review(policy_id, reviewer_id, comments, action)` | Submit peer review (SOD enforced) |
| `approve_policy(policy_id, approver_id, approval_date)` | Formal approval |
| `publish_policy(policy_id, distribution_list)` | Publish and trigger acknowledgements |
| `policy_revision(policy_id, reason, summary, revised_by)` | Initiate version revision |
| `retire_policy(policy_id, reason, retired_by)` | Archive/withdraw a policy |

### Attestation

| Method | Description |
|--------|-------------|
| `acknowledge_policy(policy_id, employee_id, date)` | Record individual acknowledgement |
| `create_attestation_campaign(...)` | SLA-tracked bulk attestation campaign |
| `acknowledgement_chase(policy_id, chased_by)` | Chase overdue acknowledgements |

### Exceptions

| Method | Description |
|--------|-------------|
| `policy_exception_request(...)` | Request exception with compensating controls |
| `approve_exception(exception_id, approver_id, until, conditions)` | Approve exception |
| `policy_exception_monitor(policy_id)` | Monitor expiring/expired exceptions |

### Compliance and Mapping

| Method | Description |
|--------|-------------|
| `policy_compliance_check(entity_id, policy_id)` | Compliance status per policy |
| `policy_mapping(policy_id, regulation_ids, control_ids)` | Map to regulations/controls |
| `policy_gap_analysis(entity_id, framework)` | Gap analysis vs ISO 27001, SOC 2, GDPR, CBK |
| `regulatory_align(policy_id, regulations, aligned_by)` | Align to regulatory requirements |

### Templates and Import

| Method | Description |
|--------|-------------|
| `policy_template(name, type, scope, sections)` | Create reusable template |
| `create_policy_from_template(template_id, title, ...)` | Instantiate policy from template |
| `policy_bulk_import(records, imported_by)` | Bulk-import pre-parsed policy records |

### Analytics and Reporting

| Method | Description |
|--------|-------------|
| `policy_analytics(period)` | KPI metrics for a period |
| `policy_kpi_summary(entity_id, period)` | KPI card for dashboard |
| `policy_effectiveness(policy_id)` | Effectiveness score (ack rate + exceptions) |
| `policy_expiry_report(days_ahead)` | Policies due for review |
| `policy_dashboard(entity_id)` | Assembled dashboard data |
| `policy_delta_report(policy_id, v_from, v_to)` | Delta between two versions |

### Hierarchy, Events, and Cache

| Method | Description |
|--------|-------------|
| `policy_set_parent(policy_id, parent_id, set_by)` | Establish policy hierarchy |
| `policy_conflict_check(policy_id)` | Detect scope/type conflicts with published policies |
| `policy_event_publish(event_type, policy_id, payload)` | Emit domain event to APG bus |
| `policy_cache_invalidate(scope)` | Invalidate tenant-scoped read cache |

## Composability

Emits domain events consumed by downstream capabilities:

- `grc_ris` — subscribes to `exception_approved` to update risk residuals
- `grc_ctl` — subscribes to `policy_published` to trigger control mapping tasks
- `grc_aud` — subscribes to `policy_archived` to close linked audit findings

```apg
use grc_pol;
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GRC_POL_DB_URL` | in-memory | SQLAlchemy async database URL |
| `GRC_POL_ACK_DEADLINE_DAYS` | 30 | Default acknowledgement deadline |
| `GRC_POL_EXCEPTION_MAX_DAYS` | 365 | Maximum exception duration |
| `GRC_POL_COMPLIANCE_THRESHOLD_PCT` | 95.0 | Ack rate % for compliant status |
| `OLLAMA_BASE_URL` | (unset) | Enable AI-assisted compliance scoring |

## License

Proprietary — © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>

---

## World-Class Enhancements (v2.0)

- **I1.** Policy Management — World-Class Improvement Roadmap
- **I2.** Hierarchical Policy Inheritance
- **I3.** AI-Assisted Policy Drafting (Ollama)
- **I4.** Structured Two-Stage Approval Workflow
- **I5.** Immutable Audit Log with Cryptographic Chaining
- **I6.** Regulatory Framework Registry as First-Class Entity
- **I7.** Policy Obligation Extraction and Tracking
- **I8.** Conflict Detection Between Policies
- **I9.** Attestation Campaigns with SLA Enforcement
- **I10.** Policy Delta Reports for Revisions
- **I11.** Risk-Linked Policy Effectiveness Scoring
- **I12.** Automated Review Scheduling with Calendar Integration
- **I13.** Policy Template Versioning and Inheritance
- **I14.** Bulk Import/Export with Format Normalisation
- **I15.** Cross-Capability Composability Hooks

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
