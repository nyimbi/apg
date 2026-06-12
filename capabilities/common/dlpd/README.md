# Data Loss Prevention (DLPD)

DLPD is APG's generated-application capability for tenant-scoped data loss
prevention. It gives composed applications a deterministic, dependency-light
surface for data classification, policy enforcement, egress inspection,
quarantine, incident response, legal hold, review, audit, and UI composition.

The generated-app runtime stores hashes and metadata, not raw sensitive content.
It is intended to make APG applications executable quickly while leaving live
network controls, storage engines, classifier model providers, and compliance
systems behind explicit adapter boundaries.

## What The Capability Provides

- Policy lifecycle for owned, tenant-scoped DLP controls.
- Built-in and custom classifier lifecycle with review guardrails.
- Deterministic pattern classification for PII, PHI, PCI, secrets, financial
  records, and source code.
- Egress inspection for email, API exports, file sharing, chat, clipboard, and
  object storage.
- File and email scanning with per-attachment classification.
- Bulk scan jobs across multiple content items.
- Endpoint event recording (file copy, print, USB write, screen capture, clipboard).
- Shadow IT detection and cloud activity monitoring.
- Cross-channel DLP analytics with false-positive feedback loop.
- Multi-format reporting export (JSON, CSV).
- High-severity blocking/quarantine rules, large-export review, and restricted
  destination review.
- Encrypted quarantine metadata with legal-hold flags.
- Incident creation, investigation, resolution, and digest-backed audit.
- ML classifier training metadata registration.
- Regex pattern library with per-tenant isolation.
- First-class DLP AI-agent composition for policy, classifier, inspection,
  quarantine, incident, privacy, legal-hold, and lifecycle review work.
- Bytewax lifecycle-batch validation for policy, classifier, inspection,
  quarantine, incident, review, and agent mutations.
- Contract-derived UI routes, view payloads, visual theme tokens, and Bytewax
  event-stream adapter evidence.

## Quick Start

```python
from capabilities.common.dlpd.service import DlpdService

service = DlpdService()
tenant_id = "tenant-dlp"

classifier = service.register_classifier(
    classifier_id="cls-secrets",
    tenant_id=tenant_id,
    name="Secrets",
    classifier_type="built_in",
    sensitivity_label="restricted",
    pattern_keys=["secrets"],
)

policy = service.register_policy(
    policy_id="pol-email",
    tenant_id=tenant_id,
    name="Email egress",
    owner="security-ops",
    channels=["email"],
    classifiers=[classifier["id"]],
    default_action="quarantine",
)

inspection = service.inspect_egress(
    inspection_id="insp-1",
    tenant_id=tenant_id,
    policy_id=policy["id"],
    channel="email",
    subject_id="user-1",
    destination="external@example.com",
    content="api_key='SECRET123456789'",
)

agent = service.register_dlp_agent(
    agent_id="agent-dlp-steward",
    tenant_id=tenant_id,
    name="DLP Steward",
    runtime="codex",
    role="dlp_steward",
    scope="tenant:tenant-dlp",
    owner="security-platform",
    purpose="review DLP lifecycle batches",
    human_approval_required=True,
)

batch = service.validate_dlpd_lifecycle_batch(
    tenant_id=tenant_id,
    event_stream="bytewax",
    mutation_count=2,
    operation="dlp_agent_batch",
)
```

## API Reference

| Method | Description |
|---|---|
| `register_policy(...)` | Create a tenant-scoped DLP policy with channels, classifiers, and default action |
| `create_policy(...)` | Alias for `register_policy` with simplified signature |
| `update_policy(...)` | Mutate channels, classifiers, default action, or status |
| `policy_effectiveness(tenant_id, policy_id)` | Compute precision, action rate, and FP metrics for a policy |
| `register_classifier(...)` | Register a built-in or custom data classifier with pattern keys |
| `bulk_register_classifiers(...)` | Register multiple classifiers in one call |
| `regex_pattern_library(...)` | Add a named, tenant-scoped regex pattern to the shared library |
| `ml_classifier_train(...)` | Record training metadata for an ML classifier model |
| `classify_content(tenant_id, content, classifier_ids)` | Classify a string against tenant classifiers |
| `evaluate_content(text, policy_ids, tenant_id)` | Evaluate text against multiple policies and return combined verdict |
| `scan_file(...)` | Scan file content against a DLP policy |
| `scan_email(...)` | Scan email headers, body, and attachments against a DLP policy |
| `endpoint_event(...)` | Record and evaluate a DLP endpoint event (file copy, print, USB, etc.) |
| `inspect_egress(...)` | Full egress inspection with auto-classification, quarantine, and incident creation |
| `review_export(...)` | Approve a large-export inspection requiring independent review |
| `bulk_scan(...)` | Scan multiple content items against a policy; returns action summary |
| `quarantine_item(...)` | Manually quarantine an item |
| `release_quarantine(...)` | Release a quarantine item with approver sign-off |
| `incident_create(...)` | Manually open a DLP incident |
| `incident_investigate(...)` | Record investigation findings on an incident |
| `resolve_incident(...)` | Resolve an incident with resolution note |
| `false_positive_feedback(...)` | Submit a false-positive report for an inspection result |
| `shadow_it_detection(...)` | Record a Shadow IT detection event |
| `cloud_activity_monitoring(...)` | Record and risk-score a cloud activity event |
| `dlp_analytics(tenant_id)` | Aggregate DLP analytics across all channels |
| `reporting_export(...)` | Export DLP report data as JSON or CSV |
| `register_dlp_agent(...)` | Register a provider-neutral DLP AI agent |
| `validate_dlpd_lifecycle_batch(...)` | Validate a Bytewax DLP lifecycle batch |
| `dashboard_summary(tenant_id)` | Full tenant dashboard snapshot |
| `health_check()` | Service liveness check |
| `list_policies/classifiers/inspections/...` | Tenant-scoped list helpers for all entity types |
| `describe(tenant_id)` | Return the full capability contract |
| `evaluate(context)` | Run capability rules against an arbitrary context dict |

## World-Class Enhancements (v2.0)

The following 15 improvements define the production-readiness roadmap for DLPD.
Each is designed to be independently implementable without breaking the existing
synchronous API surface.

1. **Async-Native Service Core** — Refactor to `AsyncDlpdService` with `async def`
   throughout; use `asyncio.gather` for fan-out operations. Sync wrappers delegate
   via `asyncio.run`. Eliminates GIL contention inside async web frameworks.

2. **Streaming Egress Inspection** — `async def inspect_egress_stream(content_stream: AsyncIterator[bytes])`
   chunks content, hashes incrementally (rolling SHA-256), and emits partial
   classification decisions. Cuts peak memory for large exports from O(N) to
   O(chunk_size) and enables early-abort on first critical hit.

3. **Repository Pattern Backends** — Abstract `PolicyRepository`, `ClassifierRepository`,
   `InspectionRepository` etc. with async `get/put/delete/list_tenant`. Concrete
   implementations: `PostgresRepository` (asyncpg) and `RedisRepository` (aioredis).
   No service logic change; swap backends at construction time.

4. **Classifier Confidence Calibration** — Replace fixed confidence values with
   Platt-scaled posteriors `P(TP|hit) = sigmoid(a*raw_score + b)` fitted from
   false-positive feedback. `ml_classifier_train` writes calibration parameters.
   `policy_effectiveness` precision becomes a live posterior.

5. **Cross-Channel Risk Correlation** — `cross_channel_risk_profile(tenant_id, subject_id)`
   joins egress inspections, shadow IT detections, cloud activity events, and endpoint
   events per user/device. Computes a composite 7-day rolling risk score — the
   foundation of UEBA alerting.

6. **Policy Version History and Rollback** — Copy-on-write `PolicyVersion` records
   on every `update_policy` mutation. Exposes `list_policy_versions` and
   `rollback_policy(policy_id, version_id, actor)`. Required for SOC 2 / ISO 27001
   change-management controls.

7. **Classifier Hot-Reload** — `PatternCache` compiles regexes on first use, keyed
   by `(tenant_id, pattern_id, regex_hash)`, invalidated on `regex_pattern_library`
   writes. `reload_classifiers(tenant_id)` flushes stale patterns. Eliminates
   repeated re-compilation under high throughput.

8. **Structured Notification Dispatch** — `async def dispatch_incident_notifications(incident_id, tenant_id)`
   backed by a `NotificationAdapter` interface with concrete implementations for
   SMTP/SendGrid, Slack webhooks, PagerDuty events API, and a no-op test stub.
   Called automatically after `_open_incident` when an adapter is configured.

9. **Tokenization and Redaction Engine** — `async def redact_content(tenant_id, content, classifier_ids, mode)`
   where `mode` is `"mask"` (`***`), `"tokenize"` (reversible vault token via `encr`
   adapter), or `"hash"` (one-way SHA-256). Turns DLPD from detection-only to an
   active data sanitization pipeline.

10. **Legal Hold Lifecycle** — First-class `LegalHold` entity with `case_reference`,
    `custodians` list, `scope` predicate, and `status` (`active`/`released`). Exposes
    `place_legal_hold`, `add_custodian`, `release_legal_hold` (requires all-custodian
    sign-off), and `legal_hold_inventory`. Required for e-discovery and GDPR
    litigation exemptions.

11. **Composite Structured-Data Scanning** — `async def scan_structured(tenant_id, data, schema_hint, policy_id, actor)`
    walks JSON/CSV/SQL field paths, applies per-field classifiers via field-name
    heuristics (`ssn`, `credit_card`, `email`), and returns a per-field sensitivity
    map. Essential for database-export DLP where isolated columns are sensitive.

12. **Automated False-Positive Suppression Loop** — When `false_positive_feedback`
    accumulates N reports for the same pattern/classifier, automatically inserts a
    tenant-local suppression entry and lowers calibrated confidence. `suppression_report(tenant_id)`
    makes the suppression inventory auditable.

13. **Differential Privacy for Analytics** — `set_analytics_privacy_budget(tenant_id, epsilon)`
    injects Laplace noise on aggregate counts before returning them to non-admin
    callers. Prevents reconstruction of individual inspection records from repeated
    aggregate queries in multi-tenant deployments.

14. **OpenTelemetry Instrumentation** — Every public service method emits spans
    annotated with `tenant_id`, `policy_id`, `severity`, `action`, and `content_hash`
    (never raw content). Inspection latency histograms and incident rate counters
    exported to OTLP. Enables SLO tracking and capacity planning.

15. **Graph-Based Policy Conflict Detection** — `async def detect_policy_conflicts(tenant_id)`
    builds a policy-classifier bipartite graph, finds overlapping classifier sets
    across policies with differing `default_action` values, and returns a conflict
    report with suggested resolutions. Surfaces contradictions before go-live rather
    than at inspection time.

## New Methods

### `evaluate_content` — Multi-Policy Verdict

Evaluates a string against multiple policies simultaneously and returns a single
`overall_action` plus per-policy breakdown. Use this when content must satisfy
several policy domains (e.g., PII + secrets + source-code) before egress.

```python
result = service.evaluate_content(
    text="SELECT ssn, card_number FROM customers WHERE ...",
    policy_ids=["pol-pii", "pol-pci", "pol-source-code"],
    tenant_id="tenant-dlp",
)
# result["overall_action"] => "block"
# result["policy_results"] => [{"policy_id": "pol-pii", "action": "block", ...}, ...]
```

### `bulk_scan` — Batch Content Scanning

Scans a list of content items against one policy in a single call and returns
an action summary plus per-item results. Efficient for scanning exported datasets
before delivery.

```python
job = service.bulk_scan(
    tenant_id="tenant-dlp",
    job_id="scan-job-001",
    policy_id="pol-email",
    items=[
        {"id": "row-1", "content": "normal text"},
        {"id": "row-2", "content": "SSN: 123-45-6789"},
        {"id": "row-3", "content": "api_key='abc123'"},
    ],
    actor="export-pipeline",
)
# job["action_summary"] => {"allow": 1, "quarantine": 1, "block": 1}
# job["results"] => [{"item_id": "row-1", "action": "allow", ...}, ...]
```

### `scan_email` — Full Email Inspection

Scans email headers, body, and attachment content in a single pass. Useful for
gateway integration where the full message is available as structured parts.

```python
result = service.scan_email(
    tenant_id="tenant-dlp",
    headers={"from": "user@corp.com", "to": "partner@external.com", "subject": "Q4 data"},
    body="Please find attached the customer export.",
    attachments=[{"name": "export.csv", "content": "ssn,name\n123-45-6789,Alice"}],
    policy_id="pol-email",
    actor="mail-gateway",
)
# result["action"] => "quarantine"
# result["severity"] => "high"
```

### `cloud_activity_monitoring` — Cloud Risk Scoring

Records a cloud provider event and maps the caller-supplied `risk_score` (0.0–1.0)
to a severity tier. Feeds the cross-channel risk profile once improvement #5 is
implemented.

```python
event = service.cloud_activity_monitoring(
    tenant_id="tenant-dlp",
    event_id="evt-s3-001",
    provider="aws",
    service="s3",
    user_id="user-42",
    action="GetObject",
    resource="s3://corp-pii-bucket/customers.csv",
    risk_score=0.85,
)
# event["severity"] => "critical"
```

### `policy_effectiveness` — Precision Metrics

Returns precision, action rate, and false-positive count for a policy. Use this
to tune `confidence_threshold` on classifiers or decide whether a policy needs
pattern refinement.

```python
metrics = service.policy_effectiveness(
    tenant_id="tenant-dlp",
    policy_id="pol-email",
)
# {
#   "precision": 0.9333,
#   "action_rate": 0.1500,
#   "false_positive_count": 2,
#   "true_positive_estimate": 14,
#   ...
# }
```

## Composition Contract

Use `get_capability_contract()` when a compiler, generator, or larger APG
application needs to inspect DLPD.

```python
from capabilities.common.dlpd.capability_contract import get_capability_contract

contract = get_capability_contract("tenant-dlp")
routes = contract["ui"]["routes"]
rules = contract["rule_engine"]["rules"]
adapters = contract["configuration"]["adapters"]
```

Important adapter evidence:

- `generated_app_runtime`: `service.DlpdService`
- `event_stream`: `bytewax`
- `agent_adapter`: `aicr_provider_neutral_dlp_agent_adapter`
- `security_framework`: `secu`
- `encryption`: `encr`
- `nlp_core`: `nlpc`
- `anomaly_detection`: `anom`
- `audit_sink`: `audl`
- `message_bus`: `mqeb`
- `compliance`: `comp`

## Agent Composition

DLPD agents are first-class records, not comments in configuration. They are
provider-neutral and may use `codex`, `claude_code`, `opencode`, or `pi` behind
the AICR adapter contract. Each agent requires a tenant, name, runtime, role,
scope, owner, purpose, and machine-contribution disclosure. Privileged roles
such as quarantine, incident response, privacy, legal hold, lifecycle batch, and
DLP steward agents enter `pending_review` unless human approval is recorded.

Supported roles:

- `policy_reviewer`
- `classifier_reviewer`
- `inspection_triage_agent`
- `quarantine_reviewer`
- `incident_response_reviewer`
- `privacy_reviewer`
- `legal_hold_reviewer`
- `lifecycle_batch_reviewer`
- `dlp_steward`

## Lifecycle Batches

DLPD lifecycle batches are explicit Bytewax-governed records. The generated
runtime accepts `policy_batch`, `classifier_batch`, `inspection_batch`,
`quarantine_batch`, `incident_batch`, `review_batch`, and `dlp_agent_batch`
operations only when they contain mutations and declare `event_stream="bytewax"`.
Broker-specific queue or broker-core routing is intentionally denied for this packet.

## Screens

The contract exposes route metadata for dashboard, policies, classifiers,
channels, inspections, incidents, quarantine, reviews, legal hold, analytics,
agents, lifecycle batches, audit, and settings. The view helpers in `views.py`
return dependency-light payloads for generated Python applications.

## Guardrails

DLPD includes deterministic rules for tenant context, policy ownership, policy
channels, classifiers, active policies, covered channels, destinations,
classifier labels, custom classifier review, classifier confidence, sensitive
classification labels, source-code review, secret/high-severity blocking,
large-export review, external/restricted destinations, quarantine encryption,
quarantine content hashes, legal hold, incident ownership/resolution,
independent review, raw-content retention denial, Bytewax batch mutation,
tenant isolation, required audit evidence, provider-neutral AI-agent
registration, privileged-agent review, and Bytewax lifecycle batches.

## Verification

Focused package checks:

```bash
./.venv/bin/python -m py_compile capabilities/common/dlpd/__init__.py capabilities/common/dlpd/capability_contract.py capabilities/common/dlpd/dlp_engine.py capabilities/common/dlpd/models.py capabilities/common/dlpd/service.py capabilities/common/dlpd/api.py capabilities/common/dlpd/views.py capabilities/common/dlpd/app.py capabilities/common/dlpd/test_capability_contract.py capabilities/common/dlpd/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/dlpd/test_capability_contract.py capabilities/common/dlpd/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/dlpd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/dlpd --json
```

---

© 2025 Datacraft — Nyimbi Odero
