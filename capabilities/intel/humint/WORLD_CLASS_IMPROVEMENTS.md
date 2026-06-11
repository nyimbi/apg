# HUMINT Capability – World-Class Improvements

**Capability**: `intel_humint` v1.1.0  
**Author**: Nyimbi Odero  
**Date**: 2026-06-11

---

## 1. Persistent Store Integration via Async SQLAlchemy

The current service is fully in-memory. All dictionaries reset on restart.
Wire `HUMINTService` to an async SQLAlchemy session factory so that every
`_audit`, source, debriefing, and intel record survives process restarts and
can be queried cross-instance. The `store` adapter slot already exists in
`__init__`; it just needs a concrete implementation backed by PostgreSQL.

**Impact**: production-readiness, data durability, horizontal scaling.

---

## 2. Event-Sourced Audit Trail via Bytewax

Every `_audit()` call produces a Python dict appended to a list. Replace
this with a Bytewax `DataflowSource` that emits immutable
`HUMINTAuditEvent` records to the `apg.intel.humint.lifecycle` topic.
Downstream consumers (alerting, compliance, SIEM) can subscribe without
polling the service. Bytewax already appears in the contract; it just is not
wired.

**Impact**: real-time observability, separation of concerns, zero-polling
compliance pipeline.

---

## 3. Source Network Graph via GraphRAG / NetworkX

`source_network_analysis` computes a flat subject-overlap list. Extend it
to build a proper directed graph (NetworkX `DiGraph` or a `grph` adapter
call) where nodes are sources and edges represent shared subjects, handlers,
or authority chains. Expose betweenness centrality and community detection
so analysts can spot broker sources and compartment leaks in O(n log n)
rather than O(n²).

**Impact**: advanced network mapping, compartment analysis, faster
compromise detection.

---

## 4. Temporal Credibility Decay Model

`adjusted_credibility` is computed once at collection time and never
decays. Intelligence staleness is a core HUMINT concern. Apply an
exponential decay `C(t) = C₀ · e^(−λt)` where `λ` is configurable per
classification level. A `TOP_SECRET` item from 90 days ago should carry
significantly lower effective credibility than yesterday's `UNCLASSIFIED`
report.

**Impact**: analytically correct credibility scores, prevents stale intel
driving decisions.

---

## 5. Multi-Factor Source Vetting with Configurable Thresholds

`source_vetting` currently derives pass/fail from a hash. Replace with
pluggable vetting modules (financial, biometric, OSINT lookup) that each
return a confidence score. The aggregate vetting score feeds directly into
the initial reliability grade rather than being a standalone record. Expose
`vetting_threshold` as a tenant-level configuration key.

**Impact**: evidence-based vetting, configurable risk tolerance per
deployment context.

---

## 6. Automated Deconfliction with Cross-Tenant Awareness

`contact_deconfliction` only looks at the current tenant's plans. In
multi-agency deployments the same source may be simultaneously handled by
a partner unit under a different tenant. A deconfliction broker (backed by
the `auth` adapter's cross-tenant permission grants) should flag cross-tenant
dual coverage before it creates compartment contamination or operational
security failures.

**Impact**: prevents duplicate handling, protects source security in
federated deployments.

---

## 7. LLM-Powered Debriefing Summarisation via `nlpc` Adapter

Debriefings currently store a `topic` string and a `credibility_score`.
Wire the `nlpc` adapter to run an async summarisation pass over the
`content` of each debriefing, extract named entities, and auto-populate
`key_actors`, `locations`, and `events` fields. Downstream `analytical_assessment`
and `humint_report` can then aggregate structured data rather than
unstructured text.

**Impact**: analyst efficiency, structured knowledge extraction, enables
semantic search via the `ragn` adapter.

---

## 8. Encrypted Source Identity Vault

Source `id`, `owner_id`, and `protection_reference` are stored as
plaintext strings. In production these are the most sensitive fields in the
entire system. Introduce an `IdentityVaultAdapter` that encrypts
identity-bearing fields at rest using AES-256-GCM, with per-tenant key
rotation. The service stores only ciphertext + key-id; the adapter handles
all crypto. The contract's `source_identity_disclosure_action_denied` rule
is rendered enforceable at the storage layer.

**Impact**: source protection at rest, regulatory compliance, key rotation
without re-keying every record.

---

## 9. Probabilistic False-Flag Scoring with Bayesian Updates

`false_flag_detection` uses deterministic bit-shifts on a content hash —
this produces stable but arbitrary risk scores. Replace with a Bayesian
classifier: maintain per-source prior `P(double|evidence)` and update it
each time a new inconsistency indicator is observed. Posterior updates are
cheap (Beta-Binomial conjugate prior) and produce calibrated probabilities
rather than hash artifacts.

**Impact**: analytically calibrated deception detection, posterior updates
propagate automatically to downstream risk scores.

---

## 10. Real-Time Source Welfare Monitoring with Alert Thresholds

`source_welfare_score` is recorded per contact report but no alert fires
when welfare trends negative. Add a welfare trend monitor that computes a
rolling 3-report moving average per source, and emits a
`WELFARE_ALERT` event (to Bytewax) when the average drops below a
tenant-configurable threshold. Handler performance scores are then computed
from the welfare trend rather than a single report.

**Impact**: proactive source welfare, handler accountability, audit evidence
for oversight bodies.

---

## 11. Classification Boundary Enforcement at API Layer

The API (`api.py`) does not currently validate that a requesting user's
clearance level is ≥ the classification of the record being accessed. Add
a `ClassificationGuard` middleware backed by the `auth` adapter that checks
`user.clearance >= record.classification` before returning any source,
debriefing, or intel item. Combine with the existing `cross_tenant_humint_write_denied`
rule for a complete MLS (Multi-Level Security) enforcement chain.

**Impact**: mandatory access control, prevents over-exposure of classified
records to under-cleared users.

---

## 12. Semantic Search over Intel Collections via RAG

`analytical_assessment` filters on exact subject string equality. Replace
with a vector similarity search using the `ragn` adapter: embed each intel
item's content fingerprint + subject at collection time, then query by
semantic proximity at assessment time. Analysts no longer need exact subject
strings; related topics surface automatically.

**Impact**: recall-oriented intelligence search, eliminates brittle string
matching, enables cross-subject pattern discovery.

---

## 13. Automated Collection Requirements Prioritisation with Gap Feedback Loop

`collection_requirements` generates requirements but never closes the loop
with actual collections. Add a feedback method that compares generated
requirements against `_intel_collections` at the end of each
`reporting_cycle`, computes a per-priority coverage ratio, and
auto-escalates uncovered requirements to the next urgency tier. This
implements a Plan-Collect-Process-Produce-Disseminate cycle at the service
layer.

**Impact**: adaptive collection management, prevents intelligence gaps from
persisting across cycles, closes the intelligence cycle loop.

---

## 14. Handler–Source Assignment Optimisation

Handlers are assigned ad-hoc via contact plans with no load balancing.
Add an `assign_handler` method that models handler capacity as a constraint
satisfaction problem: given a new source's risk level, required contact
frequency, and geographic zone, suggest the optimal handler from the active
roster using a weighted scoring function (workload, geography, cleared level,
specialisation). Expose the optimisation parameters as tenant configuration.

**Impact**: workload distribution, reduces handler overload detections,
optimises operational security by limiting handler exposure.

---

## 15. Time-Bound Authority Expiry Enforcement with Automated Suspension

`SourceAuthority.expires_at` is stored as a string but never checked at
operation time. Every sync CRUD method that validates `authority_present`
should also check `authority.expires_at > utcnow()`. If the authority has
expired, automatically transition all linked sources to `SUSPENDED` status,
emit `AUTHORITY_EXPIRED` events, and notify the approver via the `ntfy`
adapter. This closes the most significant governance gap in the current
implementation.

**Impact**: closed authority lifecycle, prevents expired mandates from
authorising active source handling, direct regulatory compliance requirement.
