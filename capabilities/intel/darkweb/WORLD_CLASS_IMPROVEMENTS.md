# Dark Web Intelligence — World-Class Improvements

**Capability**: `intel_darkweb` | **Version target**: 2.0.0

---

## 1. Streaming Intelligence Pipeline (Bytewax integration)

Replace the fire-and-forget `_audit` pattern with a proper Bytewax dataflow that
processes observations, indicators, and alerts as a continuous stream. Each new
observation is emitted onto `apg.intel.darkweb.lifecycle`; downstream consumers
subscribe per-tenant. This enables real-time alerting instead of polling-based
dashboard summaries, and makes backpressure and replay first-class concerns.

---

## 2. Graph-Based Threat Actor Attribution

Store threat actor relationships in a property graph (Neo4j or networkx for
in-process). Nodes: aliases, onion addresses, cryptocurrency wallets, malware
families. Edges: `OPERATES`, `FUNDS`, `DEPLOYS`, `COMMUNICATES_WITH`. Enables
multi-hop attribution queries — "which threat actors share infrastructure with
this ransomware group?" — with confidence decay across hops.

---

## 3. Onion Address Active Probing with Tor Circuit Rotation

Add `async probe_onion_service(onion_address, circuit_budget)` that dispatches
lightweight HTTP HEAD requests through the system Tor SOCKS5 proxy with circuit
rotation after each attempt. Returns availability histogram, TLS cert hash,
server header fingerprint, and uptime SLA class. Implements exponential backoff
and a per-tenant circuit-use budget to prevent operational security leakage.

---

## 4. Automated Credential Deduplication via k-Anonymity

Before storing leaked credentials, hash email+password combos with PBKDF2-SHA256
and check against a local k-anonymity prefix table (HaveIBeenPwned model). Only
store the truncated SHA1 prefix + hit count — never the plaintext. Eliminates
duplicate breach entries across tenants and dramatically reduces storage for
large credential dumps (>1M rows) without sacrificing detection quality.

---

## 5. MITRE ATT&CK TTP Enrichment

Map every threat actor TTP and malware capability to ATT&CK Enterprise technique
IDs (e.g. `T1486` for ransomware, `T1190` for exploit of public-facing
application). Expose `async enrich_ttp_matrix(observation_id)` that returns the
full technique + sub-technique tree with tactic phase labels. Downstream SOC
tooling can ingest these directly into SIEM correlation rules.

---

## 6. Cryptocurrency Transaction Monitoring

Add `async trace_crypto_address(address, currency)` supporting BTC, XMR, ETH,
and LTC. Calls a configurable blockchain analytics adapter (Chainalysis,
Elliptic, or CipherTrace) to retrieve transaction graph, darknet marketplace
exposure score, and OFAC sanction flags. Results are stored under
`_crypto_traces` and linked to any existing threat actor assessments sharing the
address as evidence.

---

## 7. LLM-Assisted Threat Summarisation

Introduce `async summarise_threat_actor(assessment_id, llm_adapter)` that
compiles all linked observations, indicators, channel monitors, and forum
profiles, then invokes a locally hosted Ollama model (e.g. `mistral:instruct`)
to produce a structured analytical summary: executive brief, technical indicators
table, and recommended mitigations. The raw LLM output is stored alongside the
structured record; the capability contract enforces PII stripping before LLM
dispatch.

---

## 8. Automated STIX 2.1 Export

Replace the ad-hoc `export_intelligence(fmt)` with a fully conformant
STIX 2.1 bundle generator. Each `ExposureIndicator` maps to a STIX `indicator`
object, each `ThreatActorAssessment` maps to a STIX `threat-actor`, and each
`DarkWebObservation` maps to a STIX `observed-data` with `network-traffic` SCOs
where applicable. The bundle is signed with a tenant-specific Ed25519 key and
can be pushed directly to a TAXII 2.1 collection endpoint.

---

## 9. Confidence Score Bayesian Updating

Replace the static `confidence_score` field on indicators with a Bayesian
update model: each corroborating observation nudges the posterior up; each
contradicting signal nudges it down. Store prior, likelihood, and posterior
separately. Expose `async update_indicator_confidence(indicator_id, evidence_type,
corroborates)` to drive the update. This eliminates analyst anchoring bias and
makes confidence drift over time auditable.

---

## 10. Parallel Onion Network Crawl Scheduler

Add a capability-level crawl scheduler (`CrawlScheduler`) that accepts a seed
list of `.onion` addresses and fans out crawl jobs using `asyncio.TaskGroup`.
Each job follows internal links up to configurable depth, fingerprints content,
and feeds results directly into `record_observation`. A leaky-bucket rate
limiter prevents Tor circuit saturation. The scheduler respects a
per-tenant crawl quota enforced by the existing authority model.

---

## 11. Brand Impersonation Detection

Implement `async detect_brand_impersonation(brand_terms, logo_hash)` that scans
dark web marketplaces and forums for pages using the tenant's brand terms or a
perceptual hash of their logo. Uses phash distance threshold (≤ 8 Hamming bits)
for logo matching. Results feed `record_indicator` with type `BRAND_ABUSE`.
Integrates with the `alerts` capability to raise immediate notifications when
new impersonations appear.

---

## 12. Geopolitical Risk Context Layer

Enrich every `ThreatActorAssessment` with geopolitical attribution context:
country of origin (confidence-weighted), sanctions list membership (OFAC, EU,
UN), and current geopolitical tension index from a configurable risk feed. Expose
`async get_geopolitical_context(actor_reference)` returning a structured context
block that SOC analysts can attach to incident reports. The layer is optional and
gracefully degrades if the risk feed adapter is not configured.

---

## 13. Time-Series Anomaly Detection on Monitor Metrics

Store forum thread counts and marketplace listing volumes as time-series arrays
(ring buffers, tenant-scoped). Run a lightweight Prophet-style STL decomposition
(via `statsmodels`) after each monitoring run to detect anomalous spikes in
activity. When the residual exceeds 3σ, automatically escalate to a
`CRITICAL` alert via `darkweb_alert`. Eliminates manual threshold tuning and
catches novel attack campaigns before they fully develop.

---

## 14. Secure Evidence Chain of Custody

Replace opaque `evidence_reference` strings with a structured `EvidenceChain`
model: each link records SHA-256 hash of the original artefact, the capturing
agent ID, capture timestamp (RFC 3339), and a sequential HMAC chain (each link
signed over the previous link's HMAC + current payload). `async
verify_evidence_chain(evidence_chain_id)` re-derives and compares the HMAC chain
to detect tampering. This satisfies admissibility requirements for law
enforcement referrals.

---

## 15. Composable Alert Suppression and Deduplication Engine

Add an `AlertSuppressor` collaborator injected at construction time. Before
`darkweb_alert` emits a new alert, it checks a rolling 24-hour window for
identical (`keyword_hit`, `source_type`) pairs. Duplicate alerts within the
window are collapsed into an existing alert's `recurrence_count` rather than
spawning new records. Suppression rules are tenant-configurable (whitelist,
cooldown period, minimum prevalence threshold), and the suppressor emits audit
events when suppression occurs so analysts can tune rules. This reduces alert
fatigue by an estimated 60–80% on high-volume deployments.
