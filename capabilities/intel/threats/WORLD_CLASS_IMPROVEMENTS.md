# World-Class Improvements: Threat Intelligence (`intel_threats`)

**Capability**: `intel_threats` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Malware Family Clustering via Behavioral Similarity

Current malware attribution is a flat string field. A proper implementation
clusters samples by shared behavioral fingerprints (imports, strings, PE
sections, TLSH distance) using DBSCAN or hierarchical clustering. Clusters
map to family lineages with version trees, enabling automatic variant
attribution without manual triage per-sample.

**Method**: `async def cluster_malware_family(self, sample_hashes: list[str]) -> dict[str, Any]`

---

## 2. Vulnerability Exploitation Probability (EPSS-Augmented Triage)

CVE indicators today store raw CVSS scores. A EPSS-augmented triage pipeline
scores each CVE by probability of exploitation within 30 days, combining CVSS
base metrics, CISA KEV membership, public PoC availability, and observed
in-the-wild usage. This directly drives patch prioritisation cadence.

**Method**: `async def vulnerability_triage(self, cve_ids: list[str]) -> list[dict[str, Any]]`

---

## 3. Diamond Model Structured Attribution

ATT&CK techniques alone do not constitute an attribution framework. Implementing
the Diamond Model (adversary, infrastructure, capability, victim) as a first-class
relationship structure allows analysts to reason about attribution confidence
across four axes rather than a single technique list. Each diamond instance stores
provenance, confidence, and analyst ID.

**Method**: `async def create_diamond_instance(self, adversary_id: str, infrastructure: list[str], capability: list[str], victim: str) -> dict[str, Any]`

---

## 4. Shared Infrastructure Graph Detection

Nation-state and criminal groups reuse hosting infrastructure (ASNs, certificate
authorities, registrars, nameservers) across campaigns. A graph traversal that
links campaigns and actors through shared IP ranges, ASN ownership, SSL cert
fingerprints, and registrar/registrant data surfaces clusters that simple IOC
overlap analysis misses.

**Method**: `async def shared_infrastructure_graph(self, campaign_ids: list[str]) -> dict[str, Any]`

---

## 5. Threat Hunting Hypothesis Generation

Analysts currently get mitigation hints at the tactical level. A hypothesis
engine should generate concrete, testable hunting queries (KQL, SPL, Sigma)
from observed technique patterns, target sectors, and actor TTPs. Queries
include expected field names, value patterns, and time windows, ready to drop
into a SIEM.

**Method**: `async def generate_hunting_hypotheses(self, actor_id: str, target_platform: str) -> list[dict[str, Any]]`

---

## 6. Automated PIR Satisfaction Scoring

Priority Intelligence Requirements (PIRs) are stored but never automatically
linked to the intelligence they generate. A satisfaction engine compares each
open PIR's requirement text (via keyword and semantic overlap) against new
indicators, reports, and campaign data added since the PIR was registered.
Each PIR gets a `satisfaction_score` (0–1) and a list of contributing artifacts.

**Method**: `async def score_pir_satisfaction(self, requirement_id: str) -> dict[str, Any]`

---

## 7. Temporal Decay Model for Indicator Confidence

Static confidence scores do not account for how quickly different IOC types
age. IPs rotate weekly; domains rotate monthly; file hashes are long-lived.
A temporal decay model applies a type-specific half-life function to confidence
scores, so a 90-day-old IP IOC with initial confidence 0.9 decays to ~0.3,
preventing stale IOCs from polluting high-priority alert queues.

**Method**: `async def apply_confidence_decay(self, indicator_id: str) -> dict[str, Any]`

---

## 8. Multi-Tenant Feed Access Control with TLP Enforcement

The feed system currently ignores TLP restrictions when sharing or disseminating
data. A proper access-control layer enforces TLP boundaries at the feed level:
TLP:RED indicators are never exported to feeds below amber; TLP:AMBER indicators
require explicit sharing agreements per tenant. Violations are logged as audit
events with a `tlp_violation` severity tag.

**Method**: `async def validate_feed_tlp_compliance(self, feed_id: str, target_tenant_id: str) -> dict[str, Any]`

---

## 9. Reverse Attribution Path Reconstruction

Given an unknown indicator, reconstruct the most likely threat actor by walking
backwards through indicator-campaign-actor link graphs. Scores each candidate
actor by the sum of weighted link confidences, TTP overlap with known actor
profiles, and infrastructure similarity. Outputs ranked candidate list with
confidence intervals and supporting evidence chains.

**Method**: `async def reverse_attribution(self, indicator_id: str) -> list[dict[str, Any]]`

---

## 10. STIX 2.1 Relationship Graph Export

The current STIX export produces flat bundles of indicators without
relationships. A full STIX 2.1 relationship graph export includes `relationship`
objects linking actors, campaigns, indicators, malware, tools, attack-patterns,
and course-of-action objects, enabling downstream platforms (OpenCTI, MISP,
Cortex XSOAR) to ingest the complete intelligence picture.

**Method**: `async def export_stix_relationship_graph(self, scope: dict[str, Any]) -> dict[str, Any]`

---

## 11. Behavioral Sandbox Report Parsing

Sandbox reports (Cuckoo, Any.run, Triage) contain rich behavioral data that
is currently entered manually. An automated parser extracts dropped files,
network connections, registry mutations, process trees, and MITRE technique
annotations from sandbox JSON reports, converting them directly into indicators
and campaign-technique links.

**Method**: `async def ingest_sandbox_report(self, sandbox_report: dict[str, Any], campaign_id: str | None = None) -> dict[str, Any]`

---

## 12. Analyst Workload Balancing and Queue Management

Indicator triage, PIR responses, and assessment reviews pile up without queue
management. A workload balancer assigns incoming triage tasks to available
analysts based on expertise tags, current queue depth, and SLA deadlines,
emitting Bytewax stream events for task assignment. Includes queue depth
metrics per analyst for capacity planning.

**Method**: `async def assign_triage_task(self, indicator_id: str, available_analysts: list[str]) -> dict[str, Any]`

---

## 13. Cross-Capability Threat Context Injection

`intel_threats` operates in isolation despite APG's composability model. A
context injection method pulls enrichment from sibling capabilities (`intel_alerts`,
`intel_correlation`, `intel_prediction`) via the APG capability bus, merging
alert history, anomaly scores, and predicted activity windows into the threat
context before assessment. Enables holistic risk scoring without tight coupling.

**Method**: `async def enrich_from_capability_bus(self, indicator_id: str, capabilities: list[str]) -> dict[str, Any]`

---

## 14. Adversary Simulation Playbook Generation

Rather than just mapping observed techniques, generate an adversary simulation
playbook (compatible with CALDERA, Atomic Red Team) for a given actor profile.
The playbook lists ordered attack steps with tool/technique bindings, expected
artifacts, and detection opportunities, enabling purple-team exercises directly
from production intelligence.

**Method**: `async def generate_simulation_playbook(self, actor_id: str, target_platform: str) -> dict[str, Any]`

---

## 15. Longitudinal Threat Trend Analysis

Point-in-time dashboards miss the temporal signal. A longitudinal analysis
engine buckets indicator creation, campaign activity, and actor sighting rates
into weekly/monthly bins, applies simple moving-average smoothing, and flags
acceleration or deceleration trends per sector, region, and actor motivation.
Outputs are chart-ready time-series dicts and narrative trend summaries.

**Method**: `async def longitudinal_trend_analysis(self, period_days: int, bucket: str) -> dict[str, Any]`
