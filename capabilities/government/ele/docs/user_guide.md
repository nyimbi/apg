# Electoral and Civil Registration — User Guide

**Capability ID**: `government_ele` | **Domain**: `government` | **Version**: `2.0.0`
**Author**: Nyimbi Odero | **Copyright**: © 2025 Datacraft | **Website**: www.datacraft.co.ke

---

## Description

Full-lifecycle electoral management and civil registration platform.  Covers voter registration
with biometric deduplication, polling station management, election creation, multi-method ballot
tabulation (first-past-the-post and ranked-choice), real-time result streaming over NATS, tamper-
evident audit trails backed by Merkle trees, AI-assisted biometric deduplication via Ollama, and
automated Electoral Act compliance reporting.

---

## Installation

```bash
pip install apg-government-ele
```

Or in a `pyproject.toml` workspace:

```toml
[tool.uv.workspace.dependencies]
apg-government-ele = { path = "capabilities/government/ele" }
```

---

## Quick Start

```python
from capabilities.government.ele.service import ElectoralService

svc = ElectoralService(tenant_id="ke_iebc", actor_id="admin")

# Create an election
election = svc.create_election(
    election_id="GE2027",
    tenant_id="ke_iebc",
    election_type="presidential",
    name="Kenya General Election 2027",
    polling_date="2027-08-10",
    nomination_deadline="2027-06-01",
    constituency="national",
)

# Register a voter
reg = svc.voter_registration(
    citizen_id="32145678",
    constituency="nairobi_west",
    documents=["national_id", "utility_bill"],
)

# Capture biometrics
bio = svc.biometric_capture(
    voter_id=reg["id"],
    fingerprint="<base64-encoded-fp-data>",
    photo="<base64-encoded-photo>",
)

print(bio["status"])  # "captured" or "duplicate_detected"
```

---

## Async Methods (v2.0)

All async methods require an async runtime.  Use `asyncio.run()` or an existing event loop.

### Zero-Knowledge Voter Verification

```python
import asyncio

result = asyncio.run(svc.verify_zk_credential(
    voter_id="32145678",
    zk_proof="<snark-proof-hex>",
    merkle_root="<voter-roll-root-hash>",
))
# result["proof_valid"] is True/False; no PII in response
```

Proves a voter is on the roll without revealing their identity.  Production deployments replace
the stub with a `py-snark` circuit call.

---

### Real-Time Result Streaming (NATS)

```python
payload = asyncio.run(svc.stream_result_updates(
    constituency_id="nairobi_west",
    nats_subject_prefix="apg.government.ele.results",
))
# payload["signature"] is a SHA-256 HMAC of the tally
# payload["subject"] is the NATS subject to publish to
```

Inject a live `nats.aio.client.Client` in production and call `nc.publish(payload["subject"], ...)`.

---

### AI-Assisted Biometric Deduplication

```python
score = asyncio.run(svc.ai_deduplication_score(
    voter_id="32145678",
    fingerprint_embedding=[0.12, 0.34, ...],  # 128-dim vector from Ollama/LLaVA
    face_embedding=[0.56, 0.78, ...],
    threshold=0.95,
))
# score["recommended_action"]: "pass" | "review" | "reject"
```

Uses cosine similarity for the stub.  Production deployments route embedding extraction to a
locally hosted Ollama model (e.g. `llava`).

---

### Offline-First Station Operations

```python
# On a disconnected polling station tablet:
q = asyncio.run(svc.queue_offline_operation(
    operation="vote_counting",
    payload={"station_id": "PS-001", "results": {"candidate_a": 312, "candidate_b": 198}},
    lamport_clock=7,
))

# When connectivity is restored:
sync = asyncio.run(svc.sync_offline_queue(
    nats_subject="apg.government.ele.offline_sync",
))
print(f"Synced {sync['synced_count']} operations")
```

Operations are replayed in Lamport timestamp order for causal consistency.

---

### Statistical Anomaly Detection

```python
anomalies = asyncio.run(svc.detect_statistical_anomalies(
    election_id="GE2027",
    constituency_id="nairobi_west",
    benford_alpha=0.05,
    zscore_threshold=3.0,
))
if anomalies["quarantined"]:
    print(f"{anomalies['anomalies_detected']} anomalies — results quarantined")
```

Applies Benford's Law first-digit test and Z-score outlier detection.  Quarantines results
automatically on detection.

---

### Voter Status Notifications

```python
notif = asyncio.run(svc.notify_voter_status_change(
    voter_id="32145678",
    new_status="verified",
    channel="sms",
))
# notif["subject"] is the NATS subject; ntfy capability subscriber fans out to SMS
```

---

### Tamper-Evident Voter Roll (Merkle Tree)

```python
tree = asyncio.run(svc.build_voter_roll_merkle_tree(constituency_id="nairobi_west"))
print(tree["root_hash"])  # Publish to NATS or share with observers for independent verification
```

Any modification to the voter roll produces a different root hash, making tampering detectable
by any party that holds the original hash.

---

### Ranked-Choice Tabulation

```python
# Each ballot is an ordered list of candidate IDs (most to least preferred)
ballots = [
    ["alice", "bob", "carol"],
    ["bob", "alice", "carol"],
    ["carol", "alice", "bob"],
    ["alice", "carol", "bob"],
]

result = asyncio.run(svc.tabulate_ranked_choice(
    election_id="GE2027",
    ballots=ballots,
))
print(result["winner"])       # "alice"
print(result["rounds"])       # Round-by-round tallies
```

Supports instant-runoff elimination.  Falls back to plurality if no majority is reachable.

---

### Candidate Eligibility Validation

```python
verdict = asyncio.run(svc.validate_candidate_eligibility(
    candidate_id="CAND-001",
    election_id="GE2027",
    national_id="32145678",
    birth_date="1985-03-15",
    minimum_age=35,
))
if verdict["hard_fail"]:
    print("Candidate blocked:", verdict["checks"])
```

Cross-references age from `birth_date`, civil registry presence, and duplicate candidacy.
Blocking on hard fail prevents `candidate_register` from proceeding.

---

### Automated Compliance Audit

```python
report = asyncio.run(svc.run_compliance_audit(
    election_id="GE2027",
    legal_framework="electoral_act",
))
print(f"Compliance rate: {report['compliance_rate_pct']}%")
# report["nats_subject"] is published to apg.government.ele.compliance.GE2027
```

Maps every audit event to its Electoral Act provision.  Non-compliant events include a
`remediation` recommendation.

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/government-ele/dashboard` | `government_ele:view` | Overview |
| `/government-ele/registrations` | `government_ele:register` | Registration |
| `/government-ele/deduplication` | `government_ele:deduplicate` | Registration |
| `/government-ele/polling-stations` | `government_ele:stations` | Elections |
| `/government-ele/elections` | `government_ele:elections` | Elections |
| `/government-ele/results` | `government_ele:results` | Results |
| `/government-ele/civil-registry` | `government_ele:civil` | Civil Registry |
| `/government-ele/verifications` | `government_ele:verify` | Verification |
| `/government-ele/merkle` | `government_ele:audit` | Audit |
| `/government-ele/compliance` | `government_ele:audit` | Audit |
| `/government-ele/anomalies` | `government_ele:audit` | Audit |
| `/government-ele/ranked-choice` | `government_ele:results` | Results |

---

## NATS Event Subjects

| Subject | Description |
|---------|-------------|
| `apg.government.ele.results.{constituency_id}` | Live result updates (signed) |
| `apg.government.ele.notifications.{voter_id}` | Voter status change notifications |
| `apg.government.ele.merkle.roots` | Voter roll Merkle root hashes |
| `apg.government.ele.compliance.{election_id}` | Compliance audit reports |
| `apg.government.ele.offline_sync` | Offline operation sync payloads |
| `apg.government.ele.lifecycle` | General lifecycle events |

---

## Streaming Architecture

```
Polling Station → queue_offline_operation() → [offline WAL]
                                                     ↓ connectivity restored
                                          sync_offline_queue() → NATS JetStream
                                                                       ↓
                                                     stream_result_updates() → observers
                                                     detect_statistical_anomalies() → quarantine
                                                     run_compliance_audit() → compliance report
```

All streaming uses NATS + bytewax.  NATS JetStream provides at-least-once delivery and full
replay for audit.

---

## Configuration

All keys are tenant-scoped.  Set via the `conf` capability or environment variables prefixed
with `GOVERNMENT_ELE_`.

| Key | Default | Description |
|-----|---------|-------------|
| `deduplication.duplicate_detection_threshold` | `0.95` | Match score that triggers a duplicate flag |
| `deduplication.primary_method` | `biometric_fingerprint` | Primary dedup modality |
| `deduplication.ai_model` | `ollama/llava-biometric` | Ollama model for AI dedup |
| `governance.minimum_voting_age` | `18` | Minimum age for voter registration |
| `governance.minimum_candidate_age` | `35` | Minimum age for presidential candidates |
| `merkle.publish_on_build` | `true` | Publish root to NATS after tree construction |
| `compliance.legal_framework` | `electoral_act` | Rule set for compliance auditing |
| `offline.lamport_sync` | `true` | Preserve causal order in offline sync |
| `anomaly.benford_alpha` | `0.05` | Significance level for Benford's Law test |
| `anomaly.zscore_threshold` | `3.0` | Z-score threshold for outlier detection |

---

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `duplicate_voter_denied` | `duplicate_detected=True` | deny |
| `underage_voter_denied` | `of_voting_age=False` | deny |
| `voter_biometric_required` | `biometric_present=False` | deny |
| `result_manipulation_denied` | `manipulation_detected=True` | deny |
| `cross_constituency_result_denied` | `cross_constituency=True` | deny |
| `candidate_hard_fail_blocks_registration` | `eligible=False` | deny |
| `anomaly_quarantine` | `anomalies_detected>0` | quarantine result |

---

## Composability

```apg
use government_ele;
use government_csr;   # voter card applications via portal
use government_cas;   # electoral complaints become cases
use government_law;   # electoral offences → police dockets
use intel;            # voter pattern analytics
```

Candidate eligibility checks automatically compose with `civil_events` registry and can be
extended to call `government_law` for criminal disqualification lookups.

---

## Testing

```bash
uv run pytest -vxs capabilities/government/ele/tests/ci
```

Tests cover:
- Voter registration lifecycle (register → biometric → verify)
- Deduplication (hash-based + AI scoring stub)
- Polling station setup and offline queue sync
- Result collation, transmission, and certification
- Ranked-choice tabulation (including plurality fallback)
- Merkle tree construction and root hash determinism
- Statistical anomaly detection (Benford + Z-score)
- Compliance audit report generation
- ZK credential verification stub
- Candidate eligibility cross-referencing

---

## Further Reading

- `WORLD_CLASS_IMPROVEMENTS.md` — 15 improvement plans with competitor benchmarking
- `service.py` — Complete service implementation
- `models.py` — Pydantic/dataclass data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and schemas
- `README.md` — Quick reference
- `cap_spec.md` — Capability specification
- `SPECIFICATION.md` — Detailed functional specification
