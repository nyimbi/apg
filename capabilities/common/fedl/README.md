# APG FEDL - Federated Learning

© 2025 Datacraft | Author: Nyimbi Odero

FEDL is the APG capability for privacy-preserving collaborative model training.
It lets generated applications create governed federations, attest participants,
run approved training rounds, collect participant updates, apply poisoning
defense, aggregate updates securely, publish federated models through MLCM, and
emit audit evidence without moving participant datasets into a central store.

## What It Provides

- Tenant-scoped federation coordination with coordinator, model family,
  objective metric, privacy budget, and data-residency regions.
- Participant registration with attestation, contracts, region controls, and
  compute-profile metadata.
- Approved training rounds with minimum participant checks, secure aggregation,
  privacy-budget allocation, and round state.
- Model-update intake with digesting, sample counts, quality scores, poisoning
  detection, quarantine, and audit evidence.
- Secure aggregation result records, derived federated model versions, and MLCM
  release linkage.
- Federation retirement after impact review.
- First-class federation-agent composition for Codex, Claude Code, OpenCode,
  Pi, and future AICR-adapted runtimes.
- Bytewax lifecycle-batch validation for generated-app and agent-authored FEDL
  mutations.
- Rule-engine metadata, UI route metadata, theme tokens, and generated-app
  semantic package evidence.
- Adapter configuration for AICR, MLCM, ENCR, MTEN, AUTH, AUDL, MONI, and
  Bytewax event streaming.
- Async-first extended API: 18+ `async def` methods covering DP application,
  secure aggregation, gradient compression, privacy accounting, client
  selection, convergence checking, model personalisation, security auditing,
  bulk registration, and federation export.
- Per-round communication status tracking with per-participant receipt state.
- Heterogeneous data schema registration per participant.
- Service health endpoint for operational monitoring.

## Main Files

- `SPECIFICATION.md` - complete functional scope for this packet.
- `PLAN.md` - implementation and review plan.
- `capability_contract.py` - executable configuration, rules, UI, adapters, and
  theme contract.
- `models.py` - dependency-light federated learning domain records.
- `federated_engine.py` - deterministic digest and model-version helper.
- `service.py` - `FedlService`, the generated-app runtime facade.
- `api.py` - API-shaped helper calls over the service.
- `views.py` - semantic UI view models.
- `app.py` - dynamic package evidence and self-test.

## Generated-App Usage

```python
from capabilities.common.fedl.service import FedlService

service = FedlService()
federation = service.create_federation(
	"fed-risk",
	"tenant-a",
	"Risk Model Federation",
	"ml-platform",
	"tabular-risk",
	"auc",
	privacy_epsilon_limit=6.0,
	data_residency_regions=["ke", "za", "ng"],
)
for index, region in enumerate(["ke", "za", "ng"], start=1):
	service.register_participant(
		f"node-{index}",
		"tenant-a",
		federation["id"],
		f"Node {index}",
		region,
		f"contract-{index}",
		attested=True,
	)
round_model = service.start_round(
	"round-001",
	"tenant-a",
	federation["id"],
	1,
	2.0,
	"approval-001",
)
for index in range(1, 4):
	service.submit_update(
		f"update-{index}",
		"tenant-a",
		round_model["id"],
		f"node-{index}",
		{"weights": [index, index + 1]},
		100 * index,
		0.91,
	)
aggregation = service.aggregate_updates("agg-001", "tenant-a", round_model["id"], True)
service.release_model(
	"release-001",
	"tenant-a",
	f"model:{aggregation['model_version']}",
	"mlcm://fraud-risk",
	"approval:release-001",
	"privacy-review:release-001",
)
service.register_federation_agent(
	"fedl-reviewer",
	"tenant-a",
	"FEDL Privacy Reviewer",
	"codex",
	"privacy_reviewer",
	"fed-risk releases",
	"ml-platform",
	"Review privacy budgets and release guardrails",
	human_approval_required=True,
)
service.validate_fedl_lifecycle_batch(
	"tenant-a",
	"bytewax",
	3,
	"federation_agent_batch",
	"batch-001",
)
```

## Guardrails

FEDL blocks missing tenant context, federations without coordinators, missing
model-family or objective metadata, missing data-residency regions, invalid
privacy budgets, unattested participants, participants without contracts,
participants outside allowed regions, rounds without enough participants,
rounds without approval, rounds without secure aggregation, high privacy budget
without review, privacy budget over federation limit, updates for non-running
rounds, updates from non-round participants, missing sample counts, invalid
quality scores, incomplete update sets, poisoning signals, missing aggregate
digests, model release without MLCM linkage, model release without approval,
model release without privacy review, federation retirement without impact
review, cross-tenant participation, and non-Bytewax round event streams.
Agent guardrails also block unsupported runtimes, unsupported roles, missing
scope, missing owner, missing purpose, undisclosed machine contribution, and
non-Bytewax lifecycle batches. Privileged roles without explicit human
approval are retained as `pending_review` rather than silently activated.

## Agent Composition

FEDL agents are provider-neutral governance actors. The contract currently
recognizes `codex`, `claude_code`, `opencode`, and `pi` runtime codes, but the
runtime execution remains behind AICR adapter contracts. Generated applications
compose agents through `register_federation_agent()` and inspect them through
`list_federation_agents()` or the `/fedl/agents` UI route metadata.

Supported roles include federation, participant, privacy, security, round,
aggregation, model-release, residency, and steward responsibilities. Privacy,
security, round, aggregation, model-release, and residency roles are privileged
and require human approval evidence for active status.

## Bytewax Lifecycle Batches

FEDL does not use broker-specific queue for lifecycle mutation governance. The streaming
manifest requires Bytewax with the `fedl.lifecycle` stream and declares
operation names for federation, participant, training-round, update,
aggregation, privacy-budget, release, and federation-agent batches.
Generated applications validate those batches with
`validate_fedl_lifecycle_batch()` and inspect accepted or denied evidence
through `list_lifecycle_batches()` or the `/fedl/lifecycle` route metadata.

## World-Class Enhancements (v2.0)

Fifteen targeted improvements close the gap between production FL systems and
the current in-memory implementation. They are ordered by impact.

1. **Async-First Core** — Migrate full service surface to `async def`; use
   `asyncio.gather` for bulk ops. Thin sync shims kept only where FAB forces
   synchronous dispatch.

2. **Persistent Storage via Repository Pattern** — Abstract `FedlRepository`
   protocol with pluggable backends: in-memory (tests), PostgreSQL via asyncpg/
   SQLAlchemy 2 async, Redis (hot-path cache). Wired via dependency injection.

3. **Real Differential Privacy Engine (Opacus / DP-SGD)** — Replace metadata-
   only recording with actual L2-norm gradient clipping and Gaussian/Laplace
   noise injection via Opacus or TF Privacy. Surface exact (ε, δ) per round.

4. **Cryptographic Secure Aggregation (SecAgg+)** — Implement Google SecAgg+:
   pairwise mask negotiation, Shamir secret sharing, XOR-masked upload, dropout-
   resilient reconstruction. Fall back to Paillier HE for small cohorts.

5. **Gradient Compression with Error Feedback** — Top-K sparsification,
   random-K, and PowerSGD low-rank approximation. Per-participant error-feedback
   buffers prevent compression bias accumulating across rounds.

6. **Byzantine-Robust Aggregation Rules** — Pluggable aggregation: FedAvg,
   Krum, Multi-Krum, Trimmed-Mean, Median, FLTrust. Selected at federation
   creation via `aggregation_strategy`; rule-selection evidence emitted to audit.

7. **Formal Privacy Accounting (Rényi DP / Moments Accountant)** — Replace
   ε-sum with RDP composition via Google `dp-accounting`. Surface per-round
   and cumulative (ε, δ)-DP certificates using the moments accountant.

8. **Federated Model Lineage and Provenance Graph** — Provenance DAG per model
   recording contributing participants, sample counts, DP parameters, aggregation
   strategy, and cryptographic digest. Exportable as W3C PROV-N or JSON-LD.

9. **Cross-Silo Communication Layer (gRPC / NATS)** — gRPC-based or NATS
   JetStream transport with mutual TLS, per-participant topic isolation, MAC
   update integrity, and retry/back-pressure. Service becomes true coordinator.

10. **Adaptive Client Selection with Fairness Constraints** — Stratified
    sampling balancing regional quotas, compute capacity, participation history
    (anti-starvation), and data heterogeneity proxy. Fairness metrics in analytics.

11. **Split Learning and Hybrid FL** — `split_learning_round` and
    `hybrid_fl_round` methods. Hybrid FL combines split learning for resource-
    constrained nodes with full FL for capable nodes. `learning_mode` on rounds.

12. **Real-Time Convergence Monitoring with Early Stopping** — Track per-round
    validation metrics from `model_evaluate`. EMA convergence curve with
    configurable threshold; emit early-stopping signal and `convergence_timeline`.

13. **Federated Model Distillation (FedDF / Ensemble)** — `model_distil` runs
    federated distillation using a shared unlabelled dataset; ensemble participant
    predictions into a compact student model with distillation provenance.

14. **Compliance Export (GDPR Art. 22 / Kenya DPA Evidence Pack)** —
    `compliance_export` generates a structured evidence pack: privacy notices,
    consent refs, DP certificates, data-residency proofs, release approvals, and
    audit chains. Output as JSON-LD or signed PDF via compliance adapter.

15. **Federated Hyperparameter Optimisation (FedHPO)** — `hpo_round` has the
    coordinator propose hyperparameter candidates via Bayesian optimisation;
    participants evaluate locally and return metrics only; coordinator selects
    next candidate. Integrates with Optuna or Ray Tune.

## New Methods

All extended methods are `async def`. Call them from an async context or via
`asyncio.run(...)`.

### `privacy_budget_track` — per-round DP accounting

```python
import asyncio

budget = asyncio.run(service.privacy_budget_track("tenant-a", "fed-risk"))
# {
#   "federation_id": "fed-risk",
#   "epsilon_limit": 6.0,
#   "epsilon_spent": 2.0,
#   "epsilon_remaining": 4.0,
#   "utilisation_pct": 33.33,
#   "per_round": [{"round_id": "round-001", "epsilon": 2.0, "status": "aggregated"}]
# }
```

### `differential_privacy_apply` — Gaussian mechanism metadata

```python
dp = asyncio.run(service.differential_privacy_apply(
	tenant_id="tenant-a",
	round_id="round-001",
	noise_multiplier=1.1,
	clipping_norm=1.0,
))
# {"noise_multiplier": 1.1, "clipping_norm": 1.0, "updates_noised": 3,
#  "effective_epsilon": 1.818182, "status": "applied"}
```

### `bulk_register_participants` — register a cohort atomically

```python
nodes = [
	{"id": "node-4", "name": "Node 4", "region": "ke", "contract_ref": "c4", "attested": True},
	{"id": "node-5", "name": "Node 5", "region": "za", "contract_ref": "c5", "attested": True},
]
records = asyncio.run(service.bulk_register_participants("tenant-a", "fed-risk", nodes))
```

### `fl_security_audit` — policy violation scan

```python
report = asyncio.run(service.fl_security_audit("audit-001", "tenant-a", "fed-risk"))
# {
#   "risk_level": "low",
#   "findings": [],
#   "finding_count": 0,
#   "status": "completed"
# }
```

### `communication_round` — per-participant receipt tracking

```python
status = asyncio.run(service.communication_round("tenant-a", "round-001"))
# {
#   "completion_pct": 100.0,
#   "received_count": 3,
#   "pending_count": 0,
#   "receipt_status": {"node-1": "received", "node-2": "received", "node-3": "received"}
# }
```

## Service API Reference

| Method | Sync/Async | Description |
|---|---|---|
| `create_federation` | sync | Create a governed federation |
| `register_participant` | sync | Attest and register a participant |
| `start_round` | sync | Open an approved training round |
| `submit_update` | sync | Accept a model update from a participant |
| `aggregate_updates` | sync | Aggregate accepted updates, produce model version |
| `release_model` | sync | Publish a federated model to MLCM |
| `retire_federation` | sync | Retire a federation after impact review |
| `register_federation_agent` | sync | Register a governance agent |
| `validate_fedl_lifecycle_batch` | sync | Validate Bytewax lifecycle batches |
| `list_federations` | sync | List federations for a tenant |
| `list_participants` | sync | List participants for a tenant |
| `list_rounds` | sync | List training rounds for a tenant |
| `list_updates` | sync | List model updates for a tenant |
| `list_aggregations` | sync | List aggregation results for a tenant |
| `list_models` | sync | List federated models for a tenant |
| `list_releases` | sync | List model releases for a tenant |
| `list_federation_agents` | sync | List registered agents |
| `list_lifecycle_batches` | sync | List Bytewax lifecycle batch records |
| `list_audit_events` | sync | List all audit events for a tenant |
| `dashboard_summary` | sync | Aggregate dashboard counters |
| `privacy_budget_summary` | sync | Simple per-tenant ε summary |
| `fl_round_start` | async | Async round start |
| `client_model_aggregate` | async | Async client-side aggregation trigger |
| `differential_privacy_apply` | async | Apply Gaussian DP to a round |
| `secure_aggregation` | async | Run SecAgg protocol step |
| `model_evaluate` | async | Evaluate model against a dataset |
| `gradient_compress` | async | Apply gradient compression to updates |
| `privacy_budget_track` | async | Detailed per-round privacy accounting |
| `client_select` | async | Select participant subset for next round |
| `model_version` | async | Full version history for a federation |
| `fl_analytics` | async | Federated learning analytics |
| `heterogeneous_data_handle` | async | Register per-participant schema transforms |
| `communication_round` | async | Per-participant update receipt status |
| `convergence_check` | async | Check convergence across completed rounds |
| `model_personalise` | async | Fine-tune global model per participant |
| `fl_security_audit` | async | Security policy violation scan |
| `bulk_register_participants` | async | Register multiple participants at once |
| `export_federation` | async | Full federation metadata snapshot export |
| `health_check` | async | Service health and store statistics |

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/fedl/__init__.py capabilities/common/fedl/capability_contract.py capabilities/common/fedl/models.py capabilities/common/fedl/federated_engine.py capabilities/common/fedl/service.py capabilities/common/fedl/api.py capabilities/common/fedl/views.py capabilities/common/fedl/app.py capabilities/common/fedl/test_capability_contract.py capabilities/common/fedl/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/fedl/test_capability_contract.py capabilities/common/fedl/tests/test_package_contract.py
./.venv/bin/python capabilities/common/fedl/app.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/fedl --json
./.venv/bin/apg capabilities publish-plan capabilities/common/fedl --json
```
