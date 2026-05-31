# APG FEDL - Federated Learning

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

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/fedl/__init__.py capabilities/common/fedl/capability_contract.py capabilities/common/fedl/models.py capabilities/common/fedl/federated_engine.py capabilities/common/fedl/service.py capabilities/common/fedl/api.py capabilities/common/fedl/views.py capabilities/common/fedl/app.py capabilities/common/fedl/test_capability_contract.py capabilities/common/fedl/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/fedl/test_capability_contract.py capabilities/common/fedl/tests/test_package_contract.py
./.venv/bin/python capabilities/common/fedl/app.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/fedl --json
./.venv/bin/apg capabilities publish-plan capabilities/common/fedl --json
```
