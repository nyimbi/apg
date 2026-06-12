# APG AICR - AI Core Framework

AICR is the APG AI control plane. It lets generated applications register AI
services, providers, models, workflows, and agent runtimes while enforcing
tenant, policy, approval, audit, and observability guardrails.

## What It Provides

- Tenant-scoped AI service registry with owner, endpoint, health, and model
  policy metadata.
- Provider registry for local and external providers, including Codex, Claude
  Code, OpenCode, Pi, Ollama, OpenAI, Anthropic, local, and HTTP providers.
- Model catalog with provider linkage, policy metadata, modality, evaluation,
  promotion, and retirement guardrails.
- Model metric and drift-review governance with registered-model,
  metric-name, recorder, threshold, and pending-review evidence.
- Governed inference requests with model policy checks, health checks,
  large-context review, high-risk approval, PII redaction, tool allowlists, and
  cross-tenant routing denial.
- Durable review evidence for drifted metrics, high-risk inference approvals,
  privileged AI-agent registrations, and lifecycle batch decisions.
- Workflow and agent-runtime registration for first-class AI agent composition.
- First-class AI-agent records for Codex, Claude Code, OpenCode, and Pi
  contributors, including role, scope, owner, purpose, contribution disclosure,
  and privileged approval status.
- Bytewax lifecycle batch validation for model, prompt, inference, evaluation,
  metric, safety, routing, and AI-agent mutation streams.
- Governance events for service, provider, model, workflow, agent, approval,
  and inference lifecycle actions.
- UI models for dashboards, service registry, provider registry, model catalog,
  model metric console, inference console, workflow designer, agent runtimes,
  governance, metrics, audit, and settings.

## Main Files

- `SPECIFICATION.md` - full functional scope for the current packet.
- `PLAN.md` - packet implementation and review plan.
- `capability_contract.py` - executable configuration, rules, UI, adapter, and
  theme contract.
- `service.py` - includes `AicrService`, the dependency-light generated-app
  runtime facade.
- `api_helpers.py` - API-shaped helper calls over `AicrService`.
- `views.py` - generated-app view models.
- `app.py` - dynamic semantic package evidence and self-test.

## Generated-App Usage

```python
from capabilities.common.aicr.service import AicrService

service = AicrService()
service.register_provider(
	"codex-provider",
	"tenant-a",
	"Codex",
	"codex",
	"ai-platform",
	external=True,
	credential_vault_ref="keym://codex",
	egress_policy_ref="policy://ai-egress",
)
service.register_ai_service(
	"llm-router",
	"tenant-a",
	"LLM Router",
	"ai-platform",
	model_policy={"policy_id": "safe-generation"},
)
service.register_model(
	"reasoning-model",
	"tenant-a",
	"Reasoning Model",
	"codex-provider",
	"ai-platform",
	"text",
	model_policy={"policy_id": "safe-generation"},
)
service.record_model_metric(
	"tenant-a",
	"reasoning-model",
	"accuracy",
	0.97,
	"eval-owner",
)
pending_metric = service.record_model_metric(
	"tenant-a",
	"reasoning-model",
	"population_stability_index",
	0.34,
	"eval-owner",
	drift_score=0.34,
)
assert pending_metric["status"] == "pending_review"
approval = service.request_inference(
	"request-1",
	"tenant-a",
	"llm-router",
	"workflow-owner",
	"Summarize a customer support case.",
	context_tokens=256000,
	workflow_risk="high",
)
service.decide_inference_approval(
	approval["id"],
	"tenant-a",
	"risk-reviewer",
	"approved",
	"Large context accepted for this workflow.",
)
service.run_approved_inference(approval["id"], "tenant-a")
agent = service.register_ai_agent(
	"codex-reviewer",
	"tenant-a",
	"Codex Reviewer",
	"codex",
	"model_steward",
	"model catalog triage",
	"ai-platform",
	"Keep model metadata consistent.",
)
batch = service.validate_aicr_lifecycle_batch(
	"tenant-a",
	"bytewax",
	3,
	"ai_agent_batch",
)
```

## Guardrails

AICR blocks missing tenant context, missing service owners, missing endpoints,
unsupported provider types, missing provider credentials or egress policy,
missing model owners, unregistered model providers, missing model policy,
unsupported modalities, model promotion without evaluation, model metrics
without registered model/name/recorder evidence, drift above threshold without
review evidence, model retirement without impact review, inference without
model policy, routing to unhealthy services, PII inference without redaction,
tool calls without allowlists, cross-tenant routing, workflows without steps or
registered services,
unsupported agent runtimes, first-class AI agents without supported runtime,
supported role, scope, owner, purpose, or contribution disclosure, non-Bytewax
lifecycle batches, and external agent actions without approval.

Hard deny decisions raise `PermissionError`. Review-required decisions persist
or return records with policy evidence fields: `policy_decision` or `decision`,
`matched_rules`, `review_reasons`, and `audit_evidence`. Generated
applications can use `list_pending_reviews()` and the dashboard, inference,
metric, agent, lifecycle, and governance view models to compose review queues.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/aicr/__init__.py capabilities/common/aicr/capability_contract.py capabilities/common/aicr/service.py capabilities/common/aicr/api_helpers.py capabilities/common/aicr/views.py capabilities/common/aicr/app.py capabilities/common/aicr/test_capability_contract.py capabilities/common/aicr/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/aicr/test_capability_contract.py capabilities/common/aicr/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/aicr --json
./.venv/bin/apg capabilities publish-plan capabilities/common/aicr --json
```

---

## World-Class Enhancements (v2.0)

- **I1.** AICR - World Class Improvement Roadmap
- **I2.** Async-Native Governance Methods
- **I3.** Decimal-Accurate Cost Ledger
- **I4.** Streaming Inference via AsyncGenerator
- **I5.** Policy Rule Hot-Reload Without Restart
- **I6.** Multi-Tenant Cache with TTL Eviction
- **I7.** Structured Observability via OpenTelemetry Spans
- **I8.** Model Card Generation (Factsheet)
- **I9.** Shadow Mode / Canary Inference Routing
- **I10.** Prompt Injection Detection
- **I11.** Async Background Model Health Probing
- **I12.** Governance Decision Explainability
- **I13.** Model Retirement with Impact Analysis
- **I14.** Rate Limiting Per Tenant Per Model
- **I15.** Semantic Versioning Enforcement for Models

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
