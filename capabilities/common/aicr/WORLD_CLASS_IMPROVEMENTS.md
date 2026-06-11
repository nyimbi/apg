# AICR - World Class Improvement Roadmap

15 high-impact improvements to elevate the AI Core Framework capability to production-grade, enterprise-ready infrastructure.

---

## 1. Async-Native Governance Methods

**Category:** Architecture / Performance

**Justification:**
All `AicrService` public methods are currently synchronous. The service is embedded in an async APG runtime and every sync method blocks the event loop when called from an async context. FastAPI/Flask-AppBuilder + asyncio are the target deployment environments. Blocking governance calls under load will produce measurable p99 latency spikes and limit throughput.

**Implementation:**
Convert every public method on `AicrService` to `async def`. Replace dict-level in-memory stores with `asyncio.Lock`-protected access. Add `await` to all internal helper calls. Adopt `async with self._lock:` guards on write paths. The `_record_event` helper becomes `async def _record_event(...)`. All callers in `api_helpers.py` and `app.py` receive `await`.

**Competitor Reference:**
Ray Serve and BentoML service facades are fully async-native. Seldon V2 gRPC inference protocol mandates async handlers for high-throughput routing.

---

## 2. Decimal-Accurate Cost Ledger

**Category:** Financial Correctness

**Justification:**
`cost_tracking()` computes `total_cost_usd` using floating-point arithmetic (`sum(m.value * 0.0001 for m in metrics)`). IEEE 754 accumulation errors compound at scale. A 10 million-call month produces visible rounding drift that fails SOC 2 / financial audit requirements. The project CLAUDE.md explicitly mandates `Decimal` for money.

**Implementation:**
Import `decimal.Decimal` and `decimal.ROUND_HALF_UP`. Replace the cost formula:
```python
from decimal import Decimal, ROUND_HALF_UP
COST_PER_CALL = Decimal("0.0001")
total_cost = sum(Decimal(str(m.value)) * COST_PER_CALL for m in metrics)
total_cost_usd = str(total_cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
```
Return `total_cost_usd` as a string and document that callers must parse with `Decimal`. Add `average_cost_per_call_usd` with the same treatment.

**Competitor Reference:**
Stripe Billing and AWS Cost Explorer both represent monetary values as integer micro-units or `Decimal` strings. Google Cloud Billing API uses `google.type.Money` (units + nanos) to avoid float.

---

## 3. Streaming Inference via AsyncGenerator

**Category:** Capability / UX

**Justification:**
LLM inference for long outputs (legal summarisation, code generation, report drafting) benefits enormously from token streaming. A blocking call forces the caller to wait for the full completion. Streaming cuts perceived latency by 80-90% for multi-second completions and is now a baseline expectation for any LLM-serving layer (OpenAI, Anthropic, Ollama all stream by default).

**Implementation:**
Add `async def stream_inference(self, tenant_id, service_id, prompt_summary, ...)` that yields `AsyncGenerator[dict, None]`. Internally call the Ollama `/api/generate` endpoint with `stream=True`, parse NDJSON chunks, and yield `{"token": ..., "done": False}` dicts. Terminate with `{"done": True, "total_tokens": n}`. Guard with the same policy checks as `request_inference`.

**Competitor Reference:**
Ollama `/api/generate` streaming, OpenAI `stream=True` Chat Completions, Anthropic `stream=True` Messages API. LangChain `StreamingCallbackHandler`.

---

## 4. Policy Rule Hot-Reload Without Restart

**Category:** Operability / Security

**Justification:**
`capability_contract.py` bakes governance rules at import time. Updating a drift threshold or adding a new blocked modality requires redeployment. In production AI governance, risk reviewers need to tighten policy in under a minute (e.g., emergency block of a model mid-incident). A hot-reload mechanism eliminates deployment cycles for policy changes.

**Implementation:**
Watch `capability_contract.py` (or an external `policy.json` override) using `watchfiles.awatch()` in a background task. On change, call `importlib.reload(capability_contract)` and refresh the cached contract dict inside `AicrService`. Expose `async def reload_policy(self, tenant_id)` for manual trigger via the admin API. Emit a `policy_reloaded` governance event with a diff of changed rule IDs.

**Competitor Reference:**
OPA (Open Policy Agent) supports policy bundle hot-reload via `/v1/policies` PUT. AWS Config Rules trigger re-evaluation on rule update. Falco rules can be hot-reloaded via UNIX signal.

---

## 5. Multi-Tenant Cache with TTL Eviction

**Category:** Performance / Correctness

**Justification:**
`AicrService` uses raw Python dicts with no eviction. A long-running multi-tenant deployment leaks memory indefinitely as tenants register models, agents, and events. The existing `BoundedCache` from `capabilities.common.reliability` is already imported but never used in `AicrService`. Applying it with per-tenant TTL bounds memory and prevents stale records from influencing governance decisions.

**Implementation:**
Replace `self._models`, `self._services`, `self._model_metrics`, and `self._events` with `BoundedCache(maxsize=10_000, ttl_seconds=3600)` instances. Add `async def purge_tenant_cache(self, tenant_id)` for explicit teardown. Pass `tenant_id` as a cache key prefix to keep isolation guarantees. Emit a `cache_evicted` governance event with eviction count.

**Competitor Reference:**
Redis with TTL keys (the de facto standard). Hugging Face Hub uses LRU-with-TTL for model card caches. MLflow Model Registry uses Caffeine (Java) / `cachetools.TTLCache` (Python) for metadata caching.

---

## 6. Structured Observability via OpenTelemetry Spans

**Category:** Observability / SRE

**Justification:**
All observability is currently implemented as `logging.info(...)` string messages. These are invisible to Jaeger, Grafana Tempo, or any distributed tracing system. When a governed inference request traverses service registry lookup → policy evaluation → approval → completion, operators have no way to trace latency across steps or correlate failures with specific rule IDs.

**Implementation:**
Add `opentelemetry-api` as an optional dependency. Wrap each public method with a span: `with tracer.start_as_current_span(f"aicr.{method_name}") as span:`. Set span attributes from the governance result: `span.set_attribute("aicr.decision", result["decision"])`, `span.set_attribute("aicr.matched_rules", str(result["matched_rules"]))`. Degrade gracefully when OTel is not configured (no-op tracer).

**Competitor Reference:**
BentoML instruments every inference call with OTel spans. Seldon Core 2 exports gRPC spans to Jaeger. MLflow 2.x adds trace context to experiment runs.

---

## 7. Model Card Generation (Factsheet)

**Category:** Governance / Compliance

**Justification:**
EU AI Act Article 13 (transparency obligations) and the NIST AI RMF require a machine-readable model factsheet. Currently `AicrService` stores model metadata but provides no standardised output. A `model_card()` method bridges the gap between the internal registry and external compliance tooling without requiring a separate documentation system.

**Implementation:**
Add `def model_card(self, tenant_id, model_id) -> dict[str, Any]` that assembles a structured card:
- `model_metadata`: name, modality, provider, owner, risk_profile
- `evaluation`: score, evaluator, date
- `drift_history`: last 10 drift scores with timestamps
- `governance`: pending reviews, lifecycle batch count, compliance_score from `compliance_report()`
- `intended_use`: from model_policy
- `limitations`: derived from risk_profile

Emit a `model_card_generated` event. Return format compatible with Hugging Face Model Card JSON schema.

**Competitor Reference:**
Google Model Cards (`modelcards` Python library). Hugging Face Model Card spec. IBM FactSheets (AI Fairness 360). NIST AI RMF model documentation template.

---

## 8. Shadow Mode / Canary Inference Routing

**Category:** MLOps / Risk Management

**Justification:**
The existing `ab_test_models()` registers a test but performs no actual traffic routing. Production ML teams need shadow mode: route 100% of traffic to the primary model while silently sending a copy to the challenger and comparing outputs, without exposing challenger results to users. This is the safest way to validate a new model before any real traffic split.

**Implementation:**
Add `async def shadow_inference(self, shadow_id, tenant_id, primary_model_id, shadow_model_id, prompt_summary, actor)`. Execute primary and shadow inferences concurrently via `asyncio.gather`. Return only the primary result to the caller. Store the shadow result in `self._inference_results` keyed by `shadow_id`. Add `shadow_divergence_score` computed as output similarity using `similarity_search`. Emit `shadow_inference_recorded` events with divergence.

**Competitor Reference:**
Seldon Core Shadow Deployments. AWS SageMaker Shadow Testing. Weights & Biases Model Registry canary rollouts.

---

## 9. Prompt Injection Detection

**Category:** Security

**Justification:**
The existing `_validate_inference_input()` blocks a small list of SQL/code injection patterns but has no defence against prompt injection — adversarial inputs designed to override model instructions. Prompt injection is the #1 LLM security threat in OWASP Top 10 for LLM Applications (2025). A governance layer that routes LLM inference without prompt injection guards creates audit liability.

**Implementation:**
Add `def detect_prompt_injection(self, inputs: list[str], policy: str = "strict") -> dict` that scans for:
- Role override patterns: `"ignore previous instructions"`, `"you are now"`, `"disregard your"`, `"act as"` + `"without restrictions"`
- Data exfiltration probes: `"repeat everything above"`, `"print your system prompt"`
- Delimiter injection: triple backticks, `###`, `[INST]` appearing mid-string
- Unicode homoglyphs that bypass ASCII pattern matching

Return `{"safe": bool, "flags": list[str], "risk_score": float}`. Integrate into `request_inference()` pre-check. Block `"high"` risk_score under strict policy.

**Competitor Reference:**
LangChain `PromptInjectionDetector`. Rebuff (prompt injection detection library). OWASP LLM Top 10 LLM01 mitigations.

---

## 10. Async Background Model Health Probing

**Category:** Reliability / Observability

**Justification:**
`health_check()` computes health from the in-memory `service.health` field, which is whatever was set at registration. There is no mechanism to actively probe model endpoints. A model endpoint can go down after registration; `health_check()` will still report "healthy". This creates false confidence in the governance dashboard.

**Implementation:**
Add `async def probe_service_health(self, tenant_id, service_id) -> dict` that:
1. Fetches the service endpoint from the registry
2. Issues an async HTTP HEAD/GET to the health path (e.g., `/health`) with a 2-second timeout using `aiohttp.ClientSession`
3. Updates `service.health` to `"unhealthy"` on connection error or non-2xx response
4. Emits `service_health_probed` event with latency_ms

Add `async def probe_all_services(self, tenant_id)` to probe all services concurrently via `asyncio.gather`. Expose as a background periodic task triggered every 60 seconds.

**Competitor Reference:**
Kubernetes liveness/readiness probes. Seldon Core model health endpoints. BentoML health probe runner.

---

## 11. Governance Decision Explainability

**Category:** Compliance / Auditability

**Justification:**
`evaluate_capability_rules()` returns `decision`, `matched_rules`, and `actions` but provides no human-readable explanation of *why* each rule fired. Compliance auditors reviewing a `PermissionError` trace need natural-language rationale, not opaque rule IDs. EU AI Act Article 14 (human oversight) requires that operators can explain automated decisions.

**Implementation:**
Add `def explain_decision(self, tenant_id, context: dict, rule_ids: list[str]) -> dict` that:
1. Re-runs `evaluate_capability_rules(context)`
2. For each matched rule, looks up a `"description"` and `"remediation"` field from the contract
3. Returns `{"decision": ..., "explanation": [{"rule": ..., "description": ..., "remediation": ..., "context_values": {...}}]}`

Enrich the contract with `description` and `remediation` strings for every rule. The `_raise_if_blocked()` function uses this to emit a richer `PermissionError` message.

**Competitor Reference:**
OPA `explain` query parameter. Azure Policy compliance details API. AWS Config remediation actions.

---

## 12. Model Retirement with Impact Analysis

**Category:** Governance / Change Management

**Justification:**
There is no `retire_model()` method. Operators currently hard-delete or change status manually with no impact analysis. Retiring a model that is referenced by active workflows or pending approvals can break downstream services silently. The existing guardrail comment in the README mentions "model retirement without impact review" as a blocked operation but no implementation exists.

**Implementation:**
Add `def retire_model(self, tenant_id, model_id, reason, retired_by, impact_review_ref) -> dict` that:
1. Validates `impact_review_ref` is non-empty (blocks without it)
2. Finds all workflows referencing the model via `service_ids` intersection
3. Finds all pending inference approvals referencing the model
4. Sets model status to `"retired"`
5. Emits `model_retired` event with `{"affected_workflows": [...], "affected_approvals": [...], "impact_review_ref": ...}`
6. Returns a retirement manifest with affected resource counts

**Competitor Reference:**
MLflow Model Registry `archive`/`delete` with stage transition rules. Vertex AI Model Registry deprecation workflow. Hugging Face Hub model deprecation notices.

---

## 13. Rate Limiting Per Tenant Per Model

**Category:** Security / Multi-Tenancy

**Justification:**
A misbehaving or compromised tenant can submit unlimited inference requests, overwhelming shared infrastructure and starving other tenants. There is no rate limiting at the governance layer. Token-bucket or sliding-window rate limiting at the `AicrService` level provides a last line of defence independent of API gateway configuration.

**Implementation:**
Add a `_rate_limits: dict[tuple[str, str], deque]` store (tenant_id, model_id → timestamp deque). In `request_inference()`, prepend a rate check:
```python
async def _check_rate_limit(self, tenant_id, model_id, limit=100, window_seconds=60):
    key = (tenant_id, model_id)
    now = time.monotonic()
    dq = self._rate_limits.setdefault(key, deque())
    while dq and dq[0] < now - window_seconds:
        dq.popleft()
    if len(dq) >= limit:
        raise PermissionError("inference_rate_limit_exceeded")
    dq.append(now)
```
Expose `def configure_rate_limit(self, tenant_id, model_id, limit, window_seconds)` for per-tenant tuning. Emit `rate_limit_exceeded` governance events.

**Competitor Reference:**
Kong Gateway rate limiting plugin. AWS API Gateway usage plans. Nginx limit_req_zone. Anthropic API rate limits by tier.

---

## 14. Semantic Versioning Enforcement for Models

**Category:** Data Integrity / MLOps

**Justification:**
`model_version()` accepts any string as `new_version`. This allows `"v_final_2"`, `"PROD-hotfix-mar"`, and other non-standard version strings that break downstream tooling, sorting, and changelog generation. SemVer enforcement makes model versioning machine-processable and enables automatic major/minor/patch changelog categorisation.

**Implementation:**
Add `_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+))?$")` to the module. In `model_version()`, validate both `new_version` and the stored version. Add `def next_version(self, tenant_id, model_id, bump: Literal["major","minor","patch"]) -> str` that auto-increments from the current version. Block downgrades (new_version < current_version) unless `force=True`.

**Competitor Reference:**
Docker Hub image tag SemVer convention. MLflow model version integers with alias promotion. Hugging Face model revision tags (Git SHAs + SemVer aliases).

---

## 15. Async Bulk Governance Operations with Partial-Failure Semantics

**Category:** Performance / Correctness

**Justification:**
`bulk_register_models()` is synchronous, iterates serially, and swallows exceptions into `{"error": str(exc)}` dicts without emitting governance events for failed items. At 1000-model bulk registration, the synchronous loop blocks the event loop for potentially hundreds of milliseconds. Partial-failure semantics (return successes + structured errors) are absent, making it impossible for callers to differentiate validation errors from transient failures.

**Implementation:**
Add `async def bulk_register_models_async(self, tenant_id, models, owner) -> dict` that:
1. Launches all registrations via `asyncio.gather(*tasks, return_exceptions=True)`
2. Separates `isinstance(result, Exception)` failures from successes
3. Returns `{"succeeded": [...], "failed": [{"id": ..., "error": ..., "error_type": ...}], "total": n, "success_count": m}`
4. Emits a single `bulk_model_registration_completed` governance event with counts
5. Validates that `len(models) <= 500` to prevent memory exhaustion

**Competitor Reference:**
Kubernetes batch API partial failure responses. Stripe bulk create with `idempotency_key`. AWS Batch job array status tracking.
