# BIOP - Biometric Processing

BIOP provides governed biometric processing for APG applications. It covers consent, enrollment, encrypted template metadata, liveness-backed verification, match confidence review, cross-border privacy review, first-class biometric governance agents, Bytewax lifecycle batch validation, revocation, retirement, audit evidence, and route-ready UI models.

The generated-application surface is dependency-light. `biometric_runtime.py`, `api_helpers.py`, and `view_models.py` can be imported without biometric hardware, Flask, Flask-AppBuilder, database sessions, model servers, computer-vision engines, or external AI-agent clients. Production deployments can bind sensors, matchers, HSM-backed vaults, CVSN, MFAU, AICR, AUDL, ENCR, and Bytewax adapters behind the same contract.

## What BIOP Provides

- Explicit biometric consent records with purpose, modality scope, jurisdiction scope, actor, evidence, and revocation state.
- Encrypted biometric template metadata for face, voice, fingerprint, iris, palm, behavioral, and document-derived modalities.
- Template quality, retention, retirement, and revocation guardrails.
- Liveness-backed verification with presentation-attack evidence.
- Low-confidence match routing to independent review.
- Cross-border biometric processing routing to independent privacy review.
- Provider-neutral biometric governance-agent registration for Codex, Claude Code, opencode, and Pi style runtimes.
- Bytewax-first lifecycle batch validation for consent, template, verification, liveness, review, retention, and agent changes.
- Tenant-local state isolation and audit events for each lifecycle transition.
- Deterministic rule evaluation for generated APG applications and package tests.
- UI route metadata and compact theme components for biometric operations.

## Package Structure

- `SPECIFICATION.md` defines the functional and governance requirements.
- `PLAN.md` records the implementation and review checklist.
- `capability_contract.py` declares configuration, guardrails, UI routes, theme, and adapters.
- `biometric_runtime.py` implements the generated-app runtime.
- `api_helpers.py` exposes dependency-light API helper functions.
- `view_models.py` exposes route-ready UI model helpers.
- `app.py` derives semantic model and package manifest data from the live contract.
- `test_capability_contract.py` and `tests/test_package_contract.py` provide focused verification.

## Basic Usage

```python
from capabilities.common.biop.biometric_runtime import BiopService

service = BiopService()
tenant_id = "tenant-bio"

consent = service.record_consent(
    consent_id="consent-alice",
    tenant_id=tenant_id,
    subject_id="alice",
    purpose="workforce authentication",
    modalities=["face"],
    jurisdictions=["KE"],
    granted_by="alice",
    evidence="signed-consent:v1",
)

template = service.enroll_template(
    template_id="template-alice-face",
    tenant_id=tenant_id,
    subject_id="alice",
    modality="face",
    template_hash="sha256:face-template",
    encrypted=True,
    quality_score=0.95,
    consent_id=consent["id"],
    retention_policy="workforce-biometric-365d",
)

verification = service.request_verification(
    verification_id="verify-alice-login",
    tenant_id=tenant_id,
    subject_id="alice",
    template_id=template["id"],
    modality="face",
    requested_by="access-service",
    match_confidence=0.94,
    liveness_score=0.91,
    source_jurisdiction="KE",
    target_jurisdiction="KE",
)

assert verification["status"] == "verified"

agent = service.register_biometric_agent(
    agent_id="agent-privacy",
    tenant_id=tenant_id,
    name="Privacy Reviewer",
    runtime="codex",
    role="privacy_reviewer",
    scope="cross-border biometric review",
    owner="privacy-office",
    purpose="Review biometric transfer risk",
    contribution_disclosed=True,
    human_approval_required=True,
)

batch = service.validate_biop_lifecycle_batch(
    tenant_id=tenant_id,
    event_stream="bytewax",
    mutation_count=2,
    operation="biometric_agent_batch",
)

assert agent["status"] == "active"
assert batch["required_processor"] == "bytewax"
```

## Review Flows

Low-confidence matches become `pending_match_review`. Cross-border processing becomes `pending_privacy_review`. Reviews must be requested and decided by independent actors; the runtime rejects self-approval and stale review mutations.

```python
review = service.request_match_review(
    review_id="review-low-confidence",
    tenant_id=tenant_id,
    verification_id="verify-alice-login",
    requested_by="access-service",
    justification="Confidence below tenant threshold.",
)

service.decide_match_review(
    review_id=review["id"],
    tenant_id=tenant_id,
    reviewer="identity-reviewer",
    decision="approved",
    notes="Secondary evidence supports the match.",
)
```

## Agent and Lifecycle Guardrails

BIOP agents are composition records, not hardwired tool clients. The contract supports `codex`, `claude_code`, `opencode`, and `pi` runtimes through the `aicr_provider_neutral_biometric_agent_adapter` boundary. Unsupported runtimes, unsupported roles, missing scope, missing owner, missing purpose, and missing machine-contribution disclosure are denied. Privileged roles without human approval are retained as `pending_review`.

Lifecycle batches are Bytewax-first. A broker-specific queue-routed batch is intentionally denied:

```python
from capabilities.common.biop.capability_contract import evaluate_capability_rules

result = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "validate_biop_lifecycle_batch",
    "event_stream": "legacy_queue",
    "mutation_count": 1,
})

assert result["decision"] == "deny"
assert "bytewax_biop_stream_required" in result["matched_rules"]
```

## New Methods (service.py 43–50)

| # | Method | Purpose |
|---|--------|---------|
| 43 | `fido2_credential_register` | FIDO2/WebAuthn credential with AAGUID, attestation type, CTAP 2.2 backup flags |
| 44 | `fido2_assertion_verify` | Verify assertion; sign_count rollback → credential flagged `compromised` |
| 45 | `retention_policy_set` | Per-modality retention policy (days, legal_basis, jurisdiction) |
| 45b | `retention_sweep` | Revoke templates past expiry; suitable for nightly cron |
| 46 | `verification_cost_record` | Billing line per verification using `Decimal` + ROUND_HALF_EVEN |
| 46b | `billing_summary` | Aggregate billing by modality over date range; Decimal-precise totals |
| 47 | `step_up_session_create` | Step-up session with modality cascade and TTL |
| 47b | `step_up_session_evaluate` | Fuse new verification; returns satisfied/step_up_required/failed/expired |
| 48 | `biometric_agent_register` | Register AI agent (codex/claude_code/opencode/pi) with role and disclosure |
| 48b | `biometric_agent_invoke_log` | Immutable agent invocation record linked to biometric operation |
| 49 | `pad_evidence_chain_create` | SHA-256 bind PAD indicators + challenge nonce to verification |
| 49b | `pad_evidence_chain_verify` | Recompute chain hash; `integrity_verified=False` signals tampering |
| 50 | `match_confidence_with_uncertainty` | Verification with 90% CI; flags high-uncertainty accepts for review |

### Quick examples

```python
from capabilities.common.biop.service import BiometricService

svc = BiometricService(actor_id="api", tenant_id="acme")
user = await svc.register_user("emp-007", "Amina Hassan")

# FIDO2
cred = await svc.fido2_credential_register(
    user_id=user["user_id"],
    credential_id="cred-001",
    aaguid="adce0002-35bc-c60a-648b-0b25f1f05503",
    public_key_cbor="a501...",
    attestation_type="packed",
)
result = await svc.fido2_assertion_verify(
    credential_id="cred-001",
    authenticator_data_hex="...",
    client_data_hash_hex="...",
    signature_valid=True,
    new_sign_count=1,
)
assert result["decision"] == "accept"

# Retention sweep
await svc.retention_policy_set("fingerprint", 365, "GDPR_Art9_2b", "KE")
report = await svc.retention_sweep()

# Billing (Decimal)
verif = await svc.verify(user["user_id"], "face", b"probe")
bill = await svc.verification_cost_record(verif["verification_id"])
summary = await svc.billing_summary(from_date="2026-01-01")

# Step-up
session = await svc.step_up_session_create(
    user["user_id"], "fingerprint", 0.72, required_confidence=0.90,
    step_up_modalities=["face"]
)
session = await svc.step_up_session_evaluate(
    session["session_id"], verif["verification_id"], 0.91
)

# Uncertainty-aware verification
result = await svc.match_confidence_with_uncertainty(user["user_id"], "face", b"probe")
print(result["confidence_interval"], result["high_uncertainty"])

# PAD evidence chain
challenge = await svc.issue_liveness_challenge(user["user_id"])
await svc.complete_liveness_challenge(challenge["challenge_id"], b"resp", 0.96)
v = await svc.verify(user["user_id"], "face", b"probe")
chain = await svc.pad_evidence_chain_create(v["verification_id"], challenge["challenge_id"])
check = await svc.pad_evidence_chain_verify(chain["chain_id"])
assert check["integrity_verified"]
```

## Composition Notes

BIOP depends on `mfau`, `cvsn`, `aicr`, `encr`, `audl`, and `conf`. Optional adapters include `auth`, `frec`, `moni`, `cach`, and Bytewax event streams. Applications should compose BIOP through the capability contract and dependency-light runtime helpers, not by importing production-only web, hardware integration, or external agent internals.

Raw biometric samples are outside the package runtime. BIOP stores template metadata and encrypted-template evidence only.

---

## World-Class Enhancements (v2.0)

- **I1.** APG Biometric Authentication - World-Class Revolutionary Improvements
- **I2.** Overview: Revolutionary Market Leadership
- **I3.** 🧠 Revolutionary Improvement
- **I4.** Market Status: 100% UNIQUE - NO COMPETITOR OFFERS THIS
- **I5.** The Innovation Breakthrough
- **I6.** Technical Implementation
- **I7.** Revolutionary Capabilities
- **I8.** Quantifiable Business Impact
- **I9.** Competitive Differentiation
- **I10.** 🗣️ Revolutionary Improvement
- **I11.** Market Status: 100% UNIQUE - FIRST-EVER CONVERSATIONAL BIOMETRIC INTERFACE
- **I12.** The Innovation Breakthrough
- **I13.** Technical Implementation
- **I14.** Revolutionary Capabilities
- **I15.** Example Interactions

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
