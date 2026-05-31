# Digital Forms and eSign Capability Plan

## Implementation Packet

Build one coherent lifecycle and guardrail packet that makes `esgn` usable by generated applications without external services:

1. Document the capability boundary, runtime shape, rules, UI, adapters, and verification gates.
2. Expand the executable contract with form, submission, envelope, signing, evidence, first-class signing-agent, governance, observability, adapter, UI, theme, and Bytewax lifecycle streaming surfaces.
3. Extend the runtime models for document hash, expiry, envelope state reasons, first-class signing agents, and lifecycle-batch evidence.
4. Enforce signing lifecycle guardrails in `EsgnService`: schema, publication, consent, delegation, expiry, routing order, tamper seal, cancellation/rejection, evidence sealing, first-class signing-agent governance, and Bytewax lifecycle-batch validation.
5. Expose dependency-light API helpers and view models for the new lifecycle surfaces.
6. Refresh package semantic evidence from the live contract.
7. Run focused verification only, preserving battery.

## Review Checklist

- Every state-changing operation evaluates tenant context.
- Runtime checks match contract rule names and reasons.
- Envelope signing cannot bypass routing order, duplicate-signature blocking, expiry, tamper seal, or final states.
- Evidence cannot be created before completion or without encryption, retention, and audit reference.
- First-class signing agents cannot be registered without stable ID, readable name, supported runtime, supported role, scope, owner, purpose, disclosure, and privileged-role approval handling.
- Bytewax lifecycle batches cannot be accepted without a Bytewax stream, supported operation, and at least one mutation.
- UI routes and view models expose all user-visible lifecycle surfaces.
- Documentation explains production adapter boundaries instead of implying external services run locally.
- No disallowed message-bus dependency or stale generated-baseline marker is introduced.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/esgn/__init__.py capabilities/common/esgn/capability_contract.py capabilities/common/esgn/models.py capabilities/common/esgn/signing_engine.py capabilities/common/esgn/service.py capabilities/common/esgn/api.py capabilities/common/esgn/views.py capabilities/common/esgn/app.py capabilities/common/esgn/test_capability_contract.py capabilities/common/esgn/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/esgn/test_capability_contract.py capabilities/common/esgn/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.esgn import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/esgn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/esgn --json
git diff --check -- capabilities/common/esgn docs/progress_log.md
```

## Deliberately Out Of Scope

- Live identity proofing and signer authentication providers.
- Durable document/evidence storage.
- Cryptographic key custody and HSM integration.
- External AI-agent CLI execution.
- Notification delivery.
- Browser-rendered signing UI.
- Live Bytewax worker execution.
