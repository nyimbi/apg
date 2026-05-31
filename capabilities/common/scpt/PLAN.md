# Custom Scripting Engine Capability Plan

## Implementation Packet

Build one coherent lifecycle and guardrail packet that makes `scpt` usable by
generated applications without external services:

1. Document the capability boundary, runtime shape, rules, UI, adapters, and
   verification gates.
2. Expand the executable contract with script, sandbox, package policy,
   execution, first-class scripting-agent, governance, observability, adapter,
   UI, theme, and Bytewax streaming surfaces.
3. Extend runtime models for package policy state, sandbox runtime/isolation
   metadata, script review/publication/retirement evidence, execution stream
   and cancellation evidence, scripting agents, and lifecycle batch records.
4. Enforce scripting lifecycle guardrails in `ScptService`: owner/source,
   package policy, sandbox policy, blocked imports, dangerous permissions,
   review and publication, workflow binding, Bytewax runtime events, execution
   evidence, cancellation, retirement, sandbox state, scripting-agent
   registration, privileged-agent review, and lifecycle batch validation.
5. Expose dependency-light API helpers and view models for the new lifecycle
   surfaces, including the scripting-agent roster and lifecycle batch monitor.
6. Refresh package semantic evidence from the live contract.
7. Run focused verification only, preserving battery.

## Review Checklist

- Every state-changing operation evaluates tenant context.
- Runtime checks match contract rule names and reasons.
- Scripts cannot publish without review, package policy, sandbox, and required
  approval.
- Dangerous permissions cannot bypass approval and policy.
- Blocked imports cannot be registered.
- Scripts cannot execute unless published, sandboxed, and requested by an actor.
- Execution cancellation and completion cannot omit evidence.
- Sandbox block/retire transitions require reasons.
- AI scripting agents cannot be registered without stable ID, readable name,
  supported runtime, supported role, scope, owner, purpose, and disclosure.
- Privileged scripting-agent roles require human approval evidence or remain in
  `pending_review`.
- Lifecycle batches cannot be accepted without Bytewax routing, supported
  lifecycle operation, and at least one mutation.
- Bytewax is the only batch/runtime stream accepted by the contract.
- UI routes and view models expose all user-visible lifecycle surfaces.
- Documentation explains production adapter boundaries instead of implying
  arbitrary code runs locally.
- No disallowed message-bus dependency or stale generated-baseline marker is
  introduced.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/scpt/__init__.py capabilities/common/scpt/models.py capabilities/common/scpt/script_runtime.py capabilities/common/scpt/service.py capabilities/common/scpt/api.py capabilities/common/scpt/views.py capabilities/common/scpt/capability_contract.py capabilities/common/scpt/app.py capabilities/common/scpt/test_capability_contract.py capabilities/common/scpt/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/scpt/test_capability_contract.py capabilities/common/scpt/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.scpt import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities inspect scpt --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/scpt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/scpt --json
git diff --check -- capabilities/common/scpt docs/progress_log.md
```

## Deliberately Out Of Scope

- Live Python, JavaScript, APG, WASM, or container execution.
- Package installation and vulnerability scanning.
- External AI-agent CLI execution.
- Browser-rendered script workbench behavior.
- Durable execution databases and migrations.
- Live Bytewax worker execution.
