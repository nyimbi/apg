# Custom Scripting Engine Capability

`scpt` is the APG common capability for governed custom scripting. It gives
generated applications a dependency-light runtime for registering tenant-owned
scripts, constraining them with package and sandbox policy, approving risky
permissions, binding scripts to workflows, tracking deterministic execution
metadata, composing first-class scripting agents, validating Bytewax lifecycle
batches, and exposing scripting UI models without running arbitrary host code.

## What It Provides

- Tenant-scoped package policies for allowed packages, blocked imports, secret
  access, filesystem access, network policy, and approval evidence.
- Sandboxes with runtime language, isolation mode, CPU/runtime/memory limits,
  network policy, health/state metadata, and block/retire transitions.
- Versioned script definitions with source, language, checksum, owner, review
  state, publication state, workflow bindings, requested permissions, dangerous
  permission detection, package policy, and sandbox binding.
- Approval records for publish, dangerous permission, package policy, sandbox
  exception, retirement, and workflow-binding decisions.
- Execution metadata for requested actor, event stream, input/output, logs,
  runtime, memory, timeout, cancellation, completion evidence, and audit trail.
- First-class scripting agents with provider-neutral runtime (`codex`,
  `claude_code`, `opencode`, `pi`), governance role, owner, purpose, scope,
  disclosure, privileged-role review, and audit events.
- Bytewax lifecycle batch validation for package, sandbox, script, approval,
  execution, scripting-agent, and audit mutations.
- Rule-engine, UI-route, visual-theme, Bytewax streaming, semantic-model,
  release, and publish-plan metadata for APG composition.

## Runtime Surface

Use `service.ScptService` for local generated-application behavior:

```python
from capabilities.common.scpt.service import ScptService

service = ScptService()
policy = service.create_package_policy(
    "tenant-a",
    "stdlib",
    "platform",
    allowed_packages=["json"],
)
sandbox = service.create_sandbox("tenant-a", "python-local", "platform")
script = service.create_script(
    "tenant-a",
    "normalize-payload",
    "python",
    "result = input_payload",
    "automation",
    package_policy_id=policy["id"],
    sandbox_id=sandbox["id"],
)
reviewed = service.request_script_review(
    "tenant-a",
    script["id"],
    "reviewer",
    "safe deterministic transform",
)
service.publish_script("tenant-a", reviewed["id"], "automation")
execution = service.execute_script(
    "tenant-a",
    reviewed["id"],
    sandbox["id"],
    "workflow-runner",
    {"customer_id": "C-1"},
)
service.complete_execution("tenant-a", execution["id"], output={"ok": True})
service.register_scripting_agent(
    "agent-1",
    "tenant-a",
    "Script Reviewer",
    "codex",
    "reviewer",
    script["id"],
    "platform",
    True,
    owner_ref="platform",
    purpose="Review generated scripts for publication readiness.",
)
service.validate_lifecycle_batch("tenant-a", "bytewax", 1, "scripting_agent_batch")
```

Dependency-light API helpers in `api.py` wrap the same service methods for
generated endpoints. `views.py` provides dashboard, workbench, script registry,
execution console, sandbox monitor, package policy, approval, scripting-agent,
audit, analytics, and settings view models.

## Guardrails

The deterministic rule engine blocks or flags:

- missing tenant context, script owner, script name, source, package policy,
  sandbox, review evidence, approval evidence, requested actor, execution
  evidence, cancel reason, retirement reason, and workflow-binding policy;
- dangerous permissions without approval;
- network, secret, and filesystem access without policy;
- blocked imports;
- high-resource sandboxes without review;
- execution of unpublished scripts or unavailable sandboxes;
- non-Bytewax runtime or batch event streams;
- unsupported, unnamed, unowned, purposeless, unscoped, or undisclosed
  scripting agents;
- privileged scripting agents without human approval evidence, which are held
  in review instead of silently activated;
- empty, unsupported, or non-Bytewax scripting lifecycle batches;
- cross-tenant access.

## Scripting Agents

Scripting agents are first-class APG composition citizens. SCPT records their
stable ID, name, runtime, role, owner, purpose, scripting scope, contribution
disclosure, human-approval treatment, and review status. SCPT does not invoke
external Codex, Claude Code, OpenCode, Pi, or other agent clients directly;
those integrations belong behind the AICR provider-neutral adapter contract
named in the capability contract.

Privileged roles such as `author`, `reviewer`, `policy_advisor`,
`runtime_triage`, `lifecycle_batch_reviewer`, and `script_steward` require
human approval evidence to become active immediately. Without that evidence,
SCPT keeps the agent in `pending_review`.

## Bytewax Lifecycle Batches

SCPT validates lifecycle batches through `validate_lifecycle_batch(...)`.
Accepted batches must use the Bytewax stream, include at least one mutation, and
name a configured operation such as `package_policy_batch`, `sandbox_batch`,
`script_batch`, `approval_batch`, `execution_batch`, `scripting_agent_batch`,
or `audit_batch`. The packet exposes Bytewax metadata and guardrails without
starting a live worker in dependency-light generated applications.

## Composition

Required capabilities are `wflo`, `secu`, `auth`, `audl`, and `aicr`.
Optional production adapters include `schd`, `ncod`, `moni`, and `them`.

The local package does not execute arbitrary source, install packages, spawn
containers, run WASM, call external AI CLIs, or start live Bytewax workers.
Those are production adapters behind the APG composition layer. The package
does enforce the same contract and policy shape that production adapters must
honor.

## Verification

Focused package proof:

```bash
./.venv/bin/python -m py_compile capabilities/common/scpt/__init__.py capabilities/common/scpt/models.py capabilities/common/scpt/script_runtime.py capabilities/common/scpt/service.py capabilities/common/scpt/api.py capabilities/common/scpt/views.py capabilities/common/scpt/capability_contract.py capabilities/common/scpt/app.py capabilities/common/scpt/test_capability_contract.py capabilities/common/scpt/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/scpt/test_capability_contract.py capabilities/common/scpt/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.scpt import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities inspect scpt --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/scpt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/scpt --json
```
