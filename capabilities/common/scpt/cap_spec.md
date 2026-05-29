# APG Capability Specification: SCPT - Custom Scripting Engine

`scpt` is the APG Custom Scripting Engine capability. It provides tenant-aware
script registry behavior, package allowlist policy, constrained sandboxes,
publication approvals, workflow bindings, script executions, audit events,
deterministic rules, API helpers, route metadata, and script-workbench theming.

## Executable Runtime

The package is implemented as a dependency-light Python runtime:

| Surface | File | Responsibility |
| --- | --- | --- |
| Contract | `capability_contract.py` | configuration, deterministic scripting rules, UI routes, and theme tokens |
| Runtime helpers | `script_runtime.py` | stable IDs, language/permission normalization, import inspection, syntax checks, status helpers |
| Models | `models.py` | package policies, sandboxes, scripts, approvals, executions, and audit events |
| Service | `service.py` | tenant-scoped lifecycle methods and policy enforcement |
| API helpers | `api.py` | callable package API surface for composition and generated apps |
| View models | `views.py` | dashboard, workbench, script registry, execution console, sandbox monitor, package policy, approvals, settings |
| Package entrypoint | `app.py` | publishable semantic model, component manifest, and self-test |

The runtime intentionally does not execute arbitrary user code. It records
deterministic execution metadata and keeps live interpreters, process sandboxes,
container runners, package installers, workflow engines, schedulers, AI code
generation, and audit exporters behind explicit future adapters.

## Domain Model

`ScptService` manages:

- package policies with allowlisted packages, blocked imports, secrets,
  filesystem, network, and approval metadata
- sandboxes with tenant owner, runtime limits, memory limits, network policy,
  review status, and state
- script definitions with language, source, owner, version, state, requested
  permissions, dangerous permissions, approval state, package policy, sandbox,
  workflow bindings, and tags
- script approvals for publication and dangerous permission use
- execution records with requested actor, input, output, error, runtime,
  memory, logs, and status
- audit events for policy, sandbox, script, approval, publication, binding, and
  execution operations

The compatibility `create_record` and `list_records` methods produce and list
scripts so existing package tooling can keep treating SCPT as a composable APG
package while richer scripting APIs are used by new code.

## Rule Engine

SCPT uses deterministic rule evaluation from `capability_contract.py`.

| Rule | Enforced by |
| --- | --- |
| `tenant_context_required` | all service methods that create or mutate tenant-scoped objects |
| `script_requires_owner` | script creation |
| `sandbox_required` | publication and execution |
| `dangerous_permission_requires_approval` | scripts with network, filesystem, secrets, subprocess, or system access |
| `external_network_requires_policy` | scripts or sandboxes requiring network access |
| `high_resource_script_requires_review` | sandboxes requesting memory above the configured local threshold |

Python source is parsed for syntax errors and import hints. Network,
filesystem, subprocess, secret, and system permissions are modelled as
governed metadata rather than live host access.

## UI And Theme Contract

The package publishes APG route metadata for:

- `/scpt/dashboard`
- `/scpt/workbench`
- `/scpt/scripts`
- `/scpt/executions`
- `/scpt/sandboxes`
- `/scpt/packages`
- `/scpt/approvals`
- `/scpt/settings`

The default theme is `scpt_script_workbench`. Components include script editor,
execution log, sandbox monitor, and package policy tokens. View models expose
plain dictionaries so generated Python apps, APG Studio, and future UI adapters
can compose the scripting engine without framework-specific imports.

## Adapter Boundaries

Future live integrations should attach behind explicit adapters:

- isolated Python, JavaScript, and APG execution runners
- container or WASM sandbox adapters
- package-resolution and vulnerability-scanning adapters
- workflow engine adapter for `wflo`
- scheduler adapter for `schd`
- code-generation adapter for `ncod` or AI code assistants
- security and authorization adapters for `secu` and `auth`
- audit sink adapter for `audl`

Do not make package import, tests, publish-plan, or implementation audit depend
on those live providers.

## Focused Verification

Use focused checks while developing SCPT:

```bash
rg -n "<generated-baseline marker alternation>" capabilities/common/scpt
./.venv/bin/python -m py_compile capabilities/common/scpt/__init__.py capabilities/common/scpt/models.py capabilities/common/scpt/script_runtime.py capabilities/common/scpt/service.py capabilities/common/scpt/api.py capabilities/common/scpt/views.py capabilities/common/scpt/capability_contract.py capabilities/common/scpt/app.py capabilities/common/scpt/test_capability_contract.py capabilities/common/scpt/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/scpt/test_capability_contract.py capabilities/common/scpt/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/scpt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/scpt --json
```

The baseline-marker search should return no matches. The implementation audit
should classify `scpt` as `domain_specific` and report no root warnings.
