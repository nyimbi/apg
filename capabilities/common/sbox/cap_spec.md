# Sandbox/Testing Environment Capability Specification

- **Capability Name**: Sandbox/Testing Environment
- **Capability ID**: `sbox`
- **Category**: common
- **Version**: 1.0.0
- **Theme**: `sbox_safe_testing`

## Purpose

SBOX is the APG sandbox and safe testing environment capability. It gives APG
developers, generated applications, plugin authors, AI-agent capacity builders,
and integration teams a tenant-scoped way to create isolated environments,
attach reusable templates, manage synthetic or masked datasets, execute test
runs, enforce safety policy, and retain audit evidence.

The package is executable today through the Python package runtime:

- `models.py` defines isolation profiles, sandbox templates, datasets,
  sandbox environments, sandbox runs, and audit events.
- `sandbox_runtime.py` provides deterministic IDs, normalization helpers,
  sandbox state transitions, run result classification, and sandbox risk
  scoring.
- `service.py` provides `SboxService` with lifecycle behavior for isolation
  profiles, templates, datasets, sandboxes, runs, expiration, dashboard
  summaries, compatibility records, and policy enforcement.
- `api.py` exposes dependency-light helper functions for composition tooling
  and generated Python applications.
- `views.py` exposes dashboard, console, template, dataset, run monitor,
  policy, log, and settings view models.
- `capability_contract.py` publishes tenant configuration, deterministic rules,
  UI routes, and visual theme tokens.

## Provided Services

- `sandbox_registry`
- `isolation_profiles`
- `test_runs`
- `synthetic_datasets`
- `safety_policy`
- `sbox_operations`

## Required Services And Adapter Boundaries

- `tenant_context` scopes every executable sandbox operation.
- `plgn` is the plugin-test adapter boundary.
- `secu` is the policy and isolation adapter boundary.
- `envm` is the environment provisioning adapter boundary.
- Optional `cicd`, `depl`, `logt`, and `agnt` integrations remain adapters for
  pipeline execution, deployment sandboxes, log export, and AI-agent test runs.

The package keeps those integrations local and deterministic until a future
slice verifies live providers directly.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Important groups:

- `sandboxes`: owner, template, TTL, and isolation requirements.
- `isolation`: network policy, secret redaction, data masking, and outbound
  network defaults.
- `datasets`: synthetic data, production data review, lineage, and retention
  policy.
- `governance`: tenant context, audit, long-lived review, and plugin-test
  policy requirements.
- `ui`: console, template library, run monitor, and policy center enablement.
- `theme`: default theme and tenant override policy.

## Rules

SBOX evaluates deterministic policy rules through `evaluate_capability_rules()`.

| Rule | Decision | Runtime intent |
| --- | --- | --- |
| `tenant_context_required` | deny | all sandbox operations require tenant context |
| `sandbox_requires_owner` | deny | sandbox creation requires an accountable owner |
| `sandbox_requires_isolation_profile` | deny | sandbox creation requires an isolation profile |
| `secrets_require_redaction` | deny | secret access requires redaction policy |
| `outbound_network_requires_approval` | deny | outbound sandbox network access requires approval |
| `long_lived_sandbox_requires_review` | require_review | long TTL sandboxes require lifecycle review |

The service layer also enforces dataset lineage, retention policy, production
data review, dataset masking, plugin test policy, tenant ownership, positive
test counts, and sandbox run eligibility.

## Runtime Lifecycle

The core lifecycle is:

1. Create an isolation profile with network, data, and secret controls.
2. Create a reusable sandbox template with runtime, owner, TTL, and plugin
   policy metadata.
3. Register a synthetic, masked, fixture, or reviewed production-sample
   dataset with lineage and retention.
4. Create a sandbox from a template, isolation profile, owner, TTL, datasets,
   secret/network needs, and lifecycle review evidence.
5. Compute sandbox risk from TTL, outbound network, secret access, dataset
   type, and isolation level.
6. Start a test, integration, plugin, agent, migration, or load run.
7. Complete the run with passed, failed, or blocked counts and update sandbox
   state.
8. Expire the sandbox when it is no longer valid.
9. Record audit events for isolation profile, template, dataset, sandbox, run,
   and expiration actions.

## UI

The package exposes these APG Python UI routes:

| Route | Path | Component | Permission |
| --- | --- | --- | --- |
| `dashboard` | `/sbox/dashboard` | `SBOXDashboard` | `sbox:view` |
| `sandboxes` | `/sbox/sandboxes` | `SandboxConsole` | `sbox:create` |
| `templates` | `/sbox/templates` | `TemplateLibrary` | `sbox:create` |
| `datasets` | `/sbox/datasets` | `DatasetManager` | `sbox:manage_policy` |
| `runs` | `/sbox/runs` | `RunMonitor` | `sbox:run_tests` |
| `policies` | `/sbox/policies` | `PolicyCenter` | `sbox:manage_policy` |
| `logs` | `/sbox/logs` | `SandboxLogs` | `sbox:view` |
| `settings` | `/sbox/settings` | `SBOXSettings` | `sbox:admin` |

`views.py` returns dashboard, sandbox console, template library, dataset
manager, run monitor, policy center, logs, and settings models for generated
Python app composition.

## Theme

SBOX uses `sbox_safe_testing` with compact operational density:

- sandbox cards expose container icons, TTL pills, and isolation bands;
- run monitors expose test timelines and result chips;
- dataset managers expose masked data grids and lineage chips;
- policy centers expose guardrail lists and approval chips.

Theme tokens are published by the capability contract so composed applications
can apply consistent safety, status, warning, danger, surface, text, radius, and
density values.

## Focused Verification

Use battery-conscious package verification first:

```bash
./.venv/bin/python -m py_compile capabilities/common/sbox/__init__.py capabilities/common/sbox/models.py capabilities/common/sbox/sandbox_runtime.py capabilities/common/sbox/service.py capabilities/common/sbox/api.py capabilities/common/sbox/views.py capabilities/common/sbox/capability_contract.py capabilities/common/sbox/app.py capabilities/common/sbox/test_capability_contract.py capabilities/common/sbox/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/sbox/test_capability_contract.py capabilities/common/sbox/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/sbox --json
./.venv/bin/apg capabilities publish-plan capabilities/common/sbox --json
```

## Known Non-Goals

- Live container, VM, Kubernetes, service-mesh, CI/CD, deployment, log export,
  and policy-engine integrations remain adapter boundaries.
- The package does not require live infrastructure for focused package proof.
- AI-agent sandbox execution is modeled as a run type; provider execution
  belongs behind explicit `agnt` or runtime adapters.
