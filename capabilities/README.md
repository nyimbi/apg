# APG Capabilities

`capabilities/` is the APG application component catalog. Each capability is a
package-backed building block that generated APG applications can compose into
larger products, such as ERPs, collaboration systems, AI workbenches, finance
platforms, security consoles, or industry-specific operational tools.

The catalog is intentionally executable. A capability is not just metadata: it
must define a stable contract, tenant-safe configuration, deterministic rules,
domain runtime behavior, API helpers, UI route/view metadata, visual theme
metadata, tests, release evidence, and side-effect-free publish planning.

Use the local CLI as the source of truth:

```bash
./.venv/bin/apg capabilities list --json
./.venv/bin/apg capabilities inspect <capability_id> --json
./.venv/bin/apg capabilities evaluate-rules <capability_id> --context-json '{}' --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities lifecycle-audit --json
```

## Current Registry Snapshot

The current registry exposes 146 valid capability contracts. The latest focused
implementation audit reports:

- 146 domain-specific capability packages;
- 0 materialized-baseline packages;
- 0 mixed-implementation packages;
- 0 contract-only packages;
- 0 blocking implementation gaps;
- 0 implementation-audit warnings.

The strict package-artifact operability audit also currently reports 146
operable contracts, 146 complete package artifact sets, 0 package gaps, 0
warnings, and 0 errors.

Category coverage:

| Category | Count | Purpose |
| --- | ---: | --- |
| `common` | 81 | Cross-cutting platform, AI, data, security, collaboration, workflow, and infrastructure capabilities. |
| `composition` | 6 | System-level composition hubs for access, configuration, events, gateway, orchestration, and registry behavior. |
| `fin` | 6 | Finance capabilities such as general ledger, accounts payable, accounts receivable, budgeting, cash, and reporting. |
| `hcm` | 3 | Human-capital capabilities for employee data, payroll, and time attendance. |
| `ckm` | 3 | Collaboration and knowledge-management capabilities. |
| `grc` | 2 | Governance, risk, compliance, and document-control capabilities. |
| `crm` | 1 | Customer relationship management analytics. |
| `eam` | 1 | Enterprise asset management. |
| `ecd` | 1 | ESG and sustainability management. |
| `fintech` | 27 | Fintech gateway, digital payments, digital cards, digital wallet, mobile banking, banking APIs, embedded finance, wealth management, robo advisory, portfolio management, algorithmic trading, crowdfunding, digital neobanking, digital lending, buy now pay later, agency banking, remittance, insurtech, risk management, regulatory technology, KYC, AML, fraud, compliance automation, blockchain services, cryptocurrency services, and decentralized finance. |
| `int` | 1 | Integration API management. |
| `intel` | 12 | Intelligence crawler services, open source intelligence, signals intelligence, human intelligence, geospatial intelligence, cyber intelligence, financial intelligence, social media intelligence, dark web monitoring, radio intelligence listening, digital surveillance, and real-time monitoring. |
| `pde` | 1 | Product information management. |
| `scm` | 1 | Supply-chain vendor management. |

Run this command whenever the catalog changes:

```bash
./.venv/bin/apg capabilities list --json
```

## What A Capability Is

A capability is the smallest APG unit that should be independently useful,
inspectable, testable, and composable. It owns a bounded domain concern and
publishes enough information for the compiler, generated applications, humans,
and automation to understand how to use it.

World-class APG capabilities provide:

- **Specific functionality**: a concrete domain outcome, not a generic record
  store or placeholder.
- **Stable contract**: a `capability_contract.py` with explicit capability ID,
  display name, services, dependencies, configuration, rules, UI, and theme.
- **Tenant-safe configuration**: defaults and schemas that make tenant context,
  data ownership, retention, and policy behavior explicit.
- **Rule-engine guardrails**: deterministic allow, deny, review, warn, and audit
  decisions that service methods actually enforce.
- **Executable runtime behavior**: domain records and lifecycle methods that can
  run locally without live provider credentials.
- **API helpers**: dependency-light Python helpers that generated apps and other
  capabilities can call without importing heavy web stacks.
- **UI composition metadata**: route names, paths, components, permissions,
  navigation groups, actions, and view models.
- **Theme metadata**: semantic tokens and component-level theming so applications
  can render capability screens consistently.
- **Adapter boundaries**: live systems stay behind replaceable adapter contracts.
- **Release evidence**: semantic model, package manifest, self-test evidence,
  and publish-plan output.
- **Contributor handoff**: local docs, tests, and progress entries that make the
  next change safe.

Generated shape is not enough. A package becomes an APG capability only when
another APG application can compose it without depending on private
implementation details.

## Standard Package Shape

Executable package-backed capabilities use this shape:

```text
capabilities/<category>/<code>/
  SPECIFICATION.md          # target functionality, users, lifecycles, rules, UI, adapters
  PLAN.md                   # implementation packets, review plan, test strategy
  README.md                 # local usage and contributor handoff
  cap_spec.md               # current executable runtime behavior and proof commands
  capability_contract.py    # registry contract: config, rules, UI, theme, dependencies
  models.py                 # package domain records and data contracts
  service.py                # package-owned runtime behavior and rule enforcement
  api.py                    # dependency-light API/helper surface
  views.py                  # UI/view-model composition surface
  app.py                    # package entrypoint and self-test surface
  semantic_model.json       # generated semantic evidence
  package_manifest.json     # generated package metadata
  release_report.json       # release/self-test evidence
  test_capability_contract.py
  tests/
```

Some older or specialized packages have additional runtime modules, deployment
helpers, generated artifacts, migrations, or nested package roots. Keep local
shape consistent with the package's existing style unless a cleanup plan says
otherwise.

## How Applications Use Capabilities

APG applications consume capabilities in three layers:

1. **Compile-time contract selection**
   - The compiler and CLI inspect capability contracts to understand stable
     IDs, dependencies, configuration keys, rules, routes, and theme surfaces.
   - APG source can declare or require capabilities by stable ID.

2. **Runtime composition**
   - Generated applications call package API helpers or service methods for
     local runtime behavior.
   - Live provider integrations are resolved through adapters, not by embedding
     provider calls into generated code.

3. **UI and governance composition**
   - Generated applications mount route/view metadata, permissions, actions,
     theme tokens, and rule decisions into a larger application shell.
   - Capability rules provide consistent denial, review, audit, and warning
     behavior across composed screens.

Typical flow:

```bash
# Discover a capability ID.
./.venv/bin/apg capabilities list --json

# Inspect its contract, routes, rules, dependencies, and theme.
./.venv/bin/apg capabilities inspect wflo --json

# Compile an APG application that uses the catalog.
./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg \
  --catalog capabilities \
  --output /tmp/apg-enterprise-app \
  --verify

# Run the generated smoke test.
./.venv/bin/python /tmp/apg-enterprise-app/smoke_test.py
```

## CLI Usage

List all capability contracts:

```bash
./.venv/bin/apg capabilities list --json
```

Inspect one capability:

```bash
./.venv/bin/apg capabilities inspect accs --json
```

Evaluate deterministic rules:

```bash
./.venv/bin/apg capabilities evaluate-rules accs \
  --context-json '{"tenant_context_present": true}' \
  --json
```

Validate package artifacts and operability probes:

```bash
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

This command is intentionally stricter than implementation-depth auditing. It
can fail even when every package has complete files, because it also checks
whether rule probes return the expected decision shape.

Audit implementation depth:

```bash
./.venv/bin/apg capabilities implementation-audit --json
```

Audit lifecycle evidence for the requested SPECIFICATION -> PLAN ->
implementation -> review/readiness cycle:

```bash
./.venv/bin/apg capabilities lifecycle-audit --json
```

Build a side-effect-free publish plan for one package:

```bash
./.venv/bin/apg capabilities publish-plan capabilities/common/accs --json
```

Run the standard focused proof for one package:

```bash
./.venv/bin/pytest -q \
  capabilities/<category>/<code>/test_capability_contract.py \
  capabilities/<category>/<code>/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/<category>/<code> --json
./.venv/bin/apg capabilities publish-plan capabilities/<category>/<code> --json
git diff --check -- capabilities/<category>/<code>
```

Run broader catalog proof after shared infrastructure or registry changes:

```bash
./.venv/bin/apg capabilities validate-contracts --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities lifecycle-audit --json
```

## AI Agent Composition Standard

AI agents are first-class APG citizens. A capability that exposes AI-assisted
composition, review, triage, generation, planning, or operations should model
agents explicitly in the contract and service layer.

The provider-neutral runtime set is:

```text
codex
claude_code
opencode
pi
```

Capability agent support should include:

- a top-level `agents` configuration block or capability-specific agent block;
- supported runtimes and supported roles;
- privileged-role metadata;
- a declared AICR adapter contract instead of direct vendor invocation;
- required agent ID, readable name, owner, scope, purpose, and contribution
  disclosure;
- human approval for privileged roles;
- service methods that persist agent records and enforce those guardrails;
- UI route/view metadata for agent rosters and pending review;
- tests for valid agents, unsupported runtimes, unsupported roles, missing
  ownership/purpose, missing disclosure, and privileged-role review behavior.

Agent implementation should make APG better at composing applications while
keeping live Codex, Claude Code, OpenCode, Pi, or other provider execution
behind explicit adapters.

## Bytewax Lifecycle Standard

Bytewax is the lifecycle stream-processing foundation for APG capability
batches. Do not introduce product-specific queue or broker lifecycle coupling
into capability contracts or services.

A capability lifecycle packet should define:

- `streaming.processor` or equivalent metadata set to `bytewax`;
- lifecycle stream name, for example `<capability>.lifecycle`;
- supported lifecycle operations;
- required batch mutation structure;
- rejection rules for non-Bytewax processors;
- API helpers and service methods for lifecycle batch validation;
- UI route/view metadata for lifecycle batch monitoring;
- tests for accepted Bytewax batches, empty batches, unsupported operations, and
  non-Bytewax processors.

Event buses, queues, notification systems, and delivery bridges may exist behind
adapters, but the capability lifecycle contract must remain Bytewax-oriented and
provider-neutral.

## Developing A Capability

Use one coherent packet at a time. The expected build cycle is:

1. **Read**
   - Inspect `capability_contract.py`, runtime modules, service, API, views,
     tests, `SPECIFICATION.md`, `PLAN.md`, `cap_spec.md`, and recent
     `docs/progress_log.md` entries.

2. **Specify**
   - Update `SPECIFICATION.md` before changing behavior.
   - Define the domain outcome, users, lifecycle states, data ownership,
     guardrails, UI surfaces, theme requirements, adapter boundaries, risks, and
     acceptance evidence.

3. **Plan**
   - Update `PLAN.md` with the implementation packets, test strategy, review
     checklist, and out-of-scope live providers.

4. **Implement**
   - Add or refine models, services, API helpers, views, contract metadata,
     app/self-test behavior, generated evidence, and tests.
   - Keep live providers behind adapters.
   - Preserve existing domain behavior unless a documented review finding
     requires a change.

5. **Review**
   - Perform an in-process code review before committing.
   - Look for rule metadata that service methods do not enforce, unsafe boolean
     coercion, tenant-scope gaps, implicit owner/purpose fallbacks, stale route
     metadata, missing negative tests, and stale generated evidence.

6. **Verify**
   - Run focused proof commands.
   - Inspect command output before claiming completion.
   - Document any skipped full-suite or live-provider checks as known gaps.

7. **Record**
   - Update `docs/progress_log.md` with what changed, proof commands, review
     findings, fixes, and known gaps.

8. **Commit and push**
   - Stage only the capability slice and progress log.
   - Use the Lore commit protocol.
   - Push verified slices regularly.

## Parallel Development

Parallel capability work is encouraged when ownership is clean.

Safe parallel work:

- one agent owns exactly one capability package root;
- documentation-only catalog work is separate from package runtime work;
- generated evidence is refreshed by only the owner of that package;
- shared CLI, registry, compiler, and documentation changes are coordinated.

Unsafe parallel work without coordination:

- two agents editing the same capability package;
- contract registry JSON format changes;
- CLI command or output schema changes;
- route, rule, capability ID, or theme renames;
- shared docs that define acceptance gates;
- bulk generated evidence refreshes across many packages.

Use `docs/progress_log.md` to make handoffs durable. If a future agent needs
chat history to understand the current state, the package documentation is not
complete enough.

## Review Standard

A capability change is ready only when a reviewer can answer these questions
from files and command output:

- What public capability contract changed?
- Which lifecycle, rule, or UI surface became more executable?
- Which tenant, security, approval, and audit boundaries are enforced?
- Which service methods enforce the contract rules?
- Which generated application or package API surface can consume the behavior?
- Which UI routes and theme components expose it?
- Which adapter boundaries remain deliberately external?
- Which focused commands prove the package?
- Which full-suite, browser, live-provider, or load checks were deliberately not
  run?
- Which next packet should follow?

If any answer depends on chat history, update `SPECIFICATION.md`, `PLAN.md`,
`README.md`, `cap_spec.md`, or `docs/progress_log.md` before committing.

## Troubleshooting

If `capabilities list` cannot find a package:

- confirm the package has `capability_contract.py`;
- confirm the contract exports a loadable capability object or helper expected
  by the registry;
- run `./.venv/bin/python -m py_compile` on the package files.

If contract validation fails:

- inspect the missing field in the CLI JSON output;
- verify `configuration`, `configuration_schema`, `rule_engine`, `ui`, and
  `theme` are present and serializable;
- avoid non-JSON values in contract metadata.

If implementation audit reports a baseline package:

- replace generated record-only behavior with domain-specific models and
  service methods;
- add positive lifecycle tests and negative guardrail tests;
- refresh `cap_spec.md`, semantic evidence, and release evidence.

If publish-plan reports stale evidence:

- run the package `app.py` self-test if available;
- regenerate `semantic_model.json`, `package_manifest.json`, or
  `release_report.json` using the package's existing generation path;
- validate regenerated JSON with `./.venv/bin/python -m json.tool`.

If a lifecycle packet mentions a product-specific queue or broker core
dependency:

- keep product-specific delivery behavior behind adapters;
- update contract/service metadata to use Bytewax lifecycle processing;
- add a negative guardrail test for non-Bytewax lifecycle batches.

## Completion Target

The capability-development goal is complete only when every capability has been
reviewed through the specification, plan, implementation, code-review,
verification, and documentation cycle, and the repository still proves:

```bash
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg docs audit --json
./.venv/bin/apg tooling audit --json
```

Until then, treat each verified capability packet as progress toward the larger
APG platform, not as completion of the full objective.
