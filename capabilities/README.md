# APG Capabilities

`capabilities/` contains APG's package-backed application building blocks. A
capability is a composable unit of executable behavior with a stable contract,
tenant-safe configuration, deterministic rules, UI route metadata, visual
theme metadata, package runtime code, tests, release evidence, and publish-plan
support.

The current registry contains 109 capability contracts. Use the registry and
audits as the source of truth:

```bash
./.venv/bin/apg capabilities list --json
./.venv/bin/apg capabilities inspect <capability_id> --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
./.venv/bin/apg capabilities implementation-audit --json
```

## What A Capability Provides

Every capability must make a business or platform boundary executable. A
world-class APG capability provides:

- a clear domain purpose and stable capability ID;
- explicit provided services and required dependencies;
- tenant-scoped configuration and configuration schema;
- deterministic rule-engine decisions with denial/review reasons;
- package-owned domain models and service lifecycles;
- dependency-light API helpers and view models;
- UI routes, permissions, screen ownership, and theme tokens;
- adapter boundaries for live providers and external systems;
- positive lifecycle tests and negative guardrail tests;
- package manifest, semantic model, release evidence, and publish-plan output;
- local handoff documentation for the next contributor.

Generated shape is not enough. A capability is useful when another APG
application can compose it without reading private implementation details.

## Directory Shape

Executable package-backed capabilities use this shape:

```text
capabilities/<category>/<code>/
  SPECIFICATION.md          # capability intent, scope, rules, UX, adapters
  PLAN.md                   # implementation and review plan for this package
  cap_spec.md               # current technical runtime spec and proof commands
  capability_contract.py    # registry contract: config, rules, UI, theme
  models.py                 # package domain records and data contracts
  service.py                # package-owned business/runtime behavior
  api.py                    # dependency-light API/helper surface
  views.py                  # UI/view-model composition surface
  app.py                    # package entrypoint and self-test surface
  semantic_model.json       # package semantic evidence
  package_manifest.json     # package metadata
  release_report.json       # release/self-test evidence
  test_capability_contract.py
  tests/
```

Older packages may still use `cap_spec.md` as their local specification. New
development should add `SPECIFICATION.md` before changing behavior, then keep
`cap_spec.md` aligned with the executable runtime after implementation.

## Capability Categories

The capability registry currently groups contracts as follows:

| Category | Count | Capability IDs |
| --- | ---: | --- |
| `common` | 81 | `accs`, `agnt`, `aicr`, `anom`, `apig`, `audl`, `audp`, `auth`, `bclg`, `biop`, `bkup`, `cach`, `chat`, `cicd`, `colb`, `comp`, `conf`, `conn`, `cons`, `cvsn`, `depl`, `dist`, `dlpd`, `dtwn`, `dvrl`, `edge`, `encr`, `envm`, `esgc`, `esgn`, `etlp`, `fedl`, `frec`, `geos`, `grag`, `grph`, `help`, `hlth`, `i18n`, `idfd`, `imex`, `iotd`, `keym`, `kngr`, `logt`, `mchn`, `mdm`, `meta`, `mfau`, `mlcm`, `moni`, `mqeb`, `mten`, `ncod`, `nlpc`, `ntfy`, `onto`, `plfd`, `plgn`, `pose`, `pred`, `quan`, `ragn`, `recs`, `regy`, `sbox`, `schd`, `scpt`, `scrp`, `secu`, `seop`, `shdn`, `srch`, `tens`, `them`, `usrm`, `vidc`, `walt`, `wflo`, `wsbl`, `ztna` |
| `composition` | 6 | `composition_access`, `composition_config`, `composition_events`, `composition_gateway`, `composition_orchestration`, `composition_registry` |
| `fin` | 6 | `apy_accounts_payable`, `arc_accounts_receivable`, `bfc_budgeting_forecasting`, `cbm_cash_management`, `fin_rpt`, `glr_general_ledger` |
| `hcm` | 3 | `chr_employee_data_management`, `pay_payroll`, `tat_time_attendance` |
| `ckm` | 3 | `ckm_not`, `ckm_rtc`, `ckm_wfa` |
| `grc` | 2 | `grc_doc`, `grc_rcm` |
| `crm` | 1 | `crm_adv` |
| `eam` | 1 | `eam_ast` |
| `ecd` | 1 | `ecd_esg` |
| `fintech` | 1 | `fintech_gateway` |
| `int` | 1 | `int_api` |
| `intel` | 1 | `intel_crawler` |
| `pde` | 1 | `pde_pim` |
| `scm` | 1 | `scm_ven` |

## How To Use A Capability

1. List available contracts:

   ```bash
   ./.venv/bin/apg capabilities list --json
   ```

2. Inspect a capability:

   ```bash
   ./.venv/bin/apg capabilities inspect accs --json
   ```

3. Evaluate its deterministic rules:

   ```bash
   ./.venv/bin/apg capabilities evaluate-rules accs \
     --context-json '{"tenant_context_present": true}' \
     --json
   ```

4. Validate all package artifacts:

   ```bash
   ./.venv/bin/apg capabilities audit --strict-package-artifacts --json
   ```

5. Build a side-effect-free publish plan for one package:

   ```bash
   ./.venv/bin/apg capabilities publish-plan capabilities/common/accs --json
   ```

6. Use the capability from APG source by declaring or requiring the stable
   capability ID, then compile the application:

   ```bash
   ./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg \
     --catalog capabilities \
     --output /tmp/apg-capability-app \
     --verify
   ./.venv/bin/python /tmp/apg-capability-app/smoke_test.py
   ```

## Development Workflow

Every capability should move through the same sequence:

1. **Specification**: Write or update `SPECIFICATION.md` with the domain
   outcome, users, service boundary, lifecycle, rules, UI, theme, adapters,
   risks, and acceptance gates.
2. **Plan**: Write or update `PLAN.md` with implementation packets, test
   strategy, review checklist, and out-of-scope provider work.
3. **Implementation**: Update domain models, services, API helpers, view
   models, contract metadata, and tests. Keep live providers behind adapters.
4. **Focused proof**: Run package tests, root implementation audit, and
   publish-plan.
5. **Code review**: Review the changed package for domain correctness,
   guardrail coverage, tenant safety, API/view consistency, and stale docs.
6. **Global proof**: Run capability and tooling audits when a package changes
   global readiness.
7. **Commit**: Commit only the verified package slice using the Lore protocol.

## Minimum Proof Commands

Run these after changing one package:

```bash
./.venv/bin/pytest -q capabilities/<category>/<code>/test_capability_contract.py capabilities/<category>/<code>/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/<category>/<code> --json
./.venv/bin/apg capabilities publish-plan capabilities/<category>/<code> --json
git diff --check -- capabilities/<category>/<code>
```

Run these after changing shared capability infrastructure or documentation:

```bash
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg docs audit --json
./.venv/bin/apg tooling audit --json
```

## Review Standard

A capability change is not ready until a reviewer can answer:

- What public capability contract changed?
- Which lifecycle or guardrail became more executable?
- Which tenant, security, approval, and audit boundaries are enforced?
- Which UI routes and theme surfaces expose the behavior?
- Which adapter boundaries remain deliberately external?
- Which focused commands prove the package?
- Which next packet should follow?

If the answer requires chat history, update `SPECIFICATION.md`, `PLAN.md`,
`cap_spec.md`, or the local README before committing.
