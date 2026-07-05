# APG Capabilities Overview

APG capabilities are composable business and infrastructure units. They are
represented in APG source through `capability` declarations and in the source
tree through packages under `capabilities/<domain>/<code>/`.

This page describes the current repository inventory and the contract expected
from capability packages. Use audits for exact readiness, because the tree
contains a mix of domain-specific packages, materialized baselines, mixed
packages, contract-only packages, and build artifacts.

## Current Inventory

Source-tree inventory:

- 33 top-level capability domains.
- 440 non-hidden `capabilities/<domain>/<code>` directories.
- 322 checked `cap_spec.md` files.
- 592 `capability_contract.py` files when build output copies are included.

Largest domains by source directory count:

| Domain | Count |
| --- | ---: |
| common | 105 |
| fintech | 33 |
| intel | 22 |
| fin | 21 |
| scm | 18 |
| mfg | 15 |
| government | 13 |
| crm | 13 |
| ckm | 13 |
| agriculture | 12 |
| hcm | 12 |
| composition | 11 |
| transport | 10 |
| telecom | 10 |
| realestate | 10 |
| healthcare | 9 |
| pharma | 9 |
| legal | 8 |
| insurance | 8 |
| hospitality | 8 |

Africa-first and business-facing domains include agriculture, fintech, SACCO
flows, insurance, legal, hospitality, NGO, government, healthcare, pharma, SCM,
HCM, CRM, retail, transport, telecom, energy, finance, manufacturing, mining,
and real estate.

Common infrastructure capabilities to know:

- `common/obs` - observability
- `common/dcat` - data catalog
- `common/fflag` - feature flags
- `common/gql` - GraphQL
- `common/ussd` - USSD
- `common/docint` - document intelligence
- `common/pmin` - process mining

## Capability Package Shape

A mature package commonly includes:

```text
capabilities/<domain>/<code>/
  capability_contract.py
  cap_spec.md
  README.md
  models.py
  service.py
  api.py
  views.py
  blueprint.py
  semantic_model.json
  package_manifest.json
  domain/
  database/
  alembic/
  tests/
  docs/
```

Not every package currently has every file. Document the observed package
state, not an idealized one.

## Capability Contract

The executable contract is the main composition surface. It should provide:

- stable capability id and display name
- configuration defaults
- configuration schema where practical
- deterministic rules and rule evaluation
- UI route metadata
- theme tokens
- i18n and language metadata where user-facing
- streaming metadata where applicable
- health and package metadata

Generated apps expose capability helpers for rules, configuration, health,
theme, screens, languages, approvals, and streaming when the APG source declares
capabilities.

## Add A Capability

Use the current CLI scaffold:

```bash
apg capabilities scaffold <domain> <code> --name "Display Name" --json
```

Then fill in behavior, tests, docs, and package evidence.

## Inspect And Validate Capabilities

```bash
apg capabilities list
apg capabilities search <query>
apg capabilities manifest --stats
apg capabilities contracts --json
apg capabilities inspect <capability> --json
apg capabilities evaluate-rules <capability> --context-json '{"tenant_id":"demo"}' --json
apg capabilities validate-contracts --json
apg capabilities audit --json
apg capabilities implementation-audit --json
apg capabilities lifecycle-audit --json
```

Use `implementation-audit --strict` only when materialized baseline or
contract-only packages should block the gate.

## Standards

See:

- [Capability Standards](../capability_standards.md)
- [Capability Contracts](../capability_contracts.md)
- [Capability Development Guide](../capability_development_guide.md)
- [Capability Integration Guide](../capability_integration_guide.md)
