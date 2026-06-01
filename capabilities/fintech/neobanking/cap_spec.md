# Digital Neobanking Capability Runtime Spec

`fintech_neobanking` is a package-backed APG capability with deterministic
rules, tenant-scoped runtime behavior, API helpers, UI/view metadata, visual
theme tokens, provider-neutral neobanking agents, and Bytewax lifecycle
metadata.

## Contract

- Capability ID: `fintech_neobanking`
- Display name: `Digital Neobanking`
- Version: `1.1.0`
- Runtime target: `python`
- Stream processor: `bytewax`
- Stream: `apg.fintech.neobanking.lifecycle`

## Executable Lifecycle

1. Register a bank program with owner, country, currency, and settlement
   evidence.
2. Onboard a digital customer with customer, KYC, AML, fraud, country, and
   consent evidence.
3. Open a deposit account with program, customer, type, currency, and opening
   balance controls.
4. Link supported payment rails to an account.
5. Post account transactions with risk references, direction calculation,
   balance updates, and high-impact approval gates.
6. Create savings pots and issue account statements.
7. Open service cases with customer, account, reason, reviewer, and evidence.
8. Register neobanking agents with supported runtimes and roles.

## Composition

Generated APG applications compose this capability through:

- `get_capability_contract()` for dependencies, configuration, rules, UI, theme,
  streaming, and agent metadata.
- `NeobankingService` for local executable behavior.
- `api.py` helpers for framework-free application calls.
- `views.py` helpers for route, dashboard, console, and rule models.
- `app.py` semantic model and self-test for publish/readiness evidence.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fintech/neobanking/__init__.py capabilities/fintech/neobanking/capability_contract.py capabilities/fintech/neobanking/models.py capabilities/fintech/neobanking/neobanking_runtime.py capabilities/fintech/neobanking/service.py capabilities/fintech/neobanking/api.py capabilities/fintech/neobanking/views.py capabilities/fintech/neobanking/app.py capabilities/fintech/neobanking/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/neobanking/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/neobanking/app.py
./.venv/bin/apg capabilities inspect fintech_neobanking --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/neobanking --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/neobanking --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/neobanking --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```
