# Digital Lending Capability Runtime Spec

`fintech_lending` is a package-backed APG capability with deterministic rules,
tenant-scoped runtime behavior, API helpers, UI/view metadata, visual theme
tokens, provider-neutral lending agents, and Bytewax lifecycle metadata.

## Contract

- Capability ID: `fintech_lending`
- Display name: `Digital Lending`
- Version: `1.1.0`
- Runtime target: `python`
- Stream processor: `bytewax`
- Stream: `apg.fintech.lending.lifecycle`

## Executable Lifecycle

1. Register a loan product with owner, type, currency, limits, term, rate, and
   repayment frequency.
2. Onboard a borrower with customer, KYC, country, income, and consent evidence.
3. Submit an application with affordability, bank statement, AML, fraud, and
   behavior evidence.
4. Record underwriting with score, decision, evidence, adverse-action reason
   when relevant, and human approval for final decisions.
5. Issue and accept an offer with APR, term, expiry, and borrower acceptance.
6. Record disbursement through a supported rail after funding and human
   approval evidence.
7. Schedule repayments and open collection cases when servicing requires them.
8. Register lending agents with supported runtimes and roles.

## Composition

Generated APG applications compose this capability through:

- `get_capability_contract()` for dependencies, configuration, rules, UI, theme,
  streaming, and provider-neutral agent metadata.
- `LendingService` for local executable behavior.
- `api.py` helpers for framework-free application calls.
- `views.py` helpers for route, dashboard, console, and rule models.
- `app.py` semantic model and self-test for publish/readiness evidence.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fintech/lending/__init__.py capabilities/fintech/lending/capability_contract.py capabilities/fintech/lending/models.py capabilities/fintech/lending/lending_runtime.py capabilities/fintech/lending/service.py capabilities/fintech/lending/api.py capabilities/fintech/lending/views.py capabilities/fintech/lending/app.py capabilities/fintech/lending/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/lending/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/lending/app.py
./.venv/bin/apg capabilities inspect fintech_lending --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/lending --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/lending --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/lending --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```
