# Know Your Customer Capability Specification Pointer

The active APG Know Your Customer specification is maintained in
`SPECIFICATION.md`.

## Runtime Summary

`fintech_kyc` is the dependency-light customer due-diligence capability for
generated APG applications. It owns tenant-scoped identity profiles, document
evidence, sanctions/PEP/adverse-media screening, risk scoring, verification
decisions, and provider-neutral KYC-agent evidence.

It composes with `fintech_payments` and `fintech_wallets` so onboarding
decisions can gate money movement and wallet activation without binding KYC to
live provider APIs.

## Composition Contract

Provides:

- `customer_identity_lifecycle`
- `document_verification_workflow`
- `sanctions_pep_screening`
- `kyc_risk_scoring`
- `customer_due_diligence`
- `enhanced_due_diligence`
- `kyc_agent_workflow`

Requires:

- `auth`
- `audl`
- `cons`
- `ntfy`
- `biop`
- `cvsn`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`

All lifecycle batches and events use Bytewax metadata through
`apg.fintech.kyc.lifecycle`.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fintech/kyc/__init__.py capabilities/fintech/kyc/capability_contract.py capabilities/fintech/kyc/models.py capabilities/fintech/kyc/kyc_runtime.py capabilities/fintech/kyc/service.py capabilities/fintech/kyc/api.py capabilities/fintech/kyc/views.py capabilities/fintech/kyc/app.py capabilities/fintech/kyc/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/kyc/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/kyc/app.py
./.venv/bin/apg capabilities inspect fintech_kyc --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/kyc --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/kyc --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/kyc --json
```
