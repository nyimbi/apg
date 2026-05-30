# Accounts Receivable Capability Runtime Spec

`arc_accounts_receivable` is the executable customer-to-cash capability for APG. It is composed through `capability_contract.py` and exposed through dependency-light runtime files.

## Runtime Files

- `capability_contract.py`: configuration, rules, routes, theme, streaming, provides, and requires.
- `service.py`: in-memory executable receivables lifecycle and guardrails.
- `api.py`: process-local API helper functions.
- `views.py`: framework-neutral screen models.
- `app.py`: semantic model, component manifest, and self-test.
- `tests/test_package_contract.py`: focused contract, lifecycle, guardrail, API, view, and app tests.

## Public Lifecycle

1. `create_customer`
2. `assess_credit`
3. `create_invoice`
4. `issue_invoice`
5. `record_payment`
6. `apply_cash`
7. `record_collection_activity`
8. `open_dispute`
9. `resolve_dispute`
10. `register_arc_agent`
11. `validate_agent_arc_action`
12. `validate_batch`
13. `aging_summary`
14. `dashboard_summary`

## Composition Contract

Provides:

- `customer_receivable_lifecycle`
- `credit_assessment_workflow`
- `invoice_lifecycle`
- `invoice_line_management`
- `payment_receipt_lifecycle`
- `cash_application_workflow`
- `collections_workflow`
- `dispute_resolution_workflow`
- `receivables_aging_service`
- `arc_agents`

Requires:

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `general_ledger`
- `cash_management`
- `document_management`
- `business_intelligence`
- `customer_relationship_management`

## Streaming

All lifecycle events use:

- processor: `bytewax`
- stream: `apg.fin.arc.lifecycle`
- key: `tenant_id`

The rules `arc_batch_requires_bytewax` and `arc_event_requires_bytewax` deny unsupported stream processors.

## Agent Support

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `credit_reviewer`
- `invoice_reviewer`
- `cash_application_reviewer`
- `collections_reviewer`
- `dispute_reviewer`
- `revenue_recognition_reviewer`

Privileged agent actions require recorded human approval.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fin/arc/accounts_receivable/__init__.py capabilities/fin/arc/accounts_receivable/capability_contract.py capabilities/fin/arc/accounts_receivable/service.py capabilities/fin/arc/accounts_receivable/api.py capabilities/fin/arc/accounts_receivable/views.py capabilities/fin/arc/accounts_receivable/app.py capabilities/fin/arc/accounts_receivable/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fin/arc/accounts_receivable/tests/test_package_contract.py
./.venv/bin/python capabilities/fin/arc/accounts_receivable/app.py
./.venv/bin/apg capabilities inspect arc_accounts_receivable --json
./.venv/bin/apg capabilities publish-plan capabilities/fin/arc/accounts_receivable --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fin/arc/accounts_receivable --json
```
