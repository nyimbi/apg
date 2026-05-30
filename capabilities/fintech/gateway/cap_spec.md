# Fintech Gateway Capability Runtime Spec

`fintech_gateway` is the executable payment orchestration capability for APG. It is composed through `capability_contract.py` and exposed through dependency-light runtime files.

## Runtime Files

- `capability_contract.py`: configuration, rules, routes, theme, streaming, provides, and requires.
- `service.py`: in-memory executable gateway lifecycle and guardrails.
- `api.py`: process-local API helper functions.
- `views.py`: framework-neutral screen models.
- `app.py`: semantic model, component manifest, and self-test.
- `tests/test_package_contract.py`: focused contract, lifecycle, guardrail, API, view, and app tests.

## Public Lifecycle

1. `onboard_merchant`
2. `connect_provider`
3. `tokenize_payment_method`
4. `create_payment_intent`
5. `assess_payment_risk`
6. `authorize_payment`
7. `capture_payment`
8. `refund_payment`
9. `ingest_webhook`
10. `record_settlement`
11. `open_dispute`
12. `resolve_dispute`
13. `register_gateway_agent`
14. `validate_gateway_agent_action`
15. `validate_batch`
16. `dashboard_summary`

## Composition Contract

Provides:

- `merchant_onboarding_lifecycle`
- `provider_connection_lifecycle`
- `payment_method_tokenization_workflow`
- `payment_intent_lifecycle`
- `payment_routing_workflow`
- `fraud_risk_review_workflow`
- `authorization_capture_workflow`
- `refund_lifecycle`
- `webhook_ingestion_workflow`
- `settlement_reconciliation_workflow`
- `payment_dispute_workflow`
- `gateway_agents`

Requires:

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `keym`
- `encr`
- `cash_management`
- `accounts_receivable`
- `customer_relationship_management`
- `business_intelligence`

## Streaming

All lifecycle events use:

- processor: `bytewax`
- stream: `apg.fintech.gateway.lifecycle`
- key: `tenant_id`

The rules `gateway_batch_requires_bytewax` and `gateway_event_requires_bytewax` deny unsupported stream processors.

## Agent Support

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `merchant_underwriter`
- `routing_reviewer`
- `fraud_reviewer`
- `settlement_reviewer`
- `dispute_reviewer`
- `provider_operations_reviewer`

Privileged gateway agent actions require recorded human approval.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fintech/gateway/__init__.py capabilities/fintech/gateway/capability_contract.py capabilities/fintech/gateway/service.py capabilities/fintech/gateway/api.py capabilities/fintech/gateway/views.py capabilities/fintech/gateway/app.py capabilities/fintech/gateway/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/gateway/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/gateway/app.py
./.venv/bin/apg capabilities inspect fintech_gateway --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/gateway --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/gateway --json
```
