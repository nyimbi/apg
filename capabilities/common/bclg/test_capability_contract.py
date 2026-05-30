"""Regression coverage for the BCLG executable capability contract."""

import pytest

from capabilities.common.bclg import api, register_capability
from capabilities.common.bclg import views
from capabilities.common.bclg.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.bclg.service import BclgService


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-bclg", {"transactions": {"high_value_review_threshold": 250000}})

	assert contract["capability"] == "bclg"
	assert contract["configuration"]["tenant_id"] == "tenant-bclg"
	assert contract["configuration"]["transactions"]["high_value_review_threshold"] == 250000
	assert contract["configuration_schema"]["required"] == ["tenant_id", "ledgers", "transactions", "smart_contracts", "ledger_agents", "governance", "observability", "adapters", "ui", "theme"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["configuration"]["ledger_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert set(contract["provides"]) >= {"ledger_registry", "transaction_governance", "ledger_agents"}
	assert contract["requires"] == ["encr", "keym", "comp"]
	assert len(contract["rule_engine"]["rules"]) >= 15
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"ledgers",
		"transactions",
		"transaction_reviews",
		"contracts",
		"contract_reviews",
		"keys",
		"ledger_agents",
		"audit",
		"analytics",
		"compliance",
		"settings",
	}
	assert contract["theme"]["name"] == "bclg_ledger_ops"
	assert "transaction_review_queue" in contract["theme"]["components"]
	assert "contract_review_queue" in contract["theme"]["components"]


def test_rule_engine_enforces_bclg_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_ledger",
		"ledger_owner_assigned": False,
		"key_custody_bound": False,
		"transaction_value": 200000,
		"transaction_review_recorded": False,
	})
	transaction_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "submit_transaction",
		"signature_present": False,
		"key_custody_bound": True,
	})
	transaction_review_result = evaluate_capability_rules({
		"operation": "approve_transaction_review",
		"reviewer_same_as_submitter": True,
	})
	contract_review_result = evaluate_capability_rules({
		"operation": "approve_contract_deployment",
		"reviewer_same_as_requester": True,
	})
	batch_result = evaluate_capability_rules({
		"requested_operation": "batch_ledger_mutation",
		"event_stream": "memory",
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"ledger_requires_owner",
		"key_custody_required",
		"high_value_transaction_requires_review",
	}
	assert transaction_result["matched_rules"] == ["transaction_requires_signature"]
	assert transaction_review_result["matched_rules"] == ["transaction_review_requires_independent_reviewer"]
	assert contract_review_result["matched_rules"] == ["contract_deployment_review_requires_independent_reviewer"]
	assert batch_result["matched_rules"] == ["batch_ledger_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "bclg"
	assert "keym" in registration["dependencies"]
	assert registration["ui_components"]["contracts"] == "/bclg/contracts"
	assert registration["ui_components"]["ledger_agents"] == "/bclg/agents"
	assert registration["ui_components"]["transaction_reviews"] == "/bclg/transactions/reviews"
	assert registration["ui_components"]["contract_reviews"] == "/bclg/contracts/reviews"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "bclg:manage_contracts" in registration["permissions"]
	assert "bclg:review_transactions" in registration["permissions"]
	assert "bclg:review_contracts" in registration["permissions"]


def test_service_runs_ledger_transaction_review_contract_and_audit_lifecycle():
	service = BclgService()
	ledger = service.register_ledger(
		ledger_id="supply-chain-ledger",
		tenant_id="tenant-ledger",
		name="Supply Chain Ledger",
		owner="ledger-owner",
		consensus_profile="proof-of-authority",
		network_policy="tenant-private",
		participants=["warehouse", "procurement", "finance"],
	)
	custody = service.bind_key_custody(
		binding_id="custody-1",
		tenant_id="tenant-ledger",
		ledger_id="supply-chain-ledger",
		key_id="key-001",
		custodian="key-manager",
	)
	standard = service.submit_transaction(
		transaction_id="txn-1",
		tenant_id="tenant-ledger",
		ledger_id="supply-chain-ledger",
		from_account="warehouse",
		to_account="supplier",
		amount=2500,
		asset="USD",
		signature="sig:warehouse:txn-1",
		key_custody_id="custody-1",
		compliance_tags=["invoice", "goods-received"],
		actor="ledger-operator",
	)
	high_value = service.submit_transaction(
		transaction_id="txn-high",
		tenant_id="tenant-ledger",
		ledger_id="supply-chain-ledger",
		from_account="treasury",
		to_account="supplier",
		amount=250000,
		asset="USD",
		signature="sig:treasury:txn-high",
		key_custody_id="custody-1",
		actor="treasury-operator",
	)
	review_request = service.request_transaction_review(
		review_id="review-high",
		tenant_id="tenant-ledger",
		transaction_id="txn-high",
		requested_by="treasury-operator",
		justification="High-value supplier settlement.",
	)
	approved_high_value = service.decide_transaction_review(
		review_id=review_request["id"],
		tenant_id="tenant-ledger",
		reviewer="risk-reviewer",
		decision="approved",
		notes="Approved against invoice and risk policy.",
	)
	contract_review_request = service.request_contract_deployment_approval(
		approval_id="contract-approval-1",
		tenant_id="tenant-ledger",
		ledger_id="supply-chain-ledger",
		name="SupplierSettlement",
		version="1.0.0",
		artifact_hash="sha256:contract-artifact",
		requested_by="contract-owner",
		rollback_plan="disable SupplierSettlement v1",
	)
	contract_approval = service.decide_contract_deployment_approval(
		approval_id=contract_review_request["id"],
		tenant_id="tenant-ledger",
		reviewer="contract-reviewer",
		decision="approved",
		notes="Artifact hash and rollback plan verified.",
	)
	contract = service.deploy_contract(
		contract_id="contract-1",
		tenant_id="tenant-ledger",
		ledger_id="supply-chain-ledger",
		name="SupplierSettlement",
		version="1.0.0",
		artifact_hash="sha256:contract-artifact",
		rollback_plan="disable SupplierSettlement v1",
		approval_id=contract_approval["id"],
	)
	summary = service.ledger_summary("tenant-ledger")
	model = views.dashboard_model(service, "tenant-ledger")

	assert ledger["participants"] == ["warehouse", "procurement", "finance"]
	assert custody["key_id"] == "key-001"
	assert standard["status"] == "committed"
	assert len(standard["transaction_hash"]) == 64
	assert len(standard["block_hash"]) == 64
	assert high_value["status"] == "pending_review"
	assert approved_high_value["status"] == "committed"
	assert approved_high_value["review_id"] == "review-high"
	assert len(approved_high_value["block_hash"]) == 64
	assert contract["status"] == "deployed"
	assert contract["approval_id"] == "contract-approval-1"
	assert len(contract["deployment_hash"]) == 64
	assert summary["ledger_count"] == 1
	assert summary["committed_transaction_count"] == 2
	assert summary["transaction_review_count"] == 1
	assert summary["deployed_contract_count"] == 1
	assert summary["contract_review_count"] == 1
	assert summary["ledger_agent_count"] == 0
	assert model["summary"]["audit_event_count"] >= 8


def test_service_enforces_ledger_transaction_and_contract_guardrails():
	service = BclgService()

	with pytest.raises(PermissionError, match="ledger_owner_required"):
		service.register_ledger(
			ledger_id="missing-owner",
			tenant_id="tenant-ledger",
			name="Missing Owner",
			owner="",
			consensus_profile="proof-of-authority",
			network_policy="tenant-private",
		)

	service.register_ledger(
		ledger_id="payments-ledger",
		tenant_id="tenant-ledger",
		name="Payments Ledger",
		owner="ledger-owner",
		consensus_profile="proof-of-authority",
		network_policy="tenant-private",
	)

	with pytest.raises(PermissionError, match="transaction_signature_required"):
		service.submit_transaction(
			transaction_id="unsigned",
			tenant_id="tenant-ledger",
			ledger_id="payments-ledger",
			from_account="buyer",
			to_account="seller",
			amount=100,
			signature="",
			key_custody_id="missing",
		)

	with pytest.raises(PermissionError, match="key_custody_required"):
		service.submit_transaction(
			transaction_id="uncustodied",
			tenant_id="tenant-ledger",
			ledger_id="payments-ledger",
			from_account="buyer",
			to_account="seller",
			amount=100,
			signature="sig:buyer:uncustodied",
			key_custody_id="missing",
		)

	service.bind_key_custody(
		binding_id="custody-1",
		tenant_id="tenant-ledger",
		ledger_id="payments-ledger",
		key_id="key-001",
		custodian="key-manager",
	)
	high_value = service.submit_transaction(
		transaction_id="high-value",
		tenant_id="tenant-ledger",
		ledger_id="payments-ledger",
		from_account="buyer",
		to_account="seller",
		amount=250000,
		signature="sig:buyer:high-value",
		key_custody_id="custody-1",
		transaction_review_recorded=True,
		actor="buyer",
	)
	review_request = service.request_transaction_review(
		review_id="review-rejected",
		tenant_id="tenant-ledger",
		transaction_id="high-value",
		requested_by="buyer",
		justification="High-value buyer settlement.",
	)
	with pytest.raises(ValueError, match="transaction_review_already_pending"):
		service.request_transaction_review(
			review_id="review-second",
			tenant_id="tenant-ledger",
			transaction_id="high-value",
			requested_by="buyer",
			justification="Duplicate pending review.",
		)
	with pytest.raises(PermissionError, match="independent_transaction_reviewer_required"):
		service.decide_transaction_review(
			review_id=review_request["id"],
			tenant_id="tenant-ledger",
			reviewer="buyer",
			decision="approved",
			notes="Self approved.",
		)
	with pytest.raises(ValueError, match="transaction_review_notes_required"):
		service.decide_transaction_review(
			review_id=review_request["id"],
			tenant_id="tenant-ledger",
			reviewer="risk-reviewer",
			decision="approved",
			notes="",
		)
	rejected = service.decide_transaction_review(
		review_id=review_request["id"],
		tenant_id="tenant-ledger",
		reviewer="risk-reviewer",
		decision="rejected",
		notes="Counterparty limit exceeded.",
	)
	with pytest.raises(ValueError, match="transaction_review_already_decided"):
		service.decide_transaction_review(
			review_id=review_request["id"],
			tenant_id="tenant-ledger",
			reviewer="risk-reviewer",
			decision="approved",
			notes="Changed after rejection.",
		)

	with pytest.raises(PermissionError, match="contract_review_required"):
		service.deploy_contract(
			contract_id="unreviewed-contract",
			tenant_id="tenant-ledger",
			ledger_id="payments-ledger",
			name="Escrow",
			version="1.0.0",
			artifact_hash="sha256:escrow",
			reviewed_by="contract-reviewer",
			rollback_plan="disable Escrow",
		)

	contract_review_request = service.request_contract_deployment_approval(
		approval_id="contract-review-rejected",
		tenant_id="tenant-ledger",
		ledger_id="payments-ledger",
		name="Escrow",
		version="1.0.0",
		artifact_hash="sha256:escrow",
		requested_by="contract-owner",
		rollback_plan="disable Escrow",
	)
	with pytest.raises(PermissionError, match="independent_contract_reviewer_required"):
		service.decide_contract_deployment_approval(
			approval_id=contract_review_request["id"],
			tenant_id="tenant-ledger",
			reviewer="contract-owner",
			decision="approved",
			notes="Self approved.",
		)
	rejected_contract = service.decide_contract_deployment_approval(
		approval_id=contract_review_request["id"],
		tenant_id="tenant-ledger",
		reviewer="contract-reviewer",
		decision="rejected",
		notes="Rollback plan not operationally tested.",
	)
	with pytest.raises(PermissionError, match="contract_deployment_approval_not_approved"):
		service.deploy_contract(
			contract_id="rejected-contract",
			tenant_id="tenant-ledger",
			ledger_id="payments-ledger",
			name="Escrow",
			version="1.0.0",
			artifact_hash="sha256:escrow",
			rollback_plan="disable Escrow",
			approval_id=rejected_contract["id"],
		)

	assert high_value["status"] == "pending_review"
	assert high_value["review_status"] == "required"
	assert high_value["block_hash"] is None
	assert rejected["status"] == "rejected"
	assert rejected["block_hash"] is None


def test_service_keeps_duplicate_ids_isolated_by_tenant():
	service = BclgService()
	for tenant_id in ["tenant-a", "tenant-b"]:
		service.register_ledger(
			ledger_id="same-ledger",
			tenant_id=tenant_id,
			name=f"Ledger {tenant_id}",
			owner="ledger-owner",
			consensus_profile="proof-of-authority",
			network_policy="tenant-private",
		)
		service.bind_key_custody(
			binding_id="same-custody",
			tenant_id=tenant_id,
			ledger_id="same-ledger",
			key_id=f"key-{tenant_id}",
			custodian="key-manager",
		)
		service.submit_transaction(
			transaction_id="same-transaction",
			tenant_id=tenant_id,
			ledger_id="same-ledger",
			from_account="a",
			to_account="b",
			amount=10,
			signature=f"sig:{tenant_id}",
			key_custody_id="same-custody",
		)

	assert service.list_key_custody("tenant-a")[0]["key_id"] == "key-tenant-a"
	assert service.list_key_custody("tenant-b")[0]["key_id"] == "key-tenant-b"
	assert service.list_transactions("tenant-a")[0]["tenant_id"] == "tenant-a"
	assert service.list_transactions("tenant-b")[0]["tenant_id"] == "tenant-b"
	with pytest.raises(ValueError, match="ledger already exists"):
		service.register_ledger(
			ledger_id="same-ledger",
			tenant_id="tenant-a",
			name="Duplicate",
			owner="ledger-owner",
			consensus_profile="proof-of-authority",
			network_policy="tenant-private",
		)


def test_service_registers_ledger_agents_and_enforces_bytewax_guardrail():
	service = BclgService()
	agent = service.register_ledger_agent(
		agent_id="ledger-agent-1",
		tenant_id="tenant-ledger-agent",
		name="Transaction Review Assistant",
		runtime="claude-code",
		role="transaction-reviewer",
		scope="summarize high-value transaction review evidence",
		contribution_disclosed=True,
		policy_ref="bclg-agent-policy",
	)
	batch = service.validate_batch_ledger_mutation(
		tenant_id="tenant-ledger-agent",
		event_stream="bytewax",
		mutation_count=2,
	)
	dashboard = views.dashboard_model(service, "tenant-ledger-agent")
	agents = views.ledger_agent_model(service, "tenant-ledger-agent")
	analytics = views.analytics_model(service, "tenant-ledger-agent")
	settings = views.settings_model("tenant-ledger-agent")

	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "transaction_reviewer"
	assert batch["accepted"] is True
	assert dashboard["ledger_agents"][0]["id"] == "ledger-agent-1"
	assert agents["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert analytics["summary"]["ledger_agent_count"] == 1
	assert settings["streaming"]["processor"] == "bytewax"

	with pytest.raises(PermissionError, match="ledger_agent_runtime_not_supported"):
		service.register_ledger_agent(
			agent_id="bad-runtime",
			tenant_id="tenant-ledger-agent",
			name="Bad Runtime",
			runtime="unsupported",
			role="ledger_reviewer",
			scope="ledger review",
		)

	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch_ledger_mutation(
			tenant_id="tenant-ledger-agent",
			event_stream="memory",
			mutation_count=1,
		)


def test_api_helpers_and_view_models_expose_bclg_lifecycle():
	tenant_id = "tenant-api-bclg"
	ledger = api.register_ledger({
		"id": "api-ledger",
		"tenant_id": tenant_id,
		"name": "API Ledger",
		"owner": "api-owner",
		"consensus_profile": "proof-of-authority",
		"network_policy": "tenant-private",
		"fork_monitoring_enabled": "true",
	})
	api.bind_key_custody({
		"id": "api-custody",
		"tenant_id": tenant_id,
		"ledger_id": ledger["id"],
		"key_id": "api-key",
		"custodian": "api-custodian",
	})
	transaction = api.submit_transaction({
		"id": "api-transaction",
		"tenant_id": tenant_id,
		"ledger_id": ledger["id"],
		"from_account": "treasury",
		"to_account": "supplier",
		"amount": 250000,
		"signature": "sig:api-transaction",
		"key_custody_id": "api-custody",
		"transaction_review_recorded": "true",
		"actor": "api-operator",
	})
	review_request = api.request_transaction_review({
		"id": "api-review",
		"tenant_id": tenant_id,
		"transaction_id": transaction["id"],
		"requested_by": "api-operator",
		"justification": "High-value API transaction.",
	})
	approved_transaction = api.decide_transaction_review({
		"id": review_request["id"],
		"tenant_id": tenant_id,
		"reviewer": "api-risk-reviewer",
		"decision": "approved",
		"notes": "Approved for API path.",
	})
	contract_review_request = api.request_contract_deployment_approval({
		"id": "api-contract-review",
		"tenant_id": tenant_id,
		"ledger_id": ledger["id"],
		"name": "APIEscrow",
		"version": "1.0.0",
		"artifact_hash": "sha256:api-escrow",
		"requested_by": "api-contract-owner",
		"rollback_plan": "disable APIEscrow",
	})
	contract_approval = api.decide_contract_deployment_approval({
		"id": contract_review_request["id"],
		"tenant_id": tenant_id,
		"reviewer": "api-contract-reviewer",
		"decision": "approved",
		"notes": "Approved for API deployment.",
	})
	contract = api.deploy_contract({
		"id": "api-contract",
		"tenant_id": tenant_id,
		"ledger_id": ledger["id"],
		"name": "APIEscrow",
		"version": "1.0.0",
		"artifact_hash": "sha256:api-escrow",
		"rollback_plan": "disable APIEscrow",
		"approval_id": contract_approval["id"],
	})
	dashboard = views.dashboard_model(tenant_id=tenant_id)
	transaction_reviews = views.transaction_review_model(tenant_id=tenant_id)
	contract_reviews = views.contract_review_model(tenant_id=tenant_id)
	agent = api.register_ledger_agent({
		"id": "api-ledger-agent",
		"tenant_id": tenant_id,
		"name": "API Ledger Agent",
		"runtime": "opencode",
		"role": "ledger_reviewer",
		"scope": "ledger governance review",
		"contribution_disclosed": True,
	})
	batch = api.validate_batch_ledger_mutation({
		"tenant_id": tenant_id,
		"event_stream": "bytewax",
		"mutation_count": 1,
	})

	assert transaction["status"] == "pending_review"
	assert approved_transaction["status"] == "committed"
	assert contract["status"] == "deployed"
	assert api.capability_status(tenant_id)["transaction_review_count"] == 1
	assert api.capability_status(tenant_id)["ledger_agent_count"] == 1
	assert api.capability_status(tenant_id)["streaming"]["processor"] == "bytewax"
	assert dashboard["summary"]["deployed_contract_count"] == 1
	assert dashboard["transaction_reviews"][0]["id"] == "api-review"
	assert transaction_reviews["decided_reviews"][0]["id"] == "api-review"
	assert contract_reviews["decided_approvals"][0]["id"] == "api-contract-review"
	assert agent["runtime"] == "opencode"
	assert api.list_ledger_agents(tenant_id)[0]["id"] == "api-ledger-agent"
	assert batch["accepted"] is True
