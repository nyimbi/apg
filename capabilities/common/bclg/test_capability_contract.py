"""Regression coverage for the BCLG executable capability contract."""

import pytest

from capabilities.common.bclg import register_capability
from capabilities.common.bclg.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.bclg.service import BclgService
from capabilities.common.bclg.views import dashboard_model


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-bclg", {"transactions": {"high_value_review_threshold": 250000}})

	assert contract["capability"] == "bclg"
	assert contract["configuration"]["tenant_id"] == "tenant-bclg"
	assert contract["configuration"]["transactions"]["high_value_review_threshold"] == 250000
	assert contract["configuration_schema"]["required"] == ["tenant_id", "ledgers", "transactions", "smart_contracts", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "ledgers", "transactions", "contracts", "keys", "audit", "compliance", "settings"}
	assert contract["theme"]["name"] == "bclg_ledger_ops"


def test_rule_engine_enforces_bclg_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_ledger", "ledger_owner_assigned": False, "key_custody_bound": False, "transaction_value": 200000, "transaction_review_recorded": False})
	transaction_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "submit_transaction", "signature_present": False, "key_custody_bound": True})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "ledger_requires_owner", "key_custody_required", "high_value_transaction_requires_review"}
	assert transaction_result["matched_rules"] == ["transaction_requires_signature"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "bclg"
	assert "keym" in registration["dependencies"]
	assert registration["ui_components"]["contracts"] == "/bclg/contracts"
	assert "bclg:manage_contracts" in registration["permissions"]


def test_service_registers_ledgers_commits_transactions_and_deploys_contracts():
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
	transaction = service.submit_transaction(
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
	)
	contract = service.deploy_contract(
		contract_id="contract-1",
		tenant_id="tenant-ledger",
		ledger_id="supply-chain-ledger",
		name="SupplierSettlement",
		version="1.0.0",
		artifact_hash="sha256:contract-artifact",
		reviewed_by="contract-reviewer",
		rollback_plan="disable SupplierSettlement v1",
	)
	summary = service.ledger_summary("tenant-ledger")
	model = dashboard_model(service, "tenant-ledger")

	assert ledger["participants"] == ["warehouse", "procurement", "finance"]
	assert custody["key_id"] == "key-001"
	assert transaction["status"] == "committed"
	assert transaction["review_status"] == "approved"
	assert len(transaction["transaction_hash"]) == 64
	assert len(transaction["block_hash"]) == 64
	assert contract["status"] == "deployed"
	assert len(contract["deployment_hash"]) == 64
	assert summary["ledger_count"] == 1
	assert summary["committed_transaction_count"] == 1
	assert summary["deployed_contract_count"] == 1
	assert model["summary"]["audit_event_count"] >= 4


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
	)
	approved = service.approve_transaction("high-value", reviewer="risk-reviewer")

	with pytest.raises(PermissionError, match="contract_review_required"):
		service.deploy_contract(
			contract_id="unreviewed-contract",
			tenant_id="tenant-ledger",
			ledger_id="payments-ledger",
			name="Escrow",
			version="1.0.0",
			artifact_hash="sha256:escrow",
			reviewed_by="",
			rollback_plan="disable Escrow",
		)

	assert high_value["status"] == "pending_review"
	assert high_value["review_status"] == "required"
	assert approved["status"] == "committed"
	assert len(approved["block_hash"]) == 64
