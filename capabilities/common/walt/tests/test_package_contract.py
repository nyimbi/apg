"""WALT package contract and deterministic wallet/payment runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.walt import api, views
from capabilities.common.walt.service import WaltService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_is_valid():
	module = _load_module("materialized_contract_walt", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "walt"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_walt", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "walt" in model["capabilities"]


def test_wallet_instrument_transaction_settlement_and_reconciliation_lifecycle_executes():
	service = WaltService()

	wallet = service.create_wallet(
		tenant_id="tenant-a",
		owner_ref="customer-1",
		currency="USD",
		ledger_ref="ledger://tenant-a/customer-1",
		compliance_policy_ref="policy://wallets/default",
		initial_balance="250.00",
		actor="operator-1",
	)
	instrument = service.register_instrument(
		tenant_id="tenant-a",
		wallet_id=wallet["id"],
		instrument_ref="card://token-source",
		instrument_type="card",
		token_ref="tok_123",
		encrypted=True,
		verified_by="vault-service",
	)
	transaction = service.authorize_transaction(
		tenant_id="tenant-a",
		wallet_id=wallet["id"],
		instrument_id=instrument["id"],
		amount="75.50",
		currency="USD",
		mfa_completed=True,
		risk_score=0.2,
		idempotency_key="txn-1",
		actor="cashier-1",
	)
	captured = service.capture_transaction("tenant-a", transaction["id"], "cashier-1")
	settlement = service.create_settlement_batch(
		tenant_id="tenant-a",
		transaction_ids=[captured["id"]],
		settlement_account_ref="settlement://merchant/primary",
		reconciliation_completed=True,
		created_by="settlement-ops",
	)
	reconciliation = service.record_reconciliation(
		tenant_id="tenant-a",
		settlement_batch_id=settlement["id"],
		reconciliation_ref="recon://batch/1",
		matched_count=1,
		exception_count=0,
		recorded_by="recon-ops",
	)
	summary = service.dashboard_summary("tenant-a")

	assert wallet["status"] == "active"
	assert wallet["balance"] == 250.0
	assert instrument["encrypted"] is True
	assert transaction["status"] == "authorized"
	assert captured["status"] == "captured"
	assert service.list_wallets("tenant-a")[0]["balance"] == 174.5
	assert settlement["status"] == "ready"
	assert reconciliation["status"] == "matched"
	assert summary["wallet_count"] == 1
	assert summary["instrument_count"] == 1
	assert summary["transaction_count"] == 1
	assert summary["settlement_batch_count"] == 1
	assert summary["reconciliation_count"] == 1


def test_wallet_guardrails_require_tenant_owner_encryption_mfa_reconciliation_and_balance():
	service = WaltService()

	try:
		service.create_wallet("", "owner-1", "USD", "ledger://x", "policy://x")
	except PermissionError as exc:
		assert str(exc) == "tenant_context_required"
	else:
		raise AssertionError("missing tenant was accepted")

	try:
		service.create_wallet("tenant-a", "", "USD", "ledger://x", "policy://x")
	except PermissionError as exc:
		assert str(exc) == "wallet_owner_required"
	else:
		raise AssertionError("missing wallet owner was accepted")

	wallet = service.create_wallet("tenant-a", "owner-1", "USD", "ledger://x", "policy://x", "100.00")

	try:
		service.register_instrument("tenant-a", wallet["id"], "card://raw", "card", "tok_1", False, "vault")
	except PermissionError as exc:
		assert str(exc) == "instrument_encryption_required"
	else:
		raise AssertionError("unencrypted instrument was accepted")

	instrument = service.register_instrument("tenant-a", wallet["id"], "card://ok", "card", "tok_1", True, "vault")

	try:
		service.authorize_transaction(
			tenant_id="tenant-a",
			wallet_id=wallet["id"],
			instrument_id=instrument["id"],
			amount="10001.00",
			currency="USD",
			mfa_completed=False,
			risk_score=0.1,
		)
	except PermissionError as exc:
		assert str(exc) == "high_value_mfa_required"
	else:
		raise AssertionError("high-value transaction without MFA was accepted")

	review = service.authorize_transaction(
		tenant_id="tenant-a",
		wallet_id=wallet["id"],
		instrument_id=instrument["id"],
		amount="10.00",
		currency="USD",
		mfa_completed=True,
		risk_score=0.95,
		risk_review_recorded=False,
	)
	assert review["status"] == "review_required"
	assert review["required_actions"] == ["review_transaction_risk"]

	try:
		service.authorize_transaction("tenant-a", wallet["id"], instrument["id"], "150.00", "USD", mfa_completed=True)
	except PermissionError as exc:
		assert str(exc) == "insufficient_wallet_balance"
	else:
		raise AssertionError("overdraft was accepted")

	tx = service.authorize_transaction("tenant-a", wallet["id"], instrument["id"], "20.00", "USD", mfa_completed=True)
	captured = service.capture_transaction("tenant-a", tx["id"])

	try:
		service.create_settlement_batch("tenant-a", [captured["id"]], "settlement://primary", False, "settlement-ops")
	except PermissionError as exc:
		assert str(exc) == "reconciliation_required"
	else:
		raise AssertionError("settlement without reconciliation evidence was accepted")


def test_api_and_view_models_expose_wallet_payment_surfaces():
	local_service = WaltService()
	api.SERVICE = local_service

	wallet = api.create_wallet({
		"tenant_id": "tenant-b",
		"owner_ref": "merchant-1",
		"currency": "KES",
		"ledger_ref": "ledger://merchant-1",
		"compliance_policy_ref": "policy://merchant",
		"initial_balance": "5000.00",
		"actor": "ops",
	})
	instrument = api.register_instrument({
		"tenant_id": "tenant-b",
		"wallet_id": wallet["id"],
		"instrument_ref": "mobile://merchant-1",
		"instrument_type": "mobile_money",
		"token_ref": "tok_mobile",
		"encrypted": True,
		"verified_by": "vault",
	})
	transaction = api.authorize_transaction({
		"tenant_id": "tenant-b",
		"wallet_id": wallet["id"],
		"instrument_id": instrument["id"],
		"amount": "250.00",
		"currency": "KES",
		"mfa_completed": True,
		"risk_score": 0.1,
	})
	captured = api.capture_transaction({
		"tenant_id": "tenant-b",
		"transaction_id": transaction["id"],
	})
	settlement = api.create_settlement_batch({
		"tenant_id": "tenant-b",
		"transaction_ids": [captured["id"]],
		"settlement_account_ref": "settlement://merchant-1",
		"reconciliation_completed": True,
	})
	api.record_reconciliation({
		"tenant_id": "tenant-b",
		"settlement_batch_id": settlement["id"],
		"reconciliation_ref": "recon://merchant-1",
		"matched_count": 1,
		"exception_count": 0,
	})

	status = api.capability_status("tenant-b")
	system = api.list_wallet_payments("tenant-b")
	dashboard = views.dashboard_model(local_service, "tenant-b")
	wallets = views.wallet_console_model(local_service, "tenant-b")
	transactions = views.transaction_console_model(local_service, "tenant-b")
	instruments = views.instrument_vault_model(local_service, "tenant-b")
	settlements = views.settlement_center_model(local_service, "tenant-b")
	reconciliations = views.reconciliation_queue_model(local_service, "tenant-b")
	risk = views.risk_model(local_service, "tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["wallet_count"] == 1
	assert system["summary"]["settlement_batch_count"] == 1
	assert dashboard["summary"]["total_balance"] == 4750.0
	assert wallets["wallets"][0]["currency"] == "KES"
	assert transactions["transactions"][0]["status"] == "settled"
	assert instruments["instrument_types"]
	assert settlements["settlement_batches"][0]["status"] == "reconciled"
	assert reconciliations["reconciliations"][0]["status"] == "matched"
	assert risk["review_required_transactions"] == []
	assert settings["configuration"]["tenant_id"] == "tenant-b"
