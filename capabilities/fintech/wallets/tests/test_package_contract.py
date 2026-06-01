"""Executable Digital Wallets capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_fintech_wallets", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_wallets"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "wallet_agent_workflow" in contract["provides"]
	assert "/fintech-wallets/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_same_wallet():
	module = _load_module("rules_fintech_wallets", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "wallet_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "transfer", "same_wallet": True})["matched_rules"] == ["transfer_requires_distinct_wallets"]


def test_service_executes_wallet_lifecycle():
	service_module = _load_module("service_fintech_wallets", PACKAGE_DIR / "service.py")
	service = service_module.DigitalWalletsService()

	source = service.open_wallet("wallet-a", "tenant-test", "customer-a", "consumer", "KES", 1000)
	target = service.open_wallet("wallet-b", "tenant-test", "merchant-b", "merchant", "KES", 0)
	instrument = service.register_instrument("inst-a", "tenant-test", source["id"], "mobile_money", "vault://mpesa/customer-a", "ops-1")
	credit = service.credit_wallet("credit-1", "tenant-test", source["id"], 250, "cash-in", "idem-credit-1")
	transfer = service.transfer("transfer-1", "tenant-test", source["id"], target["id"], 125)
	hold = service.place_hold("hold-1", "tenant-test", target["id"], 50, "merchant reserve")
	release = service.release_hold("hold-2", "tenant-test", target["id"], 20, "partial release")
	agent = service.register_wallet_agent("agent-1", "tenant-test", "Limits Agent", "codex", "limits_reviewer", "review transfer limits")
	batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert instrument["status"] == "active"
	assert credit["status"] == "posted"
	assert transfer["status"] == "posted"
	assert hold["status"] == "placed"
	assert release["status"] == "released"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["wallet_count"] == 2
	assert summary["total_balance"] == "1250"
	assert summary["total_available"] == "1220"
	assert summary["audit_event_count"] == 10


def test_service_guardrails_reject_invalid_wallet_actions():
	service_module = _load_module("guardrail_service_fintech_wallets", PACKAGE_DIR / "service.py")
	service = service_module.DigitalWalletsService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.open_wallet("wallet", "", "owner", "consumer", "KES")
	with pytest.raises(PermissionError, match="wallet_type_not_supported"):
		service.open_wallet("wallet", "tenant-test", "owner", "unsupported", "KES")
	wallet = service.open_wallet("wallet", "tenant-test", "owner", "consumer", "KES", 100)
	with pytest.raises(PermissionError, match="instrument_verification_required"):
		service.register_instrument("inst", "tenant-test", wallet["id"], "mobile_money", "vault://x", "")
	with pytest.raises(PermissionError, match="insufficient_available_balance"):
		service.debit_wallet("debit", "tenant-test", wallet["id"], 101, "cash-out", "idem-1")
	service.open_wallet("wallet-b", "tenant-test", "owner-b", "consumer", "KES", 0)
	with pytest.raises(PermissionError, match="distinct_wallets_required"):
		service.transfer("same", "tenant-test", wallet["id"], wallet["id"], 1)
	with pytest.raises(PermissionError, match="wallet_limit_review_required"):
		service.transfer("large", "tenant-test", wallet["id"], "wallet-b", 3000)
	usd_wallet = service.open_wallet("wallet-usd", "tenant-test", "owner-usd", "consumer", "USD", 0)
	with pytest.raises(PermissionError, match="wallet_currency_mismatch"):
		service.transfer("fx", "tenant-test", wallet["id"], usd_wallet["id"], 1)
	with pytest.raises(PermissionError, match="hold_amount_positive_required"):
		service.place_hold("negative-hold", "tenant-test", wallet["id"], -1, "invalid")
	service.place_hold("valid-hold", "tenant-test", wallet["id"], 10, "reserve")
	with pytest.raises(PermissionError, match="hold_release_amount_positive_required"):
		service.release_hold("negative-release", "tenant-test", wallet["id"], -1, "invalid")
	with pytest.raises(PermissionError, match="hold_release_exceeds_held_balance"):
		service.release_hold("over-release", "tenant-test", wallet["id"], 11, "invalid")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_wallets", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_wallets", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_wallets", PACKAGE_DIR / "app.py")

	wallet = api.open_wallet({"tenant_id": "tenant-api", "wallet_id": "api-wallet", "owner_reference": "customer-api", "wallet_type": "consumer", "currency": "KES", "initial_balance": 100})
	api.register_instrument({"tenant_id": "tenant-api", "instrument_id": "api-inst", "wallet_id": wallet["id"], "instrument_type": "mobile_money", "token_reference": "vault://api", "verified_by": "ops"})
	api.credit_wallet({"tenant_id": "tenant-api", "entry_id": "api-credit", "wallet_id": wallet["id"], "amount": 25, "idempotency_key": "api-credit"})
	agent = api.register_wallet_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Wallet Agent", "runtime": "claude_code", "role": "wallet_ops_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.wallet_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "wallet_ops_reviewer"
	assert dashboard["summary"]["wallet_count"] == 1
	assert console["wallets"][0]["id"] == "api-wallet"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_wallets"]["screens"]["agents"]["route"] == "/fintech-wallets/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_wallets", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_wallets"]["streaming"]["processor"] == "bytewax"
