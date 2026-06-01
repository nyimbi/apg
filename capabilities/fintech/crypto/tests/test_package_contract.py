"""Executable Cryptocurrency Services capability package tests."""

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
	module = _load_module("contract_fintech_crypto", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_crypto"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "crypto_agent_workflow" in contract["provides"]
	assert "/fintech-crypto/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_crypto", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "crypto_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "crypto_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_crypto_lifecycle():
	service_module = _load_module("service_fintech_crypto", PACKAGE_DIR / "service.py")
	service = service_module.CryptocurrencyServicesService()

	asset = service.register_asset("asset-1", "tenant-test", "USDC", "stablecoin", "fintech_blockchain:polygon", "contract-ref", 6, "owner-1", "asset-evidence")
	account = service.open_custody_account("account-1", "tenant-test", "custodian-ref", "mpc", "policy-ref", "owner-1", "custody-evidence")
	balance = service.record_balance("balance-1", "tenant-test", account["id"], asset["id"], 1000000, 100000, "USD", "balance-evidence")
	order = service.create_order("order-1", "tenant-test", account["id"], asset["id"], "buy", "limit", 1000000, 100, "order-policy", "requester-1", "order-evidence")
	trade = service.record_trade("trade-1", "tenant-test", order["id"], "venue-ref", 100, 1000000, 10, "executed", "settlement-ref")
	transfer = service.request_transfer("transfer-1", "tenant-test", account["id"], asset["id"], "withdrawal", "destination-ref", 1000, "approval-ref", "transfer-evidence", "approved")
	screening = service.record_screening("screening-1", "tenant-test", transfer["id"], "transaction", "review", "screening-evidence", "reviewer-1")
	price = service.record_price("price-1", "tenant-test", asset["id"], "oracle", 100, "USD", "2026-06-01T10:00:00Z", "price-evidence")
	review = service.record_review("review-1", "tenant-test", trade["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_crypto_agent("agent-1", "tenant-test", "Crypto Agent", "codex", "trade_reviewer", "review trades")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert asset["symbol"] == "USDC"
	assert account["custody_model"] == "mpc"
	assert balance["valuation_currency"] == "USD"
	assert order["status"] == "requested"
	assert trade["status"] == "executed"
	assert transfer["status"] == "approved"
	assert screening["status"] == "review"
	assert price["source"] == "oracle"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["trade_count"] == 1
	assert summary["audit_event_count"] == 10


def test_service_guardrails_reject_invalid_crypto_actions():
	service_module = _load_module("guardrail_service_fintech_crypto", PACKAGE_DIR / "service.py")
	service = service_module.CryptocurrencyServicesService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_asset("asset", "", "BTC", "native_coin", "network", "", 8, "owner", "evidence")
	with pytest.raises(PermissionError, match="asset_type_not_supported"):
		service.register_asset("asset", "tenant-test", "BAD", "unknown", "network", "", 8, "owner", "evidence")
	with pytest.raises(PermissionError, match="asset_precision_invalid"):
		service.register_asset("asset", "tenant-test", "BAD", "native_coin", "network", "", -1, "owner", "evidence")
	asset = service.register_asset("asset-ok", "tenant-test", "USDC", "stablecoin", "network", "contract", 6, "owner", "evidence")
	with pytest.raises(PermissionError, match="custody_model_not_supported"):
		service.open_custody_account("account", "tenant-test", "provider", "unknown", "policy", "owner", "evidence")
	account = service.open_custody_account("account-ok", "tenant-test", "provider", "mpc", "policy", "owner", "evidence")
	with pytest.raises(PermissionError, match="balance_amount_invalid"):
		service.record_balance("balance", "tenant-test", account["id"], asset["id"], -1, 0, "USD", "evidence")
	with pytest.raises(PermissionError, match="limit_price_required"):
		service.create_order("order", "tenant-test", account["id"], asset["id"], "buy", "limit", 1, 0, "policy", "requester", "evidence")
	order = service.create_order("order-ok", "tenant-test", account["id"], asset["id"], "buy", "market", 1, 0, "policy", "requester", "evidence")
	with pytest.raises(PermissionError, match="trade_settlement_required"):
		service.record_trade("trade", "tenant-test", order["id"], "venue", 1, 1, 0, "executed", "")
	with pytest.raises(PermissionError, match="transfer_approval_required"):
		service.request_transfer("transfer", "tenant-test", account["id"], asset["id"], "withdrawal", "destination", 1, "", "evidence")
	with pytest.raises(PermissionError, match="screening_reviewer_required"):
		service.record_screening("screening", "tenant-test", order["id"], "transaction", "review", "evidence")
	with pytest.raises(PermissionError, match="price_source_not_supported"):
		service.record_price("price", "tenant-test", asset["id"], "unknown", 1, "USD", "2026-06-01T10:00:00Z", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", order["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="crypto_agent_runtime_not_supported"):
		service.register_crypto_agent("agent", "tenant-test", "Bad Agent", "unsupported", "trade_reviewer", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_crypto", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_crypto", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_crypto", PACKAGE_DIR / "app.py")

	asset = api.register_asset({"tenant_id": "tenant-api", "asset_id": "api-asset", "symbol": "BTC", "asset_type": "native_coin", "network_reference": "fintech_blockchain:bitcoin", "precision": 8, "owner_id": "owner", "evidence_reference": "evidence"})
	account = api.open_custody_account({"tenant_id": "tenant-api", "account_id": "api-account", "provider_reference": "provider", "custody_model": "hsm", "policy_reference": "policy", "owner_id": "owner", "evidence_reference": "evidence"})
	api.record_balance({"tenant_id": "tenant-api", "balance_id": "api-balance", "account_id": account["id"], "asset_id": asset["id"], "amount_minor": 1, "valuation_minor": 1, "valuation_currency": "USD", "evidence_reference": "evidence"})
	agent = api.register_crypto_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Crypto Agent", "runtime": "claude_code", "role": "portfolio_monitor"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.crypto_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "portfolio_monitor"
	assert dashboard["summary"]["asset_count"] == 1
	assert console["balances"][0]["id"] == "api-balance"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_crypto"]["screens"]["agents"]["route"] == "/fintech-crypto/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_crypto", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_crypto"]["streaming"]["processor"] == "bytewax"
