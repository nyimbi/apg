"""Executable Decentralized Finance capability package tests."""

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
	module = _load_module("contract_fintech_defi", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_defi"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "defi_agent_workflow" in contract["provides"]
	assert "/fintech-defi/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_defi", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "defi_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "defi_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_defi_lifecycle():
	service_module = _load_module("service_fintech_defi", PACKAGE_DIR / "service.py")
	service = service_module.DecentralizedFinanceService()

	protocol = service.register_protocol("protocol-1", "tenant-test", "lending_pool", "fintech_blockchain:polygon", "aave-v3", "owner-1", "protocol-evidence", "medium")
	position = service.open_position("position-1", "tenant-test", protocol["id"], "wallet-ref", "USDC/ETH", "supply", 1000000, 0, 15000, "position-evidence")
	action = service.record_action("action-1", "tenant-test", protocol["id"], position["id"], "deposit", 1000000, "requester-1", "approval-ref", "action-evidence", "approved")
	strategy = service.register_yield_strategy("strategy-1", "tenant-test", protocol["id"], "strategy-ref", 750, "medium", "owner-1", "strategy-evidence")
	reward = service.record_reward("reward-1", "tenant-test", position["id"], "interest", "USDC", 1200, "reward-evidence")
	vote = service.record_governance_vote("proposal-1", "tenant-test", protocol["id"], "proposal-ref", "for", "voter-1", "vote-evidence")
	risk = service.record_risk_assessment("risk-1", "tenant-test", protocol["id"], "medium", "reviewer-1", "risk-evidence")
	review = service.record_review("review-1", "tenant-test", action["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_defi_agent("agent-1", "tenant-test", "DeFi Agent", "codex", "protocol_monitor", "monitor protocols")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert protocol["protocol_type"] == "lending_pool"
	assert position["position_type"] == "supply"
	assert action["status"] == "approved"
	assert strategy["max_risk_tier"] == "medium"
	assert reward["reward_type"] == "interest"
	assert vote["vote_choice"] == "for"
	assert risk["risk_tier"] == "medium"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["position_count"] == 1
	assert summary["audit_event_count"] == 9


def test_service_guardrails_reject_invalid_defi_actions():
	service_module = _load_module("guardrail_service_fintech_defi", PACKAGE_DIR / "service.py")
	service = service_module.DecentralizedFinanceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_protocol("protocol", "", "lending_pool", "network", "protocol", "owner", "evidence", "low")
	with pytest.raises(PermissionError, match="protocol_type_not_supported"):
		service.register_protocol("protocol", "tenant-test", "unknown", "network", "protocol", "owner", "evidence", "low")
	with pytest.raises(PermissionError, match="risk_tier_not_supported"):
		service.register_protocol("protocol", "tenant-test", "lending_pool", "network", "protocol", "owner", "evidence", "unknown")
	protocol = service.register_protocol("protocol-ok", "tenant-test", "lending_pool", "network", "protocol", "owner", "evidence", "low")
	with pytest.raises(PermissionError, match="position_type_not_supported"):
		service.open_position("position", "tenant-test", protocol["id"], "account", "USDC/ETH", "unknown", 1, 0, 10000, "evidence")
	with pytest.raises(PermissionError, match="position_amount_invalid"):
		service.open_position("position", "tenant-test", protocol["id"], "account", "USDC/ETH", "supply", 0, 0, 10000, "evidence")
	position = service.open_position("position-ok", "tenant-test", protocol["id"], "account", "USDC/ETH", "supply", 1, 0, 10000, "evidence")
	other_protocol = service.register_protocol("protocol-other", "tenant-test", "dex", "network", "protocol-other", "owner", "evidence", "low")
	with pytest.raises(PermissionError, match="position_protocol_mismatch"):
		service.record_action("action", "tenant-test", other_protocol["id"], position["id"], "deposit", 1, "requester", "approval", "evidence")
	with pytest.raises(PermissionError, match="action_type_not_supported"):
		service.record_action("action", "tenant-test", protocol["id"], position["id"], "unknown", 1, "requester", "approval", "evidence")
	with pytest.raises(PermissionError, match="action_approval_required"):
		service.record_action("action", "tenant-test", protocol["id"], position["id"], "deposit", 1, "requester", "", "evidence")
	with pytest.raises(PermissionError, match="target_apy_invalid"):
		service.register_yield_strategy("strategy", "tenant-test", protocol["id"], "strategy", -1, "low", "owner", "evidence")
	with pytest.raises(PermissionError, match="reward_type_not_supported"):
		service.record_reward("reward", "tenant-test", position["id"], "unknown", "USDC", 1, "evidence")
	with pytest.raises(PermissionError, match="vote_choice_not_supported"):
		service.record_governance_vote("proposal", "tenant-test", protocol["id"], "proposal", "maybe", "voter", "evidence")
	with pytest.raises(PermissionError, match="risk_reviewer_required"):
		service.record_risk_assessment("risk", "tenant-test", protocol["id"], "low", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", protocol["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="defi_agent_runtime_not_supported"):
		service.register_defi_agent("agent", "tenant-test", "Bad Agent", "unsupported", "protocol_monitor", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_defi", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_defi", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_defi", PACKAGE_DIR / "app.py")

	protocol = api.register_protocol({"tenant_id": "tenant-api", "protocol_id": "api-protocol", "protocol_type": "dex", "network_reference": "fintech_blockchain:polygon", "protocol_reference": "uniswap-v3", "owner_id": "owner", "evidence_reference": "evidence", "risk_tier": "medium"})
	position = api.open_position({"tenant_id": "tenant-api", "position_id": "api-position", "protocol_id": protocol["id"], "account_reference": "wallet", "asset_pair_reference": "USDC/ETH", "position_type": "liquidity", "amount_minor": 1, "health_factor_bps": 10000, "evidence_reference": "evidence"})
	api.record_reward({"tenant_id": "tenant-api", "reward_id": "api-reward", "position_id": position["id"], "reward_type": "fee_share", "asset_reference": "USDC", "amount_minor": 1, "evidence_reference": "evidence"})
	agent = api.register_defi_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "DeFi Agent", "runtime": "claude_code", "role": "position_reconciler"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.defi_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "position_reconciler"
	assert dashboard["summary"]["protocol_count"] == 1
	assert console["positions"][0]["id"] == "api-position"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_defi"]["screens"]["agents"]["route"] == "/fintech-defi/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_defi", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_defi"]["streaming"]["processor"] == "bytewax"
