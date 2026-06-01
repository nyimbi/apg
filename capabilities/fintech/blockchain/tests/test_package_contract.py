"""Executable Blockchain Services capability package tests."""

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
	module = _load_module("contract_fintech_blockchain", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_blockchain"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "blockchain_agent_workflow" in contract["provides"]
	assert "/fintech-blockchain/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_blockchain", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "blockchain_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "blockchain_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_blockchain_lifecycle():
	service_module = _load_module("service_fintech_blockchain", PACKAGE_DIR / "service.py")
	service = service_module.BlockchainServicesService()

	network = service.register_network("network-1", "tenant-test", "ethereum", "testnet", "11155111", "rpc-ref", "owner-1", "network-evidence")
	wallet = service.register_wallet("wallet-1", "tenant-test", network["id"], "wallet-ref", "mpc", "key-policy", "owner-1", "wallet-evidence")
	contract = service.deploy_contract("contract-1", "tenant-test", network["id"], "settlement", "artifact-ref", "owner-1", "approval-ref", "contract-evidence")
	transaction = service.record_transaction("tx-1", "tenant-test", network["id"], "0xabc", "settlement", contract["id"], 1000, "signer-1", "tx-evidence", "confirmed")
	anchor = service.anchor_evidence("anchor-1", "tenant-test", network["id"], "hash-1", transaction["id"], "2026-06-01T10:00:00Z", "anchor-evidence")
	oracle = service.register_oracle_feed("oracle-1", "tenant-test", network["id"], "price", "feed-ref", "owner-1", "oracle-evidence")
	node = service.record_node_health("node-1", "tenant-test", network["id"], "endpoint-ref", "healthy", 12345, "node-evidence")
	review = service.record_review("review-1", "tenant-test", contract["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_blockchain_agent("agent-1", "tenant-test", "Blockchain Agent", "codex", "transaction_reconciler", "reconcile chain transactions")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert network["network_type"] == "ethereum"
	assert wallet["custody_model"] == "mpc"
	assert contract["status"] == "deployed"
	assert transaction["settlement_status"] == "confirmed"
	assert anchor["reference_id"] == transaction["id"]
	assert oracle["feed_type"] == "price"
	assert node["status"] == "healthy"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["transaction_count"] == 1
	assert summary["audit_event_count"] == 9


def test_service_guardrails_reject_invalid_blockchain_actions():
	service_module = _load_module("guardrail_service_fintech_blockchain", PACKAGE_DIR / "service.py")
	service = service_module.BlockchainServicesService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_network("network", "", "ethereum", "testnet", "1", "rpc", "owner", "evidence")
	with pytest.raises(PermissionError, match="network_type_not_supported"):
		service.register_network("network", "tenant-test", "unknown", "testnet", "1", "rpc", "owner", "evidence")
	with pytest.raises(PermissionError, match="chain_id_required"):
		service.register_network("network", "tenant-test", "ethereum", "testnet", "", "rpc", "owner", "evidence")
	network = service.register_network("network-ok", "tenant-test", "ethereum", "testnet", "11155111", "rpc", "owner", "evidence")
	with pytest.raises(PermissionError, match="custody_model_not_supported"):
		service.register_wallet("wallet", "tenant-test", network["id"], "wallet", "unknown", "key-policy", "owner", "evidence")
	with pytest.raises(PermissionError, match="contract_approval_required"):
		service.deploy_contract("contract", "tenant-test", network["id"], "settlement", "artifact", "owner", "", "evidence")
	with pytest.raises(PermissionError, match="transaction_amount_invalid"):
		service.record_transaction("tx", "tenant-test", network["id"], "0xabc", "transfer", "asset", -1, "signer", "evidence", "confirmed")
	with pytest.raises(PermissionError, match="transaction_approval_required"):
		service.record_transaction("tx-high", "tenant-test", network["id"], "0xhigh", "transfer", "asset", 100000000, "signer", "evidence", "confirmed")
	with pytest.raises(PermissionError, match="payload_hash_required"):
		service.anchor_evidence("anchor", "tenant-test", network["id"], "", "ref", "2026-06-01T10:00:00Z", "evidence")
	with pytest.raises(PermissionError, match="oracle_feed_type_not_supported"):
		service.register_oracle_feed("oracle", "tenant-test", network["id"], "unknown", "source", "owner", "evidence")
	with pytest.raises(PermissionError, match="block_height_invalid"):
		service.record_node_health("node", "tenant-test", network["id"], "endpoint", "healthy", -1, "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", network["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="blockchain_agent_runtime_not_supported"):
		service.register_blockchain_agent("agent", "tenant-test", "Bad Agent", "unsupported", "transaction_reconciler", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_blockchain", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_blockchain", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_blockchain", PACKAGE_DIR / "app.py")

	network = api.register_network({"tenant_id": "tenant-api", "network_id": "api-network", "network_type": "polygon", "environment": "testnet", "chain_id": "80002", "rpc_reference": "rpc", "owner_id": "owner", "evidence_reference": "evidence"})
	api.register_wallet({"tenant_id": "tenant-api", "wallet_id": "api-wallet", "network_id": network["id"], "wallet_reference": "wallet", "custody_model": "hsm", "key_policy_reference": "key-policy", "owner_id": "owner", "evidence_reference": "evidence"})
	api.record_node_health({"tenant_id": "tenant-api", "node_id": "api-node", "network_id": network["id"], "endpoint_reference": "endpoint", "status": "healthy", "block_height": 1, "evidence_reference": "evidence"})
	agent = api.register_blockchain_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Blockchain Agent", "runtime": "claude_code", "role": "network_operator"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.blockchain_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "network_operator"
	assert dashboard["summary"]["network_count"] == 1
	assert console["wallets"][0]["id"] == "api-wallet"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_blockchain"]["screens"]["agents"]["route"] == "/fintech-blockchain/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_blockchain", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_blockchain"]["streaming"]["processor"] == "bytewax"
