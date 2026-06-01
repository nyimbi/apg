"""Process-local API helpers for APG Blockchain Services."""

from __future__ import annotations

try:
	from .service import BlockchainServicesService
except ImportError:  # pragma: no cover
	from service import BlockchainServicesService  # type: ignore


_SERVICE = BlockchainServicesService()


def service() -> BlockchainServicesService:
	return _SERVICE


def register_network(payload: dict):
	return _SERVICE.register_network(payload["network_id"], payload.get("tenant_id", "default"), payload["network_type"], payload["environment"], payload["chain_id"], payload["rpc_reference"], payload["owner_id"], payload["evidence_reference"], payload.get("policy_attached", True))


def register_wallet(payload: dict):
	return _SERVICE.register_wallet(payload["wallet_id"], payload.get("tenant_id", "default"), payload["network_id"], payload["wallet_reference"], payload["custody_model"], payload["key_policy_reference"], payload["owner_id"], payload["evidence_reference"])


def deploy_contract(payload: dict):
	return _SERVICE.deploy_contract(payload["contract_id"], payload.get("tenant_id", "default"), payload["network_id"], payload["contract_type"], payload["artifact_reference"], payload["owner_id"], payload["approval_reference"], payload["evidence_reference"])


def record_transaction(payload: dict):
	return _SERVICE.record_transaction(payload["transaction_id"], payload.get("tenant_id", "default"), payload["network_id"], payload["transaction_hash"], payload["transaction_type"], payload["asset_reference"], payload["amount_minor"], payload["signer_id"], payload["evidence_reference"], payload["settlement_status"], payload.get("approval_reference", ""))


def anchor_evidence(payload: dict):
	return _SERVICE.anchor_evidence(payload["anchor_id"], payload.get("tenant_id", "default"), payload["network_id"], payload["payload_hash"], payload["reference_id"], payload["anchored_at"], payload["evidence_reference"])


def register_oracle_feed(payload: dict):
	return _SERVICE.register_oracle_feed(payload["oracle_id"], payload.get("tenant_id", "default"), payload["network_id"], payload["feed_type"], payload["source_reference"], payload["owner_id"], payload["evidence_reference"])


def record_node_health(payload: dict):
	return _SERVICE.record_node_health(payload["node_id"], payload.get("tenant_id", "default"), payload["network_id"], payload["endpoint_reference"], payload["status"], payload["block_height"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_blockchain_agent(payload: dict):
	return _SERVICE.register_blockchain_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "blockchain operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
