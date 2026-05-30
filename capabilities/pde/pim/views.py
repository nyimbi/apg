"""View models for Product Information Management."""

from __future__ import annotations

from typing import Any


def dashboard_model(service: Any, tenant_id: str) -> dict[str, Any]:
	summary = service.dashboard_summary(tenant_id)
	return {"title": "Product Information Management", "tenant_id": tenant_id, "cards": [{"label": "Catalogs", "value": summary["catalog_count"], "icon": "folder-tree"}, {"label": "Products", "value": summary["product_count"], "icon": "package"}, {"label": "Content", "value": summary["content_count"], "icon": "file-text"}, {"label": "Channels", "value": summary["channel_count"], "icon": "send"}, {"label": "Quality", "value": summary["quality_issue_count"], "icon": "badge-check"}, {"label": "Agents", "value": summary["agent_count"], "icon": "bot"}], "streaming": summary["streaming"]}


def product_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Products", "records": service.list_records(tenant_id, "product")}


def catalog_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Catalogs", "records": service.list_records(tenant_id, "catalog")}


def content_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Content", "records": service.list_records(tenant_id, "content")}


def agent_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "PIM Agents", "records": service.list_records(tenant_id, "agent"), "policy": {"max_autonomous_scope": "inspect_prepare_and_recommend", "human_approval_required": True}}


def rules_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {"name": "PIM Rules", "rule_count": len(contract["rule_engine"]["rules"]), "rules": contract["rule_engine"]["rules"]}


def settings_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {"name": "PIM Settings", "configuration": contract["configuration"], "theme": contract["theme"], "routes": contract["ui"]["routes"]}
