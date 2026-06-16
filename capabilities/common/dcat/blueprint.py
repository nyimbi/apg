# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025

from __future__ import annotations

CAPABILITY_METADATA: dict = {
	"capability_id": "dcat",
	"domain": "common",
	"name": "Data Catalog",
	"version": "1.0.0",
	"category": "Common",
	"api_enabled": True,
	"multi_tenant": True,
	"description": (
		"Dataset registry with lineage graph, metadata tagging, data quality "
		"scoring, and Apache Atlas-compatible metadata API."
	),
}


def init_subcapability(appbuilder) -> dict:
	from .api import bp
	appbuilder.get_app.register_blueprint(bp)
	return {"success": True, "capability_id": "dcat"}


def get_capability_info() -> dict:
	return CAPABILITY_METADATA


def get_health_status() -> dict:
	return {"status": "healthy", "capability_id": "dcat"}
