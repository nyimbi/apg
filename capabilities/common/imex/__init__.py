"""APG Import/Export capability."""

from datetime import datetime, timezone

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .imex_runtime import ImexService

__version__ = "1.0.0"


capability_metadata = {
	"name": "imex",
	"version": __version__,
	"display_name": "Import/Export",
	"description": "Governed import, export, and migration lifecycle control plane",
	"category": "data_platform",
	"subcategory": "migration",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": ["etlp", "conn", "auth", "audl", "moni", "keym", "encr"],
	"provides": ["imex_operations", "bulk_transfer", "data_migration", "schema_mapping", "transfer_validation"],
	"permissions": ["imex:view", "imex:create", "imex:execute", "imex:manage", "imex:approve", "imex:admin"],
}


class ImportExportCapability:
	"""Composition registration facade for IMEX."""

	def __init__(self) -> None:
		self.capability_id = "imex"
		self.metadata = capability_metadata
		self.health_status = "ready"
		self.last_health_check = datetime.now(timezone.utc)
		self.runtime = ImexService()

	async def initialize(self) -> bool:
		self.health_status = "ready"
		return True

	async def health_check(self) -> dict:
		self.last_health_check = datetime.now(timezone.utc)
		return {
			"capability": "imex",
			"status": self.health_status,
			"timestamp": self.last_health_check.isoformat(),
			"version": __version__,
			"summary": self.runtime.dashboard_summary(),
		}


def register_capability() -> dict:
	"""Register IMEX with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "imex",
		"aliases": ["import_export", "data_migration", "bulk_transfer"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": [],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"jobs": "Create and execute governed import, export, and migration jobs",
			"endpoints": "Bind transfer endpoints to CONN-managed connections",
			"mappings": "Attach schema profiles, mappings, and quality gates",
			"validation": "Validate previews, quarantine invalid records, and record quality evidence",
			"artifacts": "Publish export artifacts with checksum and retention metadata",
			"capability_rules": "Evaluate deterministic import/export governance rules",
			"visual_theming": "Apply transfer-console theme tokens and components",
		},
		"endpoints": {
			"jobs": "/imex/api/v1/jobs",
			"mappings": "/imex/api/v1/mappings",
			"validation": "/imex/api/v1/validation",
			"monitoring": "/imex/api/v1/monitoring",
			"artifacts": "/imex/api/v1/artifacts",
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"],
	}


def get_capability_info() -> dict:
	"""Get IMEX capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


imex_capability = ImportExportCapability()

__all__ = [
	"ImportExportCapability",
	"ImexService",
	"imex_capability",
	"capability_metadata",
	"register_capability",
	"get_capability_info",
	"get_capability_contract",
	"evaluate_capability_rules",
	"__version__",
]
