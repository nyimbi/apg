#!/usr/bin/env python3
"""
APG ETLP (Extract, Transform, Load, Process) Capability
Tenant-scoped pipeline design, execution, quality, and lineage governance.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules
)
from .service import (
	ETLPAuditEventRecord,
	ETLPDatasourceRecord,
	ETLPExecutionRecord,
	ETLPLifecycleBatchRecord,
	ETLPLifecycleService,
	ETLPMappingRecord,
	ETLPPipelineAgentRecord,
	ETLPPipelineRecord,
	ETLPPublishRecord,
	ETLPQualityRecord,
	ETLPReplayRecord,
	ETLPScheduleRecord,
)

try:
	from .models import *
	from .service import ETLPService
	from .views import *
	from .api import *
	_RUNTIME_IMPORT_ERROR = None
except Exception as exc:
	_RUNTIME_IMPORT_ERROR = exc
	ETLPService = None

__version__ = "1.0.0"
__capability_name__ = "etlp"
__capability_type__ = "common"
__description__ = "Tenant-scoped data pipeline design, execution, quality, and lineage governance"

# APG Composition Engine Metadata
APG_CAPABILITY_METADATA = {
	"name": "etlp",
	"version": __version__,
	"type": "common",
	"category": "data_processing",
	"description": __description__,
	"author": "APG Platform Team",
	"license": "Proprietary",
	"dependencies": [
		"metadata",
		"aicr",  # AI capabilities from common/aicr
		"auth_rbac", 
		"audit_compliance",
		"notification",
		"real_time_collaboration"
	],
	"optional_dependencies": [
		"edge_computing",
		"time_series_analytics",
		"federated_learning"
	],
	"api_endpoints": [
		"/api/v1/etlp/pipelines",
		"/api/v1/etlp/transformations", 
		"/api/v1/etlp/executions",
		"/api/v1/etlp/datasources",
		"/api/v1/etlp/quality"
	],
	"ui_routes": [
		"/etlp/pipelines",
		"/etlp/designer",
		"/etlp/monitor",
		"/etlp/quality"
	],
	"permissions": [
		"etlp:pipeline:read",
		"etlp:pipeline:write", 
		"etlp:pipeline:execute",
		"etlp:pipeline:delete",
		"etlp:transformation:read",
		"etlp:transformation:write",
		"etlp:datasource:read",
		"etlp:datasource:write",
		"etlp:quality:read"
	],
	"features": [
			"visual_pipeline_designer",
			"pipeline_lifecycle_guardrails",
			"datasource_approval",
			"field_mapping",
			"quality_gates",
			"lineage_emission",
			"retry_replay_backfill_controls",
			"pipeline_agent_composition",
			"lifecycle_batch_validation",
			"adapter_boundaries"
	]
}


def register_capability() -> dict:
	"""Register ETLP with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "etlp",
		"aliases": ["etl", "elt", "pipeline_processing", "data_pipelines"],
		"display_name": "ETL/ELT Processing",
		"description": __description__,
		"version": __version__,
		"dependencies": contract["requires"],
		"optional_dependencies": APG_CAPABILITY_METADATA["optional_dependencies"],
		"api_prefix": contract["ui"]["api_prefix"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"pipeline_orchestration": "Design, schedule, execute, and monitor tenant pipelines",
			"transformation_governance": "Control data transformations and mapping policies",
			"quality_gates": "Apply quality rules before publish or downstream delivery",
			"lineage_emission": "Emit metadata and lineage events for catalog integration",
			"datasource_governance": "Approve and govern pipeline sources and targets",
			"execution_controls": "Enforce approval, idempotency, retry, replay, and backfill controls",
			"capability_rules": "Evaluate deterministic pipeline governance rules",
			"visual_theming": "Apply pipeline-console theme tokens and components",
			"pipeline_agent_composition": "Register first-class pipeline agents across Codex, Claude Code, opencode, Pi, and future runtime adapters",
			"lifecycle_batch_validation": "Validate ETLP lifecycle mutation batches against Bytewax-first stream rules",
			"review_evidence": "Preserve policy decisions, matched rules, review reasons, and review evidence for generated pipeline governance queues"
		},
		"endpoints": {
			"pipelines": "/etlp/api/v1/pipelines",
			"transformations": "/etlp/api/v1/transformations",
			"executions": "/etlp/api/v1/executions",
			"datasources": "/etlp/api/v1/datasources",
			"quality": "/etlp/api/v1/quality",
			"lineage": "/etlp/api/v1/lineage",
			"agents": "/etlp/api/v1/agents",
			"lifecycle": "/etlp/api/v1/lifecycle"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
		"permissions": APG_CAPABILITY_METADATA["permissions"]
	}


def get_capability_info() -> dict:
	"""Get ETLP capability information for composition and marketplace discovery."""
	info = APG_CAPABILITY_METADATA.copy()
	info["contract"] = get_capability_contract()
	info["runtime_import_error"] = str(_RUNTIME_IMPORT_ERROR) if _RUNTIME_IMPORT_ERROR else None
	return info
