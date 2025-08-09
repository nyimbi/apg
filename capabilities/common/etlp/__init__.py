#!/usr/bin/env python3
"""
APG ETLP (Extract, Transform, Load, Process) Capability
Next-generation data processing platform with AI-powered optimization

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

from .models import *
from .service import ETLPService
from .views import *
from .api import *

__version__ = "1.0.0"
__capability_name__ = "etlp"
__capability_type__ = "common"
__description__ = "AI-powered data processing and pipeline orchestration"

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
		"ai_powered_optimization",
		"real_time_collaboration",
		"multi_modal_processing",
		"self_healing_pipelines",
		"federated_processing",
		"intelligent_quality_engine",
		"zero_config_governance"
	]
}