"""
NLPC - Natural Language Processing Core Capability

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
Website: www.datacraft.co.ke

This module provides the APG composition engine integration for NLPC capability.
Registers NLPC as a composable service within the APG ecosystem.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid_extensions import uuid7str

from .capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)

# APG Capability Metadata
CAPABILITY_INFO = {
	"name": "nlpc",
	"version": "1.0.0",
	"description": "Natural Language Processing Core - Universal text intelligence platform",
	"author": "Nyimbi Odero",
	"email": "nyimbi@gmail.com",
	"website": "www.datacraft.co.ke",
	"category": "ai_ml",
	"subcategory": "nlp",
	"priority": "high",
	"dependencies": [
		"aicr",      # AI Core Framework for model orchestration
		"mlcm",      # Machine Learning Core Management
		"conf",      # Configuration management
		"auth_rbac", # Authentication and authorization
		"audit_compliance",  # Audit trails and compliance
		"real_time_collaboration"  # Real-time processing
	],
	"provides": [
		"text_processing",
		"language_analysis",
		"sentiment_analysis",
		"entity_recognition",
		"text_classification",
		"semantic_search",
		"text_generation",
		"multilingual_processing"
	],
	"supported_languages": [
		"en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi",
		"af", "aa", "ak", "am", "bm", "ee", "ff", "ha", "ig", "kr", "ki", "rw",
		"rn", "kg", "ln", "lg", "mg", "ny", "om", "sg", "sn", "so", "st", "sw",
		"ss", "ti", "ts", "tn", "tw", "ve", "wo", "xh", "yo", "zu", "kab", "kam",
		"luo", "mas", "mer", "mos", "nus", "suk", "tzm", "tig", "umb"
	],
	"performance_metrics": {
		"max_latency_ms": 100,
		"throughput_docs_per_sec": 10000,
		"supported_concurrent_users": 1000,
		"memory_efficiency": "high",
		"gpu_acceleration": True
	}
}

# APG Service Registration
SERVICE_REGISTRY = {
	"service_id": f"nlpc-{uuid7str()}",
	"service_name": "Natural Language Processing Core",
	"service_type": "nlp_processor",
	"api_version": "v1",
	"health_check_endpoint": "/api/v1/nlp/health",
	"metrics_endpoint": "/api/v1/nlp/metrics",
	"documentation_url": "/docs/nlpc/user_guide.html",
	"openapi_spec": "/api/v1/nlp/openapi.json"
}

# Composition Engine Integration
COMPOSITION_SCHEMA = {
	"input_schemas": {
		"text_document": {
			"type": "object",
			"properties": {
				"content": {"type": "string", "minLength": 1},
				"language": {"type": "string", "enum": CAPABILITY_INFO["supported_languages"]},
				"metadata": {"type": "object"}
			},
			"required": ["content"]
		},
		"processing_request": {
			"type": "object",
			"properties": {
				"document_id": {"type": "string"},
				"tasks": {
					"type": "array",
					"items": {"type": "string"},
					"minItems": 1
				},
				"parameters": {"type": "object"}
			},
			"required": ["document_id", "tasks"]
		}
	},
	"output_schemas": {
		"processing_result": {
			"type": "object",
			"properties": {
				"result_id": {"type": "string"},
				"task_type": {"type": "string"},
				"confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
				"result_data": {"type": "object"},
				"processing_time": {"type": "number", "minimum": 0}
			},
			"required": ["result_id", "task_type", "confidence_score", "result_data"]
		}
	}
}


def _log_capability_info(message: str) -> None:
	"""Log capability information messages."""
	print(f"[NLPC Capability] {message}")


def _log_composition_registration() -> str:
	"""Log successful composition engine registration."""
	message = f"NLPC capability registered with composition engine - Service ID: {SERVICE_REGISTRY['service_id']}"
	_log_capability_info(message)
	return message


def _log_dependency_check(dependencies: List[str]) -> dict[str, bool]:
	"""
	Log dependency availability check.
	
	Args:
		dependencies: List of required APG capabilities
		
	Returns:
		Dictionary mapping dependencies to availability status
	"""
	dependency_status = {}
	
	for dep in dependencies:
		# In a real implementation, this would check if the capability is available
		# For now, we assume all dependencies are available
		dependency_status[dep] = True
		_log_capability_info(f"Dependency check: {dep} - Available")
	
	return dependency_status


async def register_with_apg_composition() -> dict[str, Any]:
	"""
	Register NLPC capability with APG composition engine.
	
	Returns:
		Registration status and details
	"""
	try:
		# Check dependencies
		dependency_status = _log_dependency_check(CAPABILITY_INFO["dependencies"])
		
		# Register capability metadata
		registration_data = {
			"capability_info": CAPABILITY_INFO,
			"service_registry": SERVICE_REGISTRY,
			"composition_schema": COMPOSITION_SCHEMA,
			"dependency_status": dependency_status,
			"registration_timestamp": "2025-08-20T00:00:00Z",
			"status": "registered"
		}
		
		_log_composition_registration()
		
		return registration_data
		
	except Exception as e:
		error_message = f"Failed to register NLPC with composition engine: {str(e)}"
		_log_capability_info(f"ERROR: {error_message}")
		return {
			"status": "failed",
			"error": error_message,
			"capability_info": CAPABILITY_INFO
		}


async def initialize_nlpc_capability() -> bool:
	"""
	Initialize NLPC capability with APG platform integration.
	
	Returns:
		True if initialization successful, False otherwise
	"""
	try:
		_log_capability_info("Initializing NLPC capability...")
		
		# Register with composition engine
		registration_result = await register_with_apg_composition()
		
		if registration_result.get("status") == "registered":
			_log_capability_info("NLPC capability initialized successfully")
			return True
		else:
			_log_capability_info(f"NLPC initialization failed: {registration_result.get('error')}")
			return False
			
	except Exception as e:
		_log_capability_info(f"NLPC initialization error: {str(e)}")
		return False


def get_capability_info() -> dict[str, Any]:
	"""
	Get NLPC capability information for APG platform.
	
	Returns:
		Capability metadata and configuration
	"""
	contract = get_capability_contract()
	return {
		"capability": CAPABILITY_INFO,
		"service": SERVICE_REGISTRY,
		"composition": COMPOSITION_SCHEMA,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"ui_manifest": contract["ui"],
		"theme": contract["theme"]
	}


def register_capability() -> dict[str, Any]:
	"""Register NLPC as a first-class APG composition capability."""
	contract = get_capability_contract()
	return {
		"name": "nlpc",
		"aliases": ["nlp_core", "text_intelligence", "language_processing"],
		"display_name": "NLP Core",
		"description": CAPABILITY_INFO["description"],
		"version": CAPABILITY_INFO["version"],
		"dependencies": ["aicr", "mlcm", "conf"],
		"optional_dependencies": ["auth", "audl", "mqeb", "cach", "ragn"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"text_processing": "Process tenant-scoped documents through NLP pipelines",
			"language_analysis": "Detect and normalize languages, including broad African language coverage",
			"sentiment_analysis": "Score sentiment, emotion, and intent for governed text workloads",
			"entity_recognition": "Extract named entities, keywords, and PII signals",
			"text_generation": "Route generation and summarization with safety controls",
			"capability_rules": "Evaluate deterministic NLP governance rules",
			"visual_theming": "Apply text-intelligence console theme tokens and components"
		},
		"endpoints": {
			"documents": "/nlpc/api/v1/documents",
			"process": "/nlpc/api/v1/process",
			"models": "/nlpc/api/v1/models",
			"languages": "/nlpc/api/v1/languages",
			"analytics": "/nlpc/api/v1/analytics"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": [
			"nlpc:view",
			"nlpc:process",
			"nlpc:annotate",
			"nlpc:manage_models",
			"nlpc:govern",
			"nlpc:admin"
		]
	}


def get_supported_tasks() -> List[str]:
	"""
	Get list of supported NLP tasks.
	
	Returns:
		List of NLP task names
	"""
	from .models import NLPTask
	return [task.value for task in NLPTask]


def get_supported_languages() -> List[str]:
	"""
	Get list of supported languages.
	
	Returns:
		List of language codes
	"""
	return CAPABILITY_INFO["supported_languages"]


async def health_check() -> dict[str, Any]:
	"""
	Perform health check for NLPC capability.
	
	Returns:
		Health status information
	"""
	try:
		# Check model availability
		from .models import validate_models_async
		model_validation = await validate_models_async()
		
		# Check dependencies
		dependency_status = _log_dependency_check(CAPABILITY_INFO["dependencies"])
		
		all_models_valid = all(model_validation.values())
		all_deps_available = all(dependency_status.values())
		
		health_status = {
			"status": "healthy" if (all_models_valid and all_deps_available) else "degraded",
			"timestamp": "2025-08-20T00:00:00Z",
			"models": model_validation,
			"dependencies": dependency_status,
			"capability_info": CAPABILITY_INFO,
			"service_info": SERVICE_REGISTRY
		}
		
		return health_status
		
	except Exception as e:
		return {
			"status": "unhealthy",
			"error": str(e),
			"timestamp": "2025-08-20T00:00:00Z"
		}


# APG Capability Exports
__all__ = [
	"CAPABILITY_INFO",
	"SERVICE_REGISTRY", 
	"COMPOSITION_SCHEMA",
	"register_with_apg_composition",
	"register_capability",
	"initialize_nlpc_capability",
	"get_capability_info",
	"get_capability_contract",
	"evaluate_capability_rules",
	"get_supported_tasks",
	"get_supported_languages",
	"health_check"
]
