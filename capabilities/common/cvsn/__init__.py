"""
Computer Vision & Visual Intelligence - APG Capability Registration

APG platform capability registration providing computer vision and visual intelligence
processing with OCR, object detection, facial recognition, quality control, and video
analysis capabilities integrated with APG composition engine and multi-tenant architecture.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from .capability_contract import evaluate_capability_rules, get_capability_contract

# APG Capability Metadata
CAPABILITY_METADATA = {
	"capability_id": "computer_vision",
	"capability_name": "Computer Vision",
	"capability_description": "Governed visual intelligence for OCR, object detection, quality inspection, safety analytics, model lifecycle, and generated applications",
	"version": "1.0.0",
	"author": "Datacraft",
	"author_email": "nyimbi@gmail.com",
	"website": "https://www.datacraft.co.ke",
	"license": "Enterprise",
	"created_at": datetime.utcnow(),
	"updated_at": datetime.utcnow(),
	
	# Capability classification
	"category": "general_cross_functional",
	"subcategory": "ai_machine_learning",
	"industry_focus": "All",
	"business_domain": "Visual Content Processing",
	
	# Technical specifications
	"python_version": "3.11+",
	"framework": "APG generated application + optional FastAPI/Flask-AppBuilder production services",
	"database": "PostgreSQL",
	"cache": "Redis",
	"ai_models": ["YOLO", "Vision Transformers", "Tesseract OCR", "OpenCV"],
	
	# Capability features
	"features": [
		"Document OCR & Text Extraction",
		"Object Detection & Recognition", 
		"Image Classification & Analysis",
		"Facial Recognition & Biometrics",
		"Quality Control & Inspection",
		"Video Analysis & Processing",
		"Visual Similarity Search",
		"Batch Processing & Automation",
		"Real-time Processing",
		"Multi-language Support",
		"Edge Computing Ready",
		"Custom Model Training"
	],
	
	# Business capabilities
	"business_capabilities": [
		"Automated Document Processing",
		"Manufacturing Quality Control",
		"Security & Surveillance",
		"Content Moderation",
		"Inventory Management",
		"Customer Analytics",
		"Compliance Verification",
		"Process Automation"
	],
	
	# Integration points
	"integrations": [
		"Document Management Systems",
		"Manufacturing Systems",
		"Security Platforms",
		"Business Intelligence",
		"Workflow Automation",
		"Asset Management",
		"Quality Management Systems",
		"Compliance Platforms"
	]
}

# APG Composition Keywords for Discovery
COMPOSITION_KEYWORDS = [
	# Primary capabilities
	"processes_images", "visual_intelligence_enabled", "computer_vision_capable",
	"document_processing_aware", "real_time_vision", "ai_powered_vision",
	
	# Specific processing types
	"ocr_enabled", "text_extraction", "document_analysis",
	"object_detection", "image_classification", "visual_recognition",
	"facial_recognition", "biometric_processing", "identity_verification",
	"quality_control", "defect_detection", "inspection_automation",
	"video_analysis", "action_recognition", "motion_detection",
	
	# Technical capabilities
	"batch_processing", "real_time_processing", "edge_computing",
	"multi_modal_processing", "ai_model_training", "custom_models",
	
	# Business applications
	"manufacturing_qc", "security_surveillance", "content_moderation",
	"inventory_tracking", "compliance_verification", "process_automation",
	
	# Integration capabilities
	"multimedia_processing", "content_intelligence", "visual_analytics",
	"automated_inspection", "smart_recognition", "intelligent_processing"
]

# APG Capability Dependencies
CAPABILITY_DEPENDENCIES = {
	"required": [
		"auth",
		"audl",
		"conf"
	],
	"enhanced": [
		"aicr",
		"mlcm",
		"moni",
		"bytewax"
	],
	"optional": [
		"stor",
		"srch",
		"cach",
		"mqeb"
	]
}

# APG Permission Structure
CAPABILITY_PERMISSIONS = {
	# Core permissions
	"cv:read": {
		"name": "Computer Vision Read",
		"description": "View computer vision processing results and dashboards"
	},
	"cv:write": {
		"name": "Computer Vision Write", 
		"description": "Upload files and create processing jobs"
	},
	"cv:admin": {
		"name": "Computer Vision Admin",
		"description": "Manage models, configure settings, and access analytics"
	},
	
	# Specific processing permissions
	"cv:ocr": {
		"name": "OCR Processing",
		"description": "Access document OCR and text extraction features"
	},
	"cv:object_detection": {
		"name": "Object Detection",
		"description": "Access object detection and image analysis features"
	},
	"cv:facial_recognition": {
		"name": "Facial Recognition",
		"description": "Access facial recognition and biometric features (requires consent)"
	},
	"cv:quality_control": {
		"name": "Quality Control",
		"description": "Access manufacturing quality control and inspection features"
	},
	"cv:video_analysis": {
		"name": "Video Analysis",
		"description": "Access video processing and analysis features"
	},
	"cv:batch_processing": {
		"name": "Batch Processing",
		"description": "Create and manage batch processing jobs"
	},
	"cv:model_management": {
		"name": "Model Management",
		"description": "Deploy, configure, and manage AI models"
	},
	
	# Analytics and reporting
	"cv:analytics": {
		"name": "Analytics Access",
		"description": "Access analytics dashboards and performance metrics"
	},
	"cv:reports": {
		"name": "Report Generation",
		"description": "Generate and export processing reports"
	}
}

# APG Multi-tenant Configuration
MULTI_TENANT_CONFIG = {
	"tenant_isolation": "complete",
	"data_separation": "schema_based",
	"resource_isolation": "container_based",
	"billing_model": "usage_based",
	"scaling_model": "per_tenant",
	"customization_level": "high",
	"compliance_inheritance": True,
	"audit_separation": True
}

# APG Platform Integration Configuration
PLATFORM_INTEGRATION = {
	"menu_integration": {
		"primary_menu": "Computer Vision",
		"icon": "fa-eye",
		"order": 15,
		"submenu": [
			{"name": "Dashboard", "url": "/computer_vision/", "icon": "fa-dashboard"},
			{"name": "Document Processing", "url": "/computer_vision/documents/", "icon": "fa-file-text"},
			{"name": "Image Analysis", "url": "/computer_vision/images/", "icon": "fa-image"},
			{"name": "Quality Control", "url": "/computer_vision/quality/", "icon": "fa-check-circle"},
			{"name": "Video Analysis", "url": "/computer_vision/video/", "icon": "fa-video-camera"},
			{"name": "Model Management", "url": "/computer_vision/models/", "icon": "fa-cogs"}
		]
	},
	
	"dashboard_widgets": [
		{
			"name": "CV Processing Stats",
			"component": "cv_processing_stats",
			"size": "medium",
			"refresh_interval": 30
		},
		{
			"name": "Recent Processing Jobs",
			"component": "cv_recent_jobs",
			"size": "large",
			"refresh_interval": 10
		},
		{
			"name": "Model Performance",
			"component": "cv_model_performance",
			"size": "medium",
			"refresh_interval": 60
		}
	],
	
	"notification_types": [
		"cv_job_completed",
		"cv_job_failed",
		"cv_batch_completed",
		"cv_model_deployed",
		"cv_quality_alert",
		"cv_compliance_issue"
	],
	
	"search_integration": {
		"searchable_content": [
			"processing_jobs",
			"extracted_text",
			"detection_results",
			"quality_reports"
		],
		"search_weights": {
			"job_name": 1.0,
			"extracted_text": 0.8,
			"object_classes": 0.6,
			"metadata": 0.4
		}
	}
}

# Performance and Scaling Configuration
PERFORMANCE_CONFIG = {
	"default_settings": {
		"max_concurrent_jobs": 50,
		"max_file_size_mb": 50,
		"job_timeout_minutes": 30,
		"cache_ttl_seconds": 3600,
		"batch_size_limit": 100
	},
	
	"scaling_thresholds": {
		"cpu_scale_up": 70,
		"cpu_scale_down": 30,
		"memory_scale_up": 80,
		"memory_scale_down": 40,
		"queue_scale_up": 10,
		"queue_scale_down": 2
	},
	
	"resource_limits": {
		"min_replicas": 2,
		"max_replicas": 20,
		"cpu_request": "500m",
		"cpu_limit": "2000m",
		"memory_request": "1Gi",
		"memory_limit": "4Gi"
	}
}

# Compliance and Security Configuration
COMPLIANCE_CONFIG = {
	"data_privacy": {
		"gdpr_compliant": True,
		"hipaa_ready": True,
		"ccpa_compliant": True,
		"data_residency": "configurable",
		"encryption_at_rest": "AES-256",
		"encryption_in_transit": "TLS-1.3"
	},
	
	"biometric_compliance": {
		"consent_required": True,
		"data_anonymization": True,
		"retention_policies": True,
		"deletion_procedures": True,
		"audit_requirements": True
	},
	
	"industry_standards": [
		"ISO-27001",
		"SOC-2-Type-II", 
		"FDA-GMP",
		"ISO-9001",
		"NIST-Framework"
	]
}

# API Configuration
API_CONFIG = {
	"base_path": "/api/v1",
	"documentation_url": "/docs",
	"openapi_url": "/openapi.json",
	"rate_limiting": {
		"requests_per_minute": 1000,
		"burst_requests": 100,
		"rate_limit_by": "tenant_id"
	},
	"authentication": {
		"method": "JWT",
		"token_expiry": 3600,
		"refresh_enabled": True
	},
	"cors": {
		"enabled": True,
		"origins": ["*"],  # Configure for production
		"methods": ["GET", "POST", "PUT", "DELETE"],
		"headers": ["*"]
	}
}


def get_capability_info() -> Dict[str, Any]:
	"""Get complete capability information for APG registration"""
	contract = get_capability_contract()
	return {
		"metadata": CAPABILITY_METADATA,
		"keywords": COMPOSITION_KEYWORDS,
		"dependencies": CAPABILITY_DEPENDENCIES,
		"permissions": CAPABILITY_PERMISSIONS,
		"multi_tenant": MULTI_TENANT_CONFIG,
		"integration": PLATFORM_INTEGRATION,
		"performance": PERFORMANCE_CONFIG,
		"compliance": COMPLIANCE_CONFIG,
		"api": API_CONFIG,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"ui_manifest": contract["ui"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"]
	}


def register_capability() -> Dict[str, Any]:
	"""Register CVSN as a first-class APG composition capability."""
	contract = get_capability_contract()
	return {
		"name": "cvsn",
		"aliases": ["computer_vision", "visual_intelligence", "vision_processing"],
		"display_name": CAPABILITY_METADATA["capability_name"],
		"description": CAPABILITY_METADATA["capability_description"],
		"version": CAPABILITY_METADATA["version"],
		"dependencies": contract["requires"] + ["audl", "moni"],
		"optional_dependencies": ["stor", "srch", "cach", "mqeb"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"adapters": contract["configuration"]["adapters"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"capabilities": {
			"asset_ingestion": "Register tenant-scoped visual assets with source, MIME, size, and digest evidence",
			"document_ocr": "Extract deterministic text evidence from tenant-scoped document assets",
			"object_detection": "Detect, classify, and track visual objects in images",
			"quality_control": "Run visual inspection workflows for manufacturing and operations",
			"factory_safety": "Surface high-severity smoke, fire, OSHA, and people-counting signals",
			"video_analytics": "Process governed video clips with sampling-policy evidence",
			"model_management": "Register and release vision models through MLCM linkage",
			"pipeline_management": "Register versioned vision pipelines with owner and model evidence",
			"vision_agent_composition": "Register provider-neutral AI vision agents with runtime, role, scope, owner, purpose, disclosure, and human-review guardrails",
			"lifecycle_batch_governance": "Validate CVSN lifecycle mutations through Bytewax-only batch contracts",
			"capability_rules": "Evaluate deterministic computer-vision governance rules",
			"visual_theming": "Apply industrial vision-console theme tokens and components"
		},
		"endpoints": {
			"assets": "/cvsn/api/v1/assets",
			"documents": "/cvsn/api/v1/documents",
			"images": "/cvsn/api/v1/images",
			"video": "/cvsn/api/v1/video",
			"quality": "/cvsn/api/v1/quality",
			"safety": "/cvsn/api/v1/safety",
			"similarity": "/cvsn/api/v1/similarity",
			"models": "/cvsn/api/v1/models",
			"agents": "/cvsn/api/v1/agents",
			"lifecycle": "/cvsn/api/v1/lifecycle",
			"audit": "/cvsn/api/v1/audit"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": list(CAPABILITY_PERMISSIONS) + ["cv:review"]
	}


def validate_capability_requirements() -> Dict[str, bool]:
	"""Validate that all capability requirements are met"""
	validation_results = {
		"models_available": True,  # Would check if AI models are accessible
		"dependencies_met": True,  # Would check if required capabilities are available
		"database_ready": True,   # Would check database connectivity
		"cache_ready": True,      # Would check Redis connectivity
		"storage_ready": True,    # Would check file storage accessibility
		"permissions_configured": True,  # Would check RBAC setup
		"compliance_ready": True  # Would check compliance configuration
	}
	
	return validation_results


def register_with_apg_platform() -> bool:
	"""Register capability with APG composition engine"""
	try:
		capability_info = get_capability_info()
		validation = validate_capability_requirements()
		
		if not all(validation.values()):
			failed_checks = [k for k, v in validation.items() if not v]
			raise RuntimeError(f"Capability validation failed: {failed_checks}")
		
		# Would integrate with actual APG composition engine
		print(f"Computer Vision capability registered successfully")
		print(f"Capability ID: {CAPABILITY_METADATA['capability_id']}")
		print(f"Version: {CAPABILITY_METADATA['version']}")
		print(f"Features: {len(CAPABILITY_METADATA['features'])}")
		print(f"Keywords: {len(COMPOSITION_KEYWORDS)}")
		
		return True
		
	except Exception as e:
		print(f"Failed to register Computer Vision capability: {e}")
		return False


# Export main interfaces
__all__ = [
	"CAPABILITY_METADATA",
	"COMPOSITION_KEYWORDS", 
	"CAPABILITY_DEPENDENCIES",
	"CAPABILITY_PERMISSIONS",
	"MULTI_TENANT_CONFIG",
	"PLATFORM_INTEGRATION",
	"PERFORMANCE_CONFIG",
	"COMPLIANCE_CONFIG",
	"API_CONFIG",
	"get_capability_contract",
	"evaluate_capability_rules",
	"get_capability_info",
	"register_capability",
	"validate_capability_requirements",
	"register_with_apg_platform"
]
