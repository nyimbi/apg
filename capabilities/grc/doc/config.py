"""
APG Document Service Configuration

Configuration settings for document service with APG integration patterns,
multi-tenancy support, and intelligent processing capabilities.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class APGDocumentConfig:
	"""APG Document Service configuration following APG patterns"""
	
	# APG Integration
	apg_base_url: str = os.getenv("APG_BASE_URL", "http://localhost:8000")
	apg_auth_service: str = os.getenv("APG_AUTH_SERVICE", "auth_rbac")
	apg_audit_service: str = os.getenv("APG_AUDIT_SERVICE", "audit_compliance")
	apg_vision_service: str = os.getenv("APG_VISION_SERVICE", "common.computer_vision")
	apg_nlp_service: str = os.getenv("APG_NLP_SERVICE", "common.nlp")
	apg_ai_orchestration: str = os.getenv("APG_AI_ORCHESTRATION", "common.ai_orchestration")
	
	# Multi-tenant Configuration
	tenant_mode: str = os.getenv("APG_TENANT_MODE", "multi")  # single, multi
	tenant_isolation_level: str = os.getenv("APG_TENANT_ISOLATION", "strict")  # loose, strict
	max_tenants_per_instance: int = int(os.getenv("APG_MAX_TENANTS", "100"))
	
	# Document Processing
	max_file_size_mb: int = int(os.getenv("DOC_MAX_FILE_SIZE_MB", "100"))
	supported_formats: List[str] = field(default_factory=lambda: [
		"pdf", "docx", "doc", "txt", "rtf", "odt", 
		"jpg", "jpeg", "png", "tiff", "gif", "bmp", "webp",
		"html", "xml", "json", "csv", "xlsx", "pptx"
	])
	processing_timeout_seconds: int = int(os.getenv("DOC_PROCESSING_TIMEOUT", "300"))
	batch_processing_limit: int = int(os.getenv("DOC_BATCH_LIMIT", "50"))
	
	# Storage Configuration
	storage_backend: str = os.getenv("DOC_STORAGE_BACKEND", "filesystem")  # filesystem, s3, azure
	storage_path: str = os.getenv("DOC_STORAGE_PATH", "./storage/documents")
	encryption_enabled: bool = os.getenv("DOC_ENCRYPTION_ENABLED", "true").lower() == "true"
	backup_enabled: bool = os.getenv("DOC_BACKUP_ENABLED", "true").lower() == "true"
	retention_days_default: int = int(os.getenv("DOC_RETENTION_DAYS", "2555"))  # 7 years
	
	# AI Processing Configuration
	ocr_enabled: bool = os.getenv("DOC_OCR_ENABLED", "true").lower() == "true"
	nlp_enabled: bool = os.getenv("DOC_NLP_ENABLED", "true").lower() == "true"
	auto_classification: bool = os.getenv("DOC_AUTO_CLASSIFICATION", "true").lower() == "true"
	similarity_threshold: float = float(os.getenv("DOC_SIMILARITY_THRESHOLD", "0.85"))
	
	# Ollama Integration
	ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
	vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5-vl:latest")
	language_model: str = os.getenv("OLLAMA_LANGUAGE_MODEL", "gemma2:latest")
	analysis_model: str = os.getenv("OLLAMA_ANALYSIS_MODEL", "qwen2.5-vl:latest")
	
	# Database Configuration
	database_url: str = os.getenv("DATABASE_URL", "postgresql://localhost/apg_documents")
	connection_pool_size: int = int(os.getenv("DB_POOL_SIZE", "20"))
	connection_pool_overflow: int = int(os.getenv("DB_POOL_OVERFLOW", "30"))
	
	# Performance Configuration
	max_concurrent_processing: int = int(os.getenv("DOC_MAX_CONCURRENT", "10"))
	cache_enabled: bool = os.getenv("DOC_CACHE_ENABLED", "true").lower() == "true"
	cache_ttl_seconds: int = int(os.getenv("DOC_CACHE_TTL", "3600"))
	search_index_rebuild_interval: int = int(os.getenv("DOC_INDEX_REBUILD", "86400"))
	
	# Security Configuration
	require_authentication: bool = True  # Always true in APG
	allow_anonymous_read: bool = False   # Always false in APG
	audit_all_operations: bool = True    # Always true in APG
	encrypt_sensitive_metadata: bool = os.getenv("DOC_ENCRYPT_METADATA", "true").lower() == "true"
	
	# Collaboration Configuration
	real_time_enabled: bool = os.getenv("DOC_REALTIME_ENABLED", "true").lower() == "true"
	max_collaborators: int = int(os.getenv("DOC_MAX_COLLABORATORS", "50"))
	version_history_limit: int = int(os.getenv("DOC_VERSION_LIMIT", "100"))
	conflict_resolution: str = os.getenv("DOC_CONFLICT_RESOLUTION", "intelligent")  # manual, auto, intelligent
	
	# Workflow Configuration
	workflow_engine_enabled: bool = os.getenv("DOC_WORKFLOW_ENABLED", "true").lower() == "true"
	approval_timeout_hours: int = int(os.getenv("DOC_APPROVAL_TIMEOUT", "72"))
	auto_routing_enabled: bool = os.getenv("DOC_AUTO_ROUTING", "true").lower() == "true"
	
	# Notification Configuration
	notifications_enabled: bool = os.getenv("DOC_NOTIFICATIONS_ENABLED", "true").lower() == "true"
	email_notifications: bool = os.getenv("DOC_EMAIL_NOTIFICATIONS", "true").lower() == "true"
	push_notifications: bool = os.getenv("DOC_PUSH_NOTIFICATIONS", "true").lower() == "true"
	
	# Monitoring Configuration
	metrics_enabled: bool = os.getenv("DOC_METRICS_ENABLED", "true").lower() == "true"
	health_check_interval: int = int(os.getenv("DOC_HEALTH_CHECK_INTERVAL", "30"))
	performance_logging: bool = os.getenv("DOC_PERFORMANCE_LOGGING", "true").lower() == "true"
	
	# API Configuration
	api_rate_limit: int = int(os.getenv("DOC_API_RATE_LIMIT", "1000"))  # requests per minute
	api_timeout_seconds: int = int(os.getenv("DOC_API_TIMEOUT", "30"))
	cors_enabled: bool = os.getenv("DOC_CORS_ENABLED", "true").lower() == "true"
	
	def __post_init__(self):
		"""Validate configuration after initialization"""
		# Ensure storage path exists
		if self.storage_backend == "filesystem":
			Path(self.storage_path).mkdir(parents=True, exist_ok=True)
		
		# Validate numeric constraints
		assert self.max_file_size_mb > 0, "max_file_size_mb must be positive"
		assert self.processing_timeout_seconds > 0, "processing_timeout_seconds must be positive"
		assert self.similarity_threshold >= 0.0 and self.similarity_threshold <= 1.0, "similarity_threshold must be between 0 and 1"
		
		# Validate APG service names
		assert self.apg_auth_service, "APG auth service name is required"
		assert self.apg_audit_service, "APG audit service name is required"
	
	def get_storage_config(self) -> Dict[str, Any]:
		"""Get storage configuration dict"""
		return {
			"backend": self.storage_backend,
			"path": self.storage_path,
			"encryption_enabled": self.encryption_enabled,
			"backup_enabled": self.backup_enabled,
			"max_file_size_mb": self.max_file_size_mb,
			"retention_days": self.retention_days_default
		}
	
	def get_ai_config(self) -> Dict[str, Any]:
		"""Get AI processing configuration dict"""
		return {
			"ocr_enabled": self.ocr_enabled,
			"nlp_enabled": self.nlp_enabled,
			"auto_classification": self.auto_classification,
			"ollama_url": self.ollama_base_url,
			"vision_model": self.vision_model,
			"language_model": self.language_model,
			"analysis_model": self.analysis_model,
			"similarity_threshold": self.similarity_threshold,
			"processing_timeout": self.processing_timeout_seconds
		}
	
	def get_apg_config(self) -> Dict[str, Any]:
		"""Get APG integration configuration dict"""
		return {
			"base_url": self.apg_base_url,
			"auth_service": self.apg_auth_service,
			"audit_service": self.apg_audit_service,
			"vision_service": self.apg_vision_service,
			"nlp_service": self.apg_nlp_service,
			"ai_orchestration": self.apg_ai_orchestration,
			"tenant_mode": self.tenant_mode,
			"tenant_isolation": self.tenant_isolation_level
		}


# Global configuration instance
config = APGDocumentConfig()


def get_config() -> APGDocumentConfig:
	"""Get global configuration instance"""
	return config


def override_config(**kwargs) -> APGDocumentConfig:
	"""Override configuration for testing"""
	test_config = APGDocumentConfig(**kwargs)
	return test_config