"""
APG Workflow & Business Process Management - Standalone/Integrated Configuration

Flexible configuration system enabling both standalone and integrated utilization
with the APG platform ecosystem while maintaining full independence.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from uuid_extensions import uuid7str

from models import APGTenantContext, WBPMServiceResponse, APGBaseModel

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Enums and Classes
# =============================================================================

class DeploymentMode(str, Enum):
	"""Deployment mode configuration."""
	STANDALONE = "standalone"
	INTEGRATED = "integrated"
	HYBRID = "hybrid"


class AuthenticationMode(str, Enum):
	"""Authentication integration mode."""
	STANDALONE = "standalone"
	APG_RBAC = "apg_rbac"
	EXTERNAL_SSO = "external_sso"
	CUSTOM = "custom"


class DatabaseMode(str, Enum):
	"""Database configuration mode."""
	STANDALONE = "standalone"
	APG_SHARED = "apg_shared"
	HYBRID = "hybrid"


class NotificationMode(str, Enum):
	"""Notification system mode."""
	STANDALONE = "standalone"
	APG_NOTIFICATIONS = "apg_notifications"
	EXTERNAL = "external"
	HYBRID = "hybrid"


@dataclass
class StandaloneConfig(APGBaseModel):
	"""Configuration for standalone deployment."""
	config_id: str = field(default_factory=uuid7str)
	deployment_name: str = ""
	
	# Database Configuration
	database_url: str = ""
	database_pool_size: int = 20
	database_max_overflow: int = 30
	
	# Authentication Configuration
	auth_provider: str = "local"
	auth_settings: Dict[str, Any] = field(default_factory=dict)
	session_timeout_minutes: int = 480  # 8 hours
	
	# Security Configuration
	encryption_key: str = ""
	jwt_secret: str = ""
	cors_origins: List[str] = field(default_factory=list)
	
	# Notification Configuration
	email_provider: str = "smtp"
	email_settings: Dict[str, Any] = field(default_factory=dict)
	sms_provider: Optional[str] = None
	sms_settings: Dict[str, Any] = field(default_factory=dict)
	
	# File Storage Configuration
	file_storage_provider: str = "local"
	file_storage_settings: Dict[str, Any] = field(default_factory=dict)
	
	# Performance Configuration
	max_concurrent_workflows: int = 1000
	task_execution_timeout: int = 3600
	cache_provider: str = "memory"
	cache_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedConfig(APGBaseModel):
	"""Configuration for APG platform integration."""
	config_id: str = field(default_factory=uuid7str)
	apg_platform_url: str = ""
	
	# APG Service Integration
	auth_rbac_endpoint: str = ""
	audit_compliance_endpoint: str = ""
	notification_engine_endpoint: str = ""
	real_time_collaboration_endpoint: str = ""
	ai_orchestration_endpoint: str = ""
	document_management_endpoint: str = ""
	
	# Integration Authentication
	apg_api_key: str = ""
	apg_service_account: str = ""
	apg_integration_token: str = ""
	
	# Database Integration
	use_apg_database: bool = True
	apg_schema_prefix: str = "wbpm_"
	
	# Event Integration
	apg_event_bus_url: str = ""
	event_subscription_topics: List[str] = field(default_factory=list)
	event_publication_topics: List[str] = field(default_factory=list)
	
	# Data Sharing Configuration
	shared_user_directory: bool = True
	shared_tenant_configuration: bool = True
	cross_capability_workflows: bool = True


@dataclass
class HybridConfig(APGBaseModel):
	"""Configuration for hybrid deployment."""
	config_id: str = field(default_factory=uuid7str)
	
	# Service-specific integration settings
	authentication_mode: AuthenticationMode = AuthenticationMode.APG_RBAC
	database_mode: DatabaseMode = DatabaseMode.HYBRID
	notification_mode: NotificationMode = NotificationMode.HYBRID
	
	# APG Integration Settings
	integrated_services: Set[str] = field(default_factory=set)
	standalone_services: Set[str] = field(default_factory=set)
	
	# Fallback Configuration
	fallback_to_standalone: bool = True
	integration_health_check_interval: int = 60
	max_integration_failures: int = 3


@dataclass
class WBPMDeploymentConfiguration:
	"""Complete WBPM deployment configuration."""
	deployment_mode: DeploymentMode = DeploymentMode.INTEGRATED
	standalone_config: Optional[StandaloneConfig] = None
	integrated_config: Optional[IntegratedConfig] = None
	hybrid_config: Optional[HybridConfig] = None
	
	# Feature Toggles
	enable_visual_designer: bool = True
	enable_scheduling: bool = True
	enable_analytics: bool = True
	enable_collaboration: bool = True
	enable_mobile_api: bool = True
	enable_ai_optimization: bool = True
	
	# Performance Settings
	max_process_instances: int = 10000
	max_concurrent_tasks: int = 5000
	auto_cleanup_days: int = 90
	
	# Monitoring and Logging
	enable_detailed_logging: bool = True
	enable_performance_monitoring: bool = True
	enable_audit_trail: bool = True
	log_level: str = "INFO"


# =============================================================================
# Configuration Manager
# =============================================================================

class WBPMConfigurationManager:
	"""Manages WBPM deployment configuration and mode switching."""
	
	def __init__(self):
		self.current_config: Optional[WBPMDeploymentConfiguration] = None
		self.active_mode: Optional[DeploymentMode] = None
		self.integration_status: Dict[str, bool] = {}
		self.fallback_active: bool = False
		
		# Configuration validation rules
		self.validation_rules = self._initialize_validation_rules()
		
		# Service registry for mode switching
		self.service_registry: Dict[str, Dict[str, Callable]] = {
			DeploymentMode.STANDALONE: {},
			DeploymentMode.INTEGRATED: {},
			DeploymentMode.HYBRID: {}
		}
	
	
	async def load_configuration(self, config: WBPMDeploymentConfiguration) -> WBPMServiceResponse:
		"""Load and validate deployment configuration."""
		try:
			# Validate configuration
			validation_result = await self._validate_configuration(config)
			if not validation_result.success:
				return validation_result
			
			# Store configuration
			self.current_config = config
			self.active_mode = config.deployment_mode
			
			# Initialize services based on mode
			initialization_result = await self._initialize_services()
			if not initialization_result.success:
				return initialization_result
			
			logger.info(f"Configuration loaded successfully in {config.deployment_mode} mode")
			
			return WBPMServiceResponse(
				success=True,
				message="Configuration loaded successfully",
				data={
					"deployment_mode": config.deployment_mode,
					"features_enabled": self._get_enabled_features(),
					"integration_status": self.integration_status
				}
			)
			
		except Exception as e:
			logger.error(f"Error loading configuration: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to load configuration: {str(e)}"
			)
	
	
	async def switch_deployment_mode(self, new_mode: DeploymentMode) -> WBPMServiceResponse:
		"""Switch between deployment modes."""
		try:
			if not self.current_config:
				return WBPMServiceResponse(
					success=False,
					message="No configuration loaded"
				)
			
			old_mode = self.active_mode
			
			# Validate mode switch is possible
			if not await self._validate_mode_switch(old_mode, new_mode):
				return WBPMServiceResponse(
					success=False,
					message=f"Cannot switch from {old_mode} to {new_mode}"
				)
			
			# Shutdown current services
			await self._shutdown_services(old_mode)
			
			# Update active mode
			self.active_mode = new_mode
			self.current_config.deployment_mode = new_mode
			
			# Initialize new services
			initialization_result = await self._initialize_services()
			if not initialization_result.success:
				# Rollback on failure
				self.active_mode = old_mode
				self.current_config.deployment_mode = old_mode
				await self._initialize_services()
				return initialization_result
			
			logger.info(f"Successfully switched from {old_mode} to {new_mode}")
			
			return WBPMServiceResponse(
				success=True,
				message=f"Successfully switched to {new_mode} mode",
				data={
					"old_mode": old_mode,
					"new_mode": new_mode,
					"integration_status": self.integration_status
				}
			)
			
		except Exception as e:
			logger.error(f"Error switching deployment mode: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to switch deployment mode: {str(e)}"
			)
	
	
	async def get_service_provider(self, service_name: str) -> Optional[Callable]:
		"""Get the appropriate service provider for current mode."""
		if not self.active_mode or not self.current_config:
			return None
		
		providers = self.service_registry.get(self.active_mode, {})
		return providers.get(service_name)
	
	
	async def check_integration_health(self) -> WBPMServiceResponse:
		"""Check health of APG platform integrations."""
		try:
			health_status = {}
			
			if self.active_mode in [DeploymentMode.INTEGRATED, DeploymentMode.HYBRID]:
				integrated_config = self.current_config.integrated_config
				if integrated_config:
					# Check APG service endpoints
					health_status.update(await self._check_apg_services(integrated_config))
			
			# Update integration status
			self.integration_status.update(health_status)
			
			# Check if fallback needed
			if self.active_mode == DeploymentMode.HYBRID:
				await self._check_fallback_conditions()
			
			return WBPMServiceResponse(
				success=True,
				message="Integration health check completed",
				data={
					"health_status": health_status,
					"fallback_active": self.fallback_active,
					"overall_health": all(health_status.values())
				}
			)
			
		except Exception as e:
			logger.error(f"Error checking integration health: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to check integration health: {str(e)}"
			)
	
	
	# =============================================================================
	# Configuration Builders
	# =============================================================================
	
	@staticmethod
	def create_standalone_configuration(
		deployment_name: str,
		database_url: str,
		auth_provider: str = "local",
		email_settings: Optional[Dict[str, Any]] = None
	) -> WBPMDeploymentConfiguration:
		"""Create a standalone deployment configuration."""
		standalone_config = StandaloneConfig(
			deployment_name=deployment_name,
			database_url=database_url,
			auth_provider=auth_provider,
			email_settings=email_settings or {}
		)
		
		return WBPMDeploymentConfiguration(
			deployment_mode=DeploymentMode.STANDALONE,
			standalone_config=standalone_config
		)
	
	
	@staticmethod
	def create_integrated_configuration(
		apg_platform_url: str,
		apg_api_key: str,
		services_to_integrate: Optional[List[str]] = None
	) -> WBPMDeploymentConfiguration:
		"""Create an APG-integrated deployment configuration."""
		integrated_config = IntegratedConfig(
			apg_platform_url=apg_platform_url,
			apg_api_key=apg_api_key,
			auth_rbac_endpoint=f"{apg_platform_url}/api/auth",
			audit_compliance_endpoint=f"{apg_platform_url}/api/audit",
			notification_engine_endpoint=f"{apg_platform_url}/api/notifications",
			real_time_collaboration_endpoint=f"{apg_platform_url}/api/collaboration",
			ai_orchestration_endpoint=f"{apg_platform_url}/api/ai"
		)
		
		return WBPMDeploymentConfiguration(
			deployment_mode=DeploymentMode.INTEGRATED,
			integrated_config=integrated_config
		)
	
	
	@staticmethod
	def create_hybrid_configuration(
		apg_platform_url: str,
		apg_api_key: str,
		integrated_services: Optional[Set[str]] = None,
		standalone_fallback: bool = True
	) -> WBPMDeploymentConfiguration:
		"""Create a hybrid deployment configuration."""
		integrated_config = IntegratedConfig(
			apg_platform_url=apg_platform_url,
			apg_api_key=apg_api_key,
			auth_rbac_endpoint=f"{apg_platform_url}/api/auth",
			notification_engine_endpoint=f"{apg_platform_url}/api/notifications"
		)
		
		standalone_config = StandaloneConfig(
			deployment_name="WBPM Hybrid Fallback",
			auth_provider="local",
			email_provider="smtp"
		)
		
		hybrid_config = HybridConfig(
			integrated_services=integrated_services or {"auth", "notifications"},
			fallback_to_standalone=standalone_fallback
		)
		
		return WBPMDeploymentConfiguration(
			deployment_mode=DeploymentMode.HYBRID,
			integrated_config=integrated_config,
			standalone_config=standalone_config,
			hybrid_config=hybrid_config
		)
	
	
	# =============================================================================
	# Private Implementation Methods
	# =============================================================================
	
	async def _validate_configuration(self, config: WBPMDeploymentConfiguration) -> WBPMServiceResponse:
		"""Validate deployment configuration."""
		try:
			errors = []
			
			# Mode-specific validation
			if config.deployment_mode == DeploymentMode.STANDALONE:
				if not config.standalone_config:
					errors.append("Standalone configuration required for standalone mode")
				elif not config.standalone_config.database_url:
					errors.append("Database URL required for standalone mode")
			
			elif config.deployment_mode == DeploymentMode.INTEGRATED:
				if not config.integrated_config:
					errors.append("Integrated configuration required for integrated mode")
				elif not config.integrated_config.apg_platform_url:
					errors.append("APG platform URL required for integrated mode")
			
			elif config.deployment_mode == DeploymentMode.HYBRID:
				if not config.hybrid_config:
					errors.append("Hybrid configuration required for hybrid mode")
				if not config.integrated_config and not config.standalone_config:
					errors.append("Both integrated and standalone configs required for hybrid mode")
			
			if errors:
				return WBPMServiceResponse(
					success=False,
					message="Configuration validation failed",
					data={"errors": errors}
				)
			
			return WBPMServiceResponse(
				success=True,
				message="Configuration validation passed"
			)
			
		except Exception as e:
			logger.error(f"Error validating configuration: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Configuration validation error: {str(e)}"
			)
	
	
	async def _initialize_services(self) -> WBPMServiceResponse:
		"""Initialize services based on current configuration."""
		try:
			if not self.current_config or not self.active_mode:
				return WBPMServiceResponse(
					success=False,
					message="No configuration or mode set"
				)
			
			# Initialize based on deployment mode
			if self.active_mode == DeploymentMode.STANDALONE:
				await self._initialize_standalone_services()
			elif self.active_mode == DeploymentMode.INTEGRATED:
				await self._initialize_integrated_services()
			elif self.active_mode == DeploymentMode.HYBRID:
				await self._initialize_hybrid_services()
			
			logger.info(f"Services initialized for {self.active_mode} mode")
			
			return WBPMServiceResponse(
				success=True,
				message="Services initialized successfully"
			)
			
		except Exception as e:
			logger.error(f"Error initializing services: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to initialize services: {str(e)}"
			)
	
	
	async def _initialize_standalone_services(self) -> None:
		"""Initialize services for standalone mode."""
		config = self.current_config.standalone_config
		
		# Register standalone service providers
		self.service_registry[DeploymentMode.STANDALONE] = {
			"auth": self._create_standalone_auth_service,
			"database": self._create_standalone_database_service,
			"notifications": self._create_standalone_notification_service,
			"file_storage": self._create_standalone_storage_service,
			"cache": self._create_standalone_cache_service
		}
		
		logger.info("Standalone services registered")
	
	
	async def _initialize_integrated_services(self) -> None:
		"""Initialize services for integrated mode."""
		config = self.current_config.integrated_config
		
		# Register integrated service providers
		self.service_registry[DeploymentMode.INTEGRATED] = {
			"auth": self._create_integrated_auth_service,
			"database": self._create_integrated_database_service,
			"notifications": self._create_integrated_notification_service,
			"audit": self._create_integrated_audit_service,
			"collaboration": self._create_integrated_collaboration_service,
			"ai": self._create_integrated_ai_service
		}
		
		# Test APG integrations
		self.integration_status = await self._check_apg_services(config)
		
		logger.info("Integrated services registered")
	
	
	async def _initialize_hybrid_services(self) -> None:
		"""Initialize services for hybrid mode."""
		integrated_config = self.current_config.integrated_config
		standalone_config = self.current_config.standalone_config
		hybrid_config = self.current_config.hybrid_config
		
		# Register hybrid service providers
		self.service_registry[DeploymentMode.HYBRID] = {}
		
		# Configure integrated services
		for service in hybrid_config.integrated_services:
			if service == "auth":
				self.service_registry[DeploymentMode.HYBRID]["auth"] = self._create_hybrid_auth_service
			elif service == "notifications":
				self.service_registry[DeploymentMode.HYBRID]["notifications"] = self._create_hybrid_notification_service
		
		# Configure standalone fallbacks
		for service in hybrid_config.standalone_services:
			if service == "database":
				self.service_registry[DeploymentMode.HYBRID]["database"] = self._create_standalone_database_service
			elif service == "file_storage":
				self.service_registry[DeploymentMode.HYBRID]["file_storage"] = self._create_standalone_storage_service
		
		logger.info("Hybrid services registered")
	
	
	async def _shutdown_services(self, mode: DeploymentMode) -> None:
		"""Shutdown services for a specific mode."""
		try:
			# Clear service registry for the mode
			if mode in self.service_registry:
				self.service_registry[mode].clear()
			
			logger.info(f"Services shutdown for {mode} mode")
			
		except Exception as e:
			logger.error(f"Error shutting down services: {e}")
	
	
	async def _validate_mode_switch(self, old_mode: DeploymentMode, new_mode: DeploymentMode) -> bool:
		"""Validate if mode switch is allowed."""
		# Define allowed transitions
		allowed_transitions = {
			DeploymentMode.STANDALONE: [DeploymentMode.INTEGRATED, DeploymentMode.HYBRID],
			DeploymentMode.INTEGRATED: [DeploymentMode.STANDALONE, DeploymentMode.HYBRID],
			DeploymentMode.HYBRID: [DeploymentMode.STANDALONE, DeploymentMode.INTEGRATED]
		}
		
		return new_mode in allowed_transitions.get(old_mode, [])
	
	
	async def _check_apg_services(self, config: IntegratedConfig) -> Dict[str, bool]:
		"""Check availability of APG platform services."""
		service_status = {}
		
		# In production, would make actual HTTP calls to check service health
		services = [
			"auth_rbac",
			"audit_compliance", 
			"notification_engine",
			"real_time_collaboration",
			"ai_orchestration"
		]
		
		for service in services:
			# Simulate health check
			service_status[service] = True  # In production, would check actual endpoint
		
		return service_status
	
	
	async def _check_fallback_conditions(self) -> None:
		"""Check if fallback to standalone is needed in hybrid mode."""
		if not self.current_config.hybrid_config.fallback_to_standalone:
			return
		
		# Check integration health
		failed_services = [
			service for service, status in self.integration_status.items()
			if not status
		]
		
		# Activate fallback if too many services are failing
		if len(failed_services) >= self.current_config.hybrid_config.max_integration_failures:
			self.fallback_active = True
			logger.warning(f"Fallback activated due to failed services: {failed_services}")
		else:
			self.fallback_active = False
	
	
	def _get_enabled_features(self) -> Dict[str, bool]:
		"""Get list of enabled features."""
		if not self.current_config:
			return {}
		
		return {
			"visual_designer": self.current_config.enable_visual_designer,
			"scheduling": self.current_config.enable_scheduling,
			"analytics": self.current_config.enable_analytics,
			"collaboration": self.current_config.enable_collaboration,
			"mobile_api": self.current_config.enable_mobile_api,
			"ai_optimization": self.current_config.enable_ai_optimization
		}
	
	
	def _initialize_validation_rules(self) -> Dict[str, Any]:
		"""Initialize configuration validation rules."""
		return {
			"required_standalone_fields": ["database_url", "auth_provider"],
			"required_integrated_fields": ["apg_platform_url", "apg_api_key"],
			"required_hybrid_fields": ["integrated_services", "standalone_services"],
			"max_concurrent_workflows": 50000,
			"min_session_timeout": 15,
			"max_session_timeout": 1440
		}
	
	
	# =============================================================================
	# Service Creation Methods (Stubs for actual implementation)
	# =============================================================================
	
	def _create_standalone_auth_service(self):
		"""Create standalone authentication service."""
		# In production, would create actual auth service
		return "StandaloneAuthService"
	
	def _create_standalone_database_service(self):
		"""Create standalone database service."""
		return "StandaloneDatabaseService"
	
	def _create_standalone_notification_service(self):
		"""Create standalone notification service."""
		return "StandaloneNotificationService"
	
	def _create_standalone_storage_service(self):
		"""Create standalone file storage service."""
		return "StandaloneStorageService"
	
	def _create_standalone_cache_service(self):
		"""Create standalone cache service."""
		return "StandaloneCacheService"
	
	def _create_integrated_auth_service(self):
		"""Create APG-integrated authentication service."""
		return "APGAuthService"
	
	def _create_integrated_database_service(self):
		"""Create APG-integrated database service."""
		return "APGDatabaseService"
	
	def _create_integrated_notification_service(self):
		"""Create APG-integrated notification service."""
		return "APGNotificationService"
	
	def _create_integrated_audit_service(self):
		"""Create APG-integrated audit service."""
		return "APGAuditService"
	
	def _create_integrated_collaboration_service(self):
		"""Create APG-integrated collaboration service."""
		return "APGCollaborationService"
	
	def _create_integrated_ai_service(self):
		"""Create APG-integrated AI service."""
		return "APGAIService"
	
	def _create_hybrid_auth_service(self):
		"""Create hybrid authentication service with fallback."""
		return "HybridAuthService"
	
	def _create_hybrid_notification_service(self):
		"""Create hybrid notification service with fallback."""
		return "HybridNotificationService"


# =============================================================================
# Configuration Factory
# =============================================================================

class WBPMConfigurationFactory:
	"""Factory for creating WBPM configurations."""
	
	@staticmethod
	def create_development_config() -> WBPMDeploymentConfiguration:
		"""Create configuration for development environment."""
		return WBPMConfigurationManager.create_standalone_configuration(
			deployment_name="WBPM Development",
			database_url="postgresql://localhost:5432/wbpm_dev",
			auth_provider="local",
			email_settings={
				"smtp_host": "localhost",
				"smtp_port": 1025,  # MailHog for development
				"use_tls": False
			}
		)
	
	@staticmethod
	def create_production_standalone_config(
		database_url: str,
		smtp_host: str,
		smtp_user: str,
		smtp_password: str
	) -> WBPMDeploymentConfiguration:
		"""Create production standalone configuration."""
		return WBPMConfigurationManager.create_standalone_configuration(
			deployment_name="WBPM Production Standalone",
			database_url=database_url,
			auth_provider="ldap",
			email_settings={
				"smtp_host": smtp_host,
				"smtp_port": 587,
				"smtp_user": smtp_user,
				"smtp_password": smtp_password,
				"use_tls": True
			}
		)
	
	@staticmethod
	def create_production_integrated_config(
		apg_platform_url: str,
		apg_api_key: str
	) -> WBPMDeploymentConfiguration:
		"""Create production APG-integrated configuration."""
		return WBPMConfigurationManager.create_integrated_configuration(
			apg_platform_url=apg_platform_url,
			apg_api_key=apg_api_key,
			services_to_integrate=["auth", "audit", "notifications", "collaboration", "ai"]
		)
	
	@staticmethod
	def create_enterprise_hybrid_config(
		apg_platform_url: str,
		apg_api_key: str,
		fallback_database_url: str
	) -> WBPMDeploymentConfiguration:
		"""Create enterprise hybrid configuration."""
		config = WBPMConfigurationManager.create_hybrid_configuration(
			apg_platform_url=apg_platform_url,
			apg_api_key=apg_api_key,
			integrated_services={"auth", "notifications", "collaboration"},
			standalone_fallback=True
		)
		
		# Configure fallback database
		config.standalone_config.database_url = fallback_database_url
		
		return config


# =============================================================================
# Example Usage
# =============================================================================

async def example_configuration_usage():
	"""Example usage of WBPM configuration system."""
	
	# Create configuration manager
	config_manager = WBPMConfigurationManager()
	
	# Example 1: Standalone deployment
	standalone_config = WBPMConfigurationFactory.create_development_config()
	result = await config_manager.load_configuration(standalone_config)
	print(f"Standalone config loaded: {result.success}")
	
	# Example 2: Switch to integrated mode
	integrated_config = WBPMConfigurationFactory.create_production_integrated_config(
		apg_platform_url="https://apg.example.com",
		apg_api_key="your-api-key-here"
	)
	
	switch_result = await config_manager.load_configuration(integrated_config)
	print(f"Integrated config loaded: {switch_result.success}")
	
	# Example 3: Check integration health
	health_result = await config_manager.check_integration_health()
	print(f"Integration health: {health_result.data}")
	
	# Example 4: Hybrid configuration with fallback
	hybrid_config = WBPMConfigurationFactory.create_enterprise_hybrid_config(
		apg_platform_url="https://apg.example.com",
		apg_api_key="your-api-key-here",
		fallback_database_url="postgresql://localhost:5432/wbpm_fallback"
	)
	
	hybrid_result = await config_manager.load_configuration(hybrid_config)
	print(f"Hybrid config loaded: {hybrid_result.success}")


if __name__ == "__main__":
	asyncio.run(example_configuration_usage())