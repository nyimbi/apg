"""
APG Central Configuration - Capability Applet System

Interactive applets for managing individual APG capabilities through the central interface.
Each capability gets its own management applet with tailored UI and functionality.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import httpx
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Flask-AppBuilder for web interface
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.widgets import ListWidget, ShowWidget
from wtforms import Form, StringField, TextAreaField, SelectField, BooleanField, IntegerField
from wtforms.validators import DataRequired, Length, Optional as OptionalValidator

# Capability management
from .capability_manager import APGCapabilityManager, CapabilitySpec, CapabilityHealth, CapabilityStatus


class AppletType(Enum):
	"""Types of capability applets."""
	DASHBOARD = "dashboard"
	CONFIGURATION = "configuration"
	MONITORING = "monitoring"
	DEPLOYMENT = "deployment"
	ANALYTICS = "analytics"
	COLLABORATION = "collaboration"


@dataclass
class AppletWidget:
	"""Individual widget within an applet."""
	widget_id: str
	title: str
	widget_type: str  # chart, table, form, status, metric
	config: Dict[str, Any]
	data_source: str
	refresh_interval: int = 30  # seconds
	span: int = 6  # 1-12 grid columns


@dataclass
class AppletLayout:
	"""Layout configuration for an applet."""
	rows: List[List[AppletWidget]]
	theme: str = "default"
	responsive: bool = True


class BaseCapabilityApplet(ABC):
	"""Base class for capability management applets."""
	
	def __init__(
		self,
		capability_id: str,
		capability_spec: CapabilitySpec,
		capability_manager: APGCapabilityManager
	):
		"""Initialize base applet."""
		self.capability_id = capability_id
		self.capability_spec = capability_spec
		self.capability_manager = capability_manager
		
		# Applet metadata
		self.applet_id = f"applet_{capability_id}"
		self.title = f"{capability_spec.name} Management"
		self.description = capability_spec.description
		self.category = capability_spec.category
		self.version = capability_spec.version
		
		# UI configuration
		self.layout: Optional[AppletLayout] = None
		self.custom_routes: List[str] = []
		self.permissions: List[str] = ["read", "write"]
		
		# Initialize applet
		asyncio.create_task(self._initialize_applet())
	
	async def _initialize_applet(self):
		"""Initialize applet-specific functionality."""
		try:
			# Initialize capability connection
			await self._establish_capability_connection()
			
			# Load capability metadata
			await self._load_capability_metadata()
			
			# Setup monitoring endpoints
			await self._setup_monitoring()
			
			# Initialize configuration cache
			await self._initialize_configuration_cache()
			
			self.logger.info(f"Applet initialized successfully for {self.capability_name}")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize applet {self.capability_name}: {e}")
			raise
	
	async def get_dashboard_data(self) -> Dict[str, Any]:
		"""Get comprehensive dashboard data for the applet."""
		try:
			# Get basic capability info
			capability_info = await self._get_capability_info()
			
			# Get deployment status across all environments
			deployment_status = await self._get_deployment_status()
			
			# Get performance metrics
			performance_metrics = await self._get_performance_metrics()
			
			# Get recent activities
			recent_activities = await self._get_recent_activities()
			
			# Get configuration summary
			config_summary = await self._get_configuration_summary()
			
			# Get alerts and issues
			alerts = await self._get_alerts()
			
			return {
				"capability_info": capability_info,
				"deployment_status": deployment_status,
				"performance_metrics": performance_metrics,
				"recent_activities": recent_activities,
				"configuration_summary": config_summary,
				"alerts": alerts,
				"last_updated": datetime.now(timezone.utc).isoformat(),
				"applet_version": "1.0.0"
			}
			
		except Exception as e:
			self.logger.error(f"Failed to get dashboard data for {self.capability_name}: {e}")
			return {
				"error": str(e),
				"capability_info": {"name": self.capability_name, "status": "error"},
				"last_updated": datetime.now(timezone.utc).isoformat()
			}
	
	async def get_configuration_schema(self) -> Dict[str, Any]:
		"""Get comprehensive configuration schema for the capability."""
		try:
			# Load base schema from capability
			base_schema = await self._load_base_configuration_schema()
			
			# Get environment-specific configurations
			env_configs = await self._get_environment_configurations()
			
			# Get deployment-specific options
			deployment_options = await self._get_deployment_options()
			
			# Get security requirements
			security_requirements = await self._get_security_requirements()
			
			# Build comprehensive schema
			schema = {
				"$schema": "http://json-schema.org/draft-07/schema#",
				"title": f"{self.capability_name} Configuration Schema",
				"description": f"Complete configuration schema for {self.capability_name} capability",
				"type": "object",
				"properties": {
					"basic": {
						"title": "Basic Configuration",
						"type": "object",
						"properties": base_schema.get("properties", {})
					},
					"environments": {
						"title": "Environment Configurations",
						"type": "object",
						"properties": env_configs
					},
					"deployment": {
						"title": "Deployment Options",
						"type": "object",
						"properties": deployment_options
					},
					"security": {
						"title": "Security Settings",
						"type": "object",
						"properties": security_requirements
					},
					"monitoring": {
						"title": "Monitoring Configuration",
						"type": "object",
						"properties": {
							"metrics_enabled": {"type": "boolean", "default": True},
							"log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARN", "ERROR"], "default": "INFO"},
							"health_check_interval": {"type": "integer", "minimum": 10, "default": 30},
							"alerts": {
								"type": "object",
								"properties": {
									"email_notifications": {"type": "boolean", "default": True},
									"slack_webhook": {"type": "string", "format": "uri"},
									"severity_threshold": {"type": "string", "enum": ["INFO", "WARN", "ERROR", "CRITICAL"], "default": "WARN"}
								}
							}
						}
					}
				},
				"required": base_schema.get("required", [])
			}
			
			return schema
			
		except Exception as e:
			self.logger.error(f"Failed to get configuration schema for {self.capability_name}: {e}")
			return {
				"error": str(e),
				"type": "object",
				"properties": {}
			}
	
	async def update_configuration(
		self,
		deployment_id: str,
		configuration: Dict[str, Any]
	) -> bool:
		"""Update capability configuration with comprehensive validation and rollback."""
		try:
			# Validate configuration against schema
			schema = await self.get_configuration_schema()
			validation_result = await self._validate_configuration(configuration, schema)
			
			if not validation_result["valid"]:
				self.logger.error(f"Configuration validation failed: {validation_result['errors']}")
				return False
			
			# Backup current configuration
			backup_id = await self._backup_current_configuration(deployment_id)
			
			try:
				# Apply configuration update
				success = await self._apply_configuration_update(deployment_id, configuration)
				
				if success:
					# Verify deployment health after update
					health_check = await self._verify_deployment_health(deployment_id)
					
					if health_check["healthy"]:
						# Update configuration cache
						await self._update_configuration_cache(deployment_id, configuration)
						
						# Log successful update
						await self._log_configuration_update(deployment_id, configuration, "success")
						
						self.logger.info(f"Configuration updated successfully for deployment {deployment_id}")
						return True
					else:
						# Health check failed, rollback
						self.logger.warning(f"Health check failed after configuration update, rolling back")
						await self._rollback_configuration(deployment_id, backup_id)
						return False
				else:
					# Update failed, rollback
					await self._rollback_configuration(deployment_id, backup_id)
					return False
					
			except Exception as e:
				# Exception during update, rollback
				self.logger.error(f"Exception during configuration update: {e}")
				await self._rollback_configuration(deployment_id, backup_id)
				return False
				
		except Exception as e:
			self.logger.error(f"Failed to update configuration for deployment {deployment_id}: {e}")
			return False
	
	# ==================== Helper Methods ====================
	
	async def _establish_capability_connection(self):
		"""Establish connection to the capability service."""
		try:
			# Build capability service URL
			service_url = f"http://{self.capability_name.lower().replace(' ', '-')}-service:8080"
			
			# Test connection with health check
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.get(f"{service_url}/health")
				if response.status_code == 200:
					self.capability_endpoint = service_url
					self.connection_healthy = True
				else:
					raise Exception(f"Health check failed with status {response.status_code}")
					
		except Exception as e:
			self.logger.warning(f"Could not establish direct connection to {self.capability_name}: {e}")
			# Fallback to capability manager API
			self.capability_endpoint = f"http://capability-manager:8080/api/v1/capabilities/{self.capability_id}"
			self.connection_healthy = False
	
	async def _load_capability_metadata(self):
		"""Load capability metadata and configuration."""
		try:
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.get(f"{self.capability_endpoint}/metadata")
				if response.status_code == 200:
					self.metadata = response.json()
				else:
					# Create basic metadata
					self.metadata = {
						"name": self.capability_name,
						"version": "1.0.0",
						"description": f"APG {self.capability_name} Capability",
						"api_version": "v1",
						"endpoints": [],
						"health_check_path": "/health"
					}
		except Exception as e:
			self.logger.warning(f"Could not load metadata for {self.capability_name}: {e}")
			self.metadata = {"name": self.capability_name, "version": "unknown"}
	
	async def _setup_monitoring(self):
		"""Setup monitoring endpoints and metrics collection."""
		try:
			# Initialize metrics collector
			self.metrics_collector = {
				"performance": {},
				"health": {},
				"activities": []
			}
			
			# Setup health check scheduler
			self.health_check_task = asyncio.create_task(self._periodic_health_check())
			
		except Exception as e:
			self.logger.error(f"Failed to setup monitoring for {self.capability_name}: {e}")
	
	async def _initialize_configuration_cache(self):
		"""Initialize configuration cache for fast access."""
		try:
			self.config_cache = {}
			
			# Load current configurations from capability
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.get(f"{self.capability_endpoint}/configurations")
				if response.status_code == 200:
					configs = response.json()
					for config in configs:
						self.config_cache[config.get("deployment_id", "default")] = config
						
		except Exception as e:
			self.logger.warning(f"Could not initialize config cache for {self.capability_name}: {e}")
			self.config_cache = {}
	
	async def _get_capability_info(self) -> Dict[str, Any]:
		"""Get basic capability information."""
		return {
			"name": self.capability_name,
			"version": self.metadata.get("version", "unknown"),
			"status": "healthy" if self.connection_healthy else "degraded",
			"endpoint": self.capability_endpoint,
			"last_contact": datetime.now(timezone.utc).isoformat(),
			"metadata": self.metadata
		}
	
	async def _get_deployment_status(self) -> Dict[str, Any]:
		"""Get deployment status across all environments."""
		try:
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.get(f"{self.capability_endpoint}/deployments")
				if response.status_code == 200:
					deployments = response.json()
					
					status_summary = {
						"total_deployments": len(deployments),
						"healthy": 0,
						"degraded": 0,
						"unhealthy": 0,
						"deployments": []
					}
					
					for deployment in deployments:
						health_status = deployment.get("health", {}).get("status", "unknown")
						if health_status == "healthy":
							status_summary["healthy"] += 1
						elif health_status == "degraded":
							status_summary["degraded"] += 1
						else:
							status_summary["unhealthy"] += 1
						
						status_summary["deployments"].append({
							"id": deployment.get("id"),
							"environment": deployment.get("environment"),
							"region": deployment.get("region"),
							"status": health_status,
							"last_updated": deployment.get("last_updated")
						})
					
					return status_summary
				else:
					return {"total_deployments": 0, "healthy": 0, "degraded": 0, "unhealthy": 0, "deployments": []}
					
		except Exception as e:
			self.logger.error(f"Failed to get deployment status: {e}")
			return {"error": str(e), "total_deployments": 0, "healthy": 0, "degraded": 0, "unhealthy": 0}
	
	async def _get_performance_metrics(self) -> Dict[str, Any]:
		"""Get performance metrics for the capability."""
		try:
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.get(f"{self.capability_endpoint}/metrics")
				if response.status_code == 200:
					raw_metrics = response.json()
					
					# Process and summarize metrics
					processed_metrics = {
						"response_time": {
							"avg": raw_metrics.get("response_time_avg", 0),
							"p95": raw_metrics.get("response_time_p95", 0),
							"p99": raw_metrics.get("response_time_p99", 0)
						},
						"throughput": {
							"requests_per_second": raw_metrics.get("requests_per_second", 0),
							"total_requests": raw_metrics.get("total_requests", 0)
						},
						"error_rate": raw_metrics.get("error_rate", 0),
						"resource_usage": {
							"cpu_percent": raw_metrics.get("cpu_usage", 0),
							"memory_percent": raw_metrics.get("memory_usage", 0),
							"disk_usage": raw_metrics.get("disk_usage", 0)
						},
						"last_updated": datetime.now(timezone.utc).isoformat()
					}
					
					return processed_metrics
				else:
					return self._get_default_metrics()
					
		except Exception as e:
			self.logger.error(f"Failed to get performance metrics: {e}")
			return self._get_default_metrics()
	
	async def _get_recent_activities(self) -> List[Dict[str, Any]]:
		"""Get recent activities and events."""
		try:
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.get(f"{self.capability_endpoint}/activities?limit=10")
				if response.status_code == 200:
					activities = response.json()
					
					# Process activities
					processed_activities = []
					for activity in activities:
						processed_activities.append({
							"id": activity.get("id"),
							"type": activity.get("type"),
							"description": activity.get("description"),
							"timestamp": activity.get("timestamp"),
							"user": activity.get("user"),
							"severity": activity.get("severity", "info")
						})
					
					return processed_activities
				else:
					return []
					
		except Exception as e:
			self.logger.error(f"Failed to get recent activities: {e}")
			return []
	
	async def _get_configuration_summary(self) -> Dict[str, Any]:
		"""Get configuration summary."""
		try:
			config_count = len(self.config_cache)
			
			# Analyze configuration types
			config_types = {}
			for config in self.config_cache.values():
				config_type = config.get("type", "unknown")
				config_types[config_type] = config_types.get(config_type, 0) + 1
			
			return {
				"total_configurations": config_count,
				"by_type": config_types,
				"last_updated": max(
					[config.get("updated_at", "1970-01-01T00:00:00Z") for config in self.config_cache.values()],
					default="1970-01-01T00:00:00Z"
				)
			}
			
		except Exception as e:
			self.logger.error(f"Failed to get configuration summary: {e}")
			return {"total_configurations": 0, "by_type": {}}
	
	async def _get_alerts(self) -> List[Dict[str, Any]]:
		"""Get active alerts and issues."""
		try:
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.get(f"{self.capability_endpoint}/alerts?active=true")
				if response.status_code == 200:
					alerts = response.json()
					
					processed_alerts = []
					for alert in alerts:
						processed_alerts.append({
							"id": alert.get("id"),
							"title": alert.get("title"),
							"severity": alert.get("severity"),
							"description": alert.get("description"),
							"created_at": alert.get("created_at"),
							"acknowledged": alert.get("acknowledged", False)
						})
					
					return processed_alerts
				else:
					return []
					
		except Exception as e:
			self.logger.error(f"Failed to get alerts: {e}")
			return []
	
	async def _load_base_configuration_schema(self) -> Dict[str, Any]:
		"""Load base configuration schema from capability."""
		try:
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.get(f"{self.capability_endpoint}/schema")
				if response.status_code == 200:
					return response.json()
				else:
					# Return default schema
					return {
						"type": "object",
						"properties": {
							"name": {"type": "string", "description": "Configuration name"},
							"environment": {"type": "string", "enum": ["dev", "staging", "prod"], "default": "dev"},
							"enabled": {"type": "boolean", "default": True}
						},
						"required": ["name"]
					}
					
		except Exception as e:
			self.logger.error(f"Failed to load base schema: {e}")
			return {"type": "object", "properties": {}}
	
	async def _get_environment_configurations(self) -> Dict[str, Any]:
		"""Get environment-specific configuration options."""
		return {
			"development": {
				"type": "object",
				"properties": {
					"debug_mode": {"type": "boolean", "default": True},
					"log_level": {"type": "string", "enum": ["DEBUG", "INFO"], "default": "DEBUG"},
					"mock_external_services": {"type": "boolean", "default": True}
				}
			},
			"staging": {
				"type": "object",
				"properties": {
					"debug_mode": {"type": "boolean", "default": False},
					"log_level": {"type": "string", "enum": ["INFO", "WARN"], "default": "INFO"},
					"load_test_data": {"type": "boolean", "default": True}
				}
			},
			"production": {
				"type": "object",
				"properties": {
					"debug_mode": {"type": "boolean", "default": False},
					"log_level": {"type": "string", "enum": ["WARN", "ERROR"], "default": "WARN"},
					"high_availability": {"type": "boolean", "default": True},
					"backup_enabled": {"type": "boolean", "default": True}
				}
			}
		}
	
	async def _get_deployment_options(self) -> Dict[str, Any]:
		"""Get deployment-specific options."""
		return {
			"replicas": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
			"resource_limits": {
				"type": "object",
				"properties": {
					"cpu": {"type": "string", "pattern": "^[0-9]+m?$", "default": "500m"},
					"memory": {"type": "string", "pattern": "^[0-9]+[GMK]i?$", "default": "1Gi"}
				}
			},
			"scaling": {
				"type": "object",
				"properties": {
					"min_replicas": {"type": "integer", "minimum": 1, "default": 1},
					"max_replicas": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
					"target_cpu_utilization": {"type": "integer", "minimum": 1, "maximum": 100, "default": 70}
				}
			},
			"health_checks": {
				"type": "object",
				"properties": {
					"liveness_probe_path": {"type": "string", "default": "/health/live"},
					"readiness_probe_path": {"type": "string", "default": "/health/ready"},
					"initial_delay_seconds": {"type": "integer", "minimum": 0, "default": 30}
				}
			}
		}
	
	async def _get_security_requirements(self) -> Dict[str, Any]:
		"""Get security-specific configuration requirements."""
		return {
			"encryption": {
				"type": "object",
				"properties": {
					"enabled": {"type": "boolean", "default": True},
					"algorithm": {"type": "string", "enum": ["AES-256-GCM", "ChaCha20-Poly1305"], "default": "AES-256-GCM"},
					"key_rotation_days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 90}
				}
			},
			"authentication": {
				"type": "object",
				"properties": {
					"method": {"type": "string", "enum": ["jwt", "oauth2", "api_key"], "default": "jwt"},
					"token_expiry_hours": {"type": "integer", "minimum": 1, "maximum": 24, "default": 8}
				}
			},
			"access_control": {
				"type": "object",
				"properties": {
					"rbac_enabled": {"type": "boolean", "default": True},
					"default_permissions": {"type": "array", "items": {"type": "string"}, "default": ["read"]},
					"admin_only_operations": {"type": "array", "items": {"type": "string"}, "default": ["delete", "admin"]}
				}
			}
		}
	
	async def _validate_configuration(self, configuration: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate configuration against schema."""
		try:
			# Basic validation - in production would use jsonschema library
			errors = []
			
			# Check required fields
			required_fields = schema.get("required", [])
			for field in required_fields:
				if field not in configuration:
					errors.append(f"Missing required field: {field}")
			
			# Check data types for basic properties
			properties = schema.get("properties", {})
			for key, value in configuration.items():
				if key in properties:
					prop_schema = properties[key]
					expected_type = prop_schema.get("type")
					
					if expected_type == "string" and not isinstance(value, str):
						errors.append(f"Field '{key}' must be a string")
					elif expected_type == "integer" and not isinstance(value, int):
						errors.append(f"Field '{key}' must be an integer")
					elif expected_type == "boolean" and not isinstance(value, bool):
						errors.append(f"Field '{key}' must be a boolean")
					elif expected_type == "object" and not isinstance(value, dict):
						errors.append(f"Field '{key}' must be an object")
			
			return {
				"valid": len(errors) == 0,
				"errors": errors
			}
			
		except Exception as e:
			return {
				"valid": False,
				"errors": [f"Validation error: {str(e)}"]
			}
	
	async def _backup_current_configuration(self, deployment_id: str) -> str:
		"""Backup current configuration before update."""
		try:
			backup_id = f"backup_{deployment_id}_{int(datetime.now(timezone.utc).timestamp())}"
			
			# Get current configuration
			current_config = self.config_cache.get(deployment_id, {})
			
			# Store backup (in production would store in persistent storage)
			self.config_backups = getattr(self, 'config_backups', {})
			self.config_backups[backup_id] = {
				"deployment_id": deployment_id,
				"configuration": current_config.copy(),
				"timestamp": datetime.now(timezone.utc).isoformat()
			}
			
			return backup_id
			
		except Exception as e:
			self.logger.error(f"Failed to backup configuration: {e}")
			return ""
	
	async def _apply_configuration_update(self, deployment_id: str, configuration: Dict[str, Any]) -> bool:
		"""Apply configuration update to the capability."""
		try:
			async with httpx.AsyncClient(timeout=30.0) as client:
				response = await client.put(
					f"{self.capability_endpoint}/deployments/{deployment_id}/configuration",
					json=configuration
				)
				return response.status_code == 200
				
		except Exception as e:
			self.logger.error(f"Failed to apply configuration update: {e}")
			return False
	
	async def _verify_deployment_health(self, deployment_id: str) -> Dict[str, Any]:
		"""Verify deployment health after configuration update."""
		try:
			# Wait a moment for changes to take effect
			await asyncio.sleep(5)
			
			async with httpx.AsyncClient(timeout=20.0) as client:
				response = await client.get(f"{self.capability_endpoint}/deployments/{deployment_id}/health")
				if response.status_code == 200:
					health_data = response.json()
					return {
						"healthy": health_data.get("status") == "healthy",
						"details": health_data
					}
				else:
					return {"healthy": False, "details": {"error": "Health check failed"}}
					
		except Exception as e:
			self.logger.error(f"Health verification failed: {e}")
			return {"healthy": False, "details": {"error": str(e)}}
	
	async def _update_configuration_cache(self, deployment_id: str, configuration: Dict[str, Any]):
		"""Update configuration cache with new configuration."""
		self.config_cache[deployment_id] = {
			**configuration,
			"deployment_id": deployment_id,
			"updated_at": datetime.now(timezone.utc).isoformat()
		}
	
	async def _log_configuration_update(self, deployment_id: str, configuration: Dict[str, Any], status: str):
		"""Log configuration update for audit purposes."""
		try:
			log_entry = {
				"timestamp": datetime.now(timezone.utc).isoformat(),
				"capability": self.capability_name,
				"deployment_id": deployment_id,
				"action": "configuration_update",
				"status": status,
				"configuration_keys": list(configuration.keys()),
				"user": getattr(self, 'current_user', 'system')
			}
			
			# Store in activities log
			if not hasattr(self, 'activity_log'):
				self.activity_log = []
			
			self.activity_log.append(log_entry)
			
			# Keep only last 100 entries
			if len(self.activity_log) > 100:
				self.activity_log = self.activity_log[-100:]
				
		except Exception as e:
			self.logger.error(f"Failed to log configuration update: {e}")
	
	async def _rollback_configuration(self, deployment_id: str, backup_id: str):
		"""Rollback configuration to previous backup."""
		try:
			if not hasattr(self, 'config_backups') or backup_id not in self.config_backups:
				self.logger.error(f"Backup {backup_id} not found for rollback")
				return False
			
			backup_config = self.config_backups[backup_id]["configuration"]
			
			# Apply backup configuration
			success = await self._apply_configuration_update(deployment_id, backup_config)
			
			if success:
				# Update cache with backup configuration
				await self._update_configuration_cache(deployment_id, backup_config)
				
				# Log rollback
				await self._log_configuration_update(deployment_id, backup_config, "rollback")
				
				self.logger.info(f"Successfully rolled back configuration for deployment {deployment_id}")
				return True
			else:
				self.logger.error(f"Failed to apply rollback configuration for deployment {deployment_id}")
				return False
				
		except Exception as e:
			self.logger.error(f"Rollback failed: {e}")
			return False
	
	async def _periodic_health_check(self):
		"""Periodic health check for the capability."""
		while True:
			try:
				await asyncio.sleep(60)  # Check every minute
				
				# Perform health check
				async with httpx.AsyncClient(timeout=10.0) as client:
					response = await client.get(f"{self.capability_endpoint}/health")
					self.connection_healthy = response.status_code == 200
					
					# Update metrics
					self.metrics_collector["health"][datetime.now(timezone.utc).isoformat()] = {
						"healthy": self.connection_healthy,
						"response_time": response.elapsed.total_seconds() * 1000 if hasattr(response, 'elapsed') else 0
					}
					
			except Exception as e:
				self.connection_healthy = False
				self.logger.warning(f"Health check failed for {self.capability_name}: {e}")
			
			except asyncio.CancelledError:
				break
	
	def _get_default_metrics(self) -> Dict[str, Any]:
		"""Get default metrics when real metrics are unavailable."""
		return {
			"response_time": {"avg": 0, "p95": 0, "p99": 0},
			"throughput": {"requests_per_second": 0, "total_requests": 0},
			"error_rate": 0,
			"resource_usage": {"cpu_percent": 0, "memory_percent": 0, "disk_usage": 0},
			"last_updated": datetime.now(timezone.utc).isoformat()
		}
	
	async def get_health_status(self) -> Dict[str, CapabilityHealth]:
		"""Get health status for all deployments."""
		health_data = {}
		all_health = await self.capability_manager.get_capability_health_status()
		
		for key, health in all_health.items():
			if health.capability_id == self.capability_id:
				health_data[health.deployment_id] = health
		
		return health_data
	
	async def get_deployment_info(self) -> List[Dict[str, Any]]:
		"""Get deployment information."""
		deployments = []
		
		for deployment in self.capability_spec.deployments:
			# Get health status
			health_key = f"{self.capability_id}_{deployment.deployment_id}"
			all_health = await self.capability_manager.get_capability_health_status()
			health = all_health.get(health_key)
			
			deployment_info = {
				"deployment_id": deployment.deployment_id,
				"environment": deployment.environment.value,
				"cloud_provider": deployment.cloud_provider.value,
				"region": deployment.region,
				"endpoint_url": deployment.endpoint.url,
				"status": health.status.value if health else "unknown",
				"response_time": health.response_time_ms if health else None,
				"last_check": health.last_check.isoformat() if health and health.last_check else None,
				"resource_requirements": deployment.resource_requirements,
				"scaling_config": deployment.scaling_config
			}
			
			deployments.append(deployment_info)
		
		return deployments


class CentralConfigurationApplet(BaseCapabilityApplet):
	"""Applet for managing Central Configuration capability."""
	
	async def _initialize_applet(self):
		"""Initialize central configuration applet."""
		self.layout = AppletLayout(
			rows=[
				# Status and metrics row
				[
					AppletWidget(
						widget_id="system_status",
						title="System Status",
						widget_type="status",
						config={"status_types": ["healthy", "degraded", "unhealthy"]},
						data_source="health_status",
						span=4
					),
					AppletWidget(
						widget_id="ai_metrics",
						title="AI Engine Metrics",
						widget_type="metric",
						config={"metrics": ["predictions", "optimizations", "anomalies"]},
						data_source="ai_metrics",
						span=4
					),
					AppletWidget(
						widget_id="automation_stats",
						title="Automation Statistics",
						widget_type="metric",
						config={"metrics": ["total_actions", "success_rate", "active_rules"]},
						data_source="automation_stats",
						span=4
					)
				],
				# Configuration and analytics row
				[
					AppletWidget(
						widget_id="recent_configs",
						title="Recent Configurations",
						widget_type="table",
						config={
							"columns": ["name", "created_at", "status", "security_level"],
							"limit": 10
						},
						data_source="recent_configurations",
						span=6
					),
					AppletWidget(
						widget_id="performance_chart",
						title="Performance Trends",
						widget_type="chart",
						config={
							"chart_type": "line",
							"metrics": ["response_time", "throughput", "error_rate"]
						},
						data_source="performance_metrics",
						span=6
					)
				],
				# Deployment and health row
				[
					AppletWidget(
						widget_id="deployment_status",
						title="Deployment Status",
						widget_type="table",
						config={
							"columns": ["deployment_id", "environment", "status", "last_check"],
							"status_colors": True
						},
						data_source="deployments",
						span=8
					),
					AppletWidget(
						widget_id="quick_actions",
						title="Quick Actions",
						widget_type="form",
						config={
							"actions": [
								{"name": "Optimize Configurations", "endpoint": "/optimize"},
								{"name": "Run Health Check", "endpoint": "/health-check"},
								{"name": "Generate Report", "endpoint": "/report"}
							]
						},
						data_source="actions",
						span=4
					)
				]
			]
		)
	
	async def get_dashboard_data(self) -> Dict[str, Any]:
		"""Get dashboard data for central configuration."""
		# Get system health
		health_data = await self.get_health_status()
		system_status = "healthy"
		if any(h.status != CapabilityStatus.HEALTHY for h in health_data.values()):
			system_status = "degraded"
		
		# Mock AI metrics (would come from actual AI engine)
		ai_metrics = {
			"total_predictions": 1547,
			"optimizations_applied": 342,
			"anomalies_detected": 23,
			"success_rate": 0.94
		}
		
		# Mock automation stats (would come from automation engine)
		automation_stats = {
			"total_actions": 456,
			"successful_actions": 431,
			"active_rules": 12,
			"success_rate": 0.945
		}
		
		# Recent configurations (mock data)
		recent_configs = [
			{
				"name": "Database Pool Config",
				"created_at": "2025-01-30T10:30:00Z",
				"status": "active",
				"security_level": "confidential"
			},
			{
				"name": "Redis Cache Settings",
				"created_at": "2025-01-30T09:15:00Z",
				"status": "active",
				"security_level": "internal"
			}
		]
		
		# Performance metrics (mock data)
		performance_metrics = {
			"timestamps": ["09:00", "10:00", "11:00", "12:00", "13:00"],
			"response_time": [120, 115, 118, 125, 122],
			"throughput": [850, 920, 880, 940, 910],
			"error_rate": [0.8, 0.5, 0.7, 0.3, 0.4]
		}
		
		return {
			"system_status": system_status,
			"ai_metrics": ai_metrics,
			"automation_stats": automation_stats,
			"recent_configurations": recent_configs,
			"performance_metrics": performance_metrics,
			"deployments": await self.get_deployment_info()
		}
	
	async def get_configuration_schema(self) -> Dict[str, Any]:
		"""Get configuration schema for central configuration."""
		return {
			"type": "object",
			"properties": {
				"ai_engine": {
					"type": "object",
					"properties": {
						"enabled": {"type": "boolean", "default": True},
						"models": {
							"type": "object",
							"properties": {
								"language_model": {"type": "string", "default": "llama3.2:3b"},
								"code_model": {"type": "string", "default": "codellama:7b"},
								"embedding_model": {"type": "string", "default": "nomic-embed-text"}
							}
						},
						"optimization_threshold": {"type": "number", "default": 0.7}
					}
				},
				"automation": {
					"type": "object",
					"properties": {
						"enabled": {"type": "boolean", "default": True},
						"safety_mode": {"type": "boolean", "default": False},
						"max_concurrent_actions": {"type": "integer", "default": 5},
						"approval_required_for_critical": {"type": "boolean", "default": True}
					}
				},
				"security": {
					"type": "object",
					"properties": {
						"encryption_enabled": {"type": "boolean", "default": True},
						"quantum_resistant": {"type": "boolean", "default": False},
						"audit_all_access": {"type": "boolean", "default": True},
						"session_timeout_minutes": {"type": "integer", "default": 60}
					}
				},
				"analytics": {
					"type": "object",
					"properties": {
						"real_time_enabled": {"type": "boolean", "default": True},
						"retention_days": {"type": "integer", "default": 90},
						"anomaly_detection": {"type": "boolean", "default": True},
						"performance_monitoring": {"type": "boolean", "default": True}
					}
				}
			},
			"required": ["ai_engine", "automation", "security", "analytics"]
		}
	
	async def update_configuration(
		self,
		deployment_id: str,
		configuration: Dict[str, Any]
	) -> bool:
		"""Update central configuration."""
		return await self.capability_manager.update_capability_configuration(
			self.capability_id,
			deployment_id,
			configuration
		)


class APIServiceMeshApplet(BaseCapabilityApplet):
	"""Applet for managing API Service Mesh capability."""
	
	async def _initialize_applet(self):
		"""Initialize API service mesh applet."""
		self.layout = AppletLayout(
			rows=[
				# Service mesh overview
				[
					AppletWidget(
						widget_id="mesh_topology",
						title="Service Mesh Topology",
						widget_type="chart",
						config={
							"chart_type": "network",
							"show_health": True,
							"show_traffic": True
						},
						data_source="mesh_topology",
						span=8
					),
					AppletWidget(
						widget_id="mesh_metrics",
						title="Mesh Metrics",
						widget_type="metric",
						config={"metrics": ["total_services", "healthy_services", "rps", "latency"]},
						data_source="mesh_metrics",
						span=4
					)
				],
				# Traffic and routing
				[
					AppletWidget(
						widget_id="traffic_routing",
						title="Traffic Routing Rules",
						widget_type="table",
						config={
							"columns": ["service", "route", "weight", "status"],
							"editable": True
						},
						data_source="routing_rules",
						span=6
					),
					AppletWidget(
						widget_id="load_balancing",
						title="Load Balancing",
						widget_type="form",
						config={
							"fields": [
								{"name": "strategy", "type": "select", "options": ["round_robin", "least_connections", "weighted"]},
								{"name": "health_check_interval", "type": "number"},
								{"name": "circuit_breaker_enabled", "type": "boolean"}
							]
						},
						data_source="load_balancer_config",
						span=6
					)
				]
			]
		)
	
	async def get_dashboard_data(self) -> Dict[str, Any]:
		"""Get dashboard data for API service mesh."""
		# Mock service mesh data
		mesh_topology = {
			"nodes": [
				{"id": "central-config", "name": "Central Configuration", "status": "healthy", "type": "service"},
				{"id": "realtime-collab", "name": "Real-time Collaboration", "status": "healthy", "type": "service"},
				{"id": "api-gateway", "name": "API Gateway", "status": "healthy", "type": "gateway"}
			],
			"edges": [
				{"from": "api-gateway", "to": "central-config", "traffic": "high"},
				{"from": "api-gateway", "to": "realtime-collab", "traffic": "medium"}
			]
		}
		
		mesh_metrics = {
			"total_services": 15,
			"healthy_services": 14,
			"requests_per_second": 2450,
			"avg_latency_ms": 85,
			"error_rate": 0.3
		}
		
		routing_rules = [
			{
				"service": "central-config",
				"route": "/api/v1/configurations",
				"weight": "100%",
				"status": "active"
			},
			{
				"service": "realtime-collab",
				"route": "/api/v1/collaboration",
				"weight": "100%", 
				"status": "active"
			}
		]
		
		load_balancer_config = {
			"strategy": "round_robin",
			"health_check_interval": 30,
			"circuit_breaker_enabled": True,
			"timeout_seconds": 30
		}
		
		return {
			"mesh_topology": mesh_topology,
			"mesh_metrics": mesh_metrics,
			"routing_rules": routing_rules,
			"load_balancer_config": load_balancer_config,
			"deployments": await self.get_deployment_info()
		}
	
	async def get_configuration_schema(self) -> Dict[str, Any]:
		"""Get configuration schema for API service mesh."""
		return {
			"type": "object",
			"properties": {
				"load_balancing": {
					"type": "object",
					"properties": {
						"strategy": {"type": "string", "enum": ["round_robin", "least_connections", "weighted"], "default": "round_robin"},
						"health_check_interval": {"type": "integer", "default": 30},
						"timeout_seconds": {"type": "integer", "default": 30}
					}
				},
				"circuit_breaker": {
					"type": "object",
					"properties": {
						"enabled": {"type": "boolean", "default": True},
						"failure_threshold": {"type": "integer", "default": 5},
						"recovery_timeout": {"type": "integer", "default": 60}
					}
				},
				"rate_limiting": {
					"type": "object",
					"properties": {
						"enabled": {"type": "boolean", "default": True},
						"requests_per_second": {"type": "integer", "default": 1000},
						"burst_size": {"type": "integer", "default": 2000}
					}
				},
				"security": {
					"type": "object",
					"properties": {
						"tls_enabled": {"type": "boolean", "default": True},
						"mutual_tls": {"type": "boolean", "default": False},
						"jwt_validation": {"type": "boolean", "default": True}
					}
				}
			}
		}
	
	async def update_configuration(
		self,
		deployment_id: str,
		configuration: Dict[str, Any]
	) -> bool:
		"""Update API service mesh configuration."""
		return await self.capability_manager.update_capability_configuration(
			self.capability_id,
			deployment_id,
			configuration
		)


class RealtimeCollaborationApplet(BaseCapabilityApplet):
	"""Applet for managing Real-time Collaboration capability."""
	
	async def _initialize_applet(self):
		"""Initialize real-time collaboration applet."""
		self.layout = AppletLayout(
			rows=[
				# Connection and session metrics
				[
					AppletWidget(
						widget_id="active_sessions",
						title="Active Sessions",
						widget_type="metric",
						config={"metrics": ["total_sessions", "active_users", "concurrent_connections"]},
						data_source="session_metrics",
						span=4
					),
					AppletWidget(
						widget_id="protocol_usage",
						title="Protocol Usage",
						widget_type="chart",
						config={
							"chart_type": "pie",
							"protocols": ["websocket", "webrtc", "socketio", "mqtt"]
						},
						data_source="protocol_stats",
						span=4
					),
					AppletWidget(
						widget_id="message_throughput",
						title="Message Throughput",
						widget_type="chart",
						config={
							"chart_type": "line",
							"metrics": ["messages_per_second", "data_throughput"]
						},
						data_source="throughput_metrics",
						span=4
					)
				],
				# Protocol configuration
				[
					AppletWidget(
						widget_id="protocol_config",
						title="Protocol Configuration",
						widget_type="form",
						config={
							"protocols": [
								{"name": "websocket", "enabled": True},
								{"name": "webrtc", "enabled": True},
								{"name": "socketio", "enabled": True},
								{"name": "mqtt", "enabled": False}
							]
						},
						data_source="protocol_config",
						span=6
					),
					AppletWidget(
						widget_id="collaboration_rooms",
						title="Active Collaboration Rooms",
						widget_type="table",
						config={
							"columns": ["room_id", "participants", "protocol", "created_at", "activity"],
							"real_time_updates": True
						},
						data_source="active_rooms",
						span=6
					)
				]
			]
		)
	
	async def get_dashboard_data(self) -> Dict[str, Any]:
		"""Get dashboard data for real-time collaboration."""
		# Mock collaboration data
		session_metrics = {
			"total_sessions": 234,
			"active_users": 189,
			"concurrent_connections": 456,
			"peak_concurrent": 512
		}
		
		protocol_stats = {
			"websocket": 65,
			"webrtc": 25,
			"socketio": 8,
			"mqtt": 2
		}
		
		throughput_metrics = {
			"timestamps": ["13:00", "13:05", "13:10", "13:15", "13:20"],
			"messages_per_second": [1200, 1350, 1180, 1420, 1380],
			"data_throughput_mbps": [5.2, 6.1, 4.8, 6.8, 6.3]
		}
		
		protocol_config = {
			"websocket": {"enabled": True, "max_connections": 1000},
			"webrtc": {"enabled": True, "ice_servers": ["stun:stun.l.google.com:19302"]},
			"socketio": {"enabled": True, "cors_origins": ["*"]},
			"mqtt": {"enabled": False, "qos_level": 1}
		}
		
		active_rooms = [
			{
				"room_id": "room_config_123",
				"participants": 5,
				"protocol": "websocket",
				"created_at": "2025-01-30T13:15:00Z",
				"activity": "high"
			},
			{
				"room_id": "room_collab_456",
				"participants": 3,
				"protocol": "webrtc",
				"created_at": "2025-01-30T13:20:00Z",
				"activity": "medium"
			}
		]
		
		return {
			"session_metrics": session_metrics,
			"protocol_stats": protocol_stats,
			"throughput_metrics": throughput_metrics,
			"protocol_config": protocol_config,
			"active_rooms": active_rooms,
			"deployments": await self.get_deployment_info()
		}
	
	async def get_configuration_schema(self) -> Dict[str, Any]:
		"""Get configuration schema for real-time collaboration."""
		return {
			"type": "object",
			"properties": {
				"websocket": {
					"type": "object",
					"properties": {
						"enabled": {"type": "boolean", "default": True},
						"max_connections": {"type": "integer", "default": 1000},
						"heartbeat_interval": {"type": "integer", "default": 30},
						"compression": {"type": "boolean", "default": True}
					}
				},
				"webrtc": {
					"type": "object",
					"properties": {
						"enabled": {"type": "boolean", "default": True},
						"ice_servers": {"type": "array", "items": {"type": "string"}},
						"max_bitrate": {"type": "integer", "default": 1000000}
					}
				},
				"socketio": {
					"type": "object",
					"properties": {
						"enabled": {"type": "boolean", "default": True},
						"cors_origins": {"type": "array", "items": {"type": "string"}},
						"transport": {"type": "array", "items": {"type": "string"}}
					}
				},
				"message_retention": {
					"type": "object",
					"properties": {
						"enabled": {"type": "boolean", "default": True},
						"retention_hours": {"type": "integer", "default": 24},
						"max_message_size": {"type": "integer", "default": 1048576}
					}
				}
			}
		}
	
	async def update_configuration(
		self,
		deployment_id: str,
		configuration: Dict[str, Any]
	) -> bool:
		"""Update real-time collaboration configuration."""
		return await self.capability_manager.update_capability_configuration(
			self.capability_id,
			deployment_id,
			configuration
		)


class AppletRegistry:
	"""Registry for managing capability applets."""
	
	def __init__(self, capability_manager: APGCapabilityManager):
		"""Initialize applet registry."""
		self.capability_manager = capability_manager
		self.applets: Dict[str, BaseCapabilityApplet] = {}
		
		# Register built-in applets
		asyncio.create_task(self._register_builtin_applets())
	
	async def _register_builtin_applets(self):
		"""Register built-in capability applets."""
		await asyncio.sleep(1)  # Wait for capability manager
		
		capabilities = await self.capability_manager.get_all_capabilities()
		
		for capability_id, spec in capabilities.items():
			await self.register_applet(capability_id, spec)
	
	async def register_applet(
		self,
		capability_id: str,
		capability_spec: CapabilitySpec
	) -> BaseCapabilityApplet:
		"""Register applet for a capability."""
		# Create appropriate applet type
		if capability_id == "central_configuration":
			applet = CentralConfigurationApplet(capability_id, capability_spec, self.capability_manager)
		elif capability_id == "api_service_mesh":
			applet = APIServiceMeshApplet(capability_id, capability_spec, self.capability_manager)
		elif capability_id == "realtime_collaboration":
			applet = RealtimeCollaborationApplet(capability_id, capability_spec, self.capability_manager)
		else:
			# Generic applet for unknown capabilities
			applet = GenericCapabilityApplet(capability_id, capability_spec, self.capability_manager)
		
		self.applets[capability_id] = applet
		print(f"📱 Registered applet for {capability_spec.name}")
		
		return applet
	
	async def get_applet(self, capability_id: str) -> Optional[BaseCapabilityApplet]:
		"""Get applet for a capability."""
		return self.applets.get(capability_id)
	
	async def get_all_applets(self) -> Dict[str, BaseCapabilityApplet]:
		"""Get all registered applets."""
		return self.applets.copy()
	
	async def get_applet_dashboard_data(self, capability_id: str) -> Dict[str, Any]:
		"""Get dashboard data for a specific applet."""
		applet = self.applets.get(capability_id)
		if applet:
			return await applet.get_dashboard_data()
		return {}


class GenericCapabilityApplet(BaseCapabilityApplet):
	"""Generic applet for unknown capabilities."""
	
	async def _initialize_applet(self):
		"""Initialize generic applet."""
		self.layout = AppletLayout(
			rows=[
				[
					AppletWidget(
						widget_id="basic_info",
						title="Capability Information",
						widget_type="table",
						config={"show_metadata": True},
						data_source="capability_info",
						span=6
					),
					AppletWidget(
						widget_id="health_status",
						title="Health Status",
						widget_type="status",
						config={"show_deployments": True},
						data_source="health_status",
						span=6
					)
				],
				[
					AppletWidget(
						widget_id="configuration",
						title="Configuration",
						widget_type="form",
						config={"schema_based": True},
						data_source="configuration",
						span=12
					)
				]
			]
		)
	
	async def get_dashboard_data(self) -> Dict[str, Any]:
		"""Get dashboard data for generic capability."""
		capability_info = {
			"name": self.capability_spec.name,
			"version": self.capability_spec.version,
			"category": self.capability_spec.category,
			"description": self.capability_spec.description,
			"maintainer": self.capability_spec.maintainer
		}
		
		return {
			"capability_info": capability_info,
			"health_status": await self.get_health_status(),
			"deployments": await self.get_deployment_info(),
			"configuration": self.capability_spec.default_config
		}
	
	async def get_configuration_schema(self) -> Dict[str, Any]:
		"""Get configuration schema for generic capability."""
		return self.capability_spec.config_schema
	
	async def update_configuration(
		self,
		deployment_id: str,
		configuration: Dict[str, Any]
	) -> bool:
		"""Update generic capability configuration."""
		return await self.capability_manager.update_capability_configuration(
			self.capability_id,
			deployment_id,
			configuration
		)


# ==================== Flask-AppBuilder Integration ====================

class CapabilityAppletView(BaseView):
	"""Flask-AppBuilder view for capability applets."""
	
	route_base = "/applets"
	default_view = "list"
	
	def __init__(self, applet_registry: AppletRegistry):
		"""Initialize applet view."""
		super().__init__()
		self.applet_registry = applet_registry
	
	@expose("/")
	@has_access
	def list(self):
		"""List all capability applets."""
		applets = asyncio.run(self.applet_registry.get_all_applets())
		
		applet_data = []
		for capability_id, applet in applets.items():
			applet_data.append({
				"capability_id": capability_id,
				"title": applet.title,
				"description": applet.description,
				"category": applet.category,
				"version": applet.version
			})
		
		return render_template(
			"applets/list.html",
			applets=applet_data,
			title="APG Capability Applets"
		)
	
	@expose("/<capability_id>")
	@has_access
	def show(self, capability_id):
		"""Show specific capability applet."""
		applet = asyncio.run(self.applet_registry.get_applet(capability_id))
		
		if not applet:
			flash(f"Applet for capability '{capability_id}' not found", "error")
			return redirect(url_for(".list"))
		
		# Get dashboard data
		dashboard_data = asyncio.run(applet.get_dashboard_data())
		
		return render_template(
			"applets/dashboard.html",
			applet=applet,
			dashboard_data=dashboard_data,
			layout=applet.layout
		)
	
	@expose("/<capability_id>/configure")
	@has_access
	def configure(self, capability_id):
		"""Configure capability through applet."""
		applet = asyncio.run(self.applet_registry.get_applet(capability_id))
		
		if not applet:
			flash(f"Applet for capability '{capability_id}' not found", "error")
			return redirect(url_for(".list"))
		
		if request.method == "POST":
			# Handle configuration update
			deployment_id = request.form.get("deployment_id")
			configuration_data = {}
			
			# Extract configuration from form
			for key, value in request.form.items():
				if key.startswith("config_"):
					config_key = key[7:]  # Remove "config_" prefix
					configuration_data[config_key] = value
			
			# Update configuration
			success = asyncio.run(applet.update_configuration(deployment_id, configuration_data))
			
			if success:
				flash("Configuration updated successfully", "success")
			else:
				flash("Failed to update configuration", "error")
			
			return redirect(url_for(".show", capability_id=capability_id))
		
		# Get configuration schema
		schema = asyncio.run(applet.get_configuration_schema())
		deployments = asyncio.run(applet.get_deployment_info())
		
		return render_template(
			"applets/configure.html",
			applet=applet,
			schema=schema,
			deployments=deployments
		)
	
	@expose("/api/<capability_id>/data")
	@has_access
	def api_data(self, capability_id):
		"""API endpoint for applet data."""
		dashboard_data = asyncio.run(self.applet_registry.get_applet_dashboard_data(capability_id))
		return jsonify(dashboard_data)


# ==================== Factory Functions ====================

async def create_applet_registry(
	capability_manager: APGCapabilityManager
) -> AppletRegistry:
	"""Create and initialize applet registry."""
	registry = AppletRegistry(capability_manager)
	await asyncio.sleep(1)  # Allow initialization
	print("📱 Applet registry initialized")
	return registry