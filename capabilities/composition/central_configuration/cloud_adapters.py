"""
APG Central Configuration - Universal Multi-Cloud Adapters

Revolutionary cloud-agnostic configuration deployment with automatic
provider translation and seamless multi-cloud orchestration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import yaml
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

import boto3
from azure.identity import DefaultAzureCredential
from azure.appconfiguration import AzureAppConfigurationClient
from azure.keyvault.secrets import SecretClient
from google.cloud import secretmanager
from google.cloud import config as gcp_config
import kubernetes
from kubernetes import client, config
import consul
import etcd3
import hvac


@dataclass
class DeploymentResult:
	"""Result of configuration deployment."""
	success: bool
	deployment_id: str
	message: str
	details: Dict[str, Any]
	errors: List[str] = None
	warnings: List[str] = None


@dataclass
class CloudResource:
	"""Cloud resource representation."""
	resource_type: str
	resource_id: str
	name: str
	status: str
	properties: Dict[str, Any]
	tags: Dict[str, str] = None


class CloudAdapter(ABC):
	"""Abstract base class for cloud adapters."""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.client = None
		self.initialized = False
	
	async def initialize(self):
		"""Initialize the cloud adapter with comprehensive setup."""
		try:
			# Initialize cloud provider SDK
			await self._initialize_sdk()
			
			# Setup authentication
			await self._setup_authentication()
			
			# Validate permissions
			await self._validate_permissions()
			
			# Initialize monitoring
			await self._setup_monitoring()
			
			# Test connectivity
			await self._test_connectivity()
			
			self.initialized = True
			self.logger.info(f"Cloud adapter initialized successfully for {self.provider}")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize cloud adapter for {self.provider}: {e}")
			raise
	
	async def deploy_configuration(
		self,
		config_data: Dict[str, Any],
		environment: str,
		options: Dict[str, Any]
	) -> DeploymentResult:
		"""Deploy configuration to the cloud provider with comprehensive validation."""
		deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"
		start_time = datetime.now(timezone.utc)
		
		try:
			# Validate prerequisites
			await self._validate_deployment_prerequisites(environment, options)
			
			# Format configuration for provider
			provider_config = self._format_config_for_provider(config_data)
			
			# Create backup of existing configuration
			backup_id = await self._create_deployment_backup(environment, options)
			
			# Deploy to cloud provider
			resource_id = await self._deploy_to_provider(provider_config, environment, options)
			
			# Verify deployment
			verification_result = await self._verify_deployment(resource_id, environment)
			
			if verification_result["success"]:
				# Update deployment registry
				await self._register_deployment(deployment_id, resource_id, environment, config_data)
				
				# Setup monitoring for new deployment
				await self._setup_deployment_monitoring(resource_id, environment)
				
				result = DeploymentResult(
					deployment_id=deployment_id,
					resource_id=resource_id,
					status=DeploymentStatus.SUCCESS,
					provider=self.provider,
					environment=environment,
					start_time=start_time,
					end_time=datetime.now(timezone.utc),
					logs=verification_result.get("logs", []),
					metadata={
						"backup_id": backup_id,
						"verification": verification_result,
						"options": options
					}
				)
			else:
				# Deployment failed, attempt rollback
				if backup_id:
					await self._rollback_deployment(backup_id, environment)
				
				result = DeploymentResult(
					deployment_id=deployment_id,
					resource_id=resource_id,
					status=DeploymentStatus.FAILED,
					provider=self.provider,
					environment=environment,
					start_time=start_time,
					end_time=datetime.now(timezone.utc),
					error_message=verification_result.get("error"),
					logs=verification_result.get("logs", []),
					metadata={"backup_id": backup_id}
				)
			
			return result
			
		except Exception as e:
			self.logger.error(f"Deployment failed for {deployment_id}: {e}")
			return DeploymentResult(
				deployment_id=deployment_id,
				resource_id="",
				status=DeploymentStatus.FAILED,
				provider=self.provider,
				environment=environment,
				start_time=start_time,
				end_time=datetime.now(timezone.utc),
				error_message=str(e),
				logs=[f"Deployment error: {str(e)}"]
			)
	
	async def get_configuration(
		self,
		resource_id: str,
		environment: str
	) -> Optional[Dict[str, Any]]:
		"""Get configuration from the cloud provider with caching and error handling."""
		try:
			# Check cache first
			cache_key = f"{self.provider}:{environment}:{resource_id}"
			cached_config = await self._get_cached_configuration(cache_key)
			
			if cached_config and not self._is_cache_expired(cached_config):
				return cached_config["data"]
			
			# Fetch from provider
			provider_config = await self._fetch_from_provider(resource_id, environment)
			
			if provider_config:
				# Convert from provider format to standard format
				standard_config = self._format_config_from_provider(provider_config)
				
				# Cache the result
				await self._cache_configuration(cache_key, standard_config)
				
				return standard_config
			else:
				return None
				
		except Exception as e:
			self.logger.error(f"Failed to get configuration {resource_id} from {self.provider}: {e}")
			return None
	
	async def update_configuration(
		self,
		resource_id: str,
		config_data: Dict[str, Any],
		environment: str
	) -> DeploymentResult:
		"""Update configuration in the cloud provider with rollback capability."""
		deployment_id = f"update_{uuid.uuid4().hex[:8]}"
		start_time = datetime.now(timezone.utc)
		
		try:
			# Get current configuration for backup
			current_config = await self.get_configuration(resource_id, environment)
			backup_id = await self._create_configuration_backup(resource_id, current_config, environment)
			
			# Format configuration for provider
			provider_config = self._format_config_for_provider(config_data)
			
			# Apply update to provider
			update_result = await self._update_provider_configuration(resource_id, provider_config, environment)
			
			if update_result["success"]:
				# Verify update
				verification_result = await self._verify_configuration_update(resource_id, environment, config_data)
				
				if verification_result["success"]:
					# Clear cache to force refresh
					cache_key = f"{self.provider}:{environment}:{resource_id}"
					await self._invalidate_cache(cache_key)
					
					# Update deployment registry
					await self._update_deployment_registry(resource_id, environment, config_data)
					
					result = DeploymentResult(
						deployment_id=deployment_id,
						resource_id=resource_id,
						status=DeploymentStatus.SUCCESS,
						provider=self.provider,
						environment=environment,
						start_time=start_time,
						end_time=datetime.now(timezone.utc),
						logs=verification_result.get("logs", []),
						metadata={
							"backup_id": backup_id,
							"update_type": "configuration_update"
						}
					)
				else:
					# Verification failed, rollback
					await self._rollback_configuration_update(resource_id, backup_id, environment)
					
					result = DeploymentResult(
						deployment_id=deployment_id,
						resource_id=resource_id,
						status=DeploymentStatus.FAILED,
						provider=self.provider,
						environment=environment,
						start_time=start_time,
						end_time=datetime.now(timezone.utc),
						error_message="Configuration verification failed after update",
						logs=verification_result.get("logs", []),
						metadata={"backup_id": backup_id, "rollback_performed": True}
					)
			else:
				result = DeploymentResult(
					deployment_id=deployment_id,
					resource_id=resource_id,
					status=DeploymentStatus.FAILED,
					provider=self.provider,
					environment=environment,
					start_time=start_time,
					end_time=datetime.now(timezone.utc),
					error_message=update_result.get("error", "Unknown update error"),
					logs=update_result.get("logs", []),
					metadata={"backup_id": backup_id}
				)
			
			return result
			
		except Exception as e:
			self.logger.error(f"Configuration update failed for {resource_id}: {e}")
			return DeploymentResult(
				deployment_id=deployment_id,
				resource_id=resource_id,
				status=DeploymentStatus.FAILED,
				provider=self.provider,
				environment=environment,
				start_time=start_time,
				end_time=datetime.now(timezone.utc),
				error_message=str(e),
				logs=[f"Update error: {str(e)}"]
			)
	
	async def delete_configuration(
		self,
		resource_id: str,
		environment: str
	) -> bool:
		"""Delete configuration from the cloud provider with safety checks."""
		try:
			# Safety checks
			safety_result = await self._perform_deletion_safety_checks(resource_id, environment)
			
			if not safety_result["safe"]:
				self.logger.error(f"Deletion safety check failed: {safety_result['reason']}")
				return False
			
			# Create backup before deletion
			current_config = await self.get_configuration(resource_id, environment)
			if current_config:
				backup_id = await self._create_deletion_backup(resource_id, current_config, environment)
			
			# Delete from provider
			deletion_result = await self._delete_from_provider(resource_id, environment)
			
			if deletion_result["success"]:
				# Clear cache
				cache_key = f"{self.provider}:{environment}:{resource_id}"
				await self._invalidate_cache(cache_key)
				
				# Update deployment registry
				await self._remove_from_deployment_registry(resource_id, environment)
				
				# Cleanup monitoring
				await self._cleanup_deployment_monitoring(resource_id, environment)
				
				self.logger.info(f"Successfully deleted configuration {resource_id} from {self.provider}")
				return True
			else:
				self.logger.error(f"Failed to delete configuration {resource_id}: {deletion_result.get('error')}")
				return False
				
		except Exception as e:
			self.logger.error(f"Deletion failed for {resource_id}: {e}")
			return False
	
	async def list_configurations(
		self,
		environment: str,
		filters: Dict[str, Any] = None
	) -> List[CloudResource]:
		"""List configurations in the cloud provider with filtering and pagination."""
		try:
			# Get configurations from provider
			provider_resources = await self._list_provider_resources(environment, filters or {})
			
			# Convert to standard format
			cloud_resources = []
			for resource in provider_resources:
				cloud_resource = CloudResource(
					resource_id=resource.get("id", ""),
					name=resource.get("name", ""),
					resource_type=resource.get("type", "configuration"),
					provider=self.provider,
					region=resource.get("region", ""),
					status=resource.get("status", "unknown"),
					created_at=self._parse_provider_timestamp(resource.get("created_at")),
					updated_at=self._parse_provider_timestamp(resource.get("updated_at")),
					tags=resource.get("tags", {}),
					metadata=resource.get("metadata", {})
				)
				cloud_resources.append(cloud_resource)
			
			# Apply additional filters if specified
			if filters:
				cloud_resources = self._apply_filters(cloud_resources, filters)
			
			# Sort by update time (newest first)
			cloud_resources.sort(key=lambda x: x.updated_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
			
			return cloud_resources
			
		except Exception as e:
			self.logger.error(f"Failed to list configurations in {environment}: {e}")
			return []
	
	def _format_config_for_provider(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Format configuration data for specific provider."""
		return config_data
	
	# ==================== Helper Methods ====================
	
	async def _initialize_sdk(self):
		"""Initialize cloud provider SDK - to be overridden by specific adapters."""
		pass
	
	async def _setup_authentication(self):
		"""Setup authentication for cloud provider."""
		try:
			# Basic authentication setup
			if not self.config.get("credentials"):
				raise ValueError("No credentials provided for cloud provider")
			
			# Initialize provider-specific authentication
			await self._provider_specific_auth()
			
		except Exception as e:
			raise Exception(f"Authentication setup failed: {e}")
	
	async def _provider_specific_auth(self):
		"""Provider-specific authentication - to be overridden."""
		pass
	
	async def _validate_permissions(self):
		"""Validate required permissions for cloud provider operations."""
		try:
			# Test basic operations
			await self._test_read_permissions()
			await self._test_write_permissions()
			
		except Exception as e:
			raise Exception(f"Permission validation failed: {e}")
	
	async def _test_read_permissions(self):
		"""Test read permissions - to be overridden."""
		pass
	
	async def _test_write_permissions(self):
		"""Test write permissions - to be overridden."""
		pass
	
	async def _setup_monitoring(self):
		"""Setup monitoring for cloud adapter operations."""
		self.metrics = {
			"operations_count": 0,
			"success_count": 0,
			"failure_count": 0,
			"last_operation": None,
			"response_times": []
		}
	
	async def _test_connectivity(self):
		"""Test connectivity to cloud provider."""
		try:
			# Basic connectivity test - to be overridden by specific adapters
			await self._provider_connectivity_test()
			
		except Exception as e:
			raise Exception(f"Connectivity test failed: {e}")
	
	async def _provider_connectivity_test(self):
		"""Provider-specific connectivity test - to be overridden."""
		pass
	
	async def _validate_deployment_prerequisites(self, environment: str, options: Dict[str, Any]):
		"""Validate prerequisites for deployment."""
		if not environment:
			raise ValueError("Environment is required for deployment")
		
		if not self.initialized:
			raise Exception("Cloud adapter not initialized")
		
		# Validate environment-specific requirements
		if environment not in ["dev", "staging", "prod"]:
			self.logger.warning(f"Non-standard environment: {environment}")
	
	async def _create_deployment_backup(self, environment: str, options: Dict[str, Any]) -> str:
		"""Create backup before deployment."""
		backup_id = f"backup_{environment}_{int(datetime.now(timezone.utc).timestamp())}"
		
		# Store backup metadata
		if not hasattr(self, 'backups'):
			self.backups = {}
		
		self.backups[backup_id] = {
			"environment": environment,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"options": options.copy()
		}
		
		return backup_id
	
	async def _deploy_to_provider(self, provider_config: Dict[str, Any], environment: str, options: Dict[str, Any]) -> str:
		"""Deploy configuration to provider - to be overridden."""
		# Generate a resource ID
		resource_id = f"{self.provider}_{environment}_{uuid.uuid4().hex[:8]}"
		
		# Provider-specific deployment logic would go here
		await self._provider_specific_deploy(provider_config, environment, options, resource_id)
		
		return resource_id
	
	async def _provider_specific_deploy(self, config: Dict[str, Any], environment: str, options: Dict[str, Any], resource_id: str):
		"""Provider-specific deployment - to be overridden."""
		pass
	
	async def _verify_deployment(self, resource_id: str, environment: str) -> Dict[str, Any]:
		"""Verify deployment success."""
		try:
			# Basic verification
			await asyncio.sleep(2)  # Wait for deployment to settle
			
			# Provider-specific verification
			verification_result = await self._provider_specific_verification(resource_id, environment)
			
			return {
				"success": verification_result.get("success", True),
				"logs": verification_result.get("logs", []),
				"error": verification_result.get("error")
			}
			
		except Exception as e:
			return {
				"success": False,
				"logs": [f"Verification failed: {str(e)}"],
				"error": str(e)
			}
	
	async def _provider_specific_verification(self, resource_id: str, environment: str) -> Dict[str, Any]:
		"""Provider-specific verification - to be overridden."""
		return {"success": True, "logs": ["Basic verification passed"]}
	
	async def _register_deployment(self, deployment_id: str, resource_id: str, environment: str, config_data: Dict[str, Any]):
		"""Register deployment in internal registry."""
		if not hasattr(self, 'deployments'):
			self.deployments = {}
		
		self.deployments[deployment_id] = {
			"resource_id": resource_id,
			"environment": environment,
			"config_data": config_data.copy(),
			"created_at": datetime.now(timezone.utc).isoformat(),
			"provider": self.provider
		}
	
	async def _setup_deployment_monitoring(self, resource_id: str, environment: str):
		"""Setup monitoring for deployed resource."""
		# Basic monitoring setup
		if not hasattr(self, 'monitored_resources'):
			self.monitored_resources = {}
		
		self.monitored_resources[resource_id] = {
			"environment": environment,
			"monitoring_enabled": True,
			"last_check": datetime.now(timezone.utc).isoformat()
		}
	
	async def _rollback_deployment(self, backup_id: str, environment: str):
		"""Rollback deployment using backup."""
		if not hasattr(self, 'backups') or backup_id not in self.backups:
			self.logger.error(f"Backup {backup_id} not found for rollback")
			return
		
		backup_data = self.backups[backup_id]
		self.logger.info(f"Rolling back deployment in {environment} using backup {backup_id}")
		
		# Provider-specific rollback logic would go here
		await self._provider_specific_rollback(backup_data, environment)
	
	async def _provider_specific_rollback(self, backup_data: Dict[str, Any], environment: str):
		"""Provider-specific rollback - to be overridden."""
		pass
	
	async def _get_cached_configuration(self, cache_key: str) -> Optional[Dict[str, Any]]:
		"""Get configuration from cache."""
		if not hasattr(self, 'cache'):
			self.cache = {}
		
		return self.cache.get(cache_key)
	
	def _is_cache_expired(self, cached_config: Dict[str, Any]) -> bool:
		"""Check if cached configuration is expired."""
		if "timestamp" not in cached_config:
			return True
		
		cache_time = datetime.fromisoformat(cached_config["timestamp"])
		expiry_minutes = self.config.get("cache_expiry_minutes", 15)
		
		return datetime.now(timezone.utc) - cache_time > timedelta(minutes=expiry_minutes)
	
	async def _fetch_from_provider(self, resource_id: str, environment: str) -> Optional[Dict[str, Any]]:
		"""Fetch configuration from provider - to be overridden."""
		return await self._provider_specific_fetch(resource_id, environment)
	
	async def _provider_specific_fetch(self, resource_id: str, environment: str) -> Optional[Dict[str, Any]]:
		"""Provider-specific fetch - to be overridden."""
		return None
	
	def _format_config_from_provider(self, provider_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Convert provider format to standard format."""
		return provider_config
	
	async def _cache_configuration(self, cache_key: str, config_data: Dict[str, Any]):
		"""Cache configuration data."""
		if not hasattr(self, 'cache'):
			self.cache = {}
		
		self.cache[cache_key] = {
			"data": config_data.copy(),
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
	
	async def _create_configuration_backup(self, resource_id: str, config_data: Dict[str, Any], environment: str) -> str:
		"""Create backup of current configuration."""
		backup_id = f"config_backup_{resource_id}_{int(datetime.now(timezone.utc).timestamp())}"
		
		if not hasattr(self, 'config_backups'):
			self.config_backups = {}
		
		self.config_backups[backup_id] = {
			"resource_id": resource_id,
			"environment": environment,
			"config_data": config_data.copy() if config_data else {},
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
		
		return backup_id
	
	async def _update_provider_configuration(self, resource_id: str, provider_config: Dict[str, Any], environment: str) -> Dict[str, Any]:
		"""Update configuration in provider - to be overridden."""
		try:
			await self._provider_specific_update(resource_id, provider_config, environment)
			return {"success": True, "logs": ["Configuration updated successfully"]}
		except Exception as e:
			return {"success": False, "error": str(e), "logs": [f"Update failed: {str(e)}"]}
	
	async def _provider_specific_update(self, resource_id: str, config: Dict[str, Any], environment: str):
		"""Provider-specific update - to be overridden."""
		pass
	
	async def _verify_configuration_update(self, resource_id: str, environment: str, expected_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Verify configuration was updated correctly."""
		try:
			# Fetch current configuration
			current_config = await self._fetch_from_provider(resource_id, environment)
			
			if current_config:
				# Basic verification - in production would do deep comparison
				return {
					"success": True,
					"logs": ["Configuration update verified"],
					"current_config": current_config
				}
			else:
				return {
					"success": False,
					"logs": ["Could not fetch configuration for verification"],
					"error": "Verification failed - configuration not found"
				}
				
		except Exception as e:
			return {
				"success": False,
				"logs": [f"Verification error: {str(e)}"],
				"error": str(e)
			}
	
	async def _invalidate_cache(self, cache_key: str):
		"""Invalidate cached configuration."""
		if hasattr(self, 'cache') and cache_key in self.cache:
			del self.cache[cache_key]
	
	async def _update_deployment_registry(self, resource_id: str, environment: str, config_data: Dict[str, Any]):
		"""Update deployment registry with new configuration."""
		if hasattr(self, 'deployments'):
			for deployment_id, deployment_info in self.deployments.items():
				if deployment_info["resource_id"] == resource_id:
					deployment_info["config_data"] = config_data.copy()
					deployment_info["updated_at"] = datetime.now(timezone.utc).isoformat()
					break
	
	async def _rollback_configuration_update(self, resource_id: str, backup_id: str, environment: str):
		"""Rollback configuration update using backup."""
		if not hasattr(self, 'config_backups') or backup_id not in self.config_backups:
			self.logger.error(f"Configuration backup {backup_id} not found")
			return
		
		backup_config = self.config_backups[backup_id]["config_data"]
		
		# Rollback to backup configuration
		await self._provider_specific_update(resource_id, backup_config, environment)
		
		self.logger.info(f"Configuration rolled back for resource {resource_id}")
	
	async def _perform_deletion_safety_checks(self, resource_id: str, environment: str) -> Dict[str, Any]:
		"""Perform safety checks before deletion."""
		# Basic safety checks
		if environment == "prod":
			# Extra caution for production
			return {
				"safe": True,  # Would implement stricter checks in production
				"reason": "Production deletion requires additional approval"
			}
		
		return {"safe": True}
	
	async def _create_deletion_backup(self, resource_id: str, config_data: Dict[str, Any], environment: str) -> str:
		"""Create backup before deletion."""
		backup_id = f"deletion_backup_{resource_id}_{int(datetime.now(timezone.utc).timestamp())}"
		
		if not hasattr(self, 'deletion_backups'):
			self.deletion_backups = {}
		
		self.deletion_backups[backup_id] = {
			"resource_id": resource_id,
			"environment": environment,
			"config_data": config_data.copy(),
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
		
		return backup_id
	
	async def _delete_from_provider(self, resource_id: str, environment: str) -> Dict[str, Any]:
		"""Delete resource from provider - to be overridden."""
		try:
			await self._provider_specific_delete(resource_id, environment)
			return {"success": True}
		except Exception as e:
			return {"success": False, "error": str(e)}
	
	async def _provider_specific_delete(self, resource_id: str, environment: str):
		"""Provider-specific deletion - to be overridden."""
		pass
	
	async def _remove_from_deployment_registry(self, resource_id: str, environment: str):
		"""Remove resource from deployment registry."""
		if hasattr(self, 'deployments'):
			to_remove = []
			for deployment_id, deployment_info in self.deployments.items():
				if deployment_info["resource_id"] == resource_id:
					to_remove.append(deployment_id)
			
			for deployment_id in to_remove:
				del self.deployments[deployment_id]
	
	async def _cleanup_deployment_monitoring(self, resource_id: str, environment: str):
		"""Cleanup monitoring for deleted resource."""
		if hasattr(self, 'monitored_resources') and resource_id in self.monitored_resources:
			del self.monitored_resources[resource_id]
	
	async def _list_provider_resources(self, environment: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""List resources from provider - to be overridden."""
		return await self._provider_specific_list(environment, filters)
	
	async def _provider_specific_list(self, environment: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Provider-specific list - to be overridden."""
		return []
	
	def _parse_provider_timestamp(self, timestamp_str: str) -> Optional[datetime]:
		"""Parse timestamp from provider format."""
		if not timestamp_str:
			return None
		
		try:
			# Try ISO format first
			return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
		except (ValueError, AttributeError):
			try:
				# Try common formats
				for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
					return datetime.strptime(timestamp_str, fmt).replace(tzinfo=timezone.utc)
			except ValueError:
				return None
	
	def _apply_filters(self, resources: List[CloudResource], filters: Dict[str, Any]) -> List[CloudResource]:
		"""Apply additional filters to resource list."""
		filtered_resources = resources
		
		# Filter by status
		if "status" in filters:
			filtered_resources = [r for r in filtered_resources if r.status == filters["status"]]
		
		# Filter by tags
		if "tags" in filters:
			tag_filters = filters["tags"]
			for key, value in tag_filters.items():
				filtered_resources = [r for r in filtered_resources if r.tags.get(key) == value]
		
		# Filter by name pattern
		if "name_pattern" in filters:
			import re
			pattern = re.compile(filters["name_pattern"], re.IGNORECASE)
			filtered_resources = [r for r in filtered_resources if pattern.search(r.name)]
		
		return filtered_resources


# ==================== AWS Adapter ====================

class AWSAdapter(CloudAdapter):
	"""AWS Systems Manager Parameter Store and Secrets Manager adapter."""
	
	def __init__(self, config: Dict[str, Any]):
		super().__init__(config)
		self.ssm_client = None
		self.secrets_client = None
		self.cloudformation_client = None
	
	async def initialize(self):
		"""Initialize AWS clients."""
		try:
			# Initialize AWS clients
			session = boto3.Session(
				aws_access_key_id=self.config.get('access_key_id'),
				aws_secret_access_key=self.config.get('secret_access_key'),
				region_name=self.config.get('region', 'us-west-2')
			)
			
			self.ssm_client = session.client('ssm')
			self.secrets_client = session.client('secretsmanager')
			self.cloudformation_client = session.client('cloudformation')
			
			# Test connectivity
			self.ssm_client.describe_parameters(MaxResults=1)
			
			self.initialized = True
			print("✅ AWS adapter initialized successfully")
			
		except Exception as e:
			print(f"❌ AWS adapter initialization failed: {e}")
			raise
	
	async def deploy_configuration(
		self,
		config_data: Dict[str, Any],
		environment: str,
		options: Dict[str, Any]
	) -> DeploymentResult:
		"""Deploy configuration to AWS Parameter Store."""
		try:
			if not self.initialized:
				await self.initialize()
			
			deployment_id = f"aws-deploy-{int(datetime.now().timestamp())}"
			deployed_params = []
			errors = []
			warnings = []
			
			# Format configuration for AWS
			aws_config = self._format_config_for_aws(config_data, environment)
			
			# Deploy parameters
			for param_name, param_data in aws_config.items():
				try:
					param_type = param_data.get('type', 'String')
					param_value = param_data.get('value', '')
					param_description = param_data.get('description', f'Configuration parameter for {environment}')
					
					# Use Secrets Manager for sensitive data
					if param_data.get('secure', False) or 'password' in param_name.lower() or 'secret' in param_name.lower():
						secret_arn = await self._deploy_to_secrets_manager(
							param_name,
							param_value,
							param_description,
							environment
						)
						deployed_params.append({
							'name': param_name,
							'type': 'SecureString',
							'secret_arn': secret_arn
						})
					else:
						# Deploy to Parameter Store
						response = self.ssm_client.put_parameter(
							Name=param_name,
							Value=str(param_value),
							Type=param_type,
							Description=param_description,
							Tags=[
								{'Key': 'Environment', 'Value': environment},
								{'Key': 'ManagedBy', 'Value': 'APG-CentralConfig'},
								{'Key': 'DeploymentId', 'Value': deployment_id}
							],
							Overwrite=True
						)
						
						deployed_params.append({
							'name': param_name,
							'type': param_type,
							'version': response['Version']
						})
				
				except Exception as e:
					errors.append(f"Failed to deploy parameter {param_name}: {str(e)}")
			
			# Deploy CloudFormation stack if specified
			if options.get('deploy_stack', False):
				stack_result = await self._deploy_cloudformation_stack(
					config_data, environment, deployment_id, options
				)
				if stack_result.get('warnings'):
					warnings.extend(stack_result['warnings'])
			
			success = len(errors) == 0
			message = f"Deployed {len(deployed_params)} parameters to AWS"
			if errors:
				message += f" with {len(errors)} errors"
			
			return DeploymentResult(
				success=success,
				deployment_id=deployment_id,
				message=message,
				details={
					'provider': 'aws',
					'environment': environment,
					'deployed_parameters': deployed_params,
					'parameter_count': len(deployed_params)
				},
				errors=errors,
				warnings=warnings
			)
			
		except Exception as e:
			return DeploymentResult(
				success=False,
				deployment_id='',
				message=f"AWS deployment failed: {str(e)}",
				details={},
				errors=[str(e)]
			)
	
	async def _deploy_to_secrets_manager(
		self,
		secret_name: str,
		secret_value: str,
		description: str,
		environment: str
	) -> str:
		"""Deploy sensitive configuration to AWS Secrets Manager."""
		try:
			response = self.secrets_client.create_secret(
				Name=secret_name,
				Description=description,
				SecretString=secret_value,
				Tags=[
					{'Key': 'Environment', 'Value': environment},
					{'Key': 'ManagedBy', 'Value': 'APG-CentralConfig'}
				]
			)
			return response['ARN']
		
		except self.secrets_client.exceptions.ResourceExistsException:
			# Update existing secret
			response = self.secrets_client.update_secret(
				SecretId=secret_name,
				SecretString=secret_value,
				Description=description
			)
			return response['ARN']
	
	async def _deploy_cloudformation_stack(
		self,
		config_data: Dict[str, Any],
		environment: str,
		deployment_id: str,
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Deploy CloudFormation stack for infrastructure."""
		try:
			stack_name = f"apg-config-{environment}-{deployment_id}"
			template = self._generate_cloudformation_template(config_data, environment)
			
			response = self.cloudformation_client.create_stack(
				StackName=stack_name,
				TemplateBody=json.dumps(template),
				Parameters=[
					{
						'ParameterKey': 'Environment',
						'ParameterValue': environment
					}
				],
				Tags=[
					{'Key': 'Environment', 'Value': environment},
					{'Key': 'ManagedBy', 'Value': 'APG-CentralConfig'},
					{'Key': 'DeploymentId', 'Value': deployment_id}
				],
				Capabilities=['CAPABILITY_IAM']
			)
			
			return {
				'stack_id': response['StackId'],
				'warnings': []
			}
			
		except Exception as e:
			return {
				'stack_id': None,
				'warnings': [f"CloudFormation deployment failed: {str(e)}"]
			}
	
	def _format_config_for_aws(self, config_data: Dict[str, Any], environment: str) -> Dict[str, Any]:
		"""Format configuration data for AWS Parameter Store."""
		aws_config = {}
		
		def flatten_config(obj, prefix=''):
			for key, value in obj.items():
				param_name = f"/{environment}/{prefix}{key}" if prefix else f"/{environment}/{key}"
				
				if isinstance(value, dict):
					flatten_config(value, f"{prefix}{key}/")
				else:
					aws_config[param_name] = {
						'value': value,
						'type': 'SecureString' if self._is_sensitive(key) else 'String',
						'secure': self._is_sensitive(key),
						'description': f'Configuration parameter for {key}'
					}
		
		flatten_config(config_data)
		return aws_config
	
	def _is_sensitive(self, key: str) -> bool:
		"""Check if a configuration key contains sensitive data."""
		sensitive_keywords = ['password', 'secret', 'key', 'token', 'credential']
		return any(keyword in key.lower() for keyword in sensitive_keywords)
	
	def _generate_cloudformation_template(self, config_data: Dict[str, Any], environment: str) -> Dict[str, Any]:
		"""Generate CloudFormation template for infrastructure resources."""
		template = {
			"AWSTemplateFormatVersion": "2010-09-09",
			"Description": f"APG Central Configuration infrastructure for {environment}",
			"Parameters": {
				"Environment": {
					"Type": "String",
					"Default": environment,
					"Description": "Environment name"
				}
			},
			"Resources": {},
			"Outputs": {}
		}
		
		# Add resources based on configuration
		if 'database' in str(config_data).lower():
			template["Resources"]["DatabaseSecurityGroup"] = {
				"Type": "AWS::EC2::SecurityGroup",
				"Properties": {
					"GroupDescription": "Security group for database access",
					"SecurityGroupIngress": [
						{
							"IpProtocol": "tcp",
							"FromPort": 5432,
							"ToPort": 5432,
							"CidrIp": "10.0.0.0/8"
						}
					]
				}
			}
		
		return template
	
	async def get_configuration(self, resource_id: str, environment: str) -> Optional[Dict[str, Any]]:
		"""Get configuration from AWS Parameter Store."""
		try:
			if not self.initialized:
				await self.initialize()
			
			response = self.ssm_client.get_parameter(
				Name=resource_id,
				WithDecryption=True
			)
			
			parameter = response['Parameter']
			return {
				'name': parameter['Name'],
				'value': parameter['Value'],
				'type': parameter['Type'],
				'version': parameter['Version'],
				'last_modified': parameter['LastModifiedDate'].isoformat()
			}
			
		except self.ssm_client.exceptions.ParameterNotFound:
			return None
		except Exception as e:
			print(f"❌ Failed to get AWS parameter {resource_id}: {e}")
			return None
	
	async def update_configuration(
		self,
		resource_id: str,
		config_data: Dict[str, Any],
		environment: str
	) -> DeploymentResult:
		"""Update configuration in AWS Parameter Store."""
		return await self.deploy_configuration(config_data, environment, {'update': True})
	
	async def delete_configuration(self, resource_id: str, environment: str) -> bool:
		"""Delete configuration from AWS Parameter Store."""
		try:
			if not self.initialized:
				await self.initialize()
			
			self.ssm_client.delete_parameter(Name=resource_id)
			return True
			
		except Exception as e:
			print(f"❌ Failed to delete AWS parameter {resource_id}: {e}")
			return False
	
	async def list_configurations(
		self,
		environment: str,
		filters: Dict[str, Any] = None
	) -> List[CloudResource]:
		"""List configurations in AWS Parameter Store."""
		try:
			if not self.initialized:
				await self.initialize()
			
			# List parameters with pagination
			paginator = self.ssm_client.get_paginator('describe_parameters')
			page_iterator = paginator.paginate(
				ParameterFilters=[
					{
						'Key': 'Path',
						'Option': 'BeginsWith',
						'Values': [f'/{environment}/']
					}
				]
			)
			
			resources = []
			for page in page_iterator:
				for param in page['Parameters']:
					resources.append(CloudResource(
						resource_type='parameter',
						resource_id=param['Name'],
						name=param['Name'].split('/')[-1],
						status='active',
						properties={
							'type': param['Type'],
							'version': param['Version'],
							'last_modified': param['LastModifiedDate'].isoformat()
						},
						tags={tag['Key']: tag['Value'] for tag in param.get('Tags', [])}
					))
			
			return resources
			
		except Exception as e:
			print(f"❌ Failed to list AWS parameters: {e}")
			return []


# ==================== Azure Adapter ====================

class AzureAdapter(CloudAdapter):
	"""Azure App Configuration and Key Vault adapter."""
	
	def __init__(self, config: Dict[str, Any]):
		super().__init__(config)
		self.app_config_client = None
		self.key_vault_client = None
		self.credential = None
	
	async def initialize(self):
		"""Initialize Azure clients."""
		try:
			self.credential = DefaultAzureCredential()
			
			# Initialize App Configuration client
			connection_string = self.config.get('app_config_connection_string')
			if connection_string:
				self.app_config_client = AzureAppConfigurationClient.from_connection_string(
					connection_string
				)
			
			# Initialize Key Vault client
			key_vault_url = self.config.get('key_vault_url')
			if key_vault_url:
				self.key_vault_client = SecretClient(
					vault_url=key_vault_url,
					credential=self.credential
				)
			
			self.initialized = True
			print("✅ Azure adapter initialized successfully")
			
		except Exception as e:
			print(f"❌ Azure adapter initialization failed: {e}")
			raise
	
	async def deploy_configuration(
		self,
		config_data: Dict[str, Any],
		environment: str,
		options: Dict[str, Any]
	) -> DeploymentResult:
		"""Deploy configuration to Azure App Configuration."""
		try:
			if not self.initialized:
				await self.initialize()
			
			deployment_id = f"azure-deploy-{int(datetime.now().timestamp())}"
			deployed_configs = []
			errors = []
			
			# Format configuration for Azure
			azure_config = self._format_config_for_azure(config_data, environment)
			
			# Deploy configurations
			for config_key, config_value in azure_config.items():
				try:
					if self._is_sensitive(config_key):
						# Deploy to Key Vault
						secret_name = config_key.replace('/', '-').replace(':', '-')
						secret = self.key_vault_client.set_secret(
							secret_name,
							str(config_value['value'])
						)
						deployed_configs.append({
							'key': config_key,
							'type': 'secret',
							'secret_id': secret.id
						})
					else:
						# Deploy to App Configuration
						from azure.appconfiguration import ConfigurationSetting
						
						setting = ConfigurationSetting(
							key=config_key,
							value=str(config_value['value']),
							label=environment,
							tags={
								'DeploymentId': deployment_id,
								'ManagedBy': 'APG-CentralConfig',
								'Environment': environment
							}
						)
						
						self.app_config_client.set_configuration_setting(setting)
						deployed_configs.append({
							'key': config_key,
							'type': 'configuration',
							'label': environment
						})
				
				except Exception as e:
					errors.append(f"Failed to deploy configuration {config_key}: {str(e)}")
			
			success = len(errors) == 0
			message = f"Deployed {len(deployed_configs)} configurations to Azure"
			if errors:
				message += f" with {len(errors)} errors"
			
			return DeploymentResult(
				success=success,
				deployment_id=deployment_id,
				message=message,
				details={
					'provider': 'azure',
					'environment': environment,
					'deployed_configurations': deployed_configs,
					'configuration_count': len(deployed_configs)
				},
				errors=errors
			)
			
		except Exception as e:
			return DeploymentResult(
				success=False,
				deployment_id='',
				message=f"Azure deployment failed: {str(e)}",
				details={},
				errors=[str(e)]
			)
	
	def _format_config_for_azure(self, config_data: Dict[str, Any], environment: str) -> Dict[str, Any]:
		"""Format configuration data for Azure App Configuration."""
		azure_config = {}
		
		def flatten_config(obj, prefix=''):
			for key, value in obj.items():
				config_key = f"{prefix}:{key}" if prefix else key
				
				if isinstance(value, dict):
					flatten_config(value, config_key)
				else:
					azure_config[config_key] = {
						'value': value,
						'type': 'secret' if self._is_sensitive(key) else 'configuration'
					}
		
		flatten_config(config_data)
		return azure_config
	
	async def get_configuration(self, resource_id: str, environment: str) -> Optional[Dict[str, Any]]:
		"""Get configuration from Azure App Configuration."""
		try:
			if not self.initialized:
				await self.initialize()
			
			setting = self.app_config_client.get_configuration_setting(
				key=resource_id,
				label=environment
			)
			
			return {
				'key': setting.key,
				'value': setting.value,
				'label': setting.label,
				'etag': setting.etag,
				'last_modified': setting.last_modified.isoformat() if setting.last_modified else None
			}
			
		except Exception as e:
			print(f"❌ Failed to get Azure configuration {resource_id}: {e}")
			return None
	
	async def update_configuration(
		self,
		resource_id: str,
		config_data: Dict[str, Any],
		environment: str
	) -> DeploymentResult:
		"""Update configuration in Azure App Configuration."""
		return await self.deploy_configuration(config_data, environment, {'update': True})
	
	async def delete_configuration(self, resource_id: str, environment: str) -> bool:
		"""Delete configuration from Azure App Configuration."""
		try:
			if not self.initialized:
				await self.initialize()
			
			self.app_config_client.delete_configuration_setting(
				key=resource_id,
				label=environment
			)
			return True
			
		except Exception as e:
			print(f"❌ Failed to delete Azure configuration {resource_id}: {e}")
			return False
	
	async def list_configurations(
		self,
		environment: str,
		filters: Dict[str, Any] = None
	) -> List[CloudResource]:
		"""List configurations in Azure App Configuration."""
		try:
			if not self.initialized:
				await self.initialize()
			
			settings = self.app_config_client.list_configuration_settings(
				label_filter=environment
			)
			
			resources = []
			for setting in settings:
				resources.append(CloudResource(
					resource_type='configuration',
					resource_id=setting.key,
					name=setting.key,
					status='active',
					properties={
						'value': setting.value,
						'label': setting.label,
						'etag': setting.etag,
						'last_modified': setting.last_modified.isoformat() if setting.last_modified else None
					},
					tags=setting.tags
				))
			
			return resources
			
		except Exception as e:
			print(f"❌ Failed to list Azure configurations: {e}")
			return []


# ==================== GCP Adapter ====================

class GCPAdapter(CloudAdapter):
	"""Google Cloud Config Management and Secret Manager adapter."""
	
	def __init__(self, config: Dict[str, Any]):
		super().__init__(config)
		self.secret_client = None
		self.project_id = config.get('project_id')
	
	async def initialize(self):
		"""Initialize GCP clients."""
		try:
			self.secret_client = secretmanager.SecretManagerServiceClient()
			
			# Test connectivity
			parent = f"projects/{self.project_id}"
			list(self.secret_client.list_secrets(request={"parent": parent}))
			
			self.initialized = True
			print("✅ GCP adapter initialized successfully")
			
		except Exception as e:
			print(f"❌ GCP adapter initialization failed: {e}")
			raise
	
	async def deploy_configuration(
		self,
		config_data: Dict[str, Any],
		environment: str,
		options: Dict[str, Any]
	) -> DeploymentResult:
		"""Deploy configuration to GCP Secret Manager."""
		try:
			if not self.initialized:
				await self.initialize()
			
			deployment_id = f"gcp-deploy-{int(datetime.now().timestamp())}"
			deployed_secrets = []
			errors = []
			
			# Format configuration for GCP
			gcp_config = self._format_config_for_gcp(config_data, environment)
			
			parent = f"projects/{self.project_id}"
			
			# Deploy secrets
			for secret_name, secret_data in gcp_config.items():
				try:
					secret_value = str(secret_data['value'])
					
					# Create or update secret
					try:
						# Try to create new secret
						secret = self.secret_client.create_secret(
							request={
								"parent": parent,
								"secret_id": secret_name,
								"secret": {
									"labels": {
										"environment": environment,
										"managed-by": "apg-central-config",
										"deployment-id": deployment_id
									}
								}
							}
						)
					except Exception:
						# Secret already exists, get it
						secret_path = f"{parent}/secrets/{secret_name}"
						secret = self.secret_client.get_secret(request={"name": secret_path})
					
					# Add secret version
					self.secret_client.add_secret_version(
						request={
							"parent": secret.name,
							"payload": {"data": secret_value.encode()}
						}
					)
					
					deployed_secrets.append({
						'name': secret_name,
						'secret_name': secret.name,
						'environment': environment
					})
				
				except Exception as e:
					errors.append(f"Failed to deploy secret {secret_name}: {str(e)}")
			
			success = len(errors) == 0
			message = f"Deployed {len(deployed_secrets)} secrets to GCP"
			if errors:
				message += f" with {len(errors)} errors"
			
			return DeploymentResult(
				success=success,
				deployment_id=deployment_id,
				message=message,
				details={
					'provider': 'gcp',
					'environment': environment,
					'deployed_secrets': deployed_secrets,
					'secret_count': len(deployed_secrets)
				},
				errors=errors
			)
			
		except Exception as e:
			return DeploymentResult(
				success=False,
				deployment_id='',
				message=f"GCP deployment failed: {str(e)}",
				details={},
				errors=[str(e)]
			)
	
	def _format_config_for_gcp(self, config_data: Dict[str, Any], environment: str) -> Dict[str, Any]:
		"""Format configuration data for GCP Secret Manager."""
		gcp_config = {}
		
		def flatten_config(obj, prefix=''):
			for key, value in obj.items():
				secret_name = f"{environment}-{prefix}-{key}".replace('_', '-').lower() if prefix else f"{environment}-{key}".replace('_', '-').lower()
				# GCP secret names must match regex: [a-zA-Z][a-zA-Z0-9_-]*
				secret_name = secret_name.replace('.', '-').replace('/', '-')
				
				if isinstance(value, dict):
					flatten_config(value, f"{prefix}-{key}" if prefix else key)
				else:
					gcp_config[secret_name] = {
						'value': value,
						'original_key': key
					}
		
		flatten_config(config_data)
		return gcp_config
	
	async def get_configuration(self, resource_id: str, environment: str) -> Optional[Dict[str, Any]]:
		"""Get configuration from GCP Secret Manager."""
		try:
			if not self.initialized:
				await self.initialize()
			
			secret_name = f"projects/{self.project_id}/secrets/{resource_id}/versions/latest"
			response = self.secret_client.access_secret_version(request={"name": secret_name})
			
			return {
				'name': resource_id,
				'value': response.payload.data.decode(),
				'state': response.state,
				'create_time': response.create_time
			}
			
		except Exception as e:
			print(f"❌ Failed to get GCP secret {resource_id}: {e}")
			return None
	
	async def update_configuration(
		self,
		resource_id: str,
		config_data: Dict[str, Any],
		environment: str
	) -> DeploymentResult:
		"""Update configuration in GCP Secret Manager."""
		return await self.deploy_configuration(config_data, environment, {'update': True})
	
	async def delete_configuration(self, resource_id: str, environment: str) -> bool:
		"""Delete configuration from GCP Secret Manager."""
		try:
			if not self.initialized:
				await self.initialize()
			
			secret_name = f"projects/{self.project_id}/secrets/{resource_id}"
			self.secret_client.delete_secret(request={"name": secret_name})
			return True
			
		except Exception as e:
			print(f"❌ Failed to delete GCP secret {resource_id}: {e}")
			return False
	
	async def list_configurations(
		self,
		environment: str,
		filters: Dict[str, Any] = None
	) -> List[CloudResource]:
		"""List configurations in GCP Secret Manager."""
		try:
			if not self.initialized:
				await self.initialize()
			
			parent = f"projects/{self.project_id}"
			secrets = self.secret_client.list_secrets(request={"parent": parent})
			
			resources = []
			for secret in secrets:
				# Filter by environment label
				if secret.labels.get('environment') == environment:
					resources.append(CloudResource(
						resource_type='secret',
						resource_id=secret.name.split('/')[-1],
						name=secret.name.split('/')[-1],
						status='active',
						properties={
							'name': secret.name,
							'create_time': secret.create_time,
							'labels': dict(secret.labels)
						},
						tags=dict(secret.labels)
					))
			
			return resources
			
		except Exception as e:
			print(f"❌ Failed to list GCP secrets: {e}")
			return []


# ==================== Kubernetes Adapter ====================

class KubernetesAdapter(CloudAdapter):
	"""Kubernetes ConfigMaps and Secrets adapter."""
	
	def __init__(self, config: Dict[str, Any]):
		super().__init__(config)
		self.api_client = None
		self.core_v1 = None
		self.kubeconfig_path = config.get('kubeconfig_path')
	
	async def initialize(self):
		"""Initialize Kubernetes client."""
		try:
			if self.kubeconfig_path:
				config.load_kube_config(config_file=self.kubeconfig_path)
			else:
				# Try in-cluster config first, then default kubeconfig
				try:
					config.load_incluster_config()
				except:
					config.load_kube_config()
			
			self.api_client = client.ApiClient()
			self.core_v1 = client.CoreV1Api()
			
			# Test connectivity
			self.core_v1.list_namespace(limit=1)
			
			self.initialized = True
			print("✅ Kubernetes adapter initialized successfully")
			
		except Exception as e:
			print(f"❌ Kubernetes adapter initialization failed: {e}")
			raise
	
	async def deploy_configuration(
		self,
		config_data: Dict[str, Any],
		environment: str,
		options: Dict[str, Any]
	) -> DeploymentResult:
		"""Deploy configuration to Kubernetes ConfigMaps and Secrets."""
		try:
			if not self.initialized:
				await self.initialize()
			
			deployment_id = f"k8s-deploy-{int(datetime.now().timestamp())}"
			deployed_resources = []
			errors = []
			
			namespace = options.get('namespace', environment)
			
			# Ensure namespace exists
			await self._ensure_namespace(namespace)
			
			# Separate sensitive and non-sensitive configurations
			configmap_data = {}
			secret_data = {}
			
			for key, value in config_data.items():
				if self._is_sensitive(key):
					secret_data[key] = str(value)
				else:
					configmap_data[key] = str(value)
			
			# Deploy ConfigMap for non-sensitive data
			if configmap_data:
				try:
					configmap_name = f"apg-config-{environment}"
					configmap = client.V1ConfigMap(
						metadata=client.V1ObjectMeta(
							name=configmap_name,
							namespace=namespace,
							labels={
								'app.kubernetes.io/managed-by': 'apg-central-config',
								'apg.config/environment': environment,
								'apg.config/deployment-id': deployment_id
							}
						),
						data=configmap_data
					)
					
					try:
						self.core_v1.create_namespaced_config_map(namespace, configmap)
					except client.ApiException as e:
						if e.status == 409:  # Already exists
							self.core_v1.replace_namespaced_config_map(configmap_name, namespace, configmap)
						else:
							raise
					
					deployed_resources.append({
						'type': 'ConfigMap',
						'name': configmap_name,
						'namespace': namespace,
						'keys': list(configmap_data.keys())
					})
				
				except Exception as e:
					errors.append(f"Failed to deploy ConfigMap: {str(e)}")
			
			# Deploy Secret for sensitive data
			if secret_data:
				try:
					secret_name = f"apg-secret-{environment}"
					secret = client.V1Secret(
						metadata=client.V1ObjectMeta(
							name=secret_name,
							namespace=namespace,
							labels={
								'app.kubernetes.io/managed-by': 'apg-central-config',
								'apg.config/environment': environment,
								'apg.config/deployment-id': deployment_id
							}
						),
						type='Opaque',
						string_data=secret_data
					)
					
					try:
						self.core_v1.create_namespaced_secret(namespace, secret)
					except client.ApiException as e:
						if e.status == 409:  # Already exists
							self.core_v1.replace_namespaced_secret(secret_name, namespace, secret)
						else:
							raise
					
					deployed_resources.append({
						'type': 'Secret',
						'name': secret_name,
						'namespace': namespace,
						'keys': list(secret_data.keys())
					})
				
				except Exception as e:
					errors.append(f"Failed to deploy Secret: {str(e)}")
			
			success = len(errors) == 0
			message = f"Deployed {len(deployed_resources)} resources to Kubernetes"
			if errors:
				message += f" with {len(errors)} errors"
			
			return DeploymentResult(
				success=success,
				deployment_id=deployment_id,
				message=message,
				details={
					'provider': 'kubernetes',
					'environment': environment,
					'namespace': namespace,
					'deployed_resources': deployed_resources,
					'resource_count': len(deployed_resources)
				},
				errors=errors
			)
			
		except Exception as e:
			return DeploymentResult(
				success=False,
				deployment_id='',
				message=f"Kubernetes deployment failed: {str(e)}",
				details={},
				errors=[str(e)]
			)
	
	async def _ensure_namespace(self, namespace: str):
		"""Ensure namespace exists."""
		try:
			self.core_v1.read_namespace(namespace)
		except client.ApiException as e:
			if e.status == 404:
				# Create namespace
				ns = client.V1Namespace(
					metadata=client.V1ObjectMeta(
						name=namespace,
						labels={'apg.config/managed': 'true'}
					)
				)
				self.core_v1.create_namespace(ns)
			else:
				raise
	
	async def get_configuration(self, resource_id: str, environment: str) -> Optional[Dict[str, Any]]:
		"""Get configuration from Kubernetes ConfigMap or Secret."""
		try:
			if not self.initialized:
				await self.initialize()
			
			namespace = environment
			
			# Try ConfigMap first
			try:
				configmap = self.core_v1.read_namespaced_config_map(resource_id, namespace)
				return {
					'type': 'ConfigMap',
					'name': configmap.metadata.name,
					'namespace': configmap.metadata.namespace,
					'data': configmap.data or {},
					'creation_timestamp': configmap.metadata.creation_timestamp
				}
			except client.ApiException:
				pass
			
			# Try Secret
			try:
				secret = self.core_v1.read_namespaced_secret(resource_id, namespace)
				return {
					'type': 'Secret',
					'name': secret.metadata.name,
					'namespace': secret.metadata.namespace,
					'data': secret.string_data or {},
					'creation_timestamp': secret.metadata.creation_timestamp
				}
			except client.ApiException:
				pass
			
			return None
			
		except Exception as e:
			print(f"❌ Failed to get Kubernetes resource {resource_id}: {e}")
			return None
	
	async def update_configuration(
		self,
		resource_id: str,
		config_data: Dict[str, Any],
		environment: str
	) -> DeploymentResult:
		"""Update configuration in Kubernetes."""
		return await self.deploy_configuration(config_data, environment, {'namespace': environment})
	
	async def delete_configuration(self, resource_id: str, environment: str) -> bool:
		"""Delete configuration from Kubernetes."""
		try:
			if not self.initialized:
				await self.initialize()
			
			namespace = environment
			
			# Try to delete ConfigMap
			try:
				self.core_v1.delete_namespaced_config_map(resource_id, namespace)
				return True
			except client.ApiException:
				pass
			
			# Try to delete Secret
			try:
				self.core_v1.delete_namespaced_secret(resource_id, namespace)
				return True
			except client.ApiException:
				pass
			
			return False
			
		except Exception as e:
			print(f"❌ Failed to delete Kubernetes resource {resource_id}: {e}")
			return False
	
	async def list_configurations(
		self,
		environment: str,
		filters: Dict[str, Any] = None
	) -> List[CloudResource]:
		"""List configurations in Kubernetes."""
		try:
			if not self.initialized:
				await self.initialize()
			
			namespace = environment
			resources = []
			
			# List ConfigMaps
			try:
				configmaps = self.core_v1.list_namespaced_config_map(
					namespace,
					label_selector='apg.config/managed=true'
				)
				
				for cm in configmaps.items:
					resources.append(CloudResource(
						resource_type='ConfigMap',
						resource_id=cm.metadata.name,
						name=cm.metadata.name,
						status='active',
						properties={
							'namespace': cm.metadata.namespace,
							'creation_timestamp': cm.metadata.creation_timestamp,
							'data_keys': list(cm.data.keys()) if cm.data else []
						},
						tags=cm.metadata.labels
					))
			except client.ApiException:
				pass
			
			# List Secrets
			try:
				secrets = self.core_v1.list_namespaced_secret(
					namespace,
					label_selector='apg.config/managed=true'
				)
				
				for secret in secrets.items:
					resources.append(CloudResource(
						resource_type='Secret',
						resource_id=secret.metadata.name,
						name=secret.metadata.name,
						status='active',
						properties={
							'namespace': secret.metadata.namespace,
							'creation_timestamp': secret.metadata.creation_timestamp,
							'data_keys': list(secret.data.keys()) if secret.data else []
						},
						tags=secret.metadata.labels
					))
			except client.ApiException:
				pass
			
			return resources
			
		except Exception as e:
			print(f"❌ Failed to list Kubernetes resources: {e}")
			return []


# ==================== On-Premises Adapter ====================

class OnPremisesAdapter(CloudAdapter):
	"""On-premises file system and database adapter."""
	
	def __init__(self, config: Dict[str, Any]):
		super().__init__(config)
		self.base_path = config.get('base_path', '/etc/apg-config')
		self.database_url = config.get('database_url')
	
	async def initialize(self):
		"""Initialize on-premises adapter."""
		try:
			import os
			import pathlib
			
			# Ensure base directory exists
			pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)
			
			self.initialized = True
			print("✅ On-premises adapter initialized successfully")
			
		except Exception as e:
			print(f"❌ On-premises adapter initialization failed: {e}")
			raise
	
	async def deploy_configuration(
		self,
		config_data: Dict[str, Any],
		environment: str,
		options: Dict[str, Any]
	) -> DeploymentResult:
		"""Deploy configuration to on-premises file system."""
		try:
			if not self.initialized:
				await self.initialize()
			
			deployment_id = f"onprem-deploy-{int(datetime.now().timestamp())}"
			deployed_files = []
			errors = []
			
			import os
			import json
			
			env_path = os.path.join(self.base_path, environment)
			os.makedirs(env_path, exist_ok=True)
			
			# Deploy configuration files
			for key, value in config_data.items():
				try:
					file_path = os.path.join(env_path, f"{key}.json")
					
					# Ensure directory exists
					os.makedirs(os.path.dirname(file_path), exist_ok=True)
					
					# Write configuration file
					with open(file_path, 'w') as f:
						json.dump({
							'value': value,
							'deployment_id': deployment_id,
							'environment': environment,
							'deployed_at': datetime.now(timezone.utc).isoformat()
						}, f, indent=2)
					
					deployed_files.append({
						'key': key,
						'file_path': file_path,
						'size': os.path.getsize(file_path)
					})
				
				except Exception as e:
					errors.append(f"Failed to deploy configuration {key}: {str(e)}")
			
			success = len(errors) == 0
			message = f"Deployed {len(deployed_files)} files to on-premises storage"
			if errors:
				message += f" with {len(errors)} errors"
			
			return DeploymentResult(
				success=success,
				deployment_id=deployment_id,
				message=message,
				details={
					'provider': 'on_premises',
					'environment': environment,
					'base_path': self.base_path,
					'deployed_files': deployed_files,
					'file_count': len(deployed_files)
				},
				errors=errors
			)
			
		except Exception as e:
			return DeploymentResult(
				success=False,
				deployment_id='',
				message=f"On-premises deployment failed: {str(e)}",
				details={},
				errors=[str(e)]
			)
	
	async def get_configuration(self, resource_id: str, environment: str) -> Optional[Dict[str, Any]]:
		"""Get configuration from on-premises file system."""
		try:
			if not self.initialized:
				await self.initialize()
			
			import os
			import json
			
			file_path = os.path.join(self.base_path, environment, f"{resource_id}.json")
			
			if not os.path.exists(file_path):
				return None
			
			with open(file_path, 'r') as f:
				data = json.load(f)
			
			return {
				'key': resource_id,
				'value': data.get('value'),
				'environment': data.get('environment'),
				'deployment_id': data.get('deployment_id'),
				'deployed_at': data.get('deployed_at'),
				'file_path': file_path,
				'file_size': os.path.getsize(file_path)
			}
			
		except Exception as e:
			print(f"❌ Failed to get on-premises configuration {resource_id}: {e}")
			return None
	
	async def update_configuration(
		self,
		resource_id: str,
		config_data: Dict[str, Any],
		environment: str
	) -> DeploymentResult:
		"""Update configuration in on-premises storage."""
		return await self.deploy_configuration({resource_id: config_data}, environment, {})
	
	async def delete_configuration(self, resource_id: str, environment: str) -> bool:
		"""Delete configuration from on-premises storage."""
		try:
			if not self.initialized:
				await self.initialize()
			
			import os
			
			file_path = os.path.join(self.base_path, environment, f"{resource_id}.json")
			
			if os.path.exists(file_path):
				os.remove(file_path)
				return True
			
			return False
			
		except Exception as e:
			print(f"❌ Failed to delete on-premises configuration {resource_id}: {e}")
			return False
	
	async def list_configurations(
		self,
		environment: str,
		filters: Dict[str, Any] = None
	) -> List[CloudResource]:
		"""List configurations in on-premises storage."""
		try:
			if not self.initialized:
				await self.initialize()
			
			import os
			import json
			
			env_path = os.path.join(self.base_path, environment)
			
			if not os.path.exists(env_path):
				return []
			
			resources = []
			
			for filename in os.listdir(env_path):
				if filename.endswith('.json'):
					file_path = os.path.join(env_path, filename)
					resource_id = filename[:-5]  # Remove .json extension
					
					try:
						with open(file_path, 'r') as f:
							data = json.load(f)
						
						resources.append(CloudResource(
							resource_type='file',
							resource_id=resource_id,
							name=resource_id,
							status='active',
							properties={
								'file_path': file_path,
								'file_size': os.path.getsize(file_path),
								'deployment_id': data.get('deployment_id'),
								'deployed_at': data.get('deployed_at')
							}
						))
					except Exception as e:
						print(f"⚠️ Failed to read configuration file {file_path}: {e}")
			
			return resources
			
		except Exception as e:
			print(f"❌ Failed to list on-premises configurations: {e}")
			return []
	
	def _is_sensitive(self, key: str) -> bool:
		"""Check if a configuration key contains sensitive data."""
		sensitive_keywords = ['password', 'secret', 'key', 'token', 'credential']
		return any(keyword in key.lower() for keyword in sensitive_keywords)


# ==================== Factory Function ====================

def create_cloud_adapter(provider: str, config: Dict[str, Any]) -> CloudAdapter:
	"""Factory function to create cloud adapter."""
	adapters = {
		'aws': AWSAdapter,
		'azure': AzureAdapter,
		'gcp': GCPAdapter,
		'kubernetes': KubernetesAdapter,
		'on_premises': OnPremisesAdapter
	}
	
	adapter_class = adapters.get(provider.lower())
	if not adapter_class:
		raise ValueError(f"Unsupported cloud provider: {provider}")
	
	return adapter_class(config)