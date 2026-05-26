"""
Production Deployment Scripts for AICR

This module provides comprehensive deployment automation including:
- Multi-environment deployment orchestration
- Blue-green and canary deployment strategies
- Infrastructure provisioning and configuration
- Database migration and schema management
- SSL certificate management
- Health checks and rollback procedures
- Monitoring and alerting setup
- Security hardening and compliance checks

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import tempfile
import shutil

import aiofiles
import aiohttp
import yaml
from pydantic import BaseModel, Field, ConfigDict
from kubernetes import client, config as k8s_config
from kubernetes.client.rest import ApiException

from .production_config import (
	ProductionConfig, ConfigurationManager, Environment,
	DatabaseConfig, CacheConfig, SecurityConfig, MonitoringConfig
)
from uuid_extensions import uuid7str


class DeploymentStrategy(str, Enum):
	"""Deployment strategies."""
	ROLLING_UPDATE = "rolling_update"
	BLUE_GREEN = "blue_green"
	CANARY = "canary"
	RECREATE = "recreate"


class DeploymentStatus(str, Enum):
	"""Deployment status."""
	PENDING = "pending"
	RUNNING = "running"
	SUCCESS = "success"
	FAILED = "failed"
	ROLLED_BACK = "rolled_back"


class ValidationStatus(str, Enum):
	"""Validation status."""
	PENDING = "pending"
	RUNNING = "running"
	PASSED = "passed"
	FAILED = "failed"


class DeploymentPhase(str, Enum):
	"""Deployment phases."""
	PRE_VALIDATION = "pre_validation"
	INFRASTRUCTURE = "infrastructure"
	DATABASE_MIGRATION = "database_migration"
	APPLICATION_DEPLOYMENT = "application_deployment"
	POST_DEPLOYMENT_VALIDATION = "post_deployment_validation"
	MONITORING_SETUP = "monitoring_setup"
	CLEANUP = "cleanup"


class DeploymentResult(BaseModel):
	"""Deployment result."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	deployment_id: str = Field(default_factory=uuid7str)
	strategy: DeploymentStrategy
	environment: Environment
	status: DeploymentStatus
	start_time: datetime = Field(default_factory=datetime.utcnow)
	end_time: Optional[datetime] = None
	duration_seconds: Optional[int] = None
	success: bool = False
	error_message: Optional[str] = None
	phases: Dict[DeploymentPhase, Dict[str, Any]] = Field(default_factory=dict)
	rollback_available: bool = False
	previous_version: Optional[str] = None
	deployed_version: Optional[str] = None
	health_check_url: Optional[str] = None
	metrics: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
	"""Validation result."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	validation_id: str = Field(default_factory=uuid7str)
	status: ValidationStatus
	checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
	overall_success: bool = False
	error_count: int = 0
	warning_count: int = 0
	start_time: datetime = Field(default_factory=datetime.utcnow)
	end_time: Optional[datetime] = None
	recommendations: List[str] = Field(default_factory=list)


class DatabaseMigration(BaseModel):
	"""Database migration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	migration_id: str = Field(default_factory=uuid7str)
	version: str
	name: str
	sql_file: str
	rollback_sql_file: Optional[str] = None
	dependencies: List[str] = Field(default_factory=list)
	applied: bool = False
	applied_at: Optional[datetime] = None
	checksum: Optional[str] = None


class SSLCertificate(BaseModel):
	"""SSL certificate."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	domain: str
	cert_path: str
	key_path: str
	ca_path: Optional[str] = None
	expiry_date: datetime
	auto_renewal: bool = True
	provider: str = "letsencrypt"


class ProductionDeployer:
	"""Production deployment orchestrator."""

	def __init__(self, config: ProductionConfig):
		self.config = config
		self.config_manager = ConfigurationManager()
		self.logger = logging.getLogger(f"{__name__}.ProductionDeployer")
		self.k8s_client = None
		self._deployment_history: List[DeploymentResult] = []

	async def initialize(self) -> None:
		"""Initialize deployment environment."""
		try:
			# Load Kubernetes configuration
			if self.config.environment == Environment.PRODUCTION:
				k8s_config.load_incluster_config()
			else:
				k8s_config.load_kube_config()

			self.k8s_client = client.ApiClient()

			# Verify cluster connectivity
			v1 = client.CoreV1Api(self.k8s_client)
			namespaces = v1.list_namespace()

			self.logger.info(f"Connected to Kubernetes cluster with {len(namespaces.items)} namespaces")

		except Exception as e:
			self.logger.error(f"Failed to initialize deployment environment: {e}")
			raise

	async def deploy(
		self,
		strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
		version: Optional[str] = None,
		dry_run: bool = False,
		force: bool = False
	) -> DeploymentResult:
		"""Deploy AICR to production."""
		deployment_result = DeploymentResult(
			strategy=strategy,
			environment=self.config.environment,
			status=DeploymentStatus.RUNNING,
			deployed_version=version or "latest"
		)

		try:
			self.logger.info(f"Starting {strategy.value} deployment to {self.config.environment.value}")

			# Phase 1: Pre-deployment validation
			await self._execute_phase(
				deployment_result,
				DeploymentPhase.PRE_VALIDATION,
				self._pre_deployment_validation,
				dry_run=dry_run
			)

			# Phase 2: Infrastructure setup
			await self._execute_phase(
				deployment_result,
				DeploymentPhase.INFRASTRUCTURE,
				self._setup_infrastructure,
				dry_run=dry_run
			)

			# Phase 3: Database migration
			await self._execute_phase(
				deployment_result,
				DeploymentPhase.DATABASE_MIGRATION,
				self._run_database_migrations,
				dry_run=dry_run
			)

			# Phase 4: Application deployment
			await self._execute_phase(
				deployment_result,
				DeploymentPhase.APPLICATION_DEPLOYMENT,
				self._deploy_application,
				strategy=strategy,
				version=version,
				dry_run=dry_run,
				force=force
			)

			# Phase 5: Post-deployment validation
			await self._execute_phase(
				deployment_result,
				DeploymentPhase.POST_DEPLOYMENT_VALIDATION,
				self._post_deployment_validation,
				dry_run=dry_run
			)

			# Phase 6: Monitoring setup
			await self._execute_phase(
				deployment_result,
				DeploymentPhase.MONITORING_SETUP,
				self._setup_monitoring,
				dry_run=dry_run
			)

			# Phase 7: Cleanup
			await self._execute_phase(
				deployment_result,
				DeploymentPhase.CLEANUP,
				self._cleanup_deployment,
				dry_run=dry_run
			)

			# Mark deployment as successful
			deployment_result.status = DeploymentStatus.SUCCESS
			deployment_result.success = True
			deployment_result.end_time = datetime.utcnow()
			deployment_result.duration_seconds = int(
				(deployment_result.end_time - deployment_result.start_time).total_seconds()
			)

			self.logger.info(f"Deployment completed successfully in {deployment_result.duration_seconds} seconds")

		except Exception as e:
			deployment_result.status = DeploymentStatus.FAILED
			deployment_result.error_message = str(e)
			deployment_result.end_time = datetime.utcnow()
			deployment_result.duration_seconds = int(
				(deployment_result.end_time - deployment_result.start_time).total_seconds()
			)

			self.logger.error(f"Deployment failed: {e}")

			# Attempt rollback if not a dry run
			if not dry_run and self._deployment_history:
				self.logger.info("Attempting automatic rollback...")
				try:
					await self.rollback()
					deployment_result.status = DeploymentStatus.ROLLED_BACK
				except Exception as rollback_error:
					self.logger.error(f"Rollback failed: {rollback_error}")

		finally:
			# Add to deployment history
			self._deployment_history.append(deployment_result)

			# Clean up temporary resources
			await self._cleanup_temp_resources()

		return deployment_result

	async def rollback(self, target_version: Optional[str] = None) -> DeploymentResult:
		"""Rollback to previous deployment."""
		try:
			if not self._deployment_history:
				raise ValueError("No previous deployments found for rollback")

			# Find target deployment
			target_deployment = None
			if target_version:
				for deployment in reversed(self._deployment_history):
					if deployment.deployed_version == target_version and deployment.success:
						target_deployment = deployment
						break
			else:
				# Find last successful deployment
				for deployment in reversed(self._deployment_history):
					if deployment.success:
						target_deployment = deployment
						break

			if not target_deployment:
				raise ValueError("No suitable deployment found for rollback")

			self.logger.info(f"Rolling back to version: {target_deployment.deployed_version}")

			# Execute rollback deployment
			rollback_result = await self.deploy(
				strategy=DeploymentStrategy.ROLLING_UPDATE,
				version=target_deployment.deployed_version,
				dry_run=False,
				force=True
			)

			return rollback_result

		except Exception as e:
			self.logger.error(f"Rollback failed: {e}")
			raise

	async def validate_deployment(self) -> ValidationResult:
		"""Validate current deployment."""
		validation_result = ValidationResult(status=ValidationStatus.RUNNING)

		try:
			# Infrastructure validation
			await self._validate_infrastructure(validation_result)

			# Application health validation
			await self._validate_application_health(validation_result)

			# Database connectivity validation
			await self._validate_database_connectivity(validation_result)

			# Security validation
			await self._validate_security_configuration(validation_result)

			# Performance validation
			await self._validate_performance_metrics(validation_result)

			# Determine overall status
			validation_result.overall_success = validation_result.error_count == 0
			validation_result.status = ValidationStatus.PASSED if validation_result.overall_success else ValidationStatus.FAILED
			validation_result.end_time = datetime.utcnow()

			self.logger.info(f"Validation completed: {validation_result.status}")

		except Exception as e:
			validation_result.status = ValidationStatus.FAILED
			validation_result.error_count += 1
			validation_result.checks['validation_error'] = {
				'status': 'failed',
				'message': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}

			self.logger.error(f"Validation failed: {e}")

		return validation_result

	async def _execute_phase(
		self,
		deployment_result: DeploymentResult,
		phase: DeploymentPhase,
		phase_func: Callable,
		**kwargs
	) -> None:
		"""Execute deployment phase."""
		phase_start = datetime.utcnow()
		phase_result = {
			'status': 'running',
			'start_time': phase_start.isoformat(),
			'messages': []
		}

		deployment_result.phases[phase] = phase_result

		try:
			self.logger.info(f"Executing phase: {phase.value}")

			# Execute phase function
			await phase_func(**kwargs)

			phase_result['status'] = 'success'
			phase_result['end_time'] = datetime.utcnow().isoformat()
			phase_result['duration_seconds'] = int(
				(datetime.utcnow() - phase_start).total_seconds()
			)

			self.logger.info(f"Phase {phase.value} completed successfully")

		except Exception as e:
			phase_result['status'] = 'failed'
			phase_result['error_message'] = str(e)
			phase_result['end_time'] = datetime.utcnow().isoformat()

			self.logger.error(f"Phase {phase.value} failed: {e}")
			raise

	async def _pre_deployment_validation(self, dry_run: bool = False) -> None:
		"""Pre-deployment validation."""
		try:
			# Validate Kubernetes cluster
			await self._validate_k8s_cluster()

			# Validate configuration
			await self._validate_configuration()

			# Validate dependencies
			await self._validate_dependencies()

			# Validate resources
			await self._validate_resources()

			# Validate SSL certificates
			await self._validate_ssl_certificates()

			self.logger.info("Pre-deployment validation completed")

		except Exception as e:
			self.logger.error(f"Pre-deployment validation failed: {e}")
			raise

	async def _setup_infrastructure(self, dry_run: bool = False) -> None:
		"""Setup infrastructure."""
		try:
			# Create namespace
			await self._create_namespace(dry_run)

			# Setup RBAC
			await self._setup_rbac(dry_run)

			# Create secrets
			await self._create_secrets(dry_run)

			# Create config maps
			await self._create_config_maps(dry_run)

			# Setup persistent volumes
			await self._setup_persistent_volumes(dry_run)

			# Setup network policies
			await self._setup_network_policies(dry_run)

			self.logger.info("Infrastructure setup completed")

		except Exception as e:
			self.logger.error(f"Infrastructure setup failed: {e}")
			raise

	async def _run_database_migrations(self, dry_run: bool = False) -> None:
		"""Run database migrations."""
		try:
			# Get pending migrations
			migrations = await self._get_pending_migrations()

			if not migrations:
				self.logger.info("No pending database migrations")
				return

			# Create migration backup
			backup_id = await self._create_database_backup()
			self.logger.info(f"Database backup created: {backup_id}")

			# Apply migrations
			for migration in migrations:
				await self._apply_migration(migration, dry_run)

			# Verify migration success
			await self._verify_migrations()

			self.logger.info(f"Applied {len(migrations)} database migrations")

		except Exception as e:
			self.logger.error(f"Database migration failed: {e}")
			# Attempt to restore backup
			try:
				await self._restore_database_backup(backup_id)
				self.logger.info("Database restored from backup")
			except Exception as restore_error:
				self.logger.error(f"Database restore failed: {restore_error}")
			raise

	async def _deploy_application(
		self,
		strategy: DeploymentStrategy,
		version: Optional[str] = None,
		dry_run: bool = False,
		force: bool = False
	) -> None:
		"""Deploy application."""
		try:
			if strategy == DeploymentStrategy.ROLLING_UPDATE:
				await self._rolling_update_deployment(version, dry_run, force)
			elif strategy == DeploymentStrategy.BLUE_GREEN:
				await self._blue_green_deployment(version, dry_run, force)
			elif strategy == DeploymentStrategy.CANARY:
				await self._canary_deployment(version, dry_run, force)
			elif strategy == DeploymentStrategy.RECREATE:
				await self._recreate_deployment(version, dry_run, force)
			else:
				raise ValueError(f"Unsupported deployment strategy: {strategy}")

			self.logger.info(f"Application deployment completed using {strategy.value} strategy")

		except Exception as e:
			self.logger.error(f"Application deployment failed: {e}")
			raise

	async def _post_deployment_validation(self, dry_run: bool = False) -> None:
		"""Post-deployment validation."""
		try:
			# Wait for deployment to be ready
			await self._wait_for_deployment_ready()

			# Validate application health
			await self._validate_application_health()

			# Validate API endpoints
			await self._validate_api_endpoints()

			# Validate database connectivity
			await self._validate_database_connectivity()

			# Run smoke tests
			await self._run_smoke_tests()

			self.logger.info("Post-deployment validation completed")

		except Exception as e:
			self.logger.error(f"Post-deployment validation failed: {e}")
			raise

	async def _setup_monitoring(self, dry_run: bool = False) -> None:
		"""Setup monitoring and alerting."""
		try:
			# Deploy Prometheus ServiceMonitor
			await self._deploy_service_monitor(dry_run)

			# Setup Grafana dashboards
			await self._setup_grafana_dashboards(dry_run)

			# Configure alerting rules
			await self._configure_alerting_rules(dry_run)

			# Setup log aggregation
			await self._setup_log_aggregation(dry_run)

			self.logger.info("Monitoring setup completed")

		except Exception as e:
			self.logger.error(f"Monitoring setup failed: {e}")
			raise

	async def _cleanup_deployment(self, dry_run: bool = False) -> None:
		"""Cleanup deployment resources."""
		try:
			# Remove old deployments
			await self._cleanup_old_deployments(dry_run)

			# Clean up unused secrets
			await self._cleanup_unused_secrets(dry_run)

			# Clean up unused config maps
			await self._cleanup_unused_config_maps(dry_run)

			# Clean up temporary resources
			await self._cleanup_temp_resources()

			self.logger.info("Deployment cleanup completed")

		except Exception as e:
			self.logger.error(f"Deployment cleanup failed: {e}")
			# Don't fail the deployment for cleanup errors

	async def _rolling_update_deployment(self, version: Optional[str], dry_run: bool, force: bool) -> None:
		"""Execute rolling update deployment."""
		try:
			# Generate Kubernetes manifests
			manifests = self.config_manager.generate_kubernetes_manifests(self.config)

			# Update image version in deployment manifest
			if version:
				deployment_manifest = yaml.safe_load(manifests['deployment.yaml'])
				deployment_manifest['spec']['template']['spec']['containers'][0]['image'] = f"datacraft/aicr:{version}"
				manifests['deployment.yaml'] = yaml.dump(deployment_manifest)

			# Apply manifests
			for manifest_name, manifest_content in manifests.items():
				await self._apply_manifest(manifest_content, dry_run)

			# Wait for rollout to complete
			await self._wait_for_rollout_complete()

			self.logger.info("Rolling update deployment completed")

		except Exception as e:
			self.logger.error(f"Rolling update deployment failed: {e}")
			raise

	async def _blue_green_deployment(self, version: Optional[str], dry_run: bool, force: bool) -> None:
		"""Execute blue-green deployment."""
		try:
			# Create green environment
			green_namespace = f"{self.config.namespace}-green"

			# Deploy to green environment
			green_config = self.config.model_copy()
			green_config.namespace = green_namespace

			green_manifests = self.config_manager.generate_kubernetes_manifests(green_config)

			# Update image version
			if version:
				deployment_manifest = yaml.safe_load(green_manifests['deployment.yaml'])
				deployment_manifest['spec']['template']['spec']['containers'][0]['image'] = f"datacraft/aicr:{version}"
				green_manifests['deployment.yaml'] = yaml.dump(deployment_manifest)

			# Apply green manifests
			for manifest_content in green_manifests.values():
				await self._apply_manifest(manifest_content, dry_run)

			# Wait for green deployment to be ready
			await self._wait_for_deployment_ready(green_namespace)

			# Validate green deployment
			await self._validate_deployment_health(green_namespace)

			# Switch traffic to green
			await self._switch_traffic_to_green(green_namespace, dry_run)

			# Clean up blue environment
			await self._cleanup_blue_environment(dry_run)

			self.logger.info("Blue-green deployment completed")

		except Exception as e:
			self.logger.error(f"Blue-green deployment failed: {e}")
			# Clean up green environment on failure
			await self._cleanup_green_environment(green_namespace, dry_run)
			raise

	async def _canary_deployment(self, version: Optional[str], dry_run: bool, force: bool) -> None:
		"""Execute canary deployment."""
		try:
			# Deploy canary version
			canary_deployment = f"aicr-canary"

			# Create canary deployment manifest
			manifests = self.config_manager.generate_kubernetes_manifests(self.config)
			deployment_manifest = yaml.safe_load(manifests['deployment.yaml'])

			# Update for canary
			deployment_manifest['metadata']['name'] = canary_deployment
			deployment_manifest['spec']['replicas'] = 1  # Start with 1 replica
			deployment_manifest['spec']['selector']['matchLabels']['version'] = 'canary'
			deployment_manifest['spec']['template']['metadata']['labels']['version'] = 'canary'

			if version:
				deployment_manifest['spec']['template']['spec']['containers'][0]['image'] = f"datacraft/aicr:{version}"

			# Deploy canary
			await self._apply_manifest(yaml.dump(deployment_manifest), dry_run)

			# Wait for canary to be ready
			await self._wait_for_deployment_ready(deployment_name=canary_deployment)

			# Gradually increase canary traffic
			traffic_percentages = [10, 25, 50, 75, 100]

			for percentage in traffic_percentages:
				await self._adjust_canary_traffic(percentage, dry_run)
				await asyncio.sleep(300)  # Wait 5 minutes

				# Monitor canary metrics
				if not await self._validate_canary_metrics():
					raise Exception("Canary metrics validation failed")

			# Replace stable with canary
			await self._promote_canary_to_stable(dry_run)

			self.logger.info("Canary deployment completed")

		except Exception as e:
			self.logger.error(f"Canary deployment failed: {e}")
			# Rollback canary
			await self._rollback_canary(dry_run)
			raise

	async def _recreate_deployment(self, version: Optional[str], dry_run: bool, force: bool) -> None:
		"""Execute recreate deployment."""
		try:
			# Scale down current deployment
			await self._scale_deployment(0, dry_run)

			# Wait for pods to terminate
			await self._wait_for_pods_termination()

			# Update deployment manifest
			manifests = self.config_manager.generate_kubernetes_manifests(self.config)

			if version:
				deployment_manifest = yaml.safe_load(manifests['deployment.yaml'])
				deployment_manifest['spec']['template']['spec']['containers'][0]['image'] = f"datacraft/aicr:{version}"
				manifests['deployment.yaml'] = yaml.dump(deployment_manifest)

			# Apply updated manifest
			await self._apply_manifest(manifests['deployment.yaml'], dry_run)

			# Scale up to desired replicas
			await self._scale_deployment(self.config.auto_scaling.min_replicas, dry_run)

			# Wait for deployment to be ready
			await self._wait_for_deployment_ready()

			self.logger.info("Recreate deployment completed")

		except Exception as e:
			self.logger.error(f"Recreate deployment failed: {e}")
			raise

	async def _validate_k8s_cluster(self) -> None:
		"""Validate Kubernetes cluster."""
		try:
			v1 = client.CoreV1Api(self.k8s_client)

			# Check cluster version
			version_info = await asyncio.get_event_loop().run_in_executor(
				None, v1.get_code
			)

			# Check node status
			nodes = await asyncio.get_event_loop().run_in_executor(
				None, v1.list_node
			)

			ready_nodes = 0
			for node in nodes.items:
				for condition in node.status.conditions:
					if condition.type == "Ready" and condition.status == "True":
						ready_nodes += 1
						break

			if ready_nodes == 0:
				raise Exception("No ready nodes found in cluster")

			self.logger.info(f"Cluster validation passed: {ready_nodes} ready nodes")

		except Exception as e:
			self.logger.error(f"Cluster validation failed: {e}")
			raise

	async def _validate_configuration(self) -> None:
		"""Validate deployment configuration."""
		try:
			# Validate required fields
			if not self.config.database.host:
				raise ValueError("Database host is required")

			if not self.config.security.jwt_secret_key:
				raise ValueError("JWT secret key is required")

			# Validate resource constraints
			if self.config.auto_scaling.min_replicas > self.config.auto_scaling.max_replicas:
				raise ValueError("min_replicas cannot be greater than max_replicas")

			self.logger.info("Configuration validation passed")

		except Exception as e:
			self.logger.error(f"Configuration validation failed: {e}")
			raise

	async def _create_namespace(self, dry_run: bool) -> None:
		"""Create Kubernetes namespace."""
		try:
			v1 = client.CoreV1Api(self.k8s_client)

			namespace = client.V1Namespace(
				metadata=client.V1ObjectMeta(
					name=self.config.namespace,
					labels={
						"app": "aicr",
						"environment": self.config.environment.value
					}
				)
			)

			if not dry_run:
				try:
					await asyncio.get_event_loop().run_in_executor(
						None, v1.create_namespace, namespace
					)
					self.logger.info(f"Namespace {self.config.namespace} created")
				except ApiException as e:
					if e.status == 409:  # Already exists
						self.logger.info(f"Namespace {self.config.namespace} already exists")
					else:
						raise
			else:
				self.logger.info(f"[DRY RUN] Would create namespace: {self.config.namespace}")

		except Exception as e:
			self.logger.error(f"Failed to create namespace: {e}")
			raise

	async def _apply_manifest(self, manifest_content: str, dry_run: bool) -> None:
		"""Apply Kubernetes manifest."""
		try:
			# Write manifest to temporary file
			with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
				f.write(manifest_content)
				manifest_file = f.name

			try:
				# Apply manifest using kubectl
				cmd = ["kubectl", "apply", "-f", manifest_file]
				if dry_run:
					cmd.append("--dry-run=client")

				result = await asyncio.create_subprocess_exec(
					*cmd,
					stdout=asyncio.subprocess.PIPE,
					stderr=asyncio.subprocess.PIPE
				)

				stdout, stderr = await result.communicate()

				if result.returncode != 0:
					raise Exception(f"kubectl apply failed: {stderr.decode()}")

				if dry_run:
					self.logger.info(f"[DRY RUN] Would apply manifest: {stdout.decode().strip()}")
				else:
					self.logger.info(f"Applied manifest: {stdout.decode().strip()}")

			finally:
				# Clean up temporary file
				os.unlink(manifest_file)

		except Exception as e:
			self.logger.error(f"Failed to apply manifest: {e}")
			raise

	async def _wait_for_deployment_ready(
		self,
		namespace: Optional[str] = None,
		deployment_name: str = "aicr",
		timeout: int = 600
	) -> None:
		"""Wait for deployment to be ready."""
		try:
			namespace = namespace or self.config.namespace
			apps_v1 = client.AppsV1Api(self.k8s_client)

			start_time = time.time()

			while time.time() - start_time < timeout:
				try:
					deployment = await asyncio.get_event_loop().run_in_executor(
						None, apps_v1.read_namespaced_deployment, deployment_name, namespace
					)

					if (deployment.status.ready_replicas and
						deployment.status.ready_replicas == deployment.status.replicas):
						self.logger.info(f"Deployment {deployment_name} is ready")
						return

				except ApiException as e:
					if e.status != 404:
						raise

				await asyncio.sleep(10)

			raise Exception(f"Timeout waiting for deployment {deployment_name} to be ready")

		except Exception as e:
			self.logger.error(f"Failed to wait for deployment ready: {e}")
			raise

	async def _validate_application_health(self, validation_result: Optional[ValidationResult] = None) -> None:
		"""Validate application health."""
		try:
			health_url = f"http://aicr-service.{self.config.namespace}.svc.cluster.local{self.config.health_check.path}"

			async with aiohttp.ClientSession() as session:
				async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
					if response.status == 200:
						health_data = await response.json()

						if validation_result:
							validation_result.checks['application_health'] = {
								'status': 'passed',
								'response_time_ms': health_data.get('response_time_ms', 0),
								'timestamp': datetime.utcnow().isoformat()
							}

						self.logger.info("Application health check passed")
					else:
						raise Exception(f"Health check failed with status: {response.status}")

		except Exception as e:
			if validation_result:
				validation_result.checks['application_health'] = {
					'status': 'failed',
					'error': str(e),
					'timestamp': datetime.utcnow().isoformat()
				}
				validation_result.error_count += 1

			self.logger.error(f"Application health validation failed: {e}")
			raise

	async def _cleanup_temp_resources(self) -> None:
		"""Clean up temporary resources."""
		try:
			# Clean up temporary files
			temp_dir = Path(tempfile.gettempdir()) / "aicr_deployment"
			if temp_dir.exists():
				shutil.rmtree(temp_dir)

			self.logger.info("Temporary resources cleaned up")

		except Exception as e:
			self.logger.warning(f"Failed to clean up temporary resources: {e}")


class DeploymentManager:
	"""Deployment management interface."""

	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.DeploymentManager")
		self._active_deployments: Dict[str, ProductionDeployer] = {}

	async def create_deployment(
		self,
		environment: Environment,
		config_path: Optional[str] = None
	) -> str:
		"""Create new deployment."""
		try:
			# Load configuration
			if config_path:
				config_manager = ConfigurationManager()
				config = config_manager.load_config(config_path)
			else:
				# Use default configuration for environment
				config = self._get_default_config(environment)

			# Create deployer
			deployer = ProductionDeployer(config)
			await deployer.initialize()

			deployment_id = uuid7str()
			self._active_deployments[deployment_id] = deployer

			self.logger.info(f"Deployment created: {deployment_id}")
			return deployment_id

		except Exception as e:
			self.logger.error(f"Failed to create deployment: {e}")
			raise

	async def execute_deployment(
		self,
		deployment_id: str,
		strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
		version: Optional[str] = None,
		dry_run: bool = False
	) -> DeploymentResult:
		"""Execute deployment."""
		try:
			if deployment_id not in self._active_deployments:
				raise ValueError(f"Deployment not found: {deployment_id}")

			deployer = self._active_deployments[deployment_id]
			result = await deployer.deploy(strategy, version, dry_run)

			return result

		except Exception as e:
			self.logger.error(f"Failed to execute deployment: {e}")
			raise

	async def rollback_deployment(
		self,
		deployment_id: str,
		target_version: Optional[str] = None
	) -> DeploymentResult:
		"""Rollback deployment."""
		try:
			if deployment_id not in self._active_deployments:
				raise ValueError(f"Deployment not found: {deployment_id}")

			deployer = self._active_deployments[deployment_id]
			result = await deployer.rollback(target_version)

			return result

		except Exception as e:
			self.logger.error(f"Failed to rollback deployment: {e}")
			raise

	async def validate_deployment(self, deployment_id: str) -> ValidationResult:
		"""Validate deployment."""
		try:
			if deployment_id not in self._active_deployments:
				raise ValueError(f"Deployment not found: {deployment_id}")

			deployer = self._active_deployments[deployment_id]
			result = await deployer.validate_deployment()

			return result

		except Exception as e:
			self.logger.error(f"Failed to validate deployment: {e}")
			raise

	def _get_default_config(self, environment: Environment) -> ProductionConfig:
		"""Get default configuration for environment."""
		# This would typically load from a configuration repository
		# For now, return a basic configuration
		from .production_config import create_production_configuration

		return create_production_configuration()


# CLI interface for deployment
async def main():
	"""Main deployment CLI."""
	import argparse

	parser = argparse.ArgumentParser(description="AICR Production Deployment")
	parser.add_argument("--environment", choices=["development", "staging", "production"], required=True)
	parser.add_argument("--strategy", choices=["rolling_update", "blue_green", "canary", "recreate"], default="rolling_update")
	parser.add_argument("--version", help="Application version to deploy")
	parser.add_argument("--config", help="Configuration file path")
	parser.add_argument("--dry-run", action="store_true", help="Perform dry run")
	parser.add_argument("--validate-only", action="store_true", help="Only validate deployment")
	parser.add_argument("--rollback", help="Rollback to specified version")

	args = parser.parse_args()

	# Configure logging
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	)

	try:
		# Create deployment manager
		manager = DeploymentManager()

		# Create deployment
		deployment_id = await manager.create_deployment(
			Environment(args.environment),
			args.config
		)

		if args.rollback:
			# Execute rollback
			result = await manager.rollback_deployment(deployment_id, args.rollback)
			print(f"Rollback result: {result.status}")

		elif args.validate_only:
			# Validate only
			result = await manager.validate_deployment(deployment_id)
			print(f"Validation result: {result.status}")
			if not result.overall_success:
				print(f"Errors: {result.error_count}, Warnings: {result.warning_count}")

		else:
			# Execute deployment
			result = await manager.execute_deployment(
				deployment_id,
				DeploymentStrategy(args.strategy),
				args.version,
				args.dry_run
			)

			print(f"Deployment result: {result.status}")
			if result.success:
				print(f"Deployed version: {result.deployed_version}")
				print(f"Duration: {result.duration_seconds} seconds")
			else:
				print(f"Error: {result.error_message}")

	except Exception as e:
		print(f"Deployment failed: {e}")
		exit(1)


if __name__ == "__main__":
	asyncio.run(main())