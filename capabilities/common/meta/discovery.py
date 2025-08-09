#!/usr/bin/env python3
"""
APG Metadata Management - Discovery Service
Orchestrates metadata discovery from various data sources with AI-powered intelligence

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str

from .database import MetaDatabaseManager
from .integrations import APGMetadataIntegrationManager, MetadataEvent, MetadataEventType
from .connectors import (
	BaseConnector, ConnectorConfig, DiscoveryResult,
	PostgreSQLConnector, MySQLConnector, MongoDBConnector,
	ConnectorRegistry
)
from .models import (
	MetaAsset, MetaAssetVersion, MetaClassification,
	AssetType, SourceSystemType, AssetStatus, ClassificationType
)


class DiscoveryScheduleType(str, Enum):
	"""Discovery schedule types"""
	ONE_TIME = "one_time"
	RECURRING = "recurring"
	REAL_TIME = "real_time"
	ON_DEMAND = "on_demand"


class DiscoveryStatus(str, Enum):
	"""Discovery job status"""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	PAUSED = "paused"


@dataclass
class DiscoverySchedule:
	"""Discovery schedule configuration"""
	schedule_id: str = field(default_factory=uuid7str)
	name: str = ""
	description: Optional[str] = None
	schedule_type: DiscoveryScheduleType = DiscoveryScheduleType.ONE_TIME
	connector_configs: List[ConnectorConfig] = field(default_factory=list)
	
	# Scheduling parameters
	cron_expression: Optional[str] = None
	interval_minutes: Optional[int] = None
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	
	# Discovery parameters
	tenant_id: str = ""
	created_by: str = ""
	enable_ai_classification: bool = True
	enable_quality_assessment: bool = True
	enable_lineage_detection: bool = True
	max_parallel_connectors: int = 5
	
	# Filters and limits
	discovery_filters: Dict[str, Any] = field(default_factory=dict)
	max_assets_per_run: int = 10000
	timeout_minutes: int = 60
	
	# APG integration settings
	publish_events: bool = True
	audit_operations: bool = True
	
	# Metadata
	tags: List[str] = field(default_factory=list)
	custom_attributes: Dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	last_run: Optional[datetime] = None
	next_run: Optional[datetime] = None
	is_active: bool = True


@dataclass
class DiscoveryJob:
	"""Discovery job execution context"""
	job_id: str = field(default_factory=uuid7str)
	schedule_id: str = ""
	tenant_id: str = ""
	status: DiscoveryStatus = DiscoveryStatus.PENDING
	
	# Execution details
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	duration_seconds: Optional[float] = None
	
	# Results
	total_connectors: int = 0
	successful_connectors: int = 0
	failed_connectors: int = 0
	total_assets_discovered: int = 0
	new_assets: int = 0
	updated_assets: int = 0
	
	# Performance metrics
	assets_per_second: Optional[float] = None
	connector_results: List[DiscoveryResult] = field(default_factory=list)
	
	# Errors and warnings
	errors: List[str] = field(default_factory=list)
	warnings: List[str] = field(default_factory=list)
	
	# Progress tracking
	progress_percentage: float = 0.0
	current_connector: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for serialization"""
		return {
			"job_id": self.job_id,
			"schedule_id": self.schedule_id,
			"tenant_id": self.tenant_id,
			"status": self.status.value,
			"started_at": self.started_at.isoformat() if self.started_at else None,
			"completed_at": self.completed_at.isoformat() if self.completed_at else None,
			"duration_seconds": self.duration_seconds,
			"total_connectors": self.total_connectors,
			"successful_connectors": self.successful_connectors,
			"failed_connectors": self.failed_connectors,
			"total_assets_discovered": self.total_assets_discovered,
			"new_assets": self.new_assets,
			"updated_assets": self.updated_assets,
			"assets_per_second": self.assets_per_second,
			"errors": self.errors,
			"warnings": self.warnings,
			"progress_percentage": self.progress_percentage,
			"current_connector": self.current_connector
		}


class MetadataDiscoveryService:
	"""Advanced metadata discovery service with AI integration and real-time capabilities"""
	
	def __init__(self, 
		     db_manager: MetaDatabaseManager,
		     integration_manager: APGMetadataIntegrationManager,
		     config: Dict[str, Any] = None):
		self.db_manager = db_manager
		self.integration_manager = integration_manager
		self.config = config or {}
		
		# Connector registry for dynamic connector loading
		self.connector_registry = ConnectorRegistry()
		self._register_built_in_connectors()
		
		# Active jobs tracking
		self.active_jobs: Dict[str, DiscoveryJob] = {}
		self.job_tasks: Dict[str, asyncio.Task] = {}
		
		# Discovery schedules
		self.schedules: Dict[str, DiscoverySchedule] = {}
		self.scheduler_task: Optional[asyncio.Task] = None
		self.scheduler_running = False
		
		# Performance settings
		self.max_concurrent_jobs = config.get('max_concurrent_jobs', 3)
		self.default_timeout = config.get('default_timeout_minutes', 60)
		self.asset_batch_size = config.get('asset_batch_size', 100)
		
		# AI classification settings
		self.enable_ai_classification = config.get('enable_ai_classification', True)
		self.classification_confidence_threshold = config.get('classification_confidence_threshold', 0.8)
		
		# Change detection
		self.enable_change_detection = config.get('enable_change_detection', True)
		self.schema_change_cache: Dict[str, str] = {}  # asset_id -> schema_hash
		
		self.initialized = False
	
	def _register_built_in_connectors(self):
		"""Register built-in connector types"""
		self.connector_registry.register('postgresql', PostgreSQLConnector)
		self.connector_registry.register('mysql', MySQLConnector)
		self.connector_registry.register('mongodb', MongoDBConnector)
		# Additional connectors will be registered as they're implemented
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize the discovery service"""
		if self.initialized:
			return {"status": "already_initialized"}
		
		try:
			# Start the discovery scheduler
			await self._start_scheduler()
			
			self.initialized = True
			
			await self._log_info("Metadata discovery service initialized successfully")
			
			return {
				"status": "initialized",
				"max_concurrent_jobs": self.max_concurrent_jobs,
				"registered_connectors": list(self.connector_registry.list_connectors()),
				"scheduler_running": self.scheduler_running
			}
			
		except Exception as e:
			await self._log_error(f"Discovery service initialization failed: {str(e)}")
			raise
	
	async def shutdown(self):
		"""Shutdown the discovery service"""
		if not self.initialized:
			return
		
		try:
			# Stop scheduler
			if self.scheduler_task and not self.scheduler_task.done():
				self.scheduler_task.cancel()
				try:
					await self.scheduler_task
				except asyncio.CancelledError:
					pass
			
			# Cancel all active jobs
			for job_id, task in self.job_tasks.items():
				if not task.done():
					task.cancel()
					await self._log_info(f"Cancelled discovery job: {job_id}")
			
			# Wait for all tasks to complete
			if self.job_tasks:
				await asyncio.gather(*self.job_tasks.values(), return_exceptions=True)
			
			self.initialized = False
			await self._log_info("Metadata discovery service shutdown completed")
			
		except Exception as e:
			await self._log_error(f"Discovery service shutdown failed: {str(e)}")
	
	async def create_discovery_schedule(self, schedule: DiscoverySchedule) -> str:
		"""Create a new discovery schedule"""
		try:
			# Validate schedule
			if not schedule.connector_configs:
				raise ValueError("At least one connector configuration is required")
			
			if not schedule.tenant_id:
				raise ValueError("Tenant ID is required")
			
			# Calculate next run time
			if schedule.schedule_type == DiscoveryScheduleType.RECURRING:
				schedule.next_run = self._calculate_next_run(schedule)
			
			# Store schedule
			self.schedules[schedule.schedule_id] = schedule
			
			# Persist to database
			await self._persist_schedule(schedule)
			
			await self._log_info(f"Created discovery schedule: {schedule.schedule_id}")
			
			# Publish event
			if schedule.publish_events:
				await self.integration_manager.publish_asset_event(
					event_type=MetadataEventType.ASSET_DISCOVERED,
					asset_id=schedule.schedule_id,
					tenant_id=schedule.tenant_id,
					user_id=schedule.created_by,
					payload={
						"action": "schedule_created",
						"schedule_name": schedule.name,
						"schedule_type": schedule.schedule_type.value
					}
				)
			
			return schedule.schedule_id
			
		except Exception as e:
			await self._log_error(f"Failed to create discovery schedule: {str(e)}")
			raise
	
	async def run_discovery_job(self, 
				    schedule_id: str,
				    override_config: Dict[str, Any] = None) -> str:
		"""Run a discovery job immediately"""
		try:
			schedule = self.schedules.get(schedule_id)
			if not schedule:
				raise ValueError(f"Schedule not found: {schedule_id}")
			
			# Check concurrent job limit
			if len(self.active_jobs) >= self.max_concurrent_jobs:
				raise RuntimeError("Maximum concurrent jobs limit reached")
			
			# Create discovery job
			job = DiscoveryJob(
				schedule_id=schedule_id,
				tenant_id=schedule.tenant_id,
				total_connectors=len(schedule.connector_configs)
			)
			
			# Apply override configuration
			if override_config:
				# Apply any job-specific overrides here
				pass
			
			# Start job execution
			job.status = DiscoveryStatus.RUNNING
			job.started_at = datetime.utcnow()
			self.active_jobs[job.job_id] = job
			
			# Create and start job task
			task = asyncio.create_task(self._execute_discovery_job(job, schedule))
			self.job_tasks[job.job_id] = task
			
			await self._log_info(f"Started discovery job: {job.job_id} for schedule: {schedule_id}")
			
			return job.job_id
			
		except Exception as e:
			await self._log_error(f"Failed to start discovery job: {str(e)}")
			raise
	
	async def _execute_discovery_job(self, job: DiscoveryJob, schedule: DiscoverySchedule):
		"""Execute a discovery job with all connectors"""
		try:
			# Log operation for audit
			if schedule.audit_operations:
				await self.integration_manager.log_operation(
					operation="discovery_job_started",
					asset_id=job.job_id,
					user_id=schedule.created_by,
					tenant_id=schedule.tenant_id,
					details={
						"schedule_id": schedule.schedule_id,
						"connector_count": len(schedule.connector_configs)
					}
				)
			
			# Execute connectors in parallel (with limits)
			semaphore = asyncio.Semaphore(schedule.max_parallel_connectors)
			connector_tasks = []
			
			for i, connector_config in enumerate(schedule.connector_configs):
				task = self._execute_connector_discovery(
					job, schedule, connector_config, semaphore, i
				)
				connector_tasks.append(task)
			
			# Wait for all connectors to complete
			connector_results = await asyncio.gather(*connector_tasks, return_exceptions=True)
			
			# Process results
			for result in connector_results:
				if isinstance(result, Exception):
					job.errors.append(f"Connector execution failed: {str(result)}")
					job.failed_connectors += 1
				elif isinstance(result, DiscoveryResult):
					job.connector_results.append(result)
					job.successful_connectors += 1
					job.total_assets_discovered += result.total_assets
					
					# Process discovered assets
					await self._process_discovery_result(result, schedule, job)
			
			# Complete job
			job.completed_at = datetime.utcnow()
			job.duration_seconds = (job.completed_at - job.started_at).total_seconds()
			job.status = DiscoveryStatus.COMPLETED
			job.progress_percentage = 100.0
			
			if job.duration_seconds > 0:
				job.assets_per_second = job.total_assets_discovered / job.duration_seconds
			
			# Update schedule
			schedule.last_run = job.completed_at
			if schedule.schedule_type == DiscoveryScheduleType.RECURRING:
				schedule.next_run = self._calculate_next_run(schedule)
			
			await self._log_info(
				f"Discovery job completed: {job.job_id} - "
				f"Discovered {job.total_assets_discovered} assets in {job.duration_seconds:.2f}s"
			)
			
			# Publish completion event
			if schedule.publish_events:
				await self.integration_manager.publish_asset_event(
					event_type=MetadataEventType.ASSET_DISCOVERED,
					asset_id=job.job_id,
					tenant_id=schedule.tenant_id,
					user_id=schedule.created_by,
					payload={
						"action": "discovery_completed",
						"assets_discovered": job.total_assets_discovered,
						"duration_seconds": job.duration_seconds
					}
				)
			
		except asyncio.CancelledError:
			job.status = DiscoveryStatus.CANCELLED
			await self._log_info(f"Discovery job cancelled: {job.job_id}")
			
		except Exception as e:
			job.status = DiscoveryStatus.FAILED
			job.errors.append(f"Job execution failed: {str(e)}")
			job.completed_at = datetime.utcnow()
			
			await self._log_error(f"Discovery job failed: {job.job_id} - {str(e)}")
			
		finally:
			# Cleanup
			if job.job_id in self.active_jobs:
				del self.active_jobs[job.job_id]
			if job.job_id in self.job_tasks:
				del self.job_tasks[job.job_id]
			
			# Log completion
			if schedule.audit_operations:
				await self.integration_manager.log_operation(
					operation="discovery_job_completed",
					asset_id=job.job_id,
					user_id=schedule.created_by,
					tenant_id=schedule.tenant_id,
					details={
						"status": job.status.value,
						"assets_discovered": job.total_assets_discovered,
						"duration_seconds": job.duration_seconds
					}
				)
	
	async def _execute_connector_discovery(self,
					       job: DiscoveryJob,
					       schedule: DiscoverySchedule,
					       connector_config: ConnectorConfig,
					       semaphore: asyncio.Semaphore,
					       connector_index: int) -> DiscoveryResult:
		"""Execute discovery for a single connector"""
		async with semaphore:
			try:
				# Update progress
				job.current_connector = f"{connector_config.connection_string[:20]}..."
				progress = (connector_index / len(schedule.connector_configs)) * 100
				job.progress_percentage = min(progress, 95.0)  # Leave 5% for post-processing
				
				# Get connector instance from registry
				connector_type = self._detect_connector_type(connector_config)
				connector_class = self.connector_registry.get_connector(connector_type)
				
				if not connector_class:
					raise ValueError(f"No connector available for type: {connector_type}")
				
				# Create and configure connector
				connector: BaseConnector = connector_class(connector_config)
				
				# Test connection first
				connection_test = await connector.test_connection()
				if connection_test.get('status') != 'success':
					raise RuntimeError(f"Connection test failed: {connection_test.get('message')}")
				
				# Run discovery
				await self._log_info(f"Starting discovery with {connector_type} connector")
				result = await connector.discover_assets()
				
				await self._log_info(
					f"Connector discovery completed: {result.total_assets} assets, "
					f"{result.successful_assets} successful, {len(result.errors)} errors"
				)
				
				return result
				
			except Exception as e:
				# Create error result
				error_result = DiscoveryResult(
					connector_type=getattr(connector_config, 'connector_type', 'unknown'),
					source_system='unknown'
				)
				error_result.add_error(f"Connector execution failed: {str(e)}")
				error_result.complete_discovery()
				return error_result
	
	async def _process_discovery_result(self,
					    result: DiscoveryResult,
					    schedule: DiscoverySchedule,
					    job: DiscoveryJob):
		"""Process discovery result and create/update metadata assets"""
		try:
			async with self.db_manager.get_session(schedule.tenant_id) as session:
				for asset_metadata in result.assets:
					try:
						# Check if asset already exists
						existing_asset = await self._find_existing_asset(
							session, asset_metadata, schedule.tenant_id
						)
						
						if existing_asset:
							# Check for schema changes
							if self.enable_change_detection:
								current_hash = asset_metadata.get_schema_hash()
								previous_hash = self.schema_change_cache.get(existing_asset.id)
								
								if previous_hash and current_hash != previous_hash:
									await self._handle_schema_change(
										existing_asset, asset_metadata, session, schedule
									)
								
								self.schema_change_cache[existing_asset.id] = current_hash
							
							# Update existing asset
							await self._update_existing_asset(existing_asset, asset_metadata, session)
							job.updated_assets += 1
							
						else:
							# Create new asset
							new_asset = await self._create_new_asset(
								asset_metadata, schedule, session
							)
							job.new_assets += 1
							
							# AI-powered classification
							if schedule.enable_ai_classification and self.enable_ai_classification:
								await self._classify_asset_with_ai(new_asset, asset_metadata, session)
							
							# Quality assessment
							if schedule.enable_quality_assessment:
								await self._assess_asset_quality(new_asset, asset_metadata, session)
					
					except Exception as e:
						job.errors.append(f"Asset processing failed for {asset_metadata.name}: {str(e)}")
			
		except Exception as e:
			job.errors.append(f"Result processing failed: {str(e)}")
			await self._log_error(f"Failed to process discovery result: {str(e)}")
	
	async def _create_new_asset(self,
				    asset_metadata,
				    schedule: DiscoverySchedule,
				    session) -> MetaAsset:
		"""Create new metadata asset from discovered metadata"""
		from .models import MetaAsset
		
		# Map asset type
		asset_type = self._map_asset_type(asset_metadata.asset_type)
		
		# Create asset
		asset = MetaAsset(
			tenant_id=schedule.tenant_id,
			name=asset_metadata.name,
			display_name=asset_metadata.name.replace('_', ' ').title(),
			description=asset_metadata.description,
			asset_type=asset_type,
			source_system=asset_metadata.source_system,
			source_system_type=SourceSystemType.DATABASE,
			external_id=asset_metadata.full_name or asset_metadata.name,
			status=AssetStatus.ACTIVE,
			schema_info={
				"columns": [col.to_dict() for col in asset_metadata.columns],
				"column_count": asset_metadata.column_count,
				"row_count": asset_metadata.row_count,
				"size_bytes": asset_metadata.size_bytes
			},
			quality_score=asset_metadata.estimated_quality_score,
			tags=asset_metadata.tags,
			custom_attributes={
				"discovered_at": datetime.utcnow().isoformat(),
				"discovery_schedule": schedule.schedule_id,
				**asset_metadata.properties
			},
			created_by=schedule.created_by,
			updated_by=schedule.created_by
		)
		
		session.add(asset)
		await session.flush()  # Get the ID
		
		return asset
	
	async def _classify_asset_with_ai(self, asset: MetaAsset, asset_metadata, session):
		"""Use AI to classify the asset for sensitivity and privacy"""
		try:
			# Prepare content for classification
			content = f"{asset.name} {asset.description or ''}"
			
			# Add column information
			if asset_metadata.columns:
				column_info = " ".join([
					f"{col.name}:{col.data_type.value}" 
					for col in asset_metadata.columns[:10]  # Limit to first 10 columns
				])
				content += f" Columns: {column_info}"
			
			# Classify using AI integration
			classification_result = await self.integration_manager.ai_integration.classify_data_content(
				content=content,
				column_name=asset.name
			)
			
			# Create classification record if confidence is high enough
			if classification_result.get('confidence', 0) >= self.classification_confidence_threshold:
				from .models import MetaClassification
				
				classification_type = self._map_ai_classification(
					classification_result.get('classification', 'INTERNAL')
				)
				
				classification = MetaClassification(
					tenant_id=asset.tenant_id,
					asset_id=asset.id,
					classification_type=classification_type,
					classification_level=classification_result.get('classification', 'INTERNAL'),
					confidence_score=classification_result.get('confidence', 0.0),
					reasoning=classification_result.get('reasoning', ''),
					classified_by="system:ai_classifier",
					classification_method="ai_automated",
					status="approved" if classification_result.get('confidence', 0) > 0.9 else "pending_review"
				)
				
				session.add(classification)
				
		except Exception as e:
			await self._log_error(f"AI classification failed for asset {asset.id}: {str(e)}")
	
	async def _assess_asset_quality(self, asset: MetaAsset, asset_metadata, session):
		"""Assess and record asset quality metrics"""
		try:
			from .models import MetaQualityAssessment
			
			# Calculate quality metrics
			quality_metrics = {
				"completeness": self._calculate_completeness(asset_metadata),
				"consistency": self._calculate_consistency(asset_metadata),
				"accuracy": asset_metadata.estimated_quality_score or 0.0,
				"timeliness": self._calculate_timeliness(asset_metadata),
				"validity": self._calculate_validity(asset_metadata)
			}
			
			# Overall quality score (weighted average)
			overall_score = (
				quality_metrics["completeness"] * 0.25 +
				quality_metrics["consistency"] * 0.20 +
				quality_metrics["accuracy"] * 0.25 +
				quality_metrics["timeliness"] * 0.15 +
				quality_metrics["validity"] * 0.15
			)
			
			# Create quality assessment
			assessment = MetaQualityAssessment(
				tenant_id=asset.tenant_id,
				asset_id=asset.id,
				assessment_type="discovery_automated",
				overall_score=round(overall_score, 2),
				quality_metrics=quality_metrics,
				issues_found=[],  # Will be populated based on metrics
				recommendations=[],  # Will be populated based on metrics
				assessed_by="system:discovery_service",
				assessment_method="automated_discovery"
			)
			
			# Add issues and recommendations based on scores
			if quality_metrics["completeness"] < 80:
				assessment.issues_found.append("Low data completeness detected")
				assessment.recommendations.append("Review missing data and implement data validation rules")
			
			if quality_metrics["consistency"] < 70:
				assessment.issues_found.append("Data consistency issues detected")
				assessment.recommendations.append("Implement data standardization processes")
			
			session.add(assessment)
			
			# Update asset quality score
			asset.quality_score = round(overall_score, 2)
			
		except Exception as e:
			await self._log_error(f"Quality assessment failed for asset {asset.id}: {str(e)}")
	
	def _calculate_completeness(self, asset_metadata) -> float:
		"""Calculate data completeness score"""
		if not asset_metadata.columns:
			return 50.0
		
		# Base score
		score = 100.0
		
		# Penalize high null percentages
		for column in asset_metadata.columns:
			if column.null_percentage:
				if column.null_percentage > 50:
					score -= 10
				elif column.null_percentage > 20:
					score -= 5
		
		# Bonus for having descriptions
		described_columns = sum(1 for col in asset_metadata.columns if col.description)
		description_ratio = described_columns / len(asset_metadata.columns)
		score += description_ratio * 10
		
		return max(0.0, min(100.0, score))
	
	def _calculate_consistency(self, asset_metadata) -> float:
		"""Calculate data consistency score"""
		# Base score
		score = 85.0
		
		# Check for consistent naming patterns
		column_names = [col.name.lower() for col in asset_metadata.columns]
		
		# Penalize inconsistent naming
		snake_case = sum(1 for name in column_names if '_' in name)
		camel_case = sum(1 for name in column_names if any(c.isupper() for c in name[1:]))
		
		if snake_case > 0 and camel_case > 0:
			score -= 15  # Mixed naming conventions
		
		# Bonus for primary keys
		has_primary_key = any(col.is_primary_key for col in asset_metadata.columns)
		if has_primary_key:
			score += 10
		
		return max(0.0, min(100.0, score))
	
	def _calculate_timeliness(self, asset_metadata) -> float:
		"""Calculate data timeliness score"""
		# For discovery, we can't determine actual timeliness
		# Return neutral score
		return 75.0
	
	def _calculate_validity(self, asset_metadata) -> float:
		"""Calculate data validity score"""
		if not asset_metadata.columns:
			return 70.0
		
		score = 90.0
		
		# Check for unknown data types
		unknown_types = sum(1 for col in asset_metadata.columns 
				   if col.data_type.value == 'unknown')
		
		if unknown_types > 0:
			penalty = (unknown_types / len(asset_metadata.columns)) * 30
			score -= penalty
		
		return max(0.0, min(100.0, score))
	
	def _detect_connector_type(self, config: ConnectorConfig) -> str:
		"""Detect connector type from configuration"""
		connection_string = config.connection_string.lower()
		
		if 'postgresql' in connection_string or 'postgres' in connection_string:
			return 'postgresql'
		elif 'mysql' in connection_string:
			return 'mysql'
		elif 'mongodb' in connection_string or 'mongo' in connection_string:
			return 'mongodb'
		else:
			# Try to infer from additional params or default
			return config.additional_params.get('connector_type', 'postgresql')
	
	def _map_asset_type(self, discovered_type: str) -> AssetType:
		"""Map discovered asset type to internal asset type"""
		type_mapping = {
			'table': AssetType.TABLE,
			'view': AssetType.VIEW,
			'collection': AssetType.TABLE,  # MongoDB collections as tables
			'file': AssetType.FILE,
			'api': AssetType.API,
			'model': AssetType.ML_MODEL,
			'dashboard': AssetType.DASHBOARD,
			'report': AssetType.REPORT
		}
		
		return type_mapping.get(discovered_type.lower(), AssetType.CUSTOM)
	
	def _map_ai_classification(self, ai_classification: str) -> ClassificationType:
		"""Map AI classification to internal classification type"""
		classification_mapping = {
			'PII': ClassificationType.PII,
			'PHI': ClassificationType.PHI,
			'FINANCIAL': ClassificationType.FINANCIAL,
			'CONFIDENTIAL': ClassificationType.CONFIDENTIAL,
			'INTERNAL': ClassificationType.INTERNAL,
			'PUBLIC': ClassificationType.PUBLIC
		}
		
		return classification_mapping.get(ai_classification.upper(), ClassificationType.INTERNAL)
	
	async def _find_existing_asset(self, session, asset_metadata, tenant_id) -> Optional[MetaAsset]:
		"""Find existing asset by external ID or name"""
		from sqlalchemy import select
		
		# Try by external_id first
		external_id = asset_metadata.full_name or asset_metadata.name
		stmt = select(MetaAsset).where(
			MetaAsset.tenant_id == tenant_id,
			MetaAsset.external_id == external_id,
			MetaAsset.is_deleted == False
		)
		
		result = await session.execute(stmt)
		existing = result.scalar_one_or_none()
		
		if not existing:
			# Try by name and source system
			stmt = select(MetaAsset).where(
				MetaAsset.tenant_id == tenant_id,
				MetaAsset.name == asset_metadata.name,
				MetaAsset.source_system == asset_metadata.source_system,
				MetaAsset.is_deleted == False
			)
			
			result = await session.execute(stmt)
			existing = result.scalar_one_or_none()
		
		return existing
	
	async def _update_existing_asset(self, existing_asset, asset_metadata, session):
		"""Update existing asset with new metadata"""
		# Update basic metadata
		if asset_metadata.description:
			existing_asset.description = asset_metadata.description
		
		# Update schema info
		existing_asset.schema_info = {
			"columns": [col.to_dict() for col in asset_metadata.columns],
			"column_count": asset_metadata.column_count,
			"row_count": asset_metadata.row_count,
			"size_bytes": asset_metadata.size_bytes
		}
		
		# Update quality score if available
		if asset_metadata.estimated_quality_score:
			existing_asset.quality_score = asset_metadata.estimated_quality_score
		
		# Update timestamps
		existing_asset.updated_at = datetime.utcnow()
		existing_asset.updated_by = "system:discovery_service"
	
	async def _handle_schema_change(self, existing_asset, new_metadata, session, schedule):
		"""Handle detected schema changes"""
		try:
			# Create asset version for schema change
			from .models import MetaAssetVersion
			
			version = MetaAssetVersion(
				tenant_id=existing_asset.tenant_id,
				asset_id=existing_asset.id,
				version_number=existing_asset.current_version + 1,
				schema_info=existing_asset.schema_info,  # Previous schema
				change_summary="Schema change detected during discovery",
				created_by="system:discovery_service"
			)
			
			session.add(version)
			
			# Update asset version
			existing_asset.current_version += 1
			
			# Publish schema change event
			if schedule.publish_events:
				await self.integration_manager.publish_asset_event(
					event_type=MetadataEventType.ASSET_UPDATED,
					asset_id=existing_asset.id,
					tenant_id=existing_asset.tenant_id,
					payload={
						"change_type": "schema_change",
						"previous_version": version.version_number - 1,
						"new_version": version.version_number
					}
				)
			
		except Exception as e:
			await self._log_error(f"Schema change handling failed: {str(e)}")
	
	async def get_discovery_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
		"""Get status of a discovery job"""
		job = self.active_jobs.get(job_id)
		if job:
			return job.to_dict()
		return None
	
	async def list_active_jobs(self) -> List[Dict[str, Any]]:
		"""List all active discovery jobs"""
		return [job.to_dict() for job in self.active_jobs.values()]
	
	async def cancel_discovery_job(self, job_id: str) -> bool:
		"""Cancel a running discovery job"""
		try:
			job = self.active_jobs.get(job_id)
			task = self.job_tasks.get(job_id)
			
			if job and task and not task.done():
				task.cancel()
				if job:
					job.status = DiscoveryStatus.CANCELLED
				await self._log_info(f"Cancelled discovery job: {job_id}")
				return True
			
			return False
			
		except Exception as e:
			await self._log_error(f"Failed to cancel job {job_id}: {str(e)}")
			return False
	
	async def _start_scheduler(self):
		"""Start the discovery scheduler"""
		if self.scheduler_running:
			return
		
		self.scheduler_running = True
		self.scheduler_task = asyncio.create_task(self._scheduler_loop())
		await self._log_info("Discovery scheduler started")
	
	async def _scheduler_loop(self):
		"""Main scheduler loop"""
		while self.scheduler_running:
			try:
				await asyncio.sleep(60)  # Check every minute
				
				current_time = datetime.utcnow()
				
				for schedule in self.schedules.values():
					if not schedule.is_active:
						continue
					
					if (schedule.schedule_type == DiscoveryScheduleType.RECURRING and
						schedule.next_run and 
						current_time >= schedule.next_run):
						
						# Check if we can run (not at concurrent limit)
						if len(self.active_jobs) < self.max_concurrent_jobs:
							try:
								await self.run_discovery_job(schedule.schedule_id)
							except Exception as e:
								await self._log_error(f"Scheduled job failed: {str(e)}")
			
			except asyncio.CancelledError:
				break
			except Exception as e:
				await self._log_error(f"Scheduler loop error: {str(e)}")
	
	def _calculate_next_run(self, schedule: DiscoverySchedule) -> Optional[datetime]:
		"""Calculate next run time for recurring schedule"""
		if not schedule.interval_minutes:
			return None
		
		base_time = schedule.last_run or datetime.utcnow()
		next_run = base_time + timedelta(minutes=schedule.interval_minutes)
		
		# Check end time constraint
		if schedule.end_time and next_run > schedule.end_time:
			return None
		
		return next_run
	
	async def _persist_schedule(self, schedule: DiscoverySchedule):
		"""Persist schedule to database"""
		try:
			schedule_data = {
				'schedule_id': schedule.schedule_id,
				'name': schedule.name,
				'description': schedule.description,
				'connector_config': json.dumps({
					'connector_type': schedule.connector_config.connector_type.value,
					'connection_string': schedule.connector_config.connection_string,
					'username': schedule.connector_config.username,
					'password': schedule.connector_config.password,  # In production, encrypt this
					'database': schedule.connector_config.database,
					'host': schedule.connector_config.host,
					'port': schedule.connector_config.port,
					'custom_attributes': schedule.connector_config.custom_attributes,
					'include_patterns': schedule.connector_config.include_patterns,
					'exclude_patterns': schedule.connector_config.exclude_patterns
				}),
				'schedule_type': schedule.schedule_type.value,
				'cron_expression': schedule.cron_expression,
				'interval_minutes': schedule.interval_minutes,
				'start_time': schedule.start_time.isoformat() if schedule.start_time else None,
				'end_time': schedule.end_time.isoformat() if schedule.end_time else None,
				'is_enabled': schedule.is_enabled,
				'is_one_time': schedule.is_one_time,
				'last_run': schedule.last_run.isoformat() if schedule.last_run else None,
				'next_run': schedule.next_run.isoformat() if schedule.next_run else None,
				'created_by': schedule.created_by,
				'created_at': schedule.created_at.isoformat() if schedule.created_at else datetime.utcnow().isoformat(),
				'tenant_id': self.tenant_id
			}
			
			await self.db_manager.execute_query(
				"""
				INSERT INTO meta_discovery_schedules 
				(schedule_id, name, description, connector_config, schedule_type, cron_expression,
				 interval_minutes, start_time, end_time, is_enabled, is_one_time, last_run,
				 next_run, created_by, created_at, tenant_id)
				VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
				ON CONFLICT (schedule_id, tenant_id) DO UPDATE SET
					name = EXCLUDED.name,
					description = EXCLUDED.description,
					connector_config = EXCLUDED.connector_config,
					schedule_type = EXCLUDED.schedule_type,
					cron_expression = EXCLUDED.cron_expression,
					interval_minutes = EXCLUDED.interval_minutes,
					start_time = EXCLUDED.start_time,
					end_time = EXCLUDED.end_time,
					is_enabled = EXCLUDED.is_enabled,
					is_one_time = EXCLUDED.is_one_time,
					last_run = EXCLUDED.last_run,
					next_run = EXCLUDED.next_run,
					updated_at = NOW()
				""",
				(
					schedule_data['schedule_id'], schedule_data['name'], schedule_data['description'],
					schedule_data['connector_config'], schedule_data['schedule_type'], 
					schedule_data['cron_expression'], schedule_data['interval_minutes'],
					schedule_data['start_time'], schedule_data['end_time'], schedule_data['is_enabled'],
					schedule_data['is_one_time'], schedule_data['last_run'], schedule_data['next_run'],
					schedule_data['created_by'], schedule_data['created_at'], schedule_data['tenant_id']
				)
			)
			
			await self._log_info(f"Persisted discovery schedule: {schedule.name}")
			
		except Exception as e:
			await self._log_error(f"Failed to persist discovery schedule: {str(e)}")
	
	async def _log_info(self, message: str):
		"""Log info message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META DISCOVERY INFO: {message}")
	
	async def _log_error(self, message: str):
		"""Log error message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META DISCOVERY ERROR: {message}")


# Factory function for easy initialization
async def create_discovery_service(
	db_manager: MetaDatabaseManager,
	integration_manager: APGMetadataIntegrationManager,
	config: Dict[str, Any] = None
) -> MetadataDiscoveryService:
	"""Factory function to create and initialize discovery service"""
	discovery_service = MetadataDiscoveryService(db_manager, integration_manager, config)
	await discovery_service.initialize()
	return discovery_service