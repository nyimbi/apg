#!/usr/bin/env python3
"""
APG Metadata Management - Main Service
Unified metadata management service orchestrating all components

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str

from .database import MetaDatabaseManager, create_database_manager
from .integrations import APGMetadataIntegrationManager, create_apg_integration_manager
from .discovery import MetadataDiscoveryService, DiscoverySchedule, create_discovery_service
from .ai_classifier import AIClassificationEngine, create_ai_classifier
from .lineage_engine import DataLineageEngine, LineageEdge, create_lineage_engine
from .search_engine import MetadataSearchEngine, SearchQuery, create_search_engine
from .connectors import ConnectorConfig


class ServiceStatus(str, Enum):
	"""Service status enumeration"""
	INITIALIZING = "initializing"
	RUNNING = "running"
	DEGRADED = "degraded"
	STOPPED = "stopped"
	ERROR = "error"


@dataclass
class ServiceHealth:
	"""Service health status"""
	service_name: str = "metadata_management"
	status: ServiceStatus = ServiceStatus.STOPPED
	uptime_seconds: float = 0.0
	last_health_check: datetime = field(default_factory=datetime.utcnow)
	
	# Component health
	database_healthy: bool = False
	discovery_healthy: bool = False
	ai_classifier_healthy: bool = False
	lineage_engine_healthy: bool = False
	search_engine_healthy: bool = False
	integrations_healthy: bool = False
	
	# Performance metrics
	total_assets: int = 0
	total_discoveries: int = 0
	total_searches: int = 0
	total_classifications: int = 0
	avg_response_time_ms: float = 0.0
	
	# Error tracking
	error_count_24h: int = 0
	last_error: Optional[str] = None
	warnings: List[str] = field(default_factory=list)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for API response"""
		return {
			"service_name": self.service_name,
			"status": self.status.value,
			"uptime_seconds": self.uptime_seconds,
			"last_health_check": self.last_health_check.isoformat(),
			"components": {
				"database": self.database_healthy,
				"discovery": self.discovery_healthy,
				"ai_classifier": self.ai_classifier_healthy,
				"lineage_engine": self.lineage_engine_healthy,
				"search_engine": self.search_engine_healthy,
				"integrations": self.integrations_healthy
			},
			"metrics": {
				"total_assets": self.total_assets,
				"total_discoveries": self.total_discoveries,
				"total_searches": self.total_searches,
				"total_classifications": self.total_classifications,
				"avg_response_time_ms": self.avg_response_time_ms
			},
			"issues": {
				"error_count_24h": self.error_count_24h,
				"last_error": self.last_error,
				"warnings": self.warnings
			}
		}


class APGMetadataService:
	"""Main APG Metadata Management Service orchestrating all components"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.service_start_time = datetime.utcnow()
		
		# Component instances
		self.db_manager: Optional[MetaDatabaseManager] = None
		self.integration_manager: Optional[APGMetadataIntegrationManager] = None
		self.discovery_service: Optional[MetadataDiscoveryService] = None
		self.ai_classifier: Optional[AIClassificationEngine] = None
		self.lineage_engine: Optional[DataLineageEngine] = None
		self.search_engine: Optional[MetadataSearchEngine] = None
		
		# Service state
		self.health = ServiceHealth()
		self.initialized = False
		
		# Performance tracking
		self.request_count = 0
		self.total_response_time = 0.0
		self.error_count_24h = 0
		
		# Background tasks
		self.health_check_task: Optional[asyncio.Task] = None
		self.maintenance_task: Optional[asyncio.Task] = None
		
		# Service configuration
		self.enable_auto_discovery = config.get('enable_auto_discovery', True)
		self.enable_ai_classification = config.get('enable_ai_classification', True)
		self.enable_lineage_tracking = config.get('enable_lineage_tracking', True)
		self.enable_advanced_search = config.get('enable_advanced_search', True)
		
		# Health check interval
		self.health_check_interval = config.get('health_check_interval_seconds', 60)
		self.maintenance_interval = config.get('maintenance_interval_seconds', 3600)  # 1 hour
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize all service components"""
		if self.initialized:
			return {"status": "already_initialized"}
		
		self.health.status = ServiceStatus.INITIALIZING
		
		try:
			await self._log_info("Starting APG Metadata Management Service initialization")
			
			# Initialize core database manager
			await self._log_info("Initializing database manager...")
			self.db_manager = await create_database_manager(self.config.get('database', {}))
			self.health.database_healthy = True
			await self._log_info("✓ Database manager initialized")
			
			# Initialize APG integrations
			await self._log_info("Initializing APG integrations...")
			self.integration_manager = await create_apg_integration_manager(
				self.config.get('integrations', {}),
				self.db_manager
			)
			self.health.integrations_healthy = True
			await self._log_info("✓ APG integrations initialized")
			
			# Initialize discovery service
			if self.enable_auto_discovery:
				await self._log_info("Initializing discovery service...")
				self.discovery_service = await create_discovery_service(
					self.db_manager,
					self.integration_manager,
					self.config.get('discovery', {})
				)
				self.health.discovery_healthy = True
				await self._log_info("✓ Discovery service initialized")
			
			# Initialize AI classifier
			if self.enable_ai_classification:
				await self._log_info("Initializing AI classifier...")
				self.ai_classifier = await create_ai_classifier(
					self.db_manager,
					self.integration_manager,
					self.config.get('ai_classifier', {})
				)
				self.health.ai_classifier_healthy = True
				await self._log_info("✓ AI classifier initialized")
			
			# Initialize lineage engine
			if self.enable_lineage_tracking:
				await self._log_info("Initializing lineage engine...")
				self.lineage_engine = await create_lineage_engine(
					self.db_manager,
					self.integration_manager,
					self.config.get('lineage', {})
				)
				self.health.lineage_engine_healthy = True
				await self._log_info("✓ Lineage engine initialized")
			
			# Initialize search engine
			if self.enable_advanced_search:
				await self._log_info("Initializing search engine...")
				self.search_engine = await create_search_engine(
					self.db_manager,
					self.integration_manager,
					self.config.get('search', {})
				)
				self.health.search_engine_healthy = True
				await self._log_info("✓ Search engine initialized")
			
			# Start background tasks
			await self._start_background_tasks()
			
			# Update service status
			self.health.status = ServiceStatus.RUNNING
			self.health.last_health_check = datetime.utcnow()
			self.initialized = True
			
			await self._log_info("🚀 APG Metadata Management Service initialized successfully")
			
			return {
				"status": "initialized",
				"components_initialized": {
					"database_manager": True,
					"integration_manager": True,
					"discovery_service": self.enable_auto_discovery,
					"ai_classifier": self.enable_ai_classification,
					"lineage_engine": self.enable_lineage_tracking,
					"search_engine": self.enable_advanced_search
				},
				"service_capabilities": await self._get_service_capabilities(),
				"initialization_time_ms": (datetime.utcnow() - self.service_start_time).total_seconds() * 1000
			}
			
		except Exception as e:
			self.health.status = ServiceStatus.ERROR
			self.health.last_error = str(e)
			await self._log_error(f"Service initialization failed: {str(e)}")
			raise
	
	async def shutdown(self):
		"""Shutdown all service components gracefully"""
		if not self.initialized:
			return
		
		try:
			await self._log_info("Shutting down APG Metadata Management Service...")
			
			self.health.status = ServiceStatus.STOPPED
			
			# Stop background tasks
			if self.health_check_task and not self.health_check_task.done():
				self.health_check_task.cancel()
				try:
					await self.health_check_task
				except asyncio.CancelledError:
					pass
			
			if self.maintenance_task and not self.maintenance_task.done():
				self.maintenance_task.cancel()
				try:
					await self.maintenance_task
				except asyncio.CancelledError:
					pass
			
			# Shutdown components in reverse order
			if self.search_engine:
				# Search engine doesn't have explicit shutdown
				pass
			
			if self.lineage_engine:
				# Lineage engine doesn't have explicit shutdown
				pass
			
			if self.ai_classifier:
				# AI classifier doesn't have explicit shutdown
				pass
			
			if self.discovery_service:
				await self.discovery_service.shutdown()
			
			if self.integration_manager:
				await self.integration_manager.shutdown()
			
			if self.db_manager:
				await self.db_manager.close()
			
			self.initialized = False
			
			await self._log_info("✓ APG Metadata Management Service shutdown completed")
			
		except Exception as e:
			await self._log_error(f"Service shutdown failed: {str(e)}")
	
	# === Discovery Operations ===
	
	async def create_discovery_schedule(self, schedule: DiscoverySchedule) -> str:
		"""Create a new discovery schedule"""
		if not self.discovery_service:
			raise RuntimeError("Discovery service not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			schedule_id = await self.discovery_service.create_discovery_schedule(schedule)
			
			await self._track_performance(start_time)
			await self._log_info(f"Created discovery schedule: {schedule_id}")
			
			return schedule_id
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	async def run_discovery(self, schedule_id: str, override_config: Dict[str, Any] = None) -> str:
		"""Run a discovery job"""
		if not self.discovery_service:
			raise RuntimeError("Discovery service not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			job_id = await self.discovery_service.run_discovery_job(schedule_id, override_config)
			
			await self._track_performance(start_time)
			await self._log_info(f"Started discovery job: {job_id}")
			
			self.health.total_discoveries += 1
			
			return job_id
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	async def get_discovery_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
		"""Get discovery job status"""
		if not self.discovery_service:
			raise RuntimeError("Discovery service not initialized")
		
		return await self.discovery_service.get_discovery_job_status(job_id)
	
	# === AI Classification Operations ===
	
	async def classify_column_data(self,
				       column_name: str,
				       data_type: str,
				       sample_data: List[Any],
				       context: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Classify column data using AI"""
		if not self.ai_classifier:
			raise RuntimeError("AI classifier not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			result = await self.ai_classifier.classify_column_data(
				column_name, data_type, sample_data, context
			)
			
			await self._track_performance(start_time)
			await self._log_info(f"Classified column '{column_name}' as '{result.classification}'")
			
			self.health.total_classifications += 1
			
			return result.to_dict()
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	# === Lineage Operations ===
	
	async def add_lineage_relationship(self, edge: LineageEdge) -> str:
		"""Add a lineage relationship"""
		if not self.lineage_engine:
			raise RuntimeError("Lineage engine not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			edge_id = await self.lineage_engine.add_lineage_relationship(edge)
			
			await self._track_performance(start_time)
			await self._log_info(f"Added lineage relationship: {edge_id}")
			
			return edge_id
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	async def get_lineage_path(self,
				   asset_id: str,
				   tenant_id: str,
				   direction: str = "both",
				   max_depth: int = None) -> List[Dict[str, Any]]:
		"""Get lineage paths for an asset"""
		if not self.lineage_engine:
			raise RuntimeError("Lineage engine not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			from .lineage_engine import LineageDirection
			
			lineage_direction = LineageDirection(direction)
			paths = await self.lineage_engine.get_lineage_path(
				asset_id, tenant_id, lineage_direction, max_depth
			)
			
			await self._track_performance(start_time)
			await self._log_info(f"Retrieved {len(paths)} lineage paths for asset {asset_id}")
			
			return [path.to_dict() for path in paths]
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	async def analyze_impact(self,
				 asset_id: str,
				 tenant_id: str,
				 change_type: str = "schema_change",
				 change_details: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Perform impact analysis"""
		if not self.lineage_engine:
			raise RuntimeError("Lineage engine not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			result = await self.lineage_engine.analyze_impact(
				asset_id, tenant_id, change_type, change_details
			)
			
			await self._track_performance(start_time)
			await self._log_info(f"Impact analysis completed for {asset_id}: {result.total_impacted_assets} assets")
			
			return result.to_dict()
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	# === Search Operations ===
	
	async def search_metadata(self, query: SearchQuery) -> Dict[str, Any]:
		"""Search metadata assets"""
		if not self.search_engine:
			raise RuntimeError("Search engine not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			response = await self.search_engine.search(query)
			
			await self._track_performance(start_time)
			await self._log_info(f"Search completed: '{query.query_text}' -> {response.total_results} results")
			
			self.health.total_searches += 1
			
			return response.to_dict()
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	# === Asset Operations ===
	
	async def get_asset(self, asset_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
		"""Get metadata asset by ID"""
		start_time = asyncio.get_event_loop().time()
		
		try:
			async with self.db_manager.get_session(tenant_id) as session:
				from sqlalchemy import select
				from .models import MetaAsset
				
				stmt = select(MetaAsset).where(
					MetaAsset.id == asset_id,
					MetaAsset.tenant_id == tenant_id,
					MetaAsset.is_deleted == False
				)
				
				result = await session.execute(stmt)
				asset = result.scalar_one_or_none()
				
				if asset:
					await self._track_performance(start_time)
					return await self._asset_to_dict(asset)
				
				return None
				
		except Exception as e:
			await self._track_error(e)
			raise
	
	async def list_assets(self,
			      tenant_id: str,
			      filters: Dict[str, Any] = None,
			      limit: int = 100,
			      offset: int = 0) -> Dict[str, Any]:
		"""List metadata assets with filtering"""
		start_time = asyncio.get_event_loop().time()
		
		try:
			async with self.db_manager.get_session(tenant_id) as session:
				from sqlalchemy import select
				from .models import MetaAsset
				
				stmt = select(MetaAsset).where(
					MetaAsset.tenant_id == tenant_id,
					MetaAsset.is_deleted == False
				)
				
				# Apply filters
				if filters:
					for field, value in filters.items():
						if hasattr(MetaAsset, field):
							attr = getattr(MetaAsset, field)
							if isinstance(value, list):
								stmt = stmt.where(attr.in_(value))
							else:
								stmt = stmt.where(attr == value)
				
				# Apply pagination
				stmt = stmt.offset(offset).limit(limit)
				
				result = await session.execute(stmt)
				assets = result.scalars().all()
				
				# Get total count for pagination
				count_stmt = select(MetaAsset.id).where(
					MetaAsset.tenant_id == tenant_id,
					MetaAsset.is_deleted == False
				)
				
				if filters:
					for field, value in filters.items():
						if hasattr(MetaAsset, field):
							attr = getattr(MetaAsset, field)
							if isinstance(value, list):
								count_stmt = count_stmt.where(attr.in_(value))
							else:
								count_stmt = count_stmt.where(attr == value)
				
				total_result = await session.execute(count_stmt)
				total_count = len(total_result.scalars().all())
				
				await self._track_performance(start_time)
				
				return {
					"assets": [await self._asset_to_dict(asset) for asset in assets],
					"pagination": {
						"offset": offset,
						"limit": limit,
						"total": total_count,
						"has_more": (offset + limit) < total_count
					}
				}
				
		except Exception as e:
			await self._track_error(e)
			raise
	
	# === Health and Monitoring ===
	
	async def get_health_status(self) -> Dict[str, Any]:
		"""Get service health status"""
		await self._update_health_status()
		return self.health.to_dict()
	
	async def get_service_metrics(self) -> Dict[str, Any]:
		"""Get service performance metrics"""
		metrics = {
			"uptime_seconds": (datetime.utcnow() - self.service_start_time).total_seconds(),
			"request_count": self.request_count,
			"avg_response_time_ms": self.total_response_time / max(self.request_count, 1),
			"error_rate": self.error_count_24h / max(self.request_count, 1),
		}
		
		# Add component-specific metrics
		if self.search_engine:
			search_metrics = await self.search_engine.get_search_analytics()
			metrics["search"] = search_metrics
		
		if self.ai_classifier:
			classifier_metrics = await self.ai_classifier.get_classification_stats()
			metrics["ai_classifier"] = classifier_metrics
		
		if self.db_manager:
			db_metrics = await self.db_manager.get_database_stats()
			metrics["database"] = db_metrics
		
		return metrics
	
	# === Internal Methods ===
	
	async def _get_service_capabilities(self) -> Dict[str, Any]:
		"""Get service capabilities"""
		return {
			"auto_discovery": self.enable_auto_discovery,
			"ai_classification": self.enable_ai_classification,
			"lineage_tracking": self.enable_lineage_tracking,
			"advanced_search": self.enable_advanced_search,
			"natural_language_queries": self.enable_advanced_search,
			"real_time_lineage": self.enable_lineage_tracking,
			"federated_learning": self.enable_ai_classification,
			"apg_integration": True,
			"multi_tenant": True,
			"graph_analytics": self.enable_lineage_tracking
		}
	
	async def _start_background_tasks(self):
		"""Start background monitoring and maintenance tasks"""
		self.health_check_task = asyncio.create_task(self._health_check_loop())
		self.maintenance_task = asyncio.create_task(self._maintenance_loop())
	
	async def _health_check_loop(self):
		"""Background health check loop"""
		while self.initialized:
			try:
				await asyncio.sleep(self.health_check_interval)
				await self._update_health_status()
			except asyncio.CancelledError:
				break
			except Exception as e:
				await self._log_error(f"Health check failed: {str(e)}")
	
	async def _maintenance_loop(self):
		"""Background maintenance loop"""
		while self.initialized:
			try:
				await asyncio.sleep(self.maintenance_interval)
				await self._run_maintenance_tasks()
			except asyncio.CancelledError:
				break
			except Exception as e:
				await self._log_error(f"Maintenance task failed: {str(e)}")
	
	async def _update_health_status(self):
		"""Update service health status"""
		try:
			self.health.last_health_check = datetime.utcnow()
			self.health.uptime_seconds = (datetime.utcnow() - self.service_start_time).total_seconds()
			
			# Check database health
			if self.db_manager:
				db_health = await self.db_manager.health_check()
				self.health.database_healthy = db_health.is_healthy
			
			# Update asset count
			if self.db_manager:
				try:
					async with self.db_manager.get_session() as session:
						from sqlalchemy import select, func
						from .models import MetaAsset
						
						stmt = select(func.count(MetaAsset.id)).where(
							MetaAsset.is_deleted == False
						)
						result = await session.execute(stmt)
						self.health.total_assets = result.scalar() or 0
				except:
					pass
			
			# Calculate average response time
			if self.request_count > 0:
				self.health.avg_response_time_ms = self.total_response_time / self.request_count
			
			# Determine overall status
			component_health = [
				self.health.database_healthy,
				self.health.discovery_healthy or not self.enable_auto_discovery,
				self.health.ai_classifier_healthy or not self.enable_ai_classification,
				self.health.lineage_engine_healthy or not self.enable_lineage_tracking,
				self.health.search_engine_healthy or not self.enable_advanced_search,
				self.health.integrations_healthy
			]
			
			if all(component_health):
				self.health.status = ServiceStatus.RUNNING
			elif any(component_health):
				self.health.status = ServiceStatus.DEGRADED
			else:
				self.health.status = ServiceStatus.ERROR
			
		except Exception as e:
			self.health.status = ServiceStatus.ERROR
			self.health.last_error = str(e)
			await self._log_error(f"Health status update failed: {str(e)}")
	
	async def _run_maintenance_tasks(self):
		"""Run periodic maintenance tasks"""
		await self._log_info("Running maintenance tasks...")
		
		try:
			# Database maintenance
			if self.db_manager:
				await self.db_manager.optimize_performance()
			
			# Reset 24h error counter
			current_time = datetime.utcnow()
			if not hasattr(self, '_last_error_reset') or (current_time - self._last_error_reset).days >= 1:
				self.error_count_24h = 0
				self._last_error_reset = current_time
			
			await self._log_info("✓ Maintenance tasks completed")
			
		except Exception as e:
			await self._log_error(f"Maintenance tasks failed: {str(e)}")
	
	async def _track_performance(self, start_time: float):
		"""Track request performance"""
		response_time = (asyncio.get_event_loop().time() - start_time) * 1000
		self.request_count += 1
		self.total_response_time += response_time
	
	async def _track_error(self, error: Exception):
		"""Track error occurrence"""
		self.error_count_24h += 1
		self.health.last_error = str(error)
		await self._log_error(f"Service error: {str(error)}")
	
	async def _asset_to_dict(self, asset) -> Dict[str, Any]:
		"""Convert MetaAsset to dictionary"""
		return {
			"id": asset.id,
			"name": asset.name,
			"display_name": asset.display_name,
			"description": asset.description,
			"asset_type": asset.asset_type,
			"source_system": asset.source_system,
			"source_system_type": asset.source_system_type,
			"external_id": asset.external_id,
			"status": asset.status,
			"business_domain": asset.business_domain,
			"schema_info": asset.schema_info,
			"quality_score": asset.quality_score,
			"tags": asset.tags,
			"owner": asset.owner,
			"steward": asset.steward,
			"custom_attributes": asset.custom_attributes,
			"created_at": asset.created_at.isoformat() if asset.created_at else None,
			"updated_at": asset.updated_at.isoformat() if asset.updated_at else None,
			"created_by": asset.created_by,
			"updated_by": asset.updated_by
		}
	
	async def _log_info(self, message: str):
		"""Log info message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META SERVICE INFO: {message}")
	
	async def _log_error(self, message: str):
		"""Log error message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META SERVICE ERROR: {message}")


# Factory function for easy initialization
async def create_metadata_service(config: Dict[str, Any] = None) -> APGMetadataService:
	"""Factory function to create and initialize metadata service"""
	service = APGMetadataService(config)
	await service.initialize()
	return service


# Service singleton for global access
_metadata_service_instance: Optional[APGMetadataService] = None


async def get_metadata_service(config: Dict[str, Any] = None) -> APGMetadataService:
	"""Get or create the global metadata service instance"""
	global _metadata_service_instance
	
	if _metadata_service_instance is None:
		_metadata_service_instance = await create_metadata_service(config)
	
	return _metadata_service_instance


async def shutdown_metadata_service():
	"""Shutdown the global metadata service instance"""
	global _metadata_service_instance
	
	if _metadata_service_instance:
		await _metadata_service_instance.shutdown()
		_metadata_service_instance = None