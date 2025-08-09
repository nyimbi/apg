#!/usr/bin/env python3
"""
APG Metadata Management - APG Ecosystem Integration
Comprehensive integration with APG platform capabilities and services

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from uuid_extensions import uuid7str

from .database import MetaDatabaseManager


class APGCapability(str, Enum):
	"""APG capabilities that metadata integrates with"""
	AUTH_RBAC = "auth_rbac"
	AUDIT_COMPLIANCE = "audit_compliance"
	MDM = "mdm"
	AI_ORCHESTRATION = "ai_orchestration"
	FEDERATED_LEARNING = "federated_learning"
	NOTIFICATION_ENGINE = "notification_engine"
	REAL_TIME_COLLABORATION = "real_time_collaboration"
	VISUALIZATION_3D = "visualization_3d"


class MetadataEventType(str, Enum):
	"""Types of metadata events for APG ecosystem"""
	ASSET_DISCOVERED = "asset_discovered"
	ASSET_UPDATED = "asset_updated"
	ASSET_DELETED = "asset_deleted"
	LINEAGE_CREATED = "lineage_created"
	CLASSIFICATION_CHANGED = "classification_changed"
	QUALITY_ASSESSED = "quality_assessed"
	POLICY_VIOLATED = "policy_violated"
	USER_ACCESS = "user_access"


@dataclass
class MetadataEvent:
	"""Metadata event for APG ecosystem propagation"""
	event_id: str = field(default_factory=uuid7str)
	event_type: MetadataEventType = MetadataEventType.ASSET_UPDATED
	tenant_id: str = ""
	asset_id: Optional[str] = None
	user_id: Optional[str] = None
	timestamp: datetime = field(default_factory=datetime.utcnow)
	payload: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)
	correlation_id: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert event to dictionary for serialization"""
		return {
			"event_id": self.event_id,
			"event_type": self.event_type.value,
			"tenant_id": self.tenant_id,
			"asset_id": self.asset_id,
			"user_id": self.user_id,
			"timestamp": self.timestamp.isoformat(),
			"payload": self.payload,
			"metadata": self.metadata,
			"correlation_id": self.correlation_id
		}


class APGEventPublisher:
	"""Event publisher for APG Message Queue Event Bus integration"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.event_bus_url = config.get('apg_event_bus_url', 'http://localhost:8081/events')
		self.api_key = config.get('apg_api_key')
		self.session: Optional[aiohttp.ClientSession] = None
		self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
		self.publisher_task: Optional[asyncio.Task] = None
		self.is_running = False
		
		# Event routing configuration
		self.event_routes = {
			MetadataEventType.ASSET_DISCOVERED: ["mdm", "ai_orchestration"],
			MetadataEventType.CLASSIFICATION_CHANGED: ["audit_compliance", "auth_rbac"],
			MetadataEventType.QUALITY_ASSESSED: ["notification_engine"],
			MetadataEventType.POLICY_VIOLATED: ["audit_compliance", "notification_engine"],
			MetadataEventType.USER_ACCESS: ["audit_compliance"]
		}
	
	async def start(self):
		"""Start the event publisher"""
		if self.is_running:
			return
		
		self.session = aiohttp.ClientSession(
			timeout=aiohttp.ClientTimeout(total=30),
			headers={
				'Content-Type': 'application/json',
				'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
				'User-Agent': 'APG-Metadata/1.0.0'
			}
		)
		
		self.publisher_task = asyncio.create_task(self._publisher_worker())
		self.is_running = True
	
	async def stop(self):
		"""Stop the event publisher"""
		if not self.is_running:
			return
		
		self.is_running = False
		
		if self.publisher_task:
			self.publisher_task.cancel()
			try:
				await self.publisher_task
			except asyncio.CancelledError:
				pass
		
		if self.session:
			await self.session.close()
	
	async def publish_event(self, event: MetadataEvent) -> bool:
		"""Publish metadata event to APG ecosystem"""
		if not self.is_running:
			await self.start()
		
		try:
			await self.event_queue.put(event)
			return True
		except asyncio.QueueFull:
			await self._log_error(f"Event queue full, dropping event: {event.event_id}")
			return False
	
	async def _publisher_worker(self):
		"""Background worker to publish events to APG event bus"""
		while self.is_running:
			try:
				# Get event from queue with timeout
				try:
					event = await asyncio.wait_for(
						self.event_queue.get(), 
						timeout=1.0
					)
				except asyncio.TimeoutError:
					continue
				
				# Publish event
				success = await self._send_event_to_bus(event)
				if success:
					await self._log_info(f"Published event: {event.event_id}")
				else:
					await self._log_error(f"Failed to publish event: {event.event_id}")
				
				# Mark task done
				self.event_queue.task_done()
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				await self._log_error(f"Event publisher worker error: {str(e)}")
	
	async def _send_event_to_bus(self, event: MetadataEvent) -> bool:
		"""Send event to APG message queue event bus"""
		try:
			# Get target capabilities for this event type
			targets = self.event_routes.get(event.event_type, [])
			
			# Prepare event payload
			event_data = event.to_dict()
			event_data['targets'] = targets
			event_data['source'] = 'metadata'
			
			# Send to event bus
			async with self.session.post(
				f"{self.event_bus_url}/publish",
				json=event_data
			) as response:
				if response.status == 200:
					return True
				else:
					await self._log_error(
						f"Event bus returned {response.status}: {await response.text()}"
					)
					return False
					
		except Exception as e:
			await self._log_error(f"Failed to send event to bus: {str(e)}")
			return False
	
	async def _log_info(self, message: str):
		"""Log info message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META EVENT INFO: {message}")
	
	async def _log_error(self, message: str):
		"""Log error message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META EVENT ERROR: {message}")


class APGAuditLogger:
	"""APG Audit & Compliance integration for metadata operations"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.audit_service_url = config.get('apg_audit_service_url', 'http://localhost:8082/audit')
		self.api_key = config.get('apg_api_key')
		self.session: Optional[aiohttp.ClientSession] = None
		self.audit_queue: asyncio.Queue = asyncio.Queue(maxsize=5000)
		self.audit_task: Optional[asyncio.Task] = None
		self.is_running = False
	
	async def start(self):
		"""Start the audit logger"""
		if self.is_running:
			return
		
		self.session = aiohttp.ClientSession(
			timeout=aiohttp.ClientTimeout(total=30),
			headers={
				'Content-Type': 'application/json',
				'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
				'User-Agent': 'APG-Metadata-Audit/1.0.0'
			}
		)
		
		self.audit_task = asyncio.create_task(self._audit_worker())
		self.is_running = True
	
	async def stop(self):
		"""Stop the audit logger"""
		if not self.is_running:
			return
		
		self.is_running = False
		
		if self.audit_task:
			self.audit_task.cancel()
			try:
				await self.audit_task
			except asyncio.CancelledError:
				pass
		
		if self.session:
			await self.session.close()
	
	async def log_metadata_operation(self,
									operation: str,
									asset_id: Optional[str],
									user_id: str,
									tenant_id: str,
									details: Dict[str, Any] = None,
									sensitive: bool = False) -> bool:
		"""Log metadata operation for audit compliance"""
		audit_record = {
			"audit_id": uuid7str(),
			"timestamp": datetime.utcnow().isoformat(),
			"service": "metadata",
			"operation": operation,
			"resource_type": "metadata_asset",
			"resource_id": asset_id,
			"user_id": user_id,
			"tenant_id": tenant_id,
			"details": details or {},
			"sensitive": sensitive,
			"client_ip": None,  # Would be set by calling code
			"session_id": None  # Would be set by calling code
		}
		
		try:
			await self.audit_queue.put(audit_record)
			return True
		except asyncio.QueueFull:
			await self._log_error("Audit queue full, dropping audit record")
			return False
	
	async def _audit_worker(self):
		"""Background worker to send audit records to APG audit service"""
		while self.is_running:
			try:
				try:
					audit_record = await asyncio.wait_for(
						self.audit_queue.get(),
						timeout=1.0
					)
				except asyncio.TimeoutError:
					continue
				
				# Send to audit service
				success = await self._send_audit_record(audit_record)
				if not success:
					await self._log_error(f"Failed to send audit record: {audit_record['audit_id']}")
				
				self.audit_queue.task_done()
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				await self._log_error(f"Audit worker error: {str(e)}")
	
	async def _send_audit_record(self, record: Dict[str, Any]) -> bool:
		"""Send audit record to APG audit service"""
		try:
			async with self.session.post(
				f"{self.audit_service_url}/log",
				json=record
			) as response:
				return response.status == 201
		except Exception as e:
			await self._log_error(f"Failed to send audit record: {str(e)}")
			return False
	
	async def _log_error(self, message: str):
		"""Log error message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META AUDIT ERROR: {message}")


class APGAuthIntegration:
	"""Integration with APG Authentication & RBAC capability"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.auth_service_url = config.get('apg_auth_service_url', 'http://localhost:8083/auth')
		self.api_key = config.get('apg_api_key')
		self.session: Optional[aiohttp.ClientSession] = None
		self.permission_cache: Dict[str, Dict[str, Any]] = {}
		self.cache_ttl = 300  # 5 minutes
	
	async def initialize(self):
		"""Initialize auth integration"""
		self.session = aiohttp.ClientSession(
			timeout=aiohttp.ClientTimeout(total=10),
			headers={
				'Content-Type': 'application/json',
				'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
			}
		)
	
	async def close(self):
		"""Close auth integration"""
		if self.session:
			await self.session.close()
	
	async def check_permission(self, 
							  user_id: str,
							  tenant_id: str,
							  resource: str,
							  action: str) -> bool:
		"""Check if user has permission for metadata operation"""
		cache_key = f"{user_id}:{tenant_id}:{resource}:{action}"
		
		# Check cache first
		cached = self.permission_cache.get(cache_key)
		if cached and datetime.utcnow() < cached['expires']:
			return cached['allowed']
		
		try:
			# Call APG auth service
			permission_request = {
				"user_id": user_id,
				"tenant_id": tenant_id,
				"resource": f"metadata.{resource}",
				"action": action
			}
			
			async with self.session.post(
				f"{self.auth_service_url}/check_permission",
				json=permission_request
			) as response:
				if response.status == 200:
					result = await response.json()
					allowed = result.get('allowed', False)
					
					# Cache result
					self.permission_cache[cache_key] = {
						'allowed': allowed,
						'expires': datetime.utcnow() + timedelta(seconds=self.cache_ttl)
					}
					
					return allowed
				else:
					# Default deny if service unavailable
					return False
					
		except Exception:
			# Default deny if error
			return False
	
	async def get_user_context(self, token: str) -> Optional[Dict[str, Any]]:
		"""Get user context from APG auth service"""
		try:
			async with self.session.post(
				f"{self.auth_service_url}/validate_token",
				json={"token": token}
			) as response:
				if response.status == 200:
					return await response.json()
				else:
					return None
		except Exception:
			return None


class APGMDMIntegration:
	"""Integration with APG Master Data Management capability"""
	
	def __init__(self, config: Dict[str, Any], db_manager: MetaDatabaseManager):
		self.config = config
		self.db_manager = db_manager
		self.mdm_service_url = config.get('apg_mdm_service_url', 'http://localhost:8084/mdm')
		self.api_key = config.get('apg_api_key')
		self.session: Optional[aiohttp.ClientSession] = None
		self.sync_enabled = config.get('mdm_sync_enabled', True)
	
	async def initialize(self):
		"""Initialize MDM integration"""
		if not self.sync_enabled:
			return
		
		self.session = aiohttp.ClientSession(
			timeout=aiohttp.ClientTimeout(total=30),
			headers={
				'Content-Type': 'application/json',
				'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
			}
		)
	
	async def close(self):
		"""Close MDM integration"""
		if self.session:
			await self.session.close()
	
	async def sync_mdm_entities(self, tenant_id: str) -> Dict[str, Any]:
		"""Synchronize MDM entities as metadata assets"""
		if not self.sync_enabled or not self.session:
			return {"status": "disabled", "synced": 0}
		
		try:
			# Get all MDM entities for tenant
			async with self.session.get(
				f"{self.mdm_service_url}/entities",
				params={"tenant_id": tenant_id}
			) as response:
				if response.status != 200:
					return {"status": "error", "message": "Failed to fetch MDM entities"}
				
				entities_data = await response.json()
				entities = entities_data.get('entities', [])
				
				synced_count = 0
				errors = []
				
				# Create/update metadata assets for each MDM entity
				async with self.db_manager.get_session(tenant_id) as session:
					for entity in entities:
						try:
							await self._create_metadata_asset_from_mdm(session, entity, tenant_id)
							synced_count += 1
						except Exception as e:
							errors.append(f"Entity {entity.get('id')}: {str(e)}")
				
				return {
					"status": "success",
					"synced": synced_count,
					"total": len(entities),
					"errors": errors
				}
				
		except Exception as e:
			return {"status": "error", "message": str(e)}
	
	async def _create_metadata_asset_from_mdm(self, 
											 session, 
											 mdm_entity: Dict[str, Any], 
											 tenant_id: str):
		"""Create metadata asset from MDM entity"""
		from .models import MetaAsset, AssetType, SourceSystemType, AssetStatus
		
		# Map MDM entity to metadata asset
		asset = MetaAsset(
			tenant_id=tenant_id,
			name=mdm_entity.get('entity_name', ''),
			display_name=mdm_entity.get('display_name'),
			description=mdm_entity.get('description'),
			asset_type=AssetType.CUSTOM,
			source_system="apg_mdm",
			source_system_type=SourceSystemType.APPLICATION,
			external_id=mdm_entity.get('id'),
			status=AssetStatus.ACTIVE,
			business_domain=mdm_entity.get('business_domain', 'master_data'),
			schema_info={
				"entity_type": mdm_entity.get('entity_type'),
				"attributes": mdm_entity.get('attributes', {}),
				"mdm_version": mdm_entity.get('version')
			},
			tags=['mdm', 'master_data'] + mdm_entity.get('tags', []),
			custom_attributes={
				"mdm_entity_id": mdm_entity.get('id'),
				"quality_score": mdm_entity.get('quality_score', 0),
				"mdm_status": mdm_entity.get('status')
			},
			created_by="system:mdm_sync",
			updated_by="system:mdm_sync"
		)
		
		session.add(asset)


class APGNotificationIntegration:
	"""Integration with APG Notification Engine"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.notification_service_url = config.get('apg_notification_service_url', 'http://localhost:8085/notifications')
		self.api_key = config.get('apg_api_key')
		self.session: Optional[aiohttp.ClientSession] = None
	
	async def initialize(self):
		"""Initialize notification integration"""
		self.session = aiohttp.ClientSession(
			timeout=aiohttp.ClientTimeout(total=10),
			headers={
				'Content-Type': 'application/json',
				'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
			}
		)
	
	async def close(self):
		"""Close notification integration"""
		if self.session:
			await self.session.close()
	
	async def send_notification(self,
							   recipients: List[str],
							   title: str,
							   message: str,
							   notification_type: str = "info",
							   metadata: Dict[str, Any] = None) -> bool:
		"""Send notification through APG notification engine"""
		try:
			notification_data = {
				"recipients": recipients,
				"title": title,
				"message": message,
				"type": notification_type,
				"source": "metadata",
				"metadata": metadata or {},
				"timestamp": datetime.utcnow().isoformat()
			}
			
			async with self.session.post(
				f"{self.notification_service_url}/send",
				json=notification_data
			) as response:
				return response.status == 200
				
		except Exception:
			return False
	
	async def notify_quality_issues(self,
								   asset_name: str,
								   quality_score: float,
								   issues: List[Dict[str, Any]],
								   steward: str) -> bool:
		"""Notify data steward about quality issues"""
		if quality_score >= 80.0:  # Only notify for significant issues
			return True
		
		critical_issues = [i for i in issues if i.get('severity') == 'critical']
		issue_count = len(issues)
		critical_count = len(critical_issues)
		
		title = f"Data Quality Alert: {asset_name}"
		message = f"""
		Quality Score: {quality_score:.1f}%
		Total Issues: {issue_count}
		Critical Issues: {critical_count}
		
		Please review and remediate these quality issues.
		"""
		
		return await self.send_notification(
			recipients=[steward],
			title=title,
			message=message.strip(),
			notification_type="warning",
			metadata={
				"asset_name": asset_name,
				"quality_score": quality_score,
				"issue_count": issue_count,
				"critical_count": critical_count
			}
		)


class APGAIIntegration:
	"""Integration with APG AI Orchestration and Federated Learning"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.ai_service_url = config.get('apg_ai_service_url', 'http://localhost:8086/ai')
		self.ollama_url = config.get('ollama_url', 'http://localhost:11434')
		self.api_key = config.get('apg_api_key')
		self.session: Optional[aiohttp.ClientSession] = None
		self.local_models = config.get('local_models', {
			'classification': 'llama3.2:3b',
			'summarization': 'llama3.2:3b',
			'embedding': 'nomic-embed-text'
		})
	
	async def initialize(self):
		"""Initialize AI integration"""
		self.session = aiohttp.ClientSession(
			timeout=aiohttp.ClientTimeout(total=60),
			headers={
				'Content-Type': 'application/json',
				'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
			}
		)
	
	async def close(self):
		"""Close AI integration"""
		if self.session:
			await self.session.close()
	
	async def classify_data_content(self, 
								   content: str,
								   column_name: str = None) -> Dict[str, Any]:
		"""Use local Ollama model to classify data content"""
		try:
			# Prepare prompt for classification
			prompt = f"""
			Analyze the following data content and classify it for privacy and sensitivity:
			
			Column Name: {column_name or 'unknown'}
			Content Sample: {content[:500]}
			
			Classify as one of: PII, PHI, FINANCIAL, CONFIDENTIAL, INTERNAL, PUBLIC
			Also provide confidence score (0-1) and reasoning.
			
			Respond in JSON format:
			{{
				"classification": "classification_type",
				"confidence": 0.95,
				"reasoning": "explanation"
			}}
			"""
			
			# Call local Ollama
			async with self.session.post(
				f"{self.ollama_url}/api/generate",
				json={
					"model": self.local_models['classification'],
					"prompt": prompt,
					"stream": False
				}
			) as response:
				if response.status == 200:
					result = await response.json()
					response_text = result.get('response', '{}')
					
					try:
						return json.loads(response_text)
					except json.JSONDecodeError:
						# Fallback if JSON parsing fails
						return {
							"classification": "INTERNAL",
							"confidence": 0.5,
							"reasoning": "Failed to parse AI response"
						}
				else:
					return {
						"classification": "INTERNAL",
						"confidence": 0.0,
						"reasoning": "AI service unavailable"
					}
					
		except Exception as e:
			return {
				"classification": "INTERNAL",
				"confidence": 0.0,
				"reasoning": f"Classification failed: {str(e)}"
			}
	
	async def generate_business_description(self, 
										   schema_info: Dict[str, Any],
										   asset_name: str) -> str:
		"""Generate business-friendly description using AI"""
		try:
			prompt = f"""
			Generate a clear, business-friendly description for this data asset:
			
			Asset Name: {asset_name}
			Schema: {json.dumps(schema_info, indent=2)}
			
			Create a 2-3 sentence description that explains:
			1. What this data represents
			2. How it might be used in business context
			3. Key attributes or relationships
			
			Use clear, non-technical language that business users can understand.
			"""
			
			async with self.session.post(
				f"{self.ollama_url}/api/generate",
				json={
					"model": self.local_models['summarization'],
					"prompt": prompt,
					"stream": False
				}
			) as response:
				if response.status == 200:
					result = await response.json()
					return result.get('response', '').strip()
				else:
					return f"Data asset: {asset_name}"
					
		except Exception:
			return f"Data asset: {asset_name}"


class APGMetadataIntegrationManager:
	"""Main integration manager for all APG ecosystem services"""
	
	def __init__(self, config: Dict[str, Any], db_manager: MetaDatabaseManager):
		self.config = config
		self.db_manager = db_manager
		
		# Initialize all integration components
		self.event_publisher = APGEventPublisher(config)
		self.audit_logger = APGAuditLogger(config)
		self.auth_integration = APGAuthIntegration(config)
		self.mdm_integration = APGMDMIntegration(config, db_manager)
		self.notification_integration = APGNotificationIntegration(config)
		self.ai_integration = APGAIIntegration(config)
		
		self.initialized = False
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize all APG integrations"""
		if self.initialized:
			return {"status": "already_initialized"}
		
		try:
			# Start background services
			await self.event_publisher.start()
			await self.audit_logger.start()
			
			# Initialize synchronous integrations
			await self.auth_integration.initialize()
			await self.mdm_integration.initialize()
			await self.notification_integration.initialize()
			await self.ai_integration.initialize()
			
			self.initialized = True
			
			return {
				"status": "initialized",
				"integrations": {
					"event_publisher": "started",
					"audit_logger": "started",
					"auth_integration": "initialized",
					"mdm_integration": "initialized",
					"notification_integration": "initialized",
					"ai_integration": "initialized"
				}
			}
			
		except Exception as e:
			await self._log_error(f"Integration initialization failed: {str(e)}")
			raise
	
	async def shutdown(self):
		"""Shutdown all APG integrations"""
		if not self.initialized:
			return
		
		try:
			await self.event_publisher.stop()
			await self.audit_logger.stop()
			await self.auth_integration.close()
			await self.mdm_integration.close()
			await self.notification_integration.close()
			await self.ai_integration.close()
			
			self.initialized = False
			
		except Exception as e:
			await self._log_error(f"Integration shutdown failed: {str(e)}")
	
	async def publish_asset_event(self,
								 event_type: MetadataEventType,
								 asset_id: str,
								 tenant_id: str,
								 user_id: str = None,
								 payload: Dict[str, Any] = None) -> bool:
		"""Publish asset event to APG ecosystem"""
		event = MetadataEvent(
			event_type=event_type,
			tenant_id=tenant_id,
			asset_id=asset_id,
			user_id=user_id,
			payload=payload or {}
		)
		
		return await self.event_publisher.publish_event(event)
	
	async def log_operation(self,
						   operation: str,
						   asset_id: str,
						   user_id: str,
						   tenant_id: str,
						   details: Dict[str, Any] = None) -> bool:
		"""Log operation for audit compliance"""
		return await self.audit_logger.log_metadata_operation(
			operation=operation,
			asset_id=asset_id,
			user_id=user_id,
			tenant_id=tenant_id,
			details=details
		)
	
	async def check_user_permission(self,
								   user_id: str,
								   tenant_id: str,
								   resource: str,
								   action: str) -> bool:
		"""Check user permission through APG auth"""
		return await self.auth_integration.check_permission(
			user_id=user_id,
			tenant_id=tenant_id,
			resource=resource,
			action=action
		)
	
	async def sync_mdm_data(self, tenant_id: str) -> Dict[str, Any]:
		"""Synchronize MDM data as metadata assets"""
		return await self.mdm_integration.sync_mdm_entities(tenant_id)
	
	async def _log_error(self, message: str):
		"""Log error message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META INTEGRATION ERROR: {message}")


# Factory function for easy initialization
async def create_apg_integration_manager(
	config: Dict[str, Any],
	db_manager: MetaDatabaseManager
) -> APGMetadataIntegrationManager:
	"""Factory function to create and initialize APG integration manager"""
	integration_manager = APGMetadataIntegrationManager(config, db_manager)
	await integration_manager.initialize()
	return integration_manager