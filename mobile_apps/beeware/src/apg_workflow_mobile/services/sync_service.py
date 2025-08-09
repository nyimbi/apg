"""
Sync Service for APG Workflow Mobile

Handles real-time synchronization between mobile app and server.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any, List, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from ..models.api_response import APIResponse
from ..models.workflow import Workflow
from ..models.task import Task
from ..models.notification import Notification
from ..utils.constants import SYNC_INTERVAL, SYNC_BATCH_SIZE, MAX_SYNC_RETRIES
from ..utils.exceptions import SyncException, APIException, AuthenticationError


class SyncStatus(Enum):
	"""Sync operation status"""
	IDLE = "idle"
	SYNCING = "syncing"
	SUCCESS = "success"
	FAILED = "failed"
	CONFLICT = "conflict"


class ConflictResolution(Enum):
	"""Conflict resolution strategies"""
	SERVER_WINS = "server_wins"
	CLIENT_WINS = "client_wins"
	MERGE = "merge"
	ASK_USER = "ask_user"


@dataclass
class SyncOperation:
	"""Represents a sync operation"""
	id: str
	operation_type: str  # create, update, delete, fetch
	entity_type: str  # workflow, task, notification
	entity_id: str
	local_data: Optional[Dict[str, Any]] = None
	server_data: Optional[Dict[str, Any]] = None
	timestamp: Optional[datetime] = None
	retry_count: int = 0
	status: SyncStatus = SyncStatus.IDLE
	conflict_data: Optional[Dict[str, Any]] = None
	
	def __post_init__(self):
		if self.timestamp is None:
			self.timestamp = datetime.utcnow()
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'id': self.id,
			'operation_type': self.operation_type,
			'entity_type': self.entity_type,
			'entity_id': self.entity_id,
			'local_data': self.local_data,
			'server_data': self.server_data,
			'timestamp': self.timestamp.isoformat() if self.timestamp else None,
			'retry_count': self.retry_count,
			'status': self.status.value,
			'conflict_data': self.conflict_data
		}


@dataclass
class SyncStats:
	"""Sync statistics"""
	last_sync: Optional[datetime] = None
	successful_operations: int = 0
	failed_operations: int = 0
	conflicts_detected: int = 0
	conflicts_resolved: int = 0
	bytes_synced: int = 0
	sync_duration: float = 0.0
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'last_sync': self.last_sync.isoformat() if self.last_sync else None,
			'successful_operations': self.successful_operations,
			'failed_operations': self.failed_operations,
			'conflicts_detected': self.conflicts_detected,
			'conflicts_resolved': self.conflicts_resolved,
			'bytes_synced': self.bytes_synced,
			'sync_duration': self.sync_duration
		}


class SyncService:
	"""Service for real-time synchronization"""
	
	def __init__(self, app=None):
		self.app = app
		self.logger = logging.getLogger(__name__)
		
		# Sync state
		self.is_syncing = False
		self.auto_sync_enabled = True
		self.sync_interval = SYNC_INTERVAL
		self.sync_task: Optional[asyncio.Task] = None
		
		# Operations tracking
		self.pending_operations: List[SyncOperation] = []
		self.completed_operations: List[SyncOperation] = []
		self.failed_operations: List[SyncOperation] = []
		
		# Conflict resolution
		self.conflict_resolution_strategy = ConflictResolution.ASK_USER
		self.conflict_callbacks: Dict[str, Callable] = {}
		
		# Event callbacks
		self.sync_start_callbacks: List[Callable] = []
		self.sync_complete_callbacks: List[Callable] = []
		self.sync_error_callbacks: List[Callable] = []
		self.conflict_callbacks_list: List[Callable] = []
		
		# Statistics
		self.stats = SyncStats()
		
		# Entity change tracking
		self.entity_changes: Dict[str, datetime] = {}
		self.watched_entities: Set[str] = set()
		
		self.logger.info("Sync Service initialized")
	
	async def initialize(self):
		"""Initialize sync service"""
		try:
			# Load pending operations from offline service
			if self.app and hasattr(self.app, 'offline_service'):
				await self._load_pending_operations()
			
			# Start auto-sync if enabled
			if self.auto_sync_enabled:
				await self.start_auto_sync()
			
			self.logger.info("Sync service initialized")
			
		except asyncio.TimeoutError:
			self.logger.error("Sync service initialization timed out")
			raise SyncException("Sync service initialization timed out")
		except ConnectionError as e:
			self.logger.error(f"Network connection failed during sync service initialization: {e}")
			raise SyncException(f"Network connection failed: {e}")
		except AttributeError as e:
			self.logger.error(f"Missing required service dependency: {e}")
			raise SyncException(f"Missing required service dependency: {e}")
		except ImportError as e:
			self.logger.error(f"Failed to import required module for sync service: {e}")
			raise SyncException(f"Failed to import required module: {e}")
		except OSError as e:
			self.logger.error(f"File system error during sync service initialization: {e}")
			raise SyncException(f"File system error: {e}")
		except Exception as e:
			self.logger.error(f"Unexpected error during sync service initialization: {e}")
			raise SyncException(f"Unexpected initialization error: {e}")
	
	async def start_auto_sync(self):
		"""Start automatic synchronization"""
		if self.sync_task and not self.sync_task.done():
			return
		
		self.sync_task = asyncio.create_task(self._auto_sync_loop())
		self.logger.info("Auto-sync started")
	
	async def stop_auto_sync(self):
		"""Stop automatic synchronization"""
		if self.sync_task:
			self.sync_task.cancel()
			try:
				await self.sync_task
			except asyncio.CancelledError:
				pass
		
		self.logger.info("Auto-sync stopped")
	
	async def _auto_sync_loop(self):
		"""Automatic sync loop"""
		while True:
			try:
				await asyncio.sleep(self.sync_interval)
				
				if not self.is_syncing and self.should_sync():
					await self.sync_all()
					
			except asyncio.CancelledError:
				break
			except ConnectionError as e:
				self.logger.warning(f"Auto-sync network error (retrying): {e}")
				await self._notify_sync_error(f"Network error: {e}")
				await asyncio.sleep(30)  # Wait 30s before retry
			except TimeoutError as e:
				self.logger.warning(f"Auto-sync timeout (retrying): {e}")
				await self._notify_sync_error(f"Sync timeout: {e}")
				await asyncio.sleep(60)  # Wait 1min before retry
			except PermissionError as e:
				self.logger.error(f"Auto-sync permission denied: {e}")
				await self._notify_sync_error(f"Permission denied: {e}")
				await asyncio.sleep(300)  # Wait 5min before retry
			except OSError as e:
				self.logger.error(f"Auto-sync file system error: {e}")
				await self._notify_sync_error(f"File system error: {e}")
				await asyncio.sleep(120)  # Wait 2min before retry
			except Exception as e:
				self.logger.error(f"Unexpected auto-sync error: {e}")
				await self._notify_sync_error(f"Unexpected error: {e}")
				await asyncio.sleep(60)  # Wait 1min before retry
	
	def should_sync(self) -> bool:
		"""Determine if sync should be performed"""
		# Check if there are pending operations
		if self.pending_operations:
			return True
		
		# Check if enough time has passed since last sync
		if self.stats.last_sync:
			time_since_sync = datetime.utcnow() - self.stats.last_sync
			if time_since_sync < timedelta(minutes=5):  # Minimum 5 minutes between syncs
				return False
		
		# Check if there are entity changes
		if self.entity_changes:
			return True
		
		return False
	
	async def sync_all(self) -> Dict[str, Any]:
		"""Perform complete synchronization"""
		if self.is_syncing:
			return {"status": "already_syncing"}
		
		start_time = datetime.utcnow()
		
		try:
			self.is_syncing = True
			await self._notify_sync_start()
			
			# Reset stats for this sync
			sync_stats = SyncStats()
			
			# Check network connectivity
			if not await self._check_connectivity():
				raise SyncException("No network connectivity")
			
			# Sync workflows
			workflow_result = await self._sync_workflows()
			sync_stats.successful_operations += workflow_result.get('successful', 0)
			sync_stats.failed_operations += workflow_result.get('failed', 0)
			sync_stats.conflicts_detected += workflow_result.get('conflicts', 0)
			
			# Sync tasks
			task_result = await self._sync_tasks()
			sync_stats.successful_operations += task_result.get('successful', 0)
			sync_stats.failed_operations += task_result.get('failed', 0)
			sync_stats.conflicts_detected += task_result.get('conflicts', 0)
			
			# Sync notifications
			notification_result = await self._sync_notifications()
			sync_stats.successful_operations += notification_result.get('successful', 0)
			sync_stats.failed_operations += notification_result.get('failed', 0)
			sync_stats.conflicts_detected += notification_result.get('conflicts', 0)
			
			# Process pending operations
			operations_result = await self._process_pending_operations()
			sync_stats.successful_operations += operations_result.get('successful', 0)
			sync_stats.failed_operations += operations_result.get('failed', 0)
			
			# Update global stats
			sync_stats.last_sync = datetime.utcnow()
			sync_stats.sync_duration = (sync_stats.last_sync - start_time).total_seconds()
			
			self.stats = sync_stats
			
			result = {
				"status": "completed",
				"stats": sync_stats.to_dict(),
				"workflows": workflow_result,
				"tasks": task_result,
				"notifications": notification_result,
				"operations": operations_result
			}
			
			await self._notify_sync_complete(result)
			return result
			
		except ConnectionError as e:
			error_msg = f"Sync failed due to network error: {e}"
			self.logger.error(error_msg)
			await self._notify_sync_error(error_msg)
			return {
				"status": "failed",
				"error": error_msg,
				"error_type": "network_error",
				"retry_recommended": True,
				"stats": self.stats.to_dict()
			}
		except TimeoutError as e:
			error_msg = f"Sync failed due to timeout: {e}"
			self.logger.error(error_msg)
			await self._notify_sync_error(error_msg)
			return {
				"status": "failed",
				"error": error_msg,
				"error_type": "timeout",
				"retry_recommended": True,
				"stats": self.stats.to_dict()
			}
		except AuthenticationError as e:
			error_msg = f"Sync failed due to authentication error: {e}"
			self.logger.error(error_msg)
			await self._notify_sync_error(error_msg)
			return {
				"status": "failed",
				"error": error_msg,
				"error_type": "authentication_error",
				"retry_recommended": False,
				"stats": self.stats.to_dict()
			}
		except PermissionError as e:
			error_msg = f"Sync failed due to permission error: {e}"
			self.logger.error(error_msg)
			await self._notify_sync_error(error_msg)
			return {
				"status": "failed",
				"error": error_msg,
				"error_type": "permission_error",
				"retry_recommended": False,
				"stats": self.stats.to_dict()
			}
		except OSError as e:
			error_msg = f"Sync failed due to file system error: {e}"
			self.logger.error(error_msg)
			await self._notify_sync_error(error_msg)
			return {
				"status": "failed",
				"error": error_msg,
				"error_type": "filesystem_error",
				"retry_recommended": True,
				"stats": self.stats.to_dict()
			}
		except ValueError as e:
			error_msg = f"Sync failed due to invalid data: {e}"
			self.logger.error(error_msg)
			await self._notify_sync_error(error_msg)
			return {
				"status": "failed",
				"error": error_msg,
				"error_type": "data_validation_error",
				"retry_recommended": False,
				"stats": self.stats.to_dict()
			}
		except Exception as e:
			error_msg = f"Sync failed due to unexpected error: {e}"
			self.logger.error(error_msg, exc_info=True)
			await self._notify_sync_error(error_msg)
			return {
				"status": "failed",
				"error": error_msg,
				"error_type": "unexpected_error",
				"retry_recommended": True,
				"stats": self.stats.to_dict()
			}
		
		finally:
			self.is_syncing = False
	
	async def _check_connectivity(self) -> bool:
		"""Check network connectivity"""
		try:
			if self.app and hasattr(self.app, 'api_service'):
				return await self.app.api_service.health_check()
			return False
		except Exception:
			return False
	
	async def _sync_workflows(self) -> Dict[str, int]:
		"""Sync workflows with server"""
		result = {'successful': 0, 'failed': 0, 'conflicts': 0}
		
		try:
			if not self.app or not hasattr(self.app, 'api_service'):
				return result
			
			api_service = self.app.api_service
			
			# Get server workflows
			server_response = await api_service.get('/workflows', params={'modified_since': self._get_last_sync_timestamp()})
			
			if not server_response.success:
				self.logger.error(f"Failed to fetch workflows from server: {server_response.message}")
				result['failed'] += 1
				return result
			
			server_workflows = server_response.data.get('workflows', [])
			
			# Get local workflows from offline service
			local_workflows = []
			if hasattr(self.app, 'offline_service'):
				# This would typically get workflows from local cache
				pass
			
			# Sync each workflow
			for server_workflow_data in server_workflows:
				try:
					workflow_id = server_workflow_data['id']
					
					# Check for local version
					local_workflow = await self._get_local_workflow(workflow_id)
					
					if local_workflow:
						# Check for conflicts
						if await self._has_workflow_conflict(local_workflow, server_workflow_data):
							await self._handle_workflow_conflict(local_workflow, server_workflow_data)
							result['conflicts'] += 1
						else:
							# Update local workflow
							await self._update_local_workflow(server_workflow_data)
							result['successful'] += 1
					else:
						# Create new local workflow
						await self._create_local_workflow(server_workflow_data)
						result['successful'] += 1
						
				except KeyError as e:
					self.logger.error(f"Missing required field in workflow {server_workflow_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
				except ValueError as e:
					self.logger.error(f"Invalid workflow data for {server_workflow_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
				except TypeError as e:
					self.logger.error(f"Type error in workflow {server_workflow_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
				except Exception as e:
					self.logger.error(f"Unexpected error syncing workflow {server_workflow_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
			
			return result
			
		except ConnectionError as e:
			self.logger.error(f"Network error in workflow sync: {e}")
			result['failed'] += 1
			return result
		except TimeoutError as e:
			self.logger.error(f"Timeout in workflow sync: {e}")
			result['failed'] += 1
			return result
		except AuthenticationError as e:
			self.logger.error(f"Authentication error in workflow sync: {e}")
			result['failed'] += 1
			return result
		except Exception as e:
			self.logger.error(f"Unexpected error in workflow sync: {e}")
			result['failed'] += 1
			return result
	
	async def _sync_tasks(self) -> Dict[str, int]:
		"""Sync tasks with server"""
		result = {'successful': 0, 'failed': 0, 'conflicts': 0}
		
		try:
			if not self.app or not hasattr(self.app, 'api_service'):
				return result
			
			api_service = self.app.api_service
			
			# Get server tasks
			server_response = await api_service.get('/tasks', params={'modified_since': self._get_last_sync_timestamp()})
			
			if not server_response.success:
				self.logger.error(f"Failed to fetch tasks from server: {server_response.message}")
				result['failed'] += 1
				return result
			
			server_tasks = server_response.data.get('tasks', [])
			
			# Sync each task
			for server_task_data in server_tasks:
				try:
					task_id = server_task_data['id']
					
					# Check for local version
					local_task = await self._get_local_task(task_id)
					
					if local_task:
						# Check for conflicts
						if await self._has_task_conflict(local_task, server_task_data):
							await self._handle_task_conflict(local_task, server_task_data)
							result['conflicts'] += 1
						else:
							# Update local task
							await self._update_local_task(server_task_data)
							result['successful'] += 1
					else:
						# Create new local task
						await self._create_local_task(server_task_data)
						result['successful'] += 1
						
				except KeyError as e:
					self.logger.error(f"Missing required field in task {server_task_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
				except ValueError as e:
					self.logger.error(f"Invalid task data for {server_task_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
				except TypeError as e:
					self.logger.error(f"Type error in task {server_task_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
				except Exception as e:
					self.logger.error(f"Unexpected error syncing task {server_task_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
			
			return result
			
		except ConnectionError as e:
			self.logger.error(f"Network error in task sync: {e}")
			result['failed'] += 1
			return result
		except TimeoutError as e:
			self.logger.error(f"Timeout in task sync: {e}")
			result['failed'] += 1
			return result
		except AuthenticationError as e:
			self.logger.error(f"Authentication error in task sync: {e}")
			result['failed'] += 1
			return result
		except Exception as e:
			self.logger.error(f"Unexpected error in task sync: {e}")
			result['failed'] += 1
			return result
	
	async def _sync_notifications(self) -> Dict[str, int]:
		"""Sync notifications with server"""
		result = {'successful': 0, 'failed': 0, 'conflicts': 0}
		
		try:
			if not self.app or not hasattr(self.app, 'api_service'):
				return result
			
			api_service = self.app.api_service
			
			# Get server notifications
			server_response = await api_service.get('/notifications', params={'modified_since': self._get_last_sync_timestamp()})
			
			if not server_response.success:
				self.logger.error(f"Failed to fetch notifications from server: {server_response.message}")
				result['failed'] += 1
				return result
			
			server_notifications = server_response.data.get('notifications', [])
			
			# Sync each notification
			for server_notification_data in server_notifications:
				try:
					notification_id = server_notification_data['id']
					
					# Check for local version
					local_notification = await self._get_local_notification(notification_id)
					
					if local_notification:
						# Update local notification
						await self._update_local_notification(server_notification_data)
						result['successful'] += 1
					else:
						# Create new local notification
						await self._create_local_notification(server_notification_data)
						result['successful'] += 1
						
				except KeyError as e:
					self.logger.error(f"Missing required field in notification {server_notification_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
				except ValueError as e:
					self.logger.error(f"Invalid notification data for {server_notification_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
				except TypeError as e:
					self.logger.error(f"Type error in notification {server_notification_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
				except Exception as e:
					self.logger.error(f"Unexpected error syncing notification {server_notification_data.get('id', 'unknown')}: {e}")
					result['failed'] += 1
			
			return result
			
		except ConnectionError as e:
			self.logger.error(f"Network error in notification sync: {e}")
			result['failed'] += 1
			return result
		except TimeoutError as e:
			self.logger.error(f"Timeout in notification sync: {e}")
			result['failed'] += 1
			return result
		except AuthenticationError as e:
			self.logger.error(f"Authentication error in notification sync: {e}")
			result['failed'] += 1
			return result
		except Exception as e:
			self.logger.error(f"Unexpected error in notification sync: {e}")
			result['failed'] += 1
			return result
	
	async def _process_pending_operations(self) -> Dict[str, int]:
		"""Process pending offline operations"""
		result = {'successful': 0, 'failed': 0}
		
		if not self.pending_operations:
			return result
		
		try:
			# Process operations in batches
			for i in range(0, len(self.pending_operations), SYNC_BATCH_SIZE):
				batch = self.pending_operations[i:i + SYNC_BATCH_SIZE]
				
				for operation in batch:
					try:
						success = await self._process_sync_operation(operation)
						if success:
							result['successful'] += 1
							self.completed_operations.append(operation)
						else:
							result['failed'] += 1
							operation.retry_count += 1
							
							if operation.retry_count >= MAX_SYNC_RETRIES:
								self.failed_operations.append(operation)
							
					except ConnectionError as e:
						self.logger.error(f"Network error processing operation {operation.id}: {e}")
						result['failed'] += 1
						operation.retry_count += 1
					except TimeoutError as e:
						self.logger.error(f"Timeout processing operation {operation.id}: {e}")
						result['failed'] += 1
						operation.retry_count += 1
					except AuthenticationError as e:
						self.logger.error(f"Authentication error processing operation {operation.id}: {e}")
						result['failed'] += 1
						# Don't retry authentication errors
						operation.retry_count = MAX_SYNC_RETRIES
					except ValueError as e:
						self.logger.error(f"Invalid data in operation {operation.id}: {e}")
						result['failed'] += 1
						# Don't retry data validation errors
						operation.retry_count = MAX_SYNC_RETRIES
					except Exception as e:
						self.logger.error(f"Unexpected error processing operation {operation.id}: {e}")
						result['failed'] += 1
						operation.retry_count += 1
			
			# Remove completed operations
			self.pending_operations = [
				op for op in self.pending_operations 
				if op not in self.completed_operations and op not in self.failed_operations
			]
			
			return result
			
		except OSError as e:
			self.logger.error(f"File system error processing pending operations: {e}")
			return result
		except MemoryError as e:
			self.logger.error(f"Memory error processing pending operations: {e}")
			return result
		except Exception as e:
			self.logger.error(f"Unexpected error processing pending operations: {e}")
			return result
	
	async def _process_sync_operation(self, operation: SyncOperation) -> bool:
		"""Process individual sync operation"""
		try:
			if not self.app or not hasattr(self.app, 'api_service'):
				return False
			
			api_service = self.app.api_service
			
			if operation.entity_type == "workflow":
				return await self._sync_workflow_operation(operation, api_service)
			elif operation.entity_type == "task":
				return await self._sync_task_operation(operation, api_service)
			elif operation.entity_type == "notification":
				return await self._sync_notification_operation(operation, api_service)
			
			return False
			
		except ConnectionError as e:
			self.logger.error(f"Network error in sync operation: {e}")
			return False
		except TimeoutError as e:
			self.logger.error(f"Timeout in sync operation: {e}")
			return False
		except AuthenticationError as e:
			self.logger.error(f"Authentication error in sync operation: {e}")
			return False
		except Exception as e:
			self.logger.error(f"Unexpected error in sync operation: {e}")
			return False
	
	async def _sync_workflow_operation(self, operation: SyncOperation, api_service) -> bool:
		"""Sync workflow operation with server"""
		try:
			if operation.operation_type == "create":
				response = await api_service.post('/workflows', operation.local_data)
			elif operation.operation_type == "update":
				response = await api_service.put(f'/workflows/{operation.entity_id}', operation.local_data)
			elif operation.operation_type == "delete":
				response = await api_service.delete(f'/workflows/{operation.entity_id}')
			else:
				return False
			
			operation.status = SyncStatus.SUCCESS if response.success else SyncStatus.FAILED
			return response.success
			
		except ConnectionError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Network error in workflow operation: {e}")
			return False
		except TimeoutError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Timeout in workflow operation: {e}")
			return False
		except AuthenticationError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Authentication error in workflow operation: {e}")
			return False
		except ValueError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Invalid data in workflow operation: {e}")
			return False
		except Exception as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Unexpected error in workflow operation: {e}")
			return False
	
	async def _sync_task_operation(self, operation: SyncOperation, api_service) -> bool:
		"""Sync task operation with server"""
		try:
			if operation.operation_type == "create":
				response = await api_service.post('/tasks', operation.local_data)
			elif operation.operation_type == "update":
				response = await api_service.put(f'/tasks/{operation.entity_id}', operation.local_data)
			elif operation.operation_type == "delete":
				response = await api_service.delete(f'/tasks/{operation.entity_id}')
			else:
				return False
			
			operation.status = SyncStatus.SUCCESS if response.success else SyncStatus.FAILED
			return response.success
			
		except ConnectionError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Network error in task operation: {e}")
			return False
		except TimeoutError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Timeout in task operation: {e}")
			return False
		except AuthenticationError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Authentication error in task operation: {e}")
			return False
		except ValueError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Invalid data in task operation: {e}")
			return False
		except Exception as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Unexpected error in task operation: {e}")
			return False
	
	async def _sync_notification_operation(self, operation: SyncOperation, api_service) -> bool:
		"""Sync notification operation with server"""
		try:
			if operation.operation_type == "mark_read":
				response = await api_service.post(f'/notifications/{operation.entity_id}/read')
			elif operation.operation_type == "delete":
				response = await api_service.delete(f'/notifications/{operation.entity_id}')
			else:
				return False
			
			operation.status = SyncStatus.SUCCESS if response.success else SyncStatus.FAILED
			return response.success
			
		except ConnectionError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Network error in notification operation: {e}")
			return False
		except TimeoutError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Timeout in notification operation: {e}")
			return False
		except AuthenticationError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Authentication error in notification operation: {e}")
			return False
		except ValueError as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Invalid data in notification operation: {e}")
			return False
		except Exception as e:
			operation.status = SyncStatus.FAILED
			self.logger.error(f"Unexpected error in notification operation: {e}")
			return False
	
	async def add_sync_operation(self, operation_type: str, entity_type: str, entity_id: str, data: Optional[Dict[str, Any]] = None):
		"""Add operation to sync queue"""
		operation = SyncOperation(
			id=str(uuid.uuid4()),
			operation_type=operation_type,
			entity_type=entity_type,
			entity_id=entity_id,
			local_data=data,
			timestamp=datetime.utcnow()
		)
		
		self.pending_operations.append(operation)
		self.logger.debug(f"Added sync operation: {operation_type} {entity_type} {entity_id}")
	
	def add_sync_start_callback(self, callback: Callable):
		"""Add callback for sync start event"""
		self.sync_start_callbacks.append(callback)
	
	def add_sync_complete_callback(self, callback: Callable):
		"""Add callback for sync complete event"""
		self.sync_complete_callbacks.append(callback)
	
	def add_sync_error_callback(self, callback: Callable):
		"""Add callback for sync error event"""
		self.sync_error_callbacks.append(callback)
	
	def add_conflict_callback(self, callback: Callable):
		"""Add callback for conflict detection"""
		self.conflict_callbacks_list.append(callback)
	
	async def _notify_sync_start(self):
		"""Notify sync start callbacks"""
		for callback in self.sync_start_callbacks:
			try:
				if asyncio.iscoroutinefunction(callback):
					await callback()
				else:
					callback()
			except Exception as e:
				self.logger.warning(f"Sync start callback failed: {e}")
	
	async def _notify_sync_complete(self, result: Dict[str, Any]):
		"""Notify sync complete callbacks"""
		for callback in self.sync_complete_callbacks:
			try:
				if asyncio.iscoroutinefunction(callback):
					await callback(result)
				else:
					callback(result)
			except Exception as e:
				self.logger.warning(f"Sync complete callback failed: {e}")
	
	async def _notify_sync_error(self, error: str):
		"""Notify sync error callbacks"""
		for callback in self.sync_error_callbacks:
			try:
				if asyncio.iscoroutinefunction(callback):
					await callback(error)
				else:
					callback(error)
			except Exception as e:
				self.logger.warning(f"Sync error callback failed: {e}")
	
	def _get_last_sync_timestamp(self) -> Optional[str]:
		"""Get last sync timestamp as ISO string"""
		if self.stats.last_sync:
			return self.stats.last_sync.isoformat()
		return None
	
	async def _load_pending_operations(self):
		"""Load pending operations from offline service"""
		try:
			if hasattr(self.app, 'offline_service'):
				offline_service = self.app.offline_service
				# This would load operations from offline storage
				pass
		except OSError as e:
			self.logger.warning(f"File system error loading pending operations: {e}")
		except AttributeError as e:
			self.logger.warning(f"Missing offline service for loading operations: {e}")
		except Exception as e:
			self.logger.warning(f"Unexpected error loading pending operations: {e}")
	
	# Entity-specific helper methods (simplified implementations)
	async def _get_local_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
		"""Get local workflow data"""
		if hasattr(self.app, 'offline_service'):
			workflow = await self.app.offline_service.get_workflow(workflow_id)
			return workflow.to_dict() if workflow else None
		return None
	
	async def _get_local_task(self, task_id: str) -> Optional[Dict[str, Any]]:
		"""Get local task data"""
		if hasattr(self.app, 'offline_service'):
			task = await self.app.offline_service.get_task(task_id)
			return task.to_dict() if task else None
		return None
	
	async def _get_local_notification(self, notification_id: str) -> Optional[Dict[str, Any]]:
		"""Get local notification data"""
		if hasattr(self.app, 'offline_service'):
			notification = await self.app.offline_service.get_notification(notification_id)
			return notification.to_dict() if notification else None
		return None
	
	async def _has_workflow_conflict(self, local_data: Dict[str, Any], server_data: Dict[str, Any]) -> bool:
		"""Check if workflow has conflicts"""
		# Simple conflict detection based on modified timestamps
		local_modified = local_data.get('modified_at')
		server_modified = server_data.get('modified_at')
		
		if not local_modified or not server_modified:
			return False
		
		return local_modified != server_modified
	
	async def _has_task_conflict(self, local_data: Dict[str, Any], server_data: Dict[str, Any]) -> bool:
		"""Check if task has conflicts"""
		# Simple conflict detection based on modified timestamps
		local_modified = local_data.get('modified_at')
		server_modified = server_data.get('modified_at')
		
		if not local_modified or not server_modified:
			return False
		
		return local_modified != server_modified
	
	async def _handle_workflow_conflict(self, local_data: Dict[str, Any], server_data: Dict[str, Any]):
		"""Handle workflow conflict based on resolution strategy"""
		if self.conflict_resolution_strategy == ConflictResolution.SERVER_WINS:
			await self._update_local_workflow(server_data)
		elif self.conflict_resolution_strategy == ConflictResolution.CLIENT_WINS:
			# Push local changes to server
			pass
		elif self.conflict_resolution_strategy == ConflictResolution.ASK_USER:
			# Notify conflict callbacks
			for callback in self.conflict_callbacks_list:
				try:
					if asyncio.iscoroutinefunction(callback):
						await callback("workflow", local_data, server_data)
					else:
						callback("workflow", local_data, server_data)
				except TypeError as e:
					self.logger.warning(f"Conflict callback type error: {e}")
				except AttributeError as e:
					self.logger.warning(f"Conflict callback attribute error: {e}")
				except Exception as e:
					self.logger.warning(f"Conflict callback failed: {e}")
	
	async def _handle_task_conflict(self, local_data: Dict[str, Any], server_data: Dict[str, Any]):
		"""Handle task conflict based on resolution strategy"""
		if self.conflict_resolution_strategy == ConflictResolution.SERVER_WINS:
			await self._update_local_task(server_data)
		elif self.conflict_resolution_strategy == ConflictResolution.CLIENT_WINS:
			# Push local changes to server
			pass
		elif self.conflict_resolution_strategy == ConflictResolution.ASK_USER:
			# Notify conflict callbacks
			for callback in self.conflict_callbacks_list:
				try:
					if asyncio.iscoroutinefunction(callback):
						await callback("task", local_data, server_data)
					else:
						callback("task", local_data, server_data)
				except TypeError as e:
					self.logger.warning(f"Conflict callback type error: {e}")
				except AttributeError as e:
					self.logger.warning(f"Conflict callback attribute error: {e}")
				except Exception as e:
					self.logger.warning(f"Conflict callback failed: {e}")
	
	async def _update_local_workflow(self, server_data: Dict[str, Any]):
		"""Update local workflow with server data"""
		if hasattr(self.app, 'offline_service'):
			workflow = Workflow(**server_data)
			await self.app.offline_service.store_workflow(workflow, 'synced')
	
	async def _update_local_task(self, server_data: Dict[str, Any]):
		"""Update local task with server data"""
		if hasattr(self.app, 'offline_service'):
			task = Task(**server_data)
			await self.app.offline_service.store_task(task, 'synced')
	
	async def _update_local_notification(self, server_data: Dict[str, Any]):
		"""Update local notification with server data"""
		if hasattr(self.app, 'offline_service'):
			notification = Notification(**server_data)
			await self.app.offline_service.store_notification(notification, 'synced')
	
	async def _create_local_workflow(self, server_data: Dict[str, Any]):
		"""Create local workflow from server data"""
		await self._update_local_workflow(server_data)
	
	async def _create_local_task(self, server_data: Dict[str, Any]):
		"""Create local task from server data"""
		await self._update_local_task(server_data)
	
	async def _create_local_notification(self, server_data: Dict[str, Any]):
		"""Create local notification from server data"""
		await self._update_local_notification(server_data)
	
	def get_sync_stats(self) -> Dict[str, Any]:
		"""Get current sync statistics"""
		return self.stats.to_dict()
	
	def get_pending_operations_count(self) -> int:
		"""Get count of pending operations"""
		return len(self.pending_operations)
	
	def set_conflict_resolution_strategy(self, strategy: ConflictResolution):
		"""Set conflict resolution strategy"""
		self.conflict_resolution_strategy = strategy
		self.logger.info(f"Conflict resolution strategy set to: {strategy.value}")
	
	async def force_sync(self) -> Dict[str, Any]:
		"""Force immediate synchronization"""
		self.logger.info("Force sync initiated")
		return await self.sync_all()
	
	async def clear_sync_data(self):
		"""Clear all sync data and reset state"""
		self.pending_operations.clear()
		self.completed_operations.clear()
		self.failed_operations.clear()
		self.entity_changes.clear()
		self.stats = SyncStats()
		
		self.logger.info("Sync data cleared")
	
	async def shutdown(self):
		"""Shutdown sync service"""
		await self.stop_auto_sync()
		
		# Save any pending operations
		if hasattr(self.app, 'offline_service'):
			for operation in self.pending_operations:
				await self.app.offline_service.add_offline_operation(
					operation.operation_type,
					operation.entity_type,
					operation.entity_id,
					operation.local_data or {}
				)
		
		self.logger.info("Sync service shutdown complete")