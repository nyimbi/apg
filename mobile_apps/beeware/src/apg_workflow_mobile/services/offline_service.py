"""
Offline Service for APG Workflow Mobile

Handles offline data storage and synchronization.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
import sqlite3
import json
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from dataclasses import dataclass, asdict

from ..models.workflow import Workflow, WorkflowInstance
from ..models.task import Task
from ..models.notification import Notification
from ..models.api_response import APIResponse
from ..utils.constants import OFFLINE_DB_PATH, SYNC_BATCH_SIZE
from ..utils.exceptions import OfflineException, APIException


@dataclass
class OfflineOperation:
	"""Represents an offline operation to be synced"""
	id: str
	operation_type: str  # create, update, delete
	entity_type: str  # workflow, task, notification, etc.
	entity_id: str
	data: Dict[str, Any]
	timestamp: datetime
	retry_count: int = 0
	last_retry: Optional[datetime] = None
	error_message: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'id': self.id,
			'operation_type': self.operation_type,
			'entity_type': self.entity_type,
			'entity_id': self.entity_id,
			'data': json.dumps(self.data),
			'timestamp': self.timestamp.isoformat(),
			'retry_count': self.retry_count,
			'last_retry': self.last_retry.isoformat() if self.last_retry else None,
			'error_message': self.error_message
		}
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> "OfflineOperation":
		return cls(
			id=data['id'],
			operation_type=data['operation_type'],
			entity_type=data['entity_type'],
			entity_id=data['entity_id'],
			data=json.loads(data['data']),
			timestamp=datetime.fromisoformat(data['timestamp']),
			retry_count=data['retry_count'],
			last_retry=datetime.fromisoformat(data['last_retry']) if data['last_retry'] else None,
			error_message=data['error_message']
		)


class OfflineService:
	"""Service for offline data management and synchronization"""
	
	def __init__(self, app=None):
		self.app = app
		self.logger = logging.getLogger(__name__)
		
		# Database connection
		self.db_path = OFFLINE_DB_PATH
		self.db_connection: Optional[sqlite3.Connection] = None
		
		# Sync state
		self.is_syncing = False
		self.last_sync: Optional[datetime] = None
		self.pending_operations: List[OfflineOperation] = []
		
		self.logger.info("Offline Service initialized")
	
	async def initialize(self):
		"""Initialize offline service and database"""
		try:
			await self._init_database()
			await self._load_pending_operations()
			
			self.logger.info("Offline service initialized")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize offline service: {e}")
			raise OfflineException(f"Failed to initialize offline service: {e}")
	
	async def _init_database(self):
		"""Initialize SQLite database for offline storage"""
		try:
			# Ensure directory exists
			self.db_path.parent.mkdir(parents=True, exist_ok=True)
			
			# Connect to database
			self.db_connection = sqlite3.connect(
				str(self.db_path),
				check_same_thread=False
			)
			self.db_connection.row_factory = sqlite3.Row
			
			# Create tables
			await self._create_tables()
			
			self.logger.info("Offline database initialized")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize database: {e}")
			raise
	
	async def _create_tables(self):
		"""Create database tables for offline storage"""
		try:
			cursor = self.db_connection.cursor()
			
			# Offline operations table
			cursor.execute('''
				CREATE TABLE IF NOT EXISTS offline_operations (
					id TEXT PRIMARY KEY,
					operation_type TEXT NOT NULL,
					entity_type TEXT NOT NULL,
					entity_id TEXT NOT NULL,
					data TEXT NOT NULL,
					timestamp TEXT NOT NULL,
					retry_count INTEGER DEFAULT 0,
					last_retry TEXT,
					error_message TEXT,
					created_at TEXT DEFAULT CURRENT_TIMESTAMP
				)
			''')
			
			# Workflows cache table
			cursor.execute('''
				CREATE TABLE IF NOT EXISTS workflows_cache (
					id TEXT PRIMARY KEY,
					data TEXT NOT NULL,
					last_modified TEXT NOT NULL,
					sync_status TEXT DEFAULT 'synced',
					created_at TEXT DEFAULT CURRENT_TIMESTAMP
				)
			''')
			
			# Tasks cache table
			cursor.execute('''
				CREATE TABLE IF NOT EXISTS tasks_cache (
					id TEXT PRIMARY KEY,
					data TEXT NOT NULL,
					last_modified TEXT NOT NULL,
					sync_status TEXT DEFAULT 'synced',
					created_at TEXT DEFAULT CURRENT_TIMESTAMP
				)
			''')
			
			# Notifications cache table
			cursor.execute('''
				CREATE TABLE IF NOT EXISTS notifications_cache (
					id TEXT PRIMARY KEY,
					data TEXT NOT NULL,
					last_modified TEXT NOT NULL,
					sync_status TEXT DEFAULT 'synced',
					created_at TEXT DEFAULT CURRENT_TIMESTAMP
				)
			''')
			
			# User data cache table
			cursor.execute('''
				CREATE TABLE IF NOT EXISTS user_data_cache (
					key TEXT PRIMARY KEY,
					data TEXT NOT NULL,
					last_modified TEXT NOT NULL,
					expires_at TEXT,
					created_at TEXT DEFAULT CURRENT_TIMESTAMP
				)
			''')
			
			# File cache table
			cursor.execute('''
				CREATE TABLE IF NOT EXISTS file_cache (
					id TEXT PRIMARY KEY,
					filename TEXT NOT NULL,
					file_path TEXT NOT NULL,
					mime_type TEXT,
					file_size INTEGER,
					checksum TEXT,
					last_accessed TEXT NOT NULL,
					expires_at TEXT,
					created_at TEXT DEFAULT CURRENT_TIMESTAMP
				)
			''')
			
			# Create indexes for performance
			cursor.execute('CREATE INDEX IF NOT EXISTS idx_operations_entity ON offline_operations(entity_type, entity_id)')
			cursor.execute('CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON offline_operations(timestamp)')
			cursor.execute('CREATE INDEX IF NOT EXISTS idx_workflows_sync ON workflows_cache(sync_status)')
			cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_sync ON tasks_cache(sync_status)')
			cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_sync ON notifications_cache(sync_status)')
			
			self.db_connection.commit()
			
			self.logger.info("Database tables created successfully")
			
		except Exception as e:
			self.logger.error(f"Failed to create database tables: {e}")
			raise
	
	async def store_workflow(self, workflow: Workflow, sync_status: str = 'pending') -> bool:
		"""Store workflow in offline cache"""
		try:
			cursor = self.db_connection.cursor()
			
			workflow_data = json.dumps(workflow.to_dict())
			timestamp = datetime.utcnow().isoformat()
			
			cursor.execute('''
				INSERT OR REPLACE INTO workflows_cache 
				(id, data, last_modified, sync_status)
				VALUES (?, ?, ?, ?)
			''', (workflow.id, workflow_data, timestamp, sync_status))
			
			self.db_connection.commit()
			
			self.logger.debug(f"Stored workflow in offline cache: {workflow.id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to store workflow offline: {e}")
			return False
	
	async def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
		"""Get workflow from offline cache"""
		try:
			cursor = self.db_connection.cursor()
			
			cursor.execute(
				'SELECT data FROM workflows_cache WHERE id = ?',
				(workflow_id,)
			)
			
			row = cursor.fetchone()
			if row:
				workflow_data = json.loads(row['data'])
				return Workflow(**workflow_data)
			
			return None
			
		except Exception as e:
			self.logger.error(f"Failed to get workflow from offline cache: {e}")
			return None
	
	async def store_task(self, task: Task, sync_status: str = 'pending') -> bool:
		"""Store task in offline cache"""
		try:
			cursor = self.db_connection.cursor()
			
			task_data = json.dumps(task.to_dict())
			timestamp = datetime.utcnow().isoformat()
			
			cursor.execute('''
				INSERT OR REPLACE INTO tasks_cache 
				(id, data, last_modified, sync_status)
				VALUES (?, ?, ?, ?)
			''', (task.id, task_data, timestamp, sync_status))
			
			self.db_connection.commit()
			
			self.logger.debug(f"Stored task in offline cache: {task.id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to store task offline: {e}")
			return False
	
	async def get_task(self, task_id: str) -> Optional[Task]:
		"""Get task from offline cache"""
		try:
			cursor = self.db_connection.cursor()
			
			cursor.execute(
				'SELECT data FROM tasks_cache WHERE id = ?',
				(task_id,)
			)
			
			row = cursor.fetchone()
			if row:
				task_data = json.loads(row['data'])
				return Task(**task_data)
			
			return None
			
		except Exception as e:
			self.logger.error(f"Failed to get task from offline cache: {e}")
			return None
	
	async def store_notification(self, notification: Notification, sync_status: str = 'pending') -> bool:
		"""Store notification in offline cache"""
		try:
			cursor = self.db_connection.cursor()
			
			notification_data = json.dumps(notification.to_dict())
			timestamp = datetime.utcnow().isoformat()
			
			cursor.execute('''
				INSERT OR REPLACE INTO notifications_cache 
				(id, data, last_modified, sync_status)
				VALUES (?, ?, ?, ?)
			''', (notification.id, notification_data, timestamp, sync_status))
			
			self.db_connection.commit()
			
			self.logger.debug(f"Stored notification in offline cache: {notification.id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to store notification offline: {e}")
			return False
	
	async def get_notification(self, notification_id: str) -> Optional[Notification]:
		"""Get notification from offline cache"""
		try:
			cursor = self.db_connection.cursor()
			
			cursor.execute(
				'SELECT data FROM notifications_cache WHERE id = ?',
				(notification_id,)
			)
			
			row = cursor.fetchone()
			if row:
				notification_data = json.loads(row['data'])
				return Notification(**notification_data)
			
			return None
			
		except Exception as e:
			self.logger.error(f"Failed to get notification from offline cache: {e}")
			return None
	
	async def add_offline_operation(self, operation_type: str, entity_type: str, 
									entity_id: str, data: Dict[str, Any]) -> bool:
		"""Add operation to offline queue"""
		try:
			operation = OfflineOperation(
				id=str(uuid.uuid4()),
				operation_type=operation_type,
				entity_type=entity_type,
				entity_id=entity_id,
				data=data,
				timestamp=datetime.utcnow()
			)
			
			cursor = self.db_connection.cursor()
			op_dict = operation.to_dict()
			
			cursor.execute('''
				INSERT INTO offline_operations 
				(id, operation_type, entity_type, entity_id, data, timestamp, retry_count, last_retry, error_message)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			''', (
				op_dict['id'], op_dict['operation_type'], op_dict['entity_type'],
				op_dict['entity_id'], op_dict['data'], op_dict['timestamp'],
				op_dict['retry_count'], op_dict['last_retry'], op_dict['error_message']
			))
			
			self.db_connection.commit()
			self.pending_operations.append(operation)
			
			# Update app state
			if self.app and hasattr(self.app, 'app_state'):
				self.app.app_state.increment_pending_changes()
			
			self.logger.info(f"Added offline operation: {operation_type} {entity_type} {entity_id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to add offline operation: {e}")
			return False
	
	async def _load_pending_operations(self):
		"""Load pending operations from database"""
		try:
			cursor = self.db_connection.cursor()
			
			cursor.execute('''
				SELECT * FROM offline_operations 
				ORDER BY timestamp ASC
			''')
			
			rows = cursor.fetchall()
			self.pending_operations = []
			
			for row in rows:
				operation = OfflineOperation.from_dict(dict(row))
				self.pending_operations.append(operation)
			
			self.logger.info(f"Loaded {len(self.pending_operations)} pending operations")
			
		except Exception as e:
			self.logger.error(f"Failed to load pending operations: {e}")
	
	async def sync_pending_changes(self) -> Dict[str, Any]:
		"""Sync pending changes with server"""
		if self.is_syncing:
			return {"status": "already_syncing"}
		
		try:
			self.is_syncing = True
			self.logger.info("Starting offline sync")
			
			# Check if we have API service and network connection
			if not self.app or not hasattr(self.app, 'api_service'):
				raise OfflineException("API service not available")
			
			api_service = self.app.api_service
			
			# Check network connectivity
			if not await api_service.health_check():
				raise OfflineException("Network not available")
			
			# Process operations in batches
			total_operations = len(self.pending_operations)
			successful_operations = 0
			failed_operations = 0
			
			for i in range(0, total_operations, SYNC_BATCH_SIZE):
				batch = self.pending_operations[i:i + SYNC_BATCH_SIZE]
				
				for operation in batch:
					try:
						success = await self._sync_operation(operation, api_service)
						
						if success:
							await self._remove_operation(operation.id)
							successful_operations += 1
						else:
							await self._update_operation_retry(operation)
							failed_operations += 1
							
					except Exception as e:
						self.logger.error(f"Failed to sync operation {operation.id}: {e}")
						await self._update_operation_retry(operation, str(e))
						failed_operations += 1
				
				# Update progress
				progress = ((i + len(batch)) / total_operations) * 100
				if self.app and hasattr(self.app, 'app_state'):
					self.app.app_state.sync_status.update_progress(progress)
			
			# Update app state
			if self.app and hasattr(self.app, 'app_state'):
				pending_count = len(self.pending_operations) - successful_operations
				self.app.app_state.sync_status.pending_changes = pending_count
				self.app.app_state.sync_status.mark_sync_complete(failed_operations == 0)
			
			self.last_sync = datetime.utcnow()
			
			result = {
				"status": "completed",
				"total_operations": total_operations,
				"successful_operations": successful_operations,
				"failed_operations": failed_operations,
				"sync_time": self.last_sync.isoformat()
			}
			
			self.logger.info(f"Sync completed: {result}")
			return result
			
		except Exception as e:
			self.logger.error(f"Sync failed: {e}")
			
			if self.app and hasattr(self.app, 'app_state'):
				self.app.app_state.sync_status.mark_sync_complete(False)
				self.app.app_state.sync_status.error_message = str(e)
			
			return {
				"status": "failed",
				"error": str(e)
			}
			
		finally:
			self.is_syncing = False
	
	async def _sync_operation(self, operation: OfflineOperation, api_service) -> bool:
		"""Sync individual operation with server"""
		try:
			self.logger.debug(f"Syncing operation: {operation.operation_type} {operation.entity_type} {operation.entity_id}")
			
			if operation.entity_type == "workflow":
				return await self._sync_workflow_operation(operation, api_service)
			elif operation.entity_type == "task":
				return await self._sync_task_operation(operation, api_service)
			elif operation.entity_type == "notification":
				return await self._sync_notification_operation(operation, api_service)
			else:
				self.logger.warning(f"Unknown entity type: {operation.entity_type}")
				return False
				
		except Exception as e:
			self.logger.error(f"Failed to sync operation {operation.id}: {e}")
			return False
	
	async def _sync_workflow_operation(self, operation: OfflineOperation, api_service) -> bool:
		"""Sync workflow operation"""
		try:
			if operation.operation_type == "create":
				response = await api_service.post("/workflows", operation.data)
			elif operation.operation_type == "update":
				response = await api_service.put(f"/workflows/{operation.entity_id}", operation.data)
			elif operation.operation_type == "delete":
				response = await api_service.delete(f"/workflows/{operation.entity_id}")
			else:
				return False
			
			return response.success
			
		except Exception as e:
			self.logger.error(f"Failed to sync workflow operation: {e}")
			return False
	
	async def _sync_task_operation(self, operation: OfflineOperation, api_service) -> bool:
		"""Sync task operation"""
		try:
			if operation.operation_type == "create":
				response = await api_service.post("/tasks", operation.data)
			elif operation.operation_type == "update":
				response = await api_service.put(f"/tasks/{operation.entity_id}", operation.data)
			elif operation.operation_type == "delete":
				response = await api_service.delete(f"/tasks/{operation.entity_id}")
			else:
				return False
			
			return response.success
			
		except Exception as e:
			self.logger.error(f"Failed to sync task operation: {e}")
			return False
	
	async def _sync_notification_operation(self, operation: OfflineOperation, api_service) -> bool:
		"""Sync notification operation"""
		try:
			if operation.operation_type == "mark_read":
				response = await api_service.post(f"/notifications/{operation.entity_id}/read")
			elif operation.operation_type == "delete":
				response = await api_service.delete(f"/notifications/{operation.entity_id}")
			else:
				return False
			
			return response.success
			
		except Exception as e:
			self.logger.error(f"Failed to sync notification operation: {e}")
			return False
	
	async def _remove_operation(self, operation_id: str):
		"""Remove operation from database and memory"""
		try:
			cursor = self.db_connection.cursor()
			cursor.execute('DELETE FROM offline_operations WHERE id = ?', (operation_id,))
			self.db_connection.commit()
			
			# Remove from memory
			self.pending_operations = [op for op in self.pending_operations if op.id != operation_id]
			
			# Update app state
			if self.app and hasattr(self.app, 'app_state'):
				self.app.app_state.decrement_pending_changes()
			
		except Exception as e:
			self.logger.error(f"Failed to remove operation {operation_id}: {e}")
	
	async def _update_operation_retry(self, operation: OfflineOperation, error_message: Optional[str] = None):
		"""Update operation retry count and error"""
		try:
			operation.retry_count += 1
			operation.last_retry = datetime.utcnow()
			if error_message:
				operation.error_message = error_message
			
			cursor = self.db_connection.cursor()
			cursor.execute('''
				UPDATE offline_operations 
				SET retry_count = ?, last_retry = ?, error_message = ?
				WHERE id = ?
			''', (operation.retry_count, operation.last_retry.isoformat(), operation.error_message, operation.id))
			
			self.db_connection.commit()
			
		except Exception as e:
			self.logger.error(f"Failed to update operation retry: {e}")
	
	async def get_cache_statistics(self) -> Dict[str, Any]:
		"""Get offline cache statistics"""
		try:
			cursor = self.db_connection.cursor()
			
			# Count cached items
			cursor.execute('SELECT COUNT(*) as count FROM workflows_cache')
			workflows_count = cursor.fetchone()['count']
			
			cursor.execute('SELECT COUNT(*) as count FROM tasks_cache')
			tasks_count = cursor.fetchone()['count']
			
			cursor.execute('SELECT COUNT(*) as count FROM notifications_cache')
			notifications_count = cursor.fetchone()['count']
			
			cursor.execute('SELECT COUNT(*) as count FROM offline_operations')
			operations_count = cursor.fetchone()['count']
			
			# Get database size
			db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
			
			return {
				"workflows_cached": workflows_count,
				"tasks_cached": tasks_count,
				"notifications_cached": notifications_count,
				"pending_operations": operations_count,
				"database_size_bytes": db_size,
				"last_sync": self.last_sync.isoformat() if self.last_sync else None,
				"is_syncing": self.is_syncing
			}
			
		except Exception as e:
			self.logger.error(f"Failed to get cache statistics: {e}")
			return {}
	
	async def clear_cache(self, entity_type: Optional[str] = None):
		"""Clear offline cache"""
		try:
			cursor = self.db_connection.cursor()
			
			if entity_type == "workflows":
				cursor.execute('DELETE FROM workflows_cache')
			elif entity_type == "tasks":
				cursor.execute('DELETE FROM tasks_cache')
			elif entity_type == "notifications":
				cursor.execute('DELETE FROM notifications_cache')
			else:
				# Clear all cache
				cursor.execute('DELETE FROM workflows_cache')
				cursor.execute('DELETE FROM tasks_cache')
				cursor.execute('DELETE FROM notifications_cache')
				cursor.execute('DELETE FROM user_data_cache')
				cursor.execute('DELETE FROM file_cache')
			
			self.db_connection.commit()
			
			self.logger.info(f"Cleared offline cache: {entity_type or 'all'}")
			
		except Exception as e:
			self.logger.error(f"Failed to clear cache: {e}")
			raise OfflineException(f"Failed to clear cache: {e}")
	
	async def save_pending_changes(self):
		"""Save any pending changes before app shutdown"""
		try:
			if self.db_connection:
				self.db_connection.commit()
				
			self.logger.info("Saved pending changes")
			
		except Exception as e:
			self.logger.error(f"Failed to save pending changes: {e}")
	
	async def close(self):
		"""Close offline service and database connection"""
		try:
			await self.save_pending_changes()
			
			if self.db_connection:
				self.db_connection.close()
				self.db_connection = None
			
			self.logger.info("Offline service closed")
			
		except Exception as e:
			self.logger.error(f"Error closing offline service: {e}")
	
	def get_pending_operations_count(self) -> int:
		"""Get count of pending operations"""
		return len(self.pending_operations)
	
	def has_pending_changes(self) -> bool:
		"""Check if there are pending changes to sync"""
		return len(self.pending_operations) > 0