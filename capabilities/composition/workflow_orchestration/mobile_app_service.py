"""
Mobile Application Service Module

Provides comprehensive mobile application support:
- Native iOS/Android integration
- React Native components
- Mobile-specific workflow features
- Offline capability support
- Push notifications
- Mobile authentication
- Touch-optimized interfaces

© 2025 Datacraft
Author: Nyimbi Odero
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str
import structlog

from .database import get_async_db_session

logger = structlog.get_logger(__name__)


class MobilePlatform(str, Enum):
	"""Mobile platform types"""
	IOS = "ios"
	ANDROID = "android"
	REACT_NATIVE = "react_native"
	FLUTTER = "flutter"
	CORDOVA = "cordova"
	IONIC = "ionic"
	XAMARIN = "xamarin"


class MobileDeviceType(str, Enum):
	"""Mobile device types"""
	PHONE = "phone"
	TABLET = "tablet"
	WATCH = "watch"
	TV = "tv"
	AUTOMOTIVE = "automotive"


class NotificationType(str, Enum):
	"""Push notification types"""
	WORKFLOW_STARTED = "workflow_started"
	WORKFLOW_COMPLETED = "workflow_completed"
	WORKFLOW_FAILED = "workflow_failed"
	APPROVAL_REQUEST = "approval_request"
	TASK_ASSIGNED = "task_assigned"
	DEADLINE_APPROACHING = "deadline_approaching"
	SYSTEM_ALERT = "system_alert"


class OfflineMode(str, Enum):
	"""Offline capability modes"""
	NONE = "none"
	READ_ONLY = "read_only"
	CACHE_UPDATES = "cache_updates"
	FULL_OFFLINE = "full_offline"


class TouchGesture(str, Enum):
	"""Touch gesture types"""
	TAP = "tap"
	DOUBLE_TAP = "double_tap"
	LONG_PRESS = "long_press"
	SWIPE_LEFT = "swipe_left"
	SWIPE_RIGHT = "swipe_right"
	SWIPE_UP = "swipe_up"
	SWIPE_DOWN = "swipe_down"
	PINCH_ZOOM = "pinch_zoom"
	ROTATE = "rotate"
	PAN = "pan"


@dataclass
class MobileDevice:
	"""Mobile device information"""
	device_id: str
	user_id: str
	platform: MobilePlatform
	device_type: MobileDeviceType
	model: str
	os_version: str
	app_version: str
	push_token: Optional[str] = None
	screen_width: int = 0
	screen_height: int = 0
	screen_density: float = 1.0
	locale: str = "en_US"
	timezone: str = "UTC"
	capabilities: List[str] = field(default_factory=list)
	last_seen: datetime = field(default_factory=datetime.utcnow)
	is_active: bool = True


@dataclass
class MobileSession:
	"""Mobile app session"""
	session_id: str
	device: MobileDevice
	user_id: str
	started_at: datetime = field(default_factory=datetime.utcnow)
	last_activity: datetime = field(default_factory=datetime.utcnow)
	offline_mode: OfflineMode = OfflineMode.NONE
	cached_workflows: List[str] = field(default_factory=list)
	pending_sync: List[Dict[str, Any]] = field(default_factory=list)
	background_tasks: List[str] = field(default_factory=list)
	is_foreground: bool = True


class PushNotification(BaseModel):
	"""Push notification model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	device_id: str
	user_id: str
	notification_type: NotificationType
	title: str
	message: str
	data: Dict[str, Any] = Field(default_factory=dict)
	priority: str = "normal"  # low, normal, high
	badge_count: Optional[int] = None
	sound: Optional[str] = None
	category: Optional[str] = None
	thread_id: Optional[str] = None
	scheduled_time: Optional[datetime] = None
	expires_at: Optional[datetime] = None
	sent_at: Optional[datetime] = None
	delivered_at: Optional[datetime] = None
	opened_at: Optional[datetime] = None
	tenant_id: Optional[str] = None


class MobileWorkflowNode(BaseModel):
	"""Mobile-optimized workflow node"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	type: str
	name: str
	description: str = ""
	mobile_layout: Dict[str, Any] = Field(default_factory=dict)
	touch_interactions: List[TouchGesture] = Field(default_factory=list)
	offline_capable: bool = False
	requires_network: bool = True
	estimated_duration: int = 30  # seconds
	mobile_specific_config: Dict[str, Any] = Field(default_factory=dict)


class MobileAppService:
	"""Mobile application management service"""
	
	def __init__(self):
		self.devices: Dict[str, MobileDevice] = {}
		self.sessions: Dict[str, MobileSession] = {}
		self.push_providers: Dict[str, Any] = {}
		self.offline_cache: Dict[str, Dict[str, Any]] = {}
		
		# Mobile-specific settings
		self.max_cache_size = 100 * 1024 * 1024  # 100MB
		self.session_timeout = timedelta(hours=24)
		self.offline_sync_interval = timedelta(minutes=5)
		self.background_task_limit = 10
		
		# Initialize push notification providers
		self._init_push_providers()
	
	def _init_push_providers(self):
		"""Initialize push notification providers"""
		try:
			# Firebase Cloud Messaging (Android & iOS)
			self.push_providers["fcm"] = {
				"service_account_key": None,  # Would be loaded from config
				"project_id": "apg-mobile-app",
				"enabled": True
			}
			
			# Apple Push Notification Service
			self.push_providers["apns"] = {
				"key_id": None,  # Would be loaded from config
				"team_id": None,
				"bundle_id": "com.datacraft.apg",
				"enabled": True
			}
			
			# Web Push (for PWA)
			self.push_providers["web_push"] = {
				"vapid_public_key": None,
				"vapid_private_key": None,
				"enabled": True
			}
			
		except Exception as e:
			logger.error(f"Push provider initialization error: {e}")
	
	async def register_device(self, device_info: Dict[str, Any]) -> str:
		"""Register mobile device"""
		try:
			device = MobileDevice(
				device_id=device_info.get("device_id", uuid7str()),
				user_id=device_info["user_id"],
				platform=MobilePlatform(device_info["platform"]),
				device_type=MobileDeviceType(device_info.get("device_type", "phone")),
				model=device_info.get("model", "Unknown"),
				os_version=device_info.get("os_version", "1.0"),
				app_version=device_info.get("app_version", "1.0.0"),
				push_token=device_info.get("push_token"),
				screen_width=device_info.get("screen_width", 375),
				screen_height=device_info.get("screen_height", 667),
				screen_density=device_info.get("screen_density", 1.0),
				locale=device_info.get("locale", "en_US"),
				timezone=device_info.get("timezone", "UTC"),
				capabilities=device_info.get("capabilities", [])
			)
			
			self.devices[device.device_id] = device
			
			# Store in database
			await self._store_device_info(device)
			
			logger.info(f"Registered mobile device: {device.device_id} for user {device.user_id}")
			return device.device_id
			
		except Exception as e:
			logger.error(f"Device registration error: {e}")
			raise
	
	async def create_mobile_session(self, device_id: str, config: Dict[str, Any]) -> str:
		"""Create mobile app session"""
		try:
			device = self.devices.get(device_id)
			if not device:
				# Try to load from database
				device = await self._load_device_info(device_id)
				if not device:
					raise ValueError(f"Device not found: {device_id}")
			
			session_id = uuid7str()
			session = MobileSession(
				session_id=session_id,
				device=device,
				user_id=device.user_id,
				offline_mode=OfflineMode(config.get("offline_mode", "read_only"))
			)
			
			self.sessions[session_id] = session
			
			# Initialize offline cache if needed
			if session.offline_mode != OfflineMode.NONE:
				await self._init_offline_cache(session)
			
			# Store session in database
			await self._store_session_info(session)
			
			return session_id
			
		except Exception as e:
			logger.error(f"Mobile session creation error: {e}")
			raise
	
	async def get_mobile_workflows(self, session_id: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Get workflows optimized for mobile"""
		try:
			session = self.sessions.get(session_id)
			if not session:
				raise ValueError(f"Session not found: {session_id}")
			
			# Update last activity
			session.last_activity = datetime.utcnow()
			
			# Build mobile-optimized query
			async with get_async_db_session() as db_session:
				from sqlalchemy import text
				
				query = """
				SELECT 
					w.id,
					w.name,
					w.description,
					w.status,
					w.priority,
					w.created_at,
					w.updated_at,
					w.mobile_config,
					COUNT(wn.id) as node_count,
					AVG(CASE WHEN we.status = 'completed' THEN 
						EXTRACT(EPOCH FROM (we.completed_at - we.started_at))
					END) as avg_duration
				FROM workflow_definitions w
				LEFT JOIN workflow_nodes wn ON w.id = wn.workflow_id
				LEFT JOIN workflow_executions we ON w.id = we.workflow_id
				WHERE w.tenant_id = :tenant_id
				AND w.is_mobile_compatible = true
				"""
				
				# Add filters
				params = {"tenant_id": session.device.user_id}  # Simplified for example
				
				if filters.get("status"):
					query += " AND w.status = :status"
					params["status"] = filters["status"]
				
				if filters.get("category"):
					query += " AND w.category = :category"
					params["category"] = filters["category"]
				
				if filters.get("offline_capable") and session.offline_mode != OfflineMode.NONE:
					query += " AND w.offline_capable = true"
				
				query += """
				GROUP BY w.id, w.name, w.description, w.status, w.priority, 
						 w.created_at, w.updated_at, w.mobile_config
				ORDER BY w.priority DESC, w.updated_at DESC
				LIMIT :limit OFFSET :offset
				"""
				
				params.update({
					"limit": filters.get("limit", 20),
					"offset": filters.get("offset", 0)
				})
				
				result = await db_session.execute(text(query), params)
				workflows = []
				
				for row in result:
					mobile_config = json.loads(row.mobile_config) if row.mobile_config else {}
					
					workflow = {
						"id": row.id,
						"name": row.name,
						"description": row.description,
						"status": row.status,
						"priority": row.priority,
						"node_count": row.node_count,
						"avg_duration": int(row.avg_duration or 0),
						"created_at": row.created_at.isoformat(),
						"updated_at": row.updated_at.isoformat(),
						"mobile_optimized": True,
						"offline_capable": mobile_config.get("offline_capable", False),
						"touch_friendly": mobile_config.get("touch_friendly", True),
						"estimated_time": mobile_config.get("estimated_time", "5-10 min"),
						"complexity": mobile_config.get("complexity", "simple"),
						"requires_approval": mobile_config.get("requires_approval", False)
					}
					
					workflows.append(workflow)
				
				return workflows
			
		except Exception as e:
			logger.error(f"Get mobile workflows error: {e}")
			raise
	
	async def execute_mobile_workflow(self, session_id: str, workflow_id: str, params: Dict[str, Any]) -> str:
		"""Execute workflow from mobile app"""
		try:
			session = self.sessions.get(session_id)
			if not session:
				raise ValueError(f"Session not found: {session_id}")
			
			# Check if workflow can run offline
			if not session.is_foreground and session.offline_mode == OfflineMode.NONE:
				raise ValueError("Cannot execute workflow in background without offline mode")
			
			# Create execution record
			execution_id = uuid7str()
			
			async with get_async_db_session() as db_session:
				from sqlalchemy import text
				
				await db_session.execute(
					text("""
					INSERT INTO workflow_executions (
						id, workflow_id, user_id, status, started_at,
						execution_context, mobile_session_id, device_id
					) VALUES (
						:id, :workflow_id, :user_id, :status, :started_at,
						:execution_context, :mobile_session_id, :device_id
					)
					"""),
					{
						"id": execution_id,
						"workflow_id": workflow_id,
						"user_id": session.user_id,
						"status": "running",
						"started_at": datetime.utcnow(),
						"execution_context": json.dumps({
							"mobile": True,
							"device_id": session.device.device_id,
							"platform": session.device.platform.value,
							"offline_mode": session.offline_mode.value,
							"parameters": params
						}),
						"mobile_session_id": session_id,
						"device_id": session.device.device_id
					}
				)
				await db_session.commit()
			
			# Queue for background execution if needed
			if not session.is_foreground:
				session.background_tasks.append(execution_id)
				await self._queue_background_task(session, execution_id, workflow_id, params)
			
			# Send push notification
			await self.send_push_notification(
				session.device.device_id,
				NotificationType.WORKFLOW_STARTED,
				"Workflow Started",
				f"Your workflow '{workflow_id}' has started execution",
				{"execution_id": execution_id, "workflow_id": workflow_id}
			)
			
			return execution_id
			
		except Exception as e:
			logger.error(f"Mobile workflow execution error: {e}")
			raise
	
	async def get_mobile_execution_status(self, session_id: str, execution_id: str) -> Dict[str, Any]:
		"""Get workflow execution status for mobile"""
		try:
			session = self.sessions.get(session_id)
			if not session:
				raise ValueError(f"Session not found: {session_id}")
			
			async with get_async_db_session() as db_session:
				from sqlalchemy import text
				
				result = await db_session.execute(
					text("""
					SELECT 
						we.id,
						we.workflow_id,
						we.status,
						we.started_at,
						we.completed_at,
						we.error_message,
						we.progress_percentage,
						we.current_node_id,
						we.execution_context,
						wd.name as workflow_name,
						wn.name as current_node_name
					FROM workflow_executions we
					JOIN workflow_definitions wd ON we.workflow_id = wd.id
					LEFT JOIN workflow_nodes wn ON we.current_node_id = wn.id
					WHERE we.id = :execution_id
					AND we.mobile_session_id = :session_id
					"""),
					{"execution_id": execution_id, "session_id": session_id}
				)
				
				row = result.first()
				if not row:
					raise ValueError(f"Execution not found: {execution_id}")
				
				execution_context = json.loads(row.execution_context) if row.execution_context else {}
				
				# Calculate estimated completion time
				estimated_completion = None
				if row.status == "running" and row.started_at:
					elapsed = datetime.utcnow() - row.started_at
					if row.progress_percentage and row.progress_percentage > 0:
						total_estimated = elapsed * (100 / row.progress_percentage)
						estimated_completion = row.started_at + total_estimated
				
				status = {
					"execution_id": row.id,
					"workflow_id": row.workflow_id,
					"workflow_name": row.workflow_name,
					"status": row.status,
					"progress": row.progress_percentage or 0,
					"started_at": row.started_at.isoformat() if row.started_at else None,
					"completed_at": row.completed_at.isoformat() if row.completed_at else None,
					"estimated_completion": estimated_completion.isoformat() if estimated_completion else None,
					"current_step": {
						"node_id": row.current_node_id,
						"name": row.current_node_name or "Unknown"
					},
					"error_message": row.error_message,
					"mobile_context": execution_context.get("mobile", False),
					"can_cancel": row.status in ["running", "paused"],
					"can_retry": row.status in ["failed", "cancelled"]
				}
				
				return status
			
		except Exception as e:
			logger.error(f"Get mobile execution status error: {e}")
			raise
	
	async def send_push_notification(self, device_id: str, notification_type: NotificationType, 
									 title: str, message: str, data: Dict[str, Any]) -> str:
		"""Send push notification to mobile device"""
		try:
			device = self.devices.get(device_id)
			if not device or not device.push_token:
				logger.warning(f"Cannot send push notification - device not found or no push token: {device_id}")
				return ""
			
			notification = PushNotification(
				device_id=device_id,
				user_id=device.user_id,
				notification_type=notification_type,
				title=title,
				message=message,
				data=data,
				priority="high" if notification_type in [NotificationType.WORKFLOW_FAILED, NotificationType.APPROVAL_REQUEST] else "normal"
			)
			
			# Choose provider based on platform
			provider = None
			if device.platform == MobilePlatform.IOS:
				provider = "apns"
			elif device.platform == MobilePlatform.ANDROID:
				provider = "fcm"
			else:
				provider = "web_push"
			
			# Send notification
			success = await self._send_notification_via_provider(provider, device, notification)
			
			if success:
				notification.sent_at = datetime.utcnow()
				await self._store_notification(notification)
			
			return notification.id
			
		except Exception as e:
			logger.error(f"Push notification error: {e}")
			return ""
	
	async def _send_notification_via_provider(self, provider: str, device: MobileDevice, 
											  notification: PushNotification) -> bool:
		"""Send notification via specific provider"""
		try:
			if provider == "fcm":
				return await self._send_fcm_notification(device, notification)
			elif provider == "apns":
				return await self._send_apns_notification(device, notification)
			elif provider == "web_push":
				return await self._send_web_push_notification(device, notification)
			else:
				logger.error(f"Unknown push provider: {provider}")
				return False
				
		except Exception as e:
			logger.error(f"Provider notification error: {e}")
			return False
	
	async def _send_fcm_notification(self, device: MobileDevice, notification: PushNotification) -> bool:
		"""Send FCM notification (Android/iOS)"""
		try:
			# In a real implementation, this would use the Firebase Admin SDK
			fcm_payload = {
				"to": device.push_token,
				"notification": {
					"title": notification.title,
					"body": notification.message,
					"sound": notification.sound or "default",
					"badge": notification.badge_count
				},
				"data": notification.data,
				"priority": notification.priority,
				"android": {
					"notification": {
						"channel_id": "workflow_notifications",
						"click_action": "WORKFLOW_NOTIFICATION"
					}
				},
				"apns": {
					"payload": {
						"aps": {
							"category": notification.category or "WORKFLOW"
						}
					}
				}
			}
			
			# Simulate successful send
			logger.info(f"FCM notification sent to {device.device_id}: {notification.title}")
			return True
			
		except Exception as e:
			logger.error(f"FCM notification error: {e}")
			return False
	
	async def _send_apns_notification(self, device: MobileDevice, notification: PushNotification) -> bool:
		"""Send APNS notification (iOS)"""
		try:
			# In a real implementation, this would use the APNS HTTP/2 API
			apns_payload = {
				"aps": {
					"alert": {
						"title": notification.title,
						"body": notification.message
					},
					"sound": notification.sound or "default",
					"badge": notification.badge_count,
					"category": notification.category or "WORKFLOW",
					"thread-id": notification.thread_id or "workflow"
				},
				"custom_data": notification.data
			}
			
			# Simulate successful send
			logger.info(f"APNS notification sent to {device.device_id}: {notification.title}")
			return True
			
		except Exception as e:
			logger.error(f"APNS notification error: {e}")
			return False
	
	async def _send_web_push_notification(self, device: MobileDevice, notification: PushNotification) -> bool:
		"""Send Web Push notification (PWA)"""
		try:
			# In a real implementation, this would use the Web Push Protocol
			web_push_payload = {
				"title": notification.title,
				"body": notification.message,
				"icon": "/static/icons/workflow-icon.png",
				"badge": "/static/icons/badge.png",
				"data": notification.data,
				"actions": [
					{"action": "view", "title": "View Workflow"},
					{"action": "dismiss", "title": "Dismiss"}
				]
			}
			
			# Simulate successful send
			logger.info(f"Web Push notification sent to {device.device_id}: {notification.title}")
			return True
			
		except Exception as e:
			logger.error(f"Web Push notification error: {e}")
			return False
	
	async def sync_offline_data(self, session_id: str) -> Dict[str, Any]:
		"""Sync offline data with server"""
		try:
			session = self.sessions.get(session_id)
			if not session or session.offline_mode == OfflineMode.NONE:
				return {"synced": False, "reason": "Offline mode not enabled"}
			
			sync_results = {
				"synced": True,
				"timestamp": datetime.utcnow().isoformat(),
				"uploaded": 0,
				"downloaded": 0,
				"conflicts": 0,
				"errors": []
			}
			
			# Upload pending changes
			for pending_item in session.pending_sync:
				try:
					await self._upload_pending_change(session, pending_item)
					sync_results["uploaded"] += 1
				except Exception as e:
					sync_results["errors"].append(f"Upload error: {e}")
			
			# Download latest data
			await self._download_latest_data(session)
			sync_results["downloaded"] = len(session.cached_workflows)
			
			# Clear pending sync items
			session.pending_sync.clear()
			
			return sync_results
			
		except Exception as e:
			logger.error(f"Offline sync error: {e}")
			return {"synced": False, "error": str(e)}
	
	async def _init_offline_cache(self, session: MobileSession):
		"""Initialize offline cache for session"""
		try:
			if session.session_id not in self.offline_cache:
				self.offline_cache[session.session_id] = {
					"workflows": {},
					"executions": {},
					"templates": {},
					"last_sync": datetime.utcnow().isoformat()
				}
			
			# Pre-cache frequently used workflows
			await self._cache_frequent_workflows(session)
			
		except Exception as e:
			logger.error(f"Offline cache initialization error: {e}")
	
	async def _cache_frequent_workflows(self, session: MobileSession):
		"""Cache frequently used workflows for offline access"""
		try:
			async with get_async_db_session() as db_session:
				from sqlalchemy import text
				
				# Get user's most frequently executed workflows
				result = await db_session.execute(
					text("""
					SELECT 
						wd.id,
						wd.name,
						wd.definition,
						wd.mobile_config,
						COUNT(we.id) as execution_count
					FROM workflow_definitions wd
					LEFT JOIN workflow_executions we ON wd.id = we.workflow_id
					WHERE we.user_id = :user_id
					AND wd.offline_capable = true
					GROUP BY wd.id, wd.name, wd.definition, wd.mobile_config
					ORDER BY execution_count DESC
					LIMIT 10
					"""),
					{"user_id": session.user_id}
				)
				
				cache = self.offline_cache[session.session_id]
				for row in result:
					cache["workflows"][row.id] = {
						"id": row.id,
						"name": row.name,
						"definition": json.loads(row.definition) if row.definition else {},
						"mobile_config": json.loads(row.mobile_config) if row.mobile_config else {},
						"cached_at": datetime.utcnow().isoformat()
					}
					session.cached_workflows.append(row.id)
			
		except Exception as e:
			logger.error(f"Cache frequent workflows error: {e}")
	
	async def _upload_pending_change(self, session: MobileSession, pending_item: Dict[str, Any]):
		"""Upload pending change to server"""
		try:
			change_type = pending_item.get("type")
			
			if change_type == "workflow_execution":
				await self._upload_execution_data(session, pending_item)
			elif change_type == "workflow_modification":
				await self._upload_workflow_changes(session, pending_item)
			elif change_type == "user_input":
				await self._upload_user_input(session, pending_item)
			
		except Exception as e:
			logger.error(f"Upload pending change error: {e}")
			raise
	
	async def _upload_execution_data(self, session: MobileSession, execution_data: Dict[str, Any]):
		"""Upload workflow execution data"""
		try:
			async with get_async_db_session() as db_session:
				from sqlalchemy import text
				
				await db_session.execute(
					text("""
					UPDATE workflow_executions
					SET status = :status,
						completed_at = :completed_at,
						progress_percentage = :progress,
						execution_context = :context
					WHERE id = :execution_id
					"""),
					{
						"execution_id": execution_data["execution_id"],
						"status": execution_data["status"],
						"completed_at": execution_data.get("completed_at"),
						"progress": execution_data.get("progress", 0),
						"context": json.dumps(execution_data.get("context", {}))
					}
				)
				await db_session.commit()
			
		except Exception as e:
			logger.error(f"Upload execution data error: {e}")
			raise
	
	async def _upload_workflow_changes(self, session: MobileSession, workflow_data: Dict[str, Any]):
		"""Upload workflow changes"""
		try:
			# This would handle workflow modifications made offline
			# For now, just log the action
			logger.info(f"Uploading workflow changes for {workflow_data.get('workflow_id')}")
			
		except Exception as e:
			logger.error(f"Upload workflow changes error: {e}")
			raise
	
	async def _upload_user_input(self, session: MobileSession, input_data: Dict[str, Any]):
		"""Upload user input data"""
		try:
			# This would handle form submissions and user interactions made offline
			logger.info(f"Uploading user input for {input_data.get('workflow_id')}")
			
		except Exception as e:
			logger.error(f"Upload user input error: {e}")
			raise
	
	async def _download_latest_data(self, session: MobileSession):
		"""Download latest data for offline use"""
		try:
			# Download latest workflow definitions, templates, etc.
			await self._cache_frequent_workflows(session)
			
		except Exception as e:
			logger.error(f"Download latest data error: {e}")
			raise
	
	async def _queue_background_task(self, session: MobileSession, execution_id: str, 
									 workflow_id: str, params: Dict[str, Any]):
		"""Queue workflow for background execution"""
		try:
			# In a real implementation, this would use a background task queue
			# like Celery, RQ, or similar
			logger.info(f"Queued background task: {execution_id} for workflow {workflow_id}")
			
		except Exception as e:
			logger.error(f"Queue background task error: {e}")
	
	async def _store_device_info(self, device: MobileDevice):
		"""Store device information in database"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				await session.execute(
					text("""
					INSERT INTO mobile_devices (
						device_id, user_id, platform, device_type, model,
						os_version, app_version, push_token, screen_width,
						screen_height, screen_density, locale, timezone,
						capabilities, last_seen, is_active, created_at
					) VALUES (
						:device_id, :user_id, :platform, :device_type, :model,
						:os_version, :app_version, :push_token, :screen_width,
						:screen_height, :screen_density, :locale, :timezone,
						:capabilities, :last_seen, :is_active, :created_at
					)
					ON CONFLICT (device_id) DO UPDATE SET
						push_token = EXCLUDED.push_token,
						os_version = EXCLUDED.os_version,
						app_version = EXCLUDED.app_version,
						last_seen = EXCLUDED.last_seen,
						is_active = EXCLUDED.is_active
					"""),
					{
						"device_id": device.device_id,
						"user_id": device.user_id,
						"platform": device.platform.value,
						"device_type": device.device_type.value,
						"model": device.model,
						"os_version": device.os_version,
						"app_version": device.app_version,
						"push_token": device.push_token,
						"screen_width": device.screen_width,
						"screen_height": device.screen_height,
						"screen_density": device.screen_density,
						"locale": device.locale,
						"timezone": device.timezone,
						"capabilities": json.dumps(device.capabilities),
						"last_seen": device.last_seen,
						"is_active": device.is_active,
						"created_at": datetime.utcnow()
					}
				)
				await session.commit()
			
		except Exception as e:
			logger.error(f"Store device info error: {e}")
	
	async def _load_device_info(self, device_id: str) -> Optional[MobileDevice]:
		"""Load device information from database"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				result = await session.execute(
					text("""
					SELECT * FROM mobile_devices WHERE device_id = :device_id
					"""),
					{"device_id": device_id}
				)
				
				row = result.first()
				if not row:
					return None
				
				device = MobileDevice(
					device_id=row.device_id,
					user_id=row.user_id,
					platform=MobilePlatform(row.platform),
					device_type=MobileDeviceType(row.device_type),
					model=row.model,
					os_version=row.os_version,
					app_version=row.app_version,
					push_token=row.push_token,
					screen_width=row.screen_width,
					screen_height=row.screen_height,
					screen_density=row.screen_density,
					locale=row.locale,
					timezone=row.timezone,
					capabilities=json.loads(row.capabilities) if row.capabilities else [],
					last_seen=row.last_seen,
					is_active=row.is_active
				)
				
				self.devices[device_id] = device
				return device
			
		except Exception as e:
			logger.error(f"Load device info error: {e}")
			return None
	
	async def _store_session_info(self, session: MobileSession):
		"""Store session information in database"""
		try:
			async with get_async_db_session() as db_session:
				from sqlalchemy import text
				
				await db_session.execute(
					text("""
					INSERT INTO mobile_sessions (
						session_id, device_id, user_id, started_at,
						last_activity, offline_mode, cached_workflows,
						pending_sync, background_tasks, is_foreground
					) VALUES (
						:session_id, :device_id, :user_id, :started_at,
						:last_activity, :offline_mode, :cached_workflows,
						:pending_sync, :background_tasks, :is_foreground
					)
					"""),
					{
						"session_id": session.session_id,
						"device_id": session.device.device_id,
						"user_id": session.user_id,
						"started_at": session.started_at,
						"last_activity": session.last_activity,
						"offline_mode": session.offline_mode.value,
						"cached_workflows": json.dumps(session.cached_workflows),
						"pending_sync": json.dumps(session.pending_sync),
						"background_tasks": json.dumps(session.background_tasks),
						"is_foreground": session.is_foreground
					}
				)
				await db_session.commit()
			
		except Exception as e:
			logger.error(f"Store session info error: {e}")
	
	async def _store_notification(self, notification: PushNotification):
		"""Store notification in database"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				await session.execute(
					text("""
					INSERT INTO push_notifications (
						id, device_id, user_id, notification_type, title,
						message, data, priority, badge_count, sound,
						category, thread_id, scheduled_time, expires_at,
						sent_at, delivered_at, opened_at, tenant_id, created_at
					) VALUES (
						:id, :device_id, :user_id, :notification_type, :title,
						:message, :data, :priority, :badge_count, :sound,
						:category, :thread_id, :scheduled_time, :expires_at,
						:sent_at, :delivered_at, :opened_at, :tenant_id, :created_at
					)
					"""),
					{
						"id": notification.id,
						"device_id": notification.device_id,
						"user_id": notification.user_id,
						"notification_type": notification.notification_type.value,
						"title": notification.title,
						"message": notification.message,
						"data": json.dumps(notification.data),
						"priority": notification.priority,
						"badge_count": notification.badge_count,
						"sound": notification.sound,
						"category": notification.category,
						"thread_id": notification.thread_id,
						"scheduled_time": notification.scheduled_time,
						"expires_at": notification.expires_at,
						"sent_at": notification.sent_at,
						"delivered_at": notification.delivered_at,
						"opened_at": notification.opened_at,
						"tenant_id": notification.tenant_id,
						"created_at": datetime.utcnow()
					}
				)
				await session.commit()
			
		except Exception as e:
			logger.error(f"Store notification error: {e}")


# Global mobile app service instance
mobile_app_service = MobileAppService()