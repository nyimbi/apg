"""
Application state management

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Callable
from dataclasses import dataclass, field
import json
import threading
from pathlib import Path

from .user import User
from .workflow import Workflow, WorkflowInstance
from .task import Task
from .notification import Notification
from ..utils.constants import APP_DATA_DIR


class NetworkState(str, Enum):
	"""Network connectivity state"""
	CONNECTED = "connected"
	DISCONNECTED = "disconnected"
	CONNECTING = "connecting"
	LIMITED = "limited"
	UNKNOWN = "unknown"


class SyncState(str, Enum):
	"""Data synchronization state"""
	IDLE = "idle"
	SYNCING = "syncing"
	SYNC_SUCCESS = "sync_success"
	SYNC_FAILED = "sync_failed"
	SYNC_PENDING = "sync_pending"
	CONFLICT = "conflict"


class AppTheme(str, Enum):
	"""Application theme"""
	LIGHT = "light"
	DARK = "dark"
	AUTO = "auto"


@dataclass
class ConnectionInfo:
	"""Network connection information"""
	is_connected: bool = False
	connection_type: str = "unknown"  # wifi, cellular, ethernet, etc.
	signal_strength: Optional[int] = None  # 0-100
	bandwidth: Optional[float] = None  # Mbps
	latency: Optional[int] = None  # milliseconds
	last_check: Optional[datetime] = None
	
	def update(self, **kwargs):
		"""Update connection info"""
		for key, value in kwargs.items():
			if hasattr(self, key):
				setattr(self, key, value)
		self.last_check = datetime.utcnow()


@dataclass
class SyncStatus:
	"""Data synchronization status"""
	state: SyncState = SyncState.IDLE
	last_sync: Optional[datetime] = None
	next_sync: Optional[datetime] = None
	pending_changes: int = 0
	failed_items: int = 0
	sync_progress: float = 0.0  # 0-100
	error_message: Optional[str] = None
	
	def update_progress(self, progress: float, message: Optional[str] = None):
		"""Update sync progress"""
		self.sync_progress = max(0.0, min(100.0, progress))
		if message:
			self.error_message = message
	
	def mark_sync_complete(self, success: bool = True):
		"""Mark sync as complete"""
		self.state = SyncState.SYNC_SUCCESS if success else SyncState.SYNC_FAILED
		self.last_sync = datetime.utcnow()
		self.sync_progress = 100.0 if success else 0.0
		if success:
			self.failed_items = 0
			self.error_message = None


@dataclass
class AppSettings:
	"""Application settings and preferences"""
	theme: AppTheme = AppTheme.AUTO
	language: str = "en"
	timezone: str = "UTC"
	
	# UI preferences
	animations_enabled: bool = True
	haptic_feedback: bool = True
	sound_effects: bool = True
	auto_refresh: bool = True
	refresh_interval: int = 300  # seconds
	
	# Sync preferences
	auto_sync: bool = True
	sync_on_cellular: bool = False
	background_sync: bool = True
	
	# Security preferences
	biometric_enabled: bool = False
	auto_lock: bool = True
	auto_lock_timeout: int = 300  # seconds
	
	# Notification preferences
	push_notifications: bool = True
	notification_sound: bool = True
	notification_vibration: bool = True
	
	# Debug settings
	debug_mode: bool = False
	verbose_logging: bool = False
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert settings to dictionary"""
		return {
			"theme": self.theme.value,
			"language": self.language,
			"timezone": self.timezone,
			"animations_enabled": self.animations_enabled,
			"haptic_feedback": self.haptic_feedback,
			"sound_effects": self.sound_effects,
			"auto_refresh": self.auto_refresh,
			"refresh_interval": self.refresh_interval,
			"auto_sync": self.auto_sync,
			"sync_on_cellular": self.sync_on_cellular,
			"background_sync": self.background_sync,
			"biometric_enabled": self.biometric_enabled,
			"auto_lock": self.auto_lock,
			"auto_lock_timeout": self.auto_lock_timeout,
			"push_notifications": self.push_notifications,
			"notification_sound": self.notification_sound,
			"notification_vibration": self.notification_vibration,
			"debug_mode": self.debug_mode,
			"verbose_logging": self.verbose_logging,
		}
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
		"""Create settings from dictionary"""
		settings = cls()
		for key, value in data.items():
			if hasattr(settings, key):
				if key == "theme" and isinstance(value, str):
					setattr(settings, key, AppTheme(value))
				else:
					setattr(settings, key, value)
		return settings


@dataclass
class CacheInfo:
	"""Cache information and statistics"""
	total_size: int = 0  # bytes
	item_count: int = 0
	hit_rate: float = 0.0  # 0-100%
	last_cleanup: Optional[datetime] = None
	max_size: int = 100 * 1024 * 1024  # 100MB default
	
	def update_stats(self, size_change: int = 0, item_change: int = 0):
		"""Update cache statistics"""
		self.total_size += size_change
		self.item_count += item_change
		self.total_size = max(0, self.total_size)
		self.item_count = max(0, self.item_count)
	
	@property
	def is_full(self) -> bool:
		"""Check if cache is approaching limit"""
		return self.total_size >= (self.max_size * 0.9)
	
	@property
	def usage_percentage(self) -> float:
		"""Get cache usage as percentage"""
		if self.max_size == 0:
			return 0.0
		return (self.total_size / self.max_size) * 100


class AppState:
	"""Centralized application state management"""
	
	def __init__(self):
		self.logger = logging.getLogger(__name__)
		self._lock = threading.RLock()
		
		# Authentication state
		self._current_user: Optional[User] = None
		self._is_authenticated: bool = False
		self._auth_token: Optional[str] = None
		self._session_expiry: Optional[datetime] = None
		
		# Network and connectivity
		self._connection_info = ConnectionInfo()
		self._network_state = NetworkState.UNKNOWN
		
		# Synchronization
		self._sync_status = SyncStatus()
		
		# Application settings
		self._settings = AppSettings()
		
		# Cache information
		self._cache_info = CacheInfo()
		
		# Data caches
		self._workflows_cache: Dict[str, Workflow] = {}
		self._tasks_cache: Dict[str, Task] = {}
		self._notifications_cache: Dict[str, Notification] = {}
		self._workflow_instances_cache: Dict[str, WorkflowInstance] = {}
		
		# UI state
		self._current_screen: Optional[str] = None
		self._screen_history: List[str] = []
		self._loading_states: Set[str] = set()
		self._error_states: Dict[str, str] = {}
		
		# Event listeners
		self._listeners: Dict[str, List[Callable]] = {}
		
		# Persistence
		self._state_file = APP_DATA_DIR / "app_state.json"
		self._auto_save = True
		
		# Load saved state
		self._load_state()
		
		self.logger.info("App State initialized")
	
	# Authentication methods
	def set_current_user(self, user: User):
		"""Set current authenticated user"""
		with self._lock:
			self._current_user = user
			self._is_authenticated = True
			self._emit_event("user_changed", user)
			if self._auto_save:
				self._save_state()
	
	def clear_current_user(self):
		"""Clear current user and authentication"""
		with self._lock:
			old_user = self._current_user
			self._current_user = None
			self._is_authenticated = False
			self._auth_token = None
			self._session_expiry = None
			self._emit_event("user_cleared", old_user)
			if self._auto_save:
				self._save_state()
	
	@property
	def current_user(self) -> Optional[User]:
		"""Get current authenticated user"""
		return self._current_user
	
	@property
	def is_authenticated(self) -> bool:
		"""Check if user is authenticated"""
		return self._is_authenticated and self._current_user is not None
	
	def set_authenticated(self, authenticated: bool):
		"""Set authentication status"""
		with self._lock:
			self._is_authenticated = authenticated
			self._emit_event("auth_status_changed", authenticated)
			if self._auto_save:
				self._save_state()
	
	def set_auth_token(self, token: str, expiry: Optional[datetime] = None):
		"""Set authentication token"""
		with self._lock:
			self._auth_token = token
			self._session_expiry = expiry
			self._emit_event("token_changed", token)
	
	@property
	def auth_token(self) -> Optional[str]:
		"""Get authentication token"""
		return self._auth_token
	
	@property
	def is_session_expired(self) -> bool:
		"""Check if session is expired"""
		if not self._session_expiry:
			return False
		return datetime.utcnow() > self._session_expiry
	
	# Network state methods
	def set_network_connected(self, connected: bool, connection_info: Optional[Dict[str, Any]] = None):
		"""Set network connectivity status"""
		with self._lock:
			old_state = self._network_state
			self._connection_info.is_connected = connected
			
			if connection_info:
				self._connection_info.update(**connection_info)
			
			self._network_state = NetworkState.CONNECTED if connected else NetworkState.DISCONNECTED
			
			if old_state != self._network_state:
				self._emit_event("network_state_changed", {
					"old_state": old_state,
					"new_state": self._network_state,
					"connection_info": self._connection_info
				})
	
	@property
	def is_online(self) -> bool:
		"""Check if device is online"""
		return self._network_state == NetworkState.CONNECTED
	
	@property
	def network_state(self) -> NetworkState:
		"""Get current network state"""
		return self._network_state
	
	@property
	def connection_info(self) -> ConnectionInfo:
		"""Get connection information"""
		return self._connection_info
	
	# Sync state methods
	def set_sync_state(self, state: SyncState, **kwargs):
		"""Set synchronization state"""
		with self._lock:
			old_state = self._sync_status.state
			self._sync_status.state = state
			
			for key, value in kwargs.items():
				if hasattr(self._sync_status, key):
					setattr(self._sync_status, key, value)
			
			if old_state != state:
				self._emit_event("sync_state_changed", {
					"old_state": old_state,
					"new_state": state,
					"sync_status": self._sync_status
				})
	
	@property
	def sync_status(self) -> SyncStatus:
		"""Get synchronization status"""
		return self._sync_status
	
	def increment_pending_changes(self, count: int = 1):
		"""Increment pending changes count"""
		with self._lock:
			self._sync_status.pending_changes += count
			self._emit_event("pending_changes_updated", self._sync_status.pending_changes)
	
	def decrement_pending_changes(self, count: int = 1):
		"""Decrement pending changes count"""
		with self._lock:
			self._sync_status.pending_changes = max(0, self._sync_status.pending_changes - count)
			self._emit_event("pending_changes_updated", self._sync_status.pending_changes)
	
	# Settings methods
	@property
	def settings(self) -> AppSettings:
		"""Get application settings"""
		return self._settings
	
	def update_settings(self, **kwargs):
		"""Update application settings"""
		with self._lock:
			old_settings = self._settings.to_dict()
			
			for key, value in kwargs.items():
				if hasattr(self._settings, key):
					setattr(self._settings, key, value)
			
			new_settings = self._settings.to_dict()
			
			if old_settings != new_settings:
				self._emit_event("settings_changed", {
					"old_settings": old_settings,
					"new_settings": new_settings
				})
				
				if self._auto_save:
					self._save_state()
	
	# Cache methods
	def cache_workflow(self, workflow: Workflow):
		"""Cache workflow data"""
		with self._lock:
			self._workflows_cache[workflow.id] = workflow
			self._cache_info.update_stats(item_change=1)
			self._emit_event("workflow_cached", workflow)
	
	def get_cached_workflow(self, workflow_id: str) -> Optional[Workflow]:
		"""Get cached workflow"""
		return self._workflows_cache.get(workflow_id)
	
	def cache_task(self, task: Task):
		"""Cache task data"""
		with self._lock:
			self._tasks_cache[task.id] = task
			self._cache_info.update_stats(item_change=1)
			self._emit_event("task_cached", task)
	
	def get_cached_task(self, task_id: str) -> Optional[Task]:
		"""Get cached task"""
		return self._tasks_cache.get(task_id)
	
	def cache_notification(self, notification: Notification):
		"""Cache notification"""
		with self._lock:
			self._notifications_cache[notification.id] = notification
			self._cache_info.update_stats(item_change=1)
			self._emit_event("notification_cached", notification)
	
	def get_cached_notification(self, notification_id: str) -> Optional[Notification]:
		"""Get cached notification"""
		return self._notifications_cache.get(notification_id)
	
	def clear_cache(self, cache_type: Optional[str] = None):
		"""Clear cache data"""
		with self._lock:
			if cache_type == "workflows" or cache_type is None:
				self._workflows_cache.clear()
			if cache_type == "tasks" or cache_type is None:
				self._tasks_cache.clear()
			if cache_type == "notifications" or cache_type is None:
				self._notifications_cache.clear()
			if cache_type == "workflow_instances" or cache_type is None:
				self._workflow_instances_cache.clear()
			
			if cache_type is None:
				self._cache_info = CacheInfo()
			
			self._emit_event("cache_cleared", cache_type)
	
	@property
	def cache_info(self) -> CacheInfo:
		"""Get cache information"""
		return self._cache_info
	
	# UI state methods
	def set_current_screen(self, screen: str):
		"""Set current screen"""
		with self._lock:
			old_screen = self._current_screen
			
			if old_screen and old_screen != screen:
				self._screen_history.append(old_screen)
				# Keep history limited
				if len(self._screen_history) > 10:
					self._screen_history.pop(0)
			
			self._current_screen = screen
			self._emit_event("screen_changed", {
				"old_screen": old_screen,
				"new_screen": screen
			})
	
	@property
	def current_screen(self) -> Optional[str]:
		"""Get current screen"""
		return self._current_screen
	
	@property
	def previous_screen(self) -> Optional[str]:
		"""Get previous screen"""
		return self._screen_history[-1] if self._screen_history else None
	
	def can_go_back(self) -> bool:
		"""Check if can navigate back"""
		return len(self._screen_history) > 0
	
	def go_back(self) -> Optional[str]:
		"""Navigate to previous screen"""
		if self._screen_history:
			with self._lock:
				previous = self._screen_history.pop()
				old_screen = self._current_screen
				self._current_screen = previous
				self._emit_event("screen_changed", {
					"old_screen": old_screen,
					"new_screen": previous
				})
				return previous
		return None
	
	def set_loading(self, operation: str, loading: bool = True):
		"""Set loading state for operation"""
		with self._lock:
			if loading:
				self._loading_states.add(operation)
			else:
				self._loading_states.discard(operation)
			
			self._emit_event("loading_state_changed", {
				"operation": operation,
				"loading": loading,
				"all_loading": list(self._loading_states)
			})
	
	def is_loading(self, operation: Optional[str] = None) -> bool:
		"""Check if operation is loading"""
		if operation:
			return operation in self._loading_states
		return len(self._loading_states) > 0
	
	def set_error(self, operation: str, error: Optional[str] = None):
		"""Set error state for operation"""
		with self._lock:
			if error:
				self._error_states[operation] = error
			else:
				self._error_states.pop(operation, None)
			
			self._emit_event("error_state_changed", {
				"operation": operation,
				"error": error,
				"all_errors": dict(self._error_states)
			})
	
	def get_error(self, operation: str) -> Optional[str]:
		"""Get error for operation"""
		return self._error_states.get(operation)
	
	def clear_errors(self):
		"""Clear all error states"""
		with self._lock:
			self._error_states.clear()
			self._emit_event("errors_cleared", {})
	
	# Event system
	def add_listener(self, event_type: str, callback: Callable):
		"""Add event listener"""
		if event_type not in self._listeners:
			self._listeners[event_type] = []
		self._listeners[event_type].append(callback)
	
	def remove_listener(self, event_type: str, callback: Callable):
		"""Remove event listener"""
		if event_type in self._listeners:
			try:
				self._listeners[event_type].remove(callback)
			except ValueError:
				pass
	
	def _emit_event(self, event_type: str, data: Any):
		"""Emit event to listeners"""
		if event_type in self._listeners:
			for callback in self._listeners[event_type]:
				try:
					# Run callback in background to avoid blocking
					asyncio.create_task(self._run_callback(callback, event_type, data))
				except Exception as e:
					self.logger.error(f"Error in event callback for {event_type}: {e}")
	
	async def _run_callback(self, callback: Callable, event_type: str, data: Any):
		"""Run callback safely"""
		try:
			if asyncio.iscoroutinefunction(callback):
				await callback(event_type, data)
			else:
				callback(event_type, data)
		except Exception as e:
			self.logger.error(f"Error executing callback for {event_type}: {e}")
	
	# Persistence methods
	def _save_state(self):
		"""Save application state to disk"""
		try:
			state_data = {
				"user": self._current_user.to_dict() if self._current_user else None,
				"is_authenticated": self._is_authenticated,
				"settings": self._settings.to_dict(),
				"current_screen": self._current_screen,
				"screen_history": self._screen_history,
				"saved_at": datetime.utcnow().isoformat()
			}
			
			with open(self._state_file, 'w') as f:
				json.dump(state_data, f, indent=2)
				
		except Exception as e:
			self.logger.error(f"Failed to save app state: {e}")
	
	def _load_state(self):
		"""Load application state from disk"""
		try:
			if self._state_file.exists():
				with open(self._state_file, 'r') as f:
					state_data = json.load(f)
				
				# Load user
				if state_data.get("user"):
					self._current_user = User.from_dict(state_data["user"])
				
				# Load authentication status
				self._is_authenticated = state_data.get("is_authenticated", False)
				
				# Load settings
				if state_data.get("settings"):
					self._settings = AppSettings.from_dict(state_data["settings"])
				
				# Load UI state
				self._current_screen = state_data.get("current_screen")
				self._screen_history = state_data.get("screen_history", [])
				
				self.logger.info("App state loaded from disk")
				
		except Exception as e:
			self.logger.error(f"Failed to load app state: {e}")
	
	def clear(self):
		"""Clear all application state"""
		with self._lock:
			self.clear_current_user()
			self._network_state = NetworkState.UNKNOWN
			self._connection_info = ConnectionInfo()
			self._sync_status = SyncStatus()
			self._settings = AppSettings()
			self._cache_info = CacheInfo()
			self.clear_cache()
			self._current_screen = None
			self._screen_history.clear()
			self._loading_states.clear()
			self._error_states.clear()
			
			self._emit_event("state_cleared", {})
			
			if self._auto_save:
				self._save_state()
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert app state to dictionary"""
		return {
			"user": self._current_user.to_dict() if self._current_user else None,
			"is_authenticated": self._is_authenticated,
			"network_state": self._network_state.value,
			"connection_info": {
				"is_connected": self._connection_info.is_connected,
				"connection_type": self._connection_info.connection_type,
				"signal_strength": self._connection_info.signal_strength,
			},
			"sync_status": {
				"state": self._sync_status.state.value,
				"pending_changes": self._sync_status.pending_changes,
				"sync_progress": self._sync_status.sync_progress,
			},
			"settings": self._settings.to_dict(),
			"cache_info": {
				"total_size": self._cache_info.total_size,
				"item_count": self._cache_info.item_count,
				"usage_percentage": self._cache_info.usage_percentage,
			},
			"current_screen": self._current_screen,
			"loading_operations": list(self._loading_states),
			"error_count": len(self._error_states),
		}