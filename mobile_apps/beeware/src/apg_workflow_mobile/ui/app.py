"""
Main APG Workflow Mobile Application

BeeWare/Toga-based cross-platform mobile application.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from ..services.api_service import APIService
from ..services.workflow_service import WorkflowService
from ..services.task_service import TaskService
from ..services.notification_service import NotificationService
from ..services.offline_service import OfflineService
from ..services.biometric_service import BiometricService
from ..services.file_service import FileService
from ..services.sync_service import SyncService
from ..models.app_state import AppState
from ..utils.logger import setup_logging
from ..utils.constants import APP_NAME, APP_VERSION
from .navigation import NavigationManager
from .screens.login_screen import LoginScreen
from .screens.dashboard_screen import DashboardScreen
from .screens.workflow_list_screen import WorkflowListScreen
from .screens.task_list_screen import TaskListScreen
from .screens.settings_screen import SettingsScreen


class APGWorkflowApp(toga.App):
	"""Main APG Workflow Mobile Application"""
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Initialize logging
		self.logger = setup_logging()
		self.logger.info(f"Initializing {APP_NAME} v{APP_VERSION}")
		
		# Application state
		self.app_state = AppState()
		
		# Services
		self.api_service: Optional[APIService] = None
		self.workflow_service: Optional[WorkflowService] = None
		self.task_service: Optional[TaskService] = None
		self.notification_service: Optional[NotificationService] = None
		self.offline_service: Optional[OfflineService] = None
		self.biometric_service: Optional[BiometricService] = None
		self.file_service: Optional[FileService] = None
		self.sync_service: Optional[SyncService] = None
		
		# UI Components
		self.navigation_manager: Optional[NavigationManager] = None
		self.current_screen = None
		
		# Main UI containers
		self.main_box = None
		self.content_box = None
		self.status_bar = None
	
	async def startup(self):
		"""Initialize application on startup"""
		try:
			self.logger.info("Starting application initialization...")
			
			# Initialize services
			await self._initialize_services()
			
			# Initialize UI
			await self._initialize_ui()
			
			# Setup event handlers
			await self._setup_event_handlers()
			
			# Check authentication status
			await self._check_authentication()
			
			self.logger.info("Application initialization completed")
			
		except Exception as e:
			self.logger.error(f"Application startup failed: {e}")
			await self._show_error_dialog("Startup Error", f"Failed to initialize application: {e}")
	
	async def _initialize_services(self):
		"""Initialize all application services"""
		try:
			# Initialize API service
			self.api_service = APIService()
			
			# Initialize offline service
			self.offline_service = OfflineService(app=self)
			await self.offline_service.initialize()
			
			# Initialize other services
			self.workflow_service = WorkflowService(app=self)
			self.task_service = TaskService(app=self)
			self.notification_service = NotificationService(app=self)
			self.biometric_service = BiometricService()
			self.file_service = FileService(app=self)
			self.sync_service = SyncService(app=self)
			
			# Initialize biometric service
			await self.biometric_service.initialize()
			
			# Initialize sync service
			await self.sync_service.initialize()
			
			self.logger.info("All services initialized successfully")
			
		except Exception as e:
			self.logger.error(f"Service initialization failed: {e}")
			raise
	
	async def _initialize_ui(self):
		"""Initialize user interface"""
		try:
			# Create main application window
			self.main_window = toga.MainWindow(title=APP_NAME)
			
			# Create main layout
			self.main_box = toga.Box(style=Pack(direction=COLUMN))
			
			# Create navigation manager
			self.navigation_manager = NavigationManager(app=self)
			
			# Create status bar
			self.status_bar = toga.Box(
				style=Pack(
					direction=ROW,
					padding=5,
					background_color='#f0f0f0'
				)
			)
			
			# Status labels
			self.connection_status_label = toga.Label(
				"Offline",
				style=Pack(flex=1, text_align='left', padding=(0, 5))
			)
			
			self.sync_status_label = toga.Label(
				"No sync",
				style=Pack(flex=1, text_align='center', padding=(0, 5))
			)
			
			self.user_status_label = toga.Label(
				"Not logged in",
				style=Pack(flex=1, text_align='right', padding=(0, 5))
			)
			
			# Add status labels to status bar
			self.status_bar.add(self.connection_status_label)
			self.status_bar.add(self.sync_status_label)
			self.status_bar.add(self.user_status_label)
			
			# Create content area
			self.content_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					flex=1,
					padding=10
				)
			)
			
			# Add components to main layout
			self.main_box.add(self.content_box)
			self.main_box.add(self.status_bar)
			
			# Set main window content
			self.main_window.content = self.main_box
			
			# Initialize screens
			await self._initialize_screens()
			
			self.logger.info("UI initialized successfully")
			
		except Exception as e:
			self.logger.error(f"UI initialization failed: {e}")
			raise
	
	async def _initialize_screens(self):
		"""Initialize all application screens"""
		try:
			# Initialize screens with navigation manager
			await self.navigation_manager.initialize_screens()
			
			self.logger.info("All screens initialized")
			
		except Exception as e:
			self.logger.error(f"Screen initialization failed: {e}")
			raise
	
	async def _setup_event_handlers(self):
		"""Setup application event handlers"""
		try:
			# App state change handlers
			self.app_state.add_connection_change_callback(self._on_connection_change)
			self.app_state.add_auth_change_callback(self._on_auth_change)
			self.app_state.add_sync_change_callback(self._on_sync_change)
			
			# Sync service callbacks
			if self.sync_service:
				self.sync_service.add_sync_start_callback(self._on_sync_start)
				self.sync_service.add_sync_complete_callback(self._on_sync_complete)
				self.sync_service.add_sync_error_callback(self._on_sync_error)
			
			# Notification service callbacks
			if self.notification_service:
				self.notification_service.add_notification_callback(self._on_notification_received)
			
			self.logger.info("Event handlers setup completed")
			
		except Exception as e:
			self.logger.error(f"Event handler setup failed: {e}")
			raise
	
	async def _check_authentication(self):
		"""Check if user is already authenticated"""
		try:
			# Check for saved authentication
			if await self.app_state.is_authenticated():
				# Navigate to dashboard
				await self.navigation_manager.navigate_to('dashboard')
			else:
				# Navigate to login screen
				await self.navigation_manager.navigate_to('login')
			
		except Exception as e:
			self.logger.error(f"Authentication check failed: {e}")
			# Default to login screen
			await self.navigation_manager.navigate_to('login')
	
	async def login(self, username: str, password: str) -> bool:
		"""Perform user authentication"""
		try:
			self.logger.info(f"Attempting login for user: {username}")
			
			# Update status
			self.user_status_label.text = "Logging in..."
			
			# Authenticate with API service
			login_response = await self.api_service.login(username, password)
			
			if login_response.success:
				# Update app state
				user_data = login_response.data.get('user', {})
				await self.app_state.set_current_user(user_data)
				await self.app_state.set_authenticated(True)
				
				# Update UI
				self.user_status_label.text = f"Logged in as {user_data.get('name', username)}"
				
				# Start sync if network available
				if self.app_state.is_online():
					await self.sync_service.force_sync()
				
				self.logger.info("Login successful")
				return True
			else:
				self.user_status_label.text = "Login failed"
				self.logger.warning(f"Login failed: {login_response.message}")
				return False
				
		except Exception as e:
			self.user_status_label.text = "Login error"
			self.logger.error(f"Login error: {e}")
			return False
	
	async def logout(self):
		"""Perform user logout"""
		try:
			self.logger.info("Logging out user")
			
			# Logout from API service
			if self.api_service:
				await self.api_service.logout()
			
			# Clear app state
			await self.app_state.clear_current_user()
			await self.app_state.set_authenticated(False)
			
			# Update UI
			self.user_status_label.text = "Not logged in"
			
			# Navigate to login screen
			await self.navigation_manager.navigate_to('login')
			
			self.logger.info("Logout completed")
			
		except Exception as e:
			self.logger.error(f"Logout error: {e}")
	
	async def _on_connection_change(self, is_online: bool):
		"""Handle connection status change"""
		try:
			status_text = "Online" if is_online else "Offline"
			self.connection_status_label.text = status_text
			
			if is_online and self.sync_service:
				# Start sync when connection is restored
				await self.sync_service.force_sync()
			
		except Exception as e:
			self.logger.error(f"Connection change handler error: {e}")
	
	async def _on_auth_change(self, is_authenticated: bool):
		"""Handle authentication status change"""
		try:
			if is_authenticated:
				current_user = self.app_state.get_current_user()
				if current_user:
					self.user_status_label.text = f"Logged in as {current_user.get('name', 'User')}"
			else:
				self.user_status_label.text = "Not logged in"
			
		except Exception as e:
			self.logger.error(f"Auth change handler error: {e}")
	
	async def _on_sync_change(self, sync_info: Dict[str, Any]):
		"""Handle sync status change"""
		try:
			if sync_info.get('is_syncing', False):
				progress = sync_info.get('progress', 0)
				self.sync_status_label.text = f"Syncing... {progress:.0f}%"
			else:
				last_sync = sync_info.get('last_sync')
				if last_sync:
					self.sync_status_label.text = f"Last sync: {last_sync}"
				else:
					self.sync_status_label.text = "No sync"
			
		except Exception as e:
			self.logger.error(f"Sync change handler error: {e}")
	
	async def _on_sync_start(self):
		"""Handle sync start event"""
		self.sync_status_label.text = "Starting sync..."
	
	async def _on_sync_complete(self, result: Dict[str, Any]):
		"""Handle sync complete event"""
		stats = result.get('stats', {})
		successful = stats.get('successful_operations', 0)
		failed = stats.get('failed_operations', 0)
		
		if failed == 0:
			self.sync_status_label.text = f"Sync complete ({successful} items)"
		else:
			self.sync_status_label.text = f"Sync partial ({successful}/{successful + failed})"
	
	async def _on_sync_error(self, error: str):
		"""Handle sync error event"""
		self.sync_status_label.text = "Sync failed"
		self.logger.error(f"Sync error: {error}")
	
	async def _on_notification_received(self, notification):
		"""Handle incoming notification"""
		try:
			# Show notification to user
			await self._show_notification(
				notification.title,
				notification.message,
				notification.notification_type.value
			)
			
		except Exception as e:
			self.logger.error(f"Notification handler error: {e}")
	
	async def _show_notification(self, title: str, message: str, notification_type: str = "info"):
		"""Show notification to user"""
		try:
			# Use Toga's built-in notification system
			await self.main_window.info_dialog(title, message)
			
		except Exception as e:
			self.logger.error(f"Show notification error: {e}")
	
	async def _show_error_dialog(self, title: str, message: str):
		"""Show error dialog to user"""
		try:
			await self.main_window.error_dialog(title, message)
			
		except Exception as e:
			self.logger.error(f"Show error dialog error: {e}")
	
	async def _show_info_dialog(self, title: str, message: str):
		"""Show info dialog to user"""
		try:
			await self.main_window.info_dialog(title, message)
			
		except Exception as e:
			self.logger.error(f"Show info dialog error: {e}")
	
	async def _show_confirm_dialog(self, title: str, message: str) -> bool:
		"""Show confirmation dialog to user"""
		try:
			result = await self.main_window.confirm_dialog(title, message)
			return result
			
		except Exception as e:
			self.logger.error(f"Show confirm dialog error: {e}")
			return False
	
	def navigate_to_screen(self, screen_name: str, **kwargs):
		"""Navigate to specified screen"""
		try:
			asyncio.create_task(self.navigation_manager.navigate_to(screen_name, **kwargs))
		except Exception as e:
			self.logger.error(f"Navigation error: {e}")
	
	def get_current_screen_name(self) -> Optional[str]:
		"""Get current screen name"""
		return self.navigation_manager.current_screen if self.navigation_manager else None
	
	async def refresh_current_screen(self):
		"""Refresh current screen data"""
		try:
			if self.navigation_manager and self.navigation_manager.current_screen:
				current_screen = self.navigation_manager.get_current_screen_instance()
				if hasattr(current_screen, 'refresh'):
					await current_screen.refresh()
		except Exception as e:
			self.logger.error(f"Screen refresh error: {e}")
	
	async def shutdown(self):
		"""Shutdown application"""
		try:
			self.logger.info("Shutting down application...")
			
			# Stop sync service
			if self.sync_service:
				await self.sync_service.shutdown()
			
			# Close offline service
			if self.offline_service:
				await self.offline_service.close()
			
			# Close API service
			if self.api_service:
				await self.api_service.close()
			
			self.logger.info("Application shutdown completed")
			
		except Exception as e:
			self.logger.error(f"Shutdown error: {e}")
	
	def on_exit(self):
		"""Handle application exit"""
		asyncio.create_task(self.shutdown())


def main():
	"""Main entry point for the application"""
	app = APGWorkflowApp(
		f'APG Workflow Mobile',
		f'com.datacraft.apg.workflow.mobile',
		version=APP_VERSION,
		description='Mobile workflow management application'
	)
	
	return app


if __name__ == '__main__':
	app = main()
	app.main_loop()