"""
Main APG Workflow Mobile Application

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .services.auth_service import AuthService
from .services.api_service import APIService
from .services.offline_service import OfflineService
from .services.notification_service import NotificationService
from .ui.screens.login_screen import LoginScreen
from .ui.screens.dashboard_screen import DashboardScreen
from .ui.screens.workflow_screen import WorkflowScreen
from .ui.screens.task_screen import TaskScreen
from .ui.screens.settings_screen import SettingsScreen
from .ui.components.navigation_drawer import NavigationDrawer
from .ui.components.status_bar import StatusBar
from .ui.components.loading_overlay import LoadingOverlay
from .models.user import User
from .models.app_state import AppState
from .utils.logger import setup_logging
from .utils.constants import APP_NAME, APP_VERSION


class APGWorkflowMobile(toga.App):
	"""Main APG Workflow Mobile Application class"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Initialize logging
		setup_logging()
		self.logger = logging.getLogger(__name__)
		
		# Initialize app state
		self.app_state = AppState()
		
		# Initialize services
		self.auth_service: Optional[AuthService] = None
		self.api_service: Optional[APIService] = None
		self.offline_service: Optional[OfflineService] = None
		self.notification_service: Optional[NotificationService] = None
		
		# UI Components
		self.main_box: Optional[toga.Box] = None
		self.content_box: Optional[toga.Box] = None
		self.navigation_drawer: Optional[NavigationDrawer] = None
		self.status_bar: Optional[StatusBar] = None
		self.loading_overlay: Optional[LoadingOverlay] = None
		
		# Screens
		self.login_screen: Optional[LoginScreen] = None
		self.dashboard_screen: Optional[DashboardScreen] = None
		self.workflow_screen: Optional[WorkflowScreen] = None
		self.task_screen: Optional[TaskScreen] = None
		self.settings_screen: Optional[SettingsScreen] = None
		
		# Current screen tracking
		self.current_screen: Optional[str] = None
		
		self.logger.info(f"Initializing {APP_NAME} v{APP_VERSION}")

	def startup(self):
		"""Called when the app starts up"""
		try:
			self.logger.info("Starting APG Workflow Mobile application")
			
			# Initialize services
			self._initialize_services()
			
			# Build UI
			self._build_ui()
			
			# Check for existing authentication
			asyncio.create_task(self._check_existing_auth())
			
		except Exception as e:
			self.logger.error(f"Failed to start application: {e}")
			self._show_error("Startup Failed", f"Failed to start application: {e}")

	def _initialize_services(self):
		"""Initialize all application services"""
		try:
			# API Service (core dependency)
			self.api_service = APIService(app=self)
			
			# Authentication Service
			self.auth_service = AuthService(api_service=self.api_service, app=self)
			
			# Offline Service
			self.offline_service = OfflineService(app=self)
			
			# Notification Service
			self.notification_service = NotificationService(app=self)
			
			self.logger.info("All services initialized successfully")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize services: {e}")
			raise

	def _build_ui(self):
		"""Build the main UI structure"""
		try:
			# Create main container
			self.main_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					flex=1,
					background_color="#f5f5f5"
				)
			)
			
			# Status bar
			self.status_bar = StatusBar(app=self)
			self.main_box.add(self.status_bar.create())
			
			# Content area
			self.content_box = toga.Box(
				style=Pack(
					direction=ROW,
					flex=1
				)
			)
			
			# Navigation drawer
			self.navigation_drawer = NavigationDrawer(app=self)
			self.content_box.add(self.navigation_drawer.create())
			
			# Screen container (will be populated with current screen)
			self.screen_container = toga.Box(
				style=Pack(
					direction=COLUMN,
					flex=1,
					padding=10
				)
			)
			self.content_box.add(self.screen_container)
			
			self.main_box.add(self.content_box)
			
			# Loading overlay (initially hidden)
			self.loading_overlay = LoadingOverlay(app=self)
			
			# Create main window
			self.main_window = toga.MainWindow(title=f"{APP_NAME} v{APP_VERSION}")
			self.main_window.content = self.main_box
			
			# Initialize screens (but don't add to UI yet)
			self._initialize_screens()
			
			# Show login screen initially
			self.show_login_screen()
			
			self.main_window.show()
			
			self.logger.info("UI built successfully")
			
		except Exception as e:
			self.logger.error(f"Failed to build UI: {e}")
			raise

	def _initialize_screens(self):
		"""Initialize all application screens"""
		try:
			self.login_screen = LoginScreen(app=self)
			self.dashboard_screen = DashboardScreen(app=self)
			self.workflow_screen = WorkflowScreen(app=self)
			self.task_screen = TaskScreen(app=self)
			self.settings_screen = SettingsScreen(app=self)
			
			self.logger.info("All screens initialized")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize screens: {e}")
			raise

	async def _check_existing_auth(self):
		"""Check if user is already authenticated"""
		try:
			if await self.auth_service.check_stored_auth():
				self.logger.info("Found existing authentication, loading dashboard")
				await self.show_dashboard_screen()
			else:
				self.logger.info("No existing authentication found")
				
		except Exception as e:
			self.logger.error(f"Error checking existing auth: {e}")

	def show_login_screen(self):
		"""Show the login screen"""
		try:
			self._clear_screen_container()
			self.screen_container.add(self.login_screen.create())
			self.current_screen = "login"
			self.navigation_drawer.hide()
			self.logger.info("Login screen displayed")
			
		except Exception as e:
			self.logger.error(f"Failed to show login screen: {e}")

	async def show_dashboard_screen(self):
		"""Show the dashboard screen"""
		try:
			self._clear_screen_container()
			await self.dashboard_screen.refresh_data()
			self.screen_container.add(self.dashboard_screen.create())
			self.current_screen = "dashboard"
			self.navigation_drawer.show()
			self.navigation_drawer.set_active_item("dashboard")
			self.logger.info("Dashboard screen displayed")
			
		except Exception as e:
			self.logger.error(f"Failed to show dashboard screen: {e}")

	async def show_workflow_screen(self, workflow_id: Optional[str] = None):
		"""Show the workflow management screen"""
		try:
			self._clear_screen_container()
			if workflow_id:
				await self.workflow_screen.load_workflow(workflow_id)
			else:
				await self.workflow_screen.refresh_data()
			self.screen_container.add(self.workflow_screen.create())
			self.current_screen = "workflows"
			self.navigation_drawer.set_active_item("workflows")
			self.logger.info("Workflow screen displayed")
			
		except Exception as e:
			self.logger.error(f"Failed to show workflow screen: {e}")

	async def show_task_screen(self, task_id: Optional[str] = None):
		"""Show the task management screen"""
		try:
			self._clear_screen_container()
			if task_id:
				await self.task_screen.load_task(task_id)
			else:
				await self.task_screen.refresh_data()
			self.screen_container.add(self.task_screen.create())
			self.current_screen = "tasks"
			self.navigation_drawer.set_active_item("tasks")
			self.logger.info("Task screen displayed")
			
		except Exception as e:
			self.logger.error(f"Failed to show task screen: {e}")

	def show_settings_screen(self):
		"""Show the settings screen"""
		try:
			self._clear_screen_container()
			self.screen_container.add(self.settings_screen.create())
			self.current_screen = "settings"
			self.navigation_drawer.set_active_item("settings")
			self.logger.info("Settings screen displayed")
			
		except Exception as e:
			self.logger.error(f"Failed to show settings screen: {e}")

	def _clear_screen_container(self):
		"""Clear the current screen from the container"""
		try:
			# Remove all children from screen container
			for child in list(self.screen_container.children):
				self.screen_container.remove(child)
				
		except Exception as e:
			self.logger.error(f"Failed to clear screen container: {e}")

	def show_loading(self, message: str = "Loading..."):
		"""Show loading overlay"""
		try:
			if self.loading_overlay:
				self.loading_overlay.show(message)
				
		except Exception as e:
			self.logger.error(f"Failed to show loading overlay: {e}")

	def hide_loading(self):
		"""Hide loading overlay"""
		try:
			if self.loading_overlay:
				self.loading_overlay.hide()
				
		except Exception as e:
			self.logger.error(f"Failed to hide loading overlay: {e}")

	async def handle_logout(self):
		"""Handle user logout"""
		try:
			self.logger.info("Handling user logout")
			
			# Show loading
			self.show_loading("Logging out...")
			
			# Logout via auth service
			await self.auth_service.logout()
			
			# Clear app state
			self.app_state.clear()
			
			# Return to login screen
			self.show_login_screen()
			
			# Hide loading
			self.hide_loading()
			
			self.logger.info("User logout completed")
			
		except Exception as e:
			self.logger.error(f"Failed to handle logout: {e}")
			self.hide_loading()
			self._show_error("Logout Failed", f"Failed to logout: {e}")

	async def handle_network_change(self, is_connected: bool):
		"""Handle network connectivity changes"""
		try:
			self.app_state.set_network_connected(is_connected)
			
			if is_connected:
				self.logger.info("Network connection restored")
				self.status_bar.set_network_status(True)
				
				# Sync offline data if available
				if self.offline_service:
					await self.offline_service.sync_pending_changes()
					
			else:
				self.logger.warning("Network connection lost")
				self.status_bar.set_network_status(False)
				
		except Exception as e:
			self.logger.error(f"Failed to handle network change: {e}")

	def _show_error(self, title: str, message: str):
		"""Show error dialog"""
		try:
			self.main_window.error_dialog(title, message)
			
		except Exception as e:
			self.logger.error(f"Failed to show error dialog: {e}")

	def _show_info(self, title: str, message: str):
		"""Show info dialog"""
		try:
			self.main_window.info_dialog(title, message)
			
		except Exception as e:
			self.logger.error(f"Failed to show info dialog: {e}")

	async def _show_confirm(self, title: str, message: str) -> bool:
		"""Show confirmation dialog"""
		try:
			return await self.main_window.confirm_dialog(title, message)
			
		except Exception as e:
			self.logger.error(f"Failed to show confirm dialog: {e}")
			return False

	def on_exit(self):
		"""Called when the app is closing"""
		try:
			self.logger.info("Application shutting down")
			
			# Save any pending data
			if self.offline_service:
				asyncio.create_task(self.offline_service.save_pending_changes())
				
			# Close any open connections
			if self.api_service:
				asyncio.create_task(self.api_service.close())
				
		except Exception as e:
			self.logger.error(f"Error during app shutdown: {e}")


def main():
	"""Main entry point for the application"""
	return APGWorkflowMobile(
		formal_name="APG Workflow Manager",
		app_id="co.ke.datacraft.apg-workflow-mobile",
		app_name="apg_workflow_mobile",
		description="Mobile workflow management application for APG",
		author="Nyimbi Odero",
		version="1.0.0",
		home_page="https://www.datacraft.co.ke",
	)


if __name__ == "__main__":
	app = main()
	app.main_loop()