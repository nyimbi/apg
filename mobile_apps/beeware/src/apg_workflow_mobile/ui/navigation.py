"""
Navigation Manager for APG Workflow Mobile

Handles screen navigation and routing.

© 2025 Datacraft. All rights reserved.
"""

import logging
from typing import Dict, Any, Optional, Type
import asyncio

import toga
from toga.style import Pack
from toga.style.pack import ROW

from ..utils.exceptions import NavigationException


class NavigationManager:
	"""Manages screen navigation and routing"""
	
	def __init__(self, app):
		self.app = app
		self.logger = logging.getLogger(__name__)
		
		# Screen registry
		self.screens: Dict[str, Any] = {}
		self.screen_instances: Dict[str, Any] = {}
		
		# Navigation state
		self.current_screen: Optional[str] = None
		self.navigation_history: list = []
		self.navigation_stack: list = []
		
		# Navigation bar
		self.navigation_bar: Optional[toga.Box] = None
		self.nav_buttons: Dict[str, toga.Button] = {}
		
		self.logger.info("Navigation Manager initialized")
	
	async def initialize_screens(self):
		"""Initialize and register all application screens"""
		try:
			# Import screens (delayed import to avoid circular dependencies)
			from .screens.login_screen import LoginScreen
			from .screens.dashboard_screen import DashboardScreen
			from .screens.workflow_list_screen import WorkflowListScreen
			from .screens.task_list_screen import TaskListScreen
			from .screens.notification_screen import NotificationScreen
			from .screens.settings_screen import SettingsScreen
			from .screens.profile_screen import ProfileScreen
			from .screens.workflow_detail_screen import WorkflowDetailScreen
			from .screens.task_detail_screen import TaskDetailScreen
			
			# Register screens
			self.register_screen('login', LoginScreen)
			self.register_screen('dashboard', DashboardScreen)
			self.register_screen('workflows', WorkflowListScreen)
			self.register_screen('tasks', TaskListScreen)
			self.register_screen('notifications', NotificationScreen)
			self.register_screen('settings', SettingsScreen)
			self.register_screen('profile', ProfileScreen)
			self.register_screen('workflow_detail', WorkflowDetailScreen)
			self.register_screen('task_detail', TaskDetailScreen)
			
			# Create navigation bar
			await self._create_navigation_bar()
			
			self.logger.info("All screens registered successfully")
			
		except Exception as e:
			self.logger.error(f"Screen initialization failed: {e}")
			raise NavigationException(f"Failed to initialize screens: {e}")
	
	def register_screen(self, name: str, screen_class: Type):
		"""Register a screen class"""
		self.screens[name] = screen_class
		self.logger.debug(f"Registered screen: {name}")
	
	async def navigate_to(self, screen_name: str, **kwargs):
		"""Navigate to specified screen"""
		try:
			if screen_name not in self.screens:
				raise NavigationException(f"Screen not found: {screen_name}")
			
			self.logger.info(f"Navigating to screen: {screen_name}")
			
			# Get or create screen instance
			screen_instance = await self._get_screen_instance(screen_name)
			
			# Call screen's on_navigate method if available
			if hasattr(screen_instance, 'on_navigate'):
				await screen_instance.on_navigate(**kwargs)
			
			# Update navigation state
			if self.current_screen:
				self.navigation_history.append(self.current_screen)
			
			self.current_screen = screen_name
			self.navigation_stack.append(screen_name)
			
			# Update UI
			await self._update_screen_content(screen_instance)
			await self._update_navigation_bar()
			
			self.logger.info(f"Navigation to {screen_name} completed")
			
		except Exception as e:
			self.logger.error(f"Navigation failed: {e}")
			raise NavigationException(f"Navigation to {screen_name} failed: {e}")
	
	async def navigate_back(self):
		"""Navigate to previous screen"""
		try:
			if not self.navigation_history:
				self.logger.warning("No previous screen in history")
				return
			
			previous_screen = self.navigation_history.pop()
			self.navigation_stack.pop()  # Remove current screen
			
			await self.navigate_to(previous_screen)
			
		except Exception as e:
			self.logger.error(f"Navigate back failed: {e}")
			raise NavigationException(f"Navigate back failed: {e}")
	
	async def navigate_home(self):
		"""Navigate to home/dashboard screen"""
		try:
			# Clear navigation stack and go to dashboard
			self.navigation_history.clear()
			self.navigation_stack.clear()
			
			await self.navigate_to('dashboard')
			
		except Exception as e:
			self.logger.error(f"Navigate home failed: {e}")
			raise NavigationException(f"Navigate home failed: {e}")
	
	async def _get_screen_instance(self, screen_name: str):
		"""Get or create screen instance"""
		try:
			if screen_name not in self.screen_instances:
				screen_class = self.screens[screen_name]
				screen_instance = screen_class(app=self.app, navigation=self)
				
				# Initialize screen if it has an initialize method
				if hasattr(screen_instance, 'initialize'):
					await screen_instance.initialize()
				
				self.screen_instances[screen_name] = screen_instance
			
			return self.screen_instances[screen_name]
			
		except Exception as e:
			self.logger.error(f"Failed to get screen instance for {screen_name}: {e}")
			raise NavigationException(f"Failed to create screen {screen_name}: {e}")
	
	async def _update_screen_content(self, screen_instance):
		"""Update the main content area with new screen"""
		try:
			# Clear current content
			self.app.content_box.clear()
			
			# Add navigation bar if not login screen
			if self.current_screen != 'login' and self.navigation_bar:
				self.app.content_box.add(self.navigation_bar)
			
			# Add screen content
			if hasattr(screen_instance, 'get_content'):
				content = await screen_instance.get_content()
				self.app.content_box.add(content)
			elif hasattr(screen_instance, 'content'):
				self.app.content_box.add(screen_instance.content)
			else:
				# Fallback: create basic content
				content = toga.Label(
					f"Screen: {self.current_screen}",
					style=Pack(padding=20, text_align='center')
				)
				self.app.content_box.add(content)
			
		except Exception as e:
			self.logger.error(f"Failed to update screen content: {e}")
			raise NavigationException(f"Failed to update content: {e}")
	
	async def _create_navigation_bar(self):
		"""Create navigation bar with common actions"""
		try:
			self.navigation_bar = toga.Box(
				style=Pack(
					direction=ROW,
					padding=5,
					background_color='#2196F3'
				)
			)
			
			# Back button
			back_button = toga.Button(
				'← Back',
				on_press=self._on_back_pressed,
				style=Pack(
					padding=(5, 10),
					background_color='#1976D2',
					color='white'
				)
			)
			self.nav_buttons['back'] = back_button
			
			# Home button
			home_button = toga.Button(
				'🏠 Home',
				on_press=self._on_home_pressed,
				style=Pack(
					padding=(5, 10),
					background_color='#1976D2',
					color='white'
				)
			)
			self.nav_buttons['home'] = home_button
			
			# Screen title (flexible space)
			self.screen_title = toga.Label(
				'',
				style=Pack(
					flex=1,
					text_align='center',
					color='white',
					font_weight='bold',
					padding=(5, 10)
				)
			)
			
			# Menu button
			menu_button = toga.Button(
				'☰ Menu',
				on_press=self._on_menu_pressed,
				style=Pack(
					padding=(5, 10),
					background_color='#1976D2',
					color='white'
				)
			)
			self.nav_buttons['menu'] = menu_button
			
			# Add buttons to navigation bar
			self.navigation_bar.add(back_button)
			self.navigation_bar.add(home_button)
			self.navigation_bar.add(self.screen_title)
			self.navigation_bar.add(menu_button)
			
		except Exception as e:
			self.logger.error(f"Failed to create navigation bar: {e}")
			raise NavigationException(f"Failed to create navigation bar: {e}")
	
	async def _update_navigation_bar(self):
		"""Update navigation bar based on current screen"""
		try:
			if not self.navigation_bar or self.current_screen == 'login':
				return
			
			# Update screen title
			screen_titles = {
				'dashboard': 'Dashboard',
				'workflows': 'Workflows',
				'tasks': 'Tasks',
				'notifications': 'Notifications',
				'settings': 'Settings',
				'profile': 'Profile',
				'workflow_detail': 'Workflow Details',
				'task_detail': 'Task Details'
			}
			
			title = screen_titles.get(self.current_screen, self.current_screen.title())
			self.screen_title.text = title
			
			# Update back button visibility
			back_button = self.nav_buttons.get('back')
			if back_button:
				# Show back button if there's history or not on dashboard
				back_button.enabled = len(self.navigation_history) > 0 or self.current_screen != 'dashboard'
			
		except Exception as e:
			self.logger.error(f"Failed to update navigation bar: {e}")
	
	async def _on_back_pressed(self, widget):
		"""Handle back button press"""
		try:
			await self.navigate_back()
		except Exception as e:
			self.logger.error(f"Back button error: {e}")
	
	async def _on_home_pressed(self, widget):
		"""Handle home button press"""
		try:
			await self.navigate_home()
		except Exception as e:
			self.logger.error(f"Home button error: {e}")
	
	async def _on_menu_pressed(self, widget):
		"""Handle menu button press"""
		try:
			await self._show_navigation_menu()
		except Exception as e:
			self.logger.error(f"Menu button error: {e}")
	
	async def _show_navigation_menu(self):
		"""Show navigation menu with available screens"""
		try:
			# Create menu options based on authentication status
			menu_options = []
			
			if self.app.app_state.is_authenticated():
				menu_options = [
					('Dashboard', 'dashboard'),
					('Workflows', 'workflows'),
					('Tasks', 'tasks'),
					('Notifications', 'notifications'),
					('Profile', 'profile'),
					('Settings', 'settings'),
					('Logout', 'logout')
				]
			else:
				menu_options = [
					('Login', 'login')
				]
			
			# Show selection dialog
			# Note: Toga's selection dialog is limited, this is a simplified implementation
			# In a real implementation, you might create a custom menu screen
			
			option_labels = [option[0] for option in menu_options]
			
			# For now, just navigate to dashboard as example
			# In a real implementation, you'd show a proper menu dialog
			if self.current_screen != 'dashboard':
				await self.navigate_to('dashboard')
			
		except Exception as e:
			self.logger.error(f"Navigation menu error: {e}")
	
	def get_current_screen_instance(self):
		"""Get current screen instance"""
		if self.current_screen and self.current_screen in self.screen_instances:
			return self.screen_instances[self.current_screen]
		return None
	
	def get_navigation_history(self) -> list:
		"""Get navigation history"""
		return self.navigation_history.copy()
	
	def clear_navigation_history(self):
		"""Clear navigation history"""
		self.navigation_history.clear()
		self.navigation_stack.clear()
	
	def can_navigate_back(self) -> bool:
		"""Check if can navigate back"""
		return len(self.navigation_history) > 0
	
	async def refresh_current_screen(self):
		"""Refresh current screen"""
		try:
			if self.current_screen:
				screen_instance = await self._get_screen_instance(self.current_screen)
				if hasattr(screen_instance, 'refresh'):
					await screen_instance.refresh()
					await self._update_screen_content(screen_instance)
		except Exception as e:
			self.logger.error(f"Screen refresh error: {e}")
	
	async def show_modal(self, screen_name: str, **kwargs):
		"""Show screen as modal dialog"""
		try:
			# This is a simplified implementation
			# In a real app, you'd create a proper modal overlay
			await self.navigate_to(screen_name, **kwargs)
		except Exception as e:
			self.logger.error(f"Modal show error: {e}")
	
	async def close_modal(self):
		"""Close current modal"""
		try:
			await self.navigate_back()
		except Exception as e:
			self.logger.error(f"Modal close error: {e}")
	
	def set_screen_title(self, title: str):
		"""Set current screen title"""
		if self.screen_title:
			self.screen_title.text = title
	
	async def handle_deep_link(self, url: str):
		"""Handle deep link navigation"""
		try:
			# Parse deep link URL and navigate accordingly
			# This is a simplified implementation
			self.logger.info(f"Handling deep link: {url}")
			
			# Example: apg://workflow/123 -> navigate to workflow detail
			if url.startswith('apg://'):
				parts = url.replace('apg://', '').split('/')
				
				if len(parts) >= 2:
					entity_type = parts[0]
					entity_id = parts[1]
					
					if entity_type == 'workflow':
						await self.navigate_to('workflow_detail', workflow_id=entity_id)
					elif entity_type == 'task':
						await self.navigate_to('task_detail', task_id=entity_id)
					else:
						self.logger.warning(f"Unknown deep link entity type: {entity_type}")
				else:
					self.logger.warning(f"Invalid deep link format: {url}")
			else:
				self.logger.warning(f"Unsupported deep link scheme: {url}")
		
		except Exception as e:
			self.logger.error(f"Deep link handling error: {e}")
	
	def get_screen_names(self) -> list:
		"""Get list of registered screen names"""
		return list(self.screens.keys())
	
	def is_screen_registered(self, screen_name: str) -> bool:
		"""Check if screen is registered"""
		return screen_name in self.screens