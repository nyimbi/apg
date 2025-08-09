"""
Login Screen for APG Workflow Mobile

User authentication screen.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from typing import Optional

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .base_screen import BaseScreen
from ...utils.constants import APP_NAME


class LoginScreen(BaseScreen):
	"""User login screen"""
	
	def __init__(self, app, navigation):
		super().__init__(app, navigation)
		self.title = "Login"
		self.requires_auth = False
		
		# UI components
		self.username_input: Optional[toga.TextInput] = None
		self.password_input: Optional[toga.TextInput] = None
		self.login_button: Optional[toga.Button] = None
		self.biometric_login_button: Optional[toga.Button] = None
		self.status_label: Optional[toga.Label] = None
		
		# State
		self.is_logging_in = False
	
	async def _create_content(self):
		"""Create login screen UI"""
		try:
			# Main container
			self.content = toga.ScrollContainer(
				style=Pack(flex=1, padding=20)
			)
			
			main_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					alignment='center',
					padding=20
				)
			)
			
			# App logo/title
			logo_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					alignment='center',
					padding=(0, 0, 40, 0)
				)
			)
			
			app_title = toga.Label(
				APP_NAME,
				style=Pack(
					font_size=28,
					font_weight='bold',
					text_align='center',
					color='#2196F3',
					padding=(0, 0, 10, 0)
				)
			)
			logo_box.add(app_title)
			
			subtitle = toga.Label(
				"Mobile Workflow Management",
				style=Pack(
					font_size=16,
					text_align='center',
					color='#666666'
				)
			)
			logo_box.add(subtitle)
			main_box.add(logo_box)
			
			# Login form
			form_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					padding=20,
					background_color='white',
					width=300
				)
			)
			
			# Username field
			username_label = toga.Label(
				"Username:",
				style=Pack(
					font_weight='bold',
					padding=(0, 0, 5, 0)
				)
			)
			form_box.add(username_label)
			
			self.username_input = toga.TextInput(
				placeholder="Enter your username",
				style=Pack(
					padding=10,
					width=280
				)
			)
			form_box.add(self.username_input)
			
			# Password field
			password_label = toga.Label(
				"Password:",
				style=Pack(
					font_weight='bold',
					padding=(20, 0, 5, 0)
				)
			)
			form_box.add(password_label)
			
			# Note: Toga's TextInput doesn't have a built-in password mode
			# This is a limitation of the current Toga implementation
			self.password_input = toga.TextInput(
				placeholder="Enter your password",
				style=Pack(
					padding=10,
					width=280
				)
			)
			form_box.add(self.password_input)
			
			# Login button
			self.login_button = toga.Button(
				"Login",
				on_press=self._on_login_pressed,
				style=Pack(
					padding=15,
					background_color='#2196F3',
					color='white',
					font_size=16,
					font_weight='bold',
					width=280
				)
			)
			form_box.add(self.login_button)
			
			# Biometric login button (if available)
			if await self._is_biometric_available():
				self.biometric_login_button = toga.Button(
					"🔒 Biometric Login",
					on_press=self._on_biometric_login_pressed,
					style=Pack(
						padding=15,
						background_color='#4CAF50',
						color='white',
						font_size=14,
						width=280
					)
				)
				form_box.add(self.biometric_login_button)
			
			# Status label
			self.status_label = toga.Label(
				"",
				style=Pack(
					text_align='center',
					padding=(20, 0, 0, 0),
					color='#f44336'
				)
			)
			form_box.add(self.status_label)
			
			main_box.add(form_box)
			
			# Footer
			footer_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					alignment='center',
					padding=(40, 0, 0, 0)
				)
			)
			
			version_label = toga.Label(
				f"Version {self.app.version}",
				style=Pack(
					font_size=12,
					color='#999999',
					text_align='center'
				)
			)
			footer_box.add(version_label)
			
			copyright_label = toga.Label(
				"© 2025 Datacraft. All rights reserved.",
				style=Pack(
					font_size=10,
					color='#999999',
					text_align='center',
					padding=(5, 0, 0, 0)
				)
			)
			footer_box.add(copyright_label)
			
			main_box.add(footer_box)
			
			self.content.content = main_box
			
		except Exception as e:
			self.logger.error(f"Failed to create login UI: {e}")
			raise
	
	async def _is_biometric_available(self) -> bool:
		"""Check if biometric authentication is available"""
		try:
			if self.app.biometric_service:
				return await self.app.biometric_service.is_biometric_available()
			return False
		except Exception as e:
			self.logger.error(f"Biometric availability check failed: {e}")
			return False
	
	async def _on_login_pressed(self, widget):
		"""Handle login button press"""
		try:
			if self.is_logging_in:
				return
			
			username = self.username_input.value.strip()
			password = self.password_input.value.strip()
			
			if not username or not password:
				self.status_label.text = "Please enter both username and password"
				return
			
			await self._perform_login(username, password)
			
		except Exception as e:
			self.logger.error(f"Login button handler error: {e}")
			self.status_label.text = f"Login error: {e}"
	
	async def _on_biometric_login_pressed(self, widget):
		"""Handle biometric login button press"""
		try:
			if self.is_logging_in:
				return
			
			await self._perform_biometric_login()
			
		except Exception as e:
			self.logger.error(f"Biometric login error: {e}")
			self.status_label.text = f"Biometric login error: {e}"
	
	async def _perform_login(self, username: str, password: str):
		"""Perform username/password login"""
		try:
			self.is_logging_in = True
			self._set_login_ui_state(False)
			self.status_label.text = "Logging in..."
			
			# Perform login
			success = await self.app.login(username, password)
			
			if success:
				self.status_label.text = "Login successful!"
				
				# Clear password field for security
				self.password_input.value = ""
				
				# Navigate to dashboard
				await asyncio.sleep(0.5)  # Brief pause to show success message
				await self.navigation.navigate_to('dashboard')
			else:
				self.status_label.text = "Invalid username or password"
				self._set_login_ui_state(True)
			
		except Exception as e:
			self.logger.error(f"Login failed: {e}")
			self.status_label.text = f"Login failed: {e}"
			self._set_login_ui_state(True)
		
		finally:
			self.is_logging_in = False
	
	async def _perform_biometric_login(self):
		"""Perform biometric authentication"""
		try:
			self.is_logging_in = True
			self._set_login_ui_state(False)
			self.status_label.text = "Authenticating..."
			
			# Perform biometric authentication
			result = await self.app.biometric_service.authenticate(
				prompt="Please authenticate to login to APG Workflow"
			)
			
			if result.success:
				# For biometric login, we need to have stored credentials
				# In a real implementation, you'd retrieve stored user credentials
				# after successful biometric authentication
				
				# For demo purposes, assume we have stored credentials
				stored_username = await self._get_stored_username()
				
				if stored_username:
					# Perform silent login with stored credentials
					success = await self._perform_silent_login(stored_username)
					
					if success:
						self.status_label.text = "Biometric login successful!"
						await asyncio.sleep(0.5)
						await self.navigation.navigate_to('dashboard')
					else:
						self.status_label.text = "Biometric authentication succeeded but login failed"
						self._set_login_ui_state(True)
				else:
					self.status_label.text = "No stored credentials found. Please login with username/password first."
					self._set_login_ui_state(True)
			else:
				self.status_label.text = f"Biometric authentication failed: {result.error}"
				self._set_login_ui_state(True)
			
		except Exception as e:
			self.logger.error(f"Biometric login failed: {e}")
			self.status_label.text = f"Biometric login failed: {e}"
			self._set_login_ui_state(True)
		
		finally:
			self.is_logging_in = False
	
	async def _get_stored_username(self) -> Optional[str]:
		"""Get stored username for biometric login"""
		try:
			# In a real implementation, this would retrieve securely stored credentials
			# For demo purposes, return None
			return None
		except Exception as e:
			self.logger.error(f"Failed to get stored username: {e}")
			return None
	
	async def _perform_silent_login(self, username: str) -> bool:
		"""Perform silent login with stored credentials"""
		try:
			# In a real implementation, this would use stored encrypted credentials
			# and perform authentication with the server
			# For demo purposes, return False
			return False
		except Exception as e:
			self.logger.error(f"Silent login failed: {e}")
			return False
	
	def _set_login_ui_state(self, enabled: bool):
		"""Enable/disable login UI components"""
		try:
			if self.username_input:
				self.username_input.enabled = enabled
			if self.password_input:
				self.password_input.enabled = enabled
			if self.login_button:
				self.login_button.enabled = enabled
			if self.biometric_login_button:
				self.biometric_login_button.enabled = enabled
		except Exception as e:
			self.logger.error(f"Failed to set login UI state: {e}")
	
	async def on_navigate(self, **kwargs):
		"""Handle navigation to login screen"""
		try:
			# If already authenticated, redirect to dashboard
			if self.app.app_state.is_authenticated():
				await self.navigation.navigate_to('dashboard')
				return
			
			await super().on_navigate(**kwargs)
			
			# Clear any previous status
			if self.status_label:
				self.status_label.text = ""
			
			# Reset login state
			self.is_logging_in = False
			self._set_login_ui_state(True)
			
		except Exception as e:
			self.logger.error(f"Login screen navigation error: {e}")
	
	async def _handle_navigation_params(self, **kwargs):
		"""Handle navigation parameters"""
		try:
			# Handle logout message
			if kwargs.get('logout', False):
				self.status_label.text = "You have been logged out"
			
			# Handle error message
			error_message = kwargs.get('error')
			if error_message:
				self.status_label.text = error_message
			
			# Pre-fill username if provided
			username = kwargs.get('username')
			if username and self.username_input:
				self.username_input.value = username
			
		except Exception as e:
			self.logger.error(f"Navigation params handling error: {e}")
	
	async def _load_data(self):
		"""Load screen data (not needed for login screen)"""
		pass
	
	async def _update_content(self):
		"""Update screen content (not needed for login screen)"""
		pass