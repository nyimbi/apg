"""
Settings Screen for APG Workflow Mobile

Application settings and preferences.

© 2025 Datacraft. All rights reserved.
"""

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from .base_screen import BaseScreen


class SettingsScreen(BaseScreen):
	"""Application settings screen"""
	
	def __init__(self, app, navigation):
		super().__init__(app, navigation)
		self.title = "Settings"
		
		# Settings switches
		self.auto_sync_switch = None
		self.biometric_switch = None
		self.notifications_switch = None
		self.offline_mode_switch = None
	
	async def _create_content(self):
		"""Create settings UI"""
		self.content = toga.ScrollContainer(style=Pack(flex=1, padding=10))
		
		main_box = toga.Box(style=Pack(direction=COLUMN))
		header = self._create_header("Settings", "Configure your preferences")
		main_box.add(header)
		
		# Account section
		account_section = self._create_account_section()
		main_box.add(account_section)
		
		# Sync section
		sync_section = self._create_sync_section()
		main_box.add(sync_section)
		
		# Security section
		security_section = self._create_security_section()
		main_box.add(security_section)
		
		# App section
		app_section = self._create_app_section()
		main_box.add(app_section)
		
		# About section
		about_section = self._create_about_section()
		main_box.add(about_section)
		
		self.content.content = main_box
	
	def _create_account_section(self) -> toga.Box:
		"""Create account settings section"""
		section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 20, 0)))
		
		title = toga.Label("Account", style=Pack(font_size=18, font_weight='bold', padding=(0, 0, 10, 0)))
		section.add(title)
		
		# Profile button
		profile_btn = toga.Button("View Profile", on_press=self._on_view_profile,
								 style=Pack(padding=10, background_color='#2196F3', color='white', width=200))
		section.add(profile_btn)
		
		# Logout button
		logout_btn = toga.Button("Logout", on_press=self._on_logout,
								style=Pack(padding=10, background_color='#f44336', color='white', width=200))
		section.add(logout_btn)
		
		return section
	
	def _create_sync_section(self) -> toga.Box:
		"""Create sync settings section"""
		section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 20, 0)))
		
		title = toga.Label("Synchronization", style=Pack(font_size=18, font_weight='bold', padding=(0, 0, 10, 0)))
		section.add(title)
		
		# Auto sync setting
		auto_sync_row = toga.Box(style=Pack(direction=ROW, padding=5))
		auto_sync_label = toga.Label("Auto Sync", style=Pack(flex=1))
		# Note: Toga doesn't have a built-in Switch widget, using a button as placeholder
		self.auto_sync_switch = toga.Button("ON", on_press=self._on_toggle_auto_sync,
										   style=Pack(padding=5, background_color='#4CAF50', color='white'))
		auto_sync_row.add(auto_sync_label)
		auto_sync_row.add(self.auto_sync_switch)
		section.add(auto_sync_row)
		
		# Sync now button
		sync_now_btn = toga.Button("Sync Now", on_press=self._on_sync_now,
								  style=Pack(padding=10, background_color='#FF9800', color='white', width=200))
		section.add(sync_now_btn)
		
		return section
	
	def _create_security_section(self) -> toga.Box:
		"""Create security settings section"""
		section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 20, 0)))
		
		title = toga.Label("Security", style=Pack(font_size=18, font_weight='bold', padding=(0, 0, 10, 0)))
		section.add(title)
		
		# Biometric authentication
		biometric_row = toga.Box(style=Pack(direction=ROW, padding=5))
		biometric_label = toga.Label("Biometric Login", style=Pack(flex=1))
		self.biometric_switch = toga.Button("OFF", on_press=self._on_toggle_biometric,
										   style=Pack(padding=5, background_color='#f44336', color='white'))
		biometric_row.add(biometric_label)
		biometric_row.add(self.biometric_switch)
		section.add(biometric_row)
		
		# Change password button
		change_pwd_btn = toga.Button("Change Password", on_press=self._on_change_password,
									style=Pack(padding=10, background_color='#9C27B0', color='white', width=200))
		section.add(change_pwd_btn)
		
		return section
	
	def _create_app_section(self) -> toga.Box:
		"""Create app settings section"""
		section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 20, 0)))
		
		title = toga.Label("Application", style=Pack(font_size=18, font_weight='bold', padding=(0, 0, 10, 0)))
		section.add(title)
		
		# Notifications
		notifications_row = toga.Box(style=Pack(direction=ROW, padding=5))
		notifications_label = toga.Label("Push Notifications", style=Pack(flex=1))
		self.notifications_switch = toga.Button("ON", on_press=self._on_toggle_notifications,
											   style=Pack(padding=5, background_color='#4CAF50', color='white'))
		notifications_row.add(notifications_label)
		notifications_row.add(self.notifications_switch)
		section.add(notifications_row)
		
		# Offline mode
		offline_row = toga.Box(style=Pack(direction=ROW, padding=5))
		offline_label = toga.Label("Offline Mode", style=Pack(flex=1))
		self.offline_mode_switch = toga.Button("OFF", on_press=self._on_toggle_offline,
											  style=Pack(padding=5, background_color='#f44336', color='white'))
		offline_row.add(offline_label)
		offline_row.add(self.offline_mode_switch)
		section.add(offline_row)
		
		# Clear cache button
		clear_cache_btn = toga.Button("Clear Cache", on_press=self._on_clear_cache,
									 style=Pack(padding=10, background_color='#607D8B', color='white', width=200))
		section.add(clear_cache_btn)
		
		return section
	
	def _create_about_section(self) -> toga.Box:
		"""Create about section"""
		section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 20, 0)))
		
		title = toga.Label("About", style=Pack(font_size=18, font_weight='bold', padding=(0, 0, 10, 0)))
		section.add(title)
		
		# App version
		version_label = toga.Label(f"Version: {self.app.version}",
								  style=Pack(padding=5, color='#666666'))
		section.add(version_label)
		
		# Help button
		help_btn = toga.Button("Help & Support", on_press=self._on_help,
							  style=Pack(padding=10, background_color='#00BCD4', color='white', width=200))
		section.add(help_btn)
		
		return section
	
	async def _load_data(self):
		"""Load current settings"""
		# In a real implementation, this would load settings from storage
		pass
	
	# Event handlers
	async def _on_view_profile(self, widget):
		"""Handle view profile button"""
		await self.navigation.navigate_to('profile')
	
	async def _on_logout(self, widget):
		"""Handle logout button"""
		confirmed = await self._show_confirm("Are you sure you want to logout?")
		if confirmed:
			await self.app.logout()
	
	async def _on_toggle_auto_sync(self, widget):
		"""Handle auto sync toggle"""
		current_state = widget.text == "ON"
		new_state = not current_state
		widget.text = "ON" if new_state else "OFF"
		widget.style.background_color = '#4CAF50' if new_state else '#f44336'
		
		# Update sync service setting
		if self.app.sync_service:
			self.app.sync_service.auto_sync_enabled = new_state
	
	async def _on_sync_now(self, widget):
		"""Handle sync now button"""
		try:
			if self.app.sync_service:
				await self._show_loading("Syncing...")
				result = await self.app.sync_service.force_sync()
				await self._hide_loading()
				
				if result.get('status') == 'completed':
					await self._show_info("Sync completed successfully!")
				else:
					await self._show_error("Sync failed!")
		except Exception as e:
			await self._show_error(f"Sync error: {e}")
	
	async def _on_toggle_biometric(self, widget):
		"""Handle biometric toggle"""
		current_state = widget.text == "ON"
		new_state = not current_state
		
		if new_state:
			# Enable biometric - check if available
			if self.app.biometric_service and await self.app.biometric_service.is_biometric_available():
				widget.text = "ON"
				widget.style.background_color = '#4CAF50'
				await self._show_info("Biometric authentication enabled")
			else:
				await self._show_error("Biometric authentication not available on this device")
		else:
			widget.text = "OFF"
			widget.style.background_color = '#f44336'
			await self._show_info("Biometric authentication disabled")
	
	async def _on_change_password(self, widget):
		"""Handle change password button"""
		await self._show_info("Change password feature coming soon!")
	
	async def _on_toggle_notifications(self, widget):
		"""Handle notifications toggle"""
		current_state = widget.text == "ON"
		new_state = not current_state
		widget.text = "ON" if new_state else "OFF"
		widget.style.background_color = '#4CAF50' if new_state else '#f44336'
	
	async def _on_toggle_offline(self, widget):
		"""Handle offline mode toggle"""
		current_state = widget.text == "ON"
		new_state = not current_state
		widget.text = "ON" if new_state else "OFF"
		widget.style.background_color = '#4CAF50' if new_state else '#f44336'
	
	async def _on_clear_cache(self, widget):
		"""Handle clear cache button"""
		confirmed = await self._show_confirm("Are you sure you want to clear the cache?")
		if confirmed:
			try:
				if self.app.offline_service:
					await self.app.offline_service.clear_cache()
				if self.app.file_service:
					await self.app.file_service.clear_cache()
				await self._show_info("Cache cleared successfully!")
			except Exception as e:
				await self._show_error(f"Failed to clear cache: {e}")
	
	async def _on_help(self, widget):
		"""Handle help button"""
		await self._show_info("Help & Support feature coming soon!")