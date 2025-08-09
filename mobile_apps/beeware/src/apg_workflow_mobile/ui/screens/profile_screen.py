"""
Profile Screen for APG Workflow Mobile

User profile management and information.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from typing import Dict, Any, Optional

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .base_screen import BaseScreen


class ProfileScreen(BaseScreen):
	"""User profile management screen"""
	
	def __init__(self, app, navigation):
		super().__init__(app, navigation)
		self.title = "Profile"
		self.requires_auth = True
		
		# User data
		self.user_data: Optional[Dict[str, Any]] = None
		self.user_stats: Dict[str, Any] = {}
		self.user_preferences: Dict[str, Any] = {}
		
		# UI components
		self.profile_info_container = None
		self.stats_container = None
		self.preferences_container = None
		
		# Form inputs for editing
		self.edit_mode = False
		self.name_input = None
		self.email_input = None
		self.phone_input = None
		self.department_input = None
		self.title_input = None
	
	async def _create_content(self):
		"""Create profile UI"""
		try:
			self.content = toga.ScrollContainer(
				style=Pack(flex=1, padding=10)
			)
			
			main_box = toga.Box(
				style=Pack(direction=COLUMN)
			)
			
			# Header with user avatar and basic info
			self.header_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0), alignment='center')
			)
			main_box.add(self.header_container)
			
			# Action buttons
			self.actions_container = self._create_actions_section()
			main_box.add(self.actions_container)
			
			# Profile information section
			self.profile_info_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.profile_info_container)
			
			# User statistics section
			self.stats_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.stats_container)
			
			# Preferences section
			self.preferences_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.preferences_container)
			
			self.content.content = main_box
			
		except Exception as e:
			self.logger.error(f"Failed to create profile UI: {e}")
			raise
	
	def _create_actions_section(self) -> toga.Box:
		"""Create action buttons section"""
		actions_box = toga.Box(
			style=Pack(
				direction=ROW,
				padding=(0, 0, 20, 0),
				alignment='center'
			)
		)
		
		# Edit/Save button
		self.edit_save_button = toga.Button(
			"Edit Profile",
			on_press=self._on_edit_save_profile,
			style=Pack(
				padding=10,
				background_color='#2196F3',
				color='white',
				width=120
			)
		)
		actions_box.add(self.edit_save_button)
		
		# Change password button
		self.change_password_button = toga.Button(
			"Change Password",
			on_press=self._on_change_password,
			style=Pack(
				padding=10,
				background_color='#FF9800',
				color='white',
				width=140
			)
		)
		actions_box.add(self.change_password_button)
		
		# Refresh button
		self.refresh_button = toga.Button(
			"Refresh",
			on_press=self._on_refresh_profile,
			style=Pack(
				padding=10,
				background_color='#4CAF50',
				color='white',
				width=100
			)
		)
		actions_box.add(self.refresh_button)
		
		return actions_box
	
	async def _load_data(self):
		"""Load profile data"""
		try:
			await self._show_loading("Loading profile...")
			
			# Load user profile data
			await self._load_user_profile()
			
			# Load user statistics
			await self._load_user_statistics()
			
			# Load user preferences
			await self._load_user_preferences()
			
			await self._hide_loading()
			
		except Exception as e:
			self.logger.error(f"Failed to load profile data: {e}")
			await self._show_error(f"Failed to load profile: {e}")
	
	async def _load_user_profile(self):
		"""Load user profile information"""
		try:
			# Get current user from app state
			self.user_data = self.app.app_state.get_current_user()
			
			if not self.user_data:
				# Try to fetch from API
				if self.app.api_service:
					response = await self.app.api_service.get('/user/profile')
					if response.success:
						self.user_data = response.data.get('user', {})
						# Update app state with fresh data
						await self.app.app_state.set_current_user(self.user_data)
					else:
						raise ValueError(f"Failed to load profile: {response.message}")
				else:
					raise ValueError("No user data available and API service not accessible")
			
			self.logger.info(f"Loaded profile for user: {self.user_data.get('username', 'Unknown')}")
			
		except Exception as e:
			self.logger.error(f"Failed to load user profile: {e}")
			raise
	
	async def _load_user_statistics(self):
		"""Load user activity statistics"""
		try:
			if self.app.api_service:
				response = await self.app.api_service.get('/user/statistics')
				if response.success:
					self.user_stats = response.data.get('statistics', {})
					self.logger.info("Loaded user statistics")
				else:
					self.logger.warning(f"Failed to load statistics: {response.message}")
					# Use default stats if API fails
					self.user_stats = {
						'workflows_created': 0,
						'tasks_completed': 0,
						'tasks_assigned': 0,
						'login_count': 0,
						'last_login': None
					}
		except Exception as e:
			self.logger.error(f"Failed to load user statistics: {e}")
			# Don't raise - statistics are optional
			self.user_stats = {}
	
	async def _load_user_preferences(self):
		"""Load user preferences"""
		try:
			if self.app.api_service:
				response = await self.app.api_service.get('/user/preferences')
				if response.success:
					self.user_preferences = response.data.get('preferences', {})
					self.logger.info("Loaded user preferences")
				else:
					self.logger.warning(f"Failed to load preferences: {response.message}")
					# Use default preferences
					self.user_preferences = {
						'theme': 'light',
						'notifications': True,
						'language': 'en',
						'timezone': 'UTC'
					}
		except Exception as e:
			self.logger.error(f"Failed to load user preferences: {e}")
			# Don't raise - preferences are optional
			self.user_preferences = {}
	
	async def _update_content(self):
		"""Update all UI content with loaded data"""
		try:
			# Update header
			await self._update_header()
			
			# Update profile info
			await self._update_profile_info()
			
			# Update statistics
			await self._update_statistics_section()
			
			# Update preferences
			await self._update_preferences_section()
			
		except Exception as e:
			self.logger.error(f"Failed to update content: {e}")
	
	async def _update_header(self):
		"""Update header with user information"""
		try:
			self.header_container.clear()
			
			if self.user_data:
				# User avatar placeholder (using initials)
				name = self.user_data.get('name', self.user_data.get('username', 'User'))
				initials = ''.join([word[0].upper() for word in name.split()[:2]])
				
				avatar_box = toga.Box(
					style=Pack(
						width=80,
						height=80,
						background_color='#2196F3',
						alignment='center',
						padding=20
					)
				)
				
				avatar_label = toga.Label(
					initials,
					style=Pack(
						font_size=24,
						font_weight='bold',
						color='white',
						text_align='center'
					)
				)
				avatar_box.add(avatar_label)
				self.header_container.add(avatar_box)
				
				# User name
				name_label = toga.Label(
					name,
					style=Pack(
						font_size=24,
						font_weight='bold',
						text_align='center',
						padding=(10, 0, 5, 0)
					)
				)
				self.header_container.add(name_label)
				
				# User title/role
				title = self.user_data.get('title', self.user_data.get('role', 'User'))
				if title:
					title_label = toga.Label(
						title,
						style=Pack(
							font_size=16,
							text_align='center',
							color='#666666'
						)
					)
					self.header_container.add(title_label)
				
				# User status
				status = "Active" if self.user_data.get('active', True) else "Inactive"
				status_color = '#4CAF50' if status == "Active" else '#f44336'
				
				status_label = toga.Label(
					status,
					style=Pack(
						background_color=status_color,
						color='white',
						padding=5,
						text_align='center',
						font_weight='bold'
					)
				)
				self.header_container.add(status_label)
			
		except Exception as e:
			self.logger.error(f"Failed to update header: {e}")
	
	async def _update_profile_info(self):
		"""Update profile information section"""
		try:
			self.profile_info_container.clear()
			
			if not self.user_data:
				return
			
			# Section title
			info_title = toga.Label(
				"Profile Information",
				style=Pack(
					font_size=18,
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			self.profile_info_container.add(info_title)
			
			# Information card
			info_card = toga.Box(
				style=Pack(
					direction=COLUMN,
					padding=15,
					background_color='white'
				)
			)
			
			if self.edit_mode:
				# Edit mode - show input fields
				await self._create_edit_form(info_card)
			else:
				# View mode - show read-only information
				await self._create_info_display(info_card)
			
			self.profile_info_container.add(info_card)
			
		except Exception as e:
			self.logger.error(f"Failed to update profile info: {e}")
	
	async def _create_edit_form(self, container: toga.Box):
		"""Create edit form for profile"""
		try:
			# Name field
			name_row = self._create_form_row(
				"Full Name:",
				toga.TextInput(
					value=self.user_data.get('name', ''),
					style=Pack(flex=1, padding=5)
				)
			)
			self.name_input = name_row.children[1]
			container.add(name_row)
			
			# Email field
			email_row = self._create_form_row(
				"Email:",
				toga.TextInput(
					value=self.user_data.get('email', ''),
					style=Pack(flex=1, padding=5)
				)
			)
			self.email_input = email_row.children[1]
			container.add(email_row)
			
			# Phone field
			phone_row = self._create_form_row(
				"Phone:",
				toga.TextInput(
					value=self.user_data.get('phone', ''),
					style=Pack(flex=1, padding=5)
				)
			)
			self.phone_input = phone_row.children[1]
			container.add(phone_row)
			
			# Department field
			department_row = self._create_form_row(
				"Department:",
				toga.TextInput(
					value=self.user_data.get('department', ''),
					style=Pack(flex=1, padding=5)
				)
			)
			self.department_input = department_row.children[1]
			container.add(department_row)
			
			# Title field
			title_row = self._create_form_row(
				"Job Title:",
				toga.TextInput(
					value=self.user_data.get('title', ''),
					style=Pack(flex=1, padding=5)
				)
			)
			self.title_input = title_row.children[1]
			container.add(title_row)
			
		except Exception as e:
			self.logger.error(f"Failed to create edit form: {e}")
	
	async def _create_info_display(self, container: toga.Box):
		"""Create read-only information display"""
		try:
			# Basic information
			info_items = [
				("Username", self.user_data.get('username', 'N/A')),
				("Full Name", self.user_data.get('name', 'N/A')),
				("Email", self.user_data.get('email', 'N/A')),
				("Phone", self.user_data.get('phone', 'N/A')),
				("Department", self.user_data.get('department', 'N/A')),
				("Job Title", self.user_data.get('title', 'N/A')),
				("Employee ID", self.user_data.get('employee_id', 'N/A')),
				("Location", self.user_data.get('location', 'N/A')),
				("Manager", self.user_data.get('manager', 'N/A')),
				("Start Date", self.user_data.get('start_date', 'N/A')),
				("Last Login", self.user_data.get('last_login', 'N/A')),
				("Account Created", self.user_data.get('created_at', 'N/A'))
			]
			
			for label, value in info_items:
				if value and value != 'N/A':  # Only show fields with values
					info_row = self._create_info_row(label, str(value))
					container.add(info_row)
			
			# Roles and permissions
			roles = self.user_data.get('roles', [])
			if roles:
				roles_label = toga.Label(
					"Roles:",
					style=Pack(
						font_weight='bold',
						padding=(15, 0, 5, 0)
					)
				)
				container.add(roles_label)
				
				roles_text = ', '.join(roles) if isinstance(roles, list) else str(roles)
				roles_value = toga.Label(
					roles_text,
					style=Pack(
						padding=(0, 0, 0, 10),
						color='#333333'
					)
				)
				container.add(roles_value)
			
		except Exception as e:
			self.logger.error(f"Failed to create info display: {e}")
	
	def _create_info_row(self, label: str, value: str) -> toga.Box:
		"""Create information row"""
		row = toga.Box(
			style=Pack(
				direction=ROW,
				padding=(5, 0)
			)
		)
		
		label_widget = toga.Label(
			f"{label}:",
			style=Pack(
				width=120,
				font_weight='bold',
				text_align='right',
				padding=(0, 10, 0, 0)
			)
		)
		row.add(label_widget)
		
		value_widget = toga.Label(
			value,
			style=Pack(
				flex=1,
				color='#333333'
			)
		)
		row.add(value_widget)
		
		return row
	
	async def _update_statistics_section(self):
		"""Update user statistics section"""
		try:
			self.stats_container.clear()
			
			# Section title
			stats_title = toga.Label(
				"Activity Statistics",
				style=Pack(
					font_size=18,
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			self.stats_container.add(stats_title)
			
			if not self.user_stats:
				empty_label = toga.Label(
					"No statistics available",
					style=Pack(
						text_align='center',
						color='#999999',
						padding=20
					)
				)
				self.stats_container.add(empty_label)
				return
			
			# Statistics cards
			stats_cards = toga.Box(
				style=Pack(direction=ROW, padding=5)
			)
			
			# Workflows created
			workflows_card = self._create_stat_card(
				"Workflows Created",
				str(self.user_stats.get('workflows_created', 0)),
				"#2196F3"
			)
			stats_cards.add(workflows_card)
			
			# Tasks completed
			tasks_card = self._create_stat_card(
				"Tasks Completed",
				str(self.user_stats.get('tasks_completed', 0)),
				"#4CAF50"
			)
			stats_cards.add(tasks_card)
			
			# Tasks assigned
			assigned_card = self._create_stat_card(
				"Tasks Assigned",
				str(self.user_stats.get('tasks_assigned', 0)),
				"#FF9800"
			)
			stats_cards.add(assigned_card)
			
			self.stats_container.add(stats_cards)
			
			# Additional statistics
			additional_stats = toga.Box(
				style=Pack(
					direction=COLUMN,
					padding=10,
					background_color='#f5f5f5'
				)
			)
			
			login_count = self.user_stats.get('login_count', 0)
			login_label = toga.Label(
				f"Total Logins: {login_count}",
				style=Pack(padding=(0, 0, 5, 0))
			)
			additional_stats.add(login_label)
			
			last_login = self.user_stats.get('last_login')
			if last_login:
				last_login_label = toga.Label(
					f"Last Login: {last_login}",
					style=Pack(padding=(0, 0, 5, 0))
				)
				additional_stats.add(last_login_label)
			
			self.stats_container.add(additional_stats)
			
		except Exception as e:
			self.logger.error(f"Failed to update statistics section: {e}")
	
	def _create_stat_card(self, title: str, value: str, color: str) -> toga.Box:
		"""Create individual statistics card"""
		card = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=15,
				background_color='white',
				width=120,
				alignment='center',
				margin=(0, 5, 0, 0)
			)
		)
		
		value_label = toga.Label(
			value,
			style=Pack(
				font_size=24,
				font_weight='bold',
				color=color,
				text_align='center'
			)
		)
		card.add(value_label)
		
		title_label = toga.Label(
			title,
			style=Pack(
				font_size=12,
				text_align='center',
				color='#666666',
				padding=(5, 0, 0, 0)
			)
		)
		card.add(title_label)
		
		return card
	
	async def _update_preferences_section(self):
		"""Update user preferences section"""
		try:
			self.preferences_container.clear()
			
			# Section title
			prefs_title = toga.Label(
				"Preferences",
				style=Pack(
					font_size=18,
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			self.preferences_container.add(prefs_title)
			
			# Preferences card
			prefs_card = toga.Box(
				style=Pack(
					direction=COLUMN,
					padding=15,
					background_color='white'
				)
			)
			
			# Display current preferences
			if self.user_preferences:
				pref_items = [
					("Theme", self.user_preferences.get('theme', 'light').title()),
					("Language", self.user_preferences.get('language', 'en').upper()),
					("Timezone", self.user_preferences.get('timezone', 'UTC')),
					("Notifications", "Enabled" if self.user_preferences.get('notifications', True) else "Disabled"),
					("Email Notifications", "Enabled" if self.user_preferences.get('email_notifications', True) else "Disabled"),
					("Auto Sync", "Enabled" if self.user_preferences.get('auto_sync', True) else "Disabled")
				]
				
				for label, value in pref_items:
					pref_row = self._create_info_row(label, value)
					prefs_card.add(pref_row)
			else:
				no_prefs_label = toga.Label(
					"No preferences set",
					style=Pack(
						text_align='center',
						color='#999999',
						padding=20
					)
				)
				prefs_card.add(no_prefs_label)
			
			# Edit preferences button
			edit_prefs_btn = toga.Button(
				"Edit Preferences",
				on_press=self._on_edit_preferences,
				style=Pack(
					padding=10,
					background_color='#9C27B0',
					color='white',
					width=150
				)
			)
			prefs_card.add(edit_prefs_btn)
			
			self.preferences_container.add(prefs_card)
			
		except Exception as e:
			self.logger.error(f"Failed to update preferences section: {e}")
	
	# Event handlers
	async def _on_edit_save_profile(self, widget):
		"""Handle edit/save profile button"""
		try:
			if self.edit_mode:
				# Save changes
				await self._save_profile_changes()
				self.edit_mode = False
				self.edit_save_button.text = "Edit Profile"
				await self._update_content()
			else:
				# Enter edit mode
				self.edit_mode = True
				self.edit_save_button.text = "Save Changes"
				await self._update_content()
		except Exception as e:
			self.logger.error(f"Edit/save profile error: {e}")
			await self._show_error(f"Failed to save profile: {e}")
	
	async def _save_profile_changes(self):
		"""Save profile changes"""
		try:
			if not self.user_data:
				return
			
			# Collect changes from form inputs
			updates = {}
			
			if self.name_input and self.name_input.value.strip():
				updates['name'] = self.name_input.value.strip()
			
			if self.email_input and self.email_input.value.strip():
				updates['email'] = self.email_input.value.strip()
			
			if self.phone_input:
				updates['phone'] = self.phone_input.value.strip()
			
			if self.department_input:
				updates['department'] = self.department_input.value.strip()
			
			if self.title_input:
				updates['title'] = self.title_input.value.strip()
			
			# Validate email format
			if 'email' in updates:
				from ...utils.validators import validate_email
				is_valid, error = validate_email(updates['email'])
				if not is_valid:
					raise ValueError(f"Invalid email format: {error}")
			
			# Send update request
			if self.app.api_service and updates:
				response = await self.app.api_service.put('/user/profile', updates)
				
				if response.success:
					await self._show_info("Profile updated successfully!")
					
					# Update local user data
					self.user_data.update(updates)
					await self.app.app_state.set_current_user(self.user_data)
					
					# Reload profile data
					await self._load_user_profile()
				else:
					raise ValueError(f"Update failed: {response.message}")
			
		except Exception as e:
			self.logger.error(f"Failed to save profile changes: {e}")
			raise
	
	async def _on_change_password(self, widget):
		"""Handle change password button"""
		try:
			# In a real implementation, this would show a password change dialog
			# For now, show a simple info message
			await self._show_info(
				"Password change feature:\n\n"
				"1. Current password verification\n"
				"2. New password entry\n"
				"3. Password confirmation\n"
				"4. Secure password validation\n\n"
				"This feature will be implemented in a future update."
			)
		except Exception as e:
			self.logger.error(f"Change password error: {e}")
			await self._show_error(f"Change password error: {e}")
	
	async def _on_refresh_profile(self, widget):
		"""Handle refresh profile button"""
		try:
			await self.refresh()
			await self._show_info("Profile refreshed successfully!")
		except Exception as e:
			self.logger.error(f"Refresh profile error: {e}")
			await self._show_error(f"Failed to refresh profile: {e}")
	
	async def _on_edit_preferences(self, widget):
		"""Handle edit preferences button"""
		try:
			# In a real implementation, this would show a preferences editing dialog
			await self._show_info(
				"Preferences editing:\n\n"
				"• Theme selection (Light/Dark)\n"
				"• Language preferences\n"
				"• Timezone settings\n"
				"• Notification preferences\n"
				"• Auto-sync settings\n"
				"• Privacy settings\n\n"
				"This feature will be implemented in a future update."
			)
		except Exception as e:
			self.logger.error(f"Edit preferences error: {e}")
			await self._show_error(f"Edit preferences error: {e}")
	
	async def _handle_navigation_params(self, **kwargs):
		"""Handle navigation parameters"""
		try:
			# Check if we should refresh profile data
			refresh = kwargs.get('refresh', False)
			if refresh:
				await self.refresh()
		except Exception as e:
			self.logger.error(f"Navigation params error: {e}")
	
	async def on_navigate(self, **kwargs):
		"""Handle navigation to profile screen"""
		try:
			await super().on_navigate(**kwargs)
			
			# Always refresh profile data when navigating to this screen
			await self.refresh()
			
		except Exception as e:
			self.logger.error(f"Profile navigation error: {e}")
			await self._show_error(f"Failed to load profile: {e}")