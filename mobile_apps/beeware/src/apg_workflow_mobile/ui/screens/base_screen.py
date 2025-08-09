"""
Base Screen for APG Workflow Mobile

Base class for all application screens.

© 2025 Datacraft. All rights reserved.
"""

import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


class BaseScreen(ABC):
	"""Base class for all application screens"""
	
	def __init__(self, app, navigation):
		self.app = app
		self.navigation = navigation
		self.logger = logging.getLogger(self.__class__.__name__)
		
		# Screen state
		self.is_initialized = False
		self.is_visible = False
		self.content: Optional[toga.Widget] = None
		
		# Screen properties
		self.title = self.__class__.__name__.replace('Screen', '')
		self.requires_auth = True
		self.can_refresh = True
		
		self.logger.debug(f"Screen {self.title} created")
	
	async def initialize(self):
		"""Initialize screen (called once when screen is first created)"""
		try:
			if self.is_initialized:
				return
			
			self.logger.info(f"Initializing screen: {self.title}")
			
			# Create screen content
			await self._create_content()
			
			# Load initial data
			await self._load_data()
			
			self.is_initialized = True
			self.logger.info(f"Screen {self.title} initialized successfully")
			
		except Exception as e:
			self.logger.error(f"Screen initialization failed: {e}")
			raise
	
	@abstractmethod
	async def _create_content(self):
		"""Create screen UI content (must be implemented by subclasses)"""
		pass
	
	async def _load_data(self):
		"""Load initial screen data (can be overridden by subclasses)"""
		pass
	
	async def on_navigate(self, **kwargs):
		"""Called when navigating to this screen"""
		try:
			self.logger.debug(f"Navigating to screen: {self.title}")
			
			# Check authentication if required
			if self.requires_auth and not self.app.app_state.is_authenticated():
				self.logger.warning(f"Authentication required for {self.title}")
				await self.navigation.navigate_to('login')
				return
			
			# Update screen state
			self.is_visible = True
			
			# Handle navigation parameters
			await self._handle_navigation_params(**kwargs)
			
			# Refresh data if needed
			if self.can_refresh:
				await self.refresh()
			
		except Exception as e:
			self.logger.error(f"Navigation to {self.title} failed: {e}")
			raise
	
	async def _handle_navigation_params(self, **kwargs):
		"""Handle navigation parameters (can be overridden by subclasses)"""
		pass
	
	async def on_leave(self):
		"""Called when leaving this screen"""
		try:
			self.logger.debug(f"Leaving screen: {self.title}")
			self.is_visible = False
		except Exception as e:
			self.logger.error(f"Error leaving screen {self.title}: {e}")
	
	async def refresh(self):
		"""Refresh screen data"""
		try:
			if not self.can_refresh:
				return
			
			self.logger.debug(f"Refreshing screen: {self.title}")
			await self._load_data()
			await self._update_content()
			
		except Exception as e:
			self.logger.error(f"Screen refresh failed: {e}")
			await self._show_error(f"Failed to refresh screen: {e}")
	
	async def _update_content(self):
		"""Update screen content (can be overridden by subclasses)"""
		pass
	
	async def get_content(self) -> toga.Widget:
		"""Get screen content widget"""
		if not self.is_initialized:
			await self.initialize()
		
		return self.content
	
	def set_title(self, title: str):
		"""Set screen title"""
		self.title = title
		if self.navigation:
			self.navigation.set_screen_title(title)
	
	async def _show_loading(self, message: str = "Loading..."):
		"""Show loading indicator"""
		try:
			# Create loading overlay
			loading_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					alignment='center',
					padding=20
				)
			)
			
			loading_label = toga.Label(
				message,
				style=Pack(text_align='center', padding=(0, 0, 10, 0))
			)
			
			# Add activity indicator if available
			# Note: Toga may not have a built-in activity indicator on all platforms
			loading_box.add(loading_label)
			
			# Replace content temporarily
			if self.content:
				self.content.clear()
				self.content.add(loading_box)
			
		except Exception as e:
			self.logger.error(f"Show loading error: {e}")
	
	async def _hide_loading(self):
		"""Hide loading indicator"""
		try:
			# Restore original content
			await self._update_content()
		except Exception as e:
			self.logger.error(f"Hide loading error: {e}")
	
	async def _show_error(self, message: str, title: str = "Error"):
		"""Show error message to user"""
		try:
			await self.app.main_window.error_dialog(title, message)
		except Exception as e:
			self.logger.error(f"Show error dialog failed: {e}")
	
	async def _show_info(self, message: str, title: str = "Information"):
		"""Show info message to user"""
		try:
			await self.app.main_window.info_dialog(title, message)
		except Exception as e:
			self.logger.error(f"Show info dialog failed: {e}")
	
	async def _show_confirm(self, message: str, title: str = "Confirm") -> bool:
		"""Show confirmation dialog"""
		try:
			return await self.app.main_window.confirm_dialog(title, message)
		except Exception as e:
			self.logger.error(f"Show confirm dialog failed: {e}")
			return False
	
	def _create_header(self, title: str, subtitle: Optional[str] = None) -> toga.Box:
		"""Create standard screen header"""
		header_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=(0, 0, 20, 0)
			)
		)
		
		title_label = toga.Label(
			title,
			style=Pack(
				font_size=24,
				font_weight='bold',
				text_align='center',
				padding=(0, 0, 5, 0)
			)
		)
		header_box.add(title_label)
		
		if subtitle:
			subtitle_label = toga.Label(
				subtitle,
				style=Pack(
					font_size=16,
					text_align='center',
					color='#666666'
				)
			)
			header_box.add(subtitle_label)
		
		return header_box
	
	def _create_button(self, text: str, on_press, style_overrides: Optional[Dict] = None) -> toga.Button:
		"""Create standard button with consistent styling"""
		button_style = Pack(
			padding=10,
			background_color='#2196F3',
			color='white'
		)
		
		if style_overrides:
			for key, value in style_overrides.items():
				setattr(button_style, key, value)
		
		return toga.Button(
			text,
			on_press=on_press,
			style=button_style
		)
	
	def _create_input_field(self, placeholder: str, password: bool = False, style_overrides: Optional[Dict] = None) -> toga.TextInput:
		"""Create standard input field"""
		input_style = Pack(
			padding=5,
			flex=1
		)
		
		if style_overrides:
			for key, value in style_overrides.items():
				setattr(input_style, key, value)
		
		return toga.TextInput(
			placeholder=placeholder,
			style=input_style
		)
	
	def _create_label(self, text: str, style_overrides: Optional[Dict] = None) -> toga.Label:
		"""Create standard label"""
		label_style = Pack(
			padding=(5, 0),
			text_align='left'
		)
		
		if style_overrides:
			for key, value in style_overrides.items():
				setattr(label_style, key, value)
		
		return toga.Label(text, style=label_style)
	
	def _create_form_row(self, label_text: str, widget: toga.Widget) -> toga.Box:
		"""Create form row with label and widget"""
		row_box = toga.Box(
			style=Pack(
				direction=ROW,
				padding=(5, 0),
				alignment='center'
			)
		)
		
		label = toga.Label(
			label_text,
			style=Pack(
				width=120,
				text_align='right',
				padding=(0, 10, 0, 0)
			)
		)
		
		row_box.add(label)
		row_box.add(widget)
		
		return row_box
	
	def _create_card(self, content: toga.Widget, title: Optional[str] = None) -> toga.Box:
		"""Create card-style container"""
		card_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=15,
				background_color='#f5f5f5'
			)
		)
		
		if title:
			title_label = toga.Label(
				title,
				style=Pack(
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			card_box.add(title_label)
		
		card_box.add(content)
		
		return card_box
	
	def _create_list_item(self, title: str, subtitle: Optional[str] = None, on_press=None) -> toga.Box:
		"""Create list item with consistent styling"""
		item_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=10,
				background_color='white'
			)
		)
		
		title_label = toga.Label(
			title,
			style=Pack(
				font_weight='bold',
				padding=(0, 0, 2, 0)
			)
		)
		item_box.add(title_label)
		
		if subtitle:
			subtitle_label = toga.Label(
				subtitle,
				style=Pack(
					font_size=12,
					color='#666666'
				)
			)
			item_box.add(subtitle_label)
		
		# Add click handler if provided
		if on_press:
			# Note: Toga doesn't have click events on Box, 
			# in a real implementation you'd use a Button or handle touch events
			pass
		
		return item_box
	
	async def _async_handler(self, handler, *args, **kwargs):
		"""Wrapper for async event handlers"""
		try:
			if handler:
				if hasattr(handler, '__call__'):
					if hasattr(handler, '__code__') and handler.__code__.co_flags & 0x80:  # Check if async
						await handler(*args, **kwargs)
					else:
						handler(*args, **kwargs)
		except Exception as e:
			self.logger.error(f"Async handler error: {e}")
			await self._show_error(f"Action failed: {e}")
	
	def is_screen_visible(self) -> bool:
		"""Check if screen is currently visible"""
		return self.is_visible
	
	def get_screen_title(self) -> str:
		"""Get screen title"""
		return self.title
	
	def set_requires_auth(self, requires_auth: bool):
		"""Set whether screen requires authentication"""
		self.requires_auth = requires_auth
	
	def set_can_refresh(self, can_refresh: bool):
		"""Set whether screen can be refreshed"""
		self.can_refresh = can_refresh