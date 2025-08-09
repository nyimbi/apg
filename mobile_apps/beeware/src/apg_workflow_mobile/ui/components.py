"""
UI Components for APG Workflow Mobile

© 2025 Datacraft. All rights reserved.
"""

import toga
from typing import Optional, Callable, Any


class LoadingIndicator:
	"""Loading indicator component"""
	
	def __init__(self, message: str = "Loading..."):
		self.message = message
		self.widget = toga.ActivityIndicator()
	
	def show(self):
		"""Show loading indicator"""
		self.widget.start()
	
	def hide(self):
		"""Hide loading indicator"""
		self.widget.stop()


class ConfirmDialog:
	"""Confirmation dialog component"""
	
	def __init__(self, title: str, message: str, 
				 on_confirm: Optional[Callable] = None,
				 on_cancel: Optional[Callable] = None):
		self.title = title
		self.message = message
		self.on_confirm = on_confirm
		self.on_cancel = on_cancel
	
	async def show(self, app: toga.App) -> bool:
		"""Show confirmation dialog"""
		# This would be implemented with platform-specific dialogs
		# For now, return True as placeholder
		return True


class StatusCard:
	"""Status card component for displaying workflow/task status"""
	
	def __init__(self, title: str, status: str, icon: Optional[str] = None):
		self.title = title
		self.status = status
		self.icon = icon
		self.widget = toga.Box()
	
	def update_status(self, status: str):
		"""Update status display"""
		self.status = status


class QuickActionButton:
	"""Quick action button component"""
	
	def __init__(self, text: str, icon: Optional[str] = None, 
				 on_press: Optional[Callable] = None):
		self.text = text
		self.icon = icon
		self.on_press = on_press
		self.widget = toga.Button(text, on_press=on_press)


__all__ = [
	"LoadingIndicator",
	"ConfirmDialog", 
	"StatusCard",
	"QuickActionButton",
]