"""
Notification Screen for APG Workflow Mobile

Screen for viewing and managing notifications.

© 2025 Datacraft. All rights reserved.
"""

from typing import List, Dict, Any
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from .base_screen import BaseScreen


class NotificationScreen(BaseScreen):
	"""Notification management screen"""
	
	def __init__(self, app, navigation):
		super().__init__(app, navigation)
		self.title = "Notifications"
		self.notifications: List[Dict[str, Any]] = []
		self.notification_list_container = None
	
	async def _create_content(self):
		"""Create notification UI"""
		self.content = toga.ScrollContainer(style=Pack(flex=1, padding=10))
		
		main_box = toga.Box(style=Pack(direction=COLUMN))
		header = self._create_header("Notifications", "Stay updated with alerts")
		main_box.add(header)
		
		# Action buttons
		actions_box = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 20, 0)))
		
		mark_all_btn = toga.Button("Mark All Read", on_press=self._on_mark_all_read,
								  style=Pack(padding=10, background_color='#4CAF50', color='white'))
		clear_btn = toga.Button("Clear All", on_press=self._on_clear_all,
							   style=Pack(padding=10, background_color='#f44336', color='white'))
		
		actions_box.add(mark_all_btn)
		actions_box.add(clear_btn)
		main_box.add(actions_box)
		
		# Notification list
		self.notification_list_container = toga.Box(style=Pack(direction=COLUMN))
		main_box.add(self.notification_list_container)
		
		self.content.content = main_box
	
	async def _load_data(self):
		"""Load notifications from service"""
		try:
			if self.app.notification_service:
				response = await self.app.notification_service.get_notifications()
				if response.success:
					self.notifications = response.data.get('notifications', [])
		except Exception as e:
			await self._show_error(f"Failed to load notifications: {e}")
	
	async def _update_content(self):
		"""Update notification list display"""
		self.notification_list_container.clear()
		
		if not self.notifications:
			empty_label = toga.Label("No notifications",
									style=Pack(text_align='center', padding=20))
			self.notification_list_container.add(empty_label)
			return
		
		for notification in self.notifications:
			notif_item = self._create_notification_item(notification)
			self.notification_list_container.add(notif_item)
	
	def _create_notification_item(self, notification: Dict[str, Any]) -> toga.Box:
		"""Create notification list item"""
		item_box = toga.Box(style=Pack(direction=COLUMN, padding=10, 
									  background_color='#f5f5f5' if notification.get('read') else 'white'))
		
		title_label = toga.Label(notification.get('title', 'Notification'),
								style=Pack(font_weight='bold'))
		item_box.add(title_label)
		
		message_label = toga.Label(notification.get('message', ''),
								  style=Pack(color='#666666', padding=(5, 0)))
		item_box.add(message_label)
		
		# Actions
		actions_row = toga.Box(style=Pack(direction=ROW))
		
		time_label = toga.Label(notification.get('created_at', '')[:16],
							   style=Pack(flex=1, font_size=12, color='#999999'))
		actions_row.add(time_label)
		
		if not notification.get('read'):
			mark_read_btn = toga.Button("Mark Read",
									   on_press=lambda x, n=notification: self._on_mark_read(n),
									   style=Pack(padding=5, background_color='#2196F3', color='white'))
			actions_row.add(mark_read_btn)
		
		item_box.add(actions_row)
		return item_box
	
	async def _on_mark_all_read(self, widget):
		"""Mark all notifications as read"""
		try:
			if self.app.notification_service:
				await self.app.notification_service.mark_all_read()
				await self.refresh()
		except Exception as e:
			await self._show_error(f"Failed to mark notifications as read: {e}")
	
	async def _on_clear_all(self, widget):
		"""Clear all notifications"""
		confirmed = await self._show_confirm("Are you sure you want to clear all notifications?")
		if confirmed:
			try:
				if self.app.notification_service:
					await self.app.notification_service.clear_all()
					await self.refresh()
			except Exception as e:
				await self._show_error(f"Failed to clear notifications: {e}")
	
	async def _on_mark_read(self, notification: Dict[str, Any]):
		"""Mark single notification as read"""
		try:
			if self.app.notification_service:
				await self.app.notification_service.mark_read(notification['id'])
				await self.refresh()
		except Exception as e:
			await self._show_error(f"Failed to mark notification as read: {e}")