"""
Notification Service for APG Workflow Mobile

Handles notification management and delivery.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
import json

from ..models.notification import (
	Notification, NotificationPreferences, NotificationType, 
	NotificationPriority, NotificationChannel
)
from ..models.api_response import APIResponse
from ..services.api_service import APIService
from ..utils.exceptions import APIException, ValidationException
from ..utils.constants import URL_PATTERNS, MAX_NOTIFICATIONS, NOTIFICATION_CLEANUP_INTERVAL


class NotificationService:
	"""Service for notification management and delivery"""
	
	def __init__(self, api_service: APIService, app=None):
		self.api_service = api_service
		self.app = app
		self.logger = logging.getLogger(__name__)
		
		# Local notification storage
		self._local_notifications: List[Notification] = []
		self._notification_handlers: Dict[NotificationType, List[Callable]] = {}
		self._preferences: Optional[NotificationPreferences] = None
		
		# Cleanup task
		self._cleanup_task: Optional[asyncio.Task] = None
		
		self.logger.info("Notification Service initialized")
	
	async def initialize(self):
		"""Initialize notification service"""
		try:
			# Load user preferences
			await self._load_preferences()
			
			# Start cleanup task
			self._cleanup_task = asyncio.create_task(self._cleanup_notifications())
			
			self.logger.info("Notification service initialized")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize notification service: {e}")
	
	async def shutdown(self):
		"""Shutdown notification service"""
		try:
			if self._cleanup_task:
				self._cleanup_task.cancel()
				try:
					await self._cleanup_task
				except asyncio.CancelledError:
					pass
			
			self.logger.info("Notification service shutdown")
			
		except Exception as e:
			self.logger.error(f"Error during shutdown: {e}")
	
	async def get_notifications(self, params: Optional[Dict[str, Any]] = None) -> APIResponse:
		"""Get notifications from server"""
		try:
			self.logger.info("Fetching notifications")
			
			query_params = params or {}
			
			# Add default pagination if not provided
			if "page" not in query_params:
				query_params["page"] = 1
			if "limit" not in query_params:
				query_params["limit"] = 20
			
			response = await self.api_service.get(
				URL_PATTERNS["notifications"]["list"],
				params=query_params
			)
			
			if response.success and response.data:
				# Convert notification data to Notification objects
				notifications_data = response.data.get("notifications", [])
				notifications = []
				
				for notification_data in notifications_data:
					try:
						notification = Notification(**notification_data)
						notifications.append(notification)
					except Exception as e:
						self.logger.warning(f"Failed to parse notification {notification_data.get('id', 'unknown')}: {e}")
				
				# Update response data
				response.data = {"notifications": notifications}
				
				# Cache notifications locally
				for notification in notifications:
					self._add_local_notification(notification)
			
			self.logger.info(f"Fetched {len(notifications_data) if response.success else 0} notifications")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching notifications: {e}")
			raise APIException(f"Failed to fetch notifications: {e}")
	
	async def mark_as_read(self, notification_id: str) -> APIResponse:
		"""Mark notification as read"""
		try:
			self.logger.info(f"Marking notification as read: {notification_id}")
			
			response = await self.api_service.post(
				URL_PATTERNS["notifications"]["mark_read"].format(notification_id=notification_id)
			)
			
			if response.success:
				# Update local notification
				local_notification = self._get_local_notification(notification_id)
				if local_notification:
					local_notification.mark_as_read()
			
			self.logger.info(f"Marked notification as read: {notification_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error marking notification as read {notification_id}: {e}")
			raise APIException(f"Failed to mark notification as read: {e}")
	
	async def mark_all_as_read(self) -> APIResponse:
		"""Mark all notifications as read"""
		try:
			self.logger.info("Marking all notifications as read")
			
			response = await self.api_service.post(
				URL_PATTERNS["notifications"]["mark_all_read"]
			)
			
			if response.success:
				# Update all local notifications
				for notification in self._local_notifications:
					if not notification.is_read:
						notification.mark_as_read()
			
			self.logger.info("Marked all notifications as read")
			return response
			
		except Exception as e:
			self.logger.error(f"Error marking all notifications as read: {e}")
			raise APIException(f"Failed to mark all notifications as read: {e}")
	
	async def get_preferences(self) -> APIResponse:
		"""Get user notification preferences"""
		try:
			self.logger.info("Fetching notification preferences")
			
			response = await self.api_service.get(
				URL_PATTERNS["notifications"]["settings"]
			)
			
			if response.success and response.data:
				try:
					preferences = NotificationPreferences(**response.data)
					self._preferences = preferences
					response.data = preferences
					
				except Exception as e:
					self.logger.error(f"Failed to parse preferences data: {e}")
					raise ValidationException(f"Invalid preferences data: {e}")
			
			self.logger.info("Fetched notification preferences")
			return response
			
		except Exception as e:
			self.logger.error(f"Error fetching notification preferences: {e}")
			raise APIException(f"Failed to fetch notification preferences: {e}")
	
	async def update_preferences(self, preferences: Dict[str, Any]) -> APIResponse:
		"""Update user notification preferences"""
		try:
			self.logger.info("Updating notification preferences")
			
			response = await self.api_service.put(
				URL_PATTERNS["notifications"]["settings"],
				data=preferences
			)
			
			if response.success and response.data:
				try:
					updated_preferences = NotificationPreferences(**response.data)
					self._preferences = updated_preferences
					response.data = updated_preferences
					
				except Exception as e:
					self.logger.error(f"Failed to parse updated preferences data: {e}")
					raise ValidationException(f"Invalid preferences data: {e}")
			
			self.logger.info("Updated notification preferences")
			return response
			
		except Exception as e:
			self.logger.error(f"Error updating notification preferences: {e}")
			raise APIException(f"Failed to update notification preferences: {e}")
	
	def add_notification_handler(self, notification_type: NotificationType, handler: Callable):
		"""Add handler for specific notification type"""
		if notification_type not in self._notification_handlers:
			self._notification_handlers[notification_type] = []
		
		self._notification_handlers[notification_type].append(handler)
		self.logger.info(f"Added handler for notification type: {notification_type}")
	
	def remove_notification_handler(self, notification_type: NotificationType, handler: Callable):
		"""Remove handler for specific notification type"""
		if notification_type in self._notification_handlers:
			try:
				self._notification_handlers[notification_type].remove(handler)
				self.logger.info(f"Removed handler for notification type: {notification_type}")
			except ValueError:
				pass
	
	async def handle_incoming_notification(self, notification_data: Dict[str, Any]):
		"""Handle incoming notification from WebSocket or push"""
		try:
			notification = Notification(**notification_data)
			
			# Check if notification should be delivered based on preferences
			if self._preferences and not self._preferences.should_deliver_notification(notification):
				self.logger.info(f"Notification filtered by preferences: {notification.id}")
				return
			
			# Add to local storage
			self._add_local_notification(notification)
			
			# Call registered handlers
			await self._call_notification_handlers(notification)
			
			# Show notification in UI if app is available
			if self.app:
				await self._show_notification_in_ui(notification)
			
			self.logger.info(f"Handled incoming notification: {notification.id}")
			
		except Exception as e:
			self.logger.error(f"Error handling incoming notification: {e}")
	
	async def _call_notification_handlers(self, notification: Notification):
		"""Call registered handlers for notification type"""
		handlers = self._notification_handlers.get(notification.notification_type, [])
		
		for handler in handlers:
			try:
				if asyncio.iscoroutinefunction(handler):
					await handler(notification)
				else:
					handler(notification)
			except Exception as e:
				self.logger.error(f"Error in notification handler: {e}")
	
	async def _show_notification_in_ui(self, notification: Notification):
		"""Show notification in application UI"""
		try:
			# This would integrate with the UI framework to show notifications
			# For now, we'll just log and potentially emit an event
			
			self.logger.info(f"Showing notification in UI: {notification.title}")
			
			# Emit event if app state is available
			if self.app and hasattr(self.app, 'app_state'):
				self.app.app_state._emit_event("notification_received", notification.to_dict())
			
		except Exception as e:
			self.logger.error(f"Error showing notification in UI: {e}")
	
	def _add_local_notification(self, notification: Notification):
		"""Add notification to local storage"""
		# Remove existing notification with same ID
		self._local_notifications = [
			n for n in self._local_notifications if n.id != notification.id
		]
		
		# Add new notification at the beginning
		self._local_notifications.insert(0, notification)
		
		# Limit number of stored notifications
		if len(self._local_notifications) > MAX_NOTIFICATIONS:
			self._local_notifications = self._local_notifications[:MAX_NOTIFICATIONS]
		
		# Cache in app state if available
		if self.app and hasattr(self.app, 'app_state'):
			self.app.app_state.cache_notification(notification)
	
	def _get_local_notification(self, notification_id: str) -> Optional[Notification]:
		"""Get notification from local storage"""
		for notification in self._local_notifications:
			if notification.id == notification_id:
				return notification
		return None
	
	def get_local_notifications(self, 
									unread_only: bool = False,
									notification_type: Optional[NotificationType] = None,
									limit: Optional[int] = None) -> List[Notification]:
		"""Get notifications from local storage"""
		notifications = self._local_notifications.copy()
		
		# Filter by read status
		if unread_only:
			notifications = [n for n in notifications if not n.is_read]
		
		# Filter by type
		if notification_type:
			notifications = [n for n in notifications if n.notification_type == notification_type]
		
		# Apply limit
		if limit:
			notifications = notifications[:limit]
		
		return notifications
	
	def get_unread_count(self) -> int:
		"""Get count of unread notifications"""
		return len([n for n in self._local_notifications if not n.is_read])
	
	def clear_local_notifications(self):
		"""Clear all local notifications"""
		self._local_notifications.clear()
		self.logger.info("Cleared local notifications")
	
	async def _load_preferences(self):
		"""Load user notification preferences"""
		try:
			response = await self.get_preferences()
			if response.success and response.data:
				self._preferences = response.data
				self.logger.info("Loaded notification preferences")
		except Exception as e:
			self.logger.warning(f"Failed to load notification preferences: {e}")
			# Use default preferences
			self._preferences = NotificationPreferences(
				user_id="unknown",
				tenant_id="unknown"
			)
	
	async def _cleanup_notifications(self):
		"""Periodic cleanup of old notifications"""
		while True:
			try:
				await asyncio.sleep(NOTIFICATION_CLEANUP_INTERVAL)
				
				# Remove notifications older than 7 days
				cutoff_date = datetime.utcnow() - timedelta(days=7)
				initial_count = len(self._local_notifications)
				
				self._local_notifications = [
					n for n in self._local_notifications
					if n.created_at > cutoff_date
				]
				
				cleaned_count = initial_count - len(self._local_notifications)
				if cleaned_count > 0:
					self.logger.info(f"Cleaned up {cleaned_count} old notifications")
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Error during notification cleanup: {e}")
	
	async def create_notification(self, notification_data: Dict[str, Any]) -> APIResponse:
		"""Create new notification (admin/system use)"""
		try:
			self.logger.info("Creating notification")
			
			response = await self.api_service.post("/notifications", data=notification_data)
			
			if response.success and response.data:
				try:
					notification = Notification(**response.data)
					response.data = notification
					
					# Add to local storage
					self._add_local_notification(notification)
					
				except Exception as e:
					self.logger.error(f"Failed to parse created notification data: {e}")
					raise ValidationException(f"Invalid notification data: {e}")
			
			self.logger.info(f"Created notification: {response.data.id if response.success else 'failed'}")
			return response
			
		except ValidationException:
			raise
		except Exception as e:
			self.logger.error(f"Error creating notification: {e}")
			raise APIException(f"Failed to create notification: {e}")
	
	async def delete_notification(self, notification_id: str) -> APIResponse:
		"""Delete notification"""
		try:
			self.logger.info(f"Deleting notification: {notification_id}")
			
			response = await self.api_service.delete(f"/notifications/{notification_id}")
			
			if response.success:
				# Remove from local storage
				self._local_notifications = [
					n for n in self._local_notifications if n.id != notification_id
				]
			
			self.logger.info(f"Deleted notification: {notification_id}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error deleting notification {notification_id}: {e}")
			raise APIException(f"Failed to delete notification: {e}")
	
	async def test_notification(self, notification_type: NotificationType) -> APIResponse:
		"""Send test notification"""
		try:
			self.logger.info(f"Sending test notification: {notification_type}")
			
			test_data = {
				"notification_type": notification_type.value,
				"title": f"Test {notification_type.value} notification",
				"message": "This is a test notification to verify delivery.",
			}
			
			response = await self.api_service.post("/notifications/test", data=test_data)
			
			self.logger.info(f"Sent test notification: {notification_type}")
			return response
			
		except Exception as e:
			self.logger.error(f"Error sending test notification: {e}")
			raise APIException(f"Failed to send test notification: {e}")
	
	def get_notification_statistics(self) -> Dict[str, Any]:
		"""Get local notification statistics"""
		total_notifications = len(self._local_notifications)
		unread_notifications = self.get_unread_count()
		
		# Count by type
		type_counts = {}
		for notification in self._local_notifications:
			notification_type = notification.notification_type.value
			type_counts[notification_type] = type_counts.get(notification_type, 0) + 1
		
		# Count by priority
		priority_counts = {}
		for notification in self._local_notifications:
			priority = notification.priority.value
			priority_counts[priority] = priority_counts.get(priority, 0) + 1
		
		return {
			"total_notifications": total_notifications,
			"unread_notifications": unread_notifications,
			"read_notifications": total_notifications - unread_notifications,
			"type_counts": type_counts,
			"priority_counts": priority_counts,
			"oldest_notification": (
				self._local_notifications[-1].created_at.isoformat()
				if self._local_notifications else None
			),
			"newest_notification": (
				self._local_notifications[0].created_at.isoformat()
				if self._local_notifications else None
			),
		}