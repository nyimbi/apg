"""
Dashboard Screen for APG Workflow Mobile

Main dashboard with overview and quick actions.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from typing import Dict, Any, List

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .base_screen import BaseScreen


class DashboardScreen(BaseScreen):
	"""Main dashboard screen"""
	
	def __init__(self, app, navigation):
		super().__init__(app, navigation)
		self.title = "Dashboard"
		self.requires_auth = True
		
		# Dashboard data
		self.stats: Dict[str, Any] = {}
		self.recent_tasks: List[Dict[str, Any]] = []
		self.recent_workflows: List[Dict[str, Any]] = []
		self.notifications_count: int = 0
		
		# UI components
		self.stats_container = None
		self.quick_actions_container = None
		self.recent_items_container = None
	
	async def _create_content(self):
		"""Create dashboard UI"""
		try:
			self.content = toga.ScrollContainer(
				style=Pack(flex=1, padding=10)
			)
			
			main_box = toga.Box(
				style=Pack(direction=COLUMN)
			)
			
			# Welcome header
			welcome_text = f"Welcome back, {self._get_user_name()}!"
			header = self._create_header(welcome_text, "Here's your workflow overview")
			main_box.add(header)
			
			# Statistics cards
			self.stats_container = await self._create_stats_section()
			main_box.add(self.stats_container)
			
			# Quick actions
			self.quick_actions_container = await self._create_quick_actions_section()
			main_box.add(self.quick_actions_container)
			
			# Recent items
			self.recent_items_container = await self._create_recent_items_section()
			main_box.add(self.recent_items_container)
			
			self.content.content = main_box
			
		except Exception as e:
			self.logger.error(f"Failed to create dashboard UI: {e}")
			raise
	
	def _get_user_name(self) -> str:
		"""Get current user's name"""
		try:
			user = self.app.app_state.get_current_user()
			if user:
				return user.get('name', user.get('username', 'User'))
			return 'User'
		except:
			return 'User'
	
	async def _create_stats_section(self) -> toga.Box:
		"""Create statistics cards section"""
		stats_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=(0, 0, 20, 0)
			)
		)
		
		section_title = toga.Label(
			"Overview",
			style=Pack(
				font_size=18,
				font_weight='bold',
				padding=(0, 0, 10, 0)
			)
		)
		stats_box.add(section_title)
		
		# Stats cards container
		cards_container = toga.Box(
			style=Pack(direction=ROW, padding=5)
		)
		
		# Active workflows card
		workflows_card = self._create_stat_card(
			"Active Workflows",
			str(self.stats.get('active_workflows', 0)),
			"#2196F3"
		)
		cards_container.add(workflows_card)
		
		# Pending tasks card
		tasks_card = self._create_stat_card(
			"Pending Tasks",
			str(self.stats.get('pending_tasks', 0)),
			"#FF9800"
		)
		cards_container.add(tasks_card)
		
		# Notifications card
		notifications_card = self._create_stat_card(
			"Notifications",
			str(self.notifications_count),
			"#4CAF50"
		)
		cards_container.add(notifications_card)
		
		stats_box.add(cards_container)
		return stats_box
	
	def _create_stat_card(self, title: str, value: str, color: str) -> toga.Box:
		"""Create individual statistics card"""
		card = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=15,
				background_color='white',
				width=100,
				alignment='center'
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
	
	async def _create_quick_actions_section(self) -> toga.Box:
		"""Create quick actions section"""
		actions_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=(0, 0, 20, 0)
			)
		)
		
		section_title = toga.Label(
			"Quick Actions",
			style=Pack(
				font_size=18,
				font_weight='bold',
				padding=(0, 0, 10, 0)
			)
		)
		actions_box.add(section_title)
		
		# Action buttons container
		buttons_container = toga.Box(
			style=Pack(direction=COLUMN, padding=5)
		)
		
		# Create workflow button
		create_workflow_btn = toga.Button(
			"📋 Create New Workflow",
			on_press=self._on_create_workflow,
			style=Pack(
				padding=15,
				background_color='#2196F3',
				color='white',
				width=300
			)
		)
		buttons_container.add(create_workflow_btn)
		
		# View tasks button
		view_tasks_btn = toga.Button(
			"✅ View My Tasks",
			on_press=self._on_view_tasks,
			style=Pack(
				padding=15,
				background_color='#4CAF50',
				color='white',
				width=300
			)
		)
		buttons_container.add(view_tasks_btn)
		
		# View notifications button
		view_notifications_btn = toga.Button(
			"🔔 View Notifications",
			on_press=self._on_view_notifications,
			style=Pack(
				padding=15,
				background_color='#FF9800',
				color='white',
				width=300
			)
		)
		buttons_container.add(view_notifications_btn)
		
		# Sync data button
		sync_btn = toga.Button(
			"🔄 Sync Data",
			on_press=self._on_sync_data,
			style=Pack(
				padding=15,
				background_color='#9C27B0',
				color='white',
				width=300
			)
		)
		buttons_container.add(sync_btn)
		
		actions_box.add(buttons_container)
		return actions_box
	
	async def _create_recent_items_section(self) -> toga.Box:
		"""Create recent items section"""
		recent_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=(0, 0, 20, 0)
			)
		)
		
		section_title = toga.Label(
			"Recent Activity",
			style=Pack(
				font_size=18,
				font_weight='bold',
				padding=(0, 0, 10, 0)
			)
		)
		recent_box.add(section_title)
		
		# Recent workflows
		if self.recent_workflows:
			workflows_title = toga.Label(
				"Recent Workflows",
				style=Pack(
					font_size=14,
					font_weight='bold',
					padding=(10, 0, 5, 0)
				)
			)
			recent_box.add(workflows_title)
			
			for workflow in self.recent_workflows[:3]:  # Show max 3
				item = self._create_recent_item(
					workflow.get('name', 'Unnamed Workflow'),
					f"Status: {workflow.get('status', 'Unknown')}",
					lambda w=workflow: self._on_workflow_clicked(w)
				)
				recent_box.add(item)
		
		# Recent tasks
		if self.recent_tasks:
			tasks_title = toga.Label(
				"Recent Tasks",
				style=Pack(
					font_size=14,
					font_weight='bold',
					padding=(10, 0, 5, 0)
				)
			)
			recent_box.add(tasks_title)
			
			for task in self.recent_tasks[:3]:  # Show max 3
				item = self._create_recent_item(
					task.get('name', 'Unnamed Task'),
					f"Due: {task.get('due_date', 'No due date')}",
					lambda t=task: self._on_task_clicked(t)
				)
				recent_box.add(item)
		
		# Empty state
		if not self.recent_workflows and not self.recent_tasks:
			empty_label = toga.Label(
				"No recent activity",
				style=Pack(
					text_align='center',
					color='#999999',
					padding=20
				)
			)
			recent_box.add(empty_label)
		
		return recent_box
	
	def _create_recent_item(self, title: str, subtitle: str, on_click) -> toga.Box:
		"""Create recent item entry"""
		item_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=10,
				background_color='#f9f9f9',
				width=300
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
		
		subtitle_label = toga.Label(
			subtitle,
			style=Pack(
				font_size=12,
				color='#666666'
			)
		)
		item_box.add(subtitle_label)
		
		# Note: In a real implementation, you'd add click handling
		# Toga doesn't have built-in click events for Box widgets
		
		return item_box
	
	async def _load_data(self):
		"""Load dashboard data"""
		try:
			await self._show_loading("Loading dashboard...")
			
			# Load statistics
			await self._load_statistics()
			
			# Load recent items
			await self._load_recent_items()
			
			# Load notifications count
			await self._load_notifications_count()
			
			await self._hide_loading()
			
		except Exception as e:
			self.logger.error(f"Failed to load dashboard data: {e}")
			await self._show_error(f"Failed to load dashboard: {e}")
	
	async def _load_statistics(self):
		"""Load dashboard statistics"""
		try:
			if self.app.workflow_service:
				workflows_response = await self.app.workflow_service.get_workflows({'limit': 100})
				if workflows_response.success:
					workflows = workflows_response.data.get('workflows', [])
					self.stats['active_workflows'] = len([w for w in workflows if w.get('status') == 'active'])
			
			if self.app.task_service:
				tasks_response = await self.app.task_service.get_tasks({'status': 'pending', 'limit': 100})
				if tasks_response.success:
					self.stats['pending_tasks'] = len(tasks_response.data.get('tasks', []))
			
		except Exception as e:
			self.logger.error(f"Failed to load statistics: {e}")
	
	async def _load_recent_items(self):
		"""Load recent workflows and tasks"""
		try:
			# Load recent workflows
			if self.app.workflow_service:
				workflows_response = await self.app.workflow_service.get_workflows({
					'sort': 'modified_at',
					'order': 'desc',
					'limit': 5
				})
				if workflows_response.success:
					self.recent_workflows = workflows_response.data.get('workflows', [])
			
			# Load recent tasks
			if self.app.task_service:
				tasks_response = await self.app.task_service.get_tasks({
					'sort': 'modified_at',
					'order': 'desc',
					'limit': 5
				})
				if tasks_response.success:
					self.recent_tasks = tasks_response.data.get('tasks', [])
			
		except Exception as e:
			self.logger.error(f"Failed to load recent items: {e}")
	
	async def _load_notifications_count(self):
		"""Load unread notifications count"""
		try:
			if self.app.notification_service:
				notifications_response = await self.app.notification_service.get_notifications({
					'read': False,
					'limit': 100
				})
				if notifications_response.success:
					self.notifications_count = len(notifications_response.data.get('notifications', []))
			
		except Exception as e:
			self.logger.error(f"Failed to load notifications count: {e}")
	
	async def _update_content(self):
		"""Update dashboard content"""
		try:
			# Update statistics
			if self.stats_container:
				# Re-create stats section with updated data
				new_stats = await self._create_stats_section()
				# Replace old stats container
				# Note: This is simplified - in a real implementation you'd update individual components
			
		except Exception as e:
			self.logger.error(f"Failed to update dashboard content: {e}")
	
	# Event handlers
	async def _on_create_workflow(self, widget):
		"""Handle create workflow button"""
		try:
			# Navigate to workflow creation (for now, just go to workflows list)
			await self.navigation.navigate_to('workflows')
		except Exception as e:
			self.logger.error(f"Create workflow error: {e}")
	
	async def _on_view_tasks(self, widget):
		"""Handle view tasks button"""
		try:
			await self.navigation.navigate_to('tasks')
		except Exception as e:
			self.logger.error(f"View tasks error: {e}")
	
	async def _on_view_notifications(self, widget):
		"""Handle view notifications button"""
		try:
			await self.navigation.navigate_to('notifications')
		except Exception as e:
			self.logger.error(f"View notifications error: {e}")
	
	async def _on_sync_data(self, widget):
		"""Handle sync data button"""
		try:
			if self.app.sync_service:
				await self._show_loading("Syncing data...")
				result = await self.app.sync_service.force_sync()
				await self._hide_loading()
				
				if result.get('status') == 'completed':
					await self._show_info("Data synchronized successfully!")
					await self.refresh()
				else:
					await self._show_error(f"Sync failed: {result.get('error', 'Unknown error')}")
			else:
				await self._show_error("Sync service not available")
		except Exception as e:
			self.logger.error(f"Sync data error: {e}")
			await self._show_error(f"Sync failed: {e}")
	
	async def _on_workflow_clicked(self, workflow: Dict[str, Any]):
		"""Handle workflow item click"""
		try:
			await self.navigation.navigate_to('workflow_detail', workflow_id=workflow['id'])
		except Exception as e:
			self.logger.error(f"Workflow click error: {e}")
	
	async def _on_task_clicked(self, task: Dict[str, Any]):
		"""Handle task item click"""
		try:
			await self.navigation.navigate_to('task_detail', task_id=task['id'])
		except Exception as e:
			self.logger.error(f"Task click error: {e}")
	
	async def on_navigate(self, **kwargs):
		"""Handle navigation to dashboard"""
		try:
			await super().on_navigate(**kwargs)
			
			# Refresh data when returning to dashboard
			await self.refresh()
			
		except Exception as e:
			self.logger.error(f"Dashboard navigation error: {e}")