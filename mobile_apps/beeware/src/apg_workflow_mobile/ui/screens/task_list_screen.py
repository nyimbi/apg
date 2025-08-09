"""
Task List Screen for APG Workflow Mobile

Screen for viewing and managing tasks.

© 2025 Datacraft. All rights reserved.
"""

from typing import List, Dict, Any
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from .base_screen import BaseScreen


class TaskListScreen(BaseScreen):
	"""Task list and management screen"""
	
	def __init__(self, app, navigation):
		super().__init__(app, navigation)
		self.title = "Tasks"
		self.tasks: List[Dict[str, Any]] = []
		self.task_list_container = None
	
	async def _create_content(self):
		"""Create task list UI"""
		self.content = toga.ScrollContainer(style=Pack(flex=1, padding=10))
		
		main_box = toga.Box(style=Pack(direction=COLUMN))
		header = self._create_header("My Tasks", "Manage your assigned tasks")
		main_box.add(header)
		
		# Filter buttons
		filter_box = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 20, 0)))
		
		all_btn = toga.Button("All", on_press=self._on_filter_all,
							 style=Pack(padding=5, background_color='#2196F3', color='white'))
		pending_btn = toga.Button("Pending", on_press=self._on_filter_pending,
								 style=Pack(padding=5, background_color='#FF9800', color='white'))
		completed_btn = toga.Button("Completed", on_press=self._on_filter_completed,
								   style=Pack(padding=5, background_color='#4CAF50', color='white'))
		
		filter_box.add(all_btn)
		filter_box.add(pending_btn)
		filter_box.add(completed_btn)
		main_box.add(filter_box)
		
		# Task list
		self.task_list_container = toga.Box(style=Pack(direction=COLUMN))
		main_box.add(self.task_list_container)
		
		self.content.content = main_box
	
	async def _load_data(self):
		"""Load tasks from service"""
		try:
			if self.app.task_service:
				response = await self.app.task_service.get_tasks()
				if response.success:
					self.tasks = response.data.get('tasks', [])
		except Exception as e:
			await self._show_error(f"Failed to load tasks: {e}")
	
	async def _update_content(self):
		"""Update task list display"""
		self.task_list_container.clear()
		
		if not self.tasks:
			empty_label = toga.Label("No tasks found", 
									style=Pack(text_align='center', padding=20))
			self.task_list_container.add(empty_label)
			return
		
		for task in self.tasks:
			task_item = self._create_task_item(task)
			self.task_list_container.add(task_item)
	
	def _create_task_item(self, task: Dict[str, Any]) -> toga.Box:
		"""Create task list item"""
		item_box = toga.Box(style=Pack(direction=COLUMN, padding=10, background_color='white'))
		
		title_label = toga.Label(task.get('name', 'Unnamed Task'),
							    style=Pack(font_weight='bold'))
		item_box.add(title_label)
		
		if task.get('description'):
			desc_label = toga.Label(task['description'][:80] + '...',
								   style=Pack(color='#666666', padding=(5, 0)))
			item_box.add(desc_label)
		
		# Status and actions
		actions_row = toga.Box(style=Pack(direction=ROW))
		
		status_label = toga.Label(f"Status: {task.get('status', 'Unknown')}",
								 style=Pack(flex=1, color='#666666'))
		actions_row.add(status_label)
		
		view_btn = toga.Button("View", 
							  on_press=lambda x, t=task: self._on_view_task(t),
							  style=Pack(padding=5, background_color='#2196F3', color='white'))
		actions_row.add(view_btn)
		
		item_box.add(actions_row)
		return item_box
	
	async def _on_filter_all(self, widget):
		"""Show all tasks"""
		await self._update_content()
	
	async def _on_filter_pending(self, widget):
		"""Show pending tasks only"""
		# In a real implementation, this would filter the task list
		await self._show_info("Filter feature coming soon!")
	
	async def _on_filter_completed(self, widget):
		"""Show completed tasks only"""
		await self._show_info("Filter feature coming soon!")
	
	async def _on_view_task(self, task: Dict[str, Any]):
		"""Handle view task"""
		await self.navigation.navigate_to('task_detail', task_id=task['id'])