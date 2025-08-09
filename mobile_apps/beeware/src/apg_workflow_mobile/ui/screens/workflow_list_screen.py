"""
Workflow List Screen for APG Workflow Mobile

Screen for viewing and managing workflows.

© 2025 Datacraft. All rights reserved.
"""

from typing import List, Dict, Any
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from .base_screen import BaseScreen


class WorkflowListScreen(BaseScreen):
	"""Workflow list and management screen"""
	
	def __init__(self, app, navigation):
		super().__init__(app, navigation)
		self.title = "Workflows"
		self.workflows: List[Dict[str, Any]] = []
		self.filtered_workflows: List[Dict[str, Any]] = []
		self.search_input = None
		self.workflow_list_container = None
	
	async def _create_content(self):
		"""Create workflow list UI"""
		self.content = toga.ScrollContainer(style=Pack(flex=1, padding=10))
		
		main_box = toga.Box(style=Pack(direction=COLUMN))
		
		# Header
		header = self._create_header("Workflows", "Manage your workflow processes")
		main_box.add(header)
		
		# Search and filter section
		search_box = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 20, 0)))
		
		self.search_input = toga.TextInput(
			placeholder="Search workflows...",
			style=Pack(flex=1, padding=5)
		)
		search_box.add(self.search_input)
		
		search_btn = toga.Button(
			"Search",
			on_press=self._on_search,
			style=Pack(padding=5, background_color='#2196F3', color='white')
		)
		search_box.add(search_btn)
		
		main_box.add(search_box)
		
		# Action buttons
		actions_box = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 20, 0)))
		
		create_btn = toga.Button(
			"+ New Workflow",
			on_press=self._on_create_workflow,
			style=Pack(padding=10, background_color='#4CAF50', color='white')
		)
		actions_box.add(create_btn)
		
		refresh_btn = toga.Button(
			"🔄 Refresh",
			on_press=self._on_refresh,
			style=Pack(padding=10, background_color='#FF9800', color='white')
		)
		actions_box.add(refresh_btn)
		
		main_box.add(actions_box)
		
		# Workflow list
		self.workflow_list_container = toga.Box(style=Pack(direction=COLUMN))
		main_box.add(self.workflow_list_container)
		
		self.content.content = main_box
	
	async def _load_data(self):
		"""Load workflows from service"""
		try:
			if self.app.workflow_service:
				response = await self.app.workflow_service.get_workflows()
				if response.success:
					self.workflows = response.data.get('workflows', [])
					self.filtered_workflows = self.workflows.copy()
		except Exception as e:
			self.logger.error(f"Failed to load workflows: {e}")
			await self._show_error(f"Failed to load workflows: {e}")
	
	async def _update_content(self):
		"""Update workflow list display"""
		try:
			self.workflow_list_container.clear()
			
			if not self.filtered_workflows:
				empty_label = toga.Label(
					"No workflows found",
					style=Pack(text_align='center', padding=20, color='#999999')
				)
				self.workflow_list_container.add(empty_label)
				return
			
			for workflow in self.filtered_workflows:
				workflow_item = self._create_workflow_item(workflow)
				self.workflow_list_container.add(workflow_item)
		
		except Exception as e:
			self.logger.error(f"Failed to update workflow list: {e}")
	
	def _create_workflow_item(self, workflow: Dict[str, Any]) -> toga.Box:
		"""Create workflow list item"""
		item_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=10,
				background_color='white'
			)
		)
		
		# Title and status row
		title_row = toga.Box(style=Pack(direction=ROW))
		
		title_label = toga.Label(
			workflow.get('name', 'Unnamed Workflow'),
			style=Pack(flex=1, font_weight='bold')
		)
		title_row.add(title_label)
		
		status_label = toga.Label(
			workflow.get('status', 'Unknown').upper(),
			style=Pack(
				padding=(0, 5),
				background_color=self._get_status_color(workflow.get('status')),
				color='white'
			)
		)
		title_row.add(status_label)
		
		item_box.add(title_row)
		
		# Description
		if workflow.get('description'):
			desc_label = toga.Label(
				workflow['description'][:100] + ('...' if len(workflow.get('description', '')) > 100 else ''),
				style=Pack(padding=(5, 0), color='#666666')
			)
			item_box.add(desc_label)
		
		# Actions row
		actions_row = toga.Box(style=Pack(direction=ROW, padding=(10, 0, 0, 0)))
		
		view_btn = toga.Button(
			"View",
			on_press=lambda x, w=workflow: self._on_view_workflow(w),
			style=Pack(padding=5, background_color='#2196F3', color='white')
		)
		actions_row.add(view_btn)
		
		edit_btn = toga.Button(
			"Edit",
			on_press=lambda x, w=workflow: self._on_edit_workflow(w),
			style=Pack(padding=5, background_color='#FF9800', color='white')
		)
		actions_row.add(edit_btn)
		
		item_box.add(actions_row)
		
		return item_box
	
	def _get_status_color(self, status: str) -> str:
		"""Get color for workflow status"""
		colors = {
			'active': '#4CAF50',
			'draft': '#FF9800',
			'completed': '#2196F3',
			'cancelled': '#f44336'
		}
		return colors.get(status, '#999999')
	
	async def _on_search(self, widget):
		"""Handle search button"""
		try:
			search_term = self.search_input.value.lower()
			if search_term:
				self.filtered_workflows = [
					w for w in self.workflows
					if search_term in w.get('name', '').lower() or 
					   search_term in w.get('description', '').lower()
				]
			else:
				self.filtered_workflows = self.workflows.copy()
			
			await self._update_content()
		except Exception as e:
			self.logger.error(f"Search error: {e}")
	
	async def _on_create_workflow(self, widget):
		"""Handle create workflow button"""
		await self._show_info("Create workflow feature coming soon!")
	
	async def _on_refresh(self, widget):
		"""Handle refresh button"""
		await self.refresh()
	
	async def _on_view_workflow(self, workflow: Dict[str, Any]):
		"""Handle view workflow"""
		await self.navigation.navigate_to('workflow_detail', workflow_id=workflow['id'])
	
	async def _on_edit_workflow(self, workflow: Dict[str, Any]):
		"""Handle edit workflow"""
		await self._show_info("Edit workflow feature coming soon!")