"""
Workflow Detail Screen for APG Workflow Mobile

Detailed view and management of individual workflows.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .base_screen import BaseScreen
from ...models.workflow import Workflow, WorkflowStatus
from ...models.task import Task


class WorkflowDetailScreen(BaseScreen):
	"""Detailed workflow view and management screen"""
	
	def __init__(self, app, navigation):
		super().__init__(app, navigation)
		self.title = "Workflow Details"
		self.requires_auth = True
		
		# Data
		self.workflow: Optional[Workflow] = None
		self.workflow_id: Optional[str] = None
		self.tasks: List[Task] = []
		self.workflow_instances: List[Dict[str, Any]] = []
		
		# UI components
		self.workflow_info_container = None
		self.tasks_container = None
		self.instances_container = None
		self.actions_container = None
		
		# Form inputs for editing
		self.name_input = None
		self.description_input = None
		self.status_selection = None
		self.priority_selection = None
		self.edit_mode = False
	
	async def _create_content(self):
		"""Create workflow detail UI"""
		try:
			self.content = toga.ScrollContainer(
				style=Pack(flex=1, padding=10)
			)
			
			main_box = toga.Box(
				style=Pack(direction=COLUMN)
			)
			
			# Header with workflow title
			self.header_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.header_container)
			
			# Action buttons
			self.actions_container = self._create_actions_section()
			main_box.add(self.actions_container)
			
			# Workflow information section
			self.workflow_info_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.workflow_info_container)
			
			# Tasks section
			self.tasks_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.tasks_container)
			
			# Workflow instances section
			self.instances_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.instances_container)
			
			self.content.content = main_box
			
		except Exception as e:
			self.logger.error(f"Failed to create workflow detail UI: {e}")
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
			"Edit",
			on_press=self._on_edit_save_workflow,
			style=Pack(
				padding=10,
				background_color='#2196F3',
				color='white',
				width=80
			)
		)
		actions_box.add(self.edit_save_button)
		
		# Start workflow button
		self.start_button = toga.Button(
			"Start",
			on_press=self._on_start_workflow,
			style=Pack(
				padding=10,
				background_color='#4CAF50',
				color='white',
				width=80
			)
		)
		actions_box.add(self.start_button)
		
		# Pause/Resume button
		self.pause_resume_button = toga.Button(
			"Pause",
			on_press=self._on_pause_resume_workflow,
			style=Pack(
				padding=10,
				background_color='#FF9800',
				color='white',
				width=80
			)
		)
		actions_box.add(self.pause_resume_button)
		
		# Stop button
		self.stop_button = toga.Button(
			"Stop",
			on_press=self._on_stop_workflow,
			style=Pack(
				padding=10,
				background_color='#f44336',
				color='white',
				width=80
			)
		)
		actions_box.add(self.stop_button)
		
		# Export button
		self.export_button = toga.Button(
			"Export",
			on_press=self._on_export_workflow,
			style=Pack(
				padding=10,
				background_color='#9C27B0',
				color='white',
				width=80
			)
		)
		actions_box.add(self.export_button)
		
		return actions_box
	
	async def _load_data(self):
		"""Load workflow data"""
		try:
			if not self.workflow_id:
				raise ValueError("No workflow ID provided")
			
			await self._show_loading("Loading workflow details...")
			
			# Load workflow details
			await self._load_workflow_details()
			
			# Load workflow tasks
			await self._load_workflow_tasks()
			
			# Load workflow instances
			await self._load_workflow_instances()
			
			await self._hide_loading()
			
		except Exception as e:
			self.logger.error(f"Failed to load workflow data: {e}")
			await self._show_error(f"Failed to load workflow: {e}")
	
	async def _load_workflow_details(self):
		"""Load detailed workflow information"""
		try:
			if self.app.workflow_service:
				response = await self.app.workflow_service.get_workflow(self.workflow_id)
				if response.success:
					workflow_data = response.data.get('workflow')
					if workflow_data:
						self.workflow = Workflow(**workflow_data)
						self.logger.info(f"Loaded workflow: {self.workflow.name}")
					else:
						raise ValueError("Workflow data not found in response")
				else:
					raise ValueError(f"Failed to load workflow: {response.message}")
			else:
				raise ValueError("Workflow service not available")
				
		except Exception as e:
			self.logger.error(f"Failed to load workflow details: {e}")
			raise
	
	async def _load_workflow_tasks(self):
		"""Load tasks for this workflow"""
		try:
			if self.app.task_service:
				response = await self.app.task_service.get_tasks({
					'workflow_id': self.workflow_id,
					'limit': 50
				})
				if response.success:
					tasks_data = response.data.get('tasks', [])
					self.tasks = [Task(**task_data) for task_data in tasks_data]
					self.logger.info(f"Loaded {len(self.tasks)} tasks for workflow")
		except Exception as e:
			self.logger.error(f"Failed to load workflow tasks: {e}")
			# Don't raise - tasks are optional
	
	async def _load_workflow_instances(self):
		"""Load workflow execution instances"""
		try:
			if self.app.workflow_service:
				response = await self.app.workflow_service.get_workflow_instances(self.workflow_id)
				if response.success:
					self.workflow_instances = response.data.get('instances', [])
					self.logger.info(f"Loaded {len(self.workflow_instances)} instances")
		except Exception as e:
			self.logger.error(f"Failed to load workflow instances: {e}")
			# Don't raise - instances are optional
	
	async def _update_content(self):
		"""Update all UI content with loaded data"""
		try:
			# Update header
			await self._update_header()
			
			# Update workflow info
			await self._update_workflow_info()
			
			# Update tasks section
			await self._update_tasks_section()
			
			# Update instances section
			await self._update_instances_section()
			
			# Update action buttons
			await self._update_action_buttons()
			
		except Exception as e:
			self.logger.error(f"Failed to update content: {e}")
	
	async def _update_header(self):
		"""Update header with workflow information"""
		try:
			self.header_container.clear()
			
			if self.workflow:
				# Main title
				title_label = toga.Label(
					self.workflow.name,
					style=Pack(
						font_size=24,
						font_weight='bold',
						text_align='center',
						padding=(0, 0, 5, 0)
					)
				)
				self.header_container.add(title_label)
				
				# Status badge
				status_color = self._get_status_color(self.workflow.status)
				status_label = toga.Label(
					self.workflow.status.value.upper(),
					style=Pack(
						background_color=status_color,
						color='white',
						padding=5,
						text_align='center',
						font_weight='bold'
					)
				)
				self.header_container.add(status_label)
				
				# Set navigation title
				self.set_title(f"Workflow: {self.workflow.name}")
			
		except Exception as e:
			self.logger.error(f"Failed to update header: {e}")
	
	async def _update_workflow_info(self):
		"""Update workflow information section"""
		try:
			self.workflow_info_container.clear()
			
			if not self.workflow:
				return
			
			# Section title
			info_title = toga.Label(
				"Workflow Information",
				style=Pack(
					font_size=18,
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			self.workflow_info_container.add(info_title)
			
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
			
			self.workflow_info_container.add(info_card)
			
		except Exception as e:
			self.logger.error(f"Failed to update workflow info: {e}")
	
	async def _create_edit_form(self, container: toga.Box):
		"""Create edit form for workflow"""
		try:
			# Name field
			name_row = self._create_form_row(
				"Name:",
				toga.TextInput(
					value=self.workflow.name,
					style=Pack(flex=1, padding=5)
				)
			)
			self.name_input = name_row.children[1]
			container.add(name_row)
			
			# Description field
			desc_label = toga.Label(
				"Description:",
				style=Pack(font_weight='bold', padding=(10, 0, 5, 0))
			)
			container.add(desc_label)
			
			self.description_input = toga.MultilineTextInput(
				value=self.workflow.description or "",
				style=Pack(
					height=100,
					padding=5
				)
			)
			container.add(self.description_input)
			
			# Status selection
			status_row = toga.Box(style=Pack(direction=ROW, padding=(10, 0)))
			status_label = toga.Label(
				"Status:",
				style=Pack(width=120, font_weight='bold')
			)
			status_row.add(status_label)
			
			# Note: Toga doesn't have a built-in dropdown, using buttons as selection
			status_buttons = toga.Box(style=Pack(direction=ROW))
			
			for status in WorkflowStatus:
				is_selected = status == self.workflow.status
				status_btn = toga.Button(
					status.value.title(),
					on_press=lambda x, s=status: self._on_status_selected(s),
					style=Pack(
						padding=5,
						background_color='#2196F3' if is_selected else '#e0e0e0',
						color='white' if is_selected else 'black'
					)
				)
				status_buttons.add(status_btn)
			
			status_row.add(status_buttons)
			container.add(status_row)
			
			# Priority selection
			priority_row = toga.Box(style=Pack(direction=ROW, padding=(10, 0)))
			priority_label = toga.Label(
				"Priority:",
				style=Pack(width=120, font_weight='bold')
			)
			priority_row.add(priority_label)
			
			priority_buttons = toga.Box(style=Pack(direction=ROW))
			priorities = ['low', 'medium', 'high', 'critical']
			
			for priority in priorities:
				is_selected = priority == (self.workflow.metadata.get('priority', 'medium'))
				priority_btn = toga.Button(
					priority.title(),
					on_press=lambda x, p=priority: self._on_priority_selected(p),
					style=Pack(
						padding=5,
						background_color='#FF9800' if is_selected else '#e0e0e0',
						color='white' if is_selected else 'black'
					)
				)
				priority_buttons.add(priority_btn)
			
			priority_row.add(priority_buttons)
			container.add(priority_row)
			
		except Exception as e:
			self.logger.error(f"Failed to create edit form: {e}")
	
	async def _create_info_display(self, container: toga.Box):
		"""Create read-only information display"""
		try:
			# Basic information
			info_items = [
				("Name", self.workflow.name),
				("Description", self.workflow.description or "No description"),
				("Status", self.workflow.status.value.title()),
				("Version", self.workflow.version),
				("Created", self.workflow.created_at.strftime("%Y-%m-%d %H:%M") if self.workflow.created_at else "Unknown"),
				("Modified", self.workflow.modified_at.strftime("%Y-%m-%d %H:%M") if self.workflow.modified_at else "Unknown"),
				("Priority", self.workflow.metadata.get('priority', 'medium').title()),
				("Tags", ', '.join(self.workflow.tags) if self.workflow.tags else "None")
			]
			
			for label, value in info_items:
				info_row = self._create_info_row(label, str(value))
				container.add(info_row)
			
			# Workflow definition summary
			if self.workflow.definition:
				definition_label = toga.Label(
					"Definition Summary:",
					style=Pack(
						font_weight='bold',
						padding=(15, 0, 5, 0)
					)
				)
				container.add(definition_label)
				
				def_summary = self._create_definition_summary()
				container.add(def_summary)
			
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
	
	def _create_definition_summary(self) -> toga.Box:
		"""Create workflow definition summary"""
		summary_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=10,
				background_color='#f5f5f5'
			)
		)
		
		if self.workflow.definition:
			# Task count
			task_count = len(self.workflow.definition.get('tasks', []))
			task_count_label = toga.Label(
				f"Tasks: {task_count}",
				style=Pack(padding=(0, 0, 5, 0))
			)
			summary_box.add(task_count_label)
			
			# Trigger count
			trigger_count = len(self.workflow.definition.get('triggers', []))
			trigger_count_label = toga.Label(
				f"Triggers: {trigger_count}",
				style=Pack(padding=(0, 0, 5, 0))
			)
			summary_box.add(trigger_count_label)
			
			# Variables count
			var_count = len(self.workflow.definition.get('variables', {}))
			var_count_label = toga.Label(
				f"Variables: {var_count}",
				style=Pack(padding=(0, 0, 5, 0))
			)
			summary_box.add(var_count_label)
		
		return summary_box
	
	async def _update_tasks_section(self):
		"""Update tasks section"""
		try:
			self.tasks_container.clear()
			
			# Section title
			tasks_title = toga.Label(
				f"Tasks ({len(self.tasks)})",
				style=Pack(
					font_size=18,
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			self.tasks_container.add(tasks_title)
			
			if not self.tasks:
				empty_label = toga.Label(
					"No tasks found for this workflow",
					style=Pack(
						text_align='center',
						color='#999999',
						padding=20
					)
				)
				self.tasks_container.add(empty_label)
				return
			
			# Tasks list
			for task in self.tasks:
				task_item = self._create_task_item(task)
				self.tasks_container.add(task_item)
			
			# Add task button
			add_task_btn = toga.Button(
				"+ Add Task",
				on_press=self._on_add_task,
				style=Pack(
					padding=10,
					background_color='#4CAF50',
					color='white',
					width=120
				)
			)
			self.tasks_container.add(add_task_btn)
			
		except Exception as e:
			self.logger.error(f"Failed to update tasks section: {e}")
	
	def _create_task_item(self, task: Task) -> toga.Box:
		"""Create task list item"""
		try:
			item_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					padding=10,
					background_color='white',
					margin=(5, 0)
				)
			)
			
			# Task header
			header_row = toga.Box(style=Pack(direction=ROW))
			
			task_name = toga.Label(
				task.name,
				style=Pack(
					flex=1,
					font_weight='bold'
				)
			)
			header_row.add(task_name)
			
			status_color = self._get_task_status_color(task.status)
			status_label = toga.Label(
				task.status.value.upper(),
				style=Pack(
					background_color=status_color,
					color='white',
					padding=3,
					font_size=12
				)
			)
			header_row.add(status_label)
			
			item_box.add(header_row)
			
			# Task description
			if task.description:
				desc_label = toga.Label(
					task.description[:100] + ('...' if len(task.description) > 100 else ''),
					style=Pack(
						color='#666666',
						padding=(5, 0)
					)
				)
				item_box.add(desc_label)
			
			# Task details
			details_row = toga.Box(style=Pack(direction=ROW, padding=(5, 0)))
			
			if task.assignee:
				assignee_label = toga.Label(
					f"Assigned to: {task.assignee}",
					style=Pack(flex=1, font_size=12, color='#666666')
				)
				details_row.add(assignee_label)
			
			if task.due_date:
				due_label = toga.Label(
					f"Due: {task.due_date.strftime('%Y-%m-%d')}",
					style=Pack(font_size=12, color='#666666')
				)
				details_row.add(due_label)
			
			item_box.add(details_row)
			
			# Actions
			actions_row = toga.Box(style=Pack(direction=ROW, padding=(5, 0)))
			
			view_task_btn = toga.Button(
				"View",
				on_press=lambda x, t=task: self._on_view_task(t),
				style=Pack(
					padding=5,
					background_color='#2196F3',
					color='white'
				)
			)
			actions_row.add(view_task_btn)
			
			edit_task_btn = toga.Button(
				"Edit",
				on_press=lambda x, t=task: self._on_edit_task(t),
				style=Pack(
					padding=5,
					background_color='#FF9800',
					color='white'
				)
			)
			actions_row.add(edit_task_btn)
			
			item_box.add(actions_row)
			
			return item_box
			
		except Exception as e:
			self.logger.error(f"Failed to create task item: {e}")
			return toga.Box()
	
	async def _update_instances_section(self):
		"""Update workflow instances section"""
		try:
			self.instances_container.clear()
			
			# Section title
			instances_title = toga.Label(
				f"Execution History ({len(self.workflow_instances)})",
				style=Pack(
					font_size=18,
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			self.instances_container.add(instances_title)
			
			if not self.workflow_instances:
				empty_label = toga.Label(
					"No execution history available",
					style=Pack(
						text_align='center',
						color='#999999',
						padding=20
					)
				)
				self.instances_container.add(empty_label)
				return
			
			# Show recent instances
			recent_instances = self.workflow_instances[:5]  # Show last 5
			
			for instance in recent_instances:
				instance_item = self._create_instance_item(instance)
				self.instances_container.add(instance_item)
			
			# View all button if there are more instances
			if len(self.workflow_instances) > 5:
				view_all_btn = toga.Button(
					f"View All ({len(self.workflow_instances)} executions)",
					on_press=self._on_view_all_instances,
					style=Pack(
						padding=10,
						background_color='#607D8B',
						color='white'
					)
				)
				self.instances_container.add(view_all_btn)
			
		except Exception as e:
			self.logger.error(f"Failed to update instances section: {e}")
	
	def _create_instance_item(self, instance: Dict[str, Any]) -> toga.Box:
		"""Create workflow instance item"""
		try:
			item_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					padding=10,
					background_color='#f9f9f9',
					margin=(5, 0)
				)
			)
			
			# Instance header
			header_row = toga.Box(style=Pack(direction=ROW))
			
			instance_id = toga.Label(
				f"Run #{instance.get('id', 'Unknown')}",
				style=Pack(
					flex=1,
					font_weight='bold'
				)
			)
			header_row.add(instance_id)
			
			status = instance.get('status', 'unknown')
			status_color = self._get_instance_status_color(status)
			status_label = toga.Label(
				status.upper(),
				style=Pack(
					background_color=status_color,
					color='white',
					padding=3,
					font_size=12
				)
			)
			header_row.add(status_label)
			
			item_box.add(header_row)
			
			# Execution details
			details_row = toga.Box(style=Pack(direction=ROW, padding=(5, 0)))
			
			started_at = instance.get('started_at', '')
			if started_at:
				start_label = toga.Label(
					f"Started: {started_at[:16]}",
					style=Pack(flex=1, font_size=12, color='#666666')
				)
				details_row.add(start_label)
			
			duration = instance.get('duration')
			if duration:
				duration_label = toga.Label(
					f"Duration: {duration}",
					style=Pack(font_size=12, color='#666666')
				)
				details_row.add(duration_label)
			
			item_box.add(details_row)
			
			return item_box
			
		except Exception as e:
			self.logger.error(f"Failed to create instance item: {e}")
			return toga.Box()
	
	async def _update_action_buttons(self):
		"""Update action button states based on workflow status"""
		try:
			if not self.workflow:
				return
			
			status = self.workflow.status
			
			# Update button states based on workflow status
			self.start_button.enabled = status in [WorkflowStatus.DRAFT, WorkflowStatus.STOPPED]
			self.pause_resume_button.enabled = status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]
			self.stop_button.enabled = status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]
			
			# Update pause/resume button text
			if status == WorkflowStatus.PAUSED:
				self.pause_resume_button.text = "Resume"
				self.pause_resume_button.style.background_color = '#4CAF50'
			else:
				self.pause_resume_button.text = "Pause"
				self.pause_resume_button.style.background_color = '#FF9800'
			
		except Exception as e:
			self.logger.error(f"Failed to update action buttons: {e}")
	
	def _get_status_color(self, status: WorkflowStatus) -> str:
		"""Get color for workflow status"""
		colors = {
			WorkflowStatus.DRAFT: '#999999',
			WorkflowStatus.ACTIVE: '#4CAF50',
			WorkflowStatus.RUNNING: '#2196F3',
			WorkflowStatus.PAUSED: '#FF9800',
			WorkflowStatus.COMPLETED: '#4CAF50',
			WorkflowStatus.FAILED: '#f44336',
			WorkflowStatus.STOPPED: '#9E9E9E'
		}
		return colors.get(status, '#999999')
	
	def _get_task_status_color(self, status) -> str:
		"""Get color for task status"""
		# This would use the actual task status enum
		colors = {
			'pending': '#FF9800',
			'in_progress': '#2196F3',
			'completed': '#4CAF50',
			'failed': '#f44336',
			'cancelled': '#9E9E9E'
		}
		status_str = status.value if hasattr(status, 'value') else str(status)
		return colors.get(status_str, '#999999')
	
	def _get_instance_status_color(self, status: str) -> str:
		"""Get color for instance status"""
		colors = {
			'running': '#2196F3',
			'completed': '#4CAF50',
			'failed': '#f44336',
			'cancelled': '#9E9E9E'
		}
		return colors.get(status.lower(), '#999999')
	
	# Event handlers
	async def _on_edit_save_workflow(self, widget):
		"""Handle edit/save workflow button"""
		try:
			if self.edit_mode:
				# Save changes
				await self._save_workflow_changes()
				self.edit_mode = False
				self.edit_save_button.text = "Edit"
				await self._update_content()
			else:
				# Enter edit mode
				self.edit_mode = True
				self.edit_save_button.text = "Save"
				await self._update_content()
		except Exception as e:
			self.logger.error(f"Edit/save workflow error: {e}")
			await self._show_error(f"Failed to save workflow: {e}")
	
	async def _save_workflow_changes(self):
		"""Save workflow changes"""
		try:
			if not self.workflow:
				return
			
			# Collect changes from form inputs
			updates = {}
			
			if self.name_input and self.name_input.value.strip():
				updates['name'] = self.name_input.value.strip()
			
			if self.description_input:
				updates['description'] = self.description_input.value.strip()
			
			# Send update request
			if self.app.workflow_service and updates:
				response = await self.app.workflow_service.update_workflow(
					self.workflow_id, updates
				)
				
				if response.success:
					await self._show_info("Workflow updated successfully!")
					# Reload workflow data
					await self._load_workflow_details()
				else:
					raise ValueError(f"Update failed: {response.message}")
			
		except Exception as e:
			self.logger.error(f"Failed to save workflow changes: {e}")
			raise
	
	async def _on_start_workflow(self, widget):
		"""Handle start workflow button"""
		try:
			if self.app.workflow_service:
				response = await self.app.workflow_service.start_workflow_instance(self.workflow_id)
				if response.success:
					await self._show_info("Workflow started successfully!")
					await self.refresh()
				else:
					await self._show_error(f"Failed to start workflow: {response.message}")
		except Exception as e:
			self.logger.error(f"Start workflow error: {e}")
			await self._show_error(f"Failed to start workflow: {e}")
	
	async def _on_pause_resume_workflow(self, widget):
		"""Handle pause/resume workflow button"""
		try:
			if not self.workflow:
				return
			
			if self.workflow.status == WorkflowStatus.RUNNING:
				# Pause workflow
				if self.app.workflow_service:
					response = await self.app.workflow_service.pause_workflow_instance(self.workflow_id)
					if response.success:
						await self._show_info("Workflow paused successfully!")
					else:
						await self._show_error(f"Failed to pause workflow: {response.message}")
			elif self.workflow.status == WorkflowStatus.PAUSED:
				# Resume workflow
				if self.app.workflow_service:
					response = await self.app.workflow_service.resume_workflow_instance(self.workflow_id)
					if response.success:
						await self._show_info("Workflow resumed successfully!")
					else:
						await self._show_error(f"Failed to resume workflow: {response.message}")
			
			await self.refresh()
			
		except Exception as e:
			self.logger.error(f"Pause/resume workflow error: {e}")
			await self._show_error(f"Failed to pause/resume workflow: {e}")
	
	async def _on_stop_workflow(self, widget):
		"""Handle stop workflow button"""
		try:
			confirmed = await self._show_confirm("Are you sure you want to stop this workflow?")
			if confirmed:
				if self.app.workflow_service:
					response = await self.app.workflow_service.stop_workflow_instance(self.workflow_id)
					if response.success:
						await self._show_info("Workflow stopped successfully!")
						await self.refresh()
					else:
						await self._show_error(f"Failed to stop workflow: {response.message}")
		except Exception as e:
			self.logger.error(f"Stop workflow error: {e}")
			await self._show_error(f"Failed to stop workflow: {e}")
	
	async def _on_export_workflow(self, widget):
		"""Handle export workflow button"""
		try:
			if self.app.workflow_service:
				response = await self.app.workflow_service.export_workflow(self.workflow_id)
				if response.success:
					await self._show_info("Workflow exported successfully!")
				else:
					await self._show_error(f"Failed to export workflow: {response.message}")
		except Exception as e:
			self.logger.error(f"Export workflow error: {e}")
			await self._show_error(f"Failed to export workflow: {e}")
	
	async def _on_status_selected(self, status: WorkflowStatus):
		"""Handle status selection in edit mode"""
		if self.workflow:
			self.workflow.status = status
			await self._update_content()  # Refresh to show updated selection
	
	async def _on_priority_selected(self, priority: str):
		"""Handle priority selection in edit mode"""
		if self.workflow:
			if not self.workflow.metadata:
				self.workflow.metadata = {}
			self.workflow.metadata['priority'] = priority
			await self._update_content()  # Refresh to show updated selection
	
	async def _on_add_task(self, widget):
		"""Handle add task button"""
		# In a real implementation, this would open a task creation dialog
		await self._show_info("Add task feature coming soon!")
	
	async def _on_view_task(self, task: Task):
		"""Handle view task button"""
		await self.navigation.navigate_to('task_detail', task_id=task.id)
	
	async def _on_edit_task(self, task: Task):
		"""Handle edit task button"""
		# In a real implementation, this would open a task edit dialog
		await self._show_info("Edit task feature coming soon!")
	
	async def _on_view_all_instances(self, widget):
		"""Handle view all instances button"""
		# In a real implementation, this would show a detailed instances view
		await self._show_info("View all instances feature coming soon!")
	
	async def _handle_navigation_params(self, **kwargs):
		"""Handle navigation parameters"""
		try:
			self.workflow_id = kwargs.get('workflow_id')
			if not self.workflow_id:
				await self._show_error("No workflow ID provided")
				await self.navigation.navigate_back()
		except Exception as e:
			self.logger.error(f"Navigation params error: {e}")
	
	async def on_navigate(self, **kwargs):
		"""Handle navigation to workflow detail screen"""
		try:
			await super().on_navigate(**kwargs)
			
			if self.workflow_id:
				await self.refresh()
			else:
				await self._show_error("Invalid workflow ID")
				await self.navigation.navigate_back()
				
		except Exception as e:
			self.logger.error(f"Workflow detail navigation error: {e}")
			await self._show_error(f"Failed to load workflow: {e}")