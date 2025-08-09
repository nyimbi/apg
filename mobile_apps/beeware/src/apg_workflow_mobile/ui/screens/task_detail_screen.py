"""
Task Detail Screen for APG Workflow Mobile

Detailed view and management of individual tasks.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .base_screen import BaseScreen
from ...models.task import Task, TaskStatus, TaskPriority
from ...models.api_response import APIResponse


class TaskDetailScreen(BaseScreen):
	"""Detailed task view and management screen"""
	
	def __init__(self, app, navigation):
		super().__init__(app, navigation)
		self.title = "Task Details"
		self.requires_auth = True
		
		# Data
		self.task: Optional[Task] = None
		self.task_id: Optional[str] = None
		self.comments: List[Dict[str, Any]] = []
		self.attachments: List[Dict[str, Any]] = []
		self.activity_log: List[Dict[str, Any]] = []
		
		# UI components
		self.task_info_container = None
		self.comments_container = None
		self.attachments_container = None
		self.activity_container = None
		self.actions_container = None
		
		# Form inputs
		self.edit_mode = False
		self.name_input = None
		self.description_input = None
		self.comment_input = None
		self.assignee_input = None
		self.due_date_input = None
	
	async def _create_content(self):
		"""Create task detail UI"""
		try:
			self.content = toga.ScrollContainer(
				style=Pack(flex=1, padding=10)
			)
			
			main_box = toga.Box(
				style=Pack(direction=COLUMN)
			)
			
			# Header with task title
			self.header_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.header_container)
			
			# Action buttons
			self.actions_container = self._create_actions_section()
			main_box.add(self.actions_container)
			
			# Task information section
			self.task_info_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.task_info_container)
			
			# Comments section
			self.comments_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.comments_container)
			
			# Attachments section
			self.attachments_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.attachments_container)
			
			# Activity log section
			self.activity_container = toga.Box(
				style=Pack(direction=COLUMN, padding=(0, 0, 20, 0))
			)
			main_box.add(self.activity_container)
			
			self.content.content = main_box
			
		except Exception as e:
			self.logger.error(f"Failed to create task detail UI: {e}")
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
			on_press=self._on_edit_save_task,
			style=Pack(
				padding=10,
				background_color='#2196F3',
				color='white',
				width=80
			)
		)
		actions_box.add(self.edit_save_button)
		
		# Complete/Reopen button
		self.complete_button = toga.Button(
			"Complete",
			on_press=self._on_complete_task,
			style=Pack(
				padding=10,
				background_color='#4CAF50',
				color='white',
				width=80
			)
		)
		actions_box.add(self.complete_button)
		
		# Assign button
		self.assign_button = toga.Button(
			"Assign",
			on_press=self._on_assign_task,
			style=Pack(
				padding=10,
				background_color='#FF9800',
				color='white',
				width=80
			)
		)
		actions_box.add(self.assign_button)
		
		# Delete button
		self.delete_button = toga.Button(
			"Delete",
			on_press=self._on_delete_task,
			style=Pack(
				padding=10,
				background_color='#f44336',
				color='white',
				width=80
			)
		)
		actions_box.add(self.delete_button)
		
		return actions_box
	
	async def _load_data(self):
		"""Load task data"""
		try:
			if not self.task_id:
				raise ValueError("No task ID provided")
			
			await self._show_loading("Loading task details...")
			
			# Load task details
			await self._load_task_details()
			
			# Load task comments
			await self._load_task_comments()
			
			# Load task attachments
			await self._load_task_attachments()
			
			# Load activity log
			await self._load_activity_log()
			
			await self._hide_loading()
			
		except Exception as e:
			self.logger.error(f"Failed to load task data: {e}")
			await self._show_error(f"Failed to load task: {e}")
	
	async def _load_task_details(self):
		"""Load detailed task information"""
		try:
			if self.app.task_service:
				response = await self.app.task_service.get_task(self.task_id)
				if response.success:
					task_data = response.data.get('task')
					if task_data:
						self.task = Task(**task_data)
						self.logger.info(f"Loaded task: {self.task.name}")
					else:
						raise ValueError("Task data not found in response")
				else:
					raise ValueError(f"Failed to load task: {response.message}")
			else:
				raise ValueError("Task service not available")
				
		except Exception as e:
			self.logger.error(f"Failed to load task details: {e}")
			raise
	
	async def _load_task_comments(self):
		"""Load comments for this task"""
		try:
			if self.app.task_service:
				response = await self.app.task_service.get_task_comments(self.task_id)
				if response.success:
					self.comments = response.data.get('comments', [])
					self.logger.info(f"Loaded {len(self.comments)} comments")
		except Exception as e:
			self.logger.error(f"Failed to load task comments: {e}")
			# Don't raise - comments are optional
	
	async def _load_task_attachments(self):
		"""Load attachments for this task"""
		try:
			if self.app.task_service:
				response = await self.app.task_service.get_task_attachments(self.task_id)
				if response.success:
					self.attachments = response.data.get('attachments', [])
					self.logger.info(f"Loaded {len(self.attachments)} attachments")
		except Exception as e:
			self.logger.error(f"Failed to load task attachments: {e}")
			# Don't raise - attachments are optional
	
	async def _load_activity_log(self):
		"""Load activity log for this task"""
		try:
			if self.app.task_service:
				response = await self.app.task_service.get_task_activity(self.task_id)
				if response.success:
					self.activity_log = response.data.get('activities', [])
					self.logger.info(f"Loaded {len(self.activity_log)} activity entries")
		except Exception as e:
			self.logger.error(f"Failed to load task activity: {e}")
			# Don't raise - activity is optional
	
	async def _update_content(self):
		"""Update all UI content with loaded data"""
		try:
			# Update header
			await self._update_header()
			
			# Update task info
			await self._update_task_info()
			
			# Update comments section
			await self._update_comments_section()
			
			# Update attachments section
			await self._update_attachments_section()
			
			# Update activity section
			await self._update_activity_section()
			
			# Update action buttons
			await self._update_action_buttons()
			
		except Exception as e:
			self.logger.error(f"Failed to update content: {e}")
	
	async def _update_header(self):
		"""Update header with task information"""
		try:
			self.header_container.clear()
			
			if self.task:
				# Main title
				title_label = toga.Label(
					self.task.name,
					style=Pack(
						font_size=24,
						font_weight='bold',
						text_align='center',
						padding=(0, 0, 5, 0)
					)
				)
				self.header_container.add(title_label)
				
				# Status and priority row
				status_row = toga.Box(style=Pack(direction=ROW, alignment='center'))
				
				# Status badge
				status_color = self._get_status_color(self.task.status)
				status_label = toga.Label(
					self.task.status.value.upper(),
					style=Pack(
						background_color=status_color,
						color='white',
						padding=5,
						text_align='center',
						font_weight='bold'
					)
				)
				status_row.add(status_label)
				
				# Priority badge
				priority_color = self._get_priority_color(self.task.priority)
				priority_label = toga.Label(
					self.task.priority.value.upper(),
					style=Pack(
						background_color=priority_color,
						color='white',
						padding=5,
						text_align='center',
						font_weight='bold',
						margin=(0, 10, 0, 0)
					)
				)
				status_row.add(priority_label)
				
				self.header_container.add(status_row)
				
				# Set navigation title
				self.set_title(f"Task: {self.task.name}")
			
		except Exception as e:
			self.logger.error(f"Failed to update header: {e}")
	
	async def _update_task_info(self):
		"""Update task information section"""
		try:
			self.task_info_container.clear()
			
			if not self.task:
				return
			
			# Section title
			info_title = toga.Label(
				"Task Information",
				style=Pack(
					font_size=18,
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			self.task_info_container.add(info_title)
			
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
			
			self.task_info_container.add(info_card)
			
		except Exception as e:
			self.logger.error(f"Failed to update task info: {e}")
	
	async def _create_edit_form(self, container: toga.Box):
		"""Create edit form for task"""
		try:
			# Name field
			name_row = self._create_form_row(
				"Name:",
				toga.TextInput(
					value=self.task.name,
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
				value=self.task.description or "",
				style=Pack(
					height=100,
					padding=5
				)
			)
			container.add(self.description_input)
			
			# Assignee field
			assignee_row = self._create_form_row(
				"Assignee:",
				toga.TextInput(
					value=self.task.assignee or "",
					style=Pack(flex=1, padding=5)
				)
			)
			self.assignee_input = assignee_row.children[1]
			container.add(assignee_row)
			
			# Due date field
			due_date_row = self._create_form_row(
				"Due Date:",
				toga.TextInput(
					value=self.task.due_date.strftime("%Y-%m-%d") if self.task.due_date else "",
					placeholder="YYYY-MM-DD",
					style=Pack(flex=1, padding=5)
				)
			)
			self.due_date_input = due_date_row.children[1]
			container.add(due_date_row)
			
			# Status selection
			status_row = toga.Box(style=Pack(direction=ROW, padding=(10, 0)))
			status_label = toga.Label(
				"Status:",
				style=Pack(width=120, font_weight='bold')
			)
			status_row.add(status_label)
			
			status_buttons = toga.Box(style=Pack(direction=ROW))
			
			for status in TaskStatus:
				is_selected = status == self.task.status
				status_btn = toga.Button(
					status.value.title(),
					on_press=lambda x, s=status: self._on_status_selected(s),
					style=Pack(
						padding=5,
						background_color='#2196F3' if is_selected else '#e0e0e0',
						color='white' if is_selected else 'black',
						font_size=12
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
			
			for priority in TaskPriority:
				is_selected = priority == self.task.priority
				priority_btn = toga.Button(
					priority.value.title(),
					on_press=lambda x, p=priority: self._on_priority_selected(p),
					style=Pack(
						padding=5,
						background_color='#FF9800' if is_selected else '#e0e0e0',
						color='white' if is_selected else 'black',
						font_size=12
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
				("Name", self.task.name),
				("Description", self.task.description or "No description"),
				("Status", self.task.status.value.title()),
				("Priority", self.task.priority.value.title()),
				("Assignee", self.task.assignee or "Unassigned"),
				("Created", self.task.created_at.strftime("%Y-%m-%d %H:%M") if self.task.created_at else "Unknown"),
				("Modified", self.task.modified_at.strftime("%Y-%m-%d %H:%M") if self.task.modified_at else "Unknown"),
				("Due Date", self.task.due_date.strftime("%Y-%m-%d") if self.task.due_date else "No due date"),
				("Workflow", self.task.workflow_id or "Not part of workflow"),
				("Tags", ', '.join(self.task.tags) if self.task.tags else "None")
			]
			
			for label, value in info_items:
				info_row = self._create_info_row(label, str(value))
				container.add(info_row)
			
			# Progress information
			if hasattr(self.task, 'progress') and self.task.progress is not None:
				progress_row = self._create_progress_row(self.task.progress)
				container.add(progress_row)
			
			# Time tracking
			if hasattr(self.task, 'time_spent') or hasattr(self.task, 'estimated_hours'):
				time_info = self._create_time_tracking_info()
				container.add(time_info)
			
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
	
	def _create_progress_row(self, progress: float) -> toga.Box:
		"""Create progress display row"""
		row = toga.Box(
			style=Pack(
				direction=ROW,
				padding=(5, 0)
			)
		)
		
		label_widget = toga.Label(
			"Progress:",
			style=Pack(
				width=120,
				font_weight='bold',
				text_align='right',
				padding=(0, 10, 0, 0)
			)
		)
		row.add(label_widget)
		
		# Progress container
		progress_container = toga.Box(
			style=Pack(
				direction=ROW,
				flex=1,
				alignment='center'
			)
		)
		
		# Progress bar (simplified representation)
		progress_text = toga.Label(
			f"{progress:.1f}%",
			style=Pack(
				background_color='#4CAF50' if progress >= 50 else '#FF9800',
				color='white',
				padding=5,
				font_weight='bold'
			)
		)
		progress_container.add(progress_text)
		
		row.add(progress_container)
		return row
	
	def _create_time_tracking_info(self) -> toga.Box:
		"""Create time tracking information"""
		time_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=10,
				background_color='#f5f5f5'
			)
		)
		
		time_title = toga.Label(
			"Time Tracking",
			style=Pack(
				font_weight='bold',
				padding=(0, 0, 5, 0)
			)
		)
		time_box.add(time_title)
		
		# Estimated vs actual time
		if hasattr(self.task, 'estimated_hours'):
			estimated_label = toga.Label(
				f"Estimated: {getattr(self.task, 'estimated_hours', 0)} hours",
				style=Pack(padding=(2, 0))
			)
			time_box.add(estimated_label)
		
		if hasattr(self.task, 'time_spent'):
			spent_label = toga.Label(
				f"Time Spent: {getattr(self.task, 'time_spent', 0)} hours",
				style=Pack(padding=(2, 0))
			)
			time_box.add(spent_label)
		
		return time_box
	
	async def _update_comments_section(self):
		"""Update comments section"""
		try:
			self.comments_container.clear()
			
			# Section title
			comments_title = toga.Label(
				f"Comments ({len(self.comments)})",
				style=Pack(
					font_size=18,
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			self.comments_container.add(comments_title)
			
			# Add comment form
			add_comment_form = self._create_add_comment_form()
			self.comments_container.add(add_comment_form)
			
			# Comments list
			if not self.comments:
				empty_label = toga.Label(
					"No comments yet",
					style=Pack(
						text_align='center',
						color='#999999',
						padding=20
					)
				)
				self.comments_container.add(empty_label)
			else:
				for comment in self.comments:
					comment_item = self._create_comment_item(comment)
					self.comments_container.add(comment_item)
			
		except Exception as e:
			self.logger.error(f"Failed to update comments section: {e}")
	
	def _create_add_comment_form(self) -> toga.Box:
		"""Create add comment form"""
		form_box = toga.Box(
			style=Pack(
				direction=COLUMN,
				padding=10,
				background_color='#f9f9f9'
			)
		)
		
		form_title = toga.Label(
			"Add Comment",
			style=Pack(
				font_weight='bold',
				padding=(0, 0, 5, 0)
			)
		)
		form_box.add(form_title)
		
		# Comment input
		self.comment_input = toga.MultilineTextInput(
			placeholder="Enter your comment...",
			style=Pack(
				height=80,
				padding=5
			)
		)
		form_box.add(self.comment_input)
		
		# Add button
		add_comment_btn = toga.Button(
			"Add Comment",
			on_press=self._on_add_comment,
			style=Pack(
				padding=10,
				background_color='#4CAF50',
				color='white',
				width=120
			)
		)
		form_box.add(add_comment_btn)
		
		return form_box
	
	def _create_comment_item(self, comment: Dict[str, Any]) -> toga.Box:
		"""Create comment list item"""
		try:
			item_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					padding=10,
					background_color='white',
					margin=(5, 0)
				)
			)
			
			# Comment header
			header_row = toga.Box(style=Pack(direction=ROW))
			
			author_label = toga.Label(
				comment.get('author', 'Unknown'),
				style=Pack(
					flex=1,
					font_weight='bold'
				)
			)
			header_row.add(author_label)
			
			timestamp_label = toga.Label(
				comment.get('created_at', '')[:16],
				style=Pack(
					font_size=12,
					color='#666666'
				)
			)
			header_row.add(timestamp_label)
			
			item_box.add(header_row)
			
			# Comment content
			content_label = toga.Label(
				comment.get('content', ''),
				style=Pack(
					padding=(5, 0),
					color='#333333'
				)
			)
			item_box.add(content_label)
			
			return item_box
			
		except Exception as e:
			self.logger.error(f"Failed to create comment item: {e}")
			return toga.Box()
	
	async def _update_attachments_section(self):
		"""Update attachments section"""
		try:
			self.attachments_container.clear()
			
			# Section title
			attachments_title = toga.Label(
				f"Attachments ({len(self.attachments)})",
				style=Pack(
					font_size=18,
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			self.attachments_container.add(attachments_title)
			
			# Add attachment button
			add_attachment_btn = toga.Button(
				"+ Add Attachment",
				on_press=self._on_add_attachment,
				style=Pack(
					padding=10,
					background_color='#2196F3',
					color='white',
					width=150
				)
			)
			self.attachments_container.add(add_attachment_btn)
			
			# Attachments list
			if not self.attachments:
				empty_label = toga.Label(
					"No attachments",
					style=Pack(
						text_align='center',
						color='#999999',
						padding=20
					)
				)
				self.attachments_container.add(empty_label)
			else:
				for attachment in self.attachments:
					attachment_item = self._create_attachment_item(attachment)
					self.attachments_container.add(attachment_item)
			
		except Exception as e:
			self.logger.error(f"Failed to update attachments section: {e}")
	
	def _create_attachment_item(self, attachment: Dict[str, Any]) -> toga.Box:
		"""Create attachment list item"""
		try:
			item_box = toga.Box(
				style=Pack(
					direction=ROW,
					padding=10,
					background_color='white',
					margin=(5, 0),
					alignment='center'
				)
			)
			
			# File info
			file_info = toga.Box(style=Pack(direction=COLUMN, flex=1))
			
			filename_label = toga.Label(
				attachment.get('filename', 'Unknown file'),
				style=Pack(font_weight='bold')
			)
			file_info.add(filename_label)
			
			file_size = attachment.get('size', 0)
			size_str = self._format_file_size(file_size)
			size_label = toga.Label(
				f"Size: {size_str}",
				style=Pack(
					font_size=12,
					color='#666666'
				)
			)
			file_info.add(size_label)
			
			item_box.add(file_info)
			
			# Actions
			download_btn = toga.Button(
				"Download",
				on_press=lambda x, a=attachment: self._on_download_attachment(a),
				style=Pack(
					padding=5,
					background_color='#4CAF50',
					color='white'
				)
			)
			item_box.add(download_btn)
			
			delete_btn = toga.Button(
				"Delete",
				on_press=lambda x, a=attachment: self._on_delete_attachment(a),
				style=Pack(
					padding=5,
					background_color='#f44336',
					color='white'
				)
			)
			item_box.add(delete_btn)
			
			return item_box
			
		except Exception as e:
			self.logger.error(f"Failed to create attachment item: {e}")
			return toga.Box()
	
	def _format_file_size(self, size_bytes: int) -> str:
		"""Format file size in human readable format"""
		if size_bytes == 0:
			return "0 B"
		
		size_names = ["B", "KB", "MB", "GB"]
		i = 0
		size = float(size_bytes)
		
		while size >= 1024.0 and i < len(size_names) - 1:
			size /= 1024.0
			i += 1
		
		return f"{size:.1f} {size_names[i]}"
	
	async def _update_activity_section(self):
		"""Update activity log section"""
		try:
			self.activity_container.clear()
			
			# Section title
			activity_title = toga.Label(
				f"Activity Log ({len(self.activity_log)})",
				style=Pack(
					font_size=18,
					font_weight='bold',
					padding=(0, 0, 10, 0)
				)
			)
			self.activity_container.add(activity_title)
			
			if not self.activity_log:
				empty_label = toga.Label(
					"No activity recorded",
					style=Pack(
						text_align='center',
						color='#999999',
						padding=20
					)
				)
				self.activity_container.add(empty_label)
				return
			
			# Show recent activity (last 10 entries)
			recent_activity = self.activity_log[:10]
			
			for activity in recent_activity:
				activity_item = self._create_activity_item(activity)
				self.activity_container.add(activity_item)
			
		except Exception as e:
			self.logger.error(f"Failed to update activity section: {e}")
	
	def _create_activity_item(self, activity: Dict[str, Any]) -> toga.Box:
		"""Create activity log item"""
		try:
			item_box = toga.Box(
				style=Pack(
					direction=COLUMN,
					padding=8,
					background_color='#f5f5f5',
					margin=(3, 0)
				)
			)
			
			# Activity header
			header_row = toga.Box(style=Pack(direction=ROW))
			
			action_label = toga.Label(
				activity.get('action', 'Unknown action'),
				style=Pack(
					flex=1,
					font_weight='bold',
					font_size=12
				)
			)
			header_row.add(action_label)
			
			timestamp_label = toga.Label(
				activity.get('timestamp', '')[:16],
				style=Pack(
					font_size=10,
					color='#666666'
				)
			)
			header_row.add(timestamp_label)
			
			item_box.add(header_row)
			
			# Activity details
			details = activity.get('details', '')
			if details:
				details_label = toga.Label(
					details,
					style=Pack(
						font_size=11,
						color='#555555',
						padding=(2, 0, 0, 0)
					)
				)
				item_box.add(details_label)
			
			return item_box
			
		except Exception as e:
			self.logger.error(f"Failed to create activity item: {e}")
			return toga.Box()
	
	async def _update_action_buttons(self):
		"""Update action button states based on task status"""
		try:
			if not self.task:
				return
			
			status = self.task.status
			
			# Update complete/reopen button
			if status == TaskStatus.COMPLETED:
				self.complete_button.text = "Reopen"
				self.complete_button.style.background_color = '#FF9800'
			else:
				self.complete_button.text = "Complete"
				self.complete_button.style.background_color = '#4CAF50'
			
			# Enable/disable buttons based on status
			self.complete_button.enabled = status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]
			self.assign_button.enabled = status != TaskStatus.COMPLETED
			
		except Exception as e:
			self.logger.error(f"Failed to update action buttons: {e}")
	
	def _get_status_color(self, status: TaskStatus) -> str:
		"""Get color for task status"""
		colors = {
			TaskStatus.PENDING: '#FF9800',
			TaskStatus.IN_PROGRESS: '#2196F3',
			TaskStatus.COMPLETED: '#4CAF50',
			TaskStatus.CANCELLED: '#9E9E9E',
			TaskStatus.FAILED: '#f44336'
		}
		return colors.get(status, '#999999')
	
	def _get_priority_color(self, priority: TaskPriority) -> str:
		"""Get color for task priority"""
		colors = {
			TaskPriority.LOW: '#4CAF50',
			TaskPriority.MEDIUM: '#FF9800',
			TaskPriority.HIGH: '#f44336',
			TaskPriority.CRITICAL: '#9C27B0'
		}
		return colors.get(priority, '#999999')
	
	# Event handlers
	async def _on_edit_save_task(self, widget):
		"""Handle edit/save task button"""
		try:
			if self.edit_mode:
				# Save changes
				await self._save_task_changes()
				self.edit_mode = False
				self.edit_save_button.text = "Edit"
				await self._update_content()
			else:
				# Enter edit mode
				self.edit_mode = True
				self.edit_save_button.text = "Save"
				await self._update_content()
		except Exception as e:
			self.logger.error(f"Edit/save task error: {e}")
			await self._show_error(f"Failed to save task: {e}")
	
	async def _save_task_changes(self):
		"""Save task changes"""
		try:
			if not self.task:
				return
			
			# Collect changes from form inputs
			updates = {}
			
			if self.name_input and self.name_input.value.strip():
				updates['name'] = self.name_input.value.strip()
			
			if self.description_input:
				updates['description'] = self.description_input.value.strip()
			
			if self.assignee_input:
				updates['assignee'] = self.assignee_input.value.strip()
			
			if self.due_date_input and self.due_date_input.value.strip():
				try:
					due_date = datetime.strptime(self.due_date_input.value.strip(), "%Y-%m-%d")
					updates['due_date'] = due_date.isoformat()
				except ValueError:
					raise ValueError("Invalid due date format. Use YYYY-MM-DD")
			
			# Send update request
			if self.app.task_service and updates:
				response = await self.app.task_service.update_task(
					self.task_id, updates
				)
				
				if response.success:
					await self._show_info("Task updated successfully!")
					# Reload task data
					await self._load_task_details()
				else:
					raise ValueError(f"Update failed: {response.message}")
			
		except Exception as e:
			self.logger.error(f"Failed to save task changes: {e}")
			raise
	
	async def _on_complete_task(self, widget):
		"""Handle complete/reopen task button"""
		try:
			if not self.task:
				return
			
			if self.task.status == TaskStatus.COMPLETED:
				# Reopen task
				if self.app.task_service:
					response = await self.app.task_service.update_task(
						self.task_id, {'status': TaskStatus.IN_PROGRESS.value}
					)
					if response.success:
						await self._show_info("Task reopened successfully!")
						await self.refresh()
					else:
						await self._show_error(f"Failed to reopen task: {response.message}")
			else:
				# Complete task
				if self.app.task_service:
					response = await self.app.task_service.complete_task(self.task_id)
					if response.success:
						await self._show_info("Task completed successfully!")
						await self.refresh()
					else:
						await self._show_error(f"Failed to complete task: {response.message}")
			
		except Exception as e:
			self.logger.error(f"Complete/reopen task error: {e}")
			await self._show_error(f"Failed to update task status: {e}")
	
	async def _on_assign_task(self, widget):
		"""Handle assign task button"""
		# In a real implementation, this would show an assignment dialog
		await self._show_info("Task assignment feature coming soon!")
	
	async def _on_delete_task(self, widget):
		"""Handle delete task button"""
		try:
			confirmed = await self._show_confirm("Are you sure you want to delete this task?")
			if confirmed:
				if self.app.task_service:
					response = await self.app.task_service.delete_task(self.task_id)
					if response.success:
						await self._show_info("Task deleted successfully!")
						await self.navigation.navigate_back()
					else:
						await self._show_error(f"Failed to delete task: {response.message}")
		except Exception as e:
			self.logger.error(f"Delete task error: {e}")
			await self._show_error(f"Failed to delete task: {e}")
	
	async def _on_status_selected(self, status: TaskStatus):
		"""Handle status selection in edit mode"""
		if self.task:
			self.task.status = status
			await self._update_content()
	
	async def _on_priority_selected(self, priority: TaskPriority):
		"""Handle priority selection in edit mode"""
		if self.task:
			self.task.priority = priority
			await self._update_content()
	
	async def _on_add_comment(self, widget):
		"""Handle add comment button"""
		try:
			if not self.comment_input or not self.comment_input.value.strip():
				await self._show_error("Please enter a comment")
				return
			
			comment_text = self.comment_input.value.strip()
			
			if self.app.task_service:
				response = await self.app.task_service.add_task_comment(
					self.task_id, comment_text
				)
				
				if response.success:
					self.comment_input.value = ""  # Clear input
					await self._load_task_comments()  # Reload comments
					await self._update_comments_section()  # Refresh UI
					await self._show_info("Comment added successfully!")
				else:
					await self._show_error(f"Failed to add comment: {response.message}")
			
		except Exception as e:
			self.logger.error(f"Add comment error: {e}")
			await self._show_error(f"Failed to add comment: {e}")
	
	async def _on_add_attachment(self, widget):
		"""Handle add attachment button"""
		# In a real implementation, this would open a file picker
		await self._show_info("File attachment feature coming soon!")
	
	async def _on_download_attachment(self, attachment: Dict[str, Any]):
		"""Handle download attachment button"""
		try:
			file_id = attachment.get('id')
			if file_id and self.app.file_service:
				response = await self.app.file_service.download_file(file_id)
				if response.success:
					await self._show_info(f"Downloaded: {attachment.get('filename', 'file')}")
				else:
					await self._show_error(f"Failed to download file: {response.message}")
		except Exception as e:
			self.logger.error(f"Download attachment error: {e}")
			await self._show_error(f"Failed to download attachment: {e}")
	
	async def _on_delete_attachment(self, attachment: Dict[str, Any]):
		"""Handle delete attachment button"""
		try:
			confirmed = await self._show_confirm(
				f"Are you sure you want to delete {attachment.get('filename', 'this file')}?"
			)
			if confirmed:
				file_id = attachment.get('id')
				if file_id and self.app.task_service:
					response = await self.app.task_service.delete_task_attachment(
						self.task_id, file_id
					)
					if response.success:
						await self._load_task_attachments()
						await self._update_attachments_section()
						await self._show_info("Attachment deleted successfully!")
					else:
						await self._show_error(f"Failed to delete attachment: {response.message}")
		except Exception as e:
			self.logger.error(f"Delete attachment error: {e}")
			await self._show_error(f"Failed to delete attachment: {e}")
	
	async def _handle_navigation_params(self, **kwargs):
		"""Handle navigation parameters"""
		try:
			self.task_id = kwargs.get('task_id')
			if not self.task_id:
				await self._show_error("No task ID provided")
				await self.navigation.navigate_back()
		except Exception as e:
			self.logger.error(f"Navigation params error: {e}")
	
	async def on_navigate(self, **kwargs):
		"""Handle navigation to task detail screen"""
		try:
			await super().on_navigate(**kwargs)
			
			if self.task_id:
				await self.refresh()
			else:
				await self._show_error("Invalid task ID")
				await self.navigation.navigate_back()
				
		except Exception as e:
			self.logger.error(f"Task detail navigation error: {e}")
			await self._show_error(f"Failed to load task: {e}")