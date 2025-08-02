"""
APG Workflow Designer Service

Main service for the professional workflow designer interface.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from .canvas_engine import CanvasEngine
from .component_library import ComponentLibrary
from .property_panels import PropertyPanelManager
from .validation_engine import ValidationEngine
from .collaboration_manager import CollaborationManager
from .export_manager import ExportManager

logger = logging.getLogger(__name__)

class DesignerConfiguration(BaseModel):
	"""Configuration for the workflow designer."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core settings
	auto_save_interval: int = Field(default=30, ge=10, le=300, description="Auto-save interval in seconds")
	max_undo_steps: int = Field(default=50, ge=10, le=200, description="Maximum undo/redo steps")
	canvas_width: int = Field(default=5000, ge=1000, le=20000, description="Canvas width in pixels")
	canvas_height: int = Field(default=3000, ge=1000, le=15000, description="Canvas height in pixels")
	
	# Grid and snapping
	grid_size: int = Field(default=20, ge=5, le=50, description="Grid size for snapping")
	snap_to_grid: bool = Field(default=True, description="Enable grid snapping")
	snap_threshold: int = Field(default=10, ge=5, le=25, description="Snap distance threshold")
	
	# Performance settings
	max_nodes: int = Field(default=1000, ge=100, le=5000, description="Maximum nodes per workflow")
	render_optimization: bool = Field(default=True, description="Enable rendering optimizations")
	lazy_loading: bool = Field(default=True, description="Enable lazy loading of components")
	
	# Collaboration settings
	enable_collaboration: bool = Field(default=True, description="Enable real-time collaboration")
	max_collaborators: int = Field(default=10, ge=1, le=50, description="Maximum simultaneous collaborators")
	collaboration_timeout: int = Field(default=300, ge=60, le=3600, description="Collaboration session timeout")
	
	# UI preferences
	theme: str = Field(default="light", regex="^(light|dark|auto)$", description="UI theme")
	minimap_enabled: bool = Field(default=True, description="Show minimap")
	property_panel_position: str = Field(default="right", regex="^(left|right|bottom)$", description="Property panel position")
	component_panel_position: str = Field(default="left", regex="^(left|right|top)$", description="Component panel position")
	
	# Advanced features
	enable_ai_suggestions: bool = Field(default=True, description="Enable AI-powered suggestions")
	enable_validation: bool = Field(default=True, description="Enable real-time validation")
	enable_version_control: bool = Field(default=True, description="Enable version control")

class WorkflowDesigner:
	"""
	Professional workflow designer with drag-and-drop interface.
	
	Features:
	- Advanced canvas with infinite scroll and zoom
	- Rich component library with custom components
	- Real-time collaboration
	- AI-powered suggestions and validation
	- Export to multiple formats
	- Version control integration
	"""
	
	def __init__(self, config: DesignerConfiguration):
		self.config = config
		self.designer_id = str(uuid4())
		self.active_sessions: Dict[str, Any] = {}
		
		# Initialize core engines
		self.canvas_engine = CanvasEngine(config)
		self.component_library = ComponentLibrary(config)
		self.property_manager = PropertyPanelManager(config)
		self.validation_engine = ValidationEngine(config)
		self.collaboration_manager = CollaborationManager(config)
		self.export_manager = ExportManager(config)
		
		# State management
		self.undo_stack: List[Dict[str, Any]] = []
		self.redo_stack: List[Dict[str, Any]] = []
		self.auto_save_task: Optional[asyncio.Task] = None
		self.is_running = False
		
		logger.info(f"Workflow designer initialized with ID: {self.designer_id}")
	
	async def start(self) -> None:
		"""Start the workflow designer service."""
		try:
			self.is_running = True
			
			# Initialize all engines
			await self.canvas_engine.initialize()
			await self.component_library.initialize()
			await self.property_manager.initialize()
			await self.validation_engine.initialize()
			await self.collaboration_manager.initialize()
			await self.export_manager.initialize()
			
			# Start auto-save if enabled
			if self.config.auto_save_interval > 0:
				self.auto_save_task = asyncio.create_task(self._auto_save_loop())
			
			logger.info("Workflow designer service started successfully")
			
		except Exception as e:
			logger.error(f"Failed to start workflow designer: {e}")
			raise
	
	async def stop(self) -> None:
		"""Stop the workflow designer service."""
		try:
			self.is_running = False
			
			# Cancel auto-save task
			if self.auto_save_task:
				self.auto_save_task.cancel()
				try:
					await self.auto_save_task
				except asyncio.CancelledError:
					pass
			
			# Close all active sessions
			for session_id in list(self.active_sessions.keys()):
				await self.close_session(session_id)
			
			# Shutdown all engines
			await self.canvas_engine.shutdown()
			await self.component_library.shutdown()
			await self.property_manager.shutdown()
			await self.validation_engine.shutdown()
			await self.collaboration_manager.shutdown()
			await self.export_manager.shutdown()
			
			logger.info("Workflow designer service stopped")
			
		except Exception as e:
			logger.error(f"Error stopping workflow designer: {e}")
	
	async def create_session(self, user_id: str, workflow_id: Optional[str] = None) -> Dict[str, Any]:
		"""Create a new designer session."""
		try:
			session_id = str(uuid4())
			
			# Load existing workflow or create new
			if workflow_id:
				workflow_data = await self._load_workflow(workflow_id)
			else:
				workflow_data = await self._create_new_workflow()
			
			# Create session
			session = {
				'session_id': session_id,
				'user_id': user_id,
				'workflow_id': workflow_id,
				'workflow_data': workflow_data,
				'created_at': datetime.now(timezone.utc),
				'last_activity': datetime.now(timezone.utc),
				'canvas_state': await self.canvas_engine.get_initial_state(),
				'selected_components': [],
				'clipboard': None,
				'is_dirty': False
			}
			
			self.active_sessions[session_id] = session
			
			# Initialize collaboration if enabled
			if self.config.enable_collaboration:
				await self.collaboration_manager.join_session(session_id, user_id)
			
			logger.info(f"Created designer session {session_id} for user {user_id}")
			return {
				'session_id': session_id,
				'workflow_data': workflow_data,
				'canvas_state': session['canvas_state'],
				'configuration': self.config.model_dump()
			}
			
		except Exception as e:
			logger.error(f"Failed to create designer session: {e}")
			raise
	
	async def close_session(self, session_id: str) -> None:
		"""Close a designer session."""
		try:
			if session_id not in self.active_sessions:
				return
			
			session = self.active_sessions[session_id]
			
			# Auto-save if dirty
			if session.get('is_dirty', False):
				await self._save_session(session_id)
			
			# Leave collaboration
			if self.config.enable_collaboration:
				await self.collaboration_manager.leave_session(session_id, session['user_id'])
			
			# Clean up session
			del self.active_sessions[session_id]
			
			logger.info(f"Closed designer session {session_id}")
			
		except Exception as e:
			logger.error(f"Error closing session {session_id}: {e}")
	
	async def add_component(self, session_id: str, component_type: str, position: Dict[str, float], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Add a component to the workflow canvas."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			
			# Get component definition
			component_def = await self.component_library.get_component(component_type)
			if not component_def:
				raise ValueError(f"Component type {component_type} not found")
			
			# Create canvas node
			node = await self.canvas_engine.add_node(
				session_id=session_id,
				component_type=component_type,
				position=position,
				config=config or {}
			)
			
			# Update session state
			await self._update_session_state(session_id, {
				'action': 'add_component',
				'node': node.model_dump(),
				'timestamp': datetime.now(timezone.utc)
			})
			
			# Validate workflow
			if self.config.enable_validation:
				validation_result = await self.validation_engine.validate_workflow(session['workflow_data'])
				await self._broadcast_validation_result(session_id, validation_result)
			
			# Broadcast to collaborators
			if self.config.enable_collaboration:
				await self.collaboration_manager.broadcast_change(session_id, {
					'type': 'component_added',
					'node': node.model_dump(),
					'user_id': session['user_id']
				})
			
			logger.debug(f"Added component {component_type} to session {session_id}")
			return node.model_dump()
			
		except Exception as e:
			logger.error(f"Failed to add component: {e}")
			raise
	
	async def remove_component(self, session_id: str, node_id: str) -> None:
		"""Remove a component from the workflow canvas."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			
			# Remove from canvas
			await self.canvas_engine.remove_node(session_id, node_id)
			
			# Update session state
			await self._update_session_state(session_id, {
				'action': 'remove_component',
				'node_id': node_id,
				'timestamp': datetime.now(timezone.utc)
			})
			
			# Validate workflow
			if self.config.enable_validation:
				validation_result = await self.validation_engine.validate_workflow(session['workflow_data'])
				await self._broadcast_validation_result(session_id, validation_result)
			
			# Broadcast to collaborators
			if self.config.enable_collaboration:
				await self.collaboration_manager.broadcast_change(session_id, {
					'type': 'component_removed',
					'node_id': node_id,
					'user_id': session['user_id']
				})
			
			logger.debug(f"Removed component {node_id} from session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to remove component: {e}")
			raise
	
	async def connect_components(self, session_id: str, source_id: str, target_id: str, source_port: str = 'output', target_port: str = 'input') -> Dict[str, Any]:
		"""Create a connection between two components."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			
			# Create connection
			connection = await self.canvas_engine.add_connection(
				session_id=session_id,
				source_id=source_id,
				target_id=target_id,
				source_port=source_port,
				target_port=target_port
			)
			
			# Update session state
			await self._update_session_state(session_id, {
				'action': 'connect_components',
				'connection': connection.model_dump(),
				'timestamp': datetime.now(timezone.utc)
			})
			
			# Validate workflow
			if self.config.enable_validation:
				validation_result = await self.validation_engine.validate_workflow(session['workflow_data'])
				await self._broadcast_validation_result(session_id, validation_result)
			
			# Broadcast to collaborators
			if self.config.enable_collaboration:
				await self.collaboration_manager.broadcast_change(session_id, {
					'type': 'connection_created',
					'connection': connection.model_dump(),
					'user_id': session['user_id']
				})
			
			logger.debug(f"Connected components {source_id} -> {target_id} in session {session_id}")
			return connection.model_dump()
			
		except Exception as e:
			logger.error(f"Failed to connect components: {e}")
			raise
	
	async def update_component_config(self, session_id: str, node_id: str, config: Dict[str, Any]) -> None:
		"""Update component configuration."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			
			# Update node configuration
			await self.canvas_engine.update_node_config(session_id, node_id, config)
			
			# Update session state
			await self._update_session_state(session_id, {
				'action': 'update_config',
				'node_id': node_id,
				'config': config,
				'timestamp': datetime.now(timezone.utc)
			})
			
			# Validate workflow
			if self.config.enable_validation:
				validation_result = await self.validation_engine.validate_workflow(session['workflow_data'])
				await self._broadcast_validation_result(session_id, validation_result)
			
			# Broadcast to collaborators
			if self.config.enable_collaboration:
				await self.collaboration_manager.broadcast_change(session_id, {
					'type': 'config_updated',
					'node_id': node_id,
					'config': config,
					'user_id': session['user_id']
				})
			
			logger.debug(f"Updated config for component {node_id} in session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to update component config: {e}")
			raise
	
	async def undo(self, session_id: str) -> Optional[Dict[str, Any]]:
		"""Undo the last action."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			if not self.undo_stack:
				return None
			
			# Pop last action and push to redo
			last_action = self.undo_stack.pop()
			self.redo_stack.append(last_action)
			
			# Restore previous state
			await self._restore_session_state(session_id, last_action.get('previous_state'))
			
			logger.debug(f"Undid action in session {session_id}")
			return last_action
			
		except Exception as e:
			logger.error(f"Failed to undo: {e}")
			raise
	
	async def redo(self, session_id: str) -> Optional[Dict[str, Any]]:
		"""Redo the last undone action."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			if not self.redo_stack:
				return None
			
			# Pop from redo and execute
			action = self.redo_stack.pop()
			self.undo_stack.append(action)
			
			# Apply action
			await self._apply_action(session_id, action)
			
			logger.debug(f"Redid action in session {session_id}")
			return action
			
		except Exception as e:
			logger.error(f"Failed to redo: {e}")
			raise
	
	async def save_workflow(self, session_id: str, name: Optional[str] = None, description: Optional[str] = None) -> str:
		"""Save the current workflow."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			workflow_data = session['workflow_data']
			
			# Update metadata
			if name:
				workflow_data['name'] = name
			if description:
				workflow_data['description'] = description
			
			workflow_data['updated_at'] = datetime.now(timezone.utc).isoformat()
			workflow_data['canvas_state'] = await self.canvas_engine.get_canvas_state(session_id)
			
			# Save to database
			workflow_id = await self._save_workflow_to_database(workflow_data)
			
			# Update session
			session['workflow_id'] = workflow_id
			session['is_dirty'] = False
			
			logger.info(f"Saved workflow {workflow_id} from session {session_id}")
			return workflow_id
			
		except Exception as e:
			logger.error(f"Failed to save workflow: {e}")
			raise
	
	async def export_workflow(self, session_id: str, format_type: str) -> Dict[str, Any]:
		"""Export workflow in specified format."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			workflow_data = session['workflow_data']
			
			# Export using export manager
			export_result = await self.export_manager.export_workflow(workflow_data, format_type)
			
			logger.info(f"Exported workflow from session {session_id} as {format_type}")
			return export_result
			
		except Exception as e:
			logger.error(f"Failed to export workflow: {e}")
			raise
	
	async def get_session_info(self, session_id: str) -> Dict[str, Any]:
		"""Get information about a design session."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			canvas_state = await self.canvas_engine.get_canvas_state(session_id)
			
			return {
				'session_id': session_id,
				'user_id': session['user_id'],
				'workflow_id': session.get('workflow_id'),
				'created_at': session['created_at'].isoformat(),
				'last_activity': session['last_activity'].isoformat(),
				'is_dirty': session.get('is_dirty', False),
				'canvas_state': canvas_state,
				'collaborators': await self.collaboration_manager.get_session_collaborators(session_id) if self.config.enable_collaboration else [],
				'node_count': len(canvas_state.get('nodes', [])),
				'connection_count': len(canvas_state.get('connections', []))
			}
			
		except Exception as e:
			logger.error(f"Failed to get session info: {e}")
			raise
	
	# Private methods
	
	async def _auto_save_loop(self) -> None:
		"""Auto-save loop for dirty sessions."""
		while self.is_running:
			try:
				await asyncio.sleep(self.config.auto_save_interval)
				
				# Save all dirty sessions
				for session_id, session in self.active_sessions.items():
					if session.get('is_dirty', False):
						await self._save_session(session_id)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Error in auto-save loop: {e}")
	
	async def _save_session(self, session_id: str) -> None:
		"""Save a session's current state."""
		try:
			session = self.active_sessions[session_id]
			workflow_data = session['workflow_data']
			
			# Update canvas state
			workflow_data['canvas_state'] = await self.canvas_engine.get_canvas_state(session_id)
			workflow_data['updated_at'] = datetime.now(timezone.utc).isoformat()
			
			# Save to database
			if session.get('workflow_id'):
				await self._update_workflow_in_database(session['workflow_id'], workflow_data)
			
			session['is_dirty'] = False
			logger.debug(f"Auto-saved session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to auto-save session {session_id}: {e}")
	
	async def _update_session_state(self, session_id: str, action: Dict[str, Any]) -> None:
		"""Update session state and manage undo/redo."""
		try:
			session = self.active_sessions[session_id]
			
			# Capture current state for undo
			current_state = await self.canvas_engine.get_canvas_state(session_id)
			
			# Add to undo stack
			undo_action = {
				'action': action,
				'previous_state': current_state,
				'timestamp': datetime.now(timezone.utc)
			}
			
			self.undo_stack.append(undo_action)
			
			# Limit undo stack size
			if len(self.undo_stack) > self.config.max_undo_steps:
				self.undo_stack.pop(0)
			
			# Clear redo stack
			self.redo_stack.clear()
			
			# Mark as dirty
			session['is_dirty'] = True
			session['last_activity'] = datetime.now(timezone.utc)
			
		except Exception as e:
			logger.error(f"Failed to update session state: {e}")
	
	async def _load_workflow(self, workflow_id: str) -> Dict[str, Any]:
		"""Load workflow from database."""
		try:
			from ..database import DatabaseManager
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = """
				SELECT * FROM cr_workflows WHERE id = %s
				"""
				result = await session.execute(query, (workflow_id,))
				row = await result.fetchone()
				
				if not row:
					raise ValueError(f"Workflow {workflow_id} not found")
				
				return {
					'id': row['id'],
					'name': row['name'],
					'description': row['description'],
					'definition': row['definition'],
					'canvas_state': row.get('canvas_state', {}),
					'created_at': row['created_at'].isoformat(),
					'updated_at': row['updated_at'].isoformat()
				}
				
		except Exception as e:
			logger.error(f"Failed to load workflow {workflow_id}: {e}")
			raise
	
	async def _create_new_workflow(self) -> Dict[str, Any]:
		"""Create a new empty workflow."""
		return {
			'id': None,
			'name': 'Untitled Workflow',
			'description': '',
			'definition': {'nodes': [], 'connections': []},
			'canvas_state': {'zoom': 1.0, 'pan_x': 0, 'pan_y': 0},
			'created_at': datetime.now(timezone.utc).isoformat(),
			'updated_at': datetime.now(timezone.utc).isoformat()
		}
	
	async def _save_workflow_to_database(self, workflow_data: Dict[str, Any]) -> str:
		"""Save workflow to database."""
		try:
			from ..database import DatabaseManager
			from uuid_extensions import uuid7str
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				workflow_id = workflow_data.get('id') or uuid7str()
				
				if workflow_data.get('id'):
					# Update existing
					query = """
					UPDATE cr_workflows 
					SET name = %s, description = %s, definition = %s, 
					    canvas_state = %s, updated_at = %s
					WHERE id = %s
					"""
					await session.execute(query, (
						workflow_data['name'],
						workflow_data['description'],
						workflow_data['definition'],
						workflow_data['canvas_state'],
						datetime.now(timezone.utc),
						workflow_id
					))
				else:
					# Create new
					query = """
					INSERT INTO cr_workflows (id, name, description, definition, canvas_state, created_at, updated_at)
					VALUES (%s, %s, %s, %s, %s, %s, %s)
					"""
					await session.execute(query, (
						workflow_id,
						workflow_data['name'],
						workflow_data['description'],
						workflow_data['definition'],
						workflow_data['canvas_state'],
						datetime.now(timezone.utc),
						datetime.now(timezone.utc)
					))
				
				await session.commit()
				return workflow_id
				
		except Exception as e:
			logger.error(f"Failed to save workflow to database: {e}")
			raise
	
	async def _update_workflow_in_database(self, workflow_id: str, workflow_data: Dict[str, Any]) -> None:
		"""Update workflow in database."""
		try:
			from ..database import DatabaseManager
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = """
				UPDATE cr_workflows 
				SET definition = %s, canvas_state = %s, updated_at = %s
				WHERE id = %s
				"""
				await session.execute(query, (
					workflow_data['definition'],
					workflow_data['canvas_state'],
					datetime.now(timezone.utc),
					workflow_id
				))
				await session.commit()
				
		except Exception as e:
			logger.error(f"Failed to update workflow in database: {e}")
			raise
	
	async def _broadcast_validation_result(self, session_id: str, validation_result: Any) -> None:
		"""Broadcast validation results to session."""
		try:
			if self.config.enable_collaboration:
				await self.collaboration_manager.broadcast_change(session_id, {
					'type': 'validation_result',
					'result': validation_result.model_dump() if hasattr(validation_result, 'model_dump') else validation_result
				})
		except Exception as e:
			logger.error(f"Failed to broadcast validation result: {e}")
	
	async def _restore_session_state(self, session_id: str, state: Optional[Dict[str, Any]]) -> None:
		"""Restore session to a previous state."""
		if state:
			await self.canvas_engine.restore_state(session_id, state)
	
	async def _apply_action(self, session_id: str, action: Dict[str, Any]) -> None:
		"""Apply a specific action to the session."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			action_type = action.get('action', {}).get('action')
			action_data = action.get('action', {})
			
			if action_type == 'add_component':
				# Re-add component
				node_data = action_data.get('node', {})
				await self.add_component(
					session_id=session_id,
					component_type=node_data.get('component_type'),
					position=node_data.get('position', {}),
					config=node_data.get('config', {})
				)
				
			elif action_type == 'remove_component':
				# Re-remove component
				node_id = action_data.get('node_id')
				if node_id:
					await self.remove_component(session_id, node_id)
			
			elif action_type == 'connect_components':
				# Re-create connection
				connection_data = action_data.get('connection', {})
				await self.connect_components(
					session_id=session_id,
					source_id=connection_data.get('source_node_id'),
					target_id=connection_data.get('target_node_id'),
					source_port=connection_data.get('source_port', 'output'),
					target_port=connection_data.get('target_port', 'input')
				)
			
			elif action_type == 'remove_connection':
				# Re-remove connection
				connection_id = action_data.get('connection_id')
				if connection_id:
					await self.canvas_engine.remove_connection(session_id, connection_id)
			
			elif action_type == 'move_component':
				# Re-move component
				node_id = action_data.get('node_id')
				position = action_data.get('position', {})
				if node_id and position:
					await self.canvas_engine.move_node(session_id, node_id, position)
			
			elif action_type == 'update_config':
				# Re-update configuration
				node_id = action_data.get('node_id')
				config = action_data.get('config', {})
				if node_id:
					await self.update_component_config(session_id, node_id, config)
			
			elif action_type == 'autoLayout':
				# Re-apply auto layout
				old_positions = action_data.get('oldPositions', {})
				new_positions = action_data.get('newPositions', {})
				
				# Apply the new positions (redo operation)
				for node_id, position in new_positions.items():
					await self.canvas_engine.move_node(session_id, node_id, position)
			
			else:
				logger.warning(f"Unknown action type: {action_type}")
			
		except Exception as e:
			logger.error(f"Failed to apply action {action.get('action', {}).get('action', 'unknown')}: {e}")
			raise
	
	# Additional methods referenced by blueprint
	
	async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
		"""Get current session state."""
		try:
			if session_id not in self.active_sessions:
				return None
			
			session = self.active_sessions[session_id]
			canvas_state = await self.canvas_engine.get_canvas_state(session_id)
			
			return {
				'session_id': session_id,
				'user_id': session['user_id'],
				'workflow_id': session.get('workflow_id'),
				'workflow_data': session['workflow_data'],
				'canvas_state': canvas_state,
				'selected_components': session.get('selected_components', []),
				'is_dirty': session.get('is_dirty', False),
				'created_at': session['created_at'].isoformat(),
				'last_activity': session['last_activity'].isoformat()
			}
			
		except Exception as e:
			logger.error(f"Failed to get session state: {e}")
			return None
	
	async def move_component(self, session_id: str, component_id: str, position: Dict[str, float]) -> None:
		"""Move a component to a new position."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			
			# Move component on canvas
			await self.canvas_engine.move_node(session_id, component_id, position)
			
			# Update session state
			await self._update_session_state(session_id, {
				'action': 'move_component',
				'node_id': component_id,
				'position': position,
				'timestamp': datetime.now(timezone.utc)
			})
			
			# Broadcast to collaborators
			if self.config.enable_collaboration:
				await self.collaboration_manager.broadcast_change(session_id, {
					'type': 'component_moved',
					'node_id': component_id,
					'position': position,
					'user_id': session['user_id']
				})
			
			logger.debug(f"Moved component {component_id} in session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to move component: {e}")
			raise
	
	async def remove_connection(self, session_id: str, connection_id: str) -> None:
		"""Remove a connection from the workflow."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			
			# Remove from canvas
			await self.canvas_engine.remove_connection(session_id, connection_id)
			
			# Update session state
			await self._update_session_state(session_id, {
				'action': 'remove_connection',
				'connection_id': connection_id,
				'timestamp': datetime.now(timezone.utc)
			})
			
			# Validate workflow
			if self.config.enable_validation:
				validation_result = await self.validation_engine.validate_workflow(session['workflow_data'])
				await self._broadcast_validation_result(session_id, validation_result)
			
			# Broadcast to collaborators
			if self.config.enable_collaboration:
				await self.collaboration_manager.broadcast_change(session_id, {
					'type': 'connection_removed',
					'connection_id': connection_id,
					'user_id': session['user_id']
				})
			
			logger.debug(f"Removed connection {connection_id} from session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to remove connection: {e}")
			raise
	
	async def validate_workflow(self, session_id: str) -> Dict[str, Any]:
		"""Validate the current workflow."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			workflow_data = session['workflow_data']
			
			# Perform validation
			validation_result = await self.validation_engine.validate_workflow(workflow_data)
			
			# Broadcast result to collaborators
			if self.config.enable_collaboration:
				await self._broadcast_validation_result(session_id, validation_result)
			
			return validation_result.model_dump() if hasattr(validation_result, 'model_dump') else validation_result
			
		except Exception as e:
			logger.error(f"Failed to validate workflow: {e}")
			raise
	
	async def save_workflow(self, session_id: str, workflow_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Save workflow with optional override data."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			
			# Use provided data or session data
			if workflow_data:
				session['workflow_data'].update(workflow_data)
			
			current_workflow_data = session['workflow_data']
			
			# Update canvas state
			current_workflow_data['canvas_state'] = await self.canvas_engine.get_canvas_state(session_id)
			current_workflow_data['updated_at'] = datetime.now(timezone.utc).isoformat()
			
			# Save to database
			workflow_id = await self._save_workflow_to_database(current_workflow_data)
			
			# Update session
			session['workflow_id'] = workflow_id
			session['is_dirty'] = False
			
			logger.info(f"Saved workflow {workflow_id} from session {session_id}")
			
			return {
				'id': workflow_id,
				'name': current_workflow_data.get('name', 'Untitled Workflow'),
				'description': current_workflow_data.get('description', ''),
				'updated_at': current_workflow_data['updated_at']
			}
			
		except Exception as e:
			logger.error(f"Failed to save workflow: {e}")
			raise
	
	async def export_workflow(self, session_id: str, options: Any) -> Any:
		"""Export workflow using export manager."""
		try:
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			workflow_data = session['workflow_data']
			
			# Add current canvas state
			workflow_data['canvas_state'] = await self.canvas_engine.get_canvas_state(session_id)
			
			# Export using export manager
			export_result = await self.export_manager.export_workflow(workflow_data, options)
			
			logger.info(f"Exported workflow from session {session_id}")
			return export_result
			
		except Exception as e:
			logger.error(f"Failed to export workflow: {e}")
			raise
	
	async def get_available_components(self, category: Optional[str] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
		"""Get available workflow components."""
		try:
			components = await self.component_library.get_available_components(
				category=category,
				search=search
			)
			
			return [comp.model_dump() if hasattr(comp, 'model_dump') else comp for comp in components]
			
		except Exception as e:
			logger.error(f"Failed to get available components: {e}")
			return []
	
	async def get_component_properties(self, component_id: str) -> Dict[str, Any]:
		"""Get component property definitions."""
		try:
			properties = await self.component_library.get_component_properties(component_id)
			return properties.model_dump() if hasattr(properties, 'model_dump') else properties
			
		except Exception as e:
			logger.error(f"Failed to get component properties: {e}")
			return {}
	
	async def get_workflow_templates(self, category: Optional[str] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
		"""Get available workflow templates."""
		try:
			# This would integrate with the template management system
			from ..template_management import TemplateManager
			
			template_manager = TemplateManager()
			templates = await template_manager.get_templates(
				category=category,
				search=search
			)
			
			return [template.model_dump() if hasattr(template, 'model_dump') else template for template in templates]
			
		except Exception as e:
			logger.error(f"Failed to get workflow templates: {e}")
			return []