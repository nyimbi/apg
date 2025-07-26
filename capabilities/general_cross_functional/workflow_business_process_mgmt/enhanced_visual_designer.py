"""
APG Workflow & Business Process Management - Enhanced Visual Designer

Advanced graphical workflow designer with comprehensive BPMN 2.0 support,
timing controls, permissions management, and deep instrumentation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from uuid_extensions import uuid7str

from models import (
	APGTenantContext, WBPMProcessDefinition, WBPMProcessActivity, WBPMProcessFlow,
	WBPMServiceResponse, ActivityType, GatewayDirection, EventType, APGBaseModel
)

from workflow_scheduler import (
	WorkflowScheduler, ProcessTimer, TimingAlert, ScheduleType, TimerType, AlertType
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Enhanced Visual Designer Classes
# =============================================================================

class ElementPermission(str, Enum):
	"""Element-level permissions."""
	VIEW = "view"
	EDIT = "edit"
	DELETE = "delete"
	EXECUTE = "execute"
	ASSIGN = "assign"


class CanvasTheme(str, Enum):
	"""Visual canvas themes."""
	LIGHT = "light"
	DARK = "dark"
	HIGH_CONTRAST = "high_contrast"
	COLORBLIND_FRIENDLY = "colorblind_friendly"


class GridSize(str, Enum):
	"""Canvas grid sizes."""
	SMALL = "small"
	MEDIUM = "medium"
	LARGE = "large"
	NONE = "none"


@dataclass
class VisualPosition:
	"""Position and dimensions for visual elements."""
	x: float = 0.0
	y: float = 0.0
	width: float = 100.0
	height: float = 80.0
	z_index: int = 0


@dataclass
class VisualStyle:
	"""Visual styling properties for elements."""
	fill_color: str = "#ffffff"
	border_color: str = "#000000"
	border_width: int = 2
	border_style: str = "solid"  # solid, dashed, dotted
	text_color: str = "#000000"
	font_family: str = "Arial, sans-serif"
	font_size: int = 12
	font_weight: str = "normal"
	opacity: float = 1.0
	shadow: bool = False
	highlight: bool = False


@dataclass
class ElementPermissions:
	"""Permissions configuration for process elements."""
	view_roles: List[str] = field(default_factory=list)
	edit_roles: List[str] = field(default_factory=list)
	execute_roles: List[str] = field(default_factory=list)
	assign_roles: List[str] = field(default_factory=list)
	inherited_from_process: bool = True


@dataclass
class TimingConfiguration:
	"""Timing and SLA configuration for elements."""
	estimated_duration_minutes: Optional[int] = None
	max_duration_minutes: Optional[int] = None
	warning_threshold_percent: int = 80
	escalation_threshold_percent: int = 100
	
	# SLA Configuration
	sla_target_minutes: Optional[int] = None
	sla_critical_minutes: Optional[int] = None
	business_hours_only: bool = False
	
	# Alert Configuration
	alert_recipients: List[str] = field(default_factory=list)
	custom_alert_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EnhancedDiagramElement(APGBaseModel):
	"""Enhanced diagram element with visual and timing properties."""
	element_id: str = field(default_factory=uuid7str)
	element_type: str = ""
	name: str = ""
	description: str = ""
	
	# Visual Properties
	position: VisualPosition = field(default_factory=VisualPosition)
	style: VisualStyle = field(default_factory=VisualStyle)
	is_selected: bool = False
	is_locked: bool = False
	is_visible: bool = True
	
	# Permissions
	permissions: ElementPermissions = field(default_factory=ElementPermissions)
	
	# Timing and Instrumentation
	timing_config: TimingConfiguration = field(default_factory=TimingConfiguration)
	
	# BPMN Properties
	properties: Dict[str, Any] = field(default_factory=dict)
	documentation: str = ""
	
	# Collaboration
	last_modified_by: Optional[str] = None
	modification_timestamp: Optional[datetime] = None
	comments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProcessDiagramCanvas:
	"""Enhanced process diagram canvas with visual controls."""
	canvas_id: str = field(default_factory=uuid7str)
	name: str = ""
	description: str = ""
	
	# Canvas Configuration
	width: float = 2000.0
	height: float = 1500.0
	zoom_level: float = 1.0
	grid_size: GridSize = GridSize.MEDIUM
	theme: CanvasTheme = CanvasTheme.LIGHT
	
	# Elements
	elements: Dict[str, EnhancedDiagramElement] = field(default_factory=dict)
	connections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	
	# View State
	viewport_x: float = 0.0
	viewport_y: float = 0.0
	selected_elements: Set[str] = field(default_factory=set)
	
	# Collaboration State
	active_collaborators: Set[str] = field(default_factory=set)
	collaborative_locks: Dict[str, str] = field(default_factory=dict)  # element_id -> user_id
	
	# Timing and Instrumentation
	process_timing_config: TimingConfiguration = field(default_factory=TimingConfiguration)
	instrumentation_enabled: bool = True
	real_time_monitoring: bool = True


# =============================================================================
# Enhanced Visual Designer Service
# =============================================================================

class EnhancedVisualDesignerService:
	"""Enhanced visual designer with advanced graphical capabilities."""
	
	def __init__(self, scheduler: Optional[WorkflowScheduler] = None):
		self.scheduler = scheduler
		self.active_canvases: Dict[str, ProcessDiagramCanvas] = {}
		self.collaborative_sessions: Dict[str, Set[str]] = {}  # canvas_id -> user_ids
		
		# Designer configuration
		self.auto_save_interval = 30  # seconds
		self.collaboration_timeout = 300  # 5 minutes
		self.max_undo_history = 50
		
		# Visual configuration
		self.default_element_styles = self._initialize_default_styles()
		self.theme_configurations = self._initialize_theme_configurations()
	
	
	# =============================================================================
	# Canvas Management
	# =============================================================================
	
	async def create_canvas(
		self,
		name: str,
		context: APGTenantContext,
		template_id: Optional[str] = None,
		process_definition_id: Optional[str] = None
	) -> WBPMServiceResponse:
		"""Create a new process diagram canvas."""
		try:
			canvas = ProcessDiagramCanvas(
				tenant_id=context.tenant_id,
				created_by=context.user_id,
				updated_by=context.user_id,
				name=name
			)
			
			# Load template or existing process if specified
			if template_id:
				await self._load_template_to_canvas(canvas, template_id, context)
			elif process_definition_id:
				await self._load_process_to_canvas(canvas, process_definition_id, context)
			else:
				# Create default start event
				await self._create_default_start_event(canvas, context)
			
			# Store canvas
			self.active_canvases[canvas.canvas_id] = canvas
			
			logger.info(f"Created canvas {canvas.canvas_id}: {name}")
			
			return WBPMServiceResponse(
				success=True,
				message="Canvas created successfully",
				data={
					"canvas_id": canvas.canvas_id,
					"canvas": self._serialize_canvas(canvas)
				}
			)
			
		except Exception as e:
			logger.error(f"Error creating canvas: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create canvas: {str(e)}"
			)
	
	
	async def load_canvas(self, canvas_id: str, context: APGTenantContext) -> WBPMServiceResponse:
		"""Load an existing canvas."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			# Add user to collaborative session
			if canvas_id not in self.collaborative_sessions:
				self.collaborative_sessions[canvas_id] = set()
			self.collaborative_sessions[canvas_id].add(context.user_id)
			canvas.active_collaborators.add(context.user_id)
			
			return WBPMServiceResponse(
				success=True,
				message="Canvas loaded successfully",
				data={
					"canvas": self._serialize_canvas(canvas),
					"collaborators": list(canvas.active_collaborators)
				}
			)
			
		except Exception as e:
			logger.error(f"Error loading canvas: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to load canvas: {str(e)}"
			)
	
	
	async def save_canvas(self, canvas_id: str, context: APGTenantContext) -> WBPMServiceResponse:
		"""Save canvas to persistent storage."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			canvas.updated_by = context.user_id
			canvas.updated_at = datetime.utcnow()
			
			# In production, would save to database
			logger.info(f"Saved canvas {canvas_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Canvas saved successfully"
			)
			
		except Exception as e:
			logger.error(f"Error saving canvas: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to save canvas: {str(e)}"
			)
	
	
	# =============================================================================
	# Element Management
	# =============================================================================
	
	async def add_element(
		self,
		canvas_id: str,
		element_type: str,
		position: VisualPosition,
		context: APGTenantContext,
		element_name: Optional[str] = None,
		properties: Optional[Dict[str, Any]] = None
	) -> WBPMServiceResponse:
		"""Add a new element to the canvas."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			# Create enhanced diagram element
			element = EnhancedDiagramElement(
				tenant_id=context.tenant_id,
				created_by=context.user_id,
				updated_by=context.user_id,
				element_type=element_type,
				name=element_name or f"New {element_type}",
				position=position,
				style=self._get_default_style_for_type(element_type),
				properties=properties or {},
				last_modified_by=context.user_id,
				modification_timestamp=datetime.utcnow()
			)
			
			# Set default timing configuration based on element type
			element.timing_config = self._get_default_timing_config(element_type)
			
			# Set default permissions
			element.permissions = self._get_default_permissions(element_type, context)
			
			# Add to canvas
			canvas.elements[element.element_id] = element
			
			# Create timer if timing is configured
			if self.scheduler and element.timing_config.estimated_duration_minutes:
				await self._create_element_timer(element, context)
			
			logger.info(f"Added element {element.element_id} to canvas {canvas_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Element added successfully",
				data={
					"element_id": element.element_id,
					"element": self._serialize_element(element)
				}
			)
			
		except Exception as e:
			logger.error(f"Error adding element: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to add element: {str(e)}"
			)
	
	
	async def update_element(
		self,
		canvas_id: str,
		element_id: str,
		updates: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Update an existing element."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			if element_id not in canvas.elements:
				return WBPMServiceResponse(
					success=False,
					message="Element not found"
				)
			
			element = canvas.elements[element_id]
			
			# Check if element is locked by another user
			if element_id in canvas.collaborative_locks:
				locked_by = canvas.collaborative_locks[element_id]
				if locked_by != context.user_id:
					return WBPMServiceResponse(
						success=False,
						message=f"Element is locked by {locked_by}"
					)
			
			# Apply updates
			for key, value in updates.items():
				if key == "position" and isinstance(value, dict):
					for pos_key, pos_value in value.items():
						setattr(element.position, pos_key, pos_value)
				elif key == "style" and isinstance(value, dict):
					for style_key, style_value in value.items():
						setattr(element.style, style_key, style_value)
				elif key == "timing_config" and isinstance(value, dict):
					for timing_key, timing_value in value.items():
						setattr(element.timing_config, timing_key, timing_value)
				elif key == "permissions" and isinstance(value, dict):
					for perm_key, perm_value in value.items():
						setattr(element.permissions, perm_key, perm_value)
				elif hasattr(element, key):
					setattr(element, key, value)
			
			# Update modification tracking
			element.last_modified_by = context.user_id
			element.modification_timestamp = datetime.utcnow()
			element.updated_by = context.user_id
			element.updated_at = datetime.utcnow()
			
			# Update timer if timing configuration changed
			if self.scheduler and "timing_config" in updates:
				await self._update_element_timer(element, context)
			
			logger.info(f"Updated element {element_id} in canvas {canvas_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Element updated successfully",
				data={
					"element": self._serialize_element(element)
				}
			)
			
		except Exception as e:
			logger.error(f"Error updating element: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to update element: {str(e)}"
			)
	
	
	async def delete_element(
		self,
		canvas_id: str,
		element_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Delete an element from the canvas."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			if element_id not in canvas.elements:
				return WBPMServiceResponse(
					success=False,
					message="Element not found"
				)
			
			# Check permissions
			element = canvas.elements[element_id]
			if not self._check_permission(element, ElementPermission.DELETE, context):
				return WBPMServiceResponse(
					success=False,
					message="Insufficient permissions to delete element"
				)
			
			# Remove element
			del canvas.elements[element_id]
			
			# Remove any connections involving this element
			connections_to_remove = []
			for conn_id, connection in canvas.connections.items():
				if (connection.get('source_id') == element_id or 
					connection.get('target_id') == element_id):
					connections_to_remove.append(conn_id)
			
			for conn_id in connections_to_remove:
				del canvas.connections[conn_id]
			
			# Remove from selected elements
			canvas.selected_elements.discard(element_id)
			
			# Remove collaborative lock if exists
			if element_id in canvas.collaborative_locks:
				del canvas.collaborative_locks[element_id]
			
			logger.info(f"Deleted element {element_id} from canvas {canvas_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Element deleted successfully"
			)
			
		except Exception as e:
			logger.error(f"Error deleting element: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to delete element: {str(e)}"
			)
	
	
	# =============================================================================
	# Connection Management
	# =============================================================================
	
	async def create_connection(
		self,
		canvas_id: str,
		source_id: str,
		target_id: str,
		context: APGTenantContext,
		connection_type: str = "sequence_flow",
		properties: Optional[Dict[str, Any]] = None
	) -> WBPMServiceResponse:
		"""Create a connection between two elements."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			# Validate source and target elements exist
			if source_id not in canvas.elements or target_id not in canvas.elements:
				return WBPMServiceResponse(
					success=False,
					message="Source or target element not found"
				)
			
			# Validate connection rules
			validation_result = await self._validate_connection(
				canvas.elements[source_id],
				canvas.elements[target_id],
				connection_type
			)
			
			if not validation_result.success:
				return validation_result
			
			# Create connection
			connection_id = f"conn_{uuid7str()}"
			connection = {
				"connection_id": connection_id,
				"connection_type": connection_type,
				"source_id": source_id,
				"target_id": target_id,
				"properties": properties or {},
				"style": self._get_default_connection_style(connection_type),
				"created_by": context.user_id,
				"created_at": datetime.utcnow().isoformat(),
				"waypoints": []  # For visual routing
			}
			
			canvas.connections[connection_id] = connection
			
			logger.info(f"Created connection {connection_id} in canvas {canvas_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Connection created successfully",
				data={
					"connection_id": connection_id,
					"connection": connection
				}
			)
			
		except Exception as e:
			logger.error(f"Error creating connection: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create connection: {str(e)}"
			)
	
	
	# =============================================================================
	# Timing and Instrumentation
	# =============================================================================
	
	async def configure_process_timing(
		self,
		canvas_id: str,
		timing_config: TimingConfiguration,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Configure timing for the entire process."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			canvas.process_timing_config = timing_config
			
			# Create process-level timer if scheduler is available
			if self.scheduler and timing_config.max_duration_minutes:
				# In production, would create timer when process instance starts
				logger.info(f"Process timing configured for canvas {canvas_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Process timing configured successfully"
			)
			
		except Exception as e:
			logger.error(f"Error configuring process timing: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to configure process timing: {str(e)}"
			)
	
	
	async def configure_element_timing(
		self,
		canvas_id: str,
		element_id: str,
		timing_config: TimingConfiguration,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Configure timing for a specific element."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			if element_id not in canvas.elements:
				return WBPMServiceResponse(
					success=False,
					message="Element not found"
				)
			
			element = canvas.elements[element_id]
			element.timing_config = timing_config
			
			# Update element timer if scheduler is available
			if self.scheduler:
				await self._update_element_timer(element, context)
			
			logger.info(f"Element timing configured for {element_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Element timing configured successfully"
			)
			
		except Exception as e:
			logger.error(f"Error configuring element timing: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to configure element timing: {str(e)}"
			)
	
	
	# =============================================================================
	# Permissions Management
	# =============================================================================
	
	async def configure_element_permissions(
		self,
		canvas_id: str,
		element_id: str,
		permissions: ElementPermissions,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Configure permissions for a specific element."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			if element_id not in canvas.elements:
				return WBPMServiceResponse(
					success=False,
					message="Element not found"
				)
			
			element = canvas.elements[element_id]
			element.permissions = permissions
			
			logger.info(f"Element permissions configured for {element_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Element permissions configured successfully"
			)
			
		except Exception as e:
			logger.error(f"Error configuring element permissions: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to configure element permissions: {str(e)}"
			)
	
	
	# =============================================================================
	# Visual Operations
	# =============================================================================
	
	async def update_canvas_view(
		self,
		canvas_id: str,
		view_updates: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Update canvas view settings."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			# Apply view updates
			for key, value in view_updates.items():
				if hasattr(canvas, key):
					setattr(canvas, key, value)
			
			return WBPMServiceResponse(
				success=True,
				message="Canvas view updated successfully"
			)
			
		except Exception as e:
			logger.error(f"Error updating canvas view: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to update canvas view: {str(e)}"
			)
	
	
	async def apply_theme(
		self,
		canvas_id: str,
		theme: CanvasTheme,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Apply a theme to the canvas."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			canvas.theme = theme
			
			# Apply theme styles to all elements
			theme_config = self.theme_configurations.get(theme, {})
			for element in canvas.elements.values():
				self._apply_theme_to_element(element, theme_config)
			
			return WBPMServiceResponse(
				success=True,
				message="Theme applied successfully"
			)
			
		except Exception as e:
			logger.error(f"Error applying theme: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to apply theme: {str(e)}"
			)
	
	
	# =============================================================================
	# Collaboration Features
	# =============================================================================
	
	async def lock_element(
		self,
		canvas_id: str,
		element_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Lock an element for exclusive editing."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			if element_id not in canvas.elements:
				return WBPMServiceResponse(
					success=False,
					message="Element not found"
				)
			
			# Check if already locked
			if element_id in canvas.collaborative_locks:
				locked_by = canvas.collaborative_locks[element_id]
				if locked_by != context.user_id:
					return WBPMServiceResponse(
						success=False,
						message=f"Element is already locked by {locked_by}"
					)
			
			# Lock the element
			canvas.collaborative_locks[element_id] = context.user_id
			
			return WBPMServiceResponse(
				success=True,
				message="Element locked successfully"
			)
			
		except Exception as e:
			logger.error(f"Error locking element: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to lock element: {str(e)}"
			)
	
	
	async def unlock_element(
		self,
		canvas_id: str,
		element_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Unlock an element."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			if element_id in canvas.collaborative_locks:
				locked_by = canvas.collaborative_locks[element_id]
				if locked_by == context.user_id:
					del canvas.collaborative_locks[element_id]
				else:
					return WBPMServiceResponse(
						success=False,
						message="Cannot unlock element locked by another user"
					)
			
			return WBPMServiceResponse(
				success=True,
				message="Element unlocked successfully"
			)
			
		except Exception as e:
			logger.error(f"Error unlocking element: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to unlock element: {str(e)}"
			)
	
	
	# =============================================================================
	# Export and Import
	# =============================================================================
	
	async def export_to_bpmn(self, canvas_id: str, context: APGTenantContext) -> WBPMServiceResponse:
		"""Export canvas to BPMN 2.0 XML format."""
		try:
			if canvas_id not in self.active_canvases:
				return WBPMServiceResponse(
					success=False,
					message="Canvas not found"
				)
			
			canvas = self.active_canvases[canvas_id]
			
			# Generate BPMN XML
			bpmn_xml = await self._generate_bpmn_xml(canvas)
			
			return WBPMServiceResponse(
				success=True,
				message="BPMN export completed successfully",
				data={
					"bpmn_xml": bpmn_xml,
					"filename": f"{canvas.name.replace(' ', '_')}.bpmn"
				}
			)
			
		except Exception as e:
			logger.error(f"Error exporting to BPMN: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to export to BPMN: {str(e)}"
			)
	
	
	# =============================================================================
	# Private Helper Methods
	# =============================================================================
	
	def _serialize_canvas(self, canvas: ProcessDiagramCanvas) -> Dict[str, Any]:
		"""Serialize canvas for client transmission."""
		return {
			"canvas_id": canvas.canvas_id,
			"name": canvas.name,
			"description": canvas.description,
			"width": canvas.width,
			"height": canvas.height,
			"zoom_level": canvas.zoom_level,
			"grid_size": canvas.grid_size,
			"theme": canvas.theme,
			"viewport_x": canvas.viewport_x,
			"viewport_y": canvas.viewport_y,
			"elements": {
				element_id: self._serialize_element(element)
				for element_id, element in canvas.elements.items()
			},
			"connections": canvas.connections,
			"selected_elements": list(canvas.selected_elements),
			"active_collaborators": list(canvas.active_collaborators),
			"collaborative_locks": canvas.collaborative_locks,
			"process_timing_config": self._serialize_timing_config(canvas.process_timing_config),
			"instrumentation_enabled": canvas.instrumentation_enabled,
			"real_time_monitoring": canvas.real_time_monitoring
		}
	
	
	def _serialize_element(self, element: EnhancedDiagramElement) -> Dict[str, Any]:
		"""Serialize element for client transmission."""
		return {
			"element_id": element.element_id,
			"element_type": element.element_type,
			"name": element.name,
			"description": element.description,
			"position": {
				"x": element.position.x,
				"y": element.position.y,
				"width": element.position.width,
				"height": element.position.height,
				"z_index": element.position.z_index
			},
			"style": {
				"fill_color": element.style.fill_color,
				"border_color": element.style.border_color,
				"border_width": element.style.border_width,
				"border_style": element.style.border_style,
				"text_color": element.style.text_color,
				"font_family": element.style.font_family,
				"font_size": element.style.font_size,
				"font_weight": element.style.font_weight,
				"opacity": element.style.opacity,
				"shadow": element.style.shadow,
				"highlight": element.style.highlight
			},
			"is_selected": element.is_selected,
			"is_locked": element.is_locked,
			"is_visible": element.is_visible,
			"permissions": {
				"view_roles": element.permissions.view_roles,
				"edit_roles": element.permissions.edit_roles,
				"execute_roles": element.permissions.execute_roles,
				"assign_roles": element.permissions.assign_roles,
				"inherited_from_process": element.permissions.inherited_from_process
			},
			"timing_config": self._serialize_timing_config(element.timing_config),
			"properties": element.properties,
			"documentation": element.documentation,
			"last_modified_by": element.last_modified_by,
			"modification_timestamp": element.modification_timestamp.isoformat() if element.modification_timestamp else None,
			"comments": element.comments
		}
	
	
	def _serialize_timing_config(self, timing_config: TimingConfiguration) -> Dict[str, Any]:
		"""Serialize timing configuration."""
		return {
			"estimated_duration_minutes": timing_config.estimated_duration_minutes,
			"max_duration_minutes": timing_config.max_duration_minutes,
			"warning_threshold_percent": timing_config.warning_threshold_percent,
			"escalation_threshold_percent": timing_config.escalation_threshold_percent,
			"sla_target_minutes": timing_config.sla_target_minutes,
			"sla_critical_minutes": timing_config.sla_critical_minutes,
			"business_hours_only": timing_config.business_hours_only,
			"alert_recipients": timing_config.alert_recipients,
			"custom_alert_rules": timing_config.custom_alert_rules
		}
	
	
	def _get_default_style_for_type(self, element_type: str) -> VisualStyle:
		"""Get default visual style for element type."""
		style_map = {
			"startEvent": VisualStyle(
				fill_color="#e1f5fe",
				border_color="#01579b",
				border_width=3
			),
			"endEvent": VisualStyle(
				fill_color="#fce4ec",
				border_color="#880e4f",
				border_width=3
			),
			"userTask": VisualStyle(
				fill_color="#fff3e0",
				border_color="#e65100",
				border_width=2
			),
			"serviceTask": VisualStyle(
				fill_color="#f3e5f5",
				border_color="#4a148c",
				border_width=2
			),
			"exclusiveGateway": VisualStyle(
				fill_color="#fff8e1",
				border_color="#ff6f00",
				border_width=2
			),
			"parallelGateway": VisualStyle(
				fill_color="#e8f5e8",
				border_color="#2e7d32",
				border_width=2
			)
		}
		
		return style_map.get(element_type, VisualStyle())
	
	
	def _get_default_timing_config(self, element_type: str) -> TimingConfiguration:
		"""Get default timing configuration for element type."""
		config_map = {
			"userTask": TimingConfiguration(
				estimated_duration_minutes=60,
				max_duration_minutes=480,  # 8 hours
				warning_threshold_percent=75,
				escalation_threshold_percent=100
			),
			"serviceTask": TimingConfiguration(
				estimated_duration_minutes=5,
				max_duration_minutes=30,
				warning_threshold_percent=80,
				escalation_threshold_percent=100
			)
		}
		
		return config_map.get(element_type, TimingConfiguration())
	
	
	def _get_default_permissions(self, element_type: str, context: APGTenantContext) -> ElementPermissions:
		"""Get default permissions for element type."""
		return ElementPermissions(
			view_roles=["user"],
			edit_roles=["designer", "admin"],
			execute_roles=["user", "designer", "admin"],
			assign_roles=["manager", "admin"],
			inherited_from_process=True
		)
	
	
	def _get_default_connection_style(self, connection_type: str) -> Dict[str, Any]:
		"""Get default style for connection type."""
		return {
			"stroke_color": "#000000",
			"stroke_width": 2,
			"stroke_style": "solid",
			"arrow_head": True,
			"arrow_style": "filled"
		}
	
	
	def _check_permission(self, element: EnhancedDiagramElement, permission: ElementPermission, context: APGTenantContext) -> bool:
		"""Check if user has permission for element operation."""
		# Simplified permission check - in production would integrate with APG auth_rbac
		user_roles = context.user_roles
		
		if permission == ElementPermission.VIEW:
			return any(role in element.permissions.view_roles for role in user_roles)
		elif permission == ElementPermission.EDIT:
			return any(role in element.permissions.edit_roles for role in user_roles)
		elif permission == ElementPermission.DELETE:
			return any(role in element.permissions.edit_roles for role in user_roles)
		elif permission == ElementPermission.EXECUTE:
			return any(role in element.permissions.execute_roles for role in user_roles)
		elif permission == ElementPermission.ASSIGN:
			return any(role in element.permissions.assign_roles for role in user_roles)
		
		return False
	
	
	async def _validate_connection(
		self,
		source_element: EnhancedDiagramElement,
		target_element: EnhancedDiagramElement,
		connection_type: str
	) -> WBPMServiceResponse:
		"""Validate if connection between elements is allowed."""
		# BPMN connection rules validation
		valid_connections = {
			"startEvent": ["userTask", "serviceTask", "exclusiveGateway", "parallelGateway"],
			"userTask": ["userTask", "serviceTask", "exclusiveGateway", "parallelGateway", "endEvent"],
			"serviceTask": ["userTask", "serviceTask", "exclusiveGateway", "parallelGateway", "endEvent"],
			"exclusiveGateway": ["userTask", "serviceTask", "exclusiveGateway", "parallelGateway", "endEvent"],
			"parallelGateway": ["userTask", "serviceTask", "exclusiveGateway", "parallelGateway", "endEvent"],
			"endEvent": []
		}
		
		allowed_targets = valid_connections.get(source_element.element_type, [])
		
		if target_element.element_type not in allowed_targets:
			return WBPMServiceResponse(
				success=False,
				message=f"Invalid connection: {source_element.element_type} cannot connect to {target_element.element_type}"
			)
		
		return WBPMServiceResponse(success=True, message="Connection is valid")
	
	
	async def _create_element_timer(self, element: EnhancedDiagramElement, context: APGTenantContext) -> None:
		"""Create timer for element with timing configuration."""
		if not self.scheduler or not element.timing_config.estimated_duration_minutes:
			return
		
		from workflow_scheduler import create_process_timer
		
		timer = create_process_timer(
			process_instance_id="design_time",  # Would be actual instance ID in production
			duration_minutes=element.timing_config.estimated_duration_minutes,
			tenant_context=context,
			activity_id=element.element_id,
			alert_recipients=element.timing_config.alert_recipients
		)
		
		await self.scheduler.create_process_timer(timer)
	
	
	async def _update_element_timer(self, element: EnhancedDiagramElement, context: APGTenantContext) -> None:
		"""Update timer for element with new timing configuration."""
		if not self.scheduler:
			return
		
		# In production, would update existing timer
		logger.info(f"Timer updated for element {element.element_id}")
	
	
	def _initialize_default_styles(self) -> Dict[str, VisualStyle]:
		"""Initialize default visual styles for element types."""
		return {
			"startEvent": self._get_default_style_for_type("startEvent"),
			"endEvent": self._get_default_style_for_type("endEvent"),
			"userTask": self._get_default_style_for_type("userTask"),
			"serviceTask": self._get_default_style_for_type("serviceTask"),
			"exclusiveGateway": self._get_default_style_for_type("exclusiveGateway"),
			"parallelGateway": self._get_default_style_for_type("parallelGateway")
		}
	
	
	def _initialize_theme_configurations(self) -> Dict[CanvasTheme, Dict[str, Any]]:
		"""Initialize theme configurations."""
		return {
			CanvasTheme.LIGHT: {
				"background_color": "#ffffff",
				"grid_color": "#e0e0e0",
				"text_color": "#000000",
				"border_color": "#cccccc"
			},
			CanvasTheme.DARK: {
				"background_color": "#2e2e2e",
				"grid_color": "#505050",
				"text_color": "#ffffff",
				"border_color": "#666666"
			},
			CanvasTheme.HIGH_CONTRAST: {
				"background_color": "#000000",
				"grid_color": "#ffffff",
				"text_color": "#ffffff",
				"border_color": "#ffffff"
			},
			CanvasTheme.COLORBLIND_FRIENDLY: {
				"background_color": "#ffffff",
				"grid_color": "#e0e0e0",
				"text_color": "#000000",
				"border_color": "#0066cc"
			}
		}
	
	
	def _apply_theme_to_element(self, element: EnhancedDiagramElement, theme_config: Dict[str, Any]) -> None:
		"""Apply theme configuration to an element."""
		if "text_color" in theme_config:
			element.style.text_color = theme_config["text_color"]
		if "border_color" in theme_config:
			element.style.border_color = theme_config["border_color"]
	
	
	async def _create_default_start_event(self, canvas: ProcessDiagramCanvas, context: APGTenantContext) -> None:
		"""Create default start event for new canvas."""
		start_event = EnhancedDiagramElement(
			tenant_id=context.tenant_id,
			created_by=context.user_id,
			updated_by=context.user_id,
			element_type="startEvent",
			name="Start",
			position=VisualPosition(x=100, y=100, width=36, height=36),
			style=self._get_default_style_for_type("startEvent")
		)
		
		canvas.elements[start_event.element_id] = start_event
	
	
	async def _load_template_to_canvas(self, canvas: ProcessDiagramCanvas, template_id: str, context: APGTenantContext) -> None:
		"""Load template into canvas."""
		# In production, would load actual template
		logger.info(f"Loading template {template_id} into canvas {canvas.canvas_id}")
	
	
	async def _load_process_to_canvas(self, canvas: ProcessDiagramCanvas, process_definition_id: str, context: APGTenantContext) -> None:
		"""Load existing process into canvas."""
		# In production, would load actual process definition
		logger.info(f"Loading process {process_definition_id} into canvas {canvas.canvas_id}")
	
	
	async def _generate_bpmn_xml(self, canvas: ProcessDiagramCanvas) -> str:
		"""Generate BPMN 2.0 XML from canvas."""
		# Simplified BPMN XML generation
		xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
		xml_definitions = f'''
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"
             targetNamespace="http://datacraft.co.ke/bpmn"
             id="{canvas.canvas_id}">
  <process id="process_{canvas.canvas_id}" name="{canvas.name}">
'''
		
		# Add elements
		xml_elements = ""
		for element in canvas.elements.values():
			xml_elements += f'    <{element.element_type} id="{element.element_id}" name="{element.name}" />\n'
		
		# Add connections
		xml_flows = ""
		for connection in canvas.connections.values():
			xml_flows += f'''    <sequenceFlow id="{connection['connection_id']}" 
                               sourceRef="{connection['source_id']}" 
                               targetRef="{connection['target_id']}" />\n'''
		
		xml_footer = '''  </process>
</definitions>'''
		
		return xml_header + xml_definitions + xml_elements + xml_flows + xml_footer


# =============================================================================
# Example Usage
# =============================================================================

async def example_enhanced_designer_usage():
	"""Example usage of enhanced visual designer."""
	from workflow_scheduler import SchedulerFactory
	
	# Create tenant context
	tenant_context = APGTenantContext(
		tenant_id="example_tenant",
		user_id="designer@example.com",
		user_roles=["designer", "user"],
		permissions=["workflow_design", "workflow_execute"]
	)
	
	# Get scheduler
	scheduler = await SchedulerFactory.get_scheduler(tenant_context)
	
	# Create enhanced designer
	designer = EnhancedVisualDesignerService(scheduler)
	
	# Create new canvas
	canvas_result = await designer.create_canvas(
		name="Employee Onboarding Process",
		context=tenant_context
	)
	
	if canvas_result.success:
		canvas_id = canvas_result.data["canvas_id"]
		print(f"Canvas created: {canvas_id}")
		
		# Add user task with timing configuration
		timing_config = TimingConfiguration(
			estimated_duration_minutes=120,
			max_duration_minutes=480,
			warning_threshold_percent=75,
			sla_target_minutes=240,
			alert_recipients=["hr@example.com"]
		)
		
		element_result = await designer.add_element(
			canvas_id=canvas_id,
			element_type="userTask",
			position=VisualPosition(x=200, y=150, width=100, height=80),
			context=tenant_context,
			element_name="Complete HR Forms"
		)
		
		if element_result.success:
			element_id = element_result.data["element_id"]
			
			# Configure timing for the element
			await designer.configure_element_timing(
				canvas_id=canvas_id,
				element_id=element_id,
				timing_config=timing_config,
				context=tenant_context
			)
			
			print(f"User task added with timing: {element_id}")
		
		# Export to BPMN
		export_result = await designer.export_to_bpmn(canvas_id, tenant_context)
		if export_result.success:
			print(f"BPMN exported: {export_result.data['filename']}")


if __name__ == "__main__":
	asyncio.run(example_enhanced_designer_usage())