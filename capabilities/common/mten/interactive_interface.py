#!/usr/bin/env python3
"""
Interactive Management Interface

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Advanced interactive management interface with drag-and-drop tenant designer,
collaborative editing, real-time analytics dashboard, and mobile-responsive design.
"""

import asyncio
import json
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str

from models import Tenant, TenantStatus, TenantTier
from views import TenantCreateRequest, TenantUpdateRequest


class ComponentType(str, Enum):
	"""Interactive component types"""
	CANVAS = "canvas"
	PROPERTY_PANEL = "property_panel"
	TEMPLATE_LIBRARY = "template_library"
	ANALYTICS_DASHBOARD = "analytics_dashboard"
	COLLABORATION_PANEL = "collaboration_panel"
	MOBILE_NAVIGATOR = "mobile_navigator"


class InteractionEvent(str, Enum):
	"""User interaction events"""
	DRAG_START = "drag_start"
	DRAG_END = "drag_end"
	DROP = "drop"
	COMPONENT_SELECT = "component_select"
	PROPERTY_CHANGE = "property_change"
	TEMPLATE_APPLY = "template_apply"
	COLLABORATION_JOIN = "collaboration_join"
	COLLABORATION_LEAVE = "collaboration_leave"
	REAL_TIME_UPDATE = "real_time_update"


class OperationType(str, Enum):
	"""Collaborative operation types"""
	INSERT = "insert"
	DELETE = "delete"
	RETAIN = "retain"
	FORMAT = "format"
	PROPERTY_UPDATE = "property_update"


@dataclass
class CanvasComponent:
	"""Drag-and-drop canvas component"""
	id: str = field(default_factory=uuid7str)
	name: str = ""
	display_name: str = ""
	component_type: str = ""
	position: Dict[str, float] = field(default_factory=dict)
	properties: Dict[str, Any] = field(default_factory=dict)
	connections: List[str] = field(default_factory=list)
	is_draggable: bool = True
	is_resizable: bool = True
	created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class CollaborativeOperation:
	"""Operational transformation for collaborative editing"""
	id: str = field(default_factory=uuid7str)
	operation_type: OperationType
	position: int
	content: Any
	user_id: str
	timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
	
	def transform_against(self, other: 'CollaborativeOperation') -> 'CollaborativeOperation':
		"""Transform operation against another concurrent operation"""
		if self.timestamp <= other.timestamp:
			if other.operation_type == OperationType.INSERT and other.position <= self.position:
				return CollaborativeOperation(
					operation_type=self.operation_type,
					position=self.position + 1,
					content=self.content,
					user_id=self.user_id,
					timestamp=self.timestamp
				)
		return self


@dataclass
class CollaborationSession:
	"""Real-time collaboration session"""
	id: str = field(default_factory=uuid7str)
	tenant_id: str
	active_users: Set[str] = field(default_factory=set)
	operations: List[CollaborativeOperation] = field(default_factory=list)
	last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
	
	def add_user(self, user_id: str) -> None:
		"""Add user to collaboration session"""
		self.active_users.add(user_id)
		self.last_activity = datetime.now(UTC)
	
	def remove_user(self, user_id: str) -> None:
		"""Remove user from collaboration session"""
		self.active_users.discard(user_id)
		self.last_activity = datetime.now(UTC)
	
	def apply_operation(self, operation: CollaborativeOperation) -> None:
		"""Apply operation with conflict resolution"""
		conflicting_ops = [
			op for op in self.operations[-10:] 
			if abs((op.timestamp - operation.timestamp).total_seconds()) < 0.1
		]
		
		for conflicting_op in conflicting_ops:
			operation = operation.transform_against(conflicting_op)
		
		self.operations.append(operation)
		self.last_activity = datetime.now(UTC)


@dataclass
class AnalyticsWidget:
	"""Analytics dashboard widget"""
	id: str = field(default_factory=uuid7str)
	widget_type: str = ""
	title: str = ""
	data_source: str = ""
	configuration: Dict[str, Any] = field(default_factory=dict)
	position: Dict[str, int] = field(default_factory=dict)
	size: Dict[str, int] = field(default_factory=dict)
	refresh_interval: int = 30  # seconds
	is_real_time: bool = True


@dataclass
class DashboardLayout:
	"""Responsive dashboard layout configuration"""
	id: str = field(default_factory=uuid7str)
	name: str = ""
	device_type: str = "desktop"  # desktop, tablet, mobile
	breakpoints: Dict[str, int] = field(default_factory=lambda: {
		"desktop": 1200, "tablet": 768, "mobile": 480
	})
	widgets: List[AnalyticsWidget] = field(default_factory=list)
	is_responsive: bool = True


class DragDropDesigner:
	"""Advanced drag-and-drop tenant designer"""
	
	def __init__(self):
		self._components: Dict[str, CanvasComponent] = {}
		self._templates: Dict[str, Dict[str, Any]] = {}
		self._active_canvas: Optional[str] = None
		self._history: List[Dict[str, Any]] = []
		self._undo_stack: List[Dict[str, Any]] = []
		self._redo_stack: List[Dict[str, Any]] = []
	
	async def initialize_component_library(self) -> None:
		"""Initialize built-in component library"""
		built_in_components = [
			{
				"name": "web_server",
				"display_name": "Web Server",
				"component_type": "infrastructure",
				"default_properties": {
					"port": 80,
					"ssl_enabled": True,
					"auto_scaling": True
				}
			},
			{
				"name": "database",
				"display_name": "Database",
				"component_type": "data",
				"default_properties": {
					"engine": "postgresql",
					"backup_enabled": True,
					"encryption": True
				}
			},
			{
				"name": "load_balancer",
				"display_name": "Load Balancer",
				"component_type": "networking",
				"default_properties": {
					"algorithm": "round_robin",
					"health_check": True,
					"ssl_termination": True
				}
			},
			{
				"name": "api_gateway",
				"display_name": "API Gateway",
				"component_type": "api",
				"default_properties": {
					"rate_limiting": True,
					"authentication": True,
					"monitoring": True
				}
			},
			{
				"name": "cache",
				"display_name": "Cache Layer",
				"component_type": "performance",
				"default_properties": {
					"engine": "redis",
					"ttl": 3600,
					"clustering": True
				}
			}
		]
		
		for component_data in built_in_components:
			component = CanvasComponent(
				name=component_data["name"],
				display_name=component_data["display_name"],
				component_type=component_data["component_type"],
				properties=component_data["default_properties"]
			)
			self._components[component.id] = component
	
	async def add_component_to_canvas(
		self, 
		component_type: str, 
		position: Dict[str, float],
		properties: Optional[Dict[str, Any]] = None
	) -> CanvasComponent:
		"""Add component to canvas with real-time preview"""
		base_component = next(
			(c for c in self._components.values() if c.name == component_type), 
			None
		)
		
		if not base_component:
			raise ValueError(f"Component type {component_type} not found")
		
		new_component = CanvasComponent(
			name=base_component.name,
			display_name=base_component.display_name,
			component_type=base_component.component_type,
			position=position,
			properties={**base_component.properties, **(properties or {})}
		)
		
		self._components[new_component.id] = new_component
		await self._save_state()
		
		return new_component
	
	async def update_component_properties(
		self, 
		component_id: str, 
		properties: Dict[str, Any]
	) -> CanvasComponent:
		"""Update component properties with validation"""
		component = self._components.get(component_id)
		if not component:
			raise ValueError(f"Component {component_id} not found")
		
		# Validate properties based on component type
		validated_properties = await self._validate_component_properties(
			component.component_type, properties
		)
		
		component.properties.update(validated_properties)
		await self._save_state()
		
		return component
	
	async def connect_components(self, source_id: str, target_id: str) -> bool:
		"""Create connection between components"""
		source = self._components.get(source_id)
		target = self._components.get(target_id)
		
		if not source or not target:
			return False
		
		# Validate connection compatibility
		if await self._can_connect(source, target):
			if target_id not in source.connections:
				source.connections.append(target_id)
			await self._save_state()
			return True
		
		return False
	
	async def generate_tenant_config(self) -> Dict[str, Any]:
		"""Generate tenant configuration from canvas design"""
		config = {
			"components": {},
			"connections": [],
			"infrastructure": {},
			"services": {}
		}
		
		for component in self._components.values():
			component_config = {
				"type": component.component_type,
				"name": component.name,
				"properties": component.properties,
				"position": component.position
			}
			
			config["components"][component.id] = component_config
			
			for connection in component.connections:
				config["connections"].append({
					"source": component.id,
					"target": connection
				})
		
		return config
	
	async def _validate_component_properties(
		self, 
		component_type: str, 
		properties: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Validate component properties based on type"""
		validated = {}
		
		# Basic validation rules by component type
		validation_rules = {
			"infrastructure": {
				"port": lambda x: isinstance(x, int) and 1 <= x <= 65535,
				"ssl_enabled": lambda x: isinstance(x, bool),
				"auto_scaling": lambda x: isinstance(x, bool)
			},
			"data": {
				"engine": lambda x: isinstance(x, str) and x in ["postgresql", "mysql", "mongodb"],
				"backup_enabled": lambda x: isinstance(x, bool),
				"encryption": lambda x: isinstance(x, bool)
			}
		}
		
		rules = validation_rules.get(component_type, {})
		for key, value in properties.items():
			if key in rules:
				if rules[key](value):
					validated[key] = value
			else:
				validated[key] = value  # Allow custom properties
		
		return validated
	
	async def _can_connect(self, source: CanvasComponent, target: CanvasComponent) -> bool:
		"""Check if components can be connected"""
		# Define connection compatibility rules
		compatibility = {
			"api": ["infrastructure", "data", "performance"],
			"infrastructure": ["data", "networking", "performance"],
			"networking": ["infrastructure", "api"],
			"data": ["performance"],
			"performance": ["data"]
		}
		
		return target.component_type in compatibility.get(source.component_type, [])
	
	async def _save_state(self) -> None:
		"""Save current state for undo/redo"""
		state = {
			"components": {k: vars(v) for k, v in self._components.items()},
			"timestamp": datetime.now(UTC).isoformat()
		}
		self._history.append(state)
		self._redo_stack.clear()  # Clear redo stack on new action


class CollaborativeEditor:
	"""Real-time collaborative editing with conflict resolution"""
	
	def __init__(self):
		self._sessions: Dict[str, CollaborationSession] = {}
		self._operational_transform = OperationalTransform()
	
	async def join_session(self, tenant_id: str, user_id: str) -> CollaborationSession:
		"""Join or create collaboration session"""
		session_id = f"tenant_{tenant_id}"
		
		if session_id not in self._sessions:
			self._sessions[session_id] = CollaborationSession(
				id=session_id,
				tenant_id=tenant_id
			)
		
		session = self._sessions[session_id]
		session.add_user(user_id)
		
		return session
	
	async def leave_session(self, tenant_id: str, user_id: str) -> None:
		"""Leave collaboration session"""
		session_id = f"tenant_{tenant_id}"
		session = self._sessions.get(session_id)
		
		if session:
			session.remove_user(user_id)
			
			# Clean up empty sessions
			if not session.active_users:
				del self._sessions[session_id]
	
	async def apply_operation(
		self, 
		tenant_id: str, 
		operation: CollaborativeOperation
	) -> List[CollaborativeOperation]:
		"""Apply operation with operational transformation"""
		session_id = f"tenant_{tenant_id}"
		session = self._sessions.get(session_id)
		
		if not session:
			raise ValueError(f"No active session for tenant {tenant_id}")
		
		session.apply_operation(operation)
		
		# Return transformed operations for other users
		transformed_ops = []
		for user_id in session.active_users:
			if user_id != operation.user_id:
				transformed_op = operation  # In real implementation, apply transformation
				transformed_ops.append(transformed_op)
		
		return transformed_ops
	
	async def get_session_state(self, tenant_id: str) -> Dict[str, Any]:
		"""Get current collaboration session state"""
		session_id = f"tenant_{tenant_id}"
		session = self._sessions.get(session_id)
		
		if not session:
			return {}
		
		return {
			"active_users": list(session.active_users),
			"last_activity": session.last_activity.isoformat(),
			"operation_count": len(session.operations)
		}


class OperationalTransform:
	"""Operational transformation for conflict resolution"""
	
	def transform_operations(
		self, 
		op1: CollaborativeOperation, 
		op2: CollaborativeOperation
	) -> tuple[CollaborativeOperation, CollaborativeOperation]:
		"""Transform two concurrent operations"""
		# Simplified OT - in production would need full OT algorithm
		transformed_op1 = op1.transform_against(op2)
		transformed_op2 = op2.transform_against(op1)
		
		return transformed_op1, transformed_op2


class AnalyticsDashboard:
	"""Comprehensive tenant analytics dashboard"""
	
	def __init__(self):
		self._widgets: Dict[str, AnalyticsWidget] = {}
		self._layouts: Dict[str, DashboardLayout] = {}
		self._data_sources: Dict[str, callable] = {}
		self._real_time_data: Dict[str, Any] = {}
	
	async def initialize_default_widgets(self) -> None:
		"""Initialize default analytics widgets"""
		default_widgets = [
			{
				"widget_type": "metric_card",
				"title": "Active Tenants",
				"data_source": "tenant_count",
				"position": {"x": 0, "y": 0},
				"size": {"width": 4, "height": 2}
			},
			{
				"widget_type": "line_chart",
				"title": "Resource Usage Trend",
				"data_source": "resource_usage",
				"position": {"x": 4, "y": 0},
				"size": {"width": 8, "height": 4}
			},
			{
				"widget_type": "pie_chart",
				"title": "Tenant Distribution by Tier",
				"data_source": "tenant_tiers",
				"position": {"x": 0, "y": 2},
				"size": {"width": 4, "height": 4}
			},
			{
				"widget_type": "table",
				"title": "Recent Activities",
				"data_source": "recent_activities",
				"position": {"x": 0, "y": 6},
				"size": {"width": 12, "height": 4}
			},
			{
				"widget_type": "gauge",
				"title": "System Health Score",
				"data_source": "health_score",
				"position": {"x": 8, "y": 4},
				"size": {"width": 4, "height": 2}
			}
		]
		
		for widget_data in default_widgets:
			widget = AnalyticsWidget(
				widget_type=widget_data["widget_type"],
				title=widget_data["title"],
				data_source=widget_data["data_source"],
				position=widget_data["position"],
				size=widget_data["size"]
			)
			self._widgets[widget.id] = widget
	
	async def create_responsive_layout(self, device_type: str) -> DashboardLayout:
		"""Create responsive layout for different devices"""
		layout = DashboardLayout(
			name=f"{device_type}_layout",
			device_type=device_type,
			widgets=list(self._widgets.values())
		)
		
		# Adjust widget positions and sizes based on device type
		if device_type == "mobile":
			await self._optimize_for_mobile(layout)
		elif device_type == "tablet":
			await self._optimize_for_tablet(layout)
		else:
			await self._optimize_for_desktop(layout)
		
		self._layouts[layout.id] = layout
		return layout
	
	async def get_widget_data(self, widget_id: str) -> Dict[str, Any]:
		"""Get real-time data for widget"""
		widget = self._widgets.get(widget_id)
		if not widget:
			return {}
		
		data_source = self._data_sources.get(widget.data_source)
		if data_source:
			return await data_source()
		
		# Return fixture data for demonstration
		return await self._generate_mock_data(widget.data_source)
	
	async def update_real_time_data(self, data_source: str, data: Any) -> None:
		"""Update real-time data for dashboard"""
		self._real_time_data[data_source] = {
			"data": data,
			"timestamp": datetime.now(UTC),
			"updated_at": datetime.now(UTC).isoformat()
		}
	
	async def _optimize_for_mobile(self, layout: DashboardLayout) -> None:
		"""Optimize layout for mobile devices"""
		# Stack widgets vertically for mobile
		y_offset = 0
		for widget in layout.widgets:
			widget.position = {"x": 0, "y": y_offset}
			widget.size = {"width": 12, "height": max(2, widget.size.get("height", 2))}
			y_offset += widget.size["height"]
	
	async def _optimize_for_tablet(self, layout: DashboardLayout) -> None:
		"""Optimize layout for tablet devices"""
		# Two-column layout for tablets
		left_column_y = 0
		right_column_y = 0
		
		for i, widget in enumerate(layout.widgets):
			if i % 2 == 0:  # Left column
				widget.position = {"x": 0, "y": left_column_y}
				widget.size = {"width": 6, "height": widget.size.get("height", 2)}
				left_column_y += widget.size["height"]
			else:  # Right column
				widget.position = {"x": 6, "y": right_column_y}
				widget.size = {"width": 6, "height": widget.size.get("height", 2)}
				right_column_y += widget.size["height"]
	
	async def _optimize_for_desktop(self, layout: DashboardLayout) -> None:
		"""Optimize layout for desktop devices"""
		# Keep original positions and sizes for desktop
		pass
	
	async def _generate_mock_data(self, data_source: str) -> Dict[str, Any]:
		"""Generate fixture data for demonstration"""
		mock_data = {
			"tenant_count": {"value": 247, "change": "+12%"},
			"resource_usage": {
				"data": [
					{"time": "00:00", "cpu": 45, "memory": 62, "storage": 78},
					{"time": "04:00", "cpu": 32, "memory": 58, "storage": 80},
					{"time": "08:00", "cpu": 67, "memory": 71, "storage": 82},
					{"time": "12:00", "cpu": 89, "memory": 84, "storage": 85},
					{"time": "16:00", "cpu": 76, "memory": 79, "storage": 87},
					{"time": "20:00", "cpu": 54, "memory": 65, "storage": 89}
				]
			},
			"tenant_tiers": {
				"data": [
					{"name": "Free", "value": 45, "percentage": 18.2},
					{"name": "Standard", "value": 89, "percentage": 36.1},
					{"name": "Premium", "value": 76, "percentage": 30.8},
					{"name": "Enterprise", "value": 37, "percentage": 15.0}
				]
			},
			"health_score": {"value": 94.5, "status": "excellent"},
			"recent_activities": {
				"data": [
					{"time": "2025-01-08 10:30", "user": "admin@company.com", "action": "Created tenant", "tenant": "prod-app"},
					{"time": "2025-01-08 10:28", "user": "dev@company.com", "action": "Updated configuration", "tenant": "staging-api"},
					{"time": "2025-01-08 10:25", "user": "ops@company.com", "action": "Deployed template", "tenant": "test-env"},
					{"time": "2025-01-08 10:22", "user": "admin@company.com", "action": "Scaled resources", "tenant": "prod-web"}
				]
			}
		}
		
		return mock_data.get(data_source, {})


class InteractiveManagementInterface:
	"""Main interactive management interface orchestrator"""
	
	def __init__(self):
		self.designer = DragDropDesigner()
		self.collaborative_editor = CollaborativeEditor()
		self.analytics_dashboard = AnalyticsDashboard()
		self._event_handlers: Dict[InteractionEvent, List[callable]] = {}
	
	async def initialize(self) -> None:
		"""Initialize all interface components"""
		await self.designer.initialize_component_library()
		await self.analytics_dashboard.initialize_default_widgets()
		
		# Create responsive layouts
		for device_type in ["desktop", "tablet", "mobile"]:
			await self.analytics_dashboard.create_responsive_layout(device_type)
	
	async def handle_interaction(
		self, 
		event: InteractionEvent, 
		data: Dict[str, Any],
		user_id: str
	) -> Dict[str, Any]:
		"""Handle user interaction events"""
		response = {"success": True, "data": {}}
		
		try:
			if event == InteractionEvent.DRAG_START:
				response["data"] = await self._handle_drag_start(data, user_id)
			elif event == InteractionEvent.DROP:
				response["data"] = await self._handle_drop(data, user_id)
			elif event == InteractionEvent.PROPERTY_CHANGE:
				response["data"] = await self._handle_property_change(data, user_id)
			elif event == InteractionEvent.TEMPLATE_APPLY:
				response["data"] = await self._handle_template_apply(data, user_id)
			elif event == InteractionEvent.COLLABORATION_JOIN:
				response["data"] = await self._handle_collaboration_join(data, user_id)
			else:
				response["data"] = {"message": f"Event {event.value} handled"}
			
			# Trigger event handlers
			handlers = self._event_handlers.get(event, [])
			for handler in handlers:
				await handler(data, user_id)
				
		except Exception as e:
			response["success"] = False
			response["error"] = str(e)
		
		return response
	
	async def get_interface_state(self, tenant_id: str, device_type: str = "desktop") -> Dict[str, Any]:
		"""Get complete interface state for client"""
		# Get dashboard layout
		layout = next(
			(l for l in self.analytics_dashboard._layouts.values() 
			 if l.device_type == device_type),
			None
		)
		
		if not layout:
			layout = await self.analytics_dashboard.create_responsive_layout(device_type)
		
		# Get widget data
		widget_data = {}
		for widget in layout.widgets:
			widget_data[widget.id] = await self.analytics_dashboard.get_widget_data(widget.id)
		
		# Get collaboration state
		collaboration_state = await self.collaborative_editor.get_session_state(tenant_id)
		
		return {
			"canvas": {
				"components": {k: vars(v) for k, v in self.designer._components.items()}
			},
			"dashboard": {
				"layout": vars(layout),
				"widget_data": widget_data
			},
			"collaboration": collaboration_state,
			"device_type": device_type,
			"timestamp": datetime.now(UTC).isoformat()
		}
	
	async def _handle_drag_start(self, data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
		"""Handle drag start event"""
		return {
			"component_id": data.get("component_id"),
			"start_position": data.get("position"),
			"timestamp": datetime.now(UTC).isoformat()
		}
	
	async def _handle_drop(self, data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
		"""Handle drop event"""
		component_type = data.get("component_type")
		position = data.get("position", {"x": 0, "y": 0})
		properties = data.get("properties", {})
		
		component = await self.designer.add_component_to_canvas(
			component_type, position, properties
		)
		
		return {
			"component": vars(component),
			"message": "Component added to canvas"
		}
	
	async def _handle_property_change(self, data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
		"""Handle property change event"""
		component_id = data.get("component_id")
		properties = data.get("properties", {})
		
		component = await self.designer.update_component_properties(
			component_id, properties
		)
		
		# Create collaborative operation
		operation = CollaborativeOperation(
			operation_type=OperationType.PROPERTY_UPDATE,
			position=0,
			content={"component_id": component_id, "properties": properties},
			user_id=user_id
		)
		
		tenant_id = data.get("tenant_id")
		if tenant_id:
			transformed_ops = await self.collaborative_editor.apply_operation(
				tenant_id, operation
			)
		
		return {
			"component": vars(component),
			"message": "Properties updated"
		}
	
	async def _handle_template_apply(self, data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
		"""Handle template application"""
		template_id = data.get("template_id")
		
		# In real implementation, would load template from template system
		return {
			"template_id": template_id,
			"components_added": [],
			"message": "Template applied"
		}
	
	async def _handle_collaboration_join(self, data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
		"""Handle collaboration join"""
		tenant_id = data.get("tenant_id")
		
		session = await self.collaborative_editor.join_session(tenant_id, user_id)
		
		return {
			"session_id": session.id,
			"active_users": list(session.active_users),
			"message": "Joined collaboration session"
		}
	
	def register_event_handler(self, event: InteractionEvent, handler: callable) -> None:
		"""Register event handler for real-time updates"""
		if event not in self._event_handlers:
			self._event_handlers[event] = []
		self._event_handlers[event].append(handler)