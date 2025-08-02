"""
APG Workflow Designer Blueprint

Flask-AppBuilder blueprint for the visual workflow designer interface.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import Blueprint, render_template, request, jsonify, session
from flask_appbuilder import BaseView, ModelView, has_access, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget
from werkzeug.exceptions import BadRequest
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any

from ..models import CrWorkflow, CrWorkflowExecution, CrWorkflowTemplate
from .designer_service import WorkflowDesigner, DesignerConfiguration
from .export_manager import ExportFormat, ExportOptions

logger = logging.getLogger(__name__)

class WorkflowDesignerView(BaseView):
	"""Main workflow designer interface."""
	
	route_base = "/workflow/designer"
	default_view = "designer"
	
	def __init__(self):
		super().__init__()
		self.designer_config = DesignerConfiguration()
		self.workflow_designer = WorkflowDesigner(self.designer_config)
	
	@expose("/")
	@has_access
	def designer(self):
		"""Main designer interface."""
		try:
			# Get user session info
			user_id = session.get('user_id', 'anonymous')
			
			# Get workflow ID from query params
			workflow_id = request.args.get('workflow_id')
			template_id = request.args.get('template_id')
			
			# Initialize designer data
			designer_data = {
				'user_id': user_id,
				'workflow_id': workflow_id,
				'template_id': template_id,
				'config': self.designer_config.model_dump(),
				'api_base_url': '/api/v1/workflow/designer'
			}
			
			return self.render_template(
				'workflow_designer/designer.html',
				designer_data=designer_data,
				page_title="Workflow Designer"
			)
			
		except Exception as e:
			logger.error(f"Failed to load designer: {e}")
			return self.render_template(
				'appbuilder/general/widgets/base_error.html',
				error_message=f"Failed to load workflow designer: {e}"
			), 500
	
	@expose("/components")
	@has_access
	def components(self):
		"""Component library interface."""
		try:
			return self.render_template(
				'workflow_designer/components.html',
				page_title="Component Library"
			)
		except Exception as e:
			logger.error(f"Failed to load components: {e}")
			return self.render_template(
				'appbuilder/general/widgets/base_error.html',
				error_message=f"Failed to load component library: {e}"
			), 500
	
	@expose("/templates")
	@has_access  
	def templates(self):
		"""Workflow templates interface."""
		try:
			return self.render_template(
				'workflow_designer/templates.html',
				page_title="Workflow Templates"
			)
		except Exception as e:
			logger.error(f"Failed to load templates: {e}")
			return self.render_template(
				'appbuilder/general/widgets/base_error.html',
				error_message=f"Failed to load templates: {e}"
			), 500
	
	@expose("/help")
	@has_access
	def help(self):
		"""Designer help and documentation."""
		try:
			return self.render_template(
				'workflow_designer/help.html',
				page_title="Designer Help"
			)
		except Exception as e:
			logger.error(f"Failed to load help: {e}")
			return self.render_template(
				'appbuilder/general/widgets/base_error.html',
				error_message=f"Failed to load help: {e}"
			), 500

class WorkflowDesignerApiView(BaseView):
	"""API endpoints for the workflow designer."""
	
	route_base = "/api/v1/workflow/designer"
	
	def __init__(self):
		super().__init__()
		self.designer_config = DesignerConfiguration()
		self.workflow_designer = WorkflowDesigner(self.designer_config)
	
	@expose("/session", methods=["POST"])
	@has_access
	def create_session(self):
		"""Create a new designer session."""
		try:
			data = request.get_json()
			user_id = session.get('user_id', 'anonymous')
			workflow_id = data.get('workflow_id')
			
			session_data = asyncio.run(self.workflow_designer.create_session(
				user_id=user_id,
				workflow_id=workflow_id
			))
			
			return jsonify({
				'success': True,
				'session': session_data
			})
			
		except Exception as e:
			logger.error(f"Failed to create session: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/session/<session_id>", methods=["GET"])
	@has_access
	def get_session(self, session_id):
		"""Get session state."""
		try:
			session_data = asyncio.run(self.workflow_designer.get_session_state(session_id))
			
			if not session_data:
				return jsonify({
					'success': False,
					'error': 'Session not found'
				}), 404
			
			return jsonify({
				'success': True,
				'session': session_data
			})
			
		except Exception as e:
			logger.error(f"Failed to get session: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/session/<session_id>/component", methods=["POST"])
	@has_access
	def add_component(self, session_id):
		"""Add component to workflow."""
		try:
			data = request.get_json()
			component_type = data.get('component_type')
			position = data.get('position', {})
			config = data.get('config', {})
			
			if not component_type:
				raise BadRequest("component_type is required")
			
			result = asyncio.run(self.workflow_designer.add_component(
				session_id=session_id,
				component_type=component_type,
				position=position,
				config=config
			))
			
			return jsonify({
				'success': True,
				'component': result
			})
			
		except Exception as e:
			logger.error(f"Failed to add component: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/session/<session_id>/component/<component_id>", methods=["DELETE"])
	@has_access
	def remove_component(self, session_id, component_id):
		"""Remove component from workflow."""
		try:
			asyncio.run(self.workflow_designer.remove_component(
				session_id=session_id,
				component_id=component_id
			))
			
			return jsonify({
				'success': True
			})
			
		except Exception as e:
			logger.error(f"Failed to remove component: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/session/<session_id>/component/<component_id>/move", methods=["POST"])
	@has_access
	def move_component(self, session_id, component_id):
		"""Move component to new position."""
		try:
			data = request.get_json()
			position = data.get('position', {})
			
			asyncio.run(self.workflow_designer.move_component(
				session_id=session_id,
				component_id=component_id,
				position=position
			))
			
			return jsonify({
				'success': True
			})
			
		except Exception as e:
			logger.error(f"Failed to move component: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/session/<session_id>/connection", methods=["POST"])
	@has_access
	def add_connection(self, session_id):
		"""Add connection between components."""
		try:
			data = request.get_json()
			source_id = data.get('source_id')
			target_id = data.get('target_id')
			source_port = data.get('source_port', 'output')
			target_port = data.get('target_port', 'input')
			
			if not source_id or not target_id:
				raise BadRequest("source_id and target_id are required")
			
			result = asyncio.run(self.workflow_designer.connect_components(
				session_id=session_id,
				source_id=source_id,
				target_id=target_id,
				source_port=source_port,
				target_port=target_port
			))
			
			return jsonify({
				'success': True,
				'connection': result
			})
			
		except Exception as e:
			logger.error(f"Failed to add connection: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/session/<session_id>/connection/<connection_id>", methods=["DELETE"])
	@has_access
	def remove_connection(self, session_id, connection_id):
		"""Remove connection from workflow."""
		try:
			asyncio.run(self.workflow_designer.remove_connection(
				session_id=session_id,
				connection_id=connection_id
			))
			
			return jsonify({
				'success': True
			})
			
		except Exception as e:
			logger.error(f"Failed to remove connection: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/session/<session_id>/validate", methods=["POST"])
	@has_access
	def validate_workflow(self, session_id):
		"""Validate workflow."""
		try:
			result = asyncio.run(self.workflow_designer.validate_workflow(session_id))
			
			return jsonify({
				'success': True,
				'validation': result
			})
			
		except Exception as e:
			logger.error(f"Failed to validate workflow: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/session/<session_id>/save", methods=["POST"])
	@has_access
	def save_workflow(self, session_id):
		"""Save workflow."""
		try:
			data = request.get_json()
			workflow_data = data.get('workflow', {})
			
			result = asyncio.run(self.workflow_designer.save_workflow(
				session_id=session_id,
				workflow_data=workflow_data
			))
			
			return jsonify({
				'success': True,
				'workflow': result
			})
			
		except Exception as e:
			logger.error(f"Failed to save workflow: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/session/<session_id>/export", methods=["POST"])
	@has_access
	def export_workflow(self, session_id):
		"""Export workflow in specified format."""
		try:
			data = request.get_json()
			export_format = data.get('format', 'json')
			options = data.get('options', {})
			
			# Create export options
			export_options = ExportOptions(
				format=ExportFormat(export_format),
				**options
			)
			
			result = asyncio.run(self.workflow_designer.export_workflow(
				session_id=session_id,
				options=export_options
			))
			
			return jsonify({
				'success': True,
				'export': result.model_dump()
			})
			
		except Exception as e:
			logger.error(f"Failed to export workflow: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/components", methods=["GET"])
	@has_access
	def get_components(self):
		"""Get available components."""
		try:
			category = request.args.get('category')
			search = request.args.get('search')
			
			components = asyncio.run(self.workflow_designer.get_available_components(
				category=category,
				search=search
			))
			
			return jsonify({
				'success': True,
				'components': components
			})
			
		except Exception as e:
			logger.error(f"Failed to get components: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/component/<component_id>/properties", methods=["GET"])
	@has_access
	def get_component_properties(self, component_id):
		"""Get component property definitions."""
		try:
			properties = asyncio.run(self.workflow_designer.get_component_properties(
				component_id=component_id
			))
			
			return jsonify({
				'success': True,
				'properties': properties
			})
			
		except Exception as e:
			logger.error(f"Failed to get component properties: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose("/templates", methods=["GET"])
	@has_access
	def get_templates(self):
		"""Get available workflow templates."""
		try:
			category = request.args.get('category')
			search = request.args.get('search')
			
			templates = asyncio.run(self.workflow_designer.get_workflow_templates(
				category=category,
				search=search
			))
			
			return jsonify({
				'success': True,
				'templates': templates
			})
			
		except Exception as e:
			logger.error(f"Failed to get templates: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500

class WorkflowDesignerChartsView(DirectByChartView):
	"""Charts and analytics for workflow designer."""
	
	datamodel = SQLAInterface(CrWorkflow)
	chart_title = "Workflow Designer Analytics"
	
	definitions = [
		{
			"label": "Workflows by Status",
			"group": "status",
			"series": ["Workflow Count"]
		},
		{
			"label": "Workflows by Category",
			"group": "category",
			"series": ["Workflow Count"]
		},
		{
			"label": "Designer Usage Over Time",
			"group": "created_at",
			"series": ["Usage Count"]
		}
	]

# Create blueprint
workflow_designer_bp = Blueprint(
	'workflow_designer',
	__name__,
	template_folder='templates',
	static_folder='static',
	static_url_path='/static/workflow_designer'
)