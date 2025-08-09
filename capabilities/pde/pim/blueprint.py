"""
Product Lifecycle Management (PLM) Capability - Flask-AppBuilder Blueprint

Flask-AppBuilder blueprint for PLM capability with real-time collaboration,
3D visualization, and comprehensive APG integration.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

from flask import Blueprint, render_template, request, jsonify, session, current_app
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.views import ModelView, MultipleView
from flask_appbuilder.widgets import ListWidget, ShowWidget, FormWidget
from flask_appbuilder.forms import DynamicForm
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, BS3TextAreaFieldWidget, Select2Widget
from flask_appbuilder.actions import action
from wtforms import StringField, TextAreaField, SelectField, IntegerField, FloatField, BooleanField, DateTimeField
from wtforms.validators import DataRequired, Length, NumberRange, Optional
from wtforms.widgets import TextArea, Select
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
from uuid_extensions import uuid7str

from .models import (
	PLProduct, PLProductStructure, PLEngineeringChange, 
	PLProductConfiguration, PLCollaborationSession, PLComplianceRecord,
	PLManufacturingIntegration, PLDigitalTwinBinding
)
from .views import (
	PLMProductView, PLMProductStructureView, PLMEngineeringChangeView,
	PLMProductConfigurationView, PLMCollaborationSessionView, PLMComplianceRecordView,
	PLMManufacturingIntegrationView, PLMDigitalTwinBindingView, PLMDashboardMetricsView
)
from .service import PLMProductService, PLMEngineeringChangeService, PLMCollaborationService
from .ai_service import PLMAIService


# Flask Blueprint Definition
plm_bp = Blueprint(
	'plm', 
	__name__, 
	url_prefix='/plm',
	template_folder='templates',
	static_folder='static'
)


# Custom Widgets for PLM

class PLMCollaborationWidget(FormWidget):
	"""Custom widget for real-time collaboration interface"""
	template = 'plm/widgets/collaboration_widget.html'


class PLM3DVisualizationWidget(ShowWidget):
	"""Custom widget for 3D product visualization"""
	template = 'plm/widgets/3d_visualization_widget.html'


class PLMAnalyticsWidget(ListWidget):
	"""Custom widget for PLM analytics dashboard"""
	template = 'plm/widgets/analytics_widget.html'


# Custom Forms for PLM Views

class PLMProductForm(DynamicForm):
	"""Product creation and editing form"""
	product_name = StringField(
		'Product Name',
		validators=[DataRequired(), Length(min=3, max=200)],
		widget=BS3TextFieldWidget(),
		description="Enter a descriptive product name"
	)
	product_number = StringField(
		'Product Number',
		validators=[DataRequired(), Length(min=3, max=50)],
		widget=BS3TextFieldWidget(),
		description="Unique product identifier/SKU"
	)
	product_description = TextAreaField(
		'Description',
		validators=[Optional(), Length(max=2000)],
		widget=BS3TextAreaFieldWidget(),
		description="Detailed product description"
	)
	product_type = SelectField(
		'Product Type',
		validators=[DataRequired()],
		choices=[
			('manufactured', 'Manufactured'),
			('purchased', 'Purchased'),
			('virtual', 'Virtual'),
			('service', 'Service'),
			('kit', 'Kit'),
			('raw_material', 'Raw Material'),
			('subassembly', 'Subassembly'),
			('finished_good', 'Finished Good')
		],
		widget=Select2Widget(),
		description="Product classification"
	)
	lifecycle_phase = SelectField(
		'Lifecycle Phase',
		validators=[DataRequired()],
		choices=[
			('concept', 'Concept'),
			('design', 'Design'),
			('prototype', 'Prototype'),
			('development', 'Development'),
			('testing', 'Testing'),
			('production', 'Production'),
			('active', 'Active'),
			('mature', 'Mature'),
			('declining', 'Declining'),
			('obsolete', 'Obsolete'),
			('discontinued', 'Discontinued')
		],
		default='concept',
		widget=Select2Widget(),
		description="Current product lifecycle phase"
	)
	target_cost = FloatField(
		'Target Cost',
		validators=[Optional(), NumberRange(min=0)],
		description="Target manufacturing cost"
	)
	current_cost = FloatField(
		'Current Cost',
		validators=[Optional(), NumberRange(min=0)],
		description="Current actual cost"
	)
	unit_of_measure = StringField(
		'Unit of Measure',
		default='each',
		validators=[DataRequired()],
		description="Primary unit of measure"
	)


class PLMEngineeringChangeForm(DynamicForm):
	"""Engineering change request form"""
	change_title = StringField(
		'Change Title',
		validators=[DataRequired(), Length(min=5, max=200)],
		widget=BS3TextFieldWidget(),
		description="Brief descriptive title"
	)
	change_description = TextAreaField(
		'Change Description',
		validators=[DataRequired(), Length(min=10, max=2000)],
		widget=BS3TextAreaFieldWidget(),
		description="Detailed description of the change"
	)
	change_type = SelectField(
		'Change Type',
		validators=[DataRequired()],
		choices=[
			('design', 'Design'),
			('process', 'Process'),
			('documentation', 'Documentation'),
			('cost_reduction', 'Cost Reduction'),
			('quality_improvement', 'Quality Improvement'),
			('safety', 'Safety'),
			('regulatory', 'Regulatory'),
			('urgent', 'Urgent')
		],
		widget=Select2Widget(),
		description="Type of engineering change"
	)
	reason_for_change = TextAreaField(
		'Reason for Change',
		validators=[DataRequired(), Length(min=10, max=1000)],
		widget=BS3TextAreaFieldWidget(),
		description="Justification for this change"
	)
	business_impact = TextAreaField(
		'Business Impact',
		validators=[DataRequired()],
		widget=BS3TextAreaFieldWidget(),
		description="Expected business impact"
	)
	cost_impact = FloatField(
		'Cost Impact',
		validators=[Optional()],
		description="Estimated cost impact (positive for increase, negative for savings)"
	)
	priority = SelectField(
		'Priority',
		validators=[DataRequired()],
		choices=[
			('low', 'Low'),
			('medium', 'Medium'),
			('high', 'High'),
			('critical', 'Critical')
		],
		default='medium',
		widget=Select2Widget(),
		description="Change priority level"
	)


class PLMCollaborationSessionForm(DynamicForm):
	"""Collaboration session creation form"""
	session_name = StringField(
		'Session Name',
		validators=[DataRequired(), Length(min=3, max=200)],
		widget=BS3TextFieldWidget(),
		description="Descriptive session name"
	)
	description = TextAreaField(
		'Description',
		validators=[Optional(), Length(max=1000)],
		widget=BS3TextAreaFieldWidget(),
		description="Session description and agenda"
	)
	session_type = SelectField(
		'Session Type',
		validators=[DataRequired()],
		choices=[
			('design_review', 'Design Review'),
			('change_review', 'Change Review'),
			('brainstorming', 'Brainstorming'),
			('problem_solving', 'Problem Solving'),
			('training', 'Training'),
			('customer_meeting', 'Customer Meeting'),
			('supplier_meeting', 'Supplier Meeting')
		],
		widget=Select2Widget(),
		description="Type of collaboration session"
	)
	scheduled_start = DateTimeField(
		'Scheduled Start',
		validators=[DataRequired()],
		description="Session start date and time"
	)
	scheduled_end = DateTimeField(
		'Scheduled End',
		validators=[DataRequired()],
		description="Session end date and time"
	)
	max_participants = IntegerField(
		'Max Participants',
		validators=[DataRequired(), NumberRange(min=1, max=100)],
		default=20,
		description="Maximum number of participants"
	)
	recording_enabled = BooleanField(
		'Enable Recording',
		description="Record the collaboration session"
	)
	whiteboard_enabled = BooleanField(
		'Enable Whiteboard',
		default=True,
		description="Enable whiteboard collaboration"
	)
	file_sharing_enabled = BooleanField(
		'Enable File Sharing',
		default=True,
		description="Enable file sharing during session"
	)


# PLM Model Views

class PLMProductModelView(ModelView):
	"""Product management view"""
	datamodel = SQLAInterface(PLProduct)
	
	# List view configuration
	list_columns = [
		'product_name', 'product_number', 'product_type', 
		'lifecycle_phase', 'target_cost', 'current_cost', 'created_at'
	]
	search_columns = ['product_name', 'product_number', 'product_description']
	show_fieldsets = [
		('Product Information', {
			'fields': ['product_name', 'product_number', 'product_description', 
					  'product_type', 'lifecycle_phase', 'revision']
		}),
		('Financial Information', {
			'fields': ['target_cost', 'current_cost', 'unit_of_measure']
		}),
		('APG Integration', {
			'fields': ['manufacturing_status', 'digital_twin_id', 'compliance_records']
		}),
		('Metadata', {
			'fields': ['created_at', 'updated_at', 'created_by', 'updated_by']
		})
	]
	edit_form = PLMProductForm
	add_form = PLMProductForm
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	# Widget customization
	show_widget = PLM3DVisualizationWidget
	
	@expose('/api/products/<product_id>/create_digital_twin', methods=['POST'])
	@has_access
	def api_create_digital_twin(self, product_id: str):
		"""API endpoint to create digital twin for product"""
		try:
			# Get current user and tenant
			user_id = session.get('user_id', 'system')
			tenant_id = session.get('tenant_id', 'default_tenant')
			
			# Create digital twin via service
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			twin_id = loop.run_until_complete(
				service.create_digital_twin(product_id, user_id, tenant_id)
			)
			
			if twin_id:
				return jsonify({
					'success': True,
					'digital_twin_id': twin_id,
					'message': 'Digital twin created successfully'
				})
			else:
				return jsonify({
					'success': False,
					'message': 'Failed to create digital twin'
				}), 400
				
		except Exception as e:
			return jsonify({
				'success': False,
				'message': f'Error creating digital twin: {str(e)}'
			}), 500
	
	@action('create_digital_twins', 'Create Digital Twins', 'Create digital twins for selected products', 'fa-cube')
	def create_digital_twins_action(self, items):
		"""Bulk action to create digital twins"""
		created_count = 0
		
		for item in items:
			try:
				user_id = session.get('user_id', 'system')
				tenant_id = session.get('tenant_id', 'default_tenant')
				
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				service = PLMProductService()
				twin_id = loop.run_until_complete(
					service.create_digital_twin(item.product_id, user_id, tenant_id)
				)
				
				if twin_id:
					created_count += 1
					
			except Exception as e:
				current_app.logger.error(f"Failed to create digital twin for {item.product_id}: {e}")
		
		self.update_redirect()
		return f"Successfully created {created_count} digital twins"


class PLMEngineeringChangeModelView(ModelView):
	"""Engineering change management view"""
	datamodel = SQLAInterface(PLEngineeringChange)
	
	list_columns = [
		'change_number', 'change_title', 'change_type', 'status', 
		'priority', 'cost_impact', 'created_at'
	]
	search_columns = ['change_number', 'change_title', 'change_description']
	show_fieldsets = [
		('Change Information', {
			'fields': ['change_number', 'change_title', 'change_description', 
					  'change_type', 'change_category']
		}),
		('Affected Items', {
			'fields': ['affected_products', 'affected_documents']
		}),
		('Business Impact', {
			'fields': ['reason_for_change', 'business_impact', 'cost_impact', 'schedule_impact_days']
		}),
		('Workflow', {
			'fields': ['status', 'priority', 'urgency', 'approvers', 'approved_by']
		}),
		('Implementation', {
			'fields': ['planned_implementation_date', 'actual_implementation_date', 'implementation_notes']
		})
	]
	edit_form = PLMEngineeringChangeForm
	add_form = PLMEngineeringChangeForm
	
	@action('submit_for_approval', 'Submit for Approval', 'Submit selected changes for approval', 'fa-check')
	def submit_for_approval_action(self, items):
		"""Submit changes for approval"""
		submitted_count = 0
		
		for item in items:
			if item.status == 'draft':
				try:
					user_id = session.get('user_id', 'system')
					tenant_id = session.get('tenant_id', 'default_tenant')
					
					loop = asyncio.new_event_loop()
					asyncio.set_event_loop(loop)
					
					service = PLMEngineeringChangeService()
					success = loop.run_until_complete(
						service.submit_change_for_approval(item.change_id, user_id, tenant_id)
					)
					
					if success:
						submitted_count += 1
						
				except Exception as e:
					current_app.logger.error(f"Failed to submit change {item.change_id}: {e}")
		
		self.update_redirect()
		return f"Successfully submitted {submitted_count} changes for approval"


class PLMCollaborationSessionModelView(ModelView):
	"""Collaboration session management view"""
	datamodel = SQLAInterface(PLCollaborationSession)
	
	list_columns = [
		'session_name', 'session_type', 'host_user_id', 'scheduled_start', 
		'scheduled_end', 'status', 'participants'
	]
	search_columns = ['session_name', 'description']
	show_fieldsets = [
		('Session Information', {
			'fields': ['session_name', 'description', 'session_type', 'host_user_id']
		}),
		('Participants', {
			'fields': ['participants', 'invited_users', 'max_participants']
		}),
		('Schedule', {
			'fields': ['scheduled_start', 'scheduled_end', 'actual_start', 'actual_end']
		}),
		('Features', {
			'fields': ['recording_enabled', 'whiteboard_enabled', 'file_sharing_enabled', '3d_viewing_enabled']
		}),
		('Content', {
			'fields': ['products_discussed', 'documents_shared', 'changes_proposed', 'session_notes']
		})
	]
	edit_form = PLMCollaborationSessionForm
	add_form = PLMCollaborationSessionForm
	
	# Custom widget for collaboration interface
	show_widget = PLMCollaborationWidget
	
	@expose('/api/sessions/<session_id>/start', methods=['POST'])
	@has_access
	def api_start_session(self, session_id: str):
		"""API endpoint to start collaboration session"""
		try:
			user_id = session.get('user_id', 'system')
			tenant_id = session.get('tenant_id', 'default_tenant')
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMCollaborationService()
			room_id = loop.run_until_complete(
				service.start_collaboration_session(session_id, user_id, tenant_id)
			)
			
			if room_id:
				return jsonify({
					'success': True,
					'collaboration_room_id': room_id,
					'message': 'Collaboration session started successfully'
				})
			else:
				return jsonify({
					'success': False,
					'message': 'Failed to start collaboration session'
				}), 400
				
		except Exception as e:
			return jsonify({
				'success': False,
				'message': f'Error starting session: {str(e)}'
			}), 500
	
	@expose('/api/sessions/<session_id>/join', methods=['POST'])
	@has_access
	def api_join_session(self, session_id: str):
		"""API endpoint to join collaboration session"""
		try:
			user_id = session.get('user_id', 'system')
			tenant_id = session.get('tenant_id', 'default_tenant')
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMCollaborationService()
			success = loop.run_until_complete(
				service.join_collaboration_session(session_id, user_id, tenant_id)
			)
			
			if success:
				return jsonify({
					'success': True,
					'message': 'Successfully joined collaboration session'
				})
			else:
				return jsonify({
					'success': False,
					'message': 'Failed to join collaboration session'
				}), 400
				
		except Exception as e:
			return jsonify({
				'success': False,
				'message': f'Error joining session: {str(e)}'
			}), 500


# Dashboard and Analytics Views

class PLMDashboardView(BaseView):
	"""PLM main dashboard view"""
	route_base = '/dashboard'
	default_view = 'dashboard'
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Main PLM dashboard"""
		try:
			tenant_id = session.get('tenant_id', 'default_tenant')
			user_id = session.get('user_id', 'system')
			
			# Get dashboard metrics
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			metrics = loop.run_until_complete(
				service.get_dashboard_metrics(tenant_id, user_id)
			)
			
			return self.render_template(
				'plm/dashboard.html',
				metrics=metrics,
				tenant_id=tenant_id
			)
			
		except Exception as e:
			current_app.logger.error(f"Dashboard error: {e}")
			return self.render_template(
				'plm/dashboard.html',
				metrics={},
				error="Failed to load dashboard metrics"
			)
	
	@expose('/analytics')
	@has_access
	def analytics(self):
		"""PLM analytics view"""
		return self.render_template('plm/analytics.html')
	
	@expose('/api/metrics')
	@has_access
	def api_metrics(self):
		"""API endpoint for dashboard metrics"""
		try:
			tenant_id = session.get('tenant_id', 'default_tenant')
			user_id = session.get('user_id', 'system')
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			metrics = loop.run_until_complete(
				service.get_dashboard_metrics(tenant_id, user_id)
			)
			
			return jsonify({
				'success': True,
				'metrics': metrics
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'message': f'Error loading metrics: {str(e)}'
			}), 500


class PLMAnalyticsView(BaseView):
	"""PLM analytics and reporting view"""
	route_base = '/analytics'
	default_view = 'performance'
	
	# Custom analytics widget
	list_widget = PLMAnalyticsWidget
	
	@expose('/performance')
	@has_access
	def performance(self):
		"""Performance analytics view"""
		return self.render_template('plm/analytics/performance.html')
	
	@expose('/compliance')
	@has_access
	def compliance(self):
		"""Compliance analytics view"""
		return self.render_template('plm/analytics/compliance.html')
	
	@expose('/collaboration')
	@has_access
	def collaboration(self):
		"""Collaboration analytics view"""
		return self.render_template('plm/analytics/collaboration.html')
	
	@expose('/api/performance_metrics')
	@has_access
	def api_performance_metrics(self):
		"""API endpoint for performance metrics"""
		try:
			tenant_id = session.get('tenant_id', 'default_tenant')
			
			# Get performance metrics from AI service
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			ai_service = PLMAIService()
			metrics = loop.run_until_complete(
				ai_service.get_lifecycle_performance_insights(tenant_id)
			)
			
			return jsonify({
				'success': True,
				'metrics': metrics
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'message': f'Error loading performance metrics: {str(e)}'
			}), 500


class PLM3DVisualizationView(BaseView):
	"""3D product visualization view"""
	route_base = '/3d'
	default_view = 'viewer'
	
	@expose('/viewer/<product_id>')
	@has_access
	def viewer(self, product_id: str):
		"""3D product viewer"""
		try:
			tenant_id = session.get('tenant_id', 'default_tenant')
			user_id = session.get('user_id', 'system')
			
			# Get product data for 3D visualization
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			product = loop.run_until_complete(
				service.get_product(product_id, user_id, tenant_id)
			)
			
			if not product:
				return self.render_template('plm/error.html', message="Product not found")
			
			return self.render_template(
				'plm/3d_viewer.html',
				product=product,
				product_id=product_id
			)
			
		except Exception as e:
			current_app.logger.error(f"3D viewer error: {e}")
			return self.render_template('plm/error.html', message="Failed to load 3D viewer")
	
	@expose('/api/product/<product_id>/3d_data')
	@has_access
	def api_3d_data(self, product_id: str):
		"""API endpoint for 3D product data"""
		try:
			tenant_id = session.get('tenant_id', 'default_tenant')
			user_id = session.get('user_id', 'system')
			
			# Get 3D data from digital twin
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			data_3d = loop.run_until_complete(
				service.get_product_3d_data(product_id, user_id, tenant_id)
			)
			
			return jsonify({
				'success': True,
				'data_3d': data_3d
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'message': f'Error loading 3D data: {str(e)}'
			}), 500


# Multiple view configuration
class PLMMultipleView(MultipleView):
	"""PLM multiple view container"""
	views = [PLMDashboardView, PLMAnalyticsView, PLM3DVisualizationView]


# Register all views and API endpoints with Flask-AppBuilder
def register_plm_views(appbuilder):
	"""Register all PLM views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		PLMProductModelView,
		"Products",
		icon="fa-cube",
		category="PLM",
		category_icon="fa-industry"
	)
	
	appbuilder.add_view(
		PLMEngineeringChangeModelView,
		"Engineering Changes",
		icon="fa-exchange",
		category="PLM"
	)
	
	appbuilder.add_view(
		PLMCollaborationSessionModelView,
		"Collaboration",
		icon="fa-users",
		category="PLM"
	)
	
	# Dashboard and analytics views
	appbuilder.add_view(
		PLMDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="PLM"
	)
	
	appbuilder.add_view(
		PLMAnalyticsView,
		"Analytics",
		icon="fa-bar-chart",
		category="PLM"
	)
	
	appbuilder.add_view(
		PLM3DVisualizationView,
		"3D Viewer",
		icon="fa-cube",
		category="PLM"
	)
	
	# Multiple view
	appbuilder.add_view(
		PLMMultipleView,
		"PLM Complete",
		icon="fa-industry",
		category="PLM"
	)


# Module exports
__all__ = [
	'plm_bp',
	'PLMProductModelView',
	'PLMEngineeringChangeModelView', 
	'PLMCollaborationSessionModelView',
	'PLMDashboardView',
	'PLMAnalyticsView',
	'PLM3DVisualizationView',
	'PLMMultipleView',
	'register_plm_views'
]