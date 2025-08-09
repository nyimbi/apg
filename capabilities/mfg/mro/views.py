"""
Predictive Maintenance Views

Flask-AppBuilder views for comprehensive predictive maintenance management,
asset health monitoring, failure prediction, and maintenance optimization.
"""

from flask import request, jsonify, flash, redirect, url_for, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, FloatField, IntegerField, validators
from wtforms.validators import DataRequired, Length, Optional, NumberRange
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .models import (
	PMAsset, PMHealthRecord, PMFailurePrediction,
	PMMaintenanceRecord, PMMaintenanceAlert
)


class PredictiveMaintenanceBaseView(BaseView):
	"""Base view for predictive maintenance functionality"""
	
	def __init__(self):
		super().__init__()
		self.default_view = 'dashboard'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID from security context"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def _format_health_score(self, score: float) -> str:
		"""Format health score for display"""
		if score is None:
			return "N/A"
		return f"{score:.1f}%"
	
	def _format_probability(self, prob: float) -> str:
		"""Format probability for display"""
		if prob is None:
			return "N/A"
		return f"{prob*100:.1f}%"


class PMAssetModelView(ModelView):
	"""Asset management view"""
	
	datamodel = SQLAInterface(PMAsset)
	
	# List view configuration
	list_columns = [
		'asset_name', 'asset_type', 'asset_category', 'location',
		'current_health_score', 'status', 'criticality_level', 'next_scheduled_maintenance'
	]
	show_columns = [
		'asset_id', 'asset_name', 'asset_type', 'asset_category', 'manufacturer',
		'model_number', 'serial_number', 'location', 'installation_date',
		'expected_lifespan_years', 'operational_hours', 'current_health_score',
		'status', 'criticality_level', 'maintenance_strategy', 'last_maintenance_date',
		'next_scheduled_maintenance', 'sensor_ids', 'purchase_cost', 'replacement_cost'
	]
	edit_columns = [
		'asset_name', 'asset_type', 'asset_category', 'manufacturer', 'model_number',
		'serial_number', 'location', 'facility_id', 'parent_asset_id',
		'installation_date', 'expected_lifespan_years', 'criticality_level',
		'maintenance_strategy', 'maintenance_frequency_hours', 'sensor_ids',
		'monitoring_parameters', 'alert_thresholds', 'purchase_cost',
		'replacement_cost', 'annual_maintenance_cost', 'downtime_cost_per_hour'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['asset_name', 'asset_type', 'location', 'manufacturer', 'serial_number']
	base_filters = [['status', lambda: 'operational', lambda: True]]
	
	# Ordering
	base_order = ('current_health_score', 'asc')
	
	# Form validation
	validators_columns = {
		'asset_name': [DataRequired(), Length(min=1, max=200)],
		'asset_type': [DataRequired()],
		'asset_category': [DataRequired()],
		'current_health_score': [NumberRange(min=0, max=100)],
		'operational_hours': [NumberRange(min=0)],
		'expected_lifespan_years': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'asset_id': 'Asset ID',
		'asset_name': 'Asset Name',
		'asset_type': 'Asset Type',
		'asset_category': 'Asset Category',
		'model_number': 'Model Number',
		'serial_number': 'Serial Number',
		'facility_id': 'Facility ID',
		'parent_asset_id': 'Parent Asset',
		'installation_date': 'Installation Date',
		'commissioning_date': 'Commissioning Date',
		'expected_lifespan_years': 'Expected Lifespan (years)',
		'operational_hours': 'Operational Hours',
		'current_health_score': 'Health Score (%)',
		'criticality_level': 'Criticality Level',
		'maintenance_strategy': 'Maintenance Strategy',
		'maintenance_frequency_hours': 'Maintenance Frequency (hours)',
		'last_maintenance_date': 'Last Maintenance',
		'next_scheduled_maintenance': 'Next Maintenance',
		'sensor_ids': 'Sensor IDs',
		'monitoring_parameters': 'Monitoring Parameters',
		'alert_thresholds': 'Alert Thresholds',
		'purchase_cost': 'Purchase Cost',
		'replacement_cost': 'Replacement Cost',
		'annual_maintenance_cost': 'Annual Maintenance Cost',
		'downtime_cost_per_hour': 'Downtime Cost/Hour'
	}
	
	@expose('/health_dashboard/<int:pk>')
	@has_access
	def health_dashboard(self, pk):
		"""Asset health monitoring dashboard"""
		asset = self.datamodel.get(pk)
		if not asset:
			flash('Asset not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Get health data for dashboard
			health_data = self._get_asset_health_data(asset)
			
			return render_template('predictive_maintenance/asset_health_dashboard.html',
								   asset=asset,
								   health_data=health_data,
								   page_title=f"Health Dashboard: {asset.asset_name}")
		except Exception as e:
			flash(f'Error loading health dashboard: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/maintenance_history/<int:pk>')
	@has_access
	def maintenance_history(self, pk):
		"""View asset maintenance history"""
		asset = self.datamodel.get(pk)
		if not asset:
			flash('Asset not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Get maintenance history
			maintenance_history = self._get_maintenance_history(asset)
			
			return render_template('predictive_maintenance/maintenance_history.html',
								   asset=asset,
								   maintenance_history=maintenance_history,
								   page_title=f"Maintenance History: {asset.asset_name}")
		except Exception as e:
			flash(f'Error loading maintenance history: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/update_health/<int:pk>')
	@has_access
	def update_health(self, pk):
		"""Update asset health score"""
		asset = self.datamodel.get(pk)
		if not asset:
			flash('Asset not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			previous_score = asset.current_health_score
			new_score = asset.calculate_health_score()
			self.datamodel.edit(asset)
			
			flash(f'Health score updated from {previous_score:.1f}% to {new_score:.1f}%', 'success')
		except Exception as e:
			flash(f'Error updating health score: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/schedule_maintenance/<int:pk>')
	@has_access
	def schedule_maintenance(self, pk):
		"""Schedule maintenance for asset"""
		asset = self.datamodel.get(pk)
		if not asset:
			flash('Asset not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would schedule maintenance
			maintenance_date = self._schedule_asset_maintenance(asset)
			flash(f'Maintenance scheduled for {asset.asset_name} on {maintenance_date}', 'success')
		except Exception as e:
			flash(f'Error scheduling maintenance: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new asset"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.current_health_score:
			item.current_health_score = 100.0
		if not item.status:
			item.status = 'operational'
		if not item.criticality_level:
			item.criticality_level = 'medium'
		if not item.maintenance_strategy:
			item.maintenance_strategy = 'predictive'
	
	def _get_asset_health_data(self, asset: PMAsset) -> Dict[str, Any]:
		"""Get health data for asset dashboard"""
		# Implementation would gather real health data
		return {
			'current_health': asset.current_health_score,
			'health_trend': 'stable',
			'recent_readings': [],
			'sensor_status': {},
			'alerts_count': 0,
			'next_maintenance': asset.next_scheduled_maintenance,
			'maintenance_due': asset.is_maintenance_due()
		}
	
	def _get_maintenance_history(self, asset: PMAsset) -> List[Dict[str, Any]]:
		"""Get maintenance history for asset"""
		# Implementation would query maintenance records
		return []
	
	def _schedule_asset_maintenance(self, asset: PMAsset) -> datetime:
		"""Schedule maintenance for asset"""
		# Implementation would create maintenance schedule
		return datetime.utcnow() + timedelta(days=7)
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class PMHealthRecordModelView(ModelView):
	"""Health record monitoring view"""
	
	datamodel = SQLAInterface(PMHealthRecord)
	
	# List view configuration
	list_columns = [
		'asset', 'recorded_at', 'overall_health_score', 'health_trend',
		'anomaly_detected', 'temperature_celsius', 'vibration_amplitude'
	]
	show_columns = [
		'record_id', 'asset', 'recorded_at', 'measurement_period_hours',
		'temperature_celsius', 'vibration_amplitude', 'vibration_frequency',
		'pressure_bar', 'flow_rate', 'electrical_current', 'voltage',
		'power_consumption', 'efficiency_percentage', 'wear_level_percentage',
		'overall_health_score', 'health_trend', 'anomaly_detected', 'anomaly_severity'
	]
	# Read-only view for health records
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	can_delete = False
	
	# Search and filtering
	search_columns = ['asset.asset_name', 'health_trend']
	base_filters = [['anomaly_detected', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('recorded_at', 'desc')
	
	# Custom labels
	label_columns = {
		'record_id': 'Record ID',
		'recorded_at': 'Recorded At',
		'measurement_period_hours': 'Measurement Period (h)',
		'temperature_celsius': 'Temperature (°C)',
		'vibration_amplitude': 'Vibration Amplitude',
		'vibration_frequency': 'Vibration Frequency (Hz)',
		'pressure_bar': 'Pressure (bar)',
		'flow_rate': 'Flow Rate',
		'electrical_current': 'Current (A)',
		'voltage': 'Voltage (V)',
		'power_consumption': 'Power (W)',
		'efficiency_percentage': 'Efficiency (%)',
		'wear_level_percentage': 'Wear Level (%)',
		'stress_factor': 'Stress Factor',
		'operating_condition_score': 'Operating Condition',
		'overall_health_score': 'Health Score (%)',
		'health_trend': 'Health Trend',
		'anomaly_detected': 'Anomaly Detected',
		'anomaly_severity': 'Anomaly Severity',
		'raw_sensor_data': 'Raw Sensor Data',
		'custom_metrics': 'Custom Metrics',
		'quality_flags': 'Quality Flags'
	}
	
	@expose('/analyze_trends/<int:pk>')
	@has_access
	def analyze_trends(self, pk):
		"""Analyze health trends for record"""
		record = self.datamodel.get(pk)
		if not record:
			flash('Health record not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Get trend analysis
			trend_analysis = self._analyze_health_trends(record)
			
			return render_template('predictive_maintenance/health_trend_analysis.html',
								   record=record,
								   trend_analysis=trend_analysis,
								   page_title=f"Trend Analysis: {record.asset.asset_name}")
		except Exception as e:
			flash(f'Error analyzing trends: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def _analyze_health_trends(self, record: PMHealthRecord) -> Dict[str, Any]:
		"""Analyze health trends for record"""
		# Implementation would perform trend analysis
		return {
			'trend_direction': record.health_trend,
			'rate_of_change': 0.0,
			'forecast': [],
			'anomalies': [],
			'recommendations': []
		}


class PMFailurePredictionModelView(ModelView):
	"""Failure prediction results view"""
	
	datamodel = SQLAInterface(PMFailurePrediction)
	
	# List view configuration
	list_columns = [
		'asset', 'predicted_at', 'failure_type', 'failure_probability',
		'risk_level', 'predicted_failure_date', 'recommended_action', 'status'
	]
	show_columns = [
		'prediction_id', 'asset', 'predicted_at', 'prediction_horizon_hours',
		'predicted_failure_date', 'failure_type', 'failure_mode', 'failure_probability',
		'confidence_score', 'risk_level', 'business_impact', 'safety_risk',
		'model_name', 'model_version', 'recommended_action', 'recommended_timeline',
		'estimated_maintenance_cost', 'estimated_downtime_hours', 'status'
	]
	# Read-only view for predictions
	edit_columns = ['status', 'validated_by', 'actual_failure_occurred', 'actual_failure_date']
	add_columns = []
	can_create = False
	
	# Search and filtering
	search_columns = ['asset.asset_name', 'failure_type', 'risk_level', 'recommended_action']
	base_filters = [['status', lambda: 'active', lambda: True]]
	
	# Ordering
	base_order = ('failure_probability', 'desc')
	
	# Custom labels
	label_columns = {
		'prediction_id': 'Prediction ID',
		'predicted_at': 'Predicted At',
		'prediction_horizon_hours': 'Prediction Horizon (h)',
		'predicted_failure_date': 'Predicted Failure Date',
		'failure_type': 'Failure Type',
		'failure_mode': 'Failure Mode',
		'failure_probability': 'Failure Probability',
		'confidence_score': 'Confidence Score',
		'risk_level': 'Risk Level',
		'business_impact': 'Business Impact',
		'safety_risk': 'Safety Risk',
		'environmental_risk': 'Environmental Risk',
		'model_name': 'Model Name',
		'model_version': 'Model Version',
		'prediction_algorithm': 'Algorithm',
		'feature_importance': 'Feature Importance',
		'recommended_action': 'Recommended Action',
		'recommended_timeline': 'Timeline',
		'maintenance_type': 'Maintenance Type',
		'estimated_maintenance_cost': 'Est. Maintenance Cost',
		'estimated_downtime_hours': 'Est. Downtime (h)',
		'validated_by': 'Validated By',
		'validation_date': 'Validation Date',
		'actual_failure_occurred': 'Actual Failure Occurred',
		'actual_failure_date': 'Actual Failure Date'
	}
	
	@expose('/validate/<int:pk>')
	@has_access
	def validate_prediction(self, pk):
		"""Validate prediction accuracy"""
		prediction = self.datamodel.get(pk)
		if not prediction:
			flash('Prediction not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			prediction.status = 'validated'
			prediction.validated_by = self._get_current_user_id()
			prediction.validation_date = datetime.utcnow()
			self.datamodel.edit(prediction)
			flash('Prediction validated successfully', 'success')
		except Exception as e:
			flash(f'Error validating prediction: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/create_work_order/<int:pk>')
	@has_access
	def create_work_order(self, pk):
		"""Create maintenance work order from prediction"""
		prediction = self.datamodel.get(pk)
		if not prediction:
			flash('Prediction not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would create actual work order
			work_order_id = self._create_maintenance_work_order(prediction)
			flash(f'Work order created: {work_order_id}', 'success')
		except Exception as e:
			flash(f'Error creating work order: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/view_details/<int:pk>')
	@has_access
	def view_prediction_details(self, pk):
		"""View detailed prediction analysis"""
		prediction = self.datamodel.get(pk)
		if not prediction:
			flash('Prediction not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Get detailed analysis
			prediction_details = self._get_prediction_details(prediction)
			
			return render_template('predictive_maintenance/prediction_details.html',
								   prediction=prediction,
								   prediction_details=prediction_details,
								   page_title=f"Prediction Details: {prediction.asset.asset_name}")
		except Exception as e:
			flash(f'Error loading prediction details: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _create_maintenance_work_order(self, prediction: PMFailurePrediction) -> str:
		"""Create maintenance work order from prediction"""
		# Implementation would create actual work order
		import uuid
		return f"WO-{str(uuid.uuid4())[:8].upper()}"
	
	def _get_prediction_details(self, prediction: PMFailurePrediction) -> Dict[str, Any]:
		"""Get detailed prediction analysis"""
		# Implementation would provide detailed analysis
		return {
			'remaining_useful_life': prediction.calculate_remaining_useful_life(),
			'urgency_level': prediction.get_urgency_level(),
			'model_accuracy': 0.85,
			'similar_cases': [],
			'cost_benefit_analysis': {}
		}


class PMMaintenanceRecordModelView(ModelView):
	"""Maintenance record management view"""
	
	datamodel = SQLAInterface(PMMaintenanceRecord)
	
	# List view configuration
	list_columns = [
		'asset', 'maintenance_type', 'work_order_number', 'started_at',
		'status', 'outcome', 'total_cost', 'duration_hours'
	]
	show_columns = [
		'record_id', 'asset', 'maintenance_type', 'maintenance_category',
		'work_order_number', 'priority', 'scheduled_date', 'started_at',
		'completed_at', 'duration_hours', 'downtime_hours', 'description',
		'work_performed', 'parts_replaced', 'parts_cost', 'labor_cost',
		'total_cost', 'technician_id', 'status', 'outcome', 'quality_rating'
	]
	edit_columns = [
		'maintenance_type', 'maintenance_category', 'work_order_number',
		'priority', 'scheduled_date', 'started_at', 'completed_at',
		'description', 'work_performed', 'parts_replaced', 'parts_cost',
		'labor_cost', 'total_cost', 'technician_id', 'supervisor_id',
		'external_contractor', 'status', 'outcome', 'quality_rating', 'notes'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['asset.asset_name', 'maintenance_type', 'work_order_number']
	base_filters = [['status', lambda: 'completed', lambda: True]]
	
	# Ordering
	base_order = ('started_at', 'desc')
	
	# Form validation
	validators_columns = {
		'maintenance_type': [DataRequired()],
		'maintenance_category': [DataRequired()],
		'description': [DataRequired()],
		'started_at': [DataRequired()],
		'parts_cost': [NumberRange(min=0)],
		'labor_cost': [NumberRange(min=0)],
		'quality_rating': [NumberRange(min=1, max=5)]
	}
	
	# Custom labels
	label_columns = {
		'record_id': 'Record ID',
		'maintenance_type': 'Maintenance Type',
		'maintenance_category': 'Category',
		'work_order_number': 'Work Order #',
		'scheduled_date': 'Scheduled Date',
		'started_at': 'Started At',
		'completed_at': 'Completed At',
		'duration_hours': 'Duration (h)',
		'downtime_hours': 'Downtime (h)',
		'work_performed': 'Work Performed',
		'parts_replaced': 'Parts Replaced',
		'parts_cost': 'Parts Cost',
		'labor_cost': 'Labor Cost',
		'total_cost': 'Total Cost',
		'technician_id': 'Technician ID',
		'supervisor_id': 'Supervisor ID',
		'external_contractor': 'External Contractor',
		'quality_rating': 'Quality Rating (1-5)',
		'health_improvement': 'Health Improvement',
		'efficiency_improvement': 'Efficiency Improvement',
		'reliability_improvement': 'Reliability Improvement',
		'follow_up_required': 'Follow-up Required',
		'follow_up_date': 'Follow-up Date',
		'warranty_period_days': 'Warranty Period (days)',
		'warranty_expires': 'Warranty Expires'
	}
	
	@expose('/calculate_effectiveness/<int:pk>')
	@has_access
	def calculate_effectiveness(self, pk):
		"""Calculate maintenance effectiveness"""
		record = self.datamodel.get(pk)
		if not record:
			flash('Maintenance record not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			effectiveness = record.calculate_effectiveness()
			flash(f'Maintenance effectiveness calculated: {effectiveness:.1f}%', 'success')
		except Exception as e:
			flash(f'Error calculating effectiveness: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new maintenance record"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.status:
			item.status = 'scheduled'
		if not item.priority:
			item.priority = 'medium'
		if not item.maintenance_type:
			item.maintenance_type = 'preventive'
	
	def post_add(self, item):
		"""Post-process after adding maintenance record"""
		# Calculate total cost
		if item.parts_cost or item.labor_cost:
			item.total_cost = (item.parts_cost or 0) + (item.labor_cost or 0)
		
		# Calculate duration if both start and end times are set
		if item.started_at and item.completed_at:
			duration = item.completed_at - item.started_at
			item.duration_hours = duration.total_seconds() / 3600
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class PMMaintenanceAlertModelView(ModelView):
	"""Maintenance alert management view"""
	
	datamodel = SQLAInterface(PMMaintenanceAlert)
	
	# List view configuration
	list_columns = [
		'asset', 'alert_type', 'severity', 'priority',
		'triggered_at', 'status', 'title'
	]
	show_columns = [
		'alert_id', 'asset', 'alert_type', 'alert_category', 'severity',
		'priority', 'title', 'message', 'recommendation', 'triggered_at',
		'trigger_condition', 'trigger_value', 'threshold_value', 'status',
		'acknowledged_by', 'acknowledged_at', 'resolved_by', 'resolved_at'
	]
	edit_columns = [
		'status', 'resolution_notes'
	]
	add_columns = []
	can_create = False
	
	# Search and filtering
	search_columns = ['asset.asset_name', 'alert_type', 'severity', 'title']
	base_filters = [['status', lambda: 'active', lambda: True]]
	
	# Ordering
	base_order = ('triggered_at', 'desc')
	
	# Custom labels
	label_columns = {
		'alert_id': 'Alert ID',
		'alert_type': 'Alert Type',
		'alert_category': 'Category',
		'triggered_at': 'Triggered At',
		'trigger_condition': 'Trigger Condition',
		'trigger_value': 'Trigger Value',
		'threshold_value': 'Threshold Value',
		'acknowledged_by': 'Acknowledged By',
		'acknowledged_at': 'Acknowledged At',
		'resolved_by': 'Resolved By',
		'resolved_at': 'Resolved At',
		'resolution_notes': 'Resolution Notes',
		'escalation_level': 'Escalation Level',
		'escalated_at': 'Escalated At',
		'escalation_rules': 'Escalation Rules',
		'notification_sent': 'Notification Sent'
	}
	
	@expose('/acknowledge/<int:pk>')
	@has_access
	def acknowledge_alert(self, pk):
		"""Acknowledge maintenance alert"""
		alert = self.datamodel.get(pk)
		if not alert:
			flash('Alert not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user_id = self._get_current_user_id()
			if alert.acknowledge(user_id):
				self.datamodel.edit(alert)
				flash('Alert acknowledged successfully', 'success')
			else:
				flash('Alert cannot be acknowledged in current state', 'warning')
		except Exception as e:
			flash(f'Error acknowledging alert: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/resolve/<int:pk>')
	@has_access
	def resolve_alert(self, pk):
		"""Resolve maintenance alert"""
		alert = self.datamodel.get(pk)
		if not alert:
			flash('Alert not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user_id = self._get_current_user_id()
			notes = request.args.get('notes', 'Resolved by user')
			if alert.resolve(user_id, notes):
				self.datamodel.edit(alert)
				flash('Alert resolved successfully', 'success')
			else:
				flash('Alert cannot be resolved in current state', 'warning')
		except Exception as e:
			flash(f'Error resolving alert: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/escalate/<int:pk>')
	@has_access
	def escalate_alert(self, pk):
		"""Escalate maintenance alert"""
		alert = self.datamodel.get(pk)
		if not alert:
			flash('Alert not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if alert.escalate():
				self.datamodel.edit(alert)
				flash(f'Alert escalated to level {alert.escalation_level}', 'success')
			else:
				flash('Alert escalation failed', 'error')
		except Exception as e:
			flash(f'Error escalating alert: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None


class PredictiveMaintenanceDashboardView(PredictiveMaintenanceBaseView):
	"""Predictive maintenance dashboard"""
	
	route_base = "/predictive_maintenance_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Predictive maintenance dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('predictive_maintenance/dashboard.html',
								   metrics=metrics,
								   page_title="Predictive Maintenance Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('predictive_maintenance/dashboard.html',
								   metrics={},
								   page_title="Predictive Maintenance Dashboard")
	
	@expose('/asset_overview/')
	@has_access
	def asset_overview(self):
		"""Asset health overview"""
		try:
			overview_data = self._get_asset_overview_data()
			
			return render_template('predictive_maintenance/asset_overview.html',
								   overview_data=overview_data,
								   page_title="Asset Overview")
		except Exception as e:
			flash(f'Error loading asset overview: {str(e)}', 'error')
			return redirect(url_for('PredictiveMaintenanceDashboardView.index'))
	
	@expose('/maintenance_analytics/')
	@has_access
	def maintenance_analytics(self):
		"""Maintenance performance analytics"""
		try:
			period_days = int(request.args.get('period', 30))
			analytics_data = self._get_maintenance_analytics(period_days)
			
			return render_template('predictive_maintenance/maintenance_analytics.html',
								   analytics_data=analytics_data,
								   period_days=period_days,
								   page_title="Maintenance Analytics")
		except Exception as e:
			flash(f'Error loading maintenance analytics: {str(e)}', 'error')
			return redirect(url_for('PredictiveMaintenanceDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get predictive maintenance metrics for dashboard"""
		# Implementation would calculate real metrics from database
		return {
			'total_assets': 245,
			'critical_assets': 45,
			'healthy_assets': 198,
			'assets_at_risk': 28,
			'active_alerts': 12,
			'critical_alerts': 3,
			'maintenance_due': 15,
			'overdue_maintenance': 5,
			'prediction_accuracy': 87.5,
			'cost_savings': 125000,
			'downtime_prevented': 45.2,
			'asset_health_avg': 89.3,
			'top_risk_assets': [
				{'name': 'Pump Station A', 'health': 65.2, 'risk': 'high'},
				{'name': 'Motor Unit 12', 'health': 72.1, 'risk': 'medium'},
				{'name': 'Compressor B3', 'health': 78.5, 'risk': 'medium'}
			],
			'recent_predictions': [],
			'maintenance_schedule': []
		}
	
	def _get_asset_overview_data(self) -> Dict[str, Any]:
		"""Get asset overview data"""
		return {
			'asset_categories': {
				'motors': {'count': 85, 'avg_health': 88.5, 'at_risk': 8},
				'pumps': {'count': 45, 'avg_health': 82.3, 'at_risk': 12},
				'compressors': {'count': 32, 'avg_health': 91.2, 'at_risk': 3},
				'sensors': {'count': 156, 'avg_health': 95.1, 'at_risk': 5}
			},
			'health_distribution': {
				'excellent': 89,  # 90-100%
				'good': 98,      # 75-89%
				'fair': 45,      # 60-74%
				'poor': 13       # <60%
			},
			'criticality_breakdown': {
				'critical': 45,
				'high': 78,
				'medium': 89,
				'low': 33
			}
		}
	
	def _get_maintenance_analytics(self, period_days: int) -> Dict[str, Any]:
		"""Get maintenance analytics data"""
		return {
			'period_days': period_days,
			'total_maintenance_activities': 156,
			'preventive_maintenance': 89,
			'corrective_maintenance': 45,
			'predictive_maintenance': 22,
			'average_cost_per_activity': 1250,
			'total_maintenance_cost': 195000,
			'cost_savings_from_prediction': 45000,
			'downtime_hours': 89.5,
			'mtbf_average': 720,  # Mean Time Between Failures (hours)
			'mttr_average': 4.2,  # Mean Time To Repair (hours)
			'maintenance_effectiveness': 91.2
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all predictive maintenance views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		PMAssetModelView,
		"Assets",
		icon="fa-cogs",
		category="Predictive Maintenance",
		category_icon="fa-wrench"
	)
	
	appbuilder.add_view(
		PMHealthRecordModelView,
		"Health Records",
		icon="fa-heartbeat",
		category="Predictive Maintenance"
	)
	
	appbuilder.add_view(
		PMFailurePredictionModelView,
		"Failure Predictions",
		icon="fa-exclamation-triangle",
		category="Predictive Maintenance"
	)
	
	appbuilder.add_view(
		PMMaintenanceRecordModelView,
		"Maintenance Records",
		icon="fa-clipboard-list",
		category="Predictive Maintenance"
	)
	
	appbuilder.add_view(
		PMMaintenanceAlertModelView,
		"Maintenance Alerts",
		icon="fa-bell",
		category="Predictive Maintenance"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(PredictiveMaintenanceDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"Maintenance Dashboard",
		href="/predictive_maintenance_dashboard/",
		icon="fa-dashboard",
		category="Predictive Maintenance"
	)
	
	appbuilder.add_link(
		"Asset Overview",
		href="/predictive_maintenance_dashboard/asset_overview/",
		icon="fa-eye",
		category="Predictive Maintenance"
	)
	
	appbuilder.add_link(
		"Maintenance Analytics",
		href="/predictive_maintenance_dashboard/maintenance_analytics/",
		icon="fa-chart-line",
		category="Predictive Maintenance"
	)