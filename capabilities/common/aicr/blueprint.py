"""
Flask-Appbuilder Blueprint for the AI Core Framework (AICR) Capability
======================================================================

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

Flask-Appbuilder blueprint providing comprehensive web interface and API
endpoints for the AI Core Framework, including model management, pipeline
orchestration, monitoring dashboards, and administrative controls.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash
from flask_appbuilder import AppBuilder, BaseView, expose, has_access
from flask_appbuilder.models.mixins import AuditMixin
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.views import ModelView, SimpleFormView
from flask_appbuilder.baseviews import BaseCRUDView
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import ListWidget, FormWidget
from flask_wtf import FlaskForm
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
from wtforms import StringField, TextAreaField, SelectField, FloatField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Length, NumberRange

from .service import AICoreService
from .models import AICRModel, AICRInferenceRequest, AICRPipeline
from .monitoring import ai_monitoring_system
from .ml_pipeline import ml_pipeline_framework
from .model_marketplace import model_marketplace


# SQLAlchemy models for Flask-Appbuilder
Base = declarative_base()


class AICRModelDB(AuditMixin, Base):
	"""SQLAlchemy model for AICR models in the database."""
	__tablename__ = 'aicr_models'
	__table_args__ = {'extend_existing': True}

	id = Column(Integer, primary_key=True)
	model_id = Column(String(50), unique=True, nullable=False, default=uuid7str)
	name = Column(String(200), nullable=False)
	description = Column(Text)
	model_type = Column(String(100), nullable=False)
	framework = Column(String(50), nullable=False)
	version = Column(String(50), default="1.0.0")
	status = Column(String(50), default="inactive")
	file_path = Column(String(500))
	configuration = Column(Text)  # JSON configuration
	performance_metrics = Column(Text)  # JSON metrics
	is_active = Column(Boolean, default=True)
	deployment_count = Column(Integer, default=0)


class AICRPipelineDB(AuditMixin, Base):
	"""SQLAlchemy model for AICR pipelines in the database."""
	__tablename__ = 'aicr_pipelines'
	__table_args__ = {'extend_existing': True}

	id = Column(Integer, primary_key=True)
	pipeline_id = Column(String(50), unique=True, nullable=False, default=uuid7str)
	name = Column(String(200), nullable=False)
	description = Column(Text)
	pipeline_type = Column(String(100), nullable=False)
	status = Column(String(50), default="pending")
	configuration = Column(Text)  # JSON configuration
	stages_count = Column(Integer, default=0)
	execution_count = Column(Integer, default=0)
	last_execution = Column(DateTime)
	success_rate = Column(Float, default=0.0)
	is_active = Column(Boolean, default=True)


class AICRExecutionDB(AuditMixin, Base):
	"""SQLAlchemy model for AICR pipeline executions in the database."""
	__tablename__ = 'aicr_executions'
	__table_args__ = {'extend_existing': True}

	id = Column(Integer, primary_key=True)
	execution_id = Column(String(50), unique=True, nullable=False, default=uuid7str)
	pipeline_id = Column(String(50), ForeignKey('aicr_pipelines.pipeline_id'), nullable=False)
	status = Column(String(50), default="pending")
	started_at = Column(DateTime)
	completed_at = Column(DateTime)
	duration_seconds = Column(Float)
	stage_results = Column(Text)  # JSON results
	error_message = Column(Text)
	metrics = Column(Text)  # JSON metrics

	# Relationship
	pipeline = relationship("AICRPipelineDB", backref="executions")


class AICRMetricDB(AuditMixin, Base):
	"""SQLAlchemy model for AICR monitoring metrics in the database."""
	__tablename__ = 'aicr_metrics'
	__table_args__ = {'extend_existing': True}

	id = Column(Integer, primary_key=True)
	metric_id = Column(String(50), unique=True, nullable=False, default=uuid7str)
	metric_name = Column(String(200), nullable=False)
	metric_type = Column(String(50), nullable=False)
	value = Column(Float, nullable=False)
	timestamp = Column(DateTime, default=datetime.utcnow)
	labels = Column(Text)  # JSON labels
	source_component = Column(String(100))
	source_instance = Column(String(100))


# Flask-WTF Forms
class ModelForm(FlaskForm):
	"""Form for creating/editing AICR models."""
	name = StringField('Model Name', validators=[DataRequired(), Length(min=1, max=200)])
	description = TextAreaField('Description', validators=[Length(max=1000)])
	model_type = SelectField('Model Type', choices=[
		('classification', 'Classification'),
		('regression', 'Regression'),
		('clustering', 'Clustering'),
		('anomaly_detection', 'Anomaly Detection'),
		('time_series', 'Time Series'),
		('nlp', 'Natural Language Processing'),
		('computer_vision', 'Computer Vision'),
		('recommendation', 'Recommendation'),
		('reinforcement_learning', 'Reinforcement Learning')
	], validators=[DataRequired()])
	framework = SelectField('Framework', choices=[
		('pytorch', 'PyTorch'),
		('tensorflow', 'TensorFlow'),
		('sklearn', 'Scikit-learn'),
		('xgboost', 'XGBoost'),
		('lightgbm', 'LightGBM'),
		('onnx', 'ONNX'),
		('ollama', 'Ollama'),
		('custom', 'Custom')
	], validators=[DataRequired()])
	version = StringField('Version', default="1.0.0", validators=[DataRequired()])
	file_path = StringField('Model File Path', validators=[Length(max=500)])


class PipelineForm(FlaskForm):
	"""Form for creating/editing AICR pipelines."""
	name = StringField('Pipeline Name', validators=[DataRequired(), Length(min=1, max=200)])
	description = TextAreaField('Description', validators=[Length(max=1000)])
	pipeline_type = SelectField('Pipeline Type', choices=[
		('training', 'Model Training'),
		('inference', 'Model Inference'),
		('evaluation', 'Model Evaluation'),
		('data_processing', 'Data Processing'),
		('feature_engineering', 'Feature Engineering'),
		('hyperparameter_tuning', 'Hyperparameter Tuning'),
		('automl', 'AutoML'),
		('deployment', 'Model Deployment')
	], validators=[DataRequired()])


class ExecutionForm(FlaskForm):
	"""Form for triggering pipeline executions."""
	pipeline_id = SelectField('Pipeline', coerce=str, validators=[DataRequired()])
	execution_config = TextAreaField('Execution Configuration (JSON)', default='{}')


# Flask-Appbuilder Views
class AICRModelView(ModelView):
	"""Model view for AICR models management."""
	datamodel = SQLAInterface(AICRModelDB)

	# List configuration
	list_columns = ['name', 'model_type', 'framework', 'version', 'status', 'deployment_count', 'created_on']
	show_columns = ['name', 'description', 'model_type', 'framework', 'version', 'status',
					'file_path', 'configuration', 'performance_metrics', 'deployment_count',
					'created_on', 'changed_on', 'created_by', 'changed_by']
	edit_columns = ['name', 'description', 'model_type', 'framework', 'version', 'file_path']
	add_columns = ['name', 'description', 'model_type', 'framework', 'version', 'file_path']

	# Search and filters
	search_columns = ['name', 'model_type', 'framework', 'version']
	base_filters = [['is_active', lambda: True]]

	# Ordering
	base_order = ('created_on', 'desc')

	# Labels
	label_columns = {
		'name': 'Model Name',
		'model_type': 'Type',
		'framework': 'Framework',
		'created_on': 'Created',
		'changed_on': 'Modified',
		'deployment_count': 'Deployments'
	}

	@expose('/deploy/<model_id>')
	@has_access
	def deploy_model(self, model_id):
		"""Deploy a model to the inference engine."""
		try:
			# Get model from database
			model_db = self.datamodel.get(model_id)
			if not model_db:
				flash('Model not found', 'error')
				return redirect(url_for('.list'))

			# Deploy model using AI service
			ai_service = AICoreService()
			deployment_result = asyncio.run(ai_service.deploy_model(model_db.model_id))

			if deployment_result.get('success'):
				model_db.status = 'deployed'
				model_db.deployment_count += 1
				self.datamodel.edit(model_db)
				flash(f'Model {model_db.name} deployed successfully', 'success')
			else:
				flash(f'Failed to deploy model: {deployment_result.get("error", "Unknown error")}', 'error')

		except Exception as e:
			flash(f'Error deploying model: {str(e)}', 'error')

		return redirect(url_for('.list'))

	@expose('/undeploy/<model_id>')
	@has_access
	def undeploy_model(self, model_id):
		"""Undeploy a model from the inference engine."""
		try:
			# Get model from database
			model_db = self.datamodel.get(model_id)
			if not model_db:
				flash('Model not found', 'error')
				return redirect(url_for('.list'))

			# Undeploy model using AI service
			ai_service = AICoreService()
			result = asyncio.run(ai_service.undeploy_model(model_db.model_id))

			if result.get('success'):
				model_db.status = 'inactive'
				self.datamodel.edit(model_db)
				flash(f'Model {model_db.name} undeployed successfully', 'success')
			else:
				flash(f'Failed to undeploy model: {result.get("error", "Unknown error")}', 'error')

		except Exception as e:
			flash(f'Error undeploying model: {str(e)}', 'error')

		return redirect(url_for('.list'))


class AICRPipelineView(ModelView):
	"""Model view for AICR pipelines management."""
	datamodel = SQLAInterface(AICRPipelineDB)

	# List configuration
	list_columns = ['name', 'pipeline_type', 'status', 'stages_count', 'execution_count',
					'success_rate', 'last_execution', 'created_on']
	show_columns = ['name', 'description', 'pipeline_type', 'status', 'configuration',
					'stages_count', 'execution_count', 'success_rate', 'last_execution',
					'created_on', 'changed_on', 'created_by', 'changed_by']
	edit_columns = ['name', 'description', 'pipeline_type']
	add_columns = ['name', 'description', 'pipeline_type']

	# Search and filters
	search_columns = ['name', 'pipeline_type', 'status']
	base_filters = [['is_active', lambda: True]]

	# Ordering
	base_order = ('created_on', 'desc')

	# Labels
	label_columns = {
		'name': 'Pipeline Name',
		'pipeline_type': 'Type',
		'stages_count': 'Stages',
		'execution_count': 'Executions',
		'success_rate': 'Success Rate',
		'last_execution': 'Last Run',
		'created_on': 'Created'
	}

	@expose('/execute/<pipeline_id>')
	@has_access
	def execute_pipeline(self, pipeline_id):
		"""Execute a pipeline."""
		try:
			# Get pipeline from database
			pipeline_db = self.datamodel.get(pipeline_id)
			if not pipeline_db:
				flash('Pipeline not found', 'error')
				return redirect(url_for('.list'))

			# Execute pipeline using ML framework
			execution_id = asyncio.run(
				ml_pipeline_framework.execute_pipeline(pipeline_db.pipeline_id)
			)

			# Update pipeline execution count
			pipeline_db.execution_count += 1
			pipeline_db.last_execution = datetime.utcnow()
			self.datamodel.edit(pipeline_db)

			flash(f'Pipeline {pipeline_db.name} executed successfully. Execution ID: {execution_id}', 'success')

		except Exception as e:
			flash(f'Error executing pipeline: {str(e)}', 'error')

		return redirect(url_for('.list'))


class AICRExecutionView(ModelView):
	"""Model view for AICR pipeline executions monitoring."""
	datamodel = SQLAInterface(AICRExecutionDB)

	# List configuration
	list_columns = ['pipeline.name', 'status', 'started_at', 'completed_at', 'duration_seconds']
	show_columns = ['execution_id', 'pipeline.name', 'status', 'started_at', 'completed_at',
					'duration_seconds', 'stage_results', 'error_message', 'metrics']

	# No editing allowed for executions
	edit_columns = []
	add_columns = []

	# Search and filters
	search_columns = ['pipeline.name', 'status']

	# Ordering
	base_order = ('started_at', 'desc')

	# Labels
	label_columns = {
		'pipeline.name': 'Pipeline',
		'started_at': 'Started',
		'completed_at': 'Completed',
		'duration_seconds': 'Duration (s)',
		'stage_results': 'Results',
		'error_message': 'Error'
	}


class AICRMetricView(ModelView):
	"""Model view for AICR monitoring metrics."""
	datamodel = SQLAInterface(AICRMetricDB)

	# List configuration
	list_columns = ['metric_name', 'metric_type', 'value', 'timestamp', 'source_component']
	show_columns = ['metric_name', 'metric_type', 'value', 'timestamp', 'labels',
					'source_component', 'source_instance']

	# No editing allowed for metrics
	edit_columns = []
	add_columns = []

	# Search and filters
	search_columns = ['metric_name', 'metric_type', 'source_component']

	# Ordering
	base_order = ('timestamp', 'desc')

	# Labels
	label_columns = {
		'metric_name': 'Metric',
		'metric_type': 'Type',
		'source_component': 'Component',
		'source_instance': 'Instance'
	}


class AICRDashboardView(BaseView):
	"""Main dashboard view for AICR capability."""

	route_base = '/aicr'
	default_view = 'dashboard'

	@expose('/dashboard/')
	@has_access
	def dashboard(self):
		"""Main AICR dashboard."""
		try:
			# Get system health
			health_data = asyncio.run(ai_monitoring_system.get_system_health())

			# Get recent executions
			recent_executions = self.appbuilder.get_session.query(AICRExecutionDB)\
				.order_by(AICRExecutionDB.started_at.desc())\
				.limit(10).all()

			# Get model statistics
			total_models = self.appbuilder.get_session.query(AICRModelDB)\
				.filter(AICRModelDB.is_active == True).count()

			deployed_models = self.appbuilder.get_session.query(AICRModelDB)\
				.filter(AICRModelDB.status == 'deployed').count()

			# Get pipeline statistics
			total_pipelines = self.appbuilder.get_session.query(AICRPipelineDB)\
				.filter(AICRPipelineDB.is_active == True).count()

			running_pipelines = self.appbuilder.get_session.query(AICRPipelineDB)\
				.filter(AICRPipelineDB.status == 'running').count()

			dashboard_data = {
				'health': health_data,
				'stats': {
					'total_models': total_models,
					'deployed_models': deployed_models,
					'total_pipelines': total_pipelines,
					'running_pipelines': running_pipelines
				},
				'recent_executions': recent_executions
			}

			return self.render_template('aicr/dashboard.html', data=dashboard_data)

		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return self.render_template('aicr/dashboard.html', data={})

	@expose('/monitoring/')
	@has_access
	def monitoring(self):
		"""Real-time monitoring dashboard."""
		try:
			# Get performance summary
			performance_data = asyncio.run(ai_monitoring_system.get_performance_summary())

			# Get recent metrics
			recent_metrics = self.appbuilder.get_session.query(AICRMetricDB)\
				.order_by(AICRMetricDB.timestamp.desc())\
				.limit(100).all()

			monitoring_data = {
				'performance': performance_data,
				'recent_metrics': recent_metrics
			}

			return self.render_template('aicr/monitoring.html', data=monitoring_data)

		except Exception as e:
			flash(f'Error loading monitoring data: {str(e)}', 'error')
			return self.render_template('aicr/monitoring.html', data={})

	@expose('/marketplace/')
	@has_access
	def marketplace(self):
		"""Model marketplace dashboard."""
		try:
			# Get marketplace statistics
			marketplace_stats = asyncio.run(model_marketplace.get_marketplace_statistics())

			# Get featured models
			featured_models = asyncio.run(model_marketplace.get_featured_models(limit=10))

			marketplace_data = {
				'stats': marketplace_stats,
				'featured_models': featured_models
			}

			return self.render_template('aicr/marketplace.html', data=marketplace_data)

		except Exception as e:
			flash(f'Error loading marketplace data: {str(e)}', 'error')
			return self.render_template('aicr/marketplace.html', data={})

	@expose('/api/health')
	def api_health(self):
		"""API endpoint for system health."""
		try:
			health_data = asyncio.run(ai_monitoring_system.get_system_health())
			return jsonify(health_data)
		except Exception as e:
			return jsonify({'error': str(e)}), 500

	@expose('/api/metrics')
	def api_metrics(self):
		"""API endpoint for recent metrics."""
		try:
			# Get query parameters
			metric_names = request.args.getlist('metric_names')
			time_range_hours = int(request.args.get('time_range_hours', 1))

			# Calculate time range
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(hours=time_range_hours)

			# Get metrics from monitoring system
			metrics = asyncio.run(
				ai_monitoring_system.metrics_collector.get_metrics(
					metric_names=metric_names if metric_names else None,
					time_range=(start_time, end_time)
				)
			)

			# Convert to JSON-serializable format
			metrics_data = [
				{
					'metric_name': m.metric_name,
					'value': m.value,
					'timestamp': m.timestamp.isoformat(),
					'labels': m.labels,
					'source_component': m.source_component
				}
				for m in metrics
			]

			return jsonify({
				'metrics': metrics_data,
				'time_range': {
					'start': start_time.isoformat(),
					'end': end_time.isoformat()
				},
				'count': len(metrics_data)
			})

		except Exception as e:
			return jsonify({'error': str(e)}), 500

	@expose('/api/pipelines/<pipeline_id>/execute', methods=['POST'])
	@protect()
	def api_execute_pipeline(self, pipeline_id):
		"""API endpoint for executing pipelines."""
		try:
			# Get execution configuration from request
			execution_config = request.get_json() or {}

			# Execute pipeline
			execution_id = asyncio.run(
				ml_pipeline_framework.execute_pipeline(
					pipeline_id,
					input_data=execution_config.get('input_data'),
					execution_config=execution_config.get('execution_config')
				)
			)

			return jsonify({
				'success': True,
				'execution_id': execution_id,
				'message': f'Pipeline {pipeline_id} executed successfully'
			})

		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500

	@expose('/api/models/<model_id>/deploy', methods=['POST'])
	@protect()
	def api_deploy_model(self, model_id):
		"""API endpoint for deploying models."""
		try:
			# Deploy model using AI service
			ai_service = AICoreService()
			deployment_result = asyncio.run(ai_service.deploy_model(model_id))

			return jsonify(deployment_result)

		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500


# Blueprint registration function
def create_aicr_blueprint(appbuilder: AppBuilder) -> Blueprint:
	"""Create and configure the AICR Flask-Appbuilder blueprint.

	Args:
		appbuilder: Flask-Appbuilder instance

	Returns:
		Blueprint: Configured AICR blueprint
	"""
	# Create blueprint
	aicr_bp = Blueprint(
		'aicr',
		__name__,
		url_prefix='/aicr',
		template_folder='templates',
		static_folder='static'
	)

	# Add views to appbuilder
	appbuilder.add_view(
		AICRDashboardView,
		"AICR Dashboard",
		icon="fa-dashboard",
		category="AI Core Framework"
	)

	appbuilder.add_view(
		AICRModelView,
		"Models",
		icon="fa-cogs",
		category="AI Core Framework"
	)

	appbuilder.add_view(
		AICRPipelineView,
		"Pipelines",
		icon="fa-sitemap",
		category="AI Core Framework"
	)

	appbuilder.add_view(
		AICRExecutionView,
		"Executions",
		icon="fa-play",
		category="AI Core Framework"
	)

	appbuilder.add_view(
		AICRMetricView,
		"Metrics",
		icon="fa-line-chart",
		category="AI Core Framework"
	)

	# Add separator and links
	appbuilder.add_separator("AI Core Framework")

	appbuilder.add_link(
		"AI Monitoring",
		href="/aicr/monitoring/",
		icon="fa-heartbeat",
		category="AI Core Framework"
	)

	appbuilder.add_link(
		"Model Marketplace",
		href="/aicr/marketplace/",
		icon="fa-shopping-cart",
		category="AI Core Framework"
	)

	# Initialize AICR services if not already done
	@aicr_bp.before_app_first_request
	def initialize_aicr():
		"""Initialize AICR services on first request."""
		try:
			# Initialize monitoring system
			asyncio.run(ai_monitoring_system.initialize())

			# Initialize ML pipeline framework
			asyncio.run(ml_pipeline_framework.initialize())

			# Initialize model marketplace
			asyncio.run(model_marketplace.initialize())

			logging.info("AICR services initialized successfully")

		except Exception as e:
			logging.error(f"Failed to initialize AICR services: {e}")

	return aicr_bp


# Export blueprint creation function
__all__ = [
	'create_aicr_blueprint',
	'AICRModelView',
	'AICRPipelineView',
	'AICRExecutionView',
	'AICRMetricView',
	'AICRDashboardView',
	'AICRModelDB',
	'AICRPipelineDB',
	'AICRExecutionDB',
	'AICRMetricDB'
]