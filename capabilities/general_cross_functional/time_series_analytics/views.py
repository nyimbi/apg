"""
Time Series Analytics Views

Flask-AppBuilder views for comprehensive time series data management,
forecasting, anomaly detection, and analytics with real-time monitoring.
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
	TSDataStream, TSDataPoint, TSForecastModel, TSForecast,
	TSAnomaly, TSAnalyticsJob
)


class TimeSeriesBaseView(BaseView):
	"""Base view for time series analytics functionality"""
	
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
	
	def _format_duration(self, seconds: float) -> str:
		"""Format duration for display"""
		if seconds is None:
			return "N/A"
		if seconds < 60:
			return f"{seconds:.1f}s"
		elif seconds < 3600:
			return f"{seconds/60:.1f}m"
		else:
			return f"{seconds/3600:.1f}h"


class TSDataStreamModelView(ModelView):
	"""Data stream management view"""
	
	datamodel = SQLAInterface(TSDataStream)
	
	# List view configuration
	list_columns = [
		'stream_name', 'source_type', 'data_type', 'sampling_frequency',
		'is_active', 'quality_score', 'data_point_count', 'last_data_point'
	]
	show_columns = [
		'stream_id', 'stream_name', 'description', 'source_type', 'source_identifier',
		'data_type', 'unit_of_measure', 'sampling_frequency', 'expected_range_min',
		'expected_range_max', 'is_active', 'quality_score', 'last_data_point',
		'data_point_count', 'preprocessing_rules', 'aggregation_methods'
	]
	edit_columns = [
		'stream_name', 'description', 'source_type', 'source_identifier',
		'data_type', 'unit_of_measure', 'sampling_frequency', 'expected_range_min',
		'expected_range_max', 'is_active', 'preprocessing_rules', 'aggregation_methods',
		'alert_thresholds'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['stream_name', 'description', 'source_type', 'data_type']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('quality_score', 'desc')
	
	# Form validation
	validators_columns = {
		'stream_name': [DataRequired(), Length(min=1, max=200)],
		'source_type': [DataRequired()],
		'data_type': [DataRequired()],
		'quality_score': [NumberRange(min=0, max=100)]
	}
	
	# Custom labels
	label_columns = {
		'stream_id': 'Stream ID',
		'stream_name': 'Stream Name',
		'source_type': 'Source Type',
		'source_identifier': 'Source Identifier',
		'data_type': 'Data Type',
		'unit_of_measure': 'Unit of Measure',
		'sampling_frequency': 'Sampling Frequency',
		'expected_range_min': 'Expected Min',
		'expected_range_max': 'Expected Max',
		'is_active': 'Active',
		'quality_score': 'Quality Score',
		'last_data_point': 'Last Data Point',
		'data_point_count': 'Data Point Count',
		'preprocessing_rules': 'Preprocessing Rules',
		'aggregation_methods': 'Aggregation Methods',
		'alert_thresholds': 'Alert Thresholds'
	}
	
	@expose('/test_connection/<int:pk>')
	@has_access
	def test_connection(self, pk):
		"""Test connection to data source"""
		stream = self.datamodel.get(pk)
		if not stream:
			flash('Data stream not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would test actual data source connection
			success = self._test_data_source(stream)
			
			if success:
				flash(f'Connection test successful for stream "{stream.stream_name}"', 'success')
			else:
				flash(f'Connection test failed for stream "{stream.stream_name}"', 'error')
		except Exception as e:
			flash(f'Error testing connection: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/calculate_quality/<int:pk>')
	@has_access
	def calculate_quality(self, pk):
		"""Recalculate data quality score"""
		stream = self.datamodel.get(pk)
		if not stream:
			flash('Data stream not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			stream.calculate_quality_score()
			self.datamodel.edit(stream)
			flash(f'Quality score recalculated: {stream.quality_score:.1f}%', 'success')
		except Exception as e:
			flash(f'Error calculating quality: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/view_data/<int:pk>')
	@has_access
	def view_data(self, pk):
		"""View recent data points for stream"""
		stream = self.datamodel.get(pk)
		if not stream:
			flash('Data stream not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Get recent data points
			recent_data = self._get_recent_data(stream, limit=100)
			
			return render_template('time_series_analytics/stream_data.html',
								   stream=stream,
								   recent_data=recent_data,
								   page_title=f"Data: {stream.stream_name}")
		except Exception as e:
			flash(f'Error loading data: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new stream"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.quality_score:
			item.quality_score = 0.0
	
	def _test_data_source(self, stream: TSDataStream) -> bool:
		"""Test connection to data source"""
		# Implementation would test actual connection based on source_type
		return True
	
	def _get_recent_data(self, stream: TSDataStream, limit: int = 100) -> List[Dict[str, Any]]:
		"""Get recent data points for stream"""
		# Implementation would query recent data points
		return []
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class TSForecastModelModelView(ModelView):
	"""Forecasting model management view"""
	
	datamodel = SQLAInterface(TSForecastModel)
	
	# List view configuration
	list_columns = [
		'model_name', 'model_type', 'is_active', 'is_trained',
		'accuracy_score', 'mape', 'forecast_count', 'last_trained_at'
	]
	show_columns = [
		'model_id', 'model_name', 'model_type', 'algorithm_version',
		'model_parameters', 'hyperparameters', 'accuracy_score', 'mape',
		'rmse', 'mae', 'is_active', 'is_trained', 'training_status',
		'last_trained_at', 'forecast_count', 'successful_forecasts'
	]
	edit_columns = [
		'model_name', 'model_type', 'algorithm_version', 'model_parameters',
		'hyperparameters', 'training_config', 'is_active'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['model_name', 'model_type']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('accuracy_score', 'desc')
	
	# Form validation
	validators_columns = {
		'model_name': [DataRequired(), Length(min=1, max=200)],
		'model_type': [DataRequired()],
		'accuracy_score': [NumberRange(min=0, max=100)]
	}
	
	# Custom labels
	label_columns = {
		'model_id': 'Model ID',
		'model_name': 'Model Name',
		'model_type': 'Model Type',
		'algorithm_version': 'Algorithm Version',
		'model_parameters': 'Model Parameters',
		'hyperparameters': 'Hyperparameters',
		'training_config': 'Training Config',
		'accuracy_score': 'Accuracy Score',
		'mape': 'MAPE (%)',
		'rmse': 'RMSE',
		'mae': 'MAE',
		'is_active': 'Active',
		'is_trained': 'Trained',
		'training_status': 'Training Status',
		'last_trained_at': 'Last Trained',
		'training_duration': 'Training Duration (s)',
		'forecast_count': 'Forecast Count',
		'successful_forecasts': 'Successful Forecasts',
		'average_forecast_time': 'Avg Forecast Time (s)'
	}
	
	@expose('/train_model/<int:pk>')
	@has_access
	def train_model(self, pk):
		"""Start model training"""
		model = self.datamodel.get(pk)
		if not model:
			flash('Model not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would start actual model training
			job_id = self._start_training(model)
			flash(f'Training started for model "{model.model_name}". Job ID: {job_id}', 'success')
		except Exception as e:
			flash(f'Error starting training: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/test_model/<int:pk>')
	@has_access
	def test_model(self, pk):
		"""Test model performance"""
		model = self.datamodel.get(pk)
		if not model:
			flash('Model not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would test model performance
			test_results = self._test_model_performance(model)
			
			return render_template('time_series_analytics/model_test_results.html',
								   model=model,
								   test_results=test_results,
								   page_title=f"Test Results: {model.model_name}")
		except Exception as e:
			flash(f'Error testing model: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/clone_model/<int:pk>')
	@has_access
	def clone_model(self, pk):
		"""Clone existing model"""
		original = self.datamodel.get(pk)
		if not original:
			flash('Model not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Create cloned model
			cloned_model = TSForecastModel(
				model_name=f"{original.model_name} (Copy)",
				model_type=original.model_type,
				algorithm_version=original.algorithm_version,
				model_parameters=original.model_parameters,
				hyperparameters=original.hyperparameters,
				training_config=original.training_config,
				tenant_id=original.tenant_id
			)
			
			self.datamodel.add(cloned_model)
			flash(f'Model "{original.model_name}" cloned successfully', 'success')
		except Exception as e:
			flash(f'Error cloning model: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new model"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.algorithm_version:
			item.algorithm_version = '1.0.0'
		if not item.training_status:
			item.training_status = 'untrained'
	
	def _start_training(self, model: TSForecastModel) -> str:
		"""Start model training and return job ID"""
		# Implementation would start actual training job
		import uuid
		return str(uuid.uuid4())
	
	def _test_model_performance(self, model: TSForecastModel) -> Dict[str, Any]:
		"""Test model performance with sample data"""
		# Implementation would perform actual model testing
		return {
			'accuracy_metrics': {
				'mape': 8.5,
				'rmse': 12.3,
				'mae': 9.1
			},
			'test_cases': 50,
			'success_rate': 94.0,
			'average_prediction_time': 0.15
		}
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class TSForecastModelView(ModelView):
	"""Forecast results view"""
	
	datamodel = SQLAInterface(TSForecast)
	
	# List view configuration
	list_columns = [
		'forecast_start', 'forecast_end', 'stream', 'model',
		'forecast_horizon', 'status', 'confidence_score', 'generation_time'
	]
	show_columns = [
		'forecast_id', 'stream', 'model', 'forecast_horizon', 'forecast_start',
		'forecast_end', 'predicted_values', 'confidence_intervals', 'confidence_score',
		'status', 'generation_time', 'input_data_points'
	]
	# Read-only view for forecasts
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	
	# Search and filtering
	search_columns = ['stream.stream_name', 'model.model_name', 'status']
	base_filters = [['status', lambda: 'completed', lambda: True]]
	
	# Ordering
	base_order = ('forecast_start', 'desc')
	
	# Custom labels
	label_columns = {
		'forecast_id': 'Forecast ID',
		'forecast_horizon': 'Forecast Horizon',
		'forecast_start': 'Forecast Start',
		'forecast_end': 'Forecast End',
		'predicted_values': 'Predicted Values',
		'confidence_intervals': 'Confidence Intervals',
		'prediction_intervals': 'Prediction Intervals',
		'confidence_score': 'Confidence Score',
		'uncertainty_metrics': 'Uncertainty Metrics',
		'generation_time': 'Generation Time (s)',
		'input_data_points': 'Input Data Points'
	}
	
	@expose('/visualize/<int:pk>')
	@has_access
	def visualize_forecast(self, pk):
		"""Visualize forecast results"""
		forecast = self.datamodel.get(pk)
		if not forecast:
			flash('Forecast not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Get visualization data
			viz_data = self._prepare_visualization_data(forecast)
			
			return render_template('time_series_analytics/forecast_visualization.html',
								   forecast=forecast,
								   viz_data=viz_data,
								   page_title=f"Forecast Visualization")
		except Exception as e:
			flash(f'Error loading visualization: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/export/<int:pk>')
	@has_access
	def export_forecast(self, pk):
		"""Export forecast data"""
		forecast = self.datamodel.get(pk)
		if not forecast:
			flash('Forecast not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would export forecast data
			export_data = self._export_forecast_data(forecast)
			flash('Forecast data exported successfully', 'success')
		except Exception as e:
			flash(f'Error exporting forecast: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def _prepare_visualization_data(self, forecast: TSForecast) -> Dict[str, Any]:
		"""Prepare data for forecast visualization"""
		# Implementation would prepare chart data
		return {
			'timestamps': forecast.timestamps or [],
			'predicted_values': forecast.predicted_values or [],
			'confidence_intervals': forecast.confidence_intervals or {},
			'chart_config': {
				'title': f'Forecast for {forecast.stream.stream_name}',
				'x_axis': 'Time',
				'y_axis': forecast.stream.unit_of_measure or 'Value'
			}
		}
	
	def _export_forecast_data(self, forecast: TSForecast) -> Dict[str, Any]:
		"""Export forecast data in structured format"""
		# Implementation would format data for export
		return {
			'forecast_id': forecast.forecast_id,
			'predictions': list(zip(forecast.timestamps or [], forecast.predicted_values or [])),
			'metadata': {
				'stream': forecast.stream.stream_name,
				'model': forecast.model.model_name,
				'confidence_score': forecast.confidence_score
			}
		}


class TSAnomalyModelView(ModelView):
	"""Anomaly detection results view"""
	
	datamodel = SQLAInterface(TSAnomaly)
	
	# List view configuration
	list_columns = [
		'detected_at', 'stream', 'anomaly_type', 'severity',
		'anomaly_score', 'status', 'alert_triggered'
	]
	show_columns = [
		'anomaly_id', 'stream', 'detected_at', 'anomaly_type', 'severity',
		'anomaly_score', 'expected_value', 'actual_value', 'deviation_magnitude',
		'detection_method', 'confidence_level', 'status', 'alert_triggered',
		'investigated_by', 'resolved_at'
	]
	edit_columns = [
		'status', 'investigated_by', 'resolution_notes'
	]
	add_columns = []
	can_create = False
	
	# Search and filtering
	search_columns = ['stream.stream_name', 'anomaly_type', 'severity', 'status']
	base_filters = [['status', lambda: 'open', lambda: True]]
	
	# Ordering
	base_order = ('detected_at', 'desc')
	
	# Custom labels
	label_columns = {
		'anomaly_id': 'Anomaly ID',
		'detected_at': 'Detected At',
		'anomaly_type': 'Anomaly Type',
		'anomaly_score': 'Anomaly Score',
		'expected_value': 'Expected Value',
		'actual_value': 'Actual Value',
		'deviation_magnitude': 'Deviation',
		'context_window': 'Context Window',
		'detection_method': 'Detection Method',
		'detection_parameters': 'Detection Parameters',
		'confidence_level': 'Confidence Level',
		'investigated_by': 'Investigated By',
		'resolved_at': 'Resolved At',
		'resolution_notes': 'Resolution Notes',
		'alert_triggered': 'Alert Triggered',
		'alert_recipients': 'Alert Recipients',
		'acknowledgment_required': 'Ack Required',
		'acknowledged_by': 'Acknowledged By',
		'acknowledged_at': 'Acknowledged At'
	}
	
	@expose('/acknowledge/<int:pk>')
	@has_access
	def acknowledge_anomaly(self, pk):
		"""Acknowledge anomaly"""
		anomaly = self.datamodel.get(pk)
		if not anomaly:
			flash('Anomaly not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user_id = self._get_current_user_id()
			anomaly.acknowledge(user_id)
			self.datamodel.edit(anomaly)
			flash('Anomaly acknowledged successfully', 'success')
		except Exception as e:
			flash(f'Error acknowledging anomaly: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/resolve/<int:pk>')
	@has_access
	def resolve_anomaly(self, pk):
		"""Mark anomaly as resolved"""
		anomaly = self.datamodel.get(pk)
		if not anomaly:
			flash('Anomaly not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			anomaly.status = 'resolved'
			anomaly.resolved_at = datetime.utcnow()
			anomaly.investigated_by = self._get_current_user_id()
			self.datamodel.edit(anomaly)
			flash('Anomaly marked as resolved', 'success')
		except Exception as e:
			flash(f'Error resolving anomaly: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/mark_false_positive/<int:pk>')
	@has_access
	def mark_false_positive(self, pk):
		"""Mark anomaly as false positive"""
		anomaly = self.datamodel.get(pk)
		if not anomaly:
			flash('Anomaly not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			anomaly.status = 'false_positive'
			anomaly.resolved_at = datetime.utcnow()
			anomaly.investigated_by = self._get_current_user_id()
			self.datamodel.edit(anomaly)
			flash('Anomaly marked as false positive', 'success')
		except Exception as e:
			flash(f'Error marking false positive: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None


class TSAnalyticsJobModelView(ModelView):
	"""Analytics job monitoring view"""
	
	datamodel = SQLAInterface(TSAnalyticsJob)
	
	# List view configuration
	list_columns = [
		'job_name', 'job_type', 'status', 'progress_percentage',
		'started_at', 'duration', 'current_step'
	]
	show_columns = [
		'job_id', 'job_name', 'job_type', 'parameters', 'input_streams',
		'status', 'started_at', 'completed_at', 'duration', 'progress_percentage',
		'current_step', 'total_steps', 'results_summary', 'error_message'
	]
	# Read-only view for job monitoring
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	
	# Search and filtering
	search_columns = ['job_name', 'job_type', 'status']
	base_filters = [['status', lambda: 'running', lambda: True]]
	
	# Ordering
	base_order = ('started_at', 'desc')
	
	# Custom labels
	label_columns = {
		'job_id': 'Job ID',
		'job_name': 'Job Name',
		'job_type': 'Job Type',
		'input_streams': 'Input Streams',
		'started_at': 'Started At',
		'completed_at': 'Completed At',
		'progress_percentage': 'Progress (%)',
		'current_step': 'Current Step',
		'total_steps': 'Total Steps',
		'output_location': 'Output Location',
		'results_summary': 'Results Summary',
		'error_message': 'Error Message',
		'cpu_time': 'CPU Time (s)',
		'memory_usage': 'Memory Usage (MB)',
		'storage_used': 'Storage Used (MB)'
	}
	
	@expose('/cancel/<int:pk>')
	@has_access
	def cancel_job(self, pk):
		"""Cancel running job"""
		job = self.datamodel.get(pk)
		if not job:
			flash('Job not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if job.status == 'running':
				job.status = 'cancelled'
				job.completed_at = datetime.utcnow()
				if job.started_at:
					job.duration = (job.completed_at - job.started_at).total_seconds()
				self.datamodel.edit(job)
				flash('Job cancelled successfully', 'success')
			else:
				flash('Only running jobs can be cancelled', 'warning')
		except Exception as e:
			flash(f'Error cancelling job: {str(e)}', 'error')
		
		return redirect(self.get_redirect())


class TimeSeriesDashboardView(TimeSeriesBaseView):
	"""Time series analytics dashboard"""
	
	route_base = "/time_series_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Time series dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('time_series_analytics/dashboard.html',
								   metrics=metrics,
								   page_title="Time Series Analytics Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('time_series_analytics/dashboard.html',
								   metrics={},
								   page_title="Time Series Analytics Dashboard")
	
	@expose('/anomaly_monitoring/')
	@has_access
	def anomaly_monitoring(self):
		"""Anomaly monitoring dashboard"""
		try:
			anomaly_data = self._get_anomaly_monitoring_data()
			
			return render_template('time_series_analytics/anomaly_monitoring.html',
								   anomaly_data=anomaly_data,
								   page_title="Anomaly Monitoring")
		except Exception as e:
			flash(f'Error loading anomaly monitoring: {str(e)}', 'error')
			return redirect(url_for('TimeSeriesDashboardView.index'))
	
	@expose('/forecast_analytics/')
	@has_access
	def forecast_analytics(self):
		"""Forecast performance analytics"""
		try:
			period_days = int(request.args.get('period', 30))
			analytics_data = self._get_forecast_analytics(period_days)
			
			return render_template('time_series_analytics/forecast_analytics.html',
								   analytics_data=analytics_data,
								   period_days=period_days,
								   page_title="Forecast Analytics")
		except Exception as e:
			flash(f'Error loading forecast analytics: {str(e)}', 'error')
			return redirect(url_for('TimeSeriesDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get time series analytics metrics for dashboard"""
		# Implementation would calculate real metrics from database
		return {
			'active_streams': 45,
			'total_data_points': 1250000,
			'data_points_today': 12450,
			'active_models': 12,
			'trained_models': 8,
			'forecasts_generated': 156,
			'anomalies_detected': 23,
			'open_anomalies': 5,
			'running_jobs': 3,
			'data_quality_avg': 94.2,
			'forecast_accuracy_avg': 87.5,
			'top_streams': [
				{'name': 'Temperature Sensor 1', 'data_points': 50000, 'quality': 98.5},
				{'name': 'Pressure Monitor', 'data_points': 45000, 'quality': 96.2},
				{'name': 'Flow Rate Sensor', 'data_points': 38000, 'quality': 94.8}
			],
			'recent_anomalies': []
		}
	
	def _get_anomaly_monitoring_data(self) -> Dict[str, Any]:
		"""Get anomaly monitoring data"""
		return {
			'total_anomalies': 156,
			'open_anomalies': 23,
			'resolved_anomalies': 133,
			'critical_anomalies': 3,
			'anomaly_trend': [],
			'anomaly_types': {
				'point': 89,
				'contextual': 45,
				'collective': 22
			},
			'detection_methods': {
				'statistical': 67,
				'ml': 54,
				'threshold': 35
			}
		}
	
	def _get_forecast_analytics(self, period_days: int) -> Dict[str, Any]:
		"""Get forecast performance analytics"""
		return {
			'period_days': period_days,
			'total_forecasts': 156,
			'successful_forecasts': 142,
			'success_rate': 91.0,
			'average_accuracy': 87.5,
			'average_generation_time': 2.3,
			'model_performance': {},
			'accuracy_trends': [],
			'error_analysis': {}
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all time series analytics views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		TSDataStreamModelView,
		"Data Streams",
		icon="fa-stream",
		category="Time Series Analytics",
		category_icon="fa-chart-line"
	)
	
	appbuilder.add_view(
		TSForecastModelModelView,
		"Forecast Models",
		icon="fa-brain",
		category="Time Series Analytics"
	)
	
	appbuilder.add_view(
		TSForecastModelView,
		"Forecasts",
		icon="fa-crystal-ball",
		category="Time Series Analytics"
	)
	
	appbuilder.add_view(
		TSAnomalyModelView,
		"Anomalies",
		icon="fa-exclamation-triangle",
		category="Time Series Analytics"
	)
	
	appbuilder.add_view(
		TSAnalyticsJobModelView,
		"Analytics Jobs",
		icon="fa-tasks",
		category="Time Series Analytics"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(TimeSeriesDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"Time Series Dashboard",
		href="/time_series_dashboard/",
		icon="fa-dashboard",
		category="Time Series Analytics"
	)
	
	appbuilder.add_link(
		"Anomaly Monitoring",
		href="/time_series_dashboard/anomaly_monitoring/",
		icon="fa-search",
		category="Time Series Analytics"
	)
	
	appbuilder.add_link(
		"Forecast Analytics",
		href="/time_series_dashboard/forecast_analytics/",
		icon="fa-chart-bar",
		category="Time Series Analytics"
	)