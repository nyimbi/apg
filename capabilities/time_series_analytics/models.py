"""
Time Series Analytics Models

Database models for time series data management, forecasting models,
analytics workflows, and performance tracking.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


class TSDataStream(Model, AuditMixin, BaseMixin):
	"""
	Time series data stream definition and metadata.
	
	Manages configuration for different time series data sources,
	including sampling rates, data types, and quality metrics.
	"""
	__tablename__ = 'ts_data_stream'
	
	# Identity
	stream_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Stream Configuration
	stream_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	source_type = Column(String(50), nullable=False, index=True)  # sensor, api, database, file
	source_identifier = Column(String(500), nullable=False)
	
	# Data Characteristics
	data_type = Column(String(50), nullable=False)  # numeric, categorical, boolean
	unit_of_measure = Column(String(50), nullable=True)
	sampling_frequency = Column(String(50), nullable=True)  # 1s, 5m, 1h, 1d
	expected_range_min = Column(Float, nullable=True)
	expected_range_max = Column(Float, nullable=True)
	
	# Quality and Status
	is_active = Column(Boolean, default=True, index=True)
	quality_score = Column(Float, default=0.0)  # 0-100 data quality score
	last_data_point = Column(DateTime, nullable=True)
	data_point_count = Column(Integer, default=0)
	
	# Processing Configuration
	preprocessing_rules = Column(JSON, default=dict)
	aggregation_methods = Column(JSON, default=list)
	alert_thresholds = Column(JSON, default=dict)
	
	# Relationships
	data_points = relationship("TSDataPoint", back_populates="stream")
	forecasts = relationship("TSForecast", back_populates="stream")
	anomalies = relationship("TSAnomaly", back_populates="stream")
	
	def __repr__(self):
		return f"<TSDataStream {self.stream_name}>"
	
	def get_latest_value(self) -> Optional[float]:
		"""Get the most recent data point value"""
		if self.data_points:
			latest = max(self.data_points, key=lambda x: x.timestamp)
			return latest.value
		return None
	
	def calculate_quality_score(self):
		"""Calculate and update data quality score"""
		# Implementation would analyze data completeness, accuracy, etc.
		if self.data_point_count > 0:
			self.quality_score = min(100.0, (self.data_point_count / 1000) * 100)
		else:
			self.quality_score = 0.0


class TSDataPoint(Model, BaseMixin):
	"""
	Individual time series data points.
	
	Stores actual time series values with timestamps and quality indicators.
	Optimized for high-volume data ingestion and querying.
	"""
	__tablename__ = 'ts_data_point'
	
	# Identity
	point_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	stream_id = Column(String(36), ForeignKey('ts_data_stream.stream_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Time and Value
	timestamp = Column(DateTime, nullable=False, index=True)
	value = Column(Float, nullable=False)
	raw_value = Column(String(500), nullable=True)  # Original value before processing
	
	# Quality and Metadata
	quality_flag = Column(String(20), default='good', index=True)  # good, suspect, bad, uncertain
	confidence_score = Column(Float, default=1.0)  # 0-1 confidence in value
	source_metadata = Column(JSON, default=dict)
	processing_flags = Column(JSON, default=list)
	
	# Relationships
	stream = relationship("TSDataStream", back_populates="data_points")
	
	def __repr__(self):
		return f"<TSDataPoint {self.timestamp}: {self.value}>"


class TSForecastModel(Model, AuditMixin, BaseMixin):
	"""
	Time series forecasting model definitions and configurations.
	
	Manages different forecasting algorithms, their parameters,
	and performance metrics for model selection and optimization.
	"""
	__tablename__ = 'ts_forecast_model'
	
	# Identity
	model_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Model Definition
	model_name = Column(String(200), nullable=False)
	model_type = Column(String(50), nullable=False, index=True)  # arima, exponential_smoothing, lstm, prophet
	algorithm_version = Column(String(50), default='1.0.0')
	
	# Configuration
	model_parameters = Column(JSON, default=dict)
	hyperparameters = Column(JSON, default=dict)
	training_config = Column(JSON, default=dict)
	
	# Performance Metrics
	accuracy_score = Column(Float, default=0.0)  # 0-100 accuracy percentage
	mape = Column(Float, nullable=True)  # Mean Absolute Percentage Error
	rmse = Column(Float, nullable=True)  # Root Mean Square Error
	mae = Column(Float, nullable=True)   # Mean Absolute Error
	
	# Model Status
	is_active = Column(Boolean, default=True, index=True)
	is_trained = Column(Boolean, default=False)
	training_status = Column(String(20), default='untrained', index=True)
	last_trained_at = Column(DateTime, nullable=True)
	training_duration = Column(Float, nullable=True)  # seconds
	
	# Usage Statistics
	forecast_count = Column(Integer, default=0)
	successful_forecasts = Column(Integer, default=0)
	average_forecast_time = Column(Float, default=0.0)  # seconds
	
	# Relationships
	forecasts = relationship("TSForecast", back_populates="model")
	
	def __repr__(self):
		return f"<TSForecastModel {self.model_name} ({self.model_type})>"
	
	def calculate_accuracy(self, actual_values: List[float], predicted_values: List[float]):
		"""Calculate and update model accuracy metrics"""
		if len(actual_values) != len(predicted_values) or len(actual_values) == 0:
			return
		
		# Calculate MAPE
		mape_sum = sum(abs((actual - predicted) / actual) for actual, predicted in 
					  zip(actual_values, predicted_values) if actual != 0)
		self.mape = (mape_sum / len(actual_values)) * 100 if len(actual_values) > 0 else 0
		
		# Calculate RMSE
		mse = sum((actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values))
		self.rmse = (mse / len(actual_values)) ** 0.5
		
		# Calculate MAE
		self.mae = sum(abs(actual - predicted) for actual, predicted in zip(actual_values, predicted_values)) / len(actual_values)
		
		# Calculate overall accuracy score (inverse of normalized MAPE)
		self.accuracy_score = max(0, 100 - self.mape)


class TSForecast(Model, AuditMixin, BaseMixin):
	"""
	Time series forecast results and predictions.
	
	Stores forecast outputs including point predictions, confidence intervals,
	and uncertainty quantification for different time horizons.
	"""
	__tablename__ = 'ts_forecast'
	
	# Identity
	forecast_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	stream_id = Column(String(36), ForeignKey('ts_data_stream.stream_id'), nullable=False, index=True)
	model_id = Column(String(36), ForeignKey('ts_forecast_model.model_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Forecast Configuration
	forecast_horizon = Column(Integer, nullable=False)  # Number of periods ahead
	forecast_start = Column(DateTime, nullable=False, index=True)
	forecast_end = Column(DateTime, nullable=False, index=True)
	
	# Predictions
	predicted_values = Column(JSON, default=list)  # List of predicted values
	timestamps = Column(JSON, default=list)  # Corresponding timestamps
	confidence_intervals = Column(JSON, default=dict)  # Upper/lower bounds
	prediction_intervals = Column(JSON, default=dict)  # Wider uncertainty bounds
	
	# Forecast Quality
	confidence_score = Column(Float, default=0.0)  # Overall forecast confidence
	uncertainty_metrics = Column(JSON, default=dict)
	
	# Status and Metadata
	status = Column(String(20), default='pending', index=True)  # pending, completed, failed
	generation_time = Column(Float, nullable=True)  # Time to generate forecast
	input_data_points = Column(Integer, default=0)
	
	# Relationships
	stream = relationship("TSDataStream", back_populates="forecasts")
	model = relationship("TSForecastModel", back_populates="forecasts")
	
	def __repr__(self):
		return f"<TSForecast {self.forecast_start} to {self.forecast_end}>"
	
	def get_prediction_at_time(self, target_time: datetime) -> Optional[Dict[str, Any]]:
		"""Get prediction for specific timestamp"""
		if not self.timestamps or not self.predicted_values:
			return None
		
		for i, ts_str in enumerate(self.timestamps):
			ts = datetime.fromisoformat(ts_str)
			if ts == target_time and i < len(self.predicted_values):
				return {
					'timestamp': target_time,
					'predicted_value': self.predicted_values[i],
					'confidence_score': self.confidence_score
				}
		return None


class TSAnomaly(Model, AuditMixin, BaseMixin):
	"""
	Time series anomaly detection results.
	
	Records detected anomalies with severity scores, context information,
	and resolution tracking for monitoring and alerting systems.
	"""
	__tablename__ = 'ts_anomaly'
	
	# Identity
	anomaly_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	stream_id = Column(String(36), ForeignKey('ts_data_stream.stream_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Anomaly Details
	detected_at = Column(DateTime, nullable=False, index=True)
	anomaly_type = Column(String(50), nullable=False, index=True)  # point, contextual, collective
	severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
	anomaly_score = Column(Float, nullable=False)  # 0-1 anomaly strength
	
	# Context Information
	expected_value = Column(Float, nullable=True)
	actual_value = Column(Float, nullable=False)
	deviation_magnitude = Column(Float, nullable=False)
	context_window = Column(JSON, default=dict)  # Surrounding data context
	
	# Detection Method
	detection_method = Column(String(50), nullable=False)  # statistical, ml, threshold
	detection_parameters = Column(JSON, default=dict)
	confidence_level = Column(Float, default=0.0)
	
	# Status and Resolution
	status = Column(String(20), default='open', index=True)  # open, investigating, resolved, false_positive
	investigated_by = Column(String(36), nullable=True)
	resolved_at = Column(DateTime, nullable=True)
	resolution_notes = Column(Text, nullable=True)
	
	# Alert Information
	alert_triggered = Column(Boolean, default=False)
	alert_recipients = Column(JSON, default=list)
	acknowledgment_required = Column(Boolean, default=False)
	acknowledged_by = Column(String(36), nullable=True)
	acknowledged_at = Column(DateTime, nullable=True)
	
	# Relationships
	stream = relationship("TSDataStream", back_populates="anomalies")
	
	def __repr__(self):
		return f"<TSAnomaly {self.anomaly_type} at {self.detected_at}>"
	
	def calculate_risk_level(self) -> str:
		"""Calculate risk level based on severity and score"""
		if self.severity == 'critical' or self.anomaly_score > 0.9:
			return 'high'
		elif self.severity == 'high' or self.anomaly_score > 0.7:
			return 'medium'
		else:
			return 'low'
	
	def acknowledge(self, user_id: str):
		"""Acknowledge the anomaly"""
		self.acknowledged_by = user_id
		self.acknowledged_at = datetime.utcnow()


class TSAnalyticsJob(Model, AuditMixin, BaseMixin):
	"""
	Time series analytics job tracking and management.
	
	Manages long-running analytics tasks like model training,
	batch forecasting, and historical analysis workflows.
	"""
	__tablename__ = 'ts_analytics_job'
	
	# Identity
	job_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Job Configuration
	job_name = Column(String(200), nullable=False)
	job_type = Column(String(50), nullable=False, index=True)  # forecast, training, analysis, anomaly_detection
	parameters = Column(JSON, default=dict)
	input_streams = Column(JSON, default=list)  # List of stream IDs
	
	# Execution Status
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, cancelled
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	duration = Column(Float, nullable=True)  # seconds
	
	# Progress Tracking
	progress_percentage = Column(Float, default=0.0)
	current_step = Column(String(200), nullable=True)
	total_steps = Column(Integer, nullable=True)
	
	# Results and Output
	output_location = Column(String(500), nullable=True)
	results_summary = Column(JSON, default=dict)
	error_message = Column(Text, nullable=True)
	logs = Column(Text, nullable=True)
	
	# Resource Usage
	cpu_time = Column(Float, nullable=True)  # seconds
	memory_usage = Column(Float, nullable=True)  # MB
	storage_used = Column(Float, nullable=True)  # MB
	
	def __repr__(self):
		return f"<TSAnalyticsJob {self.job_name} ({self.job_type})>"
	
	def update_progress(self, percentage: float, current_step: str = None):
		"""Update job progress"""
		self.progress_percentage = min(100.0, max(0.0, percentage))
		if current_step:
			self.current_step = current_step
	
	def mark_completed(self, results: Dict[str, Any] = None):
		"""Mark job as completed"""
		self.status = 'completed'
		self.completed_at = datetime.utcnow()
		self.progress_percentage = 100.0
		if self.started_at:
			self.duration = (self.completed_at - self.started_at).total_seconds()
		if results:
			self.results_summary = results
	
	def mark_failed(self, error_message: str):
		"""Mark job as failed"""
		self.status = 'failed'
		self.completed_at = datetime.utcnow()
		self.error_message = error_message
		if self.started_at:
			self.duration = (self.completed_at - self.started_at).total_seconds()