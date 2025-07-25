"""
Predictive Maintenance Models

Database models for predictive maintenance management, asset health monitoring,
failure prediction, and maintenance optimization with AI-driven analytics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


class PMAsset(Model, AuditMixin, BaseMixin):
	"""
	Physical or digital asset under predictive maintenance management.
	
	Represents equipment, machinery, or systems that require monitoring
	and maintenance with complete lifecycle and health tracking.
	"""
	__tablename__ = 'pm_asset'
	
	# Identity
	asset_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset Information
	asset_name = Column(String(200), nullable=False)
	asset_type = Column(String(50), nullable=False, index=True)  # motor, pump, sensor, hvac, etc.
	asset_category = Column(String(50), nullable=False, index=True)  # critical, important, standard
	manufacturer = Column(String(100), nullable=True)
	model_number = Column(String(100), nullable=True)
	serial_number = Column(String(100), nullable=True, index=True)
	
	# Location and Context
	location = Column(String(200), nullable=True)
	facility_id = Column(String(36), nullable=True, index=True)
	parent_asset_id = Column(String(36), ForeignKey('pm_asset.asset_id'), nullable=True)
	
	# Operational Details
	installation_date = Column(DateTime, nullable=True)
	commissioning_date = Column(DateTime, nullable=True)
	expected_lifespan_years = Column(Float, nullable=True)
	operational_hours = Column(Float, default=0.0)  # Total operational hours
	
	# Health and Status
	current_health_score = Column(Float, default=100.0, index=True)  # 0-100 health percentage
	status = Column(String(20), default='operational', index=True)  # operational, maintenance, failed, decommissioned
	criticality_level = Column(String(20), default='medium', index=True)  # low, medium, high, critical
	
	# Maintenance Configuration
	maintenance_strategy = Column(String(50), default='predictive', index=True)
	maintenance_frequency_hours = Column(Float, nullable=True)
	last_maintenance_date = Column(DateTime, nullable=True)
	next_scheduled_maintenance = Column(DateTime, nullable=True, index=True)
	
	# Monitoring Configuration
	sensor_ids = Column(JSON, default=list)  # List of associated sensor IDs
	monitoring_parameters = Column(JSON, default=dict)  # Parameters to monitor
	alert_thresholds = Column(JSON, default=dict)  # Alert threshold configuration
	
	# Financial Information
	purchase_cost = Column(Float, nullable=True)
	replacement_cost = Column(Float, nullable=True)
	annual_maintenance_cost = Column(Float, nullable=True)
	downtime_cost_per_hour = Column(Float, nullable=True)
	
	# Relationships
	parent_asset = relationship("PMAsset", remote_side=[asset_id])
	child_assets = relationship("PMAsset")
	health_records = relationship("PMHealthRecord", back_populates="asset")
	maintenance_records = relationship("PMMaintenanceRecord", back_populates="asset")
	predictions = relationship("PMFailurePrediction", back_populates="asset")
	
	def __repr__(self):
		return f"<PMAsset {self.asset_name} ({self.asset_type})>"
	
	def calculate_health_score(self) -> float:
		"""Calculate current health score based on latest data"""
		# Implementation would analyze sensor data, maintenance history, etc.
		if self.health_records:
			latest_record = max(self.health_records, key=lambda x: x.recorded_at)
			self.current_health_score = latest_record.overall_health_score
		return self.current_health_score
	
	def get_maintenance_due_date(self) -> Optional[datetime]:
		"""Calculate when maintenance is due"""
		if self.maintenance_frequency_hours and self.operational_hours:
			hours_since_maintenance = self.operational_hours
			if self.last_maintenance_date:
				# Calculate based on operational hours since last maintenance
				return self.last_maintenance_date + timedelta(hours=self.maintenance_frequency_hours)
		return self.next_scheduled_maintenance
	
	def is_maintenance_due(self) -> bool:
		"""Check if maintenance is due"""
		due_date = self.get_maintenance_due_date()
		if due_date:
			return datetime.utcnow() >= due_date
		return False


class PMHealthRecord(Model, AuditMixin, BaseMixin):
	"""
	Asset health monitoring records with sensor data and computed metrics.
	
	Stores periodic health assessments including sensor readings,
	derived metrics, and automated health scoring.
	"""
	__tablename__ = 'pm_health_record'
	
	# Identity
	record_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	asset_id = Column(String(36), ForeignKey('pm_asset.asset_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Timing
	recorded_at = Column(DateTime, nullable=False, index=True)
	measurement_period_hours = Column(Float, default=1.0)  # Period over which data was collected
	
	# Sensor Data
	temperature_celsius = Column(Float, nullable=True)
	vibration_amplitude = Column(Float, nullable=True)
	vibration_frequency = Column(Float, nullable=True)
	pressure_bar = Column(Float, nullable=True)
	flow_rate = Column(Float, nullable=True)
	electrical_current = Column(Float, nullable=True)
	voltage = Column(Float, nullable=True)
	power_consumption = Column(Float, nullable=True)
	
	# Computed Metrics
	efficiency_percentage = Column(Float, nullable=True)
	wear_level_percentage = Column(Float, nullable=True)
	stress_factor = Column(Float, nullable=True)
	operating_condition_score = Column(Float, nullable=True)  # 0-100
	
	# Health Assessment
	overall_health_score = Column(Float, nullable=False, index=True)  # 0-100
	health_trend = Column(String(20), nullable=True, index=True)  # improving, stable, degrading
	anomaly_detected = Column(Boolean, default=False, index=True)
	anomaly_severity = Column(String(20), nullable=True)  # low, medium, high, critical
	
	# Raw and Custom Data
	raw_sensor_data = Column(JSON, default=dict)  # Complete sensor payload
	custom_metrics = Column(JSON, default=dict)  # Additional computed metrics
	quality_flags = Column(JSON, default=dict)  # Data quality indicators
	
	# Relationships
	asset = relationship("PMAsset", back_populates="health_records")
	
	def __repr__(self):
		return f"<PMHealthRecord {self.asset.asset_name} at {self.recorded_at}>"
	
	def calculate_health_score(self) -> float:
		"""Calculate overall health score from individual metrics"""
		scores = []
		weights = []
		
		# Temperature score (normalized)
		if self.temperature_celsius is not None:
			temp_score = max(0, 100 - abs(self.temperature_celsius - 70) * 2)  # Assuming 70°C optimal
			scores.append(temp_score)
			weights.append(0.2)
		
		# Vibration score
		if self.vibration_amplitude is not None:
			vib_score = max(0, 100 - self.vibration_amplitude * 10)  # Lower vibration is better
			scores.append(vib_score)
			weights.append(0.3)
		
		# Efficiency score
		if self.efficiency_percentage is not None:
			scores.append(self.efficiency_percentage)
			weights.append(0.3)
		
		# Operating condition score
		if self.operating_condition_score is not None:
			scores.append(self.operating_condition_score)
			weights.append(0.2)
		
		if scores:
			weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
			self.overall_health_score = min(100.0, max(0.0, weighted_score))
		else:
			self.overall_health_score = 50.0  # Default if no data
		
		return self.overall_health_score


class PMFailurePrediction(Model, AuditMixin, BaseMixin):
	"""
	AI-generated asset failure predictions with confidence intervals.
	
	Stores machine learning model predictions for potential failures
	including probability scores, time-to-failure estimates, and recommended actions.
	"""
	__tablename__ = 'pm_failure_prediction'
	
	# Identity
	prediction_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	asset_id = Column(String(36), ForeignKey('pm_asset.asset_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Prediction Details
	predicted_at = Column(DateTime, nullable=False, index=True)
	prediction_horizon_hours = Column(Float, nullable=False)  # How far into future
	predicted_failure_date = Column(DateTime, nullable=True, index=True)
	
	# Failure Information
	failure_type = Column(String(50), nullable=False, index=True)  # bearing, motor, seal, etc.
	failure_mode = Column(String(50), nullable=False)  # wear, fatigue, corrosion, overheating
	failure_probability = Column(Float, nullable=False, index=True)  # 0-1 probability
	confidence_score = Column(Float, nullable=False)  # 0-1 model confidence
	
	# Risk Assessment
	risk_level = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
	business_impact = Column(String(20), nullable=False)  # minimal, moderate, significant, severe
	safety_risk = Column(Boolean, default=False, index=True)
	environmental_risk = Column(Boolean, default=False)
	
	# Model Information
	model_name = Column(String(100), nullable=False)
	model_version = Column(String(20), nullable=False)
	prediction_algorithm = Column(String(50), nullable=False)  # random_forest, lstm, svm, etc.
	feature_importance = Column(JSON, default=dict)  # Which features drove prediction
	
	# Recommendations
	recommended_action = Column(String(20), nullable=False, index=True)  # monitor, inspect, maintain, replace
	recommended_timeline = Column(String(50), nullable=True)  # immediate, within_week, within_month
	maintenance_type = Column(String(50), nullable=True)  # preventive, corrective, replacement
	estimated_maintenance_cost = Column(Float, nullable=True)
	estimated_downtime_hours = Column(Float, nullable=True)
	
	# Status and Validation
	status = Column(String(20), default='active', index=True)  # active, validated, invalidated, expired
	validated_by = Column(String(36), nullable=True)
	validation_date = Column(DateTime, nullable=True)
	actual_failure_occurred = Column(Boolean, nullable=True)
	actual_failure_date = Column(DateTime, nullable=True)
	
	# Relationships
	asset = relationship("PMAsset", back_populates="predictions")
	
	def __repr__(self):
		return f"<PMFailurePrediction {self.failure_type} for {self.asset.asset_name}>"
	
	def calculate_remaining_useful_life(self) -> Optional[float]:
		"""Calculate remaining useful life in hours"""
		if self.predicted_failure_date:
			remaining = self.predicted_failure_date - datetime.utcnow()
			return max(0, remaining.total_seconds() / 3600)
		return None
	
	def get_urgency_level(self) -> str:
		"""Determine urgency level based on time to failure and risk"""
		rul_hours = self.calculate_remaining_useful_life()
		if rul_hours is None:
			return 'unknown'
		
		if rul_hours < 24 and self.risk_level in ['high', 'critical']:
			return 'immediate'
		elif rul_hours < 168 and self.risk_level in ['medium', 'high', 'critical']:  # 1 week
			return 'urgent'
		elif rul_hours < 720:  # 1 month
			return 'plan'
		else:
			return 'monitor'


class PMMaintenanceRecord(Model, AuditMixin, BaseMixin):
	"""
	Maintenance activity records with outcomes and performance tracking.
	
	Comprehensive tracking of all maintenance activities including
	planned, unplanned, and predictive maintenance with cost and effectiveness analysis.
	"""
	__tablename__ = 'pm_maintenance_record'
	
	# Identity
	record_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	asset_id = Column(String(36), ForeignKey('pm_asset.asset_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Maintenance Details
	maintenance_type = Column(String(50), nullable=False, index=True)  # preventive, corrective, predictive
	maintenance_category = Column(String(50), nullable=False)  # routine, repair, overhaul, replacement
	work_order_number = Column(String(50), nullable=True, index=True)
	priority = Column(String(20), default='medium', index=True)  # low, medium, high, emergency
	
	# Scheduling and Timing
	scheduled_date = Column(DateTime, nullable=True)
	started_at = Column(DateTime, nullable=False, index=True)
	completed_at = Column(DateTime, nullable=True, index=True)
	duration_hours = Column(Float, nullable=True)
	downtime_hours = Column(Float, nullable=True)
	
	# Work Details
	description = Column(Text, nullable=False)
	work_performed = Column(Text, nullable=True)
	parts_replaced = Column(JSON, default=list)  # List of parts/components
	parts_cost = Column(Float, nullable=True)
	labor_cost = Column(Float, nullable=True)
	total_cost = Column(Float, nullable=True)
	
	# Personnel
	technician_id = Column(String(36), nullable=True)
	supervisor_id = Column(String(36), nullable=True)
	external_contractor = Column(String(200), nullable=True)
	
	# Outcomes and Quality
	status = Column(String(20), default='completed', index=True)  # scheduled, in_progress, completed, cancelled
	outcome = Column(String(20), nullable=True, index=True)  # successful, partial, failed
	quality_rating = Column(Integer, nullable=True)  # 1-5 rating
	notes = Column(Text, nullable=True)
	
	# Performance Impact
	health_improvement = Column(Float, nullable=True)  # Health score change
	efficiency_improvement = Column(Float, nullable=True)  # Efficiency change
	reliability_improvement = Column(Float, nullable=True)  # Reliability change
	
	# Follow-up
	follow_up_required = Column(Boolean, default=False)
	follow_up_date = Column(DateTime, nullable=True)
	warranty_period_days = Column(Integer, nullable=True)
	warranty_expires = Column(DateTime, nullable=True)
	
	# Relationships
	asset = relationship("PMAsset", back_populates="maintenance_records")
	
	def __repr__(self):
		return f"<PMMaintenanceRecord {self.maintenance_type} for {self.asset.asset_name}>"
	
	def calculate_effectiveness(self) -> float:
		"""Calculate maintenance effectiveness score"""
		score = 0.0
		factors = 0
		
		# Outcome factor
		if self.outcome == 'successful':
			score += 40
		elif self.outcome == 'partial':
			score += 20
		factors += 1
		
		# Health improvement factor
		if self.health_improvement is not None:
			score += min(30, max(0, self.health_improvement))
		else:
			score += 15  # Neutral if unknown
		factors += 1
		
		# Efficiency improvement factor
		if self.efficiency_improvement is not None:
			score += min(20, max(0, self.efficiency_improvement))
		else:
			score += 10  # Neutral if unknown
		factors += 1
		
		# Quality rating factor
		if self.quality_rating is not None:
			score += (self.quality_rating / 5) * 10
		else:
			score += 5  # Neutral if unknown
		factors += 1
		
		return score / factors if factors > 0 else 0.0
	
	def is_warranty_valid(self) -> bool:
		"""Check if warranty is still valid"""
		if self.warranty_expires:
			return datetime.utcnow() <= self.warranty_expires
		return False


class PMMaintenanceAlert(Model, AuditMixin, BaseMixin):
	"""
	Maintenance alerts and notifications for proactive management.
	
	Automated alerts triggered by predictions, thresholds, or schedules
	with escalation rules and acknowledgment tracking.
	"""
	__tablename__ = 'pm_maintenance_alert'
	
	# Identity
	alert_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	asset_id = Column(String(36), ForeignKey('pm_asset.asset_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Alert Details
	alert_type = Column(String(50), nullable=False, index=True)  # threshold, prediction, schedule, anomaly
	alert_category = Column(String(50), nullable=False)  # maintenance_due, failure_risk, performance_degradation
	severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
	priority = Column(String(20), nullable=False, index=True)  # routine, urgent, emergency
	
	# Content
	title = Column(String(200), nullable=False)
	message = Column(Text, nullable=False)
	recommendation = Column(Text, nullable=True)
	
	# Trigger Information
	triggered_at = Column(DateTime, nullable=False, index=True)
	trigger_condition = Column(String(200), nullable=False)
	trigger_value = Column(Float, nullable=True)
	threshold_value = Column(Float, nullable=True)
	
	# Status and Handling
	status = Column(String(20), default='active', index=True)  # active, acknowledged, resolved, closed
	acknowledged_by = Column(String(36), nullable=True)
	acknowledged_at = Column(DateTime, nullable=True)
	resolved_by = Column(String(36), nullable=True)
	resolved_at = Column(DateTime, nullable=True)
	resolution_notes = Column(Text, nullable=True)
	
	# Escalation
	escalation_level = Column(Integer, default=0)
	escalated_at = Column(DateTime, nullable=True)
	escalation_rules = Column(JSON, default=dict)
	notification_sent = Column(Boolean, default=False)
	
	# Relationships
	asset = relationship("PMAsset")
	
	def __repr__(self):
		return f"<PMMaintenanceAlert {self.alert_type} for {self.asset.asset_name}>"
	
	def escalate(self) -> bool:
		"""Escalate alert to next level"""
		self.escalation_level += 1
		self.escalated_at = datetime.utcnow()
		# Implementation would trigger escalation notifications
		return True
	
	def acknowledge(self, user_id: str) -> bool:
		"""Acknowledge the alert"""
		if self.status == 'active':
			self.status = 'acknowledged'
			self.acknowledged_by = user_id
			self.acknowledged_at = datetime.utcnow()
			return True
		return False
	
	def resolve(self, user_id: str, notes: str = None) -> bool:
		"""Mark alert as resolved"""
		if self.status in ['active', 'acknowledged']:
			self.status = 'resolved'
			self.resolved_by = user_id
			self.resolved_at = datetime.utcnow()
			if notes:
				self.resolution_notes = notes
			return True
		return False