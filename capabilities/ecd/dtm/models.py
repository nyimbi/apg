"""
Digital Twin Models

Database models for comprehensive digital twin implementation including
virtual representations, real-time synchronization, simulation, and optimization.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


def uuid7str():
	"""Generate UUID7 string for consistent ID generation"""
	from uuid_extensions import uuid7
	return str(uuid7())


class DTDigitalTwin(Model, AuditMixin, BaseMixin):
	"""
	Core digital twin entity representing virtual counterpart of physical object.
	
	Stores digital twin metadata, configuration, state, and relationships
	with comprehensive lifecycle management and synchronization tracking.
	"""
	__tablename__ = 'dt_digital_twin'
	
	# Identity
	twin_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Twin Classification
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	twin_type = Column(String(50), nullable=False, index=True)  # asset, process, system, environment, product, human
	category = Column(String(100), nullable=True, index=True)  # machine, vehicle, building, etc.
	subcategory = Column(String(100), nullable=True)
	
	# Physical Entity Reference
	physical_entity_id = Column(String(200), nullable=True, index=True)  # External reference to physical entity
	physical_location = Column(JSON, default=dict)  # Geographic location data
	physical_properties = Column(JSON, default=dict)  # Physical characteristics
	
	# Twin State and Status
	current_state = Column(String(20), default='inactive', index=True)  # inactive, active, synchronizing, simulating, etc.
	health_status = Column(String(20), default='unknown', index=True)  # healthy, warning, critical, unknown
	operational_status = Column(String(20), default='offline', index=True)  # online, offline, maintenance
	
	# Configuration
	update_frequency = Column(Integer, default=60)  # Seconds between updates
	data_retention_days = Column(Integer, default=365)  # Data retention policy
	simulation_enabled = Column(Boolean, default=False)
	prediction_enabled = Column(Boolean, default=False)
	optimization_enabled = Column(Boolean, default=False)
	
	# Geometry and Visualization
	has_3d_model = Column(Boolean, default=False)
	model_file_path = Column(String(1000), nullable=True)
	model_format = Column(String(20), nullable=True)  # obj, fbx, gltf, etc.
	visualization_config = Column(JSON, default=dict)
	
	# Synchronization Status
	last_sync_time = Column(DateTime, nullable=True, index=True)
	sync_success_rate = Column(Float, default=0.0)  # 0-100%
	data_freshness_score = Column(Float, default=0.0)  # 0-100%
	total_sync_attempts = Column(Integer, default=0)
	failed_sync_attempts = Column(Integer, default=0)
	
	# Performance Metrics
	total_properties = Column(Integer, default=0)
	active_data_sources = Column(Integer, default=0)
	simulation_runs = Column(Integer, default=0)
	prediction_accuracy = Column(Float, nullable=True)  # 0-100% if predictions enabled
	
	# Lifecycle Management
	is_active = Column(Boolean, default=True, index=True)
	activation_date = Column(DateTime, nullable=True)
	deactivation_date = Column(DateTime, nullable=True)
	version = Column(String(20), default='1.0.0')
	
	# Ownership and Access
	owner_id = Column(String(36), nullable=True, index=True)  # Primary owner user ID
	responsible_team = Column(String(200), nullable=True)
	access_level = Column(String(20), default='private')  # public, private, restricted
	
	# Metadata
	tags = Column(JSON, default=list)
	custom_metadata = Column(JSON, default=dict)
	external_references = Column(JSON, default=dict)  # References to external systems
	
	# Relationships
	properties = relationship("DTTwinProperty", back_populates="twin", cascade="all, delete-orphan")
	data_sources = relationship("DTDataSource", back_populates="twin", cascade="all, delete-orphan")
	simulations = relationship("DTSimulation", back_populates="twin", cascade="all, delete-orphan")
	relationships_as_parent = relationship("DTTwinRelationship", 
										   foreign_keys="DTTwinRelationship.parent_twin_id",
										   back_populates="parent_twin",
										   cascade="all, delete-orphan")
	relationships_as_child = relationship("DTTwinRelationship",
										  foreign_keys="DTTwinRelationship.child_twin_id", 
										  back_populates="child_twin")
	
	def __repr__(self):
		return f"<DTDigitalTwin {self.name} ({self.twin_type})>"
	
	def is_healthy(self) -> bool:
		"""Check if digital twin is in healthy state"""
		return self.health_status == 'healthy' and self.current_state == 'active'
	
	def is_synchronized(self, max_age_minutes: int = 5) -> bool:
		"""Check if twin data is recently synchronized"""
		if not self.last_sync_time:
			return False
		
		age = datetime.utcnow() - self.last_sync_time
		return age.total_seconds() <= (max_age_minutes * 60)
	
	def calculate_sync_success_rate(self) -> None:
		"""Calculate synchronization success rate"""
		if self.total_sync_attempts > 0:
			success_rate = ((self.total_sync_attempts - self.failed_sync_attempts) 
							/ self.total_sync_attempts) * 100
			self.sync_success_rate = round(success_rate, 2)
	
	def get_location_summary(self) -> Dict[str, Any]:
		"""Get formatted location information"""
		location = self.physical_location or {}
		return {
			'coordinates': location.get('coordinates', [0, 0]),
			'address': location.get('address', 'Unknown'),
			'facility': location.get('facility', 'Unknown'),
			'zone': location.get('zone', 'Unknown')
		}
	
	def activate(self) -> None:
		"""Activate the digital twin"""
		self.is_active = True
		self.current_state = 'active'
		self.activation_date = datetime.utcnow()
		self.deactivation_date = None
	
	def deactivate(self, reason: str = None) -> None:
		"""Deactivate the digital twin"""
		self.is_active = False
		self.current_state = 'inactive'
		self.deactivation_date = datetime.utcnow()
		if reason:
			if 'deactivation_reason' not in self.custom_metadata:
				self.custom_metadata['deactivation_reason'] = reason
	
	def update_health_status(self, status: str, details: Dict[str, Any] = None) -> None:
		"""Update twin health status with optional details"""
		self.health_status = status
		if details:
			self.custom_metadata['health_details'] = details
			self.custom_metadata['health_updated_at'] = datetime.utcnow().isoformat()


class DTTwinProperty(Model, AuditMixin, BaseMixin):
	"""
	Individual properties of digital twins with time-series data support.
	
	Stores property values, metadata, quality indicators, and historical data
	with comprehensive validation and transformation capabilities.
	"""
	__tablename__ = 'dt_twin_property'
	
	# Identity
	property_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	twin_id = Column(String(36), ForeignKey('dt_digital_twin.twin_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Property Definition
	name = Column(String(100), nullable=False, index=True)
	display_name = Column(String(200), nullable=True)
	description = Column(Text, nullable=True)
	category = Column(String(50), nullable=True, index=True)  # sensor, computed, configuration, status
	
	# Current Value
	current_value = Column(Text, nullable=True)  # Stored as string, converted based on data_type
	data_type = Column(String(20), nullable=False)  # int, float, string, boolean, json, array
	unit = Column(String(50), nullable=True)
	precision = Column(Integer, nullable=True)  # Decimal places for numeric values
	
	# Value Constraints
	min_value = Column(Float, nullable=True)
	max_value = Column(Float, nullable=True)
	allowed_values = Column(JSON, default=list)  # For enum-like properties
	validation_rules = Column(JSON, default=dict)  # Custom validation rules
	
	# Quality and Reliability
	quality_score = Column(Float, default=1.0)  # 0-1 quality indicator
	confidence_level = Column(Float, default=1.0)  # 0-1 confidence in value
	uncertainty = Column(Float, nullable=True)  # Measurement uncertainty
	
	# Data Source Information
	source_type = Column(String(50), nullable=True)  # sensor, api, computed, manual
	source_id = Column(String(100), nullable=True)
	source_metadata = Column(JSON, default=dict)
	
	# Temporal Information
	last_updated = Column(DateTime, nullable=True, index=True)
	update_frequency = Column(Integer, nullable=True)  # Expected update frequency in seconds
	is_stale = Column(Boolean, default=False, index=True)  # Data freshness indicator
	
	# Processing Configuration
	is_computed = Column(Boolean, default=False)  # Computed from other properties
	computation_formula = Column(Text, nullable=True)  # Formula for computed properties
	dependencies = Column(JSON, default=list)  # Property dependencies
	
	# Alerting and Monitoring
	enable_monitoring = Column(Boolean, default=False)
	alert_thresholds = Column(JSON, default=dict)  # Alert threshold configuration
	anomaly_detection = Column(Boolean, default=False)
	baseline_value = Column(Float, nullable=True)
	
	# Historical Data Configuration
	store_history = Column(Boolean, default=True)
	history_retention_days = Column(Integer, default=365)
	aggregation_methods = Column(JSON, default=list)  # avg, min, max, sum, count
	
	# Status and Flags
	is_active = Column(Boolean, default=True)
	is_critical = Column(Boolean, default=False)  # Critical property for twin operation
	is_readonly = Column(Boolean, default=False)
	
	# Relationships
	twin = relationship("DTDigitalTwin", back_populates="properties")
	historical_values = relationship("DTPropertyHistory", back_populates="property", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<DTTwinProperty {self.name}={self.current_value} ({self.data_type})>"
	
	def get_typed_value(self) -> Any:
		"""Get current value converted to appropriate Python type"""
		if self.current_value is None:
			return None
		
		try:
			if self.data_type == 'int':
				return int(self.current_value)
			elif self.data_type == 'float':
				return float(self.current_value)
			elif self.data_type == 'boolean':
				return self.current_value.lower() in ('true', '1', 'yes', 'on')
			elif self.data_type == 'json':
				return json.loads(self.current_value)
			elif self.data_type == 'array':
				return json.loads(self.current_value) if isinstance(self.current_value, str) else self.current_value
			else:
				return str(self.current_value)
		except (ValueError, TypeError, json.JSONDecodeError):
			return self.current_value
	
	def set_value(self, value: Any, quality: float = 1.0, source: str = None) -> bool:
		"""Set property value with validation and type conversion"""
		# Validate value
		if not self.validate_value(value):
			return False
		
		# Convert and store value
		self.current_value = self._convert_value_to_string(value)
		self.quality_score = quality
		self.last_updated = datetime.utcnow()
		self.is_stale = False
		
		if source:
			self.source_metadata['last_source'] = source
			self.source_metadata['last_update'] = datetime.utcnow().isoformat()
		
		return True
	
	def validate_value(self, value: Any) -> bool:
		"""Validate value against constraints and rules"""
		try:
			# Type validation
			if self.data_type == 'int':
				value = int(value)
			elif self.data_type == 'float':
				value = float(value)
			elif self.data_type == 'boolean':
				value = bool(value)
			
			# Range validation for numeric types
			if self.data_type in ['int', 'float']:
				if self.min_value is not None and value < self.min_value:
					return False
				if self.max_value is not None and value > self.max_value:
					return False
			
			# Allowed values validation
			if self.allowed_values and value not in self.allowed_values:
				return False
			
			# Custom validation rules
			if self.validation_rules:
				# Implement custom validation logic here
				pass
			
			return True
			
		except (ValueError, TypeError):
			return False
	
	def _convert_value_to_string(self, value: Any) -> str:
		"""Convert value to string for storage"""
		if self.data_type in ['json', 'array'] and not isinstance(value, str):
			return json.dumps(value)
		return str(value)
	
	def is_within_normal_range(self) -> bool:
		"""Check if current value is within normal operating range"""
		if self.baseline_value is None:
			return True
		
		current = self.get_typed_value()
		if not isinstance(current, (int, float)):
			return True
		
		# Simple threshold check (can be enhanced with statistical methods)
		tolerance = self.alert_thresholds.get('normal_tolerance', 0.1)
		return abs(current - self.baseline_value) <= (self.baseline_value * tolerance)
	
	def check_staleness(self) -> None:
		"""Check if property data is stale based on update frequency"""
		if self.update_frequency and self.last_updated:
			time_since_update = (datetime.utcnow() - self.last_updated).total_seconds()
			expected_interval = self.update_frequency * 1.5  # 50% tolerance
			self.is_stale = time_since_update > expected_interval


class DTPropertyHistory(Model, AuditMixin, BaseMixin):
	"""
	Historical values for digital twin properties with time-series support.
	
	Stores time-series data for property values with aggregation support,
	compression, and efficient querying capabilities.
	"""
	__tablename__ = 'dt_property_history'
	
	# Identity
	history_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	property_id = Column(String(36), ForeignKey('dt_twin_property.property_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Time Series Data
	timestamp = Column(DateTime, nullable=False, index=True)
	value = Column(Text, nullable=False)  # Stored as string
	quality_score = Column(Float, default=1.0)
	
	# Data Source Context
	source_type = Column(String(50), nullable=True)
	source_id = Column(String(100), nullable=True)
	batch_id = Column(String(36), nullable=True, index=True)  # For batch processing
	
	# Aggregation Information
	is_aggregated = Column(Boolean, default=False)
	aggregation_method = Column(String(20), nullable=True)  # avg, min, max, sum, count
	aggregation_window = Column(String(20), nullable=True)  # 1min, 5min, 1hour, 1day
	sample_count = Column(Integer, nullable=True)  # Number of samples in aggregation
	
	# Value Statistics (for aggregated data)
	min_value = Column(Float, nullable=True)
	max_value = Column(Float, nullable=True)
	avg_value = Column(Float, nullable=True)
	std_deviation = Column(Float, nullable=True)
	
	# Data Quality Metrics
	interpolated = Column(Boolean, default=False)  # Value was interpolated
	outlier_score = Column(Float, nullable=True)  # Outlier detection score
	change_rate = Column(Float, nullable=True)  # Rate of change from previous value
	
	# Relationships
	property = relationship("DTTwinProperty", back_populates="historical_values")
	
	def __repr__(self):
		return f"<DTPropertyHistory {self.property_id} @ {self.timestamp}>"
	
	def get_typed_value(self) -> Any:
		"""Get value converted to appropriate Python type"""
		try:
			# Get data type from parent property
			if self.property:
				data_type = self.property.data_type
				
				if data_type == 'int':
					return int(self.value)
				elif data_type == 'float':
					return float(self.value)
				elif data_type == 'boolean':
					return self.value.lower() in ('true', '1', 'yes', 'on')
				elif data_type == 'json':
					return json.loads(self.value)
				elif data_type == 'array':
					return json.loads(self.value)
			
			return self.value
			
		except (ValueError, TypeError, json.JSONDecodeError):
			return self.value
	
	def calculate_statistics(self, values: List[float]) -> None:
		"""Calculate statistical metrics for aggregated data"""
		if not values:
			return
		
		import statistics
		
		self.min_value = min(values)
		self.max_value = max(values)
		self.avg_value = statistics.mean(values)
		self.sample_count = len(values)
		
		if len(values) > 1:
			self.std_deviation = statistics.stdev(values)


class DTDataSource(Model, AuditMixin, BaseMixin):
	"""
	Data sources feeding digital twin properties.
	
	Manages connections to external data sources including IoT sensors,
	APIs, databases, and files with health monitoring and configuration.
	"""
	__tablename__ = 'dt_data_source'
	
	# Identity
	source_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	twin_id = Column(String(36), ForeignKey('dt_digital_twin.twin_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Source Definition
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	source_type = Column(String(50), nullable=False, index=True)  # iot_sensor, api, database, file, manual
	
	# Connection Configuration
	connection_string = Column(String(1000), nullable=True)  # Connection details
	endpoint_url = Column(String(500), nullable=True)
	authentication_config = Column(JSON, default=dict)  # Auth configuration (encrypted)
	protocol = Column(String(50), nullable=True)  # HTTP, MQTT, TCP, etc.
	
	# Data Format and Parsing
	data_format = Column(String(50), nullable=False)  # json, xml, csv, binary
	parsing_config = Column(JSON, default=dict)  # Format-specific parsing configuration
	field_mappings = Column(JSON, default=dict)  # Map source fields to twin properties
	
	# Update Configuration
	update_frequency = Column(Integer, default=60)  # Seconds between updates
	is_real_time = Column(Boolean, default=False)
	batch_size = Column(Integer, default=1)
	timeout_seconds = Column(Integer, default=30)
	
	# Data Quality and Validation
	enable_validation = Column(Boolean, default=True)
	validation_rules = Column(JSON, default=dict)
	quality_threshold = Column(Float, default=0.8)  # Minimum quality threshold
	
	# Source Status and Health
	is_active = Column(Boolean, default=True, index=True)
	health_status = Column(String(20), default='unknown', index=True)  # healthy, warning, error, unknown
	last_success_time = Column(DateTime, nullable=True, index=True)
	last_error_time = Column(DateTime, nullable=True)
	consecutive_failures = Column(Integer, default=0)
	
	# Performance Metrics
	total_requests = Column(Integer, default=0)
	successful_requests = Column(Integer, default=0)
	failed_requests = Column(Integer, default=0)
	avg_response_time_ms = Column(Float, default=0.0)
	data_points_received = Column(Integer, default=0)
	
	# Error Handling
	last_error_message = Column(Text, nullable=True)
	max_retries = Column(Integer, default=3)
	retry_delay_seconds = Column(Integer, default=60)
	
	# Relationships
	twin = relationship("DTDigitalTwin", back_populates="data_sources")
	
	def __repr__(self):
		return f"<DTDataSource {self.name} ({self.source_type})>"
	
	def is_healthy(self) -> bool:
		"""Check if data source is healthy"""
		return (self.is_active and 
				self.health_status == 'healthy' and 
				self.consecutive_failures < 3)
	
	def calculate_success_rate(self) -> float:
		"""Calculate success rate percentage"""
		if self.total_requests > 0:
			return (self.successful_requests / self.total_requests) * 100
		return 0.0
	
	def update_performance_metrics(self, success: bool, response_time_ms: float = None) -> None:
		"""Update performance metrics after request"""
		self.total_requests += 1
		
		if success:
			self.successful_requests += 1
			self.consecutive_failures = 0
			self.last_success_time = datetime.utcnow()
			self.health_status = 'healthy'
		else:
			self.failed_requests += 1
			self.consecutive_failures += 1
			self.last_error_time = datetime.utcnow()
			
			# Update health status based on failure count
			if self.consecutive_failures >= 5:
				self.health_status = 'error'
			elif self.consecutive_failures >= 3:
				self.health_status = 'warning'
		
		# Update average response time
		if response_time_ms is not None:
			total_time = self.avg_response_time_ms * (self.total_requests - 1)
			self.avg_response_time_ms = (total_time + response_time_ms) / self.total_requests
	
	def can_retry(self) -> bool:
		"""Check if source can be retried after failure"""
		if self.consecutive_failures == 0:
			return True
		
		if self.consecutive_failures >= self.max_retries:
			return False
		
		if self.last_error_time:
			time_since_error = (datetime.utcnow() - self.last_error_time).total_seconds()
			return time_since_error >= self.retry_delay_seconds
		
		return True


class DTSimulation(Model, AuditMixin, BaseMixin):
	"""
	Simulation configurations and results for digital twins.
	
	Manages simulation parameters, execution, and results with support
	for various simulation types and scenario modeling.
	"""
	__tablename__ = 'dt_simulation'
	
	# Identity
	simulation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	twin_id = Column(String(36), ForeignKey('dt_digital_twin.twin_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Simulation Definition
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	simulation_type = Column(String(50), nullable=False, index=True)  # physics, thermal, structural, fluid, etc.
	model_type = Column(String(50), nullable=False)  # mathematical, ml, physics_engine, external
	
	# Simulation Configuration
	parameters = Column(JSON, nullable=False)  # Simulation parameters
	initial_conditions = Column(JSON, default=dict)  # Initial state
	boundary_conditions = Column(JSON, default=dict)  # Boundary conditions
	solver_config = Column(JSON, default=dict)  # Solver configuration
	
	# Time Configuration
	simulation_duration = Column(Float, nullable=False)  # Simulation time in appropriate units
	time_step = Column(Float, nullable=False)  # Time step size
	time_unit = Column(String(20), default='seconds')  # seconds, minutes, hours, days
	
	# Execution Configuration
	execution_mode = Column(String(20), default='batch')  # batch, real_time, scheduled
	priority = Column(String(20), default='normal')  # low, normal, high, urgent
	max_execution_time = Column(Integer, default=3600)  # Maximum execution time in seconds
	
	# Status and Results
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, cancelled
	progress_percentage = Column(Float, default=0.0)
	started_at = Column(DateTime, nullable=True, index=True)
	completed_at = Column(DateTime, nullable=True, index=True)
	
	# Results Storage
	results_summary = Column(JSON, default=dict)  # Summary of results
	output_data_path = Column(String(1000), nullable=True)  # Path to detailed results
	visualization_data = Column(JSON, default=dict)  # Data for visualization
	
	# Performance Metrics
	execution_time_seconds = Column(Float, nullable=True)
	memory_used_mb = Column(Float, nullable=True)
	cpu_usage_percent = Column(Float, nullable=True)
	convergence_achieved = Column(Boolean, nullable=True)
	accuracy_score = Column(Float, nullable=True)  # 0-100%
	
	# Error Handling
	error_message = Column(Text, nullable=True)
	warnings = Column(JSON, default=list)
	retry_count = Column(Integer, default=0)
	max_retries = Column(Integer, default=3)
	
	# Validation and Verification
	is_validated = Column(Boolean, default=False)
	validation_results = Column(JSON, default=dict)
	benchmark_data = Column(JSON, default=dict)  # For comparison with real data
	
	# Metadata
	created_by = Column(String(36), nullable=True, index=True)
	simulation_engine = Column(String(100), nullable=True)  # Engine/software used
	version = Column(String(20), default='1.0.0')
	
	# Relationships
	twin = relationship("DTDigitalTwin", back_populates="simulations")
	
	def __repr__(self):
		return f"<DTSimulation {self.name} ({self.simulation_type}) status={self.status}>"
	
	def is_running(self) -> bool:
		"""Check if simulation is currently running"""
		return self.status in ['pending', 'running']
	
	def is_completed(self) -> bool:
		"""Check if simulation completed successfully"""
		return self.status == 'completed'
	
	def get_duration(self) -> Optional[float]:
		"""Get simulation execution duration in seconds"""
		if self.started_at and self.completed_at:
			return (self.completed_at - self.started_at).total_seconds()
		elif self.started_at:
			return (datetime.utcnow() - self.started_at).total_seconds()
		return None
	
	def get_estimated_completion(self) -> Optional[datetime]:
		"""Estimate completion time based on progress"""
		if not self.started_at or self.progress_percentage <= 0:
			return None
		
		elapsed = (datetime.utcnow() - self.started_at).total_seconds()
		estimated_total = elapsed / (self.progress_percentage / 100)
		remaining = estimated_total - elapsed
		
		return datetime.utcnow() + timedelta(seconds=remaining)
	
	def update_progress(self, percentage: float, status: str = None) -> None:
		"""Update simulation progress"""
		self.progress_percentage = min(100.0, max(0.0, percentage))
		if status:
			self.status = status
	
	def validate_results(self, real_data: Dict[str, Any] = None) -> bool:
		"""Validate simulation results against real data"""
		if not self.results_summary:
			return False
		
		# Basic validation checks
		validation_passed = True
		validation_results = {}
		
		# Check for reasonable value ranges
		for key, value in self.results_summary.items():
			if isinstance(value, (int, float)):
				# Simple sanity check (can be enhanced with domain-specific logic)
				if value < -1e10 or value > 1e10:  # Extremely large/small values
					validation_passed = False
					validation_results[key] = "Value out of reasonable range"
		
		# Compare with real data if provided
		if real_data:
			for key, sim_value in self.results_summary.items():
				if key in real_data and isinstance(sim_value, (int, float)):
					real_value = real_data[key]
					if isinstance(real_value, (int, float)):
						error_percentage = abs((sim_value - real_value) / real_value) * 100
						validation_results[f"{key}_error_pct"] = error_percentage
						
						# Flag high error rates
						if error_percentage > 20:  # 20% threshold
							validation_results[f"{key}_high_error"] = True
		
		self.is_validated = validation_passed
		self.validation_results = validation_results
		
		return validation_passed


class DTTwinRelationship(Model, AuditMixin, BaseMixin):
	"""
	Relationships between digital twins for complex system modeling.
	
	Defines hierarchical and peer relationships between twins with
	dependency tracking and interaction modeling.
	"""
	__tablename__ = 'dt_twin_relationship'
	
	# Identity
	relationship_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Relationship Definition
	parent_twin_id = Column(String(36), ForeignKey('dt_digital_twin.twin_id'), nullable=False, index=True)
	child_twin_id = Column(String(36), ForeignKey('dt_digital_twin.twin_id'), nullable=False, index=True)
	relationship_type = Column(String(50), nullable=False, index=True)  # contains, depends_on, connects_to, controls
	
	# Relationship Properties
	strength = Column(Float, default=1.0)  # 0-1 strength of relationship
	bidirectional = Column(Boolean, default=False)
	is_critical = Column(Boolean, default=False)  # Critical for system operation
	
	# Interaction Data
	interaction_frequency = Column(Float, nullable=True)  # Interactions per time unit
	data_flow_direction = Column(String(20), default='bidirectional')  # upstream, downstream, bidirectional
	synchronization_required = Column(Boolean, default=False)
	
	# Status and Health
	is_active = Column(Boolean, default=True, index=True)
	health_status = Column(String(20), default='healthy')  # healthy, degraded, broken
	last_interaction = Column(DateTime, nullable=True)
	
	# Configuration
	properties = Column(JSON, default=dict)  # Relationship-specific properties
	constraints = Column(JSON, default=dict)  # Constraints on the relationship
	
	# Relationships
	parent_twin = relationship("DTDigitalTwin", 
							   foreign_keys=[parent_twin_id],
							   back_populates="relationships_as_parent")
	child_twin = relationship("DTDigitalTwin",
							  foreign_keys=[child_twin_id], 
							  back_populates="relationships_as_child")
	
	def __repr__(self):
		return f"<DTTwinRelationship {self.relationship_type}: {self.parent_twin_id} -> {self.child_twin_id}>"
	
	def is_healthy(self) -> bool:
		"""Check if relationship is healthy"""
		return self.is_active and self.health_status == 'healthy'
	
	def get_relationship_summary(self) -> Dict[str, Any]:
		"""Get relationship summary information"""
		return {
			'type': self.relationship_type,
			'strength': self.strength,
			'bidirectional': self.bidirectional,
			'critical': self.is_critical,
			'active': self.is_active,
			'health': self.health_status,
			'last_interaction': self.last_interaction.isoformat() if self.last_interaction else None
		}


class DTSystemConfiguration(Model, AuditMixin, BaseMixin):
	"""
	System-wide configuration for digital twin infrastructure.
	
	Stores tenant-specific configuration for digital twin processing,
	simulation engines, data retention, and performance settings.
	"""
	__tablename__ = 'dt_system_configuration'
	
	# Identity
	config_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, unique=True, index=True)
	
	# Processing Configuration
	max_concurrent_simulations = Column(Integer, default=5)
	default_update_frequency = Column(Integer, default=60)  # seconds
	batch_processing_enabled = Column(Boolean, default=True)
	real_time_processing_enabled = Column(Boolean, default=False)
	
	# Data Management
	default_retention_days = Column(Integer, default=365)
	auto_archiving_enabled = Column(Boolean, default=True)
	compression_enabled = Column(Boolean, default=True)
	backup_enabled = Column(Boolean, default=True)
	
	# Performance Settings
	max_properties_per_twin = Column(Integer, default=1000)
	max_history_records_per_property = Column(Integer, default=100000)
	aggregation_enabled = Column(Boolean, default=True)
	indexing_strategy = Column(String(50), default='time_based')
	
	# Simulation Settings
	default_simulation_timeout = Column(Integer, default=3600)  # seconds
	simulation_engines = Column(JSON, default=list)  # Available simulation engines
	validation_enabled = Column(Boolean, default=True)
	
	# Monitoring and Alerting
	health_monitoring_enabled = Column(Boolean, default=True)
	alert_thresholds = Column(JSON, default=dict)
	notification_channels = Column(JSON, default=list)
	
	# Integration Settings
	external_api_timeout = Column(Integer, default=30)  # seconds
	max_retry_attempts = Column(Integer, default=3)
	webhook_enabled = Column(Boolean, default=False)
	event_streaming_enabled = Column(Boolean, default=False)
	
	# Security and Access
	encryption_enabled = Column(Boolean, default=True)
	access_logging_enabled = Column(Boolean, default=True)
	data_anonymization_enabled = Column(Boolean, default=False)
	
	def __repr__(self):
		return f"<DTSystemConfiguration for tenant {self.tenant_id}>"
	
	def get_performance_limits(self) -> Dict[str, int]:
		"""Get performance limitation settings"""
		return {
			'max_concurrent_simulations': self.max_concurrent_simulations,
			'max_properties_per_twin': self.max_properties_per_twin,
			'max_history_records': self.max_history_records_per_property,
			'default_timeout': self.default_simulation_timeout
		}
	
	def should_archive_data(self, data_age_days: int) -> bool:
		"""Check if data should be archived based on age"""
		return (self.auto_archiving_enabled and 
				data_age_days > self.default_retention_days)
	
	def is_within_limits(self, twin_property_count: int) -> bool:
		"""Check if twin is within system limits"""
		return twin_property_count <= self.max_properties_per_twin