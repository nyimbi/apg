#!/usr/bin/env python3
"""
Enhanced IoT Management Flask-AppBuilder Blueprint
==================================================

Comprehensive IoT interface with PostgreSQL models, Flask-AppBuilder views,
and enhanced audio anomaly detection capabilities.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import BaseView, ModelView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.forms import DynamicForm
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from wtforms import StringField, TextAreaField, SelectField, IntegerField, FloatField, BooleanField, FileField
from wtforms.validators import DataRequired, Optional as OptionalValidator, NumberRange
import plotly.graph_objs as go
import plotly.utils

from blueprints.base import BaseCapabilityModel, BaseCapabilityView, BaseCapabilityModelView, uuid7str

# Enhanced PostgreSQL Models for IoT Management
class IoTProject(BaseCapabilityModel):
	"""IoT projects for organizing devices and deployments"""
	
	__tablename__ = 'iot_projects'
	
	name = Column(String(200), nullable=False)
	description = Column(Text)
	project_type = Column(String(50), default='general')  # smart_home, industrial, agriculture, healthcare
	deployment_environment = Column(String(50), default='indoor')  # indoor, outdoor, mixed
	status = Column(String(20), default='active')  # active, archived, completed, suspended
	settings = Column(JSONB, default=dict)
	
	# Project metadata
	location = Column(JSONB, default=dict)  # Geographic location
	budget = Column(Float)
	start_date = Column(DateTime)
	end_date = Column(DateTime)
	
	# Relationships
	devices = relationship("IoTDevice", back_populates="project", cascade="all, delete-orphan")
	gateways = relationship("IoTGateway", back_populates="project", cascade="all, delete-orphan")

class IoTDevice(BaseCapabilityModel):
	"""Enhanced IoT devices with comprehensive sensor and actuator support"""
	
	__tablename__ = 'iot_devices'
	
	project_id = Column(UUID(as_uuid=True), ForeignKey('iot_projects.id'), nullable=True)
	gateway_id = Column(UUID(as_uuid=True), ForeignKey('iot_gateways.id'), nullable=True)
	
	device_id = Column(String(100), nullable=False, unique=True, index=True)
	name = Column(String(200), nullable=False)
	description = Column(Text)
	
	# Device classification
	device_type = Column(String(50), nullable=False, index=True)  # sensor, actuator, gateway, hybrid
	device_category = Column(String(50), index=True)  # environmental, security, industrial, medical, audio
	manufacturer = Column(String(100))
	model = Column(String(100))
	firmware_version = Column(String(50))
	hardware_version = Column(String(50))
	
	# Network and connectivity
	connection_type = Column(String(50), default='wifi', index=True)  # wifi, ethernet, lora, zigbee, bluetooth, cellular
	ip_address = Column(String(45))  # IPv4 or IPv6
	mac_address = Column(String(17))
	network_quality = Column(Float)  # Signal strength or quality metric
	
	# Location and physical properties
	location = Column(JSONB, default=dict)  # {name, latitude, longitude, altitude, indoor/outdoor}
	installation_date = Column(DateTime)
	physical_properties = Column(JSONB, default=dict)  # Size, weight, power requirements
	
	# Sensor capabilities
	sensor_types = Column(ARRAY(String), default=list)  # temperature, humidity, pressure, audio, vibration, etc.
	measurement_range = Column(JSONB, default=dict)  # Min/max values for each sensor
	measurement_accuracy = Column(JSONB, default=dict)  # Accuracy specs for each sensor
	sampling_rate = Column(Float, default=1.0)  # Samples per second
	
	# Audio-specific capabilities
	audio_capabilities = Column(JSONB, default=dict)  # Sample rate, channels, bit depth, formats
	supports_recording = Column(Boolean, default=False)
	supports_streaming = Column(Boolean, default=False)
	microphone_specs = Column(JSONB, default=dict)  # Frequency response, sensitivity, etc.
	
	# Device status and health
	status = Column(String(20), default='offline', index=True)  # online, offline, error, maintenance
	last_seen = Column(DateTime, index=True)
	battery_level = Column(Float)  # 0-100 percentage
	temperature = Column(Float)  # Device temperature
	uptime_seconds = Column(Integer)
	
	# Communication and data settings
	data_format = Column(String(50), default='json')  # json, xml, binary, csv
	compression_enabled = Column(Boolean, default=False)
	encryption_enabled = Column(Boolean, default=True)
	
	# Device configuration
	configuration = Column(JSONB, default=dict)
	is_enabled = Column(Boolean, default=True)
	
	# Relationships
	project = relationship("IoTProject", back_populates="devices")
	gateway = relationship("IoTGateway", back_populates="devices")
	sensor_readings = relationship("IoTSensorReading", back_populates="device", cascade="all, delete-orphan")
	audio_recordings = relationship("IoTAudioRecording", back_populates="device", cascade="all, delete-orphan")
	anomaly_detections = relationship("IoTAnomalyDetection", back_populates="device", cascade="all, delete-orphan")
	
	__table_args__ = (
		Index('ix_iot_devices_type_status', 'device_type', 'status'),
		Index('ix_iot_devices_category_type', 'device_category', 'device_type'),
		Index('ix_iot_devices_location', 'project_id', 'status'),
	)

class IoTGateway(BaseCapabilityModel):
	"""IoT gateways for managing device networks"""
	
	__tablename__ = 'iot_gateways'
	
	project_id = Column(UUID(as_uuid=True), ForeignKey('iot_projects.id'), nullable=True)
	
	gateway_id = Column(String(100), nullable=False, unique=True)
	name = Column(String(200), nullable=False)
	description = Column(Text)
	
	# Gateway specifications
	gateway_type = Column(String(50), nullable=False)  # edge, cloud, hybrid
	supported_protocols = Column(ARRAY(String), default=list)  # mqtt, coap, http, websocket
	max_devices = Column(Integer, default=100)
	current_device_count = Column(Integer, default=0)
	
	# Network configuration
	ip_address = Column(String(45))
	port_range = Column(String(20))  # e.g., "1883-1893"
	ssl_enabled = Column(Boolean, default=True)
	
	# Processing capabilities
	edge_computing_enabled = Column(Boolean, default=False)
	data_processing_capacity = Column(String(50))  # low, medium, high, unlimited
	storage_capacity_gb = Column(Float)
	
	# Status and performance
	status = Column(String(20), default='offline')
	cpu_usage = Column(Float)
	memory_usage = Column(Float)
	storage_usage = Column(Float)
	network_throughput = Column(Float)  # Mbps
	
	# Relationships
	project = relationship("IoTProject", back_populates="gateways")
	devices = relationship("IoTDevice", back_populates="gateway")
	
	__table_args__ = (
		Index('ix_iot_gateways_type_status', 'gateway_type', 'status'),
	)

class IoTSensorReading(BaseCapabilityModel):
	"""Enhanced sensor readings with support for all sensor types"""
	
	__tablename__ = 'iot_sensor_readings'
	
	device_id = Column(UUID(as_uuid=True), ForeignKey('iot_devices.id'), nullable=False, index=True)
	
	sensor_id = Column(String(100), nullable=False, index=True)
	sensor_type = Column(String(50), nullable=False, index=True)  # temperature, humidity, audio, vibration, etc.
	
	# Measurement data
	value = Column(Float, nullable=False)
	unit = Column(String(20), nullable=False)
	quality = Column(Float, default=1.0)  # 0-1 quality score
	
	# Contextual information
	measurement_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
	location = Column(JSONB, default=dict)  # If different from device location
	environmental_conditions = Column(JSONB, default=dict)  # Temperature, humidity during measurement
	
	# Audio-specific fields
	audio_file_path = Column(String(1000))  # Path to audio file if applicable
	audio_duration_ms = Column(Float)
	audio_sample_rate = Column(Integer)
	audio_channels = Column(Integer)
	audio_format = Column(String(20))
	
	# Signal processing results
	frequency_analysis = Column(JSONB, default=dict)  # FFT results, dominant frequencies
	statistical_features = Column(JSONB, default=dict)  # Mean, std, min, max, etc.
	
	# Data validation
	is_validated = Column(Boolean, default=False)
	validation_flags = Column(ARRAY(String), default=list)  # outlier, noise, calibration_needed
	
	# Raw data storage
	raw_data = Column(JSONB)  # For complex sensor data
	
	# Relationships
	device = relationship("IoTDevice", back_populates="sensor_readings")
	
	__table_args__ = (
		Index('ix_iot_sensor_readings_device_sensor', 'device_id', 'sensor_type'),
		Index('ix_iot_sensor_readings_timestamp', 'measurement_timestamp'),
		Index('ix_iot_sensor_readings_device_timestamp', 'device_id', 'measurement_timestamp'),
	)

class IoTAudioRecording(BaseCapabilityModel):
	"""Audio recordings from IoT devices with analysis"""
	
	__tablename__ = 'iot_audio_recordings'
	
	device_id = Column(UUID(as_uuid=True), ForeignKey('iot_devices.id'), nullable=False, index=True)
	
	recording_id = Column(String(100), nullable=False, unique=True)
	filename = Column(String(500), nullable=False)
	file_path = Column(String(1000), nullable=False)
	file_size_bytes = Column(Integer)
	
	# Recording metadata
	duration_seconds = Column(Float, nullable=False)
	sample_rate = Column(Integer, nullable=False)
	channels = Column(Integer, default=1)
	bit_depth = Column(Integer, default=16)
	audio_format = Column(String(20), default='wav')
	
	# Recording context
	recording_trigger = Column(String(50))  # manual, scheduled, event, anomaly
	trigger_metadata = Column(JSONB, default=dict)
	environmental_noise_level = Column(Float)
	
	# Audio analysis results
	volume_analysis = Column(JSONB, default=dict)  # RMS, peak levels, dynamic range
	frequency_analysis = Column(JSONB, default=dict)  # Spectral analysis
	voice_activity_detection = Column(JSONB, default=dict)  # Speech segments
	noise_analysis = Column(JSONB, default=dict)  # Background noise characteristics
	
	# Anomaly detection results
	anomaly_score = Column(Float)
	anomaly_detected = Column(Boolean, default=False)
	anomaly_type = Column(String(100))  # unusual_sound, silence, distortion, etc.
	anomaly_confidence = Column(Float)
	
	# Sound classification
	detected_sounds = Column(JSONB, default=list)  # List of classified sounds
	sound_confidence_scores = Column(JSONB, default=dict)
	
	# Processing status
	processing_status = Column(String(20), default='pending')  # pending, processing, completed, failed
	analysis_completed_at = Column(DateTime)
	error_message = Column(Text)
	
	# Relationships
	device = relationship("IoTDevice", back_populates="audio_recordings")
	
	__table_args__ = (
		Index('ix_iot_audio_recordings_device_timestamp', 'device_id', 'created_at'),
		Index('ix_iot_audio_recordings_anomaly', 'anomaly_detected', 'anomaly_score'),
		Index('ix_iot_audio_recordings_status', 'processing_status'),
	)

class IoTAnomalyDetection(BaseCapabilityModel):
	"""Anomaly detection results for IoT devices and sensors"""
	
	__tablename__ = 'iot_anomaly_detections'
	
	device_id = Column(UUID(as_uuid=True), ForeignKey('iot_devices.id'), nullable=False, index=True)
	
	anomaly_id = Column(String(100), nullable=False, unique=True)
	anomaly_type = Column(String(50), nullable=False, index=True)  # sensor, audio, behavioral, network
	detection_method = Column(String(50), nullable=False)  # statistical, ml, rule_based, hybrid
	
	# Anomaly details
	severity = Column(String(20), default='medium', index=True)  # low, medium, high, critical
	confidence_score = Column(Float, nullable=False)
	anomaly_score = Column(Float)
	threshold_value = Column(Float)
	
	# Context and data
	affected_sensors = Column(ARRAY(String), default=list)
	anomaly_data = Column(JSONB, default=dict)  # Detailed anomaly information
	baseline_data = Column(JSONB, default=dict)  # Normal behavior baseline
	
	# Time window
	detection_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
	anomaly_start_time = Column(DateTime)
	anomaly_end_time = Column(DateTime)
	duration_seconds = Column(Float)
	
	# Response and resolution
	status = Column(String(20), default='detected', index=True)  # detected, investigating, resolved, false_positive
	response_actions = Column(JSONB, default=list)  # Actions taken in response
	resolution_notes = Column(Text)
	resolved_at = Column(DateTime)
	resolved_by = Column(String(100))
	
	# Machine learning model info
	model_name = Column(String(100))
	model_version = Column(String(50))
	feature_importance = Column(JSONB, default=dict)
	
	# Relationships
	device = relationship("IoTDevice", back_populates="anomaly_detections")
	
	__table_args__ = (
		Index('ix_iot_anomaly_detections_device_type', 'device_id', 'anomaly_type'),
		Index('ix_iot_anomaly_detections_severity_status', 'severity', 'status'),
		Index('ix_iot_anomaly_detections_timestamp', 'detection_timestamp'),
	)

class IoTAlertRule(BaseCapabilityModel):
	"""Enhanced alert rules with audio and complex conditions"""
	
	__tablename__ = 'iot_alert_rules'
	
	name = Column(String(200), nullable=False)
	description = Column(Text)
	
	# Rule scope
	device_filter = Column(JSONB, default=dict)  # Filter criteria for devices
	sensor_types = Column(ARRAY(String), default=list)  # Sensor types to monitor
	
	# Conditions
	condition_type = Column(String(50), nullable=False)  # threshold, pattern, anomaly, composite
	conditions = Column(JSONB, nullable=False)  # Detailed condition specification
	
	# Audio-specific conditions
	audio_conditions = Column(JSONB, default=dict)  # Volume thresholds, frequency patterns
	sound_patterns = Column(JSONB, default=list)  # Specific sound patterns to detect
	
	# Timing and frequency
	evaluation_interval_seconds = Column(Integer, default=60)
	time_window_seconds = Column(Integer, default=300)  # Look-back window
	cooldown_period_seconds = Column(Integer, default=600)  # Min time between alerts
	
	# Actions
	alert_actions = Column(JSONB, default=list)  # List of actions to take
	escalation_rules = Column(JSONB, default=dict)  # Escalation after time/count
	
	# Rule status
	is_enabled = Column(Boolean, default=True)
	priority = Column(String(20), default='medium')  # low, medium, high, critical
	
	# Statistics
	trigger_count = Column(Integer, default=0)
	last_triggered = Column(DateTime)
	false_positive_count = Column(Integer, default=0)
	
	__table_args__ = (
		Index('ix_iot_alert_rules_enabled_priority', 'is_enabled', 'priority'),
	)

# Enhanced Forms for IoT Operations
class IoTDeviceRegistrationForm(DynamicForm):
	"""Enhanced form for registering IoT devices"""
	
	project_id = SelectField(
		'Project',
		validators=[OptionalValidator()],
		description='Associate with existing project (optional)'
	)
	
	device_id = StringField(
		'Device ID',
		validators=[DataRequired()],
		description='Unique identifier for the device'
	)
	
	name = StringField(
		'Device Name',
		validators=[DataRequired()],
		description='Human-readable name for the device'
	)
	
	device_type = SelectField(
		'Device Type',
		choices=[
			('sensor', 'Sensor'),
			('actuator', 'Actuator'),
			('gateway', 'Gateway'),
			('hybrid', 'Hybrid (Sensor + Actuator)'),
			('audio_device', 'Audio Recording Device'),
			('environmental_monitor', 'Environmental Monitor'),
			('security_device', 'Security Device')
		],
		validators=[DataRequired()],
		description='Primary function of the device'
	)
	
	device_category = SelectField(
		'Device Category',
		choices=[
			('environmental', 'Environmental Monitoring'),
			('security', 'Security & Surveillance'),
			('industrial', 'Industrial Automation'),
			('medical', 'Medical & Healthcare'),
			('audio', 'Audio Monitoring'),
			('energy', 'Energy Management'),
			('agriculture', 'Agriculture & Farming'),
			('smart_home', 'Smart Home'),
			('transportation', 'Transportation')
		],
		validators=[OptionalValidator()],
		description='Application category for the device'
	)
	
	connection_type = SelectField(
		'Connection Type',
		choices=[
			('wifi', 'Wi-Fi'),
			('ethernet', 'Ethernet'),
			('bluetooth', 'Bluetooth'),
			('zigbee', 'Zigbee'),
			('lora', 'LoRa'),
			('cellular', 'Cellular'),
			('satellite', 'Satellite'),
			('mesh', 'Mesh Network')
		],
		default='wifi',
		description='How the device connects to the network'
	)
	
	sensor_types = SelectField(
		'Primary Sensors',
		choices=[
			('temperature', 'Temperature'),
			('humidity', 'Humidity'),
			('pressure', 'Pressure'),
			('audio', 'Audio/Microphone'),
			('vibration', 'Vibration'),
			('light', 'Light/Illuminance'),
			('motion', 'Motion/PIR'),
			('gas', 'Gas Detection'),
			('air_quality', 'Air Quality'),
			('soil_moisture', 'Soil Moisture'),
			('ph', 'pH Level'),
			('magnetic', 'Magnetic Field'),
			('proximity', 'Proximity'),
			('camera', 'Camera/Vision')
		],
		description='Primary sensor capabilities'
	)
	
	supports_audio = BooleanField(
		'Audio Recording Capable',
		default=False,
		description='Device can record audio'
	)
	
	audio_sample_rate = IntegerField(
		'Audio Sample Rate (Hz)',
		validators=[OptionalValidator(), NumberRange(8000, 96000)],
		description='Audio recording sample rate'
	)
	
	audio_channels = SelectField(
		'Audio Channels',
		choices=[
			('1', 'Mono'),
			('2', 'Stereo'),
			('4', '4 Channel'),
			('8', '8 Channel')
		],
		default='1',
		description='Number of audio channels'
	)

class IoTAudioAnalysisForm(DynamicForm):
	"""Form for audio analysis configuration"""
	
	device_id = SelectField(
		'Audio Device',
		validators=[DataRequired()],
		description='Select audio-capable device'
	)
	
	recording_duration = IntegerField(
		'Recording Duration (seconds)',
		default=30,
		validators=[NumberRange(1, 3600)],
		description='Length of audio recording'
	)
	
	analysis_types = SelectField(
		'Analysis Types',
		choices=[
			('anomaly_detection', 'Anomaly Detection'),
			('sound_classification', 'Sound Classification'),
			('noise_analysis', 'Noise Analysis'),
			('voice_activity', 'Voice Activity Detection'),
			('frequency_analysis', 'Frequency Analysis'),
			('volume_analysis', 'Volume Analysis')
		],
		description='Types of audio analysis to perform'
	)
	
	anomaly_sensitivity = SelectField(
		'Anomaly Detection Sensitivity',
		choices=[
			('low', 'Low (Fewer False Positives)'),
			('medium', 'Medium'),
			('high', 'High (More Sensitive)')
		],
		default='medium',
		description='Sensitivity level for anomaly detection'
	)
	
	background_noise_modeling = BooleanField(
		'Model Background Noise',
		default=True,
		description='Learn and model typical background noise'
	)

# Enhanced Flask-AppBuilder Views
class EnhancedIoTManagementView(BaseCapabilityView):
	"""Enhanced IoT management interface with audio capabilities"""
	
	route_base = '/enhanced_iot'
	default_view = 'dashboard'
	
	def __init__(self):
		super().__init__()
		# In real implementation, initialize IoT capability
		pass
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Enhanced IoT dashboard with audio monitoring"""
		stats = self._get_dashboard_stats()
		recent_activities = self._get_recent_activities()
		audio_alerts = self._get_recent_audio_alerts()
		
		return self.render_template(
			'enhanced_iot/dashboard.html',
			stats=stats,
			recent_activities=recent_activities,
			audio_alerts=audio_alerts
		)
	
	@expose('/audio_monitoring')
	@has_access
	def audio_monitoring(self):
		"""Audio monitoring and anomaly detection"""
		audio_devices = []  # Query audio-capable devices
		recent_recordings = []  # Recent audio recordings
		anomaly_alerts = []  # Recent audio anomalies
		
		return self.render_template(
			'enhanced_iot/audio_monitoring.html',
			audio_devices=audio_devices,
			recent_recordings=recent_recordings,
			anomaly_alerts=anomaly_alerts
		)
	
	@expose('/anomaly_detection')
	@has_access
	def anomaly_detection(self):
		"""Comprehensive anomaly detection dashboard"""
		anomaly_stats = self._get_anomaly_stats()
		recent_anomalies = []  # Query recent anomalies
		
		return self.render_template(
			'enhanced_iot/anomaly_detection.html',
			anomaly_stats=anomaly_stats,
			recent_anomalies=recent_anomalies
		)
	
	@expose('/device_health')
	@has_access
	def device_health(self):
		"""Device health monitoring and diagnostics"""
		health_metrics = self._get_device_health_metrics()
		
		return self.render_template(
			'enhanced_iot/device_health.html',
			health_metrics=health_metrics
		)
	
	def _get_dashboard_stats(self):
		"""Enhanced dashboard statistics"""
		return {
			'total_devices': 156,
			'online_devices': 142,
			'audio_devices': 23,
			'total_sensor_readings': 1247893,
			'audio_recordings_today': 127,
			'anomalies_detected_today': 8,
			'audio_anomalies_today': 3,
			'average_response_time': 245.7,
			'network_uptime': 99.2,
			'data_quality_score': 96.8
		}
	
	def _get_anomaly_stats(self):
		"""Get anomaly detection statistics"""
		return {
			'total_anomalies': 1247,
			'audio_anomalies': 89,
			'sensor_anomalies': 1034,
			'network_anomalies': 124,
			'resolved_anomalies': 1189,
			'false_positives': 58,
			'avg_detection_time': 2.3,
			'avg_resolution_time': 15.7
		}
	
	def _get_device_health_metrics(self):
		"""Get device health metrics"""
		return {
			'healthy_devices': 142,
			'warning_devices': 8,
			'critical_devices': 6,
			'avg_battery_level': 78.3,
			'avg_signal_strength': 85.2,
			'devices_needing_maintenance': 12
		}

# PostgreSQL Schema Scripts for Enhanced IoT
ENHANCED_IOT_SCHEMAS = {
	'iot_projects': """
CREATE TABLE IF NOT EXISTS iot_projects (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	name VARCHAR(200) NOT NULL,
	description TEXT,
	project_type VARCHAR(50) DEFAULT 'general',
	deployment_environment VARCHAR(50) DEFAULT 'indoor',
	status VARCHAR(20) DEFAULT 'active',
	settings JSONB DEFAULT '{}'::jsonb,
	location JSONB DEFAULT '{}'::jsonb,
	budget FLOAT,
	start_date TIMESTAMP WITH TIME ZONE,
	end_date TIMESTAMP WITH TIME ZONE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_iot_projects_name ON iot_projects(name);
CREATE INDEX IF NOT EXISTS ix_iot_projects_type ON iot_projects(project_type);
CREATE INDEX IF NOT EXISTS ix_iot_projects_status ON iot_projects(status);
""",

	'iot_devices': """
CREATE TABLE IF NOT EXISTS iot_devices (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	project_id UUID REFERENCES iot_projects(id) ON DELETE SET NULL,
	gateway_id UUID REFERENCES iot_gateways(id) ON DELETE SET NULL,
	device_id VARCHAR(100) NOT NULL UNIQUE,
	name VARCHAR(200) NOT NULL,
	description TEXT,
	device_type VARCHAR(50) NOT NULL,
	device_category VARCHAR(50),
	manufacturer VARCHAR(100),
	model VARCHAR(100),
	firmware_version VARCHAR(50),
	hardware_version VARCHAR(50),
	connection_type VARCHAR(50) DEFAULT 'wifi',
	ip_address VARCHAR(45),
	mac_address VARCHAR(17),
	network_quality FLOAT,
	location JSONB DEFAULT '{}'::jsonb,
	installation_date TIMESTAMP WITH TIME ZONE,
	physical_properties JSONB DEFAULT '{}'::jsonb,
	sensor_types TEXT[],
	measurement_range JSONB DEFAULT '{}'::jsonb,
	measurement_accuracy JSONB DEFAULT '{}'::jsonb,
	sampling_rate FLOAT DEFAULT 1.0,
	audio_capabilities JSONB DEFAULT '{}'::jsonb,
	supports_recording BOOLEAN DEFAULT FALSE,
	supports_streaming BOOLEAN DEFAULT FALSE,
	microphone_specs JSONB DEFAULT '{}'::jsonb,
	status VARCHAR(20) DEFAULT 'offline',
	last_seen TIMESTAMP WITH TIME ZONE,
	battery_level FLOAT,
	temperature FLOAT,
	uptime_seconds INTEGER,
	data_format VARCHAR(50) DEFAULT 'json',
	compression_enabled BOOLEAN DEFAULT FALSE,
	encryption_enabled BOOLEAN DEFAULT TRUE,
	configuration JSONB DEFAULT '{}'::jsonb,
	is_enabled BOOLEAN DEFAULT TRUE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_iot_devices_device_id ON iot_devices(device_id);
CREATE INDEX IF NOT EXISTS ix_iot_devices_device_type ON iot_devices(device_type);
CREATE INDEX IF NOT EXISTS ix_iot_devices_status ON iot_devices(status);
CREATE INDEX IF NOT EXISTS ix_iot_devices_connection_type ON iot_devices(connection_type);
CREATE INDEX IF NOT EXISTS ix_iot_devices_device_category ON iot_devices(device_category);
CREATE INDEX IF NOT EXISTS ix_iot_devices_last_seen ON iot_devices(last_seen);
CREATE INDEX IF NOT EXISTS ix_iot_devices_type_status ON iot_devices(device_type, status);
CREATE INDEX IF NOT EXISTS ix_iot_devices_category_type ON iot_devices(device_category, device_type);
""",

	'iot_audio_recordings': """
CREATE TABLE IF NOT EXISTS iot_audio_recordings (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	device_id UUID NOT NULL REFERENCES iot_devices(id) ON DELETE CASCADE,
	recording_id VARCHAR(100) NOT NULL UNIQUE,
	filename VARCHAR(500) NOT NULL,
	file_path VARCHAR(1000) NOT NULL,
	file_size_bytes INTEGER,
	duration_seconds FLOAT NOT NULL,
	sample_rate INTEGER NOT NULL,
	channels INTEGER DEFAULT 1,
	bit_depth INTEGER DEFAULT 16,
	audio_format VARCHAR(20) DEFAULT 'wav',
	recording_trigger VARCHAR(50),
	trigger_metadata JSONB DEFAULT '{}'::jsonb,
	environmental_noise_level FLOAT,
	volume_analysis JSONB DEFAULT '{}'::jsonb,
	frequency_analysis JSONB DEFAULT '{}'::jsonb,
	voice_activity_detection JSONB DEFAULT '{}'::jsonb,
	noise_analysis JSONB DEFAULT '{}'::jsonb,
	anomaly_score FLOAT,
	anomaly_detected BOOLEAN DEFAULT FALSE,
	anomaly_type VARCHAR(100),
	anomaly_confidence FLOAT,
	detected_sounds JSONB DEFAULT '[]'::jsonb,
	sound_confidence_scores JSONB DEFAULT '{}'::jsonb,
	processing_status VARCHAR(20) DEFAULT 'pending',
	analysis_completed_at TIMESTAMP WITH TIME ZONE,
	error_message TEXT,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_iot_audio_recordings_device_id ON iot_audio_recordings(device_id);
CREATE INDEX IF NOT EXISTS ix_iot_audio_recordings_recording_id ON iot_audio_recordings(recording_id);
CREATE INDEX IF NOT EXISTS ix_iot_audio_recordings_created_at ON iot_audio_recordings(created_at);
CREATE INDEX IF NOT EXISTS ix_iot_audio_recordings_device_timestamp ON iot_audio_recordings(device_id, created_at);
CREATE INDEX IF NOT EXISTS ix_iot_audio_recordings_anomaly ON iot_audio_recordings(anomaly_detected, anomaly_score);
CREATE INDEX IF NOT EXISTS ix_iot_audio_recordings_status ON iot_audio_recordings(processing_status);
""",

	'iot_anomaly_detections': """
CREATE TABLE IF NOT EXISTS iot_anomaly_detections (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	device_id UUID NOT NULL REFERENCES iot_devices(id) ON DELETE CASCADE,
	anomaly_id VARCHAR(100) NOT NULL UNIQUE,
	anomaly_type VARCHAR(50) NOT NULL,
	detection_method VARCHAR(50) NOT NULL,
	severity VARCHAR(20) DEFAULT 'medium',
	confidence_score FLOAT NOT NULL,
	anomaly_score FLOAT,
	threshold_value FLOAT,
	affected_sensors TEXT[],
	anomaly_data JSONB DEFAULT '{}'::jsonb,
	baseline_data JSONB DEFAULT '{}'::jsonb,
	detection_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	anomaly_start_time TIMESTAMP WITH TIME ZONE,
	anomaly_end_time TIMESTAMP WITH TIME ZONE,
	duration_seconds FLOAT,
	status VARCHAR(20) DEFAULT 'detected',
	response_actions JSONB DEFAULT '[]'::jsonb,
	resolution_notes TEXT,
	resolved_at TIMESTAMP WITH TIME ZONE,
	resolved_by VARCHAR(100),
	model_name VARCHAR(100),
	model_version VARCHAR(50),
	feature_importance JSONB DEFAULT '{}'::jsonb,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_iot_anomaly_detections_device_id ON iot_anomaly_detections(device_id);
CREATE INDEX IF NOT EXISTS ix_iot_anomaly_detections_anomaly_id ON iot_anomaly_detections(anomaly_id);
CREATE INDEX IF NOT EXISTS ix_iot_anomaly_detections_anomaly_type ON iot_anomaly_detections(anomaly_type);
CREATE INDEX IF NOT EXISTS ix_iot_anomaly_detections_severity ON iot_anomaly_detections(severity);
CREATE INDEX IF NOT EXISTS ix_iot_anomaly_detections_status ON iot_anomaly_detections(status);
CREATE INDEX IF NOT EXISTS ix_iot_anomaly_detections_detection_timestamp ON iot_anomaly_detections(detection_timestamp);
CREATE INDEX IF NOT EXISTS ix_iot_anomaly_detections_device_type ON iot_anomaly_detections(device_id, anomaly_type);
CREATE INDEX IF NOT EXISTS ix_iot_anomaly_detections_severity_status ON iot_anomaly_detections(severity, status);
"""
}

# Blueprint registration
enhanced_iot_bp = Blueprint(
	'enhanced_iot',
	__name__,
	template_folder='templates',
	static_folder='static'
)

__all__ = [
	'EnhancedIoTManagementView', 'IoTProject', 'IoTDevice', 'IoTGateway', 
	'IoTSensorReading', 'IoTAudioRecording', 'IoTAnomalyDetection', 'IoTAlertRule',
	'ENHANCED_IOT_SCHEMAS', 'enhanced_iot_bp'
]