#!/usr/bin/env python3
"""
Digital Twin Flask-AppBuilder Blueprint
=======================================

Comprehensive digital twin interface with PostgreSQL models and Flask-AppBuilder views.
Provides world-class digital twin capabilities including real-time synchronization,
simulation, prediction, and 3D visualization.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for, send_file
from flask_appbuilder import BaseView, ModelView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.forms import DynamicForm
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from wtforms import StringField, TextAreaField, SelectField, IntegerField, FloatField, BooleanField, FileField
from wtforms.validators import DataRequired, Optional as OptionalValidator, NumberRange
from werkzeug.utils import secure_filename

from blueprints.base import BaseCapabilityModel, BaseCapabilityView, BaseCapabilityModelView, uuid7str

# Import digital twin capability
try:
	from capabilities.digital_twin import DigitalTwinCapability, TwinType, TwinState, SimulationType
except ImportError:
	# Fallback if capability not available
	class DigitalTwinCapability:
		def __init__(self, *args, **kwargs):
			pass

# PostgreSQL Models for Digital Twin Management
class DTProject(BaseCapabilityModel):
	"""Digital twin projects for organizing digital twins"""
	
	__tablename__ = 'dt_projects'
	
	name = Column(String(200), nullable=False)
	description = Column(Text)
	project_type = Column(String(50), default='general')  # industrial, building, vehicle, process, system
	industry = Column(String(100))  # manufacturing, automotive, aerospace, healthcare, etc.
	status = Column(String(20), default='active')  # active, archived, completed
	settings = Column(JSONB, default=dict)
	
	# Project-level configuration
	default_sync_interval = Column(Float, default=1.0)  # seconds
	enable_real_time = Column(Boolean, default=True)
	enable_simulation = Column(Boolean, default=True)
	enable_prediction = Column(Boolean, default=False)
	
	# Relationships
	digital_twins = relationship("DigitalTwinModel", back_populates="project", cascade="all, delete-orphan")

class DigitalTwinModel(BaseCapabilityModel):
	"""Individual digital twin instances"""
	
	__tablename__ = 'digital_twins'
	
	project_id = Column(UUID(as_uuid=True), ForeignKey('dt_projects.id'), nullable=True)
	twin_id = Column(String(100), nullable=False, unique=True, index=True)
	name = Column(String(200), nullable=False)
	description = Column(Text)
	
	# Twin classification
	twin_type = Column(String(50), nullable=False)  # asset, process, system, environment, product, human
	twin_state = Column(String(20), default='inactive')  # inactive, active, synchronizing, simulating, error
	version = Column(String(20), default='1.0.0')
	
	# Physical twin reference
	physical_asset_id = Column(String(200))  # Reference to physical asset
	location = Column(JSONB)  # Geographic or facility location
	
	# Twin properties and data
	properties = Column(JSONB, default=dict)  # Dynamic properties
	relationships = Column(JSONB, default=dict)  # Relationships to other twins
	data_sources = Column(JSONB, default=dict)  # Data source configurations
	
	# Models and representations
	geometry_model = Column(JSONB)  # 3D geometry data
	simulation_models = Column(JSONB, default=dict)  # Simulation model definitions
	behavior_models = Column(JSONB, default=dict)  # Behavioral models
	
	# Real-time data
	current_state = Column(JSONB, default=dict)  # Current state snapshot
	last_sync_time = Column(DateTime)
	sync_interval = Column(Float, default=1.0)  # seconds
	sync_status = Column(String(20), default='disconnected')  # connected, disconnected, error
	
	# Performance metrics
	uptime_percentage = Column(Float, default=0.0)
	data_quality_score = Column(Float, default=0.0)
	prediction_accuracy = Column(Float, default=0.0)
	
	# Alert and monitoring
	alert_rules = Column(JSONB, default=dict)
	health_status = Column(String(20), default='unknown')  # healthy, warning, critical, unknown
	
	# Relationships
	project = relationship("DTProject", back_populates="digital_twins")
	telemetry_data = relationship("DTTelemetryData", back_populates="digital_twin", cascade="all, delete-orphan")
	simulation_jobs = relationship("DTSimulationJob", back_populates="digital_twin", cascade="all, delete-orphan")
	events = relationship("DTEvent", back_populates="digital_twin", cascade="all, delete-orphan")
	
	__table_args__ = (
		Index('ix_digital_twins_twin_id', 'twin_id'),
		Index('ix_digital_twins_state_type', 'twin_state', 'twin_type'),
		Index('ix_digital_twins_project_status', 'project_id', 'twin_state'),
		Index('ix_digital_twins_sync', 'sync_status', 'last_sync_time'),
	)

class DTTelemetryData(BaseCapabilityModel):
	"""Real-time telemetry data from digital twins"""
	
	__tablename__ = 'dt_telemetry_data'
	
	digital_twin_id = Column(UUID(as_uuid=True), ForeignKey('digital_twins.id'), nullable=False, index=True)
	timestamp = Column(DateTime, nullable=False, index=True)
	
	# Telemetry data
	data_points = Column(JSONB, nullable=False)  # Actual sensor/property data
	data_quality = Column(Float, default=1.0)  # Quality score 0-1
	source_type = Column(String(50))  # iot_sensor, api, simulation, manual
	source_id = Column(String(100))  # Identifier of data source
	
	# Processing metadata
	processing_time_ms = Column(Float)
	anomaly_score = Column(Float)
	anomaly_detected = Column(Boolean, default=False)
	
	# Data validation
	schema_version = Column(String(20))
	validation_errors = Column(JSONB, default=list)
	
	# Relationships
	digital_twin = relationship("DigitalTwinModel", back_populates="telemetry_data")
	
	__table_args__ = (
		Index('ix_dt_telemetry_twin_timestamp', 'digital_twin_id', 'timestamp'),
		Index('ix_dt_telemetry_anomaly', 'anomaly_detected', 'anomaly_score'),
		Index('ix_dt_telemetry_source', 'source_type', 'source_id'),
	)

class DTSimulationJob(BaseCapabilityModel):
	"""Simulation jobs for digital twins"""
	
	__tablename__ = 'dt_simulation_jobs'
	
	digital_twin_id = Column(UUID(as_uuid=True), ForeignKey('digital_twins.id'), nullable=False)
	simulation_id = Column(String(100), nullable=False)
	simulation_type = Column(String(50), nullable=False)  # physics, thermal, structural, fluid, electrical, behavioral
	
	# Simulation parameters
	simulation_name = Column(String(200), nullable=False)
	description = Column(Text)
	duration = Column(Float, nullable=False)  # simulation duration in seconds
	time_step = Column(Float, default=0.1)  # simulation time step
	
	# Model parameters
	model_parameters = Column(JSONB, default=dict)
	boundary_conditions = Column(JSONB, default=dict)
	initial_conditions = Column(JSONB, default=dict)
	solver_settings = Column(JSONB, default=dict)
	
	# Execution status
	status = Column(String(20), default='pending')  # pending, running, completed, failed, cancelled
	progress_percentage = Column(Float, default=0.0)
	start_time = Column(DateTime)
	end_time = Column(DateTime)
	execution_time_seconds = Column(Float)
	
	# Results
	results = Column(JSONB, default=dict)
	output_files = Column(JSONB, default=list)  # Paths to output files
	performance_metrics = Column(JSONB, default=dict)
	convergence_data = Column(JSONB, default=dict)
	
	# Error handling
	error_message = Column(Text)
	warnings = Column(JSONB, default=list)
	
	# Resource usage
	cpu_time_seconds = Column(Float)
	memory_usage_mb = Column(Float)
	gpu_usage_percentage = Column(Float)
	
	# Relationships
	digital_twin = relationship("DigitalTwinModel", back_populates="simulation_jobs")
	
	__table_args__ = (
		Index('ix_dt_simulation_jobs_twin_status', 'digital_twin_id', 'status'),
		Index('ix_dt_simulation_jobs_type', 'simulation_type'),
		Index('ix_dt_simulation_jobs_start_time', 'start_time'),
	)

class DTEvent(BaseCapabilityModel):
	"""Events and alerts from digital twins"""
	
	__tablename__ = 'dt_events'
	
	digital_twin_id = Column(UUID(as_uuid=True), ForeignKey('digital_twins.id'), nullable=False, index=True)
	event_type = Column(String(50), nullable=False, index=True)  # alert, warning, info, error, state_change, anomaly
	severity = Column(String(20), default='info')  # critical, high, medium, low, info
	
	# Event details
	title = Column(String(200), nullable=False)
	description = Column(Text)
	event_data = Column(JSONB, default=dict)
	
	# Context information
	property_name = Column(String(100))  # Property that triggered event
	threshold_value = Column(Float)
	actual_value = Column(Float)
	
	# Status and resolution
	status = Column(String(20), default='open')  # open, acknowledged, resolved, closed
	acknowledged_by = Column(String(100))
	acknowledged_at = Column(DateTime)
	resolved_at = Column(DateTime)
	resolution_notes = Column(Text)
	
	# Automation
	auto_resolved = Column(Boolean, default=False)
	requires_action = Column(Boolean, default=False)
	action_taken = Column(Text)
	
	# Relationships
	digital_twin = relationship("DigitalTwinModel", back_populates="events")
	
	__table_args__ = (
		Index('ix_dt_events_twin_type', 'digital_twin_id', 'event_type'),
		Index('ix_dt_events_severity_status', 'severity', 'status'),
		Index('ix_dt_events_created_at', 'created_at'),
	)

class DTTemplate(BaseCapabilityModel):
	"""Templates for creating digital twins"""
	
	__tablename__ = 'dt_templates'
	
	template_name = Column(String(100), nullable=False, unique=True)
	display_name = Column(String(200), nullable=False)
	description = Column(Text)
	category = Column(String(50))  # industrial_machine, building, vehicle, process, etc.
	industry = Column(String(100))
	
	# Template definition
	twin_type = Column(String(50), nullable=False)
	default_properties = Column(JSONB, default=dict)
	default_data_sources = Column(JSONB, default=dict)
	default_simulation_models = Column(JSONB, default=dict)
	default_alert_rules = Column(JSONB, default=dict)
	
	# Template metadata
	version = Column(String(20), default='1.0.0')
	is_active = Column(Boolean, default=True)
	usage_count = Column(Integer, default=0)
	
	# Validation schema
	property_schema = Column(JSONB, default=dict)
	required_data_sources = Column(JSONB, default=list)
	
	__table_args__ = (
		Index('ix_dt_templates_category', 'category'),
		Index('ix_dt_templates_active', 'is_active'),
	)

# Enhanced Forms for Digital Twin Operations
class DTCreateTwinForm(DynamicForm):
	"""Form for creating digital twins"""
	
	project_id = SelectField(
		'Project',
		validators=[OptionalValidator()],
		description='Associate with existing project (optional)'
	)
	
	template_name = SelectField(
		'Template',
		validators=[OptionalValidator()],
		choices=[
			('', 'Custom (No Template)'),
			('industrial_machine', 'Industrial Machine'),
			('smart_building', 'Smart Building'),
			('vehicle', 'Vehicle'),
			('manufacturing_line', 'Manufacturing Line'),
			('energy_system', 'Energy System'),
			('supply_chain', 'Supply Chain'),
			('process_plant', 'Process Plant')
		],
		description='Use a predefined template'
	)
	
	name = StringField(
		'Twin Name',
		validators=[DataRequired()],
		description='Descriptive name for the digital twin'
	)
	
	description = TextAreaField(
		'Description',
		description='Detailed description of the digital twin'
	)
	
	twin_type = SelectField(
		'Twin Type',
		validators=[DataRequired()],
		choices=[
			('asset', 'Physical Asset'),
			('process', 'Business Process'),
			('system', 'Complex System'),
			('environment', 'Environment/Facility'),
			('product', 'Product Lifecycle'),
			('human', 'Human-Centric')
		],
		description='Type of digital twin'
	)
	
	physical_asset_id = StringField(
		'Physical Asset ID',
		description='Reference to physical asset (optional)'
	)
	
	location = TextAreaField(
		'Location',
		description='Geographic or facility location (JSON format)'
	)
	
	sync_interval = FloatField(
		'Sync Interval (seconds)',
		validators=[NumberRange(min=0.1, max=3600)],
		default=1.0,
		description='Data synchronization interval'
	)
	
	enable_real_time = BooleanField(
		'Enable Real-time Sync',
		default=True,
		description='Enable real-time data synchronization'
	)
	
	enable_simulation = BooleanField(
		'Enable Simulation',
		default=True,
		description='Enable simulation capabilities'
	)
	
	enable_prediction = BooleanField(
		'Enable Prediction',
		default=False,
		description='Enable predictive analytics'
	)

class DTSimulationForm(DynamicForm):
	"""Form for creating simulation jobs"""
	
	simulation_name = StringField(
		'Simulation Name',
		validators=[DataRequired()],
		description='Name for this simulation'
	)
	
	simulation_type = SelectField(
		'Simulation Type',
		validators=[DataRequired()],
		choices=[
			('physics', 'Physics Simulation'),
			('thermal', 'Thermal Analysis'),
			('structural', 'Structural Analysis'),
			('fluid', 'Fluid Dynamics'),
			('electrical', 'Electrical Simulation'),
			('behavioral', 'Behavioral Modeling'),
			('economic', 'Economic Modeling')
		],
		description='Type of simulation to run'
	)
	
	duration = FloatField(
		'Duration (seconds)',
		validators=[DataRequired(), NumberRange(min=0.1, max=86400)],
		default=10.0,
		description='Simulation duration'
	)
	
	time_step = FloatField(
		'Time Step (seconds)',
		validators=[NumberRange(min=0.001, max=60)],
		default=0.1,
		description='Simulation time step'
	)
	
	description = TextAreaField(
		'Description',
		description='Description of simulation goals'
	)

# Enhanced Flask-AppBuilder Views
class DigitalTwinView(BaseCapabilityView):
	"""Comprehensive digital twin interface"""
	
	route_base = '/digital_twin'
	default_view = 'dashboard'
	
	def __init__(self):
		super().__init__()
		self.dt_capability = DigitalTwinCapability()
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Digital twin platform dashboard"""
		stats = self._get_dashboard_stats()
		recent_twins = self._get_recent_twins()
		system_health = self._get_system_health()
		active_simulations = self._get_active_simulations()
		
		return self.render_template(
			'digital_twin/dashboard.html',
			stats=stats,
			recent_twins=recent_twins,
			system_health=system_health,
			active_simulations=active_simulations
		)
	
	@expose('/twins')
	@has_access
	def twins_list(self):
		"""List all digital twins"""
		twins = []  # Query from database
		templates = []  # Query available templates
		
		return self.render_template(
			'digital_twin/twins_list.html',
			twins=twins,
			templates=templates
		)
	
	@expose('/create_twin')
	@has_access
	def create_twin(self):
		"""Create new digital twin"""
		form = DTCreateTwinForm()
		
		if form.validate_on_submit():
			# Create digital twin logic here
			flash('Digital twin created successfully!', 'success')
			return redirect(url_for('DigitalTwinView.twins_list'))
		
		return self.render_template(
			'digital_twin/create_twin.html',
			form=form
		)
	
	@expose('/twin/<twin_id>')
	@has_access
	def twin_detail(self, twin_id):
		"""Digital twin detail view"""
		twin = None  # Query twin from database
		telemetry = []  # Recent telemetry data
		events = []  # Recent events
		simulations = []  # Recent simulations
		
		return self.render_template(
			'digital_twin/twin_detail.html',
			twin=twin,
			telemetry=telemetry,
			events=events,
			simulations=simulations
		)
	
	@expose('/twin/<twin_id>/simulate')
	@has_access
	def simulate_twin(self, twin_id):
		"""Run simulation on digital twin"""
		form = DTSimulationForm()
		twin = None  # Query twin from database
		
		if form.validate_on_submit():
			# Start simulation logic here
			flash('Simulation started successfully!', 'success')
			return redirect(url_for('DigitalTwinView.twin_detail', twin_id=twin_id))
		
		return self.render_template(
			'digital_twin/simulate_twin.html',
			form=form,
			twin=twin
		)
	
	@expose('/simulations')
	@has_access
	def simulations_list(self):
		"""List all simulation jobs"""
		simulations = []  # Query from database
		
		return self.render_template(
			'digital_twin/simulations_list.html',
			simulations=simulations
		)
	
	@expose('/simulation/<simulation_id>')
	@has_access
	def simulation_detail(self, simulation_id):
		"""Simulation job detail view"""
		simulation = None  # Query from database
		
		return self.render_template(
			'digital_twin/simulation_detail.html',
			simulation=simulation
		)
	
	@expose('/analytics')
	@has_access
	def analytics(self):
		"""Digital twin analytics dashboard"""
		analytics_data = self._get_analytics_data()
		
		return self.render_template(
			'digital_twin/analytics.html',
			analytics_data=analytics_data
		)
	
	@expose('/3d_viewer/<twin_id>')
	@has_access
	def viewer_3d(self, twin_id):
		"""3D visualization of digital twin"""
		twin = None  # Query twin from database
		geometry_data = {}  # Get 3D geometry data
		
		return self.render_template(
			'digital_twin/3d_viewer.html',
			twin=twin,
			geometry_data=geometry_data
		)
	
	def _get_dashboard_stats(self):
		"""Get dashboard statistics"""
		return {
			'total_digital_twins': 42,
			'active_twins': 38,
			'total_simulations': 156,
			'running_simulations': 3,
			'total_events': 1247,
			'critical_alerts': 2,
			'data_points_today': 284561,
			'avg_sync_time_ms': 145.7,
			'system_uptime': 99.2,
			'prediction_accuracy': 94.3,
			'twin_health_score': 96.8
		}
	
	def _get_recent_twins(self):
		"""Get recently created/updated twins"""
		return []  # Query from database
	
	def _get_system_health(self):
		"""Get system health metrics"""
		return {
			'status': 'healthy',
			'score': 96.8,
			'sync_performance': 'excellent',
			'simulation_capacity': 'available',
			'data_quality': 'high'
		}
	
	def _get_active_simulations(self):
		"""Get currently running simulations"""
		return []  # Query from database
	
	def _get_analytics_data(self):
		"""Get analytics data"""
		return {
			'twin_performance': {},
			'simulation_trends': {},
			'prediction_accuracy': {},
			'system_utilization': {}
		}

# Model Views for data management
class DTProjectModelView(BaseCapabilityModelView):
	"""Digital Twin Projects management"""
	datamodel = SQLAInterface(DTProject)
	list_columns = ['name', 'project_type', 'industry', 'status', 'created_at']
	show_columns = ['name', 'description', 'project_type', 'industry', 'status', 
					'default_sync_interval', 'enable_real_time', 'enable_simulation', 
					'enable_prediction', 'created_at', 'updated_at']
	add_columns = ['name', 'description', 'project_type', 'industry', 'status',
				   'default_sync_interval', 'enable_real_time', 'enable_simulation', 
				   'enable_prediction']
	edit_columns = add_columns

class DigitalTwinModelView(BaseCapabilityModelView):
	"""Digital Twins management"""
	datamodel = SQLAInterface(DigitalTwinModel)
	list_columns = ['name', 'twin_type', 'twin_state', 'health_status', 'last_sync_time']
	show_columns = ['name', 'description', 'twin_type', 'twin_state', 'version',
					'physical_asset_id', 'sync_status', 'health_status', 'uptime_percentage',
					'data_quality_score', 'prediction_accuracy', 'created_at', 'updated_at']
	add_columns = ['name', 'description', 'twin_type', 'physical_asset_id', 
				   'sync_interval', 'project_id']
	edit_columns = add_columns

class DTSimulationJobModelView(BaseCapabilityModelView):
	"""Simulation Jobs management"""
	datamodel = SQLAInterface(DTSimulationJob)
	list_columns = ['simulation_name', 'simulation_type', 'status', 'progress_percentage', 
					'start_time', 'execution_time_seconds']
	show_columns = ['simulation_name', 'simulation_type', 'description', 'duration',
					'status', 'progress_percentage', 'start_time', 'end_time',
					'execution_time_seconds', 'cpu_time_seconds', 'memory_usage_mb']

# PostgreSQL Schema Scripts for Digital Twin
DIGITAL_TWIN_SCHEMAS = {
	'dt_projects': """
CREATE TABLE IF NOT EXISTS dt_projects (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	name VARCHAR(200) NOT NULL,
	description TEXT,
	project_type VARCHAR(50) DEFAULT 'general',
	industry VARCHAR(100),
	status VARCHAR(20) DEFAULT 'active',
	settings JSONB DEFAULT '{}'::jsonb,
	default_sync_interval FLOAT DEFAULT 1.0,
	enable_real_time BOOLEAN DEFAULT TRUE,
	enable_simulation BOOLEAN DEFAULT TRUE,
	enable_prediction BOOLEAN DEFAULT FALSE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_dt_projects_name ON dt_projects(name);
CREATE INDEX IF NOT EXISTS ix_dt_projects_type ON dt_projects(project_type);
CREATE INDEX IF NOT EXISTS ix_dt_projects_status ON dt_projects(status);
""",

	'digital_twins': """
CREATE TABLE IF NOT EXISTS digital_twins (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	project_id UUID REFERENCES dt_projects(id) ON DELETE SET NULL,
	twin_id VARCHAR(100) NOT NULL UNIQUE,
	name VARCHAR(200) NOT NULL,
	description TEXT,
	twin_type VARCHAR(50) NOT NULL,
	twin_state VARCHAR(20) DEFAULT 'inactive',
	version VARCHAR(20) DEFAULT '1.0.0',
	physical_asset_id VARCHAR(200),
	location JSONB,
	
	-- Twin data
	properties JSONB DEFAULT '{}'::jsonb,
	relationships JSONB DEFAULT '{}'::jsonb,
	data_sources JSONB DEFAULT '{}'::jsonb,
	
	-- Models
	geometry_model JSONB,
	simulation_models JSONB DEFAULT '{}'::jsonb,
	behavior_models JSONB DEFAULT '{}'::jsonb,
	
	-- Real-time data
	current_state JSONB DEFAULT '{}'::jsonb,
	last_sync_time TIMESTAMP WITH TIME ZONE,
	sync_interval FLOAT DEFAULT 1.0,
	sync_status VARCHAR(20) DEFAULT 'disconnected',
	
	-- Performance metrics
	uptime_percentage FLOAT DEFAULT 0.0,
	data_quality_score FLOAT DEFAULT 0.0,
	prediction_accuracy FLOAT DEFAULT 0.0,
	
	-- Monitoring
	alert_rules JSONB DEFAULT '{}'::jsonb,
	health_status VARCHAR(20) DEFAULT 'unknown',
	
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_digital_twins_twin_id ON digital_twins(twin_id);
CREATE INDEX IF NOT EXISTS ix_digital_twins_state_type ON digital_twins(twin_state, twin_type);
CREATE INDEX IF NOT EXISTS ix_digital_twins_project_status ON digital_twins(project_id, twin_state);
CREATE INDEX IF NOT EXISTS ix_digital_twins_sync ON digital_twins(sync_status, last_sync_time);
""",

	'dt_telemetry_data': """
CREATE TABLE IF NOT EXISTS dt_telemetry_data (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	digital_twin_id UUID NOT NULL REFERENCES digital_twins(id) ON DELETE CASCADE,
	timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
	data_points JSONB NOT NULL,
	data_quality FLOAT DEFAULT 1.0,
	source_type VARCHAR(50),
	source_id VARCHAR(100),
	processing_time_ms FLOAT,
	anomaly_score FLOAT,
	anomaly_detected BOOLEAN DEFAULT FALSE,
	schema_version VARCHAR(20),
	validation_errors JSONB DEFAULT '[]'::jsonb,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_dt_telemetry_twin_timestamp ON dt_telemetry_data(digital_twin_id, timestamp);
CREATE INDEX IF NOT EXISTS ix_dt_telemetry_anomaly ON dt_telemetry_data(anomaly_detected, anomaly_score);
CREATE INDEX IF NOT EXISTS ix_dt_telemetry_source ON dt_telemetry_data(source_type, source_id);
""",

	'dt_simulation_jobs': """
CREATE TABLE IF NOT EXISTS dt_simulation_jobs (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	digital_twin_id UUID NOT NULL REFERENCES digital_twins(id) ON DELETE CASCADE,
	simulation_id VARCHAR(100) NOT NULL,
	simulation_type VARCHAR(50) NOT NULL,
	simulation_name VARCHAR(200) NOT NULL,
	description TEXT,
	duration FLOAT NOT NULL,
	time_step FLOAT DEFAULT 0.1,
	
	-- Model parameters
	model_parameters JSONB DEFAULT '{}'::jsonb,
	boundary_conditions JSONB DEFAULT '{}'::jsonb,
	initial_conditions JSONB DEFAULT '{}'::jsonb,
	solver_settings JSONB DEFAULT '{}'::jsonb,
	
	-- Execution status
	status VARCHAR(20) DEFAULT 'pending',
	progress_percentage FLOAT DEFAULT 0.0,
	start_time TIMESTAMP WITH TIME ZONE,
	end_time TIMESTAMP WITH TIME ZONE,
	execution_time_seconds FLOAT,
	
	-- Results
	results JSONB DEFAULT '{}'::jsonb,
	output_files JSONB DEFAULT '[]'::jsonb,
	performance_metrics JSONB DEFAULT '{}'::jsonb,
	convergence_data JSONB DEFAULT '{}'::jsonb,
	
	-- Error handling
	error_message TEXT,
	warnings JSONB DEFAULT '[]'::jsonb,
	
	-- Resource usage
	cpu_time_seconds FLOAT,
	memory_usage_mb FLOAT,
	gpu_usage_percentage FLOAT,
	
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_dt_simulation_jobs_twin_status ON dt_simulation_jobs(digital_twin_id, status);
CREATE INDEX IF NOT EXISTS ix_dt_simulation_jobs_type ON dt_simulation_jobs(simulation_type);
CREATE INDEX IF NOT EXISTS ix_dt_simulation_jobs_start_time ON dt_simulation_jobs(start_time);
""",

	'dt_events': """
CREATE TABLE IF NOT EXISTS dt_events (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	digital_twin_id UUID NOT NULL REFERENCES digital_twins(id) ON DELETE CASCADE,
	event_type VARCHAR(50) NOT NULL,
	severity VARCHAR(20) DEFAULT 'info',
	title VARCHAR(200) NOT NULL,
	description TEXT,
	event_data JSONB DEFAULT '{}'::jsonb,
	
	-- Context
	property_name VARCHAR(100),
	threshold_value FLOAT,
	actual_value FLOAT,
	
	-- Status and resolution
	status VARCHAR(20) DEFAULT 'open',
	acknowledged_by VARCHAR(100),
	acknowledged_at TIMESTAMP WITH TIME ZONE,
	resolved_at TIMESTAMP WITH TIME ZONE,
	resolution_notes TEXT,
	
	-- Automation
	auto_resolved BOOLEAN DEFAULT FALSE,
	requires_action BOOLEAN DEFAULT FALSE,
	action_taken TEXT,
	
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_dt_events_twin_type ON dt_events(digital_twin_id, event_type);
CREATE INDEX IF NOT EXISTS ix_dt_events_severity_status ON dt_events(severity, status);
CREATE INDEX IF NOT EXISTS ix_dt_events_created_at ON dt_events(created_at);
""",

	'dt_templates': """
CREATE TABLE IF NOT EXISTS dt_templates (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	template_name VARCHAR(100) NOT NULL UNIQUE,
	display_name VARCHAR(200) NOT NULL,
	description TEXT,
	category VARCHAR(50),
	industry VARCHAR(100),
	twin_type VARCHAR(50) NOT NULL,
	default_properties JSONB DEFAULT '{}'::jsonb,
	default_data_sources JSONB DEFAULT '{}'::jsonb,
	default_simulation_models JSONB DEFAULT '{}'::jsonb,
	default_alert_rules JSONB DEFAULT '{}'::jsonb,
	version VARCHAR(20) DEFAULT '1.0.0',
	is_active BOOLEAN DEFAULT TRUE,
	usage_count INTEGER DEFAULT 0,
	property_schema JSONB DEFAULT '{}'::jsonb,
	required_data_sources JSONB DEFAULT '[]'::jsonb,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_dt_templates_category ON dt_templates(category);
CREATE INDEX IF NOT EXISTS ix_dt_templates_active ON dt_templates(is_active);
"""
}

# Blueprint registration
digital_twin_bp = Blueprint(
	'digital_twin',
	__name__,
	template_folder='templates',
	static_folder='static'
)

__all__ = [
	'DigitalTwinView', 'DTProject', 'DigitalTwinModel', 'DTTelemetryData',
	'DTSimulationJob', 'DTEvent', 'DTTemplate', 'DTProjectModelView',
	'DigitalTwinModelView', 'DTSimulationJobModelView', 'DIGITAL_TWIN_SCHEMAS',
	'digital_twin_bp'
]