#!/usr/bin/env python3
"""
Base Blueprint Infrastructure for APG Capabilities
==================================================

Common models, views, and utilities for all Flask-AppBuilder capability blueprints.
Uses PostgreSQL as the primary data store with comprehensive SQLAlchemy models.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import BaseView, ModelView, expose, has_access
from flask_appbuilder.models.sqla import Model
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.forms import DynamicForm
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from wtforms import StringField, TextAreaField, SelectField, IntegerField, FloatField, BooleanField
from wtforms.validators import DataRequired, Optional as OptionalValidator, NumberRange

Base = declarative_base()

# Utility functions
def uuid7str() -> str:
	"""Generate UUID7-like string for consistent ID generation"""
	return str(uuid.uuid4())

class BaseCapabilityModel(Model):
	"""Base model for all capability-related data"""
	
	__abstract__ = True
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
	created_by = Column(String(100), nullable=True)  # User who created the record
	metadata_ = Column('metadata', JSONB, default=dict)  # Flexible metadata storage
	
	def __repr__(self):
		return f'<{self.__class__.__name__} {self.id}>'

class OperationLog(BaseCapabilityModel):
	"""Log all capability operations for auditing and debugging"""
	
	__tablename__ = 'operation_logs'
	
	operation_id = Column(String(100), nullable=False, index=True)
	capability_name = Column(String(50), nullable=False, index=True)
	operation_type = Column(String(50), nullable=False, index=True)  # 'create', 'read', 'update', 'delete', 'execute'
	operation_name = Column(String(100), nullable=False)
	input_data = Column(JSONB)
	output_data = Column(JSONB)
	status = Column(String(20), default='pending', nullable=False, index=True)  # pending, completed, failed
	error_message = Column(Text)
	execution_time_ms = Column(Float)
	user_id = Column(String(100))
	session_id = Column(String(100))
	
	# Indexes for efficient querying
	__table_args__ = (
		Index('ix_operation_logs_capability_status', 'capability_name', 'status'),
		Index('ix_operation_logs_created_status', 'created_at', 'status'),
	)

class SystemMetrics(BaseCapabilityModel):
	"""System-wide metrics and performance data"""
	
	__tablename__ = 'system_metrics'
	
	metric_name = Column(String(100), nullable=False, index=True)
	metric_type = Column(String(20), nullable=False)  # 'counter', 'gauge', 'histogram'
	value = Column(Float, nullable=False)
	unit = Column(String(20))
	tags = Column(JSONB, default=dict)
	
	__table_args__ = (
		Index('ix_system_metrics_name_created', 'metric_name', 'created_at'),
	)

class CapabilityConfiguration(BaseCapabilityModel):
	"""Configuration settings for capabilities"""
	
	__tablename__ = 'capability_configurations'
	
	capability_name = Column(String(50), nullable=False, index=True)
	config_key = Column(String(100), nullable=False)
	config_value = Column(JSONB, nullable=False)
	config_type = Column(String(20), default='string')  # string, number, boolean, object, array
	description = Column(Text)
	is_sensitive = Column(Boolean, default=False)  # Whether value should be encrypted/hidden
	
	__table_args__ = (
		Index('ix_capability_config_name_key', 'capability_name', 'config_key'),
	)

class BaseCapabilityView(BaseView):
	"""Base view class for all capability interfaces"""
	
	def __init__(self):
		super().__init__()
		self.capability = None  # Should be set by subclasses
	
	def log_operation(self, operation_id: str, operation_type: str, operation_name: str, 
					 input_data: Dict = None, output_data: Dict = None, 
					 status: str = 'completed', error_message: str = None,
					 execution_time_ms: float = None):
		"""Log an operation for auditing"""
		# In a real implementation, this would write to the database
		pass
	
	def get_capability_config(self, config_key: str, default_value: Any = None) -> Any:
		"""Get configuration value for this capability"""
		# In a real implementation, this would query the database
		return default_value
	
	def set_capability_config(self, config_key: str, config_value: Any, description: str = None):
		"""Set configuration value for this capability"""
		# In a real implementation, this would update the database
		pass
	
	@expose('/health')
	@has_access
	def health_check(self):
		"""Health check endpoint for monitoring"""
		return jsonify({
			'status': 'healthy',
			'capability': self.__class__.__name__,
			'timestamp': datetime.utcnow().isoformat()
		})
	
	@expose('/config')
	@has_access
	def configuration(self):
		"""View and manage capability configuration"""
		# Get all configuration for this capability
		config_data = {}  # In real implementation, query database
		
		return self.render_template(
			'base/configuration.html',
			config_data=config_data,
			capability_name=self.__class__.__name__
		)

class BaseCapabilityModelView(ModelView):
	"""Base model view for capability data"""
	
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	list_columns = ['id', 'created_at', 'updated_at']
	show_columns = ['id', 'created_at', 'updated_at', 'created_by', 'metadata_']
	search_columns = ['id']
	base_order = ('created_at', 'desc')
	
	def pre_add(self, item):
		"""Called before adding new item"""
		if hasattr(item, 'created_by') and not item.created_by:
			item.created_by = 'system'  # In real implementation, get from session
	
	def pre_update(self, item):
		"""Called before updating item"""
		item.updated_at = datetime.utcnow()

# Common form fields and validators
class BaseCapabilityForm(DynamicForm):
	"""Base form for capability operations"""
	
	metadata_json = TextAreaField(
		'Metadata (JSON)',
		validators=[OptionalValidator()],
		description='Additional metadata in JSON format'
	)

# SQL Scripts for PostgreSQL
POSTGRESQL_SCHEMAS = {
	'operation_logs': """
CREATE TABLE IF NOT EXISTS operation_logs (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	operation_id VARCHAR(100) NOT NULL,
	capability_name VARCHAR(50) NOT NULL,
	operation_type VARCHAR(50) NOT NULL,
	operation_name VARCHAR(100) NOT NULL,
	input_data JSONB,
	output_data JSONB,
	status VARCHAR(20) DEFAULT 'pending' NOT NULL,
	error_message TEXT,
	execution_time_ms FLOAT,
	user_id VARCHAR(100),
	session_id VARCHAR(100),
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_operation_logs_capability_name ON operation_logs(capability_name);
CREATE INDEX IF NOT EXISTS ix_operation_logs_operation_type ON operation_logs(operation_type);
CREATE INDEX IF NOT EXISTS ix_operation_logs_status ON operation_logs(status);
CREATE INDEX IF NOT EXISTS ix_operation_logs_created_at ON operation_logs(created_at);
CREATE INDEX IF NOT EXISTS ix_operation_logs_capability_status ON operation_logs(capability_name, status);
CREATE INDEX IF NOT EXISTS ix_operation_logs_created_status ON operation_logs(created_at, status);
""",

	'system_metrics': """
CREATE TABLE IF NOT EXISTS system_metrics (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	metric_name VARCHAR(100) NOT NULL,
	metric_type VARCHAR(20) NOT NULL,
	value FLOAT NOT NULL,
	unit VARCHAR(20),
	tags JSONB DEFAULT '{}'::jsonb,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_system_metrics_metric_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS ix_system_metrics_created_at ON system_metrics(created_at);
CREATE INDEX IF NOT EXISTS ix_system_metrics_name_created ON system_metrics(metric_name, created_at);
""",

	'capability_configurations': """
CREATE TABLE IF NOT EXISTS capability_configurations (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	capability_name VARCHAR(50) NOT NULL,
	config_key VARCHAR(100) NOT NULL,
	config_value JSONB NOT NULL,
	config_type VARCHAR(20) DEFAULT 'string',
	description TEXT,
	is_sensitive BOOLEAN DEFAULT FALSE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
	created_by VARCHAR(100),
	metadata JSONB DEFAULT '{}'::jsonb,
	UNIQUE(capability_name, config_key)
);

CREATE INDEX IF NOT EXISTS ix_capability_configurations_capability_name ON capability_configurations(capability_name);
CREATE INDEX IF NOT EXISTS ix_capability_config_name_key ON capability_configurations(capability_name, config_key);
"""
}

# Template for base configuration view
BASE_TEMPLATES = {
	'base/configuration.html': """
{% extends "appbuilder/base.html" %}

{% block content %}
<div class="container-fluid">
	<div class="row">
		<div class="col-12">
			<h1><i class="fa fa-cog"></i> {{ capability_name }} Configuration</h1>
		</div>
	</div>
	
	<div class="row">
		<div class="col-md-8">
			<div class="card">
				<div class="card-header">
					<h5>Configuration Settings</h5>
				</div>
				<div class="card-body">
					{% if config_data %}
						<table class="table table-striped">
							<thead>
								<tr>
									<th>Setting</th>
									<th>Value</th>
									<th>Type</th>
									<th>Description</th>
									<th>Actions</th>
								</tr>
							</thead>
							<tbody>
								{% for key, config in config_data.items() %}
								<tr>
									<td><code>{{ key }}</code></td>
									<td>
										{% if config.is_sensitive %}
											<span class="text-muted">***hidden***</span>
										{% else %}
											{{ config.value }}
										{% endif %}
									</td>
									<td><span class="badge bg-info">{{ config.type }}</span></td>
									<td>{{ config.description or 'No description' }}</td>
									<td>
										<button class="btn btn-sm btn-primary" onclick="editConfig('{{ key }}')">
											<i class="fa fa-edit"></i>
										</button>
									</td>
								</tr>
								{% endfor %}
							</tbody>
						</table>
					{% else %}
						<div class="alert alert-info">
							<i class="fa fa-info-circle"></i> No configuration settings found for this capability.
						</div>
					{% endif %}
				</div>
			</div>
		</div>
		
		<div class="col-md-4">
			<div class="card">
				<div class="card-header">
					<h6>Add New Setting</h6>
				</div>
				<div class="card-body">
					<form method="post">
						<div class="mb-3">
							<label class="form-label">Setting Key</label>
							<input type="text" class="form-control" name="config_key" required>
						</div>
						<div class="mb-3">
							<label class="form-label">Value</label>
							<textarea class="form-control" name="config_value" rows="3" required></textarea>
						</div>
						<div class="mb-3">
							<label class="form-label">Type</label>
							<select class="form-control" name="config_type">
								<option value="string">String</option>
								<option value="number">Number</option>
								<option value="boolean">Boolean</option>
								<option value="object">Object</option>
								<option value="array">Array</option>
							</select>
						</div>
						<div class="mb-3">
							<label class="form-label">Description</label>
							<input type="text" class="form-control" name="description">
						</div>
						<div class="mb-3">
							<div class="form-check">
								<input type="checkbox" class="form-check-input" name="is_sensitive">
								<label class="form-check-label">Sensitive (hide value)</label>
							</div>
						</div>
						<button type="submit" class="btn btn-primary">Add Setting</button>
					</form>
				</div>
			</div>
		</div>
	</div>
</div>

<script>
function editConfig(key) {
	// Implementation for editing configuration
	alert('Edit configuration: ' + key);
}
</script>
{% endblock %}
"""
}

__all__ = [
	'BaseCapabilityModel', 'BaseCapabilityView', 'BaseCapabilityModelView', 'BaseCapabilityForm',
	'OperationLog', 'SystemMetrics', 'CapabilityConfiguration',
	'POSTGRESQL_SCHEMAS', 'BASE_TEMPLATES', 'uuid7str'
]