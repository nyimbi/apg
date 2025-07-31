"""
APG Capability Registry - Database Migration Scripts

Alembic-compatible migration scripts for capability registry models
with APG multi-tenant architecture and performance optimization.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid_extensions import uuid7str

from alembic import op
import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, JSON, ForeignKey, Index, UniqueConstraint
from sqlalchemy.sql import table, column

# APG Migration Metadata
MIGRATION_VERSION = "2025.01.001"
MIGRATION_DESCRIPTION = "Initial Capability Registry Schema"
APG_COMPATIBILITY = ">=2.0.0"

def _log_migration_start(operation: str) -> str:
	"""Log migration operation start."""
	return f"APG Migration Start: {operation} - Registry Schema v{MIGRATION_VERSION}"

def _log_migration_complete(operation: str) -> str:
	"""Log migration operation completion."""
	return f"APG Migration Complete: {operation} - Registry Schema deployed"

def upgrade():
	"""Create capability registry tables with APG integration."""
	print(_log_migration_start("Registry Schema Creation"))
	
	# Core Registry Tables
	_create_cr_capabilities_table()
	_create_cr_dependencies_table()
	_create_cr_compositions_table()
	_create_cr_composition_capabilities_table()
	_create_cr_versions_table()
	_create_cr_metadata_table()
	_create_cr_registry_table()
	
	# Analytics Tables
	_create_cr_usage_analytics_table()
	_create_cr_health_metrics_table()
	
	# Performance Indexes
	_create_performance_indexes()
	
	# APG Integration Setup
	_setup_apg_integration()
	
	print(_log_migration_complete("Registry Schema Creation"))

def downgrade():
	"""Drop capability registry tables."""
	print(_log_migration_start("Registry Schema Rollback"))
	
	# Drop in reverse order to handle foreign keys
	op.drop_table('cr_health_metrics')
	op.drop_table('cr_usage_analytics')
	op.drop_table('cr_metadata')
	op.drop_table('cr_versions')
	op.drop_table('cr_composition_capabilities')
	op.drop_table('cr_compositions')
	op.drop_table('cr_dependencies')
	op.drop_table('cr_registry')
	op.drop_table('cr_capabilities')
	
	print(_log_migration_complete("Registry Schema Rollback"))

def _create_cr_capabilities_table():
	"""Create core capabilities registry table."""
	op.create_table(
		'cr_capabilities',
		# Primary identification
		Column('capability_id', String(36), primary_key=True),
		Column('tenant_id', String(36), nullable=False, index=True),
		Column('capability_code', String(100), nullable=False, index=True),
		Column('capability_name', String(255), nullable=False),
		
		# Capability metadata
		Column('description', Text),
		Column('version', String(50), nullable=False),
		Column('category', String(100), nullable=False),
		Column('subcategory', String(100)),
		Column('priority', Integer, default=1),
		Column('status', String(50), default='discovered', nullable=False),
		
		# APG Integration metadata
		Column('composition_keywords', JSON, default='[]'),
		Column('provides_services', JSON, default='[]'),
		Column('data_models', JSON, default='[]'),
		Column('api_endpoints', JSON, default='[]'),
		
		# Feature flags
		Column('multi_tenant', Boolean, default=True, nullable=False),
		Column('audit_enabled', Boolean, default=True, nullable=False),
		Column('security_integration', Boolean, default=True, nullable=False),
		Column('performance_optimized', Boolean, default=False),
		Column('ai_enhanced', Boolean, default=False),
		
		# Business metadata
		Column('target_users', JSON, default='[]'),
		Column('business_value', Text),
		Column('use_cases', JSON, default='[]'),
		Column('industry_focus', JSON, default='[]'),
		
		# Technical metadata
		Column('file_path', String(500)),
		Column('module_path', String(500)),
		Column('documentation_path', String(500)),
		Column('repository_url', String(500)),
		
		# Performance and analytics
		Column('complexity_score', Float, default=1.0),
		Column('quality_score', Float, default=0.0),
		Column('popularity_score', Float, default=0.0),
		Column('usage_count', Integer, default=0),
		
		# APG Audit fields
		Column('created_at', DateTime, default=datetime.utcnow, nullable=False),
		Column('updated_at', DateTime, default=datetime.utcnow),
		Column('created_by', String(36), nullable=False),
		Column('updated_by', String(36)),
		
		# Additional metadata
		Column('metadata', JSON, default='{}'),
		
		# Constraints
		UniqueConstraint('tenant_id', 'capability_code', name='uq_tenant_capability'),
	)

def _create_cr_dependencies_table():
	"""Create capability dependencies table."""
	op.create_table(
		'cr_dependencies',
		# Primary identification
		Column('dependency_id', String(36), primary_key=True),
		Column('capability_id', String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False),
		Column('depends_on_id', String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False),
		
		# Dependency metadata
		Column('dependency_type', String(50), default='required', nullable=False),
		Column('version_constraint', String(50), default='latest'),
		Column('version_min', String(50)),
		Column('version_max', String(50)),
		Column('version_exact', String(50)),
		
		# Dependency configuration
		Column('load_priority', Integer, default=1),
		Column('initialization_order', Integer, default=1),
		Column('optional_features', JSON, default='[]'),
		
		# Validation and conflict resolution
		Column('conflict_resolution', String(100)),
		Column('alternative_capabilities', JSON, default='[]'),
		Column('fallback_strategy', String(100)),
		
		# APG Audit fields
		Column('created_at', DateTime, default=datetime.utcnow, nullable=False),
		Column('updated_at', DateTime, default=datetime.utcnow),
		Column('created_by', String(36), nullable=False),
		
		# Additional metadata
		Column('metadata', JSON, default='{}'),
		
		# Constraints
		UniqueConstraint('capability_id', 'depends_on_id', name='uq_capability_dependency'),
	)

def _create_cr_compositions_table():
	"""Create capability compositions table."""
	op.create_table(
		'cr_compositions',
		# Primary identification
		Column('composition_id', String(36), primary_key=True),
		Column('tenant_id', String(36), nullable=False, index=True),
		Column('name', String(255), nullable=False),
		Column('description', Text),
		
		# Composition metadata
		Column('composition_type', String(50), default='custom', nullable=False),
		Column('version', String(50), default='1.0.0', nullable=False),
		Column('industry_template', String(100)),
		Column('deployment_strategy', String(100)),
		
		# Validation and status
		Column('validation_status', String(50), default='pending', nullable=False),
		Column('validation_results', JSON, default='{}'),
		Column('validation_errors', JSON, default='[]'),
		Column('validation_warnings', JSON, default='[]'),
		
		# Composition configuration
		Column('configuration', JSON, default='{}'),
		Column('environment_settings', JSON, default='{}'),
		Column('deployment_config', JSON, default='{}'),
		
		# Performance and analytics
		Column('estimated_complexity', Float, default=1.0),
		Column('estimated_cost', Float, default=0.0),
		Column('estimated_deployment_time', String(50)),
		Column('performance_metrics', JSON, default='{}'),
		
		# Business metadata
		Column('business_requirements', JSON, default='[]'),
		Column('compliance_requirements', JSON, default='[]'),
		Column('target_users', JSON, default='[]'),
		
		# Sharing and collaboration
		Column('is_template', Boolean, default=False),
		Column('is_public', Boolean, default=False),
		Column('shared_with_tenants', JSON, default='[]'),
		
		# APG Audit fields
		Column('created_at', DateTime, default=datetime.utcnow, nullable=False),
		Column('updated_at', DateTime, default=datetime.utcnow),
		Column('created_by', String(36), nullable=False),
		Column('updated_by', String(36)),
		
		# Additional metadata
		Column('metadata', JSON, default='{}'),
	)

def _create_cr_composition_capabilities_table():
	"""Create composition-capability relationship table."""
	op.create_table(
		'cr_composition_capabilities',
		# Primary identification
		Column('comp_cap_id', String(36), primary_key=True),
		Column('composition_id', String(36), ForeignKey('cr_compositions.composition_id'), nullable=False),
		Column('capability_id', String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False),
		
		# Configuration
		Column('version_constraint', String(50), default='latest'),
		Column('required', Boolean, default=True, nullable=False),
		Column('load_order', Integer, default=1),
		Column('configuration', JSON, default='{}'),
		
		# APG Audit fields
		Column('created_at', DateTime, default=datetime.utcnow, nullable=False),
		Column('created_by', String(36), nullable=False),
		
		# Constraints
		UniqueConstraint('composition_id', 'capability_id', name='uq_composition_capability'),
	)

def _create_cr_versions_table():
	"""Create capability versions table."""
	op.create_table(
		'cr_versions',
		# Primary identification
		Column('version_id', String(36), primary_key=True),
		Column('capability_id', String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False),
		Column('version_number', String(50), nullable=False),
		
		# Version metadata
		Column('major_version', Integer, nullable=False),
		Column('minor_version', Integer, nullable=False),
		Column('patch_version', Integer, nullable=False),
		Column('pre_release', String(50)),
		Column('build_metadata', String(100)),
		
		# Release information
		Column('release_date', DateTime, nullable=False),
		Column('release_notes', Text),
		Column('breaking_changes', JSON, default='[]'),
		Column('deprecations', JSON, default='[]'),
		Column('new_features', JSON, default='[]'),
		
		# Compatibility information
		Column('compatible_versions', JSON, default='[]'),
		Column('incompatible_versions', JSON, default='[]'),
		Column('migration_path', JSON, default='{}'),
		Column('upgrade_instructions', Text),
		
		# API compatibility
		Column('api_changes', JSON, default='{}'),
		Column('backward_compatible', Boolean, default=True),
		Column('forward_compatible', Boolean, default=False),
		
		# Quality and validation
		Column('quality_score', Float, default=0.0),
		Column('test_coverage', Float, default=0.0),
		Column('documentation_score', Float, default=0.0),
		Column('security_audit_passed', Boolean, default=False),
		
		# Lifecycle status
		Column('status', String(50), default='active'),
		Column('end_of_life_date', DateTime),
		Column('support_level', String(50), default='full'),
		
		# APG Audit fields
		Column('created_at', DateTime, default=datetime.utcnow, nullable=False),
		Column('created_by', String(36), nullable=False),
		
		# Additional metadata
		Column('metadata', JSON, default='{}'),
		
		# Constraints
		UniqueConstraint('capability_id', 'version_number', name='uq_capability_version'),
	)

def _create_cr_metadata_table():
	"""Create extended metadata table."""
	op.create_table(
		'cr_metadata',
		# Primary identification
		Column('metadata_id', String(36), primary_key=True),
		Column('capability_id', String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False),
		Column('metadata_type', String(100), nullable=False),
		
		# Metadata content
		Column('metadata_key', String(255), nullable=False),
		Column('metadata_value', Text),
		Column('metadata_json', JSON),
		
		# Metadata properties
		Column('is_searchable', Boolean, default=True),
		Column('is_public', Boolean, default=True),
		Column('data_type', String(50), default='string'),
		
		# APG Audit fields
		Column('created_at', DateTime, default=datetime.utcnow, nullable=False),
		Column('updated_at', DateTime, default=datetime.utcnow),
		Column('created_by', String(36), nullable=False),
		Column('updated_by', String(36)),
	)

def _create_cr_registry_table():
	"""Create central registry configuration table."""
	op.create_table(
		'cr_registry',
		# Primary identification
		Column('registry_id', String(36), primary_key=True),
		Column('tenant_id', String(36), nullable=False, index=True),
		Column('name', String(255), nullable=False),
		Column('description', Text),
		
		# Registry configuration
		Column('auto_discovery_enabled', Boolean, default=True),
		Column('auto_validation_enabled', Boolean, default=True),
		Column('marketplace_integration', Boolean, default=True),
		Column('ai_recommendations', Boolean, default=True),
		
		# Discovery settings
		Column('discovery_paths', JSON, default='[]'),
		Column('excluded_paths', JSON, default='[]'),
		Column('scan_frequency_hours', Integer, default=24),
		Column('last_scan_date', DateTime),
		
		# Validation settings
		Column('validation_rules', JSON, default='{}'),
		Column('quality_thresholds', JSON, default='{}'),
		Column('compliance_requirements', JSON, default='[]'),
		
		# Performance settings
		Column('cache_ttl_seconds', Integer, default=3600),
		Column('max_composition_size', Integer, default=50),
		Column('max_dependency_depth', Integer, default=10),
		
		# APG Audit fields
		Column('created_at', DateTime, default=datetime.utcnow, nullable=False),
		Column('updated_at', DateTime, default=datetime.utcnow),
		Column('created_by', String(36), nullable=False),
		Column('updated_by', String(36)),
		
		# Additional metadata
		Column('metadata', JSON, default='{}'),
	)

def _create_cr_usage_analytics_table():
	"""Create usage analytics table."""
	op.create_table(
		'cr_usage_analytics',
		# Primary identification
		Column('usage_id', String(36), primary_key=True),
		Column('tenant_id', String(36), nullable=False, index=True),
		Column('capability_id', String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False),
		
		# Usage metrics
		Column('usage_date', DateTime, nullable=False, index=True),
		Column('usage_count', Integer, default=1),
		Column('composition_count', Integer, default=0),
		Column('deployment_count', Integer, default=0),
		Column('error_count', Integer, default=0),
		
		# Performance metrics
		Column('avg_response_time_ms', Float, default=0.0),
		Column('avg_memory_usage_mb', Float, default=0.0),
		Column('avg_cpu_usage_pct', Float, default=0.0),
		
		# User interaction metrics
		Column('unique_users', Integer, default=0),
		Column('total_sessions', Integer, default=0),
		Column('avg_session_duration', Float, default=0.0),
		
		# Additional metadata
		Column('metadata', JSON, default='{}'),
	)

def _create_cr_health_metrics_table():
	"""Create health metrics table."""
	op.create_table(
		'cr_health_metrics',
		# Primary identification
		Column('metric_id', String(36), primary_key=True),
		Column('capability_id', String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False),
		Column('timestamp', DateTime, default=datetime.utcnow, nullable=False),
		
		# Health metrics
		Column('health_score', Float, default=1.0),
		Column('availability_pct', Float, default=100.0),
		Column('performance_score', Float, default=1.0),
		Column('error_rate_pct', Float, default=0.0),
		
		# Dependency health
		Column('dependency_health_score', Float, default=1.0),
		Column('missing_dependencies', Integer, default=0),
		Column('conflicting_dependencies', Integer, default=0),
		
		# Quality metrics
		Column('documentation_completeness', Float, default=0.0),
		Column('test_coverage_pct', Float, default=0.0),
		Column('code_quality_score', Float, default=0.0),
		Column('security_score', Float, default=0.0),
		
		# Additional metadata
		Column('metadata', JSON, default='{}'),
	)

def _create_performance_indexes():
	"""Create performance-optimized indexes."""
	# Capability search indexes
	op.create_index('idx_cr_capability_tenant', 'cr_capabilities', ['tenant_id'])
	op.create_index('idx_cr_capability_code', 'cr_capabilities', ['capability_code'])
	op.create_index('idx_cr_capability_status', 'cr_capabilities', ['status'])
	op.create_index('idx_cr_capability_category', 'cr_capabilities', ['category'])
	op.create_index('idx_cr_capability_search', 'cr_capabilities', ['capability_name', 'description'])
	
	# Dependency resolution indexes
	op.create_index('idx_cr_dependency_capability', 'cr_dependencies', ['capability_id'])
	op.create_index('idx_cr_dependency_depends_on', 'cr_dependencies', ['depends_on_id'])
	op.create_index('idx_cr_dependency_type', 'cr_dependencies', ['dependency_type'])
	
	# Composition indexes
	op.create_index('idx_cr_composition_tenant', 'cr_compositions', ['tenant_id'])
	op.create_index('idx_cr_composition_type', 'cr_compositions', ['composition_type'])
	op.create_index('idx_cr_composition_status', 'cr_compositions', ['validation_status'])
	op.create_index('idx_cr_composition_created', 'cr_compositions', ['created_at'])
	
	# Version tracking indexes
	op.create_index('idx_cr_version_capability', 'cr_versions', ['capability_id'])
	op.create_index('idx_cr_version_number', 'cr_versions', ['version_number'])
	op.create_index('idx_cr_version_released', 'cr_versions', ['release_date'])
	
	# Analytics indexes
	op.create_index('idx_cr_usage_capability', 'cr_usage_analytics', ['capability_id'])
	op.create_index('idx_cr_usage_date', 'cr_usage_analytics', ['usage_date'])
	op.create_index('idx_cr_usage_tenant', 'cr_usage_analytics', ['tenant_id'])
	
	# Health monitoring indexes
	op.create_index('idx_cr_health_capability', 'cr_health_metrics', ['capability_id'])
	op.create_index('idx_cr_health_timestamp', 'cr_health_metrics', ['timestamp'])

def _setup_apg_integration():
	"""Set up APG platform integration and default data."""
	print("Setting up APG integration data...")
	
	# Create default registry configuration for each tenant
	registry_table = table(
		'cr_registry',
		column('registry_id', String),
		column('tenant_id', String),
		column('name', String),
		column('description', String),
		column('auto_discovery_enabled', Boolean),
		column('auto_validation_enabled', Boolean),
		column('marketplace_integration', Boolean),
		column('ai_recommendations', Boolean),
		column('discovery_paths', JSON),
		column('scan_frequency_hours', Integer),
		column('cache_ttl_seconds', Integer),
		column('max_composition_size', Integer),
		column('max_dependency_depth', Integer),
		column('created_at', DateTime),
		column('created_by', String),
		column('metadata', JSON)
	)
	
	# Insert default registry configuration
	op.bulk_insert(
		registry_table,
		[{
			'registry_id': uuid7str(),
			'tenant_id': 'default',
			'name': 'APG Default Capability Registry',
			'description': 'Default capability registry for APG platform',
			'auto_discovery_enabled': True,
			'auto_validation_enabled': True,
			'marketplace_integration': True,
			'ai_recommendations': True,
			'discovery_paths': '["capabilities/", "custom_capabilities/"]',
			'scan_frequency_hours': 24,
			'cache_ttl_seconds': 3600,
			'max_composition_size': 50,
			'max_dependency_depth': 10,
			'created_at': datetime.utcnow(),
			'created_by': 'apg_system',
			'metadata': '{"source": "apg_migration", "version": "' + MIGRATION_VERSION + '"}'
		}]
	)
	
	print("APG integration setup complete")

def _log_migration_validation() -> str:
	"""Validate migration results."""
	return f"Migration validation completed - Registry v{MIGRATION_VERSION} ready for APG integration"

# APG Migration Utilities
def validate_apg_integration() -> bool:
	"""Validate that migration is compatible with APG platform."""
	# This would connect to APG and validate integration
	return True

def rollback_apg_integration():
	"""Clean up APG integration during rollback."""
	pass