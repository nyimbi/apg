#!/usr/bin/env python3
"""
Registry (regy) - APG Flask-AppBuilder Views
============================================

APG-compatible Flask-AppBuilder views with Pydantic v2 models for service registry
management, including real-time dashboards and intelligent analytics.

Author: APG Platform Team
Copyright: © 2025 Datacraft  
Website: www.datacraft.co.ke
"""

import asyncio
import flask
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from urllib.parse import unquote

from flask import request, has_request_context, redirect, url_for, flash, render_template, jsonify
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.actions import action
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import FormWidget, ShowWidget, ListWidget
from wtforms import StringField, SelectField, TextAreaField, BooleanField, IntegerField, FloatField
from wtforms.validators import DataRequired, Optional as OptionalValidator, Length, NumberRange

from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic import PositiveInt, NonNegativeFloat
from pydantic import AfterValidator
from typing_extensions import Annotated

from .service import ServiceRegistryService
from .models import (
	ServiceRegistration, ServiceInstance, ServiceDiscoveryQuery,
	ServiceHealthStatus, ServiceEvent, ServiceMetrics, ServiceStatus,
	ServiceType, HealthCheckType, CircuitBreakerState, LoadBalanceStrategy,
	ValidatedPort, ValidatedURL, ValidatedVersion
)

# APG Integration Imports
try:
	from ..auth.decorators import has_permission
	from ..moni.widgets import RealTimeChartWidget
	from ..rtc.websocket_manager import WebSocketManager
	APG_UI_INTEGRATION = True
except ImportError:
	# Fallback for development
	def has_permission(permission): return lambda f: f
	RealTimeChartWidget = None
	WebSocketManager = None
	APG_UI_INTEGRATION = False

# APG Model Configuration Standards - CLAUDE.md compliance
APG_MODEL_CONFIG = ConfigDict(
	extra='forbid',
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
	validate_default=True,
	use_enum_values=True
)

# Pydantic v2 Models for Views - Following APG patterns

class ServiceRegistrationForm(BaseModel):
	"""Pydantic model for service registration form data."""
	model_config = APG_MODEL_CONFIG
	
	name: str = Field(description="Service name", min_length=1, max_length=100)
	display_name: str = Field(description="Human-readable service name", min_length=1, max_length=150)  
	description: Optional[str] = Field(None, description="Service description", max_length=500)
	service_type: ServiceType = Field(description="Type of service")
	namespace: str = Field(default="default", description="Service namespace", max_length=50)
	environment: str = Field(description="Deployment environment", max_length=50)
	base_path: str = Field(default="/", description="Service base path", max_length=200)
	tags: List[str] = Field(default_factory=list, description="Service tags")
	discovery_enabled: bool = Field(default=True, description="Enable service discovery")
	load_balance_strategy: LoadBalanceStrategy = Field(default=LoadBalanceStrategy.ROUND_ROBIN)
	health_check_enabled: bool = Field(default=True, description="Enable health monitoring")
	circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker")
	predictive_scaling: bool = Field(default=False, description="AI-powered predictive scaling")
	intelligent_routing: bool = Field(default=False, description="ML-optimized routing")
	
	@field_validator('name')
	@classmethod
	def validate_name(cls, v: str) -> str:
		"""Validate service name format."""
		assert v.replace('-', '').replace('_', '').isalnum(), "Service name must be alphanumeric with hyphens/underscores"
		return v.lower()

class ServiceInstanceForm(BaseModel):
	"""Pydantic model for service instance form data."""
	model_config = APG_MODEL_CONFIG
	
	instance_name: str = Field(description="Instance identifier", min_length=1, max_length=100)
	host: str = Field(description="Instance host address", min_length=1, max_length=255)
	port: ValidatedPort = Field(description="Instance port")
	base_url: ValidatedURL = Field(description="Instance base URL")
	weight: int = Field(default=100, ge=0, le=1000, description="Load balancing weight")
	max_connections: Optional[int] = Field(None, ge=0, description="Maximum connections")
	environment: Optional[str] = Field(None, description="Deployment environment", max_length=50)
	deployment_version: Optional[ValidatedVersion] = Field(None, description="Deployment version")
	tags: List[str] = Field(default_factory=list, description="Instance tags")

class ServiceDiscoveryForm(BaseModel):
	"""Pydantic model for service discovery form."""
	model_config = APG_MODEL_CONFIG
	
	service_name: Optional[str] = Field(None, description="Service name filter", max_length=100)
	service_type: Optional[ServiceType] = Field(None, description="Service type filter")
	namespace: Optional[str] = Field(None, description="Namespace filter", max_length=50)
	environment: Optional[str] = Field(None, description="Environment filter", max_length=50)
	status: Optional[ServiceStatus] = Field(None, description="Service status filter")
	healthy_only: bool = Field(default=True, description="Return only healthy services")
	min_health_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum health score")
	intelligent_ranking: bool = Field(default=False, description="AI-powered result ranking")
	predictive_filtering: bool = Field(default=False, description="Predictive availability filtering")
	limit: int = Field(default=50, ge=1, le=1000, description="Maximum results")
	offset: int = Field(default=0, ge=0, description="Result offset")
	include_instances: bool = Field(default=True, description="Include instance details")
	include_health: bool = Field(default=True, description="Include health information")
	include_metrics: bool = Field(default=False, description="Include performance metrics")

class HealthCheckForm(BaseModel):
	"""Pydantic model for health check configuration."""
	model_config = APG_MODEL_CONFIG
	
	name: str = Field(description="Health check name", min_length=1, max_length=100)
	type: HealthCheckType = Field(description="Health check method")
	enabled: bool = Field(default=True, description="Health check enabled status")
	url: Optional[ValidatedURL] = Field(None, description="Health check URL")
	interval_seconds: PositiveInt = Field(default=30, description="Check interval in seconds")
	timeout_seconds: PositiveInt = Field(default=10, description="Check timeout in seconds")
	healthy_threshold: PositiveInt = Field(default=2, description="Consecutive successes for healthy")
	unhealthy_threshold: PositiveInt = Field(default=3, description="Consecutive failures for unhealthy")
	expected_response_codes: List[int] = Field(default_factory=lambda: [200], description="Expected HTTP codes")
	adaptive_intervals: bool = Field(default=False, description="AI-powered adaptive check intervals")
	anomaly_detection: bool = Field(default=False, description="ML anomaly detection")

class CircuitBreakerForm(BaseModel):
	"""Pydantic model for circuit breaker configuration."""
	model_config = APG_MODEL_CONFIG
	
	name: str = Field(description="Circuit breaker name", min_length=1, max_length=100)
	enabled: bool = Field(default=True, description="Circuit breaker enabled status")
	failure_threshold: PositiveInt = Field(default=5, description="Failure count to open circuit")
	success_threshold: PositiveInt = Field(default=3, description="Success count to close circuit")
	timeout_seconds: PositiveInt = Field(default=60, description="Timeout before half-open retry")
	failure_rate_threshold: float = Field(default=50.0, ge=0.0, le=100.0, description="Failure rate % threshold")
	minimum_request_threshold: PositiveInt = Field(default=10, description="Minimum requests before evaluation")
	rolling_window_seconds: PositiveInt = Field(default=60, description="Rolling window for statistics")
	adaptive_thresholds: bool = Field(default=False, description="AI-optimized threshold management")
	pattern_recognition: bool = Field(default=False, description="Failure pattern recognition")
	intelligent_recovery: bool = Field(default=False, description="ML-powered recovery strategies")

# Global registry service
registry_service: Optional[ServiceRegistryService] = None

def get_registry_service() -> ServiceRegistryService:
	"""Get registry service instance with APG tenant context."""
	global registry_service
	if not registry_service:
		# In production, this would get tenant from APG auth context.
		tenant_id = 'default'
		if has_request_context():
			tenant_id = request.args.get('tenant_id', tenant_id)
		else:
			try:
				tenant_id = flask.request.args.get('tenant_id', tenant_id)
			except RuntimeError:
				pass
		registry_service = ServiceRegistryService(tenant_id)
	return registry_service

async def ensure_service_initialized():
	"""Ensure registry service is initialized.""" 
	service = get_registry_service()
	if not service.initialized:
		await service.initialize()

# Main Service Registry Views

class ServiceRegistryView(BaseView):
	"""Main service registry management view."""

	route_base = "/serviceregistryview"
	default_view = "list"
	
	# APG Flask-AppBuilder Configuration
	list_title = "Service Registry"
	show_title = "Service Details" 
	add_title = "Register New Service"
	edit_title = "Update Service"
	
	# List View Configuration
	list_columns = [
		'name', 'display_name', 'service_type', 'namespace', 
		'environment', 'status', 'health_score', 'instances_count'
	]
	show_columns = [
		'name', 'display_name', 'description', 'service_type', 'namespace',
		'environment', 'base_path', 'status', 'current_version', 'tags',
		'discovery_enabled', 'load_balance_strategy', 'instances', 'versions'
	]
	search_columns = ['name', 'display_name', 'service_type', 'namespace', 'environment']
	edit_columns = [
		'display_name', 'description', 'service_type', 'namespace', 'environment',
		'base_path', 'discovery_enabled', 'load_balance_strategy', 'health_check_enabled',
		'circuit_breaker_enabled', 'predictive_scaling', 'intelligent_routing', 'tags'
	]
	add_columns = edit_columns + ['name']
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	@expose('/list/')
	@has_access
	@has_permission('registry:list_services')
	def list(self):
		"""List all registered services with real-time updates."""
		# Get services data
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			loop.run_until_complete(ensure_service_initialized())
			service = get_registry_service()
			
			# Build query for all services
			query = ServiceDiscoveryQuery(
				tenant_id=service.tenant_id,
				limit=1000,
				include_health=True,
				include_instances=True
			)
			result = loop.run_until_complete(service.discover_services(query))
			
			# Prepare data for template
			services_data = []
			for svc in result.services:
				health_status = loop.run_until_complete(service.get_service_health(svc.id))
				services_data.append({
					'id': svc.id,
					'name': svc.name,
					'display_name': svc.display_name,
					'service_type': svc.service_type,
					'namespace': svc.namespace,
					'environment': svc.environment,
					'status': svc.status,
					'health_score': health_status.health_score if health_status else 0.0,
					'instances_count': len(svc.instances),
					'created_at': svc.created_at.strftime('%Y-%m-%d %H:%M:%S'),
					'updated_at': svc.updated_at.strftime('%Y-%m-%d %H:%M:%S')
				})
			
			# Get summary statistics
			stats = loop.run_until_complete(service.get_registry_statistics())
			
		finally:
			loop.close()
		
		return self.render_template(
			'registry/service_list.html',
			services=services_data,
			statistics=stats['service_statistics'],
			performance=stats['performance_counters'],
			title=self.list_title
		)
	
	@expose('/show/<service_id>')
	@has_access
	@has_permission('registry:view_service')
	def show(self, service_id: str):
		"""Show detailed service information."""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			loop.run_until_complete(ensure_service_initialized())
			service = get_registry_service()
			
			if service_id not in service.services:
				flash('Service not found', 'error')
				return redirect(url_for('ServiceRegistryView.list'))
			
			service_obj = service.services[service_id]
			health_status = loop.run_until_complete(service.get_service_health(service_id))
			metrics = loop.run_until_complete(service.get_service_metrics(service_id, 24))
			
		finally:
			loop.close()
		
		return self.render_template(
			'registry/service_detail.html',
			service=service_obj,
			health_status=health_status,
			metrics=metrics,
			title=f"{self.show_title} - {service_obj.name}"
		)
	
	@expose('/add', methods=['GET', 'POST'])
	@has_access
	@has_permission('registry:register_service')
	def add(self):
		"""Register a new service."""
		if request.method == 'POST':
			try:
				# Validate form data with Pydantic
				form_data = request.form.to_dict()
				
				# Handle lists (tags)
				if 'tags' in form_data:
					form_data['tags'] = [tag.strip() for tag in form_data['tags'].split(',') if tag.strip()]
				
				# Convert boolean fields
				bool_fields = ['discovery_enabled', 'health_check_enabled', 'circuit_breaker_enabled', 
							  'predictive_scaling', 'intelligent_routing']
				for field in bool_fields:
					form_data[field] = form_data.get(field) == 'on'
				
				# Validate with Pydantic
				service_form = ServiceRegistrationForm(**form_data)
				
				# Register service
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					loop.run_until_complete(ensure_service_initialized())
					service = get_registry_service()
					
					# Get current user (in production, from APG auth)
					user_id = request.args.get('user_id', 'admin')
					
					registered_service = loop.run_until_complete(
						service.register_service(service_form.model_dump(), user_id)
					)
					
					flash(f'Service "{registered_service.name}" registered successfully', 'success')
					return redirect(url_for('ServiceRegistryView.show', service_id=registered_service.id))
					
				finally:
					loop.close()
					
			except Exception as e:
				flash(f'Error registering service: {str(e)}', 'error')
		
		return self.render_template(
			'registry/service_add.html',
			service_types=[st.value for st in ServiceType],
			load_balance_strategies=[lbs.value for lbs in LoadBalanceStrategy],
			title=self.add_title
		)
	
	@expose('/edit/<service_id>', methods=['GET', 'POST'])
	@has_access
	@has_permission('registry:update_service')
	def edit(self, service_id: str):
		"""Update service configuration."""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			loop.run_until_complete(ensure_service_initialized())
			service = get_registry_service()
			
			if service_id not in service.services:
				flash('Service not found', 'error')
				return redirect(url_for('ServiceRegistryView.list'))
			
			service_obj = service.services[service_id]
			
			if request.method == 'POST':
				try:
					# Process form data
					form_data = request.form.to_dict()
					
					# Handle lists and booleans
					if 'tags' in form_data:
						form_data['tags'] = [tag.strip() for tag in form_data['tags'].split(',') if tag.strip()]
					
					bool_fields = ['discovery_enabled', 'health_check_enabled', 'circuit_breaker_enabled',
								  'predictive_scaling', 'intelligent_routing']
					for field in bool_fields:
						form_data[field] = form_data.get(field) == 'on'
					
					# Update service object
					for key, value in form_data.items():
						if hasattr(service_obj, key):
							setattr(service_obj, key, value)
					
					service_obj.updated_at = datetime.now(timezone.utc)
					service_obj.last_modified_by = request.args.get('user_id', 'admin')
					
					flash(f'Service "{service_obj.name}" updated successfully', 'success')
					return redirect(url_for('ServiceRegistryView.show', service_id=service_id))
					
				except Exception as e:
					flash(f'Error updating service: {str(e)}', 'error')
		
		finally:
			loop.close()
		
		return self.render_template(
			'registry/service_edit.html',
			service=service_obj,
			service_types=[st.value for st in ServiceType],
			load_balance_strategies=[lbs.value for lbs in LoadBalanceStrategy],
			title=f"{self.edit_title} - {service_obj.name}"
		)
	
	@expose('/delete/<service_id>')
	@has_access
	@has_permission('registry:deregister_service')
	def delete(self, service_id: str):
		"""Deregister a service."""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			loop.run_until_complete(ensure_service_initialized())
			service = get_registry_service()
			
			if service_id not in service.services:
				flash('Service not found', 'error')
				return redirect(url_for('ServiceRegistryView.list'))
			
			service_obj = service.services[service_id]
			user_id = request.args.get('user_id', 'admin')
			
			success = loop.run_until_complete(service.deregister_service(service_id, user_id))
			
			if success:
				flash(f'Service "{service_obj.name}" deregistered successfully', 'success')
			else:
				flash('Failed to deregister service', 'error')
				
		except Exception as e:
			flash(f'Error deregistering service: {str(e)}', 'error')
		finally:
			loop.close()
		
		return redirect(url_for('ServiceRegistryView.list'))

class ServiceDiscoveryView(BaseView):
	"""Service discovery interface with intelligent search."""
	
	route_base = "/discovery"
	default_view = "search"
	
	@expose('/search/', methods=['GET', 'POST'])
	@has_access
	@has_permission('registry:discover_services')
	def search(self):
		"""Intelligent service discovery interface."""
		results = []
		search_performed = False
		
		if request.method == 'POST':
			try:
				# Build discovery query from form
				form_data = request.form.to_dict()
				
				# Handle boolean fields
				bool_fields = ['healthy_only', 'intelligent_ranking', 'predictive_filtering', 
							  'include_instances', 'include_health', 'include_metrics']
				for field in bool_fields:
					form_data[field] = form_data.get(field) == 'on'
				
				# Handle numeric fields
				numeric_fields = ['min_health_score', 'limit', 'offset']
				for field in numeric_fields:
					if form_data.get(field):
						form_data[field] = float(form_data[field]) if field == 'min_health_score' else int(form_data[field])
				
				# Remove empty fields
				form_data = {k: v for k, v in form_data.items() if v not in ['', None]}
				
				# Add tenant context
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					loop.run_until_complete(ensure_service_initialized())
					service = get_registry_service()
					form_data['tenant_id'] = service.tenant_id
					
					# Validate and execute query
					query = ServiceDiscoveryQuery(**form_data)
					result = loop.run_until_complete(service.discover_services(query))
					
					results = result.services
					search_performed = True
					
					flash(f'Found {result.total_count} services in {result.query_time_ms:.2f}ms', 'info')
					
				finally:
					loop.close()
					
			except Exception as e:
				flash(f'Discovery error: {str(e)}', 'error')
		
		return self.render_template(
			'registry/service_discovery.html',
			service_types=[st.value for st in ServiceType],
			service_statuses=[ss.value for ss in ServiceStatus],
			load_balance_strategies=[lbs.value for lbs in LoadBalanceStrategy],
			results=results,
			search_performed=search_performed,
			title="Service Discovery"
		)
	
	@expose('/api/search', methods=['POST'])
	@has_access
	@has_permission('registry:discover_services')
	def api_search(self):
		"""API endpoint for AJAX service discovery."""
		try:
			data = request.get_json()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			try:
				loop.run_until_complete(ensure_service_initialized())
				service = get_registry_service()
				data['tenant_id'] = service.tenant_id
				
				query = ServiceDiscoveryQuery(**data)
				result = loop.run_until_complete(service.discover_services(query))
				
				return jsonify({
					'success': True,
					'total_count': result.total_count,
					'returned_count': result.returned_count,
					'query_time_ms': result.query_time_ms,
					'services': [svc.model_dump() for svc in result.services]
				})
				
			finally:
				loop.close()
				
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400

class ServiceHealthView(BaseView):
	"""Service health monitoring dashboard."""
	
	route_base = "/health"
	default_view = "dashboard"
	
	@expose('/dashboard/')
	@has_access 
	@has_permission('registry:view_health')
	def dashboard(self):
		"""Service health monitoring dashboard."""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			loop.run_until_complete(ensure_service_initialized())
			service = get_registry_service()
			
			# Get all services with health data
			query = ServiceDiscoveryQuery(
				tenant_id=service.tenant_id,
				limit=1000,
				include_health=True
			)
			result = loop.run_until_complete(service.discover_services(query))
			
			# Gather health information
			health_data = []
			for svc in result.services:
				health_status = loop.run_until_complete(service.get_service_health(svc.id))
				if health_status:
					health_data.append({
						'service_id': svc.id,
						'service_name': svc.name,
						'service_type': svc.service_type,
						'status': health_status.overall_status,
						'health_score': health_status.health_score,
						'response_time': health_status.response_time_ms,
						'cpu_usage': health_status.cpu_usage_percent,
						'memory_usage': health_status.memory_usage_percent,
						'circuit_breaker_state': health_status.circuit_breaker_state,
						'last_updated': health_status.last_updated.strftime('%Y-%m-%d %H:%M:%S')
					})
			
			# Calculate health statistics
			total_services = len(health_data)
			healthy_services = len([h for h in health_data if h['status'] == ServiceStatus.HEALTHY])
			degraded_services = len([h for h in health_data if h['status'] == ServiceStatus.DEGRADED])
			unhealthy_services = len([h for h in health_data if h['status'] in [ServiceStatus.UNHEALTHY, ServiceStatus.CRITICAL]])
			
			stats = {
				'total': total_services,
				'healthy': healthy_services,
				'degraded': degraded_services,
				'unhealthy': unhealthy_services,
				'health_percentage': (healthy_services / total_services * 100) if total_services > 0 else 0
			}
			
		finally:
			loop.close()
		
		return self.render_template(
			'registry/health_dashboard.html',
			health_data=health_data,
			statistics=stats,
			title="Service Health Dashboard"
		)
	
	@expose('/service/<service_id>')
	@has_access
	@has_permission('registry:view_health')
	def service_health(self, service_id: str):
		"""Detailed health view for a specific service."""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			loop.run_until_complete(ensure_service_initialized())
			service = get_registry_service()
			
			if service_id not in service.services:
				flash('Service not found', 'error')
				return redirect(url_for('ServiceHealthView.dashboard'))
			
			service_obj = service.services[service_id]
			health_status = loop.run_until_complete(service.get_service_health(service_id))
			metrics = loop.run_until_complete(service.get_service_metrics(service_id, 24))
			
			# Get recent health events
			recent_events = [e for e in service.service_events 
							if e.service_id == service_id and 'health' in e.event_type
							and e.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)]
			
		finally:
			loop.close()
		
		return self.render_template(
			'registry/service_health_detail.html',
			service=service_obj,
			health_status=health_status,
			metrics=metrics,
			recent_events=recent_events,
			title=f"Health Status - {service_obj.name}"
		)
	
	@expose('/api/health/<service_id>')
	@has_access
	@has_permission('registry:view_health')
	def api_health_status(self, service_id: str):
		"""API endpoint for real-time health status."""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			try:
				loop.run_until_complete(ensure_service_initialized())
				service = get_registry_service()
				
				health_status = loop.run_until_complete(service.get_service_health(service_id))
				if not health_status:
					return jsonify({'error': 'Service not found'}), 404
				
				return jsonify({
					'success': True,
					'health_status': health_status.model_dump()
				})
				
			finally:
				loop.close()
				
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500

class ServiceAnalyticsView(BaseView):
	"""Service metrics and analytics dashboard."""
	
	route_base = "/analytics"
	default_view = "dashboard"
	
	@expose('/dashboard/')
	@has_access
	@has_permission('registry:view_analytics')
	def dashboard(self):
		"""Service analytics dashboard."""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			loop.run_until_complete(ensure_service_initialized())
			service = get_registry_service()
			
			# Get registry statistics
			stats = loop.run_until_complete(service.get_registry_statistics())
			
			# Get service metrics for trending
			all_metrics = []
			for service_id in service.services.keys():
				metrics = loop.run_until_complete(service.get_service_metrics(service_id, 24))
				all_metrics.extend(metrics)
			
			# Calculate trends and insights
			analytics_data = {
				'registry_stats': stats,
				'total_metrics': len(all_metrics),
				'trending_services': [],  # Would calculate trending services
				'performance_insights': [],  # Would calculate performance insights
				'capacity_recommendations': []  # Would generate AI recommendations
			}
			
		finally:
			loop.close()
		
		return self.render_template(
			'registry/analytics_dashboard.html',
			analytics_data=analytics_data,
			title="Service Analytics"
		)
	
	@expose('/service/<service_id>')
	@has_access
	@has_permission('registry:view_analytics')
	def service_analytics(self, service_id: str):
		"""Detailed analytics for a specific service."""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			loop.run_until_complete(ensure_service_initialized())
			service = get_registry_service()
			
			if service_id not in service.services:
				flash('Service not found', 'error')
				return redirect(url_for('ServiceAnalyticsView.dashboard'))
			
			service_obj = service.services[service_id]
			metrics = loop.run_until_complete(service.get_service_metrics(service_id, 168))  # 1 week
			
			# Calculate analytics
			analytics = {
				'total_requests': sum(m.request_count for m in metrics),
				'total_errors': sum(m.error_count for m in metrics),
				'avg_response_time': sum(m.response_time_p50 for m in metrics) / len(metrics) if metrics else 0,
				'peak_rps': max(m.request_count for m in metrics) if metrics else 0,
				'availability': sum(m.availability_percentage for m in metrics) / len(metrics) if metrics else 100
			}
			
		finally:
			loop.close()
		
		return self.render_template(
			'registry/service_analytics_detail.html',
			service=service_obj,
			metrics=metrics,
			analytics=analytics,
			title=f"Analytics - {service_obj.name}"
		)

# Export views for APG integration
__all__ = [
	'ServiceRegistryView', 'ServiceDiscoveryView', 
	'ServiceHealthView', 'ServiceAnalyticsView',
	'ServiceRegistrationForm', 'ServiceInstanceForm', 'ServiceDiscoveryForm',
	'HealthCheckForm', 'CircuitBreakerForm'
]
