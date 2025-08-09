#!/usr/bin/env python3
"""
APG Workflow Orchestration Integration Management Views

Flask-AppBuilder views for managing integrations, webhooks, and external connections.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from flask import request, jsonify, render_template, redirect, url_for, flash, current_app
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.baseviews import expose_api
from flask_appbuilder.actions import action
from flask_babel import lazy_gettext as _
from wtforms import Form, StringField, TextAreaField, SelectField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Length, Optional as OptionalValidator, URL
from wtforms.widgets import TextArea
from pydantic import ValidationError
import sqlalchemy as sa
from sqlalchemy.orm import joinedload

# APG Framework imports
from apg.framework.base_view import APGBaseView
from apg.framework.security import require_permission
from apg.framework.widgets import APGFormWidget, APGListWidget

# Local imports
from .integration_services import (
	IntegrationService, IntegrationConfig, IntegrationType, AuthenticationType,
	WebhookConfig, IntegrationRequest, IntegrationResponse
)
from .database import WorkflowDB, WorkflowInstanceDB


logger = logging.getLogger(__name__)


class IntegrationForm(Form):
	"""Form for creating and editing integrations."""
	
	name = StringField(
		_('Integration Name'),
		validators=[DataRequired(), Length(min=1, max=255)],
		render_kw={'placeholder': 'Enter integration name', 'class': 'form-control'}
	)
	
	type = SelectField(
		_('Integration Type'),
		choices=[
			(IntegrationType.REST_API.value, 'REST API'),
			(IntegrationType.GRAPHQL.value, 'GraphQL'),
			(IntegrationType.DATABASE.value, 'Database'),
			(IntegrationType.MESSAGE_QUEUE.value, 'Message Queue'),
			(IntegrationType.FILE_SYSTEM.value, 'File System'),
			(IntegrationType.CLOUD_SERVICE.value, 'Cloud Service'),
			(IntegrationType.WEBHOOK.value, 'Webhook'),
			(IntegrationType.APG_CAPABILITY.value, 'APG Capability')
		],
		validators=[DataRequired()],
		render_kw={'class': 'form-control'}
	)
	
	endpoint = StringField(
		_('Endpoint URL'),
		validators=[DataRequired(), URL()],
		render_kw={'placeholder': 'https://api.example.com/endpoint', 'class': 'form-control'}
	)
	
	auth_type = SelectField(
		_('Authentication Type'),
		choices=[
			(AuthenticationType.NONE.value, 'None'),
			(AuthenticationType.API_KEY.value, 'API Key'),
			(AuthenticationType.BEARER_TOKEN.value, 'Bearer Token'),
			(AuthenticationType.BASIC_AUTH.value, 'Basic Auth'),
			(AuthenticationType.OAUTH2.value, 'OAuth 2.0'),
			(AuthenticationType.JWT.value, 'JWT'),
			(AuthenticationType.CERTIFICATE.value, 'Certificate'),
			(AuthenticationType.CUSTOM.value, 'Custom')
		],
		default=AuthenticationType.NONE.value,
		render_kw={'class': 'form-control'}
	)
	
	auth_config = TextAreaField(
		_('Authentication Configuration (JSON)'),
		validators=[OptionalValidator()],
		widget=TextArea(),
		render_kw={
			'placeholder': '{"key": "value"}',
			'class': 'form-control',
			'rows': 4
		}
	)
	
	headers = TextAreaField(
		_('Default Headers (JSON)'),
		validators=[OptionalValidator()],
		widget=TextArea(),
		render_kw={
			'placeholder': '{"Content-Type": "application/json"}',
			'class': 'form-control',
			'rows': 3
		}
	)
	
	timeout_seconds = IntegerField(
		_('Timeout (seconds)'),
		default=30,
		validators=[OptionalValidator()],
		render_kw={'class': 'form-control', 'min': '1', 'max': '300'}
	)
	
	retry_attempts = IntegerField(
		_('Retry Attempts'),
		default=3,
		validators=[OptionalValidator()],
		render_kw={'class': 'form-control', 'min': '0', 'max': '10'}
	)
	
	retry_delay_seconds = IntegerField(
		_('Retry Delay (seconds)'),
		default=5,
		validators=[OptionalValidator()],
		render_kw={'class': 'form-control', 'min': '1', 'max': '60'}
	)
	
	tenant_id = StringField(
		_('Tenant ID'),
		validators=[DataRequired(), Length(min=1, max=100)],
		render_kw={'placeholder': 'Tenant identifier', 'class': 'form-control'}
	)
	
	is_active = BooleanField(
		_('Active'),
		default=True,
		render_kw={'class': 'form-check-input'}
	)


class WebhookForm(Form):
	"""Form for creating and editing webhooks."""
	
	name = StringField(
		_('Webhook Name'),
		validators=[DataRequired(), Length(min=1, max=255)],
		render_kw={'placeholder': 'Enter webhook name', 'class': 'form-control'}
	)
	
	url = StringField(
		_('Webhook URL'),
		validators=[DataRequired(), URL()],
		render_kw={'placeholder': 'https://your-app.com/webhook', 'class': 'form-control'}
	)
	
	secret = StringField(
		_('Secret Key'),
		validators=[OptionalValidator()],
		render_kw={'placeholder': 'Optional secret for signature verification', 'class': 'form-control'}
	)
	
	events = TextAreaField(
		_('Events (JSON Array)'),
		validators=[DataRequired()],
		widget=TextArea(),
		render_kw={
			'placeholder': '["workflow.started", "workflow.completed", "workflow.failed"]',
			'class': 'form-control',
			'rows': 3
		}
	)
	
	tenant_id = StringField(
		_('Tenant ID'),
		validators=[DataRequired(), Length(min=1, max=100)],
		render_kw={'placeholder': 'Tenant identifier', 'class': 'form-control'}
	)
	
	retry_attempts = IntegerField(
		_('Retry Attempts'),
		default=3,
		validators=[OptionalValidator()],
		render_kw={'class': 'form-control', 'min': '0', 'max': '10'}
	)
	
	retry_delay_seconds = IntegerField(
		_('Retry Delay (seconds)'),
		default=5,
		validators=[OptionalValidator()],
		render_kw={'class': 'form-control', 'min': '1', 'max': '60'}
	)
	
	timeout_seconds = IntegerField(
		_('Timeout (seconds)'),
		default=30,
		validators=[OptionalValidator()],
		render_kw={'class': 'form-control', 'min': '1', 'max': '300'}
	)
	
	is_active = BooleanField(
		_('Active'),
		default=True,
		render_kw={'class': 'form-check-input'}
	)


class WOIntegrationsView(APGBaseView):
	"""View for managing external system integrations."""
	
	route_base = '/workflow_orchestration/integrations'
	default_view = 'list'
	
	def __init__(self):
		super().__init__()
		self.integration_service = IntegrationService()
	
	@expose('/')
	@has_access
	def list(self):
		"""List all integrations."""
		# Get integrations from service
		integrations = list(self.integration_service.integrations.values())
		
		# Filter by tenant if needed
		tenant_id = request.args.get('tenant_id')
		if tenant_id:
			integrations = [i for i in integrations if i.tenant_id == tenant_id]
		
		return self.render_template(
			'workflow_orchestration/integrations_list.html',
			title=_('Integrations'),
			integrations=integrations
		)
	
	@expose('/add', methods=['GET', 'POST'])
	@has_access
	def add(self):
		"""Add new integration."""
		form = IntegrationForm()
		
		if request.method == 'POST' and form.validate_on_submit():
			try:
				# Parse JSON fields
				auth_config = {}
				if form.auth_config.data:
					auth_config = json.loads(form.auth_config.data)
				
				headers = {}
				if form.headers.data:
					headers = json.loads(form.headers.data)
				
				# Create integration config
				config = IntegrationConfig(
					id=f"integration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
					name=form.name.data,
					type=IntegrationType(form.type.data),
					endpoint=form.endpoint.data,
					auth_type=AuthenticationType(form.auth_type.data),
					auth_config=auth_config,
					headers=headers,
					timeout_seconds=form.timeout_seconds.data or 30,
					retry_attempts=form.retry_attempts.data or 3,
					retry_delay_seconds=form.retry_delay_seconds.data or 5,
					tenant_id=form.tenant_id.data,
					is_active=form.is_active.data
				)
				
				# Register integration
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				integration_id = loop.run_until_complete(
					self.integration_service.register_integration(config)
				)
				
				flash(_('Integration created successfully'), 'success')
				return redirect(url_for('WOIntegrationsView.list'))
				
			except json.JSONDecodeError as e:
				flash(_('Invalid JSON in configuration fields'), 'error')
			except Exception as e:
				flash(_('Failed to create integration: %(error)s', error=str(e)), 'error')
		
		return self.render_template(
			'workflow_orchestration/integration_form.html',
			title=_('Add Integration'),
			form=form
		)
	
	@expose('/edit/<integration_id>', methods=['GET', 'POST'])
	@has_access
	def edit(self, integration_id):
		"""Edit existing integration."""
		if integration_id not in self.integration_service.integrations:
			flash(_('Integration not found'), 'error')
			return redirect(url_for('WOIntegrationsView.list'))
		
		config = self.integration_service.integrations[integration_id]
		form = IntegrationForm(obj=config)
		
		# Pre-populate JSON fields
		form.auth_config.data = json.dumps(config.auth_config, indent=2) if config.auth_config else ''
		form.headers.data = json.dumps(config.headers, indent=2) if config.headers else ''
		
		if request.method == 'POST' and form.validate_on_submit():
			try:
				# Parse JSON fields
				auth_config = {}
				if form.auth_config.data:
					auth_config = json.loads(form.auth_config.data)
				
				headers = {}
				if form.headers.data:
					headers = json.loads(form.headers.data)
				
				# Prepare updates
				updates = {
					'name': form.name.data,
					'type': IntegrationType(form.type.data),
					'endpoint': form.endpoint.data,
					'auth_type': AuthenticationType(form.auth_type.data),
					'auth_config': auth_config,
					'headers': headers,
					'timeout_seconds': form.timeout_seconds.data or 30,
					'retry_attempts': form.retry_attempts.data or 3,
					'retry_delay_seconds': form.retry_delay_seconds.data or 5,
					'is_active': form.is_active.data
				}
				
				# Update integration
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				success = loop.run_until_complete(
					self.integration_service.update_integration(integration_id, updates)
				)
				
				if success:
					flash(_('Integration updated successfully'), 'success')
					return redirect(url_for('WOIntegrationsView.list'))
				
			except json.JSONDecodeError as e:
				flash(_('Invalid JSON in configuration fields'), 'error')
			except Exception as e:
				flash(_('Failed to update integration: %(error)s', error=str(e)), 'error')
		
		return self.render_template(
			'workflow_orchestration/integration_form.html',
			title=_('Edit Integration'),
			form=form,
			integration_id=integration_id
		)
	
	@expose('/delete/<integration_id>', methods=['POST'])
	@has_access
	def delete(self, integration_id):
		"""Delete integration."""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			success = loop.run_until_complete(
				self.integration_service.delete_integration(integration_id)
			)
			
			if success:
				flash(_('Integration deleted successfully'), 'success')
			else:
				flash(_('Failed to delete integration'), 'error')
				
		except Exception as e:
			flash(_('Failed to delete integration: %(error)s', error=str(e)), 'error')
		
		return redirect(url_for('WOIntegrationsView.list'))
	
	@expose('/test/<integration_id>', methods=['POST'])
	@has_access
	def test(self, integration_id):
		"""Test integration connection."""
		try:
			if integration_id not in self.integration_service.integrations:
				return jsonify({'success': False, 'error': 'Integration not found'}), 404
			
			config = self.integration_service.integrations[integration_id]
			
			# Create test request
			test_request = IntegrationRequest(
				integration_id=integration_id,
				method='GET',
				payload={},
				tenant_id=config.tenant_id
			)
			
			# Make test call
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			response = loop.run_until_complete(
				self.integration_service.call_external_system(test_request)
			)
			
			return jsonify({
				'success': response.success,
				'status_code': response.status_code,
				'duration_ms': response.duration_ms,
				'error_message': response.error_message
			})
			
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose_api(name='integration_metrics', url='/api/integration/<integration_id>/metrics')
	@protect()
	def get_integration_metrics(self, integration_id):
		"""Get integration performance metrics."""
		try:
			# Get metrics from database
			session = self.appbuilder.get_session
			
			# Get recent performance data
			seven_days_ago = datetime.utcnow() - timedelta(days=7)
			
			metrics_query = session.execute(
				"""
				SELECT 
					DATE(created_at) as date,
					COUNT(*) as total_requests,
					COUNT(CASE WHEN success = true THEN 1 END) as successful_requests,
					AVG(duration_ms) as avg_response_time,
					MIN(duration_ms) as min_response_time,
					MAX(duration_ms) as max_response_time
				FROM wo_integration_audit
				WHERE integration_id = :integration_id
				AND created_at >= :start_date
				GROUP BY DATE(created_at)
				ORDER BY date DESC
				""",
				{
					'integration_id': integration_id,
					'start_date': seven_days_ago
				}
			)
			
			metrics_data = []
			for row in metrics_query.fetchall():
				metrics_data.append({
					'date': row.date.isoformat(),
					'total_requests': row.total_requests,
					'successful_requests': row.successful_requests,
					'failed_requests': row.total_requests - row.successful_requests,
					'success_rate': (row.successful_requests / row.total_requests * 100) if row.total_requests > 0 else 0,
					'avg_response_time': round(row.avg_response_time, 2) if row.avg_response_time else 0,
					'min_response_time': row.min_response_time or 0,
					'max_response_time': row.max_response_time or 0
				})
			
			return jsonify({
				'integration_id': integration_id,
				'metrics': metrics_data
			})
			
		except Exception as e:
			return jsonify({'error': str(e)}), 500


class WOWebhooksView(APGBaseView):
	"""View for managing webhook subscriptions."""
	
	route_base = '/workflow_orchestration/webhooks'
	default_view = 'list'
	
	def __init__(self):
		super().__init__()
		self.integration_service = IntegrationService()
	
	@expose('/')
	@has_access
	def list(self):
		"""List all webhooks."""
		# Get webhooks from service
		webhooks = list(self.integration_service.webhooks.values())
		
		# Filter by tenant if needed
		tenant_id = request.args.get('tenant_id')
		if tenant_id:
			webhooks = [w for w in webhooks if w.tenant_id == tenant_id]
		
		return self.render_template(
			'workflow_orchestration/webhooks_list.html',
			title=_('Webhooks'),
			webhooks=webhooks
		)
	
	@expose('/add', methods=['GET', 'POST'])
	@has_access
	def add(self):
		"""Add new webhook."""
		form = WebhookForm()
		
		if request.method == 'POST' and form.validate_on_submit():
			try:
				# Parse events JSON
				events = json.loads(form.events.data)
				
				# Create webhook config
				config = WebhookConfig(
					name=form.name.data,
					url=form.url.data,
					secret=form.secret.data or None,
					events=events,
					tenant_id=form.tenant_id.data,
					is_active=form.is_active.data,
					retry_attempts=form.retry_attempts.data or 3,
					retry_delay_seconds=form.retry_delay_seconds.data or 5,
					timeout_seconds=form.timeout_seconds.data or 30
				)
				
				# Register webhook
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				webhook_id = loop.run_until_complete(
					self.integration_service.register_webhook(config)
				)
				
				flash(_('Webhook created successfully'), 'success')
				return redirect(url_for('WOWebhooksView.list'))
				
			except json.JSONDecodeError as e:
				flash(_('Invalid JSON in events field'), 'error')
			except Exception as e:
				flash(_('Failed to create webhook: %(error)s', error=str(e)), 'error')
		
		return self.render_template(
			'workflow_orchestration/webhook_form.html',
			title=_('Add Webhook'),
			form=form
		)
	
	@expose('/deliveries')
	@has_access
	def deliveries(self):
		"""View webhook deliveries."""
		# Get recent deliveries from database
		session = self.appbuilder.get_session
		
		# Get pagination parameters
		page = int(request.args.get('page', 1))
		per_page = 25
		offset = (page - 1) * per_page
		
		# Get total count
		total_count = session.execute(
			"SELECT COUNT(*) FROM wo_webhook_deliveries"
		).scalar()
		
		# Get deliveries with webhook info
		deliveries_query = session.execute(
			"""
			SELECT 
				wd.id, wd.webhook_id, wd.event_type, wd.status,
				wd.attempts, wd.created_at, wd.last_attempt_at,
				wd.response_code, wd.error_message,
				w.name as webhook_name, w.url as webhook_url
			FROM wo_webhook_deliveries wd
			JOIN wo_webhooks w ON wd.webhook_id = w.id
			ORDER BY wd.created_at DESC
			LIMIT :limit OFFSET :offset
			""",
			{'limit': per_page, 'offset': offset}
		)
		
		deliveries = []
		for row in deliveries_query.fetchall():
			deliveries.append({
				'id': row.id,
				'webhook_id': row.webhook_id,
				'webhook_name': row.webhook_name,
				'webhook_url': row.webhook_url,
				'event_type': row.event_type,
				'status': row.status,
				'attempts': row.attempts,
				'created_at': row.created_at,
				'last_attempt_at': row.last_attempt_at,
				'response_code': row.response_code,
				'error_message': row.error_message
			})
		
		# Calculate pagination
		total_pages = (total_count + per_page - 1) // per_page
		has_prev = page > 1
		has_next = page < total_pages
		
		return self.render_template(
			'workflow_orchestration/webhook_deliveries.html',
			title=_('Webhook Deliveries'),
			deliveries=deliveries,
			pagination={
				'page': page,
				'per_page': per_page,
				'total': total_count,
				'total_pages': total_pages,
				'has_prev': has_prev,
				'has_next': has_next,
				'prev_num': page - 1 if has_prev else None,
				'next_num': page + 1 if has_next else None
			}
		)
	
	@expose('/test/<webhook_id>', methods=['POST'])
	@has_access
	def test(self, webhook_id):
		"""Test webhook delivery."""
		try:
			if webhook_id not in self.integration_service.webhooks:
				return jsonify({'success': False, 'error': 'Webhook not found'}), 404
			
			webhook = self.integration_service.webhooks[webhook_id]
			
			# Create test payload
			test_payload = {
				'event_type': 'test.webhook',
				'timestamp': datetime.utcnow().isoformat(),
				'data': {
					'message': 'This is a test webhook delivery',
					'webhook_id': webhook_id,
					'webhook_name': webhook.name
				}
			}
			
			# Trigger webhook
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			loop.run_until_complete(
				self.integration_service.trigger_webhook(
					'test.webhook',
					test_payload,
					webhook.tenant_id
				)
			)
			
			return jsonify({'success': True, 'message': 'Test webhook sent'})
			
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500


class WOAPGCapabilitiesView(APGBaseView):
	"""View for managing APG capability integrations."""
	
	route_base = '/workflow_orchestration/apg_capabilities'
	default_view = 'list'
	
	def __init__(self):
		super().__init__()
		self.integration_service = IntegrationService()
	
	@expose('/')
	@has_access
	def list(self):
		"""List all APG capabilities."""
		# Get APG capabilities from service
		capabilities = list(self.integration_service.apg_capabilities.values())
		
		return self.render_template(
			'workflow_orchestration/apg_capabilities_list.html',
			title=_('APG Capabilities'),
			capabilities=capabilities
		)
	
	@expose('/test/<capability_name>', methods=['POST'])
	@has_access
	def test(self, capability_name):
		"""Test APG capability call."""
		try:
			if capability_name not in self.integration_service.apg_capabilities:
				return jsonify({'success': False, 'error': 'Capability not found'}), 404
			
			# Get test data from request
			test_data = request.get_json() or {}
			tenant_id = test_data.get('tenant_id', 'test-tenant')
			input_data = test_data.get('input_data', {'test': True})
			
			# Call capability
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			result = loop.run_until_complete(
				self.integration_service.call_apg_capability(
					capability_name,
					input_data,
					tenant_id,
					'test-user'
				)
			)
			
			return jsonify({
				'success': True,
				'result': result
			})
			
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500


class WOIntegrationDashboardView(APGBaseView):
	"""Dashboard view for integration monitoring."""
	
	route_base = '/workflow_orchestration/integration_dashboard'
	default_view = 'dashboard'
	
	def __init__(self):
		super().__init__()
		self.integration_service = IntegrationService()
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Integration monitoring dashboard."""
		session = self.appbuilder.get_session
		
		# Get integration statistics
		total_integrations = len(self.integration_service.integrations)
		active_integrations = sum(1 for i in self.integration_service.integrations.values() if i.is_active)
		total_webhooks = len(self.integration_service.webhooks)
		active_webhooks = sum(1 for w in self.integration_service.webhooks.values() if w.is_active)
		
		# Get recent activity
		twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
		
		recent_requests = session.execute(
			"SELECT COUNT(*) FROM wo_integration_audit WHERE created_at >= :start_time",
			{'start_time': twenty_four_hours_ago}
		).scalar() or 0
		
		successful_requests = session.execute(
			"SELECT COUNT(*) FROM wo_integration_audit WHERE created_at >= :start_time AND success = true",
			{'start_time': twenty_four_hours_ago}
		).scalar() or 0
		
		failed_requests = recent_requests - successful_requests
		success_rate = (successful_requests / recent_requests * 100) if recent_requests > 0 else 0
		
		# Get webhook deliveries
		recent_deliveries = session.execute(
			"SELECT COUNT(*) FROM wo_webhook_deliveries WHERE created_at >= :start_time",
			{'start_time': twenty_four_hours_ago}
		).scalar() or 0
		
		successful_deliveries = session.execute(
			"SELECT COUNT(*) FROM wo_webhook_deliveries WHERE created_at >= :start_time AND status = 'delivered'",
			{'start_time': twenty_four_hours_ago}
		).scalar() or 0
		
		# Get top integrations by usage
		top_integrations = session.execute(
			"""
			SELECT 
				i.name,
				COUNT(ia.id) as request_count,
				AVG(ia.duration_ms) as avg_response_time
			FROM wo_integrations i
			LEFT JOIN wo_integration_audit ia ON i.id = ia.integration_id
				AND ia.created_at >= :start_time
			GROUP BY i.id, i.name
			ORDER BY request_count DESC
			LIMIT 10
			""",
			{'start_time': twenty_four_hours_ago}
		).fetchall()
		
		return self.render_template(
			'workflow_orchestration/integration_dashboard.html',
			title=_('Integration Dashboard'),
			metrics={
				'total_integrations': total_integrations,
				'active_integrations': active_integrations,
				'total_webhooks': total_webhooks,
				'active_webhooks': active_webhooks,
				'recent_requests': recent_requests,
				'successful_requests': successful_requests,
				'failed_requests': failed_requests,
				'success_rate': round(success_rate, 1),
				'recent_deliveries': recent_deliveries,
				'successful_deliveries': successful_deliveries,
				'failed_deliveries': recent_deliveries - successful_deliveries
			},
			top_integrations=top_integrations
		)
	
	@expose_api(name='integration_health', url='/api/integration/health')
	@protect()
	def get_integration_health(self):
		"""Get integration health status."""
		try:
			session = self.appbuilder.get_session
			
			# Get health metrics for each integration
			health_data = []
			
			for integration_id, config in self.integration_service.integrations.items():
				# Get recent success rate
				one_hour_ago = datetime.utcnow() - timedelta(hours=1)
				
				total_requests = session.execute(
					"SELECT COUNT(*) FROM wo_integration_audit WHERE integration_id = :id AND created_at >= :start_time",
					{'id': integration_id, 'start_time': one_hour_ago}
				).scalar() or 0
				
				successful_requests = session.execute(
					"SELECT COUNT(*) FROM wo_integration_audit WHERE integration_id = :id AND created_at >= :start_time AND success = true",
					{'id': integration_id, 'start_time': one_hour_ago}
				).scalar() or 0
				
				success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 100
				
				# Determine health status
				if not config.is_active:
					status = 'inactive'
				elif success_rate >= 95:
					status = 'healthy'
				elif success_rate >= 80:
					status = 'warning'
				else:
					status = 'unhealthy'
				
				health_data.append({
					'integration_id': integration_id,
					'name': config.name,
					'type': config.type.value,
					'status': status,
					'success_rate': round(success_rate, 1),
					'total_requests': total_requests,
					'is_active': config.is_active
				})
			
			return jsonify({
				'timestamp': datetime.utcnow().isoformat(),
				'integrations': health_data
			})
			
		except Exception as e:
			return jsonify({'error': str(e)}), 500