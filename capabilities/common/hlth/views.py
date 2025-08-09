#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - Flask-AppBuilder Views
Comprehensive health dashboard views and user interface

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from flask import request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, has_access, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import ChartView
from flask_appbuilder.widgets import ListWidget
from flask_appbuilder.actions import action

from .models import (
	HealthMetric, HealthAlert, HealthBaseline, HealthRule, 
	HealthAction, SystemComponent, HealthReport,
	HealthStatus, HealthSeverity, HealthDimension
)
from .service import SystemHealthService


class HealthDashboardView(BaseView):
	"""Main health dashboard view with real-time health visualization"""
	
	route_base = '/health/dashboard'
	default_view = 'executive'

	@expose('/executive')
	@has_access
	def executive(self):
		"""Executive health dashboard with business impact focus"""
		try:
			# Get tenant context
			tenant_id = self._get_current_tenant_id()
			
			# Create dashboard data
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			health_service = self._get_health_service()
			dashboard_data = loop.run_until_complete(
				health_service.create_health_dashboard_data(tenant_id, 'executive')
			)
			
			loop.close()
			
			return self.render_template(
				'health/executive_dashboard.html',
				dashboard_data=dashboard_data,
				page_title='Executive Health Dashboard'
			)
		
		except Exception as e:
			flash(f'Error loading executive dashboard: {str(e)}', 'error')
			return redirect(url_for('HealthDashboardView.operational'))

	@expose('/operational')
	@has_access
	def operational(self):
		"""Operational health dashboard with technical details"""
		try:
			tenant_id = self._get_current_tenant_id()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			health_service = self._get_health_service()
			dashboard_data = loop.run_until_complete(
				health_service.create_health_dashboard_data(tenant_id, 'operational')
			)
			
			loop.close()
			
			return self.render_template(
				'health/operational_dashboard.html',
				dashboard_data=dashboard_data,
				page_title='Operational Health Dashboard'
			)
		
		except Exception as e:
			flash(f'Error loading operational dashboard: {str(e)}', 'error')
			return jsonify({'error': str(e), 'status': 'failed'})

	@expose('/predictive')
	@has_access
	def predictive(self):
		"""Predictive health dashboard with forecasting and ML insights"""
		try:
			tenant_id = self._get_current_tenant_id()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			health_service = self._get_health_service()
			dashboard_data = loop.run_until_complete(
				health_service.create_health_dashboard_data(tenant_id, 'predictive')
			)
			
			# Get multi-dimensional analysis
			multi_dimensional_analysis = loop.run_until_complete(
				health_service.analyze_multi_dimensional_health(tenant_id)
			)
			
			loop.close()
			
			return self.render_template(
				'health/predictive_dashboard.html',
				dashboard_data=dashboard_data,
				multi_dimensional_analysis=multi_dimensional_analysis,
				page_title='Predictive Health Intelligence'
			)
		
		except Exception as e:
			flash(f'Error loading predictive dashboard: {str(e)}', 'error')
			return jsonify({'error': str(e), 'status': 'failed'})

	@expose('/api/dashboard_data/<dashboard_type>')
	@has_access
	def api_dashboard_data(self, dashboard_type: str):
		"""API endpoint for real-time dashboard data updates"""
		try:
			tenant_id = self._get_current_tenant_id()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			health_service = self._get_health_service()
			dashboard_data = loop.run_until_complete(
				health_service.create_health_dashboard_data(tenant_id, dashboard_type)
			)
			
			loop.close()
			
			return jsonify({
				'status': 'success',
				'data': dashboard_data,
				'timestamp': datetime.utcnow().isoformat()
			})
		
		except Exception as e:
			return jsonify({
				'status': 'error',
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			})

	def _get_current_tenant_id(self) -> str:
		"""Get current tenant ID from session or request context"""
		return request.args.get('tenant_id', 'default')

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


class HealthMetricView(ModelView):
	"""Health metric management view"""
	
	datamodel = SQLAInterface(HealthMetric)
	
	list_columns = ['timestamp', 'tenant_id', 'component_id', 'name', 'value', 'dimension', 'health_status']
	show_columns = ['timestamp', 'tenant_id', 'component_id', 'name', 'value', 'dimension', 'unit', 
	               'health_status', 'business_context', 'tags', 'metadata']
	search_columns = ['component_id', 'name', 'dimension', 'health_status']
	
	base_order = ('timestamp', 'desc')
	base_filters = [['timestamp', lambda: datetime.utcnow() - timedelta(hours=24), '>']]
	
	@action("export_metrics", "Export Metrics", "Export selected metrics to CSV", "fa-download")
	def export_metrics(self, items):
		"""Export selected health metrics"""
		try:
			# Implementation for CSV export
			flash(f'Exporting {len(items)} health metrics', 'info')
			return redirect(request.referrer)
		except Exception as e:
			flash(f'Export failed: {str(e)}', 'error')
			return redirect(request.referrer)


class HealthAlertView(ModelView):
	"""Health alert management view"""
	
	datamodel = SQLAInterface(HealthAlert)
	
	list_columns = ['timestamp', 'tenant_id', 'component_id', 'name', 'severity', 'health_status', 'status']
	show_columns = ['timestamp', 'tenant_id', 'rule_id', 'component_id', 'name', 'message', 'severity',
	               'health_status', 'source_metric', 'source_value', 'threshold_value', 'business_impact_score',
	               'escalation_level', 'status', 'acknowledged_by', 'resolved_by']
	search_columns = ['component_id', 'name', 'severity', 'status']
	
	base_order = ('timestamp', 'desc')
	
	@action("acknowledge_alerts", "Acknowledge Alerts", "Acknowledge selected alerts", "fa-check")
	def acknowledge_alerts(self, items):
		"""Acknowledge selected health alerts"""
		try:
			for alert in items:
				alert.status = 'acknowledged'
				alert.acknowledged_by = self._get_current_user()
				alert.acknowledged_timestamp = datetime.utcnow()
			
			self.datamodel.session.commit()
			flash(f'Acknowledged {len(items)} alerts', 'success')
			
		except Exception as e:
			flash(f'Acknowledgment failed: {str(e)}', 'error')
		
		return redirect(request.referrer)

	@action("resolve_alerts", "Resolve Alerts", "Mark selected alerts as resolved", "fa-check-circle")
	def resolve_alerts(self, items):
		"""Resolve selected health alerts"""
		try:
			for alert in items:
				alert.status = 'resolved'
				alert.resolved_by = self._get_current_user()
				alert.resolved_timestamp = datetime.utcnow()
			
			self.datamodel.session.commit()
			flash(f'Resolved {len(items)} alerts', 'success')
			
		except Exception as e:
			flash(f'Resolution failed: {str(e)}', 'error')
		
		return redirect(request.referrer)

	def _get_current_user(self) -> str:
		"""Get current user from Flask-AppBuilder security manager"""
		from flask import g
		return getattr(g.user, 'username', 'system') if hasattr(g, 'user') and g.user else 'system'


class SystemComponentView(ModelView):
	"""System component management view"""
	
	datamodel = SQLAInterface(SystemComponent)
	
	list_columns = ['component_id', 'tenant_id', 'name', 'component_type', 'health_status', 'status', 'discovery_timestamp']
	show_columns = ['component_id', 'tenant_id', 'name', 'description', 'component_type', 'health_status', 
	               'status', 'version', 'environment', 'tags', 'dependencies', 'business_criticality',
	               'discovery_timestamp', 'last_updated', 'metadata']
	search_columns = ['component_id', 'name', 'component_type', 'health_status']
	
	base_order = ('discovery_timestamp', 'desc')

	@action("health_assessment", "Health Assessment", "Run health assessment on selected components", "fa-heartbeat")
	def health_assessment(self, items):
		"""Run health assessment on selected components"""
		try:
			tenant_id = self._get_current_tenant_id()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			health_service = self._get_health_service()
			
			assessment_results = []
			for component in items:
				result = loop.run_until_complete(
					health_service.assess_component_health(component.component_id, tenant_id)
				)
				assessment_results.append(result)
			
			loop.close()
			
			flash(f'Health assessment completed for {len(items)} components', 'success')
			
		except Exception as e:
			flash(f'Health assessment failed: {str(e)}', 'error')
		
		return redirect(request.referrer)

	def _get_current_tenant_id(self) -> str:
		"""Get current tenant ID from request context"""
		return request.args.get('tenant_id', 'default')

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


class HealthReportView(ModelView):
	"""Health report management view"""
	
	datamodel = SQLAInterface(HealthReport)
	
	list_columns = ['report_id', 'tenant_id', 'report_type', 'overall_health_score', 'health_grade', 
	               'generation_timestamp']
	show_columns = ['report_id', 'tenant_id', 'report_type', 'time_period_hours', 'component_ids',
	               'overall_health_score', 'health_grade', 'total_components', 'healthy_components',
	               'degraded_components', 'unhealthy_components', 'critical_alerts', 'recommendations',
	               'generation_timestamp', 'metadata']
	search_columns = ['report_type', 'health_grade']
	
	base_order = ('generation_timestamp', 'desc')

	@action("generate_report", "Generate New Report", "Generate a new health report", "fa-file-text")
	def generate_report(self, items):
		"""Generate a new health report"""
		try:
			tenant_id = self._get_current_tenant_id()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			health_service = self._get_health_service()
			
			report = loop.run_until_complete(
				health_service.generate_health_report(
					tenant_id=tenant_id,
					report_type='comprehensive',
					time_period_hours=24
				)
			)
			
			loop.close()
			
			flash(f'Health report generated successfully: {report.report_id}', 'success')
			
		except Exception as e:
			flash(f'Report generation failed: {str(e)}', 'error')
		
		return redirect(request.referrer)

	def _get_current_tenant_id(self) -> str:
		"""Get current tenant ID from request context"""
		return request.args.get('tenant_id', 'default')

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


class HealthRuleView(ModelView):
	"""Health rule management view"""
	
	datamodel = SQLAInterface(HealthRule)
	
	list_columns = ['rule_id', 'tenant_id', 'name', 'dimension', 'enabled', 'severity', 'created_timestamp']
	show_columns = ['rule_id', 'tenant_id', 'name', 'description', 'dimension', 'metric_pattern',
	               'threshold_value', 'threshold_operator', 'severity', 'enabled', 'auto_resolve',
	               'escalation_policy', 'remediation_actions', 'created_timestamp', 'updated_timestamp']
	search_columns = ['name', 'dimension', 'enabled']
	
	base_order = ('created_timestamp', 'desc')

	@action("enable_rules", "Enable Rules", "Enable selected health rules", "fa-play")
	def enable_rules(self, items):
		"""Enable selected health rules"""
		try:
			for rule in items:
				rule.enabled = True
				rule.updated_timestamp = datetime.utcnow()
			
			self.datamodel.session.commit()
			flash(f'Enabled {len(items)} health rules', 'success')
			
		except Exception as e:
			flash(f'Failed to enable rules: {str(e)}', 'error')
		
		return redirect(request.referrer)

	@action("disable_rules", "Disable Rules", "Disable selected health rules", "fa-pause")
	def disable_rules(self, items):
		"""Disable selected health rules"""
		try:
			for rule in items:
				rule.enabled = False
				rule.updated_timestamp = datetime.utcnow()
			
			self.datamodel.session.commit()
			flash(f'Disabled {len(items)} health rules', 'success')
			
		except Exception as e:
			flash(f'Failed to disable rules: {str(e)}', 'error')
		
		return redirect(request.referrer)


class HealthAnalyticsView(BaseView):
	"""Advanced health analytics and insights view"""
	
	route_base = '/health/analytics'
	default_view = 'multi_dimensional'

	@expose('/multi-dimensional')
	@has_access
	def multi_dimensional(self):
		"""Multi-dimensional health analysis view"""
		try:
			tenant_id = self._get_current_tenant_id()
			component_id = request.args.get('component_id')
			time_window_hours = int(request.args.get('time_window_hours', 24))
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			health_service = self._get_health_service()
			analysis_result = loop.run_until_complete(
				health_service.analyze_multi_dimensional_health(
					tenant_id, component_id, time_window_hours
				)
			)
			
			loop.close()
			
			return self.render_template(
				'health/multi_dimensional_analysis.html',
				analysis_result=analysis_result,
				page_title='Multi-Dimensional Health Analysis'
			)
		
		except Exception as e:
			flash(f'Multi-dimensional analysis failed: {str(e)}', 'error')
			return jsonify({'error': str(e), 'status': 'failed'})

	@expose('/predictions')
	@has_access
	def predictions(self):
		"""Health predictions and forecasting view"""
		try:
			tenant_id = self._get_current_tenant_id()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			health_service = self._get_health_service()
			
			# Get prediction data
			predictions = []
			components = loop.run_until_complete(
				health_service.get_monitored_components(tenant_id)
			)
			
			for component in components[:10]:  # Limit to top 10 for performance
				prediction = loop.run_until_complete(
					health_service.predict_component_health(
						component.component_id, tenant_id
					)
				)
				predictions.append(prediction)
			
			loop.close()
			
			return self.render_template(
				'health/predictions.html',
				predictions=predictions,
				page_title='Health Predictions & Forecasting'
			)
		
		except Exception as e:
			flash(f'Predictions analysis failed: {str(e)}', 'error')
			return jsonify({'error': str(e), 'status': 'failed'})

	def _get_current_tenant_id(self) -> str:
		"""Get current tenant ID from request context"""
		return request.args.get('tenant_id', 'default')

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


class HealthChartView(ChartView):
	"""Health metrics charting view"""
	
	chart_title = 'Health Metrics Overview'
	label_columns = ['timestamp']
	group_by_columns = ['dimension']
	
	search_columns = ['component_id', 'dimension', 'health_status']
	
	definitions = [
		{
			'group': 'dimension',
			'series': 'value'
		}
	]

	def query_obj(self):
		"""Define chart query object"""
		query_obj = {
			'groupby': ['dimension'],
			'metrics': [{'aggregate': 'avg', 'column': 'value'}],
			'filters': [
				{'col': 'timestamp', 'op': '>=', 'val': (datetime.utcnow() - timedelta(hours=24)).isoformat()}
			]
		}
		return query_obj


# Export views for registration with Flask-AppBuilder
__all__ = [
	'HealthDashboardView',
	'HealthMetricView', 
	'HealthAlertView',
	'SystemComponentView',
	'HealthReportView',
	'HealthRuleView',
	'HealthAnalyticsView',
	'HealthChartView'
]