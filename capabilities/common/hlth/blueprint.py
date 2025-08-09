#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - Flask Blueprint Integration
APG-integrated Flask blueprint with composition engine registration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from flask import Blueprint, request, jsonify, render_template
from flask_appbuilder import AppBuilder, SQLA
from flask_appbuilder.security.manager import BaseSecurityManager

from .views import (
	HealthDashboardView, HealthMetricView, HealthAlertView, 
	SystemComponentView, HealthReportView, HealthRuleView,
	HealthAnalyticsView, HealthChartView
)
from .api import health_api_bp
from .models import *  # Import all models for SQLAlchemy registration


class HealthManagementBlueprint:
	"""APG System Health Management Blueprint with Flask-AppBuilder integration"""
	
	def __init__(self, appbuilder: AppBuilder, db: SQLA):
		self.appbuilder = appbuilder
		self.db = db
		
		# Create main blueprint
		self.bp = Blueprint(
			'health_management',
			__name__,
			template_folder='templates',
			static_folder='static',
			url_prefix='/health'
		)
		
		# Initialize blueprint
		self._register_views()
		self._register_api()
		self._register_permissions()
		self._register_menu_items()
	
	def _register_views(self):
		"""Register Flask-AppBuilder views"""
		
		# Dashboard Views
		self.appbuilder.add_view_no_menu(HealthDashboardView)
		
		# Data Management Views
		self.appbuilder.add_view(
			HealthMetricView,
			"Health Metrics",
			icon="fa-line-chart",
			category="Health Management",
			category_icon="fa-heartbeat"
		)
		
		self.appbuilder.add_view(
			HealthAlertView,
			"Health Alerts",
			icon="fa-exclamation-triangle",
			category="Health Management"
		)
		
		self.appbuilder.add_view(
			SystemComponentView,
			"System Components",
			icon="fa-server",
			category="Health Management"
		)
		
		self.appbuilder.add_view(
			HealthReportView,
			"Health Reports",
			icon="fa-file-text-o",
			category="Health Management"
		)
		
		self.appbuilder.add_view(
			HealthRuleView,
			"Health Rules",
			icon="fa-cogs",
			category="Health Management"
		)
		
		# Analytics Views
		self.appbuilder.add_view_no_menu(HealthAnalyticsView)
		
		# Chart Views
		self.appbuilder.add_view(
			HealthChartView,
			"Health Charts",
			icon="fa-bar-chart",
			category="Health Analytics",
			category_icon="fa-analytics"
		)
	
	def _register_api(self):
		"""Register API blueprint"""
		self.appbuilder.app.register_blueprint(health_api_bp)
	
	def _register_permissions(self):
		"""Register health management permissions"""
		
		# View permissions
		self.appbuilder.sm.add_permission_view_menu('can_list', 'HealthMetricView')
		self.appbuilder.sm.add_permission_view_menu('can_show', 'HealthMetricView')
		self.appbuilder.sm.add_permission_view_menu('can_add', 'HealthMetricView')
		self.appbuilder.sm.add_permission_view_menu('can_edit', 'HealthMetricView')
		self.appbuilder.sm.add_permission_view_menu('can_delete', 'HealthMetricView')
		
		self.appbuilder.sm.add_permission_view_menu('can_list', 'HealthAlertView')
		self.appbuilder.sm.add_permission_view_menu('can_show', 'HealthAlertView')
		self.appbuilder.sm.add_permission_view_menu('can_edit', 'HealthAlertView')
		
		self.appbuilder.sm.add_permission_view_menu('can_list', 'SystemComponentView')
		self.appbuilder.sm.add_permission_view_menu('can_show', 'SystemComponentView')
		self.appbuilder.sm.add_permission_view_menu('can_edit', 'SystemComponentView')
		
		self.appbuilder.sm.add_permission_view_menu('can_list', 'HealthReportView')
		self.appbuilder.sm.add_permission_view_menu('can_show', 'HealthReportView')
		self.appbuilder.sm.add_permission_view_menu('can_add', 'HealthReportView')
		
		self.appbuilder.sm.add_permission_view_menu('can_list', 'HealthRuleView')
		self.appbuilder.sm.add_permission_view_menu('can_show', 'HealthRuleView')
		self.appbuilder.sm.add_permission_view_menu('can_add', 'HealthRuleView')
		self.appbuilder.sm.add_permission_view_menu('can_edit', 'HealthRuleView')
		self.appbuilder.sm.add_permission_view_menu('can_delete', 'HealthRuleView')
		
		# Dashboard permissions
		self.appbuilder.sm.add_permission_view_menu('can_executive', 'HealthDashboardView')
		self.appbuilder.sm.add_permission_view_menu('can_operational', 'HealthDashboardView')
		self.appbuilder.sm.add_permission_view_menu('can_predictive', 'HealthDashboardView')
		
		# Analytics permissions
		self.appbuilder.sm.add_permission_view_menu('can_multi_dimensional', 'HealthAnalyticsView')
		self.appbuilder.sm.add_permission_view_menu('can_predictions', 'HealthAnalyticsView')
		
		# Custom action permissions
		self.appbuilder.sm.add_permission_view_menu('can_acknowledge_alerts', 'HealthAlertView')
		self.appbuilder.sm.add_permission_view_menu('can_resolve_alerts', 'HealthAlertView')
		self.appbuilder.sm.add_permission_view_menu('can_health_assessment', 'SystemComponentView')
		self.appbuilder.sm.add_permission_view_menu('can_generate_report', 'HealthReportView')
		self.appbuilder.sm.add_permission_view_menu('can_enable_rules', 'HealthRuleView')
		self.appbuilder.sm.add_permission_view_menu('can_disable_rules', 'HealthRuleView')
	
	def _register_menu_items(self):
		"""Register menu items"""
		
		# Main Health Dashboard
		self.appbuilder.add_link(
			"Executive Dashboard",
			href="/health/dashboard/executive",
			icon="fa-dashboard",
			category="Health Dashboards",
			category_icon="fa-tachometer"
		)
		
		self.appbuilder.add_link(
			"Operational Dashboard", 
			href="/health/dashboard/operational",
			icon="fa-cogs",
			category="Health Dashboards"
		)
		
		self.appbuilder.add_link(
			"Predictive Dashboard",
			href="/health/dashboard/predictive", 
			icon="fa-crystal-ball",
			category="Health Dashboards"
		)
		
		# Analytics Links
		self.appbuilder.add_link(
			"Multi-Dimensional Analysis",
			href="/health/analytics/multi-dimensional",
			icon="fa-cube",
			category="Health Analytics"
		)
		
		self.appbuilder.add_link(
			"Health Predictions",
			href="/health/analytics/predictions",
			icon="fa-line-chart",
			category="Health Analytics"
		)
		
		# API Documentation Link
		self.appbuilder.add_link(
			"API Documentation",
			href="/health/api/docs",
			icon="fa-book",
			category="Health Management"
		)
	
	def _register_roles(self):
		"""Register health management roles"""
		
		# Health Administrator Role
		health_admin_role = self.appbuilder.sm.add_role('Health Administrator')
		
		# Add all health permissions to admin role
		health_admin_permissions = [
			'can_list_HealthMetricView',
			'can_show_HealthMetricView', 
			'can_add_HealthMetricView',
			'can_edit_HealthMetricView',
			'can_delete_HealthMetricView',
			'can_list_HealthAlertView',
			'can_show_HealthAlertView',
			'can_edit_HealthAlertView',
			'can_acknowledge_alerts_HealthAlertView',
			'can_resolve_alerts_HealthAlertView',
			'can_list_SystemComponentView',
			'can_show_SystemComponentView',
			'can_edit_SystemComponentView',
			'can_health_assessment_SystemComponentView',
			'can_list_HealthReportView',
			'can_show_HealthReportView',
			'can_add_HealthReportView',
			'can_generate_report_HealthReportView',
			'can_list_HealthRuleView',
			'can_show_HealthRuleView',
			'can_add_HealthRuleView',
			'can_edit_HealthRuleView',
			'can_delete_HealthRuleView',
			'can_enable_rules_HealthRuleView',
			'can_disable_rules_HealthRuleView',
			'can_executive_HealthDashboardView',
			'can_operational_HealthDashboardView',
			'can_predictive_HealthDashboardView',
			'can_multi_dimensional_HealthAnalyticsView',
			'can_predictions_HealthAnalyticsView'
		]
		
		for permission_name in health_admin_permissions:
			permission = self.appbuilder.sm.find_permission_view_menu(permission_name, None)
			if permission:
				self.appbuilder.sm.add_permission_role(health_admin_role, permission)
		
		# Health Operator Role (read-only + alert management)
		health_operator_role = self.appbuilder.sm.add_role('Health Operator')
		
		health_operator_permissions = [
			'can_list_HealthMetricView',
			'can_show_HealthMetricView',
			'can_list_HealthAlertView',
			'can_show_HealthAlertView',
			'can_acknowledge_alerts_HealthAlertView',
			'can_list_SystemComponentView',
			'can_show_SystemComponentView',
			'can_list_HealthReportView',
			'can_show_HealthReportView',
			'can_list_HealthRuleView',
			'can_show_HealthRuleView',
			'can_executive_HealthDashboardView',
			'can_operational_HealthDashboardView'
		]
		
		for permission_name in health_operator_permissions:
			permission = self.appbuilder.sm.find_permission_view_menu(permission_name, None)
			if permission:
				self.appbuilder.sm.add_permission_role(health_operator_role, permission)
		
		# Health Viewer Role (read-only)
		health_viewer_role = self.appbuilder.sm.add_role('Health Viewer')
		
		health_viewer_permissions = [
			'can_list_HealthMetricView',
			'can_show_HealthMetricView',
			'can_list_HealthAlertView', 
			'can_show_HealthAlertView',
			'can_list_SystemComponentView',
			'can_show_SystemComponentView',
			'can_list_HealthReportView',
			'can_show_HealthReportView',
			'can_executive_HealthDashboardView'
		]
		
		for permission_name in health_viewer_permissions:
			permission = self.appbuilder.sm.find_permission_view_menu(permission_name, None)
			if permission:
				self.appbuilder.sm.add_permission_role(health_viewer_role, permission)
	
	def get_blueprint(self) -> Blueprint:
		"""Get the Flask blueprint"""
		return self.bp
	
	def initialize_database(self):
		"""Initialize database tables"""
		try:
			# Create all health management tables
			self.db.create_all()
			
			# Initialize default health rules and policies
			self._create_default_health_rules()
			
			print("[HLTH] Database initialized successfully")
		
		except Exception as e:
			print(f"[HLTH] Database initialization failed: {str(e)}")
			raise
	
	def _create_default_health_rules(self):
		"""Create default health rules"""
		try:
			from .models import HealthRule, HealthDimension, HealthSeverity
			
			default_rules = [
				{
					'name': 'High CPU Utilization',
					'description': 'Alert when CPU utilization exceeds 90%',
					'dimension': HealthDimension.PERFORMANCE,
					'metric_pattern': 'cpu_utilization',
					'threshold_value': 90.0,
					'threshold_operator': 'gt',
					'severity': HealthSeverity.HIGH,
					'enabled': True,
					'tenant_id': 'default'
				},
				{
					'name': 'High Memory Utilization', 
					'description': 'Alert when memory utilization exceeds 90%',
					'dimension': HealthDimension.PERFORMANCE,
					'metric_pattern': 'memory_utilization',
					'threshold_value': 90.0,
					'threshold_operator': 'gt', 
					'severity': HealthSeverity.HIGH,
					'enabled': True,
					'tenant_id': 'default'
				},
				{
					'name': 'High Disk Utilization',
					'description': 'Alert when disk utilization exceeds 90%',
					'dimension': HealthDimension.RESOURCE_UTILIZATION,
					'metric_pattern': 'disk_utilization',
					'threshold_value': 90.0,
					'threshold_operator': 'gt',
					'severity': HealthSeverity.HIGH,
					'enabled': True,
					'tenant_id': 'default'
				},
				{
					'name': 'Low Availability',
					'description': 'Alert when availability falls below 99%',
					'dimension': HealthDimension.AVAILABILITY,
					'metric_pattern': 'availability',
					'threshold_value': 99.0,
					'threshold_operator': 'lt',
					'severity': HealthSeverity.CRITICAL,
					'enabled': True,
					'tenant_id': 'default'
				},
				{
					'name': 'High Error Rate',
					'description': 'Alert when error rate exceeds 5%',
					'dimension': HealthDimension.RELIABILITY,
					'metric_pattern': 'error_rate',
					'threshold_value': 0.05,
					'threshold_operator': 'gt',
					'severity': HealthSeverity.HIGH,
					'enabled': True,
					'tenant_id': 'default'
				}
			]
			
			for rule_data in default_rules:
				existing_rule = self.db.session.query(HealthRule).filter_by(
					name=rule_data['name'],
					tenant_id=rule_data['tenant_id']
				).first()
				
				if not existing_rule:
					health_rule = HealthRule(**rule_data)
					self.db.session.add(health_rule)
			
			self.db.session.commit()
			print("[HLTH] Default health rules created successfully")
		
		except Exception as e:
			print(f"[HLTH] Failed to create default health rules: {str(e)}")
			self.db.session.rollback()


def create_health_blueprint(appbuilder: AppBuilder, db: SQLA) -> HealthManagementBlueprint:
	"""Factory function to create health management blueprint"""
	health_blueprint = HealthManagementBlueprint(appbuilder, db)
	return health_blueprint


# Export for APG integration
__all__ = ['HealthManagementBlueprint', 'create_health_blueprint']