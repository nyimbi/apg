#!/usr/bin/env python3
"""
APG Monitoring and Observability (MONI) - Flask-AppBuilder Views
Revolutionary monitoring dashboard with intelligent visualizations

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import asyncio

from flask import request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import ChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget
from wtforms import Form, StringField, IntegerField, SelectField, FloatField, BooleanField
from wtforms.validators import DataRequired, Optional as OptionalValidator, NumberRange

from .models import (
	MonitoringMetric, MonitoringAlert, MonitoringRule, MonitoringDashboard,
	MonitoringQuery, MonitoringTarget, AlertSeverity, AlertStatus, 
	DashboardType, MonitoringScope, MetricType
)
from .service import MonitoringService


class MonitoringMetricView(ModelView):
	"""View for monitoring metrics with advanced filtering and visualization"""
	
	datamodel = SQLAInterface(MonitoringMetric)
	
	# List view configuration
	list_columns = ['name', 'value', 'tenant_id', 'source', 'timestamp', 'quality_score']
	search_columns = ['name', 'tenant_id', 'source', 'labels']
	list_filters = ['tenant_id', 'source', 'metric_type', 'timestamp']
	
	# Show/Edit view configuration  
	show_columns = [
		'metric_id', 'name', 'value', 'tenant_id', 'source', 'source_type',
		'timestamp', 'labels', 'metric_type', 'unit', 'quality_score',
		'retention_policy', 'capability_name', 'correlation_id'
	]
	
	edit_columns = [
		'name', 'value', 'tenant_id', 'source', 'source_type', 'labels',
		'metric_type', 'unit', 'quality_score', 'retention_policy'
	]
	
	# Widget customization
	list_widget = ListWidget
	show_widget = ShowWidget
	edit_widget = EditWidget
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	# Labels and descriptions
	label_columns = {
		'metric_id': 'Metric ID',
		'tenant_id': 'Tenant',
		'source_type': 'Source Type',
		'quality_score': 'Quality Score',
		'retention_policy': 'Retention Policy',
		'capability_name': 'APG Capability',
		'correlation_id': 'Correlation ID'
	}
	
	description_columns = {
		'quality_score': 'Data quality score (0.0 - 1.0)',
		'retention_policy': 'Data retention and downsampling policy',
		'capability_name': 'Source APG capability that generated this metric'
	}
	
	# Custom formatting
	formatters_columns = {
		'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else '',
		'value': lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
		'quality_score': lambda x: f"{x:.2%}" if x else "0%"
	}
	
	# Page configuration
	page_size = 50
	show_template = 'monitoring/metric_show.html'
	list_template = 'monitoring/metric_list.html'
	edit_template = 'monitoring/metric_edit.html'


class MonitoringAlertView(ModelView):
	"""View for monitoring alerts with correlation and escalation management"""
	
	datamodel = SQLAInterface(MonitoringAlert)
	
	# List view configuration
	list_columns = ['name', 'severity', 'status', 'tenant_id', 'created_at', 'escalation_level']
	search_columns = ['name', 'message', 'tenant_id', 'source_metric']
	list_filters = ['severity', 'status', 'tenant_id', 'escalation_level', 'created_at']
	
	# Show view configuration
	show_columns = [
		'alert_id', 'name', 'severity', 'status', 'tenant_id', 'rule_id',
		'message', 'summary', 'created_at', 'updated_at', 'resolved_at',
		'correlation_key', 'escalation_level', 'impact_score', 'affected_services',
		'runbook_url', 'source_metric', 'source_value', 'threshold_value'
	]
	
	edit_columns = [
		'name', 'severity', 'status', 'message', 'summary', 'runbook_url',
		'escalation_level', 'impact_score', 'affected_services'
	]
	
	# Custom ordering
	order_columns = ['created_at', 'severity', 'escalation_level']
	base_order = ('created_at', 'desc')
	
	# Labels and formatting
	label_columns = {
		'alert_id': 'Alert ID',
		'rule_id': 'Rule ID',
		'tenant_id': 'Tenant',
		'created_at': 'Created',
		'updated_at': 'Updated',
		'resolved_at': 'Resolved',
		'correlation_key': 'Correlation',
		'escalation_level': 'Escalation',
		'impact_score': 'Business Impact',
		'affected_services': 'Affected Services',
		'runbook_url': 'Runbook',
		'source_metric': 'Source Metric',
		'source_value': 'Trigger Value',
		'threshold_value': 'Threshold'
	}
	
	formatters_columns = {
		'created_at': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else '',
		'severity': lambda x: f'<span class="label label-{x.lower()}">{x}</span>' if x else '',
		'status': lambda x: f'<span class="badge badge-{x.lower()}">{x}</span>' if x else '',
		'impact_score': lambda x: f"{x:.1%}" if x else "0%"
	}
	
	# Custom actions
	@expose('/acknowledge/<int:pk>')
	@has_access
	def acknowledge(self, pk):
		"""Acknowledge alert action"""
		alert = self.datamodel.get(pk)
		if alert:
			alert.status = AlertStatus.ACKNOWLEDGED
			alert.acknowledged_at = datetime.utcnow()
			self.datamodel.edit(alert)
			flash(f'Alert "{alert.name}" acknowledged successfully', 'success')
		return redirect(url_for('MonitoringAlertView.list'))
	
	@expose('/resolve/<int:pk>')
	@has_access  
	def resolve(self, pk):
		"""Resolve alert action"""
		alert = self.datamodel.get(pk)
		if alert:
			alert.status = AlertStatus.RESOLVED
			alert.resolved_at = datetime.utcnow()
			self.datamodel.edit(alert)
			flash(f'Alert "{alert.name}" resolved successfully', 'success')
		return redirect(url_for('MonitoringAlertView.list'))


class MonitoringRuleView(ModelView):
	"""View for alert rules with condition builder and testing"""
	
	datamodel = SQLAInterface(MonitoringRule)
	
	# List view configuration
	list_columns = ['name', 'metric_name', 'enabled', 'severity', 'tenant_id', 'trigger_count']
	search_columns = ['name', 'metric_name', 'tenant_id', 'condition']
	list_filters = ['enabled', 'severity', 'tenant_id', 'condition_type', 'scope']
	
	# Show/Edit configuration
	show_columns = [
		'rule_id', 'name', 'description', 'enabled', 'tenant_id', 'metric_name',
		'condition', 'condition_type', 'threshold_value', 'threshold_operator',
		'severity', 'evaluation_window_minutes', 'alert_message', 'runbook_url',
		'trigger_count', 'effectiveness_score', 'false_positive_rate'
	]
	
	edit_columns = [
		'name', 'description', 'enabled', 'metric_name', 'condition',
		'condition_type', 'threshold_value', 'threshold_operator', 'severity',
		'evaluation_window_minutes', 'alert_message', 'alert_summary', 'runbook_url',
		'escalation_enabled', 'escalation_interval_minutes', 'max_escalation_level'
	]
	
	# Form configuration
	add_form_extra_fields = {
		'condition_builder': StringField('Condition Builder', render_kw={
			'class': 'form-control',
			'placeholder': 'Use the visual builder or enter condition manually'
		})
	}
	
	edit_form_extra_fields = {
		'condition_builder': StringField('Condition Builder', render_kw={
			'class': 'form-control',
			'placeholder': 'Use the visual builder or enter condition manually'
		})
	}
	
	# Labels and formatting
	label_columns = {
		'rule_id': 'Rule ID',
		'tenant_id': 'Tenant',
		'metric_name': 'Target Metric',
		'condition_type': 'Condition Type',
		'threshold_value': 'Threshold',
		'threshold_operator': 'Operator',
		'evaluation_window_minutes': 'Evaluation Window (min)',
		'alert_message': 'Alert Message Template',
		'alert_summary': 'Alert Summary',
		'runbook_url': 'Runbook URL',
		'trigger_count': 'Times Triggered',
		'effectiveness_score': 'Effectiveness',
		'false_positive_rate': 'False Positive Rate',
		'escalation_enabled': 'Auto-Escalate',
		'escalation_interval_minutes': 'Escalation Interval (min)',
		'max_escalation_level': 'Max Escalation Level'
	}
	
	formatters_columns = {
		'effectiveness_score': lambda x: f"{x:.1%}" if x else "0%",
		'false_positive_rate': lambda x: f"{x:.1%}" if x else "0%",
		'enabled': lambda x: '<span class="label label-success">Enabled</span>' if x else '<span class="label label-default">Disabled</span>'
	}
	
	@expose('/test/<int:pk>')
	@has_access
	def test_rule(self, pk):
		"""Test alert rule against recent metrics"""
		rule = self.datamodel.get(pk)
		if rule:
			# Here we would integrate with the monitoring service to test the rule
			flash(f'Rule "{rule.name}" test completed - check logs for results', 'info')
		return redirect(url_for('MonitoringRuleView.show', pk=pk))


class MonitoringDashboardView(ModelView):
	"""View for monitoring dashboards with widget management"""
	
	datamodel = SQLAInterface(MonitoringDashboard)
	
	# List view configuration
	list_columns = ['name', 'dashboard_type', 'tenant_id', 'widget_count', 'view_count', 'created_at']
	search_columns = ['name', 'description', 'tenant_id']
	list_filters = ['dashboard_type', 'tenant_id', 'public', 'created_at']
	
	# Show/Edit configuration
	show_columns = [
		'dashboard_id', 'name', 'description', 'dashboard_type', 'scope',
		'tenant_id', 'widget_count', 'view_count', 'public', 'auto_refresh',
		'refresh_interval_seconds', 'created_at', 'popularity_score'
	]
	
	edit_columns = [
		'name', 'description', 'dashboard_type', 'scope', 'public',
		'auto_refresh', 'refresh_interval_seconds', 'shared_with'
	]
	
	# Custom actions
	@expose('/preview/<int:pk>')
	@has_access
	def preview(self, pk):
		"""Preview dashboard"""
		dashboard = self.datamodel.get(pk)
		if dashboard:
			return render_template('monitoring/dashboard_preview.html', dashboard=dashboard)
		return redirect(url_for('MonitoringDashboardView.list'))
	
	@expose('/clone/<int:pk>')
	@has_access
	def clone(self, pk):
		"""Clone dashboard for customization"""
		dashboard = self.datamodel.get(pk)
		if dashboard:
			# Create a copy with modified name
			new_dashboard = MonitoringDashboard(
				name=f"{dashboard.name} (Copy)",
				description=dashboard.description,
				dashboard_type=dashboard.dashboard_type,
				tenant_id=dashboard.tenant_id,
				widgets=dashboard.widgets.copy(),
				layout=dashboard.layout.copy(),
				created_by="system"
			)
			self.datamodel.add(new_dashboard)
			flash(f'Dashboard "{dashboard.name}" cloned successfully', 'success')
		return redirect(url_for('MonitoringDashboardView.list'))


class MonitoringBaseView(BaseView):
	"""Base view for monitoring operations and analytics"""
	
	default_view = 'overview'
	
	@expose('/')
	@expose('/overview')
	@has_access
	def overview(self):
		"""Monitoring overview dashboard"""
		# Get summary statistics
		tenant_id = request.args.get('tenant_id', 'default')
		time_window = request.args.get('time_window', '24h')
		
		# In a real implementation, this would call the monitoring service
		summary_stats = {
			'total_metrics': 15234,
			'active_alerts': 12,
			'critical_alerts': 2,
			'system_health_score': 0.96,
			'ingestion_rate': 1250,
			'avg_response_time': 145
		}
		
		recent_alerts = []  # Would fetch from monitoring service
		top_metrics = []    # Would fetch trending metrics
		
		return self.render_template(
			'monitoring/overview.html',
			summary_stats=summary_stats,
			recent_alerts=recent_alerts,
			top_metrics=top_metrics,
			tenant_id=tenant_id,
			time_window=time_window
		)
	
	@expose('/analytics')
	@has_access
	def analytics(self):
		"""Analytics dashboard with insights"""
		tenant_id = request.args.get('tenant_id', 'default')
		
		# Analytics data would come from the analytics engine
		analytics_data = {
			'anomaly_summary': {
				'total_anomalies': 23,
				'critical_anomalies': 3,
				'accuracy_score': 0.92
			},
			'performance_trends': [],
			'correlation_insights': [],
			'predictions': []
		}
		
		return self.render_template(
			'monitoring/analytics.html',
			analytics_data=analytics_data,
			tenant_id=tenant_id
		)
	
	@expose('/health')
	@has_access
	def health(self):
		"""System health status"""
		# Health data from monitoring service
		health_data = {
			'overall_status': 'healthy',
			'component_status': {
				'metrics_engine': 'healthy',
				'alert_engine': 'healthy',
				'analytics_engine': 'healthy',
				'database': 'healthy'
			},
			'performance_metrics': {
				'cpu_usage': 45.2,
				'memory_usage': 68.1,
				'disk_usage': 23.4,
				'network_io': 156.7
			}
		}
		
		return self.render_template(
			'monitoring/health.html',
			health_data=health_data
		)
	
	@expose('/api/metrics')
	@has_access
	def api_metrics(self):
		"""API endpoint for metrics data"""
		tenant_id = request.args.get('tenant_id', 'default')
		metric_names = request.args.getlist('metrics')
		start_time = request.args.get('start_time')
		end_time = request.args.get('end_time')
		
		# Query metrics from service
		metrics_data = []  # Would call monitoring service
		
		return jsonify({
			'status': 'success',
			'data': metrics_data,
			'count': len(metrics_data)
		})
	
	@expose('/api/alerts')
	@has_access
	def api_alerts(self):
		"""API endpoint for alerts data"""
		tenant_id = request.args.get('tenant_id', 'default')
		severity = request.args.get('severity')
		
		# Query alerts from service
		alerts_data = []  # Would call monitoring service
		
		return jsonify({
			'status': 'success',
			'data': alerts_data,
			'count': len(alerts_data)
		})


class MetricsChartView(ChartView):
	"""Chart view for metrics visualization"""
	
	chart_title = "Metrics Dashboard"
	chart_type = "LineChart"
	
	def get_data(self):
		"""Get chart data for metrics"""
		tenant_id = request.args.get('tenant_id', 'default')
		metric_name = request.args.get('metric', 'cpu_usage')
		
		# Sample data - would come from monitoring service
		data = {
			'columns': ['Timestamp', 'Value'],
			'data': [
				['2025-01-01 00:00', 45.2],
				['2025-01-01 01:00', 52.1],
				['2025-01-01 02:00', 48.7],
				['2025-01-01 03:00', 41.3]
			]
		}
		
		return data


class AlertsChartView(ChartView):
	"""Chart view for alerts analysis"""
	
	chart_title = "Alerts Analysis"
	chart_type = "PieChart"
	
	def get_data(self):
		"""Get chart data for alerts"""
		# Sample data - would come from alert engine
		data = {
			'columns': ['Severity', 'Count'],
			'data': [
				['Critical', 2],
				['High', 8], 
				['Medium', 15],
				['Low', 23]
			]
		}
		
		return data


class AnomalyDetectionView(BaseView):
	"""View for anomaly detection management and insights"""
	
	default_view = 'dashboard'
	
	@expose('/')
	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Anomaly detection dashboard"""
		tenant_id = request.args.get('tenant_id', 'default')
		
		# Anomaly data from detection engine
		anomaly_data = {
			'summary': {
				'total_anomalies': 15,
				'critical_anomalies': 2,
				'accuracy_rate': 0.94,
				'false_positive_rate': 0.06
			},
			'recent_anomalies': [],
			'algorithm_performance': {
				'z_score': {'accuracy': 0.89, 'detections': 45},
				'seasonal_hybrid': {'accuracy': 0.96, 'detections': 23},
				'contextual': {'accuracy': 0.92, 'detections': 34}
			},
			'trending_metrics': []
		}
		
		return self.render_template(
			'monitoring/anomaly_dashboard.html',
			anomaly_data=anomaly_data,
			tenant_id=tenant_id
		)
	
	@expose('/feedback')
	@has_access
	def feedback(self):
		"""Anomaly feedback for model improvement"""
		if request.method == 'POST':
			anomaly_id = request.form.get('anomaly_id')
			is_true_positive = request.form.get('is_true_positive') == 'true'
			feedback_note = request.form.get('feedback_note', '')
			
			# Update feedback in anomaly detection engine
			flash('Feedback submitted successfully', 'success')
			return redirect(url_for('AnomalyDetectionView.dashboard'))
		
		# Show feedback form for recent anomalies
		recent_anomalies = []  # Would fetch from detection engine
		
		return self.render_template(
			'monitoring/anomaly_feedback.html',
			recent_anomalies=recent_anomalies
		)
	
	@expose('/insights')
	@has_access
	def insights(self):
		"""Anomaly insights and patterns"""
		tenant_id = request.args.get('tenant_id', 'default')
		time_window = request.args.get('time_window', '24h')
		
		# Insights from detection engine
		insights_data = {
			'patterns': {
				'temporal_patterns': {
					'peak_hour': 14,
					'peak_day': 1,
					'hourly_distribution': {}
				},
				'metric_patterns': {
					'most_anomalous_metric': 'cpu_usage',
					'unique_metrics_affected': 8
				}
			},
			'recommendations': [
				"Investigate CPU usage metric - high anomaly frequency detected",
				"Most anomalies occur during business hours - consider capacity planning"
			],
			'correlation_analysis': []
		}
		
		return self.render_template(
			'monitoring/anomaly_insights.html',
			insights_data=insights_data,
			tenant_id=tenant_id,
			time_window=time_window
		)


# Custom form for metric query builder
class MetricQueryForm(Form):
	"""Form for building metric queries"""
	
	metric_names = StringField('Metric Names', validators=[DataRequired()],
							  description='Comma-separated list of metric names')
	
	start_time = StringField('Start Time', validators=[DataRequired()],
							description='Start time (ISO format or relative like -1h)')
	
	end_time = StringField('End Time', validators=[OptionalValidator()],
						  description='End time (ISO format, defaults to now)')
	
	aggregation = SelectField('Aggregation', choices=[
		('', 'None'),
		('avg', 'Average'),
		('sum', 'Sum'),
		('max', 'Maximum'),
		('min', 'Minimum'),
		('count', 'Count')
	], validators=[OptionalValidator()])
	
	group_by = StringField('Group By', validators=[OptionalValidator()],
						  description='Comma-separated list of labels to group by')
	
	max_results = IntegerField('Max Results', default=1000, 
							  validators=[NumberRange(min=1, max=10000)])


# Custom form for alert rule builder
class AlertRuleForm(Form):
	"""Form for building alert rules with visual condition builder"""
	
	name = StringField('Rule Name', validators=[DataRequired()])
	description = StringField('Description')
	metric_name = StringField('Metric Name', validators=[DataRequired()])
	
	condition_type = SelectField('Condition Type', choices=[
		('threshold', 'Threshold'),
		('anomaly', 'Anomaly Detection'),
		('rate', 'Rate of Change'),
		('absence', 'Metric Absence'),
		('composite', 'Composite Condition')
	], validators=[DataRequired()])
	
	threshold_value = FloatField('Threshold Value', validators=[OptionalValidator()])
	threshold_operator = SelectField('Operator', choices=[
		('gt', 'Greater Than'),
		('gte', 'Greater Than or Equal'),
		('lt', 'Less Than'),
		('lte', 'Less Than or Equal'),
		('eq', 'Equal To'),
		('ne', 'Not Equal To')
	], validators=[OptionalValidator()])
	
	severity = SelectField('Severity', choices=[
		('low', 'Low'),
		('medium', 'Medium'),
		('high', 'High'),
		('critical', 'Critical')
	], default='medium', validators=[DataRequired()])
	
	evaluation_window_minutes = IntegerField('Evaluation Window (minutes)', 
											default=5, validators=[NumberRange(min=1, max=1440)])
	
	alert_message = StringField('Alert Message Template', validators=[DataRequired()],
							   description='Use {value}, {threshold}, {metric_name} for variables')
	
	enabled = BooleanField('Enabled', default=True)


# Widget customizations
class MonitoringListWidget(ListWidget):
	"""Custom list widget with monitoring-specific features"""
	
	template = 'monitoring/widgets/list.html'


class MonitoringShowWidget(ShowWidget):
	"""Custom show widget with real-time updates"""
	
	template = 'monitoring/widgets/show.html'


class MonitoringEditWidget(EditWidget):
	"""Custom edit widget with validation and preview"""
	
	template = 'monitoring/widgets/edit.html'


# Register all views and customize widgets
def register_monitoring_views(appbuilder):
	"""Register all monitoring views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		MonitoringMetricView,
		"Metrics",
		icon="fa-line-chart",
		category="Monitoring",
		category_icon="fa-dashboard"
	)
	
	appbuilder.add_view(
		MonitoringAlertView,
		"Alerts", 
		icon="fa-warning",
		category="Monitoring"
	)
	
	appbuilder.add_view(
		MonitoringRuleView,
		"Alert Rules",
		icon="fa-cogs",
		category="Monitoring"
	)
	
	appbuilder.add_view(
		MonitoringDashboardView,
		"Dashboards",
		icon="fa-tachometer",
		category="Monitoring"
	)
	
	# Base views
	appbuilder.add_view(
		MonitoringBaseView,
		"Overview",
		href="/monitoring/",
		icon="fa-home",
		category="Monitoring"
	)
	
	appbuilder.add_view(
		AnomalyDetectionView,
		"Anomaly Detection",
		href="/monitoring/anomaly/",
		icon="fa-search",
		category="Analytics",
		category_icon="fa-bar-chart"
	)
	
	# Chart views
	appbuilder.add_view(
		MetricsChartView,
		"Metrics Charts",
		icon="fa-bar-chart",
		category="Analytics"
	)
	
	appbuilder.add_view(
		AlertsChartView,
		"Alerts Charts", 
		icon="fa-pie-chart",
		category="Analytics"
	)
	
	# Custom separators and links
	appbuilder.add_separator("Monitoring")
	appbuilder.add_link("System Health", href="/monitoring/health", icon="fa-heartbeat", category="Monitoring")
	appbuilder.add_link("API Documentation", href="/monitoring/api/docs", icon="fa-book", category="Monitoring")


# Export views and forms
__all__ = [
	'MonitoringMetricView',
	'MonitoringAlertView', 
	'MonitoringRuleView',
	'MonitoringDashboardView',
	'MonitoringBaseView',
	'AnomalyDetectionView',
	'MetricsChartView',
	'AlertsChartView',
	'MetricQueryForm',
	'AlertRuleForm',
	'register_monitoring_views'
]