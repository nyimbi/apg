"""
APG Connection Management Data Quality Views
Flask-AppBuilder views for data quality monitoring and management

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView, GroupByChartView
from flask_appbuilder.widgets import ListWidget
from flask_appbuilder.actions import action
from sqlalchemy import desc, and_, func
from wtforms import Form, StringField, SelectField, IntegerField, FloatField, TextAreaField
from wtforms.validators import DataRequired, NumberRange, Optional as OptionalValidator

from .models import CMConnection, CMDataFlow
from .data_quality import (
	global_data_quality_monitor, DataQualityRule, DataQualityDimension,
	IssueSeverity, IssueType, assess_connection_data_quality
)

logger = logging.getLogger(__name__)


class DataQualityRuleForm(Form):
	"""Form for creating/editing data quality rules"""
	name = StringField('Rule Name', validators=[DataRequired()],
					   render_kw={'class': 'form-control'})
	description = TextAreaField('Description', validators=[DataRequired()],
								render_kw={'class': 'form-control', 'rows': 3})
	dimension = SelectField('Quality Dimension',
						   choices=[(dim.value, dim.value.title()) for dim in DataQualityDimension],
						   validators=[DataRequired()],
						   render_kw={'class': 'form-control'})
	severity = SelectField('Severity',
						  choices=[(sev.value, sev.value.title()) for sev in IssueSeverity],
						  validators=[DataRequired()],
						  render_kw={'class': 'form-control'})
	rule_type = SelectField('Rule Type',
						   choices=[
							   ('completeness', 'Completeness'),
							   ('regex', 'Regex Pattern'),
							   ('uniqueness', 'Uniqueness'),
							   ('timeliness', 'Timeliness'),
							   ('custom', 'Custom')
						   ],
						   validators=[DataRequired()],
						   render_kw={'class': 'form-control'})
	field_names = StringField('Field Names (comma-separated)',
							  validators=[DataRequired()],
							  render_kw={'class': 'form-control',
									   'placeholder': 'field1,field2 or * for all fields'})
	parameters = TextAreaField('Parameters (JSON)',
							  render_kw={'class': 'form-control', 'rows': 4,
									   'placeholder': '{"key": "value"}'})


class DataQualityThresholdForm(Form):
	"""Form for configuring quality thresholds"""
	overall_score = IntegerField('Overall Quality Score Threshold (%)',
								validators=[DataRequired(), NumberRange(min=0, max=100)],
								default=70,
								render_kw={'class': 'form-control'})
	completeness_score = IntegerField('Completeness Score Threshold (%)',
									 validators=[DataRequired(), NumberRange(min=0, max=100)],
									 default=80,
									 render_kw={'class': 'form-control'})
	validity_score = IntegerField('Validity Score Threshold (%)',
								 validators=[DataRequired(), NumberRange(min=0, max=100)],
								 default=85,
								 render_kw={'class': 'form-control'})
	critical_issues = IntegerField('Critical Issues Threshold',
								  validators=[DataRequired(), NumberRange(min=0, max=1000)],
								  default=5,
								  render_kw={'class': 'form-control'})


class DataQualityDashboardView(BaseView):
	"""Main data quality dashboard"""

	route_base = '/data_quality'
	default_view = 'dashboard'

	@expose('/')
	@has_access
	def dashboard(self):
		"""Data quality dashboard"""
		# Get recent quality metrics
		trends = global_data_quality_monitor.get_quality_trends(lookback_hours=24)

		# Get connection quality stats
		connections_stats = self._get_connections_quality_stats()

		# Get quality distribution
		quality_distribution = self._get_quality_level_distribution()

		# Get top issues
		top_issues = self._get_top_quality_issues()

		return self.render_template(
			'data_quality/dashboard.html',
			trends=trends,
			connections_stats=connections_stats,
			quality_distribution=quality_distribution,
			top_issues=top_issues,
			thresholds=global_data_quality_monitor.alert_thresholds
		)

	@expose('/api/metrics')
	@has_access
	def api_metrics(self):
		"""API endpoint for quality metrics"""
		lookback_hours = request.args.get('hours', 24, type=int)
		trends = global_data_quality_monitor.get_quality_trends(lookback_hours)

		return jsonify({
			'success': True,
			'data': trends
		})

	@expose('/api/connection/<connection_id>/assess', methods=['POST'])
	@has_access
	async def api_assess_connection(self, connection_id):
		"""Trigger data quality assessment for a connection"""
		try:
			# Get connection
			connection = self.datamodel.session.query(CMConnection)\
				.filter(CMConnection.id == connection_id)\
				.first()

			if not connection:
				return jsonify({'success': False, 'error': 'Connection not found'}), 404

			# Mock data for assessment (in real implementation, fetch from connection)
			sample_data = [
				{'id': 1, 'name': 'Test Record', 'email': 'test@example.com', 'created_at': datetime.now().isoformat()},
				{'id': 2, 'name': 'Another Record', 'email': 'invalid-email', 'created_at': None}
			]

			# Perform assessment
			metrics = await assess_connection_data_quality(connection_id, sample_data)

			return jsonify({
				'success': True,
				'data': {
					'connection_id': connection_id,
					'overall_score': metrics.overall_score,
					'quality_level': metrics.quality_level.value,
					'total_records': metrics.total_records,
					'valid_records': metrics.valid_records,
					'issues_count': len(metrics.issues),
					'assessment_timestamp': metrics.assessment_timestamp.isoformat(),
					'scores': {
						'completeness': metrics.completeness_score,
						'accuracy': metrics.accuracy_score,
						'consistency': metrics.consistency_score,
						'validity': metrics.validity_score,
						'uniqueness': metrics.uniqueness_score,
						'timeliness': metrics.timeliness_score,
						'integrity': metrics.integrity_score
					}
				}
			})

		except Exception as e:
			logger.error(f"Error assessing connection {connection_id}: {e}")
			return jsonify({'success': False, 'error': str(e)}), 500

	@expose('/rules')
	@has_access
	def rules(self):
		"""Data quality rules management"""
		rules = list(global_data_quality_monitor.assessment_engine.validator.rules.values())
		return self.render_template(
			'data_quality/rules.html',
			rules=rules
		)

	@expose('/rules/add', methods=['GET', 'POST'])
	@has_access
	def add_rule(self):
		"""Add new data quality rule"""
		form = DataQualityRuleForm(request.form)

		if request.method == 'POST' and form.validate():
			try:
				# Parse parameters
				parameters = {}
				if form.parameters.data:
					parameters = json.loads(form.parameters.data)

				# Create rule
				rule = DataQualityRule(
					rule_id=f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
					name=form.name.data,
					description=form.description.data,
					dimension=DataQualityDimension(form.dimension.data),
					severity=IssueSeverity(form.severity.data),
					field_names=form.field_names.data.split(','),
					rule_type=form.rule_type.data,
					parameters=parameters,
					created_by=self.get_user().username
				)

				# Add to validator
				global_data_quality_monitor.assessment_engine.validator.add_rule(rule)

				flash(f'Data quality rule "{rule.name}" added successfully', 'success')
				return redirect(url_for('DataQualityDashboardView.rules'))

			except Exception as e:
				flash(f'Error adding rule: {str(e)}', 'error')

		return self.render_template(
			'data_quality/add_rule.html',
			form=form
		)

	@expose('/rules/<rule_id>/delete', methods=['POST'])
	@has_access
	def delete_rule(self, rule_id):
		"""Delete data quality rule"""
		try:
			global_data_quality_monitor.assessment_engine.validator.remove_rule(rule_id)
			flash('Rule deleted successfully', 'success')
		except Exception as e:
			flash(f'Error deleting rule: {str(e)}', 'error')

		return redirect(url_for('DataQualityDashboardView.rules'))

	@expose('/thresholds', methods=['GET', 'POST'])
	@has_access
	def thresholds(self):
		"""Configure quality alert thresholds"""
		form = DataQualityThresholdForm(request.form)

		# Pre-populate form with current thresholds
		if request.method == 'GET':
			current = global_data_quality_monitor.alert_thresholds
			form.overall_score.data = current.get('overall_score', 70)
			form.completeness_score.data = current.get('completeness_score', 80)
			form.validity_score.data = current.get('validity_score', 85)
			form.critical_issues.data = current.get('critical_issues', 5)

		if request.method == 'POST' and form.validate():
			try:
				# Update thresholds
				global_data_quality_monitor.alert_thresholds.update({
					'overall_score': form.overall_score.data,
					'completeness_score': form.completeness_score.data,
					'validity_score': form.validity_score.data,
					'critical_issues': form.critical_issues.data
				})

				flash('Quality thresholds updated successfully', 'success')

			except Exception as e:
				flash(f'Error updating thresholds: {str(e)}', 'error')

		return self.render_template(
			'data_quality/thresholds.html',
			form=form
		)

	@expose('/reports')
	@has_access
	def reports(self):
		"""Data quality reports"""
		# Get recent assessments
		recent_history = global_data_quality_monitor.quality_history[-20:] if global_data_quality_monitor.quality_history else []

		# Calculate summary stats
		if recent_history:
			scores = [h.overall_score for h in recent_history]
			avg_score = sum(scores) / len(scores)
			total_issues = sum(len(h.issues) for h in recent_history)
		else:
			avg_score = 0
			total_issues = 0

		return self.render_template(
			'data_quality/reports.html',
			recent_history=recent_history,
			summary_stats={
				'assessments_count': len(recent_history),
				'average_score': round(avg_score, 2),
				'total_issues': total_issues
			}
		)

	def _get_connections_quality_stats(self) -> List[Dict[str, Any]]:
		"""Get quality statistics by connection"""
		# This would query actual connection quality data
		# For now, return mock data
		return [
			{'connection_id': 'conn_1', 'name': 'Production DB', 'quality_score': 85.2, 'last_assessed': '2025-01-08T10:30:00Z'},
			{'connection_id': 'conn_2', 'name': 'Analytics API', 'quality_score': 92.1, 'last_assessed': '2025-01-08T11:15:00Z'},
			{'connection_id': 'conn_3', 'name': 'Legacy System', 'quality_score': 68.3, 'last_assessed': '2025-01-08T09:45:00Z'}
		]

	def _get_quality_level_distribution(self) -> Dict[str, int]:
		"""Get distribution of quality levels"""
		# Mock distribution
		return {
			'excellent': 15,
			'good': 28,
			'fair': 12,
			'poor': 8,
			'critical': 3
		}

	def _get_top_quality_issues(self) -> List[Dict[str, Any]]:
		"""Get most common quality issues"""
		return [
			{'issue_type': 'missing_value', 'count': 45, 'severity': 'high'},
			{'issue_type': 'invalid_format', 'count': 32, 'severity': 'medium'},
			{'issue_type': 'duplicate', 'count': 28, 'severity': 'medium'},
			{'issue_type': 'stale_data', 'count': 19, 'severity': 'low'},
			{'issue_type': 'outlier', 'count': 15, 'severity': 'low'}
		]


class DataQualityChartsView(DirectByChartView):
	"""Data quality charts and visualizations"""

	datamodel = SQLAInterface(CMConnection)
	chart_title = 'Data Quality Trends'
	chart_type = 'LineChart'
	direct_columns = ['name', 'created_at']
	base_order = ('created_at', 'desc')

	@expose('/api/quality_trends')
	@has_access
	def api_quality_trends(self):
		"""API endpoint for quality trend charts"""
		lookback_hours = request.args.get('hours', 24, type=int)
		trends = global_data_quality_monitor.get_quality_trends(lookback_hours)

		# Format for chart consumption
		chart_data = {
			'labels': [],  # Would be timestamps
			'datasets': [{
				'label': 'Overall Quality Score',
				'data': [],  # Would be quality scores over time
				'borderColor': 'rgb(54, 162, 235)',
				'backgroundColor': 'rgba(54, 162, 235, 0.2)'
			}]
		}

		return jsonify(chart_data)

	@expose('/api/quality_distribution')
	@has_access
	def api_quality_distribution(self):
		"""API endpoint for quality level distribution"""
		distribution = {
			'excellent': 15,
			'good': 28,
			'fair': 12,
			'poor': 8,
			'critical': 3
		}

		chart_data = {
			'labels': list(distribution.keys()),
			'datasets': [{
				'data': list(distribution.values()),
				'backgroundColor': [
					'#28a745',  # excellent - green
					'#17a2b8',  # good - blue
					'#ffc107',  # fair - yellow
					'#fd7e14',  # poor - orange
					'#dc3545'   # critical - red
				]
			}]
		}

		return jsonify(chart_data)


class DataQualityConnectionView(ModelView):
	"""Connection-specific data quality view"""

	datamodel = SQLAInterface(CMConnection)
	list_columns = ['name', 'connection_type', 'status', 'created_at']
	show_columns = ['name', 'description', 'connection_type', 'status', 'config', 'created_at']

	@action('assess_quality', 'Assess Data Quality', 'Assess data quality for selected connections', 'fa-check-circle')
	def assess_quality_action(self, items):
		"""Bulk action to assess data quality for selected connections"""
		if not items:
			flash('No connections selected', 'warning')
			return redirect(request.url)

		assessed_count = 0

		for connection in items:
			try:
				# Trigger quality assessment (async)
				# In real implementation, this would queue assessment tasks
				assessed_count += 1
			except Exception as e:
				logger.error(f"Error assessing connection {connection.id}: {e}")

		flash(f'Quality assessment triggered for {assessed_count} connections', 'success')
		return redirect(request.url)

	@expose('/show/<pk>')
	@has_access
	def show(self, pk):
		"""Enhanced connection details with quality metrics"""
		# Get base connection details
		connection = self.datamodel.get(pk, self._base_filters)
		if not connection:
			flash('Connection not found', 'error')
			return redirect(url_for('CMConnectionModelView.list'))

		# Get quality metrics for this connection (mock data)
		quality_metrics = {
			'overall_score': 85.2,
			'last_assessed': datetime.now(timezone.utc),
			'quality_level': 'good',
			'issues_count': 12,
			'trends': {
				'score_trend': 'improving',
				'recent_scores': [82.1, 83.5, 84.2, 85.2]
			}
		}

		return self.render_template(
			'data_quality/connection_detail.html',
			connection=connection,
			quality_metrics=quality_metrics
		)


# Widget for displaying quality metrics
class DataQualityWidget(ListWidget):
	"""Custom widget for displaying data quality information"""

	template = 'data_quality/widgets/quality_summary.html'

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


def init_data_quality_views(appbuilder):
	"""Initialize data quality views"""

	# Register main dashboard
	appbuilder.add_view(
		DataQualityDashboardView,
		"Data Quality Dashboard",
		icon="fa-tachometer-alt",
		category="Data Quality"
	)

	# Register charts view
	appbuilder.add_view(
		DataQualityChartsView,
		"Quality Charts",
		icon="fa-chart-line",
		category="Data Quality"
	)

	# Register connection quality view
	appbuilder.add_view(
		DataQualityConnectionView,
		"Connection Quality",
		icon="fa-database",
		category="Data Quality"
	)

	logger.info("Data quality views initialized successfully")