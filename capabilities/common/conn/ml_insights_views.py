"""
APG Connection Management ML Insights Views
Flask-AppBuilder views for machine learning insights and analytics

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import json
import logging
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget
from flask_appbuilder.actions import action
from wtforms import Form, StringField, SelectField, IntegerField, TextAreaField, SelectMultipleField
from wtforms.validators import DataRequired, NumberRange, Optional as OptionalValidator
from wtforms.widgets import CheckboxInput, ListWidget as WTFormsListWidget

from .models import CMConnection, CMDataFlow
from .ml_insights import (
	global_ml_insights_engine, generate_ml_insights, get_anomaly_insights,
	get_clustering_insights, forecast_time_series, AnalysisType, InsightSeverity
)

logger = logging.getLogger(__name__)


class MultiCheckboxField(SelectMultipleField):
	"""Custom multiple checkbox field"""
	widget = WTFormsListWidget(prefix_label=False)
	option_widget = CheckboxInput()


class MLAnalysisForm(Form):
	"""Form for configuring ML analysis"""
	connection_id = SelectField('Connection',
								choices=[],
								validators=[DataRequired()],
								render_kw={'class': 'form-control'})

	analysis_types = MultiCheckboxField(
		'Analysis Types',
		choices=[
			('anomaly_detection', 'Anomaly Detection'),
			('clustering', 'Clustering Analysis'),
			('pattern_recognition', 'Pattern Recognition'),
			('time_series_forecasting', 'Time Series Forecasting'),
			('sentiment_analysis', 'Sentiment Analysis'),
			('data_profiling', 'Data Profiling')
		],
		default=['anomaly_detection', 'pattern_recognition', 'data_profiling'],
		render_kw={'class': 'form-check-input'}
	)

	sample_size = IntegerField('Sample Size',
							   validators=[OptionalValidator(), NumberRange(min=100, max=100000)],
							   default=10000,
							   render_kw={'class': 'form-control', 'placeholder': 'Leave empty to use all data'})


class AnomalyConfigForm(Form):
	"""Form for anomaly detection configuration"""
	contamination = StringField('Contamination Rate',
							   default='0.1',
							   render_kw={'class': 'form-control', 'placeholder': '0.1 (10%)'})

	algorithm = SelectField('Algorithm',
						   choices=[
							   ('isolation_forest', 'Isolation Forest'),
							   ('local_outlier', 'Local Outlier Factor'),
							   ('one_class_svm', 'One-Class SVM')
						   ],
						   default='isolation_forest',
						   render_kw={'class': 'form-control'})


class ClusteringConfigForm(Form):
	"""Form for clustering configuration"""
	algorithm = SelectField('Algorithm',
						   choices=[
							   ('kmeans', 'K-Means'),
							   ('dbscan', 'DBSCAN'),
							   ('hierarchical', 'Hierarchical')
						   ],
						   default='kmeans',
						   render_kw={'class': 'form-control'})

	n_clusters = IntegerField('Number of Clusters',
							 validators=[OptionalValidator(), NumberRange(min=2, max=20)],
							 render_kw={'class': 'form-control', 'placeholder': 'Auto-detect if empty'})


class MLInsightsDashboardView(BaseView):
	"""Main ML insights dashboard"""

	route_base = '/ml_insights'
	default_view = 'dashboard'

	@expose('/')
	@has_access
	def dashboard(self):
		"""ML insights dashboard"""
		# Get recent insights summary
		recent_insights = self._get_recent_insights()
		insights_by_severity = self._group_insights_by_severity(recent_insights)
		top_connections = self._get_top_analyzed_connections()

		return self.render_template(
			'ml_insights/dashboard.html',
			recent_insights=recent_insights,
			insights_by_severity=insights_by_severity,
			top_connections=top_connections,
			analysis_types=self._get_analysis_type_stats()
		)

	@expose('/analyze', methods=['GET', 'POST'])
	@has_access
	def analyze(self):
		"""Run ML analysis on connection data"""
		form = MLAnalysisForm()

		# Populate connection choices
		connections = self.datamodel.session.query(CMConnection).all()
		form.connection_id.choices = [(conn.id, conn.name) for conn in connections]

		if request.method == 'POST' and form.validate():
			try:
				# Get sample data (mock for now)
				sample_data = self._get_sample_data(form.connection_id.data, form.sample_size.data)

				# Run analysis
				analysis_types = [AnalysisType(t) for t in form.analysis_types.data]
				insights = []  # Would call generate_ml_insights with real data

				flash(f'ML analysis completed. Generated {len(insights)} insights.', 'success')
				return redirect(url_for('MLInsightsDashboardView.results', job_id='mock_job_123'))

			except Exception as e:
				logger.error(f"Error running ML analysis: {e}")
				flash(f'Analysis failed: {str(e)}', 'error')

		return self.render_template(
			'ml_insights/analyze.html',
			form=form
		)

	@expose('/results/<job_id>')
	@has_access
	def results(self, job_id):
		"""Display ML analysis results"""
		# Mock results for demo
		insights = self._get_mock_insights()

		return self.render_template(
			'ml_insights/results.html',
			job_id=job_id,
			insights=insights,
			summary_stats=self._calculate_results_summary(insights)
		)

	@expose('/anomaly_detection')
	@has_access
	def anomaly_detection(self):
		"""Anomaly detection specific view"""
		form = AnomalyConfigForm()

		return self.render_template(
			'ml_insights/anomaly_detection.html',
			form=form,
			recent_anomalies=self._get_recent_anomalies()
		)

	@expose('/clustering')
	@has_access
	def clustering(self):
		"""Clustering analysis specific view"""
		form = ClusteringConfigForm()

		return self.render_template(
			'ml_insights/clustering.html',
			form=form,
			recent_clusters=self._get_recent_clusters()
		)

	@expose('/patterns')
	@has_access
	def patterns(self):
		"""Pattern recognition view"""
		patterns = self._get_discovered_patterns()

		return self.render_template(
			'ml_insights/patterns.html',
			patterns=patterns,
			pattern_categories=self._group_patterns_by_category(patterns)
		)

	@expose('/forecasting')
	@has_access
	def forecasting(self):
		"""Time series forecasting view"""
		forecasts = self._get_recent_forecasts()

		return self.render_template(
			'ml_insights/forecasting.html',
			forecasts=forecasts
		)

	@expose('/api/run_analysis', methods=['POST'])
	@has_access
	def api_run_analysis(self):
		"""API endpoint to run ML analysis"""
		try:
			data = request.get_json()
			connection_id = data.get('connection_id')
			analysis_types = data.get('analysis_types', ['anomaly_detection'])

			if not connection_id:
				return jsonify({'success': False, 'error': 'Connection ID required'}), 400

			# Mock analysis execution
			job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

			# In real implementation, this would:
			# 1. Queue analysis job
			# 2. Return job ID for tracking
			# 3. Execute analysis asynchronously

			return jsonify({
				'success': True,
				'job_id': job_id,
				'message': 'Analysis started successfully',
				'estimated_completion': '2-5 minutes'
			})

		except Exception as e:
			logger.error(f"Error starting ML analysis: {e}")
			return jsonify({'success': False, 'error': str(e)}), 500

	@expose('/api/analysis_status/<job_id>')
	@has_access
	def api_analysis_status(self, job_id):
		"""Get analysis job status"""
		# Mock status response
		status_info = {
			'job_id': job_id,
			'status': 'completed',
			'progress': 100,
			'insights_generated': 8,
			'started_at': '2025-01-08T15:30:00Z',
			'completed_at': '2025-01-08T15:33:45Z',
			'duration_seconds': 225
		}

		return jsonify(status_info)

	@expose('/api/insights/<connection_id>')
	@has_access
	def api_get_insights(self, connection_id):
		"""Get insights for a connection"""
		insights = self._get_mock_insights()

		return jsonify({
			'connection_id': connection_id,
			'insights': [
				{
					'id': insight['insight_id'],
					'title': insight['title'],
					'description': insight['description'],
					'severity': insight['severity'],
					'confidence': insight['confidence'],
					'analysis_type': insight['analysis_type'],
					'generated_at': insight['generated_at']
				}
				for insight in insights
			],
			'total': len(insights)
		})

	def _get_recent_insights(self) -> List[Dict[str, Any]]:
		"""Get recent ML insights"""
		return self._get_mock_insights()[:10]

	def _group_insights_by_severity(self, insights: List[Dict[str, Any]]) -> Dict[str, int]:
		"""Group insights by severity"""
		severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

		for insight in insights:
			severity = insight.get('severity', 'low')
			severity_counts[severity] = severity_counts.get(severity, 0) + 1

		return severity_counts

	def _get_top_analyzed_connections(self) -> List[Dict[str, Any]]:
		"""Get top analyzed connections"""
		return [
			{'name': 'Production Database', 'insights_count': 15, 'last_analyzed': '2 hours ago'},
			{'name': 'Analytics API', 'insights_count': 12, 'last_analyzed': '4 hours ago'},
			{'name': 'Customer Data Feed', 'insights_count': 8, 'last_analyzed': '1 day ago'},
		]

	def _get_analysis_type_stats(self) -> Dict[str, int]:
		"""Get analysis type statistics"""
		return {
			'anomaly_detection': 25,
			'pattern_recognition': 18,
			'clustering': 12,
			'data_profiling': 20,
			'time_series_forecasting': 8,
			'sentiment_analysis': 5
		}

	def _get_sample_data(self, connection_id: str, sample_size: int = None) -> pd.DataFrame:
		"""Get sample data from connection"""
		# Mock sample data
		import numpy as np

		size = sample_size or 1000
		data = {
			'id': range(1, size + 1),
			'value': np.random.normal(100, 15, size),
			'category': np.random.choice(['A', 'B', 'C'], size),
			'timestamp': pd.date_range('2024-01-01', periods=size, freq='h'),
			'score': np.random.uniform(0, 1, size)
		}

		return pd.DataFrame(data)

	def _get_mock_insights(self) -> List[Dict[str, Any]]:
		"""Generate mock insights for testing"""
		return [
			{
				'insight_id': 'anomaly_001',
				'title': 'Unusual Data Patterns Detected',
				'description': '23 anomalous records detected in the last batch (2.3% of total)',
				'analysis_type': 'anomaly_detection',
				'severity': 'high',
				'confidence': 0.85,
				'generated_at': '2025-01-08T15:30:00Z',
				'evidence': {
					'anomaly_count': 23,
					'anomaly_rate': 0.023,
					'affected_fields': ['value', 'score']
				},
				'recommendations': [
					'Investigate data collection process for affected records',
					'Implement automated anomaly alerts',
					'Review data validation rules'
				]
			},
			{
				'insight_id': 'cluster_002',
				'title': 'Natural Data Groupings Found',
				'description': 'Data naturally segments into 4 distinct clusters with high confidence',
				'analysis_type': 'clustering',
				'severity': 'medium',
				'confidence': 0.92,
				'generated_at': '2025-01-08T15:25:00Z',
				'evidence': {
					'num_clusters': 4,
					'silhouette_score': 0.78,
					'cluster_sizes': [45, 32, 28, 15]
				},
				'recommendations': [
					'Use cluster information for targeted processing',
					'Consider cluster-based data quality rules',
					'Optimize storage based on cluster characteristics'
				]
			},
			{
				'insight_id': 'pattern_003',
				'title': 'Recurring Sequence Pattern',
				'description': 'Identified repeating pattern in category field with 89% confidence',
				'analysis_type': 'pattern_recognition',
				'severity': 'low',
				'confidence': 0.89,
				'generated_at': '2025-01-08T15:20:00Z',
				'evidence': {
					'pattern_type': 'repeating_sequence',
					'pattern': ['A', 'B', 'A', 'C'],
					'occurrences': 12,
					'affected_field': 'category'
				},
				'recommendations': [
					'Leverage pattern for data compression',
					'Use pattern for data validation',
					'Consider pattern-based indexing'
				]
			},
			{
				'insight_id': 'quality_004',
				'title': 'Data Quality Issues Identified',
				'description': '15% of records have missing values in critical fields',
				'analysis_type': 'data_profiling',
				'severity': 'high',
				'confidence': 0.95,
				'generated_at': '2025-01-08T15:15:00Z',
				'evidence': {
					'missing_percentage': 0.15,
					'affected_fields': ['value', 'timestamp'],
					'completeness_score': 85
				},
				'recommendations': [
					'Implement data validation at source',
					'Add missing value imputation strategy',
					'Monitor data completeness metrics'
				]
			},
			{
				'insight_id': 'forecast_005',
				'title': 'Upward Trend Predicted',
				'description': 'Time series forecast shows 12% increase over next 30 days',
				'analysis_type': 'time_series_forecasting',
				'severity': 'medium',
				'confidence': 0.73,
				'generated_at': '2025-01-08T15:10:00Z',
				'evidence': {
					'trend_direction': 'increasing',
					'predicted_change': 0.12,
					'forecast_horizon': 30,
					'model_accuracy': 0.82
				},
				'recommendations': [
					'Plan for increased data volume',
					'Monitor actual vs predicted values',
					'Update capacity planning models'
				]
			}
		]

	def _calculate_results_summary(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Calculate summary statistics for results"""
		if not insights:
			return {}

		severity_counts = self._group_insights_by_severity(insights)
		avg_confidence = sum(insight['confidence'] for insight in insights) / len(insights)

		return {
			'total_insights': len(insights),
			'average_confidence': round(avg_confidence, 3),
			'severity_distribution': severity_counts,
			'analysis_types': list(set(insight['analysis_type'] for insight in insights))
		}

	def _get_recent_anomalies(self) -> List[Dict[str, Any]]:
		"""Get recent anomaly detection results"""
		return [
			{
				'connection': 'Production DB',
				'anomaly_count': 23,
				'anomaly_rate': 0.023,
				'detected_at': '2025-01-08T15:30:00Z',
				'severity': 'high'
			},
			{
				'connection': 'Analytics API',
				'anomaly_count': 8,
				'anomaly_rate': 0.008,
				'detected_at': '2025-01-08T14:15:00Z',
				'severity': 'medium'
			}
		]

	def _get_recent_clusters(self) -> List[Dict[str, Any]]:
		"""Get recent clustering results"""
		return [
			{
				'connection': 'Customer Data',
				'num_clusters': 4,
				'silhouette_score': 0.78,
				'analyzed_at': '2025-01-08T15:25:00Z'
			},
			{
				'connection': 'Transaction Data',
				'num_clusters': 6,
				'silhouette_score': 0.65,
				'analyzed_at': '2025-01-08T13:45:00Z'
			}
		]

	def _get_discovered_patterns(self) -> List[Dict[str, Any]]:
		"""Get discovered patterns"""
		return [
			{
				'pattern_id': 'seq_001',
				'pattern_type': 'repeating_sequence',
				'description': 'Recurring A-B-A-C pattern in category field',
				'confidence': 0.89,
				'frequency': 12,
				'fields': ['category'],
				'discovered_at': '2025-01-08T15:20:00Z'
			},
			{
				'pattern_id': 'const_002',
				'pattern_type': 'constant_value',
				'description': 'Status field always contains "active"',
				'confidence': 1.0,
				'frequency': 1000,
				'fields': ['status'],
				'discovered_at': '2025-01-08T15:18:00Z'
			},
			{
				'pattern_id': 'corr_003',
				'pattern_type': 'positive_correlation',
				'description': 'Strong correlation between value and score (r=0.87)',
				'confidence': 0.87,
				'frequency': 1000,
				'fields': ['value', 'score'],
				'discovered_at': '2025-01-08T15:16:00Z'
			}
		]

	def _group_patterns_by_category(self, patterns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
		"""Group patterns by category"""
		categories = {}

		for pattern in patterns:
			pattern_type = pattern['pattern_type']
			if pattern_type not in categories:
				categories[pattern_type] = []
			categories[pattern_type].append(pattern)

		return categories

	def _get_recent_forecasts(self) -> List[Dict[str, Any]]:
		"""Get recent forecasting results"""
		return [
			{
				'connection': 'Metrics Feed',
				'field': 'daily_volume',
				'trend': 'increasing',
				'predicted_change': 0.12,
				'forecast_horizon': 30,
				'confidence': 0.73,
				'generated_at': '2025-01-08T15:10:00Z'
			},
			{
				'connection': 'User Activity',
				'field': 'session_count',
				'trend': 'stable',
				'predicted_change': 0.02,
				'forecast_horizon': 14,
				'confidence': 0.85,
				'generated_at': '2025-01-08T14:30:00Z'
			}
		]


class MLInsightsChartsView(BaseView):
	"""ML insights charts and visualizations"""

	route_base = '/ml_insights/charts'

	@expose('/api/insights_timeline')
	@has_access
	def api_insights_timeline(self):
		"""API endpoint for insights timeline chart"""
		# Mock timeline data
		timeline_data = {
			'labels': ['Jan 1', 'Jan 2', 'Jan 3', 'Jan 4', 'Jan 5', 'Jan 6', 'Jan 7', 'Jan 8'],
			'datasets': [
				{
					'label': 'Critical',
					'data': [2, 1, 3, 2, 4, 1, 2, 3],
					'backgroundColor': '#dc3545',
					'borderColor': '#dc3545'
				},
				{
					'label': 'High',
					'data': [5, 7, 4, 8, 6, 9, 7, 8],
					'backgroundColor': '#fd7e14',
					'borderColor': '#fd7e14'
				},
				{
					'label': 'Medium',
					'data': [8, 12, 10, 15, 11, 14, 12, 13],
					'backgroundColor': '#ffc107',
					'borderColor': '#ffc107'
				},
				{
					'label': 'Low',
					'data': [15, 18, 20, 17, 22, 19, 21, 20],
					'backgroundColor': '#28a745',
					'borderColor': '#28a745'
				}
			]
		}

		return jsonify(timeline_data)

	@expose('/api/analysis_distribution')
	@has_access
	def api_analysis_distribution(self):
		"""API endpoint for analysis type distribution"""
		distribution_data = {
			'labels': ['Anomaly Detection', 'Pattern Recognition', 'Data Profiling', 'Clustering', 'Forecasting', 'Sentiment'],
			'datasets': [{
				'data': [25, 18, 20, 12, 8, 5],
				'backgroundColor': [
					'#FF6384',
					'#36A2EB',
					'#FFCE56',
					'#4BC0C0',
					'#9966FF',
					'#FF9F40'
				]
			}]
		}

		return jsonify(distribution_data)

	@expose('/api/confidence_scores')
	@has_access
	def api_confidence_scores(self):
		"""API endpoint for confidence score distribution"""
		confidence_data = {
			'labels': ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
			'datasets': [{
				'label': 'Number of Insights',
				'data': [2, 5, 12, 28, 35],
				'backgroundColor': 'rgba(54, 162, 235, 0.6)',
				'borderColor': 'rgba(54, 162, 235, 1)',
				'borderWidth': 1
			}]
		}

		return jsonify(confidence_data)


def init_ml_insights_views(appbuilder):
	"""Initialize ML insights views"""

	# Register main dashboard
	appbuilder.add_view(
		MLInsightsDashboardView,
		"ML Insights",
		icon="fa-brain",
		category="Analytics"
	)

	# Register charts view
	appbuilder.add_view(
		MLInsightsChartsView,
		"ML Charts",
		icon="fa-chart-line",
		category="Analytics"
	)

	logger.info("ML insights views initialized successfully")
