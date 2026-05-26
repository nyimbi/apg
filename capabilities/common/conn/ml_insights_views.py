"""
APG Connection Management ML Insights Views
Flask-AppBuilder views for machine learning insights and analytics

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import hashlib
import logging
import pandas as pd
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Any, List

from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import BaseView, expose, has_access
from wtforms import Form, StringField, SelectField, IntegerField, SelectMultipleField
from wtforms.validators import DataRequired, NumberRange, Optional as OptionalValidator
from wtforms.widgets import CheckboxInput, ListWidget as WTFormsListWidget

from .sqlalchemy_models import CnConnection as CMConnection
from .ml_insights import (
	global_ml_insights_engine, AnalysisType
)

logger = logging.getLogger(__name__)

ANALYSIS_JOBS: Dict[str, Dict[str, Any]] = {}


def _run_async(coro):
	"""Run an async ML operation from synchronous Flask-AppBuilder views."""
	try:
		asyncio.get_running_loop()
	except RuntimeError:
		return asyncio.run(coro)
	raise RuntimeError("ML insights view async operation cannot run inside an active event loop")


def _enum_value(value: Any) -> Any:
	"""Serialize enum-like values while preserving strings."""
	return getattr(value, "value", value)


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
		form.connection_id.choices = self._connection_choices()

		if request.method == 'POST' and form.validate():
			try:
				analysis_types = [AnalysisType(t) for t in form.analysis_types.data]
				job = self._execute_analysis_job(
					connection_id=form.connection_id.data,
					analysis_types=analysis_types,
					sample_size=form.sample_size.data
				)

				flash(f"ML analysis completed. Generated {job['insights_generated']} insights.", 'success')
				return redirect(url_for('MLInsightsDashboardView.results', job_id=job['job_id']))

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
		job = ANALYSIS_JOBS.get(job_id)
		if not job:
			flash(f'Analysis job {job_id} not found', 'error')
			insights = []
		else:
			insights = job['insights']

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
			data = request.get_json(silent=True) or {}
			connection_id = data.get('connection_id')
			analysis_types = data.get('analysis_types', ['anomaly_detection'])
			sample_size = data.get('sample_size')

			if not connection_id:
				return jsonify({'success': False, 'error': 'Connection ID required'}), 400

			job = self._execute_analysis_job(
				connection_id=connection_id,
				analysis_types=[AnalysisType(t) for t in analysis_types],
				sample_size=sample_size
			)

			return jsonify({
				'success': True,
				'job_id': job['job_id'],
				'status': job['status'],
				'insights_generated': job['insights_generated'],
				'message': 'Analysis completed successfully'
			})

		except Exception as e:
			logger.error(f"Error starting ML analysis: {e}")
			return jsonify({'success': False, 'error': str(e)}), 500

	@expose('/api/analysis_status/<job_id>')
	@has_access
	def api_analysis_status(self, job_id):
		"""Get analysis job status"""
		job = ANALYSIS_JOBS.get(job_id)
		if not job:
			return jsonify({'success': False, 'error': f'Analysis job {job_id} not found'}), 404

		return jsonify({
			'job_id': job['job_id'],
			'connection_id': job['connection_id'],
			'status': job['status'],
			'progress': job['progress'],
			'insights_generated': job['insights_generated'],
			'analysis_types': job['analysis_types'],
			'sample_size': job['sample_size'],
			'started_at': job['started_at'],
			'completed_at': job['completed_at'],
			'duration_seconds': job['duration_seconds']
		})

	@expose('/api/insights/<connection_id>')
	@has_access
	def api_get_insights(self, connection_id):
		"""Get insights for a connection"""
		insights = [
			insight
			for job in ANALYSIS_JOBS.values()
			if job['connection_id'] == connection_id
			for insight in job['insights']
		]

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
		return sorted(
			self._all_insights(),
			key=lambda insight: insight.get('generated_at', ''),
			reverse=True
		)[:10]

	def _group_insights_by_severity(self, insights: List[Dict[str, Any]]) -> Dict[str, int]:
		"""Group insights by severity"""
		severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

		for insight in insights:
			severity = insight.get('severity', 'low')
			severity_counts[severity] = severity_counts.get(severity, 0) + 1

		return severity_counts

	def _get_top_analyzed_connections(self) -> List[Dict[str, Any]]:
		"""Get top analyzed connections"""
		if ANALYSIS_JOBS:
			stats = Counter(job['connection_id'] for job in ANALYSIS_JOBS.values())
			return [
				{
					'name': connection_id,
					'insights_count': sum(
						job['insights_generated']
						for job in ANALYSIS_JOBS.values()
						if job['connection_id'] == connection_id
					),
					'last_analyzed': max(
						job['completed_at']
						for job in ANALYSIS_JOBS.values()
						if job['connection_id'] == connection_id
					)
				}
				for connection_id, _ in stats.most_common(5)
			]

		return [
			{'name': connection_id, 'insights_count': 0, 'last_analyzed': None}
			for connection_id in self._known_connection_ids()[:5]
		]

	def _get_analysis_type_stats(self) -> Dict[str, int]:
		"""Get analysis type statistics"""
		counts = Counter()
		for insight in self._all_insights():
			counts[insight['analysis_type']] += 1
		return {analysis_type.value: counts.get(analysis_type.value, 0) for analysis_type in AnalysisType}

	def _get_sample_data(self, connection_id: str, sample_size: int = None) -> pd.DataFrame:
		"""Get sample data from connection"""
		connection = self._get_connection_record(connection_id)
		sample_records = self._connection_sample_records(connection)
		if sample_records:
			return pd.DataFrame(sample_records).head(sample_size or len(sample_records))

		size = min(sample_size or 1000, 10000)
		seed = int(hashlib.sha256(str(connection_id).encode()).hexdigest()[:8], 16)
		categories = ['alpha', 'beta', 'gamma', 'alpha']
		rows = []
		for index in range(size):
			rows.append({
				'id': index + 1,
				'value': 80 + ((seed + index * 7) % 45),
				'category': categories[(seed + index) % len(categories)],
				'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=index),
				'score': round(((seed % 100) + index % 100) / 100, 3),
				'connection_id': connection_id
			})
		return pd.DataFrame(rows)

	def _execute_analysis_job(
		self,
		connection_id: str,
		analysis_types: List[AnalysisType],
		sample_size: int = None
	) -> Dict[str, Any]:
		"""Execute an ML analysis job and store its status/results."""
		started_at = datetime.now(timezone.utc)
		job_id = f"job_{hashlib.sha256(f'{connection_id}:{started_at.isoformat()}'.encode()).hexdigest()[:12]}"
		sample_data = self._get_sample_data(connection_id, sample_size)
		insights = _run_async(
			global_ml_insights_engine.analyze_data(
				sample_data,
				analysis_types=analysis_types,
				connection_id=connection_id
			)
		)
		insight_records = [self._insight_to_dict(insight) for insight in insights]
		completed_at = datetime.now(timezone.utc)
		job = {
			'job_id': job_id,
			'connection_id': connection_id,
			'status': 'completed',
			'progress': 100,
			'analysis_types': [_enum_value(analysis_type) for analysis_type in analysis_types],
			'sample_size': len(sample_data),
			'insights_generated': len(insight_records),
			'insights': insight_records,
			'started_at': started_at.isoformat(),
			'completed_at': completed_at.isoformat(),
			'duration_seconds': round((completed_at - started_at).total_seconds(), 3)
		}
		ANALYSIS_JOBS[job_id] = job
		return job

	def _insight_to_dict(self, insight: Any) -> Dict[str, Any]:
		"""Serialize an MLInsight dataclass to template/API data."""
		return {
			'insight_id': insight.insight_id,
			'title': insight.title,
			'description': insight.description,
			'analysis_type': _enum_value(insight.analysis_type),
			'severity': _enum_value(insight.severity),
			'confidence': insight.confidence,
			'generated_at': insight.generated_at.isoformat(),
			'evidence': insight.evidence,
			'recommendations': insight.recommendations,
			'affected_fields': insight.affected_fields,
			'metadata': insight.metadata
		}

	def _all_insights(self) -> List[Dict[str, Any]]:
		"""Return all stored insight records."""
		return [insight for job in ANALYSIS_JOBS.values() for insight in job['insights']]

	def _job_for_insight(self, insight_record: Dict[str, Any]) -> Dict[str, Any]:
		"""Return the stored job that produced an insight record."""
		insight_id = insight_record.get('insight_id')
		for job in ANALYSIS_JOBS.values():
			if any(insight.get('insight_id') == insight_id for insight in job['insights']):
				return job
		return {}

	def _get_connection_record(self, connection_id: str) -> Any:
		"""Look up a connection from the configured FAB data model when available."""
		try:
			session = self.datamodel.session
		except Exception:
			return None

		try:
			return session.query(CMConnection).filter(CMConnection.id == connection_id).first()
		except Exception:
			return None

	def _connection_sample_records(self, connection: Any) -> List[Dict[str, Any]]:
		"""Extract embedded sample records from connection metadata/config."""
		if not connection:
			return []

		for source in (
			getattr(connection, 'meta_data', None),
			getattr(connection, 'tap_config', None),
			getattr(connection, 'target_config', None),
		):
			if isinstance(source, dict):
				records = source.get('sample_records') or source.get('sample_data')
				if isinstance(records, list):
					return [record for record in records if isinstance(record, dict)]
		return []

	def _known_connection_ids(self) -> List[str]:
		"""Return known connection IDs from the configured data model."""
		try:
			connections = self.datamodel.session.query(CMConnection).all()
			return [str(connection.id) for connection in connections]
		except Exception:
			return []

	def _connection_choices(self) -> List[tuple[str, str]]:
		"""Return connection choices from the configured data model."""
		try:
			connections = self.datamodel.session.query(CMConnection).all()
			return [(str(connection.id), connection.name) for connection in connections]
		except Exception:
			return []

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
				'connection': self._job_for_insight(insight).get('connection_id'),
				'anomaly_count': insight.get('evidence', {}).get('anomaly_count', 0),
				'anomaly_rate': insight.get('evidence', {}).get('anomaly_rate', 0.0),
				'detected_at': insight.get('generated_at'),
				'severity': insight.get('severity')
			}
			for insight in self._all_insights()
			if insight.get('analysis_type') == AnalysisType.ANOMALY_DETECTION.value
		]

	def _get_recent_clusters(self) -> List[Dict[str, Any]]:
		"""Get recent clustering results"""
		return [
			{
				'connection': self._job_for_insight(insight).get('connection_id'),
				'num_clusters': insight.get('evidence', {}).get('num_clusters', 0),
				'silhouette_score': insight.get('evidence', {}).get('silhouette_score', 0.0),
				'analyzed_at': insight.get('generated_at')
			}
			for insight in self._all_insights()
			if insight.get('analysis_type') == AnalysisType.CLUSTERING.value
		]

	def _get_discovered_patterns(self) -> List[Dict[str, Any]]:
		"""Get discovered patterns"""
		return [
			{
				'pattern_id': insight.get('insight_id'),
				'pattern_type': insight.get('evidence', {}).get('pattern_type', insight.get('analysis_type')),
				'description': insight.get('description'),
				'confidence': insight.get('confidence'),
				'frequency': insight.get('evidence', {}).get('occurrences', 0),
				'fields': insight.get('affected_fields', []),
				'discovered_at': insight.get('generated_at')
			}
			for insight in self._all_insights()
			if insight.get('analysis_type') == AnalysisType.PATTERN_RECOGNITION.value
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
				'connection': self._job_for_insight(insight).get('connection_id'),
				'field': next(iter(insight.get('affected_fields', [])), None),
				'trend': insight.get('evidence', {}).get('trend_direction'),
				'predicted_change': insight.get('evidence', {}).get('predicted_change'),
				'forecast_horizon': insight.get('evidence', {}).get('forecast_horizon'),
				'confidence': insight.get('confidence'),
				'generated_at': insight.get('generated_at')
			}
			for insight in self._all_insights()
			if insight.get('analysis_type') == AnalysisType.TIME_SERIES_FORECASTING.value
		]


class MLInsightsChartsView(BaseView):
	"""ML insights charts and visualizations"""

	route_base = '/ml_insights/charts'

	@expose('/api/insights_timeline')
	@has_access
	def api_insights_timeline(self):
		"""API endpoint for insights timeline chart"""
		insights = self._all_insights()
		days = sorted({insight.get('generated_at', '')[:10] for insight in insights if insight.get('generated_at')})[-8:]
		if not days:
			days = [datetime.now(timezone.utc).date().isoformat()]
		severities = ['critical', 'high', 'medium', 'low']
		colors = {
			'critical': '#dc3545',
			'high': '#fd7e14',
			'medium': '#ffc107',
			'low': '#28a745'
		}
		timeline_data = {
			'labels': days,
			'datasets': [
				{
					'label': severity.title(),
					'data': [
						sum(
							1 for insight in insights
							if insight.get('severity') == severity
							and insight.get('generated_at', '').startswith(day)
						)
						for day in days
					],
					'backgroundColor': colors[severity],
					'borderColor': colors[severity]
				}
				for severity in severities
			]
		}

		return jsonify(timeline_data)

	@expose('/api/analysis_distribution')
	@has_access
	def api_analysis_distribution(self):
		"""API endpoint for analysis type distribution"""
		counts = Counter(insight['analysis_type'] for insight in self._all_insights())
		analysis_types = [analysis_type.value for analysis_type in AnalysisType]
		distribution_data = {
			'labels': [analysis_type.replace('_', ' ').title() for analysis_type in analysis_types],
			'datasets': [{
				'data': [counts.get(analysis_type, 0) for analysis_type in analysis_types],
				'backgroundColor': [
					'#FF6384',
					'#36A2EB',
					'#FFCE56',
					'#4BC0C0',
					'#9966FF',
					'#FF9F40',
					'#7CB342',
					'#5C6BC0',
					'#26A69A'
				]
			}]
		}

		return jsonify(distribution_data)

	@expose('/api/confidence_scores')
	@has_access
	def api_confidence_scores(self):
		"""API endpoint for confidence score distribution"""
		buckets = [0, 0, 0, 0, 0]
		for insight in self._all_insights():
			confidence = max(0.0, min(1.0, float(insight.get('confidence', 0.0))))
			bucket_index = min(int(confidence / 0.2), 4)
			buckets[bucket_index] += 1
		confidence_data = {
			'labels': ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
			'datasets': [{
				'label': 'Number of Insights',
				'data': buckets,
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
