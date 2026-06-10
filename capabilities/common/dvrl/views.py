#!/usr/bin/env python3
"""
APG Data Virtualization (DVRL) Flask-AppBuilder Views
Real Flask-AppBuilder UI views and blueprints for the DVRL capability

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Real Flask-AppBuilder imports
from flask import Blueprint, request, flash, redirect, url_for, jsonify
from flask_appbuilder import AppBuilder, BaseView, ModelView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from wtforms import Form, StringField, SelectField, TextAreaField, validators
from wtforms.widgets import TextArea

# Create DVRL Blueprint
dvrl_bp = Blueprint('dvrl', __name__, url_prefix='/dvrl', static_folder='static', template_folder='templates')


# Form Classes for DVRL
class DataSourceForm(Form):
	"""Form for creating/editing data sources"""
	name = StringField('Name', [validators.DataRequired(), validators.Length(min=1, max=100)])
	type = SelectField('Type', choices=[
		('postgresql', 'PostgreSQL'),
		('mysql', 'MySQL'),
		('mongodb', 'MongoDB'),
		('redis', 'Redis'),
		('elasticsearch', 'Elasticsearch'),
		('cassandra', 'Cassandra'),
		('s3', 'Amazon S3'),
		('api', 'REST API')
	], validators=[validators.DataRequired()])
	host = StringField('Host', [validators.Optional()])
	port = StringField('Port', [validators.Optional()])
	database = StringField('Database', [validators.Optional()])
	username = StringField('Username', [validators.Optional()])
	password = StringField('Password', [validators.Optional()])
	connection_config = TextAreaField('Connection Config (JSON)', widget=TextArea(), validators=[validators.Optional()])
	description = TextAreaField('Description', widget=TextArea(), validators=[validators.Optional()])


class NaturalLanguageQueryForm(Form):
	"""Form for natural language queries"""
	query = TextAreaField('Natural Language Query', 
		[validators.DataRequired(), validators.Length(min=5, max=1000)],
		render_kw={"placeholder": "e.g., Show me the total sales for last month", "rows": 3})
	data_sources = StringField('Target Data Sources (optional)', 
		render_kw={"placeholder": "Leave empty to search all data sources"})


class SQLQueryForm(Form):
	"""Form for SQL queries"""
	sql = TextAreaField('SQL Query', 
		[validators.DataRequired(), validators.Length(min=5, max=5000)],
		widget=TextArea(), 
		render_kw={"placeholder": "SELECT * FROM table_name", "rows": 5})
	parameters = TextAreaField('Parameters (JSON)', 
		widget=TextArea(),
		render_kw={"placeholder": '{"param1": "value1"}', "rows": 2},
		validators=[validators.Optional()])


# Real Flask-AppBuilder Views
class DVRLDashboardView(BaseView):
	"""Main DVRL dashboard view using real Flask-AppBuilder"""
	
	route_base = '/dvrl'
	default_view = 'dashboard'
	
	def __init__(self, dvrl_service):
		super().__init__()
		self.dvrl_service = dvrl_service
	
	def _run_async(self, coro):
		"""Helper to run async coroutines in sync context"""
		try:
			loop = asyncio.get_event_loop()
		except RuntimeError:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
		
		if loop.is_running():
			# If loop is already running, create a new one
			return asyncio.run_coroutine_threadsafe(coro, loop).result()
		else:
			return loop.run_until_complete(coro)
	
	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Main dashboard page"""
		try:
			# Get health status (sync wrapper for async service)
			health_status = self._run_async(self.dvrl_service.get_health_status())
			
			# Get performance metrics
			performance_metrics = self._run_async(self.dvrl_service.get_performance_metrics())
			
			# Get data sources summary
			data_sources = list(self.dvrl_service.data_sources.values())
			
			# Get recent queries (latest 10)
			recent_queries = self.dvrl_service.query_history[-10:] if hasattr(self.dvrl_service, 'query_history') else []
			
			# Calculate summary stats
			total_queries = len(getattr(self.dvrl_service, 'query_history', []))
			avg_response_time = sum(q.get('duration_ms', 0) for q in recent_queries) / len(recent_queries) if recent_queries else 0
			
			context = {
				'health_status': health_status,
				'performance_metrics': performance_metrics,
				'data_sources_count': len(data_sources),
				'total_queries': total_queries,
				'avg_response_time': round(avg_response_time, 2),
				'recent_queries': recent_queries,
				'active_connectors': performance_metrics.get('connector_framework', {}).get('total_connectors', 0)
			}
			
			return self.render_template('dvrl/dashboard.html', **context)
			
		except Exception as e:
			flash(f"Dashboard error: {str(e)}", "error")
			return self.render_template('dvrl/error.html', error=str(e))
	
	@expose('/data-sources')
	@has_access
	def data_sources(self):
		"""Data sources management page"""
		try:
			data_sources = []
			for source in self.dvrl_service.data_sources.values():
				# Get connector stats
				try:
					connector = self._run_async(self.dvrl_service.connector_manager.get_connector(source.id))
					connector_stats = self._run_async(connector.get_connection_stats()) if connector else {}
				except Exception:
					connector_stats = {}
				
				data_sources.append({
					'id': source.id,
					'name': source.name,
					'type': source.type.value,
					'status': source.status.value,
					'health': connector_stats.get('health_status', 'unknown'),
					'capabilities': connector_stats.get('capabilities', []),
					'query_count': getattr(source, 'query_count', 0),
					'created_at': source.created_at.strftime('%Y-%m-%d %H:%M') if hasattr(source, 'created_at') else 'unknown'
				})
			
			# Group by type for summary
			type_summary = {}
			for ds in data_sources:
				ds_type = ds['type']
				if ds_type not in type_summary:
					type_summary[ds_type] = {'count': 0, 'healthy': 0}
				type_summary[ds_type]['count'] += 1
				if ds['health'] == 'healthy':
					type_summary[ds_type]['healthy'] += 1
			
			context = {
				'data_sources': data_sources,
				'total_sources': len(data_sources),
				'type_summary': type_summary,
				'data_source_form': DataSourceForm()
			}
			
			return self.render_template('dvrl/data_sources.html', **context)
			
		except Exception as e:
			flash(f"Data sources error: {str(e)}", "error")
			return self.render_template('dvrl/error.html', error=str(e))
	
	@expose('/query-interface')
	@has_access
	def query_interface(self):
		"""Query interface page with NL and SQL options"""
		nl_form = NaturalLanguageQueryForm()
		sql_form = SQLQueryForm()
		
		return self.render_template('dvrl/query_interface.html', 
			nl_form=nl_form, 
			sql_form=sql_form)
	
	@expose('/execute-nl-query', methods=['POST'])
	@has_access
	def execute_nl_query(self):
		"""Execute natural language query"""
		form = NaturalLanguageQueryForm(request.form)
		
		if form.validate():
			try:
				# Build schema context
				schema_context = {}
				for ds in self.dvrl_service.data_sources.values():
					try:
						schema = self._run_async(self.dvrl_service.get_data_source_schema(ds.id))
						schema_context[ds.name] = schema
					except Exception:
						continue
				
				# Execute NL query
				result = self._run_async(
					self.dvrl_service.execute_natural_language_query(
						form.query.data,
						list(form.data_sources.data.split(',')) if form.data_sources.data else [],
						{'schema_context': schema_context}
					)
				)
				
				return jsonify({
					'success': True,
					'query_id': result.get('query_id'),
					'generated_sql': result.get('generated_sql'),
					'confidence': result.get('confidence'),
					'results': result.get('results', []),
					'explanation': result.get('explanation', '')
				})
				
			except Exception as e:
				return jsonify({
					'success': False,
					'error': str(e)
				}), 500
		else:
			return jsonify({
				'success': False,
				'error': 'Form validation failed',
				'form_errors': form.errors
			}), 400
	
	@expose('/execute-sql-query', methods=['POST'])
	@has_access
	def execute_sql_query(self):
		"""Execute SQL query"""
		form = SQLQueryForm(request.form)
		
		if form.validate():
			try:
				# Parse parameters if provided
				parameters = {}
				if form.parameters.data:
					parameters = json.loads(form.parameters.data)
				
				# Execute SQL query
				result = self._run_async(
					self.dvrl_service.execute_federated_query(
						form.sql.data,
						parameters,
						{}
					)
				)
				
				return jsonify({
					'success': True,
					'query_id': result.get('id'),
					'status': result.get('status'),
					'results': getattr(result, 'results', []),
					'rows_returned': getattr(result, 'rows_returned', 0),
					'execution_time_ms': getattr(result, 'duration_ms', 0)
				})
				
			except Exception as e:
				return jsonify({
					'success': False,
					'error': str(e)
				}), 500
		else:
			return jsonify({
				'success': False,
				'error': 'Form validation failed',
				'form_errors': form.errors
			}), 400
	
	@expose('/add-data-source', methods=['POST'])
	@has_access
	def add_data_source(self):
		"""Add new data source"""
		form = DataSourceForm(request.form)
		
		if form.validate():
			try:
				# Build connection config
				config = {
					'name': form.name.data,
					'type': form.type.data,
					'description': form.description.data or ''
				}
				
				# Add connection details
				if form.host.data:
					config['host'] = form.host.data
				if form.port.data:
					config['port'] = int(form.port.data) if form.port.data.isdigit() else form.port.data
				if form.database.data:
					config['database'] = form.database.data
				if form.username.data:
					config['username'] = form.username.data
				if form.password.data:
					config['password'] = form.password.data
				
				# Add additional JSON config if provided
				if form.connection_config.data:
					additional_config = json.loads(form.connection_config.data)
					config.update(additional_config)
				
				# Register data source
				data_source = self._run_async(
					self.dvrl_service.register_data_source(config)
				)
				
				flash(f"Data source '{data_source.name}' added successfully!", "success")
				return redirect(url_for('DVRLDashboardView.data_sources'))
				
			except Exception as e:
				flash(f"Failed to add data source: {str(e)}", "error")
		else:
			for field, errors in form.errors.items():
				for error in errors:
					flash(f"{field}: {error}", "error")
		
		return redirect(url_for('DVRLDashboardView.data_sources'))
	
	@expose('/singer-taps')
	@has_access
	def singer_taps(self):
		"""Singer taps management page"""
		try:
			# Get Singer tap manager if available
			if hasattr(self.dvrl_service, 'singer_manager'):
				singer_manager = self.dvrl_service.singer_manager
				
				available_taps = singer_manager.available_taps
				installed_taps = singer_manager.installed_taps
				
				context = {
					'available_taps': available_taps,
					'installed_taps': installed_taps,
					'total_available': len(available_taps),
					'total_installed': len(installed_taps)
				}
			else:
				context = {
					'available_taps': {},
					'installed_taps': {},
					'total_available': 0,
					'total_installed': 0,
					'error': 'Singer integration not available'
				}
			
			return self.render_template('dvrl/singer_taps.html', **context)
			
		except Exception as e:
			flash(f"Singer taps error: {str(e)}", "error")
			return self.render_template('dvrl/error.html', error=str(e))
	
	@expose('/performance-metrics')
	@has_access
	def performance_metrics(self):
		"""Performance metrics and monitoring page"""
		try:
			# Get detailed performance metrics
			metrics = self._run_async(self.dvrl_service.get_performance_metrics())
			
			# Get connector statistics
			connector_stats = self._run_async(self.dvrl_service.connector_manager.get_connector_stats()) if hasattr(self.dvrl_service, 'connector_manager') else {}
			
			context = {
				'metrics': metrics,
				'connector_stats': connector_stats,
				'last_updated': datetime.now(timezone.utc).isoformat()
			}
			
			return self.render_template('dvrl/performance_metrics.html', **context)
			
		except Exception as e:
			flash(f"Metrics error: {str(e)}", "error")
			return self.render_template('dvrl/error.html', error=str(e))


# Export view classes
__all__ = [
	"DVRLDashboardView",
	"DataSourceForm",
	"NaturalLanguageQueryForm", 
	"SQLQueryForm",
	"dvrl_bp"
]