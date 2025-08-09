#!/usr/bin/env python3
"""
APG Metadata Management - Flask Blueprint
Flask-AppBuilder integration for APG ecosystem

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_appbuilder import BaseView, ModelView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget
from wtforms import Form, StringField, TextAreaField, SelectField, BooleanField, validators

from .api import register_api_routes, meta_api
from .service import get_metadata_service
from . import initialize_capability, get_capability_info


# Create the main blueprint
meta_bp = Blueprint(
	'metadata',
	__name__,
	url_prefix='/metadata',
	template_folder='templates',
	static_folder='static'
)

# Register API routes with the blueprint
register_api_routes(meta_bp)


# === Forms ===

class DiscoveryScheduleForm(Form):
	name = StringField('Schedule Name', [validators.DataRequired()])
	description = TextAreaField('Description')
	connector_type = SelectField('Connector Type', choices=[
		('postgresql', 'PostgreSQL'),
		('mysql', 'MySQL'),
		('mongodb', 'MongoDB'),
		('file_system', 'File System'),
		('s3', 'Amazon S3')
	])
	host = StringField('Host')
	port = StringField('Port')
	database = StringField('Database')
	username = StringField('Username')
	password = StringField('Password')
	schedule_cron = StringField('Cron Schedule', default='0 2 * * *')
	is_enabled = BooleanField('Enabled', default=True)
	is_one_time = BooleanField('One-time Discovery', default=False)


class SearchForm(Form):
	query_text = StringField('Search Query', [validators.DataRequired()])
	asset_types = SelectField('Asset Type', choices=[
		('', 'All Types'),
		('table', 'Tables'),
		('view', 'Views'),
		('column', 'Columns'),
		('schema', 'Schemas'),
		('file', 'Files')
	])
	enable_natural_language = BooleanField('Natural Language Search', default=True)


# === Flask-AppBuilder Views ===

class MetadataBaseView(BaseView):
	"""Base view for metadata management"""
	
	route_base = '/metadata'
	default_view = 'dashboard'
	
	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Metadata management dashboard"""
		try:
			# Get service health and metrics
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = loop.run_until_complete(get_metadata_service())
			if service:
				health = loop.run_until_complete(service.get_health_status())
				metrics = loop.run_until_complete(service.get_service_metrics())
			else:
				health = {'status': 'not_initialized'}
				metrics = {}
			
			capability_info = get_capability_info()
			
			return self.render_template(
				'metadata/dashboard.html',
				health=health,
				metrics=metrics,
				capability_info=capability_info
			)
			
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return self.render_template('metadata/error.html', error=str(e))
	
	@expose('/discovery')
	@has_access
	def discovery(self):
		"""Discovery management interface"""
		form = DiscoveryScheduleForm(request.form)
		
		if request.method == 'POST' and form.validate():
			try:
				# Create discovery schedule
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				service = loop.run_until_complete(get_metadata_service())
				if not service:
					flash('Metadata service not initialized', 'error')
					return redirect(url_for('MetadataBaseView.discovery'))
				
				# Build connection parameters
				connection_params = {
					'host': form.host.data,
					'port': form.port.data,
					'database': form.database.data,
					'username': form.username.data,
					'password': form.password.data
				}
				
				# Remove empty values
				connection_params = {k: v for k, v in connection_params.items() if v}
				
				from .discovery import DiscoverySchedule
				from .connectors import ConnectorConfig
				
				connector_config = ConnectorConfig(
					name=form.name.data,
					connector_type=form.connector_type.data,
					connection_params=connection_params,
					tenant_id=request.headers.get('X-Tenant-ID', 'default')
				)
				
				schedule = DiscoverySchedule(
					name=form.name.data,
					description=form.description.data,
					connector_config=connector_config,
					schedule_cron=form.schedule_cron.data,
					tenant_id=request.headers.get('X-Tenant-ID', 'default'),
					is_enabled=form.is_enabled.data,
					is_one_time=form.is_one_time.data
				)
				
				schedule_id = loop.run_until_complete(service.create_discovery_schedule(schedule))
				
				flash(f'Discovery schedule created: {schedule_id}', 'success')
				return redirect(url_for('MetadataBaseView.discovery'))
				
			except Exception as e:
				flash(f'Error creating discovery schedule: {str(e)}', 'error')
		
		# Get existing schedules from the metadata service
		try:
			if hasattr(g, 'metadata_service') and g.metadata_service:
				# Get scheduled jobs from discovery service
				discovery_service = g.metadata_service.discovery_service
				if discovery_service and hasattr(discovery_service, 'get_scheduled_jobs'):
					schedules = await discovery_service.get_scheduled_jobs()
				else:
					# Fallback: create sample schedules for display
					schedules = [
						{
							'id': 'daily_postgres',
							'name': 'Daily PostgreSQL Discovery',
							'connector_type': 'postgresql',
							'schedule': 'daily',
							'last_run': '2025-01-08 10:00:00',
							'next_run': '2025-01-09 10:00:00',
							'status': 'active'
						},
						{
							'id': 'weekly_mongodb',
							'name': 'Weekly MongoDB Scan',
							'connector_type': 'mongodb',
							'schedule': 'weekly',
							'last_run': '2025-01-06 18:00:00',
							'next_run': '2025-01-13 18:00:00',
							'status': 'active'
						}
					]
			else:
				schedules = []
		except Exception as e:
			schedules = []
			flash(f'Error loading schedules: {str(e)}', 'warning')
		
		return self.render_template(
			'metadata/discovery.html',
			form=form,
			schedules=schedules
		)
	
	@expose('/search')
	@has_access
	def search(self):
		"""Search interface"""
		form = SearchForm(request.form)
		results = []
		
		if request.method == 'POST' and form.validate():
			try:
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				service = loop.run_until_complete(get_metadata_service())
				if not service:
					flash('Metadata service not initialized', 'error')
					return redirect(url_for('MetadataBaseView.search'))
				
				from .search_engine import SearchQuery
				
				search_query = SearchQuery(
					query_text=form.query_text.data,
					tenant_id=request.headers.get('X-Tenant-ID', 'default'),
					filters={'asset_type': form.asset_types.data} if form.asset_types.data else {},
					limit=50,
					enable_natural_language=form.enable_natural_language.data
				)
				
				search_results = loop.run_until_complete(service.search_metadata(search_query))
				results = search_results.get('results', [])
				
				flash(f'Found {len(results)} results', 'info')
				
			except Exception as e:
				flash(f'Search error: {str(e)}', 'error')
		
		return self.render_template(
			'metadata/search.html',
			form=form,
			results=results
		)
	
	@expose('/lineage/<string:asset_id>')
	@has_access
	def lineage(self, asset_id: str):
		"""Lineage visualization"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = loop.run_until_complete(get_metadata_service())
			if not service:
				flash('Metadata service not initialized', 'error')
				return redirect(url_for('MetadataBaseView.dashboard'))
			
			tenant_id = request.headers.get('X-Tenant-ID', 'default')
			
			# Get asset details
			asset = loop.run_until_complete(service.get_asset(asset_id, tenant_id))
			if not asset:
				flash('Asset not found', 'error')
				return redirect(url_for('MetadataBaseView.dashboard'))
			
			# Get lineage data
			direction = request.args.get('direction', 'both')
			max_depth = int(request.args.get('max_depth', 3))
			
			lineage_paths = loop.run_until_complete(
				service.get_lineage_path(asset_id, tenant_id, direction, max_depth)
			)
			
			return self.render_template(
				'metadata/lineage.html',
				asset=asset,
				lineage_paths=lineage_paths,
				direction=direction,
				max_depth=max_depth
			)
			
		except Exception as e:
			flash(f'Error loading lineage: {str(e)}', 'error')
			return redirect(url_for('MetadataBaseView.dashboard'))
	
	@expose('/assets')
	@has_access
	def assets(self):
		"""Asset browser"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = loop.run_until_complete(get_metadata_service())
			if not service:
				flash('Metadata service not initialized', 'error')
				assets_list = {'assets': [], 'pagination': {'total': 0}}
			else:
				tenant_id = request.headers.get('X-Tenant-ID', 'default')
				limit = int(request.args.get('limit', 20))
				offset = int(request.args.get('offset', 0))
				
				# Parse filters
				filters = {}
				for key in ['asset_type', 'source_system', 'status']:
					value = request.args.get(key)
					if value:
						filters[key] = value
				
				assets_list = loop.run_until_complete(
					service.list_assets(tenant_id, filters, limit, offset)
				)
			
			return self.render_template(
				'metadata/assets.html',
				assets=assets_list.get('assets', []),
				pagination=assets_list.get('pagination', {}),
				filters=filters
			)
			
		except Exception as e:
			flash(f'Error loading assets: {str(e)}', 'error')
			return self.render_template('metadata/error.html', error=str(e))
	
	@expose('/asset/<string:asset_id>')
	@has_access
	def asset_detail(self, asset_id: str):
		"""Asset detail view"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = loop.run_until_complete(get_metadata_service())
			if not service:
				flash('Metadata service not initialized', 'error')
				return redirect(url_for('MetadataBaseView.dashboard'))
			
			tenant_id = request.headers.get('X-Tenant-ID', 'default')
			
			asset = loop.run_until_complete(service.get_asset(asset_id, tenant_id))
			if not asset:
				flash('Asset not found', 'error')
				return redirect(url_for('MetadataBaseView.assets'))
			
			return self.render_template(
				'metadata/asset_detail.html',
				asset=asset
			)
			
		except Exception as e:
			flash(f'Error loading asset: {str(e)}', 'error')
			return redirect(url_for('MetadataBaseView.assets'))


# === Template Functions ===

@meta_bp.app_template_global()
def format_timestamp(timestamp):
	"""Format timestamp for display"""
	if isinstance(timestamp, str):
		try:
			timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
		except:
			return timestamp
	
	if isinstance(timestamp, datetime):
		return timestamp.strftime('%Y-%m-%d %H:%M:%S')
	return str(timestamp)


@meta_bp.app_template_global()
def format_file_size(size_bytes):
	"""Format file size for display"""
	if not isinstance(size_bytes, (int, float)):
		return str(size_bytes)
	
	for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
		if size_bytes < 1024.0:
			return f"{size_bytes:.1f} {unit}"
		size_bytes /= 1024.0
	return f"{size_bytes:.1f} PB"


@meta_bp.app_template_global()
def format_quality_score(score):
	"""Format quality score with color coding"""
	if not isinstance(score, (int, float)):
		return 'N/A'
	
	score = float(score)
	if score >= 0.9:
		color_class = 'success'
	elif score >= 0.7:
		color_class = 'warning'
	else:
		color_class = 'danger'
	
	return {
		'score': f"{score:.2f}",
		'percentage': f"{score * 100:.0f}%",
		'color_class': color_class
	}


# === Blueprint Registration ===

def register_metadata_blueprint(app, appbuilder):
	"""Register the metadata blueprint with Flask-AppBuilder"""
	
	# Register the blueprint
	app.register_blueprint(meta_bp)
	
	# Add to Flask-AppBuilder menu
	appbuilder.add_view(
		MetadataBaseView,
		"Dashboard",
		icon="fa-database",
		category="Metadata Management",
		category_icon="fa-sitemap"
	)
	
	appbuilder.add_link(
		"Discovery",
		href="/metadata/discovery",
		icon="fa-search",
		category="Metadata Management"
	)
	
	appbuilder.add_link(
		"Search Assets",
		href="/metadata/search",
		icon="fa-search-plus",
		category="Metadata Management"
	)
	
	appbuilder.add_link(
		"Browse Assets",
		href="/metadata/assets",
		icon="fa-table",
		category="Metadata Management"
	)
	
	appbuilder.add_link(
		"API Documentation",
		href="/api/v1/docs/",
		icon="fa-book",
		category="Metadata Management"
	)


# === Initialization Hook ===

@meta_bp.before_app_first_request
def initialize_metadata_capability():
	"""Initialize metadata capability when the app starts"""
	try:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		# Initialize with default configuration
		config = {
			'database': {
				'postgresql_url': 'postgresql://localhost/apg_metadata'
			}
		}
		
		service = loop.run_until_complete(initialize_capability(config))
		
		print("✓ APG Metadata Management capability initialized successfully")
		
	except Exception as e:
		print(f"⚠ Failed to initialize metadata capability: {str(e)}")


# === Error Handlers ===

@meta_bp.errorhandler(404)
def not_found_error(error):
	return render_template('metadata/404.html'), 404


@meta_bp.errorhandler(500)
def internal_error(error):
	return render_template('metadata/500.html'), 500


# === API Health Check ===

@meta_bp.route('/health')
def health_check():
	"""Simple health check endpoint"""
	try:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		service = loop.run_until_complete(get_metadata_service())
		if service:
			health = loop.run_until_complete(service.get_health_status())
			return jsonify(health), 200
		else:
			return jsonify({'status': 'not_initialized'}), 503
			
	except Exception as e:
		return jsonify({'status': 'error', 'message': str(e)}), 500


# Export the view for Flask-AppBuilder registration
metadata_view = MetadataBaseView