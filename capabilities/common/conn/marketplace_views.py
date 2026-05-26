"""
APG Connection Management Marketplace Views
Flask-AppBuilder views for capability marketplace integration

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from flask import request, jsonify, flash, redirect, url_for, abort
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget
from flask_appbuilder.actions import action
from wtforms import Form, StringField, SelectField, TextAreaField, BooleanField, FloatField
from wtforms.validators import DataRequired, NumberRange, Optional as OptionalValidator

from .marketplace import (
	global_marketplace_manager, MarketplaceSearchQuery, CapabilityType,
	LicenseType, InstallationStatus
)

logger = logging.getLogger(__name__)


class MarketplaceSearchForm(Form):
	"""Form for marketplace search"""
	query = StringField('Search Query',
						render_kw={'class': 'form-control', 'placeholder': 'Search capabilities...'})
	capability_type = SelectField('Type',
								 choices=[('', 'All Types')] + [(t.value, t.value.title()) for t in CapabilityType],
								 render_kw={'class': 'form-control'})
	license = SelectField('License',
						 choices=[('', 'All Licenses')] + [(l.value, l.value.replace('_', ' ').title()) for l in LicenseType],
						 render_kw={'class': 'form-control'})
	tags = StringField('Tags (comma-separated)',
					  render_kw={'class': 'form-control', 'placeholder': 'e.g., database, api, transform'})
	min_rating = SelectField('Minimum Rating',
							choices=[('', 'Any Rating'), ('4.5', '4.5+ Stars'), ('4.0', '4.0+ Stars'), ('3.0', '3.0+ Stars')],
							render_kw={'class': 'form-control'})
	free_only = BooleanField('Free Only', render_kw={'class': 'form-check-input'})
	verified_only = BooleanField('Verified Publishers Only', render_kw={'class': 'form-check-input'})
	sort_by = SelectField('Sort By',
						 choices=[('relevance', 'Relevance'), ('rating', 'Rating'), ('downloads', 'Downloads'), ('updated', 'Recently Updated')],
						 default='relevance',
						 render_kw={'class': 'form-control'})


class CapabilityInstallForm(Form):
	"""Form for installing capabilities"""
	version = SelectField('Version',
						 choices=[('latest', 'Latest Version')],
						 default='latest',
						 render_kw={'class': 'form-control'})
	auto_update = BooleanField('Auto-update',
							  default=True,
							  render_kw={'class': 'form-check-input'})


class MarketplaceDashboardView(BaseView):
	"""Main marketplace dashboard"""

	route_base = '/marketplace'
	default_view = 'dashboard'

	@expose('/')
	@has_access
	def dashboard(self):
		"""Marketplace dashboard"""
		return self.render_template(
			'marketplace/dashboard.html',
			featured_capabilities=[],
			installed_count=len(global_marketplace_manager.get_installed_capabilities()),
			recent_installs=self._get_recent_installs(),
			trending_categories=self._get_trending_categories()
		)

	@expose('/browse')
	@has_access
	def browse(self):
		"""Browse marketplace capabilities"""
		form = MarketplaceSearchForm(request.args)

		# Build search query
		search_query = MarketplaceSearchQuery()

		if form.query.data:
			search_query.query = form.query.data

		if form.capability_type.data:
			search_query.capability_type = CapabilityType(form.capability_type.data)

		if form.license.data:
			search_query.license = LicenseType(form.license.data)

		if form.tags.data:
			search_query.tags = [t.strip() for t in form.tags.data.split(',') if t.strip()]

		if form.min_rating.data:
			search_query.min_rating = float(form.min_rating.data)

		search_query.free_only = form.free_only.data
		search_query.verified_only = form.verified_only.data
		search_query.sort_by = form.sort_by.data
		search_query.limit = 20

		# Get page number
		page = request.args.get('page', 1, type=int)
		search_query.offset = (page - 1) * search_query.limit

		# Mock search results for now
		capabilities = self._get_mock_capabilities()
		total_results = len(capabilities)

		return self.render_template(
			'marketplace/browse.html',
			form=form,
			capabilities=capabilities,
			total_results=total_results,
			current_page=page,
			total_pages=(total_results + search_query.limit - 1) // search_query.limit
		)

	@expose('/capability/<capability_id>')
	@has_access
	def capability_detail(self, capability_id):
		"""Show detailed capability information"""
		try:
			# Mock capability details
			capability = self._get_mock_capability_detail(capability_id)

			# Check if installed
			installed_capability = global_marketplace_manager.installer.get_capability_info(capability_id)
			is_installed = installed_capability is not None

			# Get available versions
			versions = [
				{'version': '2.1.0', 'release_notes': 'Latest stable release with bug fixes'},
				{'version': '2.0.1', 'release_notes': 'Security updates'},
				{'version': '2.0.0', 'release_notes': 'Major release with new features'}
			]

			install_form = CapabilityInstallForm()
			install_form.version.choices = [('latest', 'Latest (2.1.0)')] + \
										   [(v['version'], v['version']) for v in versions]

			return self.render_template(
				'marketplace/capability_detail.html',
				capability=capability,
				is_installed=is_installed,
				installed_capability=installed_capability,
				versions=versions,
				install_form=install_form
			)

		except Exception as e:
			logger.error(f"Error loading capability {capability_id}: {e}")
			flash(f'Error loading capability: {str(e)}', 'error')
			return redirect(url_for('MarketplaceDashboardView.browse'))

	@expose('/api/install/<capability_id>', methods=['POST'])
	@has_access
	def api_install_capability(self, capability_id):
		"""API endpoint to install capability"""
		try:
			data = request.get_json()
			version = data.get('version', 'latest')
			auto_update = data.get('auto_update', True)

			# Mock installation
			from .marketplace import InstalledCapability, InstallationStatus

			installed_capability = InstalledCapability(
				capability_id=capability_id,
				name=f"Mock Capability {capability_id}",
				version=version if version != 'latest' else '2.1.0',
				installation_path=f"./installed/{capability_id}",
				status=InstallationStatus.INSTALLED,
				installed_at=datetime.now(timezone.utc),
				auto_update=auto_update
			)

			# Add to installer
			global_marketplace_manager.installer.installed_capabilities[capability_id] = installed_capability
			global_marketplace_manager.installer._save_installed_capabilities()

			return jsonify({
				'success': True,
				'message': f'Capability {capability_id} installed successfully',
				'capability': {
					'id': capability_id,
					'name': installed_capability.name,
					'version': installed_capability.version,
					'status': installed_capability.status.value
				}
			})

		except Exception as e:
			logger.error(f"Error installing capability {capability_id}: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500

	@expose('/api/uninstall/<capability_id>', methods=['POST'])
	@has_access
	def api_uninstall_capability(self, capability_id):
		"""API endpoint to uninstall capability"""
		try:
			# Remove from installer
			if capability_id in global_marketplace_manager.installer.installed_capabilities:
				del global_marketplace_manager.installer.installed_capabilities[capability_id]
				global_marketplace_manager.installer._save_installed_capabilities()

				return jsonify({
					'success': True,
					'message': f'Capability {capability_id} uninstalled successfully'
				})
			else:
				return jsonify({
					'success': False,
					'error': 'Capability not installed'
				}), 404

		except Exception as e:
			logger.error(f"Error uninstalling capability {capability_id}: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500

	@expose('/api/search')
	@has_access
	def api_search(self):
		"""API endpoint for capability search"""
		try:
			query = request.args.get('q', '')
			capability_type = request.args.get('type', '')
			tags = request.args.get('tags', '').split(',') if request.args.get('tags') else []
			limit = request.args.get('limit', 10, type=int)
			offset = request.args.get('offset', 0, type=int)

			# Mock search results
			capabilities = self._get_mock_capabilities()

			# Apply filters
			filtered_capabilities = []
			for cap in capabilities:
				if query and query.lower() not in cap['name'].lower() and query.lower() not in cap['description'].lower():
					continue
				if capability_type and cap['capability_type'] != capability_type:
					continue
				if tags and not any(tag in cap['tags'] for tag in tags):
					continue
				filtered_capabilities.append(cap)

			# Apply pagination
			paginated_capabilities = filtered_capabilities[offset:offset + limit]

			return jsonify({
				'capabilities': paginated_capabilities,
				'total': len(filtered_capabilities),
				'offset': offset,
				'limit': limit
			})

		except Exception as e:
			logger.error(f"Error in marketplace search API: {e}")
			return jsonify({'error': str(e)}), 500

	@expose('/installed')
	@has_access
	def installed_capabilities(self):
		"""Show installed capabilities"""
		installed = global_marketplace_manager.get_installed_capabilities()

		# Get update information
		updates_available = []  # Would check for updates

		return self.render_template(
			'marketplace/installed.html',
			installed_capabilities=installed,
			updates_available=updates_available
		)

	@expose('/publish')
	@has_access
	def publish_capability(self):
		"""Publish a new capability to marketplace"""
		# This would require publisher permissions
		return self.render_template(
			'marketplace/publish.html'
		)

	def _get_recent_installs(self) -> List[Dict[str, Any]]:
		"""Get recent capability installations"""
		installed = global_marketplace_manager.get_installed_capabilities()
		recent = sorted(installed, key=lambda x: x.installed_at, reverse=True)[:5]

		return [
			{
				'name': cap.name,
				'version': cap.version,
				'installed_at': cap.installed_at.strftime('%Y-%m-%d %H:%M')
			}
			for cap in recent
		]

	def _get_trending_categories(self) -> List[Dict[str, Any]]:
		"""Get trending capability categories"""
		return [
			{'name': 'Database Connectors', 'count': 45, 'icon': 'fa-database'},
			{'name': 'Data Transformers', 'count': 32, 'icon': 'fa-cogs'},
			{'name': 'API Integrations', 'count': 28, 'icon': 'fa-plug'},
			{'name': 'Validators', 'count': 19, 'icon': 'fa-check-circle'},
			{'name': 'Analytics', 'count': 15, 'icon': 'fa-chart-bar'}
		]

	def _get_mock_capabilities(self) -> List[Dict[str, Any]]:
		"""Mock marketplace capabilities for testing"""
		return [
			{
				'id': 'postgres-connector',
				'name': 'PostgreSQL Connector',
				'description': 'High-performance PostgreSQL database connector with advanced connection pooling and query optimization',
				'capability_type': 'connector',
				'status': 'published',
				'author': {'name': 'Datacraft Team', 'verified': True},
				'license': 'open_source',
				'current_version': '2.1.0',
				'rating': {'average_rating': 4.8, 'total_reviews': 156},
				'stats': {'downloads': 12543, 'installations': 8932},
				'tags': ['database', 'postgresql', 'sql', 'connector'],
				'categories': ['Database', 'Connector'],
				'price': 0.0,
				'icon_url': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg',
				'is_featured': True,
				'created_at': '2024-11-15T10:00:00Z',
				'updated_at': '2025-01-08T14:30:00Z'
			},
			{
				'id': 'json-transformer',
				'name': 'JSON Data Transformer',
				'description': 'Flexible JSON transformation engine with JSONPath, JMESPath, and custom transformation support',
				'capability_type': 'transformer',
				'status': 'published',
				'author': {'name': 'Community Contributor', 'verified': False},
				'license': 'free',
				'current_version': '1.5.2',
				'rating': {'average_rating': 4.2, 'total_reviews': 89},
				'stats': {'downloads': 7821, 'installations': 5643},
				'tags': ['json', 'transformation', 'jsonpath', 'jmespath'],
				'categories': ['Data Processing', 'Transformer'],
				'price': 0.0,
				'icon_url': 'https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/json.svg',
				'is_featured': False,
				'created_at': '2024-10-20T16:45:00Z',
				'updated_at': '2025-01-05T09:15:00Z'
			},
			{
				'id': 'rest-api-connector',
				'name': 'REST API Connector',
				'description': 'Universal REST API connector with OAuth 2.0, JWT authentication, and rate limiting support',
				'capability_type': 'connector',
				'status': 'published',
				'author': {'name': 'API Solutions Inc', 'verified': True},
				'license': 'commercial',
				'current_version': '3.0.1',
				'rating': {'average_rating': 4.6, 'total_reviews': 234},
				'stats': {'downloads': 18756, 'installations': 12453},
				'tags': ['api', 'rest', 'oauth', 'jwt', 'http'],
				'categories': ['API', 'Connector'],
				'price': 29.99,
				'currency': 'USD',
				'icon_url': 'https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/fastapi.svg',
				'is_featured': True,
				'created_at': '2024-09-10T12:30:00Z',
				'updated_at': '2025-01-07T11:20:00Z'
			},
			{
				'id': 'csv-validator',
				'name': 'CSV Data Validator',
				'description': 'Comprehensive CSV validation with schema checking, data type validation, and quality scoring',
				'capability_type': 'validator',
				'status': 'published',
				'author': {'name': 'Data Quality Team', 'verified': True},
				'license': 'open_source',
				'current_version': '1.3.4',
				'rating': {'average_rating': 4.4, 'total_reviews': 67},
				'stats': {'downloads': 4532, 'installations': 3201},
				'tags': ['csv', 'validation', 'data-quality', 'schema'],
				'categories': ['Data Quality', 'Validator'],
				'price': 0.0,
				'icon_url': 'https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/microsoftexcel.svg',
				'is_featured': False,
				'created_at': '2024-12-01T08:15:00Z',
				'updated_at': '2024-12-20T15:45:00Z'
			},
			{
				'id': 'ml-enricher',
				'name': 'ML Data Enricher',
				'description': 'Machine learning powered data enrichment with entity recognition, sentiment analysis, and prediction',
				'capability_type': 'enricher',
				'status': 'beta',
				'author': {'name': 'AI Labs', 'verified': True},
				'license': 'enterprise',
				'current_version': '0.8.0-beta',
				'rating': {'average_rating': 4.1, 'total_reviews': 23},
				'stats': {'downloads': 892, 'installations': 456},
				'tags': ['machine-learning', 'ai', 'nlp', 'enrichment'],
				'categories': ['AI/ML', 'Enricher'],
				'price': 99.99,
				'currency': 'USD',
				'icon_url': 'https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/tensorflow.svg',
				'is_featured': False,
				'created_at': '2024-12-15T14:00:00Z',
				'updated_at': '2025-01-03T10:30:00Z'
			}
		]

	def _get_mock_capability_detail(self, capability_id: str) -> Dict[str, Any]:
		"""Get detailed mock capability information"""
		capabilities = {cap['id']: cap for cap in self._get_mock_capabilities()}

		if capability_id not in capabilities:
			abort(404)

		capability = capabilities[capability_id].copy()

		# Add detailed information
		capability.update({
			'long_description': f"""
			{capability['description']}

			## Features
			- High-performance data processing
			- Advanced configuration options
			- Comprehensive error handling
			- Detailed logging and monitoring
			- REST API integration
			- Batch processing support

			## Requirements
			- Python 3.8+
			- APG Platform 1.0+
			- 512MB RAM minimum

			## Configuration
			The capability supports extensive configuration options through JSON schema validation.
			""",
			'documentation_url': f'https://docs.marketplace.apg.datacraft.co.ke/capabilities/{capability_id}',
			'source_url': f'https://github.com/datacraft/apg-{capability_id}',
			'demo_url': f'https://demo.marketplace.apg.datacraft.co.ke/{capability_id}',
			'screenshots': [
				f'https://cdn.marketplace.apg.datacraft.co.ke/{capability_id}/screenshot1.png',
				f'https://cdn.marketplace.apg.datacraft.co.ke/{capability_id}/screenshot2.png'
			],
			'requirements': {
				'python': '>=3.8',
				'apg': '>=1.0.0',
				'memory': '512MB',
				'disk': '100MB'
			},
			'configuration_schema': {
				'type': 'object',
				'properties': {
					'connection_string': {
						'type': 'string',
						'description': 'Database connection string'
					},
					'timeout': {
						'type': 'integer',
						'default': 30,
						'description': 'Connection timeout in seconds'
					},
					'retry_attempts': {
						'type': 'integer',
						'default': 3,
						'description': 'Number of retry attempts'
					}
				},
				'required': ['connection_string']
			},
			'supported_platforms': ['Linux', 'Windows', 'macOS'],
			'changelog': [
				{
					'version': '2.1.0',
					'date': '2025-01-08',
					'changes': ['Added connection pooling', 'Improved error handling', 'Performance optimizations']
				},
				{
					'version': '2.0.1',
					'date': '2024-12-15',
					'changes': ['Security fixes', 'Bug fixes']
				}
			]
		})

		return capability


class MarketplaceChartsView(BaseView):
	"""Marketplace analytics and charts"""

	route_base = '/marketplace/analytics'

	@expose('/api/download_stats')
	@has_access
	def api_download_stats(self):
		"""API endpoint for download statistics"""
		# Mock download statistics
		stats = {
			'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
			'datasets': [{
				'label': 'Total Downloads',
				'data': [1250, 1890, 2340, 2100, 2890, 3200],
				'borderColor': 'rgb(75, 192, 192)',
				'backgroundColor': 'rgba(75, 192, 192, 0.2)'
			}]
		}

		return jsonify(stats)

	@expose('/api/category_distribution')
	@has_access
	def api_category_distribution(self):
		"""API endpoint for category distribution"""
		# Mock category distribution
		distribution = {
			'labels': ['Connectors', 'Transformers', 'Validators', 'Enrichers', 'Analytics'],
			'datasets': [{
				'data': [45, 32, 28, 19, 15],
				'backgroundColor': [
					'#FF6384',
					'#36A2EB',
					'#FFCE56',
					'#4BC0C0',
					'#9966FF'
				]
			}]
		}

		return jsonify(distribution)


def init_marketplace_views(appbuilder):
	"""Initialize marketplace views"""

	# Register main dashboard
	appbuilder.add_view(
		MarketplaceDashboardView,
		"Marketplace",
		icon="fa-store",
		category="Marketplace"
	)

	# Register analytics view
	appbuilder.add_view(
		MarketplaceChartsView,
		"Analytics",
		icon="fa-chart-bar",
		category="Marketplace"
	)

	logger.info("Marketplace views initialized successfully")