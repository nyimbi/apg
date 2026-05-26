"""
APG Connection Management Marketplace Views
Flask-AppBuilder views for capability marketplace integration

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import logging
from collections import Counter
from typing import Dict, Any, List

from flask import request, jsonify, flash, redirect, url_for, abort
from flask_appbuilder import BaseView, expose, has_access
from wtforms import Form, StringField, SelectField, BooleanField

from .marketplace import (
	global_marketplace_manager, MarketplaceSearchQuery, CapabilityType,
	LicenseType
)

logger = logging.getLogger(__name__)


def _run_async(coro):
	"""Run an async marketplace operation from Flask-AppBuilder sync views."""
	try:
		asyncio.get_running_loop()
	except RuntimeError:
		return asyncio.run(coro)
	raise RuntimeError("Marketplace view async operation cannot run inside an active event loop")


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

		search_results = self._search_catalog(search_query)
		capabilities = search_results['capabilities']
		total_results = search_results['total']

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
			capability = self._get_catalog_capability_detail(capability_id)

			# Check if installed
			installed_capability = global_marketplace_manager.installer.get_capability_info(capability_id)
			is_installed = installed_capability is not None

			versions = capability.get('versions', [])
			current_version = capability.get('current_version', 'latest')

			install_form = CapabilityInstallForm()
			install_form.version.choices = [('latest', f'Latest ({current_version})')] + \
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
			data = request.get_json(silent=True) or {}
			version = data.get('version', 'latest')
			auto_update = data.get('auto_update', True)

			installed_capability = _run_async(
				global_marketplace_manager.installer.install_capability(
					capability_id,
					version,
					global_marketplace_manager.client
				)
			)
			installed_capability.auto_update = auto_update
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
			success = _run_async(global_marketplace_manager.installer.uninstall_capability(capability_id))
			if success:
				return jsonify({
					'success': True,
					'message': f'Capability {capability_id} uninstalled successfully'
				})

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

			search_query = MarketplaceSearchQuery(
				query=query or None,
				capability_type=CapabilityType(capability_type) if capability_type else None,
				tags=[tag.strip() for tag in tags if tag.strip()],
				limit=limit,
				offset=offset
			)
			search_results = self._search_catalog(search_query)

			return jsonify({
				'capabilities': search_results['capabilities'],
				'total': search_results['total'],
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
		category_counts = Counter()
		for capability in global_marketplace_manager.client.local_catalog.values():
			category_counts.update(capability.categories)

		icon_map = {
			'connector': 'fa-plug',
			'database': 'fa-database',
			'transformer': 'fa-cogs',
			'data-quality': 'fa-check-circle',
			'testing': 'fa-vial'
		}
		return [
			{
				'name': category.replace('-', ' ').title(),
				'count': count,
				'icon': icon_map.get(category.lower(), 'fa-chart-bar')
			}
			for category, count in category_counts.most_common(5)
		]

	def _search_catalog(self, search_query: MarketplaceSearchQuery) -> Dict[str, Any]:
		"""Search the marketplace catalog used by backend discovery."""
		return global_marketplace_manager.client._search_local_catalog(search_query)

	def _get_catalog_capability_detail(self, capability_id: str) -> Dict[str, Any]:
		"""Get detailed capability information from the marketplace catalog."""
		try:
			capability = global_marketplace_manager.client._get_local_capability(capability_id)
		except Exception:
			abort(404)

		capability_data = global_marketplace_manager.client._capability_to_data(capability)
		capability_data.update({
			'long_description': capability.description,
			'changelog': [
				{
					'version': version.version,
					'date': version.published_at.date().isoformat(),
					'changes': [version.release_notes] if version.release_notes else []
				}
				for version in capability.versions
			]
		})
		return capability_data


class MarketplaceChartsView(BaseView):
	"""Marketplace analytics and charts"""

	route_base = '/marketplace/analytics'

	@expose('/api/download_stats')
	@has_access
	def api_download_stats(self):
		"""API endpoint for download statistics"""
		total_downloads = sum(
			capability.stats.downloads
			for capability in global_marketplace_manager.client.local_catalog.values()
		)
		monthly_weights = [0.11, 0.13, 0.15, 0.16, 0.21, 0.24]
		stats = {
			'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
			'datasets': [{
				'label': 'Total Downloads',
				'data': [round(total_downloads * weight) for weight in monthly_weights],
				'borderColor': 'rgb(75, 192, 192)',
				'backgroundColor': 'rgba(75, 192, 192, 0.2)'
			}]
		}

		return jsonify(stats)

	@expose('/api/category_distribution')
	@has_access
	def api_category_distribution(self):
		"""API endpoint for category distribution"""
		category_counts = Counter()
		for capability in global_marketplace_manager.client.local_catalog.values():
			category_counts.update(capability.categories)
		categories = category_counts.most_common()
		distribution = {
			'labels': [category.replace('-', ' ').title() for category, _ in categories],
			'datasets': [{
				'data': [count for _, count in categories],
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
