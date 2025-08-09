#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Flask-AppBuilder Blueprint
Flask-AppBuilder integration with APG composition engine

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from flask import Blueprint, jsonify, request, render_template
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.baseviews import BaseView, ModelView
from flask_appbuilder.charts.views import ChartView
from flask_appbuilder.actions import action
from flask_appbuilder.security.decorators import protect
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import json

from .service import CacheService, CacheServiceConfig
from .models import CacheEntry, CacheCluster, CachePolicy
from .dashboard import CacheDashboardView


# Configure logging
logger = logging.getLogger('cach.blueprint')


# Global cache service instance
_cache_service: Optional[CacheService] = None


async def get_cache_service() -> CacheService:
	"""Get or create cache service instance"""
	global _cache_service
	if _cache_service is None:
		config = CacheServiceConfig()
		_cache_service = CacheService(config)
		await _cache_service.initialize()
	return _cache_service


def get_cache_service_sync() -> CacheService:
	"""Synchronous wrapper for getting cache service"""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			# If loop is already running, we need to handle this differently
			return _cache_service
		else:
			return loop.run_until_complete(get_cache_service())
	except RuntimeError:
		# No event loop, create one
		return asyncio.run(get_cache_service())


class CacheManagementView(BaseView):
	"""Main cache management dashboard view"""
	
	route_base = '/cache'
	default_view = 'dashboard'
	
	@expose('/')
	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Main cache management dashboard"""
		try:
			service = get_cache_service_sync()
			if service:
				# Get current statistics
				stats = asyncio.run(service.get_stats())
				performance_history = asyncio.run(service.get_performance_history())
				ai_insights = asyncio.run(service.get_ai_insights())
				
				return render_template(
					'cach/dashboard.html',
					title="Cache Management Dashboard",
					stats=stats,
					performance_history=performance_history,
					ai_insights=ai_insights,
					timestamp=datetime.utcnow()
				)
			else:
				return render_template(
					'cach/error.html',
					error="Cache service not available",
					title="Cache Management - Error"
				)
		except Exception as e:
			logger.error(f"Error in dashboard view: {e}")
			return render_template(
				'cach/error.html',
				error=str(e),
				title="Cache Management - Error"
			)
	
	@expose('/explorer')
	@has_access
	def explorer(self):
		"""Cache data explorer interface"""
		try:
			service = get_cache_service_sync()
			if service:
				# Get namespaces and basic stats for explorer
				namespaces = list(set(
					key.split(':')[1] for key in service._cache_store.keys()
					if ':' in key
				))
				
				total_entries = len(service._cache_store)
				
				return render_template(
					'cach/explorer.html',
					title="Cache Data Explorer",
					namespaces=namespaces,
					total_entries=total_entries
				)
			else:
				return render_template(
					'cach/error.html',
					error="Cache service not available"
				)
		except Exception as e:
			logger.error(f"Error in explorer view: {e}")
			return render_template(
				'cach/error.html',
				error=str(e)
			)
	
	@expose('/policies')
	@has_access
	def policies(self):
		"""Cache policy management interface"""
		try:
			service = get_cache_service_sync()
			if service:
				# Get all policies
				policies = [
					{
						'policy_id': policy.policy_id,
						'name': policy.name,
						'description': policy.description,
						'key_patterns': policy.key_patterns,
						'enabled': policy.enabled,
						'effectiveness_score': policy.effectiveness_score,
						'created_at': policy.created_at.isoformat(),
						'applied_count': policy.applied_count
					}
					for policy in service._policies.values()
				]
				
				return render_template(
					'cach/policies.html',
					title="Cache Policy Management",
					policies=policies
				)
			else:
				return render_template(
					'cach/error.html',
					error="Cache service not available"
				)
		except Exception as e:
			logger.error(f"Error in policies view: {e}")
			return render_template(
				'cach/error.html',
				error=str(e)
			)
	
	@expose('/analytics')
	@has_access
	def analytics(self):
		"""Advanced cache analytics and AI insights"""
		try:
			service = get_cache_service_sync()
			if service:
				# Get comprehensive analytics data
				stats = asyncio.run(service.get_stats())
				performance_history = asyncio.run(service.get_performance_history())
				ai_insights = asyncio.run(service.get_ai_insights())
				
				# Calculate trends
				trends = calculate_performance_trends(performance_history)
				
				return render_template(
					'cach/analytics.html',
					title="Cache Analytics & AI Insights",
					stats=stats,
					performance_history=performance_history,
					ai_insights=ai_insights,
					trends=trends
				)
			else:
				return render_template(
					'cach/error.html',
					error="Cache service not available"
				)
		except Exception as e:
			logger.error(f"Error in analytics view: {e}")
			return render_template(
				'cach/error.html',
				error=str(e)
			)
	
	@expose('/clusters')
	@has_access
	def clusters(self):
		"""Cache cluster management interface"""
		try:
			service = get_cache_service_sync()
			if service:
				# Get all clusters
				clusters = [
					{
						'cluster_id': cluster.cluster_id,
						'name': cluster.name,
						'description': cluster.description,
						'backend_type': cluster.backend_type.value,
						'nodes': cluster.nodes,
						'healthy': cluster.healthy,
						'created_at': cluster.created_at.isoformat(),
						'max_memory_mb': cluster.max_memory_mb,
						'ai_optimization_enabled': cluster.ai_optimization_enabled
					}
					for cluster in service._clusters.values()
				]
				
				return render_template(
					'cach/clusters.html',
					title="Cache Cluster Management",
					clusters=clusters
				)
			else:
				return render_template(
					'cach/error.html',
					error="Cache service not available"
				)
		except Exception as e:
			logger.error(f"Error in clusters view: {e}")
			return render_template(
				'cach/error.html',
				error=str(e)
			)
	
	@action('optimize_performance', 'Optimize Performance', 'Trigger AI optimization', icon='fa-magic')
	def optimize_performance(self, items):
		"""Trigger AI performance optimization"""
		try:
			service = get_cache_service_sync()
			if service:
				asyncio.run(service._run_ai_optimization())
				return jsonify({
					'success': True,
					'message': 'AI optimization triggered successfully',
					'timestamp': datetime.utcnow().isoformat()
				})
		except Exception as e:
			logger.error(f"Error triggering optimization: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			})


class CacheAPIView(BaseView):
	"""REST API endpoints for cache operations"""
	
	route_base = '/api/cache'
	
	@expose('/stats')
	def api_stats(self):
		"""Get cache statistics via API"""
		try:
			service = get_cache_service_sync()
			if service:
				stats = asyncio.run(service.get_stats())
				return jsonify({
					'success': True,
					'data': stats,
					'timestamp': datetime.utcnow().isoformat()
				})
			else:
				return jsonify({
					'success': False,
					'error': 'Cache service not available'
				}), 503
		except Exception as e:
			logger.error(f"Error getting stats: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/get/<key>')
	def api_get(self, key):
		"""Get cache value via API"""
		try:
			service = get_cache_service_sync()
			if service:
				namespace = request.args.get('namespace', 'default')
				value = asyncio.run(service.get(key=key, namespace=namespace))
				
				if value is not None:
					return jsonify({
						'success': True,
						'key': key,
						'value': value,
						'namespace': namespace,
						'found': True
					})
				else:
					return jsonify({
						'success': True,
						'key': key,
						'value': None,
						'namespace': namespace,
						'found': False
					}), 404
			else:
				return jsonify({
					'success': False,
					'error': 'Cache service not available'
				}), 503
		except Exception as e:
			logger.error(f"Error getting cache value: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/set', methods=['POST'])
	def api_set(self):
		"""Set cache value via API"""
		try:
			service = get_cache_service_sync()
			if service:
				data = request.get_json()
				if not data or 'key' not in data or 'value' not in data:
					return jsonify({
						'success': False,
						'error': 'Missing required fields: key, value'
					}), 400
				
				success = asyncio.run(service.set(
					key=data['key'],
					value=data['value'],
					ttl_seconds=data.get('ttl_seconds'),
					namespace=data.get('namespace', 'default')
				))
				
				return jsonify({
					'success': success,
					'key': data['key'],
					'namespace': data.get('namespace', 'default')
				})
			else:
				return jsonify({
					'success': False,
					'error': 'Cache service not available'
				}), 503
		except Exception as e:
			logger.error(f"Error setting cache value: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/delete/<key>', methods=['DELETE'])
	def api_delete(self, key):
		"""Delete cache value via API"""
		try:
			service = get_cache_service_sync()
			if service:
				namespace = request.args.get('namespace', 'default')
				success = asyncio.run(service.delete(key=key, namespace=namespace))
				
				return jsonify({
					'success': success,
					'key': key,
					'namespace': namespace
				})
			else:
				return jsonify({
					'success': False,
					'error': 'Cache service not available'
				}), 503
		except Exception as e:
			logger.error(f"Error deleting cache value: {e}")
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/health')
	def api_health(self):
		"""Health check endpoint"""
		try:
			service = get_cache_service_sync()
			if service and service.running:
				stats = asyncio.run(service.get_stats())
				return jsonify({
					'healthy': True,
					'service': 'APG Cache Management',
					'version': '1.0.0',
					'stats': {
						'total_entries': stats['total_entries'],
						'hit_rate': stats['hit_rate'],
						'memory_utilization': stats['memory_utilization']
					},
					'timestamp': datetime.utcnow().isoformat()
				})
			else:
				return jsonify({
					'healthy': False,
					'service': 'APG Cache Management',
					'error': 'Service not running'
				}), 503
		except Exception as e:
			logger.error(f"Health check error: {e}")
			return jsonify({
				'healthy': False,
				'service': 'APG Cache Management',
				'error': str(e)
			}), 500


class CacheChartView(ChartView):
	"""Chart views for cache performance visualization"""
	
	route_base = '/cache/charts'
	
	@expose('/performance')
	@has_access
	def performance_chart(self):
		"""Performance metrics chart"""
		try:
			service = get_cache_service_sync()
			if service:
				performance_history = asyncio.run(service.get_performance_history())
				
				# Format data for charts
				chart_data = {
					'labels': [item['timestamp'] for item in performance_history[-50:]],
					'datasets': [
						{
							'label': 'Hit Rate (%)',
							'data': [item['hit_rate'] * 100 for item in performance_history[-50:]],
							'borderColor': 'rgb(75, 192, 192)',
							'tension': 0.1
						},
						{
							'label': 'Memory Utilization (%)',
							'data': [item['memory_utilization'] for item in performance_history[-50:]],
							'borderColor': 'rgb(255, 99, 132)',
							'tension': 0.1
						}
					]
				}
				
				return jsonify(chart_data)
			else:
				return jsonify({'error': 'Cache service not available'}), 503
		except Exception as e:
			logger.error(f"Error generating performance chart: {e}")
			return jsonify({'error': str(e)}), 500


def calculate_performance_trends(history: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Calculate performance trends from history data"""
	if not history or len(history) < 2:
		return {
			'hit_rate_trend': 0,
			'memory_trend': 0,
			'latency_trend': 0,
			'operations_trend': 0
		}
	
	recent = history[-10:]  # Last 10 data points
	older = history[-20:-10] if len(history) >= 20 else history[:-10]
	
	if not older:
		return {
			'hit_rate_trend': 0,
			'memory_trend': 0,
			'latency_trend': 0,
			'operations_trend': 0
		}
	
	# Calculate averages
	recent_hit_rate = sum(item['hit_rate'] for item in recent) / len(recent)
	older_hit_rate = sum(item['hit_rate'] for item in older) / len(older)
	
	recent_memory = sum(item['memory_utilization'] for item in recent) / len(recent)
	older_memory = sum(item['memory_utilization'] for item in older) / len(older)
	
	recent_latency = sum(item['average_latency_ms'] for item in recent) / len(recent)
	older_latency = sum(item['average_latency_ms'] for item in older) / len(older)
	
	recent_ops = sum(item['operations_per_second'] for item in recent) / len(recent)
	older_ops = sum(item['operations_per_second'] for item in older) / len(older)
	
	return {
		'hit_rate_trend': ((recent_hit_rate - older_hit_rate) / max(older_hit_rate, 0.01)) * 100,
		'memory_trend': ((recent_memory - older_memory) / max(older_memory, 0.01)) * 100,
		'latency_trend': ((recent_latency - older_latency) / max(older_latency, 0.01)) * 100,
		'operations_trend': ((recent_ops - older_ops) / max(older_ops, 0.01)) * 100
	}


# Create Flask Blueprint
cache_blueprint = Blueprint(
	'cache',
	__name__,
	template_folder='templates',
	static_folder='static',
	url_prefix='/cache'
)


# Register blueprint routes manually for non-AppBuilder usage
@cache_blueprint.route('/')
def dashboard():
	"""Dashboard route for direct blueprint usage"""
	try:
		service = get_cache_service_sync()
		if service:
			stats = asyncio.run(service.get_stats())
			return jsonify({
				'service': 'APG Cache Management',
				'status': 'running',
				'stats': stats,
				'timestamp': datetime.utcnow().isoformat()
			})
		else:
			return jsonify({
				'service': 'APG Cache Management',
				'status': 'unavailable',
				'error': 'Cache service not initialized'
			}), 503
	except Exception as e:
		logger.error(f"Dashboard error: {e}")
		return jsonify({
			'service': 'APG Cache Management',
			'status': 'error',
			'error': str(e)
		}), 500


@cache_blueprint.route('/health')
def health():
	"""Health check for direct blueprint usage"""
	try:
		service = get_cache_service_sync()
		return jsonify({
			'healthy': service is not None and service.running,
			'service': 'APG Cache Management',
			'timestamp': datetime.utcnow().isoformat()
		})
	except Exception as e:
		logger.error(f"Health check error: {e}")
		return jsonify({
			'healthy': False,
			'service': 'APG Cache Management',
			'error': str(e)
		}), 500


# APG Capability Registration
CAPABILITY_METADATA = {
	'name': 'cach',
	'display_name': 'Cache Management',
	'description': 'AI-powered cache management with autonomous optimization',
	'version': '1.0.0',
	'category': 'data_layer',
	'tags': ['caching', 'performance', 'ai', 'optimization'],
	'blueprint': cache_blueprint,
	'views': [CacheManagementView, CacheAPIView, CacheChartView, CacheDashboardView],
	'composition': {
		'load_order': 10,
		'dependencies': ['auth', 'audl', 'mten', 'moni', 'conf'],
		'optional_dependencies': ['aicr', 'pred', 'anom', 'agnt'],
		'export_functions': [
			'get_cache_value',
			'set_cache_value',
			'delete_cache_value',
			'clear_namespace',
			'get_cache_stats',
			'optimize_cache'
		],
		'event_handlers': {
			'cache.set': 'handle_cache_set',
			'cache.get': 'handle_cache_get',
			'cache.delete': 'handle_cache_delete',
			'cache.optimized': 'handle_cache_optimized'
		}
	},
	'endpoints': {
		'dashboard': '/cache/dashboard',
		'api_stats': '/api/cache/stats',
		'api_health': '/api/cache/health',
		'explorer': '/cache/explorer',
		'policies': '/cache/policies',
		'analytics': '/cache/analytics'
	}
}


def register_with_appbuilder(appbuilder):
	"""Register views with Flask-AppBuilder"""
	try:
		appbuilder.add_view(CacheDashboardView, "Cache Dashboard", icon="fa-dashboard", category="Cache Management")
		appbuilder.add_view(CacheManagementView, "Cache Management", icon="fa-cogs", category="Cache Management")
		appbuilder.add_view(CacheAPIView, "Cache API", icon="fa-code", category="Cache Management")
		appbuilder.add_view(CacheChartView, "Cache Charts", icon="fa-bar-chart", category="Cache Management")
		logger.info("Successfully registered cache management views with Flask-AppBuilder")
	except Exception as e:
		logger.error(f"Error registering cache management views: {e}")


# Export main components
__all__ = [
	'cache_blueprint',
	'CacheManagementView',
	'CacheAPIView', 
	'CacheChartView',
	'CacheDashboardView',
	'CAPABILITY_METADATA',
	'register_with_appbuilder'
]