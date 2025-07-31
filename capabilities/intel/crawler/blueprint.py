"""
APG Crawler Capability - Flask-AppBuilder Blueprint
==================================================

Flask-AppBuilder integration with:
- APG-aware views and templates
- Multi-tenant dashboard interfaces
- RAG/GraphRAG management UI
- Collaborative validation workspace
- Real-time analytics and monitoring

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime

from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.actions import action
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
import asyncio

from .service import CrawlerDatabaseService
from .models import (
	CrawlTarget, ExtractedDataset, DataRecord, ValidationSession,
	RAGChunk, GraphRAGNode, KnowledgeGraph, AnalyticsInsight
)

# =====================================================
# BLUEPRINT SETUP
# =====================================================

logger = logging.getLogger(__name__)

# Flask-AppBuilder blueprint
crawler_fab_bp = Blueprint(
	'crawler_fab', 
	__name__, 
	template_folder='templates',
	static_folder='static',
	url_prefix='/crawler'
)


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def get_tenant_id():
	"""Get tenant ID from session or request"""
	# In a real APG implementation, this would come from APG auth context
	return request.args.get('tenant_id', 'default_tenant')

def run_async(coro):
	"""Run async coroutine in sync context"""
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	
	return loop.run_until_complete(coro)


# =====================================================
# CRAWL TARGET VIEWS
# =====================================================

class CrawlTargetModelView(ModelView):
	"""Crawl Target management view"""
	
	datamodel = SQLAInterface(CrawlTarget)
	
	# List view configuration
	list_columns = ['name', 'target_type', 'status', 'rag_integration_enabled', 'created_at']
	search_columns = ['name', 'description', 'target_type', 'status']
	order_columns = ['name', 'created_at', 'updated_at']
	
	# Form configuration
	add_columns = [
		'name', 'description', 'target_urls', 'target_type', 
		'business_context', 'rag_integration_enabled', 
		'graphrag_integration_enabled', 'status'
	]
	edit_columns = add_columns
	show_columns = add_columns + ['id', 'tenant_id', 'created_at', 'updated_at']
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	@action("enable_rag", "Enable RAG", "Enable RAG processing for selected targets?", "fa-cogs")
	def enable_rag_action(self, items):
		"""Bulk action to enable RAG processing"""
		for item in items:
			item.rag_integration_enabled = True
		
		self.datamodel.session.commit()
		flash(f"RAG enabled for {len(items)} targets", "success")
		return redirect(url_for('CrawlTargetModelView.list'))
	
	@action("disable_rag", "Disable RAG", "Disable RAG processing for selected targets?", "fa-times")
	def disable_rag_action(self, items):
		"""Bulk action to disable RAG processing"""
		for item in items:
			item.rag_integration_enabled = False
		
		self.datamodel.session.commit()
		flash(f"RAG disabled for {len(items)} targets", "success")
		return redirect(url_for('CrawlTargetModelView.list'))


class ExtractedDatasetModelView(ModelView):
	"""Extracted Dataset management view"""
	
	datamodel = SQLAInterface(ExtractedDataset)
	
	# List view configuration
	list_columns = ['dataset_name', 'extraction_method', 'record_count', 'validation_status', 'created_at']
	search_columns = ['dataset_name', 'extraction_method', 'validation_status']
	order_columns = ['dataset_name', 'record_count', 'created_at']
	
	# Form configuration
	show_columns = [
		'dataset_name', 'extraction_method', 'record_count', 'validation_status',
		'consensus_score', 'quality_metrics', 'created_at', 'updated_at'
	]
	
	# Read-only view (datasets are created through processing)
	base_permissions = ['can_list', 'can_show']


class DataRecordModelView(ModelView):
	"""Data Record management view"""
	
	datamodel = SQLAInterface(DataRecord)
	
	# List view configuration
	list_columns = ['record_index', 'source_url', 'content_processing_stage', 'quality_score', 'validation_status']
	search_columns = ['source_url', 'content_processing_stage', 'validation_status']
	order_columns = ['record_index', 'quality_score', 'created_at']
	
	# Show configuration
	show_columns = [
		'record_index', 'source_url', 'content_processing_stage', 
		'quality_score', 'confidence_score', 'validation_status',
		'content_fingerprint', 'created_at'
	]
	
	# Permissions
	base_permissions = ['can_list', 'can_show']


# =====================================================
# RAG AND GRAPHRAG VIEWS
# =====================================================

class RAGChunkModelView(ModelView):
	"""RAG Chunk management view"""
	
	datamodel = SQLAInterface(RAGChunk)
	
	# List view configuration
	list_columns = ['chunk_index', 'embedding_model', 'indexing_status', 'created_at']
	search_columns = ['chunk_text', 'embedding_model', 'indexing_status']
	order_columns = ['chunk_index', 'created_at']
	
	# Show configuration
	show_columns = [
		'chunk_index', 'chunk_text', 'chunk_fingerprint', 
		'embedding_model', 'vector_dimensions', 'indexing_status', 'created_at'
	]
	
	# Read-only view
	base_permissions = ['can_list', 'can_show']


class GraphRAGNodeModelView(ModelView):
	"""GraphRAG Node management view"""
	
	datamodel = SQLAInterface(GraphRAGNode)
	
	# List view configuration
	list_columns = ['node_name', 'node_type', 'entity_type', 'confidence_score', 'node_status']
	search_columns = ['node_name', 'node_type', 'entity_type', 'node_status']
	order_columns = ['node_name', 'confidence_score', 'created_at']
	
	# Show configuration
	show_columns = [
		'node_name', 'node_type', 'node_description', 'entity_type',
		'confidence_score', 'salience_score', 'node_status', 'created_at'
	]
	
	# Read-only view
	base_permissions = ['can_list', 'can_show']


class KnowledgeGraphModelView(ModelView):
	"""Knowledge Graph management view"""
	
	datamodel = SQLAInterface(KnowledgeGraph)
	
	# List view configuration
	list_columns = ['graph_name', 'domain', 'node_count', 'relation_count', 'graph_status']
	search_columns = ['graph_name', 'domain', 'graph_status']
	order_columns = ['graph_name', 'node_count', 'last_updated']
	
	# Form configuration
	add_columns = ['graph_name', 'description', 'domain']
	edit_columns = add_columns + ['graph_status']
	show_columns = [
		'graph_name', 'description', 'domain', 'node_count', 'relation_count',
		'entity_types', 'relation_types', 'graph_status', 'created_at', 'last_updated'
	]
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit']


# =====================================================
# VALIDATION VIEWS
# =====================================================

class ValidationSessionModelView(ModelView):
	"""Validation Session management view"""
	
	datamodel = SQLAInterface(ValidationSession)
	
	# List view configuration
	list_columns = ['session_name', 'session_status', 'validator_count', 'completion_percentage', 'created_at']
	search_columns = ['session_name', 'session_status']
	order_columns = ['session_name', 'completion_percentage', 'created_at']
	
	# Form configuration
	add_columns = ['session_name', 'description', 'consensus_threshold', 'quality_threshold']
	edit_columns = add_columns + ['session_status']
	show_columns = [
		'session_name', 'description', 'session_status', 'validator_count',
		'completion_percentage', 'consensus_threshold', 'quality_threshold', 'created_at'
	]
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit']


# =====================================================
# DASHBOARD VIEWS
# =====================================================

class CrawlerDashboardView(BaseView):
	"""Main crawler dashboard"""
	
	default_view = 'dashboard'
	
	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Main dashboard view"""
		tenant_id = get_tenant_id()
		
		# Get dashboard metrics (mock data for now)
		metrics = {
			'total_targets': 25,
			'active_crawls': 8,
			'total_records': 15420,
			'rag_chunks': 8950,
			'graphrag_nodes': 1240,
			'validation_sessions': 5
		}
		
		return self.render_template(
			'crawler/dashboard.html',
			metrics=metrics,
			tenant_id=tenant_id
		)
	
	@expose('/analytics')
	@has_access
	def analytics(self):
		"""Analytics dashboard"""
		tenant_id = get_tenant_id()
		
		# Mock analytics data
		analytics = {
			'crawl_performance': {
				'success_rate': 94.5,
				'average_quality': 87.2,
				'processing_speed': '2.3k records/hour'
			},
			'rag_metrics': {
				'chunk_count': 8950,
				'embedding_coverage': 98.7,
				'search_accuracy': 91.3
			},
			'graphrag_metrics': {
				'entity_count': 1240,
				'relation_count': 890,
				'graph_density': 0.65
			}
		}
		
		return self.render_template(
			'crawler/analytics.html',
			analytics=analytics,
			tenant_id=tenant_id
		)


class RAGManagementView(BaseView):
	"""RAG processing management view"""
	
	default_view = 'rag_overview'
	
	@expose('/rag')
	@has_access
	def rag_overview(self):
		"""RAG processing overview"""
		tenant_id = get_tenant_id()
		
		# Mock RAG data
		rag_data = {
			'processing_queue': 15,
			'indexed_chunks': 8950,
			'embedding_models': ['text-embedding-ada-002', 'text-embedding-3-small'],
			'vector_indexes': 3,
			'search_performance': '< 50ms average'
		}
		
		return self.render_template(
			'crawler/rag_management.html',
			rag_data=rag_data,
			tenant_id=tenant_id
		)
	
	@expose('/rag/search')
	@has_access
	def rag_search(self):
		"""RAG semantic search interface"""
		return self.render_template('crawler/rag_search.html')


class GraphRAGManagementView(BaseView):
	"""GraphRAG knowledge graph management view"""
	
	default_view = 'graphrag_overview'
	
	@expose('/graphrag')
	@has_access
	def graphrag_overview(self):
		"""GraphRAG overview"""
		tenant_id = get_tenant_id()
		
		# Mock GraphRAG data
		graphrag_data = {
			'knowledge_graphs': 3,
			'total_entities': 1240,
			'total_relations': 890,
			'entity_types': ['Organization', 'Person', 'Location', 'Product'],
			'relation_types': ['works_at', 'located_in', 'produces', 'related_to']
		}
		
		return self.render_template(
			'crawler/graphrag_management.html',
			graphrag_data=graphrag_data,
			tenant_id=tenant_id
		)


# =====================================================
# CHARTS AND ANALYTICS
# =====================================================

class CrawlerMetricsChartView(DirectByChartView):
	"""Crawler metrics charts"""
	
	chart_title = "Crawler Performance Metrics"
	chart_type = "LineChart"
	direct_columns = {
		'x_axis': 'created_at',
		'y_axis': 'record_count'
	}
	base_order = ('created_at', 'asc')


# =====================================================
# TEMPLATE FUNCTIONS
# =====================================================

@crawler_fab_bp.app_template_filter('format_processing_stage')
def format_processing_stage(stage):
	"""Format processing stage for display"""
	stage_labels = {
		'raw_extracted': 'Raw Extracted',
		'cleaned': 'Cleaned',
		'markdown_converted': 'Markdown Converted',
		'fingerprinted': 'Fingerprinted',
		'rag_processed': 'RAG Processed',
		'graphrag_processed': 'GraphRAG Processed',
		'knowledge_graph_integrated': 'Knowledge Graph Integrated'
	}
	return stage_labels.get(stage, stage.replace('_', ' ').title())

@crawler_fab_bp.app_template_filter('format_file_size')
def format_file_size(size_bytes):
	"""Format file size for display"""
	if size_bytes < 1024:
		return f"{size_bytes} B"
	elif size_bytes < 1024 * 1024:
		return f"{size_bytes / 1024:.1f} KB"
	elif size_bytes < 1024 * 1024 * 1024:
		return f"{size_bytes / (1024 * 1024):.1f} MB"
	else:
		return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


# =====================================================
# BLUEPRINT REGISTRATION
# =====================================================

def register_crawler_views(appbuilder):
	"""Register all crawler views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		CrawlTargetModelView,
		"Crawl Targets",
		icon="fa-globe",
		category="Crawler Management"
	)
	
	appbuilder.add_view(
		ExtractedDatasetModelView,
		"Datasets",
		icon="fa-database",
		category="Crawler Management"
	)
	
	appbuilder.add_view(
		DataRecordModelView,
		"Data Records",
		icon="fa-file-text",
		category="Crawler Management"
	)
	
	# RAG views
	appbuilder.add_view(
		RAGChunkModelView,
		"RAG Chunks",
		icon="fa-puzzle-piece",
		category="RAG Management"
	)
	
	appbuilder.add_view(
		RAGManagementView,
		"RAG Overview",
		icon="fa-search",
		category="RAG Management"
	)
	
	# GraphRAG views
	appbuilder.add_view(
		GraphRAGNodeModelView,
		"Graph Nodes",
		icon="fa-sitemap",
		category="GraphRAG"
	)
	
	appbuilder.add_view(
		KnowledgeGraphModelView,
		"Knowledge Graphs",
		icon="fa-share-alt",
		category="GraphRAG"
	)
	
	appbuilder.add_view(
		GraphRAGManagementView,
		"GraphRAG Overview",
		icon="fa-network-wired",
		category="GraphRAG"
	)
	
	# Validation views
	appbuilder.add_view(
		ValidationSessionModelView,
		"Validation Sessions",
		icon="fa-check-circle",
		category="Quality Management"
	)
	
	# Dashboard views
	appbuilder.add_view(
		CrawlerDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="Overview"
	)
	
	# Analytics views
	appbuilder.add_view(
		CrawlerMetricsChartView,
		"Performance Charts",
		icon="fa-chart-line",
		category="Analytics"
	)
	
	logger.info("APG Crawler Flask-AppBuilder views registered successfully")


# Export functions
__all__ = ['crawler_fab_bp', 'register_crawler_views']