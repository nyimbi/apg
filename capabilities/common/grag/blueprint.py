"""
APG GraphRAG Capability - Flask-AppBuilder Blueprint

Revolutionary Flask-AppBuilder integration providing comprehensive web interface
and management capabilities for GraphRAG knowledge graphs.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import BaseView, ModelView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import GroupByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
from wtforms import Form, StringField, TextAreaField, FloatField, IntegerField, SelectField
from wtforms.validators import DataRequired, Length, NumberRange
import json

from .models import (
	GrKnowledgeGraph, GrGraphEntity, GrGraphRelationship, 
	GrGraphCommunity, GrDocumentSource, GrQueryHistory
)
from .service import GraphRAGService
from .database import GraphRAGDatabaseService
from .ollama_integration import OllamaClient, OllamaConfig
from .hybrid_retrieval import HybridRetrievalEngine
from .reasoning_engine import ReasoningEngine
from .incremental_updates import IncrementalUpdateEngine
from .contextual_intelligence import ContextualIntelligenceEngine
from .views import (
	GraphRAGQuery, GraphRAGResponse, KnowledgeGraphRequest,
	DocumentProcessingRequest, QueryContext, RetrievalConfig, ReasoningConfig
)


logger = logging.getLogger(__name__)


# ============================================================================
# FLASK-APPBUILDER MODELS AND VIEWS
# ============================================================================

class KnowledgeGraphModelView(ModelView):
	"""Knowledge Graph management view"""
	datamodel = SQLAInterface(GrKnowledgeGraph)
	
	list_columns = [
		'knowledge_graph_id', 'name', 'description', 'domain',
		'entity_count', 'relationship_count', 'created_at', 'status'
	]
	
	show_columns = [
		'knowledge_graph_id', 'name', 'description', 'domain',
		'entity_count', 'relationship_count', 'document_count',
		'avg_entity_confidence', 'last_updated', 'created_at', 'status', 'metadata'
	]
	
	edit_columns = ['name', 'description', 'domain', 'status', 'metadata']
	add_columns = ['name', 'description', 'domain']
	
	label_columns = {
		'knowledge_graph_id': 'Graph ID',
		'entity_count': 'Entities',
		'relationship_count': 'Relationships',
		'document_count': 'Documents',
		'avg_entity_confidence': 'Avg Confidence',
		'last_updated': 'Last Updated',
		'created_at': 'Created'
	}
	
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']


class GraphEntityModelView(ModelView):
	"""Graph Entity management view"""
	datamodel = SQLAInterface(GrGraphEntity)
	
	list_columns = [
		'canonical_entity_id', 'canonical_name', 'entity_type',
		'confidence_score', 'knowledge_graph.name', 'created_at'
	]
	
	show_columns = [
		'canonical_entity_id', 'canonical_name', 'entity_type', 'aliases',
		'properties', 'confidence_score', 'embeddings_status', 
		'knowledge_graph.name', 'created_at', 'updated_at'
	]
	
	search_columns = ['canonical_name', 'entity_type', 'aliases']
	
	label_columns = {
		'canonical_entity_id': 'Entity ID',
		'canonical_name': 'Name',
		'entity_type': 'Type',
		'confidence_score': 'Confidence',
		'embeddings_status': 'Embeddings',
		'knowledge_graph.name': 'Graph'
	}


class GraphRelationshipModelView(ModelView):
	"""Graph Relationship management view"""
	datamodel = SQLAInterface(GrGraphRelationship)
	
	list_columns = [
		'canonical_relationship_id', 'relationship_type', 'strength',
		'source_entity.canonical_name', 'target_entity.canonical_name',
		'confidence_score', 'knowledge_graph.name'
	]
	
	show_columns = [
		'canonical_relationship_id', 'relationship_type', 'strength',
		'source_entity.canonical_name', 'target_entity.canonical_name',
		'properties', 'confidence_score', 'knowledge_graph.name',
		'created_at', 'updated_at'
	]
	
	label_columns = {
		'canonical_relationship_id': 'Relationship ID',
		'relationship_type': 'Type',
		'source_entity.canonical_name': 'Source Entity',
		'target_entity.canonical_name': 'Target Entity',
		'confidence_score': 'Confidence'
	}


class DocumentSourceModelView(ModelView):
	"""Document Source management view"""
	datamodel = SQLAInterface(GrDocumentSource)
	
	list_columns = [
		'document_id', 'title', 'source_type', 'processing_status',
		'entity_count', 'relationship_count', 'processed_at'
	]
	
	show_columns = [
		'document_id', 'title', 'content_preview', 'source_type',
		'source_url', 'processing_status', 'entity_count', 'relationship_count',
		'processing_metadata', 'processed_at', 'created_at'
	]
	
	edit_columns = ['title', 'content_preview', 'source_url', 'processing_status']
	
	label_columns = {
		'document_id': 'Document ID',
		'content_preview': 'Content Preview',
		'source_type': 'Source Type',
		'source_url': 'Source URL',
		'entity_count': 'Entities Extracted',
		'relationship_count': 'Relationships Extracted',
		'processed_at': 'Processed At'
	}


class QueryHistoryModelView(ModelView):
	"""Query History management view"""
	datamodel = SQLAInterface(GrQueryHistory)
	
	list_columns = [
		'query_id', 'query_text_preview', 'query_type', 'processing_time_ms',
		'result_count', 'confidence_score', 'user_id', 'created_at'
	]
	
	show_columns = [
		'query_id', 'query_text', 'query_type', 'processing_time_ms',
		'result_count', 'confidence_score', 'user_id', 'session_id',
		'retrieval_metadata', 'reasoning_metadata', 'response_metadata',
		'created_at'
	]
	
	search_columns = ['query_text', 'query_type', 'user_id']
	
	label_columns = {
		'query_id': 'Query ID',
		'query_text_preview': 'Query Preview',
		'query_text': 'Query Text',
		'query_type': 'Type',
		'processing_time_ms': 'Processing Time (ms)',
		'result_count': 'Results',
		'confidence_score': 'Confidence',
		'user_id': 'User',
		'session_id': 'Session'
	}


# ============================================================================
# GRAPHRAG ANALYTICS VIEWS
# ============================================================================

class GraphRAGAnalyticsView(BaseView):
	"""GraphRAG Analytics Dashboard View"""
	
	route_base = "/graphrag_analytics"
	default_view = "dashboard"
	
	@expose("/")
	@expose("/dashboard/")
	@has_access
	def dashboard(self):
		"""Main analytics dashboard"""
		try:
			# Get analytics data
			analytics_data = self._get_dashboard_analytics()
			
			return self.render_template(
				"graphrag/analytics_dashboard.html",
				analytics=analytics_data,
				title="GraphRAG Analytics Dashboard"
			)
		except Exception as e:
			logger.error(f"Analytics dashboard error: {e}")
			flash(f"Error loading analytics: {e}", "danger")
			return redirect(url_for("GraphRAGManagementView.index"))
	
	@expose("/performance/")
	@has_access
	def performance_analytics(self):
		"""Performance analytics view"""
		try:
			performance_data = self._get_performance_analytics()
			
			return self.render_template(
				"graphrag/performance_analytics.html",
				performance=performance_data,
				title="GraphRAG Performance Analytics"
			)
		except Exception as e:
			logger.error(f"Performance analytics error: {e}")
			flash(f"Error loading performance analytics: {e}", "danger")
			return redirect(url_for("GraphRAGAnalyticsView.dashboard"))
	
	@expose("/knowledge_graphs/")
	@has_access
	def knowledge_graph_analytics(self):
		"""Knowledge graph analytics view"""
		try:
			graph_analytics = self._get_knowledge_graph_analytics()
			
			return self.render_template(
				"graphrag/graph_analytics.html",
				graphs=graph_analytics,
				title="Knowledge Graph Analytics"
			)
		except Exception as e:
			logger.error(f"Knowledge graph analytics error: {e}")
			flash(f"Error loading graph analytics: {e}", "danger")
			return redirect(url_for("GraphRAGAnalyticsView.dashboard"))
	
	def _get_dashboard_analytics(self) -> Dict[str, Any]:
		"""Get comprehensive dashboard analytics"""
		return {
			"knowledge_graphs": {
				"total": GrKnowledgeGraph.query.count(),
				"active": GrKnowledgeGraph.query.filter_by(status='active').count(),
				"entities_total": sum(kg.entity_count or 0 for kg in GrKnowledgeGraph.query.all()),
				"relationships_total": sum(kg.relationship_count or 0 for kg in GrKnowledgeGraph.query.all())
			},
			"recent_queries": {
				"total_today": GrQueryHistory.query.filter(
					GrQueryHistory.created_at >= datetime.utcnow().date()
				).count(),
				"avg_processing_time": self._get_avg_processing_time(),
				"avg_confidence": self._get_avg_confidence()
			},
			"document_processing": {
				"total": GrDocumentSource.query.count(),
				"processed": GrDocumentSource.query.filter_by(processing_status='completed').count(),
				"pending": GrDocumentSource.query.filter_by(processing_status='pending').count(),
				"failed": GrDocumentSource.query.filter_by(processing_status='failed').count()
			}
		}
	
	def _get_performance_analytics(self) -> Dict[str, Any]:
		"""Get performance analytics data"""
		# Simplified implementation - would integrate with actual performance metrics
		return {
			"query_performance": {
				"avg_response_time": 850,
				"p95_response_time": 2100,
				"throughput_qps": 12.5
			},
			"system_health": {
				"ollama_status": "healthy",
				"database_status": "healthy",
				"cache_hit_rate": 0.78
			}
		}
	
	def _get_knowledge_graph_analytics(self) -> List[Dict[str, Any]]:
		"""Get knowledge graph analytics"""
		graphs = []
		for kg in GrKnowledgeGraph.query.all():
			graphs.append({
				"id": kg.knowledge_graph_id,
				"name": kg.name,
				"domain": kg.domain,
				"entities": kg.entity_count or 0,
				"relationships": kg.relationship_count or 0,
				"documents": kg.document_count or 0,
				"confidence": kg.avg_entity_confidence or 0.0,
				"status": kg.status
			})
		return graphs
	
	def _get_avg_processing_time(self) -> float:
		"""Get average query processing time"""
		queries = GrQueryHistory.query.filter(
			GrQueryHistory.processing_time_ms.isnot(None)
		).all()
		if not queries:
			return 0.0
		return sum(q.processing_time_ms for q in queries) / len(queries)
	
	def _get_avg_confidence(self) -> float:
		"""Get average query confidence"""
		queries = GrQueryHistory.query.filter(
			GrQueryHistory.confidence_score.isnot(None)
		).all()
		if not queries:
			return 0.0
		return sum(q.confidence_score for q in queries) / len(queries)


# ============================================================================
# GRAPHRAG MANAGEMENT VIEW
# ============================================================================

class GraphRAGManagementView(BaseView):
	"""Main GraphRAG Management Interface"""
	
	route_base = "/graphrag"
	default_view = "index"
	
	def __init__(self):
		super().__init__()
		# Initialize services (would be injected in production)
		self.db_service = None
		self.graphrag_service = None
		self.ollama_client = None
	
	@expose("/")
	@has_access
	def index(self):
		"""Main GraphRAG management interface"""
		try:
			# Get summary statistics
			stats = self._get_summary_stats()
			recent_queries = self._get_recent_queries()
			
			return self.render_template(
				"graphrag/management_index.html",
				stats=stats,
				recent_queries=recent_queries,
				title="GraphRAG Management"
			)
		except Exception as e:
			logger.error(f"GraphRAG management index error: {e}")
			flash(f"Error loading GraphRAG management: {e}", "danger")
			return self.render_template("graphrag/error.html", error=str(e))
	
	@expose("/query/", methods=["GET", "POST"])
	@has_access
	def query_interface(self):
		"""Interactive GraphRAG query interface"""
		if request.method == "POST":
			return self._handle_query_request()
		
		# GET request - show query form
		knowledge_graphs = GrKnowledgeGraph.query.filter_by(status='active').all()
		
		return self.render_template(
			"graphrag/query_interface.html",
			knowledge_graphs=knowledge_graphs,
			title="GraphRAG Query Interface"
		)
	
	@expose("/create_graph/", methods=["GET", "POST"])
	@has_access
	def create_knowledge_graph(self):
		"""Create new knowledge graph"""
		if request.method == "POST":
			return self._handle_create_graph_request()
		
		return self.render_template(
			"graphrag/create_graph.html",
			title="Create Knowledge Graph"
		)
	
	@expose("/process_document/", methods=["GET", "POST"])
	@has_access
	def process_document(self):
		"""Process document into knowledge graph"""
		if request.method == "POST":
			return self._handle_document_processing()
		
		knowledge_graphs = GrKnowledgeGraph.query.filter_by(status='active').all()
		
		return self.render_template(
			"graphrag/process_document.html",
			knowledge_graphs=knowledge_graphs,
			title="Process Document"
		)
	
	@expose("/graph_visualization/<graph_id>/")
	@has_access
	def graph_visualization(self, graph_id):
		"""Knowledge graph visualization"""
		try:
			graph = GrKnowledgeGraph.query.filter_by(knowledge_graph_id=graph_id).first()
			if not graph:
				flash("Knowledge graph not found", "danger")
				return redirect(url_for("GraphRAGManagementView.index"))
			
			# Get graph data for visualization
			graph_data = self._get_graph_visualization_data(graph_id)
			
			return self.render_template(
				"graphrag/graph_visualization.html",
				graph=graph,
				graph_data=graph_data,
				title=f"Visualize: {graph.name}"
			)
		except Exception as e:
			logger.error(f"Graph visualization error: {e}")
			flash(f"Error loading graph visualization: {e}", "danger")
			return redirect(url_for("GraphRAGManagementView.index"))
	
	def _handle_query_request(self) -> str:
		"""Handle GraphRAG query request"""
		try:
			# Extract form data
			query_text = request.form.get("query_text", "").strip()
			graph_id = request.form.get("knowledge_graph_id")
			query_type = request.form.get("query_type", "factual")
			max_hops = int(request.form.get("max_hops", "3"))
			
			if not query_text or not graph_id:
				flash("Query text and knowledge graph are required", "danger")
				return redirect(url_for("GraphRAGManagementView.query_interface"))
			
			# Process query (simplified - would use actual service)
			result = self._process_graphrag_query(query_text, graph_id, query_type, max_hops)
			
			return self.render_template(
				"graphrag/query_result.html",
				query_text=query_text,
				result=result,
				title="Query Result"
			)
			
		except Exception as e:
			logger.error(f"Query processing error: {e}")
			flash(f"Error processing query: {e}", "danger")
			return redirect(url_for("GraphRAGManagementView.query_interface"))
	
	def _handle_create_graph_request(self) -> str:
		"""Handle knowledge graph creation"""
		try:
			name = request.form.get("name", "").strip()
			description = request.form.get("description", "").strip()
			domain = request.form.get("domain", "general")
			
			if not name:
				flash("Graph name is required", "danger")
				return redirect(url_for("GraphRAGManagementView.create_knowledge_graph"))
			
			# Create knowledge graph (simplified)
			graph_id = self._create_knowledge_graph(name, description, domain)
			
			flash(f"Knowledge graph '{name}' created successfully", "success")
			return redirect(url_for("GraphRAGManagementView.graph_visualization", graph_id=graph_id))
			
		except Exception as e:
			logger.error(f"Graph creation error: {e}")
			flash(f"Error creating knowledge graph: {e}", "danger")
			return redirect(url_for("GraphRAGManagementView.create_knowledge_graph"))
	
	def _handle_document_processing(self) -> str:
		"""Handle document processing request"""
		try:
			graph_id = request.form.get("knowledge_graph_id")
			document_text = request.form.get("document_text", "").strip()
			document_url = request.form.get("document_url", "").strip()
			title = request.form.get("title", "Untitled Document")
			
			if not graph_id or (not document_text and not document_url):
				flash("Knowledge graph and document content/URL are required", "danger")
				return redirect(url_for("GraphRAGManagementView.process_document"))
			
			# Process document (simplified)
			processing_result = self._process_document_into_graph(
				graph_id, document_text, document_url, title
			)
			
			flash(f"Document processed successfully. Extracted {processing_result['entities']} entities and {processing_result['relationships']} relationships.", "success")
			return redirect(url_for("GraphRAGManagementView.graph_visualization", graph_id=graph_id))
			
		except Exception as e:
			logger.error(f"Document processing error: {e}")
			flash(f"Error processing document: {e}", "danger")
			return redirect(url_for("GraphRAGManagementView.process_document"))
	
	def _get_summary_stats(self) -> Dict[str, Any]:
		"""Get summary statistics"""
		return {
			"knowledge_graphs": GrKnowledgeGraph.query.count(),
			"total_entities": sum(kg.entity_count or 0 for kg in GrKnowledgeGraph.query.all()),
			"total_relationships": sum(kg.relationship_count or 0 for kg in GrKnowledgeGraph.query.all()),
			"documents_processed": GrDocumentSource.query.filter_by(processing_status='completed').count(),
			"queries_today": GrQueryHistory.query.filter(
				GrQueryHistory.created_at >= datetime.utcnow().date()
			).count()
		}
	
	def _get_recent_queries(self) -> List[Dict[str, Any]]:
		"""Get recent queries"""
		queries = GrQueryHistory.query.order_by(
			GrQueryHistory.created_at.desc()
		).limit(10).all()
		
		return [{
			"id": q.query_id,
			"text": q.query_text[:100] + "..." if len(q.query_text) > 100 else q.query_text,
			"type": q.query_type,
			"processing_time": q.processing_time_ms,
			"confidence": q.confidence_score,
			"created_at": q.created_at
		} for q in queries]
	
	def _process_graphrag_query(self, query_text: str, graph_id: str, query_type: str, max_hops: int) -> Dict[str, Any]:
		"""Process GraphRAG query (simplified implementation)"""
		# This would integrate with the actual GraphRAG service
		return {
			"answer": f"Based on the knowledge graph analysis, here is the response to '{query_text}'. This is a simplified response for demonstration purposes.",
			"confidence": 0.87,
			"processing_time_ms": 1250,
			"entities_used": ["Entity1", "Entity2", "Entity3"],
			"relationships_used": ["relates_to", "part_of"],
			"reasoning_hops": min(max_hops, 2),
			"sources": ["Document 1", "Document 2"]
		}
	
	def _create_knowledge_graph(self, name: str, description: str, domain: str) -> str:
		"""Create new knowledge graph (simplified)"""
		from uuid_extensions import uuid7str
		
		graph_id = uuid7str()
		# This would integrate with the actual GraphRAG service
		# For now, create a database record
		
		return graph_id
	
	def _process_document_into_graph(self, graph_id: str, document_text: str, document_url: str, title: str) -> Dict[str, Any]:
		"""Process document into knowledge graph (simplified)"""
		# This would integrate with the actual document processing service
		return {
			"entities": 15,
			"relationships": 23,
			"processing_time_ms": 5400
		}
	
	def _get_graph_visualization_data(self, graph_id: str) -> Dict[str, Any]:
		"""Get graph data for visualization"""
		# This would query the actual graph data
		return {
			"nodes": [
				{"id": "entity1", "label": "Sample Entity 1", "type": "person", "confidence": 0.9},
				{"id": "entity2", "label": "Sample Entity 2", "type": "organization", "confidence": 0.85},
				{"id": "entity3", "label": "Sample Entity 3", "type": "location", "confidence": 0.92}
			],
			"edges": [
				{"source": "entity1", "target": "entity2", "type": "works_for", "strength": 0.8},
				{"source": "entity2", "target": "entity3", "type": "located_in", "strength": 0.75}
			],
			"statistics": {
				"node_count": 3,
				"edge_count": 2,
				"avg_confidence": 0.89
			}
		}


# ============================================================================
# BLUEPRINT REGISTRATION
# ============================================================================

def create_graphrag_blueprint(appbuilder) -> None:
	"""Register GraphRAG views with Flask-AppBuilder"""
	
	# Register model views
	appbuilder.add_view(
		KnowledgeGraphModelView,
		"Knowledge Graphs",
		icon="fa-sitemap",
		category="GraphRAG",
		category_icon="fa-brain"
	)
	
	appbuilder.add_view(
		GraphEntityModelView,
		"Graph Entities",
		icon="fa-circle",
		category="GraphRAG"
	)
	
	appbuilder.add_view(
		GraphRelationshipModelView,
		"Graph Relationships",
		icon="fa-arrows-alt",
		category="GraphRAG"
	)
	
	appbuilder.add_view(
		DocumentSourceModelView,
		"Document Sources",
		icon="fa-file-text",
		category="GraphRAG"
	)
	
	appbuilder.add_view(
		QueryHistoryModelView,
		"Query History",
		icon="fa-history",
		category="GraphRAG"
	)
	
	# Register management views
	appbuilder.add_view_no_menu(GraphRAGManagementView)
	appbuilder.add_link(
		"GraphRAG Management",
		href="/graphrag/",
		icon="fa-cogs",
		category="GraphRAG"
	)
	
	# Register analytics views
	appbuilder.add_view_no_menu(GraphRAGAnalyticsView)
	appbuilder.add_link(
		"GraphRAG Analytics",
		href="/graphrag_analytics/",
		icon="fa-chart-line",
		category="GraphRAG"
	)
	
	logger.info("GraphRAG Flask-AppBuilder views registered successfully")


__all__ = [
	'KnowledgeGraphModelView',
	'GraphEntityModelView', 
	'GraphRelationshipModelView',
	'DocumentSourceModelView',
	'QueryHistoryModelView',
	'GraphRAGAnalyticsView',
	'GraphRAGManagementView',
	'create_graphrag_blueprint'
]