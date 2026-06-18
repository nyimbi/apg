#!/usr/bin/env python3
"""
APG Metadata Management - Advanced Search Engine
Natural language search with semantic understanding and contextual intelligence

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
_dc_field = field
from enum import Enum
from collections import defaultdict, Counter
import numpy as np
from uuid_extensions import uuid7str

from .database import MetaDatabaseManager
from .integrations import APGMetadataIntegrationManager


class SearchMethod(str, Enum):
	"""Search methods available"""
	FULL_TEXT = "full_text"
	SEMANTIC = "semantic"
	FACETED = "faceted"
	NATURAL_LANGUAGE = "natural_language"
	SIMILARITY = "similarity"
	GRAPH_TRAVERSAL = "graph_traversal"
	HYBRID = "hybrid"


class SearchScope(str, Enum):
	"""Search scope options"""
	ASSETS = "assets"
	COLUMNS = "columns"
	LINEAGE = "lineage"
	CLASSIFICATIONS = "classifications"
	COMMENTS = "comments"
	ALL = "all"


class SortOrder(str, Enum):
	"""Sort order options"""
	RELEVANCE = "relevance"
	NAME = "name"
	CREATED_DATE = "created_date"
	MODIFIED_DATE = "modified_date"
	POPULARITY = "popularity"
	QUALITY_SCORE = "quality_score"


@dataclass
class SearchFilter:
	"""Search filter specification"""
	field: str = ""
	operator: str = "equals"  # equals, contains, gt, lt, gte, lte, in, not_in
	value: Any = None
	boost: float = 1.0  # Boost factor for this filter


@dataclass
class FacetDefinition:
	"""Facet definition for faceted search"""
	field: str = ""
	display_name: str = ""
	facet_type: str = "terms"  # terms, range, date_histogram
	size: int = 10
	aggregation_params: Dict[str, Any] = _dc_field(default_factory=dict)


@dataclass
class SearchQuery:
	"""Comprehensive search query specification"""
	query_id: str = _dc_field(default_factory=uuid7str)
	query_text: str = ""
	tenant_id: str = ""
	user_id: str = ""
	
	# Search configuration
	search_method: SearchMethod = SearchMethod.HYBRID
	search_scope: SearchScope = SearchScope.ALL
	
	# Filtering and faceting
	filters: List[SearchFilter] = _dc_field(default_factory=list)
	facets: List[FacetDefinition] = _dc_field(default_factory=list)
	
	# Pagination and sorting
	from_index: int = 0
	size: int = 20
	sort_order: SortOrder = SortOrder.RELEVANCE
	sort_ascending: bool = False
	
	# Advanced options
	enable_highlighting: bool = True
	enable_suggestions: bool = True
	enable_autocomplete: bool = True
	min_score: float = 0.0
	boost_fields: Dict[str, float] = _dc_field(default_factory=dict)
	
	# Context
	search_context: Dict[str, Any] = _dc_field(default_factory=dict)
	
	# Timing
	created_at: datetime = _dc_field(default_factory=datetime.utcnow)


@dataclass
class SearchResult:
	"""Individual search result"""
	result_id: str = _dc_field(default_factory=uuid7str)
	asset_id: str = ""
	asset_type: str = ""
	name: str = ""
	display_name: str = ""
	description: str = ""
	source_system: str = ""
	
	# Scoring
	relevance_score: float = 0.0
	quality_score: Optional[float] = None
	popularity_score: float = 0.0
	
	# Metadata
	tags: List[str] = _dc_field(default_factory=list)
	classifications: List[str] = _dc_field(default_factory=list)
	owner: Optional[str] = None
	created_at: Optional[datetime] = None
	modified_at: Optional[datetime] = None
	
	# Search-specific
	highlighted_fields: Dict[str, List[str]] = _dc_field(default_factory=dict)
	match_reasons: List[str] = _dc_field(default_factory=list)
	column_matches: List[Dict[str, Any]] = _dc_field(default_factory=list)
	
	# Context
	business_context: Optional[str] = None
	technical_context: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for serialization"""
		return {
			"result_id": self.result_id,
			"asset_id": self.asset_id,
			"asset_type": self.asset_type,
			"name": self.name,
			"display_name": self.display_name,
			"description": self.description,
			"source_system": self.source_system,
			"relevance_score": self.relevance_score,
			"quality_score": self.quality_score,
			"popularity_score": self.popularity_score,
			"tags": self.tags,
			"classifications": self.classifications,
			"owner": self.owner,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"modified_at": self.modified_at.isoformat() if self.modified_at else None,
			"highlighted_fields": self.highlighted_fields,
			"match_reasons": self.match_reasons,
			"column_matches": self.column_matches,
			"business_context": self.business_context,
			"technical_context": self.technical_context
		}


@dataclass
class SearchFacet:
	"""Search facet result"""
	field: str = ""
	display_name: str = ""
	buckets: List[Dict[str, Any]] = _dc_field(default_factory=list)
	total_count: int = 0


@dataclass
class SearchResponse:
	"""Complete search response"""
	response_id: str = _dc_field(default_factory=uuid7str)
	query: SearchQuery = _dc_field(default_factory=SearchQuery)
	
	# Results
	results: List[SearchResult] = _dc_field(default_factory=list)
	total_results: int = 0
	max_score: float = 0.0
	
	# Facets
	facets: List[SearchFacet] = _dc_field(default_factory=list)
	
	# Suggestions and corrections
	suggested_queries: List[str] = _dc_field(default_factory=list)
	query_corrections: List[str] = _dc_field(default_factory=list)
	auto_complete_suggestions: List[str] = _dc_field(default_factory=list)
	
	# Performance metrics
	search_time_ms: float = 0.0
	from_cache: bool = False
	
	# Analytics
	search_timestamp: datetime = _dc_field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for API response"""
		return {
			"response_id": self.response_id,
			"query": {
				"query_text": self.query.query_text,
				"search_method": self.query.search_method.value,
				"search_scope": self.query.search_scope.value
			},
			"results": [result.to_dict() for result in self.results],
			"total_results": self.total_results,
			"max_score": self.max_score,
			"facets": [
				{
					"field": facet.field,
					"display_name": facet.display_name,
					"buckets": facet.buckets,
					"total_count": facet.total_count
				}
				for facet in self.facets
			],
			"suggested_queries": self.suggested_queries,
			"query_corrections": self.query_corrections,
			"auto_complete_suggestions": self.auto_complete_suggestions,
			"search_time_ms": self.search_time_ms,
			"from_cache": self.from_cache,
			"pagination": {
				"from": self.query.from_index,
				"size": self.query.size,
				"total": self.total_results,
				"has_more": (self.query.from_index + self.query.size) < self.total_results
			}
		}


class NaturalLanguageProcessor:
	"""Natural language query processor for metadata search"""
	
	def __init__(self, integration_manager: APGMetadataIntegrationManager):
		self.integration_manager = integration_manager
		
		# Common patterns for metadata queries
		self.query_patterns = {
			"find_tables_with": r"(?:find|show|get|list)\s+(?:tables?|datasets?)\s+(?:with|containing|having)\s+(.+)",
			"find_columns_like": r"(?:find|show|get|list)\s+(?:columns?|fields?)\s+(?:like|named|called|containing)\s+(.+)",
			"show_lineage": r"(?:show|find|get)\s+(?:lineage|dependencies?|flow)\s+(?:for|of)\s+(.+)",
			"find_by_owner": r"(?:find|show|get|list)\s+(?:assets?|data)\s+(?:owned?|created?)\s+by\s+(.+)",
			"find_sensitive": r"(?:find|show|get|list)\s+(?:sensitive|pii|personal|confidential)\s+(?:data|assets?|tables?)",
			"find_popular": r"(?:find|show|get|list)\s+(?:popular|frequently used|most used)\s+(?:data|assets?|tables?)",
			"find_recent": r"(?:find|show|get|list)\s+(?:recent|new|latest)\s+(?:data|assets?|tables?)",
			"data_quality": r"(?:find|show|get|list)\s+(?:low|poor|high|good)\s+(?:quality|data quality)\s+(?:data|assets?|tables?)"
		}
		
		# Entity extraction patterns
		self.entity_patterns = {
			"table_names": r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b",
			"column_names": r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b",
			"classification_terms": r"\b(pii|phi|personal|sensitive|confidential|public|internal)\b",
			"quality_terms": r"\b(high|low|good|bad|poor|excellent)\s+quality\b",
			"date_ranges": r"\b(today|yesterday|this week|this month|last week|last month|\d+\s+days?\s+ago)\b"
		}
	
	async def parse_natural_language_query(self, query_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Parse natural language query into structured search parameters"""
		parsed = {
			"intent": "general_search",
			"entities": {},
			"filters": [],
			"search_scope": SearchScope.ALL,
			"search_terms": [],
			"structured_query": query_text.lower()
		}
		
		query_lower = query_text.lower()
		
		# Detect query intent
		for intent, pattern in self.query_patterns.items():
			match = re.search(pattern, query_lower)
			if match:
				parsed["intent"] = intent
				parsed["entities"]["main_term"] = match.group(1).strip()
				break
		
		# Extract entities
		for entity_type, pattern in self.entity_patterns.items():
			matches = re.findall(pattern, query_lower, re.IGNORECASE)
			if matches:
				parsed["entities"][entity_type] = matches
		
		# Convert intent to filters and search configuration
		await self._apply_intent_to_search(parsed, context)
		
		# Use AI for complex query understanding if available
		if self.integration_manager and self.integration_manager.ai_integration:
			try:
				ai_parsed = await self._ai_parse_query(query_text, context)
				if ai_parsed:
					# Merge AI insights with rule-based parsing
					parsed.update(ai_parsed)
			except Exception:
				pass  # Fallback to rule-based parsing
		
		return parsed
	
	async def _apply_intent_to_search(self, parsed: Dict[str, Any], context: Dict[str, Any]):
		"""Apply detected intent to search configuration"""
		intent = parsed["intent"]
		
		if intent == "find_tables_with":
			parsed["search_scope"] = SearchScope.ASSETS
			parsed["filters"].append({
				"field": "asset_type",
				"operator": "in",
				"value": ["table", "view", "collection"]
			})
		
		elif intent == "find_columns_like":
			parsed["search_scope"] = SearchScope.COLUMNS
			if "main_term" in parsed["entities"]:
				parsed["search_terms"] = [parsed["entities"]["main_term"]]
		
		elif intent == "show_lineage":
			parsed["search_scope"] = SearchScope.LINEAGE
			if "main_term" in parsed["entities"]:
				parsed["search_terms"] = [parsed["entities"]["main_term"]]
		
		elif intent == "find_by_owner":
			if "main_term" in parsed["entities"]:
				parsed["filters"].append({
					"field": "owner",
					"operator": "contains",
					"value": parsed["entities"]["main_term"]
				})
		
		elif intent == "find_sensitive":
			parsed["filters"].append({
				"field": "classifications",
				"operator": "in",
				"value": ["PII", "PHI", "CONFIDENTIAL", "SENSITIVE"]
			})
		
		elif intent == "find_popular":
			parsed["filters"].append({
				"field": "popularity_score",
				"operator": "gte",
				"value": 0.7
			})
		
		elif intent == "find_recent":
			recent_date = datetime.utcnow() - timedelta(days=30)
			parsed["filters"].append({
				"field": "created_at",
				"operator": "gte",
				"value": recent_date.isoformat()
			})
		
		elif intent == "data_quality":
			quality_terms = parsed["entities"].get("quality_terms", [])
			if any("high" in term or "good" in term or "excellent" in term for term in quality_terms):
				parsed["filters"].append({
					"field": "quality_score",
					"operator": "gte",
					"value": 0.8
				})
			elif any("low" in term or "poor" in term or "bad" in term for term in quality_terms):
				parsed["filters"].append({
					"field": "quality_score",
					"operator": "lte",
					"value": 0.5
				})
	
	async def _ai_parse_query(self, query_text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		"""Use AI to parse complex natural language queries"""
		try:
			prompt = f"""
			Parse this metadata search query and extract structured information:
			Query: "{query_text}"
			
			Extract:
			1. Search intent (find_tables, find_columns, show_lineage, find_sensitive_data, etc.)
			2. Entity mentions (table names, column names, system names)
			3. Filter conditions (date ranges, quality criteria, classifications)
			4. Search scope (assets, columns, lineage, all)
			
			Respond in JSON format:
			{{
				"intent": "search_intent",
				"entities": {{"entity_type": ["entity1", "entity2"]}},
				"filters": [{{"field": "field_name", "operator": "equals", "value": "value"}}],
				"search_terms": ["term1", "term2"],
				"confidence": 0.8
			}}
			"""
			
			# Use AI integration to parse query
			ai_result = await self.integration_manager.ai_integration.classify_data_content(
				content=prompt,
				column_name="natural_language_query"
			)
			
			if ai_result and isinstance(ai_result, dict):
				try:
					# Try to parse as JSON
					return json.loads(ai_result.get("response", "{}"))
				except json.JSONDecodeError:
					return None
			
		except Exception:
			pass
		
		return None


class MetadataSearchEngine:
	"""Advanced metadata search engine with natural language capabilities"""
	
	def __init__(self,
		     db_manager: MetaDatabaseManager,
		     integration_manager: APGMetadataIntegrationManager,
		     config: Dict[str, Any] = None):
		self.db_manager = db_manager
		self.integration_manager = integration_manager
		self.config = config or {}
		
		# Natural language processor
		self.nl_processor = NaturalLanguageProcessor(integration_manager)
		
		# Search configuration
		self.enable_caching = config.get('enable_caching', True)
		self.cache_ttl = config.get('cache_ttl_seconds', 300)  # 5 minutes
		self.max_results = config.get('max_results', 1000)
		self.default_page_size = config.get('default_page_size', 20)
		
		# Scoring weights
		self.scoring_weights = {
			'text_match': 0.4,
			'quality_score': 0.2,
			'popularity': 0.2,
			'recency': 0.1,
			'ownership': 0.1
		}
		
		# Analytics tracking
		self.search_analytics: Dict[str, Any] = defaultdict(int)
		self.popular_queries: Counter = Counter()
		
		# Index for fast searching (in-memory for now)
		self.search_index: Dict[str, Any] = {}
		
		self.initialized = False
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize the search engine"""
		if self.initialized:
			return {"status": "already_initialized"}
		
		try:
			# Build search index
			await self._build_search_index()
			
			# Load search analytics
			await self._load_search_analytics()
			
			self.initialized = True
			
			await self._log_info("Metadata Search Engine initialized successfully")
			
			return {
				"status": "initialized",
				"indexed_assets": len(self.search_index.get("assets", {})),
				"caching_enabled": self.enable_caching,
				"max_results": self.max_results,
				"natural_language_enabled": True
			}
			
		except Exception as e:
			await self._log_error(f"Search Engine initialization failed: {str(e)}")
			raise
	
	async def search(self, query: SearchQuery) -> SearchResponse:
		"""Execute comprehensive metadata search"""
		start_time = asyncio.get_event_loop().time()
		
		try:
			response = SearchResponse(query=query)
			
			# Check cache first
			cache_key = self._generate_cache_key(query)
			if self.enable_caching:
				cached_response = await self._get_cached_response(cache_key)
				if cached_response:
					cached_response.from_cache = True
					self.search_analytics['cache_hits'] += 1
					return cached_response
			
			self.search_analytics['total_searches'] += 1
			
			# Parse natural language query if needed
			if query.search_method in [SearchMethod.NATURAL_LANGUAGE, SearchMethod.HYBRID]:
				nl_parsed = await self.nl_processor.parse_natural_language_query(
					query.query_text, 
					{"tenant_id": query.tenant_id, "user_id": query.user_id}
				)
				query = await self._apply_nl_parsing_to_query(query, nl_parsed)
			
			# Execute search based on method
			if query.search_method == SearchMethod.FULL_TEXT:
				results = await self._full_text_search(query)
			elif query.search_method == SearchMethod.SEMANTIC:
				results = await self._semantic_search(query)
			elif query.search_method == SearchMethod.FACETED:
				results = await self._faceted_search(query)
			elif query.search_method == SearchMethod.SIMILARITY:
				results = await self._similarity_search(query)
			elif query.search_method == SearchMethod.GRAPH_TRAVERSAL:
				results = await self._graph_traversal_search(query)
			else:  # HYBRID or default
				results = await self._hybrid_search(query)
			
			response.results = results
			response.total_results = len(results)
			response.max_score = max([r.relevance_score for r in results], default=0.0)
			
			# Apply pagination
			start_idx = query.from_index
			end_idx = start_idx + query.size
			response.results = response.results[start_idx:end_idx]
			
			# Generate facets if requested
			if query.facets:
				response.facets = await self._generate_facets(query, results)
			
			# Generate suggestions
			if query.enable_suggestions:
				response.suggested_queries = await self._generate_suggestions(query)
			
			# Generate autocomplete
			if query.enable_autocomplete:
				response.auto_complete_suggestions = await self._generate_autocomplete(query)
			
			# Calculate search time
			response.search_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
			
			# Cache response
			if self.enable_caching:
				await self._cache_response(cache_key, response)
			
			# Track analytics
			await self._track_search_analytics(query, response)
			
			await self._log_info(
				f"Search completed: '{query.query_text}' -> {response.total_results} results "
				f"in {response.search_time_ms:.2f}ms"
			)
			
			return response
			
		except Exception as e:
			await self._log_error(f"Search failed: {str(e)}")
			
			# Return empty response with error
			response = SearchResponse(query=query)
			response.search_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
			return response
	
	async def _full_text_search(self, query: SearchQuery) -> List[SearchResult]:
		"""Execute full-text search using PostgreSQL"""
		try:
			async with self.db_manager.get_session(query.tenant_id) as session:
				from sqlalchemy import select, text, func, or_, and_
				from .models import MetaAsset
				
				# Build base query
				stmt = select(MetaAsset).where(
					and_(
						MetaAsset.tenant_id == query.tenant_id,
						MetaAsset.is_deleted == False,
						MetaAsset.status == "active"
					)
				)
				
				# Add full-text search condition
				if query.query_text:
					# Use PostgreSQL full-text search
					search_vector = func.to_tsvector('english', 
						func.coalesce(MetaAsset.name, '') + ' ' +
						func.coalesce(MetaAsset.display_name, '') + ' ' +
						func.coalesce(MetaAsset.description, '')
					)
					search_query = func.plainto_tsquery('english', query.query_text)
					
					stmt = stmt.where(search_vector.op('@@')(search_query))
				
				# Apply filters
				for filter_def in query.filters:
					stmt = self._apply_filter_to_query(stmt, filter_def, MetaAsset)
				
				# Apply sorting
				if query.sort_order == SortOrder.RELEVANCE and query.query_text:
					# Rank by relevance
					rank = func.ts_rank(search_vector, search_query)
					stmt = stmt.order_by(rank.desc())
				else:
					stmt = self._apply_sorting_to_query(stmt, query, MetaAsset)
				
				# Limit results
				stmt = stmt.limit(self.max_results)
				
				# Execute query
				result = await session.execute(stmt)
				assets = result.scalars().all()
				
				# Convert to search results
				search_results = []
				for asset in assets:
					search_result = await self._convert_asset_to_search_result(asset, query)
					search_results.append(search_result)
				
				return search_results
				
		except Exception as e:
			await self._log_error(f"Full-text search failed: {str(e)}")
			return []
	
	async def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
		"""Execute semantic search using embeddings"""
		try:
			# Generate embedding for the query
			query_embedding = await self._generate_query_embedding(query.query_text)
			if not query_embedding:
				return await self._full_text_search(query)
			
			# Search for similar assets using vector similarity
			async with self.db_manager.get_session(query.tenant_id) as session:
				from sqlalchemy import select, text, func
				from .models import MetaAsset
				
				# Query for assets with embeddings and calculate similarity
				stmt = select(
					MetaAsset,
					func.coalesce(
						text(f"1 - (custom_attributes->>'embedding' <-> '{json.dumps(query_embedding)}')"),
						0.0
					).label('similarity_score')
				).where(
					MetaAsset.tenant_id == query.tenant_id,
					MetaAsset.is_deleted == False,
					text("custom_attributes ? 'embedding'")
				).order_by(
					text('similarity_score DESC')
				).limit(query.limit)
				
				result = await session.execute(stmt)
				rows = result.fetchall()
				
				search_results = []
				for row in rows:
					asset = row[0]
					similarity = float(row[1]) if row[1] else 0.0
					
					# Only include results with reasonable similarity
					if similarity >= 0.3:
						search_result = SearchResult(
							asset_id=asset.id,
							asset_name=asset.name,
							display_name=asset.display_name,
							description=asset.description,
							asset_type=asset.asset_type,
							source_system=asset.source_system,
							match_score=similarity * 0.9,  # Weight for semantic search
							match_type="semantic",
							match_details=f"Semantic similarity: {similarity:.2f}",
							tags=asset.tags or [],
							quality_score=asset.quality_score,
							tenant_id=asset.tenant_id
						)
						search_results.append(search_result)
				
				await self._log_info(f"Semantic search found {len(search_results)} results")
				return search_results
				
		except Exception as e:
			await self._log_error(f"Semantic search failed: {str(e)}")
			# Fall back to full-text search
			return await self._full_text_search(query)
	
	async def _faceted_search(self, query: SearchQuery) -> List[SearchResult]:
		"""Execute faceted search with aggregations"""
		# Start with full-text search results
		results = await self._full_text_search(query)
		
		# Faceted search would typically be implemented with Elasticsearch
		# For now, return basic results
		return results
	
	async def _similarity_search(self, query: SearchQuery) -> List[SearchResult]:
		"""Execute similarity-based search"""
		try:
			# Extract asset ID if query is asking for similar assets
			import re
			asset_id_match = re.search(r'similar to (\w+)', query.query_text.lower())
			if not asset_id_match:
				return await self._full_text_search(query)
			
			reference_asset_id = asset_id_match.group(1)
			
			async with self.db_manager.get_session(query.tenant_id) as session:
				from sqlalchemy import select, and_
				from .models import MetaAsset
				
				# Get the reference asset
				ref_stmt = select(MetaAsset).where(
					and_(
						MetaAsset.id == reference_asset_id,
						MetaAsset.tenant_id == query.tenant_id,
						MetaAsset.is_deleted == False
					)
				)
				ref_result = await session.execute(ref_stmt)
				reference_asset = ref_result.scalar_one_or_none()
				
				if not reference_asset:
					return []
				
				# Find similar assets based on multiple criteria
				similarity_stmt = select(MetaAsset).where(
					and_(
						MetaAsset.tenant_id == query.tenant_id,
						MetaAsset.is_deleted == False,
						MetaAsset.id != reference_asset_id,
						# Similar asset type
						MetaAsset.asset_type == reference_asset.asset_type
					)
				)
				
				# Add additional similarity filters
				if reference_asset.source_system:
					similarity_stmt = similarity_stmt.where(
						MetaAsset.source_system == reference_asset.source_system
					)
				
				similarity_stmt = similarity_stmt.limit(query.limit)
				
				result = await session.execute(similarity_stmt)
				similar_assets = result.scalars().all()
				
				search_results = []
				for asset in similar_assets:
					# Calculate similarity score based on multiple factors
					similarity_score = self._calculate_asset_similarity(reference_asset, asset)
					
					if similarity_score >= 0.4:  # Minimum similarity threshold
						search_result = SearchResult(
							asset_id=asset.id,
							asset_name=asset.name,
							display_name=asset.display_name,
							description=asset.description,
							asset_type=asset.asset_type,
							source_system=asset.source_system,
							match_score=similarity_score,
							match_type="similarity",
							match_details=f"Similar to {reference_asset.name} (score: {similarity_score:.2f})",
							tags=asset.tags or [],
							quality_score=asset.quality_score,
							tenant_id=asset.tenant_id
						)
						search_results.append(search_result)
				
				# Sort by similarity score
				search_results.sort(key=lambda x: x.match_score, reverse=True)
				
				await self._log_info(f"Similarity search found {len(search_results)} results")
				return search_results
				
		except Exception as e:
			await self._log_error(f"Similarity search failed: {str(e)}")
			return await self._full_text_search(query)
	
	async def _graph_traversal_search(self, query: SearchQuery) -> List[SearchResult]:
		"""Execute graph traversal search through lineage"""
		try:
			# Extract traversal patterns from query
			import re
			
			# Look for lineage-related queries
			lineage_patterns = [
				r'upstream (?:of |from )(\w+)',
				r'downstream (?:of |from )(\w+)',
				r'connected to (\w+)',
				r'related to (\w+)'
			]
			
			root_asset_id = None
			direction = "both"
			
			for pattern in lineage_patterns:
				match = re.search(pattern, query.query_text.lower())
				if match:
					root_asset_id = match.group(1)
					if 'upstream' in pattern:
						direction = "upstream"
					elif 'downstream' in pattern:
						direction = "downstream"
					break
			
			if not root_asset_id:
				return await self._full_text_search(query)
			
			# Use the lineage engine for graph traversal
			if self.integration_manager:
				# Get lineage paths
				lineage_paths = await self._get_lineage_paths(
					root_asset_id, query.tenant_id, direction, max_depth=3
				)
				
				# Extract asset IDs from lineage paths
				connected_asset_ids = set()
				for path in lineage_paths:
					for step in path.get('steps', []):
						if step.get('asset_id') != root_asset_id:
							connected_asset_ids.add(step['asset_id'])
				
				if not connected_asset_ids:
					return []
				
				# Get full asset details for connected assets
				async with self.db_manager.get_session(query.tenant_id) as session:
					from sqlalchemy import select
					from .models import MetaAsset
					
					stmt = select(MetaAsset).where(
						MetaAsset.id.in_(connected_asset_ids),
						MetaAsset.tenant_id == query.tenant_id,
						MetaAsset.is_deleted == False
					).limit(query.limit)
					
					result = await session.execute(stmt)
					connected_assets = result.scalars().all()
					
					search_results = []
					for asset in connected_assets:
						# Calculate relevance based on distance from root
						relevance_score = self._calculate_lineage_relevance(
							root_asset_id, asset.id, lineage_paths
						)
						
						search_result = SearchResult(
							asset_id=asset.id,
							asset_name=asset.name,
							display_name=asset.display_name,
							description=asset.description,
							asset_type=asset.asset_type,
							source_system=asset.source_system,
							match_score=relevance_score,
							match_type="lineage",
							match_details=f"Connected via lineage (distance: {int((1-relevance_score) * 10)})",
							tags=asset.tags or [],
							quality_score=asset.quality_score,
							tenant_id=asset.tenant_id
						)
						search_results.append(search_result)
					
					# Sort by relevance score
					search_results.sort(key=lambda x: x.match_score, reverse=True)
					
					await self._log_info(f"Graph traversal found {len(search_results)} connected assets")
					return search_results
			
			# Fall back to full-text search if lineage engine not available
			return await self._full_text_search(query)
			
		except Exception as e:
			await self._log_error(f"Graph traversal search failed: {str(e)}")
			return await self._full_text_search(query)
	
	async def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
		"""Execute hybrid search combining multiple methods"""
		try:
			# Combine results from multiple search methods
			all_results = []
			
			# Full-text search (40% weight)
			full_text_results = await self._full_text_search(query)
			for result in full_text_results:
				result.relevance_score *= 0.4
				result.match_reasons.append("full_text_match")
			all_results.extend(full_text_results)
			
			# Semantic search (30% weight) - placeholder
			# semantic_results = await self._semantic_search(query)
			# for result in semantic_results:
			#     result.relevance_score *= 0.3
			#     result.match_reasons.append("semantic_match")
			# all_results.extend(semantic_results)
			
			# Popularity boost (30% weight)
			for result in all_results:
				if result.popularity_score:
					result.relevance_score += result.popularity_score * 0.3
					if result.popularity_score > 0.7:
						result.match_reasons.append("popular_asset")
			
			# Quality score boost
			for result in all_results:
				if result.quality_score:
					result.relevance_score += (result.quality_score / 100.0) * 0.2
					if result.quality_score > 80:
						result.match_reasons.append("high_quality")
			
			# Remove duplicates and sort by relevance
			unique_results = {}
			for result in all_results:
				if result.asset_id not in unique_results:
					unique_results[result.asset_id] = result
				else:
					# Merge scores for duplicates
					existing = unique_results[result.asset_id]
					existing.relevance_score = max(existing.relevance_score, result.relevance_score)
					existing.match_reasons.extend(result.match_reasons)
			
			final_results = list(unique_results.values())
			final_results.sort(key=lambda r: r.relevance_score, reverse=True)
			
			return final_results
			
		except Exception as e:
			await self._log_error(f"Hybrid search failed: {str(e)}")
			return await self._full_text_search(query)
	
	async def _convert_asset_to_search_result(self, asset, query: SearchQuery) -> SearchResult:
		"""Convert MetaAsset to SearchResult"""
		result = SearchResult(
			asset_id=asset.id,
			asset_type=asset.asset_type,
			name=asset.name,
			display_name=asset.display_name or asset.name,
			description=asset.description or "",
			source_system=asset.source_system,
			quality_score=asset.quality_score,
			tags=asset.tags or [],
			owner=asset.owner,
			created_at=asset.created_at,
			modified_at=asset.updated_at
		)
		
		# Calculate relevance score
		result.relevance_score = await self._calculate_relevance_score(asset, query)
		
		# Calculate popularity score based on usage metrics
		result.popularity_score = await self._calculate_popularity_score(asset)
		
		# Get classifications
		try:
			async with self.db_manager.get_session(query.tenant_id) as session:
				from sqlalchemy import select
				from .models import MetaClassification
				
				classification_stmt = select(MetaClassification).where(
					MetaClassification.asset_id == asset.id
				)
				classification_result = await session.execute(classification_stmt)
				classifications = classification_result.scalars().all()
				
				result.classifications = [c.classification_type for c in classifications]
		except Exception:
			pass
		
		# Add highlighting if enabled
		if query.enable_highlighting:
			result.highlighted_fields = await self._generate_highlights(asset, query)
		
		# Add business context if available
		if asset.business_domain:
			result.business_context = f"Business domain: {asset.business_domain}"
		
		return result
	
	async def _calculate_relevance_score(self, asset, query: SearchQuery) -> float:
		"""Calculate relevance score for asset"""
		score = 0.0
		
		query_terms = query.query_text.lower().split() if query.query_text else []
		
		# Name match boost
		asset_name = asset.name.lower()
		for term in query_terms:
			if term in asset_name:
				score += 0.3
		
		# Description match boost
		if asset.description:
			asset_desc = asset.description.lower()
			for term in query_terms:
				if term in asset_desc:
					score += 0.2
		
		# Exact name match gets highest score
		if query.query_text and query.query_text.lower() == asset_name:
			score += 0.5
		
		# Quality score contribution
		if asset.quality_score:
			score += (asset.quality_score / 100.0) * 0.1
		
		return min(score, 1.0)
	
	async def _calculate_popularity_score(self, asset) -> float:
		"""Calculate popularity score based on usage metrics and activity"""
		try:
			score = 0.0
			
			# Base score from quality
			if hasattr(asset, 'quality_score') and asset.quality_score:
				score += (asset.quality_score / 100.0) * 0.3
			
			# Usage activity score (simulated from update frequency)
			if hasattr(asset, 'updated_at') and asset.updated_at:
				days_since_update = (datetime.utcnow() - asset.updated_at).days
				if days_since_update < 7:
					score += 0.4  # Recently updated
				elif days_since_update < 30:
					score += 0.2  # Updated this month
				else:
					score += 0.1  # Older content
			
			# Asset type popularity weights
			type_weights = {
				'table': 0.3,
				'view': 0.2, 
				'dashboard': 0.25,
				'report': 0.2,
				'dataset': 0.25,
				'model': 0.3,
				'pipeline': 0.2
			}
			
			asset_type = getattr(asset, 'asset_type', 'unknown').lower()
			score += type_weights.get(asset_type, 0.1)
			
			return min(score, 1.0)
			
		except Exception:
			return 0.5  # Default neutral score
	
	async def _generate_highlights(self, asset, query: SearchQuery) -> Dict[str, List[str]]:
		"""Generate highlighted text snippets"""
		highlights = {}
		
		if not query.query_text:
			return highlights
		
		query_terms = query.query_text.lower().split()
		
		# Highlight name
		name_highlights = self._highlight_text(asset.name, query_terms)
		if name_highlights:
			highlights["name"] = name_highlights
		
		# Highlight description
		if asset.description:
			desc_highlights = self._highlight_text(asset.description, query_terms)
			if desc_highlights:
				highlights["description"] = desc_highlights
		
		return highlights
	
	def _highlight_text(self, text: str, terms: List[str]) -> List[str]:
		"""Highlight matching terms in text"""
		if not text or not terms:
			return []
		
		highlighted = text
		for term in terms:
			if term in text.lower():
				# Simple highlighting - in production use more sophisticated highlighting
				pattern = re.compile(re.escape(term), re.IGNORECASE)
				highlighted = pattern.sub(f"<mark>{term}</mark>", highlighted)
		
		return [highlighted] if "<mark>" in highlighted else []
	
	async def _apply_nl_parsing_to_query(self, query: SearchQuery, nl_parsed: Dict[str, Any]) -> SearchQuery:
		"""Apply natural language parsing results to search query"""
		# Update search scope
		if "search_scope" in nl_parsed:
			query.search_scope = nl_parsed["search_scope"]
		
		# Add extracted filters
		if "filters" in nl_parsed:
			for filter_data in nl_parsed["filters"]:
				search_filter = SearchFilter(
					field=filter_data.get("field", ""),
					operator=filter_data.get("operator", "equals"),
					value=filter_data.get("value")
				)
				query.filters.append(search_filter)
		
		# Update search terms
		if "search_terms" in nl_parsed:
			additional_terms = " ".join(nl_parsed["search_terms"])
			if additional_terms:
				query.query_text = f"{query.query_text} {additional_terms}".strip()
		
		return query
	
	def _apply_filter_to_query(self, stmt, filter_def: SearchFilter, model):
		"""Apply filter to SQLAlchemy query"""
		field = getattr(model, filter_def.field, None)
		if not field:
			return stmt
		
		if filter_def.operator == "equals":
			return stmt.where(field == filter_def.value)
		elif filter_def.operator == "contains":
			return stmt.where(field.ilike(f"%{filter_def.value}%"))
		elif filter_def.operator == "gt":
			return stmt.where(field > filter_def.value)
		elif filter_def.operator == "lt":
			return stmt.where(field < filter_def.value)
		elif filter_def.operator == "gte":
			return stmt.where(field >= filter_def.value)
		elif filter_def.operator == "lte":
			return stmt.where(field <= filter_def.value)
		elif filter_def.operator == "in":
			return stmt.where(field.in_(filter_def.value))
		elif filter_def.operator == "not_in":
			return stmt.where(~field.in_(filter_def.value))
		
		return stmt
	
	def _apply_sorting_to_query(self, stmt, query: SearchQuery, model):
		"""Apply sorting to SQLAlchemy query"""
		sort_field = None
		
		if query.sort_order == SortOrder.NAME:
			sort_field = model.name
		elif query.sort_order == SortOrder.CREATED_DATE:
			sort_field = model.created_at
		elif query.sort_order == SortOrder.MODIFIED_DATE:
			sort_field = model.updated_at
		elif query.sort_order == SortOrder.QUALITY_SCORE:
			sort_field = model.quality_score
		
		if sort_field is not None:
			if query.sort_ascending:
				return stmt.order_by(sort_field.asc())
			else:
				return stmt.order_by(sort_field.desc())
		
		return stmt
	
	async def _generate_facets(self, query: SearchQuery, results: List[SearchResult]) -> List[SearchFacet]:
		"""Generate facets for search results"""
		facets = []
		
		for facet_def in query.facets:
			if facet_def.field == "asset_type":
				# Asset type facet
				type_counts = Counter(r.asset_type for r in results)
				buckets = [
					{"key": asset_type, "doc_count": count}
					for asset_type, count in type_counts.most_common(facet_def.size)
				]
				
				facets.append(SearchFacet(
					field="asset_type",
					display_name="Asset Type",
					buckets=buckets,
					total_count=sum(type_counts.values())
				))
			
			elif facet_def.field == "source_system":
				# Source system facet
				system_counts = Counter(r.source_system for r in results)
				buckets = [
					{"key": system, "doc_count": count}
					for system, count in system_counts.most_common(facet_def.size)
				]
				
				facets.append(SearchFacet(
					field="source_system",
					display_name="Source System",
					buckets=buckets,
					total_count=sum(system_counts.values())
				))
		
		return facets
	
	async def _generate_suggestions(self, query: SearchQuery) -> List[str]:
		"""Generate query suggestions"""
		suggestions = []
		
		# Popular query suggestions based on analytics
		popular_queries = self.popular_queries.most_common(5)
		for query_text, count in popular_queries:
			if query.query_text.lower() in query_text.lower():
				suggestions.append(query_text)
		
		# Query expansion suggestions
		if query.query_text:
			base_terms = query.query_text.split()
			
			# Add common metadata terms
			metadata_terms = ["data", "table", "database", "schema", "column", "field"]
			for term in metadata_terms:
				if term not in query.query_text.lower():
					suggestions.append(f"{query.query_text} {term}")
		
		return suggestions[:5]
	
	async def _generate_autocomplete(self, query: SearchQuery) -> List[str]:
		"""Generate autocomplete suggestions"""
		if not query.query_text or len(query.query_text) < 2:
			return []
		
		suggestions = []
		
		# Get asset names that start with query text
		try:
			async with self.db_manager.get_session(query.tenant_id) as session:
				from sqlalchemy import select
				from .models import MetaAsset
				
				stmt = select(MetaAsset.name).where(
					MetaAsset.tenant_id == query.tenant_id,
					MetaAsset.name.ilike(f"{query.query_text}%"),
					MetaAsset.is_deleted == False
				).limit(10)
				
				result = await session.execute(stmt)
				asset_names = result.scalars().all()
				suggestions.extend(asset_names)
		except Exception:
			pass
		
		return suggestions[:10]
	
	async def _build_search_index(self):
		"""Build in-memory search index for fast searching"""
		try:
			async with self.db_manager.get_session() as session:
				from sqlalchemy import select
				from .models import MetaAsset
				
				stmt = select(MetaAsset).where(
					MetaAsset.is_deleted == False,
					MetaAsset.status == "active"
				).limit(10000)  # Limit for initial implementation
				
				result = await session.execute(stmt)
				assets = result.scalars().all()
				
				# Build simple search index
				self.search_index["assets"] = {}
				for asset in assets:
					self.search_index["assets"][asset.id] = {
						"name": asset.name,
						"display_name": asset.display_name,
						"description": asset.description,
						"asset_type": asset.asset_type,
						"source_system": asset.source_system,
						"tags": asset.tags or [],
						"search_text": self._build_search_text(asset)
					}
			
		except Exception as e:
			await self._log_error(f"Failed to build search index: {str(e)}")
	
	def _build_search_text(self, asset) -> str:
		"""Build searchable text for asset"""
		text_parts = []
		
		if asset.name:
			text_parts.append(asset.name)
		if asset.display_name and asset.display_name != asset.name:
			text_parts.append(asset.display_name)
		if asset.description:
			text_parts.append(asset.description)
		if asset.tags:
			text_parts.extend(asset.tags)
		
		return " ".join(text_parts).lower()
	
	def _generate_cache_key(self, query: SearchQuery) -> str:
		"""Generate cache key for query"""
		import hashlib
		
		key_data = {
			"query_text": query.query_text,
			"tenant_id": query.tenant_id,
			"search_method": query.search_method.value,
			"search_scope": query.search_scope.value,
			"filters": [(f.field, f.operator, str(f.value)) for f in query.filters],
			"from_index": query.from_index,
			"size": query.size,
			"sort_order": query.sort_order.value
		}
		
		key_str = json.dumps(key_data, sort_keys=True)
		return f"search:{hashlib.sha256(key_str.encode()).hexdigest()[:16]}"
	
	async def _get_cached_response(self, cache_key: str) -> Optional[SearchResponse]:
		"""Get cached search response"""
		try:
			cached_data = await self.db_manager.cache_get(cache_key)
			if cached_data:
				# Deserialize cached response
				cached_dict = json.loads(cached_data)
				return SearchResponse(**cached_dict)
		except Exception:
			pass
		return None
	
	async def _cache_response(self, cache_key: str, response: SearchResponse):
		"""Cache search response"""
		try:
			# Serialize and cache response
			cached_data = json.dumps(response.dict() if hasattr(response, 'dict') else response.__dict__)
			await self.db_manager.cache_set(cache_key, cached_data, self.cache_ttl)
		except Exception as e:
			await self._log_error(f"Failed to cache search response: {str(e)}")
	
	async def _track_search_analytics(self, query: SearchQuery, response: SearchResponse):
		"""Track search analytics"""
		try:
			# Track popular queries
			if query.query_text:
				self.popular_queries[query.query_text.lower()] += 1
			
			# Track search patterns
			self.search_analytics[f"method_{query.search_method.value}"] += 1
			self.search_analytics[f"scope_{query.search_scope.value}"] += 1
			
			# Track result counts
			if response.total_results == 0:
				self.search_analytics['zero_results'] += 1
			elif response.total_results == 1:
				self.search_analytics['single_result'] += 1
			else:
				self.search_analytics['multiple_results'] += 1
			
		except Exception as e:
			await self._log_error(f"Failed to track search analytics: {str(e)}")
	
	async def _load_search_analytics(self):
		"""Load search analytics from storage"""
		try:
			# Load analytics from Redis cache
			analytics_data = await self.db_manager.cache_get("search_analytics")
			if analytics_data:
				self.search_analytics = json.loads(analytics_data)
			else:
				# Initialize empty analytics
				self.search_analytics = {
					"total_searches": 0,
					"successful_searches": 0,
					"popular_terms": {},
					"response_times": [],
					"last_updated": datetime.utcnow().isoformat()
				}
		except Exception as e:
			await self._log_error(f"Failed to load search analytics: {str(e)}")
			self.search_analytics = {}
	
	async def get_search_analytics(self) -> Dict[str, Any]:
		"""Get search analytics summary"""
		return {
			"total_searches": self.search_analytics.get('total_searches', 0),
			"cache_hit_rate": self.search_analytics.get('cache_hits', 0) / max(self.search_analytics.get('total_searches', 1), 1),
			"zero_results_rate": self.search_analytics.get('zero_results', 0) / max(self.search_analytics.get('total_searches', 1), 1),
			"popular_queries": dict(self.popular_queries.most_common(10)),
			"method_usage": {
				k.replace('method_', ''): v 
				for k, v in self.search_analytics.items() 
				if k.startswith('method_')
			},
			"scope_usage": {
				k.replace('scope_', ''): v 
				for k, v in self.search_analytics.items() 
				if k.startswith('scope_')
			}
		}
	
	async def _generate_query_embedding(self, query_text: str) -> Optional[List[float]]:
		"""Generate embedding for query text using local model"""
		try:
			if self.integration_manager:
				# Use Ollama to generate embeddings
				embedding = await self.integration_manager.get_text_embedding(query_text)
				return embedding
			return None
		except Exception as e:
			await self._log_error(f"Failed to generate query embedding: {str(e)}")
			return None
	
	def _calculate_asset_similarity(self, asset1, asset2) -> float:
		"""Calculate similarity score between two assets"""
		score = 0.0
		
		# Asset type similarity (40% weight)
		if asset1.asset_type == asset2.asset_type:
			score += 0.4
		
		# Source system similarity (20% weight)
		if asset1.source_system and asset2.source_system:
			if asset1.source_system == asset2.source_system:
				score += 0.2
		
		# Tag similarity (20% weight)
		if asset1.tags and asset2.tags:
			tags1 = set(asset1.tags)
			tags2 = set(asset2.tags)
			if tags1 and tags2:
				jaccard = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
				score += jaccard * 0.2
		
		# Name similarity (20% weight) - simple string similarity
		if asset1.name and asset2.name:
			name_sim = self._calculate_string_similarity(asset1.name, asset2.name)
			score += name_sim * 0.2
		
		return min(score, 1.0)
	
	def _calculate_string_similarity(self, str1: str, str2: str) -> float:
		"""Calculate string similarity using Levenshtein distance"""
		import difflib
		return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
	
	async def _get_lineage_paths(self, asset_id: str, tenant_id: str, direction: str, max_depth: int = 3) -> List[Dict]:
		"""Get lineage paths for an asset"""
		try:
			if self.integration_manager:
				# This would typically call the lineage engine
				# For now, return empty list
				return []
			return []
		except Exception as e:
			await self._log_error(f"Failed to get lineage paths: {str(e)}")
			return []
	
	def _calculate_lineage_relevance(self, root_asset_id: str, target_asset_id: str, lineage_paths: List[Dict]) -> float:
		"""Calculate relevance score based on lineage distance"""
		min_distance = float('inf')
		
		for path in lineage_paths:
			steps = path.get('steps', [])
			for i, step in enumerate(steps):
				if step.get('asset_id') == target_asset_id:
					min_distance = min(min_distance, i)
					break
		
		if min_distance == float('inf'):
			return 0.0
		
		# Convert distance to relevance score (closer = higher score)
		return max(0.1, 1.0 - (min_distance / 10.0))
	
	async def _calculate_relevance_score(self, asset, query: SearchQuery) -> float:
		"""Calculate relevance score for asset based on query"""
		score = 0.0
		
		query_text = query.query_text.lower()
		
		# Name match (highest weight)
		if asset.name and query_text in asset.name.lower():
			if asset.name.lower() == query_text:
				score += 1.0  # Exact match
			elif asset.name.lower().startswith(query_text):
				score += 0.8  # Prefix match
			else:
				score += 0.4  # Contains match
		
		# Display name match
		if asset.display_name and query_text in asset.display_name.lower():
			score += 0.3
		
		# Description match
		if asset.description and query_text in asset.description.lower():
			score += 0.2
		
		# Tag match
		if asset.tags:
			for tag in asset.tags:
				if query_text in tag.lower():
					score += 0.1
		
		# Quality score boost
		if asset.quality_score:
			score += (asset.quality_score / 100.0) * 0.1
		
		return min(score, 1.0)
	
	async def _generate_highlights(self, asset, query: SearchQuery) -> Dict[str, str]:
		"""Generate highlighted snippets for search matches"""
		highlights = {}
		query_text = query.query_text.lower()
		
		# Highlight name matches
		if asset.name and query_text in asset.name.lower():
			highlights['name'] = self._highlight_text(asset.name, query_text)
		
		# Highlight description matches
		if asset.description and query_text in asset.description.lower():
			highlights['description'] = self._highlight_text(asset.description, query_text)
		
		return highlights
	
	def _highlight_text(self, text: str, query: str) -> str:
		"""Add HTML highlighting to text"""
		import re
		pattern = re.compile(re.escape(query), re.IGNORECASE)
		return pattern.sub(f'<mark>{query}</mark>', text)
	
	async def _log_info(self, message: str):
		"""Log info message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META SEARCH INFO: {message}")
	
	async def _log_error(self, message: str):
		"""Log error message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META SEARCH ERROR: {message}")


# Factory function for easy initialization
async def create_search_engine(
	db_manager: MetaDatabaseManager,
	integration_manager: APGMetadataIntegrationManager,
	config: Dict[str, Any] = None
) -> MetadataSearchEngine:
	"""Factory function to create and initialize search engine"""
	search_engine = MetadataSearchEngine(db_manager, integration_manager, config)
	await search_engine.initialize()
	return search_engine