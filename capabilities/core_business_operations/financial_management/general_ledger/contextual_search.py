"""
APG Financial Management General Ledger - Advanced Transaction Contextual Search

Revolutionary search engine that understands natural language and context to find
any transaction, pattern, or financial insight across the entire GL system.

Features:
- Natural language query processing
- Semantic search across transaction descriptions
- Advanced filtering with business context
- Pattern recognition and anomaly detection in search results
- Time-based and relationship-based search
- Predictive search suggestions

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from decimal import Decimal
from enum import Enum
import re
import difflib
from fuzzywuzzy import fuzz
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class SearchType(Enum):
	"""Types of search queries"""
	NATURAL_LANGUAGE = "natural_language"
	SEMANTIC = "semantic"
	PATTERN = "pattern"
	ANOMALY = "anomaly"
	RELATIONSHIP = "relationship"
	TIME_SERIES = "time_series"
	AMOUNT_RANGE = "amount_range"
	COMPLIANCE = "compliance"


class ResultType(Enum):
	"""Types of search results"""
	TRANSACTION = "transaction"
	ACCOUNT = "account"
	PATTERN = "pattern"
	INSIGHT = "insight"
	ANOMALY = "anomaly"
	TREND = "trend"
	RELATIONSHIP = "relationship"


@dataclass
class SearchContext:
	"""Context for search operations"""
	user_id: str
	user_role: str
	tenant_id: str
	time_range: Optional[Tuple[datetime, datetime]] = None
	focus_accounts: List[str] = None
	business_context: str = None
	compliance_mode: bool = False


@dataclass
class SearchResult:
	"""Individual search result"""
	result_id: str
	result_type: ResultType
	title: str
	description: str
	relevance_score: float
	confidence: float
	data: Dict[str, Any]
	highlights: List[str]
	related_results: List[str]
	business_impact: str
	action_suggestions: List[str]


@dataclass
class SearchResponse:
	"""Complete search response"""
	query: str
	query_type: SearchType
	total_results: int
	processing_time_ms: float
	results: List[SearchResult]
	suggestions: List[str]
	insights: List[Dict[str, Any]]
	filters_applied: Dict[str, Any]
	search_context: SearchContext


class AdvancedTransactionSearchEngine:
	"""
	🎯 GAME CHANGER #5: Advanced Transaction Contextual Search
	
	Revolutionary search that understands natural language and business context:
	- "Show me all unusual office expenses last quarter"
	- "Find transactions similar to the ABC Corp payment"
	- "What caused the spike in travel expenses in March?"
	- "Show me all entries that might need audit review"
	"""
	
	def __init__(self, gl_service):
		self.gl_service = gl_service
		self.tenant_id = gl_service.tenant_id
		
		# Search components
		self.nlp_processor = NaturalLanguageSearchProcessor()
		self.semantic_engine = SemanticSearchEngine()
		self.pattern_detector = SearchPatternDetector()
		self.context_analyzer = SearchContextAnalyzer()
		
		# Search indexing
		self.search_index = SearchIndexManager()
		
		logger.info(f"Advanced Transaction Search Engine initialized for tenant {self.tenant_id}")
	
	async def search(self, query: str, context: SearchContext) -> SearchResponse:
		"""
		🎯 REVOLUTIONARY: Natural Language Financial Search
		
		Understands queries like:
		- "Show me rent payments over $5000 last year"
		- "Find all transactions similar to invoice #12345"
		- "What accounts have unusual activity this month?"
		- "Show me all entries that need manager approval"
		"""
		try:
			start_time = datetime.now()
			
			# Parse natural language query
			parsed_query = await self.nlp_processor.parse_search_query(query)
			
			# Determine search strategy based on query type
			search_strategy = await self._determine_search_strategy(parsed_query, context)
			
			# Execute multi-strategy search
			search_results = await self._execute_search_strategy(
				search_strategy, parsed_query, context
			)
			
			# Rank and filter results
			ranked_results = await self._rank_search_results(
				search_results, parsed_query, context
			)
			
			# Generate insights from results
			search_insights = await self._generate_search_insights(
				ranked_results, parsed_query, context
			)
			
			# Generate search suggestions
			suggestions = await self._generate_search_suggestions(
				query, ranked_results, context
			)
			
			processing_time = (datetime.now() - start_time).total_seconds() * 1000
			
			response = SearchResponse(
				query=query,
				query_type=search_strategy["primary_type"],
				total_results=len(ranked_results),
				processing_time_ms=processing_time,
				results=ranked_results,
				suggestions=suggestions,
				insights=search_insights,
				filters_applied=search_strategy.get("filters", {}),
				search_context=context
			)
			
			# Learn from search for improvement
			await self._learn_from_search(query, response, context)
			
			return response
			
		except Exception as e:
			logger.error(f"Error in search: {e}")
			raise
	
	async def search_similar_transactions(self, reference_transaction_id: str,
										context: SearchContext) -> List[SearchResult]:
		"""
		🎯 REVOLUTIONARY: Find Similar Transactions
		
		Finds transactions similar to a reference transaction based on:
		- Amount patterns
		- Description similarity
		- Account usage patterns
		- Timing patterns
		- Business context
		"""
		try:
			# Get reference transaction
			reference_tx = await self.gl_service.get_journal_entry(reference_transaction_id)
			if not reference_tx:
				return []
			
			# Extract similarity features
			similarity_features = await self._extract_similarity_features(reference_tx)
			
			# Find similar transactions using multiple algorithms
			similar_transactions = []
			
			# Amount-based similarity
			amount_similar = await self._find_amount_similar_transactions(
				similarity_features, context
			)
			similar_transactions.extend(amount_similar)
			
			# Description-based similarity
			desc_similar = await self._find_description_similar_transactions(
				similarity_features, context
			)
			similar_transactions.extend(desc_similar)
			
			# Pattern-based similarity
			pattern_similar = await self._find_pattern_similar_transactions(
				similarity_features, context
			)
			similar_transactions.extend(pattern_similar)
			
			# Account usage similarity
			account_similar = await self._find_account_usage_similar_transactions(
				similarity_features, context
			)
			similar_transactions.extend(account_similar)
			
			# Deduplicate and rank
			unique_results = await self._deduplicate_and_rank_similar(
				similar_transactions, similarity_features
			)
			
			return unique_results
			
		except Exception as e:
			logger.error(f"Error finding similar transactions: {e}")
			return []
	
	async def search_anomalies(self, search_criteria: Dict[str, Any],
							 context: SearchContext) -> List[SearchResult]:
		"""
		🎯 REVOLUTIONARY: Anomaly Detection Search
		
		Finds anomalies based on sophisticated criteria:
		- Statistical outliers in amounts
		- Unusual timing patterns
		- Unexpected account combinations
		- Compliance violations
		- Pattern deviations
		"""
		try:
			anomaly_results = []
			
			# Amount-based anomalies
			if search_criteria.get("detect_amount_anomalies", True):
				amount_anomalies = await self._detect_amount_anomalies(context)
				anomaly_results.extend(amount_anomalies)
			
			# Timing anomalies
			if search_criteria.get("detect_timing_anomalies", True):
				timing_anomalies = await self._detect_timing_anomalies(context)
				anomaly_results.extend(timing_anomalies)
			
			# Account combination anomalies
			if search_criteria.get("detect_account_anomalies", True):
				account_anomalies = await self._detect_account_combination_anomalies(context)
				anomaly_results.extend(account_anomalies)
			
			# Compliance anomalies
			if search_criteria.get("detect_compliance_anomalies", True):
				compliance_anomalies = await self._detect_compliance_anomalies(context)
				anomaly_results.extend(compliance_anomalies)
			
			# Pattern deviation anomalies
			if search_criteria.get("detect_pattern_anomalies", True):
				pattern_anomalies = await self._detect_pattern_anomalies(context)
				anomaly_results.extend(pattern_anomalies)
			
			# Rank by anomaly severity and confidence
			ranked_anomalies = await self._rank_anomaly_results(anomaly_results)
			
			return ranked_anomalies
			
		except Exception as e:
			logger.error(f"Error detecting anomalies: {e}")
			return []
	
	async def search_patterns(self, pattern_query: str, 
							context: SearchContext) -> List[SearchResult]:
		"""
		🎯 REVOLUTIONARY: Pattern Recognition Search
		
		Finds recurring patterns in transactions:
		- Monthly recurring expenses
		- Seasonal patterns
		- Vendor payment patterns
		- Account usage patterns
		- Amount clustering patterns
		"""
		try:
			# Parse pattern query
			pattern_criteria = await self._parse_pattern_query(pattern_query)
			
			pattern_results = []
			
			# Recurring transaction patterns
			if pattern_criteria.get("find_recurring", True):
				recurring_patterns = await self._find_recurring_patterns(context)
				pattern_results.extend(recurring_patterns)
			
			# Seasonal patterns
			if pattern_criteria.get("find_seasonal", True):
				seasonal_patterns = await self._find_seasonal_patterns(context)
				pattern_results.extend(seasonal_patterns)
			
			# Vendor patterns
			if pattern_criteria.get("find_vendor_patterns", True):
				vendor_patterns = await self._find_vendor_patterns(context)
				pattern_results.extend(vendor_patterns)
			
			# Amount clustering patterns
			if pattern_criteria.get("find_amount_clusters", True):
				amount_patterns = await self._find_amount_clustering_patterns(context)
				pattern_results.extend(amount_patterns)
			
			# Account relationship patterns
			if pattern_criteria.get("find_account_relationships", True):
				relationship_patterns = await self._find_account_relationship_patterns(context)
				pattern_results.extend(relationship_patterns)
			
			# Rank patterns by significance and frequency
			ranked_patterns = await self._rank_pattern_results(pattern_results)
			
			return ranked_patterns
			
		except Exception as e:
			logger.error(f"Error finding patterns: {e}")
			return []
	
	async def get_search_suggestions(self, partial_query: str,
									context: SearchContext) -> List[str]:
		"""
		🎯 REVOLUTIONARY: Predictive Search Suggestions
		
		Provides intelligent search suggestions as user types:
		- Based on user's search history
		- Common search patterns in their role
		- Available data and context
		- Business-relevant suggestions
		"""
		try:
			suggestions = []
			
			# Historical search patterns
			historical_suggestions = await self._get_historical_search_suggestions(
				partial_query, context
			)
			suggestions.extend(historical_suggestions)
			
			# Role-based suggestions
			role_suggestions = await self._get_role_based_suggestions(
				partial_query, context
			)
			suggestions.extend(role_suggestions)
			
			# Context-aware suggestions
			context_suggestions = await self._get_context_aware_suggestions(
				partial_query, context
			)
			suggestions.extend(context_suggestions)
			
			# Auto-complete suggestions
			autocomplete_suggestions = await self._get_autocomplete_suggestions(
				partial_query, context
			)
			suggestions.extend(autocomplete_suggestions)
			
			# Smart query expansion suggestions
			expansion_suggestions = await self._get_query_expansion_suggestions(
				partial_query, context
			)
			suggestions.extend(expansion_suggestions)
			
			# Deduplicate and rank suggestions
			unique_suggestions = list(set(suggestions))
			ranked_suggestions = await self._rank_suggestions(
				unique_suggestions, partial_query, context
			)
			
			return ranked_suggestions[:10]  # Top 10 suggestions
			
		except Exception as e:
			logger.error(f"Error getting search suggestions: {e}")
			return []
	
	# =====================================
	# PRIVATE HELPER METHODS
	# =====================================
	
	async def _determine_search_strategy(self, parsed_query: Dict[str, Any],
										context: SearchContext) -> Dict[str, Any]:
		"""Determine optimal search strategy based on query"""
		
		strategy = {
			"primary_type": SearchType.NATURAL_LANGUAGE,
			"secondary_types": [],
			"filters": {},
			"weights": {}
		}
		
		# Analyze query intent
		if parsed_query.get("intent") == "find_similar":
			strategy["primary_type"] = SearchType.SEMANTIC
			strategy["secondary_types"] = [SearchType.PATTERN]
		
		elif parsed_query.get("intent") == "find_anomalies":
			strategy["primary_type"] = SearchType.ANOMALY
			strategy["secondary_types"] = [SearchType.PATTERN]
		
		elif parsed_query.get("intent") == "find_patterns":
			strategy["primary_type"] = SearchType.PATTERN
			strategy["secondary_types"] = [SearchType.TIME_SERIES]
		
		elif parsed_query.get("intent") == "amount_search":
			strategy["primary_type"] = SearchType.AMOUNT_RANGE
			strategy["secondary_types"] = [SearchType.SEMANTIC]
		
		elif parsed_query.get("intent") == "compliance_search":
			strategy["primary_type"] = SearchType.COMPLIANCE
			strategy["secondary_types"] = [SearchType.ANOMALY]
		
		# Apply context-based strategy adjustments
		if context.compliance_mode:
			strategy["secondary_types"].append(SearchType.COMPLIANCE)
		
		if context.time_range:
			strategy["secondary_types"].append(SearchType.TIME_SERIES)
		
		# Set up filters based on context
		if context.focus_accounts:
			strategy["filters"]["accounts"] = context.focus_accounts
		
		if context.time_range:
			strategy["filters"]["time_range"] = context.time_range
		
		return strategy
	
	async def _execute_search_strategy(self, strategy: Dict[str, Any],
									 parsed_query: Dict[str, Any],
									 context: SearchContext) -> List[SearchResult]:
		"""Execute the determined search strategy"""
		
		all_results = []
		
		# Execute primary search
		primary_results = await self._execute_primary_search(
			strategy["primary_type"], parsed_query, context
		)
		all_results.extend(primary_results)
		
		# Execute secondary searches
		for secondary_type in strategy.get("secondary_types", []):
			secondary_results = await self._execute_secondary_search(
				secondary_type, parsed_query, context
			)
			all_results.extend(secondary_results)
		
		# Apply filters
		filtered_results = await self._apply_search_filters(
			all_results, strategy.get("filters", {}), context
		)
		
		return filtered_results
	
	async def _generate_search_insights(self, results: List[SearchResult],
									  parsed_query: Dict[str, Any],
									  context: SearchContext) -> List[Dict[str, Any]]:
		"""Generate insights from search results"""
		
		insights = []
		
		if not results:
			return insights
		
		# Amount distribution insight
		amounts = [r.data.get("amount", 0) for r in results if r.data.get("amount")]
		if amounts:
			insights.append({
				"type": "amount_distribution",
				"title": "Amount Distribution",
				"description": f"Found {len(amounts)} transactions with amounts ranging from ${min(amounts):,.2f} to ${max(amounts):,.2f}",
				"data": {
					"min_amount": min(amounts),
					"max_amount": max(amounts),
					"avg_amount": sum(amounts) / len(amounts),
					"total_amount": sum(amounts)
				}
			})
		
		# Time distribution insight
		dates = [r.data.get("date") for r in results if r.data.get("date")]
		if dates:
			insights.append({
				"type": "time_distribution",
				"title": "Time Distribution",
				"description": f"Transactions span from {min(dates)} to {max(dates)}",
				"data": {
					"earliest_date": min(dates),
					"latest_date": max(dates),
					"date_range_days": (max(dates) - min(dates)).days if isinstance(min(dates), datetime) else None
				}
			})
		
		# Account distribution insight
		accounts = [r.data.get("account_id") for r in results if r.data.get("account_id")]
		if accounts:
			account_counts = {}
			for account in accounts:
				account_counts[account] = account_counts.get(account, 0) + 1
			
			insights.append({
				"type": "account_distribution",
				"title": "Account Distribution", 
				"description": f"Transactions involve {len(set(accounts))} different accounts",
				"data": {
					"unique_accounts": len(set(accounts)),
					"most_active_account": max(account_counts.items(), key=lambda x: x[1])[0],
					"account_counts": account_counts
				}
			})
		
		# Pattern insights
		if len(results) > 5:
			pattern_insight = await self._analyze_result_patterns(results)
			if pattern_insight:
				insights.append(pattern_insight)
		
		return insights


class NaturalLanguageSearchProcessor:
	"""Processes natural language search queries"""
	
	async def parse_search_query(self, query: str) -> Dict[str, Any]:
		"""Parse natural language search query"""
		
		query_lower = query.lower()
		parsed = {
			"original_query": query,
			"intent": "general_search",
			"entities": {},
			"filters": {},
			"confidence": 0.5
		}
		
		# Intent detection
		if any(word in query_lower for word in ["similar", "like", "same as"]):
			parsed["intent"] = "find_similar"
			parsed["confidence"] = 0.8
		
		elif any(word in query_lower for word in ["unusual", "anomaly", "outlier", "strange"]):
			parsed["intent"] = "find_anomalies"
			parsed["confidence"] = 0.9
		
		elif any(word in query_lower for word in ["pattern", "recurring", "regular", "repeated"]):
			parsed["intent"] = "find_patterns"
			parsed["confidence"] = 0.8
		
		elif any(word in query_lower for word in ["compliance", "audit", "violation", "rule"]):
			parsed["intent"] = "compliance_search"
			parsed["confidence"] = 0.9
		
		# Amount extraction
		amount_patterns = [
			r'\$?([\d,]+(?:\.\d{2})?)',
			r'over\s+\$?([\d,]+)',
			r'above\s+\$?([\d,]+)',
			r'more than\s+\$?([\d,]+)',
			r'less than\s+\$?([\d,]+)',
			r'under\s+\$?([\d,]+)'
		]
		
		for pattern in amount_patterns:
			match = re.search(pattern, query_lower)
			if match:
				parsed["entities"]["amount"] = match.group(1).replace(",", "")
				break
		
		# Time period extraction
		time_patterns = [
			r'last\s+(week|month|quarter|year)',
			r'this\s+(week|month|quarter|year)',
			r'in\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
			r'(\d{4})',  # Year
			r'q[1-4]',   # Quarter
		]
		
		for pattern in time_patterns:
			match = re.search(pattern, query_lower)
			if match:
				parsed["entities"]["time_period"] = match.group(1) if match.groups() else match.group(0)
				break
		
		# Account/category extraction
		account_keywords = [
			"rent", "office", "travel", "expense", "revenue", "salary", "equipment",
			"supplies", "utilities", "insurance", "legal", "consulting", "marketing"
		]
		
		for keyword in account_keywords:
			if keyword in query_lower:
				parsed["entities"]["category"] = keyword
				break
		
		return parsed


class SemanticSearchEngine:
	"""Semantic search using embeddings and similarity"""
	
	async def semantic_search(self, query_embedding: List[float],
							context: SearchContext) -> List[SearchResult]:
		"""Perform semantic search using embeddings"""
		
		# This would integrate with actual embedding models
		# For now, we'll simulate semantic search
		
		results = []
		
		# Mock semantic search results
		results.append(SearchResult(
			result_id="semantic_001",
			result_type=ResultType.TRANSACTION,
			title="Office Rent Payment - January 2025",
			description="Monthly office rent payment to ABC Properties",
			relevance_score=0.92,
			confidence=0.88,
			data={
				"transaction_id": "je_001",
				"amount": Decimal("5000.00"),
				"date": datetime(2025, 1, 15),
				"account_id": "6000"
			},
			highlights=["office rent", "payment"],
			related_results=["semantic_002", "semantic_003"],
			business_impact="Regular operational expense",
			action_suggestions=["Review for annual increase", "Validate lease terms"]
		))
		
		return results


class SearchPatternDetector:
	"""Detects patterns in search results"""
	
	async def detect_patterns(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Detect patterns in transaction data"""
		
		patterns = []
		
		# Amount clustering patterns
		amounts = [float(t.get("amount", 0)) for t in transactions]
		if amounts:
			# Simple clustering - would use ML clustering in production
			avg_amount = sum(amounts) / len(amounts)
			patterns.append({
				"type": "amount_clustering",
				"description": f"Average transaction amount: ${avg_amount:.2f}",
				"significance": 0.7
			})
		
		# Frequency patterns
		if len(transactions) > 10:
			patterns.append({
				"type": "high_frequency",
				"description": f"High transaction frequency detected ({len(transactions)} transactions)",
				"significance": 0.8
			})
		
		return patterns


class SearchContextAnalyzer:
	"""Analyzes search context for better results"""
	
	async def analyze_context(self, context: SearchContext) -> Dict[str, Any]:
		"""Analyze search context for optimization"""
		
		analysis = {
			"user_preferences": {},
			"role_insights": {},
			"business_context": {},
			"optimization_suggestions": []
		}
		
		# Role-based context
		if context.user_role == "accountant":
			analysis["role_insights"]["focus_areas"] = ["accuracy", "compliance", "detail"]
		elif context.user_role == "manager":
			analysis["role_insights"]["focus_areas"] = ["trends", "summaries", "anomalies"]
		elif context.user_role == "auditor":
			analysis["role_insights"]["focus_areas"] = ["compliance", "risk", "documentation"]
		
		# Time-based context
		if context.time_range:
			days_span = (context.time_range[1] - context.time_range[0]).days
			if days_span > 365:
				analysis["optimization_suggestions"].append("Consider year-over-year analysis")
			elif days_span < 7:
				analysis["optimization_suggestions"].append("Consider daily trend analysis")
		
		return analysis


class SearchIndexManager:
	"""Manages search indexing for fast retrieval"""
	
	def __init__(self):
		self.transaction_index = {}
		self.account_index = {}
		self.description_index = {}
	
	async def index_transaction(self, transaction: Dict[str, Any]):
		"""Index a transaction for fast search"""
		
		transaction_id = transaction.get("id")
		if not transaction_id:
			return
		
		# Index by amount
		amount = transaction.get("amount", 0)
		amount_bucket = int(amount // 1000) * 1000  # Bucket by thousands
		
		if amount_bucket not in self.transaction_index:
			self.transaction_index[amount_bucket] = []
		self.transaction_index[amount_bucket].append(transaction_id)
		
		# Index by account
		account_id = transaction.get("account_id")
		if account_id:
			if account_id not in self.account_index:
				self.account_index[account_id] = []
			self.account_index[account_id].append(transaction_id)
		
		# Index by description keywords
		description = transaction.get("description", "").lower()
		keywords = description.split()
		for keyword in keywords:
			if len(keyword) > 3:  # Skip short words
				if keyword not in self.description_index:
					self.description_index[keyword] = []
				self.description_index[keyword].append(transaction_id)
	
	async def search_index(self, search_terms: List[str]) -> List[str]:
		"""Search the index for matching transaction IDs"""
		
		matching_ids = set()
		
		for term in search_terms:
			term_lower = term.lower()
			
			# Search description index
			if term_lower in self.description_index:
				matching_ids.update(self.description_index[term_lower])
			
			# Fuzzy search in description index
			for indexed_term in self.description_index:
				if fuzz.ratio(term_lower, indexed_term) > 80:
					matching_ids.update(self.description_index[indexed_term])
		
		return list(matching_ids)


# Export search classes
__all__ = [
	'AdvancedTransactionSearchEngine',
	'SearchContext',
	'SearchResult',
	'SearchResponse',
	'NaturalLanguageSearchProcessor',
	'SemanticSearchEngine',
	'SearchPatternDetector',
	'SearchContextAnalyzer',
	'SearchIndexManager',
	'SearchType',
	'ResultType'
]