#!/usr/bin/env python3
"""
APG Data Virtualization (DVRL) NLP Integration
Natural language query processing and semantic understanding

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid_extensions import uuid7str

# Import real NLP implementation with fallback to mock
try:
	from .real_implementations import RealAPGNLPProcessor as APGNLPProcessor
	REAL_NLP_AVAILABLE = True
except ImportError:
	REAL_NLP_AVAILABLE = False
	
	# Fallback mock APG NLP Integration
	class APGNLPProcessor:
	"""Integration with APG's nlpc capability for natural language processing"""
	
	def __init__(self, tenant_id: str, user_id: str):
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.query_patterns = self._load_query_patterns()
		self.schema_context = {}
		self.conversation_history = []
		
	def _load_query_patterns(self) -> Dict[str, Dict[str, Any]]:
		"""Load natural language query patterns"""
		return {
			'aggregation_patterns': [
				r'(?:count|number of|how many)\s+(\w+)',
				r'(?:total|sum of|add up)\s+(\w+)',
				r'(?:average|avg|mean)\s+(\w+)',
				r'(?:maximum|max|highest)\s+(\w+)',
				r'(?:minimum|min|lowest)\s+(\w+)'
			],
			'filter_patterns': [
				r'(?:where|with|having)\s+(\w+)\s+(?:is|equals?|=)\s+([^\s]+)',
				r'(\w+)\s+(?:greater than|>)\s+([^\s]+)',
				r'(\w+)\s+(?:less than|<)\s+([^\s]+)',
				r'(?:in|during)\s+(\w+)\s+(\d{4})',
				r'(?:from|since)\s+([^\s]+)\s+(?:to|until)\s+([^\s]+)'
			],
			'join_patterns': [
				r'(?:join|combine|merge)\s+(\w+)\s+(?:with|and)\s+(\w+)',
				r'(\w+)\s+(?:and|with)\s+(\w+)',
				r'(?:relate|link|connect)\s+(\w+)\s+(?:to|with)\s+(\w+)'
			],
			'time_patterns': [
				r'(?:today|yesterday|last week|this month|this year)',
				r'(?:in|during)\s+(\d{4})',
				r'(?:between|from)\s+([^\s]+)\s+(?:and|to)\s+([^\s]+)',
				r'(?:last|past)\s+(\d+)\s+(days?|weeks?|months?|years?)'
			]
		}
	
	async def process_natural_language_query(self, natural_query: str, schema_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Process natural language query and generate SQL"""
		await self._log_info(f"Processing NL query: {natural_query[:50]}...")
		
		# Store schema context for better understanding
		if schema_context:
			self.schema_context = schema_context
		
		# Clean and normalize input
		normalized_query = self._normalize_query(natural_query)
		
		# Extract intent and entities
		intent = await self._extract_intent(normalized_query)
		entities = await self._extract_entities(normalized_query)
		
		# Generate SQL query
		sql_query = await self._generate_sql(intent, entities, normalized_query)
		
		# Calculate confidence score
		confidence = await self._calculate_confidence(intent, entities, sql_query)
		
		# Store in conversation history
		conversation_entry = {
			'timestamp': datetime.now(timezone.utc).isoformat(),
			'natural_query': natural_query,
			'normalized_query': normalized_query,
			'intent': intent,
			'entities': entities,
			'generated_sql': sql_query,
			'confidence': confidence
		}
		self.conversation_history.append(conversation_entry)
		
		return {
			'original_query': natural_query,
			'normalized_query': normalized_query,
			'intent': intent,
			'entities': entities,
			'sql_query': sql_query,
			'confidence_score': confidence,
			'processing_time_ms': 45,  # Mock processing time
			'suggestions': await self._generate_suggestions(normalized_query),
			'conversation_id': uuid7str()
		}
	
	def _normalize_query(self, query: str) -> str:
		"""Normalize natural language query"""
		# Convert to lowercase and clean whitespace
		normalized = query.lower().strip()
		
		# Remove common filler words
		filler_words = ['please', 'can you', 'could you', 'i want', 'i need', 'show me', 'tell me']
		for filler in filler_words:
			normalized = normalized.replace(filler, '').strip()
		
		# Normalize spacing
		normalized = re.sub(r'\s+', ' ', normalized)
		
		return normalized
	
	async def _extract_intent(self, query: str) -> Dict[str, Any]:
		"""Extract query intent (SELECT, COUNT, etc.)"""
		intents = {
			'select': ['show', 'list', 'get', 'find', 'display', 'retrieve'],
			'count': ['count', 'number of', 'how many'],
			'sum': ['total', 'sum', 'add up'],
			'average': ['average', 'avg', 'mean'],
			'max': ['maximum', 'max', 'highest', 'largest'],
			'min': ['minimum', 'min', 'lowest', 'smallest'],
			'group': ['group by', 'group', 'categorize', 'organize'],
			'filter': ['where', 'with', 'having', 'filter']
		}
		
		detected_intents = []
		for intent_type, keywords in intents.items():
			for keyword in keywords:
				if keyword in query:
					detected_intents.append(intent_type)
					break
		
		# Default to select if no specific intent found
		if not detected_intents:
			detected_intents = ['select']
		
		return {
			'primary_intent': detected_intents[0],
			'all_intents': detected_intents,
			'confidence': 0.85
		}
	
	async def _extract_entities(self, query: str) -> Dict[str, Any]:
		"""Extract entities (tables, columns, values, etc.)"""
		entities = {
			'tables': [],
			'columns': [],
			'values': [],
			'time_ranges': [],
			'operators': []
		}
		
		# Extract table names from schema context
		if self.schema_context:
			for table_name in self.schema_context.get('tables', []):
				if table_name.lower() in query:
					entities['tables'].append(table_name)
		
		# Mock entity extraction - in production would use NER models
		common_entities = {
			'users': 'table',
			'customers': 'table',
			'orders': 'table',
			'products': 'table',
			'sales': 'table',
			'name': 'column',
			'email': 'column',
			'price': 'column',
			'date': 'column',
			'status': 'column'
		}
		
		for word, entity_type in common_entities.items():
			if word in query:
				if entity_type == 'table':
					entities['tables'].append(word)
				elif entity_type == 'column':
					entities['columns'].append(word)
		
		# Extract numeric values
		numbers = re.findall(r'\d+(?:\.\d+)?', query)
		entities['values'].extend(numbers)
		
		# Extract time expressions
		time_matches = re.findall(r'(?:today|yesterday|last week|this month|\d{4})', query)
		entities['time_ranges'].extend(time_matches)
		
		# Extract operators
		if any(op in query for op in ['greater than', '>', 'more than']):
			entities['operators'].append('>')
		if any(op in query for op in ['less than', '<', 'fewer than']):
			entities['operators'].append('<')
		if any(op in query for op in ['equals', '=', 'is']):
			entities['operators'].append('=')
		
		return entities
	
	async def _generate_sql(self, intent: Dict[str, Any], entities: Dict[str, Any], query: str) -> str:
		"""Generate SQL query from intent and entities"""
		primary_intent = intent['primary_intent']
		tables = entities.get('tables', [])
		columns = entities.get('columns', [])
		
		# Default table if none specified
		if not tables:
			tables = ['users']  # Default assumption
		
		# Generate SQL based on intent
		if primary_intent == 'count':
			sql = f"SELECT COUNT(*) FROM {tables[0]}"
		elif primary_intent in ['sum', 'average', 'max', 'min']:
			agg_func = primary_intent.upper().replace('AVERAGE', 'AVG')
			column = columns[0] if columns else 'value'
			sql = f"SELECT {agg_func}({column}) FROM {tables[0]}"
		else:
			# Default SELECT query
			if columns:
				column_list = ', '.join(columns)
			else:
				column_list = '*'
			sql = f"SELECT {column_list} FROM {tables[0]}"
		
		# Add WHERE clause if filter conditions detected
		if entities.get('operators') and entities.get('values'):
			column = columns[0] if columns else 'id'
			operator = entities['operators'][0]
			value = entities['values'][0]
			sql += f" WHERE {column} {operator} {value}"
		
		# Add time filters
		if entities.get('time_ranges'):
			time_range = entities['time_ranges'][0]
			if time_range.isdigit():  # Year
				sql += f" WHERE YEAR(created_at) = {time_range}"
			elif time_range == 'today':
				sql += " WHERE DATE(created_at) = CURRENT_DATE"
		
		# Add LIMIT for reasonable result sizes
		if primary_intent == 'select' and 'LIMIT' not in sql.upper():
			sql += " LIMIT 100"
		
		return sql
	
	async def _calculate_confidence(self, intent: Dict[str, Any], entities: Dict[str, Any], sql: str) -> float:
		"""Calculate confidence score for generated query"""
		score = 0.5  # Base score
		
		# Increase confidence if entities were found
		if entities.get('tables'):
			score += 0.2
		if entities.get('columns'):
			score += 0.2
		
		# Increase confidence based on intent clarity
		if intent.get('confidence', 0) > 0.8:
			score += 0.1
		
		# Check SQL validity (basic)
		if self._is_valid_sql_structure(sql):
			score += 0.1
		
		return min(score, 1.0)
	
	def _is_valid_sql_structure(self, sql: str) -> bool:
		"""Basic SQL structure validation"""
		sql_upper = sql.upper().strip()
		return sql_upper.startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE'))
	
	async def _generate_suggestions(self, query: str) -> List[str]:
		"""Generate query suggestions for refinement"""
		suggestions = []
		
		# Suggest more specific queries
		if 'users' in query:
			suggestions.extend([
				"Show users created this month",
				"Count users by status",
				"List top 10 most active users"
			])
		
		if 'orders' in query:
			suggestions.extend([
				"Show orders from last week",
				"Calculate total sales this year",
				"Find orders above $100"
			])
		
		# Suggest aggregation improvements
		if any(word in query for word in ['show', 'list', 'get']):
			suggestions.extend([
				"Add time filter (e.g., 'from last month')",
				"Count instead of listing",
				"Group by category"
			])
		
		return suggestions[:5]  # Limit to 5 suggestions
	
	async def get_conversation_context(self) -> Dict[str, Any]:
		"""Get conversation history and context"""
		return {
			'conversation_length': len(self.conversation_history),
			'recent_queries': [entry['natural_query'] for entry in self.conversation_history[-5:]],
			'common_intents': self._analyze_common_intents(),
			'schema_context_available': bool(self.schema_context)
		}
	
	def _analyze_common_intents(self) -> Dict[str, int]:
		"""Analyze common intents from conversation history"""
		intent_counts = {}
		for entry in self.conversation_history:
			intent = entry.get('intent', {}).get('primary_intent', 'unknown')
			intent_counts[intent] = intent_counts.get(intent, 0) + 1
		return intent_counts
	
	async def _log_info(self, message: str, context: dict = None) -> None:
		"""Log info message"""
		timestamp = datetime.now(timezone.utc).isoformat()
		print(f"[{timestamp}] NLP INFO: {message}")


# Create the NLP processor instance based on availability
if not REAL_NLP_AVAILABLE:
	# If fallback mock class was used, complete the fallback structure
	pass

class QuerySuggestionEngine:
	"""Generate intelligent query suggestions"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.query_templates = self._load_query_templates()
		
	def _load_query_templates(self) -> Dict[str, List[str]]:
		"""Load query templates for different domains"""
		return {
			'business_intelligence': [
				"Show sales by region for {time_period}",
				"Count customers who purchased {product_type}",
				"Calculate average order value in {time_period}",
				"List top 10 products by revenue",
				"Show customer retention rate"
			],
			'user_analytics': [
				"How many users signed up {time_period}?",
				"Show user activity by day of week",
				"Count active users in {time_period}",
				"List users who haven't logged in for 30 days",
				"Show user demographics breakdown"
			],
			'operational': [
				"Show system errors from {time_period}",
				"Count failed transactions today",
				"List servers with high CPU usage",
				"Show database connection pool status",
				"Find slow queries from last hour"
			]
		}
	
	async def generate_contextual_suggestions(self, schema_info: Dict[str, Any], user_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
		"""Generate contextual query suggestions based on schema"""
		suggestions = []
		
		# Analyze available tables to suggest relevant queries
		tables = schema_info.get('tables', [])
		
		for table in tables[:10]:  # Limit to avoid overwhelming
			table_suggestions = await self._generate_table_suggestions(table)
			suggestions.extend(table_suggestions)
		
		# Add domain-specific suggestions
		domain = user_context.get('domain', 'business_intelligence') if user_context else 'business_intelligence'
		domain_suggestions = await self._generate_domain_suggestions(domain)
		suggestions.extend(domain_suggestions)
		
		return suggestions
	
	async def _generate_table_suggestions(self, table_info: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate suggestions for a specific table"""
		table_name = table_info.get('name', 'table')
		columns = table_info.get('columns', [])
		
		suggestions = []
		
		# Basic queries
		suggestions.append({
			'type': 'basic',
			'query': f"Show all data from {table_name}",
			'sql_template': f"SELECT * FROM {table_name} LIMIT 100",
			'description': f"Browse {table_name} data"
		})
		
		# Count query
		suggestions.append({
			'type': 'aggregate',
			'query': f"How many records are in {table_name}?",
			'sql_template': f"SELECT COUNT(*) FROM {table_name}",
			'description': f"Count total records in {table_name}"
		})
		
		# Time-based queries if date columns exist
		date_columns = [col['name'] for col in columns if 'date' in col.get('name', '').lower() or 'time' in col.get('name', '').lower()]
		if date_columns:
			date_col = date_columns[0]
			suggestions.append({
				'type': 'temporal',
				'query': f"Show {table_name} data from last week",
				'sql_template': f"SELECT * FROM {table_name} WHERE {date_col} >= CURRENT_DATE - INTERVAL 7 DAY",
				'description': f"Recent {table_name} records"
			})
		
		return suggestions
	
	async def _generate_domain_suggestions(self, domain: str) -> List[Dict[str, Any]]:
		"""Generate domain-specific query suggestions"""
		templates = self.query_templates.get(domain, [])
		
		suggestions = []
		for template in templates[:3]:  # Limit to 3 per domain
			suggestions.append({
				'type': 'domain_template',
				'query': template.replace('{time_period}', 'this month').replace('{product_type}', 'electronics'),
				'description': f"Common {domain} query",
				'category': domain
			})
		
		return suggestions


class SemanticQueryMatcher:
	"""Match queries to similar previous queries using semantic similarity"""
	
	def __init__(self):
		self.query_embeddings = {}  # Would use actual embeddings in production
		self.similarity_threshold = 0.7
	
	async def find_similar_queries(self, query: str, query_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Find semantically similar queries from history"""
		similar_queries = []
		
		for historical_query in query_history:
			similarity = await self._calculate_similarity(query, historical_query['natural_query'])
			
			if similarity > self.similarity_threshold:
				similar_queries.append({
					'query': historical_query['natural_query'],
					'sql': historical_query.get('generated_sql', ''),
					'similarity': similarity,
					'timestamp': historical_query.get('timestamp', '')
				})
		
		# Sort by similarity
		similar_queries.sort(key=lambda x: x['similarity'], reverse=True)
		
		return similar_queries[:5]  # Top 5 similar queries
	
	async def _calculate_similarity(self, query1: str, query2: str) -> float:
		"""Calculate semantic similarity between queries"""
		# Simple similarity based on common words (mock implementation)
		words1 = set(query1.lower().split())
		words2 = set(query2.lower().split())
		
		if not words1 or not words2:
			return 0.0
		
		intersection = words1.intersection(words2)
		union = words1.union(words2)
		
		return len(intersection) / len(union)


# Export NLP integration components
__all__ = [
	"APGNLPProcessor",
	"QuerySuggestionEngine", 
	"SemanticQueryMatcher"
]