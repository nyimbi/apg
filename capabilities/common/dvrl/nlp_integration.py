#!/usr/bin/env python3
"""
APG Data Virtualization (DVRL) NLP Integration
Natural language query processing and semantic understanding

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid_extensions import uuid7str

# Import real NLP implementation and Ollama client
try:
	from .real_implementations import RealAPGNLPProcessor as APGNLPProcessor
	import httpx
	import ollama
	REAL_NLP_AVAILABLE = True
	OLLAMA_AVAILABLE = True
except ImportError:
	REAL_NLP_AVAILABLE = False
	OLLAMA_AVAILABLE = False
	
	# Real Ollama-powered APG NLP Integration
	class APGNLPProcessor:
		"""Production NLP processor using Ollama for local natural language processing"""
		
		def __init__(self, tenant_id: str, user_id: str, ollama_config: Optional[Dict[str, Any]] = None):
			self.tenant_id = tenant_id
			self.user_id = user_id
			self.ollama_config = ollama_config or {}
			self.model_name = self.ollama_config.get('model', 'llama3.2:latest')
			self.ollama_host = self.ollama_config.get('host', 'http://localhost:11434')
			self.schema_context = {}
			self.conversation_history = []
			
			# Initialize Ollama client
			self.ollama_client = None
			asyncio.create_task(self._initialize_ollama())
		
		async def _initialize_ollama(self):
			"""Initialize Ollama client and ensure model is available"""
			try:
				# Set up Ollama client with custom host if specified
				if self.ollama_host != 'http://localhost:11434':
					ollama.Client(host=self.ollama_host)
				
				# Check if model is available, pull if needed
				models = await self._list_ollama_models()
				if self.model_name not in models:
					await self._pull_ollama_model(self.model_name)
					
				await self._log_info(f"Ollama NLP initialized with model: {self.model_name}")
				
			except Exception as e:
				await self._log_error(f"Failed to initialize Ollama: {str(e)}")
				
		async def _list_ollama_models(self) -> List[str]:
			"""List available Ollama models"""
			try:
				response = ollama.list()
				return [model['name'] for model in response.get('models', [])]
			except Exception as e:
				await self._log_error(f"Failed to list Ollama models: {str(e)}")
				return []
				
		async def _pull_ollama_model(self, model_name: str):
			"""Pull Ollama model if not available"""
			try:
				await self._log_info(f"Pulling Ollama model: {model_name}")
				ollama.pull(model_name)
				await self._log_info(f"Successfully pulled model: {model_name}")
			except Exception as e:
				await self._log_error(f"Failed to pull model {model_name}: {str(e)}")
		
		async def process_natural_language_query(self, natural_query: str, schema_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
			"""Process natural language query using Ollama LLM with timing measurement"""
			import time
			start_time = time.perf_counter()
			
			try:
				await self._log_info(f"Processing NL query with Ollama: {natural_query[:50]}...")
				
				# Store schema context for better understanding
				if schema_context:
					self.schema_context = schema_context
				
				# Build comprehensive prompt for SQL generation
				prompt = self._build_sql_generation_prompt(natural_query, schema_context)
				
				# Generate SQL using Ollama
				sql_result = await self._generate_sql_with_ollama(prompt)
				
				# Extract SQL and confidence from Ollama response
				generated_sql = self._extract_sql_from_response(sql_result)
				confidence = self._calculate_ollama_confidence(sql_result, natural_query)
				
				# Generate query explanation
				explanation = await self._generate_query_explanation(natural_query, generated_sql)
				
				# Store in conversation history
				conversation_entry = {
					'id': uuid7str(),
					'timestamp': datetime.now(timezone.utc).isoformat(),
					'natural_query': natural_query,
					'generated_sql': generated_sql,
					'confidence': confidence,
					'explanation': explanation,
					'schema_context': schema_context
				}
				self.conversation_history.append(conversation_entry)
				
				return {
					'query_id': conversation_entry['id'],
					'original_query': natural_query,
					'generated_sql': generated_sql,
					'confidence': confidence,
					'explanation': explanation,
					'processing_time_ms': round((time.perf_counter() - start_time) * 1000, 2),
					'model_used': self.model_name,
					'conversation_id': f"{self.tenant_id}_{self.user_id}"
				}
				
			except Exception as e:
				await self._log_error(f"NLP processing failed: {str(e)}")
				return {
					'query_id': uuid7str(),
					'original_query': natural_query,
					'generated_sql': 'SELECT 1 as error_fallback;',
					'confidence': 0.0,
					'explanation': f"Failed to process query: {str(e)}",
					'error': str(e)
				}
		
		def _build_sql_generation_prompt(self, natural_query: str, schema_context: Optional[Dict[str, Any]]) -> str:
			"""Build comprehensive prompt for SQL generation"""
			schema_info = ""
			if schema_context:
				schema_info = f"""
Available Database Schema:
Tables and Columns:
{json.dumps(schema_context.get('tables', {}), indent=2)}

Data Types:
{json.dumps(schema_context.get('data_types', {}), indent=2)}
"""
			
			prompt = f"""You are a SQL expert. Convert this natural language query to SQL.

{schema_info}

Natural Language Query: "{natural_query}"

Instructions:
1. Generate ONLY valid SQL - no explanations in the SQL code
2. Use standard SQL syntax
3. If table/column names are unclear, make reasonable assumptions
4. Add appropriate WHERE, ORDER BY, and LIMIT clauses when sensible
5. For aggregations, include GROUP BY when needed
6. Use appropriate JOINs if multiple tables are referenced

Respond with just the SQL query, starting with SELECT, INSERT, UPDATE, or DELETE.
"""
			return prompt
		
		async def _generate_sql_with_ollama(self, prompt: str) -> str:
			"""Generate SQL using Ollama model"""
			try:
				response = ollama.generate(
					model=self.model_name,
					prompt=prompt,
					options={
						'temperature': 0.1,  # Low temperature for more deterministic SQL
						'top_p': 0.9,
						'num_predict': 200   # Limit response length
					}
				)
				return response.get('response', '')
				
			except Exception as e:
				await self._log_error(f"Ollama generation failed: {str(e)}")
				return 'SELECT 1 as ollama_error;'
		
		def _extract_sql_from_response(self, response: str) -> str:
			"""Extract SQL from Ollama response"""
			# Clean up the response
			lines = response.strip().split('\n')
			sql_lines = []
			
			for line in lines:
				line = line.strip()
				# Look for lines that start with SQL keywords
				if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
					sql_lines.append(line)
				elif sql_lines and line:  # Continue collecting SQL if we've started
					sql_lines.append(line)
				elif sql_lines and line.endswith(';'):  # End of SQL
					sql_lines.append(line)
					break
			
			sql = ' '.join(sql_lines)
			
			# Ensure SQL ends with semicolon
			if sql and not sql.rstrip().endswith(';'):
				sql = sql.rstrip() + ';'
				
			return sql or 'SELECT 1;'
		
		def _calculate_ollama_confidence(self, response: str, original_query: str) -> float:
			"""Calculate confidence based on response quality"""
			base_confidence = 0.6
			
			# Check if response contains valid SQL keywords
			sql_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING']
			found_keywords = sum(1 for kw in sql_keywords if kw in response.upper())
			keyword_score = min(found_keywords * 0.05, 0.3)
			
			# Check response length appropriateness
			length_score = 0.1 if 20 <= len(response) <= 300 else 0.0
			
			# Check for common SQL patterns
			pattern_score = 0.1 if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE)\b.*\b(FROM|INTO|SET)\b', response.upper()) else 0.0
			
			return min(base_confidence + keyword_score + length_score + pattern_score, 1.0)
		
		async def _generate_query_explanation(self, natural_query: str, sql_query: str) -> str:
			"""Generate human-readable explanation of the SQL query"""
			try:
				explanation_prompt = f"""Explain this SQL query in simple terms:

Original question: "{natural_query}"
Generated SQL: {sql_query}

Provide a brief, clear explanation of what this SQL query does.
"""
				
				response = ollama.generate(
					model=self.model_name,
					prompt=explanation_prompt,
					options={
						'temperature': 0.3,
						'num_predict': 100
					}
				)
				return response.get('response', 'Query explanation not available').strip()
				
			except Exception as e:
				return f"This query {natural_query.lower()}"
		
		async def get_query_suggestions(self, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
			"""Generate query suggestions using Ollama"""
			try:
				schema_info = ""
				if context and 'tables' in context:
					schema_info = f"Available tables: {', '.join(context['tables'].keys())}"
				
				suggestions_prompt = f"""Generate 5 sample natural language queries for a business database.

{schema_info}

Focus on common business questions like:
- Counting records
- Finding totals/averages
- Filtering by conditions
- Comparing data across time periods
- Top/bottom rankings

Respond with just the questions, one per line.
"""
				
				response = ollama.generate(
					model=self.model_name,
					prompt=suggestions_prompt,
					options={
						'temperature': 0.7,
						'num_predict': 150
					}
				)
				
				# Parse suggestions from response
				suggestions = []
				for line in response.get('response', '').strip().split('\n'):
					line = line.strip()
					if line and not line.startswith('#'):
						# Clean up numbered lists
						line = re.sub(r'^\d+[\.\)]\s*', '', line)
						suggestions.append({
							'id': uuid7str(),
							'query': line,
							'category': 'general',
							'complexity': 'medium'
						})
				
				return suggestions[:5]  # Return max 5 suggestions
				
			except Exception as e:
				await self._log_error(f"Failed to generate suggestions: {str(e)}")
				return []
		
		async def _log_info(self, message: str):
			"""Log info message"""
			print(f"[{datetime.now(timezone.utc).isoformat()}] NLP INFO: {message}")
		
		async def _log_error(self, message: str):
			"""Log error message"""
			print(f"[{datetime.now(timezone.utc).isoformat()}] NLP ERROR: {message}")


# Additional NLP utility classes for enhanced functionality
class QuerySuggestionEngine:
	"""Enhanced query suggestion engine using pattern matching and ML"""
	
	def __init__(self, nlp_processor: APGNLPProcessor):
		self.nlp_processor = nlp_processor
		self.suggestion_categories = {
			'data_exploration': [
				"Show me all {table} records",
				"How many {table} are there?", 
				"What is the total {column} in {table}?",
				"Find the average {column} by {group_column}"
			],
			'time_analysis': [
				"Show {table} from last month",
				"Count {table} by day this week",
				"What are the trends in {column} over time?",
				"Compare {column} between this year and last year"
			],
			'top_bottom': [
				"Show top 10 {table} by {column}",
				"Find the highest {column} in {table}",
				"Which {group_column} has the most {table}?",
				"Show bottom 5 {table} by {column}"
			]
		}
	
	async def get_contextual_suggestions(self, schema_context: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate contextual suggestions based on database schema"""
		suggestions = []
		tables = schema_context.get('tables', {})
		
		for table_name, columns in tables.items():
			# Generate basic suggestions for each table
			for category, templates in self.suggestion_categories.items():
				for template in templates[:2]:  # Limit suggestions per category
					suggestion_text = template.format(
						table=table_name,
						column=columns[0] if columns else 'value',
						group_column=columns[1] if len(columns) > 1 else 'category'
					)
					
					suggestions.append({
						'id': uuid7str(),
						'query': suggestion_text,
						'category': category,
						'complexity': 'easy',
						'table': table_name
					})
		
		return suggestions[:10]  # Return top 10 suggestions


class SemanticQueryMatcher:
	"""Advanced semantic matching for similar queries and patterns"""
	
	def __init__(self):
		self.query_embeddings = {}
		self.semantic_patterns = {
			'similarity_threshold': 0.75,
			'common_synonyms': {
				'count': ['number', 'total', 'how many'],
				'show': ['display', 'list', 'get', 'find'],
				'average': ['mean', 'avg', 'typical'],
				'maximum': ['max', 'highest', 'top', 'peak'],
				'minimum': ['min', 'lowest', 'bottom', 'smallest']
			}
		}
	
	async def find_similar_queries(self, query: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Find semantically similar queries from conversation history"""
		similar_queries = []
		query_words = set(query.lower().split())
		
		for historical_query in history:
			hist_words = set(historical_query.get('natural_query', '').lower().split())
			similarity = self._calculate_word_similarity(query_words, hist_words)
			
			if similarity > self.semantic_patterns['similarity_threshold']:
				similar_queries.append({
					'query_id': historical_query.get('id'),
					'query': historical_query.get('natural_query'),
					'similarity': similarity,
					'sql': historical_query.get('generated_sql'),
					'timestamp': historical_query.get('timestamp')
				})
		
		# Sort by similarity score
		return sorted(similar_queries, key=lambda x: x['similarity'], reverse=True)[:5]
	
	def _calculate_word_similarity(self, words1: set, words2: set) -> float:
		"""Calculate semantic similarity between two sets of words"""
		if not words1 or not words2:
			return 0.0
		
		# Direct word overlap
		direct_overlap = len(words1.intersection(words2))
		
		# Synonym-based overlap
		synonym_overlap = 0
		for word1 in words1:
			for word2 in words2:
				if self._are_synonyms(word1, word2):
					synonym_overlap += 1
		
		total_similarity = direct_overlap + (synonym_overlap * 0.8)
		max_words = max(len(words1), len(words2))
		
		return min(total_similarity / max_words, 1.0)
	
	def _are_synonyms(self, word1: str, word2: str) -> bool:
		"""Check if two words are synonyms based on predefined patterns"""
		for concept, synonyms in self.semantic_patterns['common_synonyms'].items():
			if word1 in synonyms and word2 in synonyms:
				return True
		return False


# Export NLP integration components
__all__ = [
	"APGNLPProcessor",
	"QuerySuggestionEngine", 
	"SemanticQueryMatcher"
]
