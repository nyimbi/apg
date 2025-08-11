#!/usr/bin/env python3
"""
Unit Tests for NLP Integration
Tests for Ollama-powered natural language processing

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from capabilities.common.dvrl.nlp_integration import APGNLPProcessor, QuerySuggestionEngine, SemanticQueryMatcher


class TestAPGNLPProcessor:
	"""Test suite for APG NLP Processor with Ollama integration"""
	
	@pytest.fixture
	async def nlp_processor(self):
		"""Create NLP processor instance for testing"""
		config = {
			'model': 'llama3.2:latest',
			'host': 'http://localhost:11434'
		}
		processor = APGNLPProcessor('test_tenant', 'test_user', config)
		return processor
	
	@patch('ollama.list')
	@patch('ollama.pull')
	async def test_initialize_ollama_model_available(self, mock_pull, mock_list, nlp_processor):
		"""Test Ollama initialization when model is available"""
		mock_list.return_value = {'models': [{'name': 'llama3.2:latest'}]}
		
		await nlp_processor._initialize_ollama()
		
		mock_list.assert_called_once()
		mock_pull.assert_not_called()  # Should not pull if model exists
	
	@patch('ollama.list')
	@patch('ollama.pull')
	async def test_initialize_ollama_model_missing(self, mock_pull, mock_list, nlp_processor):
		"""Test Ollama initialization when model needs to be pulled"""
		mock_list.return_value = {'models': []}
		
		await nlp_processor._initialize_ollama()
		
		mock_list.assert_called_once()
		mock_pull.assert_called_once_with('llama3.2:latest')
	
	@patch('ollama.generate')
	async def test_generate_sql_with_ollama_success(self, mock_generate, nlp_processor):
		"""Test successful SQL generation using Ollama"""
		mock_generate.return_value = {
			'response': 'SELECT COUNT(*) FROM users WHERE created_at > CURRENT_DATE - INTERVAL 7 DAY;'
		}
		
		prompt = "Count users created in the last week"
		result = await nlp_processor._generate_sql_with_ollama(prompt)
		
		assert 'SELECT COUNT(*)' in result
		assert 'users' in result
		assert 'INTERVAL 7 DAY' in result
		mock_generate.assert_called_once()
	
	@patch('ollama.generate')
	async def test_generate_sql_with_ollama_error(self, mock_generate, nlp_processor):
		"""Test SQL generation error handling"""
		mock_generate.side_effect = Exception("Ollama connection failed")
		
		prompt = "Count users"
		result = await nlp_processor._generate_sql_with_ollama(prompt)
		
		assert result == 'SELECT 1 as ollama_error;'
	
	def test_extract_sql_from_response_valid(self, nlp_processor):
		"""Test SQL extraction from Ollama response"""
		response = """
		Here's the SQL query you requested:
		
		SELECT name, COUNT(*) as total 
		FROM products 
		WHERE category = 'electronics' 
		GROUP BY name 
		ORDER BY total DESC;
		
		This query will show you the product counts by name.
		"""
		
		result = nlp_processor._extract_sql_from_response(response)
		
		assert result.startswith('SELECT')
		assert 'FROM products' in result
		assert result.endswith(';')
	
	def test_extract_sql_from_response_no_sql(self, nlp_processor):
		"""Test SQL extraction when no SQL is found"""
		response = "I don't understand your request."
		
		result = nlp_processor._extract_sql_from_response(response)
		
		assert result == 'SELECT 1;'
	
	def test_calculate_ollama_confidence_high(self, nlp_processor):
		"""Test confidence calculation for high-quality response"""
		response = "SELECT COUNT(*) FROM users WHERE created_at > '2024-01-01'"
		original_query = "Count users created after 2024"
		
		confidence = nlp_processor._calculate_ollama_confidence(response, original_query)
		
		assert confidence > 0.8
	
	def test_calculate_ollama_confidence_low(self, nlp_processor):
		"""Test confidence calculation for low-quality response"""
		response = "Error generating SQL"
		original_query = "Count users"
		
		confidence = nlp_processor._calculate_ollama_confidence(response, original_query)
		
		assert confidence < 0.7
	
	@patch('ollama.generate')
	async def test_process_natural_language_query_complete(self, mock_generate, nlp_processor):
		"""Test complete natural language query processing"""
		# Mock SQL generation
		mock_generate.side_effect = [
			{'response': 'SELECT COUNT(*) FROM users;'},  # SQL generation
			{'response': 'This query counts all users in the database'}  # Explanation
		]
		
		schema_context = {
			'tables': {
				'users': ['id', 'name', 'email', 'created_at']
			}
		}
		
		result = await nlp_processor.process_natural_language_query(
			"How many users are there?",
			schema_context
		)
		
		assert result['original_query'] == "How many users are there?"
		assert 'SELECT COUNT(*)' in result['generated_sql']
		assert result['confidence'] > 0
		assert 'query_id' in result
		assert result['model_used'] == 'llama3.2:latest'
	
	@patch('ollama.generate')
	async def test_get_query_suggestions(self, mock_generate, nlp_processor):
		"""Test query suggestions generation"""
		mock_generate.return_value = {
			'response': """How many customers do we have?
What is the total revenue this month?
Show me the top 10 products by sales
Which region has the highest sales?
What is the average order value?"""
		}
		
		context = {
			'tables': {
				'customers': ['id', 'name'],
				'orders': ['id', 'total', 'customer_id']
			}
		}
		
		suggestions = await nlp_processor.get_query_suggestions(context)
		
		assert len(suggestions) <= 5
		assert all('query' in suggestion for suggestion in suggestions)
		assert all('id' in suggestion for suggestion in suggestions)


class TestQuerySuggestionEngine:
	"""Test suite for Query Suggestion Engine"""
	
	@pytest.fixture
	def suggestion_engine(self):
		"""Create suggestion engine for testing"""
		mock_processor = Mock()
		return QuerySuggestionEngine(mock_processor)
	
	async def test_get_contextual_suggestions(self, suggestion_engine):
		"""Test contextual suggestion generation"""
		schema_context = {
			'tables': {
				'users': ['id', 'name', 'email'],
				'orders': ['id', 'user_id', 'total', 'created_at']
			}
		}
		
		suggestions = await suggestion_engine.get_contextual_suggestions(schema_context)
		
		assert len(suggestions) > 0
		assert len(suggestions) <= 10
		
		# Check structure of suggestions
		for suggestion in suggestions:
			assert 'id' in suggestion
			assert 'query' in suggestion
			assert 'category' in suggestion
			assert 'complexity' in suggestion
			assert 'table' in suggestion
		
		# Check that suggestions reference actual tables
		suggestion_texts = [s['query'] for s in suggestions]
		assert any('users' in text for text in suggestion_texts)
		assert any('orders' in text for text in suggestion_texts)


class TestSemanticQueryMatcher:
	"""Test suite for Semantic Query Matcher"""
	
	@pytest.fixture
	def query_matcher(self):
		"""Create query matcher for testing"""
		return SemanticQueryMatcher()
	
	def test_calculate_word_similarity_identical(self, query_matcher):
		"""Test similarity calculation for identical queries"""
		words1 = {'count', 'users', 'total'}
		words2 = {'count', 'users', 'total'}
		
		similarity = query_matcher._calculate_word_similarity(words1, words2)
		
		assert similarity == 1.0
	
	def test_calculate_word_similarity_no_overlap(self, query_matcher):
		"""Test similarity calculation for completely different queries"""
		words1 = {'count', 'users'}
		words2 = {'show', 'products'}
		
		similarity = query_matcher._calculate_word_similarity(words1, words2)
		
		assert similarity == 0.0
	
	def test_calculate_word_similarity_partial(self, query_matcher):
		"""Test similarity calculation for partially overlapping queries"""
		words1 = {'count', 'users', 'today'}
		words2 = {'count', 'orders', 'today'}
		
		similarity = query_matcher._calculate_word_similarity(words1, words2)
		
		assert 0.0 < similarity < 1.0
	
	def test_are_synonyms_true(self, query_matcher):
		"""Test synonym detection for actual synonyms"""
		assert query_matcher._are_synonyms('count', 'number') == True
		assert query_matcher._are_synonyms('show', 'display') == True
		assert query_matcher._are_synonyms('maximum', 'highest') == True
	
	def test_are_synonyms_false(self, query_matcher):
		"""Test synonym detection for non-synonyms"""
		assert query_matcher._are_synonyms('count', 'show') == False
		assert query_matcher._are_synonyms('user', 'product') == False
	
	async def test_find_similar_queries(self, query_matcher):
		"""Test finding similar queries in history"""
		history = [
			{
				'id': 'q1',
				'natural_query': 'count total users',
				'generated_sql': 'SELECT COUNT(*) FROM users',
				'timestamp': '2025-01-01T10:00:00Z'
			},
			{
				'id': 'q2', 
				'natural_query': 'show me all products',
				'generated_sql': 'SELECT * FROM products',
				'timestamp': '2025-01-01T11:00:00Z'
			},
			{
				'id': 'q3',
				'natural_query': 'how many users do we have',
				'generated_sql': 'SELECT COUNT(*) FROM users',
				'timestamp': '2025-01-01T12:00:00Z'
			}
		]
		
		query = 'count number of users'
		similar_queries = await query_matcher.find_similar_queries(query, history)
		
		# Should find q1 and q3 as similar (both about counting users)
		assert len(similar_queries) >= 1
		similar_ids = [q['query_id'] for q in similar_queries]
		assert 'q1' in similar_ids or 'q3' in similar_ids
		
		# Check similarity scores
		for similar_query in similar_queries:
			assert similar_query['similarity'] > 0.75
			assert 'query_id' in similar_query
			assert 'similarity' in similar_query


# Integration test combining all components
class TestNLPIntegrationComplete:
	"""Integration tests for complete NLP workflow"""
	
	@patch('ollama.generate')
	@patch('ollama.list')
	async def test_complete_nlp_workflow(self, mock_list, mock_generate):
		"""Test complete NLP workflow from query to response"""
		# Setup mocks
		mock_list.return_value = {'models': [{'name': 'llama3.2:latest'}]}
		mock_generate.side_effect = [
			{'response': 'SELECT name, COUNT(*) FROM users GROUP BY name ORDER BY COUNT(*) DESC LIMIT 10;'},
			{'response': 'This query shows the top 10 most common user names'}
		]
		
		# Initialize processor
		processor = APGNLPProcessor('test_tenant', 'test_user')
		await processor._initialize_ollama()
		
		# Test query processing
		schema_context = {
			'tables': {
				'users': ['id', 'name', 'email', 'created_at']
			}
		}
		
		result = await processor.process_natural_language_query(
			"Show me the most popular user names",
			schema_context
		)
		
		# Verify complete workflow
		assert result['success'] is not False  # Should not have explicit failure
		assert 'SELECT' in result['generated_sql']
		assert 'users' in result['generated_sql']
		assert result['confidence'] > 0
		assert len(processor.conversation_history) == 1
	
	async def test_suggestion_and_matching_integration(self):
		"""Test integration between suggestion engine and query matcher"""
		mock_processor = Mock()
		suggestion_engine = QuerySuggestionEngine(mock_processor)
		query_matcher = SemanticQueryMatcher()
		
		# Generate suggestions
		schema_context = {
			'tables': {
				'sales': ['id', 'amount', 'date'],
				'customers': ['id', 'name', 'region']
			}
		}
		
		suggestions = await suggestion_engine.get_contextual_suggestions(schema_context)
		
		# Test finding similar suggestions
		if len(suggestions) >= 2:
			query = suggestions[0]['query']
			# Create fake history from other suggestions
			history = [
				{
					'id': s['id'],
					'natural_query': s['query'],
					'generated_sql': f"SELECT * FROM {s['table']}",
					'timestamp': '2025-01-01T10:00:00Z'
				}
				for s in suggestions[1:]
			]
			
			similar = await query_matcher.find_similar_queries(query, history)
			
			# Should be able to process without errors
			assert isinstance(similar, list)


if __name__ == '__main__':
	pytest.main([__file__])