"""
APG Document Content Management - Semantic Search Service

Contextual and semantic search capabilities with NLP, vector embeddings,
and intelligent query understanding for content discovery.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import (
	DCMDocument, DCMSemanticSearch, DCMSearchIndex, DCMContentIntelligence,
	ValidatedConfidenceScore
)


class SemanticSearchEngine:
	"""Contextual and semantic search engine with NLP capabilities"""
	
	def __init__(self, apg_ai_client=None, apg_rag_client=None):
		"""Initialize semantic search engine with APG AI/ML integration"""
		self.apg_ai_client = apg_ai_client
		self.apg_rag_client = apg_rag_client
		self.logger = logging.getLogger(__name__)
		
		# Search configuration
		self.max_results = 100
		self.min_similarity_threshold = 0.1
		self.query_expansion_limit = 10
		
		# TF-IDF vectorizer for fallback search
		self.tfidf_vectorizer = TfidfVectorizer(
			max_features=10000,
			stop_words='english',
			ngram_range=(1, 3),
			lowercase=True
		)
		
		# Search analytics
		self.search_stats = {
			'total_searches': 0,
			'average_response_time': 0.0,
			'average_results_count': 0.0,
			'click_through_rates': []
		}
	
	async def search_documents(
		self,
		query: str,
		user_id: str,
		tenant_id: str,
		search_options: Optional[Dict[str, Any]] = None
	) -> DCMSemanticSearch:
		"""Perform semantic search across documents"""
		start_time = time.time()
		search_options = search_options or {}
		
		try:
			# Preprocess and analyze query
			processed_query = await self._preprocess_query(query)
			intent = await self._classify_query_intent(processed_query)
			
			# Generate vector embedding for semantic search
			query_embedding = await self._generate_query_embedding(processed_query)
			
			# Expand query with semantic alternatives
			expanded_terms = await self._expand_query_semantically(processed_query)
			
			# Perform multi-modal search
			search_results = await self._execute_semantic_search(
				processed_query,
				query_embedding,
				expanded_terms,
				intent,
				search_options,
				tenant_id,
				user_id
			)
			
			# Rank and filter results
			ranked_results = await self._rank_search_results(
				search_results,
				query_embedding,
				intent,
				user_id
			)
			
			# Create search record
			search_record = DCMSemanticSearch(
				tenant_id=tenant_id,
				created_by=user_id,
				updated_by=user_id,
				query_text=query,
				user_id=user_id,
				vector_embedding=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
				semantic_expansion=expanded_terms,
				intent_classification=intent,
				result_document_ids=[r['document_id'] for r in ranked_results],
				relevance_scores=[r['relevance_score'] for r in ranked_results],
				context_matches=[r['context_match'] for r in ranked_results],
				response_time_ms=int((time.time() - start_time) * 1000),
				result_count=len(ranked_results)
			)
			
			# Update search statistics
			self._update_search_stats(search_record)
			
			self.logger.info(f"Semantic search completed: {len(ranked_results)} results in {search_record.response_time_ms}ms")
			return search_record
			
		except Exception as e:
			self.logger.error(f"Semantic search error: {str(e)}")
			
			# Return error search record
			return DCMSemanticSearch(
				tenant_id=tenant_id,
				created_by=user_id,
				updated_by=user_id,
				query_text=query,
				user_id=user_id,
				vector_embedding=[],
				semantic_expansion=[],
				intent_classification="error",
				result_document_ids=[],
				relevance_scores=[],
				context_matches=[],
				response_time_ms=int((time.time() - start_time) * 1000),
				result_count=0
			)
	
	async def _preprocess_query(self, query: str) -> str:
		"""Preprocess and clean search query"""
		# Basic text cleaning
		cleaned_query = re.sub(r'[^\w\s-]', ' ', query)
		cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
		
		# Apply APG AI text preprocessing if available
		if self.apg_ai_client:
			try:
				processed = await self.apg_ai_client.preprocess_text(cleaned_query)
				return processed
			except Exception as e:
				self.logger.warning(f"AI preprocessing failed: {str(e)}")
		
		return cleaned_query.lower()
	
	async def _classify_query_intent(self, query: str) -> str:
		"""Classify user query intent using APG AI"""
		if self.apg_ai_client:
			try:
				intent_result = await self.apg_ai_client.classify_intent(query)
				return intent_result.get('intent', 'general_search')
			except Exception as e:
				self.logger.warning(f"Intent classification failed: {str(e)}")
		
		# Fallback intent classification
		if any(word in query.lower() for word in ['who', 'what', 'when', 'where', 'how', 'why']):
			return 'question'
		elif any(word in query.lower() for word in ['find', 'search', 'look for', 'locate']):
			return 'find_document'
		elif any(word in query.lower() for word in ['show', 'list', 'display']):
			return 'browse'
		else:
			return 'general_search'
	
	async def _generate_query_embedding(self, query: str) -> Union[np.ndarray, List[float]]:
		"""Generate vector embedding for query using APG AI"""
		if self.apg_ai_client:
			try:
				embedding = await self.apg_ai_client.generate_embedding(query)
				return np.array(embedding) if isinstance(embedding, list) else embedding
			except Exception as e:
				self.logger.warning(f"Embedding generation failed: {str(e)}")
		
		# Fallback to TF-IDF if APG AI unavailable
		try:
			tfidf_vector = self.tfidf_vectorizer.fit_transform([query])
			return tfidf_vector.toarray()[0]
		except:
			# Return zero vector as last resort
			return np.zeros(300)  # Standard embedding dimension
	
	async def _expand_query_semantically(self, query: str) -> List[str]:
		"""Expand query with semantically related terms"""
		expanded_terms = []
		
		if self.apg_ai_client:
			try:
				expansion = await self.apg_ai_client.expand_query(query)
				expanded_terms.extend(expansion.get('synonyms', []))
				expanded_terms.extend(expansion.get('related_terms', []))
				expanded_terms.extend(expansion.get('hypernyms', []))
			except Exception as e:
				self.logger.warning(f"Query expansion failed: {str(e)}")
		
		# Fallback expansion using simple word associations
		expansion_map = {
			'document': ['file', 'paper', 'record', 'report'],
			'contract': ['agreement', 'deal', 'terms', 'legal'],
			'invoice': ['bill', 'payment', 'charge', 'receipt'],
			'policy': ['procedure', 'guideline', 'rule', 'standard'],
			'manual': ['guide', 'handbook', 'instructions', 'documentation']
		}
		
		query_words = query.lower().split()
		for word in query_words:
			if word in expansion_map:
				expanded_terms.extend(expansion_map[word])
		
		# Remove duplicates and limit results
		expanded_terms = list(set(expanded_terms))[:self.query_expansion_limit]
		return expanded_terms
	
	async def _execute_semantic_search(
		self,
		query: str,
		query_embedding: Union[np.ndarray, List[float]],
		expanded_terms: List[str],
		intent: str,
		search_options: Dict[str, Any],
		tenant_id: str,
		user_id: str
	) -> List[Dict[str, Any]]:
		"""Execute the actual semantic search"""
		search_results = []
		
		try:
			# Search using vector similarity if embeddings available
			if isinstance(query_embedding, np.ndarray) and query_embedding.any():
				vector_results = await self._vector_similarity_search(
					query_embedding, 
					tenant_id,
					search_options
				)
				search_results.extend(vector_results)
			
			# Full-text search with expanded terms
			text_results = await self._full_text_search(
				query,
				expanded_terms,
				tenant_id,
				search_options
			)
			search_results.extend(text_results)
			
			# Metadata-based search
			metadata_results = await self._metadata_search(
				query,
				intent,
				tenant_id,
				search_options
			)
			search_results.extend(metadata_results)
			
			# Use APG RAG for contextual search if available
			if self.apg_rag_client:
				rag_results = await self._rag_contextual_search(
					query,
					tenant_id,
					search_options
				)
				search_results.extend(rag_results)
			
			# Remove duplicates based on document_id
			unique_results = {}
			for result in search_results:
				doc_id = result['document_id']
				if doc_id not in unique_results or result['relevance_score'] > unique_results[doc_id]['relevance_score']:
					unique_results[doc_id] = result
			
			return list(unique_results.values())
			
		except Exception as e:
			self.logger.error(f"Search execution error: {str(e)}")
			return []
	
	async def _vector_similarity_search(
		self,
		query_embedding: np.ndarray,
		tenant_id: str,
		search_options: Dict[str, Any]
	) -> List[Dict[str, Any]]:
		"""Perform vector similarity search"""
		results = []
		
		# This would typically query a vector database like Chroma, Pinecone, or Weaviate
		# For now, we'll simulate the process
		
		try:
			# In a real implementation, this would:
			# 1. Query vector database for similar embeddings
			# 2. Return documents with similarity scores
			# 3. Include context matching information
			
			# Simulated vector search results
			# In production, replace with actual vector database query
			similarity_results = [
				{
					'document_id': 'doc-1',
					'similarity_score': 0.95,
					'embedding': query_embedding.tolist()
				},
				{
					'document_id': 'doc-2', 
					'similarity_score': 0.87,
					'embedding': query_embedding.tolist()
				}
			]
			
			for result in similarity_results:
				if result['similarity_score'] >= self.min_similarity_threshold:
					results.append({
						'document_id': result['document_id'],
						'relevance_score': result['similarity_score'],
						'search_type': 'vector_similarity',
						'context_match': {
							'type': 'semantic_similarity',
							'score': result['similarity_score'],
							'method': 'vector_embedding'
						}
					})
			
		except Exception as e:
			self.logger.error(f"Vector search error: {str(e)}")
		
		return results
	
	async def _full_text_search(
		self,
		query: str,
		expanded_terms: List[str],
		tenant_id: str,
		search_options: Dict[str, Any]
	) -> List[Dict[str, Any]]:
		"""Perform full-text search with query expansion"""
		results = []
		
		try:
			# Combine original query with expanded terms
			search_terms = [query] + expanded_terms
			
			# In a real implementation, this would query the search index
			# For now, we'll simulate full-text search results
			
			text_results = [
				{
					'document_id': 'doc-3',
					'match_score': 0.8,
					'matched_terms': ['query', 'term1'],
					'highlights': ['...matched text snippet...']
				},
				{
					'document_id': 'doc-4',
					'match_score': 0.7,
					'matched_terms': ['query'],
					'highlights': ['...another match...']
				}
			]
			
			for result in text_results:
				results.append({
					'document_id': result['document_id'],
					'relevance_score': result['match_score'],
					'search_type': 'full_text',
					'context_match': {
						'type': 'text_match',
						'matched_terms': result['matched_terms'],
						'highlights': result['highlights'],
						'score': result['match_score']
					}
				})
			
		except Exception as e:
			self.logger.error(f"Full-text search error: {str(e)}")
		
		return results
	
	async def _metadata_search(
		self,
		query: str,
		intent: str,
		tenant_id: str,
		search_options: Dict[str, Any]
	) -> List[Dict[str, Any]]:
		"""Search based on document metadata"""
		results = []
		
		try:
			# Extract potential metadata terms from query
			metadata_terms = self._extract_metadata_terms(query)
			
			# In a real implementation, this would search metadata fields
			# For now, simulate metadata search
			
			metadata_results = [
				{
					'document_id': 'doc-5',
					'metadata_matches': {'author': 'john doe', 'category': 'contract'},
					'match_score': 0.9
				}
			]
			
			for result in metadata_results:
				results.append({
					'document_id': result['document_id'],
					'relevance_score': result['match_score'],
					'search_type': 'metadata',
					'context_match': {
						'type': 'metadata_match',
						'matched_fields': result['metadata_matches'],
						'score': result['match_score']
					}
				})
			
		except Exception as e:
			self.logger.error(f"Metadata search error: {str(e)}")
		
		return results
	
	async def _rag_contextual_search(
		self,
		query: str,
		tenant_id: str,
		search_options: Dict[str, Any]
	) -> List[Dict[str, Any]]:
		"""Perform RAG-based contextual search using APG RAG"""
		results = []
		
		if self.apg_rag_client:
			try:
				# Use APG RAG for contextual document retrieval
				rag_response = await self.apg_rag_client.retrieve_documents(
					query=query,
					tenant_id=tenant_id,
					max_results=20,
					context_window=search_options.get('context_window', 500)
				)
				
				for doc in rag_response.get('documents', []):
					results.append({
						'document_id': doc['id'],
						'relevance_score': doc.get('score', 0.5),
						'search_type': 'rag_contextual',
						'context_match': {
							'type': 'contextual_relevance',
							'context_snippet': doc.get('context', ''),
							'score': doc.get('score', 0.5),
							'reasoning': doc.get('reasoning', '')
						}
					})
				
			except Exception as e:
				self.logger.error(f"RAG search error: {str(e)}")
		
		return results
	
	async def _rank_search_results(
		self,
		search_results: List[Dict[str, Any]],
		query_embedding: Union[np.ndarray, List[float]],
		intent: str,
		user_id: str
	) -> List[Dict[str, Any]]:
		"""Rank and re-score search results using multiple factors"""
		if not search_results:
			return []
		
		try:
			# Apply intent-based boosting
			intent_boost = self._get_intent_boost_factors(intent)
			
			# Apply personalization if available
			personalization_scores = await self._get_personalization_scores(user_id, search_results)
			
			# Calculate final scores
			for result in search_results:
				base_score = result['relevance_score']
				
				# Apply intent boost
				search_type = result.get('search_type', 'unknown')
				intent_factor = intent_boost.get(search_type, 1.0)
				
				# Apply personalization
				personalization_factor = personalization_scores.get(result['document_id'], 1.0)
				
				# Apply recency boost for time-sensitive queries
				recency_factor = 1.0  # Would calculate based on document age
				
				# Calculate final score
				final_score = base_score * intent_factor * personalization_factor * recency_factor
				result['final_relevance_score'] = min(final_score, 1.0)
			
			# Sort by final score
			ranked_results = sorted(
				search_results,
				key=lambda x: x['final_relevance_score'],
				reverse=True
			)
			
			# Limit results
			return ranked_results[:self.max_results]
			
		except Exception as e:
			self.logger.error(f"Result ranking error: {str(e)}")
			return search_results[:self.max_results]
	
	def _extract_metadata_terms(self, query: str) -> Dict[str, List[str]]:
		"""Extract potential metadata search terms from query"""
		metadata_terms = {
			'authors': [],
			'categories': [],
			'dates': [],
			'types': []
		}
		
		# Simple pattern matching for metadata extraction
		author_patterns = [r'by (\w+)', r'author:(\w+)', r'written by (\w+)']
		for pattern in author_patterns:
			matches = re.findall(pattern, query, re.IGNORECASE)
			metadata_terms['authors'].extend(matches)
		
		# Category patterns
		category_patterns = [r'category:(\w+)', r'type:(\w+)']
		for pattern in category_patterns:
			matches = re.findall(pattern, query, re.IGNORECASE)
			metadata_terms['categories'].extend(matches)
		
		return metadata_terms
	
	def _get_intent_boost_factors(self, intent: str) -> Dict[str, float]:
		"""Get boosting factors based on query intent"""
		boost_factors = {
			'question': {
				'rag_contextual': 1.3,
				'full_text': 1.2,
				'vector_similarity': 1.1,
				'metadata': 0.9
			},
			'find_document': {
				'metadata': 1.3,
				'full_text': 1.2,
				'vector_similarity': 1.0,
				'rag_contextual': 0.9
			},
			'browse': {
				'metadata': 1.4,
				'full_text': 1.0,
				'vector_similarity': 0.9,
				'rag_contextual': 0.8
			},
			'general_search': {
				'vector_similarity': 1.2,
				'full_text': 1.1,
				'rag_contextual': 1.0,
				'metadata': 0.9
			}
		}
		
		return boost_factors.get(intent, boost_factors['general_search'])
	
	async def _get_personalization_scores(
		self, 
		user_id: str, 
		search_results: List[Dict[str, Any]]
	) -> Dict[str, float]:
		"""Get personalization scores based on user behavior"""
		personalization_scores = {}
		
		try:
			# In a real implementation, this would:
			# 1. Analyze user's past interactions with documents
			# 2. Consider user's role and permissions
			# 3. Factor in collaborative filtering
			# 4. Apply content preferences
			
			# For now, return neutral scores
			for result in search_results:
				personalization_scores[result['document_id']] = 1.0
			
		except Exception as e:
			self.logger.error(f"Personalization scoring error: {str(e)}")
		
		return personalization_scores
	
	def _update_search_stats(self, search_record: DCMSemanticSearch):
		"""Update search performance statistics"""
		self.search_stats['total_searches'] += 1
		
		# Update average response time
		total_time = (self.search_stats['average_response_time'] * 
					 (self.search_stats['total_searches'] - 1) + 
					 search_record.response_time_ms)
		self.search_stats['average_response_time'] = total_time / self.search_stats['total_searches']
		
		# Update average results count
		total_results = (self.search_stats['average_results_count'] * 
						(self.search_stats['total_searches'] - 1) + 
						search_record.result_count)
		self.search_stats['average_results_count'] = total_results / self.search_stats['total_searches']
	
	async def record_click_through(self, search_id: str, document_id: str, position: int):
		"""Record click-through for search analytics"""
		try:
			# This would update the search record with click-through data
			# and be used to improve ranking algorithms
			self.logger.info(f"Click-through recorded: search={search_id}, doc={document_id}, pos={position}")
		except Exception as e:
			self.logger.error(f"Click-through recording error: {str(e)}")
	
	async def record_search_satisfaction(self, search_id: str, rating: int):
		"""Record user satisfaction with search results"""
		try:
			# This would update the search record with satisfaction rating
			# and be used to improve search quality
			self.logger.info(f"Search satisfaction recorded: search={search_id}, rating={rating}")
		except Exception as e:
			self.logger.error(f"Satisfaction recording error: {str(e)}")
	
	async def get_search_analytics(self) -> Dict[str, Any]:
		"""Get search performance analytics"""
		return {
			"total_searches": self.search_stats['total_searches'],
			"average_response_time_ms": self.search_stats['average_response_time'],
			"average_results_count": self.search_stats['average_results_count'],
			"click_through_rate": (
				sum(self.search_stats['click_through_rates']) / 
				len(self.search_stats['click_through_rates'])
			) if self.search_stats['click_through_rates'] else 0.0
		}
	
	async def suggest_query_completions(self, partial_query: str, user_id: str) -> List[str]:
		"""Suggest query completions based on partial input"""
		suggestions = []
		
		try:
			if self.apg_ai_client:
				# Use APG AI for intelligent suggestions
				ai_suggestions = await self.apg_ai_client.suggest_completions(partial_query)
				suggestions.extend(ai_suggestions.get('suggestions', []))
			
			# Add fallback suggestions based on common patterns
			common_completions = [
				f"{partial_query} document",
				f"{partial_query} file",
				f"{partial_query} report",
				f"{partial_query} policy"
			]
			
			suggestions.extend(common_completions)
			
			# Remove duplicates and limit
			suggestions = list(set(suggestions))[:5]
			
		except Exception as e:
			self.logger.error(f"Query suggestion error: {str(e)}")
		
		return suggestions