"""
Performance and accuracy tests for NLPC capability.

Tests throughput, latency, memory usage, accuracy benchmarks,
and scalability following APG performance testing standards.
"""

import pytest
import asyncio
import time
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from uuid_extensions import uuid7str
from unittest.mock import patch
import json

from ...service import NLPCService
from ...models import (
	NLPDocument, ProcessingRequest, ProcessingResult, ContextSession,
	NLPTask, ProcessingStatus, LanguageCode, PriorityLevel
)
from ..conftest import (
	assert_processing_result_valid, create_test_document, create_test_request
)

class TestLatencyPerformance:
	"""Test processing latency across different scenarios"""
	
	async def test_single_document_latency(self, nlpc_service, performance_test_data):
		"""Test latency for single document processing"""
		test_cases = [
			('small', performance_test_data['small_text']),
			('medium', performance_test_data['medium_text']),
			('large', performance_test_data['large_text'])
		]
		
		latency_results = {}
		
		for size, text in test_cases:
			document = create_test_document(text)
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
			
			# Measure latency
			start_time = time.perf_counter()
			result = await nlpc_service.secure_process_document(
				document, request, security_context
			)
			end_time = time.perf_counter()
			
			latency_ms = (end_time - start_time) * 1000
			latency_results[size] = latency_ms
			
			# Verify result quality
			assert_processing_result_valid(result)
			assert result.status == ProcessingStatus.COMPLETED
		
		# Performance assertions
		expected_latencies = performance_test_data['expected_latencies']
		
		assert latency_results['small'] < expected_latencies['small']
		assert latency_results['medium'] < expected_latencies['medium']
		assert latency_results['large'] < expected_latencies['large']
		
		# Latency should scale reasonably with document size
		assert latency_results['small'] < latency_results['medium']
		assert latency_results['medium'] < latency_results['large']
	
	async def test_task_specific_latency(self, nlpc_service, performance_test_data):
		"""Test latency for different NLP tasks"""
		text = performance_test_data['medium_text']
		document = create_test_document(text)
		security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
		
		tasks_to_test = [
			NLPTask.LANGUAGE_DETECTION,
			NLPTask.TOKENIZATION,
			NLPTask.SENTIMENT_ANALYSIS,
			NLPTask.NAMED_ENTITY_RECOGNITION,
			NLPTask.TEXT_CLASSIFICATION,
			NLPTask.KEYWORD_EXTRACTION
		]
		
		task_latencies = {}
		
		for task in tasks_to_test:
			request = create_test_request([task])
			
			start_time = time.perf_counter()
			result = await nlpc_service.secure_process_document(
				document, request, security_context
			)
			end_time = time.perf_counter()
			
			latency_ms = (end_time - start_time) * 1000
			task_latencies[task.value] = latency_ms
			
			assert_processing_result_valid(result)
		
		# Language detection should be fastest
		assert task_latencies['language_detection'] < 100  # < 100ms
		
		# Tokenization should be very fast
		assert task_latencies['tokenization'] < 50  # < 50ms
		
		# Complex tasks should still be reasonable
		assert task_latencies['named_entity_recognition'] < 500  # < 500ms
		assert task_latencies['text_classification'] < 300  # < 300ms
	
	async def test_concurrent_processing_latency(self, nlpc_service, performance_test_data):
		"""Test latency under concurrent processing load"""
		text = performance_test_data['medium_text']
		security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
		
		# Test with different concurrency levels
		concurrency_levels = [1, 5, 10, 20]
		latency_by_concurrency = {}
		
		for num_concurrent in concurrency_levels:
			# Create concurrent tasks
			tasks = []
			for i in range(num_concurrent):
				document = create_test_document(f"{text} - Document {i}")
				request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
				
				task = nlpc_service.secure_process_document(
					document, request, security_context
				)
				tasks.append(task)
			
			# Measure concurrent execution time
			start_time = time.perf_counter()
			results = await asyncio.gather(*tasks, return_exceptions=True)
			end_time = time.perf_counter()
			
			total_time_ms = (end_time - start_time) * 1000
			avg_latency_ms = total_time_ms / num_concurrent
			
			latency_by_concurrency[num_concurrent] = avg_latency_ms
			
			# Verify all completed successfully
			successful_results = [r for r in results if not isinstance(r, Exception)]
			assert len(successful_results) == num_concurrent
		
		# Latency should not degrade dramatically with concurrency
		single_latency = latency_by_concurrency[1]
		concurrent_latency = latency_by_concurrency[10]
		
		# Allow up to 3x degradation for 10x concurrency
		assert concurrent_latency < single_latency * 3

class TestThroughputPerformance:
	"""Test processing throughput and scalability"""
	
	async def test_documents_per_second_throughput(self, nlpc_service, performance_test_data):
		"""Test documents processed per second throughput"""
		text = performance_test_data['medium_text']
		security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
		
		num_documents = 50
		documents = []
		requests = []
		
		# Prepare documents and requests
		for i in range(num_documents):
			document = create_test_document(f"{text} - Throughput test {i}")
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			documents.append(document)
			requests.append(request)
		
		# Process in parallel batches
		batch_size = 10
		start_time = time.perf_counter()
		
		all_results = []
		for i in range(0, num_documents, batch_size):
			batch_docs = documents[i:i+batch_size]
			batch_requests = requests[i:i+batch_size]
			
			batch_tasks = []
			for doc, req in zip(batch_docs, batch_requests):
				task = nlpc_service.secure_process_document(doc, req, security_context)
				batch_tasks.append(task)
			
			batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
			all_results.extend(batch_results)
		
		end_time = time.perf_counter()
		
		# Calculate throughput
		total_time_seconds = end_time - start_time
		documents_per_second = num_documents / total_time_seconds
		
		# Verify results
		successful_results = [r for r in all_results if not isinstance(r, Exception)]
		success_rate = len(successful_results) / num_documents
		
		assert success_rate > 0.95  # 95% success rate
		assert documents_per_second > 5  # Minimum 5 documents/second
		
		# Log performance metrics
		print(f"Processed {num_documents} documents in {total_time_seconds:.2f}s")
		print(f"Throughput: {documents_per_second:.2f} documents/second")
		print(f"Success rate: {success_rate * 100:.1f}%")
	
	async def test_batch_processing_throughput(self, nlpc_service, performance_test_data):
		"""Test batch processing throughput optimization"""
		texts = [
			performance_test_data['small_text'],
			performance_test_data['medium_text']
		] * 25  # 50 documents total
		
		documents = [create_test_document(text) for text in texts]
		security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
		
		pipeline_config = {
			'tasks': [NLPTask.SENTIMENT_ANALYSIS],
			'priority': PriorityLevel.MEDIUM,
			'options': {'batch_optimization': True},
			'parallel_processing': True,
			'max_workers': 5
		}
		
		# Measure batch processing performance
		start_time = time.perf_counter()
		results = await nlpc_service.orchestrate_nlp_pipeline(
			documents, pipeline_config, security_context
		)
		end_time = time.perf_counter()
		
		# Calculate metrics
		total_time = end_time - start_time
		throughput = len(documents) / total_time
		
		assert len(results) == len(documents)
		assert throughput > 8  # Should be faster than individual processing
		
		# Verify batch optimization benefits
		successful_results = [r for r in results if r.get('status') == 'completed']
		assert len(successful_results) == len(documents)
	
	async def test_streaming_throughput(self, nlpc_service):
		"""Test streaming processing throughput"""
		# Simulate streaming chunks
		chunks = [
			f"Streaming chunk {i}: This is test content for streaming processing."
			for i in range(100)
		]
		
		security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
		
		# Create streaming session
		session = await nlpc_service.create_context_session('test-tenant')
		
		processed_chunks = []
		start_time = time.perf_counter()
		
		# Process chunks with streaming
		for chunk in chunks:
			document = create_test_document(chunk)
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			
			result = await nlpc_service.process_with_context(
				document, request, session.session_id
			)
			processed_chunks.append(result)
		
		end_time = time.perf_counter()
		
		# Calculate streaming metrics
		total_time = end_time - start_time
		chunks_per_second = len(chunks) / total_time
		
		assert len(processed_chunks) == len(chunks)
		assert chunks_per_second > 10  # Should process > 10 chunks/second
		
		# Verify context session was updated
		context_data = session.context_data
		assert len(context_data) > 0

class TestMemoryPerformance:
	"""Test memory usage and optimization"""
	
	async def test_memory_usage_single_document(self, nlpc_service, performance_test_data):
		"""Test memory usage for single document processing"""
		process = psutil.Process()
		
		# Measure baseline memory
		baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
		
		# Process large document
		document = create_test_document(performance_test_data['large_text'])
		request = create_test_request([NLPTask.NAMED_ENTITY_RECOGNITION])
		security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
		
		result = await nlpc_service.secure_process_document(
			document, request, security_context
		)
		
		# Measure peak memory
		peak_memory = process.memory_info().rss / 1024 / 1024  # MB
		memory_increase = peak_memory - baseline_memory
		
		assert_processing_result_valid(result)
		assert memory_increase < 500  # Should not use more than 500MB for single document
	
	async def test_memory_usage_concurrent_processing(self, nlpc_service, performance_test_data):
		"""Test memory usage under concurrent processing"""
		process = psutil.Process()
		baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
		
		# Create multiple concurrent tasks
		num_concurrent = 20
		tasks = []
		
		for i in range(num_concurrent):
			document = create_test_document(performance_test_data['medium_text'])
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
			
			task = nlpc_service.secure_process_document(document, request, security_context)
			tasks.append(task)
		
		# Execute concurrently and monitor memory
		memory_samples = []
		
		async def monitor_memory():
			for _ in range(10):  # Sample 10 times during processing
				await asyncio.sleep(0.1)
				current_memory = process.memory_info().rss / 1024 / 1024
				memory_samples.append(current_memory)
		
		# Run processing and monitoring concurrently
		results, _ = await asyncio.gather(
			asyncio.gather(*tasks, return_exceptions=True),
			monitor_memory()
		)
		
		peak_memory = max(memory_samples)
		memory_increase = peak_memory - baseline_memory
		
		# Memory should not grow excessively with concurrency
		assert memory_increase < 1000  # < 1GB increase
		
		# Verify processing succeeded
		successful_results = [r for r in results if not isinstance(r, Exception)]
		assert len(successful_results) == num_concurrent
	
	async def test_cache_memory_management(self, nlpc_service, performance_test_data):
		"""Test cache memory management and cleanup"""
		# Initialize performance system with limited cache
		await nlpc_service.initialize_performance_system({
			'cache_enabled': True,
			'cache_size': 10,  # Small cache to force eviction
			'cache_ttl': 300
		})
		
		process = psutil.Process()
		baseline_memory = process.memory_info().rss / 1024 / 1024
		
		# Fill cache beyond capacity
		documents = []
		for i in range(20):  # More than cache capacity
			document = create_test_document(f"Cache test document {i}: {performance_test_data['small_text']}")
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
			
			await nlpc_service.process_with_performance_optimization(
				document, request, security_context
			)
		
		# Check memory after cache operations
		current_memory = process.memory_info().rss / 1024 / 1024
		memory_increase = current_memory - baseline_memory
		
		# Memory should not grow excessively due to cache management
		assert memory_increase < 200  # < 200MB increase
		
		# Verify cache stats
		cache_stats = await nlpc_service._get_cache_statistics()
		assert cache_stats['current_size'] <= 10  # Should respect size limit

class TestAccuracyBenchmarks:
	"""Test NLP accuracy across different tasks and languages"""
	
	async def test_sentiment_analysis_accuracy(self, nlpc_service):
		"""Test sentiment analysis accuracy with known examples"""
		test_cases = [
			("I absolutely love this product! It's amazing!", "positive", 0.8),
			("This is the worst experience I've ever had.", "negative", 0.8),
			("The product is okay, nothing special.", "neutral", 0.6),
			("Outstanding quality and excellent customer service!", "positive", 0.9),
			("Terrible quality, waste of money!", "negative", 0.9)
		]
		
		correct_predictions = 0
		confidence_scores = []
		
		for text, expected_sentiment, min_confidence in test_cases:
			document = create_test_document(text)
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
			
			result = await nlpc_service.secure_process_document(
				document, request, security_context
			)
			
			assert_processing_result_valid(result)
			assert 'sentiment' in result.results
			
			predicted_sentiment = result.results['sentiment']
			confidence = result.results.get('confidence', 0)
			
			if predicted_sentiment == expected_sentiment:
				correct_predictions += 1
			
			confidence_scores.append(confidence)
			
			# Check minimum confidence threshold
			assert confidence >= min_confidence
		
		# Calculate accuracy
		accuracy = correct_predictions / len(test_cases)
		avg_confidence = statistics.mean(confidence_scores)
		
		assert accuracy >= 0.8  # 80% accuracy minimum
		assert avg_confidence >= 0.7  # 70% average confidence
	
	async def test_named_entity_recognition_accuracy(self, nlpc_service):
		"""Test named entity recognition accuracy"""
		test_cases = [
			{
				'text': "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
				'expected_entities': [
					{'text': 'Apple Inc.', 'label': 'ORG'},
					{'text': 'Steve Jobs', 'label': 'PERSON'},
					{'text': 'Cupertino', 'label': 'GPE'},
					{'text': 'California', 'label': 'GPE'}
				]
			},
			{
				'text': "Microsoft Corporation is headquartered in Redmond, Washington.",
				'expected_entities': [
					{'text': 'Microsoft Corporation', 'label': 'ORG'},
					{'text': 'Redmond', 'label': 'GPE'},
					{'text': 'Washington', 'label': 'GPE'}
				]
			}
		]
		
		total_expected = 0
		total_found = 0
		total_correct = 0
		
		for test_case in test_cases:
			document = create_test_document(test_case['text'])
			request = create_test_request([NLPTask.NAMED_ENTITY_RECOGNITION])
			security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
			
			result = await nlpc_service.secure_process_document(
				document, request, security_context
			)
			
			assert_processing_result_valid(result)
			assert 'entities' in result.results
			
			predicted_entities = result.results['entities']
			expected_entities = test_case['expected_entities']
			
			total_expected += len(expected_entities)
			total_found += len(predicted_entities)
			
			# Count correct predictions
			for expected in expected_entities:
				for predicted in predicted_entities:
					if (expected['text'].lower() in predicted.get('text', '').lower() and
						expected['label'] == predicted.get('label')):
						total_correct += 1
						break
		
		# Calculate metrics
		precision = total_correct / total_found if total_found > 0 else 0
		recall = total_correct / total_expected if total_expected > 0 else 0
		f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
		
		assert precision >= 0.7  # 70% precision
		assert recall >= 0.6     # 60% recall
		assert f1_score >= 0.65  # 65% F1 score
	
	async def test_language_detection_accuracy(self, nlpc_service):
		"""Test language detection accuracy"""
		test_cases = [
			("Hello world, how are you today?", LanguageCode.EN),
			("Hola mundo, ¿cómo estás hoy?", LanguageCode.ES),
			("Bonjour le monde, comment allez-vous aujourd'hui?", LanguageCode.FR),
			("Hallo Welt, wie geht es dir heute?", LanguageCode.DE),
			("Ciao mondo, come stai oggi?", LanguageCode.IT)
		]
		
		correct_predictions = 0
		confidence_scores = []
		
		for text, expected_language in test_cases:
			result = await nlpc_service._enhanced_language_detection(text)
			
			detected_language = LanguageCode(result['language'])
			confidence = result['confidence']
			
			if detected_language == expected_language:
				correct_predictions += 1
			
			confidence_scores.append(confidence)
			
			# Confidence should be reasonable for clear cases
			assert confidence >= 0.8
		
		# Calculate accuracy
		accuracy = correct_predictions / len(test_cases)
		avg_confidence = statistics.mean(confidence_scores)
		
		assert accuracy >= 0.9  # 90% accuracy for clear cases
		assert avg_confidence >= 0.9  # 90% average confidence
	
	async def test_multilingual_processing_accuracy(self, nlpc_service):
		"""Test processing accuracy across multiple languages"""
		multilingual_test_cases = [
			{
				'text': "J'adore ce produit, il est fantastique!",
				'language': LanguageCode.FR,
				'task': NLPTask.SENTIMENT_ANALYSIS,
				'expected': 'positive'
			},
			{
				'text': "Estoy muy decepcionado con este servicio.",
				'language': LanguageCode.ES,
				'task': NLPTask.SENTIMENT_ANALYSIS,
				'expected': 'negative'
			},
			{
				'text': "Ich bin sehr zufrieden mit dem Ergebnis.",
				'language': LanguageCode.DE,
				'task': NLPTask.SENTIMENT_ANALYSIS,
				'expected': 'positive'
			}
		]
		
		correct_predictions = 0
		
		for test_case in multilingual_test_cases:
			document = create_test_document(test_case['text'])
			document.language = test_case['language']
			
			request = create_test_request([test_case['task']])
			security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
			
			result = await nlpc_service.secure_process_document(
				document, request, security_context
			)
			
			assert_processing_result_valid(result)
			
			if test_case['task'] == NLPTask.SENTIMENT_ANALYSIS:
				predicted = result.results.get('sentiment')
				if predicted == test_case['expected']:
					correct_predictions += 1
		
		# Multilingual accuracy should be reasonable
		accuracy = correct_predictions / len(multilingual_test_cases)
		assert accuracy >= 0.7  # 70% accuracy across languages

class TestScalabilityBenchmarks:
	"""Test system scalability and load handling"""
	
	async def test_load_scalability(self, nlpc_service, performance_test_data):
		"""Test system behavior under increasing load"""
		load_levels = [10, 50, 100, 200]
		performance_metrics = {}
		
		for num_requests in load_levels:
			documents = []
			requests = []
			
			# Prepare requests
			for i in range(num_requests):
				document = create_test_document(performance_test_data['medium_text'])
				request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
				documents.append(document)
				requests.append(request)
			
			security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
			
			# Measure performance under load
			start_time = time.perf_counter()
			
			# Process in batches to avoid overwhelming the system
			batch_size = 10
			all_results = []
			
			for i in range(0, num_requests, batch_size):
				batch_docs = documents[i:i+batch_size]
				batch_requests = requests[i:i+batch_size]
				
				batch_tasks = []
				for doc, req in zip(batch_docs, batch_requests):
					task = nlpc_service.secure_process_document(doc, req, security_context)
					batch_tasks.append(task)
				
				batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
				all_results.extend(batch_results)
			
			end_time = time.perf_counter()
			
			# Calculate metrics
			total_time = end_time - start_time
			throughput = num_requests / total_time
			successful_results = [r for r in all_results if not isinstance(r, Exception)]
			success_rate = len(successful_results) / num_requests
			
			performance_metrics[num_requests] = {
				'throughput': throughput,
				'success_rate': success_rate,
				'total_time': total_time
			}
			
			# Basic performance requirements
			assert success_rate >= 0.95  # 95% success rate
			assert throughput >= 1  # At least 1 request/second
		
		# Verify scalability characteristics
		# System should handle higher loads with reasonable degradation
		small_load_throughput = performance_metrics[10]['throughput']
		large_load_throughput = performance_metrics[200]['throughput']
		
		# Throughput shouldn't degrade more than 50% at 20x load
		assert large_load_throughput >= small_load_throughput * 0.5
	
	async def test_context_session_scalability(self, nlpc_service):
		"""Test scalability of context session management"""
		num_sessions = 50
		documents_per_session = 10
		
		# Create multiple context sessions
		sessions = []
		for i in range(num_sessions):
			session = await nlpc_service.create_context_session(
				f"tenant-{i % 5}",  # 5 different tenants
				{'max_context_length': 2000}
			)
			sessions.append(session)
		
		# Process documents across sessions
		start_time = time.perf_counter()
		all_tasks = []
		
		for session in sessions:
			for j in range(documents_per_session):
				document = create_test_document(f"Context test document {j}")
				request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
				
				task = nlpc_service.process_with_context(
					document, request, session.session_id
				)
				all_tasks.append(task)
		
		results = await asyncio.gather(*all_tasks, return_exceptions=True)
		end_time = time.perf_counter()
		
		# Verify performance
		total_requests = num_sessions * documents_per_session
		total_time = end_time - start_time
		throughput = total_requests / total_time
		
		successful_results = [r for r in results if not isinstance(r, Exception)]
		success_rate = len(successful_results) / total_requests
		
		assert success_rate >= 0.95
		assert throughput >= 5  # 5 requests/second with context
		
		# Verify context sessions are managed properly
		assert len(nlpc_service.context_sessions) == num_sessions
	
	async def test_cache_performance_scalability(self, nlpc_service):
		"""Test cache performance under scale"""
		# Initialize with reasonable cache size
		await nlpc_service.initialize_performance_system({
			'cache_enabled': True,
			'cache_size': 100,
			'cache_ttl': 600
		})
		
		num_unique_documents = 200  # More than cache size
		num_repeated_requests = 500  # Many repeated requests
		
		# Create unique documents first (to populate cache)
		unique_documents = []
		for i in range(num_unique_documents):
			document = create_test_document(f"Unique document {i} for cache testing.")
			unique_documents.append(document)
		
		security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
		
		# First pass: populate cache
		first_pass_start = time.perf_counter()
		first_pass_tasks = []
		
		for doc in unique_documents:
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			task = nlpc_service.process_with_performance_optimization(
				doc, request, security_context
			)
			first_pass_tasks.append(task)
		
		await asyncio.gather(*first_pass_tasks, return_exceptions=True)

		first_pass_time = time.perf_counter() - first_pass_start
		
		# Second pass: should benefit from cache
		second_pass_start = time.perf_counter()
		second_pass_tasks = []
		
		# Repeat some documents (should hit cache)
		for i in range(num_repeated_requests):
			doc_index = i % min(50, num_unique_documents)  # Cycle through first 50
			doc = unique_documents[doc_index]
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			
			task = nlpc_service.process_with_performance_optimization(
				doc, request, security_context
			)
			second_pass_tasks.append(task)
		
		await asyncio.gather(*second_pass_tasks, return_exceptions=True)

		second_pass_time = time.perf_counter() - second_pass_start
		
		# Calculate performance improvement
		first_pass_throughput = num_unique_documents / first_pass_time
		second_pass_throughput = num_repeated_requests / second_pass_time
		
		# Cache should provide significant speedup
		cache_speedup = second_pass_throughput / first_pass_throughput
		assert cache_speedup >= 2  # At least 2x speedup from caching
		
		# Verify cache statistics
		cache_stats = await nlpc_service._get_cache_statistics()
		assert cache_stats['hit_rate'] > 0.3  # At least 30% hit rate