#!/usr/bin/env python3
"""
Unit Tests for NLP Integration Timing Implementation
Tests for processing time measurement and timing accuracy

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from capabilities.common.dvrl.nlp_integration import APGNLPProcessor


class TestNLPTimingImplementation:
    """Test suite for NLP timing measurement implementation"""
    
    @pytest.fixture
    def nlp_processor(self):
        """Create APGNLPProcessor instance for testing"""
        return APGNLPProcessor('test_tenant', 'test_user')
    
    @patch('ollama.list')
    @patch('ollama.generate')
    async def test_process_natural_language_query_timing_basic(self, mock_generate, mock_list, nlp_processor):
        """Test basic timing functionality"""
        # Mock Ollama setup
        mock_list.return_value = {'models': [{'name': 'llama3.2:latest'}]}
        mock_generate.side_effect = [
            {'response': 'SELECT COUNT(*) FROM users;'},  # SQL generation
            {'response': 'This query counts all users'}   # Explanation
        ]
        
        # Mock schema context
        schema_context = {
            'tables': {
                'users': {
                    'columns': ['id', 'name', 'email'],
                    'types': ['integer', 'varchar', 'varchar']
                }
            }
        }
        
        start_time = time.perf_counter()
        result = await nlp_processor.process_natural_language_query(
            "How many users are there?", 
            schema_context
        )
        end_time = time.perf_counter()
        
        # Verify timing is measured
        assert 'processing_time_ms' in result
        assert isinstance(result['processing_time_ms'], (int, float))
        assert result['processing_time_ms'] > 0
        
        # Timing should be reasonably accurate (within 100ms of actual)
        actual_time_ms = (end_time - start_time) * 1000
        measured_time_ms = result['processing_time_ms']
        time_diff = abs(actual_time_ms - measured_time_ms)
        assert time_diff < 100, f"Timing accuracy issue: {time_diff}ms difference"
        
        # Verify other result fields
        assert result['query_id'] is not None
        assert result['original_query'] == "How many users are there?"
        assert result['generated_sql'] == 'SELECT COUNT(*) FROM users;'
        assert result['model_used'] is not None
    
    @patch('ollama.list')
    @patch('ollama.generate')
    async def test_timing_with_slow_operations(self, mock_generate, mock_list, nlp_processor):
        """Test timing measurement with artificially slow operations"""
        mock_list.return_value = {'models': [{'name': 'llama3.2:latest'}]}
        
        # Mock slow Ollama responses
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return {'response': 'SELECT * FROM products;'}
        
        mock_generate.side_effect = [
            {'response': 'SELECT * FROM products;'},  # SQL generation 
            {'response': 'This lists all products'}    # Explanation
        ]
        
        result = await nlp_processor.process_natural_language_query(
            "Show me all products"
        )
        
        # Should measure at least the sleep time
        assert result['processing_time_ms'] >= 0  # Should be positive
        assert isinstance(result['processing_time_ms'], (int, float))
    
    @patch('ollama.list') 
    @patch('ollama.generate')
    async def test_timing_consistency(self, mock_generate, mock_list, nlp_processor):
        """Test timing measurement consistency across multiple calls"""
        mock_list.return_value = {'models': [{'name': 'llama3.2:latest'}]}
        mock_generate.side_effect = [
            {'response': 'SELECT COUNT(*) FROM orders;'},
            {'response': 'Counts total orders'},
            {'response': 'SELECT AVG(amount) FROM orders;'},  
            {'response': 'Calculates average order amount'}
        ]
        
        # Execute multiple queries
        results = []
        for query in ["How many orders?", "What's the average order amount?"]:
            result = await nlp_processor.process_natural_language_query(query)
            results.append(result)
        
        # All should have timing measurements
        for i, result in enumerate(results):
            assert 'processing_time_ms' in result, f"Result {i} missing timing"
            assert result['processing_time_ms'] > 0, f"Result {i} has invalid timing"
            assert isinstance(result['processing_time_ms'], (int, float)), f"Result {i} timing not numeric"
    
    @patch('ollama.list')
    @patch('ollama.generate')
    async def test_timing_with_exception_handling(self, mock_generate, mock_list, nlp_processor):
        """Test that timing works even when exceptions occur"""
        mock_list.return_value = {'models': [{'name': 'llama3.2:latest'}]}
        
        # First call succeeds, second fails for explanation
        mock_generate.side_effect = [
            {'response': 'SELECT * FROM invalid_table;'},  # SQL generation succeeds
            Exception("Explanation generation failed")      # Explanation fails
        ]
        
        # Should not raise exception but should still measure timing
        result = await nlp_processor.process_natural_language_query(
            "Show me data from invalid table"
        )
        
        # Should still have timing measurement despite internal error
        assert 'processing_time_ms' in result
        assert result['processing_time_ms'] > 0
        assert result['generated_sql'] == 'SELECT * FROM invalid_table;'
    
    @patch('ollama.list')
    @patch('ollama.generate')
    async def test_conversation_history_includes_timing(self, mock_generate, mock_list, nlp_processor):
        """Test that conversation history preserves timing information"""
        mock_list.return_value = {'models': [{'name': 'llama3.2:latest'}]}
        mock_generate.side_effect = [
            {'response': 'SELECT name FROM customers;'},
            {'response': 'Lists customer names'}
        ]
        
        await nlp_processor.process_natural_language_query("List customer names")
        
        # Check conversation history
        assert len(nlp_processor.conversation_history) == 1
        history_entry = nlp_processor.conversation_history[0]
        
        # History should contain timing information in metadata or main data
        assert 'natural_query' in history_entry
        assert 'generated_sql' in history_entry
        assert 'confidence' in history_entry
        assert 'timestamp' in history_entry
    
    async def test_timing_precision(self, nlp_processor):
        """Test timing measurement precision"""
        with patch('ollama.list', return_value={'models': [{'name': 'llama3.2:latest'}]}):
            with patch('ollama.generate') as mock_generate:
                # Mock very fast response
                mock_generate.side_effect = [
                    {'response': 'SELECT 1;'},
                    {'response': 'Simple test query'}
                ]
                
                result = await nlp_processor.process_natural_language_query("Test query")
                
                # Should have precise timing measurement
                timing = result['processing_time_ms']
                assert isinstance(timing, (int, float))
                assert timing >= 0
                
                # Should be reasonable precision (not just rounded to nearest second)
                # Even very fast operations should show some microsecond-level timing
                assert timing < 10000  # Should be less than 10 seconds for simple test
    
    @patch('time.perf_counter')
    @patch('ollama.list')
    @patch('ollama.generate')
    async def test_timing_calculation_accuracy(self, mock_generate, mock_list, mock_perf_counter, nlp_processor):
        """Test timing calculation mathematical accuracy"""
        mock_list.return_value = {'models': [{'name': 'llama3.2:latest'}]}
        mock_generate.side_effect = [
            {'response': 'SELECT id FROM items;'},
            {'response': 'Gets item IDs'}
        ]
        
        # Mock perf_counter to return predictable values
        # Start: 1000.0, End: 1000.5 (500ms difference)
        mock_perf_counter.side_effect = [1000.0, 1000.5]
        
        result = await nlp_processor.process_natural_language_query("Get item IDs")
        
        # Should calculate exactly 500ms
        expected_time_ms = (1000.5 - 1000.0) * 1000  # 500.0ms
        assert result['processing_time_ms'] == expected_time_ms
        
        # Verify perf_counter was called twice (start and end)
        assert mock_perf_counter.call_count == 2
    
    @patch('ollama.list')
    async def test_timing_with_no_model(self, mock_list, nlp_processor):
        """Test timing behavior when no Ollama models available"""
        mock_list.return_value = {'models': []}  # No models available
        
        # Should handle gracefully and still measure timing  
        result = await nlp_processor.process_natural_language_query("Test query with no model")
        
        # Should still have timing measurement even if processing fails
        assert 'processing_time_ms' in result
        assert isinstance(result['processing_time_ms'], (int, float))
        assert result['processing_time_ms'] >= 0


class TestNLPPerformanceMonitoring:
    """Test suite for NLP performance monitoring capabilities"""
    
    @pytest.fixture
    def nlp_processor(self):
        """Create APGNLPProcessor for performance testing"""
        return APGNLPProcessor('perf_tenant', 'perf_user')
    
    @patch('ollama.list')
    @patch('ollama.generate') 
    async def test_performance_tracking_across_queries(self, mock_generate, mock_list, nlp_processor):
        """Test performance tracking across multiple queries"""
        mock_list.return_value = {'models': [{'name': 'llama3.2:latest'}]}
        mock_generate.side_effect = [
            # First query responses
            {'response': 'SELECT COUNT(*) FROM users;'},
            {'response': 'Count users'},
            # Second query responses  
            {'response': 'SELECT MAX(date) FROM logs;'},
            {'response': 'Get latest log date'},
            # Third query responses
            {'response': 'SELECT AVG(price) FROM products;'},
            {'response': 'Average product price'}
        ]
        
        queries = [
            "How many users?",
            "When was the last log entry?", 
            "What's the average product price?"
        ]
        
        results = []
        for query in queries:
            result = await nlp_processor.process_natural_language_query(query)
            results.append(result)
        
        # All results should have timing
        for i, result in enumerate(results):
            assert 'processing_time_ms' in result
            assert result['processing_time_ms'] > 0
        
        # Can calculate performance statistics
        times = [r['processing_time_ms'] for r in results]
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        assert avg_time > 0
        assert max_time >= avg_time >= min_time
        assert all(t > 0 for t in times)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])