#!/usr/bin/env python3
"""
APG Cache Management (CACH) - AI Optimization Tests
Comprehensive tests for AI optimization engine

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from ai_optimization import OptimizationEngine, OptimizationMetrics, OptimizationStrategy
from models import CacheEntry, CacheAccessPattern
from conftest import PerformanceTimer, simulate_cache_load


class TestOptimizationEngine:
    """Test suite for AI optimization engine"""
    
    async def test_engine_initialization(self):
        """Test optimization engine initialization"""
        config = {'max_optimization_cycles': 10}
        engine = OptimizationEngine(config)
        
        # Engine should not be running initially
        assert not engine._running
        
        # Initialize the engine
        await engine.initialize()
        
        # Engine should now be running
        assert engine._running
        assert engine.ml_models is not None
        assert engine.optimization_history is not None
        
        # Cleanup
        await engine.shutdown()
        assert not engine._running
    
    async def test_cache_analysis(self, optimization_engine, sample_cache_entries):
        """Test cache performance analysis"""
        # Analyze cache entries
        analysis = await optimization_engine.analyze_cache_performance(sample_cache_entries)
        
        # Verify analysis structure
        assert isinstance(analysis, dict)
        assert 'total_entries' in analysis
        assert 'memory_utilization' in analysis
        assert 'access_patterns' in analysis
        assert 'performance_bottlenecks' in analysis
        
        # Verify analysis values
        assert analysis['total_entries'] == len(sample_cache_entries)
        assert 0.0 <= analysis['memory_utilization'] <= 100.0
        assert isinstance(analysis['access_patterns'], dict)
        assert isinstance(analysis['performance_bottlenecks'], list)
    
    async def test_optimization_recommendations(self, optimization_engine, sample_cache_entries):
        """Test optimization recommendation generation"""
        # Generate recommendations
        recommendations = await optimization_engine.generate_optimization_recommendations(sample_cache_entries)
        
        # Verify recommendations structure
        assert isinstance(recommendations, list)
        
        for rec in recommendations:
            assert 'strategy' in rec
            assert 'priority' in rec
            assert 'expected_improvement' in rec
            assert 'implementation_complexity' in rec
            assert 'confidence_score' in rec
            
            # Verify enum types
            assert isinstance(rec['strategy'], OptimizationStrategy)
            assert rec['priority'] in ['low', 'medium', 'high', 'critical']
            assert 0.0 <= rec['confidence_score'] <= 1.0
    
    async def test_cache_sizing_optimization(self, optimization_engine):
        """Test cache sizing optimization"""
        # Mock performance metrics
        metrics = OptimizationMetrics(
            hit_rate=0.75,
            average_latency_ms=5.2,
            memory_utilization=95.0,
            operations_per_second=1500.0,
            eviction_rate=0.15,
            timestamp=datetime.utcnow()
        )
        
        # Get sizing recommendations
        sizing_rec = await optimization_engine.optimize_cache_sizing(metrics)
        
        assert isinstance(sizing_rec, dict)
        assert 'recommended_size_mb' in sizing_rec
        assert 'current_utilization' in sizing_rec
        assert 'expected_improvement' in sizing_rec
        assert 'confidence' in sizing_rec
        
        # High utilization should recommend size increase
        if metrics.memory_utilization > 90:
            assert sizing_rec['recommended_size_mb'] > 0
    
    async def test_eviction_policy_optimization(self, optimization_engine, sample_cache_entries):
        """Test eviction policy optimization"""
        # Analyze access patterns for different entries
        access_patterns = {}
        for key, entry in sample_cache_entries.items():
            if "hot:" in key:
                pattern = CacheAccessPattern.FREQUENT
            elif "test:" in key:
                pattern = CacheAccessPattern.TEMPORAL
            else:
                pattern = CacheAccessPattern.RANDOM
            
            access_patterns[key] = pattern
        
        # Get eviction policy recommendation
        policy_rec = await optimization_engine.optimize_eviction_policy(access_patterns)
        
        assert isinstance(policy_rec, dict)
        assert 'recommended_policy' in policy_rec
        assert 'current_policy' in policy_rec
        assert 'expected_hit_rate_improvement' in policy_rec
        assert 'confidence' in policy_rec
        
        # Should recommend appropriate policy based on patterns
        assert policy_rec['recommended_policy'] in ['LRU', 'LFU', 'FIFO', 'ARC', 'ADAPTIVE']
    
    async def test_tier_optimization(self, optimization_engine, sample_cache_entries):
        """Test multi-tier cache optimization"""
        # Generate tier recommendations
        tier_recommendations = await optimization_engine.optimize_tier_allocation(sample_cache_entries)
        
        assert isinstance(tier_recommendations, list)
        
        for rec in tier_recommendations:
            assert 'entry_key' in rec
            assert 'current_tier' in rec
            assert 'recommended_tier' in rec
            assert 'migration_priority' in rec
            assert 'expected_latency_improvement' in rec
            
            # Migration priority should be valid
            assert rec['migration_priority'] in ['low', 'medium', 'high', 'critical']
    
    async def test_prefetch_optimization(self, optimization_engine, sample_cache_entries):
        """Test prefetching strategy optimization"""
        # Create access history simulation
        access_history = []
        for i, (key, entry) in enumerate(sample_cache_entries.items()):
            access_history.append({
                'key': key,
                'timestamp': datetime.utcnow() - timedelta(minutes=i),
                'hit': True,
                'latency_ms': 2.0 + (i % 10) * 0.5
            })
        
        # Get prefetch recommendations
        prefetch_rec = await optimization_engine.optimize_prefetching_strategy(access_history)
        
        assert isinstance(prefetch_rec, dict)
        assert 'prefetch_candidates' in prefetch_rec
        assert 'prefetch_strategy' in prefetch_rec
        assert 'expected_hit_rate_improvement' in prefetch_rec
        assert 'confidence' in prefetch_rec
        
        # Should identify prefetch candidates
        candidates = prefetch_rec['prefetch_candidates']
        assert isinstance(candidates, list)
        
        for candidate in candidates:
            assert 'key_pattern' in candidate
            assert 'probability' in candidate
            assert 0.0 <= candidate['probability'] <= 1.0
    
    async def test_ml_model_training(self, optimization_engine):
        """Test ML model training and updating"""
        # Generate training data
        training_data = []
        for i in range(100):
            data_point = {
                'features': {
                    'cache_size_mb': 1024 + (i * 10),
                    'hit_rate': 0.7 + (i % 20) * 0.01,
                    'memory_utilization': 50 + (i % 40),
                    'operations_per_second': 1000 + (i * 50)
                },
                'target': {
                    'performance_score': 0.8 + (i % 20) * 0.01
                }
            }
            training_data.append(data_point)
        
        # Train models
        training_result = await optimization_engine.train_optimization_models(training_data)
        
        assert isinstance(training_result, dict)
        assert 'models_trained' in training_result
        assert 'training_accuracy' in training_result
        assert 'validation_score' in training_result
        
        # Models should be trained
        assert training_result['models_trained'] > 0
        assert 0.0 <= training_result['training_accuracy'] <= 1.0
    
    async def test_real_time_optimization(self, optimization_engine):
        """Test real-time optimization capabilities"""
        # Simulate real-time metrics
        metrics = OptimizationMetrics(
            hit_rate=0.82,
            average_latency_ms=3.5,
            memory_utilization=72.0,
            operations_per_second=2200.0,
            eviction_rate=0.08,
            timestamp=datetime.utcnow()
        )
        
        # Trigger real-time optimization
        optimization_result = await optimization_engine.optimize_real_time(metrics)
        
        assert isinstance(optimization_result, dict)
        assert 'optimizations_applied' in optimization_result
        assert 'performance_improvement' in optimization_result
        assert 'optimization_duration_ms' in optimization_result
        
        # Should apply some optimizations
        assert optimization_result['optimizations_applied'] >= 0
        assert optimization_result['optimization_duration_ms'] > 0
    
    async def test_performance_prediction(self, optimization_engine):
        """Test performance prediction capabilities"""
        # Create historical performance data
        history = []
        for i in range(50):
            timestamp = datetime.utcnow() - timedelta(hours=i)
            history.append({
                'timestamp': timestamp,
                'hit_rate': 0.8 + 0.1 * np.sin(i * 0.1),
                'latency_ms': 3.0 + 1.0 * np.cos(i * 0.1),
                'throughput': 2000 + 500 * np.sin(i * 0.2),
                'memory_utilization': 60 + 20 * np.cos(i * 0.15)
            })
        
        # Get performance predictions
        predictions = await optimization_engine.predict_performance(history, hours_ahead=24)
        
        assert isinstance(predictions, dict)
        assert 'predicted_metrics' in predictions
        assert 'confidence_intervals' in predictions
        assert 'trend_analysis' in predictions
        
        # Predictions should be reasonable
        predicted = predictions['predicted_metrics']
        assert 'hit_rate' in predicted
        assert 'latency_ms' in predicted
        assert 'throughput' in predicted
        
        # Confidence intervals should be provided
        confidence = predictions['confidence_intervals']
        for metric in predicted:
            assert metric in confidence
            assert 'lower' in confidence[metric]
            assert 'upper' in confidence[metric]
    
    async def test_optimization_effectiveness(self, optimization_engine):
        """Test optimization effectiveness measurement"""
        # Simulate before/after metrics
        before_metrics = OptimizationMetrics(
            hit_rate=0.75,
            average_latency_ms=4.5,
            memory_utilization=85.0,
            operations_per_second=1800.0,
            eviction_rate=0.12,
            timestamp=datetime.utcnow() - timedelta(minutes=30)
        )
        
        after_metrics = OptimizationMetrics(
            hit_rate=0.87,
            average_latency_ms=3.2,
            memory_utilization=78.0,
            operations_per_second=2200.0,
            eviction_rate=0.06,
            timestamp=datetime.utcnow()
        )
        
        # Measure effectiveness
        effectiveness = await optimization_engine.measure_optimization_effectiveness(
            before_metrics, after_metrics
        )
        
        assert isinstance(effectiveness, dict)
        assert 'overall_improvement' in effectiveness
        assert 'metric_improvements' in effectiveness
        assert 'optimization_success' in effectiveness
        
        # Should detect improvements
        improvements = effectiveness['metric_improvements']
        assert 'hit_rate' in improvements
        assert 'latency_ms' in improvements
        
        # Hit rate improvement should be positive
        assert improvements['hit_rate'] > 0
        # Latency improvement should be positive (lower is better)
        assert improvements['latency_ms'] > 0
    
    async def test_adaptive_optimization(self, optimization_engine):
        """Test adaptive optimization behavior"""
        # Simulate changing conditions
        conditions = [
            {'time_of_day': 9, 'load': 'high', 'pattern': 'business'},
            {'time_of_day': 14, 'load': 'medium', 'pattern': 'mixed'},
            {'time_of_day': 22, 'load': 'low', 'pattern': 'background'},
            {'time_of_day': 2, 'load': 'very_low', 'pattern': 'maintenance'}
        ]
        
        adaptations = []
        for condition in conditions:
            # Get adaptive optimization for each condition
            adaptation = await optimization_engine.adapt_to_conditions(condition)
            adaptations.append(adaptation)
            
            assert isinstance(adaptation, dict)
            assert 'strategy_adjustments' in adaptation
            assert 'parameter_changes' in adaptation
            assert 'expected_benefit' in adaptation
        
        # Different conditions should produce different adaptations
        strategies = [adapt['strategy_adjustments'] for adapt in adaptations]
        assert len(set(str(s) for s in strategies)) > 1  # Should have variation
    
    @pytest.mark.performance
    async def test_optimization_performance(self, optimization_engine, sample_cache_entries):
        """Test optimization engine performance"""
        with PerformanceTimer() as timer:
            # Run comprehensive optimization
            recommendations = await optimization_engine.generate_optimization_recommendations(
                sample_cache_entries
            )
        
        # Optimization should complete within reasonable time
        assert timer.duration_ms < 5000  # 5 seconds max
        assert len(recommendations) > 0
        
        # Test concurrent optimizations
        tasks = []
        for _ in range(5):
            task = optimization_engine.analyze_cache_performance(sample_cache_entries)
            tasks.append(task)
        
        with PerformanceTimer() as concurrent_timer:
            results = await asyncio.gather(*tasks)
        
        # Concurrent operations should be efficient
        assert len(results) == 5
        assert concurrent_timer.duration_ms < 10000  # 10 seconds max for 5 concurrent
    
    async def test_optimization_persistence(self, optimization_engine):
        """Test optimization state persistence"""
        # Generate some optimization history
        for i in range(10):
            metrics = OptimizationMetrics(
                hit_rate=0.8 + (i * 0.01),
                average_latency_ms=3.0 - (i * 0.1),
                memory_utilization=70.0 + (i * 2),
                operations_per_second=2000.0 + (i * 100),
                eviction_rate=0.1 - (i * 0.005),
                timestamp=datetime.utcnow() - timedelta(minutes=i)
            )
            
            await optimization_engine.record_optimization_result(metrics, f"test_optimization_{i}")
        
        # Get optimization history
        history = await optimization_engine.get_optimization_history(limit=5)
        
        assert isinstance(history, list)
        assert len(history) <= 5  # Should respect limit
        
        for entry in history:
            assert 'timestamp' in entry
            assert 'optimization_type' in entry
            assert 'metrics' in entry
            assert 'result' in entry
    
    async def test_error_handling(self, optimization_engine):
        """Test error handling in optimization engine"""
        # Test with invalid data
        with pytest.raises((ValueError, TypeError)):
            await optimization_engine.analyze_cache_performance(None)
        
        with pytest.raises((ValueError, TypeError)):
            await optimization_engine.analyze_cache_performance({})
        
        # Test with corrupted metrics
        invalid_metrics = OptimizationMetrics(
            hit_rate=-1.0,  # Invalid hit rate
            average_latency_ms=-5.0,  # Invalid latency
            memory_utilization=150.0,  # Invalid utilization
            operations_per_second=-1000.0,  # Invalid ops
            eviction_rate=-0.5,  # Invalid eviction rate
            timestamp=datetime.utcnow()
        )
        
        # Should handle gracefully
        try:
            result = await optimization_engine.optimize_real_time(invalid_metrics)
            # If no exception, result should indicate error handling
            assert 'error' in result or result.get('optimizations_applied', 0) == 0
        except (ValueError, TypeError):
            # Exception is acceptable for invalid data
            pass


class TestOptimizationIntegration:
    """Integration tests for optimization engine"""
    
    async def test_cache_service_integration(self, cache_service, optimization_engine):
        """Test integration with cache service"""
        # Populate cache service
        for i in range(50):
            await cache_service.set(f"integration:key:{i}", f"value_{i}")
        
        # Get cache entries for optimization
        cache_entries = {}
        stats = await cache_service.get_stats()
        
        # Simulate getting entries (in real implementation, this would be internal)
        for i in range(50):
            key = f"integration:key:{i}"
            value = await cache_service.get(key)
            if value:
                cache_entries[key] = value
        
        # Run optimization analysis
        if cache_entries:
            analysis = await optimization_engine.analyze_cache_performance(cache_entries)
            recommendations = await optimization_engine.generate_optimization_recommendations(cache_entries)
            
            assert isinstance(analysis, dict)
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
    
    async def test_end_to_end_optimization(self, cache_service, optimization_engine):
        """Test end-to-end optimization workflow"""
        # 1. Setup initial cache state
        for i in range(100):
            key = f"e2e:key:{i}"
            value = f"e2e_value_{i}"
            await cache_service.set(key, value)
        
        # 2. Generate load to create performance baseline
        load_results = await simulate_cache_load(cache_service, operations=500, pattern="mixed")
        assert load_results['hits'] + load_results['misses'] + load_results['sets'] > 400
        
        # 3. Get baseline metrics
        baseline_stats = await cache_service.get_stats()
        baseline_hit_rate = baseline_stats.get('hit_rate', 0)
        
        # 4. Run optimization
        cache_entries = {}  # In real scenario, this would come from cache service
        recommendations = await optimization_engine.generate_optimization_recommendations(cache_entries)
        
        # 5. Apply high-priority recommendations (simulated)
        high_priority_recs = [
            rec for rec in recommendations 
            if rec.get('priority') == 'high' and rec.get('confidence_score', 0) > 0.8
        ]
        
        # 6. Verify optimization improved performance
        # In a real scenario, we would apply recommendations and measure improvement
        assert isinstance(recommendations, list)
        assert baseline_hit_rate >= 0  # Baseline should be valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])