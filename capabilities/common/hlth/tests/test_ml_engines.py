#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - ML Engines Tests
Comprehensive tests for machine learning and analytics engines

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from ml_engines import (
    HealthPredictionEngine, AdvancedAnalyticsEngine, MLModelType
)
from optimization_engine import (
    ResourceOptimizationEngine, OptimizationType
)


class TestHealthPredictionEngine:
    """Test suite for ML prediction engine"""
    
    async def test_engine_initialization(self, prediction_engine):
        """Test prediction engine initialization"""
        assert prediction_engine is not None
        assert hasattr(prediction_engine, 'models')
        assert hasattr(prediction_engine, 'scalers')
        
        # Check models are initialized
        expected_models = ['health_score', 'failure_prediction', 'anomaly_detection']
        for model_name in expected_models:
            assert model_name in prediction_engine.models
            assert prediction_engine.models[model_name]['type'] is not None
    
    async def test_health_score_prediction(self, prediction_engine):
        """Test health score prediction"""
        prediction = await prediction_engine.predict_health_score(
            component_id='test-component',
            tenant_id='test-tenant',
            prediction_window_hours=24
        )
        
        assert 'component_id' in prediction
        assert 'predicted_health_score' in prediction
        assert 'confidence' in prediction
        assert 'risk_level' in prediction
        
        # Validate prediction values
        score = prediction['predicted_health_score']
        assert 0 <= score <= 100
        
        confidence = prediction['confidence']
        assert 0 <= confidence <= 1
        
        assert prediction['risk_level'] in ['low', 'medium', 'high', 'critical']
    
    async def test_anomaly_detection(self, prediction_engine):
        """Test anomaly detection functionality"""
        detection_result = await prediction_engine.detect_anomalies(
            component_id='test-component',
            tenant_id='test-tenant',
            time_window_hours=24
        )
        
        assert 'component_id' in detection_result
        assert 'anomalies_detected' in detection_result
        assert 'anomalies' in detection_result
        assert 'overall_anomaly_score' in detection_result
        
        # Validate anomaly data
        anomaly_count = detection_result['anomalies_detected']
        assert isinstance(anomaly_count, int)
        assert anomaly_count >= 0
        
        if anomaly_count > 0:
            anomalies = detection_result['anomalies']
            assert len(anomalies) == anomaly_count
            
            for anomaly in anomalies:
                assert 'timestamp' in anomaly
                assert 'anomaly_score' in anomaly
                assert 'severity' in anomaly
                assert anomaly['severity'] in ['low', 'medium', 'high']
    
    async def test_failure_prediction(self, prediction_engine):
        """Test failure probability prediction"""
        prediction = await prediction_engine.predict_failure_probability(
            component_id='test-component',
            tenant_id='test-tenant',
            prediction_window_hours=48
        )
        
        assert 'component_id' in prediction
        assert 'failure_probability' in prediction
        assert 'confidence' in prediction
        assert 'risk_level' in prediction
        
        # Validate prediction values
        failure_prob = prediction['failure_probability']
        assert 0 <= failure_prob <= 1
        
        confidence = prediction['confidence']
        assert 0 <= confidence <= 1
        
        if 'time_to_failure_estimate' in prediction:
            ttf = prediction['time_to_failure_estimate']
            assert ttf is None or isinstance(ttf, (int, float))
    
    async def test_model_training(self, prediction_engine):
        """Test ML model training"""
        training_result = await prediction_engine.train_models('test-tenant')
        
        assert 'status' in training_result
        assert training_result['status'] in ['completed', 'insufficient_data', 'failed']
        
        if training_result['status'] == 'completed':
            assert 'training_results' in training_result
            assert 'models_trained' in training_result
            assert isinstance(training_result['models_trained'], int)
    
    @patch('ml_engines.ML_AVAILABLE', False)
    async def test_baseline_prediction_fallback(self):
        """Test baseline prediction when ML libraries are not available"""
        engine = HealthPredictionEngine()
        
        prediction = await engine.predict_health_score(
            component_id='fallback-test',
            tenant_id='test-tenant'
        )
        
        # Should still return valid prediction using baseline methods
        assert 'predicted_health_score' in prediction
        assert 'confidence' in prediction
        assert prediction['confidence'] >= 0.5  # Should have reasonable confidence
    
    async def test_feature_extraction(self, prediction_engine):
        """Test feature extraction for predictions"""
        features = await prediction_engine._extract_features_for_component(
            'test-component', 'test-tenant'
        )
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check for expected features
        expected_features = [
            'cpu_utilization', 'memory_utilization', 'error_rate',
            'response_time', 'availability_score'
        ]
        
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
    
    async def test_prediction_confidence_calculation(self, prediction_engine):
        """Test prediction confidence calculation"""
        model_info = prediction_engine.models['health_score']
        features = {
            'cpu_utilization': 75.0,
            'memory_utilization': 60.0,
            'error_rate': 0.02
        }
        
        confidence = prediction_engine._calculate_prediction_confidence(
            model_info, features
        )
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)
    
    async def test_concurrent_predictions(self, prediction_engine):
        """Test concurrent prediction requests"""
        tasks = []
        
        for i in range(10):
            task = prediction_engine.predict_health_score(
                component_id=f'concurrent-component-{i}',
                tenant_id='concurrent-test',
                prediction_window_hours=24
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check all predictions succeeded
        successful_predictions = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_predictions) == 10
        
        # Verify all predictions are valid
        for prediction in successful_predictions:
            assert 'predicted_health_score' in prediction
            assert 'confidence' in prediction


class TestAdvancedAnalyticsEngine:
    """Test suite for advanced analytics engine"""
    
    async def test_analytics_engine_initialization(self, prediction_engine):
        """Test analytics engine initialization"""
        analytics_engine = AdvancedAnalyticsEngine(prediction_engine)
        
        assert analytics_engine.prediction_engine == prediction_engine
        assert hasattr(analytics_engine, 'analytics_cache')
    
    async def test_health_insights_generation(self, prediction_engine):
        """Test comprehensive health insights generation"""
        analytics_engine = AdvancedAnalyticsEngine(prediction_engine)
        
        insights = await analytics_engine.generate_health_insights(
            tenant_id='test-tenant',
            time_window_hours=168
        )
        
        assert 'tenant_id' in insights
        assert 'analysis_period_hours' in insights
        
        # Check for key insights sections
        expected_sections = [
            'overall_health_trend', 'top_risk_components',
            'performance_bottlenecks', 'cost_optimization_opportunities'
        ]
        
        for section in expected_sections:
            assert section in insights
    
    async def test_health_trend_analysis(self, prediction_engine):
        """Test health trend analysis"""
        analytics_engine = AdvancedAnalyticsEngine(prediction_engine)
        
        trend = await analytics_engine._analyze_health_trend(
            'test-tenant', 168
        )
        
        assert 'trend_direction' in trend
        assert 'trend_strength' in trend
        assert 'average_health_score' in trend
        assert 'health_volatility' in trend
        
        assert trend['trend_direction'] in ['improving', 'stable', 'degrading']
        assert 0 <= trend['trend_strength'] <= 1
        assert 0 <= trend['average_health_score'] <= 100
    
    async def test_risk_component_identification(self, prediction_engine):
        """Test identification of high-risk components"""
        analytics_engine = AdvancedAnalyticsEngine(prediction_engine)
        
        risk_components = await analytics_engine._identify_top_risk_components(
            'test-tenant'
        )
        
        assert isinstance(risk_components, list)
        
        if len(risk_components) > 0:
            component = risk_components[0]
            assert 'component_id' in component
            assert 'risk_score' in component
            assert 'primary_risks' in component
            assert 'time_to_failure_estimate' in component
            
            assert 0 <= component['risk_score'] <= 1
            assert isinstance(component['primary_risks'], list)


class TestResourceOptimizationEngine:
    """Test suite for resource optimization engine"""
    
    async def test_optimization_engine_initialization(self, optimization_engine):
        """Test optimization engine initialization"""
        assert optimization_engine is not None
        assert hasattr(optimization_engine, 'optimization_history')
        assert hasattr(optimization_engine, 'cost_models')
        assert hasattr(optimization_engine, 'performance_baselines')
    
    async def test_optimization_opportunity_analysis(self, optimization_engine):
        """Test optimization opportunity analysis"""
        analysis = await optimization_engine.analyze_optimization_opportunities(
            tenant_id='test-tenant'
        )
        
        assert 'tenant_id' in analysis
        assert 'total_opportunities' in analysis
        assert 'total_estimated_savings' in analysis
        assert 'optimizations' in analysis
        
        # Validate opportunities
        opportunities = analysis['optimizations']
        assert isinstance(opportunities, list)
        
        if len(opportunities) > 0:
            optimization = opportunities[0]
            assert 'recommendation_id' in optimization
            assert 'optimization_type' in optimization
            assert 'title' in optimization
            assert 'description' in optimization
            assert 'expected_benefits' in optimization
            assert 'implementation_effort' in optimization
            assert 'risk_level' in optimization
    
    async def test_resource_optimization_analysis(self, optimization_engine):
        """Test resource-specific optimization analysis"""
        recommendations = await optimization_engine._analyze_resource_optimization(
            'test-component', 'test-tenant'
        )
        
        assert isinstance(recommendations, list)
        
        if len(recommendations) > 0:
            recommendation = recommendations[0]
            assert recommendation.component_id == 'test-component'
            assert recommendation.tenant_id == 'test-tenant'
            assert recommendation.optimization_type in OptimizationType
            assert 0 <= recommendation.priority_score <= 1
            assert 0 <= recommendation.confidence <= 1
    
    async def test_cpu_optimization_analysis(self, optimization_engine):
        """Test CPU optimization analysis"""
        resource_metrics = {
            'cpu_utilization_avg': 15.0,  # Low utilization
            'cpu_utilization_peak': 35.0,
            'cpu_cores_allocated': 4
        }
        
        recommendations = await optimization_engine._analyze_cpu_optimization(
            'test-component', 'test-tenant', resource_metrics
        )
        
        assert isinstance(recommendations, list)
        
        # Should recommend downsizing for low utilization
        if len(recommendations) > 0:
            rec = recommendations[0]
            assert 'downsize' in rec.title.lower() or 'reduce' in rec.title.lower()
            assert rec.optimization_type == OptimizationType.RESOURCE_SCALING
    
    async def test_memory_optimization_analysis(self, optimization_engine):
        """Test memory optimization analysis"""
        resource_metrics = {
            'memory_utilization_avg': 70.0,
            'memory_utilization_peak': 85.0,
            'memory_gb_allocated': 16,
            'memory_trend_7d': 8.0  # High trend indicates potential leak
        }
        
        recommendations = await optimization_engine._analyze_memory_optimization(
            'test-component', 'test-tenant', resource_metrics
        )
        
        assert isinstance(recommendations, list)
        
        # Should detect memory leak
        if len(recommendations) > 0:
            rec = recommendations[0]
            assert 'memory leak' in rec.title.lower() or 'leak' in rec.description.lower()
            assert rec.optimization_type == OptimizationType.PERFORMANCE_TUNING
    
    async def test_cost_optimization_analysis(self, optimization_engine):
        """Test cost optimization analysis"""
        recommendations = await optimization_engine._analyze_cost_optimization(
            'test-component', 'test-tenant'
        )
        
        assert isinstance(recommendations, list)
        
        if len(recommendations) > 0:
            rec = recommendations[0]
            assert rec.optimization_type == OptimizationType.COST_OPTIMIZATION
            assert rec.estimated_savings > 0
    
    async def test_recommendation_ranking(self, optimization_engine):
        """Test optimization recommendation ranking"""
        from optimization_engine import OptimizationRecommendation
        
        # Create test recommendations with different priorities
        recommendations = [
            OptimizationRecommendation(
                recommendation_id='test-1',
                component_id='test',
                tenant_id='test',
                optimization_type=OptimizationType.RESOURCE_SCALING,
                title='Low Priority',
                description='Test',
                current_state={},
                recommended_state={},
                expected_benefits={},
                implementation_effort='low',
                risk_level='low',
                estimated_savings=100,
                implementation_steps=[],
                prerequisites=[],
                monitoring_metrics=[],
                rollback_plan='',
                priority_score=0.3,
                confidence=0.8,
                created_at=datetime.utcnow(),
                estimated_implementation_time=1
            ),
            OptimizationRecommendation(
                recommendation_id='test-2',
                component_id='test',
                tenant_id='test',
                optimization_type=OptimizationType.PERFORMANCE_TUNING,
                title='High Priority',
                description='Test',
                current_state={},
                recommended_state={},
                expected_benefits={},
                implementation_effort='medium',
                risk_level='low',
                estimated_savings=1000,
                implementation_steps=[],
                prerequisites=[],
                monitoring_metrics=[],
                rollback_plan='',
                priority_score=0.9,
                confidence=0.9,
                created_at=datetime.utcnow(),
                estimated_implementation_time=4
            )
        ]
        
        ranked = optimization_engine._rank_optimizations(recommendations)
        
        assert len(ranked) == 2
        assert ranked[0].priority_score > ranked[1].priority_score
        assert ranked[0].title == 'High Priority'
    
    async def test_concurrent_optimization_analysis(self, optimization_engine):
        """Test concurrent optimization analysis"""
        tasks = []
        
        for i in range(5):
            task = optimization_engine.analyze_optimization_opportunities(
                tenant_id=f'concurrent-tenant-{i}'
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check all analyses succeeded
        successful_analyses = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_analyses) == 5
        
        # Verify all results are valid
        for analysis in successful_analyses:
            assert 'tenant_id' in analysis
            assert 'total_opportunities' in analysis


class TestMLIntegrationPerformance:
    """Performance tests for ML engines"""
    
    async def test_prediction_performance(self, prediction_engine):
        """Test prediction engine performance"""
        start_time = datetime.utcnow()
        
        # Run multiple predictions
        tasks = []
        for i in range(20):
            task = prediction_engine.predict_health_score(
                component_id=f'perf-component-{i}',
                tenant_id='performance-test'
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Performance assertions
        assert duration < 10  # Should complete within 10 seconds
        assert len(results) == 20
        
        # Check throughput
        throughput = len(results) / duration
        assert throughput > 2  # At least 2 predictions per second
    
    async def test_analytics_performance(self, prediction_engine):
        """Test analytics engine performance"""
        analytics_engine = AdvancedAnalyticsEngine(prediction_engine)
        
        start_time = datetime.utcnow()
        
        insights = await analytics_engine.generate_health_insights(
            'performance-test', 168
        )
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete analytics within reasonable time
        assert duration < 15  # 15 seconds max for comprehensive insights
        assert 'tenant_id' in insights
    
    async def test_optimization_performance(self, optimization_engine):
        """Test optimization engine performance"""
        start_time = datetime.utcnow()
        
        analysis = await optimization_engine.analyze_optimization_opportunities(
            'performance-test'
        )
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete optimization analysis quickly
        assert duration < 5  # 5 seconds max
        assert 'optimizations' in analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])