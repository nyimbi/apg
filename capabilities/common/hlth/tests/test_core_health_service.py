#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - Core Service Tests
Comprehensive tests for core health management functionality

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from conftest import (
    assert_health_metric_valid, generate_mock_metrics, 
    generate_mock_components
)
from models import (
    HealthMetric, HealthAlert, SystemComponent,
    HealthStatus, HealthSeverity, HealthDimension, ComponentType
)


class TestSystemHealthService:
    """Test suite for SystemHealthService core functionality"""
    
    async def test_service_initialization(self, health_service):
        """Test health service initialization"""
        assert health_service.initialized is True
        assert health_service.service_id is not None
        assert health_service.startup_time is not None
        
        # Check core components are initialized
        assert hasattr(health_service, '_health_baselines')
        assert hasattr(health_service, '_component_registry')
        assert hasattr(health_service, '_health_rules')
        assert hasattr(health_service, '_active_alerts')
    
    async def test_health_metric_processing(self, health_service, sample_health_metric):
        """Test health metric processing"""
        result = await health_service.process_health_metric(sample_health_metric)
        
        assert result['status'] == 'success'
        assert 'health_score' in result
        assert 'processing_time_ms' in result
        assert 'alert_triggered' in result
        
        # Verify metric is stored
        stored_metrics = health_service._health_metrics.get(
            sample_health_metric.tenant_id, {}
        )
        assert sample_health_metric.component_id in stored_metrics
    
    async def test_component_health_assessment(self, health_service, sample_system_component):
        """Test component health assessment"""
        # Register component first
        await health_service.register_system_component(sample_system_component)
        
        # Assess component health
        assessment = await health_service.assess_component_health(
            sample_system_component.component_id,
            sample_system_component.tenant_id
        )
        
        assert 'overall_health_score' in assessment
        assert 'health_status' in assessment
        assert 'assessment_timestamp' in assessment
        assert 'dimension_scores' in assessment
        
        # Check health score is valid
        score = assessment['overall_health_score']
        assert 0 <= score <= 100
    
    async def test_alert_generation_and_processing(self, health_service, sample_health_metric):
        """Test alert generation from health metrics"""
        # Use a high value to trigger an alert
        sample_health_metric.value = 95.0
        
        result = await health_service.process_health_metric(sample_health_metric)
        
        # Should trigger alert for high CPU
        if result.get('alert_triggered'):
            alerts = health_service._active_alerts.get(sample_health_metric.tenant_id, {})
            assert len(alerts) > 0
            
            # Check alert properties
            alert = list(alerts.values())[0]
            assert alert.severity in [HealthSeverity.HIGH, HealthSeverity.CRITICAL]
            assert alert.component_id == sample_health_metric.component_id
    
    async def test_health_baseline_establishment(self, health_service):
        """Test health baseline establishment"""
        tenant_id = "baseline-test"
        component_id = "baseline-component"
        
        # Generate historical metrics
        metrics = []
        for i in range(50):  # Enough for baseline
            metric = HealthMetric(
                tenant_id=tenant_id,
                component_id=component_id,
                name="cpu_utilization",
                value=50.0 + (i % 10),  # Stable around 50%
                dimension=HealthDimension.PERFORMANCE
            )
            metrics.append(metric)
            await health_service.process_health_metric(metric)
        
        # Establish baseline
        baseline = await health_service._establish_component_baseline(
            component_id, tenant_id, "cpu_utilization"
        )
        
        assert baseline is not None
        assert 45 <= baseline.baseline_value <= 55  # Should be around 50
        assert baseline.confidence_level > 0.8
    
    async def test_predictive_health_analysis(self, health_service, sample_system_component):
        """Test predictive health analysis"""
        # Register component
        await health_service.register_system_component(sample_system_component)
        
        # Generate trend data
        for i in range(10):
            metric = HealthMetric(
                tenant_id=sample_system_component.tenant_id,
                component_id=sample_system_component.component_id,
                name="memory_utilization",
                value=60.0 + i * 2,  # Increasing trend
                dimension=HealthDimension.PERFORMANCE
            )
            await health_service.process_health_metric(metric)
        
        # Get prediction
        prediction = await health_service.predict_component_health(
            sample_system_component.component_id,
            sample_system_component.tenant_id,
            prediction_window_hours=24
        )
        
        assert 'predicted_health_score' in prediction
        assert 'confidence' in prediction
        assert 'risk_factors' in prediction
        assert prediction['predicted_health_score'] <= 100
    
    async def test_multi_dimensional_health_analysis(self, health_service, sample_system_component):
        """Test multi-dimensional health analysis"""
        # Register component
        await health_service.register_system_component(sample_system_component)
        
        # Add metrics across dimensions
        dimensions = [
            (HealthDimension.PERFORMANCE, "cpu_utilization", 80),
            (HealthDimension.AVAILABILITY, "uptime_percentage", 99.5),
            (HealthDimension.SECURITY, "security_score", 85),
        ]
        
        for dimension, name, value in dimensions:
            metric = HealthMetric(
                tenant_id=sample_system_component.tenant_id,
                component_id=sample_system_component.component_id,
                name=name,
                value=value,
                dimension=dimension
            )
            await health_service.process_health_metric(metric)
        
        # Analyze multi-dimensional health
        analysis = await health_service.analyze_multi_dimensional_health(
            sample_system_component.tenant_id
        )
        
        assert 'dimension_scores' in analysis
        assert 'overall_health_score' in analysis
        assert 'correlations' in analysis
        
        # Check all dimensions are represented
        dimension_names = [d.value for d in HealthDimension]
        for dim_name in dimension_names:
            assert dim_name in analysis['dimension_scores']
    
    async def test_autonomous_remediation(self, health_service, sample_health_alert):
        """Test autonomous remediation system"""
        # Process alert that should trigger remediation
        remediation_result = await health_service._trigger_autonomous_remediation(
            sample_health_alert
        )
        
        assert 'remediation_actions' in remediation_result
        assert 'success' in remediation_result
        
        if remediation_result['success']:
            actions = remediation_result['remediation_actions']
            assert len(actions) > 0
            assert all('action_type' in action for action in actions)
    
    async def test_health_report_generation(self, health_service):
        """Test health report generation"""
        tenant_id = "report-test"
        
        # Generate test data
        components = generate_mock_components(3, tenant_id)
        for component in components:
            await health_service.register_system_component(component)
        
        metrics = generate_mock_metrics(20, tenant_id)
        for metric in metrics:
            await health_service.process_health_metric(metric)
        
        # Generate report
        report = await health_service.generate_health_report(
            tenant_id=tenant_id,
            report_type='comprehensive',
            time_period_hours=24
        )
        
        assert report.report_id is not None
        assert report.tenant_id == tenant_id
        assert report.overall_health_score >= 0
        assert report.total_components == len(components)
        assert len(report.recommendations) > 0
    
    async def test_component_discovery(self, health_service):
        """Test component discovery functionality"""
        discovery_result = await health_service._discover_system_components()
        
        assert 'discovered_components' in discovery_result.__dict__
        assert 'discovery_timestamp' in discovery_result.__dict__
        assert 'discovery_method' in discovery_result.__dict__
        
        # Components should be registered
        components = discovery_result.discovered_components
        for component in components:
            assert component.component_id is not None
            assert component.tenant_id is not None
    
    async def test_health_correlation_analysis(self, health_service):
        """Test health correlation analysis"""
        tenant_id = "correlation-test"
        
        # Create correlated components
        components = [
            SystemComponent(
                component_id="web-server",
                tenant_id=tenant_id,
                name="Web Server",
                component_type=ComponentType.SERVICE,
                dependencies=["database"]
            ),
            SystemComponent(
                component_id="database", 
                tenant_id=tenant_id,
                name="Database",
                component_type=ComponentType.DATABASE
            )
        ]
        
        for component in components:
            await health_service.register_system_component(component)
        
        # Generate correlated health events
        for i in range(10):
            # Database issues affect web server
            db_metric = HealthMetric(
                tenant_id=tenant_id,
                component_id="database",
                name="response_time",
                value=100 + i * 10,  # Degrading
                dimension=HealthDimension.PERFORMANCE
            )
            
            web_metric = HealthMetric(
                tenant_id=tenant_id,
                component_id="web-server", 
                name="response_time",
                value=50 + i * 15,  # Also degrading
                dimension=HealthDimension.PERFORMANCE
            )
            
            await health_service.process_health_metric(db_metric)
            await health_service.process_health_metric(web_metric)
        
        # Analyze correlations
        correlations = await health_service._analyze_health_correlations(tenant_id)
        
        assert len(correlations) > 0
        correlation = correlations[0]
        assert correlation.correlation_strength > 0.5
        assert {"web-server", "database"}.issubset({correlation.component_a_id, correlation.component_b_id})
    
    async def test_performance_with_load(self, health_service, performance_test_data):
        """Test service performance under load"""
        start_time = datetime.utcnow()
        
        # Process components
        for component in performance_test_data['components']:
            await health_service.register_system_component(component)
        
        # Process metrics in batches
        batch_size = 10
        metrics = performance_test_data['metrics']
        
        for i in range(0, len(metrics), batch_size):
            batch = metrics[i:i + batch_size]
            tasks = [
                health_service.process_health_metric(metric)
                for metric in batch
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Performance assertions
        assert processing_time < 60  # Should complete within 60 seconds
        
        # Check throughput
        total_operations = len(performance_test_data['components']) + len(metrics)
        throughput = total_operations / processing_time
        assert throughput > 10  # At least 10 operations per second
    
    async def test_concurrent_operations(self, health_service):
        """Test concurrent health operations"""
        tenant_id = "concurrent-test"
        
        # Create concurrent tasks
        tasks = []
        
        # Concurrent metric processing
        for i in range(20):
            metric = HealthMetric(
                tenant_id=tenant_id,
                component_id=f"concurrent-component-{i % 5}",
                name="test_metric",
                value=50.0 + i,
                dimension=HealthDimension.PERFORMANCE
            )
            tasks.append(health_service.process_health_metric(metric))
        
        # Concurrent component registration
        for i in range(5):
            component = SystemComponent(
                component_id=f"concurrent-component-{i}",
                tenant_id=tenant_id,
                name=f"Concurrent Component {i}",
                component_type=ComponentType.SERVICE
            )
            tasks.append(health_service.register_system_component(component))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        
        # Verify successful processing
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(tasks)
    
    async def test_error_handling_and_recovery(self, health_service):
        """Test error handling and recovery mechanisms"""
        # Test invalid metric processing
        invalid_metric = HealthMetric(
            tenant_id="",  # Invalid empty tenant
            component_id="test",
            name="test_metric",
            value=50.0,
            dimension=HealthDimension.PERFORMANCE
        )
        
        result = await health_service.process_health_metric(invalid_metric)
        assert 'error' in result or result['status'] != 'success'
        
        # Test service recovery after error
        valid_metric = HealthMetric(
            tenant_id="recovery-test",
            component_id="test-component",
            name="recovery_metric",
            value=75.0,
            dimension=HealthDimension.PERFORMANCE
        )
        
        recovery_result = await health_service.process_health_metric(valid_metric)
        assert recovery_result['status'] == 'success'
    
    async def test_data_consistency(self, health_service):
        """Test data consistency across operations"""
        tenant_id = "consistency-test"
        component_id = "consistency-component"
        
        # Register component
        component = SystemComponent(
            component_id=component_id,
            tenant_id=tenant_id,
            name="Consistency Test Component",
            component_type=ComponentType.SERVICE
        )
        await health_service.register_system_component(component)
        
        # Process multiple metrics
        metric_values = [60, 70, 80, 75, 85]
        for i, value in enumerate(metric_values):
            metric = HealthMetric(
                tenant_id=tenant_id,
                component_id=component_id,
                name="consistency_metric",
                value=value,
                dimension=HealthDimension.PERFORMANCE
            )
            await health_service.process_health_metric(metric)
        
        # Verify data consistency
        assessment = await health_service.assess_component_health(component_id, tenant_id)
        
        # Component should exist in registry
        assert tenant_id in health_service._component_registry
        assert component_id in health_service._component_registry[tenant_id]
        
        # Metrics should be stored
        assert tenant_id in health_service._health_metrics
        assert component_id in health_service._health_metrics[tenant_id]
        
        # Health score should reflect recent metrics
        assert assessment['overall_health_score'] > 0


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    async def test_metric_processing_throughput(self, health_service):
        """Benchmark metric processing throughput"""
        start_time = datetime.utcnow()
        metric_count = 1000
        
        tasks = []
        for i in range(metric_count):
            metric = HealthMetric(
                tenant_id="benchmark",
                component_id=f"bench-component-{i % 10}",
                name="benchmark_metric",
                value=50.0 + (i % 50),
                dimension=HealthDimension.PERFORMANCE
            )
            tasks.append(health_service.process_health_metric(metric))
        
        # Process in batches to avoid overwhelming
        batch_size = 100
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch, return_exceptions=True)

        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        throughput = metric_count / duration
        
        print(f"Metric processing throughput: {throughput:.2f} metrics/second")
        assert throughput > 50  # Should process at least 50 metrics per second
    
    async def test_memory_usage_stability(self, health_service):
        """Test memory usage stability under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process significant load
        for batch in range(10):
            tasks = []
            for i in range(100):
                metric = HealthMetric(
                    tenant_id="memory-test",
                    component_id=f"memory-component-{i}",
                    name="memory_metric",
                    value=60.0,
                    dimension=HealthDimension.PERFORMANCE
                )
                tasks.append(health_service.process_health_metric(metric))
            
            await asyncio.gather(*tasks, return_exceptions=True)

            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            # Memory shouldn't increase excessively
            assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"
        
        print(f"Memory usage: {initial_memory:.2f}MB -> {current_memory:.2f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])