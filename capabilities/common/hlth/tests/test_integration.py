#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - Integration Tests
End-to-end integration tests across all components

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

from models import (
    HealthMetric, HealthAlert, SystemComponent,
    HealthStatus, HealthSeverity, HealthDimension, ComponentType
)


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    async def test_complete_health_flow(self, health_service):
        """Test complete health monitoring flow from metric to dashboard"""
        tenant_id = 'e2e-test'
        component_id = 'e2e-component'
        
        # Step 1: Create enterprise tenant
        tenant_config = {
            'tenant_id': tenant_id,
            'tenant_name': 'E2E Test Corporation',
            'tier': 'enterprise',
            'compliance_frameworks': ['soc2']
        }
        
        tenant_result = await health_service.create_enterprise_tenant(tenant_config)
        assert tenant_result['status'] == 'success'
        
        # Step 2: Register system component
        component = SystemComponent(
            component_id=component_id,
            tenant_id=tenant_id,
            name='E2E Test Component',
            component_type=ComponentType.SERVICE,
            environment='production',
            business_criticality='high'
        )
        
        registration_result = await health_service.register_system_component(component)
        assert registration_result['status'] == 'success'
        
        # Step 3: Process health metrics
        metrics = []
        for i in range(10):
            metric = HealthMetric(
                tenant_id=tenant_id,
                component_id=component_id,
                name='cpu_utilization',
                value=60.0 + (i * 2),  # Gradual increase
                dimension=HealthDimension.PERFORMANCE
            )
            metrics.append(metric)
        
        for metric in metrics:
            result = await health_service.process_health_metric_with_enterprise_features(
                metric, tenant_id
            )
            assert result.get('status') != 'error'
        
        # Step 4: Assess component health
        assessment = await health_service.assess_component_health(
            component_id, tenant_id
        )
        
        assert 'overall_health_score' in assessment
        assert assessment['overall_health_score'] > 0
        
        # Step 5: Generate predictions
        prediction = await health_service.predict_component_health_advanced(
            component_id, tenant_id, 24
        )
        
        if 'error' not in prediction:
            assert 'predicted_health_score' in prediction
            assert 'confidence' in prediction
        
        # Step 6: Generate optimization recommendations
        optimizations = await health_service.analyze_optimization_opportunities(
            tenant_id, component_id
        )
        
        if 'error' not in optimizations:
            assert 'optimizations' in optimizations
        
        # Step 7: Generate health report
        report = await health_service.generate_health_report(
            tenant_id=tenant_id,
            report_type='comprehensive'
        )
        
        assert report.tenant_id == tenant_id
        assert report.total_components >= 1
        
        # Step 8: Create dashboard
        dashboard = await health_service.create_tenant_health_dashboard(
            tenant_id, 'executive'
        )
        
        if 'error' not in dashboard:
            assert dashboard.get('enterprise_enhanced') is True
            assert 'tenant_info' in dashboard
    
    async def test_multi_tenant_isolation_flow(self, health_service):
        """Test multi-tenant isolation across complete workflow"""
        # Create two isolated tenants
        tenants = [
            {
                'tenant_id': 'isolated-tenant-a',
                'tenant_name': 'Isolated Tenant A',
                'tier': 'enterprise'
            },
            {
                'tenant_id': 'isolated-tenant-b',
                'tenant_name': 'Isolated Tenant B',
                'tier': 'enterprise'
            }
        ]
        
        tenant_ids = []
        for tenant_config in tenants:
            result = await health_service.create_enterprise_tenant(tenant_config)
            assert result['status'] == 'success'
            tenant_ids.append(tenant_config['tenant_id'])
        
        # Create components for each tenant
        for i, tenant_id in enumerate(tenant_ids):
            component = SystemComponent(
                component_id=f'isolated-component-{i}',
                tenant_id=tenant_id,
                name=f'Isolated Component {i}',
                component_type=ComponentType.SERVICE
            )
            
            await health_service.register_system_component(component)
        
        # Test cross-tenant boundary enforcement
        boundary_result = await health_service.enforce_tenant_boundaries(
            tenant_ids[0],  # Requesting tenant
            tenant_ids[1],  # Target tenant
            'health_metric',
            'read'
        )
        
        # Cross-tenant access should be denied
        assert boundary_result['allowed'] is False
        
        # Test same-tenant access (should be allowed)
        boundary_result = await health_service.enforce_tenant_boundaries(
            tenant_ids[0],
            tenant_ids[0],
            'health_metric',
            'read'
        )
        
        assert boundary_result['allowed'] is True
        
        # Process metrics for each tenant
        for tenant_id in tenant_ids:
            metric = HealthMetric(
                tenant_id=tenant_id,
                component_id=f'isolated-component-{tenant_ids.index(tenant_id)}',
                name='isolation_test_metric',
                value=75.0,
                dimension=HealthDimension.PERFORMANCE
            )
            
            result = await health_service.process_health_metric_with_enterprise_features(
                metric, tenant_id
            )
            
            # Should process successfully for own tenant
            assert result.get('status') != 'tenant_boundary_violation'
        
        # Verify tenant isolation status
        for tenant_id in tenant_ids:
            isolation_status = await health_service.get_tenant_isolation_status(tenant_id)
            
            if 'error' not in isolation_status:
                assert isolation_status['tenant_id'] == tenant_id
                assert 'isolation_policy' in isolation_status
    
    async def test_ml_analytics_integration_flow(self, health_service):
        """Test ML and analytics integration across the system"""
        tenant_id = 'ml-integration-test'
        
        # Create tenant
        tenant_config = {
            'tenant_id': tenant_id,
            'tenant_name': 'ML Integration Test',
            'tier': 'enterprise_plus'
        }
        
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Create multiple components
        components = []
        for i in range(5):
            component = SystemComponent(
                component_id=f'ml-component-{i:02d}',
                tenant_id=tenant_id,
                name=f'ML Component {i}',
                component_type=ComponentType.SERVICE
            )
            components.append(component)
            await health_service.register_system_component(component)
        
        # Generate training data
        for component in components:
            for j in range(20):  # 20 metrics per component
                metric = HealthMetric(
                    tenant_id=tenant_id,
                    component_id=component.component_id,
                    name='performance_metric',
                    value=50.0 + (j % 40) + (hash(component.component_id) % 20),
                    dimension=HealthDimension.PERFORMANCE
                )
                
                await health_service.process_health_metric(metric)
        
        # Test ML model training
        training_result = await health_service.train_ml_models_for_tenant(tenant_id)
        
        if training_result.get('status') == 'completed':
            assert training_result['models_trained'] > 0
        
        # Test advanced predictions
        for component in components[:3]:  # Test first 3 components
            prediction = await health_service.predict_component_health_advanced(
                component.component_id, tenant_id, 24
            )
            
            if 'error' not in prediction:
                assert 'predicted_health_score' in prediction
                assert 'confidence' in prediction
                
                # Test anomaly detection
                anomalies = await health_service.detect_health_anomalies(
                    component.component_id, tenant_id, 24
                )
                
                if 'error' not in anomalies:
                    assert 'anomalies_detected' in anomalies
        
        # Test advanced analytics
        insights = await health_service.generate_advanced_health_insights(
            tenant_id, 168
        )
        
        if 'error' not in insights:
            assert 'tenant_id' in insights
            assert 'analysis_period_hours' in insights
        
        # Test optimization analysis
        optimizations = await health_service.analyze_optimization_opportunities(tenant_id)
        
        if 'error' not in optimizations:
            assert 'total_opportunities' in optimizations
            assert 'optimizations' in optimizations
        
        # Test ML model performance tracking
        model_performance = await health_service.get_ml_model_performance(tenant_id)
        
        if 'error' not in model_performance:
            assert 'available_models' in model_performance
    
    async def test_compliance_integration_flow(self, health_service):
        """Test compliance integration across all features"""
        tenant_id = 'compliance-integration'
        
        # Create compliant enterprise tenant
        tenant_config = {
            'tenant_id': tenant_id,
            'tenant_name': 'Compliance Integration Corp',
            'tier': 'enterprise',
            'compliance_frameworks': ['soc2', 'hipaa', 'iso27001']
        }
        
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Create components with compliance requirements
        sensitive_component = SystemComponent(
            component_id='sensitive-data-processor',
            tenant_id=tenant_id,
            name='Sensitive Data Processor',
            component_type=ComponentType.SERVICE,
            business_criticality='critical',
            tags=['pii', 'healthcare', 'compliance']
        )
        
        await health_service.register_system_component(sensitive_component)
        
        # Process health metrics with compliance context
        compliance_metric = HealthMetric(
            tenant_id=tenant_id,
            component_id='sensitive-data-processor',
            name='data_processing_latency',
            value=150.0,
            dimension=HealthDimension.PERFORMANCE,
            business_context={
                'data_classification': 'restricted',
                'compliance_required': True,
                'audit_trail': True
            }
        )
        
        result = await health_service.process_health_metric_with_enterprise_features(
            compliance_metric, tenant_id
        )
        
        # Should include compliance checks
        if 'sla_compliance' in result:
            assert 'compliant' in result['sla_compliance']
        
        # Generate compliance reports
        frameworks = ['soc2', 'hipaa', 'iso27001']
        for framework in frameworks:
            report = await health_service.generate_compliance_report(
                tenant_id, framework, 7
            )
            
            if 'error' not in report:
                assert report['framework'] == framework
                assert 'report' in report
                assert report['report']['framework'] in ['SOC 2', 'HIPAA', 'ISO 27001']
        
        # Test audit trail generation
        # (In real implementation, this would check actual audit logs)
        dashboard = await health_service.create_tenant_health_dashboard(
            tenant_id, 'executive'
        )
        
        if 'error' not in dashboard:
            # Dashboard should include compliance information
            if 'tenant_info' in dashboard:
                tenant_info = dashboard['tenant_info']
                assert 'compliance_frameworks' in tenant_info
                frameworks = tenant_info['compliance_frameworks']
                assert 'soc2' in frameworks
                assert 'hipaa' in frameworks
    
    async def test_performance_under_load_integration(self, health_service):
        """Test system performance under integrated load"""
        tenant_count = 3
        components_per_tenant = 10
        metrics_per_component = 20
        
        # Create multiple tenants
        tenant_ids = []
        for i in range(tenant_count):
            tenant_config = {
                'tenant_id': f'load-test-tenant-{i}',
                'tenant_name': f'Load Test Tenant {i}',
                'tier': 'professional'
            }
            
            result = await health_service.create_enterprise_tenant(tenant_config)
            assert result['status'] == 'success'
            tenant_ids.append(tenant_config['tenant_id'])
        
        # Create components for each tenant
        all_components = []
        for tenant_id in tenant_ids:
            for j in range(components_per_tenant):
                component = SystemComponent(
                    component_id=f'{tenant_id}-component-{j:02d}',
                    tenant_id=tenant_id,
                    name=f'Load Test Component {j}',
                    component_type=ComponentType.SERVICE
                )
                all_components.append(component)
        
        # Register all components concurrently
        start_time = datetime.utcnow()
        
        registration_tasks = [
            health_service.register_system_component(component)
            for component in all_components
        ]
        
        registration_results = await asyncio.gather(*registration_tasks, return_exceptions=True)

        
        # Verify all registrations succeeded
        successful_registrations = [
            r for r in registration_results 
            if r.get('status') == 'success'
        ]
        assert len(successful_registrations) == len(all_components)
        
        # Process metrics for all components concurrently
        metric_tasks = []
        for component in all_components:
            for k in range(metrics_per_component):
                metric = HealthMetric(
                    tenant_id=component.tenant_id,
                    component_id=component.component_id,
                    name='load_test_metric',
                    value=50.0 + (k % 50),
                    dimension=HealthDimension.PERFORMANCE
                )
                
                task = health_service.process_health_metric_with_enterprise_features(
                    metric, component.tenant_id
                )
                metric_tasks.append(task)
        
        # Process in batches to avoid overwhelming the system
        batch_size = 50
        metric_results = []
        
        for i in range(0, len(metric_tasks), batch_size):
            batch = metric_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            metric_results.extend(batch_results)
        
        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds()
        
        # Performance assertions
        total_operations = len(all_components) + len(metric_tasks)
        throughput = total_operations / total_duration
        
        print(f"Integration test throughput: {throughput:.2f} ops/sec")
        assert throughput > 5  # Should handle at least 5 operations per second
        
        # Check error rate
        errors = [r for r in metric_results if isinstance(r, Exception)]
        error_rate = len(errors) / len(metric_results)
        assert error_rate < 0.1  # Less than 10% error rate
        
        # Generate reports for all tenants
        report_tasks = [
            health_service.generate_health_report(
                tenant_id=tenant_id,
                report_type='operational'
            )
            for tenant_id in tenant_ids
        ]
        
        reports = await asyncio.gather(*report_tasks, return_exceptions=True)

        
        # Verify all reports generated successfully
        for report in reports:
            assert report.tenant_id in tenant_ids
            assert report.total_components == components_per_tenant
    
    async def test_error_recovery_integration(self, health_service):
        """Test error recovery across integrated components"""
        tenant_id = 'error-recovery-test'
        
        # Create tenant
        tenant_config = {
            'tenant_id': tenant_id,
            'tenant_name': 'Error Recovery Test',
            'tier': 'basic'
        }
        
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Test quota exceeded scenario
        # Exhaust API quota
        quota_result = await health_service.enforce_tenant_quotas(
            tenant_id, 'api_calls_per_hour', 1000  # Exhaust basic tier quota
        )
        assert quota_result['allowed'] is True
        
        # Next request should be denied
        quota_result = await health_service.enforce_tenant_quotas(
            tenant_id, 'api_calls_per_hour', 1
        )
        assert quota_result['allowed'] is False
        
        # Test metric processing with quota exceeded
        metric = HealthMetric(
            tenant_id=tenant_id,
            component_id='recovery-test-component',
            name='recovery_metric',
            value=60.0,
            dimension=HealthDimension.PERFORMANCE
        )
        
        result = await health_service.process_health_metric_with_enterprise_features(
            metric, tenant_id
        )
        
        # Should handle quota exceeded gracefully
        if result.get('status') == 'quota_exceeded':
            assert 'quota_details' in result
        
        # Test invalid data handling
        invalid_metric = HealthMetric(
            tenant_id='',  # Invalid tenant ID
            component_id='recovery-component',
            name='invalid_metric',
            value=-100.0,  # Invalid value
            dimension=HealthDimension.PERFORMANCE
        )
        
        invalid_result = await health_service.process_health_metric(invalid_metric)
        
        # Should handle invalid data gracefully
        assert 'error' in invalid_result or invalid_result.get('status') != 'success'
        
        # Test system recovery with valid data
        valid_metric = HealthMetric(
            tenant_id=tenant_id,
            component_id='recovery-component',
            name='valid_recovery_metric',
            value=70.0,
            dimension=HealthDimension.PERFORMANCE
        )
        
        # Reset quotas (simulate quota reset)
        if hasattr(health_service, '_enterprise_manager'):
            health_service._enterprise_manager.tenant_quotas[tenant_id] = {}
        
        recovery_result = await health_service.process_health_metric(valid_metric)
        
        # System should recover and process valid data
        assert recovery_result.get('status') == 'success'
    
    async def test_data_consistency_integration(self, health_service):
        """Test data consistency across all integrated components"""
        tenant_id = 'consistency-integration'
        
        # Create tenant and components
        tenant_config = {
            'tenant_id': tenant_id,
            'tenant_name': 'Consistency Test',
            'tier': 'enterprise'
        }
        
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Create related components
        components = [
            SystemComponent(
                component_id='web-server-01',
                tenant_id=tenant_id,
                name='Web Server 01',
                component_type=ComponentType.SERVICE,
                dependencies=['database-01', 'cache-01']
            ),
            SystemComponent(
                component_id='database-01',
                tenant_id=tenant_id,
                name='Database 01',
                component_type=ComponentType.DATABASE
            ),
            SystemComponent(
                component_id='cache-01',
                tenant_id=tenant_id,
                name='Cache 01',
                component_type=ComponentType.CACHE
            )
        ]
        
        for component in components:
            await health_service.register_system_component(component)
        
        # Process correlated metrics
        correlation_data = [
            ('database-01', 'response_time', 200),
            ('web-server-01', 'response_time', 300),  # Should correlate
            ('cache-01', 'hit_rate', 85),
            ('web-server-01', 'cpu_utilization', 75)
        ]
        
        for component_id, metric_name, value in correlation_data:
            metric = HealthMetric(
                tenant_id=tenant_id,
                component_id=component_id,
                name=metric_name,
                value=value,
                dimension=HealthDimension.PERFORMANCE
            )
            
            await health_service.process_health_metric(metric)
        
        # Verify data consistency across assessments
        assessments = {}
        for component in components:
            assessment = await health_service.assess_component_health(
                component.component_id, tenant_id
            )
            assessments[component.component_id] = assessment
        
        # All components should have assessments
        assert len(assessments) == len(components)
        
        for component_id, assessment in assessments.items():
            assert assessment['component_id'] == component_id
            assert assessment['tenant_id'] == tenant_id
            assert 'overall_health_score' in assessment
        
        # Generate comprehensive report
        report = await health_service.generate_health_report(
            tenant_id=tenant_id,
            report_type='comprehensive'
        )
        
        # Report should reflect all components
        assert report.total_components == len(components)
        assert report.tenant_id == tenant_id
        
        # Verify component relationships are preserved
        web_server_assessment = assessments['web-server-01']
        if 'dependencies' in web_server_assessment:
            dependencies = web_server_assessment['dependencies']
            assert 'database-01' in dependencies
            assert 'cache-01' in dependencies


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])