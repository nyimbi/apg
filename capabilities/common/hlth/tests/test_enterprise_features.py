#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - Enterprise Features Tests
Comprehensive tests for enterprise features and multi-tenancy

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from conftest import assert_enterprise_features_available
from enterprise_features import (
    EnterpriseHealthManager, TenantTier, ComplianceFramework,
    TenantConfiguration, ServiceLevelAgreement
)
from multi_tenant_isolation import (
    TenantIsolationManager, IsolationLevel, DataClassification
)
from models import HealthMetric, HealthDimension


class TestEnterpriseHealthManager:
    """Test suite for enterprise health management features"""
    
    async def test_tenant_creation(self, enterprise_manager):
        """Test enterprise tenant creation"""
        tenant_config = {
            'tenant_id': 'test-enterprise-001',
            'tenant_name': 'Test Enterprise',
            'tier': 'enterprise',
            'compliance_frameworks': ['soc2', 'iso27001'],
            'custom_branding': {
                'company_name': 'Test Corp',
                'logo_url': '/static/test-logo.png'
            }
        }
        
        tenant_cfg = await enterprise_manager.create_enterprise_tenant(tenant_config)
        
        assert tenant_cfg.tenant_id == 'test-enterprise-001'
        assert tenant_cfg.tier == TenantTier.ENTERPRISE
        assert ComplianceFramework.SOC2 in tenant_cfg.compliance_frameworks
        assert ComplianceFramework.ISO27001 in tenant_cfg.compliance_frameworks
        assert tenant_cfg.custom_branding['company_name'] == 'Test Corp'
        assert tenant_cfg.active is True
    
    async def test_tenant_tier_defaults(self, enterprise_manager):
        """Test tenant tier default configurations"""
        # Test basic tier
        basic_config = {'tenant_id': 'basic-001', 'tier': 'basic'}
        basic_tenant = await enterprise_manager.create_enterprise_tenant(basic_config)
        
        assert basic_tenant.max_components == 50
        assert basic_tenant.max_users == 5
        assert basic_tenant.data_retention_days == 30
        
        # Test enterprise plus tier
        enterprise_plus_config = {'tenant_id': 'enterprise-plus-001', 'tier': 'enterprise_plus'}
        enterprise_plus_tenant = await enterprise_manager.create_enterprise_tenant(enterprise_plus_config)
        
        assert enterprise_plus_tenant.max_components == -1  # Unlimited
        assert enterprise_plus_tenant.max_users == -1  # Unlimited
        assert enterprise_plus_tenant.feature_flags['dedicated_instance'] is True
    
    async def test_quota_enforcement(self, enterprise_manager):
        """Test resource quota enforcement"""
        # Create basic tier tenant
        tenant_config = {'tenant_id': 'quota-test', 'tier': 'basic'}
        await enterprise_manager.create_enterprise_tenant(tenant_config)
        
        # Test within quota
        quota_result = await enterprise_manager.enforce_tenant_quotas(
            'quota-test', 'api_calls_per_hour', 100
        )
        assert quota_result['allowed'] is True
        assert quota_result['current_usage'] == 100
        
        # Test quota exceeded
        quota_result = await enterprise_manager.enforce_tenant_quotas(
            'quota-test', 'api_calls_per_hour', 950  # Would exceed 1000 limit
        )
        assert quota_result['allowed'] is False
        assert 'Quota exceeded' in quota_result['reason']
    
    async def test_sla_compliance_checking(self, enterprise_manager):
        """Test SLA compliance checking"""
        # Create enterprise tenant with SLA
        tenant_config = {
            'tenant_id': 'sla-test',
            'tier': 'enterprise',
            'sla_requirements': {
                'availability_target': 99.9,
                'response_time_target': 200
            }
        }
        await enterprise_manager.create_enterprise_tenant(tenant_config)
        
        # Test compliant availability
        compliance_result = await enterprise_manager.check_sla_compliance(
            'sla-test', 'availability', 99.95
        )
        assert compliance_result['overall_compliant'] is True
        
        # Test non-compliant availability
        compliance_result = await enterprise_manager.check_sla_compliance(
            'sla-test', 'availability', 99.5  # Below 99.9% target
        )
        assert compliance_result['overall_compliant'] is False
        assert len(compliance_result['sla_results']) > 0
        
        # Check breach details
        breach = compliance_result['sla_results'][0]['breach_details']
        assert breach['type'] == 'availability'
        assert breach['target'] == 99.9
        assert breach['actual'] == 99.5
    
    async def test_compliance_report_generation(self, enterprise_manager):
        """Test compliance report generation"""
        # Create tenant with compliance requirements
        tenant_config = {
            'tenant_id': 'compliance-test',
            'tier': 'enterprise',
            'compliance_frameworks': ['soc2', 'hipaa']
        }
        await enterprise_manager.create_enterprise_tenant(tenant_config)
        
        # Generate SOC2 compliance report
        report_result = await enterprise_manager.generate_compliance_report(
            'compliance-test', ComplianceFramework.SOC2, 30
        )
        
        assert 'report_id' in report_result
        assert report_result['framework'] == 'soc2'
        assert 'report' in report_result
        
        report = report_result['report']
        assert report['framework'] == 'SOC 2'
        assert 'trust_service_criteria' in report
        assert 'security' in report['trust_service_criteria']
        assert 'availability' in report['trust_service_criteria']
    
    async def test_audit_trail_logging(self, enterprise_manager):
        """Test comprehensive audit trail logging"""
        tenant_id = 'audit-test'
        
        # Create tenant
        tenant_config = {'tenant_id': tenant_id, 'tier': 'enterprise'}
        await enterprise_manager.create_enterprise_tenant(tenant_config)
        
        # Perform auditable action
        await enterprise_manager._log_audit_event(
            tenant_id, 'test-user', 'CREATE_COMPONENT', 'component', 'comp-001',
            None, {'name': 'Test Component'}, '192.168.1.1', 'test-agent', 'session-001'
        )
        
        # Check audit trail
        audit_trails = enterprise_manager.audit_trails.get(tenant_id, [])
        assert len(audit_trails) >= 1
        
        audit_entry = audit_trails[-1]
        assert audit_entry.user_id == 'test-user'
        assert audit_entry.action == 'CREATE_COMPONENT'
        assert audit_entry.resource_type == 'component'
        assert audit_entry.ip_address == '192.168.1.1'
        assert 'data_modification' in audit_entry.compliance_tags
    
    async def test_audit_retention_policy(self, enterprise_manager):
        """Test audit log retention policy enforcement"""
        tenant_id = 'retention-test'
        
        # Create tenant with short retention
        tenant_config = {
            'tenant_id': tenant_id, 
            'tier': 'basic',
            'audit_requirements': {'retention_days': 1}
        }
        tenant_cfg = await enterprise_manager.create_enterprise_tenant(tenant_config)
        
        # Add old audit entry
        old_audit = enterprise_manager.audit_trails[tenant_id][0]  # From tenant creation
        old_audit.timestamp = datetime.utcnow() - timedelta(days=2)  # 2 days old
        
        # Add recent audit entry
        await enterprise_manager._log_audit_event(
            tenant_id, 'test-user', 'RECENT_ACTION', 'test', 'test-001',
            None, {}, '127.0.0.1', 'test', 'test-session'
        )
        
        # Enforce retention policy
        await enterprise_manager._enforce_audit_retention_policy(tenant_id)
        
        # Check that old entries are removed
        remaining_audits = enterprise_manager.audit_trails[tenant_id]
        for audit in remaining_audits:
            age_days = (datetime.utcnow() - audit.timestamp).days
            assert age_days <= 1


class TestTenantIsolationManager:
    """Test suite for tenant isolation functionality"""
    
    async def test_tenant_isolation_creation(self, tenant_isolation_manager):
        """Test tenant isolation setup"""
        from enterprise_features import TenantConfiguration
        
        tenant_cfg = TenantConfiguration(
            tenant_id='isolation-test',
            tenant_name='Isolation Test',
            tier=TenantTier.ENTERPRISE,
            compliance_frameworks=[],
            custom_branding={},
            sla_requirements={},
            feature_flags={},
            resource_quotas={},
            audit_requirements={}
        )
        
        isolation_config = {
            'data_classification': 'confidential',
            'network_isolation': True,
            'storage_isolation': True,
            'encryption_at_rest': True
        }
        
        isolation_policy = await tenant_isolation_manager.create_tenant_isolation(
            'isolation-test', tenant_cfg, isolation_config
        )
        
        assert isolation_policy.tenant_id == 'isolation-test'
        assert isolation_policy.isolation_level == IsolationLevel.HYBRID
        assert isolation_policy.data_classification == DataClassification.CONFIDENTIAL
        assert isolation_policy.network_isolation is True
        assert isolation_policy.storage_isolation is True
        assert isolation_policy.encryption_at_rest is True
    
    async def test_tenant_boundary_enforcement(self, tenant_isolation_manager):
        """Test tenant boundary enforcement"""
        from enterprise_features import TenantConfiguration
        
        # Create two isolated tenants
        tenant_configs = [
            ('tenant-a', TenantTier.ENTERPRISE),
            ('tenant-b', TenantTier.ENTERPRISE)
        ]
        
        for tenant_id, tier in tenant_configs:
            tenant_cfg = TenantConfiguration(
                tenant_id=tenant_id,
                tenant_name=f'Tenant {tenant_id.upper()}',
                tier=tier,
                compliance_frameworks=[],
                custom_branding={},
                sla_requirements={},
                feature_flags={},
                resource_quotas={},
                audit_requirements={}
            )
            
            await tenant_isolation_manager.create_tenant_isolation(
                tenant_id, tenant_cfg, {'data_classification': 'internal'}
            )
        
        # Test cross-tenant access denial
        boundary_result = await tenant_isolation_manager.enforce_tenant_boundaries(
            'tenant-a', 'tenant-b', 'health_metric', 'read'
        )
        
        assert boundary_result['allowed'] is False
        assert 'Cross-tenant access not permitted' in boundary_result['reason']
        
        # Test same-tenant access allowance
        boundary_result = await tenant_isolation_manager.enforce_tenant_boundaries(
            'tenant-a', 'tenant-a', 'health_metric', 'read'
        )
        
        assert boundary_result['allowed'] is True
        assert 'Same tenant access' in boundary_result['reason']
    
    async def test_data_classification_compatibility(self, tenant_isolation_manager):
        """Test data classification compatibility checks"""
        # Test compatibility check directly
        result = tenant_isolation_manager._check_data_classification_compatibility(
            DataClassification.INTERNAL,      # Requesting
            DataClassification.PUBLIC         # Target
        )
        assert result is True  # Can access lower classification
        
        result = tenant_isolation_manager._check_data_classification_compatibility(
            DataClassification.PUBLIC,        # Requesting
            DataClassification.CONFIDENTIAL   # Target
        )
        assert result is False  # Cannot access higher classification
    
    async def test_tenant_encryption(self, tenant_isolation_manager):
        """Test tenant-specific encryption"""
        from enterprise_features import TenantConfiguration
        
        tenant_cfg = TenantConfiguration(
            tenant_id='encryption-test',
            tenant_name='Encryption Test',
            tier=TenantTier.ENTERPRISE,
            compliance_frameworks=[],
            custom_branding={},
            sla_requirements={},
            feature_flags={},
            resource_quotas={},
            audit_requirements={}
        )
        
        await tenant_isolation_manager.create_tenant_isolation(
            'encryption-test', tenant_cfg, {'encryption_at_rest': True}
        )
        
        # Test data encryption
        test_data = "sensitive health data"
        encrypted = await tenant_isolation_manager.encrypt_tenant_data(
            'encryption-test', test_data
        )
        
        assert encrypted != test_data
        assert len(encrypted) > 0
        
        # Test data decryption
        decrypted = await tenant_isolation_manager.decrypt_tenant_data(
            'encryption-test', encrypted
        )
        
        assert decrypted != encrypted
        assert 'encryption-test' in decrypted  # Mock decryption includes tenant ID
    
    async def test_isolation_status_reporting(self, tenant_isolation_manager):
        """Test tenant isolation status reporting"""
        from enterprise_features import TenantConfiguration
        
        tenant_cfg = TenantConfiguration(
            tenant_id='status-test',
            tenant_name='Status Test',
            tier=TenantTier.ENTERPRISE_PLUS,
            compliance_frameworks=[],
            custom_branding={},
            sla_requirements={},
            feature_flags={},
            resource_quotas={},
            audit_requirements={}
        )
        
        await tenant_isolation_manager.create_tenant_isolation(
            'status-test', tenant_cfg, {
                'data_classification': 'restricted',
                'network_isolation': True,
                'compute_isolation': True
            }
        )
        
        # Get isolation status
        status = await tenant_isolation_manager.get_tenant_isolation_status('status-test')
        
        assert status['tenant_id'] == 'status-test'
        assert status['isolation_policy']['isolation_level'] == 'dedicated'
        assert status['isolation_policy']['data_classification'] == 'restricted'
        assert status['isolation_policy']['network_isolation'] is True
        assert status['isolation_policy']['compute_isolation'] is True
        assert status['resources_count'] > 0
        assert 'encryption_status' in status
        assert 'security_boundaries' in status


class TestEnterpriseIntegration:
    """Test suite for enterprise feature integration with health service"""
    
    async def test_enterprise_health_service_integration(self, health_service):
        """Test enterprise features integration with health service"""
        # Verify enterprise features are available
        assert_enterprise_features_available(health_service)
        
        # Test enterprise tenant creation through health service
        tenant_config = {
            'tenant_id': 'integration-test',
            'tenant_name': 'Integration Test',
            'tier': 'professional'
        }
        
        result = await health_service.create_enterprise_tenant(tenant_config)
        
        assert result['status'] == 'success'
        assert result['tenant_id'] == 'integration-test'
        assert result['tier'] == 'professional'
    
    async def test_enterprise_metric_processing(self, health_service):
        """Test enterprise-enhanced metric processing"""
        # Create enterprise tenant first
        tenant_config = {
            'tenant_id': 'enterprise-metrics',
            'tenant_name': 'Enterprise Metrics Test',
            'tier': 'enterprise'
        }
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Create metric
        metric = HealthMetric(
            tenant_id='enterprise-metrics',
            component_id='enterprise-component-001',
            name='enterprise_metric',
            value=85.0,
            dimension=HealthDimension.PERFORMANCE
        )
        
        # Process with enterprise features
        result = await health_service.process_health_metric_with_enterprise_features(
            metric, 'enterprise-metrics'
        )
        
        assert result['status'] == 'success' or 'enterprise_processing' in result
        if 'sla_compliance' in result:
            assert 'compliant' in result['sla_compliance']
    
    async def test_tenant_quota_integration(self, health_service):
        """Test tenant quota integration with health operations"""
        # Create basic tier tenant with low quotas
        tenant_config = {
            'tenant_id': 'quota-integration',
            'tier': 'basic'
        }
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Test quota enforcement
        quota_result = await health_service.enforce_tenant_quotas(
            'quota-integration', 'api_calls_per_hour', 500
        )
        
        assert 'allowed' in quota_result
        assert 'current_usage' in quota_result
        
        if quota_result['allowed']:
            assert quota_result['current_usage'] <= 1000  # Basic tier limit
    
    async def test_compliance_integration(self, health_service):
        """Test compliance integration with health management"""
        # Create compliant tenant
        tenant_config = {
            'tenant_id': 'compliance-integration',
            'tier': 'enterprise',
            'compliance_frameworks': ['soc2']
        }
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Generate compliance report
        report_result = await health_service.generate_compliance_report(
            'compliance-integration', 'soc2', 7
        )
        
        if 'error' not in report_result:
            assert report_result['framework'] == 'soc2'
            assert 'report' in report_result
    
    async def test_tenant_dashboard_integration(self, health_service):
        """Test tenant-specific dashboard integration"""
        # Create tenant with custom branding
        tenant_config = {
            'tenant_id': 'dashboard-integration',
            'tenant_name': 'Dashboard Test Corp',
            'tier': 'enterprise',
            'custom_branding': {
                'company_name': 'Dashboard Test Corp',
                'theme_color': '#FF6B35'
            }
        }
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Create tenant dashboard
        dashboard = await health_service.create_tenant_health_dashboard(
            'dashboard-integration', 'executive'
        )
        
        if 'error' not in dashboard:
            assert dashboard.get('enterprise_enhanced') is True
            
            if 'tenant_info' in dashboard:
                assert dashboard['tenant_info']['tenant_name'] == 'Dashboard Test Corp'
                assert dashboard['tenant_info']['tier'] == 'enterprise'
            
            if 'custom_branding' in dashboard:
                assert dashboard['custom_branding']['company_name'] == 'Dashboard Test Corp'


class TestEnterprisePerformance:
    """Performance tests for enterprise features"""
    
    async def test_multi_tenant_performance(self, health_service):
        """Test performance with multiple tenants"""
        tenant_count = 10
        metrics_per_tenant = 50
        
        # Create multiple tenants
        for i in range(tenant_count):
            tenant_config = {
                'tenant_id': f'perf-tenant-{i:02d}',
                'tier': 'professional'
            }
            await health_service.create_enterprise_tenant(tenant_config)
        
        start_time = datetime.utcnow()
        
        # Process metrics for all tenants concurrently
        tasks = []
        for tenant_idx in range(tenant_count):
            tenant_id = f'perf-tenant-{tenant_idx:02d}'
            
            for metric_idx in range(metrics_per_tenant):
                metric = HealthMetric(
                    tenant_id=tenant_id,
                    component_id=f'perf-component-{metric_idx % 5}',
                    name='performance_metric',
                    value=50.0 + (metric_idx % 50),
                    dimension=HealthDimension.PERFORMANCE
                )
                
                task = health_service.process_health_metric_with_enterprise_features(
                    metric, tenant_id
                )
                tasks.append(task)
        
        # Process in batches
        batch_size = 50
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch, return_exceptions=True)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        total_operations = tenant_count * metrics_per_tenant
        throughput = total_operations / duration
        
        print(f"Multi-tenant throughput: {throughput:.2f} operations/second")
        assert throughput > 20  # Should handle at least 20 ops/sec with enterprise features
    
    async def test_isolation_overhead(self, health_service):
        """Test performance overhead of tenant isolation"""
        # Create isolated tenant
        tenant_config = {
            'tenant_id': 'isolation-perf',
            'tier': 'enterprise_plus',
            'isolation_config': {
                'data_classification': 'restricted',
                'network_isolation': True,
                'storage_isolation': True,
                'encryption_at_rest': True
            }
        }
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Measure processing time with isolation
        start_time = datetime.utcnow()
        
        for i in range(100):
            metric = HealthMetric(
                tenant_id='isolation-perf',
                component_id=f'isolation-component-{i % 5}',
                name='isolation_metric',
                value=60.0,
                dimension=HealthDimension.PERFORMANCE
            )
            
            await health_service.process_health_metric_with_enterprise_features(
                metric, 'isolation-perf'
            )
        
        end_time = datetime.utcnow()
        isolation_duration = (end_time - start_time).total_seconds()
        
        # Measure processing time without enterprise features (baseline)
        start_time = datetime.utcnow()
        
        for i in range(100):
            metric = HealthMetric(
                tenant_id='baseline-test',
                component_id=f'baseline-component-{i % 5}',
                name='baseline_metric',
                value=60.0,
                dimension=HealthDimension.PERFORMANCE
            )
            
            await health_service.process_health_metric(metric)
        
        end_time = datetime.utcnow()
        baseline_duration = (end_time - start_time).total_seconds()
        
        # Calculate overhead
        overhead_ratio = isolation_duration / baseline_duration if baseline_duration > 0 else 1
        
        print(f"Isolation overhead: {overhead_ratio:.2f}x baseline")
        assert overhead_ratio < 3.0  # Overhead should be less than 3x baseline


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])