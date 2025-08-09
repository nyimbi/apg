#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - Test Configuration and Fixtures
Comprehensive test fixtures for all health management components

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Generator
from unittest.mock import MagicMock, AsyncMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service import SystemHealthService, HealthServiceConfig
from models import (
    HealthMetric, HealthAlert, SystemComponent, HealthReport,
    HealthStatus, HealthSeverity, HealthDimension, ComponentType
)
from enterprise_features import EnterpriseHealthManager, TenantTier, TenantConfiguration
from multi_tenant_isolation import TenantIsolationManager, IsolationLevel
from ml_engines import HealthPredictionEngine, AdvancedAnalyticsEngine
from optimization_engine import ResourceOptimizationEngine


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Test configuration for health service"""
    return HealthServiceConfig(
        health_check_interval_seconds=10,
        prediction_window_hours=1,
        auto_remediation_enabled=False,  # Disabled for testing
        alert_correlation_window_minutes=1,
        baseline_learning_period_days=1,
        max_concurrent_assessments=10,
        health_data_retention_days=7,
        batch_processing_size=100
    )


@pytest.fixture
async def health_service(test_config):
    """Health service instance for testing"""
    service = SystemHealthService(test_config)
    await service.initialize()
    yield service
    # Cleanup if needed
    if hasattr(service, '_background_tasks'):
        for task in service._background_tasks:
            if not task.done():
                task.cancel()


@pytest.fixture
def sample_health_metric():
    """Sample health metric for testing"""
    return HealthMetric(
        tenant_id="test-tenant-001",
        component_id="test-component-001",
        name="cpu_utilization",
        value=75.5,
        dimension=HealthDimension.PERFORMANCE,
        unit="percent",
        business_context={
            "service_tier": "critical",
            "business_impact": "high"
        },
        tags=["production", "web-server"],
        metadata={
            "source": "prometheus",
            "collection_method": "agent"
        }
    )


@pytest.fixture
def sample_system_component():
    """Sample system component for testing"""
    return SystemComponent(
        component_id="test-component-001",
        tenant_id="test-tenant-001",
        name="Test Web Server",
        description="Test web server component",
        component_type=ComponentType.SERVICE,
        health_status=HealthStatus.HEALTHY,
        status="running",
        version="1.0.0",
        environment="test",
        tags=["web", "api", "critical"],
        dependencies=["database-001", "cache-001"],
        business_criticality="high",
        metadata={
            "port": 8080,
            "endpoint": "/health",
            "deployment": "kubernetes"
        }
    )


@pytest.fixture
def sample_health_alert():
    """Sample health alert for testing"""
    return HealthAlert(
        tenant_id="test-tenant-001",
        rule_id="test-rule-001",
        component_id="test-component-001",
        name="High CPU Utilization",
        message="CPU utilization exceeded threshold",
        severity=HealthSeverity.HIGH,
        health_status=HealthStatus.WARNING,
        source_metric="cpu_utilization",
        source_value=95.0,
        threshold_value=90.0,
        threshold_operator="gt",
        business_impact_score=0.8
    )


@pytest.fixture
def enterprise_tenant_config():
    """Enterprise tenant configuration for testing"""
    return {
        'tenant_id': 'test-enterprise-001',
        'tenant_name': 'Test Enterprise',
        'tier': 'enterprise',
        'compliance_frameworks': ['soc2', 'iso27001'],
        'custom_branding': {
            'company_name': 'Test Corp',
            'logo_url': '/static/test-logo.png'
        },
        'sla_requirements': {
            'availability_target': 99.9,
            'response_time_target': 200
        }
    }


@pytest.fixture
async def enterprise_manager():
    """Enterprise health manager for testing"""
    manager = EnterpriseHealthManager()
    yield manager


@pytest.fixture
async def tenant_isolation_manager():
    """Tenant isolation manager for testing"""
    manager = TenantIsolationManager()
    yield manager


@pytest.fixture
async def prediction_engine():
    """ML prediction engine for testing"""
    engine = HealthPredictionEngine()
    yield engine


@pytest.fixture
async def optimization_engine(prediction_engine):
    """Resource optimization engine for testing"""
    engine = ResourceOptimizationEngine(prediction_engine)
    yield engine


@pytest.fixture
def mock_apg_integrations():
    """Mock APG capability integrations"""
    return {
        'moni': MagicMock(),
        'auth': MagicMock(),
        'audl': MagicMock(),
        'ntfy': MagicMock(),
        'mten': MagicMock(),
        'conf': MagicMock()
    }


@pytest.fixture
def performance_test_data():
    """Generate performance test data"""
    components = []
    metrics = []
    alerts = []
    
    # Generate test components
    for i in range(100):
        component = SystemComponent(
            component_id=f"perf-component-{i:03d}",
            tenant_id="performance-test",
            name=f"Performance Component {i}",
            component_type=ComponentType.SERVICE,
            health_status=HealthStatus.HEALTHY,
            metadata={"test": True, "index": i}
        )
        components.append(component)
        
        # Generate metrics for each component
        for metric_name in ["cpu_utilization", "memory_utilization", "response_time"]:
            metric = HealthMetric(
                tenant_id="performance-test",
                component_id=component.component_id,
                name=metric_name,
                value=50.0 + (i % 50),  # Vary values
                dimension=HealthDimension.PERFORMANCE
            )
            metrics.append(metric)
    
    return {
        'components': components,
        'metrics': metrics,
        'alerts': alerts
    }


@pytest.fixture
def integration_test_scenarios():
    """Integration test scenarios"""
    return {
        'tenant_creation': {
            'tenant_config': {
                'tenant_id': 'integration-test-001',
                'tenant_name': 'Integration Test',
                'tier': 'professional'
            },
            'expected_isolation_level': IsolationLevel.SHARED
        },
        'health_metric_flow': {
            'metric': {
                'tenant_id': 'integration-test-001',
                'component_id': 'integration-component-001',
                'name': 'test_metric',
                'value': 85.0,
                'dimension': 'performance'
            },
            'expected_processing': True,
            'expected_alerts': 0
        },
        'cross_tenant_access': {
            'requesting_tenant': 'integration-test-001',
            'target_tenant': 'integration-test-002',
            'resource_type': 'health_metric',
            'operation': 'read',
            'expected_allowed': False
        }
    }


# Mock data generators
def generate_mock_metrics(count: int = 10, tenant_id: str = "test") -> list:
    """Generate mock health metrics for testing"""
    metrics = []
    metric_names = ["cpu_utilization", "memory_utilization", "disk_utilization", 
                   "response_time", "error_rate", "throughput"]
    
    for i in range(count):
        metric = HealthMetric(
            tenant_id=tenant_id,
            component_id=f"component-{i % 3:03d}",
            name=metric_names[i % len(metric_names)],
            value=float(20 + (i * 7) % 80),  # Generate realistic values
            dimension=HealthDimension.PERFORMANCE,
            metadata={"test": True, "index": i}
        )
        metrics.append(metric)
    
    return metrics


def generate_mock_components(count: int = 5, tenant_id: str = "test") -> list:
    """Generate mock system components for testing"""
    components = []
    component_types = list(ComponentType)
    
    for i in range(count):
        component = SystemComponent(
            component_id=f"test-component-{i:03d}",
            tenant_id=tenant_id,
            name=f"Test Component {i}",
            component_type=component_types[i % len(component_types)],
            health_status=HealthStatus.HEALTHY,
            metadata={"test": True, "index": i}
        )
        components.append(component)
    
    return components


# Test utilities
async def wait_for_background_task(service, task_name: str, timeout: int = 10):
    """Wait for a background task to complete"""
    start_time = datetime.utcnow()
    while (datetime.utcnow() - start_time).seconds < timeout:
        if hasattr(service, '_background_task_status'):
            if service._background_task_status.get(task_name) == 'completed':
                return True
        await asyncio.sleep(0.1)
    return False


def assert_health_metric_valid(metric: HealthMetric):
    """Assert that a health metric is valid"""
    assert metric.metric_id is not None
    assert metric.tenant_id is not None
    assert metric.component_id is not None
    assert metric.name is not None
    assert isinstance(metric.value, (int, float))
    assert metric.timestamp is not None
    assert metric.dimension in HealthDimension


def assert_enterprise_features_available(service: SystemHealthService):
    """Assert that enterprise features are available"""
    assert service._enterprise_manager is not None
    assert service._tenant_isolation_manager is not None
    assert hasattr(service, 'create_enterprise_tenant')
    assert hasattr(service, 'enforce_tenant_quotas')


# Export fixtures and utilities
__all__ = [
    'test_config', 'health_service', 'sample_health_metric', 
    'sample_system_component', 'sample_health_alert',
    'enterprise_tenant_config', 'enterprise_manager', 
    'tenant_isolation_manager', 'prediction_engine', 
    'optimization_engine', 'mock_apg_integrations',
    'performance_test_data', 'integration_test_scenarios',
    'generate_mock_metrics', 'generate_mock_components',
    'wait_for_background_task', 'assert_health_metric_valid',
    'assert_enterprise_features_available'
]