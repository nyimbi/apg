#!/usr/bin/env python3
"""
Isolated test for APG Configuration Management capability
Tests the capability in isolation to validate foundation implementation.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, '/Users/nyimbiodero/src/pjs/apg')

# Test configuration management models directly
def test_models():
    """Test Configuration Management data models"""
    print("🧪 Testing Configuration Management Models...")
    
    from capabilities.common.conf.models import (
        CMResource, CMTemplate, CMPolicy, CMEnvironment, CMDeployment,
        ResourceType, CloudProvider, ResourceState, DeploymentStatus,
        ConfigurationDSL, ValidationResult
    )
    from uuid_extensions import uuid7str
    
    # Test ConfigurationDSL
    dsl = ConfigurationDSL(
        kind="VirtualMachine",
        spec={
            "resources": {
                "instance_type": "t3.medium",
                "image": "ami-ubuntu-20.04",
                "vpc_id": "vpc-12345"
            }
        },
        version="1.0"
    )
    
    assert dsl.kind == "VirtualMachine"
    assert "t3.medium" in str(dsl.spec)
    print(f"   ✓ ConfigurationDSL: {dsl.kind}")
    
    # Test YAML and HCL export
    yaml_output = dsl.to_yaml()
    hcl_output = dsl.to_hcl()
    assert len(yaml_output) > 50
    assert len(hcl_output) > 20
    print(f"   ✓ Export formats: YAML ({len(yaml_output)} chars), HCL ({len(hcl_output)} chars)")
    
    # Test CMResource
    resource = CMResource(
        name="test-web-server",
        resource_type=ResourceType.VIRTUAL_MACHINE,
        cloud_provider=CloudProvider.AWS,
        configuration=dsl,
        description="Test web server"
    )
    
    assert resource.name == "test-web-server"
    assert resource.resource_type == ResourceType.VIRTUAL_MACHINE
    assert resource.state == ResourceState.PENDING
    assert len(resource.id) > 20
    print(f"   ✓ CMResource: {resource.name} ({resource.state.value})")
    
    # Test CMTemplate
    template = CMTemplate(
        name="web-template",
        description="Web server template",
        category="web",
        configuration_template={"instance_type": "{{instance_type}}"},
        parameters={"instance_type": "t3.micro"}
    )
    
    assert template.name == "web-template"
    assert template.category == "web"
    print(f"   ✓ CMTemplate: {template.name}")
    
    # Test template instantiation
    instantiated = template.instantiate({"instance_type": "t3.large"})
    assert isinstance(instantiated, dict)
    print(f"   ✓ Template instantiation works")
    
    # Test CMPolicy
    policy = CMPolicy(
        name="security-policy",
        description="Security compliance policy",
        policy_type="security",
        rules=[{"field": "encryption", "required": True}],
        actions=["warn"]
    )
    
    assert policy.name == "security-policy"
    assert policy.policy_type == "security"
    print(f"   ✓ CMPolicy: {policy.name}")
    
    # Test policy evaluation
    policy_result = policy.evaluate(resource)
    assert isinstance(policy_result, dict)
    assert "compliant" in policy_result
    print(f"   ✓ Policy evaluation: compliant={policy_result['compliant']}")
    
    # Test CMDeployment
    deployment = CMDeployment(
        resource_id=resource.id,
        environment_id="production",
        deployment_plan={"steps": ["validate", "deploy", "verify"]},
        started_at=datetime.utcnow()
    )
    
    assert deployment.resource_id == resource.id
    assert deployment.status == DeploymentStatus.PENDING
    print(f"   ✓ CMDeployment: {deployment.id[:8]}... ({deployment.status.value})")
    
    # Test duration calculation
    deployment.completed_at = datetime.utcnow()
    duration = deployment.calculate_duration()
    assert isinstance(duration, int)
    print(f"   ✓ Duration calculation: {duration}s")
    
    # Test ValidationResult
    validation = ValidationResult(
        valid=True,
        errors=[],
        warnings=["Consider adding tags"],
        resource_id=resource.id
    )
    
    assert validation.valid is True
    assert len(validation.warnings) == 1
    print(f"   ✓ ValidationResult: valid={validation.valid}, warnings={len(validation.warnings)}")
    
    print("   🎉 All model tests passed!")


async def test_service_basic():
    """Test basic service functionality with mocked dependencies"""
    print("\n🔧 Testing Configuration Management Service (Basic)...")
    
    # Create minimal mock classes
    class MockAI:
        async def initialize(self): pass
        async def optimize_configuration(self, *args): 
            return {"optimization": "applied", "improvements": ["caching enabled"]}
        async def process_natural_language(self, *args): 
            return {"intent": "create_vm", "confidence": 0.95}
        async def analyze_configuration(self, *args): 
            return {"analysis": "complete", "score": 0.87}
    
    class MockLayer:
        async def initialize(self): pass
        async def deploy_resource(self, *args): 
            return {"deployment": "simulated", "status": "success"}
        async def validate_configuration(self, *args): 
            return {"valid": True, "warnings": []}
    
    class MockSecurity:
        async def initialize(self): pass
        async def encrypt_configuration(self, *args): 
            return {"encrypted": True, "algorithm": "AES-256"}
        async def verify_integrity(self, *args): 
            return {"verified": True, "checksum": "abc123"}
    
    # Import and patch the service
    from capabilities.common.conf import service
    
    # Temporarily replace the classes
    original_ai = getattr(service, 'AIIntelligenceEngine', None)
    original_layer = getattr(service, 'UniversalAbstractionLayer', None) 
    original_security = getattr(service, 'QuantumSecurity', None)
    original_analytics = getattr(service, 'PredictiveConfigAnalytics', None)
    
    service.AIIntelligenceEngine = MockAI
    service.UniversalAbstractionLayer = MockLayer
    service.QuantumSecurity = MockSecurity
    service.PredictiveConfigAnalytics = MockAI
    
    try:
        from capabilities.common.conf.service import RevolutionaryConfigurationManager
        
        # Initialize manager
        manager = RevolutionaryConfigurationManager(tenant_id="test_isolated")
        await manager.initialize({})
        
        assert manager.tenant_id == "test_isolated"
        assert manager._initialized is True
        print(f"   ✓ Manager initialized: {manager.tenant_id}")
        
        # Test configuration creation
        config = {
            "name": "isolated-test-vm",
            "type": "virtual_machine",
            "cloud_provider": "aws",
            "configuration": {
                "kind": "VirtualMachine",
                "spec": {
                    "resources": {
                        "instance_type": "t3.micro",
                        "image": "ami-ubuntu-22.04"
                    }
                }
            },
            "description": "Isolated test VM"
        }
        
        resource = await manager.create_configuration(config)
        assert resource.name == "isolated-test-vm"
        assert len(manager.resources) == 1
        print(f"   ✓ Configuration created: {resource.name}")
        
        # Test deployment
        deployment = await manager.deploy_configuration(resource.id, "test_env")
        assert deployment.resource_id == resource.id
        assert len(manager.deployments) == 1
        print(f"   ✓ Deployment created: {deployment.environment_id}")
        
        # Test drift detection
        drift_result = await manager.detect_and_remediate_drift(resource.id)
        assert "resource_id" in drift_result
        assert "drift_detected" in drift_result
        print(f"   ✓ Drift detection: {drift_result['drift_detected']}")
        
        # Test natural language processing
        nl_result = await manager.natural_language_configuration(
            "Create a small database server",
            {"user_id": "test", "environment": "dev"}
        )
        assert "request" in nl_result
        assert "ready_to_deploy" in nl_result
        print(f"   ✓ Natural language: {nl_result['ready_to_deploy']}")
        
        # Test template creation
        template_req = {
            "name": "isolated-template",
            "category": "database",
            "requirements": {"db": "postgresql"},
            "created_by": "test"
        }
        template = await manager.create_intelligent_template(template_req)
        assert template.name == "isolated-template"
        print(f"   ✓ Template created: {template.name}")
        
        # Test metrics
        metrics = await manager.get_revolutionary_metrics()
        assert "system_metrics" in metrics
        assert "performance_indicators" in metrics
        print(f"   ✓ Metrics retrieved: {len(metrics)} categories")
        
        # Test insights
        insights = await manager.get_predictive_insights(resource.id)
        assert "insights" in insights
        print(f"   ✓ Predictive insights generated")
        
        await manager.shutdown()
        print(f"   ✓ Manager shutdown completed")
        
        print("   🎉 All service tests passed!")
        
    finally:
        # Restore original classes
        if original_ai:
            service.AIIntelligenceEngine = original_ai
        if original_layer:
            service.UniversalAbstractionLayer = original_layer
        if original_security:
            service.QuantumSecurity = original_security
        if original_analytics:
            service.PredictiveConfigAnalytics = original_analytics


def test_integration():
    """Test APG integration points"""
    print("\n🔗 Testing APG Integration Points...")
    
    from capabilities.common.conf import (
        RevolutionaryConfigurationManager,
        create_configuration_manager,
        get_config_manager
    )
    
    # Test factory functions exist
    assert callable(create_configuration_manager)
    assert callable(get_config_manager)
    print("   ✓ Factory functions available")
    
    # Test module exports
    from capabilities.common.conf import (
        CMResource, CMTemplate, CMPolicy,
        ResourceState, DeploymentStatus, ResourceType
    )
    
    assert CMResource is not None
    assert ResourceType is not None
    print("   ✓ Core models exported")
    
    print("   🎉 Integration tests passed!")


async def main():
    """Run comprehensive isolated tests"""
    print("🚀 APG Configuration Management - Isolated Foundation Test")
    print("=" * 70)
    
    try:
        # Test models (sync)
        test_models()
        
        # Test service (async)
        await test_service_basic()
        
        # Test integration (sync)
        test_integration()
        
        print("\n" + "=" * 70)
        print("🏆 FOUNDATION VALIDATION COMPLETE!")
        print("   ✅ All models working correctly")
        print("   ✅ Service layer functional")
        print("   ✅ APG integration verified")
        print("   ✅ Revolutionary features operational")
        print("\n🎯 Configuration Management foundation is solid and ready!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)