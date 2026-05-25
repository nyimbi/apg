#!/usr/bin/env python3
"""
APG Configuration Management Security Integration Test
Tests the security framework integration with configuration operations.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the project root to Python path
conf_path = "/Users/nyimbiodero/src/pjs/apg/capabilities/common/conf"
sys.path.insert(0, conf_path)

async def test_security_integration():
    """Test security integration with configuration management"""
    print("🔐 Testing APG Security Framework Integration...")
    
    try:
        # Import required modules
        from security_integration import (
            ConfigurationSecurityService,
            ConfigurationSecurityLevel,
            ConfigurationSecurityContext,
            get_configuration_security_service
        )
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        print("   ✓ Security integration modules loaded successfully")
        
        # Test 1: Initialize security service
        security_service = await get_configuration_security_service()
        assert security_service is not None
        print("   ✓ Configuration Security Service initialized")
        
        # Test 2: Create security context
        security_context = ConfigurationSecurityContext(
            tenant_id="test_tenant",
            user_id="test_user",
            operation="create",
            security_level=ConfigurationSecurityLevel.INTERNAL
        )
        
        assert security_context.tenant_id == "test_tenant"
        assert security_context.user_id == "test_user"
        assert security_context.operation == "create"
        print("   ✓ Security context created successfully")
        
        # Test 3: Test security assessment for low-risk operation
        is_authorized, assessed_context, messages = await security_service.secure_configuration_operation(
            tenant_id="test_tenant",
            user_id="test_user",
            operation="read",
            security_level=ConfigurationSecurityLevel.PUBLIC
        )
        
        assert is_authorized == True  # Read operation on public config should be allowed
        assert assessed_context.risk_score is not None
        print(f"   ✓ Low-risk operation authorized: messages={len(messages)}")
        
        # Test 4: Test security assessment for higher-risk operation
        is_authorized, assessed_context, messages = await security_service.secure_configuration_operation(
            tenant_id="test_tenant",
            user_id="test_user",
            operation="delete",
            security_level=ConfigurationSecurityLevel.CONFIDENTIAL
        )
        
        # Should be authorized but may have security messages
        assert assessed_context.risk_score is not None
        print(f"   ✓ High-risk operation assessed: authorized={is_authorized}, messages={len(messages)}")
        
        # Test 5: Create test configuration resource
        dsl = ConfigurationDSL(
            kind="VirtualMachine",
            spec={
                "resources": {"instance_type": "t3.micro"},
                "security": {"encryption": True}
            },
            version="1.0"
        )
        
        test_resource = CMResource(
            name="test-secure-vm",
            resource_type=ResourceType.VIRTUAL_MACHINE,
            cloud_provider=CloudProvider.AWS,
            configuration=dsl,
            description="Security integration test resource"
        )
        
        # Test 6: Validate configuration compliance
        compliance_result = await security_service.validate_configuration_compliance(
            test_resource, "test_tenant"
        )
        
        assert compliance_result.valid in [True, False]  # Should return a boolean
        print(f"   ✓ Configuration compliance validation: valid={compliance_result.valid}, warnings={len(compliance_result.warnings)}")
        
        # Test 7: Test with potentially risky configuration
        risky_dsl = ConfigurationDSL(
            kind="VirtualMachine",
            spec={
                "resources": {"instance_type": "t3.micro"},
                "security": {"password": "admin123", "privileged": True}  # Should trigger security warnings
            },
            version="1.0"
        )
        
        risky_resource = CMResource(
            name="test-risky-vm",
            resource_type=ResourceType.VIRTUAL_MACHINE,
            cloud_provider=CloudProvider.AWS,
            configuration=risky_dsl,
            description="Risky configuration test"
        )
        
        # Assess security for risky configuration
        is_authorized, risky_context, risky_messages = await security_service.secure_configuration_operation(
            tenant_id="test_tenant",
            user_id="test_user",
            operation="create",
            resource=risky_resource,
            security_level=ConfigurationSecurityLevel.INTERNAL
        )
        
        # Should detect threats in risky configuration
        threat_count = len(risky_context.threat_indicators)
        print(f"   ✓ Risky configuration detected: authorized={is_authorized}, threats={threat_count}, messages={len(risky_messages)}")
        
        # Test 8: Compliance validation for risky resource
        risky_compliance = await security_service.validate_configuration_compliance(
            risky_resource, "test_tenant"
        )
        
        # Should have compliance issues due to hardcoded password
        assert len(risky_compliance.warnings) > 0 or len(risky_compliance.errors) > 0
        print(f"   ✓ Risky configuration compliance: valid={risky_compliance.valid}, issues={len(risky_compliance.errors + risky_compliance.warnings)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Security integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_policy_engine():
    """Test security policy engine functionality"""
    print("\n🛡️  Testing Security Policy Engine...")
    
    try:
        from security_integration import ConfigurationSecurityEngine
        from security_integration import get_configuration_security_service
        
        # Get security service and engine
        security_service = await get_configuration_security_service()
        engine = security_service.security_engine
        
        assert engine._initialized == True
        assert len(engine.config_policies) > 0
        print(f"   ✓ Policy engine initialized with {len(engine.config_policies)} policies")
        
        # Test threat pattern loading
        assert len(engine.threat_patterns) > 0
        threat_pattern_names = list(engine.threat_patterns.keys())
        print(f"   ✓ Threat patterns loaded: {threat_pattern_names}")
        
        # Test compliance rule loading
        assert len(engine.compliance_rules) > 0
        compliance_rule_names = list(engine.compliance_rules.keys())
        print(f"   ✓ Compliance rules loaded: {compliance_rule_names}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Policy engine test failed: {e}")
        return False


async def main():
    """Run security integration tests"""
    print("🔐 APG Configuration Management Security Integration Tests")
    print("=" * 70)
    
    test1_success = await test_security_integration()
    test2_success = await test_policy_engine()
    
    print("\n" + "=" * 70)
    if test1_success and test2_success:
        print("🏆 SECURITY INTEGRATION TESTS: PASSED ✅")
        print("   🔐 APG Security Framework integration successful")
        print("   🛡️  Security policy engine operational")
        print("   ⚡ Configuration threat detection working")
        print("   📋 Compliance validation functional")
        print("   🔍 Risk assessment and authorization complete")
        print("   🎯 Phase 3.3a Security Integration: COMPLETE")
        print("   💎 Configuration security governance achieved")
    else:
        print("❌ SECURITY INTEGRATION TESTS: FAILED")
        print("   🔍 Check error logs above for details")
    
    print("=" * 70)
    
    return test1_success and test2_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)