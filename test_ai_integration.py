#!/usr/bin/env python3
"""
Test AI Engine Integration with Service Layer
Validates that the advanced AI engine integrates correctly with the service layer.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, '/Users/nyimbiodero/src/pjs/apg')

async def test_ai_integration():
    """Test AI engine integration with service layer"""
    print("🧠 Testing AI Engine Integration...")
    
    try:
        # Direct import approach
        conf_path = "/Users/nyimbiodero/src/pjs/apg/capabilities/common/conf"
        sys.path.insert(0, conf_path)
        
        from ai_engine_advanced import AIIntelligenceEngine
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        # Initialize AI engine
        ai_engine = AIIntelligenceEngine(tenant_id="test_ai")
        await ai_engine.initialize()
        print("   ✓ AI Engine initialized")
        
        # Create test resource with proper spec structure
        dsl = ConfigurationDSL(
            kind="TestVM",
            spec={
                "resources": {"instance_type": "t3.micro", "image": "ami-test"}
            },
            version="1.0"
        )
        
        resource = CMResource(
            name="test-vm-ai",
            resource_type=ResourceType.VIRTUAL_MACHINE,
            cloud_provider=CloudProvider.AWS,
            configuration=dsl,
            description="AI integration test VM"
        )
        
        print("   ✓ Test resource created")
        
        # Test optimize_configuration (compatibility method)
        optimization = await ai_engine.optimize_configuration(resource, {"historical_data": {}})
        print(f"   ✓ Configuration optimization: {len(optimization)} recommendations")
        
        # Test generate_deployment_plan (compatibility method)
        deployment_plan = await ai_engine.generate_deployment_plan(resource, "production")
        assert "phases" in deployment_plan
        assert len(deployment_plan["phases"]) == 3
        print(f"   ✓ Deployment plan generated: {len(deployment_plan['phases'])} phases")
        
        # Test drift detection (compatibility method)
        drift_analysis = await ai_engine.detect_configuration_drift(resource)
        assert "has_drift" in drift_analysis
        assert "details" in drift_analysis
        print(f"   ✓ Drift detection: has_drift={drift_analysis['has_drift']}")
        
        # Test remediation plan generation
        remediation_plan = await ai_engine.generate_remediation_plan(drift_analysis)
        assert "actions" in remediation_plan
        print(f"   ✓ Remediation plan: {len(remediation_plan['actions'])} actions")
        
        # Test natural language processing (compatibility method)
        nl_intent = await ai_engine.parse_natural_language_intent(
            "Create a web server with load balancing", 
            {"environment": "production"}
        )
        assert isinstance(nl_intent, dict)
        print("   ✓ Natural language intent parsing")
        
        # Test configuration generation from intent
        config_from_intent = await ai_engine.generate_configuration_from_intent(nl_intent)
        assert isinstance(config_from_intent, dict)
        print("   ✓ Configuration generation from intent")
        
        # Test configuration generation from requirements
        requirements = {
            "name": "test-web-app",
            "type": "web_application",
            "requirements": {"scaling": "auto", "database": "postgresql"}
        }
        generated_config = await ai_engine.generate_configuration_from_requirements(requirements)
        assert isinstance(generated_config, dict)
        print("   ✓ Configuration generation from requirements")
        
        # Test AI metrics
        ai_metrics = await ai_engine.get_metrics()
        assert "ml_models" in ai_metrics
        print(f"   ✓ AI metrics retrieved: {len(ai_metrics['ml_models'])} models")
        
        # Test predictive insights
        insights = await ai_engine.generate_predictive_insights(resource.id, {"forecast_days": 7})
        assert "predictions" in insights
        print("   ✓ Predictive insights generated")
        
        # Shutdown
        await ai_engine.shutdown()
        print("   ✓ AI Engine shutdown completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ AI integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_service_with_ai():
    """Test service layer with advanced AI engine"""
    print("\n🔧 Testing Service Layer with Advanced AI...")
    
    try:
        conf_path = "/Users/nyimbiodero/src/pjs/apg/capabilities/common/conf"
        sys.path.insert(0, conf_path)
        
        # Import using absolute imports to avoid relative import issues
        import importlib.util
        service_spec = importlib.util.spec_from_file_location("service", f"{conf_path}/service.py")
        service_module = importlib.util.module_from_spec(service_spec)
        service_spec.loader.exec_module(service_module)
        RevolutionaryConfigurationManager = service_module.RevolutionaryConfigurationManager
        
        # Initialize manager with AI engine
        manager = RevolutionaryConfigurationManager(tenant_id="test_service_ai")
        await manager.initialize({})
        print("   ✓ Configuration Manager initialized with AI")
        
        # Test configuration creation with AI optimization
        config_data = {
            "name": "ai-optimized-web-server",
            "type": "web_server",
            "cloud_provider": "aws",
            "configuration": {
                "kind": "WebServer",
                "spec": {
                    "instance_type": "t3.medium",
                    "auto_scaling": {"min": 2, "max": 10}
                }
            },
            "created_by": "test_ai_user"
        }
        
        resource = await manager.create_configuration(config_data)
        assert resource.name == "ai-optimized-web-server"
        print(f"   ✓ AI-optimized configuration created: {resource.name}")
        
        # Test natural language configuration
        nl_result = await manager.natural_language_configuration(
            "Set up a high-availability database cluster with automated backups",
            {"environment": "production", "budget_limit": 5000}
        )
        assert "request" in nl_result
        assert "generated_configuration" in nl_result
        print("   ✓ Natural language configuration processing")
        
        # Test intelligent template creation
        template_requirements = {
            "name": "ai-generated-microservice-template",
            "description": "Auto-generated microservice deployment template",
            "category": "microservices",
            "requirements": {
                "scalability": "high",
                "monitoring": "enabled",
                "security": "enterprise"
            },
            "created_by": "ai_system"
        }
        
        template = await manager.create_intelligent_template(template_requirements)
        assert template.name == "ai-generated-microservice-template"
        print(f"   ✓ AI-generated template created: {template.name}")
        
        # Test predictive insights
        insights = await manager.get_predictive_insights(resource.id)
        assert "insights" in insights
        assert "generated_at" in insights
        print("   ✓ Predictive insights from service layer")
        
        # Test revolutionary metrics including AI metrics
        metrics = await manager.get_revolutionary_metrics()
        assert "ai_intelligence" in metrics
        assert "system_metrics" in metrics
        assert "performance_indicators" in metrics
        print(f"   ✓ Revolutionary metrics with AI: {len(metrics)} categories")
        
        await manager.shutdown()
        print("   ✓ Service shutdown completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Service AI integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run AI integration tests"""
    print("🚀 APG Configuration Management - AI Engine Integration Test")
    print("=" * 70)
    
    ai_test_success = await test_ai_integration()
    service_test_success = await test_service_with_ai()
    
    print("\n" + "=" * 70)
    if ai_test_success and service_test_success:
        print("🏆 AI INTEGRATION TESTS: PASSED ✅")
        print("   🧠 Advanced AI Engine fully operational")
        print("   🔗 Service layer integration successful")
        print("   🚀 Phase 3.1 AI Intelligence Engine: COMPLETE")
        print("   🎯 Ready for Phase 3.2: Universal Abstraction Layer")
    else:
        print("❌ AI INTEGRATION TESTS: FAILED")
        print("   🔍 Check error logs above for details")
    
    print("=" * 70)
    
    return ai_test_success and service_test_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)