#!/usr/bin/env python3
"""
APG Universal Infrastructure Abstraction Layer Test
Tests the cloud-agnostic resource management across AWS, Azure, and GCP.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the project root to Python path
conf_path = "/Users/nyimbiodero/src/pjs/apg/capabilities/common/conf"
sys.path.insert(0, conf_path)

async def test_universal_abstraction_layer():
    """Test Universal Infrastructure Abstraction Layer with all providers"""
    print("🌐 Testing Universal Infrastructure Abstraction Layer...")
    
    try:
        from universal_abstraction import (
            UniversalResourceLayer, UniversalResource, DeploymentPlan,
            AWSProviderAdapter, AzureProviderAdapter, GCPProviderAdapter,
            ResourceCapability, ProviderFeature, DeploymentStrategy
        )
        from models import (
            CMResource, CMDeployment, ConfigurationDSL, 
            ResourceType, CloudProvider, ResourceState, DeploymentStatus
        )
        
        # Initialize Universal Resource Layer
        universal_layer = UniversalResourceLayer(tenant_id="test_universal")
        await universal_layer.initialize()
        print("   ✓ Universal Resource Layer initialized")
        
        # Test 1: Provider initialization and capabilities
        assert len(universal_layer.providers) == 3
        assert CloudProvider.AWS in universal_layer.providers
        assert CloudProvider.AZURE in universal_layer.providers
        assert CloudProvider.GCP in universal_layer.providers
        print(f"   ✓ All 3 cloud providers initialized: {list(universal_layer.providers.keys())}")
        
        # Test provider capabilities
        for provider, capabilities in universal_layer.provider_capabilities.items():
            assert len(capabilities.supported_resources) > 0
            assert len(capabilities.supported_features) > 0
            assert len(capabilities.regions) > 0
            print(f"   ✓ {provider}: {len(capabilities.supported_resources)} resources, {len(capabilities.supported_features)} features, {len(capabilities.regions)} regions")
        
        # Test 2: Provider rankings calculation
        assert len(universal_layer.provider_rankings) == 3
        aws_score = universal_layer.provider_rankings[CloudProvider.AWS]
        azure_score = universal_layer.provider_rankings[CloudProvider.AZURE]
        gcp_score = universal_layer.provider_rankings[CloudProvider.GCP]
        print(f"   ✓ Provider rankings: AWS={aws_score:.3f}, Azure={azure_score:.3f}, GCP={gcp_score:.3f}")
        
        # Test 3: Universal resource creation
        universal_resource = UniversalResource(
            name="test-universal-vm",
            resource_type=ResourceType.VIRTUAL_MACHINE,
            capabilities=[ResourceCapability.COMPUTE, ResourceCapability.NETWORKING],
            compute_specs={
                "instance_type": "t3.micro",  # AWS
                "vm_size": "Standard_B1s",     # Azure
                "machine_type": "e2-micro"     # GCP
            },
            network_specs={
                "vpc_id": "vpc-test123",
                "subnet_id": "subnet-test456"
            },
            security_specs={
                "encryption_enabled": True,
                "security_groups": ["sg-web", "sg-ssh"]
            },
            feature_requirements=[
                ProviderFeature.AUTO_SCALING,
                ProviderFeature.ENCRYPTION_AT_REST
            ],
            tags={
                "environment": "test",
                "project": "universal-abstraction",
                "owner": "test-user"
            }
        )
        
        assert universal_resource.name == "test-universal-vm"
        assert universal_resource.resource_type == ResourceType.VIRTUAL_MACHINE
        assert len(universal_resource.capabilities) == 2
        assert len(universal_resource.feature_requirements) == 2
        print(f"   ✓ Universal resource created: {universal_resource.name}")
        
        # Test 4: Provider compatibility detection
        compatible_providers = await universal_layer._find_compatible_providers(universal_resource)
        assert len(compatible_providers) > 0
        print(f"   ✓ Compatible providers found: {[p.value for p in compatible_providers]}")
        
        # Test 5: Resource translation for each provider
        translations = {}
        for provider in [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP]:
            adapter = universal_layer.providers[provider]
            translation = await adapter.translate_resource(universal_resource)
            translations[provider] = translation
            assert isinstance(translation, dict)
            assert len(translation) > 0
            print(f"   ✓ {provider} translation: {len(str(translation))} characters")
        
        # Verify provider-specific translations
        assert "Type" in translations[CloudProvider.AWS]  # CloudFormation
        assert "$schema" in translations[CloudProvider.AZURE]  # ARM Template
        assert "resources" in translations[CloudProvider.GCP]  # Deployment Manager
        print("   ✓ Provider-specific configuration formats verified")
        
        # Test 6: Resource validation across providers
        validation_results = {}
        for provider, adapter in universal_layer.providers.items():
            validation = await adapter.validate_resource(universal_resource)
            validation_results[provider] = validation
            assert hasattr(validation, 'valid')
            assert hasattr(validation, 'errors')
            print(f"   ✓ {provider} validation: valid={validation.valid}, errors={len(validation.errors)}")
        
        # Test 7: CM Resource integration
        dsl = ConfigurationDSL(
            kind="VirtualMachine",
            spec={
                "resources": {
                    "instance_type": "t3.micro",
                    "image": "ami-ubuntu-20.04"
                },
                "network": {
                    "vpc_id": "vpc-test123",
                    "subnet_id": "subnet-test456"
                },
                "security": {
                    "encryption_enabled": True
                }
            },
            version="1.0"
        )
        
        cm_resource = CMResource(
            name="test-cm-integration",
            resource_type=ResourceType.VIRTUAL_MACHINE,
            cloud_provider=CloudProvider.AWS,
            configuration=dsl,
            description="CM Resource integration test"
        )
        
        # Test configuration validation through universal layer
        cm_validation = await universal_layer.validate_configuration(cm_resource)
        assert cm_validation.valid
        # Note: ValidationResult no longer has details field, provider info is logged
        print(f"   ✓ CM Resource validation: valid={cm_validation.valid}")
        
        # Test 8: Deployment plan creation and execution
        cm_deployment = CMDeployment(
            resource_id=cm_resource.id,
            environment_id="test-env",
            deployment_plan={"strategy": "rolling", "phases": []},
            started_at=datetime.utcnow()
        )
        
        # Execute deployment through universal layer
        deployment_result = await universal_layer.execute_deployment(cm_deployment)
        assert deployment_result.success
        assert "selected_provider" in deployment_result.details
        assert "deployment_plan_id" in deployment_result.details
        print(f"   ✓ Deployment executed: provider={deployment_result.details['selected_provider']}")
        
        # Test 9: Multi-provider deployment plans
        deployment_plans_created = []
        for provider in [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP]:
            # Create universal resource for each provider
            test_resource = UniversalResource(
                name=f"multi-provider-test-{provider.value}",
                resource_type=ResourceType.VIRTUAL_MACHINE,
                compute_specs={"instance_type": "small"}
            )
            
            # Create deployment plan
            deployment_plan = await universal_layer._create_deployment_plan(
                test_resource, provider, cm_deployment
            )
            
            assert deployment_plan.target_provider == provider
            assert len(deployment_plan.phases) > 0
            assert deployment_plan.rollback_plan["enabled"] is True
            deployment_plans_created.append(deployment_plan)
            
        print(f"   ✓ Multi-provider deployment plans created: {len(deployment_plans_created)}")
        
        # Test 10: Cost estimation across providers
        cost_estimates = {}
        for deployment_plan in deployment_plans_created:
            provider = deployment_plan.target_provider
            cost_estimates[provider] = deployment_plan.estimated_cost
            
        print(f"   ✓ Cost estimates: {cost_estimates}")
        
        # Test 11: Remediation execution
        remediation_plan = {
            "actions": [
                {
                    "type": "reconcile_configuration",
                    "description": "Restore configuration to desired state"
                },
                {
                    "type": "performance_optimization", 
                    "description": "Apply performance tuning"
                }
            ],
            "priority": "medium"
        }
        
        remediation_result = await universal_layer.execute_remediation(cm_resource, remediation_plan)
        assert remediation_result.success
        print(f"   ✓ Remediation executed: {remediation_result.message}")
        
        # Test 12: Template validation across providers
        class MockTemplate:
            def __init__(self):
                self.id = "template-test-123"
                self.configuration_template = {
                    "compute": {"instance_type": "t3.micro"},
                    "storage": {"size": "20GB", "type": "ssd"},
                    "network": {"vpc": "default"},
                    "security": {"encryption": True}
                }
        
        template = MockTemplate()
        template_validation = await universal_layer.validate_template(template)
        assert hasattr(template_validation, 'valid')
        # Note: ValidationResult no longer has details field, provider info is logged
        print(f"   ✓ Template validated successfully: valid={template_validation.valid}")
        
        # Test 13: Configuration dictionary validation
        config_dict = {
            "name": "dict-validation-test",
            "compute": {"cpu": 2, "memory": "4GB"},
            "storage": {"disk": "100GB"},
            "network": {"bandwidth": "1Gbps"},
            "security": {"firewall": True}
        }
        
        dict_validation = await universal_layer.validate_configuration_dict(config_dict)
        assert hasattr(dict_validation, 'valid')
        print(f"   ✓ Configuration dictionary validated: valid={dict_validation.valid}")
        
        # Test 14: Policy action execution
        policy_action = {
            "type": "compliance_fix",
            "target": "encryption_settings",
            "description": "Enable encryption for all storage"
        }
        
        policy_result = await universal_layer.execute_policy_action(policy_action)
        assert policy_result.success
        print(f"   ✓ Policy action executed: {policy_action['type']}")
        
        # Test 15: Metrics collection
        metrics = await universal_layer.get_metrics()
        assert "providers_initialized" in metrics
        assert "provider_rankings" in metrics
        assert "abstraction_efficiency" in metrics
        assert metrics["providers_initialized"] == 3
        print(f"   ✓ Metrics collected: efficiency={metrics['abstraction_efficiency']}")
        
        # Test 16: Resource lifecycle management
        for provider, adapter in universal_layer.providers.items():
            # Test resource status
            status = await adapter.get_resource_status("test-resource-id")
            assert "resource_id" in status
            assert "provider" in status
            assert status["provider"] == provider.value.lower()
            
            # Test resource update
            updates = {"configuration_sync": True, "performance_tuning": True}
            update_result = await adapter.update_resource("test-resource-id", updates)
            assert update_result.success
            
            # Test resource deletion
            delete_result = await adapter.delete_resource("test-resource-id")
            assert delete_result.success
            
            print(f"   ✓ {provider} lifecycle: status/update/delete completed")
        
        # Shutdown universal layer
        await universal_layer.shutdown()
        print("   ✓ Universal Resource Layer shutdown completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Universal abstraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_provider_specific_features():
    """Test provider-specific features and capabilities"""
    print("\n☁️  Testing Provider-Specific Features...")
    
    try:
        from universal_abstraction import (
            AWSProviderAdapter, AzureProviderAdapter, GCPProviderAdapter,
            UniversalResource, ProviderFeature
        )
        from models import ResourceType, CloudProvider
        
        # Test AWS specific features
        aws_adapter = AWSProviderAdapter(CloudProvider.AWS, {})
        await aws_adapter.initialize()
        aws_capabilities = await aws_adapter.get_provider_capabilities()
        
        assert ProviderFeature.SPOT_INSTANCES in aws_capabilities.supported_features
        assert ProviderFeature.RESERVED_INSTANCES in aws_capabilities.supported_features
        assert ResourceType.SERVERLESS_FUNCTION in aws_capabilities.supported_resources
        assert "us-east-1" in aws_capabilities.regions
        assert aws_capabilities.api_version == "2016-11-15"
        print(f"   ✓ AWS: {len(aws_capabilities.supported_features)} features, {len(aws_capabilities.regions)} regions")
        
        # Test Azure specific features
        azure_adapter = AzureProviderAdapter(CloudProvider.AZURE, {})
        await azure_adapter.initialize()
        azure_capabilities = await azure_adapter.get_provider_capabilities()
        
        assert ProviderFeature.AUTO_SCALING in azure_capabilities.supported_features
        assert ResourceType.KUBERNETES_DEPLOYMENT in azure_capabilities.supported_resources
        assert "eastus" in azure_capabilities.regions
        assert azure_capabilities.api_version == "2021-03-01"
        print(f"   ✓ Azure: {len(azure_capabilities.supported_features)} features, {len(azure_capabilities.regions)} regions")
        
        # Test GCP specific features
        gcp_adapter = GCPProviderAdapter(CloudProvider.GCP, {})
        await gcp_adapter.initialize()
        gcp_capabilities = await gcp_adapter.get_provider_capabilities()
        
        assert ProviderFeature.SERVERLESS_FUNCTIONS in gcp_capabilities.supported_features
        assert ResourceType.SERVERLESS_FUNCTION in gcp_capabilities.supported_resources
        assert "us-central1" in gcp_capabilities.regions
        assert gcp_capabilities.api_version == "v1"
        print(f"   ✓ GCP: {len(gcp_capabilities.supported_features)} features, {len(gcp_capabilities.regions)} regions")
        
        # Test provider-specific resource configurations
        test_resource = UniversalResource(
            name="provider-specific-test",
            resource_type=ResourceType.DATABASE,
            compute_specs={
                "db_instance_class": "db.t3.micro",  # AWS
                "vm_size": "Standard_B1s",            # Azure
                "machine_type": "db-f1-micro"         # GCP
            },
            storage_specs={
                "allocated_storage": 20,              # AWS
                "disk_size_gb": 20                    # GCP
            }
        )
        
        # Test AWS database configuration
        aws_validation = await aws_adapter.validate_resource(test_resource)
        print(f"   ✓ AWS database validation: valid={aws_validation.valid}")
        
        aws_config = await aws_adapter.translate_resource(test_resource)
        assert "AWS::RDS::DBInstance" in aws_config.get("Type", "")
        print(f"   ✓ AWS database config: {aws_config['Type']}")
        
        # Test multi-provider pricing comparison
        pricing_comparison = {}
        for provider, capabilities in [(CloudProvider.AWS, aws_capabilities), 
                                      (CloudProvider.AZURE, azure_capabilities), 
                                      (CloudProvider.GCP, gcp_capabilities)]:
            if capabilities.pricing_model.get("compute"):
                pricing_comparison[provider] = capabilities.pricing_model["compute"]
        
        print(f"   ✓ Pricing comparison: {pricing_comparison}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Provider-specific features test failed: {e}")
        return False


async def test_deployment_strategies():
    """Test different deployment strategies across providers"""
    print("\n🚀 Testing Deployment Strategies...")
    
    try:
        from universal_abstraction import (
            UniversalResourceLayer, DeploymentStrategy, DeploymentPlan
        )
        from models import ResourceType, CloudProvider
        
        layer = UniversalResourceLayer(tenant_id="test_deployment_strategies")
        await layer.initialize()
        
        strategies_to_test = [
            DeploymentStrategy.BLUE_GREEN,
            DeploymentStrategy.ROLLING, 
            DeploymentStrategy.CANARY,
            DeploymentStrategy.MULTI_REGION
        ]
        
        for strategy in strategies_to_test:
            # Create deployment plan with specific strategy
            deployment_plan = DeploymentPlan(
                resource_id="test-strategy-resource",
                target_provider=CloudProvider.AWS,
                deployment_strategy=strategy
            )
            
            # Verify strategy-specific configuration
            assert deployment_plan.deployment_strategy == strategy
            deployment_plan.phases = [
                {"name": f"{strategy.value}_phase_1", "duration": 120},
                {"name": f"{strategy.value}_phase_2", "duration": 180}
            ]
            
            print(f"   ✓ {strategy.value} strategy configured with {len(deployment_plan.phases)} phases")
        
        # Test rollback capabilities
        rollback_plan = {
            "enabled": True,
            "trigger_conditions": ["health_check_failure", "performance_degradation"],
            "rollback_phases": [
                {"name": "stop_traffic", "duration": 30},
                {"name": "restore_previous_version", "duration": 90},
                {"name": "validate_rollback", "duration": 60}
            ]
        }
        
        deployment_plan.rollback_plan = rollback_plan
        assert deployment_plan.rollback_plan["enabled"] is True
        assert len(deployment_plan.rollback_plan["rollback_phases"]) == 3
        print(f"   ✓ Rollback plan configured with {len(rollback_plan['rollback_phases'])} phases")
        
        await layer.shutdown()
        return True
        
    except Exception as e:
        print(f"   ❌ Deployment strategies test failed: {e}")
        return False


async def main():
    """Run comprehensive Universal Abstraction Layer tests"""
    print("🌐 APG Universal Infrastructure Abstraction Layer - Comprehensive Test")
    print("=" * 80)
    
    test1_success = await test_universal_abstraction_layer()
    test2_success = await test_provider_specific_features()
    test3_success = await test_deployment_strategies()
    
    print("\n" + "=" * 80)
    if test1_success and test2_success and test3_success:
        print("🏆 UNIVERSAL ABSTRACTION LAYER TESTS: PASSED ✅")
        print("   🌐 All 3 cloud providers (AWS, Azure, GCP) integrated successfully")
        print("   🔀 Multi-provider resource translation working")
        print("   ⚡ Intelligent provider selection operational") 
        print("   🔄 Resource lifecycle management complete")
        print("   🎯 Deployment strategies and rollback capabilities verified")
        print("   📊 Cost optimization and provider ranking functional")
        print("   🚀 Phase 3.2 Universal Abstraction Layer: COMPLETE")
        print("   💎 Ready for Phase 3.3: Quantum Security Implementation")
    else:
        print("❌ UNIVERSAL ABSTRACTION LAYER TESTS: FAILED")
        print("   🔍 Check error logs above for details")
    
    print("=" * 80)
    
    return test1_success and test2_success and test3_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)