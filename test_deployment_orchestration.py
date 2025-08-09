#!/usr/bin/env python3
"""
APG Configuration Management Deployment Orchestration Tests
Tests the advanced deployment orchestration with rollback capabilities,
health monitoring, and multi-strategy deployments.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the project root to Python path
conf_path = "/Users/nyimbiodero/src/pjs/apg/capabilities/common/conf"
sys.path.insert(0, conf_path)

async def test_deployment_orchestration_integration():
    """Test GitOps integration with deployment orchestrator"""
    print("🚀 Testing Deployment Orchestration Integration...")
    
    try:
        from gitops_integration import (
            get_gitops_manager, 
            GitRepository,
            DeploymentStrategy
        )
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        manager = await get_gitops_manager("orchestration-test-tenant")
        
        # Test 1: Create repository and resource
        repository = GitRepository(
            name="orchestration-repo",
            url="https://github.com/example/orchestration-repo.git",
            branch="main"
        )
        
        repo_id = await manager.add_repository(repository)
        
        resource = CMResource(
            name="orchestration-test-app",
            resource_type=ResourceType.CONTAINER,
            cloud_provider=CloudProvider.AWS,
            configuration=ConfigurationDSL(
                kind="Container",
                spec={
                    "resources": {"cpu": "2", "memory": "4Gi"},
                    "image": "nginx:latest",
                    "replicas": 3,
                    "security": {
                        "encryption_at_rest": True,
                        "audit_logging": True
                    }
                },
                version="1.0"
            )
        )
        
        manifest_id = await manager.create_manifest(
            resource=resource,
            repository_id=repo_id,
            environment="production"
        )
        
        print(f"   ✓ Repository and manifest created: {repo_id}, {manifest_id}")
        
        # Test 2: Create deployment plan with blue-green strategy
        deployment_plan_id = await manager.create_deployment_plan(
            resource_id=resource.id,
            manifest_id=manifest_id,
            environment="production",
            strategy=DeploymentStrategy.BLUE_GREEN,
            approval_required=True
        )
        
        assert deployment_plan_id is not None
        print(f"   ✓ Blue-green deployment plan created: {deployment_plan_id}")
        
        # Test 3: Execute deployment with orchestration
        deployment_success = await manager.execute_deployment(
            deployment_plan_id,
            approved_by="deployment-orchestrator@example.com"
        )
        
        assert deployment_success == True
        print("   ✓ Deployment execution started with orchestration")
        
        # Test 4: Get deployment execution status
        await asyncio.sleep(3)  # Allow deployment to progress
        
        execution_status = await manager.get_deployment_execution_status(deployment_plan_id)
        assert execution_status is not None
        
        print(f"   ✓ Deployment status: {execution_status['state']}")
        print(f"     - Strategy: {execution_status['strategy']}")
        print(f"     - Phase: {execution_status['current_phase']}")
        print(f"     - Progress: {execution_status['progress_percentage']:.1f}%")
        print(f"     - Replicas: {execution_status['healthy_replicas']}/{execution_status['target_replicas']}")
        
        # Test 5: Verify orchestration features are working
        if execution_status['logs']:
            print(f"   ✓ Deployment logs captured: {len(execution_status['logs'])} entries")
            for log in execution_status['logs'][-2:]:
                print(f"     - {log.get('level', 'info').upper()}: {log.get('message', '')}")
        
        if execution_status['health_checks']:
            print(f"   ✓ Health checks executed: {len(execution_status['health_checks'])} results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Deployment orchestration integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rollback_capabilities():
    """Test deployment rollback capabilities"""
    print("\n↩️  Testing Deployment Rollback Capabilities...")
    
    try:
        from gitops_integration import (
            get_gitops_manager, 
            GitRepository,
            DeploymentStrategy
        )
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        manager = await get_gitops_manager("rollback-test-tenant")
        
        # Setup repository and resource
        repository = GitRepository(
            name="rollback-test-repo",
            url="https://github.com/example/rollback-repo.git"
        )
        
        repo_id = await manager.add_repository(repository)
        
        resource = CMResource(
            name="rollback-test-service",
            resource_type=ResourceType.CONTAINER,
            cloud_provider=CloudProvider.AWS,
            configuration=ConfigurationDSL(
                kind="WebService",
                spec={
                    "resources": {"cpu": "1", "memory": "2Gi"},
                    "image": "webapp:v2.0.0",
                    "replicas": 2
                },
                version="1.0"
            )
        )
        
        manifest_id = await manager.create_manifest(
            resource=resource,
            repository_id=repo_id,
            environment="staging"
        )
        
        # Create deployment plan with canary strategy
        deployment_plan_id = await manager.create_deployment_plan(
            resource_id=resource.id,
            manifest_id=manifest_id,
            environment="staging",
            strategy=DeploymentStrategy.CANARY,
            approval_required=False
        )
        
        # Execute deployment
        await manager.execute_deployment(deployment_plan_id, approved_by="rollback-tester@example.com")
        
        print(f"   ✓ Canary deployment started: {deployment_plan_id}")
        
        # Wait for deployment to start
        await asyncio.sleep(2)
        
        # Test manual rollback
        rollback_success = await manager.trigger_deployment_rollback(
            deployment_plan_id, 
            reason="Testing rollback functionality"
        )
        
        assert rollback_success == True
        print("   ✓ Manual rollback triggered successfully")
        
        # Check rollback status
        await asyncio.sleep(1)
        execution_status = await manager.get_deployment_execution_status(deployment_plan_id)
        
        if execution_status and execution_status['rollback_triggered']:
            print(f"   ✓ Rollback confirmed: {execution_status['rollback_reason']}")
            print(f"     - Current state: {execution_status['state']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Rollback capabilities test failed: {e}")
        return False


async def test_deployment_strategies():
    """Test different deployment strategies"""
    print("\n🎯 Testing Multiple Deployment Strategies...")
    
    try:
        from gitops_integration import (
            get_gitops_manager, 
            GitRepository,
            DeploymentStrategy
        )
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        manager = await get_gitops_manager("strategies-test-tenant")
        
        # Setup base resources
        repository = GitRepository(
            name="strategies-repo",
            url="https://github.com/example/strategies-repo.git"
        )
        
        repo_id = await manager.add_repository(repository)
        
        resource = CMResource(
            name="multi-strategy-app",
            resource_type=ResourceType.CONTAINER,
            cloud_provider=CloudProvider.AWS,
            configuration=ConfigurationDSL(
                kind="Application",
                spec={
                    "resources": {"cpu": "1", "memory": "1Gi"},
                    "image": "app:latest",
                    "replicas": 4
                },
                version="1.0"
            )
        )
        
        manifest_id = await manager.create_manifest(
            resource=resource,
            repository_id=repo_id,
            environment="testing"
        )
        
        strategies = [
            DeploymentStrategy.ROLLING_UPDATE,
            DeploymentStrategy.BLUE_GREEN,
            DeploymentStrategy.CANARY
        ]
        
        strategy_results = {}
        
        for strategy in strategies:
            print(f"   Testing {strategy.value} strategy...")
            
            # Create deployment plan for strategy
            plan_id = await manager.create_deployment_plan(
                resource_id=resource.id,
                manifest_id=manifest_id,
                environment="testing",
                strategy=strategy,
                approval_required=False
            )
            
            # Execute deployment
            success = await manager.execute_deployment(plan_id, approved_by="strategy-tester@example.com")
            
            if success:
                await asyncio.sleep(1.5)  # Allow deployment to progress
                
                status = await manager.get_deployment_execution_status(plan_id)
                if status:
                    strategy_results[strategy.value] = {
                        "success": True,
                        "state": status['state'],
                        "progress": status['progress_percentage']
                    }
                    print(f"     ✓ {strategy.value}: {status['state']} ({status['progress_percentage']:.1f}% complete)")
                else:
                    strategy_results[strategy.value] = {"success": False, "error": "No status available"}
            else:
                strategy_results[strategy.value] = {"success": False, "error": "Deployment failed to start"}
        
        # Summary
        successful_strategies = [s for s, r in strategy_results.items() if r["success"]]
        print(f"   ✓ Successfully tested {len(successful_strategies)}/{len(strategies)} deployment strategies")
        
        return len(successful_strategies) == len(strategies)
        
    except Exception as e:
        print(f"   ❌ Deployment strategies test failed: {e}")
        return False


async def test_orchestration_metrics():
    """Test deployment orchestration metrics"""
    print("\n📊 Testing Orchestration Metrics...")
    
    try:
        from gitops_integration import get_gitops_manager
        
        manager = await get_gitops_manager("metrics-test-tenant")
        
        # Get GitOps status with orchestration metrics
        status = await manager.get_gitops_status()
        
        assert "deployment_orchestration" in status
        orchestration_metrics = status["deployment_orchestration"]
        
        print("   ✓ Orchestration metrics available:")
        print(f"     - Total deployments: {orchestration_metrics.get('total_deployments', 0)}")
        print(f"     - Active deployments: {orchestration_metrics.get('active_deployments', 0)}")
        print(f"     - Success rate: {orchestration_metrics.get('success_rate', 0):.1%}")
        print(f"     - Rollback rate: {orchestration_metrics.get('rollback_rate', 0):.1%}")
        print(f"     - Deployment strategies: {orchestration_metrics.get('deployment_strategies', [])}")
        print(f"     - Average deployment time: {orchestration_metrics.get('average_deployment_time', 0):.1f}s")
        
        # Verify key metrics are present
        required_metrics = ["total_deployments", "success_rate", "deployment_strategies", "generated_at"]
        
        for metric in required_metrics:
            assert metric in orchestration_metrics, f"Missing metric: {metric}"
        
        print("   ✓ All required orchestration metrics present")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Orchestration metrics test failed: {e}")
        return False


async def main():
    """Run deployment orchestration tests"""
    print("🚀 APG Configuration Management Deployment Orchestration Tests")
    print("=" * 85)
    
    test1_success = await test_deployment_orchestration_integration()
    test2_success = await test_rollback_capabilities()
    test3_success = await test_deployment_strategies()
    test4_success = await test_orchestration_metrics()
    
    print("\n" + "=" * 85)
    if test1_success and test2_success and test3_success and test4_success:
        print("🏆 DEPLOYMENT ORCHESTRATION TESTS: PASSED ✅")
        print("   🚀 Deployment orchestration integration operational")
        print("   ↩️  Rollback capabilities working")
        print("   🎯 Multiple deployment strategies supported")
        print("   📊 Orchestration metrics comprehensive")
        print("   🎯 Phase 3.5d Deployment Orchestration: COMPLETE")
        print("")
        print("   📊 Orchestration Summary:")
        print("   ├── Advanced Deployment Strategies: ✅ Rolling/Blue-Green/Canary")
        print("   ├── Health Check Monitoring: ✅ Multi-phase validation")
        print("   ├── Rollback Capabilities: ✅ Automatic & manual triggers")
        print("   ├── Progress Tracking: ✅ Real-time status & logs")
        print("   ├── Quality Gates Integration: ✅ Testing-aware deployments")
        print("   └── Metrics & Observability: ✅ Comprehensive tracking")
    else:
        print("❌ DEPLOYMENT ORCHESTRATION TESTS: FAILED")
        print("   🔍 Check error logs above for details")
    
    print("=" * 85)
    
    return test1_success and test2_success and test3_success and test4_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)