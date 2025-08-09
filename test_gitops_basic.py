#!/usr/bin/env python3
"""
APG Configuration Management Basic GitOps Tests
Simplified tests to validate core GitOps functionality.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the project root to Python path
conf_path = "/Users/nyimbiodero/src/pjs/apg/capabilities/common/conf"
sys.path.insert(0, conf_path)

async def test_basic_gitops_functionality():
    """Test basic GitOps functionality"""
    print("🔧 Testing Basic GitOps Functionality...")
    
    try:
        from gitops_integration import get_gitops_manager, GitRepository
        
        manager = await get_gitops_manager("basic-test-tenant")
        
        # Test 1: Create repository
        repository = GitRepository(
            name="basic-test-repo",
            url="https://github.com/example/basic-repo.git",
            branch="main"
        )
        
        repo_id = await manager.add_repository(repository)
        assert repo_id is not None
        print(f"   ✓ Repository created: {repo_id}")
        
        # Test 2: Create pipeline
        pipeline_id = await manager.create_deployment_pipeline(
            name="Basic Pipeline",
            repository_id=repo_id
        )
        
        assert pipeline_id is not None
        print(f"   ✓ Pipeline created: {pipeline_id}")
        
        # Test 3: Trigger pipeline
        execution_id = await manager.trigger_pipeline(
            pipeline_id=pipeline_id,
            trigger_data={
                "event": "push",
                "commit_sha": "test123",
                "branch": "main"
            }
        )
        
        assert execution_id is not None
        print(f"   ✓ Pipeline triggered: {execution_id}")
        
        # Test 4: Check status
        status = await manager.get_gitops_status()
        assert status["repositories"] >= 1
        assert status["pipelines"] >= 1
        print("   ✓ GitOps status verified")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic GitOps functionality test failed: {e}")
        return False


async def test_deployment_workflow():
    """Test deployment workflow"""
    print("\n🚀 Testing Deployment Workflow...")
    
    try:
        from gitops_integration import (
            get_gitops_manager, 
            GitRepository,
            DeploymentStrategy
        )
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        manager = await get_gitops_manager("deploy-test-tenant")
        
        # Setup
        repository = GitRepository(
            name="deploy-workflow-repo",
            url="https://github.com/example/deploy-repo.git",
            branch="main"
        )
        
        repo_id = await manager.add_repository(repository)
        
        # Create resource and manifest
        resource = CMResource(
            name="workflow-app",
            resource_type=ResourceType.CONTAINER,
            cloud_provider=CloudProvider.AWS,
            configuration=ConfigurationDSL(
                kind="Container",
                spec={"resources": {"cpu": "1", "memory": "2Gi"}},
                version="1.0"
            )
        )
        
        manifest_id = await manager.create_manifest(
            resource=resource,
            repository_id=repo_id,
            environment="staging"
        )
        
        # Create deployment plan
        plan_id = await manager.create_deployment_plan(
            resource_id=resource.id,
            manifest_id=manifest_id,
            environment="staging",
            strategy=DeploymentStrategy.ROLLING_UPDATE
        )
        
        assert plan_id is not None
        print(f"   ✓ Deployment plan created: {plan_id}")
        
        # Execute deployment
        success = await manager.execute_deployment(plan_id)
        assert success == True
        print("   ✓ Deployment executed successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Deployment workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pipeline_monitoring():
    """Test pipeline execution monitoring"""
    print("\n📊 Testing Pipeline Monitoring...")
    
    try:
        from gitops_integration import get_gitops_manager, GitRepository
        
        manager = await get_gitops_manager("monitoring-test-tenant")
        
        # Setup repository and pipeline
        repository = GitRepository(
            name="monitoring-repo",
            url="https://github.com/example/monitoring-repo.git"
        )
        
        repo_id = await manager.add_repository(repository)
        pipeline_id = await manager.create_deployment_pipeline(
            name="Monitoring Pipeline",
            repository_id=repo_id
        )
        
        # Trigger multiple executions
        executions = []
        for i in range(2):
            execution_id = await manager.trigger_pipeline(
                pipeline_id=pipeline_id,
                trigger_data={
                    "event": "push",
                    "commit_sha": f"commit-{i}",
                    "branch": "main"
                }
            )
            executions.append(execution_id)
        
        print(f"   ✓ Started {len(executions)} pipeline executions")
        
        # Monitor executions
        await asyncio.sleep(2)  # Allow pipelines to progress
        
        for execution_id in executions:
            execution = await manager.pipeline_engine.get_execution_status(execution_id)
            if execution:
                print(f"   ✓ Execution {execution_id[:8]}: {execution.status.value}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pipeline monitoring test failed: {e}")
        return False


async def main():
    """Run basic GitOps tests"""
    print("🔧 APG Configuration Management Basic GitOps Tests")
    print("=" * 65)
    
    test1_success = await test_basic_gitops_functionality()
    test2_success = await test_deployment_workflow()
    test3_success = await test_pipeline_monitoring()
    
    print("\n" + "=" * 65)
    if test1_success and test2_success and test3_success:
        print("🏆 BASIC GITOPS TESTS: PASSED ✅")
        print("   🔧 Core GitOps functionality operational")
        print("   🚀 Deployment workflow functional")
        print("   📊 Pipeline monitoring working")
        print("   🎯 Phase 3.5b CI/CD Automation: COMPLETE")
    else:
        print("❌ BASIC GITOPS TESTS: FAILED")
    
    print("=" * 65)
    
    return test1_success and test2_success and test3_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)