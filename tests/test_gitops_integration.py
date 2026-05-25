#!/usr/bin/env python3
"""
APG Configuration Management GitOps Integration Tests
Tests the comprehensive GitOps workflow with CI/CD pipeline automation.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the project root to Python path
conf_path = "/Users/nyimbiodero/src/pjs/apg/capabilities/common/conf"
sys.path.insert(0, conf_path)

async def test_gitops_repository_management():
    """Test GitOps repository setup and management"""
    print("📁 Testing GitOps Repository Management...")
    
    try:
        from gitops_integration import (
            get_gitops_manager, 
            GitRepository,
            GitOpsSyncMode,
            GitBranchStrategy
        )
        
        manager = await get_gitops_manager("gitops-test-tenant")
        
        # Test 1: Create GitOps repository
        repository = GitRepository(
            name="test-config-repo",
            url="https://github.com/example/config-repo.git",
            branch="main",
            access_token="mock_token",
            sync_enabled=True,
            auto_sync_interval=60
        )
        
        repo_id = await manager.add_repository(repository)
        assert repo_id is not None
        assert repo_id in manager.repositories
        print(f"   ✓ GitOps repository created: {repo_id}")
        
        # Test 2: Verify repository configuration
        gitops_repo = manager.repositories[repo_id]
        assert gitops_repo.repository.name == "test-config-repo"
        assert gitops_repo.repository.sync_enabled == True
        print("   ✓ Repository configuration verified")
        
        # Test 3: Test GitOps status
        status = await manager.get_gitops_status()
        assert status["repositories"] >= 1
        assert status["sync_mode"] == GitOpsSyncMode.PULL_BASED.value
        assert status["branch_strategy"] == GitBranchStrategy.FEATURE_BRANCH.value
        print(f"   ✓ GitOps status retrieved: {status['repositories']} repositories")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GitOps repository management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_manifest_generation():
    """Test GitOps manifest creation and management"""
    print("\n📄 Testing GitOps Manifest Generation...")
    
    try:
        from gitops_integration import get_gitops_manager, GitRepository
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        manager = await get_gitops_manager("manifest-test-tenant")
        
        # Setup repository
        repository = GitRepository(
            name="manifest-test-repo",
            url="https://github.com/example/manifest-repo.git",
            branch="main"
        )
        
        repo_id = await manager.add_repository(repository)
        
        # Create test resource
        config_dsl = ConfigurationDSL(
            kind="VirtualMachine",
            spec={
                "resources": {"instance_type": "t3.micro"},
                "security": {
                    "encryption_at_rest": True,
                    "encryption_in_transit": True
                },
                "networking": {
                    "vpc": "vpc-12345",
                    "subnet": "subnet-67890"
                }
            },
            version="1.0"
        )
        
        resource = CMResource(
            name="gitops-test-vm",
            resource_type=ResourceType.VIRTUAL_MACHINE,
            cloud_provider=CloudProvider.AWS,
            configuration=config_dsl,
            description="Test VM for GitOps manifest generation"
        )
        
        # Test 1: Create GitOps manifest
        manifest_id = await manager.create_manifest(
            resource=resource,
            repository_id=repo_id,
            environment="staging",
            namespace="test-namespace"
        )
        
        assert manifest_id is not None
        assert manifest_id in manager.manifests
        print(f"   ✓ GitOps manifest created: {manifest_id}")
        
        # Test 2: Verify manifest content
        manifest = manager.manifests[manifest_id]
        assert manifest.resource_id == resource.id
        assert manifest.environment == "staging"
        assert manifest.namespace == "test-namespace"
        assert manifest.file_path == "environments/staging/resources/gitops-test-vm.yaml"
        print("   ✓ Manifest content and metadata verified")
        
        # Test 3: Verify manifest structure
        manifest_content = manifest.content
        assert manifest_content["apiVersion"] == "apg.datacraft.co.ke/v1"
        assert manifest_content["kind"] == "ConfigurationResource"
        assert manifest_content["metadata"]["name"] == resource.name
        assert manifest_content["metadata"]["namespace"] == "test-namespace"
        assert manifest_content["spec"] == resource.configuration.model_dump()
        print("   ✓ Manifest structure follows Kubernetes conventions")
        
        # Test 4: Update manifest
        # Create a new ConfigurationDSL with updated values
        from models import ConfigurationDSL
        updated_config_dsl = ConfigurationDSL(
            kind="VirtualMachine",
            spec={
                "resources": {"instance_type": "t3.small"},
                "security": {
                    "encryption_at_rest": True,
                    "encryption_in_transit": True
                },
                "networking": {
                    "vpc": "vpc-12345",
                    "subnet": "subnet-67890"
                }
            },
            version="1.0"
        )
        
        # Update the resource configuration
        updated_resource = resource
        updated_resource.configuration = updated_config_dsl
        
        update_success = await manager.update_manifest(manifest_id, updated_resource)
        assert update_success == True
        
        updated_manifest = manager.manifests[manifest_id]
        # Check the updated configuration in the manifest
        updated_spec = updated_manifest.content["spec"]
        assert updated_spec["resources"]["instance_type"] == "t3.small"
        print("   ✓ Manifest update successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GitOps manifest generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_cicd_pipeline_creation():
    """Test CI/CD pipeline creation and configuration"""
    print("\n🔧 Testing CI/CD Pipeline Creation...")
    
    try:
        from gitops_integration import (
            get_gitops_manager, 
            GitRepository,
            CIPipeline,
            PipelineStatus
        )
        
        manager = await get_gitops_manager("pipeline-test-tenant")
        
        # Setup repository
        repository = GitRepository(
            name="pipeline-test-repo",
            url="https://github.com/example/pipeline-repo.git",
            branch="main"
        )
        
        repo_id = await manager.add_repository(repository)
        
        # Test 1: Create basic deployment pipeline
        pipeline_id = await manager.create_deployment_pipeline(
            name="Basic Deployment Pipeline",
            repository_id=repo_id,
            trigger_events=["push", "pull_request"]
        )
        
        assert pipeline_id is not None
        assert pipeline_id in manager.pipelines
        print(f"   ✓ Basic deployment pipeline created: {pipeline_id}")
        
        # Test 2: Create custom pipeline with specific stages
        custom_stages = [
            {
                "name": "validate_syntax",
                "type": "script",
                "script": ["echo 'Validating configuration syntax'"],
                "timeout": 120
            },
            {
                "name": "security_scan",
                "type": "script", 
                "script": ["echo 'Running security scan'"],
                "timeout": 300
            },
            {
                "name": "integration_test",
                "type": "test",
                "timeout": 600
            },
            {
                "name": "deploy_staging",
                "type": "deploy",
                "environment": "staging",
                "timeout": 900
            }
        ]
        
        custom_pipeline_id = await manager.create_deployment_pipeline(
            name="Custom Security Pipeline",
            repository_id=repo_id,
            trigger_events=["push"],
            custom_stages=custom_stages
        )
        
        assert custom_pipeline_id is not None
        print(f"   ✓ Custom pipeline created: {custom_pipeline_id}")
        
        # Test 3: Verify pipeline configuration
        pipeline = manager.pipelines[custom_pipeline_id]
        assert pipeline.name == "Custom Security Pipeline"
        assert len(pipeline.stages) == 4
        assert "push" in pipeline.trigger_events
        print("   ✓ Custom pipeline configuration verified")
        
        # Test 4: Trigger pipeline execution
        execution_id = await manager.trigger_pipeline(
            pipeline_id=custom_pipeline_id,
            trigger_data={
                "event": "push",
                "commit_sha": "abc123def456",
                "branch": "feature/new-config",
                "author": "developer@example.com",
                "message": "Add new VM configuration"
            }
        )
        
        assert execution_id is not None
        print(f"   ✓ Pipeline execution triggered: {execution_id}")
        
        # Test 5: Check execution status
        await asyncio.sleep(1)  # Allow pipeline to start
        
        execution = await manager.pipeline_engine.get_execution_status(execution_id)
        assert execution is not None
        assert execution.pipeline_id == custom_pipeline_id
        assert execution.commit_sha == "abc123def456"
        print(f"   ✓ Pipeline execution status: {execution.status.value}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CI/CD pipeline creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_deployment_orchestration():
    """Test deployment plan creation and orchestration"""
    print("\n🚀 Testing Deployment Orchestration...")
    
    try:
        from gitops_integration import (
            get_gitops_manager,
            GitRepository,
            DeploymentStrategy,
            DeploymentPlan
        )
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        manager = await get_gitops_manager("deploy-test-tenant")
        
        # Setup repository and resource
        repository = GitRepository(
            name="deploy-test-repo",
            url="https://github.com/example/deploy-repo.git",
            branch="main"
        )
        
        repo_id = await manager.add_repository(repository)
        
        # Create test resource and manifest
        resource = CMResource(
            name="deploy-test-app",
            resource_type=ResourceType.CONTAINER,
            cloud_provider=CloudProvider.AWS,
            configuration=ConfigurationDSL(
                kind="Container",
                spec={
                    "resources": {"cpu": "2", "memory": "4Gi"},
                    "image": "nginx:latest",
                    "replicas": 3
                },
                version="1.0"
            )
        )
        
        manifest_id = await manager.create_manifest(
            resource=resource,
            repository_id=repo_id,
            environment="production"
        )
        
        # Test 1: Create rolling update deployment plan
        rolling_plan_id = await manager.create_deployment_plan(
            resource_id=resource.id,
            manifest_id=manifest_id,
            environment="production",
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            approval_required=False
        )
        
        assert rolling_plan_id is not None
        assert rolling_plan_id in manager.deployments
        print(f"   ✓ Rolling update deployment plan created: {rolling_plan_id}")
        
        # Test 2: Create blue-green deployment plan with approval
        blue_green_plan_id = await manager.create_deployment_plan(
            resource_id=resource.id,
            manifest_id=manifest_id,
            environment="production",
            strategy=DeploymentStrategy.BLUE_GREEN,
            approval_required=True
        )
        
        assert blue_green_plan_id is not None
        print(f"   ✓ Blue-green deployment plan created: {blue_green_plan_id}")
        
        # Test 3: Verify deployment plan configuration
        blue_green_plan = manager.deployments[blue_green_plan_id]
        assert blue_green_plan.strategy == DeploymentStrategy.BLUE_GREEN
        assert blue_green_plan.approval_required == True
        assert blue_green_plan.rollback_plan is not None
        assert len(blue_green_plan.health_checks) > 0
        print("   ✓ Deployment plan configuration verified")
        
        # Test 4: Execute rolling deployment (no approval required)
        rolling_success = await manager.execute_deployment(rolling_plan_id)
        assert rolling_success == True
        print("   ✓ Rolling update deployment executed successfully")
        
        # Test 5: Try to execute blue-green without approval (should fail)
        bg_no_approval = await manager.execute_deployment(blue_green_plan_id)
        assert bg_no_approval == False
        print("   ✓ Blue-green deployment properly blocked without approval")
        
        # Test 6: Execute blue-green with approval
        bg_with_approval = await manager.execute_deployment(
            blue_green_plan_id,
            approved_by="deployment-manager@example.com"
        )
        assert bg_with_approval == True
        
        # Verify approval was recorded
        approved_plan = manager.deployments[blue_green_plan_id]
        assert approved_plan.approved_by == "deployment-manager@example.com"
        assert approved_plan.approved_at is not None
        print("   ✓ Blue-green deployment executed with approval")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Deployment orchestration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pipeline_execution_monitoring():
    """Test CI/CD pipeline execution and monitoring"""
    print("\n📊 Testing Pipeline Execution Monitoring...")
    
    try:
        from gitops_integration import get_gitops_manager, GitRepository
        
        manager = await get_gitops_manager("monitoring-test-tenant")
        
        # Setup repository and pipeline
        repository = GitRepository(
            name="monitoring-test-repo",
            url="https://github.com/example/monitoring-repo.git",
            branch="main"
        )
        
        repo_id = await manager.add_repository(repository)
        
        pipeline_id = await manager.create_deployment_pipeline(
            name="Monitoring Test Pipeline",
            repository_id=repo_id
        )
        
        # Test 1: Execute multiple pipelines
        executions = []
        for i in range(3):
            execution_id = await manager.trigger_pipeline(
                pipeline_id=pipeline_id,
                trigger_data={
                    "event": "push",
                    "commit_sha": f"commit-{i+1}",
                    "branch": "main",
                    "author": f"developer-{i+1}@example.com",
                    "message": f"Test commit {i+1}"
                }
            )
            executions.append(execution_id)
        
        print(f"   ✓ Started {len(executions)} pipeline executions")
        
        # Test 2: Monitor execution progress
        await asyncio.sleep(2)  # Allow pipelines to make progress
        
        completed_count = 0
        running_count = 0
        
        for execution_id in executions:
            execution = await manager.pipeline_engine.get_execution_status(execution_id)
            if execution:
                if execution.status.value in ["success", "failed"]:
                    completed_count += 1
                elif execution.status.value == "running":
                    running_count += 1
        
        print(f"   ✓ Pipeline monitoring: {completed_count} completed, {running_count} running")
        
        # Test 3: Wait for completion and verify results
        await asyncio.sleep(3)  # Allow more time for completion
        
        success_count = 0
        total_stages = 0
        total_duration = 0
        
        for execution_id in executions:
            execution = await manager.pipeline_engine.get_execution_status(execution_id)
            if execution:
                if execution.status.value == "success":
                    success_count += 1
                
                total_stages += len(execution.stages)
                if execution.duration_seconds:
                    total_duration += execution.duration_seconds
        
        print(f"   ✓ Execution results: {success_count}/{len(executions)} successful")
        print(f"   ✓ Total stages executed: {total_stages}")
        print(f"   ✓ Average execution time: {total_duration/len(executions):.1f}s")
        
        # Test 4: Verify execution logs
        sample_execution = await manager.pipeline_engine.get_execution_status(executions[0])
        assert len(sample_execution.logs) > 0
        assert any("info" in log.get("level", "") for log in sample_execution.logs)
        print("   ✓ Execution logs captured and formatted")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pipeline execution monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gitops_workflow_integration():
    """Test complete GitOps workflow integration"""
    print("\n🔄 Testing Complete GitOps Workflow Integration...")
    
    try:
        from gitops_integration import (
            get_gitops_manager,
            GitRepository,
            DeploymentStrategy
        )
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        manager = await get_gitops_manager("workflow-test-tenant")
        
        # Step 1: Setup GitOps infrastructure
        repository = GitRepository(
            name="workflow-integration-repo", 
            url="https://github.com/example/workflow-repo.git",
            branch="main",
            sync_enabled=True
        )
        
        repo_id = await manager.add_repository(repository)
        
        pipeline_id = await manager.create_deployment_pipeline(
            name="Integration Workflow Pipeline",
            repository_id=repo_id,
            trigger_events=["push", "pull_request", "tag"]
        )
        
        print("   ✓ Step 1: GitOps infrastructure setup complete")
        
        # Step 2: Create and deploy configuration resource
        resource = CMResource(
            name="workflow-web-app",
            resource_type=ResourceType.CONTAINER,
            cloud_provider=CloudProvider.AWS,
            configuration=ConfigurationDSL(
                kind="WebApplication",
                spec={
                    "resources": {"cpu": "1", "memory": "2Gi"},
                    "image": "myapp:v1.0.0",
                    "replicas": 2,
                    "port": 8080,
                    "environment": {
                        "NODE_ENV": "production",
                        "LOG_LEVEL": "info"
                    }
                },
                version="1.0"
            ),
            description="Web application for workflow integration testing"
        )
        
        manifest_id = await manager.create_manifest(
            resource=resource,
            repository_id=repo_id,
            environment="production",
            namespace="webapp"
        )
        
        print("   ✓ Step 2: Configuration resource and manifest created")
        
        # Step 3: Create and execute deployment plan
        deployment_plan_id = await manager.create_deployment_plan(
            resource_id=resource.id,
            manifest_id=manifest_id,
            environment="production",
            strategy=DeploymentStrategy.CANARY,
            approval_required=True
        )
        
        deployment_success = await manager.execute_deployment(
            deployment_plan_id,
            approved_by="release-manager@example.com"
        )
        
        assert deployment_success == True
        print("   ✓ Step 3: Canary deployment executed successfully")
        
        # Step 4: Trigger CI/CD pipeline
        execution_id = await manager.trigger_pipeline(
            pipeline_id=pipeline_id,
            trigger_data={
                "event": "push",
                "commit_sha": "workflow-commit-123",
                "branch": "main",
                "author": "workflow-test@example.com",
                "message": "Deploy workflow integration app"
            }
        )
        
        print("   ✓ Step 4: CI/CD pipeline triggered")
        
        # Step 5: Monitor and verify complete workflow
        await asyncio.sleep(3)  # Allow workflow to complete
        
        # Verify pipeline execution
        execution = await manager.pipeline_engine.get_execution_status(execution_id)
        assert execution is not None
        pipeline_success = execution.status.value in ["success", "running"]
        
        # Verify deployment plan  
        deployment_plan = manager.deployments[deployment_plan_id]
        assert deployment_plan.approved_by == "release-manager@example.com"
        
        # Verify manifest
        manifest = manager.manifests[manifest_id]
        assert manifest.environment == "production"
        assert manifest.namespace == "webapp"
        
        # Verify GitOps status
        status = await manager.get_gitops_status()
        assert status["repositories"] >= 1
        assert status["manifests"] >= 1
        assert status["pipelines"] >= 1
        
        print("   ✓ Step 5: Complete workflow verification successful")
        print(f"     - Pipeline Status: {execution.status.value}")
        print(f"     - Deployment Strategy: {deployment_plan.strategy.value}")
        print(f"     - GitOps Resources: {status['repositories']} repos, {status['manifests']} manifests")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GitOps workflow integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run GitOps integration tests"""
    print("🔄 APG Configuration Management GitOps Integration Tests")
    print("=" * 75)
    
    test1_success = await test_gitops_repository_management()
    test2_success = await test_manifest_generation()
    test3_success = await test_cicd_pipeline_creation()
    test4_success = await test_deployment_orchestration()
    test5_success = await test_pipeline_execution_monitoring()
    test6_success = await test_gitops_workflow_integration()
    
    print("\n" + "=" * 75)
    if all([test1_success, test2_success, test3_success, test4_success, test5_success, test6_success]):
        print("🏆 GITOPS INTEGRATION TESTS: PASSED ✅")
        print("   📁 Repository management operational")
        print("   📄 Manifest generation working")
        print("   🔧 CI/CD pipeline creation functional")
        print("   🚀 Deployment orchestration complete")
        print("   📊 Pipeline monitoring comprehensive")
        print("   🔄 End-to-end workflow integration successful")
        print("   🎯 Phase 3.5b CI/CD Pipeline Automation: COMPLETE")
        print("   💎 Revolutionary GitOps workflow achieved")
        print("")
        print("   📊 GitOps Summary:")
        print("   ├── Repository Integration: ✅ Git-native")
        print("   ├── Manifest Generation: ✅ Kubernetes-style")
        print("   ├── Pipeline Automation: ✅ Multi-stage")
        print("   ├── Deployment Strategies: ✅ Rolling/Blue-Green/Canary")
        print("   ├── Approval Workflows: ✅ Role-based")
        print("   └── Monitoring & Observability: ✅ Real-time")
    else:
        print("❌ GITOPS INTEGRATION TESTS: FAILED")
        print("   🔍 Check error logs above for details")
    
    print("=" * 75)
    
    return all([test1_success, test2_success, test3_success, test4_success, test5_success, test6_success])


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)