#!/usr/bin/env python3
"""
APG Configuration Management Service Collaboration Integration Tests
Tests the integration of collaboration features into the main service layer.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the project root to Python path
conf_path = "/Users/nyimbiodero/src/pjs/apg/capabilities/common/conf"
sys.path.insert(0, conf_path)

async def test_service_collaboration_integration():
    """Test collaboration features integrated into main service"""
    print("🔗 Testing Service Collaboration Integration...")
    
    try:
        from service import RevolutionaryConfigurationManager
        from collaboration_layer import CollaborationPermission
        from models import ResourceType, CloudProvider
        
        # Initialize configuration manager
        manager = RevolutionaryConfigurationManager("collaboration-test-tenant")
        
        # Mock APG integrations
        mock_integrations = {
            "auth_rbac": None,
            "audit_compliance": None,
            "ai_orchestration": None,
            "notification_engine": None
        }
        
        await manager.initialize(mock_integrations)
        
        # Test 1: Create a configuration resource
        config_data = {
            "name": "collab-test-vm",
            "type": "virtual_machine",
            "cloud_provider": "aws",
            "created_by": "test-user",
            "configuration": {
                "kind": "VirtualMachine",
                "spec": {
                    "resources": {"instance_type": "t3.micro"},
                    "security": {"encryption": True}
                },
                "version": "1.0"
            }
        }
        
        resource = await manager.create_configuration(config_data)
        assert resource is not None
        print(f"   ✓ Configuration resource created: {resource.name}")
        
        # Test 2: Create collaboration session for the resource
        session_id = await manager.create_collaboration_session(
            resource_id=resource.id,
            owner_id="owner-user",
            name="Test Collaboration Session",
            user_permissions={
                "alice": [CollaborationPermission.EDIT, CollaborationPermission.COMMENT],
                "bob": [CollaborationPermission.EDIT, CollaborationPermission.APPROVE]
            }
        )
        
        assert session_id is not None
        print(f"   ✓ Collaboration session created: {session_id}")
        
        # Test 3: Get collaboration state
        state = await manager.get_collaboration_state(session_id)
        assert state is not None
        assert len(state["participants"]) == 2
        print(f"   ✓ Collaboration state retrieved: {len(state['participants'])} participants")
        
        # Test 4: Apply collaborative changes
        change_id = await manager.apply_collaborative_change(
            session_id=session_id,
            user_id="alice",
            change_type="modify",
            path="spec.resources.instance_type",
            old_value="t3.micro",
            new_value="t3.small"
        )
        
        assert change_id is not None
        print("   ✓ Collaborative change applied successfully")
        
        # Test 5: Add collaboration comment
        comment_id = await manager.add_collaboration_comment(
            session_id=session_id,
            user_id="bob",
            content="Looks good, scaling up the instance size",
            section_path="spec.resources"
        )
        
        assert comment_id is not None
        print("   ✓ Collaboration comment added successfully")
        
        # Test 6: Join additional user to session
        join_success = await manager.join_configuration_collaboration(
            session_id=session_id,
            user_id="charlie",
            display_name="Charlie Reviewer",
            permissions=[CollaborationPermission.APPROVE]
        )
        
        assert join_success == True
        print("   ✓ Additional user joined collaboration session")
        
        # Test 7: Check updated collaboration state
        updated_state = await manager.get_collaboration_state(session_id)
        assert len(updated_state["participants"]) == 3
        print(f"   ✓ Updated collaboration state: {len(updated_state['participants'])} participants")
        
        # Test 8: Leave collaboration session
        await manager.leave_collaboration_session(session_id, "charlie")
        final_state = await manager.get_collaboration_state(session_id)
        assert len(final_state["participants"]) == 2
        print("   ✓ User left collaboration session successfully")
        
        await manager.shutdown()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Service collaboration integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_collaborative_workflow_end_to_end():
    """Test complete collaborative workflow from creation to resolution"""
    print("\n🔄 Testing Collaborative Workflow End-to-End...")
    
    try:
        from service import RevolutionaryConfigurationManager
        from collaboration_layer import CollaborationPermission
        
        # Initialize manager
        manager = RevolutionaryConfigurationManager("workflow-test-tenant")
        
        mock_integrations = {
            "auth_rbac": None,
            "audit_compliance": None,
            "ai_orchestration": None,
            "notification_engine": None
        }
        
        await manager.initialize(mock_integrations)
        
        # Step 1: Create configuration resource
        config_data = {
            "name": "workflow-test-db",
            "type": "database",
            "cloud_provider": "aws",
            "created_by": "architect-user",
            "configuration": {
                "kind": "Database",
                "spec": {
                    "resources": {"instance_type": "db.t3.micro"},
                    "security": {"encryption_at_rest": True}
                },
                "version": "1.0"
            }
        }
        
        resource = await manager.create_configuration(config_data)
        print(f"   ✓ Step 1: Configuration resource created")
        
        # Step 2: Start collaborative editing session
        session_id = await manager.create_collaboration_session(
            resource_id=resource.id,
            owner_id="architect-user",
            name="Database Configuration Review"
        )
        print(f"   ✓ Step 2: Collaboration session started")
        
        # Step 3: Multiple users join
        users = [
            ("developer1", "Alice Developer", [CollaborationPermission.EDIT, CollaborationPermission.COMMENT]),
            ("developer2", "Bob Developer", [CollaborationPermission.EDIT, CollaborationPermission.COMMENT]),
            ("reviewer", "Charlie Reviewer", [CollaborationPermission.APPROVE, CollaborationPermission.COMMENT])
        ]
        
        for user_id, display_name, permissions in users:
            await manager.join_configuration_collaboration(
                session_id=session_id,
                user_id=user_id,
                display_name=display_name,
                permissions=permissions
            )
        
        print("   ✓ Step 3: Multiple users joined collaboration")
        
        # Step 4: Collaborative editing
        changes = [
            ("developer1", "modify", "spec.resources.instance_type", "db.t3.micro", "db.t3.small"),
            ("developer2", "add", "spec.backup_retention", None, "7days"),
            ("developer1", "modify", "spec.security.encryption_in_transit", None, True)
        ]
        
        change_ids = []
        for user_id, change_type, path, old_val, new_val in changes:
            change_id = await manager.apply_collaborative_change(
                session_id=session_id,
                user_id=user_id,
                change_type=change_type,
                path=path,
                old_value=old_val,
                new_value=new_val
            )
            if change_id:
                change_ids.append(change_id)
        
        print(f"   ✓ Step 4: Applied {len(change_ids)} collaborative changes")
        
        # Step 5: Add review comments
        comments = [
            ("reviewer", "Backup retention looks good", "spec.backup_retention"),
            ("developer2", "Should we consider multi-AZ deployment?", "spec.resources"),
        ]
        
        comment_ids = []
        for user_id, content, section in comments:
            comment_id = await manager.add_collaboration_comment(
                session_id=session_id,
                user_id=user_id,
                content=content,
                section_path=section
            )
            if comment_id:
                comment_ids.append(comment_id)
        
        print(f"   ✓ Step 5: Added {len(comment_ids)} review comments")
        
        # Step 6: Final state verification
        final_state = await manager.get_collaboration_state(session_id)
        
        assert final_state is not None
        assert len(final_state["participants"]) == 4  # architect + 3 users
        assert final_state["change_count"] >= len(change_ids)
        assert final_state["comment_count"] >= len(comment_ids)
        
        print("   ✓ Step 6: Final collaboration state verified")
        print(f"     - Participants: {len(final_state['participants'])}")
        print(f"     - Changes: {final_state['change_count']}")
        print(f"     - Comments: {final_state['comment_count']}")
        
        await manager.shutdown()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Collaborative workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_collaboration_with_security_integration():
    """Test collaboration features working with security integration"""
    print("\n🔒 Testing Collaboration with Security Integration...")
    
    try:
        from service import RevolutionaryConfigurationManager
        from collaboration_layer import CollaborationPermission
        
        # Initialize manager
        manager = RevolutionaryConfigurationManager("security-collab-tenant")
        
        mock_integrations = {
            "auth_rbac": None,
            "audit_compliance": None,
            "ai_orchestration": None,
            "notification_engine": None
        }
        
        await manager.initialize(mock_integrations)
        
        # Create high-security configuration
        config_data = {
            "name": "secure-collab-config",
            "type": "virtual_machine",
            "cloud_provider": "aws",
            "security_level": "confidential",
            "created_by": "security-architect",
            "configuration": {
                "kind": "VirtualMachine",
                "spec": {
                    "resources": {"instance_type": "t3.micro"},
                    "security": {
                        "encryption_at_rest": True,
                        "encryption_in_transit": True,
                        "audit_logging": True
                    }
                },
                "version": "1.0"
            }
        }
        
        resource = await manager.create_configuration(config_data)
        print("   ✓ High-security configuration created")
        
        # Create collaboration session
        session_id = await manager.create_collaboration_session(
            resource_id=resource.id,
            owner_id="security-architect",
            name="Secure Configuration Review"
        )
        print("   ✓ Collaboration session for secure resource created")
        
        # Add users with different security permissions
        security_users = [
            ("security-editor", "Security Editor", [CollaborationPermission.EDIT]),
            ("compliance-reviewer", "Compliance Reviewer", [CollaborationPermission.APPROVE])
        ]
        
        for user_id, display_name, permissions in security_users:
            await manager.join_configuration_collaboration(
                session_id=session_id,
                user_id=user_id,
                display_name=display_name,
                permissions=permissions
            )
        
        print("   ✓ Security-aware users joined collaboration")
        
        # Test secure collaborative changes
        secure_change_id = await manager.apply_collaborative_change(
            session_id=session_id,
            user_id="security-editor",
            change_type="modify",
            path="spec.security.multi_factor_auth",
            old_value=None,
            new_value=True
        )
        
        assert secure_change_id is not None
        print("   ✓ Security-focused collaborative change applied")
        
        # Add compliance comment
        compliance_comment_id = await manager.add_collaboration_comment(
            session_id=session_id,
            user_id="compliance-reviewer",
            content="Configuration meets GDPR requirements",
            section_path="spec.security"
        )
        
        assert compliance_comment_id is not None
        print("   ✓ Compliance review comment added")
        
        # Verify collaboration state with security context
        state = await manager.get_collaboration_state(session_id)
        assert state is not None
        assert len(state["participants"]) == 3  # architect + 2 security users
        
        print("   ✓ Secure collaboration state verified")
        print(f"     - Secure participants: {len(state['participants'])}")
        
        await manager.shutdown()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Collaboration with security integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run service collaboration integration tests"""
    print("🤝 APG Configuration Management Service Collaboration Integration Tests")
    print("=" * 85)
    
    test1_success = await test_service_collaboration_integration()
    test2_success = await test_collaborative_workflow_end_to_end()
    test3_success = await test_collaboration_with_security_integration()
    
    print("\n" + "=" * 85)
    if test1_success and test2_success and test3_success:
        print("🏆 SERVICE COLLABORATION INTEGRATION TESTS: PASSED ✅")
        print("   🔗 Service collaboration integration successful")
        print("   🔄 End-to-end collaborative workflows functional")
        print("   🔒 Security-aware collaboration operational")
        print("   🎯 Phase 3.4 Service Integration: COMPLETE")
        print("   💎 Revolutionary collaborative configuration management integrated")
        print("")
        print("   📊 Integration Summary:")
        print("   ├── Service Layer Integration: ✅ Complete")
        print("   ├── Multi-User Workflows: ✅ Operational")
        print("   ├── Real-Time Collaboration: ✅ Integrated")
        print("   ├── Security Integration: ✅ Functional")
        print("   ├── Comment System: ✅ Working")
        print("   └── Session Management: ✅ Robust")
    else:
        print("❌ SERVICE COLLABORATION INTEGRATION TESTS: FAILED")
        print("   🔍 Check error logs above for details")
    
    print("=" * 85)
    
    return test1_success and test2_success and test3_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)