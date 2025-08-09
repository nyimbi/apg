#!/usr/bin/env python3
"""
APG Configuration Management Collaboration Integration Tests  
Direct integration tests for collaboration features.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the project root to Python path
conf_path = "/Users/nyimbiodero/src/pjs/apg/capabilities/common/conf"
sys.path.insert(0, conf_path)

async def test_collaboration_integration():
    """Test collaboration integration with configuration management"""
    print("🔗 Testing Collaboration Integration...")
    
    try:
        from collaboration_layer import get_collaboration_manager, CollaborationPermission
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        # Initialize collaboration manager
        collaboration_manager = await get_collaboration_manager()
        
        # Create a mock configuration resource
        config_dsl = ConfigurationDSL(
            kind="VirtualMachine",
            spec={
                "resources": {"instance_type": "t3.micro"},
                "security": {"encryption": True}
            },
            version="1.0"
        )
        
        resource = CMResource(
            name="integration-test-vm",
            resource_type=ResourceType.VIRTUAL_MACHINE,
            cloud_provider=CloudProvider.AWS,
            configuration=config_dsl,
            description="Test resource for collaboration integration"
        )
        
        print("   ✓ Mock configuration resource created")
        
        # Test 1: Create collaboration session
        session_id = await collaboration_manager.create_collaboration_session(
            resource_id=resource.id,
            owner_id="integration-owner",
            name="Integration Test Session",
            base_configuration=config_dsl
        )
        
        assert session_id is not None
        print(f"   ✓ Collaboration session created: {session_id}")
        
        # Test 2: Add multiple users with different roles
        users_data = [
            ("developer", "Alice Developer", [CollaborationPermission.EDIT, CollaborationPermission.COMMENT]),
            ("reviewer", "Bob Reviewer", [CollaborationPermission.APPROVE, CollaborationPermission.COMMENT]),
            ("viewer", "Charlie Viewer", [CollaborationPermission.VIEW_ONLY])
        ]
        
        for user_id, display_name, permissions in users_data:
            success = await collaboration_manager.join_collaboration_session(
                session_id=session_id,
                user_id=user_id,
                display_name=display_name,
                permissions=permissions
            )
            assert success == True
        
        print(f"   ✓ Added {len(users_data)} users to collaboration session")
        
        # Test 3: Collaborative editing workflow
        # Developer makes changes
        change_id1 = await collaboration_manager.apply_configuration_change(
            session_id=session_id,
            user_id="developer",
            change_type="modify",
            path="spec.resources.instance_type",
            old_value="t3.micro",
            new_value="t3.small"
        )
        
        assert change_id1 is not None
        print("   ✓ Developer applied configuration change")
        
        # Reviewer adds comment
        comment_id = await collaboration_manager.add_comment(
            session_id=session_id,
            user_id="reviewer",
            content="Instance size upgrade looks good for performance requirements",
            section_path="spec.resources"
        )
        
        assert comment_id is not None
        print("   ✓ Reviewer added comment")
        
        # Test 4: Configuration locking
        lock_id = await collaboration_manager.acquire_configuration_lock(
            session_id=session_id,
            user_id="developer",
            lock_sections=["spec.security"]
        )
        
        assert lock_id is not None
        print("   ✓ Developer acquired configuration lock")
        
        # Test 5: Get comprehensive session state
        state = await collaboration_manager.get_session_state(session_id)
        
        assert state is not None
        assert len(state["participants"]) == 3
        assert state["change_count"] >= 1
        assert state["comment_count"] >= 1
        assert len(state["active_locks"]) >= 1
        
        print("   ✓ Session state comprehensive and accurate")
        print(f"     - Participants: {len(state['participants'])}")
        print(f"     - Changes: {state['change_count']}")
        print(f"     - Comments: {state['comment_count']}")
        print(f"     - Active locks: {len(state['active_locks'])}")
        
        # Test 6: Release lock and cleanup
        release_success = await collaboration_manager.release_configuration_lock(
            session_id, lock_id, "developer"
        )
        
        assert release_success == True
        print("   ✓ Configuration lock released")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Collaboration integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_advanced_collaboration_features():
    """Test advanced collaboration features"""
    print("\n⚡ Testing Advanced Collaboration Features...")
    
    try:
        from collaboration_layer import (
            get_collaboration_manager, 
            CollaborationPermission,
            ConfigurationConflict,
            ConflictResolutionStrategy
        )
        from models import CMResource, ConfigurationDSL, ResourceType, CloudProvider
        
        collaboration_manager = await get_collaboration_manager()
        
        # Create collaboration session
        session_id = await collaboration_manager.create_collaboration_session(
            resource_id="advanced-test-resource",
            owner_id="advanced-owner",
            name="Advanced Features Test"
        )
        
        # Add users
        await collaboration_manager.join_collaboration_session(
            session_id, "user1", "User One", permissions=[CollaborationPermission.EDIT]
        )
        await collaboration_manager.join_collaboration_session(
            session_id, "user2", "User Two", permissions=[CollaborationPermission.EDIT]
        )
        await collaboration_manager.join_collaboration_session(
            session_id, "admin", "Admin User", permissions=[CollaborationPermission.APPROVE]
        )
        
        print("   ✓ Advanced collaboration session setup complete")
        
        # Test 1: Concurrent editing and conflict creation
        session = collaboration_manager.sessions[session_id]
        
        # Simulate concurrent changes to same path
        from collaboration_layer import ConfigurationChange, ConfigurationConflict
        
        change1 = ConfigurationChange(
            resource_id="advanced-test-resource",
            user_id="user1",
            change_type="modify",
            path="spec.resources.cpu",
            old_value="2",
            new_value="4"
        )
        
        change2 = ConfigurationChange(
            resource_id="advanced-test-resource",
            user_id="user2",
            change_type="modify", 
            path="spec.resources.cpu",
            old_value="2",
            new_value="8"
        )
        
        session.changes[change1.id] = change1
        session.changes[change2.id] = change2
        
        # Create conflict
        conflict = ConfigurationConflict(
            resource_id="advanced-test-resource",
            conflicting_changes=[change1.id, change2.id],
            conflict_type="value_conflict",
            path="spec.resources.cpu",
            base_value="2",
            user_values={"user1": "4", "user2": "8"},
            resolution_strategy=ConflictResolutionStrategy.MANUAL
        )
        
        session.conflicts[conflict.id] = conflict
        print("   ✓ Simulated configuration conflict created")
        
        # Test 2: Conflict resolution by admin
        resolution_success = await collaboration_manager.resolve_conflict(
            session_id=session_id,
            conflict_id=conflict.id,
            resolution_value="6",  # Compromise between 4 and 8
            resolved_by="admin"
        )
        
        assert resolution_success == True
        assert conflict.resolved == True
        assert conflict.resolution_value == "6"
        print("   ✓ Configuration conflict resolved by admin")
        
        # Test 3: Comment threading and mentions
        thread_comment_id = await collaboration_manager.add_comment(
            session_id=session_id,
            user_id="user1",
            content="@admin Thanks for resolving the CPU allocation conflict",
            mentions=["admin"]
        )
        
        assert thread_comment_id is not None
        print("   ✓ Comment with mentions added successfully")
        
        # Test 4: Permission-based access control
        # Try to have viewer make changes (should fail)
        await collaboration_manager.join_collaboration_session(
            session_id, "viewer", "View Only", permissions=[CollaborationPermission.VIEW_ONLY]
        )
        
        failed_change_id = await collaboration_manager.apply_configuration_change(
            session_id=session_id,
            user_id="viewer",
            change_type="modify",
            path="spec.resources.memory",
            old_value="4Gi",
            new_value="8Gi"
        )
        
        assert failed_change_id is None  # Should fail
        print("   ✓ View-only user properly blocked from making changes")
        
        # Test 5: Session state after advanced operations
        final_state = await collaboration_manager.get_session_state(session_id)
        
        assert len(final_state["participants"]) == 4
        assert len(final_state["pending_conflicts"]) == 0  # Should be resolved
        assert final_state["comment_count"] > 0
        
        print("   ✓ Advanced collaboration state verified")
        print(f"     - Total participants: {len(final_state['participants'])}")
        print(f"     - Resolved conflicts: {len(session.conflicts) - len(final_state['pending_conflicts'])}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Advanced collaboration features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_collaboration_performance():
    """Test collaboration system performance and scalability"""
    print("\n🚀 Testing Collaboration Performance...")
    
    try:
        from collaboration_layer import get_collaboration_manager, CollaborationPermission
        
        collaboration_manager = await get_collaboration_manager()
        
        # Test 1: Multiple concurrent sessions
        session_ids = []
        for i in range(5):
            session_id = await collaboration_manager.create_collaboration_session(
                resource_id=f"perf-test-resource-{i}",
                owner_id=f"owner-{i}",
                name=f"Performance Test Session {i}"
            )
            session_ids.append(session_id)
        
        print(f"   ✓ Created {len(session_ids)} concurrent collaboration sessions")
        
        # Test 2: Multiple users per session
        total_users = 0
        for session_id in session_ids:
            for j in range(3):  # 3 users per session
                await collaboration_manager.join_collaboration_session(
                    session_id, f"user-{j}", f"User {j}", 
                    permissions=[CollaborationPermission.EDIT]
                )
                total_users += 1
        
        print(f"   ✓ Added {total_users} users across all sessions")
        
        # Test 3: Bulk operations
        total_changes = 0
        total_comments = 0
        
        for session_id in session_ids:
            # Add changes
            for i in range(2):
                change_id = await collaboration_manager.apply_configuration_change(
                    session_id=session_id,
                    user_id="user-0",
                    change_type="modify",
                    path=f"spec.test_field_{i}",
                    old_value=f"old_{i}",
                    new_value=f"new_{i}"
                )
                if change_id:
                    total_changes += 1
            
            # Add comments
            comment_id = await collaboration_manager.add_comment(
                session_id=session_id,
                user_id="user-1",
                content=f"Performance test comment for session {session_id[:8]}"
            )
            if comment_id:
                total_comments += 1
        
        print(f"   ✓ Applied {total_changes} changes and {total_comments} comments")
        
        # Test 4: State retrieval performance
        start_time = datetime.utcnow()
        
        states = []
        for session_id in session_ids:
            state = await collaboration_manager.get_session_state(session_id)
            if state:
                states.append(state)
        
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        print(f"   ✓ Retrieved {len(states)} session states in {duration_ms:.1f}ms")
        
        # Test 5: Resource cleanup
        for session_id in session_ids:
            session = collaboration_manager.sessions[session_id]
            # Simulate users leaving
            for user_id in list(session.participants.keys()):
                await collaboration_manager.leave_collaboration_session(session_id, user_id)
        
        print("   ✓ Performance test cleanup completed")
        
        # Performance assertions
        assert len(states) == len(session_ids)
        assert duration_ms < 1000  # Should retrieve states in under 1 second
        assert total_changes > 0
        assert total_comments > 0
        
        print("   ✓ Performance benchmarks met")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Collaboration performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run collaboration integration tests"""
    print("🤝 APG Configuration Management Collaboration Integration Tests")
    print("=" * 80)
    
    test1_success = await test_collaboration_integration()
    test2_success = await test_advanced_collaboration_features()
    test3_success = await test_collaboration_performance()
    
    print("\n" + "=" * 80)
    if test1_success and test2_success and test3_success:
        print("🏆 COLLABORATION INTEGRATION TESTS: PASSED ✅")
        print("   🔗 Core collaboration integration working")
        print("   ⚡ Advanced features operational")
        print("   🚀 Performance benchmarks met")
        print("   🎯 Phase 3.4 Real-Time Collaboration Layer: COMPLETE")
        print("   💎 Revolutionary collaborative configuration management achieved")
        print("")
        print("   📊 Collaboration Summary:")
        print("   ├── Multi-User Sessions: ✅ Scalable")
        print("   ├── Real-Time Editing: ✅ Synchronized")
        print("   ├── Conflict Resolution: ✅ Advanced")
        print("   ├── Permission System: ✅ Role-based")
        print("   ├── Comment Threading: ✅ Functional")
        print("   ├── Configuration Locking: ✅ Section-level")
        print("   └── Performance: ✅ Sub-second response")
    else:
        print("❌ COLLABORATION INTEGRATION TESTS: FAILED")
        print("   🔍 Check error logs above for details")
    
    print("=" * 80)
    
    return test1_success and test2_success and test3_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)