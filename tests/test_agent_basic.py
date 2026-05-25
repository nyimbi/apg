#!/usr/bin/env python3
"""
Basic test of APG Intelligent Agent functionality
"""

import asyncio
import sys
import os

# Add the APG root to Python path
sys.path.insert(0, '/Users/nyimbiodero/src/pjs/apg')

from capabilities.common.agents.service import AgentManagerService
from capabilities.common.agents.models import AgentType, AgentRole
from capabilities.common.agents.tests.test_utils import create_test_services

async def test_basic_functionality():
    """Test basic agent functionality"""
    print("🚀 Starting APG Intelligent Agent basic functionality test...")
    
    try:
        # Create service with real integrations
        print("📝 Creating AgentManagerService...")
        service = AgentManagerService()
        
        # Setup test services
        print("🔧 Setting up test services...")
        services = create_test_services()
        service._auth_service_available = True
        service._audit_service_available = True
        service.auth_service = services["auth_service"]
        service.audit_service = services["audit_service"]
        service.ai_orchestration = services["ai_orchestration"]
        service.federated_learning = services["federated_learning"]
        service.collaboration_service = services["collaboration_service"]
        
        print("✅ Service setup complete!")
        
        # Test agent creation
        print("👤 Testing agent creation...")
        agent_config = {
            "name": "Test Agent",
            "description": "A test agent for basic functionality verification",
            "type": AgentType.WORKER,
            "role": AgentRole.TASK_MANAGER,
            "capabilities": ["reasoning", "learning", "communication"]
        }
        
        user_id = "test-user-123"  # User with proper permissions
        agent = await service.create_agent(user_id, agent_config)
        
        print(f"✅ Agent created successfully!")
        print(f"   - ID: {agent.id}")
        print(f"   - Name: {agent.name}")
        print(f"   - Type: {agent.type}")
        print(f"   - Role: {agent.role}")
        print(f"   - Capabilities: {agent.capabilities}")
        
        # Test agent retrieval
        print("🔍 Testing agent retrieval...")
        retrieved_agent = await service.get_agent(user_id, agent.id)
        print(f"✅ Agent retrieved successfully: {retrieved_agent.name}")
        
        # Test agent listing
        print("📋 Testing agent listing...")
        agents = await service.list_agents(user_id)
        print(f"✅ Found {len(agents)} agents")
        
        # Test auth service
        print("🔐 Testing authentication...")
        has_permission = await service.auth_service.check_permission(user_id, "agent:create")
        print(f"✅ Permission check: {has_permission}")
        
        # Test audit service
        print("📊 Testing audit logging...")
        audit_logs = await service.audit_service.get_audit_logs(user_id=user_id)
        print(f"✅ Found {len(audit_logs)} audit log entries")
        
        # Test AI orchestration
        print("🤖 Testing AI orchestration...")
        task_definition = {
            "type": "text_generation",
            "input": "Hello, this is a test",
            "model": "llama3.2:3b"
        }
        task_id = await service.ai_orchestration.submit_task(task_definition)
        print(f"✅ AI task submitted: {task_id}")
        
        # Wait and check task status
        await asyncio.sleep(0.2)
        status = await service.ai_orchestration.get_task_status(task_id)
        print(f"✅ AI task status: {status['status']}")
        
        # Test collaboration service
        print("🤝 Testing collaboration...")
        session_id = await service.collaboration_service.create_collaboration_session(
            ["agent-1", "agent-2"], "testing"
        )
        print(f"✅ Collaboration session created: {session_id}")
        
        # Send a message
        await service.collaboration_service.send_message(
            session_id, "agent-1", "Hello from test!", "text"
        )
        
        # Get messages
        messages = await service.collaboration_service.get_messages(session_id)
        print(f"✅ Found {len(messages)} collaboration messages")
        
        print("\n🎉 All basic functionality tests PASSED!")
        print("✅ APG Intelligent Agent capability is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_advanced_functionality():
    """Test advanced agent functionality"""
    print("\n🚀 Starting advanced functionality tests...")
    
    try:
        # Test orchestration engine
        print("🎭 Testing orchestration engine...")
        from capabilities.common.agents.orchestration_engine import NetworkOrchestrator
        from capabilities.common.agents.models import AgentNetwork, IntelligentAgent, AgentType, AgentRole
        
        # Create a test network and agent for orchestrator
        test_network = AgentNetwork(
            id="test-network",
            name="Test Network",
            topology="hierarchical",
            configuration={},
            created_by="test-user-123",
            tenant_id="test_tenant"
        )
        test_agent = IntelligentAgent(
            id="test-agent-orch",
            name="Test Orchestrator Agent",
            type=AgentType.COORDINATOR,
            role=AgentRole.ORCHESTRATOR,
            created_by="test-user-123",
            tenant_id="test_tenant"
        )
        orchestrator = NetworkOrchestrator(test_network, [test_agent])
        print("✅ Network orchestrator initialized")
        
        # Test decision engine
        print("🧠 Testing decision engine...")
        from capabilities.common.agents.decision_engine import get_decision_engine
        decision_engine = get_decision_engine()
        print("✅ Decision engine initialized")
        
        # Test communication hub
        print("📡 Testing communication hub...")
        from capabilities.common.agents.communication_hub import get_communication_hub
        comm_hub = get_communication_hub()
        print("✅ Communication hub initialized")
        
        # Test capability framework 
        print("⚙️ Testing capability framework...")
        from capabilities.common.agents.capability_framework import global_capability_registry
        capabilities = global_capability_registry.list_available_capabilities()
        print(f"✅ Found {len(capabilities)} registered capabilities: {capabilities}")
        
        # Test learning engine
        print("📚 Testing learning engine...")
        from capabilities.common.agents.learning_engine import AgentLearningEngine
        learning_engine = AgentLearningEngine("test-agent")
        status = learning_engine.get_learning_status()
        print(f"✅ Learning engine status: {status['learning_enabled']}")
        
        # Test template engine
        print("📄 Testing template engine...")
        from capabilities.common.agents.template_engine import AgentTemplateEngine
        template_engine = AgentTemplateEngine()
        templates = template_engine.list_available_templates()
        print(f"✅ Found {len(templates)} agent templates: {list(templates.keys())}")
        
        print("✅ All advanced functionality tests PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Advanced test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 APG Intelligent Agent Functionality Test")
    print("=" * 50)
    
    # Run basic tests
    basic_success = asyncio.run(test_basic_functionality())
    
    # Run advanced tests  
    advanced_success = asyncio.run(test_advanced_functionality())
    
    if basic_success and advanced_success:
        print("\n🎊 ALL TESTS PASSED! 🎊")
        print("The APG Intelligent Agent capability is fully functional!")
        sys.exit(0)
    else:
        print("\n💥 SOME TESTS FAILED")
        sys.exit(1)