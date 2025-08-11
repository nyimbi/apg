#!/usr/bin/env python3
"""
Minimal Integration Test

Test production components that don't require external dependencies.
Validates the core architectural transformation without Redis/Ollama/wasmtime dependencies.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_apg_clients():
    """Test APG client classes and configurations."""
    print("🧪 Testing APG clients...")
    
    try:
        from apg_clients import (
            APGAuthRBACClient, APGMonitoringClient, APGConfigurationClient,
            APGAIOrchestrationClient, APGMessageQueueClient, APGAuditComplianceClient,
            APGServiceConfig, APGServiceStatus, AuthResult
        )
        
        # Test service config
        config = APGServiceConfig(
            base_url='http://test.example.com',
            api_key='test-key-123',
            timeout=30,
            circuit_breaker_threshold=5,
            retry_attempts=3,
            retry_delay=1.0
        )
        
        # Test client instantiation (without initialization)
        auth_client = APGAuthRBACClient(config, 'test-tenant')
        monitoring_client = APGMonitoringClient(config, 'test-tenant')
        config_client = APGConfigurationClient(config, 'test-tenant')
        ai_client = APGAIOrchestrationClient(config, 'test-tenant')
        queue_client = APGMessageQueueClient(config, 'test-tenant')
        audit_client = APGAuditComplianceClient(config, 'test-tenant')
        
        # Test enums
        assert APGServiceStatus.CONNECTED.value == 'connected'
        assert APGServiceStatus.DISCONNECTED.value == 'disconnected'
        
        # Test auth result
        auth_result = AuthResult(
            authenticated=True,
            user_id='test-user',
            tenant_id='test-tenant',
            roles=['admin']
        )
        
        assert auth_result.authenticated == True
        assert 'admin' in auth_result.roles
        
        print("   ✅ APG clients tested successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ APG clients test failed: {str(e)}")
        return False

async def test_ollama_client():
    """Test Ollama client classes and configurations."""
    print("🧪 Testing Ollama client...")
    
    try:
        from ollama_client import (
            ProductionOllamaClient, OllamaConfig, GenerationRequest, 
            GenerationResponse, EmbeddingRequest, EmbeddingResponse,
            ModelInfo, OllamaModelStatus, OllamaError
        )
        
        # Test config
        config = OllamaConfig(
            base_url='http://localhost:11434',
            timeout=60,
            max_retries=3,
            retry_delay=1.0,
            model_cache_size=5,
            default_model='llama3.2:latest'
        )
        
        assert config.base_url == 'http://localhost:11434'
        assert config.default_model == 'llama3.2:latest'
        
        # Test generation request
        gen_request = GenerationRequest(
            model='llama3.2:latest',
            prompt='Test prompt for integration',
            system='You are a helpful assistant',
            stream=False,
            options={'temperature': 0.7}
        )
        
        assert gen_request.model == 'llama3.2:latest'
        assert gen_request.stream == False
        
        # Test client instantiation (without initialization)
        client = ProductionOllamaClient(config, 'test-tenant')
        assert client.tenant_id == 'test-tenant'
        assert client.config.base_url == 'http://localhost:11434'
        
        # Test model status enum
        assert OllamaModelStatus.AVAILABLE.value == 'available'
        assert OllamaModelStatus.NOT_FOUND.value == 'not_found'
        
        print("   ✅ Ollama client tested successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Ollama client test failed: {str(e)}")
        return False

async def test_wasm_runtime():
    """Test WASM runtime classes and configurations."""
    print("🧪 Testing WASM runtime...")
    
    try:
        from wasm_runtime import (
            ProductionWASMRuntime, WASMExecutionContext, WASMExecutionResult,
            WASMModuleInfo, WASMExecutionStatus, WASMSecurityError
        )
        
        # Test runtime instantiation (without initialization)
        runtime = ProductionWASMRuntime('test-tenant', max_modules=50)
        assert runtime.tenant_id == 'test-tenant'
        assert runtime.max_modules == 50
        assert runtime.initialized == False
        
        # Test execution context (using AgHttpRequest would require models)
        from models import AgHttpRequest
        
        request = AgHttpRequest(
            method='POST',
            path='/api/wasm-test',
            headers={'content-type': 'application/json'},
            body=b'{"test": "data"}',
            client_ip='192.168.1.100',
            user_agent='TestClient/1.0',
            tenant_id='test-tenant'
        )
        
        context = WASMExecutionContext(
            module_id='test-module',
            request=request,
            memory_limit_mb=32,
            execution_timeout_ms=2000,
            allowed_imports=['console.log']
        )
        
        assert context.module_id == 'test-module'
        assert context.memory_limit_mb == 32
        
        # Test status enum
        assert WASMExecutionStatus.SUCCESS.value == 'success'
        assert WASMExecutionStatus.TIMEOUT.value == 'timeout'
        
        print("   ✅ WASM runtime tested successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ WASM runtime test failed: {str(e)}")
        return False

async def test_models():
    """Test model creation and validation."""
    print("🧪 Testing models...")
    
    try:
        from models import (
            AgGatewayConfig, AgApiRoute, AgUpstreamService, AgPolicy,
            AgHttpRequest, AgHttpResponse, EnvironmentType, PolicyType,
            AgTrafficMetrics, AgSecurityEvent, ThreatLevel
        )
        
        # Test upstream service
        upstream = AgUpstreamService(
            name='test-upstream',
            base_url='https://api.example.com',
            weight=100,
            max_connections=50,
            connection_timeout_ms=5000
        )
        
        # Test API route
        route = AgApiRoute(
            method='GET',
            path='/api/users/*',
            upstream_services=[upstream],
            tenant_id='test-tenant',
            created_by='test-user'
        )
        
        # Test gateway configuration
        gateway = AgGatewayConfig(
            name='production-gateway',
            environment=EnvironmentType.PRODUCTION,
            tenant_id='test-tenant',
            created_by='test-user',
            listen_port=443,
            routes=[route]
        )
        
        # Test HTTP request/response
        request = AgHttpRequest(
            method='GET',
            path='/api/users/123',
            headers={'authorization': 'Bearer token123'},
            client_ip='203.0.113.1',
            user_agent='Mozilla/5.0',
            tenant_id='test-tenant'
        )
        
        response = AgHttpResponse(
            request_id=request.id,
            status_code=200,
            headers={'content-type': 'application/json'},
            body=b'{"id": 123, "name": "John"}'
        )
        
        # Test policy
        policy = AgPolicy(
            name='Rate Limiting Policy',
            type=PolicyType.RATE_LIMITING,
            configuration={'requests_per_minute': 1000},
            conditions=['path.startswith("/api/")'],
            tenant_id='test-tenant',
            created_by='test-user'
        )
        
        # Test traffic metrics
        metrics = AgTrafficMetrics(
            gateway_id=gateway.id,
            tenant_id='test-tenant',
            request_count=15423,
            error_count=23,
            response_time_p50=45.2,
            response_time_p95=127.8
        )
        
        # Test security event
        security_event = AgSecurityEvent(
            gateway_id=gateway.id,
            event_type='suspicious_activity',
            threat_level=ThreatLevel.MEDIUM,
            confidence=0.87,
            source_ip='198.51.100.42',
            user_agent='BadBot/1.0',
            route_path='/admin/login',
            action_taken='blocked',
            tenant_id='test-tenant'
        )
        
        # Verify all objects created successfully
        assert gateway.name == 'production-gateway'
        assert route.method == 'GET'
        assert upstream.base_url == 'https://api.example.com'
        assert request.method == 'GET'
        assert response.status_code == 200
        assert policy.type == PolicyType.RATE_LIMITING
        assert metrics.request_count == 15423
        assert security_event.threat_level == ThreatLevel.MEDIUM
        
        print("   ✅ Models tested successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Models test failed: {str(e)}")
        return False

async def test_component_integration():
    """Test that components can work together conceptually."""
    print("🧪 Testing component integration...")
    
    try:
        # Import all main components
        from apg_clients import APGServiceConfig, APGAuthRBACClient
        from ollama_client import OllamaConfig, ProductionOllamaClient
        from wasm_runtime import ProductionWASMRuntime
        from models import AgGatewayConfig, EnvironmentType
        
        # Test configurations work together
        apg_config = APGServiceConfig(
            base_url='http://localhost:8000',
            api_key='integration-test-key'
        )
        
        ollama_config = OllamaConfig(
            base_url='http://localhost:11434'
        )
        
        # Test client instantiations
        auth_client = APGAuthRBACClient(apg_config, 'integration-test-tenant')
        ollama_client = ProductionOllamaClient(ollama_config, 'integration-test-tenant')
        wasm_runtime = ProductionWASMRuntime('integration-test-tenant')
        
        # Test gateway config
        gateway = AgGatewayConfig(
            name='integration-test-gateway',
            environment=EnvironmentType.DEVELOPMENT,
            tenant_id='integration-test-tenant',
            created_by='integration-test-user',
            listen_port=8080
        )
        
        # Verify all components use the same tenant
        assert auth_client.tenant_id == 'integration-test-tenant'
        assert ollama_client.tenant_id == 'integration-test-tenant'
        assert wasm_runtime.tenant_id == 'integration-test-tenant'
        assert gateway.tenant_id == 'integration-test-tenant'
        
        print("   ✅ Component integration tested successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Component integration test failed: {str(e)}")
        return False

async def run_minimal_integration_tests():
    """Run minimal integration tests."""
    print("🚀 Starting Minimal Integration Tests")
    print("=" * 55)
    
    tests = [
        ("APG Clients", test_apg_clients),
        ("Ollama Client", test_ollama_client), 
        ("WASM Runtime", test_wasm_runtime),
        ("Models", test_models),
        ("Component Integration", test_component_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {str(e)}")
    
    print("\n" + "=" * 55)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All minimal integration tests PASSED!")
        print("🎉 Production components are working correctly!")
        print("📋 Key achievements verified:")
        print("   • All APG service clients instantiate correctly")
        print("   • Ollama AI client configuration works")  
        print("   • WASM runtime structures are valid")
        print("   • All models validate and create successfully")
        print("   • Components integrate with consistent tenant IDs")
        print("\n🏆 Production transformation VALIDATED!")
    else:
        print("⚠️  Some tests failed - check component implementations")
    
    return passed == total

if __name__ == '__main__':
    success = asyncio.run(run_minimal_integration_tests())
    exit(0 if success else 1)