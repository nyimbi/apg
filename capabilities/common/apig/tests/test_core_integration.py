#!/usr/bin/env python3
"""
Core Integration Test

Simple integration test focusing on core functionality without external dependencies.
Tests that all production components can be imported and initialized.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_production_imports():
    """Test that all production components can be imported."""
    print("🧪 Testing production component imports...")
    
    try:
        # Test APG clients import
        from apg_clients import (
            APGAuthRBACClient, APGMonitoringClient, APGConfigurationClient,
            APGAIOrchestrationClient, APGMessageQueueClient, APGAuditComplianceClient,
            APGServiceConfig
        )
        print("   ✅ APG clients imported successfully")
        
        # Test Ollama client import
        from ollama_client import (
            ProductionOllamaClient, OllamaConfig, GenerationRequest
        )
        print("   ✅ Ollama client imported successfully")
        
        # Test WASM runtime import (may fail without wasmtime)
        try:
            from wasm_runtime import (
                ProductionWASMRuntime, WASMExecutionContext
            )
            print("   ✅ WASM runtime imported successfully")
        except ImportError as e:
            print(f"   ⚠️  WASM runtime import failed (expected without wasmtime): {str(e)}")
        
        # Test models import
        from models import (
            AgGatewayConfig, AgApiRoute, AgPolicy, AgHttpRequest, 
            EnvironmentType, PolicyType
        )
        print("   ✅ Models imported successfully")
        
        # Test service import (may fail without dependencies)
        try:
            from service import ProductionAPGIntelligentGatewayService
            print("   ✅ Production service imported successfully")
            return True
        except ImportError as e:
            print(f"   ⚠️  Production service import failed: {str(e)}")
            return False
            
    except Exception as e:
        print(f"   ❌ Import failed: {str(e)}")
        return False

async def test_apg_service_config():
    """Test APG service configuration."""
    print("🧪 Testing APG service configuration...")
    
    try:
        from apg_clients import APGServiceConfig
        
        config = APGServiceConfig(
            base_url='http://localhost:8000',
            api_key='test-key',
            timeout=30,
            circuit_breaker_threshold=5
        )
        
        assert config.base_url == 'http://localhost:8000'
        assert config.api_key == 'test-key'
        assert config.timeout == 30
        assert config.circuit_breaker_threshold == 5
        
        print("   ✅ APG service configuration created successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ APG service configuration failed: {str(e)}")
        return False

async def test_model_creation():
    """Test model creation and validation."""
    print("🧪 Testing model creation...")
    
    try:
        from models import AgGatewayConfig, AgApiRoute, AgUpstreamService, EnvironmentType
        
        # Create upstream service
        upstream = AgUpstreamService(
            name='test-upstream',
            base_url='http://localhost:3000',
            weight=100
        )
        
        # Create API route
        route = AgApiRoute(
            method='GET',
            path='/api/test',
            upstream_services=[upstream],
            tenant_id='test-tenant',
            created_by='test-user'
        )
        
        # Create gateway config
        gateway = AgGatewayConfig(
            name='test-gateway',
            environment=EnvironmentType.DEVELOPMENT,
            tenant_id='test-tenant',
            created_by='test-user',
            listen_port=8080,
            routes=[route]
        )
        
        # Verify properties
        assert gateway.name == 'test-gateway'
        assert gateway.environment == EnvironmentType.DEVELOPMENT
        assert gateway.tenant_id == 'test-tenant'
        assert len(gateway.routes) == 1
        assert gateway.routes[0].method == 'GET'
        
        print("   ✅ Models created and validated successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {str(e)}")
        return False

async def test_ollama_client_config():
    """Test Ollama client configuration."""
    print("🧪 Testing Ollama client configuration...")
    
    try:
        from ollama_client import OllamaConfig, GenerationRequest
        
        # Test configuration
        config = OllamaConfig(
            base_url='http://localhost:11434',
            timeout=60,
            max_retries=3
        )
        
        assert config.base_url == 'http://localhost:11434'
        assert config.timeout == 60
        assert config.max_retries == 3
        
        # Test generation request
        request = GenerationRequest(
            model='llama3.2:latest',
            prompt='Test prompt',
            system='Test system prompt'
        )
        
        assert request.model == 'llama3.2:latest'
        assert request.prompt == 'Test prompt'
        assert request.system == 'Test system prompt'
        
        print("   ✅ Ollama client configuration created successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Ollama client configuration failed: {str(e)}")
        return False

async def test_service_instantiation():
    """Test service instantiation without initialization."""
    print("🧪 Testing service instantiation...")
    
    try:
        from service import ProductionAPGIntelligentGatewayService
        
        # Create service instance (without initialization)
        service = ProductionAPGIntelligentGatewayService(
            tenant_id='test-tenant-123',
            user_id='test-user-456',
            config={
                'apg_base_url': 'http://localhost:8000',
                'enable_wasm': False,  # Disable to avoid wasmtime dependency
                'enable_ai': False     # Disable to avoid Ollama dependency
            }
        )
        
        # Verify basic properties
        assert service.tenant_id == 'test-tenant-123'
        assert service.user_id == 'test-user-456'
        assert service.initialized == False
        assert isinstance(service.gateway_configs, dict)
        assert isinstance(service.policies, dict)
        assert isinstance(service.traffic_metrics, dict)
        
        print("   ✅ Service instantiated successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Service instantiation failed: {str(e)}")
        return False

async def run_core_integration_tests():
    """Run all core integration tests."""
    print("🚀 Starting Core Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_production_imports),
        ("APG Config Tests", test_apg_service_config),
        ("Model Creation Tests", test_model_creation),
        ("Ollama Config Tests", test_ollama_client_config),
        ("Service Instantiation Tests", test_service_instantiation)
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
        
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All core integration tests PASSED!")
        print("🎉 Production components are correctly structured and importable")
    else:
        print("⚠️  Some tests failed - check dependencies and imports")
    
    return passed == total

if __name__ == '__main__':
    success = asyncio.run(run_core_integration_tests())
    exit(0 if success else 1)