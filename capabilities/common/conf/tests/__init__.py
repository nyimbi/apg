"""
APG Configuration Management Tests

Comprehensive test suite for the revolutionary configuration management system
ensuring >95% code coverage and APG quality standards compliance.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from typing import Dict, Any

# Test fixtures and utilities
@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop()
    return loop

@pytest.fixture
async def config_manager():
    """Create test configuration manager"""
    from ..service import create_configuration_manager
    manager = await create_configuration_manager(tenant_id="test_tenant")
    await manager.initialize({})
    return manager

@pytest.fixture
def sample_configuration() -> Dict[str, Any]:
    """Sample configuration for testing"""
    return {
        "name": "test-vm",
        "type": "virtual_machine", 
        "cloud_provider": "aws",
        "configuration": {
            "kind": "VirtualMachine",
            "spec": {
                "resources": {
                    "instance_type": "t3.micro",
                    "image": "ami-12345",
                    "vpc_id": "vpc-test"
                }
            }
        },
        "description": "Test virtual machine configuration"
    }

# Test markers for different test categories
pytestmark = [
    pytest.mark.asyncio,  # All tests are async by default
]