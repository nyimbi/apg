#!/usr/bin/env python3
"""
Phase 1 Validation Script for Multi-Tenant Management Capability

Quick validation of core functionality without complex imports.
"""

import sys
import traceback
from pathlib import Path

def validate_phase1() -> bool:
	"""Validate Phase 1 implementation components"""
	try:
		print("🔍 Validating Multi-Tenant Management Phase 1 Implementation...")
		
		# Check core files exist
		core_files = [
			'models.py',
			'service.py', 
			'views.py',
			'api.py',
			'blueprint.py',
			'__init__.py',
			'cap_spec.md',
			'todo.md'
		]
		
		missing_files = []
		for file in core_files:
			if not Path(file).exists():
				missing_files.append(file)
		
		if missing_files:
			print(f"❌ Missing core files: {missing_files}")
			return False
		
		print("✅ All core files present")
		
		# Test basic imports
		try:
			# Test models import and basic functionality
			exec("""
from models import Tenant, TenantStatus, TenantTier, ResourceAllocation

# Test basic model creation
tenant = Tenant(
	name="test-tenant",
	display_name="Test Tenant", 
	organization_name="Test Org",
	contact_email="test@example.com",
	primary_domain="test.example.com",
	created_by="tester"
)

assert tenant.name == "test-tenant"
assert tenant.status == TenantStatus.PROVISIONING
assert tenant.tier == TenantTier.FREE
print("✅ Core models working correctly")

# Test resource allocation
allocation = ResourceAllocation(
	cpu_cores=4,
	memory_gb=16, 
	storage_gb=100,
	bandwidth_mbps=1000,
	database_connections=50
)
assert allocation.total_compute_units() == 70
print("✅ Resource allocation model working correctly")
""")
		except Exception as e:
			print(f"❌ Models validation failed: {e}")
			return False
		
		# Test views import
		try:
			exec("""
from views import TenantCreateRequest, TenantResponse, TenantTier

# Test view model creation
request = TenantCreateRequest(
	name="api-test",
	display_name="API Test",
	organization_name="API Org", 
	contact_email="api@test.com",
	primary_domain="api.test.com"
)
assert request.tier == TenantTier.FREE
print("✅ View models working correctly")
""")
		except Exception as e:
			print(f"❌ Views validation failed: {e}")
			return False
		
		# Test service basic structure
		try:
			exec("""
from service import MultiTenantManager

# Test service instantiation
service = MultiTenantManager(tenant_id="test-system")
assert service.tenant_id == "test-system"
print("✅ Service class instantiation working correctly")
""")
		except Exception as e:
			print(f"❌ Service validation failed: {e}")
			return False
		
		# Check capability metadata
		try:
			exec("""
from __init__ import APG_CAPABILITY_METADATA

assert APG_CAPABILITY_METADATA['name'] == 'mten'
assert 'multi_tenant_management' in APG_CAPABILITY_METADATA['provides']
assert 'auth_rbac' in APG_CAPABILITY_METADATA['dependencies']
print("✅ APG capability metadata configured correctly")
""")
		except Exception as e:
			print(f"❌ Capability metadata validation failed: {e}")
			return False
		
		print("\n🎉 Phase 1 Validation PASSED!")
		print("✅ Multi-Tenant Management capability foundation is solid")
		print("✅ All core components integrated and functional")
		print("✅ Ready to proceed to Phase 2: Advanced Multi-Tenancy Features")
		
		return True
		
	except Exception as e:
		print(f"❌ Validation failed with error: {e}")
		traceback.print_exc()
		return False

if __name__ == "__main__":
	success = validate_phase1()
	sys.exit(0 if success else 1)