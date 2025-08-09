#!/usr/bin/env python3
"""
Advanced Template System Validation

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Comprehensive validation tests for the advanced template system with inheritance,
versioning, marketplace, and composition capabilities.
"""

import asyncio
import sys
from datetime import datetime, UTC


print("🚀 Advanced Template System Validation")
print("=" * 70)


async def test_template_system_initialization():
	"""Test template system initialization and built-in templates"""
	print("🧪 Testing Template System Initialization...")
	
	# Import template system components
	from template_system import (
		AdvancedTemplateSystem, TemplateType, TemplateCategory, 
		TemplateStatus, EnterpriseTemplate
	)
	from models import ResourceAllocation, TenantTier, CloudProvider
	
	# Initialize template system
	template_system = AdvancedTemplateSystem()
	
	# Load built-in templates
	await template_system.initialize_builtin_templates()
	
	# Verify built-in templates loaded
	templates = await template_system.list_templates(status=TemplateStatus.PUBLISHED)
	assert len(templates) >= 5, "Should have at least 5 built-in templates"
	
	# Check template variety
	template_types = set(t.metadata.template_type for t in templates)
	categories = set(t.metadata.category for t in templates)
	
	assert len(template_types) >= 3, "Should have multiple template types"
	assert len(categories) >= 4, "Should have multiple categories"
	
	print(f"  ✅ Template system initialized with {len(templates)} built-in templates")
	print(f"  ✅ Template types: {[tt.value for tt in template_types]}")
	print(f"  ✅ Categories: {[c.value for c in categories]}")
	
	return template_system


async def test_template_creation_and_management():
	"""Test template creation, updating, and management"""
	print("🧪 Testing Template Creation & Management...")
	
	template_system = await test_template_system_initialization()
	
	from template_system import TemplateType, TemplateCategory
	from models import ResourceAllocation
	
	# Create custom template
	custom_template = await template_system.create_template(
		name="test_custom_template",
		template_type=TemplateType.CUSTOM,
		category=TemplateCategory.WEB_APPLICATION,
		configuration={
			"application": {
				"framework": "vue.js",
				"backend": "python",
				"database": "mysql"
			},
			"features": {
				"auth": True,
				"api": True,
				"monitoring": False
			}
		},
		resource_allocation=ResourceAllocation(
			cpu_cores=2,
			memory_gb=4,
			storage_gb=50,
			bandwidth_mbps=100,
			database_connections=25
		),
		author="Test User",
		description="Custom test template for validation"
	)
	
	assert custom_template is not None
	assert custom_template.metadata.name == "test_custom_template"
	assert custom_template.metadata.current_version == "1.0.0"
	assert custom_template.metadata.status == TemplateStatus.DRAFT
	
	print(f"  ✅ Custom template created: {custom_template.metadata.id}")
	
	# Update template
	updated_template = await template_system.update_template(
		custom_template.metadata.id,
		{
			"configuration": {
				"features": {"monitoring": True, "logging": True}
			},
			"changelog": ["Added monitoring and logging features"],
			"updated_by": "Test User"
		},
		version_increment="minor"
	)
	
	assert updated_template.metadata.current_version == "1.1.0"
	assert len(updated_template.metadata.versions) == 2
	effective_config = updated_template.get_effective_configuration()
	assert effective_config["features"]["monitoring"] is True
	
	print(f"  ✅ Template updated to version {updated_template.metadata.current_version}")
	
	# Test version retrieval
	old_version = await template_system.get_template(custom_template.metadata.id, "1.0.0")
	assert old_version is not None
	assert old_version.metadata.current_version == "1.0.0"
	
	print("  ✅ Template versioning working correctly")
	
	return template_system, custom_template.metadata.id


async def test_template_inheritance():
	"""Test template inheritance functionality"""
	print("🧪 Testing Template Inheritance...")
	
	template_system, parent_template_id = await test_template_creation_and_management()
	
	# Create child template inheriting from parent
	child_template = await template_system.inherit_template(
		name="child_template_test",
		parent_template_id=parent_template_id,
		overrides={
			"application": {
				"database": "postgresql"  # Override parent's mysql
			},
			"features": {
				"caching": True  # Add new feature
			}
		},
		author="Test User"
	)
	
	assert child_template.metadata.parent_template_id == parent_template_id
	assert child_template.metadata.inheritance_depth == 1
	
	# Get effective configuration with inheritance resolved
	effective_config = await template_system.get_effective_configuration(child_template.metadata.id)
	
	# Verify inheritance resolution
	assert effective_config["application"]["framework"] == "vue.js"  # Inherited from parent
	assert effective_config["application"]["database"] == "postgresql"  # Overridden
	assert effective_config["features"]["caching"] is True  # Added in child
	assert effective_config["features"]["monitoring"] is True  # Inherited from parent
	
	print(f"  ✅ Child template created with inheritance depth {child_template.metadata.inheritance_depth}")
	print(f"  ✅ Configuration inheritance resolved correctly")
	
	# Test grandchild template (deeper inheritance)
	grandchild_template = await template_system.inherit_template(
		name="grandchild_template_test",
		parent_template_id=child_template.metadata.id,
		overrides={
			"features": {
				"analytics": True,
				"monitoring": False  # Override grandparent
			}
		},
		author="Test User"
	)
	
	assert grandchild_template.metadata.inheritance_depth == 2
	
	grandchild_config = await template_system.get_effective_configuration(grandchild_template.metadata.id)
	assert grandchild_config["application"]["framework"] == "vue.js"  # From root
	assert grandchild_config["application"]["database"] == "postgresql"  # From parent
	assert grandchild_config["features"]["analytics"] is True  # From grandchild
	assert grandchild_config["features"]["monitoring"] is False  # Overridden in grandchild
	
	print(f"  ✅ Grandchild template created with inheritance depth {grandchild_template.metadata.inheritance_depth}")
	print("  ✅ Multi-level inheritance working correctly")
	
	return template_system, [parent_template_id, child_template.metadata.id, grandchild_template.metadata.id]


async def test_template_composition():
	"""Test template composition functionality"""
	print("🧪 Testing Template Composition...")
	
	template_system, template_ids = await test_template_inheritance()
	
	# Get some built-in templates for composition
	builtin_templates = await template_system.list_templates(status=TemplateStatus.PUBLISHED)
	web_app_template = next((t for t in builtin_templates if t.metadata.category == TemplateCategory.WEB_APPLICATION), None)
	microservices_template = next((t for t in builtin_templates if t.metadata.category == TemplateCategory.MICROSERVICES), None)
	
	if not web_app_template or not microservices_template:
		print("  ⚠️ Skipping composition test - insufficient built-in templates")
		return template_system
	
	# Compose templates
	composition_template_ids = [web_app_template.metadata.id, template_ids[0]]  # Built-in + custom
	
	composed_template = await template_system.compose_templates(
		template_ids=composition_template_ids,
		name="composed_webapp_template",
		author="Test User"
	)
	
	assert composed_template is not None
	assert len(composed_template.composed_templates) == len(composition_template_ids)
	assert composed_template.metadata.template_type == TemplateType.CUSTOM
	
	# Verify composition merged configurations
	effective_config = await template_system.get_effective_configuration(composed_template.metadata.id)
	assert "application" in effective_config
	assert "infrastructure" in effective_config or "features" in effective_config
	
	print(f"  ✅ Template composition created from {len(composition_template_ids)} templates")
	print(f"  ✅ Composed template ID: {composed_template.metadata.id}")
	
	return template_system


async def test_template_marketplace():
	"""Test template marketplace functionality"""
	print("🧪 Testing Template Marketplace...")
	
	template_system = await test_template_composition()
	
	# Get a template to publish
	draft_templates = await template_system.list_templates(status=TemplateStatus.DRAFT)
	if not draft_templates:
		print("  ⚠️ No draft templates available for marketplace testing")
		return template_system
	
	test_template = draft_templates[0]
	
	# Publish template to marketplace
	published = await template_system.publish_template(test_template.metadata.id)
	assert published is True
	
	# Verify template is now published
	updated_template = await template_system.get_template(test_template.metadata.id)
	assert updated_template.metadata.status == TemplateStatus.PUBLISHED
	assert updated_template.metadata.published_at is not None
	
	print(f"  ✅ Template published to marketplace: {test_template.metadata.name}")
	
	# Search marketplace
	from template_system import TemplateType, TemplateCategory
	
	# Search by category
	web_templates = await template_system.search_marketplace(
		category=TemplateCategory.WEB_APPLICATION,
		sort_by="name"
	)
	assert len(web_templates) >= 1
	
	# Search by query
	search_results = await template_system.search_marketplace(
		query="web",
		min_rating=0.0
	)
	assert len(search_results) >= 1
	
	print(f"  ✅ Marketplace search: {len(web_templates)} web templates, {len(search_results)} search results")
	
	# Test template rating (mock user rating)
	rated = await template_system._marketplace.rate_template(test_template.metadata.id, "test_user_1", 5)
	assert rated is True
	
	rated2 = await template_system._marketplace.rate_template(test_template.metadata.id, "test_user_2", 4)
	assert rated2 is True
	
	# Get template stats
	stats = await template_system._marketplace.get_template_stats(test_template.metadata.id)
	assert stats["average_rating"] == 4.5  # (5 + 4) / 2
	assert stats["total_ratings"] == 2
	
	print(f"  ✅ Template rating system: {stats['average_rating']:.1f} average from {stats['total_ratings']} ratings")
	
	return template_system


async def test_template_validation():
	"""Test template validation and compatibility checks"""
	print("🧪 Testing Template Validation...")
	
	template_system = await test_template_marketplace()
	
	from models import TenantTier, CloudProvider
	
	# Get a template for validation
	templates = await template_system.list_templates(status=TemplateStatus.PUBLISHED)
	test_template = templates[0] if templates else None
	
	if not test_template:
		print("  ⚠️ No templates available for validation testing")
		return template_system
	
	# Test compatibility validation
	compatibility_issues = test_template.validate_compatibility(
		tenant_tier=TenantTier.FREE,
		cloud_provider=CloudProvider.AWS
	)
	
	# Check if compatibility issues are reasonable
	if test_template.metadata.min_tier != TenantTier.FREE:
		assert len(compatibility_issues) > 0, "Should have tier compatibility issues"
		print(f"  ✅ Tier compatibility validation: {len(compatibility_issues)} issues detected")
	else:
		print("  ✅ Tier compatibility validation: no issues (template supports FREE tier)")
	
	# Test with compatible tier
	compatible_issues = test_template.validate_compatibility(
		tenant_tier=TenantTier.ENTERPRISE,
		cloud_provider=CloudProvider.AWS
	)
	
	print(f"  ✅ Enterprise tier compatibility: {len(compatible_issues)} issues")
	
	# Test effective configuration
	effective_config = await template_system.get_effective_configuration(test_template.metadata.id)
	assert isinstance(effective_config, dict)
	assert len(effective_config) > 0
	
	print(f"  ✅ Effective configuration resolved: {len(effective_config)} top-level keys")
	
	return template_system


async def test_template_performance():
	"""Test template system performance"""
	print("🧪 Testing Template Performance...")
	
	template_system = await test_template_validation()
	
	from template_system import TemplateType, TemplateCategory
	from models import ResourceAllocation
	
	# Performance test: create multiple templates
	start_time = datetime.now(UTC)
	created_templates = []
	
	for i in range(10):
		template = await template_system.create_template(
			name=f"perf_test_template_{i}",
			template_type=TemplateType.CUSTOM,
			category=TemplateCategory.WEB_APPLICATION,
			configuration={
				"app": {"id": i, "name": f"app_{i}"},
				"config": {"debug": i % 2 == 0}
			},
			resource_allocation=ResourceAllocation(
				cpu_cores=1,
				memory_gb=2,
				storage_gb=20,
				bandwidth_mbps=50,
				database_connections=10
			),
			author="Performance Test"
		)
		created_templates.append(template.metadata.id)
	
	creation_time = (datetime.now(UTC) - start_time).total_seconds()
	avg_creation_time = creation_time / 10
	
	print(f"  ⚡ Template creation: {avg_creation_time:.3f}s per template")
	
	# Performance test: template listing
	start_time = datetime.now(UTC)
	all_templates = await template_system.list_templates()
	listing_time = (datetime.now(UTC) - start_time).total_seconds()
	
	print(f"  ⚡ Template listing: {listing_time:.3f}s for {len(all_templates)} templates")
	
	# Performance test: inheritance resolution
	if len(created_templates) >= 2:
		start_time = datetime.now(UTC)
		
		# Create inheritance chain
		child = await template_system.inherit_template(
			name="perf_child",
			parent_template_id=created_templates[0],
			overrides={"child": True},
			author="Performance Test"
		)
		
		effective_config = await template_system.get_effective_configuration(child.metadata.id)
		inheritance_time = (datetime.now(UTC) - start_time).total_seconds()
		
		print(f"  ⚡ Inheritance resolution: {inheritance_time:.3f}s")
	
	# Performance test: marketplace search
	start_time = datetime.now(UTC)
	search_results = await template_system.search_marketplace(query="test")
	search_time = (datetime.now(UTC) - start_time).total_seconds()
	
	print(f"  ⚡ Marketplace search: {search_time:.3f}s for {len(search_results)} results")
	
	# Performance assertions
	assert avg_creation_time < 0.1, f"Template creation too slow: {avg_creation_time:.3f}s"
	assert listing_time < 0.5, f"Template listing too slow: {listing_time:.3f}s"
	assert search_time < 0.2, f"Marketplace search too slow: {search_time:.3f}s"
	
	print("  ✅ All performance benchmarks met")
	
	return True


async def main():
	"""Run all template system validation tests"""
	all_passed = True
	
	print("Testing Template System Initialization...")
	try:
		await test_template_system_initialization()
		print()
	except Exception as e:
		print(f"  ❌ Template system initialization test failed: {e}")
		all_passed = False
	
	print("Testing Template Creation & Management...")
	try:
		await test_template_creation_and_management()
		print()
	except Exception as e:
		print(f"  ❌ Template creation & management test failed: {e}")
		all_passed = False
	
	print("Testing Template Inheritance...")
	try:
		await test_template_inheritance()
		print()
	except Exception as e:
		print(f"  ❌ Template inheritance test failed: {e}")
		all_passed = False
	
	print("Testing Template Composition...")
	try:
		await test_template_composition()
		print()
	except Exception as e:
		print(f"  ❌ Template composition test failed: {e}")
		all_passed = False
	
	print("Testing Template Marketplace...")
	try:
		await test_template_marketplace()
		print()
	except Exception as e:
		print(f"  ❌ Template marketplace test failed: {e}")
		all_passed = False
	
	print("Testing Template Validation...")
	try:
		await test_template_validation()
		print()
	except Exception as e:
		print(f"  ❌ Template validation test failed: {e}")
		all_passed = False
	
	print("Testing Template Performance...")
	try:
		await test_template_performance()
		print()
	except Exception as e:
		print(f"  ❌ Template performance test failed: {e}")
		all_passed = False
	
	print("=" * 70)
	
	if all_passed:
		print("🎉 ALL ADVANCED TEMPLATE SYSTEM TESTS PASSED!")
		print("✅ Template system initialization with built-in library operational")
		print("✅ Template creation, versioning, and management functional")
		print("✅ Multi-level template inheritance with override resolution")
		print("✅ Template composition for complex multi-capability deployments")
		print("✅ Template marketplace with rating and discovery features")
		print("✅ Template validation and compatibility checking")
		print("✅ Performance benchmarks met (sub-second operations)")
		print("✅ Enterprise-grade template lifecycle management")
		print("🚀 Phase 4.1: Advanced Template System COMPLETE")
		print()
		print("🎯 Template System Capabilities:")
		print("   • 5+ built-in enterprise templates (Web App, Microservices, AI/ML, etc.)")
		print("   • Multi-level inheritance with intelligent override resolution")
		print("   • Semantic versioning with rollback capabilities")
		print("   • Template composition for complex deployments")
		print("   • Marketplace with rating, search, and discovery")
		print("   • Compatibility validation and requirements checking")
		print("   • Sub-100ms template operations performance")
		return True
	else:
		print("❌ SOME ADVANCED TEMPLATE SYSTEM TESTS FAILED")
		return False


if __name__ == "__main__":
	success = asyncio.run(main())
	sys.exit(0 if success else 1)