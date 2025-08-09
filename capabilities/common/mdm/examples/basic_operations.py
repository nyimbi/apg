#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Basic Operations Example
Demonstrates core CRUD operations for entities

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime
from uuid_extensions import uuid7str

from ..service import MDMService
from ..models import MdEntityCreate, MdEntityUpdate, EntityType, EntityStatus
from ..database import MDMDatabaseManager
from ..integrations import APGIntegrationManager


async def setup_mdm_service():
	"""Initialize MDM service for examples"""
	print("🚀 Initializing APG MDM Service...")
	
	# Database configuration
	db_config = {
		"database_url": "postgresql://mdm_user:password@localhost:5432/apg_mdm",
		"pool_size": 10,
		"max_overflow": 20
	}
	
	# APG integration configuration  
	apg_config = {
		"mqeb_url": "http://localhost:8080",
		"redis_url": "redis://localhost:6379/0",
		"audl_url": "http://localhost:8081",
		"conf_url": "http://localhost:8082"
	}
	
	# Initialize components
	db_manager = MDMDatabaseManager(db_config)
	integration_manager = APGIntegrationManager(apg_config)
	
	# Create MDM service
	mdm_service = MDMService(
		db_manager=db_manager,
		integration_manager=integration_manager,
		config={"enable_ai": True, "enable_caching": True}
	)
	
	await mdm_service.initialize()
	
	# Verify health
	health = await mdm_service.health_check()
	print(f"✅ MDM Service Status: {health['status']}")
	
	return mdm_service


async def example_create_entities(mdm_service: MDMService):
	"""Example: Creating different types of entities"""
	print("\n📝 Creating Entities Example")
	print("=" * 50)
	
	tenant_id = "demo-tenant"
	user_id = "demo-user"
	
	# Example 1: Create a Person entity
	print("\n1. Creating Person Entity...")
	
	person_data = MdEntityCreate(
		tenant_id=tenant_id,
		entity_type=EntityType.PERSON,
		entity_name="Sarah Chen",
		entity_description="Senior Software Engineer",
		business_key="EMP-2024-001",
		source_system="hr_system",
		status=EntityStatus.ACTIVE,
		attributes={
			"first_name": "Sarah",
			"last_name": "Chen",
			"email": "sarah.chen@company.com",
			"phone": "+1-555-987-6543",
			"employee_id": "E001234",
			"department": "Engineering",
			"title": "Senior Software Engineer",
			"hire_date": "2024-01-15",
			"manager": "john.doe@company.com",
			"location": "San Francisco, CA",
			"skills": ["Python", "Machine Learning", "Cloud Architecture"],
			"security_clearance": "Level 2"
		},
		tags=["employee", "engineering", "senior", "active"],
		data_classification="confidential"
	)
	
	context = mdm_service.create_operation_context(
		tenant_id=tenant_id,
		user_id=user_id,
		operation_type="create_entity",
		source_system="basic_example"
	)
	
	result = await mdm_service.entity_service.create_entity(person_data, context)
	
	if result["status"] == "success":
		person_id = result["entity_id"]
		print(f"✅ Person created: {person_id}")
	else:
		print(f"❌ Error: {result['message']}")
		return
	
	# Example 2: Create a Customer entity
	print("\n2. Creating Customer Entity...")
	
	customer_data = MdEntityCreate(
		tenant_id=tenant_id,
		entity_type=EntityType.CUSTOMER,
		entity_name="TechCorp Industries",
		entity_description="Technology consulting company",
		business_key="CUST-2024-501",
		source_system="crm_system",
		status=EntityStatus.ACTIVE,
		attributes={
			"company_name": "TechCorp Industries",
			"industry": "Technology Consulting",
			"revenue": 15000000,
			"employees": 250,
			"website": "https://techcorp.example.com",
			"primary_contact": "ceo@techcorp.example.com",
			"phone": "+1-555-TECH-001",
			"address": {
				"street": "123 Innovation Drive",
				"city": "Austin",
				"state": "TX",
				"postal_code": "73301",
				"country": "USA"
			},
			"contract_value": 2500000,
			"contract_start": "2024-02-01",
			"contract_end": "2026-01-31",
			"account_manager": "sarah.chen@company.com",
			"payment_terms": "Net 30",
			"customer_since": "2020-06-15"
		},
		tags=["customer", "enterprise", "technology", "active"],
		data_classification="confidential"
	)
	
	result = await mdm_service.entity_service.create_entity(customer_data, context)
	
	if result["status"] == "success":
		customer_id = result["entity_id"]
		print(f"✅ Customer created: {customer_id}")
	else:
		print(f"❌ Error: {result['message']}")
		return
	
	# Example 3: Create a Product entity
	print("\n3. Creating Product Entity...")
	
	product_data = MdEntityCreate(
		tenant_id=tenant_id,
		entity_type=EntityType.PRODUCT,
		entity_name="CloudSync Pro 2024",
		entity_description="Enterprise cloud synchronization software",
		business_key="PROD-CS-2024",
		source_system="product_catalog",
		status=EntityStatus.ACTIVE,
		attributes={
			"product_name": "CloudSync Pro 2024",
			"sku": "CS-PRO-2024-ENT",
			"category": "Enterprise Software",
			"subcategory": "Cloud Infrastructure",
			"version": "2024.1.0",
			"price": 2999.99,
			"currency": "USD",
			"license_type": "Enterprise",
			"features": [
				"Real-time synchronization",
				"Multi-cloud support",
				"Enterprise security",
				"API integration",
				"Advanced monitoring"
			],
			"technical_specs": {
				"supported_platforms": ["Windows", "Linux", "macOS"],
				"min_memory": "8GB",
				"min_storage": "100GB",
				"network_requirements": "High-speed internet"
			},
			"release_date": "2024-03-01",
			"support_level": "Premium",
			"warranty_months": 36
		},
		tags=["product", "software", "enterprise", "2024"],
		data_classification="public"
	)
	
	result = await mdm_service.entity_service.create_entity(product_data, context)
	
	if result["status"] == "success":
		product_id = result["entity_id"]
		print(f"✅ Product created: {product_id}")
	else:
		print(f"❌ Error: {result['message']}")
		return
	
	return person_id, customer_id, product_id


async def example_retrieve_entities(mdm_service: MDMService, entity_ids):
	"""Example: Retrieving entities with various options"""
	print("\n🔍 Retrieving Entities Example")
	print("=" * 50)
	
	tenant_id = "demo-tenant"
	person_id, customer_id, product_id = entity_ids
	
	# Example 1: Basic entity retrieval
	print("\n1. Basic Entity Retrieval...")
	
	result = await mdm_service.entity_service.get_entity(person_id, tenant_id)
	
	if result["status"] == "success":
		entity = result["entity"]
		print(f"✅ Retrieved: {entity['entity_name']}")
		print(f"   Type: {entity['entity_type']}")
		print(f"   Business Key: {entity['business_key']}")
		print(f"   Quality Score: {entity['quality_score']:.1f}%")
		print(f"   Status: {entity['status']}")
	else:
		print(f"❌ Error: {result['message']}")
	
	# Example 2: Entity retrieval with all includes
	print("\n2. Detailed Entity Retrieval (with versions, quality, cross-refs)...")
	
	result = await mdm_service.entity_service.get_entity(
		customer_id, 
		tenant_id,
		include_versions=True,
		include_quality=True,
		include_cross_refs=True
	)
	
	if result["status"] == "success":
		entity = result["entity"]
		print(f"✅ Retrieved detailed data for: {entity['entity_name']}")
		
		# Show versions if available
		if entity.get("versions"):
			print(f"   Versions: {len(entity['versions'])} history entries")
		
		# Show quality assessment if available
		if entity.get("quality_assessment"):
			qa = entity["quality_assessment"]
			print(f"   Quality Assessment: {qa.get('overall_score', 'N/A')}% "
			      f"({qa.get('quality_status', 'unknown')})")
		
		# Show cross-references if available
		if entity.get("cross_references"):
			print(f"   Cross-references: {len(entity['cross_references'])} external links")
	else:
		print(f"❌ Error: {result['message']}")


async def example_update_entities(mdm_service: MDMService, entity_ids):
	"""Example: Updating entities with different scenarios"""
	print("\n✏️  Updating Entities Example")
	print("=" * 50)
	
	tenant_id = "demo-tenant"
	user_id = "demo-user"
	person_id, customer_id, product_id = entity_ids
	
	# Example 1: Update person attributes
	print("\n1. Updating Person Attributes...")
	
	person_update = MdEntityUpdate(
		entity_name="Sarah Chen, Ph.D.",  # Promotion!
		entity_description="Principal Software Engineer & Team Lead",
		attributes={
			"title": "Principal Software Engineer",
			"department": "Engineering - AI/ML Team",
			"phone": "+1-555-987-6543",  # Keep same
			"education": "Ph.D. Computer Science, Stanford University",
			"certifications": ["AWS Solutions Architect", "Google Cloud Professional"],
			"promotion_date": "2024-06-01",
			"salary_band": "L7",
			"team_size": 8
		},
		tags=["employee", "engineering", "principal", "team_lead", "active"]
	)
	
	context = mdm_service.create_operation_context(
		tenant_id=tenant_id,
		user_id=user_id,
		operation_type="update_entity",
		entity_id=person_id,
		source_system="hr_promotion_update"
	)
	
	result = await mdm_service.entity_service.update_entity(person_id, person_update, context)
	
	if result["status"] == "success":
		print(f"✅ Person updated successfully")
		print(f"   Updated at: {result['updated_at']}")
	else:
		print(f"❌ Error: {result['message']}")
	
	# Example 2: Update customer contract information
	print("\n2. Updating Customer Contract...")
	
	customer_update = MdEntityUpdate(
		attributes={
			"contract_value": 3500000,  # Contract extension
			"contract_end": "2027-01-31",  # Extended by 1 year
			"payment_terms": "Net 15",  # Better terms
			"last_contract_review": "2024-05-15",
			"contract_amendments": [
				{"date": "2024-05-15", "change": "Value increase to $3.5M"},
				{"date": "2024-05-15", "change": "Extended term by 12 months"}
			],
			"account_notes": "Excellent client relationship, expanded services"
		},
		tags=["customer", "enterprise", "technology", "expanded", "active"]
	)
	
	context = mdm_service.create_operation_context(
		tenant_id=tenant_id,
		user_id=user_id,
		operation_type="update_entity",
		entity_id=customer_id,
		source_system="contract_management_update"
	)
	
	result = await mdm_service.entity_service.update_entity(customer_id, customer_update, context)
	
	if result["status"] == "success":
		print(f"✅ Customer contract updated successfully")
	else:
		print(f"❌ Error: {result['message']}")
	
	# Example 3: Update product with new version
	print("\n3. Updating Product Version...")
	
	product_update = MdEntityUpdate(
		entity_name="CloudSync Pro 2024.2",
		entity_description="Enterprise cloud synchronization software - Summer 2024 Release",
		attributes={
			"version": "2024.2.0",
			"price": 3299.99,  # Price increase with new features
			"features": [
				"Real-time synchronization",
				"Multi-cloud support", 
				"Enterprise security",
				"API integration",
				"Advanced monitoring",
				"AI-powered optimization",  # New feature
				"Enhanced dashboard"        # New feature
			],
			"release_date": "2024-07-15",
			"changelog": [
				"Added AI-powered sync optimization",
				"New enterprise dashboard",
				"Performance improvements up to 40%",
				"Enhanced security protocols"
			],
			"min_memory": "12GB"  # Increased requirements
		},
		tags=["product", "software", "enterprise", "2024", "latest"]
	)
	
	context = mdm_service.create_operation_context(
		tenant_id=tenant_id,
		user_id=user_id,
		operation_type="update_entity",
		entity_id=product_id,
		source_system="product_release_update"
	)
	
	result = await mdm_service.entity_service.update_entity(product_id, product_update, context)
	
	if result["status"] == "success":
		print(f"✅ Product updated to new version")
	else:
		print(f"❌ Error: {result['message']}")


async def example_search_entities(mdm_service: MDMService):
	"""Example: Searching entities with various criteria"""
	print("\n🔎 Searching Entities Example")
	print("=" * 50)
	
	tenant_id = "demo-tenant"
	
	# Example 1: Search by entity type
	print("\n1. Search by Entity Type (People)...")
	
	search_criteria = {
		"entity_type": EntityType.PERSON,
		"limit": 10,
		"offset": 0,
		"sort_by": "created_at",
		"sort_order": "desc"
	}
	
	result = await mdm_service.entity_service.search_entities(tenant_id, search_criteria)
	
	if result["status"] == "success":
		entities = result["entities"]
		pagination = result["pagination"]
		
		print(f"✅ Found {pagination['total_count']} people")
		for entity in entities:
			print(f"   • {entity['entity_name']} ({entity['business_key']})")
			print(f"     Quality: {entity['quality_score']:.1f}% | Status: {entity['status']}")
	else:
		print(f"❌ Error: {result['message']}")
	
	# Example 2: Search by name pattern
	print("\n2. Search by Name Pattern...")
	
	search_criteria = {
		"entity_name": "Chen",  # Partial name match
		"limit": 5,
		"offset": 0
	}
	
	result = await mdm_service.entity_service.search_entities(tenant_id, search_criteria)
	
	if result["status"] == "success":
		entities = result["entities"]
		print(f"✅ Found {len(entities)} entities matching 'Chen'")
		for entity in entities:
			print(f"   • {entity['entity_name']} ({entity['entity_type']})")
	else:
		print(f"❌ Error: {result['message']}")
	
	# Example 3: Complex multi-criteria search
	print("\n3. Complex Multi-Criteria Search...")
	
	search_criteria = {
		"entity_type": EntityType.CUSTOMER,
		"source_system": "crm_system",
		"status": EntityStatus.ACTIVE,
		"min_quality_score": 80.0,
		"tags": ["enterprise"],  # Must have enterprise tag
		"limit": 20,
		"offset": 0,
		"sort_by": "quality_score",
		"sort_order": "desc"
	}
	
	result = await mdm_service.entity_service.search_entities(tenant_id, search_criteria)
	
	if result["status"] == "success":
		entities = result["entities"]
		print(f"✅ Found {len(entities)} enterprise customers from CRM with 80%+ quality")
		for entity in entities:
			print(f"   • {entity['entity_name']}")
			print(f"     Quality: {entity['quality_score']:.1f}% | Tags: {', '.join(entity['tags'])}")
	else:
		print(f"❌ Error: {result['message']}")
	
	# Example 4: Paginated search
	print("\n4. Paginated Search Example...")
	
	page_size = 2
	page_num = 1
	
	search_criteria = {
		"limit": page_size,
		"offset": (page_num - 1) * page_size,
		"sort_by": "entity_name",
		"sort_order": "asc"
	}
	
	result = await mdm_service.entity_service.search_entities(tenant_id, search_criteria)
	
	if result["status"] == "success":
		entities = result["entities"]
		pagination = result["pagination"]
		
		print(f"✅ Page {page_num} of {pagination.get('total_pages', '?')}")
		print(f"   Showing {len(entities)} of {pagination['total_count']} total entities")
		print(f"   Has next page: {pagination['has_next']}")
		print(f"   Has previous page: {pagination['has_previous']}")
		
		for entity in entities:
			print(f"   • {entity['entity_name']}")
	else:
		print(f"❌ Error: {result['message']}")


async def example_delete_entities(mdm_service: MDMService, entity_ids):
	"""Example: Deleting entities (soft delete)"""
	print("\n🗑️  Deleting Entities Example")
	print("=" * 50)
	
	tenant_id = "demo-tenant"
	user_id = "demo-user"
	person_id, customer_id, product_id = entity_ids
	
	# Note: APG MDM uses soft deletes by default for audit trail
	print("\n1. Soft Deleting Product Entity...")
	
	context = mdm_service.create_operation_context(
		tenant_id=tenant_id,
		user_id=user_id,
		operation_type="delete_entity",
		entity_id=product_id,
		source_system="product_retirement"
	)
	
	result = await mdm_service.entity_service.delete_entity(product_id, context)
	
	if result["status"] == "success":
		print(f"✅ Product soft deleted successfully")
		print(f"   Entity ID: {result['entity_id']}")
		print(f"   Deleted at: {result.get('deleted_at', 'N/A')}")
		
		# Verify the entity is now marked as deleted
		verify_result = await mdm_service.entity_service.get_entity(product_id, tenant_id)
		if verify_result["status"] == "success":
			entity = verify_result["entity"]
			print(f"   Verified status: {entity['status']} (should be 'deleted')")
	else:
		print(f"❌ Error: {result['message']}")
	
	print("\n💡 Note: APG MDM uses soft deletes to maintain audit trails.")
	print("   Deleted entities remain in the database but are marked as 'deleted'.")
	print("   They won't appear in normal searches but can be retrieved for audit purposes.")


async def main():
	"""Run all basic operations examples"""
	print("🌟 APG Master Data Management - Basic Operations Examples")
	print("=" * 70)
	
	try:
		# Initialize MDM service
		mdm_service = await setup_mdm_service()
		
		# Run examples
		entity_ids = await example_create_entities(mdm_service)
		await example_retrieve_entities(mdm_service, entity_ids)
		await example_update_entities(mdm_service, entity_ids)
		await example_search_entities(mdm_service)
		await example_delete_entities(mdm_service, entity_ids)
		
		print("\n✅ All basic operations examples completed successfully!")
		print("\n🚀 Next steps:")
		print("   • Try quality_assessment.py for data quality examples")
		print("   • Try duplicate_detection.py for entity matching examples")
		print("   • Try golden_records.py for master record examples")
		
	except Exception as e:
		print(f"\n❌ Error running examples: {str(e)}")
		raise
	finally:
		# Clean up
		if 'mdm_service' in locals():
			await mdm_service.shutdown()
			print("\n🛑 MDM service shut down")


if __name__ == "__main__":
	# Run the examples
	asyncio.run(main())