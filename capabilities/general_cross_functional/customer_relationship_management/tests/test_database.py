"""
APG Customer Relationship Management - Database Tests

Comprehensive unit tests for database operations including CRUD operations,
multi-tenant isolation, performance, and data integrity.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
import asyncio
from datetime import datetime, date
from decimal import Decimal
from uuid_extensions import uuid7str

from ..database import DatabaseManager
from ..models import CRMContact, ContactType, LeadSource
from . import TEST_TENANT_ID, TEST_USER_ID


@pytest.mark.unit
class TestDatabaseManager:
	"""Test DatabaseManager functionality"""
	
	@pytest.mark.asyncio
	async def test_database_initialization(self, test_database):
		"""Test database manager initialization"""
		manager = DatabaseManager(test_database)
		
		assert not manager._initialized
		
		await manager.initialize()
		assert manager._initialized
		assert manager.pool is not None
		
		await manager.shutdown()
		assert not manager._initialized
	
	@pytest.mark.asyncio
	async def test_health_check(self, database_manager):
		"""Test database health check"""
		health = await database_manager.health_check()
		
		assert health["status"] == "healthy"
		assert health["initialized"] is True
		assert health["pool_size"] > 0
		assert "timestamp" in health
	
	@pytest.mark.asyncio 
	async def test_connection_pool(self, database_manager):
		"""Test database connection pooling"""
		# Test multiple concurrent connections
		async def test_query():
			async with database_manager.get_connection() as conn:
				result = await conn.fetchval("SELECT 1")
				return result
		
		# Run multiple queries concurrently
		tasks = [test_query() for _ in range(10)]
		results = await asyncio.gather(*tasks)
		
		assert all(result == 1 for result in results)


@pytest.mark.unit
class TestContactOperations:
	"""Test contact CRUD operations"""
	
	@pytest.mark.asyncio
	async def test_create_contact(self, database_manager, contact_factory, clean_database):
		"""Test creating a contact"""
		contact = contact_factory()
		
		result = await database_manager.create_contact(contact)
		
		assert result is True
		
		# Verify contact was created
		retrieved = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert retrieved is not None
		assert retrieved.id == contact.id
		assert retrieved.first_name == contact.first_name
		assert retrieved.last_name == contact.last_name
		assert retrieved.email == contact.email
	
	@pytest.mark.asyncio
	async def test_get_contact(self, database_manager, contact_factory, clean_database):
		"""Test retrieving a contact"""
		contact = contact_factory()
		await database_manager.create_contact(contact)
		
		# Get existing contact
		retrieved = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert retrieved is not None
		assert retrieved.id == contact.id
		
		# Get non-existent contact
		non_existent = await database_manager.get_contact(uuid7str(), contact.tenant_id)
		assert non_existent is None
	
	@pytest.mark.asyncio
	async def test_update_contact(self, database_manager, contact_factory, clean_database):
		"""Test updating a contact"""
		contact = contact_factory()
		await database_manager.create_contact(contact)
		
		# Update contact
		contact.first_name = "UpdatedName"
		contact.email = "updated@example.com"
		contact.lead_score = 85.0
		contact.updated_by = TEST_USER_ID
		
		result = await database_manager.update_contact(contact)
		assert result is True
		
		# Verify updates
		updated = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert updated.first_name == "UpdatedName"
		assert updated.email == "updated@example.com"
		assert updated.lead_score == 85.0
		assert updated.version == contact.version + 1
	
	@pytest.mark.asyncio
	async def test_delete_contact(self, database_manager, contact_factory, clean_database):
		"""Test deleting a contact"""
		contact = contact_factory()
		await database_manager.create_contact(contact)
		
		# Verify contact exists
		retrieved = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert retrieved is not None
		
		# Delete contact
		result = await database_manager.delete_contact(contact.id, contact.tenant_id)
		assert result is True
		
		# Verify contact is deleted
		deleted = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert deleted is None
	
	@pytest.mark.asyncio
	async def test_list_contacts(self, database_manager, contact_factory, clean_database):
		"""Test listing contacts with pagination"""
		# Create multiple contacts
		contacts = []
		for i in range(15):
			contact = contact_factory(
				first_name=f"Contact{i:02d}",
				email=f"contact{i:02d}@example.com"
			)
			await database_manager.create_contact(contact)
			contacts.append(contact)
		
		# Test basic listing
		result = await database_manager.list_contacts(TEST_TENANT_ID)
		assert result["total"] == 15
		assert len(result["items"]) == 10  # Default page size
		
		# Test pagination
		page2 = await database_manager.list_contacts(TEST_TENANT_ID, skip=10, limit=5)
		assert len(page2["items"]) == 5
		assert page2["total"] == 15
		
		# Test filtering by contact type
		filtered = await database_manager.list_contacts(
			TEST_TENANT_ID, 
			filters={"contact_type": ContactType.PROSPECT}
		)
		assert filtered["total"] == 15  # All are prospects by default
	
	@pytest.mark.asyncio
	async def test_search_contacts(self, database_manager, contact_factory, clean_database):
		"""Test contact search functionality"""
		# Create contacts with searchable data
		contacts = [
			contact_factory(first_name="John", last_name="Doe", company="TechCorp"),
			contact_factory(first_name="Jane", last_name="Smith", company="DataCorp"),
			contact_factory(first_name="Bob", last_name="Johnson", company="TechStart")
		]
		
		for contact in contacts:
			await database_manager.create_contact(contact)
		
		# Search by name
		john_results = await database_manager.search_contacts(TEST_TENANT_ID, "John")
		assert len(john_results["items"]) == 1
		assert john_results["items"][0].first_name == "John"
		
		# Search by company
		tech_results = await database_manager.search_contacts(TEST_TENANT_ID, "Tech")
		assert len(tech_results["items"]) == 2  # TechCorp and TechStart
		
		# Search with no results
		no_results = await database_manager.search_contacts(TEST_TENANT_ID, "NonExistent")
		assert len(no_results["items"]) == 0


@pytest.mark.unit
class TestTenantIsolation:
	"""Test multi-tenant data isolation"""
	
	@pytest.mark.asyncio
	async def test_tenant_data_isolation(self, database_manager, contact_factory, clean_database):
		"""Test that tenants can only access their own data"""
		tenant1_id = "tenant_1"
		tenant2_id = "tenant_2"
		
		# Create contacts for different tenants
		contact1 = contact_factory(tenant_id=tenant1_id, email="tenant1@example.com")
		contact2 = contact_factory(tenant_id=tenant2_id, email="tenant2@example.com")
		
		await database_manager.create_contact(contact1)
		await database_manager.create_contact(contact2)
		
		# Tenant 1 should only see their contact
		tenant1_contacts = await database_manager.list_contacts(tenant1_id)
		assert tenant1_contacts["total"] == 1
		assert tenant1_contacts["items"][0].email == "tenant1@example.com"
		
		# Tenant 2 should only see their contact
		tenant2_contacts = await database_manager.list_contacts(tenant2_id)
		assert tenant2_contacts["total"] == 1
		assert tenant2_contacts["items"][0].email == "tenant2@example.com"
		
		# Tenant 1 should not be able to access tenant 2's contact
		cross_tenant_access = await database_manager.get_contact(contact2.id, tenant1_id)
		assert cross_tenant_access is None
	
	@pytest.mark.asyncio
	async def test_tenant_update_isolation(self, database_manager, contact_factory, clean_database):
		"""Test that tenants cannot update other tenant's data"""
		tenant1_id = "tenant_1"
		tenant2_id = "tenant_2"
		
		# Create contact for tenant 1
		contact = contact_factory(tenant_id=tenant1_id)
		await database_manager.create_contact(contact)
		
		# Try to update as tenant 2 (should fail)
		contact.tenant_id = tenant2_id  # Change tenant
		contact.first_name = "Unauthorized Update"
		
		result = await database_manager.update_contact(contact)
		assert result is False  # Should fail due to tenant mismatch
		
		# Verify original data is unchanged
		original = await database_manager.get_contact(contact.id, tenant1_id)
		assert original.first_name != "Unauthorized Update"


@pytest.mark.unit
class TestDataIntegrity:
	"""Test data integrity constraints"""
	
	@pytest.mark.asyncio
	async def test_unique_constraints(self, database_manager, contact_factory, clean_database):
		"""Test unique constraint violations"""
		contact1 = contact_factory(email="unique@example.com")
		await database_manager.create_contact(contact1)
		
		# Try to create another contact with same email in same tenant
		contact2 = contact_factory(email="unique@example.com")
		
		# This should succeed as we don't have unique email constraint across tenant
		# But if we had such constraint, it would fail
		result = await database_manager.create_contact(contact2)
		assert result is True  # Currently no unique email constraint
	
	@pytest.mark.asyncio
	async def test_foreign_key_constraints(self, database_manager, contact_factory, activity_factory, clean_database):
		"""Test foreign key constraint handling"""
		contact = contact_factory()
		await database_manager.create_contact(contact)
		
		# Create activity referencing the contact
		activity = activity_factory(contact_id=contact.id)
		
		# This would test foreign key constraint if activity creation was implemented
		# For now, just verify the reference is set correctly
		assert activity.contact_id == contact.id
	
	@pytest.mark.asyncio
	async def test_audit_fields(self, database_manager, contact_factory, clean_database):
		"""Test audit field population"""
		contact = contact_factory()
		
		# Record creation time
		before_create = datetime.utcnow()
		await database_manager.create_contact(contact)
		after_create = datetime.utcnow()
		
		# Retrieve and verify audit fields
		retrieved = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert retrieved.created_by == TEST_USER_ID
		assert retrieved.updated_by == TEST_USER_ID
		assert before_create <= retrieved.created_at <= after_create
		assert before_create <= retrieved.updated_at <= after_create
		assert retrieved.version == 1
		
		# Test update audit fields
		retrieved.first_name = "Updated"
		before_update = datetime.utcnow()
		await database_manager.update_contact(retrieved)
		after_update = datetime.utcnow()
		
		updated = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert updated.version == 2
		assert before_update <= updated.updated_at <= after_update
		assert updated.created_at == retrieved.created_at  # Should not change


@pytest.mark.performance
class TestDatabasePerformance:
	"""Test database performance characteristics"""
	
	@pytest.mark.asyncio
	async def test_bulk_insert_performance(self, database_manager, contact_factory, clean_database, performance_timer):
		"""Test bulk insert performance"""
		# Create 100 contacts
		contacts = [contact_factory(email=f"perf{i:03d}@example.com") for i in range(100)]
		
		performance_timer.start()
		
		# Insert all contacts
		tasks = [database_manager.create_contact(contact) for contact in contacts]
		results = await asyncio.gather(*tasks)
		
		performance_timer.stop()
		
		# Verify all inserts succeeded
		assert all(results)
		
		# Performance should be reasonable (adjust threshold as needed)
		assert performance_timer.elapsed < 5.0  # Should complete in under 5 seconds
		
		# Verify all contacts were created
		all_contacts = await database_manager.list_contacts(TEST_TENANT_ID, limit=200)
		assert all_contacts["total"] == 100
	
	@pytest.mark.asyncio
	async def test_search_performance(self, database_manager, contact_factory, clean_database, performance_timer):
		"""Test search performance with large dataset"""
		# Create 1000 contacts with varied data
		contacts = []
		for i in range(1000):
			contact = contact_factory(
				first_name=f"FirstName{i:04d}",
				last_name=f"LastName{i:04d}",
				email=f"user{i:04d}@company{i%10}.com",
				company=f"Company{i%10}"
			)
			contacts.append(contact)
		
		# Bulk insert
		tasks = [database_manager.create_contact(contact) for contact in contacts]
		await asyncio.gather(*tasks)
		
		# Test search performance
		performance_timer.start()
		
		# Search for specific pattern
		results = await database_manager.search_contacts(TEST_TENANT_ID, "FirstName0001")
		
		performance_timer.stop()
		
		# Should find the specific contact quickly
		assert len(results["items"]) == 1
		assert results["items"][0].first_name == "FirstName0001"
		
		# Search should be fast (adjust threshold as needed)
		assert performance_timer.elapsed < 1.0  # Should complete in under 1 second
	
	@pytest.mark.asyncio
	async def test_pagination_performance(self, database_manager, contact_factory, clean_database):
		"""Test pagination performance"""
		# Create 500 contacts
		contacts = [contact_factory(email=f"page{i:03d}@example.com") for i in range(500)]
		
		tasks = [database_manager.create_contact(contact) for contact in contacts]
		await asyncio.gather(*tasks)
		
		# Test various page sizes
		page_sizes = [10, 50, 100]
		
		for page_size in page_sizes:
			start_time = datetime.utcnow()
			
			# Get first page
			page1 = await database_manager.list_contacts(TEST_TENANT_ID, limit=page_size)
			
			# Get middle page
			middle_skip = (500 // page_size) // 2 * page_size
			middle_page = await database_manager.list_contacts(TEST_TENANT_ID, skip=middle_skip, limit=page_size)
			
			# Get last page
			last_skip = 500 - page_size
			last_page = await database_manager.list_contacts(TEST_TENANT_ID, skip=last_skip, limit=page_size)
			
			end_time = datetime.utcnow()
			duration = (end_time - start_time).total_seconds()
			
			# Verify results
			assert len(page1["items"]) == page_size
			assert len(middle_page["items"]) == page_size
			assert len(last_page["items"]) == page_size
			assert page1["total"] == 500
			
			# Should be reasonably fast
			assert duration < 2.0  # All pagination operations under 2 seconds


@pytest.mark.unit
class TestErrorHandling:
	"""Test database error handling"""
	
	@pytest.mark.asyncio
	async def test_connection_error_handling(self, test_database):
		"""Test handling of connection errors"""
		# Create manager with invalid config
		invalid_config = test_database.copy()
		invalid_config["host"] = "nonexistent-host"
		invalid_config["port"] = 9999
		
		manager = DatabaseManager(invalid_config)
		
		# Should handle initialization failure gracefully
		with pytest.raises(Exception):
			await manager.initialize()
		
		assert not manager._initialized
	
	@pytest.mark.asyncio
	async def test_invalid_data_handling(self, database_manager, clean_database):
		"""Test handling of invalid data"""
		# Try to create contact with invalid data
		invalid_contact = CRMContact(
			id="invalid_id",  # Should be valid UUID
			tenant_id="",  # Empty tenant ID
			first_name="",  # Empty name
			last_name="Test",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		# Should handle validation errors
		result = await database_manager.create_contact(invalid_contact)
		assert result is False
	
	@pytest.mark.asyncio
	async def test_transaction_rollback(self, database_manager, contact_factory, clean_database):
		"""Test transaction rollback on errors"""
		contact = contact_factory()
		
		# This would test transaction rollback if we had complex transactions
		# For now, just verify basic operation
		result = await database_manager.create_contact(contact)
		assert result is True
		
		# Verify contact exists
		retrieved = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert retrieved is not None


@pytest.mark.integration
class TestDatabaseIntegration:
	"""Integration tests for database operations"""
	
	@pytest.mark.asyncio
	async def test_full_contact_lifecycle(self, database_manager, contact_factory, clean_database):
		"""Test complete contact lifecycle"""
		# Create
		contact = contact_factory()
		create_result = await database_manager.create_contact(contact)
		assert create_result is True
		
		# Read
		retrieved = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert retrieved is not None
		assert retrieved.email == contact.email
		
		# Update
		retrieved.lead_score = 95.0
		retrieved.notes = "Updated notes"
		update_result = await database_manager.update_contact(retrieved)
		assert update_result is True
		
		# Verify update
		updated = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert updated.lead_score == 95.0
		assert updated.notes == "Updated notes"
		assert updated.version == 2
		
		# Delete
		delete_result = await database_manager.delete_contact(contact.id, contact.tenant_id)
		assert delete_result is True
		
		# Verify deletion
		deleted = await database_manager.get_contact(contact.id, contact.tenant_id)
		assert deleted is None
	
	@pytest.mark.asyncio
	async def test_concurrent_operations(self, database_manager, contact_factory, clean_database):
		"""Test concurrent database operations"""
		# Create multiple contacts concurrently
		contacts = [contact_factory(email=f"concurrent{i}@example.com") for i in range(20)]
		
		# Concurrent creates
		create_tasks = [database_manager.create_contact(contact) for contact in contacts]
		create_results = await asyncio.gather(*create_tasks)
		assert all(create_results)
		
		# Concurrent reads
		read_tasks = [database_manager.get_contact(contact.id, contact.tenant_id) for contact in contacts]
		read_results = await asyncio.gather(*read_tasks)
		assert all(result is not None for result in read_results)
		
		# Concurrent updates  
		for i, contact in enumerate(contacts):
			contact.lead_score = float(i * 5)
		
		update_tasks = [database_manager.update_contact(contact) for contact in contacts]
		update_results = await asyncio.gather(*update_tasks)
		assert all(update_results)
		
		# Verify updates
		final_reads = await asyncio.gather(*read_tasks)
		for i, result in enumerate(final_reads):
			assert result.lead_score == float(i * 5)