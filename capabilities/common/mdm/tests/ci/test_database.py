#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Database Testing
Unit tests for database manager and operations

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from uuid_extensions import uuid7str

from ...database import MDMDatabaseManager
from ...models import MdEntity, EntityType, EntityStatus


class TestMDMDatabaseManager:
	"""Test database manager initialization and configuration"""
	
	async def test_database_initialization(self):
		"""Test database manager initialization"""
		config = {
			"database_url": "postgresql://test:test@localhost:5432/mdm_test",
			"pool_size": 10,
			"max_overflow": 20,
			"pool_timeout": 30,
			"pool_recycle": 3600
		}
		
		db_manager = MDMDatabaseManager(config)
		await db_manager.initialize()
		
		assert db_manager.engine is not None
		assert db_manager.session_factory is not None
		assert db_manager.is_initialized is True
		
		await db_manager.close()
	
	async def test_database_health_check(self, test_db_manager):
		"""Test database health check"""
		health_result = await test_db_manager.health_check()
		
		assert health_result["status"] == "healthy"
		assert health_result["database"] == "connected"
		assert "connection_pool" in health_result
		assert "response_time_ms" in health_result
		assert health_result["response_time_ms"] < 1000  # Should be fast
	
	async def test_database_stats_generation(self, test_db_manager, test_tenant_id):
		"""Test database statistics generation"""
		# Create test data
		entities = []
		for i in range(5):
			entity = MdEntity(
				entity_id=uuid7str(),
				tenant_id=test_tenant_id,
				entity_type=EntityType.PERSON if i < 3 else EntityType.CUSTOMER,
				entity_name=f"Entity {i}",
				business_key=f"ENT-{i:03d}",
				source_system="test_system",
				status=EntityStatus.ACTIVE,
				data_classification="internal",
				quality_score=80.0 + (i * 2),  # Varying quality scores
				created_by="test_user",
				updated_by="test_user"
			)
			entities.append(entity)
		
		async with test_db_manager.get_session() as session:
			session.add_all(entities)
			await session.commit()
		
		# Get statistics
		stats = await test_db_manager.get_database_stats(test_tenant_id)
		
		assert "entity_statistics" in stats
		assert "quality_statistics" in stats
		assert "recent_activity" in stats
		assert "timestamp" in stats
		
		entity_stats = stats["entity_statistics"]
		assert len(entity_stats) > 0
		
		# Find person statistics
		person_stats = next(
			(stat for stat in entity_stats if stat["entity_type"] == "person"),
			None
		)
		assert person_stats is not None
		assert person_stats["total_entities"] == 3
		assert person_stats["avg_quality_score"] >= 80.0
	
	async def test_database_performance_monitoring(self, test_db_manager):
		"""Test database performance monitoring"""
		# Execute some operations and check performance
		start_time = datetime.utcnow()
		
		# Simulate database load
		async with test_db_manager.get_session() as session:
			for i in range(10):
				# Simple query to test response time
				result = await session.execute("SELECT 1")
				assert result.scalar() == 1
		
		end_time = datetime.utcnow()
		operation_time = (end_time - start_time).total_seconds() * 1000
		
		# Performance should be reasonable
		assert operation_time < 1000  # Less than 1 second for 10 simple queries
	
	async def test_database_connection_pooling(self, test_db_manager):
		"""Test database connection pooling behavior"""
		# Test concurrent connections
		async def test_query(session_id: int):
			async with test_db_manager.get_session() as session:
				result = await session.execute(f"SELECT {session_id}")
				return result.scalar()
		
		# Create multiple concurrent database operations
		tasks = [test_query(i) for i in range(5)]
		results = await asyncio.gather(*tasks, return_exceptions=True)

		
		# All operations should complete successfully
		assert results == list(range(5))
	
	async def test_database_transaction_isolation(self, test_db_manager, test_tenant_id):
		"""Test database transaction isolation"""
		entity_id = uuid7str()
		
		# Start two separate transactions
		async def transaction_1():
			async with test_db_manager.get_session() as session:
				entity = MdEntity(
					entity_id=entity_id,
					tenant_id=test_tenant_id,
					entity_type=EntityType.PERSON,
					entity_name="Transaction Test 1",
					business_key="TXN-001",
					source_system="test",
					status=EntityStatus.ACTIVE,
					data_classification="internal",
					created_by="txn1",
					updated_by="txn1"
				)
				session.add(entity)
				await session.commit()
				return "txn1_committed"
		
		async def transaction_2():
			async with test_db_manager.get_session() as session:
				# Try to read the entity created in transaction 1
				result = await session.execute(
					session.query(MdEntity).filter(MdEntity.entity_id == entity_id)
				)
				entity = result.scalar_one_or_none()
				return entity is not None
		
		# Execute transaction 1, then transaction 2
		result1 = await transaction_1()
		result2 = await transaction_2()
		
		assert result1 == "txn1_committed"
		assert result2 is True  # Should be able to read committed data
	
	async def test_database_error_handling(self, test_db_manager):
		"""Test database error handling"""
		# Test invalid query
		try:
			async with test_db_manager.get_session() as session:
				await session.execute("SELECT * FROM non_existent_table")
			assert False, "Should have raised an exception"
		except Exception as e:
			assert "non_existent_table" in str(e).lower()
	
	async def test_database_session_cleanup(self, test_db_manager):
		"""Test proper session cleanup"""
		# Create and use a session
		session = test_db_manager.session_factory()
		
		try:
			result = await session.execute("SELECT 1")
			assert result.scalar() == 1
		finally:
			await session.close()
		
		# Session should be closed
		assert session.is_active is False


class TestDatabaseQueries:
	"""Test database query operations"""
	
	async def test_entity_crud_operations(self, test_db_manager, test_tenant_id):
		"""Test Create, Read, Update, Delete operations"""
		entity_id = uuid7str()
		
		# CREATE
		entity = MdEntity(
			entity_id=entity_id,
			tenant_id=test_tenant_id,
			entity_type=EntityType.CUSTOMER,
			entity_name="Test Customer",
			business_key="CUST-001",
			source_system="crm_system",
			status=EntityStatus.ACTIVE,
			attributes={"industry": "Technology", "revenue": 1000000},
			data_classification="confidential",
			quality_score=85.0,
			created_by="test_user",
			updated_by="test_user"
		)
		
		async with test_db_manager.get_session() as session:
			session.add(entity)
			await session.commit()
		
		# READ
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity).filter(MdEntity.entity_id == entity_id)
			)
			retrieved_entity = result.scalar_one()
			
			assert retrieved_entity.entity_name == "Test Customer"
			assert retrieved_entity.attributes["industry"] == "Technology"
			assert retrieved_entity.quality_score == 85.0
		
		# UPDATE
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity).filter(MdEntity.entity_id == entity_id)
			)
			update_entity = result.scalar_one()
			
			update_entity.entity_name = "Updated Customer"
			update_entity.quality_score = 90.0
			update_entity.attributes["revenue"] = 1500000
			
			await session.commit()
		
		# Verify UPDATE
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity).filter(MdEntity.entity_id == entity_id)
			)
			updated_entity = result.scalar_one()
			
			assert updated_entity.entity_name == "Updated Customer"
			assert updated_entity.quality_score == 90.0
			assert updated_entity.attributes["revenue"] == 1500000
		
		# DELETE
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity).filter(MdEntity.entity_id == entity_id)
			)
			delete_entity = result.scalar_one()
			
			await session.delete(delete_entity)
			await session.commit()
		
		# Verify DELETE
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity).filter(MdEntity.entity_id == entity_id)
			)
			deleted_entity = result.scalar_one_or_none()
			
			assert deleted_entity is None
	
	async def test_complex_entity_queries(self, test_db_manager, test_tenant_id):
		"""Test complex database queries with filters and sorting"""
		# Create test entities with varying attributes
		entities = []
		for i in range(10):
			entity = MdEntity(
				entity_id=uuid7str(),
				tenant_id=test_tenant_id,
				entity_type=EntityType.PERSON if i % 2 == 0 else EntityType.CUSTOMER,
				entity_name=f"Entity {i:02d}",
				business_key=f"ENT-{i:03d}",
				source_system="test_system" if i < 5 else "other_system",
				status=EntityStatus.ACTIVE if i % 3 != 0 else EntityStatus.INACTIVE,
				quality_score=60.0 + (i * 4),  # Scores from 60 to 96
				attributes={"index": i, "category": "test"},
				data_classification="internal",
				created_by="test_user",
				updated_by="test_user"
			)
			entities.append(entity)
		
		async with test_db_manager.get_session() as session:
			session.add_all(entities)
			await session.commit()
		
		# Test filtering by entity type
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity)
				.filter(MdEntity.tenant_id == test_tenant_id)
				.filter(MdEntity.entity_type == EntityType.PERSON)
				.order_by(MdEntity.entity_name)
			)
			person_entities = result.scalars().all()
			
			assert len(person_entities) == 5  # Every even index
			assert all(e.entity_type == EntityType.PERSON for e in person_entities)
		
		# Test filtering by quality score range
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity)
				.filter(MdEntity.tenant_id == test_tenant_id)
				.filter(MdEntity.quality_score >= 80.0)
				.order_by(MdEntity.quality_score.desc())
			)
			high_quality_entities = result.scalars().all()
			
			assert len(high_quality_entities) >= 5  # Entities with scores 80+
			assert all(e.quality_score >= 80.0 for e in high_quality_entities)
		
		# Test filtering by source system and status
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity)
				.filter(MdEntity.tenant_id == test_tenant_id)
				.filter(MdEntity.source_system == "test_system")
				.filter(MdEntity.status == EntityStatus.ACTIVE)
			)
			filtered_entities = result.scalars().all()
			
			assert len(filtered_entities) >= 3
			assert all(e.source_system == "test_system" for e in filtered_entities)
			assert all(e.status == EntityStatus.ACTIVE for e in filtered_entities)
	
	async def test_database_json_queries(self, test_db_manager, test_tenant_id):
		"""Test JSON attribute queries"""
		# Create entities with complex JSON attributes
		entities = []
		for i in range(3):
			attributes = {
				"personal_info": {
					"age": 25 + i * 5,
					"location": {
						"city": "New York" if i == 0 else ("Boston" if i == 1 else "Chicago"),
						"state": "NY" if i == 0 else ("MA" if i == 1 else "IL")
					}
				},
				"preferences": {
					"communication": ["email"] if i == 0 else ["phone", "email"],
					"language": "en"
				},
				"scores": {
					"satisfaction": 8 + i,
					"engagement": 7 + i * 0.5
				}
			}
			
			entity = MdEntity(
				entity_id=uuid7str(),
				tenant_id=test_tenant_id,
				entity_type=EntityType.PERSON,
				entity_name=f"Person {i}",
				business_key=f"PERSON-{i:03d}",
				source_system="test_system",
				status=EntityStatus.ACTIVE,
				attributes=attributes,
				data_classification="internal",
				created_by="test_user",
				updated_by="test_user"
			)
			entities.append(entity)
		
		async with test_db_manager.get_session() as session:
			session.add_all(entities)
			await session.commit()
		
		# Test JSON path queries (PostgreSQL specific)
		async with test_db_manager.get_session() as session:
			# Find entities with age > 30
			result = await session.execute(
				session.query(MdEntity)
				.filter(MdEntity.tenant_id == test_tenant_id)
				.filter(MdEntity.attributes["personal_info"]["age"].astext.cast(Integer) > 30)
			)
			older_entities = result.scalars().all()
			
			assert len(older_entities) >= 1
			for entity in older_entities:
				age = entity.attributes["personal_info"]["age"]
				assert age > 30
		
		# Test JSON array contains queries
		async with test_db_manager.get_session() as session:
			# Find entities with phone in communication preferences
			result = await session.execute(
				session.query(MdEntity)
				.filter(MdEntity.tenant_id == test_tenant_id)
				.filter(MdEntity.attributes["preferences"]["communication"].op("?")("phone"))
			)
			phone_entities = result.scalars().all()
			
			assert len(phone_entities) >= 1
			for entity in phone_entities:
				comm_prefs = entity.attributes["preferences"]["communication"]
				assert "phone" in comm_prefs
	
	async def test_database_pagination(self, test_db_manager, test_tenant_id):
		"""Test database pagination queries"""
		# Create 25 test entities
		entities = []
		for i in range(25):
			entity = MdEntity(
				entity_id=uuid7str(),
				tenant_id=test_tenant_id,
				entity_type=EntityType.PRODUCT,
				entity_name=f"Product {i:02d}",
				business_key=f"PROD-{i:03d}",
				source_system="catalog_system",
				status=EntityStatus.ACTIVE,
				data_classification="public",
				created_by="test_user",
				updated_by="test_user"
			)
			entities.append(entity)
		
		async with test_db_manager.get_session() as session:
			session.add_all(entities)
			await session.commit()
		
		# Test first page
		page_size = 10
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity)
				.filter(MdEntity.tenant_id == test_tenant_id)
				.order_by(MdEntity.entity_name)
				.limit(page_size)
				.offset(0)
			)
			page_1_entities = result.scalars().all()
			
			assert len(page_1_entities) == page_size
			assert page_1_entities[0].entity_name == "Product 00"
		
		# Test second page
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity)
				.filter(MdEntity.tenant_id == test_tenant_id)
				.order_by(MdEntity.entity_name)
				.limit(page_size)
				.offset(page_size)
			)
			page_2_entities = result.scalars().all()
			
			assert len(page_2_entities) == page_size
			assert page_2_entities[0].entity_name == "Product 10"
		
		# Test final page
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntity)
				.filter(MdEntity.tenant_id == test_tenant_id)
				.order_by(MdEntity.entity_name)
				.limit(page_size)
				.offset(2 * page_size)
			)
			page_3_entities = result.scalars().all()
			
			assert len(page_3_entities) == 5  # Remaining entities
			assert page_3_entities[0].entity_name == "Product 20"


class TestDatabaseConstraints:
	"""Test database constraints and data integrity"""
	
	async def test_unique_constraints(self, test_db_manager, test_tenant_id):
		"""Test unique constraints enforcement"""
		entity_1 = MdEntity(
			entity_id=uuid7str(),
			tenant_id=test_tenant_id,
			entity_type=EntityType.PERSON,
			entity_name="Unique Test",
			business_key="UNIQUE-001",
			source_system="test_system",
			status=EntityStatus.ACTIVE,
			data_classification="internal",
			created_by="test_user",
			updated_by="test_user"
		)
		
		entity_2 = MdEntity(
			entity_id=uuid7str(),  # Different entity_id
			tenant_id=test_tenant_id,  # Same tenant
			entity_type=EntityType.PERSON,
			entity_name="Unique Test 2",
			business_key="UNIQUE-001",  # Same business_key
			source_system="test_system",  # Same source_system
			status=EntityStatus.ACTIVE,
			data_classification="internal",
			created_by="test_user",
			updated_by="test_user"
		)
		
		async with test_db_manager.get_session() as session:
			session.add(entity_1)
			await session.commit()
			
			# Should fail due to unique constraint on (tenant_id, business_key, source_system)
			try:
				session.add(entity_2)
				await session.commit()
				assert False, "Should have failed unique constraint"
			except Exception as e:
				await session.rollback()
				assert "unique" in str(e).lower() or "duplicate" in str(e).lower()
	
	async def test_foreign_key_constraints(self, test_db_manager, test_tenant_id):
		"""Test foreign key constraint enforcement"""
		from ...models import MdEntityVersion
		
		# Try to create version without parent entity
		orphan_version = MdEntityVersion(
			version_id=uuid7str(),
			entity_id=uuid7str(),  # Non-existent entity
			tenant_id=test_tenant_id,
			version_number=1,
			version_type="create",
			change_description="Orphan version",
			created_by="test_user"
		)
		
		async with test_db_manager.get_session() as session:
			try:
				session.add(orphan_version)
				await session.commit()
				assert False, "Should have failed foreign key constraint"
			except Exception as e:
				await session.rollback()
				assert "foreign key" in str(e).lower() or "constraint" in str(e).lower()
	
	async def test_not_null_constraints(self, test_db_manager, test_tenant_id):
		"""Test NOT NULL constraints"""
		# Try to create entity without required fields
		try:
			incomplete_entity = MdEntity(
				entity_id=uuid7str(),
				tenant_id=test_tenant_id,
				# Missing entity_type (required)
				entity_name="Incomplete Entity",
				business_key="INC-001",
				source_system="test_system",
				status=EntityStatus.ACTIVE,
				data_classification="internal",
				created_by="test_user",
				updated_by="test_user"
			)
			
			async with test_db_manager.get_session() as session:
				session.add(incomplete_entity)
				await session.commit()
				
			assert False, "Should have failed NOT NULL constraint"
		except Exception as e:
			assert "not null" in str(e).lower() or "null value" in str(e).lower()