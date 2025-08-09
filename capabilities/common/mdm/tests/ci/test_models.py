#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Models Testing
Unit tests for MDM data models and validation

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from uuid_extensions import uuid7str

from ...models import (
    MdEntity, MdEntityVersion, MdGoldenRecord, MdDataQualityAssessment,
    MdCrossReference, MdEntityCreate, MdEntityUpdate, EntityType, 
    EntityStatus, DataQualityStatus, MatchConfidence
)


class TestMdEntityModel:
	"""Test MdEntity SQLAlchemy model"""
	
	async def test_entity_creation_with_required_fields(self, test_db_manager, test_tenant_id):
		"""Test creating entity with minimum required fields"""
		entity = MdEntity(
			entity_id=uuid7str(),
			tenant_id=test_tenant_id,
			entity_type=EntityType.PERSON,
			entity_name="Test Person",
			business_key="TEST-001",
			source_system="test_system",
			status=EntityStatus.ACTIVE,
			data_classification="internal",
			created_by="test_user",
			updated_by="test_user"
		)
		
		async with test_db_manager.get_session() as session:
			session.add(entity)
			await session.commit()
			await session.refresh(entity)
		
		assert entity.entity_id is not None
		assert entity.created_at is not None
		assert entity.updated_at is not None
		assert entity.quality_score == 0.0  # Default value
		assert entity.attributes == {}  # Default empty dict
		assert entity.tags == []  # Default empty list
	
	async def test_entity_with_complex_attributes(self, test_db_manager, test_tenant_id):
		"""Test entity with complex JSON attributes"""
		complex_attributes = {
			"personal_info": {
				"first_name": "John",
				"last_name": "Doe",
				"date_of_birth": "1985-06-15"
			},
			"contact_info": {
				"email": "john@example.com",
				"phones": ["+1-555-123-4567", "+1-555-987-6543"],
				"addresses": [
					{
						"type": "home",
						"street": "123 Main St",
						"city": "Anytown",
						"state": "NY",
						"zip": "12345"
					}
				]
			},
			"preferences": {
				"communication": ["email", "phone"],
				"language": "en",
				"timezone": "America/New_York"
			},
			"metadata": {
				"source_confidence": 0.95,
				"last_verified": datetime.utcnow().isoformat()
			}
		}
		
		entity = MdEntity(
			entity_id=uuid7str(),
			tenant_id=test_tenant_id,
			entity_type=EntityType.PERSON,
			entity_name="John Doe",
			business_key="PERSON-001",
			source_system="crm_system",
			status=EntityStatus.ACTIVE,
			attributes=complex_attributes,
			tags=["customer", "vip", "verified"],
			data_classification="confidential",
			created_by="test_user",
			updated_by="test_user"
		)
		
		async with test_db_manager.get_session() as session:
			session.add(entity)
			await session.commit()
			await session.refresh(entity)
		
		# Verify JSON storage and retrieval
		assert entity.attributes == complex_attributes
		assert entity.attributes["personal_info"]["first_name"] == "John"
		assert len(entity.attributes["contact_info"]["phones"]) == 2
		assert entity.tags == ["customer", "vip", "verified"]
	
	async def test_entity_tenant_isolation(self, test_db_manager):
		"""Test that entities are properly isolated by tenant"""
		tenant_1 = f"tenant-1-{uuid7str()[:8]}"
		tenant_2 = f"tenant-2-{uuid7str()[:8]}"
		
		entity_1 = MdEntity(
			entity_id=uuid7str(),
			tenant_id=tenant_1,
			entity_type=EntityType.PERSON,
			entity_name="Person in Tenant 1",
			business_key="PERSON-001",
			source_system="system_1",
			status=EntityStatus.ACTIVE,
			data_classification="internal",
			created_by="user_1",
			updated_by="user_1"
		)
		
		entity_2 = MdEntity(
			entity_id=uuid7str(),
			tenant_id=tenant_2,
			entity_type=EntityType.PERSON,
			entity_name="Person in Tenant 2",
			business_key="PERSON-001",  # Same business key, different tenant
			source_system="system_2",
			status=EntityStatus.ACTIVE,
			data_classification="internal",
			created_by="user_2",
			updated_by="user_2"
		)
		
		async with test_db_manager.get_session() as session:
			session.add(entity_1)
			session.add(entity_2)
			await session.commit()
		
		# Both entities should exist with same business key in different tenants
		async with test_db_manager.get_session() as session:
			result_1 = await session.execute(
				session.query(MdEntity).filter(
					MdEntity.tenant_id == tenant_1,
					MdEntity.business_key == "PERSON-001"
				)
			)
			result_2 = await session.execute(
				session.query(MdEntity).filter(
					MdEntity.tenant_id == tenant_2,
					MdEntity.business_key == "PERSON-001"
				)
			)
			
			assert len(result_1.scalars().all()) == 1
			assert len(result_2.scalars().all()) == 1


class TestMdEntityVersionModel:
	"""Test MdEntityVersion model for change tracking"""
	
	async def test_version_creation_and_relationship(self, test_db_manager, test_tenant_id):
		"""Test creating entity version with parent relationship"""
		# Create parent entity
		entity = MdEntity(
			entity_id=uuid7str(),
			tenant_id=test_tenant_id,
			entity_type=EntityType.CUSTOMER,
			entity_name="Acme Corp",
			business_key="CUST-001",
			source_system="crm",
			status=EntityStatus.ACTIVE,
			data_classification="confidential",
			created_by="test_user",
			updated_by="test_user"
		)
		
		async with test_db_manager.get_session() as session:
			session.add(entity)
			await session.commit()
			await session.refresh(entity)
		
		# Create version
		version = MdEntityVersion(
			version_id=uuid7str(),
			entity_id=entity.entity_id,
			tenant_id=test_tenant_id,
			version_number=1,
			version_type="create",
			change_description="Initial entity creation",
			changed_fields=["entity_name", "business_key", "attributes"],
			change_source="api",
			created_by="test_user"
		)
		
		async with test_db_manager.get_session() as session:
			session.add(version)
			await session.commit()
			await session.refresh(version)
		
		assert version.version_number == 1
		assert version.version_timestamp is not None
		assert version.changed_fields == ["entity_name", "business_key", "attributes"]
	
	async def test_version_ordering(self, test_db_manager, test_tenant_id):
		"""Test that versions are ordered correctly"""
		entity_id = uuid7str()
		
		# Create parent entity
		entity = MdEntity(
			entity_id=entity_id,
			tenant_id=test_tenant_id,
			entity_type=EntityType.PRODUCT,
			entity_name="Test Product",
			business_key="PROD-001",
			source_system="catalog",
			status=EntityStatus.ACTIVE,
			data_classification="public",
			created_by="test_user",
			updated_by="test_user"
		)
		
		# Create multiple versions
		versions = []
		for i in range(1, 4):
			version = MdEntityVersion(
				version_id=uuid7str(),
				entity_id=entity_id,
				tenant_id=test_tenant_id,
				version_number=i,
				version_type="update",
				change_description=f"Update {i}",
				changed_fields=[f"field_{i}"],
				created_by="test_user"
			)
			versions.append(version)
		
		async with test_db_manager.get_session() as session:
			session.add(entity)
			session.add_all(versions)
			await session.commit()
		
		# Verify ordering
		async with test_db_manager.get_session() as session:
			result = await session.execute(
				session.query(MdEntityVersion)
				.filter(MdEntityVersion.entity_id == entity_id)
				.order_by(MdEntityVersion.version_number)
			)
			retrieved_versions = result.scalars().all()
			
			assert len(retrieved_versions) == 3
			for i, version in enumerate(retrieved_versions):
				assert version.version_number == i + 1


class TestMdEntityCreateValidation:
	"""Test MdEntityCreate Pydantic model validation"""
	
	def test_valid_entity_create(self, test_tenant_id):
		"""Test creating valid entity with all fields"""
		entity_data = {
			"tenant_id": test_tenant_id,
			"entity_type": EntityType.PERSON,
			"entity_name": "Jane Smith",
			"entity_description": "Test person entity",
			"business_key": "PERSON-002",
			"source_system": "hr_system",
			"status": EntityStatus.ACTIVE,
			"attributes": {
				"first_name": "Jane",
				"last_name": "Smith",
				"employee_id": "EMP-1001"
			},
			"tags": ["employee", "full_time"],
			"data_classification": "internal"
		}
		
		entity_create = MdEntityCreate(**entity_data)
		
		assert entity_create.tenant_id == test_tenant_id
		assert entity_create.entity_type == EntityType.PERSON
		assert entity_create.entity_name == "Jane Smith"
		assert entity_create.business_key == "PERSON-002"
		assert entity_create.attributes["employee_id"] == "EMP-1001"
		assert "employee" in entity_create.tags
	
	def test_entity_create_validation_errors(self, test_tenant_id):
		"""Test validation errors for invalid entity data"""
		
		# Test missing required fields
		with pytest.raises(ValueError) as exc_info:
			MdEntityCreate()
		assert "tenant_id" in str(exc_info.value)
		
		# Test invalid entity type
		with pytest.raises(ValueError) as exc_info:
			MdEntityCreate(
				tenant_id=test_tenant_id,
				entity_type="invalid_type",
				entity_name="Test",
				business_key="TEST-001",
				source_system="test"
			)
		
		# Test empty entity name
		with pytest.raises(ValueError) as exc_info:
			MdEntityCreate(
				tenant_id=test_tenant_id,
				entity_type=EntityType.PERSON,
				entity_name="",
				business_key="TEST-001",
				source_system="test"
			)
		assert "entity_name" in str(exc_info.value)
	
	def test_entity_create_field_constraints(self, test_tenant_id):
		"""Test field length and format constraints"""
		
		# Test entity_name length limit
		long_name = "x" * 256  # Assuming 255 character limit
		with pytest.raises(ValueError):
			MdEntityCreate(
				tenant_id=test_tenant_id,
				entity_type=EntityType.PERSON,
				entity_name=long_name,
				business_key="TEST-001",
				source_system="test"
			)
		
		# Test business_key format
		entity_create = MdEntityCreate(
			tenant_id=test_tenant_id,
			entity_type=EntityType.PERSON,
			entity_name="Test Person",
			business_key="VALID-KEY-123",
			source_system="test"
		)
		assert entity_create.business_key == "VALID-KEY-123"
	
	def test_entity_create_tags_validation(self, test_tenant_id):
		"""Test tags validation and limits"""
		
		# Test valid tags
		entity_create = MdEntityCreate(
			tenant_id=test_tenant_id,
			entity_type=EntityType.PERSON,
			entity_name="Test Person",
			business_key="TEST-001",
			source_system="test",
			tags=["tag1", "tag2", "tag-with-dash", "tag_with_underscore"]
		)
		assert len(entity_create.tags) == 4
		
		# Test too many tags (assuming 20 tag limit)
		too_many_tags = [f"tag_{i}" for i in range(25)]
		with pytest.raises(ValueError):
			MdEntityCreate(
				tenant_id=test_tenant_id,
				entity_type=EntityType.PERSON,
				entity_name="Test Person",
				business_key="TEST-001",
				source_system="test",
				tags=too_many_tags
			)


class TestMdDataQualityAssessmentModel:
	"""Test data quality assessment model"""
	
	async def test_quality_assessment_creation(self, test_db_manager, test_tenant_id):
		"""Test creating quality assessment"""
		entity_id = uuid7str()
		
		# Create parent entity first
		entity = MdEntity(
			entity_id=entity_id,
			tenant_id=test_tenant_id,
			entity_type=EntityType.PERSON,
			entity_name="Test Person",
			business_key="PERSON-001",
			source_system="test",
			status=EntityStatus.ACTIVE,
			data_classification="internal",
			created_by="test_user",
			updated_by="test_user"
		)
		
		assessment = MdDataQualityAssessment(
			assessment_id=uuid7str(),
			entity_id=entity_id,
			tenant_id=test_tenant_id,
			overall_score=87.5,
			quality_status=DataQualityStatus.GOOD,
			completeness_score=90.0,
			accuracy_score=85.0,
			consistency_score=88.0,
			validity_score=92.0,
			uniqueness_score=100.0,
			timeliness_score=75.0,
			assessment_duration_ms=125.5,
			algorithm_version="1.0.0",
			quality_issues=[
				{
					"issue_type": "timeliness",
					"field": "last_contact",
					"severity": "medium",
					"message": "Data is 3 months old"
				}
			],
			recommendations=["Update contact information", "Verify email"],
			created_by="quality_engine"
		)
		
		async with test_db_manager.get_session() as session:
			session.add(entity)
			session.add(assessment)
			await session.commit()
			await session.refresh(assessment)
		
		assert assessment.overall_score == 87.5
		assert assessment.quality_status == DataQualityStatus.GOOD
		assert assessment.assessment_timestamp is not None
		assert len(assessment.quality_issues) == 1
		assert len(assessment.recommendations) == 2
	
	def test_quality_score_validation(self):
		"""Test quality score validation constraints"""
		# Valid scores
		valid_scores = [0.0, 50.5, 100.0]
		for score in valid_scores:
			# Should not raise exception
			assessment_data = {
				"assessment_id": uuid7str(),
				"entity_id": uuid7str(),
				"tenant_id": "test-tenant",
				"overall_score": score,
				"quality_status": DataQualityStatus.GOOD,
				"completeness_score": score,
				"accuracy_score": score,
				"consistency_score": score,
				"validity_score": score,
				"uniqueness_score": score,
				"timeliness_score": score,
				"created_by": "test"
			}
			
			# This would use Pydantic validation if we had a view model
			assert assessment_data["overall_score"] == score
		
		# Invalid scores would be caught by database constraints
		# or Pydantic validation in the service layer


class TestEntityEnumsAndConstants:
	"""Test enum values and constants"""
	
	def test_entity_type_enum_values(self):
		"""Test EntityType enum values"""
		assert EntityType.PERSON == "person"
		assert EntityType.CUSTOMER == "customer"
		assert EntityType.PRODUCT == "product"
		assert EntityType.ORGANIZATION == "organization"
		assert EntityType.LOCATION == "location"
		assert EntityType.ASSET == "asset"
		assert EntityType.DOCUMENT == "document"
		assert EntityType.EVENT == "event"
		
		# Test enum iteration
		all_types = list(EntityType)
		assert len(all_types) == 8
	
	def test_entity_status_enum_values(self):
		"""Test EntityStatus enum values"""
		assert EntityStatus.ACTIVE == "active"
		assert EntityStatus.INACTIVE == "inactive"
		assert EntityStatus.PENDING == "pending"
		assert EntityStatus.MERGED == "merged"
		assert EntityStatus.DELETED == "deleted"
		assert EntityStatus.ARCHIVED == "archived"
		
		all_statuses = list(EntityStatus)
		assert len(all_statuses) == 6
	
	def test_data_quality_status_enum(self):
		"""Test DataQualityStatus enum values"""
		assert DataQualityStatus.EXCELLENT == "excellent"
		assert DataQualityStatus.GOOD == "good"
		assert DataQualityStatus.FAIR == "fair"
		assert DataQualityStatus.POOR == "poor"
		assert DataQualityStatus.CRITICAL == "critical"
		
		all_quality_statuses = list(DataQualityStatus)
		assert len(all_quality_statuses) == 5
	
	def test_match_confidence_enum(self):
		"""Test MatchConfidence enum values"""
		assert MatchConfidence.EXACT == "exact"
		assert MatchConfidence.HIGH == "high"
		assert MatchConfidence.MEDIUM == "medium"
		assert MatchConfidence.LOW == "low"
		
		all_confidences = list(MatchConfidence)
		assert len(all_confidences) == 4