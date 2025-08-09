#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Service Layer Testing
Unit tests for MDM service operations and business logic

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from uuid_extensions import uuid7str

from ...service import (
    MDMService, EntityService, QualityService, MatchingService, 
    AuditService, MDMOperationType, MDMOperationContext
)
from ...models import (
    MdEntityCreate, MdEntityUpdate, EntityType, EntityStatus, 
    DataQualityStatus, MatchConfidence
)


class TestMDMService:
	"""Test main MDM service orchestration"""
	
	async def test_service_initialization(self, test_db_manager, mock_integration_manager):
		"""Test MDM service initialization"""
		service = MDMService(
			db_manager=test_db_manager,
			integration_manager=mock_integration_manager,
			config={'enable_ai': False}
		)
		
		await service.initialize()
		
		assert service.is_initialized is True
		assert service.entity_service is not None
		assert service.quality_service is not None
		assert service.matching_service is not None
		assert service.audit_service is not None
		
		await service.shutdown()
	
	async def test_service_health_check(self, test_mdm_service):
		"""Test service health check"""
		health = await test_mdm_service.health_check()
		
		assert health["status"] == "healthy"
		assert health["services"]["entity_service"] == "healthy"
		assert health["services"]["quality_service"] == "healthy"
		assert health["services"]["matching_service"] == "healthy"
		assert health["services"]["audit_service"] == "healthy"
		assert health["database"] == "connected"
		assert "uptime_seconds" in health
		assert "version" in health
	
	async def test_operation_context_creation(self, test_mdm_service, test_tenant_id, test_user_id):
		"""Test operation context creation"""
		context = test_mdm_service.create_operation_context(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			operation_type=MDMOperationType.CREATE_ENTITY,
			entity_type="person",
			source_system="test_api",
			client_ip="192.168.1.100",
			user_agent="pytest/1.0"
		)
		
		assert isinstance(context, MDMOperationContext)
		assert context.tenant_id == test_tenant_id
		assert context.user_id == test_user_id
		assert context.operation_type == MDMOperationType.CREATE_ENTITY
		assert context.entity_type == "person"
		assert context.source_system == "test_api"
		assert context.client_ip == "192.168.1.100"
		assert context.operation_id is not None


class TestEntityService:
	"""Test entity service operations"""
	
	async def test_create_entity_success(self, test_mdm_service, sample_entity_data, 
	                                   test_tenant_id, test_user_id):
		"""Test successful entity creation"""
		entity_create = MdEntityCreate(**sample_entity_data)
		context = test_mdm_service.create_operation_context(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			operation_type=MDMOperationType.CREATE_ENTITY,
			source_system="test"
		)
		
		result = await test_mdm_service.entity_service.create_entity(entity_create, context)
		
		assert result["status"] == "success"
		assert "entity_id" in result
		assert result["entity_id"] is not None
		assert "created_at" in result
		
		# Verify entity was actually created
		get_result = await test_mdm_service.entity_service.get_entity(
			result["entity_id"], test_tenant_id
		)
		assert get_result["status"] == "success"
		assert get_result["entity"]["entity_name"] == sample_entity_data["entity_name"]
	
	async def test_create_entity_validation_errors(self, test_mdm_service, test_tenant_id, test_user_id):
		"""Test entity creation with validation errors"""
		# Invalid entity data - missing required fields
		invalid_entity_data = {
			"tenant_id": test_tenant_id,
			"entity_type": EntityType.PERSON,
			# Missing entity_name, business_key, source_system
		}
		
		context = test_mdm_service.create_operation_context(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			operation_type=MDMOperationType.CREATE_ENTITY,
			source_system="test"
		)
		
		with pytest.raises(ValueError):
			entity_create = MdEntityCreate(**invalid_entity_data)
	
	async def test_get_entity_success(self, test_mdm_service, created_test_entity, 
	                                test_tenant_id):
		"""Test successful entity retrieval"""
		entity_id = created_test_entity["entity_id"]
		
		result = await test_mdm_service.entity_service.get_entity(entity_id, test_tenant_id)
		
		assert result["status"] == "success"
		assert result["entity"]["entity_id"] == entity_id
		assert result["entity"]["entity_name"] == created_test_entity["entity_name"]
		assert result["entity"]["tenant_id"] == test_tenant_id
	
	async def test_get_entity_with_includes(self, test_mdm_service, created_test_entity, 
	                                      test_tenant_id):
		"""Test entity retrieval with optional includes"""
		entity_id = created_test_entity["entity_id"]
		
		result = await test_mdm_service.entity_service.get_entity(
			entity_id, test_tenant_id,
			include_versions=True,
			include_quality=True,
			include_cross_refs=True
		)
		
		assert result["status"] == "success"
		assert "versions" in result["entity"]
		assert "quality_assessment" in result["entity"]
		assert "cross_references" in result["entity"]
	
	async def test_get_entity_not_found(self, test_mdm_service, test_tenant_id):
		"""Test entity retrieval for non-existent entity"""
		non_existent_id = uuid7str()
		
		result = await test_mdm_service.entity_service.get_entity(non_existent_id, test_tenant_id)
		
		assert result["status"] == "error"
		assert result["error_code"] == "ENTITY_NOT_FOUND"
		assert non_existent_id in result["message"]
	
	async def test_update_entity_success(self, test_mdm_service, created_test_entity, 
	                                   test_tenant_id, test_user_id):
		"""Test successful entity update"""
		entity_id = created_test_entity["entity_id"]
		
		update_data = MdEntityUpdate(
			entity_name="Updated Entity Name",
			entity_description="Updated description",
			attributes={
				**created_test_entity["attributes"],
				"updated_field": "new_value",
				"timestamp": datetime.utcnow().isoformat()
			},
			tags=created_test_entity["tags"] + ["updated"]
		)
		
		context = test_mdm_service.create_operation_context(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			operation_type=MDMOperationType.UPDATE_ENTITY,
			entity_id=entity_id,
			source_system="test"
		)
		
		result = await test_mdm_service.entity_service.update_entity(
			entity_id, update_data, context
		)
		
		assert result["status"] == "success"
		assert result["entity_id"] == entity_id
		assert "updated_at" in result
		
		# Verify update was applied
		get_result = await test_mdm_service.entity_service.get_entity(entity_id, test_tenant_id)
		updated_entity = get_result["entity"]
		
		assert updated_entity["entity_name"] == "Updated Entity Name"
		assert updated_entity["entity_description"] == "Updated description"
		assert updated_entity["attributes"]["updated_field"] == "new_value"
		assert "updated" in updated_entity["tags"]
	
	async def test_delete_entity_success(self, test_mdm_service, created_test_entity,
	                                   test_tenant_id, test_user_id):
		"""Test successful entity deletion"""
		entity_id = created_test_entity["entity_id"]
		
		context = test_mdm_service.create_operation_context(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			operation_type=MDMOperationType.DELETE_ENTITY,
			entity_id=entity_id,
			source_system="test"
		)
		
		result = await test_mdm_service.entity_service.delete_entity(entity_id, context)
		
		assert result["status"] == "success"
		assert result["entity_id"] == entity_id
		
		# Verify entity is marked as deleted
		get_result = await test_mdm_service.entity_service.get_entity(entity_id, test_tenant_id)
		assert get_result["status"] == "success"  # Soft delete - still retrievable
		assert get_result["entity"]["status"] == EntityStatus.DELETED
	
	async def test_search_entities_basic(self, test_mdm_service, multiple_test_entities, 
	                                   test_tenant_id):
		"""Test basic entity search"""
		search_criteria = {
			"entity_type": EntityType.PERSON,
			"limit": 10,
			"offset": 0
		}
		
		result = await test_mdm_service.entity_service.search_entities(
			test_tenant_id, search_criteria
		)
		
		assert result["status"] == "success"
		assert "entities" in result
		assert "pagination" in result
		assert len(result["entities"]) == 5  # All test entities
		
		# Verify all returned entities are PERSON type
		for entity in result["entities"]:
			assert entity["entity_type"] == EntityType.PERSON
	
	async def test_search_entities_with_filters(self, test_mdm_service, multiple_test_entities,
	                                          test_tenant_id):
		"""Test entity search with various filters"""
		# Search by name pattern
		search_criteria = {
			"entity_name": "Person 1",  # Should match "Person 1"
			"limit": 10,
			"offset": 0
		}
		
		result = await test_mdm_service.entity_service.search_entities(
			test_tenant_id, search_criteria
		)
		
		assert result["status"] == "success"
		assert len(result["entities"]) == 1
		assert result["entities"][0]["entity_name"] == "Person 1"
		
		# Search by business key pattern
		search_criteria = {
			"business_key": "PERSON-00",  # Should match PERSON-001, PERSON-002, etc.
			"limit": 10,
			"offset": 0
		}
		
		result = await test_mdm_service.entity_service.search_entities(
			test_tenant_id, search_criteria
		)
		
		assert result["status"] == "success"
		assert len(result["entities"]) >= 3  # Should match multiple entities
	
	async def test_search_entities_pagination(self, test_mdm_service, multiple_test_entities,
	                                        test_tenant_id):
		"""Test entity search pagination"""
		# First page
		search_criteria = {
			"limit": 2,
			"offset": 0,
			"sort_by": "entity_name",
			"sort_order": "asc"
		}
		
		page_1 = await test_mdm_service.entity_service.search_entities(
			test_tenant_id, search_criteria
		)
		
		assert page_1["status"] == "success"
		assert len(page_1["entities"]) == 2
		assert page_1["pagination"]["has_next"] is True
		assert page_1["pagination"]["has_previous"] is False
		
		# Second page
		search_criteria["offset"] = 2
		page_2 = await test_mdm_service.entity_service.search_entities(
			test_tenant_id, search_criteria
		)
		
		assert page_2["status"] == "success"
		assert len(page_2["entities"]) == 2
		assert page_2["pagination"]["has_previous"] is True
		
		# Entities should be different between pages
		page_1_ids = {e["entity_id"] for e in page_1["entities"]}
		page_2_ids = {e["entity_id"] for e in page_2["entities"]}
		assert page_1_ids.isdisjoint(page_2_ids)
	
	async def test_batch_entity_operations(self, test_mdm_service, test_tenant_id, test_user_id):
		"""Test batch entity operations"""
		# Create multiple entities in batch
		entities_data = []
		for i in range(3):
			entity_data = MdEntityCreate(
				tenant_id=test_tenant_id,
				entity_type=EntityType.PRODUCT,
				entity_name=f"Batch Product {i}",
				business_key=f"BATCH-{i:03d}",
				source_system="batch_system",
				status=EntityStatus.ACTIVE,
				attributes={"batch_index": i},
				data_classification="public"
			)
			entities_data.append(entity_data)
		
		context = test_mdm_service.create_operation_context(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			operation_type=MDMOperationType.BATCH_CREATE,
			source_system="test_batch"
		)
		
		result = await test_mdm_service.entity_service.batch_create_entities(
			entities_data, context
		)
		
		assert result["status"] == "success"
		assert result["total_processed"] == 3
		assert result["successful"] == 3
		assert result["failed"] == 0
		assert len(result["entity_ids"]) == 3


class TestQualityService:
	"""Test quality service operations"""
	
	async def test_assess_entity_quality_success(self, test_mdm_service, created_test_entity,
	                                           test_tenant_id, quality_assessment_data):
		"""Test successful quality assessment"""
		entity_id = created_test_entity["entity_id"]
		entity_attributes = created_test_entity["attributes"]
		entity_type = created_test_entity["entity_type"]
		
		result = await test_mdm_service.quality_service.assess_quality(
			entity_id, test_tenant_id, entity_attributes, entity_type
		)
		
		assert result["status"] == "success"
		assert "overall_score" in result
		assert "quality_status" in result
		assert "completeness_score" in result
		assert "accuracy_score" in result
		assert "consistency_score" in result
		assert "validity_score" in result
		assert "uniqueness_score" in result
		assert "timeliness_score" in result
		assert "assessment_duration_ms" in result
		
		# Scores should be valid ranges
		assert 0.0 <= result["overall_score"] <= 100.0
		assert result["quality_status"] in ["excellent", "good", "fair", "poor", "critical"]
	
	async def test_quality_assessment_with_issues(self, test_mdm_service, test_tenant_id):
		"""Test quality assessment with data quality issues"""
		# Create entity with known quality issues
		poor_quality_attributes = {
			"first_name": "",  # Completeness issue
			"email": "invalid-email",  # Validity issue
			"phone": "123",  # Validity issue
			"last_updated": "2020-01-01",  # Timeliness issue
		}
		
		result = await test_mdm_service.quality_service.assess_quality(
			uuid7str(), test_tenant_id, poor_quality_attributes, EntityType.PERSON
		)
		
		assert result["status"] == "success"
		assert result["overall_score"] < 70.0  # Should be lower due to issues
		assert len(result.get("quality_issues", [])) > 0
		assert len(result.get("recommendations", [])) > 0
		
		# Check for expected issue types
		issue_types = {issue["issue_type"] for issue in result.get("quality_issues", [])}
		assert "completeness" in issue_types  # Empty first_name
		assert "validity" in issue_types  # Invalid email/phone
	
	async def test_batch_quality_assessment(self, test_mdm_service, multiple_test_entities,
	                                      test_tenant_id):
		"""Test batch quality assessment"""
		entity_ids = [entity["entity_id"] for entity in multiple_test_entities]
		
		result = await test_mdm_service.quality_service.batch_assess_quality(
			entity_ids, test_tenant_id
		)
		
		assert result["status"] == "success"
		assert "assessments" in result
		assert len(result["assessments"]) == len(entity_ids)
		assert "summary" in result
		
		# Verify each assessment
		for assessment in result["assessments"]:
			assert "entity_id" in assessment
			assert "overall_score" in assessment
			assert "quality_status" in assessment
			assert assessment["entity_id"] in entity_ids


class TestMatchingService:
	"""Test matching service operations"""
	
	async def test_find_duplicates_success(self, test_mdm_service, created_test_entity,
	                                     test_tenant_id, duplicate_candidates_data):
		"""Test successful duplicate detection"""
		entity_id = created_test_entity["entity_id"]
		entity_data = created_test_entity
		
		result = await test_mdm_service.matching_service.find_duplicates(
			entity_id, test_tenant_id, entity_data
		)
		
		assert result["status"] == "success"
		assert "total_candidates" in result
		assert "high_confidence_matches" in result
		assert "medium_confidence_matches" in result
		assert "low_confidence_matches" in result
		assert "match_candidates" in result
		assert "detection_timestamp" in result
		
		# Counts should be non-negative
		assert result["total_candidates"] >= 0
		assert result["high_confidence_matches"] >= 0
		assert result["medium_confidence_matches"] >= 0
		assert result["low_confidence_matches"] >= 0
	
	async def test_calculate_match_score(self, test_mdm_service, test_tenant_id):
		"""Test match score calculation"""
		entity_1 = {
			"entity_name": "John Doe",
			"attributes": {
				"first_name": "John",
				"last_name": "Doe",
				"email": "john.doe@example.com",
				"phone": "+1-555-123-4567"
			}
		}
		
		entity_2 = {
			"entity_name": "John D. Doe",
			"attributes": {
				"first_name": "John",
				"last_name": "Doe",
				"email": "john.doe@example.com",
				"phone": "+1-555-123-4567"
			}
		}
		
		result = await test_mdm_service.matching_service.calculate_match_score(
			entity_1, entity_2, EntityType.PERSON
		)
		
		assert result["status"] == "success"
		assert "match_score" in result
		assert "confidence" in result
		assert "similarity_details" in result
		
		# High similarity expected
		assert result["match_score"] >= 90.0
		assert result["confidence"] in ["exact", "high", "medium", "low"]
	
	async def test_match_score_with_different_entities(self, test_mdm_service, test_tenant_id):
		"""Test match score for very different entities"""
		entity_1 = {
			"entity_name": "John Doe",
			"attributes": {
				"first_name": "John",
				"last_name": "Doe",
				"email": "john.doe@example.com"
			}
		}
		
		entity_2 = {
			"entity_name": "Jane Smith",
			"attributes": {
				"first_name": "Jane",
				"last_name": "Smith",
				"email": "jane.smith@different.com"
			}
		}
		
		result = await test_mdm_service.matching_service.calculate_match_score(
			entity_1, entity_2, EntityType.PERSON
		)
		
		assert result["status"] == "success"
		assert result["match_score"] < 50.0  # Should be low similarity
		assert result["confidence"] in ["low", "medium"]


class TestAuditService:
	"""Test audit service operations"""
	
	async def test_log_operation_success(self, test_mdm_service, test_tenant_id, test_user_id):
		"""Test successful audit logging"""
		context = test_mdm_service.create_operation_context(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			operation_type=MDMOperationType.CREATE_ENTITY,
			entity_id=uuid7str(),
			entity_type="person",
			source_system="test_audit"
		)
		
		result = await test_mdm_service.audit_service.log_operation(
			context,
			operation_details={
				"action": "entity_created",
				"entity_name": "Test Entity",
				"changes": ["name", "attributes"]
			},
			outcome="success"
		)
		
		assert result["status"] == "success"
		assert "audit_id" in result
		assert result["audit_id"] is not None
	
	async def test_get_audit_trail(self, test_mdm_service, created_test_entity, test_tenant_id):
		"""Test audit trail retrieval"""
		entity_id = created_test_entity["entity_id"]
		
		result = await test_mdm_service.audit_service.get_audit_trail(
			entity_id, test_tenant_id
		)
		
		assert result["status"] == "success"
		assert "audit_entries" in result
		assert len(result["audit_entries"]) >= 1  # At least creation entry
		
		# Verify audit entry structure
		first_entry = result["audit_entries"][0]
		assert "audit_id" in first_entry
		assert "event_type" in first_entry
		assert "event_timestamp" in first_entry
		assert "user_id" in first_entry
		assert "operation_details" in first_entry


class TestServiceIntegration:
	"""Test service layer integration and error handling"""
	
	async def test_service_performance_benchmarks(self, test_mdm_service, sample_entity_data,
	                                            test_tenant_id, test_user_id,
	                                            performance_benchmarks):
		"""Test service operation performance"""
		entity_create = MdEntityCreate(**sample_entity_data)
		context = test_mdm_service.create_operation_context(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			operation_type=MDMOperationType.CREATE_ENTITY,
			source_system="perf_test"
		)
		
		# Measure entity creation time
		start_time = datetime.utcnow()
		result = await test_mdm_service.entity_service.create_entity(entity_create, context)
		end_time = datetime.utcnow()
		
		creation_time_ms = (end_time - start_time).total_seconds() * 1000
		
		assert result["status"] == "success"
		assert creation_time_ms <= performance_benchmarks["entity_creation_max_ms"]
		
		# Measure entity retrieval time
		entity_id = result["entity_id"]
		start_time = datetime.utcnow()
		get_result = await test_mdm_service.entity_service.get_entity(entity_id, test_tenant_id)
		end_time = datetime.utcnow()
		
		retrieval_time_ms = (end_time - start_time).total_seconds() * 1000
		
		assert get_result["status"] == "success"
		assert retrieval_time_ms <= performance_benchmarks["entity_retrieval_max_ms"]
	
	async def test_service_error_handling(self, test_mdm_service, test_tenant_id):
		"""Test service error handling and recovery"""
		# Test invalid entity ID format
		result = await test_mdm_service.entity_service.get_entity(
			"invalid-uuid", test_tenant_id
		)
		
		assert result["status"] == "error"
		assert "error_code" in result
		assert "message" in result
		
		# Test invalid tenant isolation
		valid_entity_id = uuid7str()
		wrong_tenant = f"wrong-tenant-{uuid7str()[:8]}"
		
		result = await test_mdm_service.entity_service.get_entity(
			valid_entity_id, wrong_tenant
		)
		
		assert result["status"] == "error"
		assert result["error_code"] == "ENTITY_NOT_FOUND"
	
	async def test_concurrent_operations(self, test_mdm_service, test_tenant_id, test_user_id):
		"""Test concurrent service operations"""
		# Create multiple entities concurrently
		async def create_entity(index: int):
			entity_data = MdEntityCreate(
				tenant_id=test_tenant_id,
				entity_type=EntityType.PRODUCT,
				entity_name=f"Concurrent Product {index}",
				business_key=f"CONC-{index:03d}",
				source_system="concurrent_test",
				status=EntityStatus.ACTIVE,
				attributes={"index": index},
				data_classification="public"
			)
			
			context = test_mdm_service.create_operation_context(
				tenant_id=test_tenant_id,
				user_id=f"{test_user_id}-{index}",
				operation_type=MDMOperationType.CREATE_ENTITY,
				source_system="concurrent_test"
			)
			
			return await test_mdm_service.entity_service.create_entity(entity_data, context)
		
		# Execute concurrent operations
		tasks = [create_entity(i) for i in range(5)]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# All operations should succeed
		successful_results = [r for r in results if not isinstance(r, Exception)]
		assert len(successful_results) == 5
		
		for result in successful_results:
			assert result["status"] == "success"
			assert "entity_id" in result
		
		# Verify all entities were created with unique IDs
		entity_ids = {r["entity_id"] for r in successful_results}
		assert len(entity_ids) == 5  # All unique