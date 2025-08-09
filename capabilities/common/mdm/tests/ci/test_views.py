#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Views Testing
Unit tests for Pydantic view models and serialization

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from uuid_extensions import uuid7str

from pydantic import ValidationError

from ...views import (
    # Base models
    MDMBaseResponse, PaginationMeta,
    
    # Entity views
    EntitySummaryView, EntityDetailView, EntityVersionView, EntitySearchResultView,
    
    # Quality views
    QualityIssueView, QualityAssessmentView, QualityBatchAssessmentView,
    
    # Duplicate detection views
    DuplicateCandidateView, DuplicateDetectionResultView,
    
    # Golden record views
    GoldenRecordView,
    
    # Cross-reference views
    CrossReferenceView,
    
    # Audit views
    AuditLogView, DataLineageView,
    
    # Analytics views
    EntityStatisticsView, QualityTrendsView,
    
    # Bulk operation views
    BulkOperationStatusView,
    
    # Configuration views
    MatchRuleView, SurvivorshipRuleView,
    
    # Response containers
    EntityResponse, EntityListResponse, QualityAssessmentResponse
)

from ...models import EntityType, EntityStatus, DataQualityStatus, MatchConfidence


class TestBaseModels:
	"""Test base Pydantic models"""
	
	def test_mdm_base_response_creation(self):
		"""Test MDM base response creation"""
		response = MDMBaseResponse(
			success=True,
			message="Operation completed successfully",
			request_id="req-123"
		)
		
		assert response.success is True
		assert response.message == "Operation completed successfully"
		assert response.request_id == "req-123"
		assert response.timestamp is not None
		assert isinstance(response.timestamp, datetime)
	
	def test_mdm_base_response_defaults(self):
		"""Test MDM base response with defaults"""
		response = MDMBaseResponse()
		
		assert response.success is True
		assert response.message == ""
		assert response.request_id is None
		assert response.timestamp is not None
	
	def test_pagination_meta_calculation(self):
		"""Test pagination metadata calculations"""
		pagination = PaginationMeta(
			total_count=100,
			offset=20,
			limit=10,
			has_next=True,
			has_previous=True
		)
		
		assert pagination.total_count == 100
		assert pagination.offset == 20
		assert pagination.limit == 10
		assert pagination.has_next is True
		assert pagination.has_previous is True
		assert pagination.total_pages == 10  # ceil(100/10)
		assert pagination.current_page == 3   # floor(20/10) + 1
	
	def test_pagination_meta_edge_cases(self):
		"""Test pagination metadata edge cases"""
		# First page
		pagination = PaginationMeta(
			total_count=25,
			offset=0,
			limit=10,
			has_next=True,
			has_previous=False
		)
		
		assert pagination.current_page == 1
		assert pagination.total_pages == 3
		
		# Last page with partial results
		pagination = PaginationMeta(
			total_count=25,
			offset=20,
			limit=10,
			has_next=False,
			has_previous=True
		)
		
		assert pagination.current_page == 3
		assert pagination.total_pages == 3
	
	def test_pagination_validation_errors(self):
		"""Test pagination validation errors"""
		# Negative total_count
		with pytest.raises(ValidationError) as exc_info:
			PaginationMeta(
				total_count=-1,
				offset=0,
				limit=10,
				has_next=False,
				has_previous=False
			)
		assert "total_count" in str(exc_info.value)
		
		# Invalid limit
		with pytest.raises(ValidationError) as exc_info:
			PaginationMeta(
				total_count=100,
				offset=0,
				limit=0,  # Must be >= 1
				has_next=False,
				has_previous=False
			)
		assert "limit" in str(exc_info.value)
		
		# Limit too large
		with pytest.raises(ValidationError) as exc_info:
			PaginationMeta(
				total_count=100,
				offset=0,
				limit=1001,  # Must be <= 1000
				has_next=False,
				has_previous=False
			)
		assert "limit" in str(exc_info.value)


class TestEntityViews:
	"""Test entity view models"""
	
	def test_entity_summary_view_creation(self):
		"""Test entity summary view creation"""
		entity_summary = EntitySummaryView(
			entity_id=uuid7str(),
			entity_type=EntityType.PERSON,
			entity_name="John Doe",
			business_key="PERSON-001",
			source_system="crm_system",
			status=EntityStatus.ACTIVE,
			quality_score=85.5,
			is_golden_record=True,
			data_classification="confidential",
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			tags=["customer", "vip"]
		)
		
		assert entity_summary.entity_type == EntityType.PERSON
		assert entity_summary.entity_name == "John Doe"
		assert entity_summary.quality_score == 85.5
		assert entity_summary.is_golden_record is True
		assert len(entity_summary.tags) == 2
		assert "customer" in entity_summary.tags
	
	def test_entity_summary_view_validation(self):
		"""Test entity summary view validation"""
		# Quality score out of range
		with pytest.raises(ValidationError) as exc_info:
			EntitySummaryView(
				entity_id=uuid7str(),
				entity_type=EntityType.PERSON,
				entity_name="Test",
				business_key="TEST-001",
				source_system="test",
				status=EntityStatus.ACTIVE,
				quality_score=150.0,  # > 100.0
				data_classification="internal",
				created_at=datetime.utcnow(),
				updated_at=datetime.utcnow()
			)
		assert "quality_score" in str(exc_info.value)
		
		# Too many tags
		with pytest.raises(ValidationError) as exc_info:
			EntitySummaryView(
				entity_id=uuid7str(),
				entity_type=EntityType.PERSON,
				entity_name="Test",
				business_key="TEST-001",
				source_system="test",
				status=EntityStatus.ACTIVE,
				quality_score=85.0,
				data_classification="internal",
				created_at=datetime.utcnow(),
				updated_at=datetime.utcnow(),
				tags=[f"tag_{i}" for i in range(15)]  # > 10 tags
			)
		assert "tags" in str(exc_info.value)
	
	def test_entity_detail_view_creation(self):
		"""Test entity detail view creation"""
		entity_id = uuid7str()
		tenant_id = f"tenant-{uuid7str()[:8]}"
		
		entity_detail = EntityDetailView(
			entity_id=entity_id,
			tenant_id=tenant_id,
			entity_type=EntityType.CUSTOMER,
			entity_name="Acme Corporation",
			entity_description="Technology company",
			business_key="CUST-001",
			source_system="crm_system",
			status=EntityStatus.ACTIVE,
			attributes={
				"industry": "Technology",
				"revenue": 5000000,
				"employees": 150,
				"website": "https://acme.com"
			},
			tags=["enterprise", "technology", "customer"],
			data_classification="confidential",
			quality_score=92.5,
			last_quality_check=datetime.utcnow() - timedelta(days=1),
			is_golden_record=True,
			golden_record_id=uuid7str(),
			created_at=datetime.utcnow() - timedelta(days=30),
			updated_at=datetime.utcnow() - timedelta(hours=2),
			created_by="system",
			updated_by="user-123",
			audit_trail_id=uuid7str()
		)
		
		assert entity_detail.entity_id == entity_id
		assert entity_detail.tenant_id == tenant_id
		assert entity_detail.entity_type == EntityType.CUSTOMER
		assert entity_detail.attributes["industry"] == "Technology"
		assert entity_detail.attributes["revenue"] == 5000000
		assert entity_detail.is_golden_record is True
		assert entity_detail.golden_record_id is not None
		assert len(entity_detail.tags) == 3
	
	def test_entity_version_view_creation(self):
		"""Test entity version view creation"""
		version_view = EntityVersionView(
			version_id=uuid7str(),
			version_number=3,
			version_timestamp=datetime.utcnow(),
			version_type="update",
			created_by="user-456",
			change_description="Updated contact information",
			changed_fields=["attributes.email", "attributes.phone", "updated_at"],
			quality_score_snapshot=88.5,
			change_source="api"
		)
		
		assert version_view.version_number == 3
		assert version_view.version_type == "update"
		assert version_view.change_description == "Updated contact information"
		assert len(version_view.changed_fields) == 3
		assert "attributes.email" in version_view.changed_fields
		assert version_view.quality_score_snapshot == 88.5
	
	def test_entity_version_validation(self):
		"""Test entity version view validation"""
		# Version number must be >= 1
		with pytest.raises(ValidationError) as exc_info:
			EntityVersionView(
				version_id=uuid7str(),
				version_number=0,  # Invalid
				version_timestamp=datetime.utcnow(),
				version_type="create",
				created_by="user"
			)
		assert "version_number" in str(exc_info.value)
	
	def test_entity_search_result_view_creation(self):
		"""Test entity search result view creation"""
		entities = [
			EntitySummaryView(
				entity_id=uuid7str(),
				entity_type=EntityType.PERSON,
				entity_name=f"Person {i}",
				business_key=f"PERSON-{i:03d}",
				source_system="test_system",
				status=EntityStatus.ACTIVE,
				quality_score=80.0 + i,
				data_classification="internal",
				created_at=datetime.utcnow(),
				updated_at=datetime.utcnow()
			)
			for i in range(1, 4)
		]
		
		pagination = PaginationMeta(
			total_count=10,
			offset=0,
			limit=3,
			has_next=True,
			has_previous=False
		)
		
		search_result = EntitySearchResultView(
			entities=entities,
			pagination=pagination,
			search_criteria={"entity_type": "person", "limit": 3},
			search_duration_ms=125.5,
			total_quality_score_avg=81.0,
			entity_type_breakdown={"person": 3}
		)
		
		assert len(search_result.entities) == 3
		assert search_result.pagination.total_count == 10
		assert search_result.search_criteria["entity_type"] == "person"
		assert search_result.search_duration_ms == 125.5
		assert search_result.total_quality_score_avg == 81.0
		assert search_result.entity_type_breakdown["person"] == 3


class TestQualityViews:
	"""Test quality assessment view models"""
	
	def test_quality_issue_view_creation(self):
		"""Test quality issue view creation"""
		quality_issue = QualityIssueView(
			issue_type="completeness",
			field="email",
			severity="high",
			message="Email field is empty",
			recommendation="Collect email address from customer",
			auto_fixable=False
		)
		
		assert quality_issue.issue_type == "completeness"
		assert quality_issue.field == "email"
		assert quality_issue.severity == "high"
		assert quality_issue.auto_fixable is False
	
	def test_quality_issue_validation(self):
		"""Test quality issue validation"""
		# Invalid severity
		with pytest.raises(ValidationError) as exc_info:
			QualityIssueView(
				issue_type="completeness",
				field="email",
				severity="invalid",  # Not in allowed values
				message="Test message"
			)
		assert "severity" in str(exc_info.value)
	
	def test_quality_assessment_view_creation(self):
		"""Test quality assessment view creation"""
		assessment_id = uuid7str()
		entity_id = uuid7str()
		tenant_id = f"tenant-{uuid7str()[:8]}"
		
		quality_issues = [
			QualityIssueView(
				issue_type="timeliness",
				field="last_updated",
				severity="medium",
				message="Data is 6 months old",
				recommendation="Update from source system"
			),
			QualityIssueView(
				issue_type="validity",
				field="phone",
				severity="low",
				message="Phone format is non-standard",
				recommendation="Standardize phone format",
				auto_fixable=True
			)
		]
		
		quality_assessment = QualityAssessmentView(
			assessment_id=assessment_id,
			entity_id=entity_id,
			tenant_id=tenant_id,
			overall_score=82.5,
			quality_status=DataQualityStatus.GOOD,
			completeness_score=90.0,
			accuracy_score=85.0,
			consistency_score=80.0,
			validity_score=78.0,
			uniqueness_score=95.0,
			timeliness_score=75.0,
			assessment_timestamp=datetime.utcnow(),
			assessment_duration_ms=175.5,
			assessment_algorithm="ai_enhanced",
			algorithm_version="1.2.0",
			quality_issues=quality_issues,
			recommendations=[
				"Update contact information",
				"Verify email address",
				"Standardize data formats"
			],
			priority_issues=["Data freshness", "Phone format"],
			auto_fix_suggestions=[
				{"field": "phone", "action": "format_standardization", "confidence": 0.95}
			]
		)
		
		assert quality_assessment.assessment_id == assessment_id
		assert quality_assessment.overall_score == 82.5
		assert quality_assessment.quality_status == DataQualityStatus.GOOD
		assert len(quality_assessment.quality_issues) == 2
		assert len(quality_assessment.recommendations) == 3
		assert quality_assessment.assessment_algorithm == "ai_enhanced"
		
		# Test individual dimension scores
		assert quality_assessment.completeness_score == 90.0
		assert quality_assessment.accuracy_score == 85.0
		assert quality_assessment.validity_score == 78.0
	
	def test_quality_assessment_score_validation(self):
		"""Test quality assessment score validation"""
		# Invalid overall score
		with pytest.raises(ValidationError) as exc_info:
			QualityAssessmentView(
				assessment_id=uuid7str(),
				entity_id=uuid7str(),
				tenant_id="test-tenant",
				overall_score=-10.0,  # < 0.0
				quality_status=DataQualityStatus.POOR,
				completeness_score=50.0,
				accuracy_score=50.0,
				consistency_score=50.0,
				validity_score=50.0,
				uniqueness_score=50.0,
				timeliness_score=50.0,
				assessment_timestamp=datetime.utcnow()
			)
		assert "overall_score" in str(exc_info.value)


class TestDuplicateDetectionViews:
	"""Test duplicate detection view models"""
	
	def test_duplicate_candidate_view_creation(self):
		"""Test duplicate candidate view creation"""
		duplicate_candidate = DuplicateCandidateView(
			candidate_id=uuid7str(),
			candidate_name="John D. Doe",
			candidate_business_key="PERSON-002",
			candidate_source_system="hr_system",
			match_score=92.5,
			confidence=MatchConfidence.HIGH,
			matching_attributes=["first_name", "last_name", "email"],
			similarity_details={
				"name_similarity": 95.0,
				"email_similarity": 100.0,
				"phone_similarity": 0.0
			},
			recommended_action="merge",
			match_explanation="High similarity in name and email address",
			last_updated=datetime.utcnow()
		)
		
		assert duplicate_candidate.match_score == 92.5
		assert duplicate_candidate.confidence == MatchConfidence.HIGH
		assert len(duplicate_candidate.matching_attributes) == 3
		assert "email" in duplicate_candidate.matching_attributes
		assert duplicate_candidate.similarity_details["name_similarity"] == 95.0
		assert duplicate_candidate.recommended_action == "merge"
	
	def test_duplicate_candidate_validation(self):
		"""Test duplicate candidate validation"""
		# Invalid recommended action
		with pytest.raises(ValidationError) as exc_info:
			DuplicateCandidateView(
				candidate_id=uuid7str(),
				candidate_name="Test",
				candidate_business_key="TEST-001",
				candidate_source_system="test",
				match_score=85.0,
				confidence=MatchConfidence.MEDIUM,
				recommended_action="invalid_action"  # Not in allowed values
			)
		assert "recommended_action" in str(exc_info.value)
	
	def test_duplicate_detection_result_view_creation(self):
		"""Test duplicate detection result view creation"""
		candidates = [
			DuplicateCandidateView(
				candidate_id=uuid7str(),
				candidate_name="Similar Entity 1",
				candidate_business_key="SIM-001",
				candidate_source_system="system1",
				match_score=95.0,
				confidence=MatchConfidence.HIGH,
				recommended_action="merge"
			),
			DuplicateCandidateView(
				candidate_id=uuid7str(),
				candidate_name="Similar Entity 2",
				candidate_business_key="SIM-002",
				candidate_source_system="system2",
				match_score=75.0,
				confidence=MatchConfidence.MEDIUM,
				recommended_action="review"
			)
		]
		
		detection_result = DuplicateDetectionResultView(
			detection_id=uuid7str(),
			entity_id=uuid7str(),
			entity_name="Original Entity",
			tenant_id="test-tenant",
			total_candidates=2,
			high_confidence_matches=1,
			medium_confidence_matches=1,
			low_confidence_matches=0,
			match_candidates=candidates,
			detection_timestamp=datetime.utcnow(),
			detection_duration_ms=350.0,
			algorithm_version="2.1.0",
			detection_rules_applied=["name_similarity", "email_matching", "phone_matching"],
			next_review_date=datetime.utcnow() + timedelta(days=30)
		)
		
		assert detection_result.total_candidates == 2
		assert detection_result.high_confidence_matches == 1
		assert detection_result.medium_confidence_matches == 1
		assert len(detection_result.match_candidates) == 2
		assert detection_result.detection_duration_ms == 350.0
		assert len(detection_result.detection_rules_applied) == 3


class TestGoldenRecordViews:
	"""Test golden record view models"""
	
	def test_golden_record_view_creation(self):
		"""Test golden record view creation"""
		golden_record = GoldenRecordView(
			golden_record_id=uuid7str(),
			tenant_id="test-tenant",
			entity_type=EntityType.CUSTOMER,
			golden_record_name="Acme Corporation (Golden)",
			business_key="GOLDEN-CUST-001",
			consolidated_attributes={
				"company_name": "Acme Corporation",
				"industry": "Technology",
				"revenue": 5000000,
				"primary_contact": "ceo@acme.com",
				"phone": "+1-555-123-4567",
				"website": "https://www.acme.com"
			},
			source_entity_ids=[uuid7str(), uuid7str(), uuid7str()],
			overall_quality_score=95.5,
			consolidation_confidence=92.0,
			data_completeness=88.0,
			survivorship_rules={
				"company_name": {"strategy": "most_trusted_source", "source_priority": ["crm", "erp"]},
				"revenue": {"strategy": "most_recent", "confidence_threshold": 0.8},
				"contact_info": {"strategy": "ai_determined", "model_version": "1.2"}
			},
			consolidation_method="ai_determined",
			created_at=datetime.utcnow() - timedelta(days=10),
			updated_at=datetime.utcnow() - timedelta(hours=4),
			last_consolidation=datetime.utcnow() - timedelta(hours=4),
			created_by="consolidation_engine",
			is_active=True,
			approval_status="auto_approved",
			approved_by="system",
			approved_at=datetime.utcnow() - timedelta(days=9)
		)
		
		assert golden_record.entity_type == EntityType.CUSTOMER
		assert golden_record.overall_quality_score == 95.5
		assert golden_record.consolidation_confidence == 92.0
		assert len(golden_record.source_entity_ids) == 3
		assert golden_record.consolidated_attributes["company_name"] == "Acme Corporation"
		assert golden_record.survivorship_rules["company_name"]["strategy"] == "most_trusted_source"
		assert golden_record.is_active is True
		assert golden_record.approval_status == "auto_approved"


class TestCrossReferenceViews:
	"""Test cross-reference view models"""
	
	def test_cross_reference_view_creation(self):
		"""Test cross-reference view creation"""
		cross_ref = CrossReferenceView(
			cross_reference_id=uuid7str(),
			entity_id=uuid7str(),
			source_system="external_system",
			source_entity_id="EXT-12345",
			source_entity_type="customer",
			confidence_score=95.0,
			is_primary_reference=True,
			reference_quality="excellent",
			created_at=datetime.utcnow() - timedelta(days=5),
			updated_at=datetime.utcnow() - timedelta(hours=2),
			last_verified=datetime.utcnow() - timedelta(hours=2),
			created_by="integration_service",
			is_active=True,
			verification_method="automated"
		)
		
		assert cross_ref.source_system == "external_system"
		assert cross_ref.source_entity_id == "EXT-12345"
		assert cross_ref.confidence_score == 95.0
		assert cross_ref.is_primary_reference is True
		assert cross_ref.reference_quality == "excellent"
		assert cross_ref.is_active is True
	
	def test_cross_reference_validation(self):
		"""Test cross-reference validation"""
		# Invalid reference quality
		with pytest.raises(ValidationError) as exc_info:
			CrossReferenceView(
				cross_reference_id=uuid7str(),
				entity_id=uuid7str(),
				source_system="test",
				source_entity_id="TEST-001",
				confidence_score=85.0,
				reference_quality="invalid_quality",  # Not in allowed values
				created_at=datetime.utcnow(),
				updated_at=datetime.utcnow(),
				created_by="test"
			)
		assert "reference_quality" in str(exc_info.value)


class TestAnalyticsViews:
	"""Test analytics and statistics view models"""
	
	def test_entity_statistics_view_creation(self):
		"""Test entity statistics view creation"""
		statistics = EntityStatisticsView(
			tenant_id="test-tenant",
			total_entities=1500,
			entities_by_type={
				"person": 800,
				"customer": 400,
				"product": 200,
				"organization": 100
			},
			entities_by_status={
				"active": 1400,
				"inactive": 50,
				"deleted": 30,
				"merged": 20
			},
			entities_by_source={
				"crm_system": 600,
				"erp_system": 400,
				"web_portal": 300,
				"api_imports": 200
			},
			average_quality_score=84.5,
			quality_distribution={
				"excellent": 300,
				"good": 800,
				"fair": 300,
				"poor": 80,
				"critical": 20
			},
			golden_records_count=450,
			duplicate_candidates_count=75,
			data_freshness_stats={
				"avg_age_days": 45.2,
				"stale_entities_count": 120,
				"fresh_entities_count": 1380
			},
			growth_trends={
				"monthly_growth_rate": 0.05,
				"new_entities_this_month": 150,
				"quality_improvement_rate": 0.02
			}
		)
		
		assert statistics.total_entities == 1500
		assert statistics.entities_by_type["person"] == 800
		assert statistics.entities_by_status["active"] == 1400
		assert statistics.average_quality_score == 84.5
		assert statistics.quality_distribution["excellent"] == 300
		assert statistics.golden_records_count == 450
		assert statistics.data_freshness_stats["avg_age_days"] == 45.2
	
	def test_quality_trends_view_creation(self):
		"""Test quality trends view creation"""
		trend_data = [
			{"date": "2024-01-01", "avg_score": 82.0, "assessments": 100},
			{"date": "2024-01-02", "avg_score": 82.5, "assessments": 105},
			{"date": "2024-01-03", "avg_score": 83.0, "assessments": 98},
			{"date": "2024-01-04", "avg_score": 83.2, "assessments": 110}
		]
		
		quality_trends = QualityTrendsView(
			tenant_id="test-tenant",
			time_period="daily",
			trend_data=trend_data,
			overall_trend="improving",
			trend_percentage=1.5,  # 1.5% improvement
			quality_dimension_trends={
				"completeness": {"trend": "stable", "change": 0.1},
				"accuracy": {"trend": "improving", "change": 2.0},
				"timeliness": {"trend": "declining", "change": -0.5}
			},
			top_quality_issues=[
				{"issue_type": "timeliness", "frequency": 45, "severity": "medium"},
				{"issue_type": "completeness", "frequency": 32, "severity": "high"},
				{"issue_type": "validity", "frequency": 28, "severity": "low"}
			],
			improvement_recommendations=[
				"Focus on data freshness for timeliness improvements",
				"Implement automated completeness checks",
				"Add validation rules for critical fields"
			]
		)
		
		assert quality_trends.time_period == "daily"
		assert len(quality_trends.trend_data) == 4
		assert quality_trends.overall_trend == "improving"
		assert quality_trends.trend_percentage == 1.5
		assert quality_trends.quality_dimension_trends["accuracy"]["trend"] == "improving"
		assert len(quality_trends.top_quality_issues) == 3
		assert len(quality_trends.improvement_recommendations) == 3


class TestBulkOperationViews:
	"""Test bulk operation view models"""
	
	def test_bulk_operation_status_view_creation(self):
		"""Test bulk operation status view creation"""
		bulk_operation = BulkOperationStatusView(
			operation_id=uuid7str(),
			tenant_id="test-tenant",
			operation_type="create",
			status="completed",
			total_items=1000,
			processed_items=1000,
			successful_items=985,
			failed_items=15,
			started_at=datetime.utcnow() - timedelta(minutes=10),
			completed_at=datetime.utcnow(),
			progress_percentage=100.0,
			results=[
				{"entity_id": uuid7str(), "status": "success", "message": "Created successfully"},
				{"entity_id": uuid7str(), "status": "success", "message": "Created successfully"}
			],
			errors=[
				{"line": 456, "error": "Validation failed", "details": "Missing required field: business_key"},
				{"line": 789, "error": "Duplicate key", "details": "Business key already exists"}
			],
			warnings=[
				{"line": 123, "warning": "Data quality issue", "details": "Email format is non-standard"}
			],
			processing_rate_per_second=16.4,
			estimated_time_remaining_seconds=0.0
		)
		
		assert bulk_operation.operation_type == "create"
		assert bulk_operation.status == "completed"
		assert bulk_operation.total_items == 1000
		assert bulk_operation.successful_items == 985
		assert bulk_operation.failed_items == 15
		assert bulk_operation.progress_percentage == 100.0
		assert len(bulk_operation.results) == 2
		assert len(bulk_operation.errors) == 2
		assert len(bulk_operation.warnings) == 1
		assert bulk_operation.processing_rate_per_second == 16.4


class TestResponseContainers:
	"""Test response container models"""
	
	def test_entity_response_creation(self):
		"""Test entity response container"""
		entity_data = EntityDetailView(
			entity_id=uuid7str(),
			tenant_id="test-tenant",
			entity_type=EntityType.PERSON,
			entity_name="Test Entity",
			business_key="TEST-001",
			source_system="test",
			status=EntityStatus.ACTIVE,
			quality_score=85.0,
			data_classification="internal",
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			created_by="test",
			updated_by="test"
		)
		
		response = EntityResponse(
			success=True,
			message="Entity retrieved successfully",
			data=entity_data,
			request_id="req-123"
		)
		
		assert response.success is True
		assert response.data is not None
		assert response.data.entity_name == "Test Entity"
		assert response.request_id == "req-123"
	
	def test_entity_list_response_creation(self):
		"""Test entity list response container"""
		entities = [
			EntitySummaryView(
				entity_id=uuid7str(),
				entity_type=EntityType.PERSON,
				entity_name=f"Entity {i}",
				business_key=f"ENT-{i:03d}",
				source_system="test",
				status=EntityStatus.ACTIVE,
				quality_score=80.0,
				data_classification="internal",
				created_at=datetime.utcnow(),
				updated_at=datetime.utcnow()
			)
			for i in range(1, 4)
		]
		
		search_result = EntitySearchResultView(
			entities=entities,
			pagination=PaginationMeta(
				total_count=3,
				offset=0,
				limit=10,
				has_next=False,
				has_previous=False
			),
			search_criteria={"entity_type": "person"}
		)
		
		response = EntityListResponse(
			success=True,
			message="Search completed successfully",
			data=search_result
		)
		
		assert response.success is True
		assert response.data is not None
		assert len(response.data.entities) == 3
		assert response.data.pagination.total_count == 3
	
	def test_quality_assessment_response_creation(self):
		"""Test quality assessment response container"""
		quality_data = QualityAssessmentView(
			assessment_id=uuid7str(),
			entity_id=uuid7str(),
			tenant_id="test-tenant",
			overall_score=85.0,
			quality_status=DataQualityStatus.GOOD,
			completeness_score=90.0,
			accuracy_score=80.0,
			consistency_score=85.0,
			validity_score=85.0,
			uniqueness_score=95.0,
			timeliness_score=75.0,
			assessment_timestamp=datetime.utcnow()
		)
		
		response = QualityAssessmentResponse(
			success=True,
			message="Quality assessment completed",
			data=quality_data
		)
		
		assert response.success is True
		assert response.data is not None
		assert response.data.overall_score == 85.0
		assert response.data.quality_status == DataQualityStatus.GOOD