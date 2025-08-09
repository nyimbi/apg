#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Views Layer
Pydantic models for API serialization and data transformation

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import re

from pydantic import BaseModel, Field, ConfigDict, validator, root_validator
from pydantic.types import EmailStr, UUID4
from uuid_extensions import uuid7str

from .models import EntityType, EntityStatus, DataQualityStatus, MatchConfidence


# Base Response Models

class MDMBaseResponse(BaseModel):
    """Base response model with common fields"""
    model_config = ConfigDict(
        extra='forbid', 
        validate_by_name=True,
        validate_by_alias=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class PaginationMeta(BaseModel):
    """Pagination metadata"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    total_count: int = Field(..., ge=0)
    offset: int = Field(..., ge=0)
    limit: int = Field(..., ge=1, le=1000)
    has_next: bool
    has_previous: bool
    total_pages: Optional[int] = None
    current_page: Optional[int] = None
    
    @validator('total_pages', always=True)
    def calculate_total_pages(cls, v, values):
        if 'limit' in values and 'total_count' in values:
            import math
            return math.ceil(values['total_count'] / values['limit'])
        return v
    
    @validator('current_page', always=True)
    def calculate_current_page(cls, v, values):
        if 'limit' in values and 'offset' in values:
            import math
            return math.floor(values['offset'] / values['limit']) + 1
        return v


# Entity Views

class EntitySummaryView(BaseModel):
    """Lightweight entity view for list operations"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    entity_id: str
    entity_type: EntityType
    entity_name: str
    business_key: str
    source_system: str
    status: EntityStatus
    quality_score: float = Field(..., ge=0.0, le=100.0)
    is_golden_record: bool = False
    data_classification: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = Field(default_factory=list, max_items=10)


class EntityDetailView(BaseModel):
    """Detailed entity view with full information"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    entity_id: str
    tenant_id: str
    entity_type: EntityType
    entity_name: str
    entity_description: Optional[str] = None
    business_key: str
    source_system: str
    status: EntityStatus
    attributes: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    data_classification: str
    quality_score: float = Field(..., ge=0.0, le=100.0)
    last_quality_check: Optional[datetime] = None
    is_golden_record: bool = False
    golden_record_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    audit_trail_id: Optional[str] = None
    
    # Optional nested data
    versions: Optional[List['EntityVersionView']] = None
    quality_assessment: Optional['QualityAssessmentView'] = None
    cross_references: Optional[List['CrossReferenceView']] = None
    duplicate_candidates: Optional[List['DuplicateCandidateView']] = None


class EntityVersionView(BaseModel):
    """Entity version history view"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    version_id: str
    version_number: int = Field(..., ge=1)
    version_timestamp: datetime
    version_type: str  # create, update, merge, split, delete
    created_by: str
    change_description: Optional[str] = None
    changed_fields: List[str] = Field(default_factory=list)
    quality_score_snapshot: Optional[float] = Field(None, ge=0.0, le=100.0)
    change_source: Optional[str] = None


class EntitySearchResultView(BaseModel):
    """Entity search results with pagination"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    entities: List[EntitySummaryView]
    pagination: PaginationMeta
    search_criteria: Dict[str, Any]
    search_duration_ms: Optional[float] = None
    total_quality_score_avg: Optional[float] = Field(None, ge=0.0, le=100.0)
    entity_type_breakdown: Optional[Dict[str, int]] = None


# Quality Assessment Views

class QualityIssueView(BaseModel):
    """Individual data quality issue"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    issue_type: str  # completeness, accuracy, consistency, validity, uniqueness, timeliness
    field: str
    severity: str  # low, medium, high, critical
    message: str
    recommendation: Optional[str] = None
    auto_fixable: bool = False
    
    @validator('severity')
    def validate_severity(cls, v):
        allowed = ['low', 'medium', 'high', 'critical']
        if v not in allowed:
            raise ValueError(f'Severity must be one of: {", ".join(allowed)}')
        return v


class QualityAssessmentView(BaseModel):
    """Comprehensive quality assessment view"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    assessment_id: str
    entity_id: str
    tenant_id: str
    overall_score: float = Field(..., ge=0.0, le=100.0)
    quality_status: DataQualityStatus
    
    # Dimension scores
    completeness_score: float = Field(..., ge=0.0, le=100.0)
    accuracy_score: float = Field(..., ge=0.0, le=100.0)
    consistency_score: float = Field(..., ge=0.0, le=100.0)
    validity_score: float = Field(..., ge=0.0, le=100.0)
    uniqueness_score: float = Field(..., ge=0.0, le=100.0)
    timeliness_score: float = Field(..., ge=0.0, le=100.0)
    
    # Assessment metadata
    assessment_timestamp: datetime
    assessment_duration_ms: Optional[float] = Field(None, ge=0.0)
    assessment_algorithm: str = "ai_enhanced"
    algorithm_version: str = "1.0.0"
    
    # Issues and recommendations
    quality_issues: List[QualityIssueView] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    priority_issues: List[str] = Field(default_factory=list)
    auto_fix_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Trend analysis (if available)
    score_trend: Optional[Dict[str, float]] = None  # previous scores for trend analysis
    improvement_suggestions: Optional[List[str]] = None


class QualityBatchAssessmentView(BaseModel):
    """Batch quality assessment results"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    batch_id: str
    tenant_id: str
    assessments: List[QualityAssessmentView]
    summary: Dict[str, Any]
    assessment_duration_total_ms: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Duplicate Detection Views

class DuplicateCandidateView(BaseModel):
    """Duplicate candidate match result"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    candidate_id: str
    candidate_name: str
    candidate_business_key: str
    candidate_source_system: str
    match_score: float = Field(..., ge=0.0, le=100.0)
    confidence: MatchConfidence
    matching_attributes: List[str] = Field(default_factory=list)
    similarity_details: Dict[str, float] = Field(default_factory=dict)
    recommended_action: str  # merge, review, ignore, investigate
    match_explanation: Optional[str] = None
    last_updated: Optional[datetime] = None
    
    @validator('recommended_action')
    def validate_action(cls, v):
        allowed = ['merge', 'review', 'ignore', 'investigate']
        if v not in allowed:
            raise ValueError(f'Recommended action must be one of: {", ".join(allowed)}')
        return v


class DuplicateDetectionResultView(BaseModel):
    """Complete duplicate detection results"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    detection_id: str
    entity_id: str
    entity_name: str
    tenant_id: str
    total_candidates: int = Field(..., ge=0)
    high_confidence_matches: int = Field(..., ge=0)
    medium_confidence_matches: int = Field(..., ge=0)
    low_confidence_matches: int = Field(..., ge=0)
    match_candidates: List[DuplicateCandidateView] = Field(default_factory=list)
    detection_timestamp: datetime
    detection_duration_ms: Optional[float] = Field(None, ge=0.0)
    algorithm_version: str = "1.0.0"
    detection_rules_applied: List[str] = Field(default_factory=list)
    next_review_date: Optional[datetime] = None


# Golden Record Views

class GoldenRecordView(BaseModel):
    """Golden record view with consolidation details"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    golden_record_id: str
    tenant_id: str
    entity_type: EntityType
    golden_record_name: str
    business_key: str
    consolidated_attributes: Dict[str, Any] = Field(default_factory=dict)
    source_entity_ids: List[str] = Field(default_factory=list)
    
    # Quality and confidence metrics
    overall_quality_score: float = Field(..., ge=0.0, le=100.0)
    consolidation_confidence: float = Field(..., ge=0.0, le=100.0)
    data_completeness: float = Field(..., ge=0.0, le=100.0)
    
    # Survivorship information
    survivorship_rules: Dict[str, Any] = Field(default_factory=dict)
    consolidation_method: str = "ai_determined"
    
    # Lifecycle
    created_at: datetime
    updated_at: datetime
    last_consolidation: datetime
    created_by: str
    
    # Status and approval
    is_active: bool = True
    approval_status: str = "auto_approved"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Related data
    contributing_entities: Optional[List[EntitySummaryView]] = None
    quality_assessment: Optional[QualityAssessmentView] = None


# Cross-Reference Views

class CrossReferenceView(BaseModel):
    """Cross-system reference mapping view"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    cross_reference_id: str
    entity_id: str
    source_system: str
    source_entity_id: str
    source_entity_type: Optional[str] = None
    confidence_score: float = Field(..., ge=0.0, le=100.0)
    is_primary_reference: bool = False
    reference_quality: str = "high"
    created_at: datetime
    updated_at: datetime
    last_verified: Optional[datetime] = None
    created_by: str
    is_active: bool = True
    verification_method: Optional[str] = None
    
    @validator('reference_quality')
    def validate_quality(cls, v):
        allowed = ['low', 'medium', 'high', 'excellent']
        if v not in allowed:
            raise ValueError(f'Reference quality must be one of: {", ".join(allowed)}')
        return v


# Audit and Lineage Views

class AuditLogView(BaseModel):
    """Audit log entry view"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    audit_id: str
    tenant_id: str
    event_type: str
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    event_timestamp: datetime
    event_description: str
    event_details: Dict[str, Any] = Field(default_factory=dict)
    user_id: str
    user_name: Optional[str] = None
    source_system: Optional[str] = None
    client_ip: Optional[str] = None
    operation_id: Optional[str] = None
    data_sensitivity: str = "internal"
    compliance_tags: List[str] = Field(default_factory=list)


class DataLineageView(BaseModel):
    """Data lineage relationship view"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    lineage_id: str
    source_entity_id: str
    source_entity_name: str
    target_entity_id: str
    target_entity_name: str
    relationship_type: str  # derived_from, merged_into, split_from, etc.
    transformation_type: str
    transformation_details: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(..., ge=0.0, le=100.0)
    created_at: datetime
    created_by: str
    is_verified: bool = False
    verification_method: Optional[str] = None


# Analytics and Reporting Views

class EntityStatisticsView(BaseModel):
    """Entity statistics and metrics"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    tenant_id: str
    total_entities: int = Field(..., ge=0)
    entities_by_type: Dict[str, int] = Field(default_factory=dict)
    entities_by_status: Dict[str, int] = Field(default_factory=dict)
    entities_by_source: Dict[str, int] = Field(default_factory=dict)
    average_quality_score: float = Field(..., ge=0.0, le=100.0)
    quality_distribution: Dict[str, int] = Field(default_factory=dict)  # excellent, good, fair, poor, critical
    golden_records_count: int = Field(..., ge=0)
    duplicate_candidates_count: int = Field(..., ge=0)
    data_freshness_stats: Dict[str, Any] = Field(default_factory=dict)
    growth_trends: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class QualityTrendsView(BaseModel):
    """Quality trends and analytics"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    tenant_id: str
    time_period: str  # daily, weekly, monthly
    trend_data: List[Dict[str, Any]] = Field(default_factory=list)
    overall_trend: str  # improving, declining, stable
    trend_percentage: float  # positive or negative change
    quality_dimension_trends: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    top_quality_issues: List[Dict[str, Any]] = Field(default_factory=list)
    improvement_recommendations: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# Bulk Operation Views

class BulkOperationStatusView(BaseModel):
    """Bulk operation status and results"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    operation_id: str
    tenant_id: str
    operation_type: str  # create, update, delete, merge
    status: str  # pending, processing, completed, failed, partial
    total_items: int = Field(..., ge=0)
    processed_items: int = Field(..., ge=0)
    successful_items: int = Field(..., ge=0)
    failed_items: int = Field(..., ge=0)
    
    # Operation details
    started_at: datetime
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    progress_percentage: float = Field(..., ge=0.0, le=100.0)
    
    # Results
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance metrics
    processing_rate_per_second: Optional[float] = Field(None, ge=0.0)
    estimated_time_remaining_seconds: Optional[float] = Field(None, ge=0.0)


# Configuration Views

class MatchRuleView(BaseModel):
    """Matching rule configuration view"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    rule_id: str
    tenant_id: str
    rule_name: str
    rule_description: Optional[str] = None
    entity_type: EntityType
    rule_config: Dict[str, Any] = Field(default_factory=dict)
    matching_attributes: List[str] = Field(default_factory=list)
    weight_config: Dict[str, float] = Field(default_factory=dict)
    
    # Thresholds
    exact_match_threshold: float = Field(..., ge=0.0, le=100.0)
    high_confidence_threshold: float = Field(..., ge=0.0, le=100.0)
    medium_confidence_threshold: float = Field(..., ge=0.0, le=100.0)
    minimum_match_threshold: float = Field(..., ge=0.0, le=100.0)
    
    # Performance metrics
    rule_version: str = "1.0.0"
    performance_stats: Dict[str, float] = Field(default_factory=dict)
    last_performance_review: Optional[datetime] = None
    
    # Status
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    created_by: str


class SurvivorshipRuleView(BaseModel):
    """Survivorship rule configuration view"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    rule_id: str
    tenant_id: str
    rule_name: str
    rule_description: Optional[str] = None
    entity_type: EntityType
    survivorship_strategy: str = "ai_determined"
    attribute_rules: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    source_system_rankings: Dict[str, float] = Field(default_factory=dict)
    rule_conditions: Dict[str, Any] = Field(default_factory=dict)
    fallback_strategy: str = "most_recent"
    conflict_resolution: Dict[str, Any] = Field(default_factory=dict)
    rule_effectiveness: float = Field(..., ge=0.0, le=100.0)
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    created_by: str


# Response Containers

class EntityResponse(MDMBaseResponse):
    """Single entity response"""
    data: Optional[EntityDetailView] = None


class EntityListResponse(MDMBaseResponse):
    """Entity list response with pagination"""
    data: Optional[EntitySearchResultView] = None


class QualityAssessmentResponse(MDMBaseResponse):
    """Quality assessment response"""
    data: Optional[QualityAssessmentView] = None


class QualityBatchResponse(MDMBaseResponse):
    """Batch quality assessment response"""
    data: Optional[QualityBatchAssessmentView] = None


class DuplicateDetectionResponse(MDMBaseResponse):
    """Duplicate detection response"""
    data: Optional[DuplicateDetectionResultView] = None


class GoldenRecordResponse(MDMBaseResponse):
    """Golden record response"""
    data: Optional[GoldenRecordView] = None


class BulkOperationResponse(MDMBaseResponse):
    """Bulk operation response"""
    data: Optional[BulkOperationStatusView] = None


class AnalyticsResponse(MDMBaseResponse):
    """Analytics and statistics response"""
    data: Optional[Union[EntityStatisticsView, QualityTrendsView]] = None


# Update forward references
EntityDetailView.model_rebuild()


# Export all view classes
__all__ = [
    # Base models
    'MDMBaseResponse', 'PaginationMeta',
    
    # Entity views
    'EntitySummaryView', 'EntityDetailView', 'EntityVersionView', 'EntitySearchResultView',
    
    # Quality views
    'QualityIssueView', 'QualityAssessmentView', 'QualityBatchAssessmentView',
    
    # Duplicate detection views
    'DuplicateCandidateView', 'DuplicateDetectionResultView',
    
    # Golden record views
    'GoldenRecordView',
    
    # Cross-reference views
    'CrossReferenceView',
    
    # Audit and lineage views
    'AuditLogView', 'DataLineageView',
    
    # Analytics views
    'EntityStatisticsView', 'QualityTrendsView',
    
    # Bulk operation views
    'BulkOperationStatusView',
    
    # Configuration views
    'MatchRuleView', 'SurvivorshipRuleView',
    
    # Response containers
    'EntityResponse', 'EntityListResponse', 'QualityAssessmentResponse',
    'QualityBatchResponse', 'DuplicateDetectionResponse', 'GoldenRecordResponse',
    'BulkOperationResponse', 'AnalyticsResponse'
]