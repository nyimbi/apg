"""
APG Facial Recognition - Pydantic Views and API Models

Comprehensive Pydantic v2 models for API validation, request/response schemas,
and data serialization with strict validation and security measures.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from pydantic.types import EmailStr, PositiveFloat, PositiveInt

# Configuration for all models
class BaseConfig:
    model_config = ConfigDict(
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True,
        str_strip_whitespace=True,
        frozen=False
    )

# Enums for consistent validation
class EnrollmentStatus(str, Enum):
    NOT_ENROLLED = "not_enrolled"
    PENDING_ENROLLMENT = "pending_enrollment"
    ENROLLED = "enrolled"
    ENROLLMENT_FAILED = "enrollment_failed"
    EXPIRED = "expired"

class VerificationTypeEnum(str, Enum):
    AUTHENTICATION = "authentication"
    IDENTIFICATION = "identification"
    WATCHLIST = "watchlist"

class ProcessingStatusEnum(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class LivenessResultEnum(str, Enum):
    LIVE = "live"
    SPOOF = "spoof"
    UNKNOWN = "unknown"

class EmotionTypeEnum(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    CONTEMPT = "contempt"
    CONFUSED = "confused"

class PrivacyLevelEnum(str, Enum):
    BASIC = "basic"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"
    ZERO_KNOWLEDGE = "zero_knowledge"

class ProcessingModeEnum(str, Enum):
    ON_DEVICE = "on_device"
    FEDERATED = "federated"
    HOMOMORPHIC = "homomorphic"
    DIFFERENTIAL_PRIVATE = "differential_private"

# Base Models
class TimestampedModel(BaseModel, BaseConfig):
    """Base model with timestamp fields"""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None

class TenantModel(BaseModel, BaseConfig):
    """Base model with tenant isolation"""
    tenant_id: str = Field(..., min_length=1, max_length=100)

# User Management Models
class UserCreateRequest(BaseModel, BaseConfig):
    """Request model for creating a new user"""
    external_user_id: str = Field(..., min_length=1, max_length=100)
    full_name: str = Field(..., min_length=1, max_length=200)
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = Field(None, max_length=20)
    consent_given: bool = Field(default=False)
    privacy_settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class UserUpdateRequest(BaseModel, BaseConfig):
    """Request model for updating a user"""
    full_name: Optional[str] = Field(None, min_length=1, max_length=200)
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = Field(None, max_length=20)
    consent_given: Optional[bool] = None
    privacy_settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class UserResponse(TimestampedModel, TenantModel):
    """Response model for user data"""
    id: str = Field(default_factory=uuid7str)
    external_user_id: str
    full_name: str
    email: Optional[str] = None
    phone_number: Optional[str] = None
    enrollment_status: EnrollmentStatus = EnrollmentStatus.NOT_ENROLLED
    consent_given: bool = False
    consent_timestamp: Optional[datetime] = None
    privacy_settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    last_verification: Optional[datetime] = None

# Enrollment Models
class EnrollmentRequest(BaseModel, BaseConfig):
    """Request model for face enrollment"""
    user_id: str = Field(..., min_length=1)
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to image file")
    enrollment_type: str = Field(default="standard")
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    device_info: Optional[Dict[str, Any]] = None
    location_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode='after')
    def validate_image_source(self):
        """Ensure either image_data or image_url is provided"""
        if not self.image_data and not self.image_url:
            raise ValueError("Either image_data or image_url must be provided")
        return self

class EnrollmentResponse(TimestampedModel):
    """Response model for enrollment result"""
    success: bool
    enrollment_id: Optional[str] = None
    template_id: Optional[str] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time_ms: Optional[PositiveFloat] = None
    error: Optional[str] = None
    enrollment_timestamp: Optional[datetime] = None

# Verification Models
class VerificationRequest(BaseModel, BaseConfig):
    """Request model for face verification"""
    user_id: str = Field(..., min_length=1)
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to image file")
    verification_type: VerificationTypeEnum = VerificationTypeEnum.AUTHENTICATION
    require_liveness: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    business_context: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None
    location_data: Optional[Dict[str, Any]] = None
    privacy_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode='after')
    def validate_image_source(self):
        """Ensure either image_data or image_url is provided"""
        if not self.image_data and not self.image_url:
            raise ValueError("Either image_data or image_url must be provided")
        return self

class VerificationResponse(TimestampedModel):
    """Response model for verification result"""
    success: bool
    verified: bool = False
    verification_id: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    liveness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    liveness_result: Optional[LivenessResultEnum] = None
    processing_time_ms: Optional[PositiveFloat] = None
    failure_reason: Optional[str] = None
    risk_analysis: Optional[Dict[str, Any]] = None
    emotion_analysis: Optional[Dict[str, Any]] = None
    contextual_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    verification_timestamp: Optional[datetime] = None

# Identification Models
class IdentificationRequest(BaseModel, BaseConfig):
    """Request model for face identification (1:N matching)"""
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to image file")
    max_candidates: PositiveInt = Field(default=10, le=50)
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    search_scope: Optional[List[str]] = Field(None, description="List of user IDs to search")
    device_info: Optional[Dict[str, Any]] = None
    location_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode='after')
    def validate_image_source(self):
        """Ensure either image_data or image_url is provided"""
        if not self.image_data and not self.image_url:
            raise ValueError("Either image_data or image_url must be provided")
        return self

class IdentificationCandidate(BaseModel, BaseConfig):
    """Model for identification candidate"""
    user_id: str
    external_user_id: str
    full_name: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    template_id: str
    rank: PositiveInt

class IdentificationResponse(TimestampedModel):
    """Response model for identification result"""
    success: bool
    candidates: List[IdentificationCandidate] = []
    query_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time_ms: Optional[PositiveFloat] = None
    search_timestamp: Optional[datetime] = None
    error: Optional[str] = None

# Emotion Analysis Models
class EmotionAnalysisRequest(BaseModel, BaseConfig):
    """Request model for emotion analysis"""
    verification_id: Optional[str] = None
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to image file")
    video_frames: Optional[List[str]] = Field(None, description="List of base64 encoded frames")
    analysis_config: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None

class EmotionScores(BaseModel, BaseConfig):
    """Model for emotion scores"""
    neutral: float = Field(..., ge=0.0, le=1.0)
    happy: float = Field(..., ge=0.0, le=1.0)
    sad: float = Field(..., ge=0.0, le=1.0)
    angry: float = Field(..., ge=0.0, le=1.0)
    fearful: float = Field(..., ge=0.0, le=1.0)
    disgusted: float = Field(..., ge=0.0, le=1.0)
    surprised: float = Field(..., ge=0.0, le=1.0)
    contempt: float = Field(..., ge=0.0, le=1.0)
    confused: float = Field(..., ge=0.0, le=1.0)

class StressAnalysis(BaseModel, BaseConfig):
    """Model for stress analysis results"""
    overall_stress_level: str = Field(..., pattern="^(low|medium|high)$")
    stress_score: float = Field(..., ge=0.0, le=1.0)
    stress_indicators: List[str] = []
    physiological_markers: Dict[str, float] = {}

class EmotionAnalysisResponse(TimestampedModel):
    """Response model for emotion analysis"""
    success: bool = True
    analysis_id: str = Field(default_factory=uuid7str)
    primary_emotion: EmotionTypeEnum
    emotion_confidence: float = Field(..., ge=0.0, le=1.0)
    emotion_scores: EmotionScores
    stress_analysis: StressAnalysis
    micro_expressions: List[Dict[str, Any]] = []
    behavioral_insights: Optional[Dict[str, Any]] = None
    risk_indicators: List[str] = []
    recommendations: List[str] = []
    processing_time_ms: Optional[PositiveFloat] = None
    error: Optional[str] = None

# Collaborative Verification Models
class CollaborationRequest(BaseModel, BaseConfig):
    """Request model for collaborative verification"""
    verification_id: str = Field(..., min_length=1)
    workflow_type: str = Field(default="standard_approval")
    required_approvals: PositiveInt = Field(default=1, le=10)
    timeout_minutes: PositiveInt = Field(default=30, le=240)
    participant_roles: List[str] = []
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ParticipantResponse(BaseModel, BaseConfig):
    """Model for participant response"""
    collaboration_id: str = Field(..., min_length=1)
    participant_id: str = Field(..., min_length=1)
    decision: str = Field(..., pattern="^(approve|reject)$")
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    comments: Optional[str] = None
    additional_evidence: Optional[Dict[str, Any]] = None

class CollaborationStatus(BaseModel, BaseConfig):
    """Model for collaboration status"""
    collaboration_id: str
    status: str
    workflow_type: str
    participants: int
    responses_received: int
    approvals: int
    rejections: int
    consensus_score: float = Field(..., ge=0.0, le=1.0)
    final_decision: Optional[str] = None
    timeout_at: datetime
    initiated_at: datetime
    completed_at: Optional[datetime] = None

class CollaborationResponse(TimestampedModel):
    """Response model for collaboration operations"""
    success: bool
    collaboration_id: str
    status: str
    message: Optional[str] = None
    participants_invited: Optional[int] = None
    timeout_at: Optional[datetime] = None
    error: Optional[str] = None

# Privacy and Consent Models
class ConsentRequest(BaseModel, BaseConfig):
    """Request model for managing user consent"""
    user_id: str = Field(..., min_length=1)
    consent_given: bool
    consent_method: str = Field(default="explicit")
    allowed_purposes: List[str] = ["identity_verification"]
    allowed_data_categories: List[str] = ["facial_biometric"]
    consent_expiry: Optional[datetime] = None
    granular_control: Optional[Dict[str, bool]] = None
    legal_basis: str = Field(default="consent")

class ConsentResponse(TimestampedModel):
    """Response model for consent operations"""
    success: bool
    consent_id: str = Field(default_factory=uuid7str)
    consent_status: str
    data_subject_rights: Dict[str, bool]
    error: Optional[str] = None

class PrivacyProcessingRequest(BaseModel, BaseConfig):
    """Request model for privacy-preserving processing"""
    user_id: str = Field(..., min_length=1)
    biometric_data: str = Field(..., description="Base64 encoded biometric data")
    privacy_level: PrivacyLevelEnum = PrivacyLevelEnum.ENHANCED
    processing_mode: ProcessingModeEnum = ProcessingModeEnum.FEDERATED
    processing_purpose: str = Field(default="identity_verification")
    data_categories: List[str] = ["facial_biometric"]
    retention_policy: str = Field(default="short_term")
    legal_basis: str = Field(default="consent")

class PrivacyProcessingResponse(TimestampedModel):
    """Response model for privacy processing"""
    success: bool
    processing_id: str = Field(default_factory=uuid7str)
    privacy_metadata: Dict[str, Any]
    processing_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class DataSubjectRequest(BaseModel, BaseConfig):
    """Request model for data subject rights (GDPR)"""
    user_id: str = Field(..., min_length=1)
    request_type: str = Field(..., pattern="^(access|rectification|erasure|portability|objection|restriction)$")
    request_reason: Optional[str] = None
    specific_data_categories: Optional[List[str]] = None
    verification_method: str = Field(default="identity_verification")

class DataSubjectResponse(TimestampedModel):
    """Response model for data subject rights requests"""
    success: bool
    request_id: str = Field(default_factory=uuid7str)
    request_type: str
    processing_time_days: int = Field(default=30)
    data_export: Optional[Dict[str, Any]] = None
    completion_status: Optional[str] = None
    error: Optional[str] = None

# Analytics and Reporting Models
class AnalyticsRequest(BaseModel, BaseConfig):
    """Request model for analytics data"""
    metric_type: str = Field(..., min_length=1)
    time_period: str = Field(default="last_30_days")
    filters: Optional[Dict[str, Any]] = None
    group_by: Optional[List[str]] = None
    aggregation: str = Field(default="count")

class AnalyticsMetric(BaseModel, BaseConfig):
    """Model for individual analytics metric"""
    name: str
    value: Union[int, float, str]
    unit: Optional[str] = None
    trend: Optional[str] = None
    change_percentage: Optional[float] = None

class AnalyticsResponse(TimestampedModel):
    """Response model for analytics data"""
    success: bool
    metrics: List[AnalyticsMetric]
    time_period: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None

# System Configuration Models
class SystemSettingsRequest(BaseModel, BaseConfig):
    """Request model for system settings"""
    category: str = Field(..., min_length=1)
    settings: Dict[str, Any]
    apply_immediately: bool = Field(default=True)

class SystemSettingsResponse(BaseModel, BaseConfig):
    """Response model for system settings"""
    success: bool
    updated_settings: Dict[str, Any]
    requires_restart: bool = Field(default=False)
    error: Optional[str] = None

class HealthCheckResponse(BaseModel, BaseConfig):
    """Response model for system health check"""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    version: str
    uptime_seconds: PositiveInt
    components: Dict[str, str]
    metrics: Dict[str, Union[int, float]]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Batch Processing Models
class BatchProcessingRequest(BaseModel, BaseConfig):
    """Request model for batch operations"""
    operation_type: str = Field(..., min_length=1)
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000)
    processing_options: Optional[Dict[str, Any]] = None
    callback_url: Optional[str] = None
    priority: str = Field(default="normal", pattern="^(low|normal|high|critical)$")

class BatchProcessingResponse(TimestampedModel):
    """Response model for batch operations"""
    success: bool
    batch_id: str = Field(default_factory=uuid7str)
    total_items: PositiveInt
    status: str = Field(default="queued")
    estimated_completion: Optional[datetime] = None
    error: Optional[str] = None

class BatchStatusResponse(BaseModel, BaseConfig):
    """Response model for batch status"""
    batch_id: str
    status: str
    total_items: PositiveInt
    processed_items: int = Field(default=0, ge=0)
    successful_items: int = Field(default=0, ge=0)
    failed_items: int = Field(default=0, ge=0)
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_summary: Optional[Dict[str, Any]] = None

# Error Response Models
class ErrorDetail(BaseModel, BaseConfig):
    """Model for error details"""
    code: str
    message: str
    field: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel, BaseConfig):
    """Standard error response model"""
    success: bool = False
    error: str
    error_code: str
    details: List[ErrorDetail] = []
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = None

# Validation Models
class ValidationResult(BaseModel, BaseConfig):
    """Model for validation results"""
    is_valid: bool
    score: float = Field(..., ge=0.0, le=1.0)
    issues: List[str] = []
    recommendations: List[str] = []

class ImageQualityResponse(BaseModel, BaseConfig):
    """Response model for image quality assessment"""
    success: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    resolution_score: float = Field(..., ge=0.0, le=1.0)
    sharpness_score: float = Field(..., ge=0.0, le=1.0)
    brightness_score: float = Field(..., ge=0.0, le=1.0)
    contrast_score: float = Field(..., ge=0.0, le=1.0)
    pose_score: float = Field(..., ge=0.0, le=1.0)
    occlusion_score: float = Field(..., ge=0.0, le=1.0)
    quality_issues: List[str] = []
    usable_for_recognition: bool
    error: Optional[str] = None

# List and Pagination Models
class PaginationRequest(BaseModel, BaseConfig):
    """Request model for pagination"""
    page: PositiveInt = Field(default=1)
    page_size: PositiveInt = Field(default=20, le=100)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    filters: Optional[Dict[str, Any]] = None

class PaginationMeta(BaseModel, BaseConfig):
    """Model for pagination metadata"""
    page: PositiveInt
    page_size: PositiveInt
    total_items: int = Field(..., ge=0)
    total_pages: PositiveInt
    has_next: bool
    has_previous: bool

class PaginatedResponse(BaseModel, BaseConfig):
    """Generic paginated response model"""
    items: List[Dict[str, Any]]
    pagination: PaginationMeta
    success: bool = True
    error: Optional[str] = None

# Export all models for easy import
__all__ = [
    # Enums
    'EnrollmentStatus', 'VerificationTypeEnum', 'ProcessingStatusEnum',
    'LivenessResultEnum', 'EmotionTypeEnum', 'PrivacyLevelEnum', 'ProcessingModeEnum',
    
    # Base Models
    'TimestampedModel', 'TenantModel', 'BaseConfig',
    
    # User Models
    'UserCreateRequest', 'UserUpdateRequest', 'UserResponse',
    
    # Enrollment Models
    'EnrollmentRequest', 'EnrollmentResponse',
    
    # Verification Models
    'VerificationRequest', 'VerificationResponse',
    
    # Identification Models
    'IdentificationRequest', 'IdentificationCandidate', 'IdentificationResponse',
    
    # Emotion Models
    'EmotionAnalysisRequest', 'EmotionScores', 'StressAnalysis', 'EmotionAnalysisResponse',
    
    # Collaboration Models
    'CollaborationRequest', 'ParticipantResponse', 'CollaborationStatus', 'CollaborationResponse',
    
    # Privacy Models
    'ConsentRequest', 'ConsentResponse', 'PrivacyProcessingRequest', 'PrivacyProcessingResponse',
    'DataSubjectRequest', 'DataSubjectResponse',
    
    # Analytics Models
    'AnalyticsRequest', 'AnalyticsMetric', 'AnalyticsResponse',
    
    # System Models
    'SystemSettingsRequest', 'SystemSettingsResponse', 'HealthCheckResponse',
    
    # Batch Models
    'BatchProcessingRequest', 'BatchProcessingResponse', 'BatchStatusResponse',
    
    # Error Models
    'ErrorDetail', 'ErrorResponse',
    
    # Validation Models
    'ValidationResult', 'ImageQualityResponse',
    
    # Pagination Models
    'PaginationRequest', 'PaginationMeta', 'PaginatedResponse'
]