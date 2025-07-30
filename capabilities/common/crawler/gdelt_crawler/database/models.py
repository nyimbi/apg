"""
GDELT Database Models and Schema Definitions
============================================

SQLAlchemy models and database schema definitions for GDELT data storage
compatible with the information_units table structure and ML Deep Scorer
integration requirements.

Key Features:
- **Information Units Compatibility**: Full compatibility with existing schema
- **ML Deep Scorer Integration**: Support for all ML extraction fields
- **GDELT-Specific Fields**: Specialized fields for GDELT data types
- **Performance Optimization**: Proper indexing and constraints
- **Type Safety**: Comprehensive type definitions and validation
- **Relationship Management**: Foreign key relationships and constraints

Supported Models:
- **InformationUnit**: Main model for GDELT records
- **DataSource**: Data source definitions and metadata
- **GDELTEvent**: GDELT-specific event model extensions
- **ProcessingLog**: ETL processing tracking and monitoring

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import (UUID, JSONB, TIMESTAMP, ARRAY, BYTEA, BIGINT, NUMERIC)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid
from geoalchemy2 import Geometry
from sqlalchemy import Date, Time

# Base class for all models
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class DataSource(Base):
    """Data source model for tracking GDELT and other sources."""

    __tablename__ = 'data_sources'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    source_type = Column(String(50))
    base_url = Column(Text)
    api_endpoint = Column(Text)
    api_key = Column(String(255))
    crawl_depth = Column(Integer, default=5)
    country = Column(String(100))
    auth_config = Column(JSONB)
    rate_limit_per_hour = Column(Integer)
    data_format = Column(String(20))
    reliability_score = Column(NUMERIC(3, 2), default=0.5)
    bias_score = Column(Integer, default=0)
    collection_frequency_hours = Column(Integer, default=24)
    health_status = Column(String(20), default='unhealthy')
    last_health_check = Column(TIMESTAMP(timezone=True))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    description = Column(String(250))

    information_units = relationship("InformationUnit", back_populates="data_source")

    def __repr__(self):
        return f"<DataSource(name='{self.name}', reliability={self.reliability_score})>"


class InformationUnit(Base):
    """
    Main model for GDELT information units compatible with ML Deep Scorer.
    This model includes all fields from the information_units table schema
    including ML Deep Scorer event extraction fields and thinking traces.
    """

    __tablename__ = "information_units"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    data_source_id = Column(UUID(as_uuid=True), ForeignKey("data_sources.id"))
    parent_unit_id = Column(
        UUID(as_uuid=True), ForeignKey("information_units.id", ondelete="SET NULL")
    )
    external_id = Column(String(255), unique=True)
    unit_type = Column(String(50), nullable=False)
    title = Column(Text)
    authors = Column(ARRAY(Text))
    publication_name = Column(String(255))
    content = Column(Text)
    content_url = Column(Text, unique=True)
    url_hash = Column(String(250))
    local_storage_path = Column(Text)
    content_hash = Column(String(64))
    summary = Column(Text)
    keywords = Column(ARRAY(Text))
    tags = Column(ARRAY(Text))
    source_domain = Column(String(255))
    language_code = Column(String(10))
    detected_language_code = Column(String(10))
    translation_source_language_code = Column(String(10))
    published_at = Column(TIMESTAMP(timezone=True))
    discovered_at = Column(TIMESTAMP(timezone=True), default=func.now())
    ingested_at = Column(TIMESTAMP(timezone=True), default=func.now())
    scraped_at = Column(TIMESTAMP(timezone=True))
    processed_at = Column(TIMESTAMP(timezone=True))
    capture_method = Column(String(50))
    scraper_name = Column(String(100))
    scraper_version = Column(String(50))
    scrape_config = Column(JSONB)
    raw_content_snapshot = Column(Text)
    http_status_code = Column(Integer)
    response_headers = Column(JSONB)
    social_media_platform = Column(String(50))
    social_media_user_handle = Column(String(150))
    social_media_user_display_name = Column(String(255))
    social_media_user_id_str = Column(String(100))
    likes_count = Column(Integer)
    shares_count = Column(Integer)
    comments_count = Column(Integer)
    views_count = Column(Integer)
    parent_entry_external_id = Column(String(255))
    quoted_entry_external_id = Column(String(255))
    thread_id = Column(String(255))
    is_reply = Column(Boolean)
    is_quote = Column(Boolean)
    is_retweet = Column(Boolean)
    hashtags = Column(ARRAY(Text))
    user_mentions = Column(ARRAY(Text))
    attached_media_details = Column(JSONB)
    article_section = Column(String(100))
    edition = Column(String(50))
    page_numbers = Column(String(50))
    paywall_status = Column(String(20), default="unknown")
    relevance_score = Column(NUMERIC(5, 4))
    confidence_score = Column(NUMERIC(5, 4))
    sentiment_score = Column(NUMERIC(5, 4))
    classification_level = Column(String(20), default="public")
    verification_status = Column(String(20), default="unverified")
    verification_sources = Column(ARRAY(Text))
    verification_details = Column(JSONB)
    geotags = Column(JSONB)
    entities_extracted = Column(JSONB)
    extraction_status = Column(String(20), default="pending")
    metadata_json = Column('metadata', JSONB)
    encrypted_content = Column(BYTEA)
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(TIMESTAMP(timezone=True), default=func.now())
    updated_at = Column(
        TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now()
    )
    event_id = Column(String(255))
    event_nature = Column(String(100))
    event_summary_extracted = Column(Text)
    event_severity = Column(String(20))
    summary_30_words = Column(Text)
    summary_100_words = Column(Text)
    cross_border_classification = Column(String(30))
    conflict_classification = Column(String(30))
    cross_border_details = Column(Text)
    conflict_continuity_details = Column(Text)
    related_incidents_mentioned = Column(ARRAY(Text))
    event_date_extracted = Column(Date)
    event_time_extracted = Column(Time)
    time_of_day_extracted = Column(String(20))
    event_duration_extracted = Column(String(100))
    event_timeline = Column(ARRAY(Text))
    location_name_extracted = Column(String(255))
    specific_location_extracted = Column(Text)
    latitude_extracted = Column(NUMERIC(10, 8))
    longitude_extracted = Column(NUMERIC(11, 8))
    country_extracted = Column(String(100))
    region_state_extracted = Column(String(100))
    city_district_extracted = Column(String(100))
    proximity_landmarks = Column(ARRAY(Text))
    proximity_borders_extracted = Column(Text)
    proximity_strategic_locations = Column(ARRAY(Text))
    geocoding_confidence = Column(NUMERIC(5, 4))
    geocoding_source = Column(String(100))
    geocoding_accuracy_radius_meters = Column(NUMERIC(10, 2))
    geocoding_country_code = Column(String(5))
    geocoding_admin_level_1 = Column(String(100))
    geocoding_admin_level_2 = Column(String(100))
    geocoding_place_type = Column(String(50))
    location_point = Column(Geometry("POINT", srid=4326))
    fatalities_count_extracted = Column(Integer)
    casualties_count_extracted = Column(Integer)
    missing_count_extracted = Column(Integer)
    people_displaced_extracted = Column(Integer)
    people_affected_extracted = Column(Integer)
    perpetrator_names_extracted = Column(ARRAY(Text))
    perpetrator_groups_extracted = Column(ARRAY(Text))
    victim_names_extracted = Column(ARRAY(Text))
    victim_groups_extracted = Column(ARRAY(Text))
    witness_names_extracted = Column(ARRAY(Text))
    authority_figures_extracted = Column(ARRAY(Text))
    victim_ages_extracted = Column(ARRAY(Text))
    perpetrator_ages_extracted = Column(ARRAY(Text))
    victim_genders_extracted = Column(ARRAY(Text))
    perpetrator_genders_extracted = Column(ARRAY(Text))
    ethnic_groups_involved_extracted = Column(ARRAY(Text))
    religious_groups_involved_extracted = Column(ARRAY(Text))
    occupational_groups_extracted = Column(ARRAY(Text))
    weapons_methods_extracted = Column(ARRAY(Text))
    motives_reasons_extracted = Column(ARRAY(Text))
    event_triggers_extracted = Column(ARRAY(Text))
    tactical_approach_extracted = Column(Text)
    planning_level_extracted = Column(String(50))
    background_context_extracted = Column(Text)
    historical_significance_extracted = Column(Text)
    political_factors_extracted = Column(ARRAY(Text))
    social_factors_extracted = Column(ARRAY(Text))
    economic_factors_extracted = Column(ARRAY(Text))
    cultural_factors_extracted = Column(ARRAY(Text))
    environmental_factors_extracted = Column(ARRAY(Text))
    immediate_aftermath_extracted = Column(Text)
    infrastructure_damage_extracted = Column(ARRAY(Text))
    economic_impact_extracted = Column(Text)
    social_impact_extracted = Column(Text)
    psychological_impact_extracted = Column(Text)
    local_authority_response_extracted = Column(Text)
    national_government_response_extracted = Column(Text)
    international_reactions_extracted = Column(ARRAY(Text))
    humanitarian_response_extracted = Column(Text)
    security_measures_extracted = Column(ARRAY(Text))
    arrests_made_extracted = Column(Integer)
    charges_filed_extracted = Column(ARRAY(Text))
    legal_actions_extracted = Column(ARRAY(Text))
    trial_information_extracted = Column(Text)
    convictions_extracted = Column(ARRAY(Text))
    sentences_extracted = Column(ARRAY(Text))
    investigations_launched_extracted = Column(ARRAY(Text))
    photographic_evidence_extracted = Column(Boolean)
    video_evidence_extracted = Column(Boolean)
    audio_evidence_extracted = Column(Boolean)
    social_media_evidence_extracted = Column(ARRAY(Text))
    witness_testimonies_extracted = Column(ARRAY(Text))
    official_statements_extracted = Column(ARRAY(Text))
    media_coverage_extracted = Column(ARRAY(Text))
    source_name_extracted = Column(String(255))
    article_author_extracted = Column(String(255))
    publication_date_extracted = Column(Date)
    mentioned_sources_extracted = Column(ARRAY(Text))
    eyewitnesses_extracted = Column(ARRAY(Text))
    verification_level_extracted = Column(String(20))
    source_reliability_extracted = Column(String(20))
    international_actors_extracted = Column(ARRAY(Text))
    cross_border_elements_extracted = Column(ARRAY(Text))
    diplomatic_implications_extracted = Column(ARRAY(Text))
    treaty_violations_extracted = Column(ARRAY(Text))
    humanitarian_aid_extracted = Column(Text)
    refugee_displacement_extracted = Column(Text)
    humanitarian_access_extracted = Column(Text)
    protection_concerns_extracted = Column(ARRAY(Text))
    key_quotes_extracted = Column(ARRAY(Text))
    official_statements_text_extracted = Column(ARRAY(Text))
    eyewitness_accounts_extracted = Column(ARRAY(Text))
    extraction_confidence_score = Column(NUMERIC(5, 4))
    extraction_strategy_used = Column(String(30))
    extraction_model_used = Column(String(100))
    extraction_processing_time_ms = Column(Integer)
    extraction_api_calls_used = Column(Integer)
    extraction_errors = Column(ARRAY(Text))
    extraction_timestamp = Column(TIMESTAMP(timezone=True))
    completeness_score = Column(NUMERIC(5, 4))
    data_quality_score = Column(NUMERIC(5, 4))
    gdelt_global_event_id = Column(BIGINT)
    gdelt_sql_date = Column(Integer)
    gdelt_date_added = Column(BIGINT)
    actor1_code = Column(String(100))
    actor1_name = Column(String(255))
    actor1_country_code = Column(String(3))
    actor1_known_group_code = Column(String(100))
    actor1_ethnic_code = Column(String(100))
    actor1_religion1_code = Column(String(100))
    actor1_religion2_code = Column(String(100))
    actor1_type1_code = Column(String(100))
    actor1_type2_code = Column(String(100))
    actor1_type3_code = Column(String(100))
    actor2_code = Column(String(100))
    actor2_name = Column(String(255))
    actor2_country_code = Column(String(3))
    actor2_known_group_code = Column(String(100))
    actor2_ethnic_code = Column(String(100))
    actor2_religion1_code = Column(String(100))
    actor2_religion2_code = Column(String(100))
    actor2_type1_code = Column(String(100))
    actor2_type2_code = Column(String(100))
    actor2_type3_code = Column(String(100))
    gdelt_is_root_event = Column(Integer)
    gdelt_event_code = Column(String(10))
    gdelt_event_base_code = Column(String(10))
    gdelt_event_root_code = Column(String(10))
    gdelt_quad_class = Column(Integer)
    gdelt_goldstein_scale = Column(NUMERIC(4, 1))
    gdelt_num_mentions = Column(Integer, default=0)
    gdelt_num_sources = Column(Integer, default=0)
    gdelt_num_articles = Column(Integer, default=0)
    gdelt_avg_tone = Column(NUMERIC(15, 10))
    actor1_geo_type = Column(Integer)
    actor1_geo_fullname = Column(String(255))
    actor1_geo_country_code = Column(String(3))
    actor1_geo_adm1_code = Column(String(10))
    actor1_geo_adm2_code = Column(String(10))
    actor1_geo_lat = Column(NUMERIC(10, 6))
    actor1_geo_long = Column(NUMERIC(10, 6))
    actor1_geo_feature_id = Column(String(20))
    actor2_geo_type = Column(Integer)
    actor2_geo_fullname = Column(String(255))
    actor2_geo_country_code = Column(String(3))
    actor2_geo_adm1_code = Column(String(10))
    actor2_geo_adm2_code = Column(String(10))
    actor2_geo_lat = Column(NUMERIC(10, 6))
    actor2_geo_long = Column(NUMERIC(10, 6))
    actor2_geo_feature_id = Column(String(20))
    action_geo_type = Column(Integer)
    action_geo_fullname = Column(String(255))
    action_geo_country_code = Column(String(3))
    action_geo_adm1_code = Column(String(10))
    action_geo_adm2_code = Column(String(10))
    action_geo_lat = Column(NUMERIC(10, 6))
    action_geo_long = Column(NUMERIC(10, 6))
    action_geo_feature_id = Column(String(20))
    gdelt_content_crawl_status = Column(String(20), default="pending")
    content_crawl_status = Column(String)
    content_crawl_at = Column(DateTime)
    reliability_score = Column(Float)
    goldstein_score = Column(Float)
    valid_from = Column(DateTime)

    # Relationships
    data_source = relationship("DataSource", back_populates="information_units")
    parent_unit = relationship("InformationUnit", remote_side=[id])

    # Constraints
    __table_args__ = (
        UniqueConstraint("external_id", name="unique_external_id"),
        CheckConstraint(
            "arrests_made_extracted IS NULL OR arrests_made_extracted >= 0",
            name="information_units_arrests_made_extracted_check",
        ),
        CheckConstraint(
            "capture_method = ANY (ARRAY['api', 'web_scrape_static', 'web_scrape_dynamic', 'rss_feed', 'manual_entry', 'third_party_aggregator', 'email_submission', 'user_upload'])",
            name="information_units_capture_method_check",
        ),
        CheckConstraint(
            "casualties_count_extracted IS NULL OR casualties_count_extracted >= 0",
            name="information_units_casualties_count_extracted_check",
        ),
        CheckConstraint(
            "classification_level = ANY (ARRAY['public', 'restricted', 'confidential', 'secret', 'internal_use_only', 'top_secret'])",
            name="information_units_classification_level_check",
        ),
        CheckConstraint(
            "completeness_score IS NULL OR (completeness_score >= 0 AND completeness_score <= 1)",
            name="information_units_completeness_score_check",
        ),
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)",
            name="information_units_confidence_score_check",
        ),
        CheckConstraint(
            "conflict_classification = ANY (ARRAY['singular_incident', 'escalation', 'ongoing_conflict', 'retaliation', 'spillover', 'unknown'])",
            name="information_units_conflict_classification_check",
        ),
        CheckConstraint(
            "cross_border_classification = ANY (ARRAY['domestic_only', 'cross_border_direct', 'cross_border_impact', 'refugee_displacement', 'international_involvement', 'border_area', 'unknown'])",
            name="information_units_cross_border_classification_check",
        ),
        CheckConstraint(
            "data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 1)",
            name="information_units_data_quality_score_check",
        ),
        CheckConstraint(
            "event_severity = ANY (ARRAY['Low', 'Medium', 'High', 'Critical'])",
            name="information_units_event_severity_check",
        ),
        CheckConstraint(
            "extraction_confidence_score IS NULL OR (extraction_confidence_score >= 0 AND extraction_confidence_score <= 1)",
            name="information_units_extraction_confidence_score_check",
        ),
        CheckConstraint(
            "extraction_status = ANY (ARRAY['pending', 'in_progress', 'completed', 'failed', 'partial'])",
            name="information_units_extraction_status_check",
        ),
        CheckConstraint(
            "extraction_strategy_used = ANY (ARRAY['single_call', 'multi_sequential', 'multi_parallel', 'adaptive', 'hybrid'])",
            name="information_units_extraction_strategy_used_check",
        ),
        CheckConstraint(
            "fatalities_count_extracted IS NULL OR fatalities_count_extracted >= 0",
            name="information_units_fatalities_count_extracted_check",
        ),
        CheckConstraint(
            "gdelt_quad_class IS NULL OR (gdelt_quad_class >= 1 AND gdelt_quad_class <= 4)",
            name="information_units_gdelt_quad_class_check",
        ),
        CheckConstraint(
            "missing_count_extracted IS NULL OR missing_count_extracted >= 0",
            name="information_units_missing_count_extracted_check",
        ),
        CheckConstraint(
            "paywall_status = ANY (ARRAY['unknown', 'none', 'soft', 'hard', 'metered', 'freemium'])",
            name="information_units_paywall_status_check",
        ),
        CheckConstraint(
            "people_affected_extracted IS NULL OR people_affected_extracted >= 0",
            name="information_units_people_affected_extracted_check",
        ),
        CheckConstraint(
            "people_displaced_extracted IS NULL OR people_displaced_extracted >= 0",
            name="information_units_people_displaced_extracted_check",
        ),
        CheckConstraint(
            "planning_level_extracted = ANY (ARRAY['spontaneous', 'planned', 'coordinated', 'unknown'])",
            name="information_units_planning_level_extracted_check",
        ),
        CheckConstraint(
            "relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 1)",
            name="information_units_relevance_score_check",
        ),
        CheckConstraint(
            "sentiment_score IS NULL OR (sentiment_score >= -1 AND sentiment_score <= 1)",
            name="information_units_sentiment_score_check",
        ),
        CheckConstraint(
            "source_reliability_extracted = ANY (ARRAY['Low', 'Medium', 'High', 'Unknown'])",
            name="information_units_source_reliability_extracted_check",
        ),
        CheckConstraint(
            "time_of_day_extracted = ANY (ARRAY['morning', 'noon', 'afternoon', 'evening', 'night', 'dusk', 'dawn'])",
            name="information_units_time_of_day_extracted_check",
        ),
        CheckConstraint(
            "unit_type = ANY (ARRAY['article', 'social_media_post', 'report', 'text_extract', 'image', 'video', 'audio', 'document_file', 'dataset_entry', 'field_report', 'other_raw_data', 'news_article', 'blog_post', 'tweet', 'forum_post', 'academic_publication', 'press_release', 'official_statement', 'web_page', 'audio_clip', 'other_text_document'])",
            name="information_units_unit_type_check",
        ),
        CheckConstraint(
            "verification_level_extracted = ANY (ARRAY['Unverified', 'Partially', 'Verified', 'Disputed'])",
            name="information_units_verification_level_extracted_check",
        ),
        CheckConstraint(
            "verification_status = ANY (ARRAY['unverified', 'pending_verification', 'verified_human', 'human_verified', 'verified_automated', 'machine_verified', 'disputed', 'debunked', 'confirmed_false', 'uncertain', 'retracted'])",
            name="information_units_verification_status_check",
        ),
    )

    def __repr__(self):
        return f"<InformationUnit(id='{self.id}', title='{self.title[:50]}...')>"

    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                result[column.name] = value.isoformat()
            elif isinstance(value, uuid.UUID):
                result[column.name] = str(value)
            else:
                result[column.name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InformationUnit':
        """Create model instance from dictionary."""
        # Filter out None values and unknown columns
        valid_columns = {c.name for c in cls.__table__.columns}
        filtered_data = {k: v for k, v in data.items() if k in valid_columns and v is not None}
        
        return cls(**filtered_data)


class ProcessingLog(Base):
    """Model for tracking ETL processing operations."""
    
    __tablename__ = 'processing_logs'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Processing information
    operation_type = Column(String(50), nullable=False)  # 'etl', 'ml_scoring', 'conflict_analysis'
    status = Column(String(20), nullable=False)  # 'started', 'completed', 'failed'
    
    # File and dataset information
    file_path = Column(String(500))
    dataset_type = Column(String(50))
    file_size_bytes = Column(Integer)
    
    # Processing metrics
    records_processed = Column(Integer, default=0)
    records_succeeded = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    processing_time_seconds = Column(Float)
    
    # Error information
    error_message = Column(Text)
    error_details = Column(JSONB)
    
    # Metadata
    metadata_json = Column('metadata', JSONB)
    
    # Timestamps
    started_at = Column(TIMESTAMP(timezone=True), nullable=False)
    completed_at = Column(TIMESTAMP(timezone=True))
    created_at = Column(TIMESTAMP(timezone=True), default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_processing_logs_operation_type', 'operation_type'),
        Index('idx_processing_logs_status', 'status'),
        Index('idx_processing_logs_started_at', 'started_at'),
        Index('idx_processing_logs_dataset_type', 'dataset_type'),
    )
    
    def __repr__(self):
        return f"<ProcessingLog(operation='{self.operation_type}', status='{self.status}')>"


class GDELTEventSummary(Base):
    """Summary model for aggregated GDELT event statistics."""
    
    __tablename__ = 'gdelt_event_summaries'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Time aggregation
    date = Column(DateTime(timezone=True), nullable=False)
    aggregation_level = Column(String(20), nullable=False)  # 'daily', 'weekly', 'monthly'
    
    # Geographic aggregation
    country = Column(String(10))
    region = Column(String(50))
    
    # Event aggregation
    event_type = Column(String(50))
    conflict_level = Column(String(20))
    
    # Metrics
    total_events = Column(Integer, default=0)
    total_mentions = Column(Integer, default=0)
    total_sources = Column(Integer, default=0)
    average_tone = Column(Float)
    average_goldstein = Column(Float)
    
    # Conflict metrics
    conflict_events = Column(Integer, default=0)
    fatalities = Column(Integer, default=0)
    casualties = Column(Integer, default=0)
    displaced_people = Column(Integer, default=0)
    
    # Metadata
    metadata_json = Column('metadata', JSONB)
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('date', 'aggregation_level', 'country', 'event_type', 
                        name='uq_gdelt_summary_unique'),
        Index('idx_gdelt_summaries_date', 'date'),
        Index('idx_gdelt_summaries_country', 'country'),
        Index('idx_gdelt_summaries_event_type', 'event_type'),
        Index('idx_gdelt_summaries_conflict_level', 'conflict_level'),
    )
    
    def __repr__(self):
        return f"<GDELTEventSummary(date='{self.date}', country='{self.country}', events={self.total_events})>"


# Utility functions for model operations
def create_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)


def drop_tables(engine):
    """Drop all tables from the database."""
    Base.metadata.drop_all(engine)


def get_table_info() -> Dict[str, Any]:
    """Get information about all defined tables."""
    tables = {}
    for table_name, table in Base.metadata.tables.items():
        tables[table_name] = {
            'columns': [col.name for col in table.columns],
            'indexes': [idx.name for idx in table.indexes],
            'constraints': [const.name for const in table.constraints if const.name]
        }
    return tables


# Export all models and utilities
__all__ = [
    'Base',
    'DataSource',
    'InformationUnit', 
    'ProcessingLog',
    'GDELTEventSummary',
    'create_tables',
    'drop_tables',
    'get_table_info'
]