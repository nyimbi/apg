"""
Quality Management Models

Database models for quality management functionality including quality control plans,
inspections, non-conformances, CAPA, and regulatory compliance.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, Integer, DateTime, Date, Numeric, Text, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from uuid_extensions import uuid7str

from ...core_financials.general_ledger.models import BaseModel as SQLBaseModel

class QualityControlPlanStatus(str, Enum):
	"""Quality control plan status"""
	DRAFT = "draft"
	ACTIVE = "active"
	INACTIVE = "inactive"  
	OBSOLETE = "obsolete"

class InspectionType(str, Enum):
	"""Inspection type enumeration"""
	INCOMING = "incoming"
	IN_PROCESS = "in_process"
	FINAL = "final"
	AUDIT = "audit"
	CUSTOMER_RETURN = "customer_return"

class InspectionStatus(str, Enum):
	"""Inspection status enumeration"""
	SCHEDULED = "scheduled"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	PASSED = "passed"
	FAILED = "failed"
	ON_HOLD = "on_hold"

class NonConformanceStatus(str, Enum):
	"""Non-conformance status"""
	OPEN = "open"
	INVESTIGATING = "investigating"
	CORRECTIVE_ACTION = "corrective_action"
	VERIFIED = "verified"
	CLOSED = "closed"

class NonConformanceSeverity(str, Enum):
	"""Non-conformance severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"

class CAPAStatus(str, Enum):
	"""CAPA status enumeration"""
	INITIATED = "initiated"
	INVESTIGATION = "investigation"
	ACTION_PLAN = "action_plan"
	IMPLEMENTATION = "implementation"
	VERIFICATION = "verification"
	COMPLETED = "completed"
	CANCELLED = "cancelled"

# SQLAlchemy Models

class MFQQualityControlPlan(SQLBaseModel):
	"""Quality control plans for products and processes"""
	__tablename__ = 'mfq_quality_control_plans'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Plan identification
	plan_number = Column(String(50), nullable=False, unique=True)
	plan_name = Column(String(200), nullable=False)
	version = Column(String(20), nullable=False, default="1.0")
	revision = Column(String(10), nullable=False, default="A")
	
	# Plan scope
	product_id = Column(String(36), index=True)
	process_id = Column(String(36), index=True)
	facility_id = Column(String(36), index=True)
	
	# Plan details
	plan_type = Column(String(30), nullable=False)  # product, process, supplier
	inspection_frequency = Column(String(50))  # per_batch, daily, weekly, etc.
	sample_size = Column(Integer)
	sampling_method = Column(String(50))
	
	# Effectivity
	effective_date = Column(Date, nullable=False)
	expiry_date = Column(Date)
	status = Column(String(20), nullable=False, default="draft")
	
	# Approval
	approved_by = Column(String(36))
	approved_at = Column(DateTime)
	
	# Documentation
	description = Column(Text)
	regulatory_requirements = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	inspection_points = relationship("MFQInspectionPoint", back_populates="quality_plan")
	inspections = relationship("MFQInspection", back_populates="quality_plan")
	
	__table_args__ = (
		Index('idx_mfq_qcp_tenant_product', 'tenant_id', 'product_id'),
		Index('idx_mfq_qcp_plan_number', 'plan_number'),
		Index('idx_mfq_qcp_status_effective', 'status', 'effective_date'),
	)

class MFQInspectionPoint(SQLBaseModel):
	"""Inspection points and test specifications"""
	__tablename__ = 'mfq_inspection_points'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Plan reference
	quality_plan_id = Column(String(36), ForeignKey('mfq_quality_control_plans.id'), nullable=False, index=True)
	
	# Inspection point details
	sequence_number = Column(Integer, nullable=False)
	characteristic_name = Column(String(200), nullable=False)
	characteristic_type = Column(String(30), nullable=False)  # dimensional, visual, functional, chemical
	
	# Test method
	test_method = Column(String(100))
	measuring_instrument = Column(String(100))
	test_procedure = Column(String(200))
	
	# Specification limits
	nominal_value = Column(Numeric(15, 6))
	lower_spec_limit = Column(Numeric(15, 6))
	upper_spec_limit = Column(Numeric(15, 6))
	unit_of_measure = Column(String(20))
	
	# Control limits (for SPC)
	lower_control_limit = Column(Numeric(15, 6))
	upper_control_limit = Column(Numeric(15, 6))
	
	# Inspection requirements
	is_critical = Column(Boolean, default=False)
	is_major = Column(Boolean, default=False)
	is_minor = Column(Boolean, default=False)
	acceptance_criteria = Column(Text)
	
	# Documentation
	inspection_instructions = Column(Text)
	regulatory_reference = Column(String(200))
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	
	# Relationships
	quality_plan = relationship("MFQQualityControlPlan", back_populates="inspection_points")
	test_results = relationship("MFQTestResult", back_populates="inspection_point")
	
	__table_args__ = (
		Index('idx_mfq_ip_tenant_plan', 'tenant_id', 'quality_plan_id'),
		Index('idx_mfq_ip_sequence', 'quality_plan_id', 'sequence_number'),
	)

class MFQInspection(SQLBaseModel):
	"""Quality inspections and testing activities"""
	__tablename__ = 'mfq_inspections'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Inspection identification
	inspection_number = Column(String(50), nullable=False, unique=True)
	quality_plan_id = Column(String(36), ForeignKey('mfq_quality_control_plans.id'), index=True)
	
	# Inspection type and scope
	inspection_type = Column(String(30), nullable=False)
	inspection_reason = Column(String(100))
	
	# Product/batch information
	product_id = Column(String(36), index=True)
	batch_lot_number = Column(String(100), index=True)
	production_order_id = Column(String(36), index=True)
	supplier_id = Column(String(36), index=True)
	
	# Quantities
	lot_size = Column(Numeric(15, 4))
	sample_size = Column(Numeric(15, 4))
	defective_quantity = Column(Numeric(15, 4), default=0)
	
	# Scheduling
	scheduled_date = Column(DateTime)
	started_at = Column(DateTime)
	completed_at = Column(DateTime)
	
	# Personnel
	inspector_id = Column(String(36), nullable=False)
	supervisor_id = Column(String(36))
	
	# Status and results
	status = Column(String(20), nullable=False, default="scheduled")
	overall_result = Column(String(20))  # pass, fail, conditional
	
	# Location
	facility_id = Column(String(36), index=True)
	location = Column(String(100))
	
	# Documentation
	inspection_notes = Column(Text)
	corrective_action_required = Column(Boolean, default=False)
	
	# Regulatory
	regulatory_compliance = Column(Boolean, default=True)
	regulatory_notes = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	quality_plan = relationship("MFQQualityControlPlan", back_populates="inspections")
	test_results = relationship("MFQTestResult", back_populates="inspection")
	non_conformances = relationship("MFQNonConformance", back_populates="inspection")
	
	__table_args__ = (
		Index('idx_mfq_insp_tenant_status', 'tenant_id', 'status'),
		Index('idx_mfq_insp_number', 'inspection_number'),
		Index('idx_mfq_insp_product_batch', 'product_id', 'batch_lot_number'),
		Index('idx_mfq_insp_dates', 'scheduled_date', 'completed_at'),
	)

class MFQTestResult(SQLBaseModel):
	"""Individual test results for inspection points"""
	__tablename__ = 'mfq_test_results'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# References
	inspection_id = Column(String(36), ForeignKey('mfq_inspections.id'), nullable=False, index=True)
	inspection_point_id = Column(String(36), ForeignKey('mfq_inspection_points.id'), nullable=False, index=True)
	
	# Test execution
	test_sequence = Column(Integer, nullable=False)
	tested_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	tested_by = Column(String(36), nullable=False)
	
	# Results
	measured_value = Column(Numeric(15, 6))
	text_result = Column(String(500))  # For non-numeric results
	pass_fail_result = Column(String(10))  # pass, fail, n/a
	
	# Test conditions
	test_temperature = Column(Numeric(8, 2))
	test_humidity = Column(Numeric(5, 2))
	test_conditions = Column(Text)
	
	# Equipment used
	measuring_equipment_id = Column(String(36))
	calibration_due_date = Column(Date)
	
	# Statistical data
	is_out_of_spec = Column(Boolean, default=False)
	is_out_of_control = Column(Boolean, default=False)
	deviation_from_nominal = Column(Numeric(15, 6))
	
	# Comments
	test_notes = Column(Text)
	
	# Audit trail
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	
	# Relationships
	inspection = relationship("MFQInspection", back_populates="test_results")
	inspection_point = relationship("MFQInspectionPoint", back_populates="test_results")
	
	__table_args__ = (
		Index('idx_mfq_tr_tenant_inspection', 'tenant_id', 'inspection_id'),
		Index('idx_mfq_tr_point_sequence', 'inspection_point_id', 'test_sequence'),
		Index('idx_mfq_tr_out_of_spec', 'is_out_of_spec'),
	)

class MFQNonConformance(SQLBaseModel):
	"""Non-conformance tracking and management"""
	__tablename__ = 'mfq_non_conformances'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# NC identification
	nc_number = Column(String(50), nullable=False, unique=True)
	title = Column(String(200), nullable=False)
	
	# Source information
	inspection_id = Column(String(36), ForeignKey('mfq_inspections.id'), index=True)
	source_type = Column(String(30), nullable=False)  # inspection, customer_complaint, internal_audit
	source_reference = Column(String(100))
	
	# Product/process information
	product_id = Column(String(36), index=True)
	batch_lot_number = Column(String(100), index=True)
	production_order_id = Column(String(36), index=True)
	supplier_id = Column(String(36), index=True)
	
	# NC details
	description = Column(Text, nullable=False)
	root_cause_analysis = Column(Text)
	severity = Column(String(20), nullable=False, default="medium")
	category = Column(String(50))  # material, process, documentation, etc.
	
	# Quantities affected
	quantity_affected = Column(Numeric(15, 4))
	quantity_quarantined = Column(Numeric(15, 4))
	quantity_scrapped = Column(Numeric(15, 4))
	quantity_reworked = Column(Numeric(15, 4))
	
	# Dates and timeline
	discovered_date = Column(Date, nullable=False)
	reported_date = Column(Date, nullable=False)
	required_resolution_date = Column(Date)
	actual_resolution_date = Column(Date)
	
	# Personnel
	reported_by = Column(String(36), nullable=False)
	assigned_to = Column(String(36))
	quality_manager = Column(String(36))
	
	# Status and disposition
	status = Column(String(20), nullable=False, default="open")
	disposition = Column(String(50))  # accept, reject, rework, return_to_supplier
	disposition_notes = Column(Text)
	
	# Cost impact
	estimated_cost_impact = Column(Numeric(15, 2))
	actual_cost_impact = Column(Numeric(15, 2))
	
	# Customer impact
	customer_notification_required = Column(Boolean, default=False)
	customer_notified = Column(Boolean, default=False)
	customer_notification_date = Column(Date)
	
	# Regulatory impact
	regulatory_reportable = Column(Boolean, default=False)
	regulatory_reported = Column(Boolean, default=False)
	regulatory_reference = Column(String(100))
	
	# Resolution
	immediate_action = Column(Text)
	containment_action = Column(Text)
	resolution_notes = Column(Text)
	
	# Verification
	verified_by = Column(String(36))
	verified_at = Column(DateTime)
	verification_notes = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	inspection = relationship("MFQInspection", back_populates="non_conformances")
	capa_records = relationship("MFQCAPARecord", back_populates="non_conformance")
	
	__table_args__ = (
		Index('idx_mfq_nc_tenant_status', 'tenant_id', 'status'),
		Index('idx_mfq_nc_number', 'nc_number'),
		Index('idx_mfq_nc_product_batch', 'product_id', 'batch_lot_number'),
		Index('idx_mfq_nc_severity', 'severity'),
		Index('idx_mfq_nc_dates', 'discovered_date', 'required_resolution_date'),
	)

class MFQCAPARecord(SQLBaseModel):
	"""Corrective and Preventive Action (CAPA) records"""
	__tablename__ = 'mfq_capa_records'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# CAPA identification
	capa_number = Column(String(50), nullable=False, unique=True)
	title = Column(String(200), nullable=False)
	capa_type = Column(String(20), nullable=False)  # corrective, preventive
	
	# Source information
	non_conformance_id = Column(String(36), ForeignKey('mfq_non_conformances.id'), index=True)
	source_type = Column(String(30), nullable=False)  # nc, audit, risk_assessment
	source_reference = Column(String(100))
	
	# Problem definition
	problem_description = Column(Text, nullable=False)
	impact_assessment = Column(Text)
	risk_level = Column(String(20), default="medium")
	
	# Root cause analysis
	root_cause_description = Column(Text)
	root_cause_method = Column(String(50))  # 5_why, fishbone, fta
	contributing_factors = Column(Text)
	
	# Action plan
	corrective_action_plan = Column(Text)
	preventive_action_plan = Column(Text)
	implementation_plan = Column(Text)
	
	# Personnel
	initiated_by = Column(String(36), nullable=False)
	capa_owner = Column(String(36), nullable=False)
	quality_manager = Column(String(36))
	
	# Timeline
	initiated_date = Column(Date, nullable=False)
	target_completion_date = Column(Date, nullable=False)
	actual_completion_date = Column(Date)
	
	# Status tracking
	status = Column(String(20), nullable=False, default="initiated")
	percent_complete = Column(Integer, default=0)
	
	# Implementation tracking
	actions_implemented = Column(Text)
	resources_required = Column(Text)
	training_required = Column(Text)
	
	# Effectiveness verification
	verification_method = Column(String(100))
	verification_criteria = Column(Text)
	verification_results = Column(Text)
	is_effective = Column(Boolean)
	effectiveness_verified_by = Column(String(36))
	effectiveness_verified_at = Column(DateTime)
	
	# Cost tracking
	estimated_cost = Column(Numeric(15, 2))
	actual_cost = Column(Numeric(15, 2))
	cost_benefit_analysis = Column(Text)
	
	# Documentation
	supporting_documents = Column(Text)  # JSON array of document references
	
	# Closure
	closed_by = Column(String(36))
	closed_at = Column(DateTime)
	closure_notes = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	non_conformance = relationship("MFQNonConformance", back_populates="capa_records")
	
	__table_args__ = (
		Index('idx_mfq_capa_tenant_status', 'tenant_id', 'status'),
		Index('idx_mfq_capa_number', 'capa_number'),
		Index('idx_mfq_capa_nc', 'non_conformance_id'),
		Index('idx_mfq_capa_dates', 'initiated_date', 'target_completion_date'),
	)

# Pydantic Models for API

class QualityControlPlanCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	plan_number: str = Field(..., min_length=1, max_length=50)
	plan_name: str = Field(..., min_length=1, max_length=200)
	version: str = Field(default="1.0", max_length=20)
	revision: str = Field(default="A", max_length=10)
	product_id: str | None = None
	process_id: str | None = None
	facility_id: str | None = None
	plan_type: str = Field(..., min_length=1, max_length=30)
	inspection_frequency: str | None = None
	sample_size: int | None = None
	sampling_method: str | None = None
	effective_date: date
	expiry_date: date | None = None
	description: str | None = None
	regulatory_requirements: str | None = None

class InspectionCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	inspection_number: str = Field(..., min_length=1, max_length=50)
	quality_plan_id: str | None = None
	inspection_type: InspectionType
	inspection_reason: str | None = None
	product_id: str | None = None
	batch_lot_number: str | None = None
	production_order_id: str | None = None
	supplier_id: str | None = None
	lot_size: Decimal | None = None
	sample_size: Decimal | None = None
	scheduled_date: datetime | None = None
	inspector_id: str = Field(..., min_length=36, max_length=36)
	supervisor_id: str | None = None
	facility_id: str | None = None
	location: str | None = None
	inspection_notes: str | None = None
	regulatory_compliance: bool = True
	regulatory_notes: str | None = None

class NonConformanceCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	nc_number: str = Field(..., min_length=1, max_length=50)
	title: str = Field(..., min_length=1, max_length=200)
	inspection_id: str | None = None
	source_type: str = Field(..., min_length=1, max_length=30)
	source_reference: str | None = None
	product_id: str | None = None
	batch_lot_number: str | None = None
	production_order_id: str | None = None
	supplier_id: str | None = None
	description: str = Field(..., min_length=1)
	severity: NonConformanceSeverity = NonConformanceSeverity.MEDIUM
	category: str | None = None
	quantity_affected: Decimal | None = None
	discovered_date: date
	reported_date: date
	required_resolution_date: date | None = None
	reported_by: str = Field(..., min_length=36, max_length=36)
	assigned_to: str | None = None
	disposition: str | None = None
	disposition_notes: str | None = None
	customer_notification_required: bool = False
	regulatory_reportable: bool = False
	immediate_action: str | None = None
	containment_action: str | None = None

class CAPARecordCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	capa_number: str = Field(..., min_length=1, max_length=50)
	title: str = Field(..., min_length=1, max_length=200)
	capa_type: str = Field(..., regex="^(corrective|preventive)$")
	non_conformance_id: str | None = None
	source_type: str = Field(..., min_length=1, max_length=30)
	source_reference: str | None = None
	problem_description: str = Field(..., min_length=1)
	impact_assessment: str | None = None
	risk_level: str = Field(default="medium", regex="^(low|medium|high|critical)$")
	root_cause_description: str | None = None
	root_cause_method: str | None = None
	contributing_factors: str | None = None
	corrective_action_plan: str | None = None
	preventive_action_plan: str | None = None
	implementation_plan: str | None = None
	initiated_by: str = Field(..., min_length=36, max_length=36)
	capa_owner: str = Field(..., min_length=36, max_length=36)
	quality_manager: str | None = None
	initiated_date: date
	target_completion_date: date
	verification_method: str | None = None
	verification_criteria: str | None = None
	resources_required: str | None = None
	training_required: str | None = None
	estimated_cost: Decimal | None = None

__all__ = [
	"QualityControlPlanStatus", "InspectionType", "InspectionStatus", 
	"NonConformanceStatus", "NonConformanceSeverity", "CAPAStatus",
	"MFQQualityControlPlan", "MFQInspectionPoint", "MFQInspection", "MFQTestResult",
	"MFQNonConformance", "MFQCAPARecord",
	"QualityControlPlanCreate", "InspectionCreate", "NonConformanceCreate", "CAPARecordCreate"
]