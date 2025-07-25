"""
Bill of Materials (BOM) Models

Database models for BOM management including product structures, component relationships,
engineering changes, and version control.
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

class BOMStatus(str, Enum):
	"""BOM status enumeration"""
	DRAFT = "draft"
	ACTIVE = "active"
	INACTIVE = "inactive"
	OBSOLETE = "obsolete"
	PENDING_APPROVAL = "pending_approval"

class ComponentType(str, Enum):
	"""Component type in BOM"""
	RAW_MATERIAL = "raw_material"
	PURCHASED_PART = "purchased_part"
	MANUFACTURED_PART = "manufactured_part"
	ASSEMBLY = "assembly"
	SUBASSEMBLY = "subassembly"
	PHANTOM = "phantom"

class BOMUsageType(str, Enum):
	"""BOM usage type"""
	MANUFACTURING = "manufacturing"
	ENGINEERING = "engineering"
	COSTING = "costing"
	PLANNING = "planning"

class ChangeOrderStatus(str, Enum):
	"""Engineering change order status"""
	DRAFT = "draft"
	SUBMITTED = "submitted"
	APPROVED = "approved"
	REJECTED = "rejected"
	IMPLEMENTED = "implemented"
	CANCELLED = "cancelled"

# SQLAlchemy Models

class MFBBillOfMaterials(SQLBaseModel):
	"""Bill of Materials header"""
	__tablename__ = 'mfb_bill_of_materials'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# BOM identification
	bom_number = Column(String(50), nullable=False, unique=True)
	bom_name = Column(String(200), nullable=False)
	version = Column(String(20), nullable=False, default="1.0")
	revision = Column(String(10), nullable=False, default="A")
	
	# Parent product information
	parent_product_id = Column(String(36), nullable=False, index=True)
	parent_sku = Column(String(100), nullable=False)
	parent_name = Column(String(200), nullable=False)
	
	# BOM properties
	bom_type = Column(String(30), nullable=False, default="manufacturing")
	usage_type = Column(String(30), nullable=False, default="manufacturing")
	unit_of_measure = Column(String(20), nullable=False)
	base_quantity = Column(Numeric(15, 4), nullable=False, default=1)
	
	# Validity and effectivity
	effective_date = Column(Date, nullable=False)
	expiry_date = Column(Date)
	status = Column(String(20), nullable=False, default="draft")
	
	# Manufacturing details
	facility_id = Column(String(36), index=True)
	routing_id = Column(String(36))
	alternate_bom_id = Column(String(36))
	
	# Cost information
	material_cost = Column(Numeric(15, 4))
	labor_cost = Column(Numeric(15, 4))
	overhead_cost = Column(Numeric(15, 4))
	total_cost = Column(Numeric(15, 4))
	cost_calculated_at = Column(DateTime)
	
	# Engineering information
	engineering_drawing = Column(String(200))
	specification_document = Column(String(200))
	
	# Approval workflow
	approved_by = Column(String(36))
	approved_at = Column(DateTime)
	approval_notes = Column(Text)
	
	# Notes and descriptions
	description = Column(Text)
	manufacturing_notes = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	components = relationship("MFBBOMComponent", back_populates="bom")
	change_orders = relationship("MFBEngineeringChangeOrder", back_populates="affected_bom")
	
	__table_args__ = (
		Index('idx_mfb_bom_tenant_product', 'tenant_id', 'parent_product_id'),
		Index('idx_mfb_bom_number', 'bom_number'),
		Index('idx_mfb_bom_status_effective', 'status', 'effective_date'),
	)

class MFBBOMComponent(SQLBaseModel):
	"""BOM component line items"""
	__tablename__ = 'mfb_bom_components'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# BOM reference
	bom_id = Column(String(36), ForeignKey('mfb_bill_of_materials.id'), nullable=False, index=True)
	
	# Component identification
	component_product_id = Column(String(36), nullable=False, index=True)
	component_sku = Column(String(100), nullable=False)
	component_name = Column(String(200), nullable=False)
	component_type = Column(String(30), nullable=False)
	
	# BOM structure
	sequence_number = Column(Integer, nullable=False)
	level_number = Column(Integer, nullable=False, default=1)
	parent_component_id = Column(String(36), ForeignKey('mfb_bom_components.id'))
	
	# Quantity requirements
	quantity_per = Column(Numeric(15, 6), nullable=False)
	unit_of_measure = Column(String(20), nullable=False)
	scrap_factor_pct = Column(Numeric(5, 2), default=0)
	yield_factor_pct = Column(Numeric(5, 2), default=100)
	
	# Reference designators (for electronics)
	reference_designators = Column(String(500))
	
	# Component properties
	is_optional = Column(Boolean, default=False)
	is_phantom = Column(Boolean, default=False)
	is_bulk_item = Column(Boolean, default=False)
	is_tooling = Column(Boolean, default=False)
	
	# Effectivity
	effective_date = Column(Date, nullable=False)
	expiry_date = Column(Date)
	
	# Sourcing information
	preferred_supplier_id = Column(String(36))
	supplier_part_number = Column(String(100))
	make_or_buy = Column(String(10), default="buy")
	
	# Cost information
	unit_cost = Column(Numeric(12, 4))
	extended_cost = Column(Numeric(15, 4))
	
	# Manufacturing details
	operation_sequence = Column(Integer)
	work_center_id = Column(String(36))
	assembly_notes = Column(Text)
	
	# Change tracking
	change_order_id = Column(String(36))
	change_type = Column(String(20))  # add, modify, delete
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	bom = relationship("MFBBillOfMaterials", back_populates="components")
	parent_component = relationship("MFBBOMComponent", remote_side=[id])
	child_components = relationship("MFBBOMComponent")
	substitutes = relationship("MFBComponentSubstitute", back_populates="primary_component")
	
	__table_args__ = (
		Index('idx_mfb_comp_tenant_bom', 'tenant_id', 'bom_id'),
		Index('idx_mfb_comp_product', 'component_product_id'),
		Index('idx_mfb_comp_sequence', 'bom_id', 'sequence_number'),
		Index('idx_mfb_comp_level', 'level_number'),
	)

class MFBComponentSubstitute(SQLBaseModel):
	"""Component substitution alternatives"""
	__tablename__ = 'mfb_component_substitutes'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Primary component reference
	primary_component_id = Column(String(36), ForeignKey('mfb_bom_components.id'), nullable=False, index=True)
	
	# Substitute component details
	substitute_product_id = Column(String(36), nullable=False, index=True)
	substitute_sku = Column(String(100), nullable=False)
	substitute_name = Column(String(200), nullable=False)
	
	# Substitution properties
	substitution_type = Column(String(20), nullable=False)  # direct, functional, form_fit_function
	substitution_ratio = Column(Numeric(10, 4), default=1)  # quantity ratio
	priority = Column(Integer, default=1)  # preference order
	
	# Effectivity
	effective_date = Column(Date, nullable=False)
	expiry_date = Column(Date)
	
	# Approval and validation
	is_approved = Column(Boolean, default=False)
	approved_by = Column(String(36))
	approved_at = Column(DateTime)
	
	# Cost impact
	cost_difference = Column(Numeric(12, 4))
	cost_difference_pct = Column(Numeric(5, 2))
	
	# Notes
	substitution_notes = Column(Text)
	engineering_notes = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	
	# Relationships
	primary_component = relationship("MFBBOMComponent", back_populates="substitutes")
	
	__table_args__ = (
		Index('idx_mfb_sub_tenant_primary', 'tenant_id', 'primary_component_id'),
		Index('idx_mfb_sub_product', 'substitute_product_id'),
		Index('idx_mfb_sub_priority', 'priority'),
	)

class MFBEngineeringChangeOrder(SQLBaseModel):
	"""Engineering change orders for BOM modifications"""
	__tablename__ = 'mfb_engineering_change_orders'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# ECO identification
	eco_number = Column(String(50), nullable=False, unique=True)
	eco_title = Column(String(200), nullable=False)
	change_reason = Column(String(100), nullable=False)
	
	# Change details
	change_description = Column(Text, nullable=False)
	impact_analysis = Column(Text)
	implementation_plan = Column(Text)
	
	# Affected items
	affected_bom_id = Column(String(36), ForeignKey('mfb_bill_of_materials.id'), index=True)
	affected_product_ids = Column(Text)  # JSON array of product IDs
	
	# Priority and urgency
	priority = Column(String(20), default="normal")
	urgency = Column(String(20), default="normal")
	business_impact = Column(String(20), default="low")
	
	# Dates and timeline
	requested_date = Column(Date, nullable=False)
	required_implementation_date = Column(Date)
	planned_implementation_date = Column(Date)
	actual_implementation_date = Column(Date)
	
	# Status and workflow
	status = Column(String(20), nullable=False, default="draft")
	current_approver = Column(String(36))
	
	# Cost impact
	estimated_cost = Column(Numeric(15, 2))
	actual_cost = Column(Numeric(15, 2))
	cost_category = Column(String(50))
	
	# Approval workflow
	submitted_by = Column(String(36), nullable=False)
	submitted_at = Column(DateTime)
	approved_by = Column(String(36))
	approved_at = Column(DateTime)
	rejected_by = Column(String(36))
	rejected_at = Column(DateTime)
	rejection_reason = Column(Text)
	
	# Implementation tracking
	implemented_by = Column(String(36))
	implemented_at = Column(DateTime)
	implementation_notes = Column(Text)
	
	# Document attachments
	attachments = Column(Text)  # JSON array of document references
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	affected_bom = relationship("MFBBillOfMaterials", back_populates="change_orders")
	
	__table_args__ = (
		Index('idx_mfb_eco_tenant_status', 'tenant_id', 'status'),
		Index('idx_mfb_eco_number', 'eco_number'),
		Index('idx_mfb_eco_bom', 'affected_bom_id'),
		Index('idx_mfb_eco_dates', 'requested_date', 'required_implementation_date'),
	)

# Pydantic Models for API

class BillOfMaterialsCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	bom_number: str = Field(..., min_length=1, max_length=50)
	bom_name: str = Field(..., min_length=1, max_length=200)
	version: str = Field(default="1.0", max_length=20)
	revision: str = Field(default="A", max_length=10)
	parent_product_id: str = Field(..., min_length=36, max_length=36)
	parent_sku: str = Field(..., min_length=1, max_length=100)
	parent_name: str = Field(..., min_length=1, max_length=200)
	bom_type: str = Field(default="manufacturing", max_length=30)
	usage_type: BOMUsageType = BOMUsageType.MANUFACTURING
	unit_of_measure: str = Field(..., min_length=1, max_length=20)
	base_quantity: Decimal = Field(default=Decimal('1'), gt=0)
	effective_date: date
	expiry_date: date | None = None
	facility_id: str | None = None
	routing_id: str | None = None
	alternate_bom_id: str | None = None
	engineering_drawing: str | None = None
	specification_document: str | None = None
	description: str | None = None
	manufacturing_notes: str | None = None

class BOMComponentCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	component_product_id: str = Field(..., min_length=36, max_length=36)
	component_sku: str = Field(..., min_length=1, max_length=100)
	component_name: str = Field(..., min_length=1, max_length=200)
	component_type: ComponentType
	sequence_number: int = Field(..., ge=1)
	level_number: int = Field(default=1, ge=1)
	parent_component_id: str | None = None
	quantity_per: Decimal = Field(..., gt=0)
	unit_of_measure: str = Field(..., min_length=1, max_length=20)
	scrap_factor_pct: Decimal = Field(default=Decimal('0'), ge=0, le=100)
	yield_factor_pct: Decimal = Field(default=Decimal('100'), gt=0, le=100)
	reference_designators: str | None = None
	is_optional: bool = False
	is_phantom: bool = False
	is_bulk_item: bool = False
	is_tooling: bool = False
	effective_date: date
	expiry_date: date | None = None
	preferred_supplier_id: str | None = None
	supplier_part_number: str | None = None
	make_or_buy: str = Field(default="buy", regex="^(make|buy)$")
	unit_cost: Decimal | None = None
	operation_sequence: int | None = None
	work_center_id: str | None = None
	assembly_notes: str | None = None

class ComponentSubstituteCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	substitute_product_id: str = Field(..., min_length=36, max_length=36)
	substitute_sku: str = Field(..., min_length=1, max_length=100)
	substitute_name: str = Field(..., min_length=1, max_length=200)
	substitution_type: str = Field(..., regex="^(direct|functional|form_fit_function)$")
	substitution_ratio: Decimal = Field(default=Decimal('1'), gt=0)
	priority: int = Field(default=1, ge=1)
	effective_date: date
	expiry_date: date | None = None
	cost_difference: Decimal | None = None
	cost_difference_pct: Decimal | None = None
	substitution_notes: str | None = None
	engineering_notes: str | None = None

class EngineeringChangeOrderCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	eco_number: str = Field(..., min_length=1, max_length=50)
	eco_title: str = Field(..., min_length=1, max_length=200)
	change_reason: str = Field(..., min_length=1, max_length=100)
	change_description: str = Field(..., min_length=1)
	impact_analysis: str | None = None
	implementation_plan: str | None = None
	affected_bom_id: str | None = None
	affected_product_ids: str | None = None
	priority: str = Field(default="normal", regex="^(low|normal|high|urgent|critical)$")
	urgency: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
	business_impact: str = Field(default="low", regex="^(low|medium|high|critical)$")
	requested_date: date
	required_implementation_date: date | None = None
	planned_implementation_date: date | None = None
	estimated_cost: Decimal | None = None
	cost_category: str | None = None

__all__ = [
	"BOMStatus", "ComponentType", "BOMUsageType", "ChangeOrderStatus",
	"MFBBillOfMaterials", "MFBBOMComponent", "MFBComponentSubstitute", "MFBEngineeringChangeOrder",
	"BillOfMaterialsCreate", "BOMComponentCreate", "ComponentSubstituteCreate", "EngineeringChangeOrderCreate"
]