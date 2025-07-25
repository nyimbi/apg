"""
Recipe & Formula Management Models

Database models for recipe and formula management functionality including
master recipes, formulas, ingredients, process instructions, and batch records.
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

class RecipeStatus(str, Enum):
	"""Recipe status enumeration"""
	DRAFT = "draft"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	ACTIVE = "active"
	INACTIVE = "inactive"
	OBSOLETE = "obsolete"

class FormulaType(str, Enum):
	"""Formula type enumeration"""
	MASTER = "master"
	PILOT = "pilot"
	PRODUCTION = "production"
	RESEARCH = "research"

class IngredientType(str, Enum):
	"""Ingredient type enumeration"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	EXCIPIENT = "excipient"
	ADDITIVE = "additive"
	SOLVENT = "solvent"
	CATALYST = "catalyst"

class ProcessStepType(str, Enum):
	"""Process step type enumeration"""
	WEIGHING = "weighing"
	MIXING = "mixing"
	HEATING = "heating"
	COOLING = "cooling"
	REACTION = "reaction"
	FILTRATION = "filtration"
	DRYING = "drying"
	PACKAGING = "packaging"

class BatchRecordStatus(str, Enum):
	"""Batch record status enumeration"""
	CREATED = "created"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	REVIEWED = "reviewed"
	APPROVED = "approved"
	REJECTED = "rejected"

# SQLAlchemy Models

class MFRMasterRecipe(SQLBaseModel):
	"""Master recipes for process manufacturing"""
	__tablename__ = 'mfr_master_recipes'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Recipe identification
	recipe_number = Column(String(50), nullable=False, unique=True)
	recipe_name = Column(String(200), nullable=False)
	version = Column(String(20), nullable=False, default="1.0")
	revision = Column(String(10), nullable=False, default="A")
	
	# Product information
	product_id = Column(String(36), nullable=False, index=True)
	product_sku = Column(String(100), nullable=False)
	product_name = Column(String(200), nullable=False)
	
	# Recipe properties
	recipe_type = Column(String(30), nullable=False, default="production")
	formula_type = Column(String(30), nullable=False, default="master")
	batch_size = Column(Numeric(15, 4), nullable=False)
	batch_unit = Column(String(20), nullable=False)
	yield_percentage = Column(Numeric(5, 2), default=100)
	
	# Timing
	total_process_time_minutes = Column(Integer)
	mixing_time_minutes = Column(Integer)
	reaction_time_minutes = Column(Integer)
	
	# Conditions
	process_temperature_min = Column(Numeric(8, 2))
	process_temperature_max = Column(Numeric(8, 2))
	process_pressure_min = Column(Numeric(10, 4))
	process_pressure_max = Column(Numeric(10, 4))  
	ph_min = Column(Numeric(4, 2))
	ph_max = Column(Numeric(4, 2))
	
	# Effectivity
	effective_date = Column(Date, nullable=False)
	expiry_date = Column(Date)
	status = Column(String(20), nullable=False, default="draft")
	
	# Facility constraints
	facility_id = Column(String(36), index=True)
	equipment_required = Column(Text)  # JSON array of equipment IDs
	
	# Regulatory information
	regulatory_approval_number = Column(String(100))
	gmp_required = Column(Boolean, default=False)
	fda_approved = Column(Boolean, default=False)
	allergen_information = Column(Text)
	
	# Quality specifications
	quality_specifications = Column(Text)  # JSON object with specs
	critical_control_points = Column(Text)  # JSON array of CCPs
	
	# Approval workflow
	developed_by = Column(String(36), nullable=False)
	reviewed_by = Column(String(36))
	reviewed_at = Column(DateTime)
	approved_by = Column(String(36))
	approved_at = Column(DateTime)
	
	# Documentation
	description = Column(Text)
	process_notes = Column(Text)
	safety_instructions = Column(Text)
	environmental_considerations = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	ingredients = relationship("MFRRecipeIngredient", back_populates="recipe")
	process_steps = relationship("MFRProcessStep", back_populates="recipe")
	batch_records = relationship("MFRBatchRecord", back_populates="recipe")
	
	__table_args__ = (
		Index('idx_mfr_recipe_tenant_product', 'tenant_id', 'product_id'),
		Index('idx_mfr_recipe_number', 'recipe_number'),
		Index('idx_mfr_recipe_status_effective', 'status', 'effective_date'),
	)

class MFRRecipeIngredient(SQLBaseModel):
	"""Recipe ingredients and material specifications"""
	__tablename__ = 'mfr_recipe_ingredients'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Recipe reference
	recipe_id = Column(String(36), ForeignKey('mfr_master_recipes.id'), nullable=False, index=True)
	
	# Ingredient identification
	sequence_number = Column(Integer, nullable=False)
	ingredient_id = Column(String(36), nullable=False, index=True)
	ingredient_sku = Column(String(100), nullable=False)
	ingredient_name = Column(String(200), nullable=False)
	ingredient_type = Column(String(30), nullable=False)
	
	# Quantities
	target_quantity = Column(Numeric(15, 6), nullable=False)
	minimum_quantity = Column(Numeric(15, 6))
	maximum_quantity = Column(Numeric(15, 6))
	unit_of_measure = Column(String(20), nullable=False)
	
	# Percentages
	percentage_by_weight = Column(Numeric(8, 4))
	percentage_by_volume = Column(Numeric(8, 4))
	
	# Specifications
	grade_specification = Column(String(100))
	purity_percentage = Column(Numeric(5, 2))
	mesh_size = Column(String(20))
	
	# Addition parameters
	addition_order = Column(Integer)
	addition_method = Column(String(100))
	addition_temperature = Column(Numeric(8, 2))
	addition_rate = Column(String(100))
	addition_time_minutes = Column(Integer)
	
	# Quality requirements
	is_critical_ingredient = Column(Boolean, default=False)
	testing_required = Column(Boolean, default=False)
	certificate_of_analysis_required = Column(Boolean, default=False)
	
	# Regulatory
	is_controlled_substance = Column(Boolean, default=False)
	regulatory_classification = Column(String(50))
	
	# Sourcing
	preferred_supplier_id = Column(String(36))
	supplier_part_number = Column(String(100))
	
	# Substitution
	is_substitutable = Column(Boolean, default=False)
	substitute_ingredients = Column(Text)  # JSON array of substitute ingredient IDs
	
	# Cost
	unit_cost = Column(Numeric(12, 4))
	extended_cost = Column(Numeric(15, 4))
	
	# Notes and instructions
	handling_instructions = Column(Text)
	storage_requirements = Column(Text)
	safety_notes = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	recipe = relationship("MFRMasterRecipe", back_populates="ingredients")
	
	__table_args__ = (
		Index('idx_mfr_ing_tenant_recipe', 'tenant_id', 'recipe_id'),
		Index('idx_mfr_ing_sequence', 'recipe_id', 'sequence_number'),
		Index('idx_mfr_ing_ingredient', 'ingredient_id'),
	)

class MFRProcessStep(SQLBaseModel):
	"""Recipe process steps and instructions"""
	__tablename__ = 'mfr_process_steps'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Recipe reference
	recipe_id = Column(String(36), ForeignKey('mfr_master_recipes.id'), nullable=False, index=True)
	
	# Step identification
	step_number = Column(Integer, nullable=False)
	step_name = Column(String(200), nullable=False)
	step_type = Column(String(30), nullable=False)
	
	# Instructions
	instruction_text = Column(Text, nullable=False)
	detailed_procedure = Column(Text)
	
	# Timing
	duration_minutes = Column(Integer)
	minimum_time_minutes = Column(Integer)
	maximum_time_minutes = Column(Integer)
	
	# Process parameters
	temperature_setpoint = Column(Numeric(8, 2))
	temperature_tolerance = Column(Numeric(5, 2))
	pressure_setpoint = Column(Numeric(10, 4))
	pressure_tolerance = Column(Numeric(8, 4))
	speed_rpm = Column(Integer)
	ph_target = Column(Numeric(4, 2))
	ph_tolerance = Column(Numeric(3, 2))
	
	# Equipment requirements
	equipment_required = Column(String(200))
	equipment_settings = Column(Text)  # JSON object with settings
	
	# Quality controls
	in_process_checks = Column(Text)  # JSON array of check requirements
	critical_control_point = Column(Boolean, default=False)
	acceptance_criteria = Column(Text)
	
	# Materials involved
	ingredients_added = Column(Text)  # JSON array of ingredient IDs
	materials_consumed = Column(Text)  # JSON array of consumable material IDs
	
	# Safety and environmental
	safety_precautions = Column(Text)
	environmental_controls = Column(Text)
	ppe_required = Column(String(200))
	
	# Hold points and approvals
	requires_approval = Column(Boolean, default=False)
	approval_role = Column(String(100))
	hold_point = Column(Boolean, default=False)
	
	# Parallel/sequential control
	predecessor_steps = Column(Text)  # JSON array of prerequisite step numbers
	can_run_parallel = Column(Boolean, default=False)
	parallel_with_steps = Column(Text)  # JSON array of parallel step numbers
	
	# Documentation
	attachments = Column(Text)  # JSON array of document references
	notes = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	recipe = relationship("MFRMasterRecipe", back_populates="process_steps")
	
	__table_args__ = (
		Index('idx_mfr_step_tenant_recipe', 'tenant_id', 'recipe_id'),
		Index('idx_mfr_step_number', 'recipe_id', 'step_number'),
	)

class MFRBatchRecord(SQLBaseModel):
	"""Batch production records for recipe execution"""
	__tablename__ = 'mfr_batch_records'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Batch identification
	batch_number = Column(String(50), nullable=False, unique=True)
	lot_number = Column(String(50), index=True)
	
	# Recipe reference
	recipe_id = Column(String(36), ForeignKey('mfr_master_recipes.id'), nullable=False, index=True)
	recipe_version = Column(String(20), nullable=False)
	
	# Production order
	production_order_id = Column(String(36), index=True)
	
	# Batch details
	batch_size_planned = Column(Numeric(15, 4), nullable=False)
	batch_size_actual = Column(Numeric(15, 4))
	batch_unit = Column(String(20), nullable=False)
	
	# Dates and scheduling
	scheduled_start_date = Column(DateTime)
	actual_start_date = Column(DateTime)
	scheduled_end_date = Column(DateTime)
	actual_end_date = Column(DateTime)
	
	# Personnel
	batch_supervisor = Column(String(36), nullable=False)
	quality_representative = Column(String(36))
	operators = Column(Text)  # JSON array of operator IDs
	
	# Location and equipment
	facility_id = Column(String(36), nullable=False, index=True)
	production_line = Column(String(100))
	equipment_used = Column(Text)  # JSON array of equipment IDs
	
	# Status and results  
	status = Column(String(20), nullable=False, default="created")
	yield_percentage = Column(Numeric(5, 2))
	quantity_produced = Column(Numeric(15, 4))
	quantity_rejected = Column(Numeric(15, 4))
	
	# Quality results
	quality_status = Column(String(20))  # pass, fail, pending
	quality_notes = Column(Text)
	quality_approved_by = Column(String(36))
	quality_approved_at = Column(DateTime)
	
	# Deviations and exceptions
	deviations_reported = Column(Boolean, default=False)
	deviation_summary = Column(Text)
	investigation_required = Column(Boolean, default=False)
	
	# Environmental conditions
	ambient_temperature = Column(Numeric(6, 2))
	ambient_humidity = Column(Numeric(5, 2))
	environmental_notes = Column(Text)
	
	# Material consumption
	materials_consumed = Column(Text)  # JSON with actual quantities used
	material_waste = Column(Numeric(15, 4))
	
	# Documentation and signatures
	electronic_signatures = Column(Text)  # JSON array of signature records
	attachments = Column(Text)  # JSON array of document references
	
	# Review and approval
	reviewed_by = Column(String(36))
	reviewed_at = Column(DateTime)
	approved_by = Column(String(36))
	approved_at = Column(DateTime)
	approval_notes = Column(Text)
	
	# Audit trail
	created_by = Column(String(36), nullable=False)
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_by = Column(String(36))
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	recipe = relationship("MFRMasterRecipe", back_populates="batch_records")
	step_executions = relationship("MFRBatchStepExecution", back_populates="batch_record")
	
	__table_args__ = (
		Index('idx_mfr_batch_tenant_status', 'tenant_id', 'status'),
		Index('idx_mfr_batch_number', 'batch_number'),
		Index('idx_mfr_batch_recipe', 'recipe_id'),
		Index('idx_mfr_batch_dates', 'actual_start_date', 'actual_end_date'),
	)

class MFRBatchStepExecution(SQLBaseModel):
	"""Execution records for batch process steps"""
	__tablename__ = 'mfr_batch_step_executions'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Batch record reference
	batch_record_id = Column(String(36), ForeignKey('mfr_batch_records.id'), nullable=False, index=True)
	
	# Step information
	step_number = Column(Integer, nullable=False)
	step_name = Column(String(200), nullable=False)
	step_type = Column(String(30), nullable=False)
	
	# Execution timing
	started_at = Column(DateTime)
	completed_at = Column(DateTime)
	duration_minutes = Column(Integer)
	
	# Process parameters achieved
	actual_temperature = Column(Numeric(8, 2))
	actual_pressure = Column(Numeric(10, 4))
	actual_speed_rpm = Column(Integer)
	actual_ph = Column(Numeric(4, 2))
	
	# Personnel
	executed_by = Column(String(36), nullable=False)
	verified_by = Column(String(36))
	
	# Materials used
	ingredients_consumed = Column(Text)  # JSON with actual quantities
	yield_obtained = Column(Numeric(15, 4))
	
	# Quality checks
	in_process_test_results = Column(Text)  # JSON with test data
	quality_status = Column(String(20))  # pass, fail, n/a
	
	# Deviations
	deviation_occurred = Column(Boolean, default=False)
	deviation_description = Column(Text)
	corrective_action_taken = Column(Text)
	
	# Equipment and conditions
	equipment_used = Column(String(200))
	environmental_conditions = Column(Text)  # JSON with temp, humidity, etc.
	
	# Comments and observations
	operator_comments = Column(Text)
	observations = Column(Text)
	
	# Electronic signature
	electronic_signature = Column(Text)  # JSON signature record
	signature_timestamp = Column(DateTime)
	
	# Audit trail
	created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	# Relationships
	batch_record = relationship("MFRBatchRecord", back_populates="step_executions")
	
	__table_args__ = (
		Index('idx_mfr_exec_tenant_batch', 'tenant_id', 'batch_record_id'),
		Index('idx_mfr_exec_step', 'batch_record_id', 'step_number'),
	)

# Pydantic Models for API

class MasterRecipeCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	recipe_number: str = Field(..., min_length=1, max_length=50)
	recipe_name: str = Field(..., min_length=1, max_length=200)
	version: str = Field(default="1.0", max_length=20)
	revision: str = Field(default="A", max_length=10)
	product_id: str = Field(..., min_length=36, max_length=36)
	product_sku: str = Field(..., min_length=1, max_length=100)
	product_name: str = Field(..., min_length=1, max_length=200)
	recipe_type: str = Field(default="production", max_length=30)
	formula_type: FormulaType = FormulaType.MASTER
	batch_size: Decimal = Field(..., gt=0)
	batch_unit: str = Field(..., min_length=1, max_length=20)
	yield_percentage: Decimal = Field(default=Decimal('100'), gt=0, le=100)
	total_process_time_minutes: int | None = None
	mixing_time_minutes: int | None = None
	reaction_time_minutes: int | None = None
	process_temperature_min: Decimal | None = None
	process_temperature_max: Decimal | None = None
	process_pressure_min: Decimal | None = None
	process_pressure_max: Decimal | None = None
	ph_min: Decimal | None = None
	ph_max: Decimal | None = None
	effective_date: date
	expiry_date: date | None = None
	facility_id: str | None = None
	equipment_required: str | None = None
	regulatory_approval_number: str | None = None
	gmp_required: bool = False
	fda_approved: bool = False
	allergen_information: str | None = None
	quality_specifications: str | None = None
	critical_control_points: str | None = None
	description: str | None = None
	process_notes: str | None = None
	safety_instructions: str | None = None
	environmental_considerations: str | None = None

class RecipeIngredientCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	sequence_number: int = Field(..., ge=1)
	ingredient_id: str = Field(..., min_length=36, max_length=36)
	ingredient_sku: str = Field(..., min_length=1, max_length=100)
	ingredient_name: str = Field(..., min_length=1, max_length=200)
	ingredient_type: IngredientType
	target_quantity: Decimal = Field(..., gt=0)
	minimum_quantity: Decimal | None = None
	maximum_quantity: Decimal | None = None
	unit_of_measure: str = Field(..., min_length=1, max_length=20)
	percentage_by_weight: Decimal | None = None
	percentage_by_volume: Decimal | None = None
	grade_specification: str | None = None
	purity_percentage: Decimal | None = Field(None, ge=0, le=100)
	mesh_size: str | None = None
	addition_order: int | None = None
	addition_method: str | None = None
	addition_temperature: Decimal | None = None
	addition_rate: str | None = None
	addition_time_minutes: int | None = None
	is_critical_ingredient: bool = False
	testing_required: bool = False
	certificate_of_analysis_required: bool = False
	is_controlled_substance: bool = False
	regulatory_classification: str | None = None
	preferred_supplier_id: str | None = None
	supplier_part_number: str | None = None
	is_substitutable: bool = False
	substitute_ingredients: str | None = None
	unit_cost: Decimal | None = None
	handling_instructions: str | None = None
	storage_requirements: str | None = None
	safety_notes: str | None = None

class ProcessStepCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	step_number: int = Field(..., ge=1)
	step_name: str = Field(..., min_length=1, max_length=200)
	step_type: ProcessStepType
	instruction_text: str = Field(..., min_length=1)
	detailed_procedure: str | None = None
	duration_minutes: int | None = None
	minimum_time_minutes: int | None = None
	maximum_time_minutes: int | None = None
	temperature_setpoint: Decimal | None = None
	temperature_tolerance: Decimal | None = None
	pressure_setpoint: Decimal | None = None
	pressure_tolerance: Decimal | None = None
	speed_rpm: int | None = None
	ph_target: Decimal | None = None
	ph_tolerance: Decimal | None = None
	equipment_required: str | None = None
	equipment_settings: str | None = None
	in_process_checks: str | None = None
	critical_control_point: bool = False
	acceptance_criteria: str | None = None
	ingredients_added: str | None = None
	materials_consumed: str | None = None
	safety_precautions: str | None = None
	environmental_controls: str | None = None
	ppe_required: str | None = None
	requires_approval: bool = False
	approval_role: str | None = None
	hold_point: bool = False
	predecessor_steps: str | None = None
	can_run_parallel: bool = False
	parallel_with_steps: str | None = None
	attachments: str | None = None
	notes: str | None = None

class BatchRecordCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	batch_number: str = Field(..., min_length=1, max_length=50)
	lot_number: str | None = None
	recipe_id: str = Field(..., min_length=36, max_length=36)
	recipe_version: str = Field(..., min_length=1, max_length=20)
	production_order_id: str | None = None
	batch_size_planned: Decimal = Field(..., gt=0)
	batch_unit: str = Field(..., min_length=1, max_length=20)
	scheduled_start_date: datetime | None = None
	scheduled_end_date: datetime | None = None
	batch_supervisor: str = Field(..., min_length=36, max_length=36)
	quality_representative: str | None = None
	operators: str | None = None
	facility_id: str = Field(..., min_length=36, max_length=36)
	production_line: str | None = None
	equipment_used: str | None = None
	environmental_notes: str | None = None

__all__ = [
	"RecipeStatus", "FormulaType", "IngredientType", "ProcessStepType", "BatchRecordStatus",
	"MFRMasterRecipe", "MFRRecipeIngredient", "MFRProcessStep", "MFRBatchRecord", "MFRBatchStepExecution",
	"MasterRecipeCreate", "RecipeIngredientCreate", "ProcessStepCreate", "BatchRecordCreate"
]