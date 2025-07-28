"""
APG Employee Data Management - Revolutionary AI-Powered Models

Enhanced database models with AI intelligence, predictive analytics, and immersive
features for 10x improvement over market leaders like Workday and BambooHR.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, JSON, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, VECTOR
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from annotated_types import Annotated

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class HRDepartment(Model, AuditMixin, BaseMixin):
	"""
	Department structure for organizational hierarchy.
	
	Supports hierarchical department structure with parent-child relationships.
	"""
	__tablename__ = 'hr_edm_department'
	
	# Identity
	department_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Department Information
	department_code = Column(String(20), nullable=False, index=True)
	department_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Hierarchy
	parent_department_id = Column(String(36), ForeignKey('hr_edm_department.department_id'), nullable=True, index=True)
	level = Column(Integer, default=0)
	path = Column(String(500), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	cost_center = Column(String(20), nullable=True)
	budget_allocation = Column(DECIMAL(15, 2), nullable=True)
	
	# Manager
	manager_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=True, index=True)
	
	# Location
	location = Column(String(200), nullable=True)
	address = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'department_code', name='uq_department_code_tenant'),
	)
	
	# Relationships
	parent_department = relationship("HRDepartment", remote_side=[department_id])
	child_departments = relationship("HRDepartment")
	employees = relationship("HREmployee", back_populates="department")
	positions = relationship("HRPosition", back_populates="department")
	manager = relationship("HREmployee", foreign_keys=[manager_id])
	
	def __repr__(self):
		return f"<HRDepartment {self.department_name}>"


class HRPosition(Model, AuditMixin, BaseMixin):
	"""
	Position/job definitions within the organization.
	
	Defines roles, responsibilities, and requirements for positions.
	"""
	__tablename__ = 'hr_edm_position'
	
	# Identity
	position_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Position Information
	position_code = Column(String(20), nullable=False, index=True)
	position_title = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	responsibilities = Column(Text, nullable=True)
	requirements = Column(Text, nullable=True)
	
	# Classification
	department_id = Column(String(36), ForeignKey('hr_edm_department.department_id'), nullable=False, index=True)
	job_level = Column(String(50), nullable=True)  # Executive, Manager, Individual Contributor
	job_family = Column(String(100), nullable=True)  # Engineering, Sales, Marketing, etc.
	
	# Compensation
	min_salary = Column(DECIMAL(12, 2), nullable=True)
	max_salary = Column(DECIMAL(12, 2), nullable=True)
	currency_code = Column(String(3), default='USD')
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_exempt = Column(Boolean, default=True)  # Exempt from overtime
	reports_to_position_id = Column(String(36), ForeignKey('hr_edm_position.position_id'), nullable=True, index=True)
	
	# Headcount
	authorized_headcount = Column(Integer, default=1)
	current_headcount = Column(Integer, default=0)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'position_code', name='uq_position_code_tenant'),
	)
	
	# Relationships
	department = relationship("HRDepartment", back_populates="positions")
	employees = relationship("HREmployee", back_populates="position")
	reports_to_position = relationship("HRPosition", remote_side=[position_id])
	direct_reports = relationship("HRPosition")
	required_skills = relationship("HRPositionSkill", back_populates="position")
	
	def __repr__(self):
		return f"<HRPosition {self.position_title}>"


class HREmployee(Model, AuditMixin, BaseMixin):
	"""
	Core employee record with personal and employment information.
	
	Central employee entity linking to all other HR modules.
	"""
	__tablename__ = 'hr_edm_employee'
	
	# Identity
	employee_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Employee Number
	employee_number = Column(String(20), nullable=False, index=True)  # EMP000001
	badge_id = Column(String(20), nullable=True, index=True)
	
	# Personal Information
	first_name = Column(String(100), nullable=False, index=True)
	middle_name = Column(String(100), nullable=True)
	last_name = Column(String(100), nullable=False, index=True) 
	preferred_name = Column(String(100), nullable=True)
	full_name = Column(String(300), nullable=False, index=True)  # Computed field
	
	# Contact Information
	personal_email = Column(String(200), nullable=True, index=True)
	work_email = Column(String(200), nullable=True, index=True)
	phone_mobile = Column(String(20), nullable=True)
	phone_home = Column(String(20), nullable=True)
	phone_work = Column(String(20), nullable=True)
	
	# Demographics
	date_of_birth = Column(Date, nullable=True)
	gender = Column(String(20), nullable=True)
	marital_status = Column(String(20), nullable=True)
	nationality = Column(String(100), nullable=True)
	
	# Address
	address_line1 = Column(String(200), nullable=True)
	address_line2 = Column(String(200), nullable=True)
	city = Column(String(100), nullable=True)
	state_province = Column(String(100), nullable=True)
	postal_code = Column(String(20), nullable=True)
	country = Column(String(100), nullable=True)
	
	# Employment Information
	department_id = Column(String(36), ForeignKey('hr_edm_department.department_id'), nullable=False, index=True)
	position_id = Column(String(36), ForeignKey('hr_edm_position.position_id'), nullable=False, index=True)
	manager_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=True, index=True)
	
	# Employment Dates
	hire_date = Column(Date, nullable=False, index=True)
	start_date = Column(Date, nullable=True)  # May differ from hire_date
	termination_date = Column(Date, nullable=True, index=True)
	rehire_date = Column(Date, nullable=True)
	
	# Employment Status
	employment_status = Column(String(20), default='Active', index=True)  # Active, Inactive, Terminated, Leave
	employment_type = Column(String(20), default='Full-Time', index=True)  # Full-Time, Part-Time, Contract, Intern
	work_location = Column(String(20), default='Office', index=True)  # Office, Remote, Hybrid
	
	# Compensation
	base_salary = Column(DECIMAL(12, 2), nullable=True)
	hourly_rate = Column(DECIMAL(8, 2), nullable=True)
	currency_code = Column(String(3), default='USD')
	pay_frequency = Column(String(20), default='Monthly')  # Weekly, Bi-Weekly, Monthly, Annual
	
	# Benefits Eligibility
	benefits_eligible = Column(Boolean, default=True)
	benefits_start_date = Column(Date, nullable=True)
	
	# Tax Information
	tax_id = Column(String(50), nullable=True)  # SSN, TIN, etc.
	tax_country = Column(String(3), default='USA')
	tax_state = Column(String(50), nullable=True)
	
	# Performance & Review
	probation_end_date = Column(Date, nullable=True)
	next_review_date = Column(Date, nullable=True)
	performance_rating = Column(String(20), nullable=True)
	
	# System Fields
	is_active = Column(Boolean, default=True, index=True)
	is_system_user = Column(Boolean, default=False)
	system_user_id = Column(String(36), nullable=True)  # Link to auth system
	
	# Photo and Documents
	photo_url = Column(String(500), nullable=True)
	documents_folder = Column(String(500), nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'employee_number', name='uq_employee_number_tenant'),
		UniqueConstraint('tenant_id', 'work_email', name='uq_work_email_tenant'),
	)
	
	# Relationships
	department = relationship("HRDepartment", back_populates="employees")
	position = relationship("HRPosition", back_populates="employees")
	manager = relationship("HREmployee", remote_side=[employee_id])
	direct_reports = relationship("HREmployee")
	
	personal_info = relationship("HRPersonalInfo", back_populates="employee", uselist=False)
	emergency_contacts = relationship("HREmergencyContact", back_populates="employee")
	employment_history = relationship("HREmploymentHistory", back_populates="employee")
	employee_skills = relationship("HREmployeeSkill", back_populates="employee")
	employee_certifications = relationship("HREmployeeCertification", back_populates="employee")
	
	def __repr__(self):
		return f"<HREmployee {self.full_name} ({self.employee_number})>"


class HRPersonalInfo(Model, AuditMixin, BaseMixin):
	"""
	Extended personal information for employees.
	
	Sensitive personal data stored separately for access control.
	"""
	__tablename__ = 'hr_edm_personal_info'
	
	# Identity
	personal_info_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, unique=True, index=True)
	
	# Identification
	passport_number = Column(String(50), nullable=True)
	passport_country = Column(String(3), nullable=True)
	passport_expiry = Column(Date, nullable=True)
	drivers_license = Column(String(50), nullable=True)
	drivers_license_state = Column(String(50), nullable=True)
	drivers_license_expiry = Column(Date, nullable=True)
	
	# Banking (for payroll)
	bank_name = Column(String(200), nullable=True)
	bank_account_number = Column(String(50), nullable=True)
	bank_routing_number = Column(String(20), nullable=True)
	bank_account_type = Column(String(20), nullable=True)  # Checking, Savings
	
	# Insurance
	health_insurance_id = Column(String(50), nullable=True)
	life_insurance_beneficiary = Column(String(200), nullable=True)
	
	# Veteran Status
	veteran_status = Column(String(50), nullable=True)
	disability_status = Column(String(50), nullable=True)
	
	# Visa/Work Authorization
	work_authorization = Column(String(50), nullable=True)
	visa_type = Column(String(50), nullable=True)
	visa_expiry = Column(Date, nullable=True)
	i9_verified = Column(Boolean, default=False)
	i9_verification_date = Column(Date, nullable=True)
	
	# Relationships
	employee = relationship("HREmployee", back_populates="personal_info")
	
	def __repr__(self):
		return f"<HRPersonalInfo for {self.employee_id}>"


class HREmergencyContact(Model, AuditMixin, BaseMixin):
	"""
	Emergency contact information for employees.
	"""
	__tablename__ = 'hr_edm_emergency_contact'
	
	# Identity
	contact_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, index=True)
	
	# Contact Information
	contact_name = Column(String(200), nullable=False)
	relationship = Column(String(50), nullable=False)  # Spouse, Parent, Sibling, Friend, etc.
	phone_primary = Column(String(20), nullable=False)
	phone_secondary = Column(String(20), nullable=True)
	email = Column(String(200), nullable=True)
	
	# Address
	address_line1 = Column(String(200), nullable=True)
	address_line2 = Column(String(200), nullable=True)
	city = Column(String(100), nullable=True)
	state_province = Column(String(100), nullable=True)
	postal_code = Column(String(20), nullable=True)
	country = Column(String(100), nullable=True)
	
	# Priority
	is_primary = Column(Boolean, default=False)
	priority_order = Column(Integer, default=1)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Relationships
	employee = relationship("HREmployee", back_populates="emergency_contacts")
	
	def __repr__(self):
		return f"<HREmergencyContact {self.contact_name} for {self.employee_id}>"


class HREmploymentHistory(Model, AuditMixin, BaseMixin):
	"""
	Track employment history and position changes.
	
	Maintains audit trail of all position, department, and compensation changes.
	"""
	__tablename__ = 'hr_edm_employment_history'
	
	# Identity
	history_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, index=True)
	
	# Change Information
	change_type = Column(String(50), nullable=False, index=True)  # Hire, Promotion, Transfer, Termination, etc.
	effective_date = Column(Date, nullable=False, index=True)
	reason = Column(String(200), nullable=True)
	notes = Column(Text, nullable=True)
	
	# Previous Values
	previous_department_id = Column(String(36), nullable=True)
	previous_position_id = Column(String(36), nullable=True)
	previous_manager_id = Column(String(36), nullable=True)
	previous_salary = Column(DECIMAL(12, 2), nullable=True)
	previous_status = Column(String(50), nullable=True)
	
	# New Values
	new_department_id = Column(String(36), nullable=True)
	new_position_id = Column(String(36), nullable=True)
	new_manager_id = Column(String(36), nullable=True)
	new_salary = Column(DECIMAL(12, 2), nullable=True)
	new_status = Column(String(50), nullable=True)
	
	# Approval
	approved_by = Column(String(36), nullable=True)  # Employee ID of approver
	approval_date = Column(Date, nullable=True)
	
	# Relationships
	employee = relationship("HREmployee", back_populates="employment_history")
	
	def __repr__(self):
		return f"<HREmploymentHistory {self.change_type} for {self.employee_id}>"


class HRSkill(Model, AuditMixin, BaseMixin):
	"""
	Skills catalog for competency management.
	
	Defines available skills that can be assigned to employees and positions.
	"""
	__tablename__ = 'hr_edm_skill'
	
	# Identity
	skill_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Skill Information
	skill_code = Column(String(20), nullable=False, index=True)
	skill_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Classification
	skill_category = Column(String(100), nullable=True, index=True)  # Technical, Leadership, Communication, etc.
	skill_type = Column(String(50), nullable=True)  # Hard, Soft
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_core_competency = Column(Boolean, default=False)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'skill_code', name='uq_skill_code_tenant'),
	)
	
	# Relationships
	employee_skills = relationship("HREmployeeSkill", back_populates="skill")
	position_skills = relationship("HRPositionSkill", back_populates="skill")
	
	def __repr__(self):
		return f"<HRSkill {self.skill_name}>"


class HREmployeeSkill(Model, AuditMixin, BaseMixin):
	"""
	Employee skill assignments with proficiency levels.
	
	Links employees to skills with proficiency ratings and validation.
	"""
	__tablename__ = 'hr_edm_employee_skill'
	
	# Identity
	employee_skill_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, index=True)
	skill_id = Column(String(36), ForeignKey('hr_edm_skill.skill_id'), nullable=False, index=True)
	
	# Proficiency
	proficiency_level = Column(String(20), nullable=False)  # Beginner, Intermediate, Advanced, Expert
	proficiency_score = Column(Integer, nullable=True)  # 1-10 scale
	
	# Validation
	self_assessed = Column(Boolean, default=True)
	manager_validated = Column(Boolean, default=False)
	validated_by = Column(String(36), nullable=True)  # Employee ID of validator
	validation_date = Column(Date, nullable=True)
	
	# Experience
	years_experience = Column(DECIMAL(4, 2), nullable=True)
	last_used_date = Column(Date, nullable=True)
	
	# Evidence
	evidence_notes = Column(Text, nullable=True)
	certification_reference = Column(String(36), nullable=True)  # Link to certification
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_primary = Column(Boolean, default=False)  # Primary skill for role
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'employee_id', 'skill_id', name='uq_employee_skill'),
	)
	
	# Relationships
	employee = relationship("HREmployee", back_populates="employee_skills")
	skill = relationship("HRSkill", back_populates="employee_skills")
	
	def __repr__(self):
		return f"<HREmployeeSkill {self.employee_id} - {self.skill_id}>"


class HRPositionSkill(Model, AuditMixin, BaseMixin):
	"""
	Position skill requirements.
	
	Defines required and preferred skills for positions.
	"""
	__tablename__ = 'hr_edm_position_skill'
	
	# Identity
	position_skill_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	position_id = Column(String(36), ForeignKey('hr_edm_position.position_id'), nullable=False, index=True)
	skill_id = Column(String(36), ForeignKey('hr_edm_skill.skill_id'), nullable=False, index=True)
	
	# Requirements
	requirement_level = Column(String(20), nullable=False)  # Required, Preferred, Nice-to-Have
	minimum_proficiency = Column(String(20), nullable=False)  # Beginner, Intermediate, Advanced, Expert
	minimum_years_experience = Column(DECIMAL(4, 2), nullable=True)
	
	# Priority
	priority = Column(Integer, default=1)  # 1 = highest priority
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'position_id', 'skill_id', name='uq_position_skill'),
	)
	
	# Relationships
	position = relationship("HRPosition", back_populates="required_skills")
	skill = relationship("HRSkill", back_populates="position_skills")
	
	def __repr__(self):
		return f"<HRPositionSkill {self.position_id} - {self.skill_id}>"


class HRCertification(Model, AuditMixin, BaseMixin):
	"""
	Certification catalog for professional certifications.
	
	Defines available certifications that employees can obtain.
	"""
	__tablename__ = 'hr_edm_certification'
	
	# Identity
	certification_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Certification Information
	certification_code = Column(String(20), nullable=False, index=True)
	certification_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Provider
	issuing_organization = Column(String(200), nullable=False)
	organization_website = Column(String(500), nullable=True)
	
	# Classification
	certification_category = Column(String(100), nullable=True, index=True)  # Technical, Professional, Safety, etc.
	industry = Column(String(100), nullable=True)
	
	# Validity
	validity_period_months = Column(Integer, nullable=True)  # Months before renewal required
	is_renewable = Column(Boolean, default=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_continuing_education = Column(Boolean, default=False)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'certification_code', name='uq_certification_code_tenant'),
	)
	
	# Relationships
	employee_certifications = relationship("HREmployeeCertification", back_populates="certification")
	
	def __repr__(self):
		return f"<HRCertification {self.certification_name}>"


class HREmployeeCertification(Model, AuditMixin, BaseMixin):
	"""
	Employee certification records.
	
	Tracks certifications obtained by employees with validity and renewal tracking.
	"""
	__tablename__ = 'hr_edm_employee_certification'
	
	# Identity
	employee_certification_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, index=True)
	certification_id = Column(String(36), ForeignKey('hr_edm_certification.certification_id'), nullable=False, index=True)
	
	# Certificate Details
	certificate_number = Column(String(100), nullable=True)
	issued_date = Column(Date, nullable=False, index=True)
	expiry_date = Column(Date, nullable=True, index=True)
	renewal_date = Column(Date, nullable=True)
	
	# Status
	status = Column(String(20), default='Active', index=True)  # Active, Expired, Suspended, Revoked
	
	# Verification
	verified = Column(Boolean, default=False)
	verified_by = Column(String(36), nullable=True)  # Employee ID of verifier
	verification_date = Column(Date, nullable=True)
	verification_notes = Column(Text, nullable=True)
	
	# Scoring
	score = Column(String(20), nullable=True)  # Pass/Fail, Percentage, Grade
	score_details = Column(Text, nullable=True)
	
	# Documentation
	certificate_file_path = Column(String(500), nullable=True)
	documentation_notes = Column(Text, nullable=True)
	
	# Cost Tracking
	cost = Column(DECIMAL(10, 2), nullable=True)
	reimbursed = Column(Boolean, default=False)
	reimbursement_amount = Column(DECIMAL(10, 2), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'employee_id', 'certification_id', 'issued_date', name='uq_employee_certification'),
	)
	
	# Relationships
	employee = relationship("HREmployee", back_populates="employee_certifications")
	certification = relationship("HRCertification", back_populates="employee_certifications")
	
	def __repr__(self):
		return f"<HREmployeeCertification {self.employee_id} - {self.certification_id}>"


# ============================================================================
# REVOLUTIONARY AI-POWERED MODELS FOR 10X IMPROVEMENT
# ============================================================================

class EmployeeEngagementLevel(str, Enum):
	"""Employee engagement levels for predictive analytics."""
	DISENGAGED = "disengaged"
	SOMEWHAT_ENGAGED = "somewhat_engaged"
	ENGAGED = "engaged"
	HIGHLY_ENGAGED = "highly_engaged"
	CHAMPION = "champion"


class RiskLevel(str, Enum):
	"""Risk assessment levels."""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class AIInsightType(str, Enum):
	"""Types of AI-generated insights."""
	CAREER_RECOMMENDATION = "career_recommendation"
	SKILL_DEVELOPMENT = "skill_development"
	PERFORMANCE_PREDICTION = "performance_prediction"
	RETENTION_RISK = "retention_risk"
	PROMOTION_READINESS = "promotion_readiness"
	LEARNING_SUGGESTION = "learning_suggestion"
	COMPENSATION_ANALYSIS = "compensation_analysis"
	TEAM_DYNAMICS = "team_dynamics"


class HREmployeeAIProfile(Model, AuditMixin, BaseMixin):
	"""
	AI-powered employee profile with predictive insights and recommendations.
	
	Revolutionary feature #1: AI-Powered Employee Intelligence Engine
	"""
	__tablename__ = 'hr_edm_ai_profile'
	
	# Identity
	ai_profile_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, unique=True, index=True)
	
	# AI Embeddings for Semantic Search
	profile_embedding = Column(VECTOR(1536), nullable=True)  # OpenAI embeddings
	skills_embedding = Column(VECTOR(1536), nullable=True)
	career_embedding = Column(VECTOR(1536), nullable=True)
	
	# Predictive Analytics
	retention_risk_score = Column(DECIMAL(5, 4), nullable=True, index=True)  # 0.0000 to 1.0000
	performance_prediction = Column(DECIMAL(5, 4), nullable=True)
	promotion_readiness_score = Column(DECIMAL(5, 4), nullable=True)
	engagement_score = Column(DECIMAL(5, 4), nullable=True)
	engagement_level = Column(String(20), nullable=True, index=True)
	
	# Career Path Intelligence
	suggested_career_paths = Column(JSONB, nullable=True)
	skill_gap_analysis = Column(JSONB, nullable=True)
	learning_recommendations = Column(JSONB, nullable=True)
	
	# Behavioral Analytics
	communication_style = Column(String(50), nullable=True)
	work_preferences = Column(JSONB, nullable=True)
	collaboration_patterns = Column(JSONB, nullable=True)
	productivity_metrics = Column(JSONB, nullable=True)
	
	# AI Model Metadata
	last_ai_analysis = Column(DateTime, nullable=True, index=True)
	ai_model_version = Column(String(20), nullable=True)
	confidence_score = Column(DECIMAL(5, 4), nullable=True)
	
	# Relationships
	employee = relationship("HREmployee", foreign_keys=[employee_id])
	ai_insights = relationship("HREmployeeAIInsight", back_populates="ai_profile")
	
	def __repr__(self):
		return f"<HREmployeeAIProfile {self.employee_id}>"


class HREmployeeAIInsight(Model, AuditMixin, BaseMixin):
	"""
	AI-generated insights and recommendations for employees.
	
	Revolutionary feature #1: AI-Powered Employee Intelligence Engine
	"""
	__tablename__ = 'hr_edm_ai_insight'
	
	# Identity
	insight_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	ai_profile_id = Column(String(36), ForeignKey('hr_edm_ai_profile.ai_profile_id'), nullable=False, index=True)
	
	# Insight Details
	insight_type = Column(String(50), nullable=False, index=True)
	title = Column(String(200), nullable=False)
	description = Column(Text, nullable=False)
	recommendation = Column(Text, nullable=True)
	
	# Scoring and Priority
	confidence_score = Column(DECIMAL(5, 4), nullable=False)
	priority_score = Column(Integer, nullable=False, default=1)  # 1-10
	impact_assessment = Column(String(20), nullable=True)  # Low, Medium, High, Critical
	
	# Timeline and Actions
	suggested_action_date = Column(Date, nullable=True)
	expiry_date = Column(Date, nullable=True)
	action_taken = Column(Boolean, default=False)
	action_notes = Column(Text, nullable=True)
	
	# Supporting Data
	supporting_data = Column(JSONB, nullable=True)
	related_metrics = Column(JSONB, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True, index=True)
	is_dismissed = Column(Boolean, default=False)
	dismissed_by = Column(String(36), nullable=True)
	dismissed_date = Column(DateTime, nullable=True)
	
	# Relationships
	ai_profile = relationship("HREmployeeAIProfile", back_populates="ai_insights")
	
	def __repr__(self):
		return f"<HREmployeeAIInsight {self.insight_type} - {self.title[:30]}>"


class HROrganizationalVisualization(Model, AuditMixin, BaseMixin):
	"""
	3D/AR organizational visualization data and configurations.
	
	Revolutionary feature #2: Immersive Employee Experience Platform
	"""
	__tablename__ = 'hr_edm_org_visualization'
	
	# Identity
	visualization_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Visualization Configuration
	visualization_name = Column(String(200), nullable=False)
	visualization_type = Column(String(50), nullable=False)  # 3d_org_chart, ar_directory, vr_workspace
	description = Column(Text, nullable=True)
	
	# 3D Layout Data
	layout_data = Column(JSONB, nullable=False)  # Node positions, connections, animations
	visual_theme = Column(String(50), default='modern')
	color_scheme = Column(JSONB, nullable=True)
	
	# AR/VR Settings
	ar_markers = Column(JSONB, nullable=True)
	vr_environment = Column(String(50), nullable=True)
	interaction_config = Column(JSONB, nullable=True)
	
	# Permissions and Sharing
	is_public = Column(Boolean, default=False)
	shared_with_departments = Column(ARRAY(String), nullable=True)
	created_by = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False)
	
	# Usage Analytics
	view_count = Column(Integer, default=0)
	last_accessed = Column(DateTime, nullable=True)
	average_interaction_time = Column(Integer, nullable=True)  # seconds
	
	# Configuration
	is_active = Column(Boolean, default=True, index=True)
	
	# Relationships
	creator = relationship("HREmployee", foreign_keys=[created_by])
	
	def __repr__(self):
		return f"<HROrganizationalVisualization {self.visualization_name}>"


class HRConversationalSession(Model, AuditMixin, BaseMixin):
	"""
	Conversational AI session tracking for natural language HR interactions.
	
	Revolutionary feature #3: Conversational HR Assistant
	"""
	__tablename__ = 'hr_edm_conversation_session'
	
	# Identity
	session_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, index=True)
	
	# Session Details
	session_start = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	session_end = Column(DateTime, nullable=True)
	session_duration_seconds = Column(Integer, nullable=True)
	
	# Interaction Mode
	interaction_mode = Column(String(20), nullable=False)  # text, voice, hybrid
	language_code = Column(String(10), default='en-US')
	device_type = Column(String(50), nullable=True)  # mobile, desktop, tablet
	
	# Conversation Analytics
	total_messages = Column(Integer, default=0)
	user_satisfaction_score = Column(Integer, nullable=True)  # 1-5
	resolution_achieved = Column(Boolean, nullable=True)
	escalated_to_human = Column(Boolean, default=False)
	
	# AI Performance Metrics
	average_response_time_ms = Column(Integer, nullable=True)
	ai_confidence_average = Column(DECIMAL(5, 4), nullable=True)
	successful_task_completion = Column(Boolean, nullable=True)
	
	# Context and State
	conversation_context = Column(JSONB, nullable=True)
	session_metadata = Column(JSONB, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True, index=True)
	
	# Relationships
	employee = relationship("HREmployee", foreign_keys=[employee_id])
	messages = relationship("HRConversationalMessage", back_populates="session")
	
	def __repr__(self):
		return f"<HRConversationalSession {self.session_id}>"


class HRConversationalMessage(Model, AuditMixin, BaseMixin):
	"""
	Individual messages within conversational AI sessions.
	
	Revolutionary feature #3: Conversational HR Assistant
	"""
	__tablename__ = 'hr_edm_conversation_message'
	
	# Identity
	message_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	session_id = Column(String(36), ForeignKey('hr_edm_conversation_session.session_id'), nullable=False, index=True)
	
	# Message Details
	message_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	message_type = Column(String(20), nullable=False)  # user_text, user_voice, ai_response, system
	message_content = Column(Text, nullable=False)
	
	# Voice Message Data
	audio_file_path = Column(String(500), nullable=True)
	transcription_confidence = Column(DECIMAL(5, 4), nullable=True)
	audio_duration_seconds = Column(Integer, nullable=True)
	
	# AI Processing
	intent_detected = Column(String(100), nullable=True)
	entities_extracted = Column(JSONB, nullable=True)
	ai_confidence_score = Column(DECIMAL(5, 4), nullable=True)
	processing_time_ms = Column(Integer, nullable=True)
	
	# Response Generation
	response_generated = Column(Boolean, default=False)
	response_type = Column(String(50), nullable=True)  # text, voice, action, data
	actions_triggered = Column(JSONB, nullable=True)
	
	# Quality and Feedback
	user_feedback = Column(Integer, nullable=True)  # 1-5 rating
	feedback_notes = Column(Text, nullable=True)
	flagged_for_review = Column(Boolean, default=False)
	
	# Relationships
	session = relationship("HRConversationalSession", back_populates="messages")
	
	def __repr__(self):
		return f"<HRConversationalMessage {self.message_type} - {self.message_content[:30]}>"


class HRPredictiveAnalyticsModel(Model, AuditMixin, BaseMixin):
	"""
	ML model definitions and performance tracking for predictive analytics.
	
	Revolutionary feature #5: Predictive People Analytics
	"""
	__tablename__ = 'hr_edm_predictive_model'
	
	# Identity
	model_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Model Definition
	model_name = Column(String(200), nullable=False, index=True)
	model_type = Column(String(50), nullable=False)  # retention_prediction, performance_forecast, skill_recommendation
	model_version = Column(String(20), nullable=False)
	description = Column(Text, nullable=True)
	
	# Model Configuration
	algorithm = Column(String(100), nullable=False)  # random_forest, neural_network, xgboost
	hyperparameters = Column(JSONB, nullable=True)
	feature_columns = Column(JSONB, nullable=False)
	target_column = Column(String(100), nullable=False)
	
	# Training Data
	training_data_source = Column(String(200), nullable=True)
	training_date = Column(DateTime, nullable=True)
	training_records_count = Column(Integer, nullable=True)
	validation_split = Column(DECIMAL(3, 2), default=0.2)
	
	# Performance Metrics
	accuracy_score = Column(DECIMAL(5, 4), nullable=True)
	precision_score = Column(DECIMAL(5, 4), nullable=True)
	recall_score = Column(DECIMAL(5, 4), nullable=True)
	f1_score = Column(DECIMAL(5, 4), nullable=True)
	auc_score = Column(DECIMAL(5, 4), nullable=True)
	
	# Model Artifacts
	model_file_path = Column(String(500), nullable=True)
	feature_importance = Column(JSONB, nullable=True)
	model_metrics = Column(JSONB, nullable=True)
	
	# Deployment Status
	is_active = Column(Boolean, default=False, index=True)
	is_production = Column(Boolean, default=False)
	deployment_date = Column(DateTime, nullable=True)
	last_prediction_date = Column(DateTime, nullable=True)
	
	# Usage Statistics
	total_predictions = Column(Integer, default=0)
	average_prediction_time_ms = Column(Integer, nullable=True)
	
	def __repr__(self):
		return f"<HRPredictiveAnalyticsModel {self.model_name} v{self.model_version}>"


class HRGlobalComplianceRule(Model, AuditMixin, BaseMixin):
	"""
	Global workforce compliance rules and automated monitoring.
	
	Revolutionary feature #10: Global Workforce Management
	"""
	__tablename__ = 'hr_edm_global_compliance'
	
	# Identity
	rule_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Rule Definition
	rule_name = Column(String(200), nullable=False, index=True)
	rule_category = Column(String(100), nullable=False)  # labor_law, tax_compliance, data_privacy, safety
	jurisdiction = Column(String(10), nullable=False, index=True)  # Country code (US, UK, DE, etc.)
	region = Column(String(100), nullable=True)  # State, Province, Region
	
	# Rule Logic
	rule_description = Column(Text, nullable=False)
	validation_logic = Column(JSONB, nullable=False)  # JSON-based rule engine
	severity_level = Column(String(20), default='medium')  # low, medium, high, critical
	
	# Automation Settings
	auto_check_enabled = Column(Boolean, default=True)
	auto_fix_enabled = Column(Boolean, default=False)
	notification_required = Column(Boolean, default=True)
	escalation_required = Column(Boolean, default=False)
	
	# Effective Dates
	effective_from = Column(Date, nullable=False, index=True)
	effective_to = Column(Date, nullable=True, index=True)
	
	# Compliance Tracking
	last_check_date = Column(DateTime, nullable=True)
	violations_count = Column(Integer, default=0)
	last_violation_date = Column(DateTime, nullable=True)
	
	# Documentation
	regulation_reference = Column(String(500), nullable=True)
	documentation_url = Column(String(500), nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True, index=True)
	
	# Relationships
	compliance_violations = relationship("HRComplianceViolation", back_populates="compliance_rule")
	
	def __repr__(self):
		return f"<HRGlobalComplianceRule {self.rule_name} - {self.jurisdiction}>"


class HRComplianceViolation(Model, AuditMixin, BaseMixin):
	"""
	Compliance violation tracking and resolution management.
	
	Revolutionary feature #10: Global Workforce Management
	"""
	__tablename__ = 'hr_edm_compliance_violation'
	
	# Identity
	violation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	rule_id = Column(String(36), ForeignKey('hr_edm_global_compliance.rule_id'), nullable=False, index=True)
	
	# Violation Details
	violation_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	severity_level = Column(String(20), nullable=False, index=True)
	violation_description = Column(Text, nullable=False)
	
	# Affected Entity
	entity_type = Column(String(50), nullable=False)  # employee, department, policy
	entity_id = Column(String(36), nullable=False, index=True)
	entity_name = Column(String(200), nullable=True)
	
	# Detection Details
	detection_method = Column(String(50), nullable=False)  # automated, manual, audit
	detected_by_system = Column(Boolean, default=True)
	detected_by_user = Column(String(36), nullable=True)
	
	# Resolution Tracking
	status = Column(String(20), default='open', index=True)  # open, investigating, resolved, dismissed
	assigned_to = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=True)
	resolution_notes = Column(Text, nullable=True)
	resolution_date = Column(DateTime, nullable=True)
	
	# Impact Assessment
	risk_level = Column(String(20), nullable=False)
	potential_penalties = Column(Text, nullable=True)
	business_impact = Column(Text, nullable=True)
	
	# Remediation
	remediation_plan = Column(Text, nullable=True)
	remediation_deadline = Column(Date, nullable=True)
	remediation_cost = Column(DECIMAL(12, 2), nullable=True)
	
	# Follow-up
	requires_reporting = Column(Boolean, default=False)
	reported_to_authority = Column(Boolean, default=False)
	report_date = Column(DateTime, nullable=True)
	
	# Relationships
	compliance_rule = relationship("HRGlobalComplianceRule", back_populates="compliance_violations")
	assigned_employee = relationship("HREmployee", foreign_keys=[assigned_to])
	
	def __repr__(self):
		return f"<HRComplianceViolation {self.violation_id} - {self.severity_level}>"


class HRWorkflowAutomation(Model, AuditMixin, BaseMixin):
	"""
	Intelligent workflow automation configurations and execution tracking.
	
	Revolutionary feature #8: Privacy-First Data Architecture
	"""
	__tablename__ = 'hr_edm_workflow_automation'
	
	# Identity
	workflow_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Workflow Definition
	workflow_name = Column(String(200), nullable=False, index=True)
	workflow_type = Column(String(50), nullable=False)  # onboarding, offboarding, promotion, training
	description = Column(Text, nullable=True)
	
	# Workflow Configuration
	trigger_conditions = Column(JSONB, nullable=False)
	workflow_steps = Column(JSONB, nullable=False)
	approval_chain = Column(JSONB, nullable=True)
	notification_settings = Column(JSONB, nullable=True)
	
	# Automation Settings
	is_automated = Column(Boolean, default=True)
	auto_approve_conditions = Column(JSONB, nullable=True)
	escalation_rules = Column(JSONB, nullable=True)
	timeout_settings = Column(JSONB, nullable=True)
	
	# AI Enhancement
	ai_optimization_enabled = Column(Boolean, default=False)
	ai_recommendation_engine = Column(String(100), nullable=True)
	learning_mode = Column(Boolean, default=False)
	
	# Performance Metrics
	total_executions = Column(Integer, default=0)
	successful_executions = Column(Integer, default=0)
	average_completion_time_hours = Column(DECIMAL(8, 2), nullable=True)
	user_satisfaction_score = Column(DECIMAL(3, 2), nullable=True)
	
	# Version Control
	version = Column(String(20), default='1.0.0')
	is_active = Column(Boolean, default=True, index=True)
	created_by = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False)
	
	# Relationships
	creator = relationship("HREmployee", foreign_keys=[created_by])
	executions = relationship("HRWorkflowExecution", back_populates="workflow")
	
	def __repr__(self):
		return f"<HRWorkflowAutomation {self.workflow_name}>"


class HRWorkflowExecution(Model, AuditMixin, BaseMixin):
	"""
	Individual workflow execution instances and tracking.
	
	Revolutionary feature #8: Privacy-First Data Architecture
	"""
	__tablename__ = 'hr_edm_workflow_execution'
	
	# Identity
	execution_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	workflow_id = Column(String(36), ForeignKey('hr_edm_workflow_automation.workflow_id'), nullable=False, index=True)
	
	# Execution Details
	execution_start = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	execution_end = Column(DateTime, nullable=True)
	execution_duration_hours = Column(DECIMAL(8, 2), nullable=True)
	
	# Trigger Information
	triggered_by = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=True)
	trigger_event = Column(String(100), nullable=False)
	trigger_data = Column(JSONB, nullable=True)
	
	# Subject Information
	subject_type = Column(String(50), nullable=False)  # employee, department, position
	subject_id = Column(String(36), nullable=False, index=True)
	subject_name = Column(String(200), nullable=True)
	
	# Execution Status
	status = Column(String(20), default='running', index=True)  # running, completed, failed, cancelled
	current_step = Column(String(100), nullable=True)
	steps_completed = Column(Integer, default=0)
	total_steps = Column(Integer, nullable=True)
	
	# Progress Tracking
	step_history = Column(JSONB, nullable=True)
	approval_history = Column(JSONB, nullable=True)
	notification_history = Column(JSONB, nullable=True)
	
	# Results and Metrics
	completion_percentage = Column(Integer, default=0)
	user_satisfaction_rating = Column(Integer, nullable=True)  # 1-5
	outcome_summary = Column(Text, nullable=True)
	
	# Error Handling
	error_count = Column(Integer, default=0)
	last_error = Column(Text, nullable=True)
	retry_count = Column(Integer, default=0)
	
	# Relationships
	workflow = relationship("HRWorkflowAutomation", back_populates="executions")
	triggered_by_employee = relationship("HREmployee", foreign_keys=[triggered_by])
	
	def __repr__(self):
		return f"<HRWorkflowExecution {self.execution_id} - {self.status}>"


class HRAnalyticsDashboard(Model, AuditMixin, BaseMixin):
	"""
	Dynamic analytics dashboard configurations with AI-powered insights.
	
	Revolutionary feature #4: Real-Time Collaborative Workspaces
	"""
	__tablename__ = 'hr_edm_analytics_dashboard'
	
	# Identity
	dashboard_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Dashboard Configuration
	dashboard_name = Column(String(200), nullable=False, index=True)
	dashboard_type = Column(String(50), nullable=False)  # executive, hr_manager, employee_self_service
	description = Column(Text, nullable=True)
	
	# Layout and Widgets
	layout_config = Column(JSONB, nullable=False)
	widget_config = Column(JSONB, nullable=False)
	filter_config = Column(JSONB, nullable=True)
	
	# AI Enhancement
	ai_insights_enabled = Column(Boolean, default=True)
	auto_refresh_enabled = Column(Boolean, default=True)
	refresh_interval_minutes = Column(Integer, default=15)
	predictive_alerts_enabled = Column(Boolean, default=False)
	
	# Sharing and Permissions
	is_public = Column(Boolean, default=False)
	shared_with_roles = Column(ARRAY(String), nullable=True)
	shared_with_users = Column(ARRAY(String), nullable=True)
	owner_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False)
	
	# Usage Analytics
	view_count = Column(Integer, default=0)
	last_viewed = Column(DateTime, nullable=True)
	average_session_duration_minutes = Column(Integer, nullable=True)
	
	# Personalization
	user_customizations = Column(JSONB, nullable=True)
	auto_personalization_enabled = Column(Boolean, default=True)
	
	# Status
	is_active = Column(Boolean, default=True, index=True)
	
	# Relationships
	owner = relationship("HREmployee", foreign_keys=[owner_id])
	
	def __repr__(self):
		return f"<HRAnalyticsDashboard {self.dashboard_name}>"


# ============================================================================
# PYDANTIC V2 MODELS FOR API VALIDATION
# ============================================================================

class EmployeeProfilePydantic(BaseModel):
	"""Pydantic model for employee profile validation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	employee_id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employee_number: str
	first_name: str
	last_name: str
	work_email: str | None = None
	department_id: str
	position_id: str
	hire_date: date
	employment_status: str = "Active"
	is_active: bool = True


class AIInsightPydantic(BaseModel):
	"""Pydantic model for AI insight validation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	insight_id: str = Field(default_factory=uuid7str)
	tenant_id: str
	insight_type: AIInsightType
	title: str
	description: str
	confidence_score: Annotated[float, AfterValidator(lambda v: 0.0 <= v <= 1.0)]
	priority_score: Annotated[int, AfterValidator(lambda v: 1 <= v <= 10)]
	is_active: bool = True


class ConversationSessionPydantic(BaseModel):
	"""Pydantic model for conversational session validation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	session_id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employee_id: str
	interaction_mode: str
	language_code: str = "en-US"
	device_type: str | None = None
	is_active: bool = True


class WorkflowAutomationPydantic(BaseModel):
	"""Pydantic model for workflow automation validation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	workflow_id: str = Field(default_factory=uuid7str)
	tenant_id: str
	workflow_name: str
	workflow_type: str
	trigger_conditions: dict[str, Any]
	workflow_steps: list[dict[str, Any]]
	is_automated: bool = True
	is_active: bool = True


# ============================================================================
# ASYNC HELPER FUNCTIONS
# ============================================================================

async def _log_model_operation(operation: str, model_name: str, record_id: str) -> None:
	"""Log model operations for audit trails."""
	print(f"[MODEL_LOG] {operation}: {model_name} - {record_id} at {datetime.utcnow()}")


async def _validate_tenant_access(tenant_id: str, user_id: str) -> bool:
	"""Validate tenant access for multi-tenant security."""
	# Implementation would check user's tenant permissions
	return True


async def _generate_ai_embeddings(text_content: str) -> list[float] | None:
	"""Generate AI embeddings for semantic search."""
	# Implementation would call APG AI orchestration service
	return None