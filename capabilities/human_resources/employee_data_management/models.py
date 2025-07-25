"""
Employee Data Management Models

Database models for employee information, organizational structure, skills,
certifications, and employment history management.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Enum
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

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