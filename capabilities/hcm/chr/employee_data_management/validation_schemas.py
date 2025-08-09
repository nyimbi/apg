"""
APG Employee Data Management - Advanced Validation Schemas

Comprehensive validation schemas using Pydantic v2 with AI-enhanced
validation rules and intelligent data transformation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import re
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union, Annotated
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, AfterValidator
from annotated_types import MinLen, MaxLen, Ge, Le
from uuid_extensions import uuid7str


# ============================================================================
# CUSTOM VALIDATORS AND TRANSFORMERS
# ============================================================================

def validate_email(email: str) -> str:
	"""Validate and normalize email address."""
	if not email:
		raise ValueError("Email cannot be empty")
	
	email = email.strip().lower()
	email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
	
	if not re.match(email_pattern, email):
		raise ValueError(f"Invalid email format: {email}")
	
	return email

def validate_phone_number(phone: str) -> str:
	"""Validate and normalize phone number."""
	if not phone:
		return phone
	
	# Remove all non-digit characters except +
	cleaned = re.sub(r'[^\d+]', '', phone.strip())
	
	# Basic validation - should have at least 10 digits
	if len(re.sub(r'[^\d]', '', cleaned)) < 10:
		raise ValueError(f"Phone number too short: {phone}")
	
	return cleaned

def validate_employee_number(emp_num: str) -> str:
	"""Validate employee number format."""
	if not emp_num:
		raise ValueError("Employee number cannot be empty")
	
	emp_num = emp_num.strip().upper()
	
	# Standard format: EMP followed by 6 digits
	if not re.match(r'^EMP\d{6}$', emp_num):
		raise ValueError(f"Invalid employee number format: {emp_num}. Expected: EMP######")
	
	return emp_num

def validate_salary(salary: Union[str, int, float, Decimal]) -> Decimal:
	"""Validate and convert salary to Decimal."""
	if salary is None:
		return None
	
	try:
		salary_decimal = Decimal(str(salary))
		
		if salary_decimal < 0:
			raise ValueError("Salary cannot be negative")
		
		if salary_decimal > Decimal('10000000'):  # 10M cap
			raise ValueError("Salary exceeds maximum allowed amount")
		
		return salary_decimal.quantize(Decimal('0.01'))  # Round to 2 decimal places
		
	except (ValueError, TypeError) as e:
		raise ValueError(f"Invalid salary format: {salary}")

def validate_date_not_future(date_value: date) -> date:
	"""Validate that date is not in the future."""
	if date_value and date_value > date.today():
		raise ValueError(f"Date cannot be in the future: {date_value}")
	return date_value

def validate_reasonable_age(birth_date: date) -> date:
	"""Validate reasonable age range."""
	if not birth_date:
		return birth_date
	
	today = date.today()
	age = (today - birth_date).days / 365.25
	
	if age < 16:
		raise ValueError(f"Employee too young: {age:.1f} years")
	
	if age > 100:
		raise ValueError(f"Employee age unrealistic: {age:.1f} years")
	
	return birth_date

def normalize_name(name: str) -> str:
	"""Normalize and validate name field."""
	if not name:
		raise ValueError("Name cannot be empty")
	
	# Remove extra whitespace and title case
	normalized = ' '.join(name.strip().split()).title()
	
	# Basic validation - only letters, spaces, hyphens, apostrophes
	if not re.match(r'^[a-zA-Z\s\-\'\.]+$', normalized):
		raise ValueError(f"Invalid characters in name: {name}")
	
	if len(normalized) > 100:
		raise ValueError(f"Name too long: {name}")
	
	return normalized

def validate_postal_code(postal_code: str, country: str = None) -> str:
	"""Validate postal code based on country."""
	if not postal_code:
		return postal_code
	
	postal_code = postal_code.strip().upper()
	
	# Country-specific patterns
	patterns = {
		'USA': r'^\d{5}(-\d{4})?$',
		'US': r'^\d{5}(-\d{4})?$',
		'CANADA': r'^[A-Z]\d[A-Z] \d[A-Z]\d$',
		'CA': r'^[A-Z]\d[A-Z] \d[A-Z]\d$',
		'UK': r'^[A-Z]{1,2}\d{1,2}[A-Z]? \d[A-Z]{2}$',
		'GERMANY': r'^\d{5}$',
		'DE': r'^\d{5}$'
	}
	
	if country and country.upper() in patterns:
		pattern = patterns[country.upper()]
		if not re.match(pattern, postal_code):
			raise ValueError(f"Invalid postal code for {country}: {postal_code}")
	
	return postal_code


# ============================================================================
# EMPLOYMENT STATUS AND TYPE ENUMS
# ============================================================================

class EmploymentStatus(str, Enum):
	"""Valid employment statuses."""
	ACTIVE = "Active"
	INACTIVE = "Inactive"
	TERMINATED = "Terminated"
	ON_LEAVE = "On Leave"
	SUSPENDED = "Suspended"

class EmploymentType(str, Enum):
	"""Valid employment types."""
	FULL_TIME = "Full-Time"
	PART_TIME = "Part-Time"
	CONTRACT = "Contract"
	INTERN = "Intern"
	CONSULTANT = "Consultant"
	TEMPORARY = "Temporary"

class WorkLocation(str, Enum):
	"""Valid work locations."""
	OFFICE = "Office"
	REMOTE = "Remote"
	HYBRID = "Hybrid"
	FIELD = "Field"

class PayFrequency(str, Enum):
	"""Valid pay frequencies."""
	WEEKLY = "Weekly"
	BIWEEKLY = "Bi-weekly"
	SEMIMONTHLY = "Semi-monthly"
	MONTHLY = "Monthly"
	QUARTERLY = "Quarterly"
	ANNUALLY = "Annually"

class Gender(str, Enum):
	"""Valid gender options."""
	MALE = "Male"
	FEMALE = "Female"
	NON_BINARY = "Non-binary"
	PREFER_NOT_TO_SAY = "Prefer not to say"
	OTHER = "Other"

class MaritalStatus(str, Enum):
	"""Valid marital status options."""
	SINGLE = "Single"
	MARRIED = "Married"
	DIVORCED = "Divorced"
	WIDOWED = "Widowed"
	SEPARATED = "Separated"
	DOMESTIC_PARTNERSHIP = "Domestic Partnership"


# ============================================================================
# CORE VALIDATION SCHEMAS
# ============================================================================

class EmployeePersonalInfoSchema(BaseModel):
	"""Validation schema for personal information."""
	model_config = ConfigDict(
		extra='forbid',
		validate_assignment=True,
		str_strip_whitespace=True
	)
	
	first_name: Annotated[str, MinLen(1), MaxLen(100), AfterValidator(normalize_name)]
	middle_name: Optional[Annotated[str, MaxLen(100), AfterValidator(normalize_name)]] = None
	last_name: Annotated[str, MinLen(1), MaxLen(100), AfterValidator(normalize_name)]
	preferred_name: Optional[Annotated[str, MaxLen(100), AfterValidator(normalize_name)]] = None
	
	date_of_birth: Optional[Annotated[date, AfterValidator(validate_reasonable_age)]] = None
	gender: Optional[Gender] = None
	marital_status: Optional[MaritalStatus] = None
	nationality: Optional[Annotated[str, MaxLen(100)]] = None
	
	@field_validator('first_name', 'last_name')
	@classmethod
	def validate_required_names(cls, v):
		if not v or not v.strip():
			raise ValueError("Name cannot be empty")
		return v
	
	@model_validator(mode='after')
	def validate_name_consistency(self):
		"""Ensure name fields are consistent."""
		if self.preferred_name and len(self.preferred_name) > 50:
			raise ValueError("Preferred name too long")
		return self


class EmployeeContactInfoSchema(BaseModel):
	"""Validation schema for contact information."""
	model_config = ConfigDict(
		extra='forbid',
		validate_assignment=True,
		str_strip_whitespace=True
	)
	
	personal_email: Optional[Annotated[str, AfterValidator(validate_email)]] = None
	work_email: Annotated[str, AfterValidator(validate_email)]
	
	phone_mobile: Optional[Annotated[str, AfterValidator(validate_phone_number)]] = None
	phone_home: Optional[Annotated[str, AfterValidator(validate_phone_number)]] = None
	phone_work: Optional[Annotated[str, AfterValidator(validate_phone_number)]] = None
	
	address_line1: Optional[Annotated[str, MaxLen(200)]] = None
	address_line2: Optional[Annotated[str, MaxLen(200)]] = None
	city: Optional[Annotated[str, MaxLen(100)]] = None
	state_province: Optional[Annotated[str, MaxLen(100)]] = None
	postal_code: Optional[str] = None
	country: Optional[Annotated[str, MaxLen(100)]] = None
	
	@field_validator('postal_code')
	@classmethod
	def validate_postal_code_format(cls, v, info):
		if v and 'country' in info.data:
			return validate_postal_code(v, info.data['country'])
		return v
	
	@model_validator(mode='after')
	def validate_address_completeness(self):
		"""Validate address completeness."""
		address_fields = [self.address_line1, self.city, self.state_province, self.country]
		filled_fields = [f for f in address_fields if f]
		
		# If any address field is filled, require core address fields
		if filled_fields and not all([self.address_line1, self.city, self.country]):
			raise ValueError("If providing address, please include at least address line 1, city, and country")
		
		return self


class EmployeeEmploymentInfoSchema(BaseModel):
	"""Validation schema for employment information."""
	model_config = ConfigDict(
		extra='forbid',
		validate_assignment=True,
		str_strip_whitespace=True
	)
	
	employee_number: Annotated[str, AfterValidator(validate_employee_number)]
	badge_id: Optional[Annotated[str, MaxLen(20)]] = None
	
	department_id: Annotated[str, MinLen(1)]
	position_id: Annotated[str, MinLen(1)]
	manager_id: Optional[str] = None
	
	hire_date: Annotated[date, AfterValidator(validate_date_not_future)]
	start_date: Optional[Annotated[date, AfterValidator(validate_date_not_future)]] = None
	termination_date: Optional[date] = None
	rehire_date: Optional[Annotated[date, AfterValidator(validate_date_not_future)]] = None
	
	employment_status: EmploymentStatus = EmploymentStatus.ACTIVE
	employment_type: EmploymentType = EmploymentType.FULL_TIME
	work_location: WorkLocation = WorkLocation.OFFICE
	
	@model_validator(mode='after')
	def validate_employment_dates(self):
		"""Validate employment date logic."""
		if self.start_date and self.hire_date:
			if self.start_date < self.hire_date:
				raise ValueError("Start date cannot be before hire date")
		
		if self.termination_date and self.hire_date:
			if self.termination_date < self.hire_date:
				raise ValueError("Termination date cannot be before hire date")
		
		if self.rehire_date and self.termination_date:
			if self.rehire_date < self.termination_date:
				raise ValueError("Rehire date cannot be before termination date")
		
		# Employment status consistency
		if self.employment_status == EmploymentStatus.TERMINATED and not self.termination_date:
			raise ValueError("Terminated employees must have a termination date")
		
		if self.employment_status == EmploymentStatus.ACTIVE and self.termination_date:
			if self.termination_date <= date.today():
				raise ValueError("Active employees cannot have a past termination date")
		
		return self
	
	@field_validator('manager_id')
	@classmethod
	def validate_manager_not_self(cls, v, info):
		"""Ensure employee is not their own manager."""
		if v and 'employee_id' in info.data and v == info.data.get('employee_id'):
			raise ValueError("Employee cannot be their own manager")
		return v


class EmployeeCompensationSchema(BaseModel):
	"""Validation schema for compensation information."""
	model_config = ConfigDict(
		extra='forbid',
		validate_assignment=True,
		str_strip_whitespace=True
	)
	
	base_salary: Optional[Annotated[Decimal, AfterValidator(validate_salary)]] = None
	hourly_rate: Optional[Annotated[Decimal, Ge(Decimal('0')), Le(Decimal('1000'))]] = None
	currency_code: Annotated[str, Field(default='USD', pattern=r'^[A-Z]{3}$')] = 'USD'
	pay_frequency: PayFrequency = PayFrequency.MONTHLY
	
	benefits_eligible: bool = True
	benefits_start_date: Optional[date] = None
	
	@model_validator(mode='after')
	def validate_compensation_logic(self):
		"""Validate compensation business rules."""
		if not self.base_salary and not self.hourly_rate:
			raise ValueError("Either base salary or hourly rate must be provided")
		
		if self.base_salary and self.hourly_rate:
			raise ValueError("Cannot have both base salary and hourly rate")
		
		if self.benefits_start_date and not self.benefits_eligible:
			raise ValueError("Benefits start date provided but employee not eligible for benefits")
		
		return self


class EmployeeTaxInfoSchema(BaseModel):
	"""Validation schema for tax information."""
	model_config = ConfigDict(
		extra='forbid',
		validate_assignment=True,
		str_strip_whitespace=True
	)
	
	tax_id: Optional[Annotated[str, MaxLen(50)]] = None
	tax_country: Annotated[str, Field(default='USA', max_length=3)] = 'USA'
	tax_state: Optional[Annotated[str, MaxLen(50)]] = None
	
	@field_validator('tax_id')
	@classmethod
	def validate_tax_id_format(cls, v, info):
		"""Validate tax ID format based on country."""
		if not v:
			return v
		
		tax_country = info.data.get('tax_country', 'USA')
		
		# US SSN format
		if tax_country in ['USA', 'US']:
			# Remove hyphens and validate format
			clean_ssn = re.sub(r'[^\d]', '', v)
			if len(clean_ssn) != 9:
				raise ValueError("US Tax ID (SSN) must be 9 digits")
			return f"{clean_ssn[:3]}-{clean_ssn[3:5]}-{clean_ssn[5:]}"
		
		return v


class EmployeePerformanceSchema(BaseModel):
	"""Validation schema for performance information."""
	model_config = ConfigDict(
		extra='forbid',
		validate_assignment=True
	)
	
	probation_end_date: Optional[date] = None
	next_review_date: Optional[date] = None
	performance_rating: Optional[Annotated[str, MaxLen(20)]] = None
	
	@field_validator('performance_rating')
	@classmethod
	def validate_performance_rating(cls, v):
		"""Validate performance rating values."""
		if v:
			valid_ratings = [
				'Exceptional', 'Exceeds Expectations', 'Meets Expectations',
				'Below Expectations', 'Unsatisfactory', '5', '4', '3', '2', '1'
			]
			if v not in valid_ratings:
				raise ValueError(f"Invalid performance rating: {v}")
		return v


# ============================================================================
# COMPREHENSIVE EMPLOYEE VALIDATION SCHEMA
# ============================================================================

class ComprehensiveEmployeeSchema(BaseModel):
	"""Complete employee validation schema combining all components."""
	model_config = ConfigDict(
		extra='forbid',
		validate_assignment=True,
		str_strip_whitespace=True
	)
	
	# Core identification
	tenant_id: Annotated[str, MinLen(1)]
	employee_id: Optional[str] = Field(default_factory=uuid7str)
	
	# Personal information
	first_name: Annotated[str, MinLen(1), MaxLen(100), AfterValidator(normalize_name)]
	middle_name: Optional[Annotated[str, MaxLen(100), AfterValidator(normalize_name)]] = None
	last_name: Annotated[str, MinLen(1), MaxLen(100), AfterValidator(normalize_name)]
	preferred_name: Optional[Annotated[str, MaxLen(100), AfterValidator(normalize_name)]] = None
	full_name: Optional[Annotated[str, MaxLen(300)]] = None
	
	# Demographics
	date_of_birth: Optional[Annotated[date, AfterValidator(validate_reasonable_age)]] = None
	gender: Optional[Gender] = None
	marital_status: Optional[MaritalStatus] = None
	nationality: Optional[Annotated[str, MaxLen(100)]] = None
	
	# Contact information
	personal_email: Optional[Annotated[str, AfterValidator(validate_email)]] = None
	work_email: Annotated[str, AfterValidator(validate_email)]
	phone_mobile: Optional[Annotated[str, AfterValidator(validate_phone_number)]] = None
	phone_home: Optional[Annotated[str, AfterValidator(validate_phone_number)]] = None
	phone_work: Optional[Annotated[str, AfterValidator(validate_phone_number)]] = None
	
	# Address
	address_line1: Optional[Annotated[str, MaxLen(200)]] = None
	address_line2: Optional[Annotated[str, MaxLen(200)]] = None
	city: Optional[Annotated[str, MaxLen(100)]] = None
	state_province: Optional[Annotated[str, MaxLen(100)]] = None
	postal_code: Optional[str] = None
	country: Optional[Annotated[str, MaxLen(100)]] = None
	
	# Employment information
	employee_number: Annotated[str, AfterValidator(validate_employee_number)]
	badge_id: Optional[Annotated[str, MaxLen(20)]] = None
	department_id: Annotated[str, MinLen(1)]
	position_id: Annotated[str, MinLen(1)]
	manager_id: Optional[str] = None
	
	# Employment dates
	hire_date: Annotated[date, AfterValidator(validate_date_not_future)]
	start_date: Optional[Annotated[date, AfterValidator(validate_date_not_future)]] = None
	termination_date: Optional[date] = None
	rehire_date: Optional[Annotated[date, AfterValidator(validate_date_not_future)]] = None
	
	# Employment status
	employment_status: EmploymentStatus = EmploymentStatus.ACTIVE
	employment_type: EmploymentType = EmploymentType.FULL_TIME
	work_location: WorkLocation = WorkLocation.OFFICE
	
	# Compensation
	base_salary: Optional[Annotated[Decimal, AfterValidator(validate_salary)]] = None
	hourly_rate: Optional[Annotated[Decimal, Ge(Decimal('0')), Le(Decimal('1000'))]] = None
	currency_code: Annotated[str, Field(default='USD', pattern=r'^[A-Z]{3}$')] = 'USD'
	pay_frequency: PayFrequency = PayFrequency.MONTHLY
	
	# Benefits and tax
	benefits_eligible: bool = True
	benefits_start_date: Optional[date] = None
	tax_id: Optional[Annotated[str, MaxLen(50)]] = None
	tax_country: Annotated[str, Field(default='USA', max_length=3)] = 'USA'
	tax_state: Optional[Annotated[str, MaxLen(50)]] = None
	
	# Performance
	probation_end_date: Optional[date] = None
	next_review_date: Optional[date] = None
	performance_rating: Optional[Annotated[str, MaxLen(20)]] = None
	
	# System fields
	is_active: bool = True
	is_system_user: bool = False
	system_user_id: Optional[str] = None
	photo_url: Optional[Annotated[str, MaxLen(500)]] = None
	documents_folder: Optional[Annotated[str, MaxLen(500)]] = None
	
	@model_validator(mode='after')
	def validate_comprehensive_employee(self):
		"""Comprehensive validation across all employee data."""
		
		# Auto-generate full name if not provided
		if not self.full_name:
			middle_part = f" {self.middle_name}" if self.middle_name else ""
			self.full_name = f"{self.first_name}{middle_part} {self.last_name}"
		
		# Validate employment date logic
		if self.start_date and self.hire_date:
			if self.start_date < self.hire_date:
				raise ValueError("Start date cannot be before hire date")
		
		if self.termination_date and self.hire_date:
			if self.termination_date < self.hire_date:
				raise ValueError("Termination date cannot be before hire date")
		
		# Employment status consistency
		if self.employment_status == EmploymentStatus.TERMINATED and not self.termination_date:
			raise ValueError("Terminated employees must have a termination date")
		
		if self.employment_status == EmploymentStatus.ACTIVE and self.termination_date:
			if self.termination_date <= date.today():
				raise ValueError("Active employees cannot have a past termination date")
		
		# Compensation validation
		if not self.base_salary and not self.hourly_rate:
			raise ValueError("Either base salary or hourly rate must be provided")
		
		if self.base_salary and self.hourly_rate:
			raise ValueError("Cannot have both base salary and hourly rate")
		
		# Benefits logic
		if self.benefits_start_date and not self.benefits_eligible:
			raise ValueError("Benefits start date provided but employee not eligible for benefits")
		
		# Manager validation
		if self.manager_id and self.manager_id == self.employee_id:
			raise ValueError("Employee cannot be their own manager")
		
		# Address completeness
		address_fields = [self.address_line1, self.city, self.state_province, self.country]
		filled_fields = [f for f in address_fields if f]
		
		if filled_fields and not all([self.address_line1, self.city, self.country]):
			raise ValueError("If providing address, please include at least address line 1, city, and country")
		
		return self


# ============================================================================
# UPDATE AND PARTIAL VALIDATION SCHEMAS
# ============================================================================

class EmployeeUpdateSchema(BaseModel):
	"""Schema for partial employee updates."""
	model_config = ConfigDict(
		extra='forbid',
		validate_assignment=True,
		str_strip_whitespace=True
	)
	
	# Only include fields that can be updated
	first_name: Optional[Annotated[str, MinLen(1), MaxLen(100), AfterValidator(normalize_name)]] = None
	middle_name: Optional[Annotated[str, MaxLen(100), AfterValidator(normalize_name)]] = None
	last_name: Optional[Annotated[str, MinLen(1), MaxLen(100), AfterValidator(normalize_name)]] = None
	preferred_name: Optional[Annotated[str, MaxLen(100), AfterValidator(normalize_name)]] = None
	
	personal_email: Optional[Annotated[str, AfterValidator(validate_email)]] = None
	work_email: Optional[Annotated[str, AfterValidator(validate_email)]] = None
	phone_mobile: Optional[Annotated[str, AfterValidator(validate_phone_number)]] = None
	phone_home: Optional[Annotated[str, AfterValidator(validate_phone_number)]] = None
	phone_work: Optional[Annotated[str, AfterValidator(validate_phone_number)]] = None
	
	address_line1: Optional[Annotated[str, MaxLen(200)]] = None
	address_line2: Optional[Annotated[str, MaxLen(200)]] = None
	city: Optional[Annotated[str, MaxLen(100)]] = None
	state_province: Optional[Annotated[str, MaxLen(100)]] = None
	postal_code: Optional[str] = None
	country: Optional[Annotated[str, MaxLen(100)]] = None
	
	department_id: Optional[Annotated[str, MinLen(1)]] = None
	position_id: Optional[Annotated[str, MinLen(1)]] = None
	manager_id: Optional[str] = None
	
	employment_status: Optional[EmploymentStatus] = None
	employment_type: Optional[EmploymentType] = None
	work_location: Optional[WorkLocation] = None
	
	base_salary: Optional[Annotated[Decimal, AfterValidator(validate_salary)]] = None
	hourly_rate: Optional[Annotated[Decimal, Ge(Decimal('0')), Le(Decimal('1000'))]] = None
	
	benefits_eligible: Optional[bool] = None
	benefits_start_date: Optional[date] = None
	
	performance_rating: Optional[Annotated[str, MaxLen(20)]] = None
	next_review_date: Optional[date] = None
	
	is_active: Optional[bool] = None
	photo_url: Optional[Annotated[str, MaxLen(500)]] = None
	
	@field_validator('postal_code')
	@classmethod
	def validate_postal_code_format(cls, v, info):
		if v and 'country' in info.data:
			return validate_postal_code(v, info.data['country'])
		return v


class EmployeeSearchSchema(BaseModel):
	"""Schema for employee search criteria."""
	model_config = ConfigDict(extra='forbid')
	
	# Search criteria
	search_text: Optional[str] = None  # Free text search
	first_name: Optional[str] = None
	last_name: Optional[str] = None
	email: Optional[str] = None
	employee_number: Optional[str] = None
	
	department_id: Optional[str] = None
	position_id: Optional[str] = None
	manager_id: Optional[str] = None
	
	employment_status: Optional[EmploymentStatus] = None
	employment_type: Optional[EmploymentType] = None
	work_location: Optional[WorkLocation] = None
	
	hire_date_from: Optional[date] = None
	hire_date_to: Optional[date] = None
	
	is_active: Optional[bool] = None
	
	# Search options
	include_inactive: bool = False
	limit: Annotated[int, Ge(1), Le(1000)] = 100
	offset: Annotated[int, Ge(0)] = 0
	sort_by: Optional[str] = 'last_name'
	sort_order: Annotated[str, Field(pattern=r'^(asc|desc)$')] = 'asc'


# ============================================================================
# BULK OPERATION SCHEMAS
# ============================================================================

class BulkEmployeeImportSchema(BaseModel):
	"""Schema for bulk employee import."""
	model_config = ConfigDict(extra='forbid')
	
	employees: List[ComprehensiveEmployeeSchema]
	import_options: Dict[str, Any] = Field(default_factory=dict)
	validation_mode: Annotated[str, Field(pattern=r'^(strict|lenient|auto_correct)$')] = 'strict'
	
	@field_validator('employees')
	@classmethod
	def validate_employee_list(cls, v):
		if not v:
			raise ValueError("Employee list cannot be empty")
		if len(v) > 1000:
			raise ValueError("Cannot import more than 1000 employees at once")
		return v


class EmployeeValidationResultSchema(BaseModel):
	"""Schema for validation results."""
	model_config = ConfigDict(extra='forbid')
	
	is_valid: bool
	employee_data: Dict[str, Any]
	validation_errors: List[str] = Field(default_factory=list)
	validation_warnings: List[str] = Field(default_factory=list)
	auto_corrections: List[Dict[str, Any]] = Field(default_factory=list)
	quality_score: float = Field(ge=0.0, le=1.0, default=1.0)
	confidence_score: float = Field(ge=0.0, le=1.0, default=1.0)


# ============================================================================
# VALIDATION HELPER FUNCTIONS
# ============================================================================

def validate_employee_data(employee_data: Dict[str, Any], strict: bool = True) -> EmployeeValidationResultSchema:
	"""Validate employee data against comprehensive schema."""
	try:
		# Attempt validation
		validated_employee = ComprehensiveEmployeeSchema(**employee_data)
		
		return EmployeeValidationResultSchema(
			is_valid=True,
			employee_data=validated_employee.model_dump(),
			quality_score=1.0,
			confidence_score=1.0
		)
		
	except ValidationError as e:
		errors = []
		warnings = []
		
		for error in e.errors():
			error_msg = f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}"
			
			if strict:
				errors.append(error_msg)
			else:
				# In lenient mode, treat some errors as warnings
				if error['type'] in ['missing', 'string_too_short', 'value_error']:
					warnings.append(error_msg)
				else:
					errors.append(error_msg)
		
		return EmployeeValidationResultSchema(
			is_valid=len(errors) == 0,
			employee_data=employee_data,
			validation_errors=errors,
			validation_warnings=warnings,
			quality_score=max(0.0, 1.0 - len(errors) * 0.1 - len(warnings) * 0.05),
			confidence_score=0.8 if warnings else 0.5
		)

def validate_employee_update(update_data: Dict[str, Any]) -> EmployeeValidationResultSchema:
	"""Validate employee update data."""
	try:
		validated_update = EmployeeUpdateSchema(**update_data)
		
		return EmployeeValidationResultSchema(
			is_valid=True,
			employee_data=validated_update.model_dump(exclude_none=True),
			quality_score=1.0,
			confidence_score=1.0
		)
		
	except ValidationError as e:
		errors = [f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}" for error in e.errors()]
		
		return EmployeeValidationResultSchema(
			is_valid=False,
			employee_data=update_data,
			validation_errors=errors,
			quality_score=max(0.0, 1.0 - len(errors) * 0.2),
			confidence_score=0.5
		)