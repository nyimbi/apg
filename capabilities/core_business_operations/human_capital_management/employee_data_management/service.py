"""
Employee Data Management Service

Business logic for employee data management including employee lifecycle,
organizational structure, skills management, and reporting.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from .models import (
	HREmployee, HRDepartment, HRPosition, HRPersonalInfo, HREmergencyContact,
	HREmploymentHistory, HRSkill, HREmployeeSkill, HRPositionSkill,
	HRCertification, HREmployeeCertification
)
from ...auth_rbac.models import db


class EmployeeDataManagementService:
	"""Service class for Employee Data Management operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	# Employee Management
	
	def create_employee(self, employee_data: Dict[str, Any]) -> HREmployee:
		"""Create a new employee record"""
		
		# Generate employee number if not provided
		if 'employee_number' not in employee_data:
			employee_data['employee_number'] = self._generate_employee_number()
		
		# Compute full name
		full_name = f"{employee_data['first_name']}"
		if employee_data.get('middle_name'):
			full_name += f" {employee_data['middle_name']}"
		full_name += f" {employee_data['last_name']}"
		
		employee = HREmployee(
			tenant_id=self.tenant_id,
			employee_number=employee_data['employee_number'],
			badge_id=employee_data.get('badge_id'),
			first_name=employee_data['first_name'],
			middle_name=employee_data.get('middle_name'),
			last_name=employee_data['last_name'],
			preferred_name=employee_data.get('preferred_name'),
			full_name=full_name,
			personal_email=employee_data.get('personal_email'),
			work_email=employee_data.get('work_email'),
			phone_mobile=employee_data.get('phone_mobile'),
			phone_home=employee_data.get('phone_home'),
			phone_work=employee_data.get('phone_work'),
			date_of_birth=employee_data.get('date_of_birth'),
			gender=employee_data.get('gender'),
			marital_status=employee_data.get('marital_status'),
			nationality=employee_data.get('nationality'),
			address_line1=employee_data.get('address_line1'),
			address_line2=employee_data.get('address_line2'),
			city=employee_data.get('city'),
			state_province=employee_data.get('state_province'),
			postal_code=employee_data.get('postal_code'),
			country=employee_data.get('country'),
			department_id=employee_data['department_id'],
			position_id=employee_data['position_id'],
			manager_id=employee_data.get('manager_id'),
			hire_date=employee_data['hire_date'],
			start_date=employee_data.get('start_date', employee_data['hire_date']),
			employment_status=employee_data.get('employment_status', 'Active'),
			employment_type=employee_data.get('employment_type', 'Full-Time'),
			work_location=employee_data.get('work_location', 'Office'),
			base_salary=employee_data.get('base_salary'),
			hourly_rate=employee_data.get('hourly_rate'),
			currency_code=employee_data.get('currency_code', 'USD'),
			pay_frequency=employee_data.get('pay_frequency', 'Monthly'),
			benefits_eligible=employee_data.get('benefits_eligible', True),
			tax_id=employee_data.get('tax_id'),
			tax_country=employee_data.get('tax_country', 'USA'),
			tax_state=employee_data.get('tax_state'),
			is_active=employee_data.get('is_active', True)
		)
		
		# Set probation end date if applicable
		if employee_data.get('probation_period_days'):
			employee.probation_end_date = employee.hire_date + timedelta(days=employee_data['probation_period_days'])
		
		# Set benefits start date
		if employee.benefits_eligible and not employee_data.get('benefits_start_date'):
			employee.benefits_start_date = employee.start_date
		
		db.session.add(employee)
		db.session.flush()  # Get the employee_id
		
		# Create employment history record
		self._create_employment_history_record(
			employee.employee_id,
			'Hire',
			employee.hire_date,
			f"Initial hire as {employee.position.position_title}",
			new_department_id=employee.department_id,
			new_position_id=employee.position_id,
			new_manager_id=employee.manager_id,
			new_salary=employee.base_salary,
			new_status=employee.employment_status
		)
		
		db.session.commit()
		return employee
	
	def get_employee(self, employee_id: str) -> Optional[HREmployee]:
		"""Get employee by ID"""
		return HREmployee.query.filter_by(
			tenant_id=self.tenant_id,
			employee_id=employee_id
		).first()
	
	def get_employee_by_number(self, employee_number: str) -> Optional[HREmployee]:
		"""Get employee by employee number"""
		return HREmployee.query.filter_by(
			tenant_id=self.tenant_id,
			employee_number=employee_number
		).first()
	
	def get_employee_by_email(self, email: str) -> Optional[HREmployee]:
		"""Get employee by work email"""
		return HREmployee.query.filter_by(
			tenant_id=self.tenant_id,
			work_email=email
		).first()
	
	def get_employees(self, 
					  active_only: bool = True,
					  department_id: Optional[str] = None,
					  position_id: Optional[str] = None,
					  manager_id: Optional[str] = None,
					  limit: Optional[int] = None,
					  offset: Optional[int] = None) -> List[HREmployee]:
		"""Get employees with optional filtering"""
		
		query = HREmployee.query.filter_by(tenant_id=self.tenant_id)
		
		if active_only:
			query = query.filter_by(is_active=True)
		
		if department_id:
			query = query.filter_by(department_id=department_id)
		
		if position_id:
			query = query.filter_by(position_id=position_id)
		
		if manager_id:
			query = query.filter_by(manager_id=manager_id)
		
		query = query.order_by(HREmployee.last_name, HREmployee.first_name)
		
		if limit:
			query = query.limit(limit)
		
		if offset:
			query = query.offset(offset)
		
		return query.all()
	
	def update_employee(self, employee_id: str, updates: Dict[str, Any]) -> HREmployee:
		"""Update employee information"""
		employee = self.get_employee(employee_id)
		if not employee:
			raise ValueError(f"Employee {employee_id} not found")
		
		# Track changes that need employment history
		position_changed = 'position_id' in updates and updates['position_id'] != employee.position_id
		department_changed = 'department_id' in updates and updates['department_id'] != employee.department_id
		manager_changed = 'manager_id' in updates and updates['manager_id'] != employee.manager_id
		salary_changed = ('base_salary' in updates and updates['base_salary'] != employee.base_salary) or \
						('hourly_rate' in updates and updates['hourly_rate'] != employee.hourly_rate)
		status_changed = 'employment_status' in updates and updates['employment_status'] != employee.employment_status
		
		# Store previous values for history
		previous_values = {}
		if position_changed or department_changed or manager_changed or salary_changed or status_changed:
			previous_values = {
				'department_id': employee.department_id,
				'position_id': employee.position_id,
				'manager_id': employee.manager_id,
				'salary': employee.base_salary or employee.hourly_rate,
				'status': employee.employment_status
			}
		
		# Update full name if name components changed
		if any(field in updates for field in ['first_name', 'middle_name', 'last_name']):
			first_name = updates.get('first_name', employee.first_name)
			middle_name = updates.get('middle_name', employee.middle_name)
			last_name = updates.get('last_name', employee.last_name)
			
			full_name = first_name
			if middle_name:
				full_name += f" {middle_name}"
			full_name += f" {last_name}"
			updates['full_name'] = full_name
		
		# Apply updates
		for field, value in updates.items():
			if hasattr(employee, field):
				setattr(employee, field, value)
		
		# Create employment history record if significant changes
		if position_changed or department_changed or manager_changed or salary_changed or status_changed:
			change_type = 'Transfer' if department_changed else 'Promotion' if position_changed else 'Update'
			
			self._create_employment_history_record(
				employee_id,
				change_type,
				updates.get('effective_date', date.today()),
				updates.get('change_reason', f"{change_type} - system update"),
				previous_department_id=previous_values.get('department_id'),
				previous_position_id=previous_values.get('position_id'),
				previous_manager_id=previous_values.get('manager_id'),
				previous_salary=previous_values.get('salary'),
				previous_status=previous_values.get('status'),
				new_department_id=employee.department_id,
				new_position_id=employee.position_id,
				new_manager_id=employee.manager_id,
				new_salary=employee.base_salary or employee.hourly_rate,
				new_status=employee.employment_status
			)
		
		db.session.commit()
		return employee
	
	def terminate_employee(self, employee_id: str, termination_data: Dict[str, Any]) -> HREmployee:
		"""Terminate an employee"""
		employee = self.get_employee(employee_id)
		if not employee:
			raise ValueError(f"Employee {employee_id} not found")
		
		employee.termination_date = termination_data['termination_date']
		employee.employment_status = 'Terminated'
		employee.is_active = False
		
		# Create employment history record
		self._create_employment_history_record(
			employee_id,
			'Termination',
			termination_data['termination_date'],
			termination_data.get('reason', 'Employment terminated'),
			previous_status='Active',
			new_status='Terminated'
		)
		
		db.session.commit()
		return employee
	
	# Department Management
	
	def create_department(self, department_data: Dict[str, Any]) -> HRDepartment:
		"""Create a new department"""
		department = HRDepartment(
			tenant_id=self.tenant_id,
			department_code=department_data['department_code'],
			department_name=department_data['department_name'],
			description=department_data.get('description'),
			parent_department_id=department_data.get('parent_department_id'),
			cost_center=department_data.get('cost_center'),
			budget_allocation=department_data.get('budget_allocation'),
			manager_id=department_data.get('manager_id'),
			location=department_data.get('location'),
			address=department_data.get('address'),
			is_active=department_data.get('is_active', True)
		)
		
		# Calculate hierarchy level and path
		if department.parent_department_id:
			parent = self.get_department(department.parent_department_id)
			department.level = parent.level + 1
			department.path = f"{parent.path}/{department.department_code}" if parent.path else department.department_code
		else:
			department.level = 0
			department.path = department.department_code
		
		db.session.add(department)
		db.session.commit()
		return department
	
	def get_department(self, department_id: str) -> Optional[HRDepartment]:
		"""Get department by ID"""
		return HRDepartment.query.filter_by(
			tenant_id=self.tenant_id,
			department_id=department_id
		).first()
	
	def get_departments(self, include_inactive: bool = False) -> List[HRDepartment]:
		"""Get departments"""
		query = HRDepartment.query.filter_by(tenant_id=self.tenant_id)
		
		if not include_inactive:
			query = query.filter_by(is_active=True)
		
		return query.order_by(HRDepartment.department_code).all()
	
	def get_department_hierarchy(self, parent_id: Optional[str] = None) -> List[HRDepartment]:
		"""Get departments in hierarchical structure"""
		if parent_id:
			return HRDepartment.query.filter_by(
				tenant_id=self.tenant_id,
				parent_department_id=parent_id,
				is_active=True
			).order_by(HRDepartment.department_code).all()
		else:
			return HRDepartment.query.filter_by(
				tenant_id=self.tenant_id,
				parent_department_id=None,
				is_active=True
			).order_by(HRDepartment.department_code).all()
	
	# Position Management
	
	def create_position(self, position_data: Dict[str, Any]) -> HRPosition:
		"""Create a new position"""
		position = HRPosition(
			tenant_id=self.tenant_id,
			position_code=position_data['position_code'],
			position_title=position_data['position_title'],
			description=position_data.get('description'),
			responsibilities=position_data.get('responsibilities'),
			requirements=position_data.get('requirements'),
			department_id=position_data['department_id'],
			job_level=position_data.get('job_level'),
			job_family=position_data.get('job_family'),
			min_salary=position_data.get('min_salary'),
			max_salary=position_data.get('max_salary'),
			currency_code=position_data.get('currency_code', 'USD'),
			is_active=position_data.get('is_active', True),
			is_exempt=position_data.get('is_exempt', True),
			reports_to_position_id=position_data.get('reports_to_position_id'),
			authorized_headcount=position_data.get('authorized_headcount', 1)
		)
		
		db.session.add(position)
		db.session.commit()
		return position
	
	def get_position(self, position_id: str) -> Optional[HRPosition]:
		"""Get position by ID"""
		return HRPosition.query.filter_by(
			tenant_id=self.tenant_id,
			position_id=position_id
		).first()
	
	def get_positions(self, 
					  department_id: Optional[str] = None,
					  include_inactive: bool = False) -> List[HRPosition]:
		"""Get positions"""
		query = HRPosition.query.filter_by(tenant_id=self.tenant_id)
		
		if department_id:
			query = query.filter_by(department_id=department_id)
		
		if not include_inactive:
			query = query.filter_by(is_active=True)
		
		return query.order_by(HRPosition.position_title).all()
	
	# Skills Management
	
	def create_skill(self, skill_data: Dict[str, Any]) -> HRSkill:
		"""Create a new skill"""
		skill = HRSkill(
			tenant_id=self.tenant_id,
			skill_code=skill_data['skill_code'],
			skill_name=skill_data['skill_name'],
			description=skill_data.get('description'),
			skill_category=skill_data.get('skill_category'),
			skill_type=skill_data.get('skill_type'),
			is_active=skill_data.get('is_active', True),
			is_core_competency=skill_data.get('is_core_competency', False)
		)
		
		db.session.add(skill)
		db.session.commit()
		return skill
	
	def assign_skill_to_employee(self, employee_id: str, skill_id: str, skill_data: Dict[str, Any]) -> HREmployeeSkill:
		"""Assign a skill to an employee"""
		employee_skill = HREmployeeSkill(
			tenant_id=self.tenant_id,
			employee_id=employee_id,
			skill_id=skill_id,
			proficiency_level=skill_data['proficiency_level'],
			proficiency_score=skill_data.get('proficiency_score'),
			self_assessed=skill_data.get('self_assessed', True),
			years_experience=skill_data.get('years_experience'),
			last_used_date=skill_data.get('last_used_date'),
			evidence_notes=skill_data.get('evidence_notes'),
			is_primary=skill_data.get('is_primary', False)
		)
		
		db.session.add(employee_skill)
		db.session.commit()
		return employee_skill
	
	# Certification Management
	
	def create_certification(self, cert_data: Dict[str, Any]) -> HRCertification:
		"""Create a new certification"""
		certification = HRCertification(
			tenant_id=self.tenant_id,
			certification_code=cert_data['certification_code'],
			certification_name=cert_data['certification_name'],
			description=cert_data.get('description'),
			issuing_organization=cert_data['issuing_organization'],
			organization_website=cert_data.get('organization_website'),
			certification_category=cert_data.get('certification_category'),
			industry=cert_data.get('industry'),
			validity_period_months=cert_data.get('validity_period_months'),
			is_renewable=cert_data.get('is_renewable', True),
			is_active=cert_data.get('is_active', True)
		)
		
		db.session.add(certification)
		db.session.commit()
		return certification
	
	def assign_certification_to_employee(self, employee_id: str, certification_id: str, cert_data: Dict[str, Any]) -> HREmployeeCertification:
		"""Assign a certification to an employee"""
		employee_cert = HREmployeeCertification(
			tenant_id=self.tenant_id,
			employee_id=employee_id,
			certification_id=certification_id,
			certificate_number=cert_data.get('certificate_number'),
			issued_date=cert_data['issued_date'],
			expiry_date=cert_data.get('expiry_date'),
			status=cert_data.get('status', 'Active'),
			score=cert_data.get('score'),
			score_details=cert_data.get('score_details'),
			certificate_file_path=cert_data.get('certificate_file_path'),
			cost=cert_data.get('cost'),
			reimbursed=cert_data.get('reimbursed', False)
		)
		
		db.session.add(employee_cert)
		db.session.commit()
		return employee_cert
	
	# Reporting and Analytics
	
	def get_employee_count(self, active_only: bool = True) -> int:
		"""Get total employee count"""
		query = HREmployee.query.filter_by(tenant_id=self.tenant_id)
		
		if active_only:
			query = query.filter_by(is_active=True)
		
		return query.count()
	
	def get_new_hires_count(self, days: int = 30) -> int:
		"""Get count of new hires in the last N days"""
		since_date = date.today() - timedelta(days=days)
		
		return HREmployee.query.filter(
			HREmployee.tenant_id == self.tenant_id,
			HREmployee.hire_date >= since_date,
			HREmployee.is_active == True
		).count()
	
	def get_upcoming_reviews_count(self, days: int = 30) -> int:
		"""Get count of upcoming performance reviews"""
		until_date = date.today() + timedelta(days=days)
		
		return HREmployee.query.filter(
			HREmployee.tenant_id == self.tenant_id,
			HREmployee.next_review_date <= until_date,
			HREmployee.next_review_date >= date.today(),
			HREmployee.is_active == True
		).count()
	
	def get_department_headcount_report(self) -> List[Dict[str, Any]]:
		"""Get headcount by department"""
		result = db.session.query(
			HRDepartment.department_name,
			HRDepartment.department_code,
			func.count(HREmployee.employee_id).label('headcount'),
			func.count(func.nullif(HREmployee.employment_status, 'Active')).label('inactive_count')
		).join(
			HREmployee, HRDepartment.department_id == HREmployee.department_id
		).filter(
			HRDepartment.tenant_id == self.tenant_id,
			HREmployee.tenant_id == self.tenant_id
		).group_by(
			HRDepartment.department_id,
			HRDepartment.department_name,
			HRDepartment.department_code
		).all()
		
		return [
			{
				'department_name': row.department_name,
				'department_code': row.department_code,
				'total_headcount': row.headcount,
				'active_headcount': row.headcount - row.inactive_count,
				'inactive_headcount': row.inactive_count
			}
			for row in result
		]
	
	def get_turnover_report(self, months: int = 12) -> Dict[str, Any]:
		"""Get employee turnover report"""
		since_date = date.today() - timedelta(days=months * 30)
		
		# Get terminations in period
		terminations = HREmployee.query.filter(
			HREmployee.tenant_id == self.tenant_id,
			HREmployee.termination_date >= since_date,
			HREmployee.termination_date <= date.today()
		).count()
		
		# Get average headcount during period
		avg_headcount = self.get_employee_count(active_only=False)
		
		# Calculate turnover rate
		turnover_rate = (terminations / avg_headcount * 100) if avg_headcount > 0 else 0
		
		return {
			'period_months': months,
			'terminations': terminations,
			'average_headcount': avg_headcount,
			'turnover_rate_percent': round(turnover_rate, 2)
		}
	
	# Private Helper Methods
	
	def _generate_employee_number(self) -> str:
		"""Generate next employee number"""
		# Get the highest current employee number
		last_employee = HREmployee.query.filter_by(
			tenant_id=self.tenant_id
		).order_by(HREmployee.employee_number.desc()).first()
		
		if last_employee and last_employee.employee_number.startswith('EMP'):
			try:
				last_number = int(last_employee.employee_number[3:])
				next_number = last_number + 1
			except (ValueError, IndexError):
				next_number = 1
		else:
			next_number = 1
		
		return f"EMP{next_number:06d}"
	
	def _create_employment_history_record(self, 
										  employee_id: str,
										  change_type: str,
										  effective_date: date,
										  reason: str,
										  previous_department_id: Optional[str] = None,
										  previous_position_id: Optional[str] = None,
										  previous_manager_id: Optional[str] = None,
										  previous_salary: Optional[Decimal] = None,
										  previous_status: Optional[str] = None,
										  new_department_id: Optional[str] = None,
										  new_position_id: Optional[str] = None,
										  new_manager_id: Optional[str] = None,
										  new_salary: Optional[Decimal] = None,
										  new_status: Optional[str] = None) -> HREmploymentHistory:
		"""Create employment history record"""
		
		history = HREmploymentHistory(
			tenant_id=self.tenant_id,
			employee_id=employee_id,
			change_type=change_type,
			effective_date=effective_date,
			reason=reason,
			previous_department_id=previous_department_id,
			previous_position_id=previous_position_id,
			previous_manager_id=previous_manager_id,
			previous_salary=previous_salary,
			previous_status=previous_status,
			new_department_id=new_department_id,
			new_position_id=new_position_id,
			new_manager_id=new_manager_id,
			new_salary=new_salary,
			new_status=new_status
		)
		
		db.session.add(history)
		return history