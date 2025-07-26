"""
Employee Data Management Views

Flask-AppBuilder views for employee data management including CRUD operations,
dashboards, and reporting interfaces.
"""

from flask import request, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
from wtforms import validators
from typing import Dict, Any, List, Optional

from .models import (
	HREmployee, HRDepartment, HRPosition, HRPersonalInfo, HREmergencyContact,
	HREmploymentHistory, HRSkill, HREmployeeSkill, HRPositionSkill,
	HRCertification, HREmployeeCertification
)
from .service import EmployeeDataManagementService


class HREmployeeModelView(ModelView):
	"""Employee management view"""
	
	datamodel = SQLAInterface(HREmployee)
	
	# List configuration
	list_columns = [
		'employee_number', 'full_name', 'department.department_name', 
		'position.position_title', 'employment_status', 'hire_date'
	]
	
	search_columns = [
		'employee_number', 'first_name', 'last_name', 'work_email',
		'department.department_name', 'position.position_title'
	]
	
	list_filters = [
		'employment_status', 'employment_type', 'department', 'position', 'hire_date'
	]
	
	# Show configuration
	show_columns = [
		'employee_number', 'badge_id', 'full_name', 'preferred_name',
		'work_email', 'personal_email', 'phone_work', 'phone_mobile',
		'department', 'position', 'manager', 'employment_status', 'employment_type',
		'hire_date', 'start_date', 'base_salary', 'currency_code', 'is_active'
	]
	
	# Edit configuration  
	edit_columns = [
		'employee_number', 'badge_id', 'first_name', 'middle_name', 'last_name',
		'preferred_name', 'work_email', 'personal_email', 'phone_work', 
		'phone_mobile', 'phone_home', 'department', 'position', 'manager',
		'employment_status', 'employment_type', 'work_location',
		'base_salary', 'hourly_rate', 'currency_code', 'pay_frequency',
		'benefits_eligible', 'is_active'
	]
	
	# Add configuration
	add_columns = [
		'first_name', 'middle_name', 'last_name', 'work_email', 'personal_email',
		'phone_mobile', 'department', 'position', 'manager', 'hire_date',
		'employment_status', 'employment_type', 'base_salary', 'currency_code'
	]
	
	# Form validation
	validators_columns = {
		'first_name': [validators.DataRequired()],
		'last_name': [validators.DataRequired()],
		'hire_date': [validators.DataRequired()],
		'department': [validators.DataRequired()],
		'position': [validators.DataRequired()]
	}
	
	# Formatters
	formatters_columns = {
		'base_salary': lambda x: f"${x:,.2f}" if x else "N/A",
		'hire_date': lambda x: x.strftime('%Y-%m-%d') if x else "N/A"
	}
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	def pre_add(self, item):
		"""Pre-process before adding new employee"""
		item.tenant_id = self.get_tenant_id()
		
		# Generate full name
		full_name = item.first_name
		if item.middle_name:
			full_name += f" {item.middle_name}"
		full_name += f" {item.last_name}"
		item.full_name = full_name
		
		# Generate employee number if not provided
		if not item.employee_number:
			service = EmployeeDataManagementService(self.get_tenant_id())
			item.employee_number = service._generate_employee_number()
	
	def pre_update(self, item):
		"""Pre-process before updating employee"""
		# Update full name if name components changed
		full_name = item.first_name
		if item.middle_name:
			full_name += f" {item.middle_name}"
		full_name += f" {item.last_name}"
		item.full_name = full_name
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		# TODO: Implement tenant resolution from session/context
		return "default_tenant"


class HRDepartmentModelView(ModelView):
	"""Department management view"""
	
	datamodel = SQLAInterface(HRDepartment)
	
	# List configuration
	list_columns = [
		'department_code', 'department_name', 'parent_department.department_name',
		'manager.full_name', 'location', 'is_active'
	]
	
	search_columns = [
		'department_code', 'department_name', 'description', 'location'
	]
	
	list_filters = ['is_active', 'parent_department', 'level']
	
	# Show configuration
	show_columns = [
		'department_code', 'department_name', 'description', 'parent_department',
		'level', 'path', 'manager', 'location', 'address', 'cost_center',
		'budget_allocation', 'is_active'
	]
	
	# Edit configuration
	edit_columns = [
		'department_code', 'department_name', 'description', 'parent_department',
		'manager', 'location', 'address', 'cost_center', 'budget_allocation',
		'is_active'
	]
	
	# Add configuration
	add_columns = [
		'department_code', 'department_name', 'description', 'parent_department',
		'location', 'cost_center', 'budget_allocation'
	]
	
	# Form validation
	validators_columns = {
		'department_code': [validators.DataRequired(), validators.Length(max=20)],
		'department_name': [validators.DataRequired(), validators.Length(max=200)]
	}
	
	# Formatters
	formatters_columns = {
		'budget_allocation': lambda x: f"${x:,.2f}" if x else "N/A"
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new department"""
		item.tenant_id = self.get_tenant_id()
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class HRPositionModelView(ModelView):
	"""Position management view"""
	
	datamodel = SQLAInterface(HRPosition)
	
	# List configuration
	list_columns = [
		'position_code', 'position_title', 'department.department_name',
		'job_level', 'authorized_headcount', 'current_headcount', 'is_active'
	]
	
	search_columns = [
		'position_code', 'position_title', 'description', 'job_level', 'job_family'
	]
	
	list_filters = ['department', 'job_level', 'job_family', 'is_active']
	
	# Show configuration
	show_columns = [
		'position_code', 'position_title', 'description', 'responsibilities',
		'requirements', 'department', 'job_level', 'job_family',
		'min_salary', 'max_salary', 'currency_code', 'is_exempt',
		'reports_to_position', 'authorized_headcount', 'current_headcount',
		'is_active'
	]
	
	# Edit configuration
	edit_columns = [
		'position_code', 'position_title', 'description', 'responsibilities',
		'requirements', 'department', 'job_level', 'job_family',
		'min_salary', 'max_salary', 'currency_code', 'is_exempt',
		'reports_to_position', 'authorized_headcount', 'is_active'
	]
	
	# Add configuration
	add_columns = [
		'position_code', 'position_title', 'description', 'department',
		'job_level', 'min_salary', 'max_salary', 'authorized_headcount'
	]
	
	# Form validation
	validators_columns = {
		'position_code': [validators.DataRequired(), validators.Length(max=20)],
		'position_title': [validators.DataRequired(), validators.Length(max=200)],
		'department': [validators.DataRequired()]
	}
	
	# Formatters
	formatters_columns = {
		'min_salary': lambda x: f"${x:,.2f}" if x else "N/A",
		'max_salary': lambda x: f"${x:,.2f}" if x else "N/A"
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new position"""
		item.tenant_id = self.get_tenant_id()
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class HRSkillModelView(ModelView):
	"""Skills management view"""
	
	datamodel = SQLAInterface(HRSkill)
	
	# List configuration
	list_columns = [
		'skill_code', 'skill_name', 'skill_category', 'skill_type',
		'is_core_competency', 'is_active'
	]
	
	search_columns = ['skill_code', 'skill_name', 'description', 'skill_category']
	list_filters = ['skill_category', 'skill_type', 'is_core_competency', 'is_active']
	
	# Show configuration
	show_columns = [
		'skill_code', 'skill_name', 'description', 'skill_category',
		'skill_type', 'is_core_competency', 'is_active'
	]
	
	# Edit configuration
	edit_columns = [
		'skill_code', 'skill_name', 'description', 'skill_category',
		'skill_type', 'is_core_competency', 'is_active'
	]
	
	# Add configuration
	add_columns = [
		'skill_code', 'skill_name', 'description', 'skill_category', 'skill_type'
	]
	
	# Form validation
	validators_columns = {
		'skill_code': [validators.DataRequired(), validators.Length(max=20)],
		'skill_name': [validators.DataRequired(), validators.Length(max=200)]
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new skill"""
		item.tenant_id = self.get_tenant_id()
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class HRCertificationModelView(ModelView):
	"""Certifications management view"""
	
	datamodel = SQLAInterface(HRCertification)
	
	# List configuration
	list_columns = [
		'certification_code', 'certification_name', 'issuing_organization',
		'certification_category', 'validity_period_months', 'is_active'
	]
	
	search_columns = [
		'certification_code', 'certification_name', 'issuing_organization',
		'certification_category'
	]
	
	list_filters = ['certification_category', 'industry', 'is_renewable', 'is_active']
	
	# Show configuration
	show_columns = [
		'certification_code', 'certification_name', 'description',
		'issuing_organization', 'organization_website', 'certification_category',
		'industry', 'validity_period_months', 'is_renewable', 'is_active'
	]
	
	# Edit configuration
	edit_columns = [
		'certification_code', 'certification_name', 'description',
		'issuing_organization', 'organization_website', 'certification_category',
		'industry', 'validity_period_months', 'is_renewable', 'is_active'
	]
	
	# Add configuration
	add_columns = [
		'certification_code', 'certification_name', 'issuing_organization',
		'certification_category', 'validity_period_months'
	]
	
	# Form validation
	validators_columns = {
		'certification_code': [validators.DataRequired(), validators.Length(max=20)],
		'certification_name': [validators.DataRequired(), validators.Length(max=200)],
		'issuing_organization': [validators.DataRequired(), validators.Length(max=200)]
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new certification"""
		item.tenant_id = self.get_tenant_id()
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class HREmployeeSkillModelView(ModelView):
	"""Employee skills management view"""
	
	datamodel = SQLAInterface(HREmployeeSkill)
	
	# List configuration
	list_columns = [
		'employee.full_name', 'skill.skill_name', 'proficiency_level',
		'years_experience', 'manager_validated', 'is_primary'
	]
	
	search_columns = [
		'employee.full_name', 'skill.skill_name', 'proficiency_level',
		'evidence_notes'
	]
	
	list_filters = [
		'skill', 'proficiency_level', 'self_assessed', 'manager_validated',
		'is_primary', 'is_active'
	]
	
	# Show configuration
	show_columns = [
		'employee', 'skill', 'proficiency_level', 'proficiency_score',
		'years_experience', 'last_used_date', 'self_assessed',
		'manager_validated', 'validated_by', 'validation_date',
		'evidence_notes', 'is_primary', 'is_active'
	]
	
	# Edit configuration
	edit_columns = [
		'employee', 'skill', 'proficiency_level', 'proficiency_score',
		'years_experience', 'last_used_date', 'evidence_notes',
		'manager_validated', 'is_primary', 'is_active'
	]
	
	# Add configuration
	add_columns = [
		'employee', 'skill', 'proficiency_level', 'years_experience',
		'evidence_notes'
	]
	
	# Form validation
	validators_columns = {
		'employee': [validators.DataRequired()],
		'skill': [validators.DataRequired()],
		'proficiency_level': [validators.DataRequired()]
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new employee skill"""
		item.tenant_id = self.get_tenant_id()
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class HREmployeeCertificationModelView(ModelView):
	"""Employee certifications management view"""
	
	datamodel = SQLAInterface(HREmployeeCertification)
	
	# List configuration
	list_columns = [
		'employee.full_name', 'certification.certification_name',
		'issued_date', 'expiry_date', 'status', 'verified'
	]
	
	search_columns = [
		'employee.full_name', 'certification.certification_name',
		'certificate_number', 'status'
	]
	
	list_filters = [
		'certification', 'status', 'verified', 'issued_date', 'expiry_date'
	]
	
	# Show configuration
	show_columns = [
		'employee', 'certification', 'certificate_number', 'issued_date',
		'expiry_date', 'renewal_date', 'status', 'verified', 'verified_by',
		'verification_date', 'score', 'cost', 'reimbursed', 'is_active'
	]
	
	# Edit configuration
	edit_columns = [
		'employee', 'certification', 'certificate_number', 'issued_date',
		'expiry_date', 'status', 'verified', 'score', 'cost',
		'reimbursed', 'reimbursement_amount', 'is_active'
	]
	
	# Add configuration
	add_columns = [
		'employee', 'certification', 'certificate_number', 'issued_date',
		'expiry_date', 'status', 'score', 'cost'
	]
	
	# Form validation
	validators_columns = {
		'employee': [validators.DataRequired()],
		'certification': [validators.DataRequired()],
		'issued_date': [validators.DataRequired()]
	}
	
	# Formatters
	formatters_columns = {
		'cost': lambda x: f"${x:,.2f}" if x else "N/A",
		'issued_date': lambda x: x.strftime('%Y-%m-%d') if x else "N/A",
		'expiry_date': lambda x: x.strftime('%Y-%m-%d') if x else "N/A"
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new employee certification"""
		item.tenant_id = self.get_tenant_id()
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class HREmployeeDashboardView(BaseView):
	"""Employee Data Management Dashboard"""
	
	route_base = "/hr/employee_dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Display employee dashboard with key metrics"""
		
		service = EmployeeDataManagementService(self.get_tenant_id())
		
		# Get dashboard metrics
		dashboard_data = {
			'total_employees': service.get_employee_count(active_only=False),
			'active_employees': service.get_employee_count(active_only=True),
			'new_hires_30_days': service.get_new_hires_count(days=30),
			'upcoming_reviews': service.get_upcoming_reviews_count(days=30),
			'department_headcount': service.get_department_headcount_report(),
			'turnover_report': service.get_turnover_report(months=12)
		}
		
		return self.render_template(
			'hr_employee_dashboard.html',
			dashboard_data=dashboard_data,
			title="Employee Data Management Dashboard"
		)
	
	@expose('/org_chart')
	@has_access
	def org_chart(self):
		"""Display organizational chart"""
		
		service = EmployeeDataManagementService(self.get_tenant_id())
		
		# Get departments and employees for org chart
		departments = service.get_departments()
		employees = service.get_employees()
		
		# Build org chart data structure
		org_data = self._build_org_chart_data(departments, employees)
		
		return self.render_template(
			'hr_org_chart.html',
			org_data=org_data,
			title="Organizational Chart"
		)
	
	def _build_org_chart_data(self, departments: List, employees: List) -> Dict[str, Any]:
		"""Build organizational chart data structure"""
		
		# Create department hierarchy
		dept_hierarchy = {}
		for dept in departments:
			dept_hierarchy[dept.department_id] = {
				'name': dept.department_name,
				'code': dept.department_code,
				'manager_id': dept.manager_id,
				'parent_id': dept.parent_department_id,
				'employees': []
			}
		
		# Add employees to departments
		for emp in employees:
			if emp.department_id in dept_hierarchy:
				dept_hierarchy[emp.department_id]['employees'].append({
					'id': emp.employee_id,
					'name': emp.full_name,
					'title': emp.position.position_title if emp.position else 'N/A',
					'manager_id': emp.manager_id,
					'is_manager': any(e.manager_id == emp.employee_id for e in employees)
				})
		
		return dept_hierarchy
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class HREmploymentHistoryModelView(ModelView):
	"""Employment history view (read-only)"""
	
	datamodel = SQLAInterface(HREmploymentHistory)
	
	# List configuration
	list_columns = [
		'employee.full_name', 'change_type', 'effective_date',
		'reason', 'approved_by'
	]
	
	search_columns = [
		'employee.full_name', 'change_type', 'reason'
	]
	
	list_filters = [
		'change_type', 'effective_date', 'employee'
	]
	
	# Show configuration
	show_columns = [
		'employee', 'change_type', 'effective_date', 'reason', 'notes',
		'previous_department_id', 'previous_position_id', 'previous_salary',
		'new_department_id', 'new_position_id', 'new_salary',
		'approved_by', 'approval_date'
	]
	
	# Read-only view
	base_permissions = ['can_list', 'can_show']
	
	# Formatters
	formatters_columns = {
		'effective_date': lambda x: x.strftime('%Y-%m-%d') if x else "N/A",
		'previous_salary': lambda x: f"${x:,.2f}" if x else "N/A",
		'new_salary': lambda x: f"${x:,.2f}" if x else "N/A"
	}