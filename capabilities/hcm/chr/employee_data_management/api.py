"""
Employee Data Management API

REST API endpoints for employee data management operations.
"""

from flask import request, jsonify, current_app
from flask_appbuilder import ModelRestApi
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.security.decorators import has_access_api
from typing import Dict, Any, List, Optional
from datetime import datetime, date

from .models import (
	HREmployee, HRDepartment, HRPosition, HRPersonalInfo, HREmergencyContact,
	HREmploymentHistory, HRSkill, HREmployeeSkill, HRPositionSkill,
	HRCertification, HREmployeeCertification
)
from .service import EmployeeDataManagementService


class HREmployeeRestApi(ModelRestApi):
	"""Employee REST API"""
	
	datamodel = SQLAInterface(HREmployee)
	resource_name = 'employees'
	allow_browser_login = True
	
	list_columns = [
		'employee_id', 'employee_number', 'full_name', 'work_email',
		'department.department_name', 'position.position_title',
		'employment_status', 'hire_date', 'is_active'
	]
	
	show_columns = [
		'employee_id', 'employee_number', 'badge_id', 'full_name',
		'first_name', 'middle_name', 'last_name', 'preferred_name',
		'work_email', 'personal_email', 'phone_work', 'phone_mobile',
		'department', 'position', 'manager', 'employment_status',
		'employment_type', 'work_location', 'hire_date', 'start_date',
		'base_salary', 'currency_code', 'is_active'
	]
	
	add_columns = [
		'first_name', 'middle_name', 'last_name', 'work_email',
		'personal_email', 'phone_mobile', 'department_id', 'position_id',
		'manager_id', 'hire_date', 'employment_status', 'employment_type',
		'base_salary', 'currency_code'
	]
	
	edit_columns = add_columns + ['employee_number', 'is_active']
	
	search_columns = [
		'employee_number', 'first_name', 'last_name', 'work_email',
		'department.department_name', 'position.position_title'
	]
	
	filters_converter_class_map = {
		'employment_status': 'enum',
		'employment_type': 'enum',
		'is_active': 'boolean'
	}


class HRDepartmentRestApi(ModelRestApi):
	"""Department REST API"""
	
	datamodel = SQLAInterface(HRDepartment)
	resource_name = 'departments'
	allow_browser_login = True
	
	list_columns = [
		'department_id', 'department_code', 'department_name',
		'parent_department.department_name', 'manager.full_name',
		'location', 'is_active'
	]
	
	show_columns = [
		'department_id', 'department_code', 'department_name', 'description',
		'parent_department', 'level', 'path', 'manager', 'location',
		'address', 'cost_center', 'budget_allocation', 'is_active'
	]
	
	add_columns = [
		'department_code', 'department_name', 'description',
		'parent_department_id', 'manager_id', 'location', 'address',
		'cost_center', 'budget_allocation'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = [
		'department_code', 'department_name', 'description', 'location'
	]


class HRPositionRestApi(ModelRestApi):
	"""Position REST API"""
	
	datamodel = SQLAInterface(HRPosition)
	resource_name = 'positions'
	allow_browser_login = True
	
	list_columns = [
		'position_id', 'position_code', 'position_title',
		'department.department_name', 'job_level', 'authorized_headcount',
		'current_headcount', 'is_active'
	]
	
	show_columns = [
		'position_id', 'position_code', 'position_title', 'description',
		'responsibilities', 'requirements', 'department', 'job_level',
		'job_family', 'min_salary', 'max_salary', 'currency_code',
		'is_exempt', 'reports_to_position', 'authorized_headcount',
		'current_headcount', 'is_active'
	]
	
	add_columns = [
		'position_code', 'position_title', 'description', 'responsibilities',
		'requirements', 'department_id', 'job_level', 'job_family',
		'min_salary', 'max_salary', 'currency_code', 'is_exempt',
		'reports_to_position_id', 'authorized_headcount'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = [
		'position_code', 'position_title', 'description', 'job_level'
	]


class HRSkillRestApi(ModelRestApi):
	"""Skill REST API"""
	
	datamodel = SQLAInterface(HRSkill)
	resource_name = 'skills'
	allow_browser_login = True
	
	list_columns = [
		'skill_id', 'skill_code', 'skill_name', 'skill_category',
		'skill_type', 'is_core_competency', 'is_active'
	]
	
	show_columns = [
		'skill_id', 'skill_code', 'skill_name', 'description',
		'skill_category', 'skill_type', 'is_core_competency', 'is_active'
	]
	
	add_columns = [
		'skill_code', 'skill_name', 'description', 'skill_category',
		'skill_type', 'is_core_competency'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = [
		'skill_code', 'skill_name', 'description', 'skill_category'
	]


class HRCertificationRestApi(ModelRestApi):
	"""Certification REST API"""
	
	datamodel = SQLAInterface(HRCertification)
	resource_name = 'certifications'
	allow_browser_login = True
	
	list_columns = [
		'certification_id', 'certification_code', 'certification_name',
		'issuing_organization', 'certification_category',
		'validity_period_months', 'is_active'
	]
	
	show_columns = [
		'certification_id', 'certification_code', 'certification_name',
		'description', 'issuing_organization', 'organization_website',
		'certification_category', 'industry', 'validity_period_months',
		'is_renewable', 'is_active'
	]
	
	add_columns = [
		'certification_code', 'certification_name', 'description',
		'issuing_organization', 'organization_website',
		'certification_category', 'industry', 'validity_period_months',
		'is_renewable'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = [
		'certification_code', 'certification_name', 'issuing_organization',
		'certification_category'
	]


class HREmployeeSkillRestApi(ModelRestApi):
	"""Employee Skill REST API"""
	
	datamodel = SQLAInterface(HREmployeeSkill)
	resource_name = 'employee_skills'
	allow_browser_login = True
	
	list_columns = [
		'employee_skill_id', 'employee.full_name', 'skill.skill_name',
		'proficiency_level', 'years_experience', 'manager_validated', 'is_primary'
	]
	
	show_columns = [
		'employee_skill_id', 'employee', 'skill', 'proficiency_level',
		'proficiency_score', 'years_experience', 'last_used_date',
		'self_assessed', 'manager_validated', 'evidence_notes',
		'is_primary', 'is_active'
	]
	
	add_columns = [
		'employee_id', 'skill_id', 'proficiency_level', 'proficiency_score',
		'years_experience', 'last_used_date', 'evidence_notes', 'is_primary'
	]
	
	edit_columns = add_columns + ['manager_validated', 'is_active']
	
	search_columns = [
		'employee.full_name', 'skill.skill_name', 'proficiency_level'
	]


class HREmployeeCertificationRestApi(ModelRestApi):
	"""Employee Certification REST API"""
	
	datamodel = SQLAInterface(HREmployeeCertification)
	resource_name = 'employee_certifications'
	allow_browser_login = True
	
	list_columns = [
		'employee_certification_id', 'employee.full_name',
		'certification.certification_name', 'issued_date', 'expiry_date',
		'status', 'verified'
	]
	
	show_columns = [
		'employee_certification_id', 'employee', 'certification',
		'certificate_number', 'issued_date', 'expiry_date', 'renewal_date',
		'status', 'verified', 'score', 'cost', 'reimbursed', 'is_active'
	]
	
	add_columns = [
		'employee_id', 'certification_id', 'certificate_number',
		'issued_date', 'expiry_date', 'status', 'score', 'cost'
	]
	
	edit_columns = add_columns + ['verified', 'reimbursed', 'is_active']
	
	search_columns = [
		'employee.full_name', 'certification.certification_name',
		'certificate_number', 'status'
	]


class HREmploymentHistoryRestApi(ModelRestApi):
	"""Employment History REST API (read-only)"""
	
	datamodel = SQLAInterface(HREmploymentHistory)
	resource_name = 'employment_history'
	allow_browser_login = True
	
	# Read-only API
	base_permissions = ['can_get', 'can_info']
	
	list_columns = [
		'history_id', 'employee.full_name', 'change_type',
		'effective_date', 'reason', 'approved_by'
	]
	
	show_columns = [
		'history_id', 'employee', 'change_type', 'effective_date',
		'reason', 'notes', 'previous_department_id', 'previous_position_id',
		'previous_salary', 'new_department_id', 'new_position_id',
		'new_salary', 'approved_by', 'approval_date'
	]
	
	search_columns = [
		'employee.full_name', 'change_type', 'reason'
	]


class EmployeeDataManagementApi(BaseApi):
	"""Custom API endpoints for Employee Data Management"""
	
	resource_name = 'edm'
	
	@expose('/dashboard', methods=['GET'])
	@has_access_api
	def dashboard(self):
		"""Get dashboard data"""
		
		service = EmployeeDataManagementService(self.get_tenant_id())
		
		dashboard_data = {
			'total_employees': service.get_employee_count(active_only=False),
			'active_employees': service.get_employee_count(active_only=True),
			'new_hires_30_days': service.get_new_hires_count(days=30),
			'upcoming_reviews': service.get_upcoming_reviews_count(days=30),
			'department_headcount': service.get_department_headcount_report(),
			'turnover_report': service.get_turnover_report(months=12)
		}
		
		return self.response(200, **dashboard_data)
	
	@expose('/org_chart', methods=['GET'])
	@has_access_api
	def org_chart(self):
		"""Get organizational chart data"""
		
		service = EmployeeDataManagementService(self.get_tenant_id())
		
		departments = service.get_departments()
		employees = service.get_employees()
		
		# Build org chart structure
		org_data = self._build_org_chart_data(departments, employees)
		
		return self.response(200, org_chart=org_data)
	
	@expose('/employees/<employee_id>/skills', methods=['GET'])  
	@has_access_api
	def employee_skills(self, employee_id: str):
		"""Get skills for specific employee"""
		
		service = EmployeeDataManagementService(self.get_tenant_id())
		employee = service.get_employee(employee_id)
		
		if not employee:
			return self.response_404()
		
		skills_data = []
		for emp_skill in employee.employee_skills:
			if emp_skill.is_active:
				skills_data.append({
					'skill_id': emp_skill.skill_id,
					'skill_name': emp_skill.skill.skill_name,
					'skill_category': emp_skill.skill.skill_category,
					'proficiency_level': emp_skill.proficiency_level,
					'proficiency_score': emp_skill.proficiency_score,
					'years_experience': float(emp_skill.years_experience) if emp_skill.years_experience else None,
					'manager_validated': emp_skill.manager_validated,
					'is_primary': emp_skill.is_primary
				})
		
		return self.response(200, skills=skills_data)
	
	@expose('/employees/<employee_id>/certifications', methods=['GET'])
	@has_access_api
	def employee_certifications(self, employee_id: str):
		"""Get certifications for specific employee"""
		
		service = EmployeeDataManagementService(self.get_tenant_id())
		employee = service.get_employee(employee_id)
		
		if not employee:
			return self.response_404()
		
		certs_data = []
		for emp_cert in employee.employee_certifications:
			if emp_cert.is_active:
				certs_data.append({
					'certification_id': emp_cert.certification_id,
					'certification_name': emp_cert.certification.certification_name,
					'issuing_organization': emp_cert.certification.issuing_organization,
					'certificate_number': emp_cert.certificate_number,
					'issued_date': emp_cert.issued_date.isoformat() if emp_cert.issued_date else None,
					'expiry_date': emp_cert.expiry_date.isoformat() if emp_cert.expiry_date else None,
					'status': emp_cert.status,
					'verified': emp_cert.verified,
					'score': emp_cert.score
				})
		
		return self.response(200, certifications=certs_data)
	
	@expose('/reports/headcount_by_department', methods=['GET'])
	@has_access_api
	def headcount_by_department(self):
		"""Get headcount report by department"""
		
		service = EmployeeDataManagementService(self.get_tenant_id())
		headcount_data = service.get_department_headcount_report()
		
		return self.response(200, headcount=headcount_data)
	
	@expose('/reports/turnover', methods=['GET'])
	@has_access_api
	def turnover_report(self):
		"""Get turnover report"""
		
		months = request.args.get('months', 12, type=int)
		
		service = EmployeeDataManagementService(self.get_tenant_id())
		turnover_data = service.get_turnover_report(months=months)
		
		return self.response(200, **turnover_data)
	
	@expose('/employees/<employee_id>/terminate', methods=['POST'])
	@has_access_api
	def terminate_employee(self, employee_id: str):
		"""Terminate an employee"""
		
		data = request.json
		
		if not data or 'termination_date' not in data:
			return self.response_400(message="termination_date is required")
		
		try:
			termination_date = datetime.strptime(data['termination_date'], '%Y-%m-%d').date()
		except ValueError:
			return self.response_400(message="Invalid date format. Use YYYY-MM-DD")
		
		service = EmployeeDataManagementService(self.get_tenant_id())
		
		try:
			employee = service.terminate_employee(employee_id, {
				'termination_date': termination_date,
				'reason': data.get('reason', 'Employment terminated')
			})
			
			return self.response(200, message="Employee terminated successfully", employee_id=employee.employee_id)
			
		except ValueError as e:
			return self.response_404(message=str(e))
		except Exception as e:
			current_app.logger.error(f"Error terminating employee: {e}")
			return self.response_500(message="Internal server error")
	
	def _build_org_chart_data(self, departments: List, employees: List) -> Dict[str, Any]:
		"""Build organizational chart data structure"""
		
		# Create department hierarchy
		dept_hierarchy = {}
		for dept in departments:
			dept_hierarchy[dept.department_id] = {
				'id': dept.department_id,
				'name': dept.department_name,
				'code': dept.department_code,
				'manager_id': dept.manager_id,
				'parent_id': dept.parent_department_id,
				'level': dept.level,
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
					'employment_status': emp.employment_status,
					'is_manager': any(e.manager_id == emp.employee_id for e in employees)
				})
		
		return dept_hierarchy
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		# TODO: Implement tenant resolution
		return "default_tenant"


def register_api_views(appbuilder: AppBuilder):
	"""Register Employee Data Management API views"""
	
	# Register model REST APIs
	appbuilder.add_api(HREmployeeRestApi)
	appbuilder.add_api(HRDepartmentRestApi)
	appbuilder.add_api(HRPositionRestApi)
	appbuilder.add_api(HRSkillRestApi)
	appbuilder.add_api(HRCertificationRestApi)
	appbuilder.add_api(HREmployeeSkillRestApi)
	appbuilder.add_api(HREmployeeCertificationRestApi)
	appbuilder.add_api(HREmploymentHistoryRestApi)
	
	# Register custom API
	appbuilder.add_api(EmployeeDataManagementApi)