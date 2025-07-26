"""
Regulatory Compliance Blueprint

Flask blueprint registration for regulatory compliance sub-capability.
Registers views, API endpoints, and URL routes.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	PHRCRegulatoryFrameworkModelView, PHRCSubmissionModelView,
	PHRCAuditModelView, PHRCAuditFindingModelView,
	PHRCDeviationModelView, PHRCCorrectiveActionModelView,
	PHRCInspectionModelView, PHRCComplianceDashboardView,
	PHRCRegulatoryContactModelView, PHRCRegulatoryReportModelView,
	PHRCSubmissionStatusChartView, PHRCDeviationSeverityChartView
)
from .api import regulatory_compliance_api


def register_views(appbuilder: AppBuilder):
	"""Register regulatory compliance views with Flask-AppBuilder"""
	
	# Main dashboard
	appbuilder.add_view_no_menu(PHRCComplianceDashboardView())
	appbuilder.add_link(
		"Regulatory Dashboard",
		href="/pharmaceutical/regulatory/dashboard/",
		icon="fa-shield-alt",
		category="Regulatory Compliance",
		category_icon="fa-shield-alt"
	)
	
	# Regulatory frameworks
	appbuilder.add_view(
		PHRCRegulatoryFrameworkModelView,
		"Regulatory Frameworks",
		icon="fa-gavel",
		category="Regulatory Compliance"
	)
	
	# Submissions management
	appbuilder.add_view(
		PHRCSubmissionModelView,
		"Regulatory Submissions",
		icon="fa-file-medical-alt",
		category="Regulatory Compliance"
	)
	
	# Audit management
	appbuilder.add_view(
		PHRCAuditModelView,
		"Compliance Audits",
		icon="fa-search",
		category="Regulatory Compliance"
	)
	
	appbuilder.add_view(
		PHRCAuditFindingModelView,
		"Audit Findings",
		icon="fa-exclamation-circle",
		category="Regulatory Compliance"
	)
	
	# Deviation and CAPA management
	appbuilder.add_view(
		PHRCDeviationModelView,
		"Quality Deviations",
		icon="fa-exclamation-triangle",
		category="Regulatory Compliance"
	)
	
	appbuilder.add_view(
		PHRCCorrectiveActionModelView,
		"Corrective Actions (CAPA)",
		icon="fa-tools",
		category="Regulatory Compliance"
	)
	
	# Inspection management
	appbuilder.add_view(
		PHRCInspectionModelView,
		"Regulatory Inspections",
		icon="fa-clipboard-check",
		category="Regulatory Compliance"
	)
	
	# Contact management
	appbuilder.add_view(
		PHRCRegulatoryContactModelView,
		"Regulatory Contacts",
		icon="fa-address-book",
		category="Regulatory Compliance"
	)
	
	# Reporting
	appbuilder.add_view(
		PHRCRegulatoryReportModelView,
		"Regulatory Reports",
		icon="fa-chart-line",
		category="Regulatory Compliance"
	)
	
	# Charts and analytics
	appbuilder.add_view(
		PHRCSubmissionStatusChartView,
		"Submission Status Chart",
		icon="fa-chart-pie",
		category="Regulatory Analytics"
	)
	
	appbuilder.add_view(
		PHRCDeviationSeverityChartView,
		"Deviation Severity Chart",
		icon="fa-chart-bar",
		category="Regulatory Analytics"
	)


def create_blueprint() -> Blueprint:
	"""Create Flask blueprint for regulatory compliance"""
	
	regulatory_bp = Blueprint(
		'regulatory_compliance',
		__name__,
		url_prefix='/pharmaceutical/regulatory',
		template_folder='templates',
		static_folder='static'
	)
	
	# Register API blueprint
	regulatory_bp.register_blueprint(regulatory_compliance_api, url_prefix='/api')
	
	return regulatory_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register regulatory compliance permissions"""
	
	permissions = [
		# Regulatory Framework permissions
		('can_list', 'PHRCRegulatoryFrameworkModelView'),
		('can_show', 'PHRCRegulatoryFrameworkModelView'),
		('can_add', 'PHRCRegulatoryFrameworkModelView'),
		('can_edit', 'PHRCRegulatoryFrameworkModelView'),
		('can_delete', 'PHRCRegulatoryFrameworkModelView'),
		
		# Submission permissions
		('can_list', 'PHRCSubmissionModelView'),
		('can_show', 'PHRCSubmissionModelView'),
		('can_add', 'PHRCSubmissionModelView'),
		('can_edit', 'PHRCSubmissionModelView'),
		('can_delete', 'PHRCSubmissionModelView'),
		('can_submit', 'PHRCSubmissionModelView'),
		
		# Audit permissions
		('can_list', 'PHRCAuditModelView'),
		('can_show', 'PHRCAuditModelView'),
		('can_add', 'PHRCAuditModelView'),
		('can_edit', 'PHRCAuditModelView'),
		('can_delete', 'PHRCAuditModelView'),
		
		('can_list', 'PHRCAuditFindingModelView'),
		('can_show', 'PHRCAuditFindingModelView'),
		('can_add', 'PHRCAuditFindingModelView'),
		('can_edit', 'PHRCAuditFindingModelView'),
		('can_delete', 'PHRCAuditFindingModelView'),
		
		# Deviation permissions
		('can_list', 'PHRCDeviationModelView'),
		('can_show', 'PHRCDeviationModelView'),
		('can_add', 'PHRCDeviationModelView'),
		('can_edit', 'PHRCDeviationModelView'),
		('can_delete', 'PHRCDeviationModelView'),
		('can_assign_investigation', 'PHRCDeviationModelView'),
		('can_complete_investigation', 'PHRCDeviationModelView'),
		
		# CAPA permissions
		('can_list', 'PHRCCorrectiveActionModelView'),
		('can_show', 'PHRCCorrectiveActionModelView'),
		('can_add', 'PHRCCorrectiveActionModelView'),
		('can_edit', 'PHRCCorrectiveActionModelView'),
		('can_delete', 'PHRCCorrectiveActionModelView'),
		('can_complete', 'PHRCCorrectiveActionModelView'),
		('can_verify_effectiveness', 'PHRCCorrectiveActionModelView'),
		
		# Inspection permissions
		('can_list', 'PHRCInspectionModelView'),
		('can_show', 'PHRCInspectionModelView'),
		('can_add', 'PHRCInspectionModelView'),
		('can_edit', 'PHRCInspectionModelView'),
		('can_delete', 'PHRCInspectionModelView'),
		
		# Contact permissions
		('can_list', 'PHRCRegulatoryContactModelView'),
		('can_show', 'PHRCRegulatoryContactModelView'),
		('can_add', 'PHRCRegulatoryContactModelView'),
		('can_edit', 'PHRCRegulatoryContactModelView'),
		('can_delete', 'PHRCRegulatoryContactModelView'),
		
		# Report permissions
		('can_list', 'PHRCRegulatoryReportModelView'),
		('can_show', 'PHRCRegulatoryReportModelView'),
		('can_add', 'PHRCRegulatoryReportModelView'),
		('can_edit', 'PHRCRegulatoryReportModelView'),
		('can_delete', 'PHRCRegulatoryReportModelView'),
		
		# Dashboard permissions
		('can_index', 'PHRCComplianceDashboardView'),
		('can_api_metrics', 'PHRCComplianceDashboardView'),
		
		# Chart permissions
		('can_chart', 'PHRCSubmissionStatusChartView'),
		('can_chart', 'PHRCDeviationSeverityChartView'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure():
	"""Get menu structure for regulatory compliance"""
	
	return {
		'name': 'Regulatory Compliance',
		'icon': 'fa-shield-alt',
		'items': [
			{
				'name': 'Regulatory Dashboard',
				'href': '/pharmaceutical/regulatory/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_index on PHRCComplianceDashboardView'
			},
			{
				'name': 'Regulatory Frameworks',
				'href': '/phrcregulatoryframeworkmodelview/list/',
				'icon': 'fa-gavel',
				'permission': 'can_list on PHRCRegulatoryFrameworkModelView'
			},
			{
				'name': 'Submissions',
				'icon': 'fa-file-medical-alt',
				'items': [
					{
						'name': 'All Submissions',
						'href': '/phrcsubmissionmodelview/list/',
						'icon': 'fa-list',
						'permission': 'can_list on PHRCSubmissionModelView'
					},
					{
						'name': 'New Submission',
						'href': '/phrcsubmissionmodelview/add/',
						'icon': 'fa-plus',
						'permission': 'can_add on PHRCSubmissionModelView'
					}
				]
			},
			{
				'name': 'Audits & Inspections',
				'icon': 'fa-search',
				'items': [
					{
						'name': 'Compliance Audits',
						'href': '/phrcauditmodelview/list/',
						'icon': 'fa-search',
						'permission': 'can_list on PHRCAuditModelView'
					},
					{
						'name': 'Audit Findings',
						'href': '/phrcauditfindingmodelview/list/',
						'icon': 'fa-exclamation-circle',
						'permission': 'can_list on PHRCAuditFindingModelView'
					},
					{
						'name': 'Regulatory Inspections',
						'href': '/phrcinspectionmodelview/list/',
						'icon': 'fa-clipboard-check',
						'permission': 'can_list on PHRCInspectionModelView'
					}
				]
			},
			{
				'name': 'Deviations & CAPA',
				'icon': 'fa-exclamation-triangle',
				'items': [
					{
						'name': 'Quality Deviations',
						'href': '/phrcdeviationmodelview/list/',
						'icon': 'fa-exclamation-triangle',
						'permission': 'can_list on PHRCDeviationModelView'
					},
					{
						'name': 'Corrective Actions',
						'href': '/phrccorrectiveactionmodelview/list/',
						'icon': 'fa-tools',
						'permission': 'can_list on PHRCCorrectiveActionModelView'
					}
				]
			},
			{
				'name': 'Reports & Analytics',
				'icon': 'fa-chart-line',
				'items': [
					{
						'name': 'Regulatory Reports',
						'href': '/phrcregulatoryreportmodelview/list/',
						'icon': 'fa-file-alt',
						'permission': 'can_list on PHRCRegulatoryReportModelView'
					},
					{
						'name': 'Submission Status Chart',
						'href': '/phrcsubmissionstatuschartview/chart/',
						'icon': 'fa-chart-pie',
						'permission': 'can_chart on PHRCSubmissionStatusChartView'
					},
					{
						'name': 'Deviation Severity Chart',
						'href': '/phrcdeviationseveritychartview/chart/',
						'icon': 'fa-chart-bar',
						'permission': 'can_chart on PHRCDeviationSeverityChartView'
					}
				]
			},
			{
				'name': 'Configuration',
				'icon': 'fa-cog',
				'items': [
					{
						'name': 'Regulatory Contacts',
						'href': '/phrcregulatorycontactmodelview/list/',
						'icon': 'fa-address-book',
						'permission': 'can_list on PHRCRegulatoryContactModelView'
					}
				]
			}
		]
	}


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize regulatory compliance sub-capability"""
	
	# Register views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Initialize default data if needed
	_init_default_data(appbuilder)


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default regulatory compliance data"""
	
	try:
		from .models import PHRCRegulatoryFramework, PHRCComplianceControl
		from ....auth_rbac.models import db
		
		# Create default regulatory frameworks if they don't exist
		default_frameworks = [
			{
				'framework_code': 'FDA',
				'framework_name': 'Food and Drug Administration',
				'region': 'United States',
				'description': 'US FDA regulations for pharmaceutical products',
				'is_active': True,
				'version': '2024.1'
			},
			{
				'framework_code': 'EMA',
				'framework_name': 'European Medicines Agency',
				'region': 'European Union',
				'description': 'EU EMA regulations for medicinal products',
				'is_active': True,
				'version': '2024.1'
			},
			{
				'framework_code': 'GMP',
				'framework_name': 'Good Manufacturing Practice',
				'region': 'Global',
				'description': 'International GMP standards',
				'is_active': True,
				'version': '2024.1'
			}
		]
		
		for framework_data in default_frameworks:
			existing = PHRCRegulatoryFramework.query.filter_by(
				framework_code=framework_data['framework_code']
			).first()
			
			if not existing:
				framework = PHRCRegulatoryFramework(
					tenant_id='default_tenant',
					**framework_data
				)
				db.session.add(framework)
		
		# Create default compliance controls
		default_controls = [
			{
				'control_code': 'PH-RC-001',
				'control_name': 'Electronic Signature Validation',
				'description': 'Validates electronic signatures per 21 CFR Part 11',
				'control_type': 'Automated',
				'severity': 'Critical',
				'is_active': True
			},
			{
				'control_code': 'PH-RC-002',
				'control_name': 'Audit Trail Completeness',
				'description': 'Ensures complete audit trails for all regulatory data',
				'control_type': 'Automated',
				'severity': 'Critical',
				'is_active': True
			}
		]
		
		for control_data in default_controls:
			existing = PHRCComplianceControl.query.filter_by(
				control_code=control_data['control_code']
			).first()
			
			if not existing:
				# Get FDA framework for default association
				fda_framework = PHRCRegulatoryFramework.query.filter_by(
					framework_code='FDA'
				).first()
				
				if fda_framework:
					control = PHRCComplianceControl(
						tenant_id='default_tenant',
						framework_id=fda_framework.framework_id,
						**control_data
					)
					db.session.add(control)
		
		db.session.commit()
		print("Default regulatory compliance data initialized")
		
	except Exception as e:
		print(f"Error initializing regulatory compliance data: {e}")
		db.session.rollback()