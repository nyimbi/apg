"""
Pharmaceutical Specific Blueprint

Flask blueprint registration for pharmaceutical industry-specific capability.
Registers all sub-capability views, API endpoints, and URL routes.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

# Import sub-capability blueprint registration functions
from .regulatory_compliance.blueprint import register_views as register_regulatory_views
from .product_serialization_tracking.blueprint import register_views as register_serialization_views
from .clinical_trials_management.blueprint import register_views as register_clinical_views
from .rd_management.blueprint import register_views as register_rd_views
from .batch_release_management.blueprint import register_views as register_batch_views


def register_views(appbuilder: AppBuilder):
	"""Register all pharmaceutical views with Flask-AppBuilder"""
	
	# Register sub-capability views
	register_regulatory_views(appbuilder)
	register_serialization_views(appbuilder)
	register_clinical_views(appbuilder)
	register_rd_views(appbuilder)
	register_batch_views(appbuilder)


def create_blueprint() -> Blueprint:
	"""Create Flask blueprint for Pharmaceutical Specific capability"""
	
	pharmaceutical_bp = Blueprint(
		'pharmaceutical_specific',
		__name__,
		url_prefix='/pharmaceutical',
		template_folder='templates',
		static_folder='static'
	)
	
	return pharmaceutical_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register pharmaceutical-specific permissions"""
	
	# High-level pharmaceutical permissions
	base_permissions = [
		# Capability-level permissions
		('can_access', 'PharmaceuticalDashboardView'),
		('can_manage_regulatory', 'PharmaceuticalCapability'),
		('can_manage_clinical', 'PharmaceuticalCapability'),
		('can_manage_serialization', 'PharmaceuticalCapability'),
		('can_manage_rd', 'PharmaceuticalCapability'),
		('can_manage_batch_release', 'PharmaceuticalCapability'),
		
		# Compliance permissions
		('can_approve_submissions', 'PharmaceuticalCapability'),
		('can_electronic_sign', 'PharmaceuticalCapability'),
		('can_override_controls', 'PharmaceuticalCapability'),
		('can_audit_access', 'PharmaceuticalCapability'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in base_permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure():
	"""Get menu structure for Pharmaceutical Specific capability"""
	
	return {
		'name': 'Pharmaceutical',
		'icon': 'fa-pills',
		'items': [
			{
				'name': 'Pharmaceutical Dashboard',
				'href': '/pharmaceutical/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_access on PharmaceuticalDashboardView'
			},
			{
				'name': 'Regulatory Compliance',
				'icon': 'fa-shield-alt',
				'items': [
					{
						'name': 'Regulatory Submissions',
						'href': '/pharmaceutical/regulatory/submissions/',
						'icon': 'fa-file-medical-alt',
						'permission': 'ph.regulatory.read'
					},
					{
						'name': 'Compliance Audits',
						'href': '/pharmaceutical/regulatory/audits/',
						'icon': 'fa-search',
						'permission': 'ph.regulatory.read'
					},
					{
						'name': 'Regulatory Reporting',
						'href': '/pharmaceutical/regulatory/reports/',
						'icon': 'fa-chart-line',
						'permission': 'ph.regulatory.read'
					}
				]
			},
			{
				'name': 'Product Serialization',
				'icon': 'fa-barcode',
				'items': [
					{
						'name': 'Serial Number Management',
						'href': '/pharmaceutical/serialization/serials/',
						'icon': 'fa-hashtag',
						'permission': 'ph.serialization.read'
					},
					{
						'name': 'Aggregation Management',
						'href': '/pharmaceutical/serialization/aggregation/',
						'icon': 'fa-boxes',
						'permission': 'ph.serialization.read'
					},
					{
						'name': 'Track & Trace',
						'href': '/pharmaceutical/serialization/tracking/',
						'icon': 'fa-route',
						'permission': 'ph.serialization.read'
					}
				]
			},
			{
				'name': 'Clinical Trials',
				'icon': 'fa-flask',
				'items': [
					{
						'name': 'Protocol Management',
						'href': '/pharmaceutical/clinical/protocols/',
						'icon': 'fa-file-medical',
						'permission': 'ph.clinical.read'
					},
					{
						'name': 'Patient Management',
						'href': '/pharmaceutical/clinical/patients/',
						'icon': 'fa-users',
						'permission': 'ph.clinical.read'
					},
					{
						'name': 'Data Collection',
						'href': '/pharmaceutical/clinical/data/',
						'icon': 'fa-database',
						'permission': 'ph.clinical.read'
					},
					{
						'name': 'Adverse Events',
						'href': '/pharmaceutical/clinical/adverse-events/',
						'icon': 'fa-exclamation-triangle',
						'permission': 'ph.clinical.read'
					}
				]
			},
			{
				'name': 'R&D Management',
				'icon': 'fa-microscope',
				'items': [
					{
						'name': 'Research Projects',
						'href': '/pharmaceutical/rd/projects/',
						'icon': 'fa-project-diagram',
						'permission': 'ph.rd.read'
					},
					{
						'name': 'Formulations',
						'href': '/pharmaceutical/rd/formulations/',
						'icon': 'fa-vial',
						'permission': 'ph.rd.read'
					},
					{
						'name': 'IP Management',
						'href': '/pharmaceutical/rd/intellectual-property/',
						'icon': 'fa-copyright',
						'permission': 'ph.rd.read'
					},
					{
						'name': 'Lab Data',
						'href': '/pharmaceutical/rd/lab-data/',
						'icon': 'fa-chart-area',
						'permission': 'ph.rd.read'
					}
				]
			},
			{
				'name': 'Batch Release',
				'icon': 'fa-check-circle',
				'items': [
					{
						'name': 'Batch Records',
						'href': '/pharmaceutical/batch/records/',
						'icon': 'fa-clipboard-list',
						'permission': 'ph.batch.read'
					},
					{
						'name': 'Quality Testing',
						'href': '/pharmaceutical/batch/testing/',
						'icon': 'fa-vials',
						'permission': 'ph.batch.read'
					},
					{
						'name': 'Release Approval',
						'href': '/pharmaceutical/batch/approval/',
						'icon': 'fa-stamp',
						'permission': 'ph.batch.approve'
					},
					{
						'name': 'Certificate of Analysis',
						'href': '/pharmaceutical/batch/coa/',
						'icon': 'fa-certificate',
						'permission': 'ph.batch.read'
					}
				]
			}
		]
	}


def init_capability(appbuilder: AppBuilder):
	"""Initialize Pharmaceutical Specific capability"""
	
	# Register views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Initialize default data if needed
	_init_default_data(appbuilder)


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default pharmaceutical data if needed"""
	
	try:
		# Initialize regulatory frameworks
		_init_regulatory_frameworks()
		
		# Initialize compliance controls
		_init_compliance_controls()
		
		# Initialize serialization standards
		_init_serialization_standards()
		
		print("Default pharmaceutical data initialized")
		
	except Exception as e:
		print(f"Error initializing pharmaceutical data: {e}")


def _init_regulatory_frameworks():
	"""Initialize regulatory frameworks"""
	
	from .regulatory_compliance.models import PHRCRegulatoryFramework
	from ...auth_rbac.models import db
	
	frameworks = [
		{
			'framework_code': 'FDA',
			'framework_name': 'Food and Drug Administration',
			'region': 'United States',
			'description': 'US FDA regulations for pharmaceutical products',
			'is_active': True
		},
		{
			'framework_code': 'EMA', 
			'framework_name': 'European Medicines Agency',
			'region': 'European Union',
			'description': 'EU EMA regulations for medicinal products',
			'is_active': True
		},
		{
			'framework_code': 'GMP',
			'framework_name': 'Good Manufacturing Practice',
			'region': 'Global',
			'description': 'International GMP standards',
			'is_active': True
		},
		{
			'framework_code': 'GCP',
			'framework_name': 'Good Clinical Practice',
			'region': 'Global', 
			'description': 'International GCP standards for clinical trials',
			'is_active': True
		}
	]
	
	for framework_data in frameworks:
		existing = PHRCRegulatoryFramework.query.filter_by(
			framework_code=framework_data['framework_code']
		).first()
		
		if not existing:
			framework = PHRCRegulatoryFramework(
				tenant_id='default_tenant',
				**framework_data
			)
			db.session.add(framework)
	
	db.session.commit()


def _init_compliance_controls():
	"""Initialize compliance controls"""
	
	from .regulatory_compliance.models import PHRCComplianceControl
	from ...auth_rbac.models import db
	
	controls = [
		{
			'control_code': 'PH-001',
			'control_name': 'Electronic Signature Validation',
			'description': 'Validates electronic signatures per 21 CFR Part 11',
			'control_type': 'Automated',
			'severity': 'Critical',
			'is_active': True
		},
		{
			'control_code': 'PH-002',
			'control_name': 'Audit Trail Completeness', 
			'description': 'Ensures complete audit trails for all regulatory data',
			'control_type': 'Automated',
			'severity': 'Critical',
			'is_active': True
		},
		{
			'control_code': 'PH-003',
			'control_name': 'Data Integrity Check',
			'description': 'Validates data integrity using ALCOA+ principles',
			'control_type': 'Automated',
			'severity': 'High',
			'is_active': True
		}
	]
	
	for control_data in controls:
		existing = PHRCComplianceControl.query.filter_by(
			control_code=control_data['control_code']
		).first()
		
		if not existing:
			control = PHRCComplianceControl(
				tenant_id='default_tenant',
				**control_data
			)
			db.session.add(control)
	
	db.session.commit()


def _init_serialization_standards():
	"""Initialize serialization standards""" 
	
	from .product_serialization_tracking.models import PHPSSerializationStandard
	from ...auth_rbac.models import db
	
	standards = [
		{
			'standard_code': 'GS1',
			'standard_name': 'GS1 Global Standards',
			'description': 'GS1 standards for product identification and data exchange',
			'version': '2024.1',
			'is_active': True
		},
		{
			'standard_code': 'FDA_UDI',
			'standard_name': 'FDA Unique Device Identification',
			'description': 'FDA UDI requirements for medical devices',
			'version': '2024.1', 
			'is_active': True
		},
		{
			'standard_code': 'EU_FMD',
			'standard_name': 'EU Falsified Medicines Directive',
			'description': 'EU FMD serialization requirements',
			'version': '2024.1',
			'is_active': True
		}
	]
	
	for standard_data in standards:
		existing = PHPSSerializationStandard.query.filter_by(
			standard_code=standard_data['standard_code']
		).first()
		
		if not existing:
			standard = PHPSSerializationStandard(
				tenant_id='default_tenant',
				**standard_data
			)
			db.session.add(standard)
	
	db.session.commit()