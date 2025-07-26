"""
Fixed Asset Management Blueprint

Flask blueprint registration for Fixed Asset Management sub-capability.
Registers all views, API endpoints, and URL routes for FAM functionality.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	FAMAssetModelView, FAMAssetCategoryModelView, FAMDepreciationMethodModelView,
	FAMAssetAcquisitionModelView, FAMAssetDisposalModelView, FAMAssetTransferModelView,
	FAMAssetMaintenanceModelView, FAMAssetInsuranceModelView, FAMAssetValuationModelView,
	FAMAssetLeaseModelView, FAMDepreciationReportView, FAMDashboardView
)
from .api import create_api_blueprint


def register_views(appbuilder: AppBuilder):
	"""Register Fixed Asset Management views with Flask-AppBuilder"""
	
	# Dashboard
	appbuilder.add_view_no_menu(FAMDashboardView())
	appbuilder.add_link(
		"FAM Dashboard",
		href="/fam/dashboard/",
		icon="fa-dashboard",
		category="Fixed Assets",
		category_icon="fa-building"
	)
	
	# Asset Management
	appbuilder.add_view(
		FAMAssetModelView,
		"Fixed Assets",
		icon="fa-building",
		category="Fixed Assets"
	)
	
	appbuilder.add_view(
		FAMAssetCategoryModelView,
		"Asset Categories",
		icon="fa-tags",
		category="Fixed Assets"
	)
	
	appbuilder.add_view(
		FAMDepreciationMethodModelView,
		"Depreciation Methods",
		icon="fa-calculator",
		category="Fixed Assets"
	)
	
	# Asset Lifecycle
	appbuilder.add_view(
		FAMAssetAcquisitionModelView,
		"Asset Acquisitions",
		icon="fa-plus-circle",
		category="Fixed Assets"
	)
	
	appbuilder.add_view(
		FAMAssetDisposalModelView,
		"Asset Disposals",
		icon="fa-minus-circle",
		category="Fixed Assets"
	)
	
	appbuilder.add_view(
		FAMAssetTransferModelView,
		"Asset Transfers",
		icon="fa-exchange-alt",
		category="Fixed Assets"
	)
	
	# Asset Services
	appbuilder.add_view(
		FAMAssetMaintenanceModelView,
		"Maintenance Records",
		icon="fa-wrench",
		category="Fixed Assets"
	)
	
	appbuilder.add_view(
		FAMAssetInsuranceModelView,
		"Insurance Tracking",
		icon="fa-shield-alt",
		category="Fixed Assets"
	)
	
	appbuilder.add_view(
		FAMAssetValuationModelView,
		"Asset Valuations",
		icon="fa-chart-line",
		category="Fixed Assets"
	)
	
	appbuilder.add_view(
		FAMAssetLeaseModelView,
		"Lease Assets",
		icon="fa-file-contract",
		category="Fixed Assets"
	)
	
	# Reports
	appbuilder.add_view_no_menu(FAMDepreciationReportView())
	appbuilder.add_link(
		"Depreciation Report",
		href="/fam/depreciation_report/",
		icon="fa-chart-bar",
		category="Fixed Assets"
	)


def register_api_blueprint(app):
	"""Register API blueprint with Flask app"""
	api_bp = create_api_blueprint()
	app.register_blueprint(api_bp)


def create_blueprint() -> Blueprint:
	"""Create Flask blueprint for Fixed Asset Management"""
	
	fam_bp = Blueprint(
		'fixed_asset_management',
		__name__,
		url_prefix='/fam',
		template_folder='templates',
		static_folder='static'
	)
	
	return fam_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register Fixed Asset Management permissions"""
	
	permissions = [
		# Asset permissions
		('can_list', 'FAMAssetModelView'),
		('can_show', 'FAMAssetModelView'),
		('can_add', 'FAMAssetModelView'),
		('can_edit', 'FAMAssetModelView'),
		('can_delete', 'FAMAssetModelView'),
		('can_transfer_asset', 'FAMAssetModelView'),
		('can_schedule_maintenance', 'FAMAssetModelView'),
		('can_dispose_asset', 'FAMAssetModelView'),
		('can_depreciation_history', 'FAMAssetModelView'),
		
		# Category permissions
		('can_list', 'FAMAssetCategoryModelView'),
		('can_show', 'FAMAssetCategoryModelView'),
		('can_add', 'FAMAssetCategoryModelView'),
		('can_edit', 'FAMAssetCategoryModelView'),
		('can_delete', 'FAMAssetCategoryModelView'),
		
		# Depreciation Method permissions
		('can_list', 'FAMDepreciationMethodModelView'),
		('can_show', 'FAMDepreciationMethodModelView'),
		('can_add', 'FAMDepreciationMethodModelView'),
		('can_edit', 'FAMDepreciationMethodModelView'),
		('can_delete', 'FAMDepreciationMethodModelView'),
		
		# Acquisition permissions
		('can_list', 'FAMAssetAcquisitionModelView'),
		('can_show', 'FAMAssetAcquisitionModelView'),
		('can_add', 'FAMAssetAcquisitionModelView'),
		('can_edit', 'FAMAssetAcquisitionModelView'),
		('can_delete', 'FAMAssetAcquisitionModelView'),
		
		# Disposal permissions
		('can_list', 'FAMAssetDisposalModelView'),
		('can_show', 'FAMAssetDisposalModelView'),
		('can_add', 'FAMAssetDisposalModelView'),
		('can_edit', 'FAMAssetDisposalModelView'),
		('can_delete', 'FAMAssetDisposalModelView'),
		
		# Transfer permissions
		('can_list', 'FAMAssetTransferModelView'),
		('can_show', 'FAMAssetTransferModelView'),
		('can_add', 'FAMAssetTransferModelView'),
		('can_edit', 'FAMAssetTransferModelView'),
		('can_delete', 'FAMAssetTransferModelView'),
		
		# Maintenance permissions
		('can_list', 'FAMAssetMaintenanceModelView'),
		('can_show', 'FAMAssetMaintenanceModelView'),
		('can_add', 'FAMAssetMaintenanceModelView'),
		('can_edit', 'FAMAssetMaintenanceModelView'),
		('can_delete', 'FAMAssetMaintenanceModelView'),
		
		# Insurance permissions
		('can_list', 'FAMAssetInsuranceModelView'),
		('can_show', 'FAMAssetInsuranceModelView'),
		('can_add', 'FAMAssetInsuranceModelView'),
		('can_edit', 'FAMAssetInsuranceModelView'),
		('can_delete', 'FAMAssetInsuranceModelView'),
		
		# Valuation permissions
		('can_list', 'FAMAssetValuationModelView'),
		('can_show', 'FAMAssetValuationModelView'),
		('can_add', 'FAMAssetValuationModelView'),
		('can_edit', 'FAMAssetValuationModelView'),
		('can_delete', 'FAMAssetValuationModelView'),
		
		# Lease permissions
		('can_list', 'FAMAssetLeaseModelView'),
		('can_show', 'FAMAssetLeaseModelView'),
		('can_add', 'FAMAssetLeaseModelView'),
		('can_edit', 'FAMAssetLeaseModelView'),
		('can_delete', 'FAMAssetLeaseModelView'),
		
		# Report permissions
		('can_index', 'FAMDepreciationReportView'),
		('can_api_depreciation_schedule', 'FAMDepreciationReportView'),
		('can_calculate_monthly', 'FAMDepreciationReportView'),
		('can_index', 'FAMDashboardView'),
		('can_api_summary', 'FAMDashboardView'),
		('can_api_maintenance_alerts', 'FAMDashboardView'),
		('can_api_insurance_alerts', 'FAMDashboardView'),
		
		# API permissions
		('can_get_list', 'FAMAssetApi'),
		('can_get_asset', 'FAMAssetApi'),
		('can_create_asset', 'FAMAssetApi'),
		('can_update_asset', 'FAMAssetApi'),
		('can_delete_asset', 'FAMAssetApi'),
		('can_transfer_asset', 'FAMAssetApi'),
		('can_get_depreciation_history', 'FAMAssetApi'),
		
		('can_get_list', 'FAMAssetCategoryApi'),
		('can_get_category', 'FAMAssetCategoryApi'),
		('can_create_category', 'FAMAssetCategoryApi'),
		('can_update_category', 'FAMAssetCategoryApi'),
		('can_delete_category', 'FAMAssetCategoryApi'),
		
		('can_get_list', 'FAMDepreciationMethodApi'),
		('can_get_method', 'FAMDepreciationMethodApi'),
		('can_create_method', 'FAMDepreciationMethodApi'),
		('can_update_method', 'FAMDepreciationMethodApi'),
		('can_delete_method', 'FAMDepreciationMethodApi'),
		
		('can_get_list', 'FAMAssetAcquisitionApi'),
		('can_get_acquisition', 'FAMAssetAcquisitionApi'),
		('can_create_acquisition', 'FAMAssetAcquisitionApi'),
		('can_update_acquisition', 'FAMAssetAcquisitionApi'),
		('can_delete_acquisition', 'FAMAssetAcquisitionApi'),
		
		('can_get_list', 'FAMAssetDisposalApi'),
		('can_get_disposal', 'FAMAssetDisposalApi'),
		('can_create_disposal', 'FAMAssetDisposalApi'),
		('can_update_disposal', 'FAMAssetDisposalApi'),
		('can_delete_disposal', 'FAMAssetDisposalApi'),
		
		('can_get_list', 'FAMAssetTransferApi'),
		('can_get_transfer', 'FAMAssetTransferApi'),
		('can_create_transfer', 'FAMAssetTransferApi'),
		('can_update_transfer', 'FAMAssetTransferApi'),
		('can_delete_transfer', 'FAMAssetTransferApi'),
		
		('can_get_list', 'FAMAssetMaintenanceApi'),
		('can_get_maintenance', 'FAMAssetMaintenanceApi'),
		('can_schedule_maintenance', 'FAMAssetMaintenanceApi'),
		('can_complete_maintenance', 'FAMAssetMaintenanceApi'),
		('can_update_maintenance', 'FAMAssetMaintenanceApi'),
		('can_delete_maintenance', 'FAMAssetMaintenanceApi'),
		
		('can_get_list', 'FAMAssetInsuranceApi'),
		('can_get_insurance', 'FAMAssetInsuranceApi'),
		('can_create_insurance', 'FAMAssetInsuranceApi'),
		('can_update_insurance', 'FAMAssetInsuranceApi'),
		('can_delete_insurance', 'FAMAssetInsuranceApi'),
		('can_get_renewals_due', 'FAMAssetInsuranceApi'),
		
		('can_get_list', 'FAMAssetValuationApi'),
		('can_get_valuation', 'FAMAssetValuationApi'),
		('can_create_valuation', 'FAMAssetValuationApi'),
		('can_update_valuation', 'FAMAssetValuationApi'),
		('can_delete_valuation', 'FAMAssetValuationApi'),
		
		('can_get_list', 'FAMAssetLeaseApi'),
		('can_get_lease', 'FAMAssetLeaseApi'),
		('can_create_lease', 'FAMAssetLeaseApi'),
		('can_update_lease', 'FAMAssetLeaseApi'),
		('can_delete_lease', 'FAMAssetLeaseApi'),
		
		('can_calculate_depreciation', 'FAMDepreciationApi'),
		('can_post_depreciation', 'FAMDepreciationApi'),
		('can_get_depreciation_schedule', 'FAMDepreciationApi'),
		('can_get_depreciation_history', 'FAMDepreciationApi'),
		
		('can_get_dashboard_summary', 'FAMDashboardApi'),
		('can_get_asset_summary', 'FAMDashboardApi'),
		('can_get_maintenance_alerts', 'FAMDashboardApi'),
		('can_get_insurance_alerts', 'FAMDashboardApi'),
		('can_get_assets_by_location', 'FAMDashboardApi'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure():
	"""Get menu structure for Fixed Asset Management"""
	
	return {
		'name': 'Fixed Assets',
		'icon': 'fa-building',
		'items': [
			{
				'name': 'FAM Dashboard',
				'href': '/fam/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_index on FAMDashboardView'
			},
			{
				'name': 'Fixed Assets',
				'href': '/famassetmodelview/list/',
				'icon': 'fa-building',
				'permission': 'can_list on FAMAssetModelView'
			},
			{
				'name': 'Asset Categories',
				'href': '/famassetcategorymodelview/list/',
				'icon': 'fa-tags',
				'permission': 'can_list on FAMAssetCategoryModelView'
			},
			{
				'name': 'Depreciation Methods',
				'href': '/famdepreciationmethodmodelview/list/',
				'icon': 'fa-calculator',
				'permission': 'can_list on FAMDepreciationMethodModelView'
			},
			{
				'name': 'Asset Acquisitions',
				'href': '/famassetacquisitionmodelview/list/',
				'icon': 'fa-plus-circle',
				'permission': 'can_list on FAMAssetAcquisitionModelView'
			},
			{
				'name': 'Asset Disposals',
				'href': '/famassetdisposalmodelview/list/',
				'icon': 'fa-minus-circle',
				'permission': 'can_list on FAMAssetDisposalModelView'
			},
			{
				'name': 'Asset Transfers',
				'href': '/famassettransfermodelview/list/',
				'icon': 'fa-exchange-alt',
				'permission': 'can_list on FAMAssetTransferModelView'
			},
			{
				'name': 'Maintenance Records',
				'href': '/famassetmaintenancemodelview/list/',
				'icon': 'fa-wrench',
				'permission': 'can_list on FAMAssetMaintenanceModelView'
			},
			{
				'name': 'Insurance Tracking',
				'href': '/famassetinsurancemodelview/list/',
				'icon': 'fa-shield-alt',
				'permission': 'can_list on FAMAssetInsuranceModelView'
			},
			{
				'name': 'Asset Valuations',
				'href': '/famassetvaluationmodelview/list/',
				'icon': 'fa-chart-line',
				'permission': 'can_list on FAMAssetValuationModelView'
			},
			{
				'name': 'Lease Assets',
				'href': '/famassetleasemodelview/list/',
				'icon': 'fa-file-contract',
				'permission': 'can_list on FAMAssetLeaseModelView'
			},
			{
				'name': 'Depreciation Report',
				'href': '/fam/depreciation_report/',
				'icon': 'fa-chart-bar',
				'permission': 'can_index on FAMDepreciationReportView'
			}
		]
	}


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Fixed Asset Management sub-capability"""
	
	# Register views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Initialize default data if needed
	_init_default_data(appbuilder)


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default FAM data if needed"""
	
	from .models import CFAMAssetCategory, CFAMDepreciationMethod
	from ...auth_rbac.models import db
	from . import get_default_asset_categories, get_default_depreciation_methods
	
	try:
		# Check if asset categories already exist (use a default tenant for now)
		existing_categories = CFAMAssetCategory.query.filter_by(tenant_id='default_tenant').count()
		
		if existing_categories == 0:
			# Create default asset categories
			default_categories = get_default_asset_categories()
			
			for category_data in default_categories:
				category = CFAMAssetCategory(
					tenant_id='default_tenant',
					category_code=category_data['code'],
					category_name=category_data['name'],
					description=category_data['description'],
					default_useful_life_years=category_data['useful_life_years'],
					allow_depreciation=category_data['useful_life_years'] > 0,
					is_active=True
				)
				db.session.add(category)
			
			db.session.commit()
			print("Default FAM asset categories created")
		
		# Check if depreciation methods already exist
		existing_methods = CFAMDepreciationMethod.query.filter_by(tenant_id='default_tenant').count()
		
		if existing_methods == 0:
			# Create default depreciation methods
			default_methods = get_default_depreciation_methods()
			
			for method_data in default_methods:
				method = CFAMDepreciationMethod(
					tenant_id='default_tenant',
					method_code=method_data['code'],
					method_name=method_data['name'],
					description=method_data['description'],
					formula=method_data['formula'],
					is_active=method_data['is_active'],
					is_system=True
				)
				db.session.add(method)
			
			db.session.commit()
			print("Default FAM depreciation methods created")
			
	except Exception as e:
		print(f"Error initializing default FAM data: {e}")
		db.session.rollback()


def create_default_assets(tenant_id: str, appbuilder: AppBuilder):
	"""Create default assets for a tenant"""
	
	from .service import FixedAssetManagementService
	from . import get_default_asset_categories
	
	try:
		fam_service = FixedAssetManagementService(tenant_id)
		
		# Check if assets already exist
		existing_assets = fam_service.get_assets()
		if len(existing_assets) > 0:
			return
		
		# Get asset categories
		categories = CFAMAssetCategory.query.filter_by(tenant_id=tenant_id).all()
		if not categories:
			return  # No categories to work with
		
		# Create sample assets for each category
		building_category = next((c for c in categories if c.category_code == 'BUILDING'), None)
		equipment_category = next((c for c in categories if c.category_code == 'EQUIPMENT'), None)
		vehicle_category = next((c for c in categories if c.category_code == 'VEHICLE'), None)
		computer_category = next((c for c in categories if c.category_code == 'COMPUTER'), None)
		
		sample_assets = []
		
		if building_category:
			sample_assets.append({
				'asset_number': '001001',
				'asset_name': 'Main Office Building',
				'description': 'Corporate headquarters building',
				'category_id': building_category.category_id,
				'acquisition_cost': 500000.00,
				'acquisition_date': date(2020, 1, 15),
				'placed_in_service_date': date(2020, 2, 1),
				'location': 'Corporate HQ',
				'department': 'Corporate',
				'useful_life_years': 39
			})
		
		if equipment_category:
			sample_assets.append({
				'asset_number': '002001',
				'asset_name': 'Production Line A',
				'description': 'Manufacturing production line equipment',
				'category_id': equipment_category.category_id,
				'acquisition_cost': 75000.00,
				'acquisition_date': date(2021, 3, 10),
				'placed_in_service_date': date(2021, 3, 15),
				'location': 'Factory Floor 1',
				'department': 'Manufacturing',
				'manufacturer': 'Industrial Equipment Co',
				'model': 'PL-2000',
				'serial_number': 'IE123456',
				'useful_life_years': 7
			})
		
		if vehicle_category:
			sample_assets.append({
				'asset_number': '003001',
				'asset_name': 'Delivery Truck #1',
				'description': 'Ford Transit delivery truck',
				'category_id': vehicle_category.category_id,
				'acquisition_cost': 35000.00,
				'acquisition_date': date(2022, 6, 1),
				'placed_in_service_date': date(2022, 6, 1),
				'location': 'Fleet Garage',
				'department': 'Logistics',
				'manufacturer': 'Ford',
				'model': 'Transit 350',
				'serial_number': 'VIN123456789',
				'year_manufactured': 2022,
				'useful_life_years': 5
			})
		
		if computer_category:
			sample_assets.append({
				'asset_number': '004001',
				'asset_name': 'Office Computer Bundle',
				'description': '10 Dell desktop computers for office use',
				'category_id': computer_category.category_id,
				'acquisition_cost': 12000.00,
				'acquisition_date': date(2023, 1, 15),
				'placed_in_service_date': date(2023, 1, 20),
				'location': 'Office Floor 2',
				'department': 'Administration',
				'manufacturer': 'Dell',
				'model': 'OptiPlex 7000',
				'useful_life_years': 3
			})
		
		for asset_data in sample_assets:
			fam_service.create_asset(asset_data)
		
		print(f"Default assets created for tenant {tenant_id}")
		
	except Exception as e:
		print(f"Error creating default assets: {e}")


def setup_fam_integration(appbuilder: AppBuilder):
	"""Set up FAM integration with other modules"""
	
	try:
		# Set up GL integration
		from ..general_ledger.models import CFGLAccount
		from ...auth_rbac.models import db
		from . import get_default_gl_account_mappings
		
		# Ensure required GL accounts exist
		gl_mappings = get_default_gl_account_mappings()
		
		for account_type, account_code in gl_mappings.items():
			existing_account = CFGLAccount.query.filter_by(
				tenant_id='default_tenant',
				account_code=account_code
			).first()
			
			if not existing_account:
				print(f"Warning: GL account {account_code} for {account_type} not found")
		
		print("FAM-GL integration check completed")
		
	except Exception as e:
		print(f"Error setting up FAM integration: {e}")


def get_fam_configuration():
	"""Get FAM configuration settings"""
	
	from . import SUBCAPABILITY_META
	
	return SUBCAPABILITY_META['configuration']


def validate_fam_setup(tenant_id: str) -> Dict[str, Any]:
	"""Validate FAM setup for a tenant"""
	
	from .service import FixedAssetManagementService
	from ..general_ledger.models import CFGLAccount
	from . import get_default_gl_account_mappings
	
	validation_results = {
		'valid': True,
		'errors': [],
		'warnings': []
	}
	
	try:
		fam_service = FixedAssetManagementService(tenant_id)
		
		# Check if required GL accounts exist
		gl_mappings = get_default_gl_account_mappings()
		missing_accounts = []
		
		for account_type, account_code in gl_mappings.items():
			account = CFGLAccount.query.filter_by(
				tenant_id=tenant_id,
				account_code=account_code
			).first()
			
			if not account:
				missing_accounts.append(f"{account_type} ({account_code})")
		
		if missing_accounts:
			validation_results['warnings'].append(
				f"Recommended GL accounts not found: {', '.join(missing_accounts)}"
			)
		
		# Check if asset categories exist
		categories = CFAMAssetCategory.query.filter_by(tenant_id=tenant_id).count()
		if categories == 0:
			validation_results['warnings'].append("No asset categories configured")
		
		# Check if depreciation methods exist
		methods = CFAMDepreciationMethod.query.filter_by(tenant_id=tenant_id).count()
		if methods == 0:
			validation_results['warnings'].append("No depreciation methods configured")
		
		# Check if assets exist
		assets = fam_service.get_assets()
		if len(assets) == 0:
			validation_results['warnings'].append("No assets configured")
		
	except Exception as e:
		validation_results['errors'].append(f"Validation error: {str(e)}")
		validation_results['valid'] = False
	
	return validation_results