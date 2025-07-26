"""
Fixed Asset Management Sub-Capability

Manages the complete lifecycle of company fixed assets including acquisition,
depreciation, disposal, maintenance, and reporting. Provides comprehensive 
asset tracking and financial controls for compliance and operational efficiency.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Fixed Asset Management',
	'code': 'FAM',
	'version': '1.0.0',
	'capability': 'core_financials',
	'description': 'Manages the complete lifecycle of company fixed assets including acquisition, depreciation, disposal, maintenance, and reporting.',
	'industry_focus': 'All',
	'dependencies': ['general_ledger'],
	'optional_dependencies': ['accounts_payable', 'cash_management', 'procurement'],
	'database_tables': [
		'cf_fam_asset',
		'cf_fam_asset_category',
		'cf_fam_depreciation_method',
		'cf_fam_depreciation',
		'cf_fam_asset_acquisition',
		'cf_fam_asset_disposal',
		'cf_fam_asset_transfer',
		'cf_fam_asset_maintenance',
		'cf_fam_asset_insurance',
		'cf_fam_asset_valuation',
		'cf_fam_asset_lease'
	],
	'api_endpoints': [
		'/api/core_financials/fam/assets',
		'/api/core_financials/fam/categories',
		'/api/core_financials/fam/depreciation',
		'/api/core_financials/fam/acquisitions',
		'/api/core_financials/fam/disposals',
		'/api/core_financials/fam/transfers',
		'/api/core_financials/fam/maintenance',
		'/api/core_financials/fam/insurance',
		'/api/core_financials/fam/valuations',
		'/api/core_financials/fam/leases',
		'/api/core_financials/fam/reports'
	],
	'views': [
		'FAMAssetModelView',
		'FAMAssetCategoryModelView',
		'FAMDepreciationMethodModelView',
		'FAMAssetAcquisitionModelView',
		'FAMAssetDisposalModelView',
		'FAMAssetTransferModelView',
		'FAMAssetMaintenanceModelView',
		'FAMAssetInsuranceModelView',
		'FAMAssetValuationModelView',
		'FAMAssetLeaseModelView',
		'FAMDepreciationReportView',
		'FAMDashboardView'
	],
	'permissions': [
		'fam.read',
		'fam.write',
		'fam.asset_admin',
		'fam.depreciation_admin',
		'fam.acquisition_approve',
		'fam.disposal_approve',
		'fam.transfer_approve',
		'fam.maintenance_admin',
		'fam.insurance_admin',
		'fam.valuation_admin',
		'fam.lease_admin',
		'fam.reports',
		'fam.admin'
	],
	'menu_items': [
		{
			'name': 'FAM Dashboard',
			'endpoint': 'FAMDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'fam.read'
		},
		{
			'name': 'Fixed Assets',
			'endpoint': 'FAMAssetModelView.list',
			'icon': 'fa-building',
			'permission': 'fam.read'
		},
		{
			'name': 'Asset Categories',
			'endpoint': 'FAMAssetCategoryModelView.list',
			'icon': 'fa-tags',
			'permission': 'fam.read'
		},
		{
			'name': 'Depreciation Methods',
			'endpoint': 'FAMDepreciationMethodModelView.list',
			'icon': 'fa-calculator',
			'permission': 'fam.read'
		},
		{
			'name': 'Asset Acquisitions',
			'endpoint': 'FAMAssetAcquisitionModelView.list',
			'icon': 'fa-plus-circle',
			'permission': 'fam.read'
		},
		{
			'name': 'Asset Disposals',
			'endpoint': 'FAMAssetDisposalModelView.list',
			'icon': 'fa-minus-circle',
			'permission': 'fam.read'
		},
		{
			'name': 'Asset Transfers',
			'endpoint': 'FAMAssetTransferModelView.list',
			'icon': 'fa-exchange-alt',
			'permission': 'fam.read'
		},
		{
			'name': 'Maintenance Records',
			'endpoint': 'FAMAssetMaintenanceModelView.list',
			'icon': 'fa-wrench',
			'permission': 'fam.read'
		},
		{
			'name': 'Insurance Tracking',
			'endpoint': 'FAMAssetInsuranceModelView.list',
			'icon': 'fa-shield-alt',
			'permission': 'fam.read'
		},
		{
			'name': 'Asset Valuations',
			'endpoint': 'FAMAssetValuationModelView.list',
			'icon': 'fa-chart-line',
			'permission': 'fam.read'
		},
		{
			'name': 'Lease Assets',
			'endpoint': 'FAMAssetLeaseModelView.list',
			'icon': 'fa-file-contract',
			'permission': 'fam.read'
		},
		{
			'name': 'Depreciation Report',
			'endpoint': 'FAMDepreciationReportView.index',
			'icon': 'fa-chart-bar',
			'permission': 'fam.reports'
		}
	],
	'configuration': {
		'auto_asset_numbering': True,
		'auto_depreciation_calculation': True,
		'depreciation_posting_frequency': 'monthly',  # monthly, quarterly, yearly
		'require_acquisition_approval': True,
		'require_disposal_approval': True,
		'require_transfer_approval': False,
		'maintenance_scheduling': True,
		'insurance_tracking': True,
		'asset_tagging': True,
		'barcode_scanning': True,
		'photo_attachments': True,
		'document_attachments': True,
		'physical_inventory': True,
		'multi_location_support': True,
		'cost_center_tracking': True,
		'department_tracking': True,
		'project_tracking': True,
		'asset_categories_required': True,
		'depreciation_methods': ['straight_line', 'declining_balance', 'sum_of_years', 'units_of_production'],
		'default_depreciation_method': 'straight_line',
		'minimum_capitalization_amount': 1000.00,
		'currency_support': True,
		'multi_company_support': True,
		'tax_depreciation_tracking': True,
		'lease_accounting': True,  # ASC 842 / IFRS 16 support
		'impairment_testing': True,
		'component_depreciation': True,  # Depreciate components separately
		'group_depreciation': True,     # Group similar assets
		'disposal_gain_loss_tracking': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# Check required dependencies
	if 'general_ledger' not in available_subcapabilities:
		errors.append("General Ledger sub-capability is required for FAM operations")
	
	# Check optional dependencies
	if 'accounts_payable' not in available_subcapabilities:
		warnings.append("Accounts Payable integration not available - manual asset acquisition entry required")
	
	if 'cash_management' not in available_subcapabilities:
		warnings.append("Cash Management integration not available - manual cash transaction recording required")
	
	if 'procurement' not in available_subcapabilities:
		warnings.append("Procurement integration not available - manual purchase order linkage required")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_asset_categories() -> List[Dict[str, Any]]:
	"""Get default asset categories"""
	return [
		{
			'code': 'BUILDING',
			'name': 'Buildings & Improvements',
			'description': 'Real estate, buildings, and structural improvements',
			'useful_life_years': 39,
			'depreciation_method': 'straight_line',
			'gl_asset_account': '1510',
			'gl_depreciation_account': '1515',
			'gl_expense_account': '6510'
		},
		{
			'code': 'EQUIPMENT',
			'name': 'Machinery & Equipment',
			'description': 'Manufacturing equipment, machinery, and tools',
			'useful_life_years': 7,
			'depreciation_method': 'straight_line',
			'gl_asset_account': '1520',
			'gl_depreciation_account': '1525',
			'gl_expense_account': '6520'
		},
		{
			'code': 'VEHICLE',
			'name': 'Vehicles',
			'description': 'Company cars, trucks, and other vehicles',
			'useful_life_years': 5,
			'depreciation_method': 'straight_line',
			'gl_asset_account': '1530',
			'gl_depreciation_account': '1535',
			'gl_expense_account': '6530'
		},
		{
			'code': 'COMPUTER',
			'name': 'Computer Equipment',
			'description': 'Computers, servers, and IT hardware',
			'useful_life_years': 3,
			'depreciation_method': 'straight_line',
			'gl_asset_account': '1540',
			'gl_depreciation_account': '1545',
			'gl_expense_account': '6540'
		},
		{
			'code': 'FURNITURE',
			'name': 'Furniture & Fixtures',
			'description': 'Office furniture, fixtures, and fittings',
			'useful_life_years': 7,
			'depreciation_method': 'straight_line',
			'gl_asset_account': '1550',
			'gl_depreciation_account': '1555',
			'gl_expense_account': '6550'
		},
		{
			'code': 'SOFTWARE',
			'name': 'Software & Licenses',
			'description': 'Software licenses and intangible assets',
			'useful_life_years': 3,
			'depreciation_method': 'straight_line',
			'gl_asset_account': '1560',
			'gl_depreciation_account': '1565',
			'gl_expense_account': '6560'
		},
		{
			'code': 'LAND',
			'name': 'Land',
			'description': 'Land and land rights (non-depreciable)',
			'useful_life_years': 0,
			'depreciation_method': 'none',
			'gl_asset_account': '1505',
			'gl_depreciation_account': None,
			'gl_expense_account': None
		},
		{
			'code': 'LEASEHOLD',
			'name': 'Leasehold Improvements',
			'description': 'Improvements to leased property',
			'useful_life_years': 10,
			'depreciation_method': 'straight_line',
			'gl_asset_account': '1570',
			'gl_depreciation_account': '1575',
			'gl_expense_account': '6570'
		}
	]

def get_default_depreciation_methods() -> List[Dict[str, Any]]:
	"""Get default depreciation methods"""
	return [
		{
			'code': 'SL',
			'name': 'Straight Line',
			'description': 'Equal depreciation over useful life',
			'formula': 'straight_line',
			'is_active': True
		},
		{
			'code': 'DB',
			'name': 'Declining Balance',
			'description': 'Accelerated depreciation method',
			'formula': 'declining_balance',
			'is_active': True
		},
		{
			'code': 'DDB',
			'name': 'Double Declining Balance',
			'description': 'Double declining balance method',
			'formula': 'double_declining_balance',
			'is_active': True
		},
		{
			'code': 'SYD',
			'name': 'Sum of Years Digits',
			'description': 'Sum of years digits method',
			'formula': 'sum_of_years_digits',
			'is_active': True
		},
		{
			'code': 'UNITS',
			'name': 'Units of Production',
			'description': 'Based on usage/production units',
			'formula': 'units_of_production',
			'is_active': True
		},
		{
			'code': 'NONE',
			'name': 'No Depreciation',
			'description': 'Non-depreciable assets (e.g., land)',
			'formula': 'none',
			'is_active': True
		}
	]

def get_default_maintenance_types() -> List[Dict[str, Any]]:
	"""Get default maintenance types"""
	return [
		{
			'code': 'PREVENTIVE',
			'name': 'Preventive Maintenance',
			'description': 'Scheduled preventive maintenance',
			'schedule_type': 'recurring',
			'is_active': True
		},
		{
			'code': 'CORRECTIVE',
			'name': 'Corrective Maintenance',
			'description': 'Repair and corrective maintenance',
			'schedule_type': 'on_demand',
			'is_active': True
		},
		{
			'code': 'EMERGENCY',
			'name': 'Emergency Maintenance',
			'description': 'Emergency repairs and fixes',
			'schedule_type': 'immediate',
			'is_active': True
		},
		{
			'code': 'INSPECTION',
			'name': 'Inspection',
			'description': 'Regular asset inspections',
			'schedule_type': 'recurring',
			'is_active': True
		},
		{
			'code': 'CALIBRATION',
			'name': 'Calibration',
			'description': 'Equipment calibration and testing',
			'schedule_type': 'recurring',
			'is_active': True
		}
	]

def get_default_disposal_reasons() -> List[Dict[str, Any]]:
	"""Get default disposal reasons"""
	return [
		{
			'code': 'SOLD',
			'name': 'Sold',
			'description': 'Asset sold to third party',
			'requires_proceeds': True
		},
		{
			'code': 'SCRAPPED',
			'name': 'Scrapped',
			'description': 'Asset scrapped or destroyed',
			'requires_proceeds': False
		},
		{
			'code': 'TRADED',
			'name': 'Trade-in',
			'description': 'Asset traded for new asset',
			'requires_proceeds': True
		},
		{
			'code': 'DONATED',
			'name': 'Donated',
			'description': 'Asset donated to charity',
			'requires_proceeds': False
		},
		{
			'code': 'STOLEN',
			'name': 'Stolen/Lost',
			'description': 'Asset stolen or lost',
			'requires_proceeds': False
		},
		{
			'code': 'OBSOLETE',
			'name': 'Obsolete',
			'description': 'Asset became obsolete',
			'requires_proceeds': False
		}
	]

def get_default_gl_account_mappings() -> Dict[str, str]:
	"""Get default GL account mappings for FAM transactions"""
	return {
		# Asset accounts
		'buildings': '1510',
		'accumulated_depreciation_buildings': '1515',
		'equipment': '1520',
		'accumulated_depreciation_equipment': '1525',
		'vehicles': '1530',
		'accumulated_depreciation_vehicles': '1535',
		'computers': '1540',
		'accumulated_depreciation_computers': '1545',
		'furniture': '1550',
		'accumulated_depreciation_furniture': '1555',
		'software': '1560',
		'accumulated_depreciation_software': '1565',
		'land': '1505',
		'leasehold_improvements': '1570',
		'accumulated_depreciation_leasehold': '1575',
		
		# Expense accounts
		'depreciation_expense_buildings': '6510',
		'depreciation_expense_equipment': '6520',
		'depreciation_expense_vehicles': '6530',
		'depreciation_expense_computers': '6540',
		'depreciation_expense_furniture': '6550',
		'depreciation_expense_software': '6560',
		'depreciation_expense_leasehold': '6570',
		'maintenance_expense': '6580',
		'insurance_expense': '6590',
		
		# Gain/Loss accounts
		'gain_on_disposal': '4110',  # Other Income
		'loss_on_disposal': '6900',  # Other Expense
		
		# Construction in progress
		'construction_in_progress': '1590',
		
		# Lease accounts (ASC 842)
		'right_of_use_assets': '1580',
		'lease_liability': '2140',
		'lease_expense': '6595'
	}

def get_asset_status_options() -> List[Dict[str, Any]]:
	"""Get asset status options"""
	return [
		{
			'code': 'ACTIVE',
			'name': 'Active',
			'description': 'Asset in active use',
			'depreciate': True
		},
		{
			'code': 'INACTIVE',
			'name': 'Inactive',
			'description': 'Asset not in use but still owned',
			'depreciate': False
		},
		{
			'code': 'MAINTENANCE',
			'name': 'Under Maintenance',
			'description': 'Asset under maintenance',
			'depreciate': True
		},
		{
			'code': 'CONSTRUCTION',
			'name': 'Under Construction',
			'description': 'Asset under construction',
			'depreciate': False
		},
		{
			'code': 'DISPOSED',
			'name': 'Disposed',
			'description': 'Asset disposed or sold',
			'depreciate': False
		},
		{
			'code': 'RETIRED',
			'name': 'Retired',
			'description': 'Asset retired from service',
			'depreciate': False
		}
	]

def get_lease_types() -> List[Dict[str, Any]]:
	"""Get lease types for ASC 842 compliance"""
	return [
		{
			'code': 'FINANCE',
			'name': 'Finance Lease',
			'description': 'Finance lease (formerly capital lease)',
			'accounting_treatment': 'finance'
		},
		{
			'code': 'OPERATING',
			'name': 'Operating Lease',
			'description': 'Operating lease',
			'accounting_treatment': 'operating'
		}
	]