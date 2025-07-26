"""
Procurement & Purchasing Capability

Enterprise-grade procurement and purchasing management capability that handles
the entire requisition-to-pay process including requisitioning, purchase order
management, vendor management, sourcing & supplier selection, and contract management.

Sub-capabilities:
- Requisitioning: Employee requisition management and approval workflows
- Purchase Order Management: PO creation, tracking, receiving, and three-way matching
- Vendor Management: Vendor master data, performance tracking, and relationships
- Sourcing & Supplier Selection: Supplier evaluation, RFQ/RFP processes, and selection
- Contract Management: Contract lifecycle, terms, compliance, and renewals

Integration with Core Financials for seamless AP processing and financial reporting.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

# Capability Metadata
CAPABILITY_INFO = {
	'code': 'PP',
	'name': 'Procurement & Purchasing',
	'description': 'Complete procurement and purchasing management with requisition-to-pay processes',
	'version': '1.0.0',
	'category': 'ERP Core',
	'status': 'Active',
	'dependencies': [
		'core_financials.accounts_payable',  # For invoice processing
		'core_financials.general_ledger',    # For GL integration
		'auth_rbac'                          # For access control
	],
	'subcapabilities': [
		{
			'code': 'PPR',
			'name': 'Requisitioning',
			'description': 'Employee requisition management and approval workflows'
		},
		{
			'code': 'PPO',
			'name': 'Purchase Order Management', 
			'description': 'PO creation, tracking, receiving, and three-way matching'
		},
		{
			'code': 'PPV',
			'name': 'Vendor Management',
			'description': 'Vendor master data, performance tracking, and relationships'
		},
		{
			'code': 'PPS',
			'name': 'Sourcing & Supplier Selection',
			'description': 'Supplier evaluation, RFQ/RFP processes, and selection'
		},
		{
			'code': 'PPC',
			'name': 'Contract Management',
			'description': 'Contract lifecycle, terms, compliance, and renewals'
		}
	],
	'models': [
		'PPRRequisition', 'PPRRequisitionLine', 'PPRApprovalWorkflow',
		'PPOPurchaseOrder', 'PPOPurchaseOrderLine', 'PPOReceipt', 'PPOReceiptLine',
		'PPVVendor', 'PPVVendorContact', 'PPVVendorPerformance', 'PPVVendorCategory',
		'PPSRFQHeader', 'PPSRFQLine', 'PPSBid', 'PPSBidLine', 'PPSSupplierEvaluation',
		'PPCContract', 'PPCContractLine', 'PPCContractAmendment', 'PPCContractRenewal'
	],
	'integrations': {
		'accounts_payable': 'Automatic invoice creation from receipts',
		'general_ledger': 'Budget checking and encumbrance accounting',
		'inventory_management': 'Stock level monitoring and replenishment',
		'project_management': 'Project-based procurement tracking'
	}
}


def get_capability_info() -> Dict[str, Any]:
	"""Get Procurement & Purchasing capability information"""
	return CAPABILITY_INFO.copy()


def get_subcapabilities() -> List[str]:
	"""Get list of available sub-capabilities"""
	return [sub['code'].lower() for sub in CAPABILITY_INFO['subcapabilities']]


def validate_composition(subcapabilities: List[str]) -> Dict[str, Any]:
	"""
	Validate sub-capability composition for business logic dependencies.
	
	Args:
		subcapabilities: List of sub-capability codes to validate
		
	Returns:
		Dictionary with validation results including:
		- valid: Boolean indicating if composition is valid
		- errors: List of error messages for invalid compositions
		- warnings: List of warning messages for suboptimal compositions
		- recommendations: List of recommended additional sub-capabilities
	"""
	
	if not subcapabilities:
		return {
			'valid': False,
			'errors': ['At least one sub-capability must be specified'],
			'warnings': [],
			'recommendations': ['Start with requisitioning and vendor_management']
		}
	
	available = get_subcapabilities()
	invalid = [sub for sub in subcapabilities if sub not in available]
	
	if invalid:
		return {
			'valid': False,
			'errors': [f'Invalid sub-capabilities: {invalid}. Available: {available}'],
			'warnings': [],
			'recommendations': []
		}
	
	errors = []
	warnings = []
	recommendations = []
	
	# Business logic validation rules
	if 'purchase_order_management' in subcapabilities:
		if 'vendor_management' not in subcapabilities:
			errors.append('Purchase Order Management requires Vendor Management for vendor master data')
		if 'requisitioning' not in subcapabilities:
			warnings.append('Purchase Order Management works best with Requisitioning for demand management')
	
	if 'sourcing_supplier_selection' in subcapabilities:
		if 'vendor_management' not in subcapabilities:
			errors.append('Sourcing & Supplier Selection requires Vendor Management for supplier data')
		if 'purchase_order_management' not in subcapabilities:
			warnings.append('Sourcing & Supplier Selection should include Purchase Order Management for complete process')
	
	if 'contract_management' in subcapabilities:
		if 'vendor_management' not in subcapabilities:
			errors.append('Contract Management requires Vendor Management for vendor relationships')
		if 'purchase_order_management' not in subcapabilities:
			warnings.append('Contract Management works best with Purchase Order Management for contract compliance')
	
	# Recommendations for optimal functionality
	if 'requisitioning' in subcapabilities and 'purchase_order_management' not in subcapabilities:
		recommendations.append('Add Purchase Order Management for complete req-to-PO workflow')
	
	if 'vendor_management' in subcapabilities and 'contract_management' not in subcapabilities:
		recommendations.append('Add Contract Management for comprehensive vendor relationship management')
	
	# Check for complete procurement suite
	complete_suite = {'requisitioning', 'purchase_order_management', 'vendor_management', 'sourcing_supplier_selection', 'contract_management'}
	if set(subcapabilities) == complete_suite:
		recommendations.append('Complete procurement suite - excellent for comprehensive procurement operations')
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings,
		'recommendations': recommendations
	}


def get_integration_requirements(subcapabilities: List[str]) -> Dict[str, List[str]]:
	"""
	Get external integration requirements for sub-capability composition.
	
	Args:
		subcapabilities: List of sub-capability codes
		
	Returns:
		Dictionary mapping integration types to required external capabilities
	"""
	
	requirements = {
		'required': [],
		'recommended': [],
		'optional': []
	}
	
	# Core Financials integration requirements
	if any(sub in subcapabilities for sub in ['purchase_order_management', 'requisitioning']):
		requirements['required'].extend([
			'core_financials.accounts_payable',
			'core_financials.general_ledger'
		])
	
	if 'contract_management' in subcapabilities:
		requirements['recommended'].append('core_financials.budgeting_forecasting')
	
	# Other ERP module integrations
	if 'requisitioning' in subcapabilities:
		requirements['recommended'].extend([
			'inventory_management',
			'project_management'
		])
	
	if 'vendor_management' in subcapabilities:
		requirements['optional'].extend([
			'quality_management',
			'logistics_management'
		])
	
	return requirements


def get_business_metrics() -> Dict[str, Any]:
	"""Get key business metrics tracked by Procurement & Purchasing capability"""
	
	return {
		'operational_metrics': [
			'Total Purchase Volume',
			'Number of Active Vendors',
			'Average PO Processing Time',
			'Requisition Approval Cycle Time',
			'Contract Compliance Rate',
			'Three-Way Match Exception Rate'
		],
		'financial_metrics': [
			'Cost Savings from Negotiations',
			'Spend Under Management',
			'Contract Utilization Rate',
			'Budget vs Actual Spending',
			'Payment Terms Optimization',
			'Early Payment Discount Capture'
		],
		'quality_metrics': [
			'Vendor Performance Score',
			'On-Time Delivery Rate',
			'Quality Rejection Rate',
			'Vendor Diversity Index',
			'Supplier Risk Score',
			'Contract Renewal Success Rate'
		],
		'compliance_metrics': [
			'Policy Compliance Rate',
			'Audit Findings Resolution',
			'Regulatory Compliance Score',
			'Contract Amendment Frequency',
			'Emergency Purchase Rate',
			'Segregation of Duties Violations'
		]
	}


def get_dashboard_widgets() -> List[Dict[str, Any]]:
	"""Get default dashboard widgets for Procurement & Purchasing capability"""
	
	return [
		{
			'name': 'Procurement Summary',
			'type': 'summary_cards',
			'metrics': ['total_spend_ytd', 'active_pos', 'pending_requisitions', 'active_vendors'],
			'size': 'full_width'
		},
		{
			'name': 'Purchase Order Status',
			'type': 'donut_chart',
			'data_source': 'po_status_distribution',
			'size': 'half_width'
		},
		{
			'name': 'Vendor Performance',
			'type': 'bar_chart',
			'data_source': 'top_vendors_by_performance',
			'size': 'half_width'
		},
		{
			'name': 'Spend Analysis',
			'type': 'line_chart',
			'data_source': 'monthly_spend_trend',
			'size': 'full_width'
		},
		{
			'name': 'Contract Expiration Alert',
			'type': 'table',
			'data_source': 'contracts_expiring_soon',
			'size': 'half_width'
		},
		{
			'name': 'Approval Workflow',
			'type': 'workflow_status',
			'data_source': 'pending_approvals',
			'size': 'half_width'
		}
	]


def get_workflow_definitions() -> Dict[str, Dict[str, Any]]:
	"""Get standard workflow definitions for procurement processes"""
	
	return {
		'requisition_approval': {
			'name': 'Requisition Approval Workflow',
			'description': 'Standard approval workflow for purchase requisitions',
			'steps': [
				{'name': 'Manager Approval', 'role': 'manager', 'condition': 'amount > 1000'},
				{'name': 'Department Head Approval', 'role': 'dept_head', 'condition': 'amount > 5000'},
				{'name': 'Finance Approval', 'role': 'finance', 'condition': 'amount > 25000'},
				{'name': 'Executive Approval', 'role': 'executive', 'condition': 'amount > 100000'}
			],
			'parallel_approval': False,
			'escalation_hours': 48
		},
		'po_approval': {
			'name': 'Purchase Order Approval Workflow', 
			'description': 'Standard approval workflow for purchase orders',
			'steps': [
				{'name': 'Procurement Review', 'role': 'procurement', 'condition': 'always'},
				{'name': 'Manager Approval', 'role': 'manager', 'condition': 'amount > 5000'},
				{'name': 'Finance Approval', 'role': 'finance', 'condition': 'amount > 50000'},
				{'name': 'Executive Approval', 'role': 'executive', 'condition': 'amount > 250000'}
			],
			'parallel_approval': False,
			'escalation_hours': 24
		},
		'contract_approval': {
			'name': 'Contract Approval Workflow',
			'description': 'Standard approval workflow for vendor contracts',
			'steps': [
				{'name': 'Legal Review', 'role': 'legal', 'condition': 'value > 10000'},
				{'name': 'Procurement Approval', 'role': 'procurement_manager', 'condition': 'always'},
				{'name': 'Finance Approval', 'role': 'finance_director', 'condition': 'value > 100000'},
				{'name': 'Executive Approval', 'role': 'executive', 'condition': 'value > 500000'}
			],
			'parallel_approval': True,
			'escalation_hours': 72
		}
	}


# Import all models for capability registration
def get_all_models():
	"""Import and return all models from sub-capabilities"""
	models = []
	
	try:
		from .requisitioning.models import *
		models.extend([
			PPRRequisition, PPRRequisitionLine, PPRApprovalWorkflow, PPRRequisitionComment
		])
	except ImportError:
		pass
	
	try:
		from .purchase_order_management.models import *
		models.extend([
			PPOPurchaseOrder, PPOPurchaseOrderLine, PPOReceipt, PPOReceiptLine,
			PPOThreeWayMatch, PPOChangeOrder
		])
	except ImportError:
		pass
	
	try:
		from .vendor_management.models import *
		models.extend([
			PPVVendor, PPVVendorContact, PPVVendorPerformance, PPVVendorCategory,
			PPVVendorQualification, PPVVendorInsurance
		])
	except ImportError:
		pass
	
	try:
		from .sourcing_supplier_selection.models import *
		models.extend([
			PPSRFQHeader, PPSRFQLine, PPSBid, PPSBidLine, PPSSupplierEvaluation,
			PPSEvaluationCriteria, PPSAwardRecommendation
		])
	except ImportError:
		pass
	
	try:
		from .contract_management.models import *
		models.extend([
			PPCContract, PPCContractLine, PPCContractAmendment, PPCContractRenewal,
			PPCContractMilestone, PPCContractDocument
		])
	except ImportError:
		pass
	
	return models


# Export key classes and functions
__all__ = [
	'get_capability_info',
	'get_subcapabilities', 
	'validate_composition',
	'get_integration_requirements',
	'get_business_metrics',
	'get_dashboard_widgets',
	'get_workflow_definitions',
	'get_all_models',
	'CAPABILITY_INFO'
]