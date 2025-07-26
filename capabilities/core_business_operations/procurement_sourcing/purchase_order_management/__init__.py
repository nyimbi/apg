"""
Purchase Order Management Sub-capability

Comprehensive purchase order management with creation, tracking, receiving,
and three-way matching functionality. Integrates with Vendor Management
and Core Financials for complete procurement-to-pay processes.

Key Features:
- Purchase order creation and management
- Automated PO generation from requisitions
- Goods receipt processing and matching
- Three-way matching (PO, Receipt, Invoice)
- Change order management
- Vendor performance tracking
- Integration with AP for invoice processing
"""

from typing import Dict, Any

# Sub-capability metadata
SUBCAPABILITY_INFO = {
	'code': 'PPO',
	'name': 'Purchase Order Management',
	'description': 'PO creation, tracking, receiving, and three-way matching',
	'version': '1.0.0',
	'models': [
		'PPOPurchaseOrder',
		'PPOPurchaseOrderLine', 
		'PPOReceipt',
		'PPOReceiptLine',
		'PPOThreeWayMatch',
		'PPOChangeOrder'
	],
	'views': [
		'PurchaseOrderView',
		'PurchaseOrderLineView',
		'ReceiptView',
		'ReceiptLineView',
		'ThreeWayMatchView',
		'ChangeOrderView',
		'PurchaseOrderDashboardView'
	],
	'permissions': [
		'can_create_po',
		'can_edit_po',
		'can_approve_po',
		'can_receive_goods',
		'can_process_three_way_match',
		'can_create_change_order',
		'can_view_all_pos',
		'can_override_matching'
	]
}


def get_subcapability_info() -> Dict[str, Any]:
	"""Get Purchase Order Management sub-capability information"""
	return SUBCAPABILITY_INFO.copy()