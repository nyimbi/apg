"""
Order Processing Sub-capability

Manages the workflow from order receipt through fulfillment to invoicing
including inventory allocation, picking, packing, and shipping.
"""

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Order Processing',
	'code': 'SOP',
	'version': '1.0.0',
	'description': 'Order workflow management from receipt to fulfillment',
	'parent_capability': 'sales_order_management',
	'industry_focus': 'All'
}