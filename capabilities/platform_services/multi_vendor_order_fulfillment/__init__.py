"""
Multi-Vendor Order Fulfillment Sub-Capability

Coordinates and tracks orders that involve multiple sellers/vendors
and diverse fulfillment processes.
"""

SUBCAPABILITY_META = {
	'name': 'Multi-Vendor Order Fulfillment',
	'code': 'PSO',
	'description': 'Complex order fulfillment coordination across multiple vendors',
	'models': [
		'PSOrder',
		'PSOrderItem',
		'PSOrderVendor',
		'PSOrderShipment',
		'PSOrderTracking',
		'PSOrderSplit',
		'PSOrderFulfillment',
		'PSOrderReturn',
		'PSOrderStatus',
		'PSOrderCommunication'
	]
}