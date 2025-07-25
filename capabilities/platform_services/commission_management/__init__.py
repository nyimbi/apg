"""
Commission Management Sub-Capability

Calculates and tracks commissions for marketplace transactions,
ensuring accurate payouts to the platform and sellers.
"""

SUBCAPABILITY_META = {
	'name': 'Commission Management',
	'code': 'PSM',
	'description': 'Commission calculation and payout management for marketplace transactions',
	'models': [
		'PSCommissionRule',
		'PSCommissionTransaction',
		'PSCommissionPayout',
		'PSCommissionAdjustment',
		'PSCommissionReport',
		'PSCommissionTier',
		'PSCommissionSchedule',
		'PSCommissionFee'
	]
}