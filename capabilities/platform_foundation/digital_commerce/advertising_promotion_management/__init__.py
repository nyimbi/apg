"""
Advertising & Promotion Management Sub-Capability

Manages internal platform advertising (e.g., sponsored listings)
and promotional campaigns.
"""

SUBCAPABILITY_META = {
	'name': 'Advertising & Promotion Management',
	'code': 'PSA',
	'description': 'Platform advertising, promotions, and sponsored content management',
	'models': [
		'PSAdvertisement',
		'PSPromotion',
		'PSCoupon',
		'PSCampaign',
		'PSAdPlacement',
		'PSAdAnalytics',
		'PSPromotionRule',
		'PSLoyaltyProgram'
	]
}