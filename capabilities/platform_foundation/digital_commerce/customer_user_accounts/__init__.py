"""
Customer User Accounts Sub-Capability

Manages customer profiles, purchase history, preferences, and communication
for personalized e-commerce experiences.
"""

SUBCAPABILITY_META = {
	'name': 'Customer User Accounts',
	'code': 'PSC',
	'description': 'Customer profile management with personalization and purchase history',
	'models': [
		'PSCustomer',
		'PSCustomerAddress',
		'PSCustomerPreference',
		'PSCustomerGroup',
		'PSCustomerWishlist',
		'PSCustomerReview',
		'PSCustomerCommunication',
		'PSCustomerSession',
		'PSCustomerLoyalty',
		'PSCustomerSupport'
	]
}