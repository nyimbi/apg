"""
Seller/Vendor Management Sub-Capability

Onboarding, management, and performance tracking of third-party
sellers/vendors on a marketplace platform.
"""

SUBCAPABILITY_META = {
	'name': 'Seller/Vendor Management',
	'code': 'PSV',
	'description': 'Multi-vendor marketplace seller onboarding and management',
	'models': [
		'PSVendor',
		'PSVendorProfile',
		'PSVendorVerification',
		'PSVendorContract',
		'PSVendorPerformance',
		'PSVendorPayout',
		'PSVendorProduct',
		'PSVendorStore',
		'PSVendorDocument',
		'PSVendorCommunication'
	]
}