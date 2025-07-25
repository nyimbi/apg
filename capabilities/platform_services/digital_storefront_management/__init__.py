"""
Digital Storefront Management Sub-Capability

Manages the front-end user experience, product display, branding, and content
of an e-commerce site or marketplace storefront.
"""

SUBCAPABILITY_META = {
	'name': 'Digital Storefront Management',
	'code': 'PSD',
	'description': 'Front-end storefront experience, branding, content, and display management',
	'models': [
		'PSStorefront',
		'PSStorefrontTheme',
		'PSStorefrontPage',
		'PSStorefrontWidget',
		'PSStorefrontNavigation',
		'PSStorefrontBanner',
		'PSStorefrontLayout',
		'PSStorefrontSEO'
	]
}