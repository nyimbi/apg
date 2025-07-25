"""
Payment Gateway Integration Sub-Capability

Connects to various payment processors to handle secure online transactions
and multiple payment methods.
"""

SUBCAPABILITY_META = {
	'name': 'Payment Gateway Integration',
	'code': 'PSG',
	'description': 'Secure payment processing with multiple gateway support',
	'models': [
		'PSPaymentGateway',
		'PSPaymentMethod',
		'PSPaymentTransaction',
		'PSPaymentRefund',
		'PSPaymentSubscription',
		'PSPaymentWallet',
		'PSPaymentCard',
		'PSPaymentWebhook'
	]
}