"""
Dispute Resolution Sub-Capability

Manages conflicts, returns, and refunds between buyers and sellers
on a platform.
"""

SUBCAPABILITY_META = {
	'name': 'Dispute Resolution',
	'code': 'PSD',
	'description': 'Conflict resolution system for buyer-seller disputes',
	'models': [
		'PSDispute',
		'PSDisputeMessage',
		'PSDisputeEvidence',
		'PSDisputeResolution',
		'PSDisputeEscalation',
		'PSDisputeMediation',
		'PSDisputeRefund',
		'PSDisputePolicy'
	]
}