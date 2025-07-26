"""
Ratings & Reviews Management Sub-Capability

Collects, moderates, and displays customer ratings and reviews
for products and sellers.
"""

SUBCAPABILITY_META = {
	'name': 'Ratings & Reviews Management',
	'code': 'PSR',
	'description': 'Customer review and rating system with moderation',
	'models': [
		'PSReview',
		'PSReviewRating',
		'PSReviewComment',
		'PSReviewModeration',
		'PSReviewHelpful',
		'PSReviewMedia',
		'PSReviewTemplate',
		'PSReviewIncentive'
	]
}