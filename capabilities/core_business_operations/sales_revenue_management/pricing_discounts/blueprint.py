"""
Pricing & Discounts Blueprint Registration

Flask blueprint registration for Pricing & Discounts sub-capability.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

# Create blueprint
pricing_discounts_bp = Blueprint(
	'pricing_discounts',
	__name__,
	url_prefix='/sales_order_management/pricing_discounts'
)

def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Pricing & Discounts sub-capability with Flask-AppBuilder"""
	
	# Import views (would be implemented)
	from .views import (
		SPDPricingStrategyView, SPDDiscountRuleView, SPDCampaignView
	)
	
	# Register views with AppBuilder
	appbuilder.add_view(
		SPDPricingStrategyView,
		"Pricing Strategies",
		icon="fa-calculator",
		category="Pricing & Discounts",
		category_icon="fa-tags"
	)
	
	appbuilder.add_view(
		SPDDiscountRuleView,
		"Discount Rules",
		icon="fa-percent",
		category="Pricing & Discounts"
	)
	
	appbuilder.add_view(
		SPDCampaignView,
		"Campaigns",
		icon="fa-bullhorn",
		category="Pricing & Discounts"
	)
	
	# Register blueprint
	appbuilder.get_app.register_blueprint(pricing_discounts_bp)