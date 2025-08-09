"""
Pricing & Discounts Views

Flask-AppBuilder views for pricing strategy and discount management.
"""

from flask_appbuilder import ModelView
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .models import SPDPricingStrategy, SPDDiscountRule, SPDCampaign


class SPDPricingStrategyView(ModelView):
	"""Pricing strategy management view"""
	
	datamodel = SQLAInterface(SPDPricingStrategy)
	
	list_columns = [
		'strategy_code', 'strategy_name', 'pricing_method',
		'markup_percentage', 'is_active', 'is_default'
	]
	search_columns = ['strategy_code', 'strategy_name']
	list_filters = ['pricing_method', 'is_active', 'is_default']
	base_order = ('strategy_name', 'asc')


class SPDDiscountRuleView(ModelView):
	"""Discount rule management view"""
	
	datamodel = SQLAInterface(SPDDiscountRule)
	
	list_columns = [
		'rule_code', 'rule_name', 'discount_type',
		'discount_percentage', 'effective_date', 'is_active'
	]
	search_columns = ['rule_code', 'rule_name']
	list_filters = ['discount_type', 'is_active', 'effective_date']
	base_order = ('rule_name', 'asc')


class SPDCampaignView(ModelView):
	"""Campaign management view"""
	
	datamodel = SQLAInterface(SPDCampaign)
	
	list_columns = [
		'campaign_code', 'campaign_name', 'campaign_type',
		'start_date', 'end_date', 'budget_amount', 'is_active'
	]
	search_columns = ['campaign_code', 'campaign_name']
	list_filters = ['campaign_type', 'is_active', 'start_date']
	base_order = ('start_date', 'desc')