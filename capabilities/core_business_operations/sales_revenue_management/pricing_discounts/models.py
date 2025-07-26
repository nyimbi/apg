"""
Pricing & Discounts Models

Database models for pricing strategies, discount rules, promotional campaigns,
and dynamic pricing engines.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ....auth_rbac.models import BaseMixin, AuditMixin, Model


class SPDPricingStrategy(Model, AuditMixin, BaseMixin):
	"""
	Pricing strategy definitions for different market segments.
	
	Manages comprehensive pricing strategies including cost-plus,
	competitive pricing, value-based pricing, and dynamic pricing.
	"""
	__tablename__ = 'so_pd_pricing_strategy'
	
	# Identity
	strategy_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Strategy Information
	strategy_code = Column(String(20), nullable=False, index=True)
	strategy_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	
	# Pricing Method
	pricing_method = Column(String(20), nullable=False)  # COST_PLUS, COMPETITIVE, VALUE_BASED, DYNAMIC
	
	# Base Calculations
	markup_percentage = Column(DECIMAL(5, 2), default=0.00)
	margin_percentage = Column(DECIMAL(5, 2), default=0.00)
	minimum_margin = Column(DECIMAL(5, 2), default=0.00)
	
	# Dynamic Pricing Configuration
	dynamic_factors = Column(Text, nullable=True)  # JSON configuration
	
	# Application Rules
	item_categories = Column(Text, nullable=True)  # JSON array of category IDs
	customer_types = Column(Text, nullable=True)  # JSON array of customer types
	geographic_regions = Column(Text, nullable=True)  # JSON array of regions
	
	# Effective Dates
	effective_date = Column(Date, nullable=True)
	expiration_date = Column(Date, nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_default = Column(Boolean, default=False)
	priority = Column(Integer, default=0)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'strategy_code', name='uq_spd_strategy_code_tenant'),
	)
	
	def __repr__(self):
		return f"<SPDPricingStrategy {self.strategy_code} - {self.strategy_name}>"


class SPDDiscountRule(Model, AuditMixin, BaseMixin):
	"""
	Discount rule definitions for automated discount application.
	
	Manages complex discount rules including quantity breaks,
	customer-specific discounts, and promotional discounts.
	"""
	__tablename__ = 'so_pd_discount_rule'
	
	# Identity
	rule_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Rule Information
	rule_code = Column(String(20), nullable=False, index=True)
	rule_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	
	# Discount Type
	discount_type = Column(String(20), nullable=False)  # PERCENTAGE, FIXED_AMOUNT, QUANTITY_BREAK
	
	# Discount Values
	discount_percentage = Column(DECIMAL(5, 2), default=0.00)
	discount_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Quantity Break Configuration
	quantity_breaks = Column(Text, nullable=True)  # JSON configuration
	
	# Application Conditions
	minimum_quantity = Column(DECIMAL(12, 4), default=0.0000)
	minimum_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Item/Customer Filters
	item_codes = Column(Text, nullable=True)  # JSON array
	item_categories = Column(Text, nullable=True)  # JSON array
	customer_ids = Column(Text, nullable=True)  # JSON array
	customer_types = Column(Text, nullable=True)  # JSON array
	
	# Combinability
	can_combine_with_other_discounts = Column(Boolean, default=False)
	priority = Column(Integer, default=0)
	maximum_discount_percentage = Column(DECIMAL(5, 2), nullable=True)
	
	# Usage Limits
	max_uses_per_customer = Column(Integer, nullable=True)
	max_total_uses = Column(Integer, nullable=True)
	current_usage_count = Column(Integer, default=0)
	
	# Effective Dates
	effective_date = Column(Date, nullable=False)
	expiration_date = Column(Date, nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	requires_approval = Column(Boolean, default=False)
	auto_apply = Column(Boolean, default=True)
	
	# Relationships
	campaign_id = Column(String(36), ForeignKey('so_pd_campaign.campaign_id'), nullable=True, index=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'rule_code', name='uq_spd_discount_rule_code_tenant'),
	)
	
	def __repr__(self):
		return f"<SPDDiscountRule {self.rule_code} - {self.discount_percentage}%>"


class SPDCampaign(Model, AuditMixin, BaseMixin):
	"""
	Marketing campaigns with promotional pricing and discounts.
	
	Manages time-bound promotional campaigns with specific
	pricing and discount configurations.
	"""
	__tablename__ = 'so_pd_campaign'
	
	# Identity
	campaign_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Campaign Information
	campaign_code = Column(String(20), nullable=False, index=True)
	campaign_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Campaign Type
	campaign_type = Column(String(20), default='PROMOTIONAL')  # PROMOTIONAL, SEASONAL, CLEARANCE
	
	# Campaign Dates
	start_date = Column(Date, nullable=False, index=True)
	end_date = Column(Date, nullable=False, index=True)
	
	# Target Audience
	target_customer_types = Column(Text, nullable=True)  # JSON array
	target_geographic_regions = Column(Text, nullable=True)  # JSON array
	
	# Budget and Limits
	budget_amount = Column(DECIMAL(15, 2), nullable=True)
	spent_amount = Column(DECIMAL(15, 2), default=0.00)
	max_discount_per_order = Column(DECIMAL(15, 2), nullable=True)
	
	# Performance Tracking
	total_orders = Column(Integer, default=0)
	total_revenue = Column(DECIMAL(15, 2), default=0.00)
	total_discount_given = Column(DECIMAL(15, 2), default=0.00)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	auto_start = Column(Boolean, default=True)
	auto_end = Column(Boolean, default=True)
	
	# Relationships
	discount_rules = relationship("SPDDiscountRule", back_populates="campaign")
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'campaign_code', name='uq_spd_campaign_code_tenant'),
	)
	
	def __repr__(self):
		return f"<SPDCampaign {self.campaign_code} - {self.campaign_name}>"


# Create back-reference for campaign relationship
SPDDiscountRule.campaign = relationship("SPDCampaign", back_populates="discount_rules")