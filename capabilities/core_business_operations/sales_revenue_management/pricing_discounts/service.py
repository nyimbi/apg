"""
Pricing & Discounts Service

Business logic for pricing strategies, discount calculations,
and promotional campaign management.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from .models import SPDPricingStrategy, SPDDiscountRule, SPDCampaign


class PricingDiscountsService:
	"""Service class for pricing and discount operations"""
	
	def __init__(self, db_session: Session):
		self.db = db_session
	
	def calculate_item_price(self, tenant_id: str, item_data: Dict[str, Any], 
							customer_data: Dict[str, Any], quantity: Decimal) -> Dict[str, Any]:
		"""Calculate item price with applicable discounts"""
		
		# Get applicable pricing strategy
		strategy = self._get_pricing_strategy(tenant_id, item_data, customer_data)
		
		# Calculate base price
		base_price = self._calculate_base_price(strategy, item_data)
		
		# Apply quantity breaks and discounts
		discounts = self._get_applicable_discounts(tenant_id, item_data, customer_data, quantity, base_price * quantity)
		
		# Calculate final price
		total_discount = sum(discount['amount'] for discount in discounts)
		final_price = max(0, base_price - (total_discount / quantity) if quantity > 0 else base_price)
		
		return {
			'list_price': item_data.get('list_price', base_price),
			'base_price': base_price,
			'final_price': final_price,
			'total_discount': total_discount,
			'discounts_applied': discounts,
			'pricing_strategy': strategy.strategy_code if strategy else None
		}
	
	def _get_pricing_strategy(self, tenant_id: str, item_data: Dict[str, Any], 
							 customer_data: Dict[str, Any]) -> Optional[SPDPricingStrategy]:
		"""Get applicable pricing strategy"""
		# Implementation would check item categories, customer types, etc.
		return self.db.query(SPDPricingStrategy).filter(
			and_(
				SPDPricingStrategy.tenant_id == tenant_id,
				SPDPricingStrategy.is_active == True,
				SPDPricingStrategy.is_default == True
			)
		).first()
	
	def _calculate_base_price(self, strategy: SPDPricingStrategy, item_data: Dict[str, Any]) -> Decimal:
		"""Calculate base price using strategy"""
		if not strategy:
			return Decimal(str(item_data.get('list_price', 0)))
		
		if strategy.pricing_method == 'COST_PLUS':
			cost_price = Decimal(str(item_data.get('cost_price', 0)))
			return cost_price * (1 + strategy.markup_percentage / 100)
		
		return Decimal(str(item_data.get('list_price', 0)))
	
	def _get_applicable_discounts(self, tenant_id: str, item_data: Dict[str, Any],
								customer_data: Dict[str, Any], quantity: Decimal, 
								line_amount: Decimal) -> List[Dict[str, Any]]:
		"""Get applicable discount rules"""
		discounts = []
		
		# Query discount rules
		rules = self.db.query(SPDDiscountRule).filter(
			and_(
				SPDDiscountRule.tenant_id == tenant_id,
				SPDDiscountRule.is_active == True,
				SPDDiscountRule.effective_date <= date.today()
			)
		).all()
		
		for rule in rules:
			if self._rule_applies(rule, item_data, customer_data, quantity, line_amount):
				discount_amount = self._calculate_discount_amount(rule, quantity, line_amount)
				if discount_amount > 0:
					discounts.append({
						'rule_code': rule.rule_code,
						'rule_name': rule.rule_name,
						'amount': discount_amount,
						'type': rule.discount_type
					})
		
		return discounts
	
	def _rule_applies(self, rule: SPDDiscountRule, item_data: Dict[str, Any],
					 customer_data: Dict[str, Any], quantity: Decimal, line_amount: Decimal) -> bool:
		"""Check if discount rule applies"""
		# Check quantity and amount thresholds
		if quantity < rule.minimum_quantity:
			return False
		
		if line_amount < rule.minimum_amount:
			return False
		
		# Check expiration
		if rule.expiration_date and date.today() > rule.expiration_date:
			return False
		
		# Additional filters would be implemented here
		return True
	
	def _calculate_discount_amount(self, rule: SPDDiscountRule, quantity: Decimal, line_amount: Decimal) -> Decimal:
		"""Calculate discount amount for rule"""
		if rule.discount_type == 'PERCENTAGE':
			return line_amount * (rule.discount_percentage / 100)
		elif rule.discount_type == 'FIXED_AMOUNT':
			return rule.discount_amount
		
		return Decimal('0.00')