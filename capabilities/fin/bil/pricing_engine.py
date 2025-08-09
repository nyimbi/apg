"""
APG Billing Pricing Engine

Advanced pricing rules engine supporting tiered pricing, volume discounts,
dynamic pricing, promotional campaigns, and AI-powered pricing optimization.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from uuid_extensions import uuid7str

from .models import BLPlan, BLUsage, BLPricingRule, BLDiscount, BillingCurrency


class PricingType(Enum):
	"""Pricing model types"""
	FLAT = "flat"
	TIERED = "tiered"
	VOLUME = "volume"
	GRADUATED = "graduated"
	PACKAGE = "package"
	METERED = "metered"


class DiscountType(Enum):
	"""Discount types"""
	PERCENTAGE = "percentage"
	FIXED_AMOUNT = "fixed_amount"
	FREE_USAGE = "free_usage"
	FREE_MONTHS = "free_months"


class PricingError(Exception):
	"""Pricing calculation error"""
	pass


class PricingRule:
	"""Enhanced pricing rule with complex logic"""
	
	def __init__(self, rule_data: Dict[str, Any]):
		self.id = rule_data.get('id', uuid7str())
		self.name = rule_data['name']
		self.pricing_type = PricingType(rule_data['pricing_type'])
		self.metric_name = rule_data['metric_name']
		self.currency = BillingCurrency(rule_data.get('currency', 'USD'))
		self.active = rule_data.get('active', True)
		self.effective_date = datetime.fromisoformat(rule_data.get('effective_date', datetime.utcnow().isoformat()))
		self.expiry_date = datetime.fromisoformat(rule_data['expiry_date']) if rule_data.get('expiry_date') else None
		
		# Pricing tiers for tiered/graduated pricing
		self.tiers = rule_data.get('tiers', [])
		
		# Flat rate pricing
		self.flat_rate = Decimal(str(rule_data.get('flat_rate', 0)))
		
		# Package pricing
		self.package_size = rule_data.get('package_size', 0)
		self.package_price = Decimal(str(rule_data.get('package_price', 0)))
		
		# Minimum and maximum charges
		self.minimum_charge = Decimal(str(rule_data.get('minimum_charge', 0)))
		self.maximum_charge = Decimal(str(rule_data.get('maximum_charge', 0))) if rule_data.get('maximum_charge') else None
		
		# Free allowance
		self.free_allowance = rule_data.get('free_allowance', 0)
		
		# Conditions for rule application
		self.conditions = rule_data.get('conditions', {})
	
	def is_applicable(self, usage_data: Dict[str, Any]) -> bool:
		"""Check if pricing rule applies to usage data"""
		# Check if rule is active
		if not self.active:
			return False
		
		# Check effective dates
		now = datetime.utcnow()
		if now < self.effective_date:
			return False
		if self.expiry_date and now > self.expiry_date:
			return False
		
		# Check metric name
		if usage_data.get('metric_name') != self.metric_name:
			return False
		
		# Check conditions
		for condition_key, condition_value in self.conditions.items():
			if condition_key in usage_data:
				if usage_data[condition_key] != condition_value:
					return False
		
		return True
	
	def calculate_price(self, quantity: Decimal) -> Dict[str, Any]:
		"""Calculate price based on quantity and pricing type"""
		if self.pricing_type == PricingType.FLAT:
			return self._calculate_flat_price(quantity)
		elif self.pricing_type == PricingType.TIERED:
			return self._calculate_tiered_price(quantity)
		elif self.pricing_type == PricingType.VOLUME:
			return self._calculate_volume_price(quantity)
		elif self.pricing_type == PricingType.GRADUATED:
			return self._calculate_graduated_price(quantity)
		elif self.pricing_type == PricingType.PACKAGE:
			return self._calculate_package_price(quantity)
		elif self.pricing_type == PricingType.METERED:
			return self._calculate_metered_price(quantity)
		else:
			raise PricingError(f"Unknown pricing type: {self.pricing_type}")
	
	def _calculate_flat_price(self, quantity: Decimal) -> Dict[str, Any]:
		"""Calculate flat rate pricing"""
		billable_quantity = max(Decimal('0'), quantity - self.free_allowance)
		total_price = billable_quantity * self.flat_rate
		
		# Apply minimum/maximum charges
		total_price = max(total_price, self.minimum_charge)
		if self.maximum_charge:
			total_price = min(total_price, self.maximum_charge)
		
		return {
			'total_price': total_price,
			'billable_quantity': billable_quantity,
			'effective_rate': self.flat_rate,
			'pricing_breakdown': [{
				'tier': 1,
				'quantity': billable_quantity,
				'rate': self.flat_rate,
				'amount': total_price
			}]
		}
	
	def _calculate_tiered_price(self, quantity: Decimal) -> Dict[str, Any]:
		"""Calculate tiered pricing (all usage priced at the tier rate)"""
		billable_quantity = max(Decimal('0'), quantity - self.free_allowance)
		
		# Find applicable tier
		applicable_tier = None
		for tier in sorted(self.tiers, key=lambda t: t['min_quantity']):
			if billable_quantity >= tier['min_quantity']:
				max_qty = tier.get('max_quantity')
				if max_qty is None or billable_quantity <= max_qty:
					applicable_tier = tier
					break
		
		if not applicable_tier:
			raise PricingError("No applicable pricing tier found")
		
		rate = Decimal(str(applicable_tier['rate']))
		total_price = billable_quantity * rate
		
		# Apply minimum/maximum charges
		total_price = max(total_price, self.minimum_charge)
		if self.maximum_charge:
			total_price = min(total_price, self.maximum_charge)
		
		return {
			'total_price': total_price,
			'billable_quantity': billable_quantity,
			'effective_rate': rate,
			'pricing_breakdown': [{
				'tier': applicable_tier.get('tier_number', 1),
				'quantity': billable_quantity,
				'rate': rate,
				'amount': total_price
			}]
		}
	
	def _calculate_volume_price(self, quantity: Decimal) -> Dict[str, Any]:
		"""Calculate volume pricing (better rate for higher volume)"""
		return self._calculate_tiered_price(quantity)  # Same logic as tiered
	
	def _calculate_graduated_price(self, quantity: Decimal) -> Dict[str, Any]:
		"""Calculate graduated pricing (each tier priced separately)"""
		billable_quantity = max(Decimal('0'), quantity - self.free_allowance)
		remaining_quantity = billable_quantity
		total_price = Decimal('0')
		pricing_breakdown = []
		
		for tier in sorted(self.tiers, key=lambda t: t['min_quantity']):
			if remaining_quantity <= 0:
				break
			
			tier_min = tier['min_quantity']
			tier_max = tier.get('max_quantity')
			tier_rate = Decimal(str(tier['rate']))
			
			# Calculate quantity for this tier
			if tier_max is None:
				tier_quantity = remaining_quantity
			else:
				tier_quantity = min(remaining_quantity, tier_max - tier_min + 1)
			
			if tier_quantity > 0:
				tier_amount = tier_quantity * tier_rate
				total_price += tier_amount
				remaining_quantity -= tier_quantity
				
				pricing_breakdown.append({
					'tier': tier.get('tier_number', len(pricing_breakdown) + 1),
					'quantity': tier_quantity,
					'rate': tier_rate,
					'amount': tier_amount
				})
		
		# Apply minimum/maximum charges
		total_price = max(total_price, self.minimum_charge)
		if self.maximum_charge:
			total_price = min(total_price, self.maximum_charge)
		
		effective_rate = total_price / billable_quantity if billable_quantity > 0 else Decimal('0')
		
		return {
			'total_price': total_price,
			'billable_quantity': billable_quantity,
			'effective_rate': effective_rate,
			'pricing_breakdown': pricing_breakdown
		}
	
	def _calculate_package_price(self, quantity: Decimal) -> Dict[str, Any]:
		"""Calculate package pricing (sold in bundles)"""
		billable_quantity = max(Decimal('0'), quantity - self.free_allowance)
		
		if self.package_size <= 0:
			raise PricingError("Invalid package size")
		
		# Calculate number of packages needed
		packages_needed = int((billable_quantity + self.package_size - 1) // self.package_size)  # Ceiling division
		total_price = packages_needed * self.package_price
		
		# Apply minimum/maximum charges
		total_price = max(total_price, self.minimum_charge)
		if self.maximum_charge:
			total_price = min(total_price, self.maximum_charge)
		
		return {
			'total_price': total_price,
			'billable_quantity': billable_quantity,
			'packages_purchased': packages_needed,
			'package_size': self.package_size,
			'package_price': self.package_price,
			'pricing_breakdown': [{
				'tier': 1,
				'quantity': packages_needed,
				'rate': self.package_price,
				'amount': total_price,
				'description': f"{packages_needed} packages of {self.package_size} units"
			}]
		}
	
	def _calculate_metered_price(self, quantity: Decimal) -> Dict[str, Any]:
		"""Calculate metered pricing (same as flat for now)"""
		return self._calculate_flat_price(quantity)


class DiscountRule:
	"""Discount rule for promotions and campaigns"""
	
	def __init__(self, discount_data: Dict[str, Any]):
		self.id = discount_data.get('id', uuid7str())
		self.code = discount_data.get('code')
		self.name = discount_data['name']
		self.discount_type = DiscountType(discount_data['discount_type'])
		self.value = Decimal(str(discount_data['value']))
		self.active = discount_data.get('active', True)
		self.start_date = datetime.fromisoformat(discount_data.get('start_date', datetime.utcnow().isoformat()))
		self.end_date = datetime.fromisoformat(discount_data['end_date']) if discount_data.get('end_date') else None
		self.usage_limit = discount_data.get('usage_limit')
		self.usage_count = discount_data.get('usage_count', 0)
		self.minimum_amount = Decimal(str(discount_data.get('minimum_amount', 0)))
		self.maximum_discount = Decimal(str(discount_data.get('maximum_discount', 0))) if discount_data.get('maximum_discount') else None
		self.applicable_plans = discount_data.get('applicable_plans', [])
		self.applicable_metrics = discount_data.get('applicable_metrics', [])
	
	def is_applicable(self, context: Dict[str, Any]) -> bool:
		"""Check if discount is applicable"""
		# Check if active
		if not self.active:
			return False
		
		# Check dates
		now = datetime.utcnow()
		if now < self.start_date:
			return False
		if self.end_date and now > self.end_date:
			return False
		
		# Check usage limit
		if self.usage_limit and self.usage_count >= self.usage_limit:
			return False
		
		# Check minimum amount
		total_amount = context.get('total_amount', Decimal('0'))
		if total_amount < self.minimum_amount:
			return False
		
		# Check applicable plans
		if self.applicable_plans and context.get('plan_id') not in self.applicable_plans:
			return False
		
		# Check applicable metrics
		if self.applicable_metrics and context.get('metric_name') not in self.applicable_metrics:
			return False
		
		return True
	
	def calculate_discount(self, amount: Decimal) -> Decimal:
		"""Calculate discount amount"""
		if self.discount_type == DiscountType.PERCENTAGE:
			discount = amount * (self.value / 100)
		elif self.discount_type == DiscountType.FIXED_AMOUNT:
			discount = self.value
		else:
			discount = Decimal('0')  # Other types handled differently
		
		# Apply maximum discount limit
		if self.maximum_discount:
			discount = min(discount, self.maximum_discount)
		
		return min(discount, amount)  # Don't exceed the original amount


class PricingEngine:
	"""Advanced pricing engine with rule management"""
	
	def __init__(self):
		self.pricing_rules: List[PricingRule] = []
		self.discount_rules: List[DiscountRule] = []
		self.logger = logging.getLogger(f"{__name__}.PricingEngine")
		
		# AI pricing optimization
		self._ai_optimization_enabled = False
		self._initialize_ai_integration()
	
	def _initialize_ai_integration(self) -> None:
		"""Initialize AI integration for pricing optimization"""
		try:
			from capabilities.common.ai_orchestration import get_orchestration_service
			self.ai_orchestration = get_orchestration_service()
			self._ai_optimization_enabled = True
			self.logger.info("✅ AI pricing optimization enabled")
		except ImportError:
			self.logger.warning("⚠️  AI orchestration not available for pricing optimization")
	
	def add_pricing_rule(self, rule_data: Dict[str, Any]) -> PricingRule:
		"""Add a new pricing rule"""
		rule = PricingRule(rule_data)
		self.pricing_rules.append(rule)
		self.logger.info(f"Added pricing rule: {rule.name}")
		return rule
	
	def add_discount_rule(self, discount_data: Dict[str, Any]) -> DiscountRule:
		"""Add a new discount rule"""
		discount = DiscountRule(discount_data)
		self.discount_rules.append(discount)
		self.logger.info(f"Added discount rule: {discount.name}")
		return discount
	
	def get_applicable_pricing_rules(self, usage_data: Dict[str, Any]) -> List[PricingRule]:
		"""Get all applicable pricing rules for usage data"""
		applicable_rules = []
		for rule in self.pricing_rules:
			if rule.is_applicable(usage_data):
				applicable_rules.append(rule)
		return applicable_rules
	
	def get_applicable_discounts(self, context: Dict[str, Any]) -> List[DiscountRule]:
		"""Get all applicable discount rules"""
		applicable_discounts = []
		for discount in self.discount_rules:
			if discount.is_applicable(context):
				applicable_discounts.append(discount)
		return applicable_discounts
	
	async def calculate_usage_price(self, usage_records: List[BLUsage], 
								   plan: BLPlan = None, customer_context: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Calculate total price for usage records"""
		try:
			total_price = Decimal('0')
			pricing_breakdown = []
			applied_discounts = []
			
			# Group usage by metric
			usage_by_metric = {}
			for usage in usage_records:
				metric = usage.metric_name
				if metric not in usage_by_metric:
					usage_by_metric[metric] = Decimal('0')
				usage_by_metric[metric] += usage.quantity
			
			# Calculate price for each metric
			for metric_name, total_quantity in usage_by_metric.items():
				usage_data = {
					'metric_name': metric_name,
					'quantity': total_quantity,
					**(customer_context or {})
				}
				
				# Find applicable pricing rules
				applicable_rules = self.get_applicable_pricing_rules(usage_data)
				
				if not applicable_rules:
					self.logger.warning(f"No pricing rules found for metric: {metric_name}")
					continue
				
				# Use the first applicable rule (could be enhanced with priority)
				pricing_rule = applicable_rules[0]
				price_result = pricing_rule.calculate_price(total_quantity)
				
				total_price += price_result['total_price']
				pricing_breakdown.append({
					'metric_name': metric_name,
					'rule_name': pricing_rule.name,
					'quantity': total_quantity,
					'price': price_result['total_price'],
					'details': price_result
				})
			
			# Apply discounts
			discount_context = {
				'total_amount': total_price,
				'plan_id': plan.id if plan else None,
				**(customer_context or {})
			}
			
			applicable_discounts = self.get_applicable_discounts(discount_context)
			total_discount = Decimal('0')
			
			for discount in applicable_discounts:
				discount_amount = discount.calculate_discount(total_price - total_discount)
				if discount_amount > 0:
					total_discount += discount_amount
					applied_discounts.append({
						'discount_name': discount.name,
						'discount_code': discount.code,
						'discount_type': discount.discount_type.value,
						'discount_amount': discount_amount
					})
			
			final_price = max(Decimal('0'), total_price - total_discount)
			
			return {
				'subtotal': total_price,
				'total_discount': total_discount,
				'final_price': final_price,
				'pricing_breakdown': pricing_breakdown,
				'applied_discounts': applied_discounts,
				'calculation_timestamp': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			self.logger.error(f"Pricing calculation failed: {e}")
			raise PricingError(f"Pricing calculation failed: {e}")
	
	async def optimize_pricing_with_ai(self, historical_data: Dict[str, Any], 
									  optimization_goals: List[str] = None) -> Dict[str, Any]:
		"""Use AI to optimize pricing based on historical data"""
		if not self._ai_optimization_enabled:
			return {"error": "AI optimization not available"}
		
		try:
			# Prepare data for AI analysis
			ai_task = {
				"type": "pricing_optimization",
				"input": {
					"historical_usage": historical_data.get('usage_data', []),
					"revenue_data": historical_data.get('revenue_data', []),
					"customer_behavior": historical_data.get('customer_behavior', []),
					"market_data": historical_data.get('market_data', {}),
					"current_pricing_rules": [
						{
							"name": rule.name,
							"metric": rule.metric_name,
							"type": rule.pricing_type.value,
							"tiers": rule.tiers
						} for rule in self.pricing_rules
					],
					"optimization_goals": optimization_goals or ["maximize_revenue", "minimize_churn"]
				},
				"model": "llama3.2:3b"
			}
			
			# Submit to AI orchestration
			task_id = await self.ai_orchestration.submit_task(ai_task)
			
			# Wait for completion (simplified)
			await asyncio.sleep(3)
			task_result = await self.ai_orchestration.get_task_status(task_id)
			
			if task_result.get("status") == "completed":
				return task_result.get("result", {})
			else:
				return {"status": "processing", "task_id": task_id}
		
		except Exception as e:
			self.logger.error(f"AI pricing optimization failed: {e}")
			return {"error": f"AI optimization failed: {e}"}
	
	async def simulate_pricing_changes(self, new_rules: List[Dict[str, Any]], 
									  historical_usage: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Simulate the impact of pricing changes"""
		try:
			# Create temporary rules
			temp_rules = [PricingRule(rule_data) for rule_data in new_rules]
			original_rules = self.pricing_rules.copy()
			
			# Replace rules temporarily
			self.pricing_rules = temp_rules
			
			# Calculate impact on historical usage
			total_original_revenue = Decimal('0')
			total_new_revenue = Decimal('0')
			
			# Restore original rules to calculate baseline
			self.pricing_rules = original_rules
			
			for usage_data in historical_usage:
				# Calculate with original rules
				original_result = await self.calculate_usage_price([])  # Simplified
				total_original_revenue += original_result.get('final_price', Decimal('0'))
			
			# Switch to new rules
			self.pricing_rules = temp_rules
			
			for usage_data in historical_usage:
				# Calculate with new rules
				new_result = await self.calculate_usage_price([])  # Simplified
				total_new_revenue += new_result.get('final_price', Decimal('0'))
			
			# Restore original rules
			self.pricing_rules = original_rules
			
			# Calculate impact
			revenue_change = total_new_revenue - total_original_revenue
			revenue_change_percentage = (revenue_change / total_original_revenue) * 100 if total_original_revenue > 0 else Decimal('0')
			
			return {
				'original_revenue': total_original_revenue,
				'projected_revenue': total_new_revenue,
				'revenue_change': revenue_change,
				'revenue_change_percentage': float(revenue_change_percentage),
				'simulation_date': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			self.logger.error(f"Pricing simulation failed: {e}")
			raise PricingError(f"Pricing simulation failed: {e}")


# Global pricing engine instance
_pricing_engine_instance: Optional[PricingEngine] = None

def get_pricing_engine() -> PricingEngine:
	"""Get global pricing engine instance"""
	global _pricing_engine_instance
	if _pricing_engine_instance is None:
		_pricing_engine_instance = PricingEngine()
		
		# Initialize with default pricing rules
		default_rules = [
			{
				'name': 'API Calls - Tiered Pricing',
				'pricing_type': 'tiered',
				'metric_name': 'api_calls',
				'currency': 'USD',
				'tiers': [
					{'tier_number': 1, 'min_quantity': 0, 'max_quantity': 10000, 'rate': 0.001},
					{'tier_number': 2, 'min_quantity': 10001, 'max_quantity': 100000, 'rate': 0.0008},
					{'tier_number': 3, 'min_quantity': 100001, 'max_quantity': None, 'rate': 0.0005}
				],
				'free_allowance': 1000,
				'minimum_charge': 0
			},
			{
				'name': 'Storage - Graduated Pricing',
				'pricing_type': 'graduated',
				'metric_name': 'storage_gb',
				'currency': 'USD',
				'tiers': [
					{'tier_number': 1, 'min_quantity': 0, 'max_quantity': 100, 'rate': 0.10},
					{'tier_number': 2, 'min_quantity': 101, 'max_quantity': 1000, 'rate': 0.08},
					{'tier_number': 3, 'min_quantity': 1001, 'max_quantity': None, 'rate': 0.06}
				],
				'free_allowance': 10,
				'minimum_charge': 0
			}
		]
		
		for rule_data in default_rules:
			_pricing_engine_instance.add_pricing_rule(rule_data)
	
	return _pricing_engine_instance


__all__ = [
	'PricingEngine',
	'PricingRule',
	'DiscountRule',
	'PricingType',
	'DiscountType',
	'get_pricing_engine',
	'PricingError'
]