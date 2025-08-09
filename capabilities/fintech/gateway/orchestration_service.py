"""
Payment Orchestration and Smart Routing 2.0 Service
Advanced payment orchestration with intelligent routing, failover, and optimization.

Copyright (c) 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, validator
from uuid_extensions import uuid7str

logger = logging.getLogger(__name__)


class PaymentMethod(str, Enum):
	CARD = "card"
	BANK_TRANSFER = "bank_transfer"
	DIGITAL_WALLET = "digital_wallet"
	CRYPTOCURRENCY = "cryptocurrency"
	BNPL = "buy_now_pay_later"
	APM = "alternative_payment_method"


class ProviderTier(str, Enum):
	PRIMARY = "primary"
	SECONDARY = "secondary"
	TERTIARY = "tertiary"
	BACKUP = "backup"


class RoutingStrategy(str, Enum):
	COST_OPTIMIZATION = "cost_optimization"
	SUCCESS_RATE_OPTIMIZATION = "success_rate_optimization"
	LATENCY_OPTIMIZATION = "latency_optimization"
	BALANCED = "balanced"
	CUSTOM = "custom"


class FailoverTrigger(str, Enum):
	PROVIDER_DOWN = "provider_down"
	HIGH_LATENCY = "high_latency"
	LOW_SUCCESS_RATE = "low_success_rate"
	COST_THRESHOLD = "cost_threshold"
	MANUAL = "manual"


class OrchestrationRule(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	priority: int = Field(ge=1, le=100)
	conditions: Dict[str, Any]
	actions: Dict[str, Any]
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class PaymentProvider(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	provider_code: str
	tier: ProviderTier
	supported_methods: List[PaymentMethod]
	supported_currencies: List[str]
	supported_countries: List[str]
	base_cost: Decimal = Field(default=Decimal("0.00"))
	percentage_cost: Decimal = Field(default=Decimal("0.00"))
	fixed_cost: Decimal = Field(default=Decimal("0.00"))
	max_amount: Optional[Decimal] = None
	min_amount: Optional[Decimal] = None
	settlement_time: int = Field(default=24, description="Hours")
	capabilities: Dict[str, Any] = Field(default_factory=dict)
	health_score: float = Field(default=1.0, ge=0.0, le=1.0)
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class RoutingDecision(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	transaction_id: str
	primary_provider: str
	fallback_providers: List[str] = Field(default_factory=list)
	routing_strategy: RoutingStrategy
	decision_factors: Dict[str, Any] = Field(default_factory=dict)
	estimated_cost: Decimal = Field(default=Decimal("0.00"))
	estimated_success_rate: float = Field(default=0.0)
	estimated_latency: int = Field(default=0, description="Milliseconds")
	created_at: datetime = Field(default_factory=datetime.utcnow)


class TransactionRoute(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	transaction_id: str
	routing_decision_id: str
	provider_id: str
	attempt_number: int = Field(default=1)
	status: str = Field(default="pending")
	response_time: Optional[int] = None
	error_code: Optional[str] = None
	error_message: Optional[str] = None
	cost: Optional[Decimal] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	completed_at: Optional[datetime] = None


class ProviderHealthMetrics(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	provider_id: str
	success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
	average_response_time: int = Field(default=0, description="Milliseconds")
	error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
	uptime: float = Field(default=1.0, ge=0.0, le=1.0)
	total_transactions: int = Field(default=0)
	successful_transactions: int = Field(default=0)
	failed_transactions: int = Field(default=0)
	last_updated: datetime = Field(default_factory=datetime.utcnow)


class SmartRoutingConfig(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	default_strategy: RoutingStrategy = RoutingStrategy.BALANCED
	cost_weight: float = Field(default=0.3, ge=0.0, le=1.0)
	success_rate_weight: float = Field(default=0.4, ge=0.0, le=1.0)
	latency_weight: float = Field(default=0.3, ge=0.0, le=1.0)
	min_success_rate_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
	max_latency_threshold: int = Field(default=5000, description="Milliseconds")
	max_cost_threshold: Decimal = Field(default=Decimal("10.00"))
	failover_enabled: bool = True
	retry_attempts: int = Field(default=3, ge=1, le=10)
	retry_delay: int = Field(default=1000, description="Milliseconds")


class PaymentOrchestrationService:
	"""Advanced payment orchestration and smart routing service."""
	
	def __init__(self):
		self._providers: Dict[str, PaymentProvider] = {}
		self._rules: Dict[str, OrchestrationRule] = {}
		self._routing_decisions: Dict[str, RoutingDecision] = {}
		self._transaction_routes: Dict[str, List[TransactionRoute]] = {}
		self._provider_metrics: Dict[str, ProviderHealthMetrics] = {}
		self._config = SmartRoutingConfig()
		self._active_transactions: Dict[str, Dict[str, Any]] = {}
		
		# Initialize default providers and rules
		asyncio.create_task(self._initialize_default_configuration())
	
	async def _initialize_default_configuration(self) -> None:
		"""Initialize default providers and orchestration rules."""
		# Default providers
		default_providers = [
			PaymentProvider(
				name="Stripe",
				provider_code="stripe",
				tier=ProviderTier.PRIMARY,
				supported_methods=[PaymentMethod.CARD, PaymentMethod.DIGITAL_WALLET],
				supported_currencies=["USD", "EUR", "GBP", "KES"],
				supported_countries=["US", "GB", "KE", "DE", "FR"],
				percentage_cost=Decimal("2.9"),
				fixed_cost=Decimal("0.30"),
				settlement_time=2,
				capabilities={
					"3ds": True,
					"recurring": True,
					"refunds": True,
					"disputes": True
				}
			),
			PaymentProvider(
				name="PayPal",
				provider_code="paypal",
				tier=ProviderTier.SECONDARY,
				supported_methods=[PaymentMethod.DIGITAL_WALLET, PaymentMethod.CARD],
				supported_currencies=["USD", "EUR", "GBP"],
				supported_countries=["US", "GB", "DE", "FR"],
				percentage_cost=Decimal("3.4"),
				fixed_cost=Decimal("0.35"),
				settlement_time=24,
				capabilities={
					"buyer_protection": True,
					"recurring": True,
					"refunds": True
				}
			),
			PaymentProvider(
				name="Adyen",
				provider_code="adyen",
				tier=ProviderTier.PRIMARY,
				supported_methods=[PaymentMethod.CARD, PaymentMethod.BANK_TRANSFER, PaymentMethod.APM],
				supported_currencies=["USD", "EUR", "GBP", "KES"],
				supported_countries=["US", "GB", "KE", "NL", "DE"],
				percentage_cost=Decimal("2.6"),
				fixed_cost=Decimal("0.10"),
				settlement_time=1,
				capabilities={
					"3ds": True,
					"local_methods": True,
					"real_time_settlement": True
				}
			)
		]
		
		for provider in default_providers:
			await self.register_provider(provider)
		
		# Default orchestration rules
		default_rules = [
			OrchestrationRule(
				name="High Value Transaction Route",
				description="Route high-value transactions to most reliable providers",
				priority=95,
				conditions={
					"amount": {"gte": 1000.00},
					"provider_tier": ["primary"]
				},
				actions={
					"strategy": "success_rate_optimization",
					"require_3ds": True,
					"fraud_check": "enhanced"
				}
			),
			OrchestrationRule(
				name="Currency-Specific Routing",
				description="Route transactions based on currency optimization",
				priority=80,
				conditions={
					"currency": "KES"
				},
				actions={
					"preferred_providers": ["adyen"],
					"local_method_priority": True
				}
			),
			OrchestrationRule(
				name="Cost Optimization for Low Value",
				description="Optimize costs for transactions under $100",
				priority=70,
				conditions={
					"amount": {"lt": 100.00}
				},
				actions={
					"strategy": "cost_optimization",
					"skip_premium_features": True
				}
			)
		]
		
		for rule in default_rules:
			await self.add_orchestration_rule(rule)
	
	async def register_provider(self, provider: PaymentProvider) -> str:
		"""Register a new payment provider."""
		self._providers[provider.id] = provider
		
		# Initialize health metrics
		self._provider_metrics[provider.id] = ProviderHealthMetrics(
			provider_id=provider.id
		)
		
		logger.info(f"Registered payment provider: {provider.name} ({provider.provider_code})")
		return provider.id
	
	async def update_provider(self, provider_id: str, updates: Dict[str, Any]) -> bool:
		"""Update provider configuration."""
		if provider_id not in self._providers:
			return False
		
		provider = self._providers[provider_id]
		for key, value in updates.items():
			if hasattr(provider, key):
				setattr(provider, key, value)
		
		provider.updated_at = datetime.utcnow()
		logger.info(f"Updated provider {provider_id}: {list(updates.keys())}")
		return True
	
	async def add_orchestration_rule(self, rule: OrchestrationRule) -> str:
		"""Add a new orchestration rule."""
		self._rules[rule.id] = rule
		logger.info(f"Added orchestration rule: {rule.name}")
		return rule.id
	
	async def update_orchestration_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
		"""Update an orchestration rule."""
		if rule_id not in self._rules:
			return False
		
		rule = self._rules[rule_id]
		for key, value in updates.items():
			if hasattr(rule, key):
				setattr(rule, key, value)
		
		rule.updated_at = datetime.utcnow()
		logger.info(f"Updated orchestration rule {rule_id}: {list(updates.keys())}")
		return True
	
	async def calculate_optimal_route(self, transaction_data: Dict[str, Any]) -> RoutingDecision:
		"""Calculate the optimal payment route using smart routing."""
		transaction_id = transaction_data.get('id', uuid7str())
		amount = Decimal(str(transaction_data.get('amount', 0)))
		currency = transaction_data.get('currency', 'USD')
		country = transaction_data.get('country', 'US')
		payment_method = PaymentMethod(transaction_data.get('payment_method', PaymentMethod.CARD))
		
		# Get eligible providers
		eligible_providers = await self._get_eligible_providers(
			amount, currency, country, payment_method
		)
		
		if not eligible_providers:
			raise ValueError("No eligible payment providers found")
		
		# Apply orchestration rules
		routing_context = await self._apply_orchestration_rules(transaction_data, eligible_providers)
		
		# Calculate routing scores
		provider_scores = await self._calculate_provider_scores(
			eligible_providers, transaction_data, routing_context
		)
		
		# Sort providers by score
		sorted_providers = sorted(
			provider_scores.items(),
			key=lambda x: x[1]['total_score'],
			reverse=True
		)
		
		primary_provider_id = sorted_providers[0][0]
		fallback_providers = [p[0] for p in sorted_providers[1:]]
		
		decision = RoutingDecision(
			transaction_id=transaction_id,
			primary_provider=primary_provider_id,
			fallback_providers=fallback_providers,
			routing_strategy=routing_context.get('strategy', self._config.default_strategy),
			decision_factors=sorted_providers[0][1],
			estimated_cost=sorted_providers[0][1]['cost'],
			estimated_success_rate=sorted_providers[0][1]['success_rate'],
			estimated_latency=sorted_providers[0][1]['latency']
		)
		
		self._routing_decisions[decision.id] = decision
		logger.info(f"Calculated optimal route for transaction {transaction_id}: {primary_provider_id}")
		
		return decision
	
	async def _get_eligible_providers(
		self,
		amount: Decimal,
		currency: str,
		country: str,
		payment_method: PaymentMethod
	) -> List[PaymentProvider]:
		"""Get providers eligible for the transaction."""
		eligible = []
		
		for provider in self._providers.values():
			if not provider.is_active:
				continue
			
			# Check currency support
			if currency not in provider.supported_currencies:
				continue
			
			# Check country support
			if country not in provider.supported_countries:
				continue
			
			# Check payment method support
			if payment_method not in provider.supported_methods:
				continue
			
			# Check amount limits
			if provider.min_amount and amount < provider.min_amount:
				continue
			if provider.max_amount and amount > provider.max_amount:
				continue
			
			# Check health score threshold
			metrics = self._provider_metrics.get(provider.id)
			if metrics and metrics.success_rate < self._config.min_success_rate_threshold:
				continue
			
			eligible.append(provider)
		
		return eligible
	
	async def _apply_orchestration_rules(
		self,
		transaction_data: Dict[str, Any],
		eligible_providers: List[PaymentProvider]
	) -> Dict[str, Any]:
		"""Apply orchestration rules to determine routing context."""
		context = {
			'strategy': self._config.default_strategy,
			'preferred_providers': [],
			'excluded_providers': [],
			'special_requirements': {}
		}
		
		# Sort rules by priority (highest first)
		sorted_rules = sorted(
			[rule for rule in self._rules.values() if rule.is_active],
			key=lambda r: r.priority,
			reverse=True
		)
		
		for rule in sorted_rules:
			if await self._rule_matches_transaction(rule, transaction_data):
				# Apply rule actions
				for action_key, action_value in rule.actions.items():
					if action_key == 'strategy':
						context['strategy'] = RoutingStrategy(action_value)
					elif action_key == 'preferred_providers':
						context['preferred_providers'].extend(action_value)
					elif action_key == 'excluded_providers':
						context['excluded_providers'].extend(action_value)
					else:
						context['special_requirements'][action_key] = action_value
				
				logger.debug(f"Applied orchestration rule: {rule.name}")
		
		return context
	
	async def _rule_matches_transaction(
		self,
		rule: OrchestrationRule,
		transaction_data: Dict[str, Any]
	) -> bool:
		"""Check if a rule matches the transaction."""
		for condition_key, condition_value in rule.conditions.items():
			transaction_value = transaction_data.get(condition_key)
			
			if isinstance(condition_value, dict):
				# Range conditions
				if 'gte' in condition_value and transaction_value < condition_value['gte']:
					return False
				if 'gt' in condition_value and transaction_value <= condition_value['gt']:
					return False
				if 'lte' in condition_value and transaction_value > condition_value['lte']:
					return False
				if 'lt' in condition_value and transaction_value >= condition_value['lt']:
					return False
			elif isinstance(condition_value, list):
				# List membership
				if transaction_value not in condition_value:
					return False
			else:
				# Exact match
				if transaction_value != condition_value:
					return False
		
		return True
	
	async def _calculate_provider_scores(
		self,
		providers: List[PaymentProvider],
		transaction_data: Dict[str, Any],
		routing_context: Dict[str, Any]
	) -> Dict[str, Dict[str, Any]]:
		"""Calculate routing scores for providers."""
		scores = {}
		strategy = routing_context.get('strategy', self._config.default_strategy)
		
		for provider in providers:
			metrics = self._provider_metrics.get(provider.id, ProviderHealthMetrics(provider_id=provider.id))
			
			# Calculate individual scores
			cost_score = await self._calculate_cost_score(provider, transaction_data)
			success_rate_score = metrics.success_rate
			latency_score = await self._calculate_latency_score(provider, metrics)
			
			# Apply strategy weights
			if strategy == RoutingStrategy.COST_OPTIMIZATION:
				weights = {'cost': 0.7, 'success_rate': 0.2, 'latency': 0.1}
			elif strategy == RoutingStrategy.SUCCESS_RATE_OPTIMIZATION:
				weights = {'cost': 0.1, 'success_rate': 0.8, 'latency': 0.1}
			elif strategy == RoutingStrategy.LATENCY_OPTIMIZATION:
				weights = {'cost': 0.1, 'success_rate': 0.3, 'latency': 0.6}
			else:  # BALANCED or default
				weights = {
					'cost': self._config.cost_weight,
					'success_rate': self._config.success_rate_weight,
					'latency': self._config.latency_weight
				}
			
			# Calculate weighted total score
			total_score = (
				cost_score * weights['cost'] +
				success_rate_score * weights['success_rate'] +
				latency_score * weights['latency']
			)
			
			# Apply preference bonuses
			if provider.provider_code in routing_context.get('preferred_providers', []):
				total_score *= 1.2
			
			# Apply tier bonuses
			tier_bonus = {
				ProviderTier.PRIMARY: 1.1,
				ProviderTier.SECONDARY: 1.0,
				ProviderTier.TERTIARY: 0.9,
				ProviderTier.BACKUP: 0.8
			}
			total_score *= tier_bonus.get(provider.tier, 1.0)
			
			scores[provider.id] = {
				'total_score': total_score,
				'cost_score': cost_score,
				'success_rate': success_rate_score,
				'latency_score': latency_score,
				'cost': await self._calculate_transaction_cost(provider, transaction_data),
				'latency': metrics.average_response_time,
				'health_score': provider.health_score
			}
		
		return scores
	
	async def _calculate_cost_score(self, provider: PaymentProvider, transaction_data: Dict[str, Any]) -> float:
		"""Calculate cost-based score (higher is better, so invert)."""
		total_cost = await self._calculate_transaction_cost(provider, transaction_data)
		
		# Normalize cost score (assuming max reasonable cost of $50)
		max_cost = 50.0
		normalized_cost = min(float(total_cost), max_cost) / max_cost
		
		# Invert so lower cost = higher score
		return 1.0 - normalized_cost
	
	async def _calculate_transaction_cost(self, provider: PaymentProvider, transaction_data: Dict[str, Any]) -> Decimal:
		"""Calculate total transaction cost for provider."""
		amount = Decimal(str(transaction_data.get('amount', 0)))
		
		percentage_fee = amount * (provider.percentage_cost / 100)
		total_cost = provider.base_cost + percentage_fee + provider.fixed_cost
		
		return total_cost
	
	async def _calculate_latency_score(self, provider: PaymentProvider, metrics: ProviderHealthMetrics) -> float:
		"""Calculate latency-based score (lower latency = higher score)."""
		latency_ms = metrics.average_response_time or 1000  # Default 1 second
		
		# Normalize latency score (assuming max reasonable latency of 10 seconds)
		max_latency = 10000  # 10 seconds in ms
		normalized_latency = min(latency_ms, max_latency) / max_latency
		
		# Invert so lower latency = higher score
		return 1.0 - normalized_latency
	
	async def execute_payment_route(self, routing_decision: RoutingDecision, payment_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute payment using the routing decision with failover."""
		transaction_id = routing_decision.transaction_id
		self._active_transactions[transaction_id] = {
			'routing_decision': routing_decision,
			'payment_data': payment_data,
			'start_time': datetime.utcnow(),
			'attempts': []
		}
		
		# Try primary provider first
		providers_to_try = [routing_decision.primary_provider] + routing_decision.fallback_providers
		
		for attempt_num, provider_id in enumerate(providers_to_try, 1):
			if attempt_num > self._config.retry_attempts:
				break
			
			try:
				result = await self._attempt_payment(
					transaction_id, provider_id, payment_data, attempt_num
				)
				
				if result['status'] == 'success':
					await self._record_successful_transaction(transaction_id, provider_id, result)
					return result
				
				# Check if we should continue with failover
				should_failover = await self._should_trigger_failover(result, provider_id)
				if not should_failover:
					await self._record_failed_transaction(transaction_id, provider_id, result)
					return result
				
				# Wait before next attempt
				if attempt_num < len(providers_to_try):
					await asyncio.sleep(self._config.retry_delay / 1000)
				
			except Exception as e:
				logger.error(f"Payment attempt failed for provider {provider_id}: {str(e)}")
				await self._record_failed_transaction(transaction_id, provider_id, {
					'status': 'error',
					'error': str(e)
				})
		
		# All attempts failed
		return {
			'status': 'failed',
			'error': 'All payment providers failed',
			'attempts': len(self._active_transactions[transaction_id]['attempts'])
		}
	
	async def _attempt_payment(
		self,
		transaction_id: str,
		provider_id: str,
		payment_data: Dict[str, Any],
		attempt_number: int
	) -> Dict[str, Any]:
		"""Attempt payment with specific provider."""
		start_time = datetime.utcnow()
		
		# Create transaction route record
		route = TransactionRoute(
			transaction_id=transaction_id,
			routing_decision_id=self._routing_decisions[transaction_id].id,
			provider_id=provider_id,
			attempt_number=attempt_number
		)
		
		if transaction_id not in self._transaction_routes:
			self._transaction_routes[transaction_id] = []
		self._transaction_routes[transaction_id].append(route)
		
		try:
			# Simulate payment processing (in real implementation, call actual provider APIs)
			provider = self._providers[provider_id]
			processing_time = await self._simulate_payment_processing(provider, payment_data)
			
			# Simulate success/failure based on provider health
			metrics = self._provider_metrics[provider_id]
			success_probability = metrics.success_rate or 0.9
			
			import random
			is_successful = random.random() < success_probability
			
			end_time = datetime.utcnow()
			response_time = int((end_time - start_time).total_seconds() * 1000)
			
			if is_successful:
				result = {
					'status': 'success',
					'provider_id': provider_id,
					'transaction_id': transaction_id,
					'provider_transaction_id': uuid7str(),
					'response_time': response_time,
					'cost': await self._calculate_transaction_cost(provider, payment_data)
				}
				route.status = 'completed'
			else:
				result = {
					'status': 'failed',
					'provider_id': provider_id,
					'transaction_id': transaction_id,
					'error_code': 'PAYMENT_DECLINED',
					'error_message': 'Payment was declined by the provider',
					'response_time': response_time
				}
				route.status = 'failed'
				route.error_code = result['error_code']
				route.error_message = result['error_message']
			
			route.response_time = response_time
			route.completed_at = end_time
			
			# Update provider metrics
			await self._update_provider_metrics(provider_id, is_successful, response_time)
			
			return result
			
		except Exception as e:
			route.status = 'error'
			route.error_message = str(e)
			route.completed_at = datetime.utcnow()
			raise
	
	async def _simulate_payment_processing(self, provider: PaymentProvider, payment_data: Dict[str, Any]) -> int:
		"""Simulate payment processing time."""
		base_time = 500  # Base 500ms
		
		# Add variability based on provider tier
		tier_multiplier = {
			ProviderTier.PRIMARY: 1.0,
			ProviderTier.SECONDARY: 1.2,
			ProviderTier.TERTIARY: 1.5,
			ProviderTier.BACKUP: 2.0
		}
		
		processing_time = int(base_time * tier_multiplier.get(provider.tier, 1.0))
		
		# Simulate network latency
		import random
		jitter = random.randint(-100, 200)
		processing_time += jitter
		
		await asyncio.sleep(max(processing_time, 100) / 1000)  # Min 100ms
		return processing_time
	
	async def _should_trigger_failover(self, result: Dict[str, Any], provider_id: str) -> bool:
		"""Determine if failover should be triggered."""
		if not self._config.failover_enabled:
			return False
		
		# Check specific error conditions that warrant failover
		failover_conditions = [
			result.get('error_code') in ['PROVIDER_UNAVAILABLE', 'TIMEOUT', 'SYSTEM_ERROR'],
			result.get('response_time', 0) > self._config.max_latency_threshold,
			result.get('status') == 'error'
		]
		
		return any(failover_conditions)
	
	async def _record_successful_transaction(
		self,
		transaction_id: str,
		provider_id: str,
		result: Dict[str, Any]
	) -> None:
		"""Record successful transaction metrics."""
		if transaction_id in self._active_transactions:
			transaction_info = self._active_transactions[transaction_id]
			transaction_info['status'] = 'completed'
			transaction_info['provider_used'] = provider_id
			transaction_info['end_time'] = datetime.utcnow()
			transaction_info['result'] = result
		
		logger.info(f"Payment successful for transaction {transaction_id} via {provider_id}")
	
	async def _record_failed_transaction(
		self,
		transaction_id: str,
		provider_id: str,
		result: Dict[str, Any]
	) -> None:
		"""Record failed transaction metrics."""
		if transaction_id in self._active_transactions:
			transaction_info = self._active_transactions[transaction_id]
			transaction_info['attempts'].append({
				'provider_id': provider_id,
				'result': result,
				'timestamp': datetime.utcnow()
			})
		
		logger.warning(f"Payment failed for transaction {transaction_id} via {provider_id}: {result.get('error_message')}")
	
	async def _update_provider_metrics(self, provider_id: str, success: bool, response_time: int) -> None:
		"""Update provider health metrics."""
		metrics = self._provider_metrics.get(provider_id)
		if not metrics:
			return
		
		# Update counters
		metrics.total_transactions += 1
		if success:
			metrics.successful_transactions += 1
		else:
			metrics.failed_transactions += 1
		
		# Recalculate rates
		metrics.success_rate = metrics.successful_transactions / metrics.total_transactions
		metrics.error_rate = metrics.failed_transactions / metrics.total_transactions
		
		# Update average response time (using exponential moving average)
		alpha = 0.1  # Smoothing factor
		if metrics.average_response_time == 0:
			metrics.average_response_time = response_time
		else:
			metrics.average_response_time = int(
				alpha * response_time + (1 - alpha) * metrics.average_response_time
			)
		
		metrics.last_updated = datetime.utcnow()
		
		# Update provider health score
		provider = self._providers[provider_id]
		provider.health_score = await self._calculate_provider_health_score(metrics)
	
	async def _calculate_provider_health_score(self, metrics: ProviderHealthMetrics) -> float:
		"""Calculate overall provider health score."""
		success_weight = 0.5
		latency_weight = 0.3
		uptime_weight = 0.2
		
		# Normalize latency score (lower is better)
		latency_score = max(0, 1 - (metrics.average_response_time / 5000))  # 5 second max
		
		health_score = (
			metrics.success_rate * success_weight +
			latency_score * latency_weight +
			metrics.uptime * uptime_weight
		)
		
		return min(max(health_score, 0.0), 1.0)
	
	async def get_provider_health_report(self, provider_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get comprehensive provider health report."""
		if provider_id:
			if provider_id not in self._providers:
				raise ValueError(f"Provider {provider_id} not found")
			
			provider = self._providers[provider_id]
			metrics = self._provider_metrics.get(provider_id, ProviderHealthMetrics(provider_id=provider_id))
			
			return {
				'provider': provider.dict(),
				'metrics': metrics.dict(),
				'recommendations': await self._generate_provider_recommendations(provider_id)
			}
		else:
			# Return report for all providers
			report = {}
			for pid in self._providers.keys():
				report[pid] = await self.get_provider_health_report(pid)
			
			# Add summary statistics
			report['summary'] = await self._generate_health_summary()
			return report
	
	async def _generate_provider_recommendations(self, provider_id: str) -> List[str]:
		"""Generate recommendations for provider optimization."""
		recommendations = []
		provider = self._providers[provider_id]
		metrics = self._provider_metrics.get(provider_id)
		
		if not metrics:
			return ["Insufficient data for recommendations"]
		
		if metrics.success_rate < 0.9:
			recommendations.append(f"Success rate ({metrics.success_rate:.1%}) is below optimal. Consider reviewing configuration.")
		
		if metrics.average_response_time > 3000:
			recommendations.append(f"Average response time ({metrics.average_response_time}ms) is high. Check connectivity.")
		
		if provider.health_score < 0.8:
			recommendations.append("Overall health score is concerning. Consider demoting tier or investigating issues.")
		
		if metrics.total_transactions < 100:
			recommendations.append("Limited transaction history. Monitor closely as volume increases.")
		
		return recommendations or ["Provider performance is optimal"]
	
	async def _generate_health_summary(self) -> Dict[str, Any]:
		"""Generate overall health summary."""
		if not self._provider_metrics:
			return {"status": "No data available"}
		
		all_metrics = list(self._provider_metrics.values())
		total_transactions = sum(m.total_transactions for m in all_metrics)
		total_successful = sum(m.successful_transactions for m in all_metrics)
		
		avg_success_rate = total_successful / total_transactions if total_transactions > 0 else 0
		avg_response_time = statistics.mean([m.average_response_time for m in all_metrics if m.average_response_time > 0]) if all_metrics else 0
		
		healthy_providers = sum(1 for p in self._providers.values() if p.health_score > 0.8)
		total_providers = len(self._providers)
		
		return {
			'total_providers': total_providers,
			'healthy_providers': healthy_providers,
			'overall_success_rate': avg_success_rate,
			'average_response_time': int(avg_response_time),
			'total_transactions_processed': total_transactions,
			'health_percentage': (healthy_providers / total_providers * 100) if total_providers > 0 else 0
		}
	
	async def get_routing_analytics(self, time_range: Optional[Dict[str, datetime]] = None) -> Dict[str, Any]:
		"""Get comprehensive routing analytics."""
		start_time = (time_range or {}).get('start', datetime.utcnow() - timedelta(days=7))
		end_time = (time_range or {}).get('end', datetime.utcnow())
		
		# Filter routing decisions by time range
		filtered_decisions = [
			decision for decision in self._routing_decisions.values()
			if start_time <= decision.created_at <= end_time
		]
		
		if not filtered_decisions:
			return {"message": "No routing data available for the specified time range"}
		
		# Calculate analytics
		total_decisions = len(filtered_decisions)
		strategy_distribution = {}
		provider_usage = {}
		cost_analysis = {
			'total_estimated_cost': Decimal('0'),
			'average_cost': Decimal('0'),
			'cost_by_provider': {}
		}
		
		for decision in filtered_decisions:
			# Strategy distribution
			strategy = decision.routing_strategy.value
			strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
			
			# Provider usage
			provider_usage[decision.primary_provider] = provider_usage.get(decision.primary_provider, 0) + 1
			
			# Cost analysis
			cost_analysis['total_estimated_cost'] += decision.estimated_cost
			provider_id = decision.primary_provider
			if provider_id not in cost_analysis['cost_by_provider']:
				cost_analysis['cost_by_provider'][provider_id] = Decimal('0')
			cost_analysis['cost_by_provider'][provider_id] += decision.estimated_cost
		
		cost_analysis['average_cost'] = cost_analysis['total_estimated_cost'] / total_decisions
		
		# Performance metrics
		avg_success_rate = statistics.mean([d.estimated_success_rate for d in filtered_decisions])
		avg_latency = statistics.mean([d.estimated_latency for d in filtered_decisions])
		
		return {
			'time_range': {
				'start': start_time.isoformat(),
				'end': end_time.isoformat()
			},
			'total_routing_decisions': total_decisions,
			'strategy_distribution': strategy_distribution,
			'provider_usage': provider_usage,
			'cost_analysis': {
				'total_estimated_cost': float(cost_analysis['total_estimated_cost']),
				'average_cost': float(cost_analysis['average_cost']),
				'cost_by_provider': {k: float(v) for k, v in cost_analysis['cost_by_provider'].items()}
			},
			'performance_metrics': {
				'average_estimated_success_rate': avg_success_rate,
				'average_estimated_latency_ms': avg_latency
			},
			'top_providers': sorted(provider_usage.items(), key=lambda x: x[1], reverse=True)[:5]
		}
	
	async def optimize_routing_configuration(self) -> Dict[str, Any]:
		"""Analyze performance and suggest routing optimizations."""
		recommendations = []
		
		# Analyze provider performance
		for provider_id, provider in self._providers.items():
			metrics = self._provider_metrics.get(provider_id)
			if not metrics:
				continue
			
			if metrics.success_rate < 0.85 and provider.tier == ProviderTier.PRIMARY:
				recommendations.append({
					'type': 'provider_demotion',
					'provider_id': provider_id,
					'current_tier': provider.tier.value,
					'suggested_tier': ProviderTier.SECONDARY.value,
					'reason': f'Success rate ({metrics.success_rate:.1%}) below threshold for primary tier'
				})
			
			if metrics.average_response_time > 5000:
				recommendations.append({
					'type': 'latency_concern',
					'provider_id': provider_id,
					'current_latency': metrics.average_response_time,
					'reason': 'High latency may impact user experience'
				})
		
		# Analyze routing strategy effectiveness
		recent_decisions = [
			d for d in self._routing_decisions.values()
			if d.created_at > datetime.utcnow() - timedelta(days=7)
		]
		
		if recent_decisions:
			strategy_performance = {}
			for decision in recent_decisions:
				strategy = decision.routing_strategy.value
				if strategy not in strategy_performance:
					strategy_performance[strategy] = {'count': 0, 'total_cost': Decimal('0'), 'total_success_rate': 0.0}
				
				strategy_performance[strategy]['count'] += 1
				strategy_performance[strategy]['total_cost'] += decision.estimated_cost
				strategy_performance[strategy]['total_success_rate'] += decision.estimated_success_rate
			
			# Calculate averages and find best strategy
			best_strategy = None
			best_score = 0
			
			for strategy, perf in strategy_performance.items():
				avg_cost = perf['total_cost'] / perf['count']
				avg_success_rate = perf['total_success_rate'] / perf['count']
				
				# Combined score (success rate weighted more heavily)
				score = avg_success_rate * 0.7 + (1 - min(float(avg_cost), 10) / 10) * 0.3
				
				if score > best_score:
					best_score = score
					best_strategy = strategy
			
			if best_strategy and best_strategy != self._config.default_strategy.value:
				recommendations.append({
					'type': 'strategy_optimization',
					'current_strategy': self._config.default_strategy.value,
					'suggested_strategy': best_strategy,
					'expected_improvement': f'{(best_score - 0.8) * 100:.1f}% better performance'
				})
		
		# Check for unused providers
		unused_providers = []
		for provider_id, provider in self._providers.items():
			metrics = self._provider_metrics.get(provider_id)
			if metrics and metrics.total_transactions == 0:
				unused_providers.append(provider_id)
		
		if unused_providers:
			recommendations.append({
				'type': 'unused_providers',
				'provider_ids': unused_providers,
				'suggestion': 'Consider removing or investigating why these providers are not being used'
			})
		
		return {
			'recommendations': recommendations,
			'analysis_timestamp': datetime.utcnow().isoformat(),
			'analysis_period': '7 days',
			'total_recommendations': len(recommendations)
		}
	
	async def simulate_routing_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
		"""Simulate routing behavior under different scenarios."""
		scenario_type = scenario.get('type', 'load_test')
		
		if scenario_type == 'load_test':
			return await self._simulate_load_test(scenario)
		elif scenario_type == 'provider_failure':
			return await self._simulate_provider_failure(scenario)
		elif scenario_type == 'cost_analysis':
			return await self._simulate_cost_analysis(scenario)
		else:
			raise ValueError(f"Unknown scenario type: {scenario_type}")
	
	async def _simulate_load_test(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
		"""Simulate load testing scenario."""
		transaction_count = scenario.get('transaction_count', 1000)
		concurrent_requests = scenario.get('concurrent_requests', 50)
		
		results = {
			'total_transactions': transaction_count,
			'concurrent_requests': concurrent_requests,
			'provider_distribution': {},
			'strategy_usage': {},
			'estimated_total_cost': Decimal('0'),
			'estimated_success_rate': 0.0,
			'estimated_avg_latency': 0.0
		}
		
		# Generate sample transactions
		sample_transactions = []
		for i in range(transaction_count):
			transaction = {
				'id': uuid7str(),
				'amount': float(50 + (i % 450)),  # $50-$500 range
				'currency': 'USD',
				'country': 'US',
				'payment_method': PaymentMethod.CARD.value
			}
			sample_transactions.append(transaction)
		
		# Process in batches to simulate concurrency
		batch_size = concurrent_requests
		total_cost = Decimal('0')
		total_success_rate = 0.0
		total_latency = 0.0
		
		for i in range(0, transaction_count, batch_size):
			batch = sample_transactions[i:i + batch_size]
			
			# Process batch concurrently
			batch_results = await asyncio.gather(*[
				self.calculate_optimal_route(tx) for tx in batch
			])
			
			for decision in batch_results:
				# Update distribution tracking
				provider_id = decision.primary_provider
				results['provider_distribution'][provider_id] = results['provider_distribution'].get(provider_id, 0) + 1
				
				strategy = decision.routing_strategy.value
				results['strategy_usage'][strategy] = results['strategy_usage'].get(strategy, 0) + 1
				
				# Accumulate metrics
				total_cost += decision.estimated_cost
				total_success_rate += decision.estimated_success_rate
				total_latency += decision.estimated_latency
		
		# Calculate averages
		results['estimated_total_cost'] = float(total_cost)
		results['estimated_success_rate'] = total_success_rate / transaction_count
		results['estimated_avg_latency'] = total_latency / transaction_count
		
		return results
	
	async def _simulate_provider_failure(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
		"""Simulate provider failure scenario."""
		failed_provider_id = scenario.get('failed_provider_id')
		if not failed_provider_id or failed_provider_id not in self._providers:
			raise ValueError("Invalid or missing failed_provider_id")
		
		# Temporarily mark provider as inactive
		original_status = self._providers[failed_provider_id].is_active
		self._providers[failed_provider_id].is_active = False
		
		try:
			# Test sample transactions
			sample_transactions = [
				{
					'id': uuid7str(),
					'amount': 100.0,
					'currency': 'USD',
					'country': 'US',
					'payment_method': PaymentMethod.CARD.value
				}
				for _ in range(100)
			]
			
			results = {
				'failed_provider_id': failed_provider_id,
				'total_test_transactions': len(sample_transactions),
				'fallback_distribution': {},
				'estimated_impact': {
					'cost_increase': Decimal('0'),
					'success_rate_change': 0.0,
					'latency_change': 0.0
				}
			}
			
			# Process transactions with failed provider
			fallback_decisions = await asyncio.gather(*[
				self.calculate_optimal_route(tx) for tx in sample_transactions
			])
			
			# Analyze fallback behavior
			for decision in fallback_decisions:
				fallback_provider = decision.primary_provider
				results['fallback_distribution'][fallback_provider] = results['fallback_distribution'].get(fallback_provider, 0) + 1
			
			# Calculate impact metrics (simplified)
			results['estimated_impact']['cost_increase'] = float(sum(d.estimated_cost for d in fallback_decisions) / len(fallback_decisions))
			results['estimated_impact']['success_rate_change'] = sum(d.estimated_success_rate for d in fallback_decisions) / len(fallback_decisions)
			results['estimated_impact']['latency_change'] = sum(d.estimated_latency for d in fallback_decisions) / len(fallback_decisions)
			
			return results
			
		finally:
			# Restore original provider status
			self._providers[failed_provider_id].is_active = original_status
	
	async def _simulate_cost_analysis(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
		"""Simulate cost analysis scenario."""
		transaction_volumes = scenario.get('volumes', [100, 1000, 10000])
		
		results = {
			'volume_analysis': {},
			'cost_breakdown': {},
			'optimization_potential': {}
		}
		
		for volume in transaction_volumes:
			# Generate transactions
			transactions = [
				{
					'id': uuid7str(),
					'amount': 75.0,  # Fixed amount for consistent comparison
					'currency': 'USD',
					'country': 'US',
					'payment_method': PaymentMethod.CARD.value
				}
				for _ in range(volume)
			]
			
			# Test with different strategies
			strategies = [RoutingStrategy.COST_OPTIMIZATION, RoutingStrategy.BALANCED]
			volume_results = {}
			
			for strategy in strategies:
				# Temporarily change default strategy
				original_strategy = self._config.default_strategy
				self._config.default_strategy = strategy
				
				try:
					decisions = await asyncio.gather(*[
						self.calculate_optimal_route(tx) for tx in transactions
					])
					
					total_cost = sum(d.estimated_cost for d in decisions)
					avg_success_rate = sum(d.estimated_success_rate for d in decisions) / len(decisions)
					
					volume_results[strategy.value] = {
						'total_cost': float(total_cost),
						'average_cost': float(total_cost / volume),
						'average_success_rate': avg_success_rate
					}
					
				finally:
					self._config.default_strategy = original_strategy
			
			results['volume_analysis'][volume] = volume_results
		
		return results
	
	async def get_orchestration_status(self) -> Dict[str, Any]:
		"""Get current orchestration service status."""
		return {
			'service_status': 'active',
			'total_providers': len(self._providers),
			'active_providers': sum(1 for p in self._providers.values() if p.is_active),
			'total_rules': len([r for r in self._rules.values() if r.is_active]),
			'configuration': {
				'default_strategy': self._config.default_strategy.value,
				'failover_enabled': self._config.failover_enabled,
				'retry_attempts': self._config.retry_attempts,
				'success_rate_threshold': self._config.min_success_rate_threshold
			},
			'active_transactions': len(self._active_transactions),
			'total_routing_decisions': len(self._routing_decisions),
			'uptime': "24/7",  # In real implementation, track actual uptime
			'last_updated': datetime.utcnow().isoformat()
		}