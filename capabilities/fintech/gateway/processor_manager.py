"""
Payment Processor Manager - Multi-Processor Orchestration Engine

Intelligent routing, failover, and management of multiple payment processors
including MPESA, Stripe, Adyen, and PayPal for APG payment gateway.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Type
from enum import Enum
from uuid_extensions import uuid7str

from .payment_processor import (
	AbstractPaymentProcessor,
	PaymentResult,
	ProcessorHealth,
	ProcessorStatus,
	ProcessorCapability
)
from .models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType
from .mpesa_processor import MPESAPaymentProcessor, create_mpesa_processor
from .stripe_processor import StripePaymentProcessor, create_stripe_processor
from .adyen_processor import AdyenPaymentProcessor, create_adyen_processor
from .paypal_processor import PayPalPaymentProcessor, create_paypal_processor

class RoutingStrategy(str, Enum):
	"""Payment processor routing strategy"""
	ROUND_ROBIN = "round_robin"
	LEAST_LOADED = "least_loaded"
	BEST_SUCCESS_RATE = "best_success_rate"
	LOWEST_COST = "lowest_cost"
	GEOGRAPHIC = "geographic"
	PAYMENT_METHOD_OPTIMIZED = "payment_method_optimized"
	SMART_ROUTING = "smart_routing"

class ProcessorPriority(str, Enum):
	"""Processor priority levels"""
	PRIMARY = "primary"
	SECONDARY = "secondary"
	FALLBACK = "fallback"
	DISABLED = "disabled"

class PaymentProcessorManager:
	"""
	Intelligent payment processor orchestration and management
	
	Handles multiple payment processors with smart routing, automatic failover,
	performance monitoring, and adaptive optimization.
	"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.manager_id = uuid7str()
		
		# Processor registry
		self._processors: Dict[str, AbstractPaymentProcessor] = {}
		self._processor_configs: Dict[str, Dict[str, Any]] = {}
		self._processor_priorities: Dict[str, ProcessorPriority] = {}
		
		# Routing configuration
		self.routing_strategy = RoutingStrategy(config.get("routing_strategy", RoutingStrategy.SMART_ROUTING))
		self.enable_failover = config.get("enable_failover", True)
		self.max_retry_attempts = config.get("max_retry_attempts", 3)
		self.circuit_breaker_threshold = config.get("circuit_breaker_threshold", 0.5)  # 50% failure rate
		
		# Performance tracking
		self._routing_stats: Dict[str, Dict[str, Any]] = {}
		self._last_used_processor: Optional[str] = None
		self._round_robin_index = 0
		
		# Circuit breaker state
		self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
		
		self._initialized = False
		
		self._log_manager_created()
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize payment processor manager and all processors"""
		self._log_manager_initialization_start()
		
		try:
			# Initialize processors based on configuration
			await self._initialize_processors()
			
			# Set up routing rules
			await self._setup_routing_rules()
			
			# Initialize monitoring
			await self._setup_monitoring()
			
			self._initialized = True
			
			self._log_manager_initialization_complete()
			
			return {
				"status": "initialized",
				"manager_id": self.manager_id,
				"processors_count": len(self._processors),
				"active_processors": [name for name, proc in self._processors.items() 
									if proc._health.status == ProcessorStatus.HEALTHY],
				"routing_strategy": self.routing_strategy.value,
				"failover_enabled": self.enable_failover
			}
			
		except Exception as e:
			self._log_manager_initialization_error(str(e))
			raise
	
	async def process_payment(
		self,
		transaction: PaymentTransaction,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any] | None = None,
		preferred_processor: str | None = None
	) -> PaymentResult:
		"""
		Process payment with intelligent processor routing
		
		Args:
			transaction: Payment transaction to process
			payment_method: Payment method to use
			additional_data: Additional processor-specific data
			preferred_processor: Preferred processor name (optional)
			
		Returns:
			PaymentResult with transaction outcome
		"""
		if not self._initialized:
			raise RuntimeError("Payment processor manager not initialized")
		
		self._log_payment_routing_start(transaction.id, payment_method.type)
		
		# Determine processing order
		processor_order = await self._determine_processor_order(
			transaction, payment_method, preferred_processor
		)
		
		last_error = None
		for attempt, processor_name in enumerate(processor_order):
			processor = self._processors.get(processor_name)
			if not processor:
				continue
			
			# Check circuit breaker
			if self._is_circuit_breaker_open(processor_name):
				self._log_circuit_breaker_skip(processor_name)
				continue
			
			try:
				self._log_processor_attempt(processor_name, attempt + 1, transaction.id)
				
				# Process payment
				result = await processor.process_payment(transaction, payment_method, additional_data)
				
				# Record routing stats
				await self._record_routing_attempt(processor_name, result.success, result.error_code)
				
				if result.success:
					self._log_payment_routing_success(transaction.id, processor_name)
					return result
				else:
					last_error = result
					self._log_processor_failure(processor_name, result.error_message or "Unknown error")
					
					# Update circuit breaker
					await self._update_circuit_breaker(processor_name, False)
					
					# If this is not a retriable error, break
					if not await self._is_retriable_error(result):
						break
			
			except Exception as e:
				error_message = str(e)
				self._log_processor_exception(processor_name, error_message)
				
				# Update circuit breaker
				await self._update_circuit_breaker(processor_name, False)
				
				last_error = PaymentResult(
					success=False,
					status=PaymentStatus.FAILED,
					error_code="processor_exception",
					error_message=error_message
				)
		
		# All processors failed
		self._log_payment_routing_failed(transaction.id, len(processor_order))
		
		return last_error or PaymentResult(
			success=False,
			status=PaymentStatus.FAILED,
			error_code="all_processors_failed",
			error_message="All payment processors failed to process the transaction"
		)
	
	async def capture_payment(
		self,
		transaction_id: str,
		amount: int | None = None,
		processor_name: str | None = None
	) -> PaymentResult:
		"""Capture payment using specified or detected processor"""
		if processor_name:
			processor = self._processors.get(processor_name)
			if processor:
				return await processor.capture_payment(transaction_id, amount)
		
		# Try to capture with all processors if processor not specified
		for name, processor in self._processors.items():
			try:
				result = await processor.capture_payment(transaction_id, amount)
				if result.success:
					return result
			except Exception:
				continue
		
		return PaymentResult(
			success=False,
			status=PaymentStatus.FAILED,
			error_code="capture_failed",
			error_message="Failed to capture payment with any processor"
		)
	
	async def refund_payment(
		self,
		transaction_id: str,
		amount: int | None = None,
		reason: str | None = None,
		processor_name: str | None = None
	) -> PaymentResult:
		"""Refund payment using specified or detected processor"""
		if processor_name:
			processor = self._processors.get(processor_name)
			if processor:
				return await processor.refund_payment(transaction_id, amount, reason)
		
		# Try to refund with all processors if processor not specified
		for name, processor in self._processors.items():
			try:
				result = await processor.refund_payment(transaction_id, amount, reason)
				if result.success:
					return result
			except Exception:
				continue
		
		return PaymentResult(
			success=False,
			status=PaymentStatus.FAILED,
			error_code="refund_failed",
			error_message="Failed to refund payment with any processor"
		)
	
	async def get_processor_health(self) -> Dict[str, ProcessorHealth]:
		"""Get health status of all processors"""
		health_status = {}
		
		for name, processor in self._processors.items():
			try:
				health_status[name] = await processor.health_check()
			except Exception as e:
				# Create error health status
				health_status[name] = ProcessorHealth(
					status=ProcessorStatus.ERROR,
					last_error=str(e)
				)
		
		return health_status
	
	async def get_routing_statistics(self) -> Dict[str, Any]:
		"""Get routing and performance statistics"""
		return {
			"routing_strategy": self.routing_strategy.value,
			"total_processors": len(self._processors),
			"active_processors": len([p for p in self._processors.values() 
									if p._health.status == ProcessorStatus.HEALTHY]),
			"processor_stats": self._routing_stats,
			"circuit_breakers": self._circuit_breakers,
			"last_used_processor": self._last_used_processor
		}
	
	# Processor management methods
	
	async def add_processor(
		self,
		name: str,
		processor_type: str,
		config: Dict[str, Any],
		priority: ProcessorPriority = ProcessorPriority.SECONDARY
	) -> bool:
		"""Add a new payment processor"""
		try:
			processor = await self._create_processor(processor_type, config)
			await processor.initialize()
			
			self._processors[name] = processor
			self._processor_configs[name] = config
			self._processor_priorities[name] = priority
			self._routing_stats[name] = {"attempts": 0, "successes": 0, "failures": 0}
			
			self._log_processor_added(name, processor_type)
			return True
			
		except Exception as e:
			self._log_processor_add_error(name, str(e))
			return False
	
	async def remove_processor(self, name: str) -> bool:
		"""Remove a payment processor"""
		if name in self._processors:
			del self._processors[name]
			del self._processor_configs[name]
			del self._processor_priorities[name]
			if name in self._routing_stats:
				del self._routing_stats[name]
			
			self._log_processor_removed(name)
			return True
		
		return False
	
	async def update_processor_priority(self, name: str, priority: ProcessorPriority) -> bool:
		"""Update processor priority"""
		if name in self._processors:
			self._processor_priorities[name] = priority
			self._log_processor_priority_updated(name, priority)
			return True
		
		return False
	
	# Private helper methods
	
	async def _initialize_processors(self):
		"""Initialize all configured processors"""
		processor_configs = self.config.get("processors", {})
		
		for name, config in processor_configs.items():
			processor_type = config.get("type")
			if not processor_type:
				continue
			
			try:
				processor = await self._create_processor(processor_type, config)
				await processor.initialize()
				
				self._processors[name] = processor
				self._processor_configs[name] = config
				self._processor_priorities[name] = ProcessorPriority(
					config.get("priority", ProcessorPriority.SECONDARY)
				)
				self._routing_stats[name] = {"attempts": 0, "successes": 0, "failures": 0}
				
				self._log_processor_initialized(name, processor_type)
				
			except Exception as e:
				self._log_processor_initialization_error(name, str(e))
	
	async def _create_processor(self, processor_type: str, config: Dict[str, Any]) -> AbstractPaymentProcessor:
		"""Create processor instance based on type"""
		processor_factories = {
			"mpesa": create_mpesa_processor,
			"stripe": create_stripe_processor,
			"adyen": create_adyen_processor,
			"paypal": create_paypal_processor
		}
		
		factory = processor_factories.get(processor_type.lower())
		if not factory:
			raise ValueError(f"Unknown processor type: {processor_type}")
		
		return factory(config)
	
	async def _determine_processor_order(
		self,
		transaction: PaymentTransaction,
		payment_method: PaymentMethod,
		preferred_processor: str | None
	) -> List[str]:
		"""Determine the order of processors to try"""
		available_processors = await self._get_available_processors(payment_method.type)
		
		if preferred_processor and preferred_processor in available_processors:
			# Move preferred processor to front
			available_processors.remove(preferred_processor)
			available_processors.insert(0, preferred_processor)
			return available_processors
		
		if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
			return await self._round_robin_routing(available_processors)
		elif self.routing_strategy == RoutingStrategy.BEST_SUCCESS_RATE:
			return await self._success_rate_routing(available_processors)
		elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
			return await self._least_loaded_routing(available_processors)
		elif self.routing_strategy == RoutingStrategy.PAYMENT_METHOD_OPTIMIZED:
			return await self._payment_method_routing(available_processors, payment_method.type)
		elif self.routing_strategy == RoutingStrategy.SMART_ROUTING:
			return await self._smart_routing(available_processors, transaction, payment_method)
		else:
			# Default to priority-based routing
			return await self._priority_routing(available_processors)
	
	async def _get_available_processors(self, payment_method_type: PaymentMethodType) -> List[str]:
		"""Get processors that support the payment method and are healthy"""
		available = []
		
		for name, processor in self._processors.items():
			# Check if processor is disabled
			if self._processor_priorities.get(name) == ProcessorPriority.DISABLED:
				continue
			
			# Check if processor supports payment method
			if payment_method_type not in processor.supported_payment_methods:
				continue
			
			# Check if processor is healthy (not under circuit breaker)
			if self._is_circuit_breaker_open(name):
				continue
			
			available.append(name)
		
		return available
	
	async def _round_robin_routing(self, processors: List[str]) -> List[str]:
		"""Round-robin routing strategy"""
		if not processors:
			return []
		
		# Sort by priority first
		processors = await self._priority_routing(processors)
		
		# Apply round-robin within same priority
		self._round_robin_index = (self._round_robin_index + 1) % len(processors)
		return processors[self._round_robin_index:] + processors[:self._round_robin_index]
	
	async def _success_rate_routing(self, processors: List[str]) -> List[str]:
		"""Route based on success rate"""
		processor_success_rates = []
		
		for name in processors:
			stats = self._routing_stats.get(name, {"attempts": 0, "successes": 0})
			if stats["attempts"] > 0:
				success_rate = stats["successes"] / stats["attempts"]
			else:
				success_rate = 1.0  # New processors get benefit of doubt
			
			processor_success_rates.append((name, success_rate))
		
		# Sort by success rate (descending)
		processor_success_rates.sort(key=lambda x: x[1], reverse=True)
		return [name for name, _ in processor_success_rates]
	
	async def _least_loaded_routing(self, processors: List[str]) -> List[str]:
		"""Route to least loaded processor"""
		processor_loads = []
		
		for name in processors:
			processor = self._processors[name]
			# Use transaction count as load metric
			load = processor._transaction_count
			processor_loads.append((name, load))
		
		# Sort by load (ascending)
		processor_loads.sort(key=lambda x: x[1])
		return [name for name, _ in processor_loads]
	
	async def _payment_method_routing(self, processors: List[str], payment_method_type: PaymentMethodType) -> List[str]:
		"""Route based on payment method optimization"""
		# MPESA optimized for mobile money
		if payment_method_type == PaymentMethodType.MPESA:
			mpesa_processors = [p for p in processors if p.lower() == "mpesa"]
			other_processors = [p for p in processors if p.lower() != "mpesa"]
			return mpesa_processors + other_processors
		
		# PayPal optimized for PayPal payments
		elif payment_method_type == PaymentMethodType.PAYPAL:
			paypal_processors = [p for p in processors if p.lower() == "paypal"]
			other_processors = [p for p in processors if p.lower() != "paypal"]
			return paypal_processors + other_processors
		
		# Cards optimized for Stripe/Adyen
		elif payment_method_type in [PaymentMethodType.CREDIT_CARD, PaymentMethodType.DEBIT_CARD]:
			card_processors = [p for p in processors if p.lower() in ["stripe", "adyen"]]
			other_processors = [p for p in processors if p.lower() not in ["stripe", "adyen"]]
			return card_processors + other_processors
		
		else:
			return await self._priority_routing(processors)
	
	async def _smart_routing(
		self,
		processors: List[str],
		transaction: PaymentTransaction,
		payment_method: PaymentMethod
	) -> List[str]:
		"""Advanced smart routing combining multiple factors"""
		processor_scores = []
		
		for name in processors:
			score = 0.0
			
			# Success rate factor (40%)
			stats = self._routing_stats.get(name, {"attempts": 0, "successes": 0})
			if stats["attempts"] > 0:
				success_rate = stats["successes"] / stats["attempts"]
			else:
				success_rate = 1.0
			score += success_rate * 0.4
			
			# Payment method optimization (30%)
			processor = self._processors[name]
			if payment_method.type == PaymentMethodType.MPESA and name.lower() == "mpesa":
				score += 0.3
			elif payment_method.type == PaymentMethodType.PAYPAL and name.lower() == "paypal":
				score += 0.3
			elif payment_method.type in [PaymentMethodType.CREDIT_CARD, PaymentMethodType.DEBIT_CARD] and name.lower() in ["stripe", "adyen"]:
				score += 0.3
			
			# Response time factor (20%)
			avg_response_time = processor._calculate_average_response_time()
			if avg_response_time > 0:
				# Lower response time = higher score
				response_time_score = max(0, 1 - (avg_response_time / 5000))  # 5 seconds max
				score += response_time_score * 0.2
			else:
				score += 0.2  # New processors get benefit
			
			# Priority factor (10%)
			priority = self._processor_priorities.get(name, ProcessorPriority.SECONDARY)
			if priority == ProcessorPriority.PRIMARY:
				score += 0.1
			elif priority == ProcessorPriority.SECONDARY:
				score += 0.05
			
			processor_scores.append((name, score))
		
		# Sort by score (descending)
		processor_scores.sort(key=lambda x: x[1], reverse=True)
		return [name for name, _ in processor_scores]
	
	async def _priority_routing(self, processors: List[str]) -> List[str]:
		"""Route based on processor priority"""
		primary = []
		secondary = []
		fallback = []
		
		for name in processors:
			priority = self._processor_priorities.get(name, ProcessorPriority.SECONDARY)
			if priority == ProcessorPriority.PRIMARY:
				primary.append(name)
			elif priority == ProcessorPriority.SECONDARY:
				secondary.append(name)
			elif priority == ProcessorPriority.FALLBACK:
				fallback.append(name)
		
		return primary + secondary + fallback
	
	async def _setup_routing_rules(self):
		"""Set up routing rules and configurations"""
		# Initialize circuit breakers
		for name in self._processors:
			self._circuit_breakers[name] = {
				"state": "closed",  # closed, open, half_open
				"failure_count": 0,
				"last_failure_time": None,
				"next_attempt_time": None
			}
	
	async def _setup_monitoring(self):
		"""Set up processor monitoring"""
		# Initialize routing statistics
		for name in self._processors:
			if name not in self._routing_stats:
				self._routing_stats[name] = {"attempts": 0, "successes": 0, "failures": 0}
	
	async def _record_routing_attempt(self, processor_name: str, success: bool, error_code: str | None):
		"""Record routing attempt statistics"""
		if processor_name not in self._routing_stats:
			self._routing_stats[processor_name] = {"attempts": 0, "successes": 0, "failures": 0}
		
		stats = self._routing_stats[processor_name]
		stats["attempts"] += 1
		
		if success:
			stats["successes"] += 1
			await self._update_circuit_breaker(processor_name, True)
		else:
			stats["failures"] += 1
			await self._update_circuit_breaker(processor_name, False)
		
		self._last_used_processor = processor_name
	
	async def _is_retriable_error(self, result: PaymentResult) -> bool:
		"""Determine if error is retriable"""
		# Non-retriable errors
		non_retriable_codes = [
			"insufficient_funds",
			"card_declined",
			"invalid_card",
			"expired_card",
			"authentication_failed"
		]
		
		return result.error_code not in non_retriable_codes
	
	def _is_circuit_breaker_open(self, processor_name: str) -> bool:
		"""Check if circuit breaker is open for processor"""
		breaker = self._circuit_breakers.get(processor_name)
		if not breaker:
			return False
		
		if breaker["state"] == "open":
			# Check if it's time to try again (half-open)
			if breaker["next_attempt_time"] and datetime.now() >= breaker["next_attempt_time"]:
				breaker["state"] = "half_open"
				return False
			return True
		
		return False
	
	async def _update_circuit_breaker(self, processor_name: str, success: bool):
		"""Update circuit breaker state"""
		breaker = self._circuit_breakers.get(processor_name)
		if not breaker:
			return
		
		if success:
			# Reset on success
			breaker["failure_count"] = 0
			breaker["state"] = "closed"
		else:
			# Increment failure count
			breaker["failure_count"] += 1
			breaker["last_failure_time"] = datetime.now()
			
			# Open circuit breaker if threshold exceeded
			if breaker["failure_count"] >= self.max_retry_attempts:
				breaker["state"] = "open"
				# Next attempt in 5 minutes
				breaker["next_attempt_time"] = datetime.now() + timedelta(minutes=5)
	
	# Logging methods following APG patterns
	
	def _log_manager_created(self):
		"""Log manager creation"""
		print(f"🎯 Payment Processor Manager created")
		print(f"   Manager ID: {self.manager_id}")
		print(f"   Routing Strategy: {self.routing_strategy.value}")
		print(f"   Failover Enabled: {self.enable_failover}")
	
	def _log_manager_initialization_start(self):
		"""Log manager initialization start"""
		print(f"🚀 Initializing Payment Processor Manager...")
		print(f"   Strategy: {self.routing_strategy.value}")
		print(f"   Max Retries: {self.max_retry_attempts}")
	
	def _log_manager_initialization_complete(self):
		"""Log manager initialization complete"""
		print(f"✅ Payment Processor Manager initialized successfully")
		print(f"   Active Processors: {len(self._processors)}")
		print(f"   Processors: {', '.join(self._processors.keys())}")
	
	def _log_manager_initialization_error(self, error: str):
		"""Log manager initialization error"""
		print(f"❌ Payment Processor Manager initialization failed: {error}")
	
	def _log_payment_routing_start(self, transaction_id: str, payment_method_type: PaymentMethodType):
		"""Log payment routing start"""
		print(f"🎯 Routing payment: {transaction_id}")
		print(f"   Method: {payment_method_type.value}")
		print(f"   Strategy: {self.routing_strategy.value}")
	
	def _log_payment_routing_success(self, transaction_id: str, processor_name: str):
		"""Log successful payment routing"""
		print(f"✅ Payment routed successfully: {transaction_id}")
		print(f"   Processor: {processor_name}")
	
	def _log_payment_routing_failed(self, transaction_id: str, attempts: int):
		"""Log failed payment routing"""
		print(f"❌ Payment routing failed: {transaction_id}")
		print(f"   Attempts: {attempts}")
		print(f"   All processors exhausted")
	
	def _log_processor_attempt(self, processor_name: str, attempt: int, transaction_id: str):
		"""Log processor attempt"""
		print(f"🔄 Attempting processor: {processor_name} (attempt {attempt}) for {transaction_id}")
	
	def _log_processor_failure(self, processor_name: str, error: str):
		"""Log processor failure"""
		print(f"❌ Processor failed: {processor_name} - {error}")
	
	def _log_processor_exception(self, processor_name: str, error: str):
		"""Log processor exception"""
		print(f"💥 Processor exception: {processor_name} - {error}")
	
	def _log_circuit_breaker_skip(self, processor_name: str):
		"""Log circuit breaker skip"""
		print(f"🚫 Circuit breaker open, skipping: {processor_name}")
	
	def _log_processor_added(self, name: str, processor_type: str):
		"""Log processor added"""
		print(f"➕ Processor added: {name} ({processor_type})")
	
	def _log_processor_removed(self, name: str):
		"""Log processor removed"""
		print(f"➖ Processor removed: {name}")
	
	def _log_processor_priority_updated(self, name: str, priority: ProcessorPriority):
		"""Log processor priority update"""
		print(f"🔄 Processor priority updated: {name} -> {priority.value}")
	
	def _log_processor_initialized(self, name: str, processor_type: str):
		"""Log processor initialization"""
		print(f"✅ Processor initialized: {name} ({processor_type})")
	
	def _log_processor_initialization_error(self, name: str, error: str):
		"""Log processor initialization error"""
		print(f"❌ Processor initialization failed: {name} - {error}")
	
	def _log_processor_add_error(self, name: str, error: str):
		"""Log processor add error"""
		print(f"❌ Failed to add processor: {name} - {error}")

# Factory function for creating processor manager
def create_processor_manager(config: Dict[str, Any]) -> PaymentProcessorManager:
	"""Factory function to create payment processor manager"""
	return PaymentProcessorManager(config)

def _log_processor_manager_module_loaded():
	"""Log processor manager module loaded"""
	print("🎯 Payment Processor Manager module loaded")
	print("   - Multi-processor orchestration")
	print("   - Intelligent routing strategies")
	print("   - Automatic failover & circuit breakers")
	print("   - Performance monitoring & optimization")

# Execute module loading log
_log_processor_manager_module_loaded()