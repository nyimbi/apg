"""
Abstract Payment Processor Base Class - APG Payment Gateway

Provides the foundational interface for all payment processors in the APG
payment gateway, enabling seamless multi-processor orchestration and routing.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from uuid_extensions import uuid7str

from .models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType

class ProcessorCapability(str, Enum):
	"""Payment processor capability enumeration"""
	AUTHORIZATION = "authorization"
	CAPTURE = "capture"
	REFUND = "refund"
	PARTIAL_REFUND = "partial_refund"
	VOID = "void"
	RECURRING = "recurring"
	SUBSCRIPTION = "subscription"
	TOKENIZATION = "tokenization"
	THREE_D_SECURE = "3ds"
	FRAUD_PROTECTION = "fraud_protection"
	MULTI_CURRENCY = "multi_currency"
	INSTALLMENTS = "installments"
	MOBILE_MONEY = "mobile_money"
	BANK_TRANSFER = "bank_transfer"
	DIGITAL_WALLET = "digital_wallet"
	CRYPTOCURRENCY = "cryptocurrency"

class ProcessorStatus(str, Enum):
	"""Payment processor status enumeration"""
	INITIALIZING = "initializing"
	HEALTHY = "healthy"
	DEGRADED = "degraded"
	UNAVAILABLE = "unavailable"
	MAINTENANCE = "maintenance"
	ERROR = "error"

class PaymentResult:
	"""
	Standardized payment result across all processors
	"""
	
	def __init__(
		self,
		success: bool,
		status: PaymentStatus,
		processor_transaction_id: str | None = None,
		processor_response: Dict[str, Any] | None = None,
		error_code: str | None = None,
		error_message: str | None = None,
		requires_action: bool = False,
		action_type: str | None = None,
		action_data: Dict[str, Any] | None = None,
		metadata: Dict[str, Any] | None = None
	):
		self.success = success
		self.status = status
		self.processor_transaction_id = processor_transaction_id
		self.processor_response = processor_response or {}
		self.error_code = error_code
		self.error_message = error_message
		self.requires_action = requires_action
		self.action_type = action_type
		self.action_data = action_data or {}
		self.metadata = metadata or {}
		self.timestamp = datetime.now(timezone.utc)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert payment result to dictionary"""
		return {
			"success": self.success,
			"status": self.status.value if isinstance(self.status, PaymentStatus) else self.status,
			"processor_transaction_id": self.processor_transaction_id,
			"processor_response": self.processor_response,
			"error_code": self.error_code,
			"error_message": self.error_message,
			"requires_action": self.requires_action,
			"action_type": self.action_type,
			"action_data": self.action_data,
			"metadata": self.metadata,
			"timestamp": self.timestamp.isoformat()
		}

class ProcessorHealth:
	"""
	Payment processor health status and metrics
	"""
	
	def __init__(
		self,
		status: ProcessorStatus,
		success_rate: float = 0.0,
		average_response_time: float = 0.0,
		last_successful_transaction: datetime | None = None,
		error_count: int = 0,
		last_error: str | None = None,
		uptime_percentage: float = 100.0,
		supported_currencies: List[str] | None = None,
		supported_countries: List[str] | None = None,
		maintenance_window: Dict[str, Any] | None = None
	):
		self.status = status
		self.success_rate = success_rate
		self.average_response_time = average_response_time
		self.last_successful_transaction = last_successful_transaction
		self.error_count = error_count
		self.last_error = last_error
		self.uptime_percentage = uptime_percentage
		self.supported_currencies = supported_currencies or []
		self.supported_countries = supported_countries or []
		self.maintenance_window = maintenance_window or {}
		self.last_check = datetime.now(timezone.utc)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert processor health to dictionary"""
		return {
			"status": self.status.value,
			"success_rate": self.success_rate,
			"average_response_time": self.average_response_time,
			"last_successful_transaction": self.last_successful_transaction.isoformat() if self.last_successful_transaction else None,
			"error_count": self.error_count,
			"last_error": self.last_error,
			"uptime_percentage": self.uptime_percentage,
			"supported_currencies": self.supported_currencies,
			"supported_countries": self.supported_countries,
			"maintenance_window": self.maintenance_window,
			"last_check": self.last_check.isoformat()
		}

class AbstractPaymentProcessor(ABC):
	"""
	Abstract base class for all payment processors in APG Payment Gateway
	
	This class defines the standard interface that all payment processors must implement,
	ensuring consistent behavior across different payment providers (Stripe, Adyen, MPESA, etc.)
	"""
	
	def __init__(
		self,
		processor_name: str,
		supported_payment_methods: List[PaymentMethodType],
		supported_currencies: List[str],
		supported_countries: List[str],
		capabilities: List[ProcessorCapability],
		config: Dict[str, Any] | None = None
	):
		self.processor_name = processor_name
		self.supported_payment_methods = supported_payment_methods
		self.supported_currencies = supported_currencies
		self.supported_countries = supported_countries
		self.capabilities = capabilities
		self.config = config or {}
		
		# Internal state
		self._initialized = False
		self._health = ProcessorHealth(ProcessorStatus.INITIALIZING)
		self.processor_id = uuid7str()
		
		# Performance metrics
		self._transaction_count = 0
		self._success_count = 0
		self._error_count = 0
		self._total_response_time = 0.0
		
		self._log_processor_created()
	
	@abstractmethod
	async def initialize(self) -> Dict[str, Any]:
		"""
		Initialize the payment processor
		
		Returns:
			Dict containing initialization status and configuration
		"""
		pass
	
	@abstractmethod
	async def process_payment(
		self,
		transaction: PaymentTransaction,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any] | None = None
	) -> PaymentResult:
		"""
		Process a payment transaction
		
		Args:
			transaction: Payment transaction to process
			payment_method: Payment method to use
			additional_data: Additional processor-specific data
			
		Returns:
			PaymentResult with transaction outcome
		"""
		pass
	
	@abstractmethod
	async def capture_payment(
		self,
		transaction_id: str,
		amount: int | None = None
	) -> PaymentResult:
		"""
		Capture a previously authorized payment
		
		Args:
			transaction_id: ID of transaction to capture
			amount: Amount to capture (None for full amount)
			
		Returns:
			PaymentResult with capture outcome
		"""
		pass
	
	@abstractmethod
	async def refund_payment(
		self,
		transaction_id: str,
		amount: int | None = None,
		reason: str | None = None
	) -> PaymentResult:
		"""
		Refund a completed payment
		
		Args:
			transaction_id: ID of transaction to refund
			amount: Amount to refund (None for full amount)
			reason: Reason for refund
			
		Returns:
			PaymentResult with refund outcome
		"""
		pass
	
	@abstractmethod
	async def void_payment(self, transaction_id: str) -> PaymentResult:
		"""
		Void an authorized but not captured payment
		
		Args:
			transaction_id: ID of transaction to void
			
		Returns:
			PaymentResult with void outcome
		"""
		pass
	
	@abstractmethod
	async def get_transaction_status(self, transaction_id: str) -> PaymentResult:
		"""
		Get the current status of a transaction
		
		Args:
			transaction_id: ID of transaction to query
			
		Returns:
			PaymentResult with current transaction status
		"""
		pass
	
	@abstractmethod
	async def tokenize_payment_method(
		self,
		payment_method_data: Dict[str, Any],
		customer_id: str
	) -> Dict[str, Any]:
		"""
		Tokenize payment method for future use
		
		Args:
			payment_method_data: Raw payment method data
			customer_id: Customer ID for tokenization
			
		Returns:
			Dict containing tokenization result
		"""
		pass
	
	@abstractmethod
	async def health_check(self) -> ProcessorHealth:
		"""
		Perform health check and return processor status
		
		Returns:
			ProcessorHealth with current status and metrics
		"""
		pass
	
	# Common functionality implemented in base class
	
	async def supports_payment_method(self, payment_method_type: PaymentMethodType) -> bool:
		"""Check if processor supports given payment method type"""
		return payment_method_type in self.supported_payment_methods
	
	async def supports_currency(self, currency: str) -> bool:
		"""Check if processor supports given currency"""
		return currency.upper() in [c.upper() for c in self.supported_currencies]
	
	async def supports_country(self, country: str) -> bool:
		"""Check if processor supports given country"""
		return country.upper() in [c.upper() for c in self.supported_countries]
	
	async def has_capability(self, capability: ProcessorCapability) -> bool:
		"""Check if processor has given capability"""
		return capability in self.capabilities
	
	def get_processor_info(self) -> Dict[str, Any]:
		"""Get basic processor information"""
		return {
			"id": self.processor_id,
			"name": self.processor_name,
			"initialized": self._initialized,
			"supported_payment_methods": [pm.value for pm in self.supported_payment_methods],
			"supported_currencies": self.supported_currencies,
			"supported_countries": self.supported_countries,
			"capabilities": [cap.value for cap in self.capabilities],
			"health_status": self._health.status.value,
			"success_rate": self._calculate_success_rate(),
			"average_response_time": self._calculate_average_response_time()
		}
	
	def _record_transaction_start(self) -> datetime:
		"""Record the start of a transaction for metrics"""
		self._transaction_count += 1
		return datetime.now()
	
	def _record_transaction_success(self, start_time: datetime):
		"""Record successful transaction for metrics"""
		self._success_count += 1
		response_time = (datetime.now() - start_time).total_seconds() * 1000
		self._total_response_time += response_time
		self._health.last_successful_transaction = datetime.now(timezone.utc)
		
		# Update health status if it was degraded
		if self._health.status in [ProcessorStatus.DEGRADED, ProcessorStatus.ERROR]:
			self._health.status = ProcessorStatus.HEALTHY
	
	def _record_transaction_error(self, error: str):
		"""Record transaction error for metrics"""
		self._error_count += 1
		self._health.error_count += 1
		self._health.last_error = error
		
		# Update health status based on error rate
		success_rate = self._calculate_success_rate()
		if success_rate < 0.5:  # Less than 50% success rate
			self._health.status = ProcessorStatus.ERROR
		elif success_rate < 0.8:  # Less than 80% success rate
			self._health.status = ProcessorStatus.DEGRADED
	
	def _calculate_success_rate(self) -> float:
		"""Calculate current success rate"""
		if self._transaction_count == 0:
			return 0.0
		return self._success_count / self._transaction_count
	
	def _calculate_average_response_time(self) -> float:
		"""Calculate average response time in milliseconds"""
		if self._success_count == 0:
			return 0.0
		return self._total_response_time / self._success_count
	
	def _update_health_metrics(self):
		"""Update health metrics"""
		self._health.success_rate = self._calculate_success_rate()
		self._health.average_response_time = self._calculate_average_response_time()
		self._health.error_count = self._error_count
		
		# Calculate uptime percentage (simplified)
		if self._health.status == ProcessorStatus.HEALTHY:
			self._health.uptime_percentage = min(100.0, self._health.uptime_percentage + 0.1)
		else:
			self._health.uptime_percentage = max(0.0, self._health.uptime_percentage - 1.0)
	
	# Logging methods following APG patterns
	
	def _log_processor_created(self):
		"""Log processor creation"""
		print(f"⚙️  Payment processor created: {self.processor_name}")
		print(f"   ID: {self.processor_id}")
		print(f"   Methods: {len(self.supported_payment_methods)}")
		print(f"   Currencies: {len(self.supported_currencies)}")
		print(f"   Capabilities: {len(self.capabilities)}")
	
	def _log_transaction_processing(self, transaction_id: str, amount: int, currency: str):
		"""Log transaction processing start"""
		print(f"💳 Processing payment: {transaction_id}")
		print(f"   Processor: {self.processor_name}")
		print(f"   Amount: {amount} {currency}")
	
	def _log_transaction_success(self, transaction_id: str, processor_transaction_id: str | None = None):
		"""Log successful transaction"""
		print(f"✅ Payment processed successfully: {transaction_id}")
		if processor_transaction_id:
			print(f"   Processor ID: {processor_transaction_id}")
	
	def _log_transaction_error(self, transaction_id: str, error: str):
		"""Log transaction error"""
		print(f"❌ Payment processing failed: {transaction_id}")
		print(f"   Error: {error}")
		print(f"   Processor: {self.processor_name}")
	
	def _log_health_check(self, status: ProcessorStatus):
		"""Log health check result"""
		print(f"🏥 Health check: {self.processor_name} - {status.value}")
		if status != ProcessorStatus.HEALTHY:
			print(f"   Success Rate: {self._calculate_success_rate():.1%}")
			print(f"   Error Count: {self._error_count}")

# Utility functions for processor management

def validate_processor_config(config: Dict[str, Any], required_fields: List[str]) -> bool:
	"""
	Validate processor configuration has required fields
	
	Args:
		config: Processor configuration dictionary
		required_fields: List of required field names
		
	Returns:
		True if all required fields are present
	"""
	return all(field in config and config[field] for field in required_fields)

def create_payment_result_success(
	status: PaymentStatus,
	processor_transaction_id: str | None = None,
	metadata: Dict[str, Any] | None = None
) -> PaymentResult:
	"""Create a successful payment result"""
	return PaymentResult(
		success=True,
		status=status,
		processor_transaction_id=processor_transaction_id,
		metadata=metadata or {}
	)

def create_payment_result_error(
	status: PaymentStatus,
	error_code: str,
	error_message: str,
	metadata: Dict[str, Any] | None = None
) -> PaymentResult:
	"""Create an error payment result"""
	return PaymentResult(
		success=False,
		status=status,
		error_code=error_code,
		error_message=error_message,
		metadata=metadata or {}
	)

def create_payment_result_action_required(
	status: PaymentStatus,
	action_type: str,
	action_data: Dict[str, Any],
	processor_transaction_id: str | None = None
) -> PaymentResult:
	"""Create a payment result that requires additional action"""
	return PaymentResult(
		success=False,
		status=status,
		processor_transaction_id=processor_transaction_id,
		requires_action=True,
		action_type=action_type,
		action_data=action_data
	)

def _log_payment_processor_module_loaded():
	"""Log payment processor module loaded"""
	print("⚙️  Abstract Payment Processor module loaded")
	print("   - Standardized processor interface")
	print("   - Health monitoring and metrics")
	print("   - Multi-processor orchestration support")
	print("   - APG composition engine integration")

# Execute module loading log
_log_payment_processor_module_loaded()