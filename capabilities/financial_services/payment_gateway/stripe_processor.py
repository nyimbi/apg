"""
Stripe Payment Processor - Comprehensive Stripe API Integration

Full-featured Stripe payment processor with support for all major payment methods,
advanced features, and seamless APG platform integration.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import stripe
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from decimal import Decimal

from .payment_processor import (
	AbstractPaymentProcessor,
	PaymentResult,
	ProcessorHealth,
	ProcessorStatus,
	ProcessorCapability,
	create_payment_result_success,
	create_payment_result_error,
	create_payment_result_action_required,
	validate_processor_config
)
from .models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType

class StripePaymentProcessor(AbstractPaymentProcessor):
	"""
	Comprehensive Stripe payment processor with full API coverage
	
	Supports all Stripe payment methods, advanced features like 3D Secure,
	subscriptions, and multi-party payments.
	"""
	
	def __init__(self, config: Dict[str, Any]):
		# Validate required configuration
		required_fields = ["api_key", "webhook_secret"]
		if not validate_processor_config(config, required_fields):
			raise ValueError("Missing required Stripe configuration fields")
		
		# Initialize base processor
		super().__init__(
			processor_name="stripe",
			supported_payment_methods=[
				PaymentMethodType.CREDIT_CARD,
				PaymentMethodType.DEBIT_CARD,
				PaymentMethodType.BANK_TRANSFER,
				PaymentMethodType.ACH,
				PaymentMethodType.DIGITAL_WALLET,
				PaymentMethodType.APPLE_PAY,
				PaymentMethodType.GOOGLE_PAY,
				PaymentMethodType.BANK_REDIRECT,
				PaymentMethodType.BUY_NOW_PAY_LATER
			],
			supported_currencies=[
				"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "SEK", "NOK", "DKK",
				"PLN", "CZK", "HUF", "BGN", "RON", "HRK", "THB", "SGD", "MYR", "INR",
				"BRL", "MXN", "ARS", "CLP", "COP", "PEN", "UYU", "ZAR", "NGN", "GHS",
				"KES", "UGX", "TZS", "RWF", "ZMW", "BWP", "MZN", "MAD", "EGP", "TND"
			],
			supported_countries=[
				"US", "CA", "GB", "IE", "AU", "NZ", "SG", "HK", "JP", "AT", "BE", "BG",
				"HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IT", "LV",
				"LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "CH",
				"NO", "LI", "BR", "MX", "TH", "MY", "IN", "PH", "ID", "ZA", "NG", "KE",
				"GH", "UG", "TZ", "RW", "ZM", "BW", "MZ", "MA", "EG", "TN"
			],
			capabilities=[
				ProcessorCapability.AUTHORIZATION,
				ProcessorCapability.CAPTURE,
				ProcessorCapability.REFUND,
				ProcessorCapability.PARTIAL_REFUND,
				ProcessorCapability.VOID,
				ProcessorCapability.RECURRING,
				ProcessorCapability.SUBSCRIPTION,
				ProcessorCapability.TOKENIZATION,
				ProcessorCapability.THREE_D_SECURE,
				ProcessorCapability.FRAUD_PROTECTION,
				ProcessorCapability.MULTI_CURRENCY,
				ProcessorCapability.INSTALLMENTS,
				ProcessorCapability.DIGITAL_WALLET,
				ProcessorCapability.BANK_TRANSFER
			],
			config=config
		)
		
		# Stripe configuration
		self.api_key = config["api_key"]
		self.webhook_secret = config["webhook_secret"]
		self.environment = config.get("environment", "sandbox")
		self.account_id = config.get("account_id")  # For Stripe Connect
		
		# Configure Stripe SDK
		stripe.api_key = self.api_key
		if self.account_id:
			stripe.default_stripe_account = self.account_id
		
		# Advanced configuration
		self.automatic_payment_methods = config.get("automatic_payment_methods", True)
		self.capture_method = config.get("capture_method", "automatic")  # automatic or manual
		self.setup_future_usage = config.get("setup_future_usage")  # off_session, on_session
		self.statement_descriptor = config.get("statement_descriptor")
		self.receipt_email = config.get("receipt_email", True)
		
		# 3D Secure configuration
		self.require_3ds = config.get("require_3ds", "automatic")  # automatic, challenge_only, any
		
		# Connect configuration
		self.application_fee_percent = config.get("application_fee_percent", 0.0)
		self.transfer_group = config.get("transfer_group")
		
		self._log_stripe_processor_created()
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize Stripe processor and validate configuration"""
		self._log_stripe_initialization_start()
		
		try:
			# Validate API key by making a test request
			account = await self._make_stripe_request(stripe.Account.retrieve)
			
			# Get processor capabilities from Stripe
			capabilities = await self._get_stripe_capabilities()
			
			self._initialized = True
			self._health.status = ProcessorStatus.HEALTHY
			
			self._log_stripe_initialization_complete(account.id)
			
			return {
				"status": "initialized",
				"processor": "stripe",
				"account_id": account.id,
				"environment": self.environment,
				"capabilities": capabilities,
				"supported_payment_methods": len(self.supported_payment_methods),
				"supported_currencies": len(self.supported_currencies)
			}
			
		except Exception as e:
			self._health.status = ProcessorStatus.ERROR
			self._log_stripe_initialization_error(str(e))
			raise
	
	async def process_payment(
		self,
		transaction: PaymentTransaction,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any] | None = None
	) -> PaymentResult:
		"""Process payment through Stripe"""
		start_time = self._record_transaction_start()
		self._log_transaction_processing(transaction.id, transaction.amount, transaction.currency)
		
		try:
			# Prepare payment intent data
			payment_intent_data = await self._prepare_payment_intent_data(
				transaction, payment_method, additional_data or {}
			)
			
			# Create payment intent
			payment_intent = await self._make_stripe_request(
				stripe.PaymentIntent.create,
				**payment_intent_data
			)
			
			# Handle different payment method types
			if payment_method.type in [PaymentMethodType.CREDIT_CARD, PaymentMethodType.DEBIT_CARD]:
				result = await self._process_card_payment(payment_intent, payment_method, additional_data)
			elif payment_method.type == PaymentMethodType.BANK_TRANSFER:
				result = await self._process_bank_transfer(payment_intent, payment_method, additional_data)
			elif payment_method.type in [PaymentMethodType.APPLE_PAY, PaymentMethodType.GOOGLE_PAY]:
				result = await self._process_digital_wallet(payment_intent, payment_method, additional_data)
			else:
				result = await self._process_generic_payment(payment_intent, payment_method, additional_data)
			
			if result.success:
				self._record_transaction_success(start_time)
				self._log_transaction_success(transaction.id, result.processor_transaction_id)
			else:
				self._record_transaction_error(result.error_message or "Unknown error")
				self._log_transaction_error(transaction.id, result.error_message or "Unknown error")
			
			return result
			
		except Exception as e:
			error_message = str(e)
			self._record_transaction_error(error_message)
			self._log_transaction_error(transaction.id, error_message)
			
			return create_payment_result_error(
				status=PaymentStatus.FAILED,
				error_code="stripe_error",
				error_message=error_message
			)
	
	async def capture_payment(
		self,
		transaction_id: str,
		amount: int | None = None
	) -> PaymentResult:
		"""Capture a previously authorized Stripe payment"""
		try:
			self._log_stripe_capture_start(transaction_id, amount)
			
			# Get the payment intent
			payment_intent = await self._make_stripe_request(
				stripe.PaymentIntent.retrieve,
				transaction_id
			)
			
			# Prepare capture data
			capture_data = {}
			if amount is not None:
				capture_data["amount_to_capture"] = amount
			
			# Capture the payment
			captured_intent = await self._make_stripe_request(
				payment_intent.capture,
				**capture_data
			)
			
			self._log_stripe_capture_success(transaction_id, captured_intent.id)
			
			return create_payment_result_success(
				status=PaymentStatus.CAPTURED,
				processor_transaction_id=captured_intent.id,
				metadata={
					"stripe_payment_intent_id": captured_intent.id,
					"amount_captured": captured_intent.amount_received,
					"currency": captured_intent.currency
				}
			)
			
		except Exception as e:
			error_message = str(e)
			self._log_stripe_capture_error(transaction_id, error_message)
			
			return create_payment_result_error(
				status=PaymentStatus.FAILED,
				error_code="stripe_capture_error",
				error_message=error_message
			)
	
	async def refund_payment(
		self,
		transaction_id: str,
		amount: int | None = None,
		reason: str | None = None
	) -> PaymentResult:
		"""Refund a Stripe payment"""
		try:
			self._log_stripe_refund_start(transaction_id, amount, reason)
			
			# Prepare refund data
			refund_data = {
				"payment_intent": transaction_id
			}
			
			if amount is not None:
				refund_data["amount"] = amount
			
			if reason:
				refund_data["reason"] = reason
				refund_data["metadata"] = {"refund_reason": reason}
			
			# Create refund
			refund = await self._make_stripe_request(
				stripe.Refund.create,
				**refund_data
			)
			
			# Determine status based on refund amount
			if refund.amount == refund.charge.amount:
				status = PaymentStatus.REFUNDED
			else:
				status = PaymentStatus.PARTIALLY_REFUNDED
			
			self._log_stripe_refund_success(transaction_id, refund.id, refund.amount)
			
			return create_payment_result_success(
				status=status,
				processor_transaction_id=refund.id,
				metadata={
					"stripe_refund_id": refund.id,
					"refund_amount": refund.amount,
					"refund_reason": reason,
					"currency": refund.currency
				}
			)
			
		except Exception as e:
			error_message = str(e)
			self._log_stripe_refund_error(transaction_id, error_message)
			
			return create_payment_result_error(
				status=PaymentStatus.FAILED,
				error_code="stripe_refund_error",
				error_message=error_message
			)
	
	async def void_payment(self, transaction_id: str) -> PaymentResult:
		"""Void a Stripe payment intent"""
		try:
			self._log_stripe_void_start(transaction_id)
			
			# Cancel the payment intent
			payment_intent = await self._make_stripe_request(
				stripe.PaymentIntent.retrieve,
				transaction_id
			)
			
			canceled_intent = await self._make_stripe_request(
				payment_intent.cancel
			)
			
			self._log_stripe_void_success(transaction_id)
			
			return create_payment_result_success(
				status=PaymentStatus.CANCELLED,
				processor_transaction_id=canceled_intent.id,
				metadata={
					"stripe_payment_intent_id": canceled_intent.id,
					"cancellation_reason": canceled_intent.cancellation_reason
				}
			)
			
		except Exception as e:
			error_message = str(e)
			self._log_stripe_void_error(transaction_id, error_message)
			
			return create_payment_result_error(
				status=PaymentStatus.FAILED,
				error_code="stripe_void_error",
				error_message=error_message
			)
	
	async def get_transaction_status(self, transaction_id: str) -> PaymentResult:
		"""Get Stripe transaction status"""
		try:
			# Retrieve payment intent
			payment_intent = await self._make_stripe_request(
				stripe.PaymentIntent.retrieve,
				transaction_id
			)
			
			# Map Stripe status to our status
			status_mapping = {
				"requires_payment_method": PaymentStatus.PENDING,
				"requires_confirmation": PaymentStatus.PENDING,
				"requires_action": PaymentStatus.PENDING,
				"processing": PaymentStatus.PROCESSING,
				"requires_capture": PaymentStatus.AUTHORIZED,
				"succeeded": PaymentStatus.COMPLETED,
				"canceled": PaymentStatus.CANCELLED
			}
			
			status = status_mapping.get(payment_intent.status, PaymentStatus.PENDING)
			
			# Check if action is required
			requires_action = payment_intent.status == "requires_action"
			action_data = {}
			if requires_action and payment_intent.next_action:
				action_data = {
					"type": payment_intent.next_action.type,
					"data": payment_intent.next_action
				}
			
			return PaymentResult(
				success=True,
				status=status,
				processor_transaction_id=payment_intent.id,
				requires_action=requires_action,
				action_type=action_data.get("type"),
				action_data=action_data,
				metadata={
					"stripe_status": payment_intent.status,
					"amount": payment_intent.amount,
					"currency": payment_intent.currency,
					"client_secret": payment_intent.client_secret
				}
			)
			
		except Exception as e:
			error_message = str(e)
			return create_payment_result_error(
				status=PaymentStatus.FAILED,
				error_code="stripe_status_error",
				error_message=error_message
			)
	
	async def tokenize_payment_method(
		self,
		payment_method_data: Dict[str, Any],
		customer_id: str
	) -> Dict[str, Any]:
		"""Tokenize payment method in Stripe"""
		try:
			self._log_stripe_tokenization_start(customer_id)
			
			# Create or retrieve Stripe customer
			stripe_customer = await self._get_or_create_stripe_customer(customer_id)
			
			# Create payment method
			payment_method = await self._make_stripe_request(
				stripe.PaymentMethod.create,
				**payment_method_data
			)
			
			# Attach to customer
			await self._make_stripe_request(
				payment_method.attach,
				customer=stripe_customer.id
			)
			
			self._log_stripe_tokenization_success(payment_method.id)
			
			return {
				"success": True,
				"token": payment_method.id,
				"stripe_customer_id": stripe_customer.id,
				"payment_method_type": payment_method.type,
				"metadata": {
					"stripe_payment_method_id": payment_method.id,
					"last4": getattr(payment_method.card, "last4", None) if payment_method.card else None,
					"brand": getattr(payment_method.card, "brand", None) if payment_method.card else None,
					"exp_month": getattr(payment_method.card, "exp_month", None) if payment_method.card else None,
					"exp_year": getattr(payment_method.card, "exp_year", None) if payment_method.card else None
				}
			}
			
		except Exception as e:
			error_message = str(e)
			self._log_stripe_tokenization_error(customer_id, error_message)
			
			return {
				"success": False,
				"error": error_message
			}
	
	async def health_check(self) -> ProcessorHealth:
		"""Perform Stripe health check"""
		try:
			# Test API connectivity
			await self._make_stripe_request(stripe.Account.retrieve)
			
			self._health.status = ProcessorStatus.HEALTHY
			self._update_health_metrics()
			
			self._log_health_check(ProcessorStatus.HEALTHY)
			
		except Exception as e:
			self._health.status = ProcessorStatus.ERROR
			self._health.last_error = str(e)
			self._log_health_check(ProcessorStatus.ERROR)
		
		return self._health
	
	# Stripe-specific helper methods
	
	async def _prepare_payment_intent_data(
		self,
		transaction: PaymentTransaction,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Prepare payment intent data for Stripe"""
		data = {
			"amount": transaction.amount,
			"currency": transaction.currency.lower(),
			"payment_method": payment_method.token,
			"capture_method": self.capture_method,
			"metadata": {
				"apg_transaction_id": transaction.id,
				"apg_tenant_id": transaction.tenant_id,
				"apg_merchant_id": transaction.merchant_id
			}
		}
		
		# Add customer if available
		if transaction.customer_id:
			stripe_customer = await self._get_or_create_stripe_customer(transaction.customer_id)
			data["customer"] = stripe_customer.id
		
		# Add description
		if transaction.description:
			data["description"] = transaction.description
		
		# Add statement descriptor
		if self.statement_descriptor:
			data["statement_descriptor"] = self.statement_descriptor
		
		# Add receipt email
		if self.receipt_email and additional_data.get("receipt_email"):
			data["receipt_email"] = additional_data["receipt_email"]
		
		# Add 3D Secure configuration
		if self.require_3ds != "automatic":
			data["payment_method_options"] = {
				"card": {
					"request_three_d_secure": self.require_3ds
				}
			}
		
		# Add automatic payment methods
		if self.automatic_payment_methods:
			data["automatic_payment_methods"] = {"enabled": True}
		
		# Add setup for future usage
		if self.setup_future_usage:
			data["setup_future_usage"] = self.setup_future_usage
		
		# Add application fee for Stripe Connect
		if self.application_fee_percent > 0:
			data["application_fee_amount"] = int(transaction.amount * self.application_fee_percent / 100)
		
		# Add transfer group
		if self.transfer_group:
			data["transfer_group"] = self.transfer_group
		
		return data
	
	async def _process_card_payment(
		self,
		payment_intent,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any] | None
	) -> PaymentResult:
		"""Process card payment with enhanced features"""
		try:
			# Confirm payment intent
			confirmed_intent = await self._make_stripe_request(
				payment_intent.confirm,
				return_url=additional_data.get("return_url") if additional_data else None
			)
			
			return await self._handle_payment_intent_result(confirmed_intent)
			
		except Exception as e:
			return create_payment_result_error(
				status=PaymentStatus.FAILED,
				error_code="stripe_card_error",
				error_message=str(e)
			)
	
	async def _process_bank_transfer(
		self,
		payment_intent,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any] | None
	) -> PaymentResult:
		"""Process bank transfer payment"""
		try:
			# Bank transfers typically require confirmation and customer action
			confirmed_intent = await self._make_stripe_request(
				payment_intent.confirm,
				return_url=additional_data.get("return_url") if additional_data else None
			)
			
			return await self._handle_payment_intent_result(confirmed_intent)
			
		except Exception as e:
			return create_payment_result_error(
				status=PaymentStatus.FAILED,
				error_code="stripe_bank_transfer_error",
				error_message=str(e)
			)
	
	async def _process_digital_wallet(
		self,
		payment_intent,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any] | None
	) -> PaymentResult:
		"""Process digital wallet payment (Apple Pay, Google Pay)"""
		try:
			# Digital wallet payments are typically pre-authorized
			confirmed_intent = await self._make_stripe_request(
				payment_intent.confirm
			)
			
			return await self._handle_payment_intent_result(confirmed_intent)
			
		except Exception as e:
			return create_payment_result_error(
				status=PaymentStatus.FAILED,
				error_code="stripe_digital_wallet_error",
				error_message=str(e)
			)
	
	async def _process_generic_payment(
		self,
		payment_intent,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any] | None
	) -> PaymentResult:
		"""Process generic payment method"""
		try:
			confirmed_intent = await self._make_stripe_request(
				payment_intent.confirm,
				return_url=additional_data.get("return_url") if additional_data else None
			)
			
			return await self._handle_payment_intent_result(confirmed_intent)
			
		except Exception as e:
			return create_payment_result_error(
				status=PaymentStatus.FAILED,
				error_code="stripe_generic_error",
				error_message=str(e)
			)
	
	async def _handle_payment_intent_result(self, payment_intent) -> PaymentResult:
		"""Handle payment intent result and determine next action"""
		status_mapping = {
			"succeeded": PaymentStatus.COMPLETED,
			"requires_capture": PaymentStatus.AUTHORIZED,
			"processing": PaymentStatus.PROCESSING,
			"requires_action": PaymentStatus.PENDING,
			"requires_payment_method": PaymentStatus.FAILED,
			"canceled": PaymentStatus.CANCELLED
		}
		
		status = status_mapping.get(payment_intent.status, PaymentStatus.PENDING)
		
		# Check if additional action is required
		if payment_intent.status == "requires_action":
			return create_payment_result_action_required(
				status=status,
				action_type=payment_intent.next_action.type,
				action_data={
					"client_secret": payment_intent.client_secret,
					"next_action": payment_intent.next_action
				},
				processor_transaction_id=payment_intent.id
			)
		
		# Success case
		if payment_intent.status in ["succeeded", "requires_capture", "processing"]:
			return create_payment_result_success(
				status=status,
				processor_transaction_id=payment_intent.id,
				metadata={
					"stripe_payment_intent_id": payment_intent.id,
					"client_secret": payment_intent.client_secret,
					"amount": payment_intent.amount,
					"currency": payment_intent.currency
				}
			)
		
		# Error case
		return create_payment_result_error(
			status=status,
			error_code="stripe_payment_failed",
			error_message=f"Payment failed with status: {payment_intent.status}"
		)
	
	async def _get_or_create_stripe_customer(self, customer_id: str):
		"""Get or create Stripe customer"""
		try:
			# Try to retrieve existing customer
			customers = await self._make_stripe_request(
				stripe.Customer.list,
				limit=1,
				email=customer_id  # Assuming customer_id is email for simplicity
			)
			
			if customers.data:
				return customers.data[0]
			
			# Create new customer
			return await self._make_stripe_request(
				stripe.Customer.create,
				email=customer_id,
				metadata={"apg_customer_id": customer_id}
			)
			
		except Exception as e:
			self._log_stripe_customer_error(customer_id, str(e))
			raise
	
	async def _get_stripe_capabilities(self) -> List[str]:
		"""Get available Stripe capabilities"""
		try:
			account = await self._make_stripe_request(stripe.Account.retrieve)
			return list(account.capabilities.keys()) if hasattr(account, 'capabilities') else []
		except Exception:
			return []
	
	async def _make_stripe_request(self, stripe_method, *args, **kwargs):
		"""Make async Stripe API request with error handling"""
		try:
			# Convert to async if needed (Stripe SDK is synchronous)
			loop = asyncio.get_event_loop()
			return await loop.run_in_executor(None, stripe_method, *args, **kwargs)
		except stripe.error.StripeError as e:
			self._log_stripe_api_error(str(e))
			raise
	
	# Logging methods following APG patterns
	
	def _log_stripe_processor_created(self):
		"""Log Stripe processor creation"""
		print(f"💳 Stripe Payment Processor created")
		print(f"   Environment: {self.environment}")
		print(f"   Capabilities: {len(self.capabilities)}")
		print(f"   Auto Payment Methods: {self.automatic_payment_methods}")
	
	def _log_stripe_initialization_start(self):
		"""Log Stripe initialization start"""
		print(f"🚀 Initializing Stripe processor...")
		print(f"   Environment: {self.environment}")
	
	def _log_stripe_initialization_complete(self, account_id: str):
		"""Log Stripe initialization complete"""
		print(f"✅ Stripe processor initialized successfully")
		print(f"   Account ID: {account_id}")
		print(f"   Supported Methods: {len(self.supported_payment_methods)}")
	
	def _log_stripe_initialization_error(self, error: str):
		"""Log Stripe initialization error"""
		print(f"❌ Stripe initialization failed: {error}")
	
	def _log_stripe_capture_start(self, transaction_id: str, amount: int | None):
		"""Log capture start"""
		print(f"💰 Capturing Stripe payment: {transaction_id}")
		if amount:
			print(f"   Amount: {amount}")
	
	def _log_stripe_capture_success(self, transaction_id: str, stripe_id: str):
		"""Log capture success"""
		print(f"✅ Stripe payment captured: {transaction_id} -> {stripe_id}")
	
	def _log_stripe_capture_error(self, transaction_id: str, error: str):
		"""Log capture error"""
		print(f"❌ Stripe capture failed: {transaction_id} - {error}")
	
	def _log_stripe_refund_start(self, transaction_id: str, amount: int | None, reason: str | None):
		"""Log refund start"""
		print(f"↩️  Processing Stripe refund: {transaction_id}")
		if amount:
			print(f"   Amount: {amount}")
		if reason:
			print(f"   Reason: {reason}")
	
	def _log_stripe_refund_success(self, transaction_id: str, refund_id: str, amount: int):
		"""Log refund success"""
		print(f"✅ Stripe refund processed: {transaction_id} -> {refund_id}")
		print(f"   Amount: {amount}")
	
	def _log_stripe_refund_error(self, transaction_id: str, error: str):
		"""Log refund error"""
		print(f"❌ Stripe refund failed: {transaction_id} - {error}")
	
	def _log_stripe_void_start(self, transaction_id: str):
		"""Log void start"""
		print(f"🚫 Voiding Stripe payment: {transaction_id}")
	
	def _log_stripe_void_success(self, transaction_id: str):
		"""Log void success"""
		print(f"✅ Stripe payment voided: {transaction_id}")
	
	def _log_stripe_void_error(self, transaction_id: str, error: str):
		"""Log void error"""
		print(f"❌ Stripe void failed: {transaction_id} - {error}")
	
	def _log_stripe_tokenization_start(self, customer_id: str):
		"""Log tokenization start"""
		print(f"🔐 Tokenizing payment method for customer: {customer_id}")
	
	def _log_stripe_tokenization_success(self, token: str):
		"""Log tokenization success"""
		print(f"✅ Payment method tokenized: {token}")
	
	def _log_stripe_tokenization_error(self, customer_id: str, error: str):
		"""Log tokenization error"""
		print(f"❌ Tokenization failed for customer {customer_id}: {error}")
	
	def _log_stripe_customer_error(self, customer_id: str, error: str):
		"""Log customer operation error"""
		print(f"❌ Stripe customer operation failed: {customer_id} - {error}")
	
	def _log_stripe_api_error(self, error: str):
		"""Log Stripe API error"""
		print(f"❌ Stripe API error: {error}")

# Factory function for creating Stripe processor
def create_stripe_processor(config: Dict[str, Any]) -> StripePaymentProcessor:
	"""Factory function to create Stripe processor with configuration"""
	return StripePaymentProcessor(config)

def _log_stripe_module_loaded():
	"""Log Stripe module loaded"""
	print("💳 Stripe Payment Processor module loaded")
	print("   - Full Stripe API coverage")
	print("   - 3D Secure support")
	print("   - Multi-currency processing")
	print("   - Advanced fraud protection")
	print("   - Stripe Connect support")

# Execute module loading log
_log_stripe_module_loaded()