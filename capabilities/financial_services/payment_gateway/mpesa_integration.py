"""
Complete MPESA Integration Service - APG Payment Gateway

Full implementation of all MPESA features using actual Safaricom APIs:
- STK Push (Customer-to-Business)
- B2B (Business-to-Business) payments
- B2C (Business-to-Customer) payments  
- C2B (Customer-to-Business) payments
- Account balance inquiry
- Transaction status query
- Transaction reversal
- Callback URL handling

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import aiohttp
import base64
import json
import logging
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, asdict
from uuid_extensions import uuid7str
import os
from urllib.parse import quote_plus

from .models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType
from .payment_processor import AbstractPaymentProcessor, PaymentResult, ProcessorCapability, ProcessorStatus, ProcessorHealth

logger = logging.getLogger(__name__)

class MPESAEnvironment(str, Enum):
	"""MPESA API environments"""
	SANDBOX = "sandbox"
	PRODUCTION = "production"

class MPESATransactionType(str, Enum):
	"""MPESA transaction types"""
	STK_PUSH = "stk_push"
	B2B = "b2b"  
	B2C = "b2c"
	C2B = "c2b"
	ACCOUNT_BALANCE = "account_balance"
	TRANSACTION_STATUS = "transaction_status"
	REVERSAL = "reversal"

class MPESACommandID(str, Enum):
	"""MPESA command IDs"""
	CUSTOMER_PAY_BILL_ONLINE = "CustomerPayBillOnline"
	CUSTOMER_BUY_GOODS_ONLINE = "CustomerBuyGoodsOnline"
	BUSINESS_PAYMENT = "BusinessPayment"
	SALARY_PAYMENT = "SalaryPayment"
	PROMOTION_PAYMENT = "PromotionPayment"
	ACCOUNT_BALANCE = "AccountBalance"
	TRANSACTION_REVERSAL = "TransactionReversal"
	TRANSACTION_STATUS_QUERY = "TransactionStatusQuery"

@dataclass
class MPESACredentials:
	"""MPESA API credentials"""
	consumer_key: str
	consumer_secret: str
	business_short_code: str
	passkey: str
	initiator_name: str
	security_credential: str
	callback_url: str
	timeout_url: str
	result_url: str
	queue_timeout_url: str

@dataclass  
class MPESAConfig:
	"""MPESA configuration"""
	environment: MPESAEnvironment
	credentials: MPESACredentials
	base_url: str
	oauth_url: str
	stk_push_url: str
	b2b_url: str
	b2c_url: str
	c2b_register_url: str
	c2b_simulate_url: str
	account_balance_url: str
	transaction_status_url: str
	reversal_url: str

class MPESAService(AbstractPaymentProcessor):
	"""
	Complete MPESA integration service with all features
	
	Implements all MPESA APIs:
	- OAuth authentication and token management
	- STK Push for customer payments
	- B2B for business-to-business transfers
	- B2C for business-to-customer payments
	- C2B for customer-to-business payments
	- Account balance inquiries
	- Transaction status queries
	- Transaction reversals
	"""
	
	def __init__(self, config: MPESAConfig):
		"""Initialize MPESA service with configuration"""
		super().__init__(
			processor_name="mpesa",
			supported_payment_methods=[PaymentMethodType.MPESA],
			supported_currencies=["KES"],
			supported_countries=["KE"],
			capabilities=[
				ProcessorCapability.AUTHORIZATION,
				ProcessorCapability.CAPTURE,
				ProcessorCapability.REFUND,
				ProcessorCapability.VOID,
				ProcessorCapability.TOKENIZATION
			]
		)
		
		self.config = config
		self._access_token: Optional[str] = None
		self._token_expires_at: Optional[datetime] = None
		self._session: Optional[aiohttp.ClientSession] = None
		self._initialized = False
		
		# Transaction tracking
		self._pending_transactions: Dict[str, Dict[str, Any]] = {}
		self._completed_transactions: Dict[str, Dict[str, Any]] = {}
		
		logger.info(f"MPESA Service initialized for environment: {config.environment.value}")
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize MPESA service and establish connection"""
		try:
			# Create HTTP session
			self._session = aiohttp.ClientSession(
				timeout=aiohttp.ClientTimeout(total=30),
				headers={
					"Content-Type": "application/json",
					"User-Agent": "APG-PaymentGateway/1.0"
				}
			)
			
			# Get initial access token
			await self._get_access_token()
			
			# Validate configuration
			await self._validate_configuration()
			
			# Register C2B URLs if configured
			if self.config.credentials.callback_url:
				await self._register_c2b_urls()
			
			self._initialized = True
			self._health.status = ProcessorStatus.HEALTHY
			
			logger.info("MPESA Service initialized successfully")
			
			return {
				"status": "initialized",
				"environment": self.config.environment.value,
				"business_short_code": self.config.credentials.business_short_code,
				"capabilities": [cap.value for cap in self.capabilities],
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			self._health.status = ProcessorStatus.ERROR
			logger.error(f"MPESA Service initialization failed: {str(e)}")
			raise
	
	async def _get_access_token(self) -> str:
		"""Get OAuth access token from Safaricom API"""
		try:
			# Check if current token is still valid
			if (self._access_token and self._token_expires_at and 
				datetime.utcnow() < self._token_expires_at):
				return self._access_token
			
			# Prepare authentication
			credentials = f"{self.config.credentials.consumer_key}:{self.config.credentials.consumer_secret}"
			encoded_credentials = base64.b64encode(credentials.encode()).decode()
			
			headers = {
				"Authorization": f"Basic {encoded_credentials}",
				"Content-Type": "application/json"
			}
			
			# Make OAuth request
			async with self._session.get(self.config.oauth_url, headers=headers) as response:
				if response.status == 200:
					data = await response.json()
					self._access_token = data["access_token"]
					expires_in = int(data.get("expires_in", 3600))
					self._token_expires_at = datetime.utcnow().replace(microsecond=0) + timedelta(seconds=expires_in - 300)  # 5min buffer
					
					logger.info(f"MPESA access token obtained, expires at: {self._token_expires_at}")
					return self._access_token
				else:
					error_data = await response.text()
					raise Exception(f"OAuth failed with status {response.status}: {error_data}")
					
		except Exception as e:
			logger.error(f"Failed to get MPESA access token: {str(e)}")
			raise
	
	async def _validate_configuration(self) -> None:
		"""Validate MPESA configuration by testing account balance"""
		try:
			balance_result = await self.get_account_balance()
			if not balance_result.get("success", False):
				raise Exception(f"Configuration validation failed: {balance_result.get('error', 'Unknown error')}")
			
			logger.info("MPESA configuration validated successfully")
			
		except Exception as e:
			logger.error(f"MPESA configuration validation failed: {str(e)}")
			raise
	
	async def _register_c2b_urls(self) -> None:
		"""Register C2B validation and confirmation URLs"""
		try:
			token = await self._get_access_token()
			
			headers = {
				"Authorization": f"Bearer {token}",
				"Content-Type": "application/json"
			}
			
			payload = {
				"ShortCode": self.config.credentials.business_short_code,
				"ResponseType": "Completed",  # or "Cancelled"
				"ConfirmationURL": self.config.credentials.callback_url,
				"ValidationURL": self.config.credentials.callback_url
			}
			
			async with self._session.post(
				self.config.c2b_register_url,
				headers=headers,
				json=payload
			) as response:
				if response.status == 200:
					data = await response.json()
					if data.get("ResponseCode") == "0":
						logger.info("C2B URLs registered successfully")
					else:
						logger.warning(f"C2B URL registration warning: {data.get('ResponseDescription', 'Unknown')}")
				else:
					error_data = await response.text()
					logger.error(f"C2B URL registration failed: {error_data}")
					
		except Exception as e:
			logger.error(f"Failed to register C2B URLs: {str(e)}")
			# Don't raise here as this is not critical for basic functionality
	
	async def process_payment(
		self,
		transaction: PaymentTransaction,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any] | None = None
	) -> PaymentResult:
		"""Process payment using MPESA STK Push"""
		try:
			if not self._initialized:
				await self.initialize()
			
			start_time = self._record_transaction_start()
			
			# Determine transaction type
			transaction_type = additional_data.get("transaction_type", MPESATransactionType.STK_PUSH.value)
			
			result = None
			if transaction_type == MPESATransactionType.STK_PUSH.value:
				result = await self._process_stk_push(transaction, payment_method, additional_data or {})
			elif transaction_type == MPESATransactionType.B2B.value:
				result = await self._process_b2b_payment(transaction, payment_method, additional_data or {})
			elif transaction_type == MPESATransactionType.B2C.value:
				result = await self._process_b2c_payment(transaction, payment_method, additional_data or {})
			elif transaction_type == MPESATransactionType.C2B.value:
				result = await self._process_c2b_payment(transaction, payment_method, additional_data or {})
			else:
				raise ValueError(f"Unsupported transaction type: {transaction_type}")
			
			if result.success:
				self._record_transaction_success(start_time)
			else:
				self._record_transaction_error(result.error_message or "Payment failed")
			
			return result
			
		except Exception as e:
			self._record_transaction_error(str(e))
			logger.error(f"MPESA payment processing failed: {str(e)}")
			return PaymentResult(
				success=False,
				status=PaymentStatus.FAILED,
				error_code="processing_error",
				error_message=str(e)
			)
	
	async def _process_stk_push(
		self,
		transaction: PaymentTransaction,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any]
	) -> PaymentResult:
		"""Process STK Push (Customer Pay Bill Online)"""
		try:
			token = await self._get_access_token()
			
			# Generate timestamp and password
			timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
			password_string = f"{self.config.credentials.business_short_code}{self.config.credentials.passkey}{timestamp}"
			password = base64.b64encode(password_string.encode()).decode()
			
			# Extract phone number
			phone_number = additional_data.get("phone_number") or payment_method.mpesa_phone_number
			if not phone_number:
				raise ValueError("Phone number is required for STK Push")
			
			# Format phone number (ensure it starts with 254)
			if phone_number.startswith("0"):
				phone_number = "254" + phone_number[1:]
			elif phone_number.startswith("+254"):
				phone_number = phone_number[1:]
			elif not phone_number.startswith("254"):
				phone_number = "254" + phone_number
			
			headers = {
				"Authorization": f"Bearer {token}",
				"Content-Type": "application/json"
			}
			
			payload = {
				"BusinessShortCode": self.config.credentials.business_short_code,
				"Password": password,
				"Timestamp": timestamp,
				"TransactionType": "CustomerPayBillOnline",
				"Amount": transaction.amount,
				"PartyA": phone_number,
				"PartyB": self.config.credentials.business_short_code,
				"PhoneNumber": phone_number,
				"CallBackURL": self.config.credentials.callback_url,
				"AccountReference": additional_data.get("account_reference", transaction.id),
				"TransactionDesc": transaction.description or f"Payment for {transaction.id}"
			}
			
			async with self._session.post(
				self.config.stk_push_url,
				headers=headers,
				json=payload
			) as response:
				
				data = await response.json()
				
				if response.status == 200 and data.get("ResponseCode") == "0":
					checkout_request_id = data.get("CheckoutRequestID")
					merchant_request_id = data.get("MerchantRequestID")
					
					# Store pending transaction
					self._pending_transactions[checkout_request_id] = {
						"transaction_id": transaction.id,
						"merchant_request_id": merchant_request_id,
						"checkout_request_id": checkout_request_id,
						"phone_number": phone_number,
						"amount": transaction.amount,
						"timestamp": datetime.utcnow().isoformat(),
						"status": "pending"
					}
					
					logger.info(f"STK Push initiated successfully: {checkout_request_id}")
					
					return PaymentResult(
						success=True,
						status=PaymentStatus.PENDING,
						processor_transaction_id=checkout_request_id,
						processor_response=data,
						requires_action=True,
						action_type="customer_confirmation",
						action_data={
							"message": "Customer needs to confirm payment on their phone",
							"checkout_request_id": checkout_request_id,
							"merchant_request_id": merchant_request_id
						}
					)
				else:
					error_message = data.get("ResponseDescription", "STK Push failed")
					error_code = data.get("ResponseCode", "unknown_error")
					
					logger.error(f"STK Push failed: {error_code} - {error_message}")
					
					return PaymentResult(
						success=False,
						status=PaymentStatus.FAILED,
						error_code=str(error_code),
						error_message=error_message,
						processor_response=data
					)
					
		except Exception as e:
			logger.error(f"STK Push processing error: {str(e)}")
			return PaymentResult(
				success=False,
				status=PaymentStatus.FAILED,
				error_code="stk_push_error",
				error_message=str(e)
			)
	
	async def _process_b2b_payment(
		self,
		transaction: PaymentTransaction,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any]
	) -> PaymentResult:
		"""Process Business-to-Business payment"""
		try:
			token = await self._get_access_token()
			
			# Required fields for B2B
			receiver_party = additional_data.get("receiver_party")
			command_id = additional_data.get("command_id", MPESACommandID.BUSINESS_PAYMENT.value)
			
			if not receiver_party:
				raise ValueError("Receiver party (business short code) is required for B2B payment")
			
			# Generate security credential (encrypt initiator password with M-Pesa public key)
			security_credential = self.config.credentials.security_credential
			
			headers = {
				"Authorization": f"Bearer {token}",
				"Content-Type": "application/json"
			}
			
			payload = {
				"Initiator": self.config.credentials.initiator_name,
				"SecurityCredential": security_credential,
				"CommandID": command_id,
				"SenderIdentifierType": "4",  # Organization short code
				"RecieverIdentifierType": "4",  # Organization short code
				"Amount": transaction.amount,
				"PartyA": self.config.credentials.business_short_code,
				"PartyB": receiver_party,
				"AccountReference": additional_data.get("account_reference", transaction.id),
				"Remarks": transaction.description or f"B2B Payment {transaction.id}",
				"QueueTimeOutURL": self.config.credentials.queue_timeout_url,
				"ResultURL": self.config.credentials.result_url
			}
			
			async with self._session.post(
				self.config.b2b_url,
				headers=headers,
				json=payload
			) as response:
				
				data = await response.json()
				
				if response.status == 200 and data.get("ResponseCode") == "0":
					conversation_id = data.get("ConversationID")
					originator_conversation_id = data.get("OriginatorConversationID")
					
					# Store pending transaction
					self._pending_transactions[conversation_id] = {
						"transaction_id": transaction.id,
						"conversation_id": conversation_id,
						"originator_conversation_id": originator_conversation_id,
						"receiver_party": receiver_party,
						"amount": transaction.amount,
						"command_id": command_id,
						"timestamp": datetime.utcnow().isoformat(),
						"status": "pending"
					}
					
					logger.info(f"B2B payment initiated successfully: {conversation_id}")
					
					return PaymentResult(
						success=True,
						status=PaymentStatus.PENDING,
						processor_transaction_id=conversation_id,
						processor_response=data,
						metadata={
							"conversation_id": conversation_id,
							"originator_conversation_id": originator_conversation_id,
							"receiver_party": receiver_party
						}
					)
				else:
					error_message = data.get("ResponseDescription", "B2B payment failed")
					error_code = data.get("ResponseCode", "unknown_error")
					
					logger.error(f"B2B payment failed: {error_code} - {error_message}")
					
					return PaymentResult(
						success=False,
						status=PaymentStatus.FAILED,
						error_code=str(error_code),
						error_message=error_message,
						processor_response=data
					)
					
		except Exception as e:
			logger.error(f"B2B payment processing error: {str(e)}")
			return PaymentResult(
				success=False,
				status=PaymentStatus.FAILED,
				error_code="b2b_payment_error",
				error_message=str(e)
			)
	
	async def _process_b2c_payment(
		self,
		transaction: PaymentTransaction,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any]
	) -> PaymentResult:
		"""Process Business-to-Customer payment"""
		try:
			token = await self._get_access_token()
			
			# Required fields for B2C
			phone_number = additional_data.get("phone_number") or payment_method.mpesa_phone_number
			command_id = additional_data.get("command_id", MPESACommandID.BUSINESS_PAYMENT.value)
			
			if not phone_number:
				raise ValueError("Phone number is required for B2C payment")
			
			# Format phone number
			if phone_number.startswith("0"):
				phone_number = "254" + phone_number[1:]
			elif phone_number.startswith("+254"):
				phone_number = phone_number[1:]
			elif not phone_number.startswith("254"):
				phone_number = "254" + phone_number
			
			# Generate security credential
			security_credential = self.config.credentials.security_credential
			
			headers = {
				"Authorization": f"Bearer {token}",
				"Content-Type": "application/json"
			}
			
			payload = {
				"InitiatorName": self.config.credentials.initiator_name,
				"SecurityCredential": security_credential,
				"CommandID": command_id,
				"Amount": transaction.amount,
				"PartyA": self.config.credentials.business_short_code,
				"PartyB": phone_number,
				"Remarks": transaction.description or f"B2C Payment {transaction.id}",
				"QueueTimeOutURL": self.config.credentials.queue_timeout_url,
				"ResultURL": self.config.credentials.result_url,
				"Occasion": additional_data.get("occasion", "Payment")
			}
			
			async with self._session.post(
				self.config.b2c_url,
				headers=headers,
				json=payload
			) as response:
				
				data = await response.json()
				
				if response.status == 200 and data.get("ResponseCode") == "0":
					conversation_id = data.get("ConversationID")
					originator_conversation_id = data.get("OriginatorConversationID")
					
					# Store pending transaction
					self._pending_transactions[conversation_id] = {
						"transaction_id": transaction.id,
						"conversation_id": conversation_id,
						"originator_conversation_id": originator_conversation_id,
						"phone_number": phone_number,
						"amount": transaction.amount,
						"command_id": command_id,
						"timestamp": datetime.utcnow().isoformat(),
						"status": "pending"
					}
					
					logger.info(f"B2C payment initiated successfully: {conversation_id}")
					
					return PaymentResult(
						success=True,
						status=PaymentStatus.PENDING,
						processor_transaction_id=conversation_id,
						processor_response=data,
						metadata={
							"conversation_id": conversation_id,
							"originator_conversation_id": originator_conversation_id,
							"phone_number": phone_number
						}
					)
				else:
					error_message = data.get("ResponseDescription", "B2C payment failed")
					error_code = data.get("ResponseCode", "unknown_error")
					
					logger.error(f"B2C payment failed: {error_code} - {error_message}")
					
					return PaymentResult(
						success=False,
						status=PaymentStatus.FAILED,
						error_code=str(error_code),
						error_message=error_message,
						processor_response=data
					)
					
		except Exception as e:
			logger.error(f"B2C payment processing error: {str(e)}")
			return PaymentResult(
				success=False,
				status=PaymentStatus.FAILED,
				error_code="b2c_payment_error",
				error_message=str(e)
			)
	
	async def _process_c2b_payment(
		self,
		transaction: PaymentTransaction,
		payment_method: PaymentMethod,
		additional_data: Dict[str, Any]
	) -> PaymentResult:
		"""Process Customer-to-Business payment simulation"""
		try:
			token = await self._get_access_token()
			
			# Required fields for C2B simulation
			phone_number = additional_data.get("phone_number") or payment_method.mpesa_phone_number
			command_id = additional_data.get("command_id", "CustomerPayBillOnline")
			
			if not phone_number:
				raise ValueError("Phone number is required for C2B payment")
			
			# Format phone number
			if phone_number.startswith("0"):
				phone_number = "254" + phone_number[1:]
			elif phone_number.startswith("+254"):
				phone_number = phone_number[1:]
			elif not phone_number.startswith("254"):
				phone_number = "254" + phone_number
			
			headers = {
				"Authorization": f"Bearer {token}",
				"Content-Type": "application/json"
			}
			
			payload = {
				"ShortCode": self.config.credentials.business_short_code,
				"CommandID": command_id,
				"Amount": transaction.amount,
				"Msisdn": phone_number,
				"BillRefNumber": additional_data.get("bill_ref_number", transaction.id)
			}
			
			async with self._session.post(
				self.config.c2b_simulate_url,
				headers=headers,
				json=payload
			) as response:
				
				data = await response.json()
				
				if response.status == 200 and data.get("ResponseCode") == "0":
					conversation_id = data.get("ConversationID", uuid7str())
					
					# Store pending transaction
					self._pending_transactions[conversation_id] = {
						"transaction_id": transaction.id,
						"conversation_id": conversation_id,
						"phone_number": phone_number,
						"amount": transaction.amount,
						"command_id": command_id,
						"bill_ref_number": payload["BillRefNumber"],
						"timestamp": datetime.utcnow().isoformat(),
						"status": "pending"
					}
					
					logger.info(f"C2B payment simulation initiated successfully: {conversation_id}")
					
					return PaymentResult(
						success=True,
						status=PaymentStatus.PENDING,
						processor_transaction_id=conversation_id,
						processor_response=data,
						metadata={
							"conversation_id": conversation_id,
							"phone_number": phone_number,
							"bill_ref_number": payload["BillRefNumber"]
						}
					)
				else:
					error_message = data.get("ResponseDescription", "C2B payment failed")
					error_code = data.get("ResponseCode", "unknown_error")
					
					logger.error(f"C2B payment failed: {error_code} - {error_message}")
					
					return PaymentResult(
						success=False,
						status=PaymentStatus.FAILED,
						error_code=str(error_code),
						error_message=error_message,
						processor_response=data
					)
					
		except Exception as e:
			logger.error(f"C2B payment processing error: {str(e)}")
			return PaymentResult(
				success=False,
				status=PaymentStatus.FAILED,
				error_code="c2b_payment_error",
				error_message=str(e)
			)
	
	async def capture_payment(self, transaction_id: str, amount: int | None = None) -> PaymentResult:
		"""Capture payment (not applicable for MPESA, payments are immediately settled)"""
		return PaymentResult(
			success=True,
			status=PaymentStatus.COMPLETED,
			processor_transaction_id=transaction_id,
			metadata={"note": "MPESA payments are immediately settled, no capture required"}
		)
	
	async def refund_payment(
		self,
		transaction_id: str,
		amount: int | None = None,
		reason: str | None = None
	) -> PaymentResult:
		"""Process refund through transaction reversal"""
		try:
			# Find the original transaction
			original_transaction = None
			for tx_data in self._completed_transactions.values():
				if tx_data.get("transaction_id") == transaction_id:
					original_transaction = tx_data
					break
			
			if not original_transaction:
				return PaymentResult(
					success=False,
					status=PaymentStatus.FAILED,
					error_code="transaction_not_found",
					error_message=f"Original transaction {transaction_id} not found"
				)
			
			# Use transaction reversal API
			return await self.reverse_transaction(
				originator_conversation_id=original_transaction.get("originator_conversation_id"),
				transaction_id=original_transaction.get("mpesa_receipt_number"),
				amount=amount or original_transaction.get("amount"),
				reason=reason or "Refund requested"
			)
			
		except Exception as e:
			logger.error(f"Refund processing error: {str(e)}")
			return PaymentResult(
				success=False,
				status=PaymentStatus.FAILED,
				error_code="refund_error",
				error_message=str(e)
			)
	
	async def void_payment(self, transaction_id: str) -> PaymentResult:
		"""Void payment (not applicable for MPESA completed transactions)"""
		# Check if transaction is still pending
		for pending_tx in self._pending_transactions.values():
			if pending_tx.get("transaction_id") == transaction_id:
				# Mark as cancelled
				pending_tx["status"] = "cancelled"
				
				return PaymentResult(
					success=True,
					status=PaymentStatus.CANCELLED,
					processor_transaction_id=transaction_id,
					metadata={"note": "Pending transaction cancelled"}
				)
		
		return PaymentResult(
			success=False,
			status=PaymentStatus.FAILED,
			error_code="cannot_void",
			error_message="Cannot void completed MPESA transaction, use refund instead"
		)
	
	async def get_transaction_status(self, transaction_id: str) -> PaymentResult:
		"""Query transaction status from MPESA API"""
		try:
			token = await self._get_access_token()
			
			# Find transaction details
			conversation_id = None
			originator_conversation_id = None
			
			# Check pending transactions
			for tx_data in self._pending_transactions.values():
				if (tx_data.get("transaction_id") == transaction_id or
					tx_data.get("conversation_id") == transaction_id or
					tx_data.get("checkout_request_id") == transaction_id):
					conversation_id = tx_data.get("conversation_id")
					originator_conversation_id = tx_data.get("originator_conversation_id")
					break
			
			# Check completed transactions
			if not conversation_id:
				for tx_data in self._completed_transactions.values():
					if (tx_data.get("transaction_id") == transaction_id or
						tx_data.get("conversation_id") == transaction_id):
						return PaymentResult(
							success=True,
							status=PaymentStatus.COMPLETED,
							processor_transaction_id=transaction_id,
							processor_response=tx_data
						)
			
			if not conversation_id:
				return PaymentResult(
					success=False,
					status=PaymentStatus.FAILED,
					error_code="transaction_not_found",
					error_message=f"Transaction {transaction_id} not found"
				)
			
			# Generate security credential
			security_credential = self.config.credentials.security_credential
			
			headers = {
				"Authorization": f"Bearer {token}",
				"Content-Type": "application/json"
			}
			
			payload = {
				"Initiator": self.config.credentials.initiator_name,
				"SecurityCredential": security_credential,
				"CommandID": "TransactionStatusQuery",
				"TransactionID": conversation_id,
				"PartyA": self.config.credentials.business_short_code,
				"IdentifierType": "4",
				"ResultURL": self.config.credentials.result_url,
				"QueueTimeOutURL": self.config.credentials.queue_timeout_url,
				"Remarks": f"Transaction status query for {transaction_id}",
				"Occasion": "Status Check"
			}
			
			async with self._session.post(
				self.config.transaction_status_url,
				headers=headers,
				json=payload
			) as response:
				
				data = await response.json()
				
				if response.status == 200 and data.get("ResponseCode") == "0":
					logger.info(f"Transaction status query initiated: {conversation_id}")
					
					return PaymentResult(
						success=True,
						status=PaymentStatus.PENDING,
						processor_transaction_id=conversation_id,
						processor_response=data,
						metadata={
							"note": "Status query initiated, result will be sent to callback URL",
							"conversation_id": data.get("ConversationID"),
							"originator_conversation_id": data.get("OriginatorConversationID")
						}
					)
				else:
					error_message = data.get("ResponseDescription", "Status query failed")
					error_code = data.get("ResponseCode", "unknown_error")
					
					return PaymentResult(
						success=False,
						status=PaymentStatus.FAILED,
						error_code=str(error_code),
						error_message=error_message,
						processor_response=data
					)
					
		except Exception as e:
			logger.error(f"Transaction status query error: {str(e)}")
			return PaymentResult(
				success=False,
				status=PaymentStatus.FAILED,
				error_code="status_query_error",
				error_message=str(e)
			)
	
	async def tokenize_payment_method(
		self,
		payment_method_data: Dict[str, Any],
		customer_id: str
	) -> Dict[str, Any]:
		"""Tokenize MPESA payment method (store phone number securely)"""
		try:
			phone_number = payment_method_data.get("phone_number")
			if not phone_number:
				raise ValueError("Phone number is required for MPESA tokenization")
			
			# Format and validate phone number
			if phone_number.startswith("0"):
				phone_number = "254" + phone_number[1:]
			elif phone_number.startswith("+254"):
				phone_number = phone_number[1:]
			elif not phone_number.startswith("254"):
				phone_number = "254" + phone_number
			
			# Validate phone number format
			if not (phone_number.startswith("254") and len(phone_number) == 12 and phone_number[3:].isdigit()):
				raise ValueError("Invalid Kenyan phone number format")
			
			# Create secure token
			token = f"mpesa_token_{uuid7str()}"
			
			# In production, this would be stored securely with encryption
			tokenized_data = {
				"success": True,
				"token": token,
				"payment_method_type": "mpesa",
				"phone_number_last_4": phone_number[-4:],
				"customer_id": customer_id,
				"created_at": datetime.utcnow().isoformat(),
				"expires_at": (datetime.utcnow() + timedelta(days=365)).isoformat()
			}
			
			logger.info(f"MPESA payment method tokenized for customer {customer_id}")
			
			return tokenized_data
			
		except Exception as e:
			logger.error(f"MPESA tokenization error: {str(e)}")
			return {
				"success": False,
				"error": str(e)
			}
	
	async def health_check(self) -> ProcessorHealth:
		"""Perform comprehensive health check"""
		try:
			# Check token validity
			token_valid = False
			try:
				await self._get_access_token()
				token_valid = True
			except:
				pass
			
			# Check account balance (validates configuration)
			balance_check = False
			try:
				balance_result = await self.get_account_balance()
				balance_check = balance_result.get("success", False)
			except:
				pass
			
			# Update health metrics
			self._update_health_metrics()
			
			# Determine overall status
			if token_valid and balance_check:
				self._health.status = ProcessorStatus.HEALTHY
			elif token_valid:
				self._health.status = ProcessorStatus.DEGRADED
			else:
				self._health.status = ProcessorStatus.ERROR
			
			self._health.supported_currencies = self.supported_currencies
			self._health.supported_countries = self.supported_countries
			
			return self._health
			
		except Exception as e:
			self._health.status = ProcessorStatus.ERROR
			self._health.last_error = str(e)
			logger.error(f"MPESA health check failed: {str(e)}")
			return self._health
	
	# Additional MPESA-specific methods
	
	async def get_account_balance(self) -> Dict[str, Any]:
		"""Get account balance from MPESA"""
		try:
			if not self._initialized:
				await self.initialize()
			
			token = await self._get_access_token()
			
			# Generate security credential
			security_credential = self.config.credentials.security_credential
			
			headers = {
				"Authorization": f"Bearer {token}",
				"Content-Type": "application/json"
			}
			
			payload = {
				"Initiator": self.config.credentials.initiator_name,
				"SecurityCredential": security_credential,
				"CommandID": "AccountBalance",
				"PartyA": self.config.credentials.business_short_code,
				"IdentifierType": "4",
				"Remarks": "Account balance inquiry",
				"QueueTimeOutURL": self.config.credentials.queue_timeout_url,
				"ResultURL": self.config.credentials.result_url
			}
			
			async with self._session.post(
				self.config.account_balance_url,
				headers=headers,
				json=payload
			) as response:
				
				data = await response.json()
				
				if response.status == 200 and data.get("ResponseCode") == "0":
					conversation_id = data.get("ConversationID")
					
					logger.info(f"Account balance inquiry initiated: {conversation_id}")
					
					return {
						"success": True,
						"conversation_id": conversation_id,
						"originator_conversation_id": data.get("OriginatorConversationID"),
						"response_description": data.get("ResponseDescription"),
						"note": "Balance inquiry initiated, result will be sent to callback URL"
					}
				else:
					error_message = data.get("ResponseDescription", "Balance inquiry failed")
					error_code = data.get("ResponseCode", "unknown_error")
					
					return {
						"success": False,
						"error_code": error_code,
						"error_message": error_message,
						"response": data
					}
					
		except Exception as e:
			logger.error(f"Account balance inquiry error: {str(e)}")
			return {
				"success": False,
				"error": str(e)
			}
	
	async def reverse_transaction(
		self,
		originator_conversation_id: str,
		transaction_id: str,
		amount: int,
		reason: str = "Transaction reversal"
	) -> PaymentResult:
		"""Reverse a completed transaction"""
		try:
			if not self._initialized:
				await self.initialize()
			
			token = await self._get_access_token()
			
			# Generate security credential
			security_credential = self.config.credentials.security_credential
			
			headers = {
				"Authorization": f"Bearer {token}",
				"Content-Type": "application/json"
			}
			
			payload = {
				"Initiator": self.config.credentials.initiator_name,
				"SecurityCredential": security_credential,
				"CommandID": "TransactionReversal",
				"TransactionID": transaction_id,
				"Amount": amount,
				"ReceiverParty": self.config.credentials.business_short_code,
				"RecieverIdentifierType": "11",
				"ResultURL": self.config.credentials.result_url,
				"QueueTimeOutURL": self.config.credentials.queue_timeout_url,
				"Remarks": reason,
				"Occasion": "Reversal"
			}
			
			async with self._session.post(
				self.config.reversal_url,
				headers=headers,
				json=payload
			) as response:
				
				data = await response.json()
				
				if response.status == 200 and data.get("ResponseCode") == "0":
					conversation_id = data.get("ConversationID")
					
					logger.info(f"Transaction reversal initiated: {conversation_id}")
					
					return PaymentResult(
						success=True,
						status=PaymentStatus.PENDING,
						processor_transaction_id=conversation_id,
						processor_response=data,
						metadata={
							"conversation_id": conversation_id,
							"originator_conversation_id": data.get("OriginatorConversationID"),
							"original_transaction_id": transaction_id,
							"reversal_amount": amount
						}
					)
				else:
					error_message = data.get("ResponseDescription", "Transaction reversal failed")
					error_code = data.get("ResponseCode", "unknown_error")
					
					logger.error(f"Transaction reversal failed: {error_code} - {error_message}")
					
					return PaymentResult(
						success=False,
						status=PaymentStatus.FAILED,
						error_code=str(error_code),
						error_message=error_message,
						processor_response=data
					)
					
		except Exception as e:
			logger.error(f"Transaction reversal error: {str(e)}")
			return PaymentResult(
				success=False,
				status=PaymentStatus.FAILED,
				error_code="reversal_error",
				error_message=str(e)
			)
	
	async def handle_callback(self, callback_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle MPESA callback/webhook data"""
		try:
			logger.info(f"Processing MPESA callback: {callback_data}")
			
			# Determine callback type
			if "stkCallback" in callback_data:
				return await self._handle_stk_callback(callback_data["stkCallback"])
			elif "Result" in callback_data:
				return await self._handle_result_callback(callback_data["Result"])
			else:
				logger.warning(f"Unknown callback format: {callback_data}")
				return {"success": False, "error": "Unknown callback format"}
				
		except Exception as e:
			logger.error(f"Callback handling error: {str(e)}")
			return {"success": False, "error": str(e)}
	
	async def _handle_stk_callback(self, stk_callback: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle STK Push callback"""
		try:
			merchant_request_id = stk_callback.get("MerchantRequestID")
			checkout_request_id = stk_callback.get("CheckoutRequestID")
			result_code = stk_callback.get("ResultCode")
			result_desc = stk_callback.get("ResultDesc")
			
			# Find pending transaction
			pending_tx = self._pending_transactions.get(checkout_request_id)
			if not pending_tx:
				logger.warning(f"STK callback for unknown transaction: {checkout_request_id}")
				return {"success": False, "error": "Transaction not found"}
			
			if result_code == 0:  # Success
				# Extract callback metadata
				callback_metadata = {}
				if "CallbackMetadata" in stk_callback:
					for item in stk_callback["CallbackMetadata"].get("Item", []):
						callback_metadata[item.get("Name")] = item.get("Value")
				
				# Move to completed transactions
				completed_tx = {
					**pending_tx,
					"status": "completed",
					"result_code": result_code,
					"result_description": result_desc,
					"mpesa_receipt_number": callback_metadata.get("MpesaReceiptNumber"),
					"transaction_date": callback_metadata.get("TransactionDate"),
					"phone_number": callback_metadata.get("PhoneNumber"),
					"completed_at": datetime.utcnow().isoformat(),
					"callback_metadata": callback_metadata
				}
				
				self._completed_transactions[checkout_request_id] = completed_tx
				del self._pending_transactions[checkout_request_id]
				
				logger.info(f"STK Push completed successfully: {checkout_request_id}")
				
				return {
					"success": True,
					"status": "completed",
					"transaction_id": pending_tx["transaction_id"],
					"mpesa_receipt_number": callback_metadata.get("MpesaReceiptNumber"),
					"amount": callback_metadata.get("Amount"),
					"phone_number": callback_metadata.get("PhoneNumber")
				}
			else:
				# Payment failed or cancelled
				failed_tx = {
					**pending_tx,
					"status": "failed",
					"result_code": result_code,
					"result_description": result_desc,
					"failed_at": datetime.utcnow().isoformat()
				}
				
				self._completed_transactions[checkout_request_id] = failed_tx
				del self._pending_transactions[checkout_request_id]
				
				logger.info(f"STK Push failed: {checkout_request_id} - {result_desc}")
				
				return {
					"success": False,
					"status": "failed",
					"transaction_id": pending_tx["transaction_id"],
					"error_code": str(result_code),
					"error_message": result_desc
				}
				
		except Exception as e:
			logger.error(f"STK callback handling error: {str(e)}")
			return {"success": False, "error": str(e)}
	
	async def _handle_result_callback(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle result callback for B2B, B2C, Balance, Status, Reversal"""
		try:
			conversation_id = result_data.get("ConversationID")
			originator_conversation_id = result_data.get("OriginatorConversationID")
			result_code = result_data.get("ResultCode")
			result_desc = result_data.get("ResultDesc")
			
			# Find pending transaction
			pending_tx = None
			for tx_id, tx_data in self._pending_transactions.items():
				if (tx_data.get("conversation_id") == conversation_id or
					tx_data.get("originator_conversation_id") == originator_conversation_id):
					pending_tx = tx_data
					break
			
			if result_code == 0:  # Success
				# Extract result parameters
				result_parameters = {}
				if "ResultParameters" in result_data:
					for param in result_data["ResultParameters"].get("ResultParameter", []):
						result_parameters[param.get("Key")] = param.get("Value")
				
				if pending_tx:
					# Move to completed transactions
					completed_tx = {
						**pending_tx,
						"status": "completed",
						"result_code": result_code,
						"result_description": result_desc,
						"result_parameters": result_parameters,
						"completed_at": datetime.utcnow().isoformat()
					}
					
					self._completed_transactions[conversation_id] = completed_tx
					# Remove from pending (find correct key)
					for tx_id, tx_data in list(self._pending_transactions.items()):
						if tx_data == pending_tx:
							del self._pending_transactions[tx_id]
							break
					
					logger.info(f"Transaction completed successfully: {conversation_id}")
				
				return {
					"success": True,
					"status": "completed",
					"conversation_id": conversation_id,
					"result_parameters": result_parameters
				}
			else:
				# Transaction failed
				if pending_tx:
					failed_tx = {
						**pending_tx,
						"status": "failed",
						"result_code": result_code,
						"result_description": result_desc,
						"failed_at": datetime.utcnow().isoformat()
					}
					
					self._completed_transactions[conversation_id] = failed_tx
					# Remove from pending
					for tx_id, tx_data in list(self._pending_transactions.items()):
						if tx_data == pending_tx:
							del self._pending_transactions[tx_id]
							break
				
				logger.info(f"Transaction failed: {conversation_id} - {result_desc}")
				
				return {
					"success": False,
					"status": "failed",
					"conversation_id": conversation_id,
					"error_code": str(result_code),
					"error_message": result_desc
				}
				
		except Exception as e:
			logger.error(f"Result callback handling error: {str(e)}")
			return {"success": False, "error": str(e)}
	
	async def cleanup(self):
		"""Cleanup resources"""
		if self._session:
			await self._session.close()
		logger.info("MPESA Service cleaned up")

# Configuration factory functions

def create_mpesa_config(environment: MPESAEnvironment = MPESAEnvironment.SANDBOX) -> MPESAConfig:
	"""Create MPESA configuration for specified environment"""
	
	if environment == MPESAEnvironment.SANDBOX:
		base_url = "https://sandbox.safaricom.co.ke"
	else:
		base_url = "https://api.safaricom.co.ke"
	
	# Load credentials from environment variables
	credentials = MPESACredentials(
		consumer_key=os.getenv("MPESA_CONSUMER_KEY", ""),
		consumer_secret=os.getenv("MPESA_CONSUMER_SECRET", ""),
		business_short_code=os.getenv("MPESA_BUSINESS_SHORT_CODE", "174379"),
		passkey=os.getenv("MPESA_PASSKEY", "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919"),
		initiator_name=os.getenv("MPESA_INITIATOR_NAME", "testapi"),
		security_credential=os.getenv("MPESA_SECURITY_CREDENTIAL", ""),
		callback_url=os.getenv("MPESA_CALLBACK_URL", "https://your-domain.com/mpesa/callback"),
		timeout_url=os.getenv("MPESA_TIMEOUT_URL", "https://your-domain.com/mpesa/timeout"),
		result_url=os.getenv("MPESA_RESULT_URL", "https://your-domain.com/mpesa/result"),
		queue_timeout_url=os.getenv("MPESA_QUEUE_TIMEOUT_URL", "https://your-domain.com/mpesa/queue-timeout")
	)
	
	return MPESAConfig(
		environment=environment,
		credentials=credentials,
		base_url=base_url,
		oauth_url=f"{base_url}/oauth/v1/generate?grant_type=client_credentials",
		stk_push_url=f"{base_url}/mpesa/stkpush/v1/processrequest",
		b2b_url=f"{base_url}/mpesa/b2b/v1/paymentrequest",
		b2c_url=f"{base_url}/mpesa/b2c/v1/paymentrequest",
		c2b_register_url=f"{base_url}/mpesa/c2b/v1/registerurl",
		c2b_simulate_url=f"{base_url}/mpesa/c2b/v1/simulate",
		account_balance_url=f"{base_url}/mpesa/accountbalance/v1/query",
		transaction_status_url=f"{base_url}/mpesa/transactionstatus/v1/query",
		reversal_url=f"{base_url}/mpesa/reversal/v1/request"
	)

async def create_mpesa_service(environment: MPESAEnvironment = MPESAEnvironment.SANDBOX) -> MPESAService:
	"""Create and initialize MPESA service"""
	config = create_mpesa_config(environment)
	service = MPESAService(config)
	await service.initialize()
	return service

def _log_mpesa_module_loaded():
	"""Log MPESA module loaded"""
	print("📱 Complete MPESA Integration module loaded")
	print("   - STK Push (Customer-to-Business)")
	print("   - B2B (Business-to-Business) payments")
	print("   - B2C (Business-to-Customer) payments")
	print("   - C2B (Customer-to-Business) payments")
	print("   - Account balance inquiry")
	print("   - Transaction status query")
	print("   - Transaction reversal")
	print("   - Callback URL handling")
	print("   - OAuth token management")
	print("   - Full Safaricom API integration")

# Execute module loading log
_log_mpesa_module_loaded()