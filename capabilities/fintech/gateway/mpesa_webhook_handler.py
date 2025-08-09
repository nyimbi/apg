"""
MPESA Webhook Handler - APG Payment Gateway

Handles all MPESA callback URLs and webhook processing:
- STK Push callbacks
- B2B result callbacks  
- B2C result callbacks
- C2B validation and confirmation
- Account balance result callbacks
- Transaction status result callbacks
- Transaction reversal result callbacks

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify
from uuid_extensions import uuid7str

from .mpesa_integration import MPESAService

logger = logging.getLogger(__name__)

class MPESAWebhookHandler:
	"""
	Complete MPESA webhook handler for all callback types
	"""
	
	def __init__(self, mpesa_service: MPESAService):
		"""Initialize webhook handler with MPESA service"""
		self.mpesa_service = mpesa_service
		self.webhook_logs: Dict[str, Dict[str, Any]] = {}
		
	async def handle_stk_push_callback(self, callback_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle STK Push callback from MPESA"""
		try:
			webhook_id = uuid7str()
			timestamp = datetime.utcnow().isoformat()
			
			# Log incoming webhook
			self.webhook_logs[webhook_id] = {
				"id": webhook_id,
				"type": "stk_push_callback",
				"timestamp": timestamp,
				"data": callback_data,
				"processed": False
			}
			
			logger.info(f"Received STK Push callback: {webhook_id}")
			
			# Process with MPESA service
			result = await self.mpesa_service.handle_callback(callback_data)
			
			# Update log with result
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"result": result,
				"processed_at": datetime.utcnow().isoformat()
			})
			
			# Return MPESA expected response
			return {
				"ResultCode": 0,
				"ResultDesc": "Success"
			}
			
		except Exception as e:
			logger.error(f"STK Push callback handling error: {str(e)}")
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"error": str(e),
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 1,
				"ResultDesc": f"Error processing callback: {str(e)}"
			}
	
	async def handle_b2b_result_callback(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle B2B payment result callback"""
		try:
			webhook_id = uuid7str()
			timestamp = datetime.utcnow().isoformat()
			
			# Log incoming webhook  
			self.webhook_logs[webhook_id] = {
				"id": webhook_id,
				"type": "b2b_result_callback",
				"timestamp": timestamp,
				"data": result_data,
				"processed": False
			}
			
			logger.info(f"Received B2B result callback: {webhook_id}")
			
			# Process with MPESA service
			result = await self.mpesa_service.handle_callback({"Result": result_data})
			
			# Update log with result
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"result": result,
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 0,
				"ResultDesc": "Success"
			}
			
		except Exception as e:
			logger.error(f"B2B result callback handling error: {str(e)}")
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"error": str(e),
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 1,
				"ResultDesc": f"Error processing callback: {str(e)}"
			}
	
	async def handle_b2c_result_callback(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle B2C payment result callback"""
		try:
			webhook_id = uuid7str()
			timestamp = datetime.utcnow().isoformat()
			
			# Log incoming webhook
			self.webhook_logs[webhook_id] = {
				"id": webhook_id,
				"type": "b2c_result_callback", 
				"timestamp": timestamp,
				"data": result_data,
				"processed": False
			}
			
			logger.info(f"Received B2C result callback: {webhook_id}")
			
			# Process with MPESA service
			result = await self.mpesa_service.handle_callback({"Result": result_data})
			
			# Update log with result
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"result": result,
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 0,
				"ResultDesc": "Success"
			}
			
		except Exception as e:
			logger.error(f"B2C result callback handling error: {str(e)}")
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"error": str(e),
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 1,
				"ResultDesc": f"Error processing callback: {str(e)}"
			}
	
	async def handle_c2b_validation(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle C2B validation callback"""
		try:
			webhook_id = uuid7str()
			timestamp = datetime.utcnow().isoformat()
			
			# Log incoming webhook
			self.webhook_logs[webhook_id] = {
				"id": webhook_id,
				"type": "c2b_validation",
				"timestamp": timestamp,
				"data": validation_data,
				"processed": False
			}
			
			logger.info(f"Received C2B validation: {webhook_id}")
			
			# Extract validation parameters
			transaction_type = validation_data.get("TransactionType")
			trans_id = validation_data.get("TransID")
			trans_time = validation_data.get("TransTime")
			trans_amount = validation_data.get("TransAmount")
			business_short_code = validation_data.get("BusinessShortCode")
			bill_ref_number = validation_data.get("BillRefNumber")
			invoice_number = validation_data.get("InvoiceNumber")
			org_account_balance = validation_data.get("OrgAccountBalance")
			third_party_trans_id = validation_data.get("ThirdPartyTransID")
			msisdn = validation_data.get("MSISDN")
			first_name = validation_data.get("FirstName")
			middle_name = validation_data.get("MiddleName")
			last_name = validation_data.get("LastName")
			
			# Perform validation logic
			validation_result = await self._validate_c2b_transaction({
				"transaction_type": transaction_type,
				"trans_id": trans_id,
				"amount": trans_amount,
				"business_short_code": business_short_code,
				"bill_ref_number": bill_ref_number,
				"msisdn": msisdn,
				"customer_name": f"{first_name or ''} {middle_name or ''} {last_name or ''}".strip()
			})
			
			# Update log with result
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"validation_result": validation_result,
				"processed_at": datetime.utcnow().isoformat()
			})
			
			if validation_result["valid"]:
				return {
					"ResultCode": 0,
					"ResultDesc": "Accepted"
				}
			else:
				return {
					"ResultCode": 1,
					"ResultDesc": validation_result.get("reason", "Validation failed")
				}
				
		except Exception as e:
			logger.error(f"C2B validation handling error: {str(e)}")
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"error": str(e),
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 1,
				"ResultDesc": f"Validation error: {str(e)}"
			}
	
	async def handle_c2b_confirmation(self, confirmation_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle C2B confirmation callback"""
		try:
			webhook_id = uuid7str()
			timestamp = datetime.utcnow().isoformat()
			
			# Log incoming webhook
			self.webhook_logs[webhook_id] = {
				"id": webhook_id,
				"type": "c2b_confirmation",
				"timestamp": timestamp,
				"data": confirmation_data,
				"processed": False
			}
			
			logger.info(f"Received C2B confirmation: {webhook_id}")
			
			# Extract confirmation parameters
			transaction_type = confirmation_data.get("TransactionType")
			trans_id = confirmation_data.get("TransID")
			trans_time = confirmation_data.get("TransTime")
			trans_amount = confirmation_data.get("TransAmount")
			business_short_code = confirmation_data.get("BusinessShortCode")
			bill_ref_number = confirmation_data.get("BillRefNumber")
			invoice_number = confirmation_data.get("InvoiceNumber")
			org_account_balance = confirmation_data.get("OrgAccountBalance")
			third_party_trans_id = confirmation_data.get("ThirdPartyTransID")
			msisdn = confirmation_data.get("MSISDN")
			first_name = confirmation_data.get("FirstName")
			middle_name = confirmation_data.get("MiddleName")
			last_name = confirmation_data.get("LastName")
			
			# Process confirmation
			confirmation_result = await self._process_c2b_confirmation({
				"transaction_type": transaction_type,
				"trans_id": trans_id,
				"trans_time": trans_time,
				"amount": trans_amount,
				"business_short_code": business_short_code,
				"bill_ref_number": bill_ref_number,
				"invoice_number": invoice_number,
				"org_account_balance": org_account_balance,
				"third_party_trans_id": third_party_trans_id,
				"msisdn": msisdn,
				"customer_name": f"{first_name or ''} {middle_name or ''} {last_name or ''}".strip()
			})
			
			# Update log with result
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"confirmation_result": confirmation_result,
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 0,
				"ResultDesc": "Success"
			}
			
		except Exception as e:
			logger.error(f"C2B confirmation handling error: {str(e)}")
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"error": str(e),
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 1,
				"ResultDesc": f"Confirmation error: {str(e)}"
			}
	
	async def handle_account_balance_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle account balance result callback"""
		try:
			webhook_id = uuid7str()
			timestamp = datetime.utcnow().isoformat()
			
			# Log incoming webhook
			self.webhook_logs[webhook_id] = {
				"id": webhook_id,
				"type": "account_balance_result",
				"timestamp": timestamp,
				"data": result_data,
				"processed": False
			}
			
			logger.info(f"Received account balance result: {webhook_id}")
			
			# Process with MPESA service
			result = await self.mpesa_service.handle_callback({"Result": result_data})
			
			# Update log with result
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"result": result,
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 0,
				"ResultDesc": "Success"
			}
			
		except Exception as e:
			logger.error(f"Account balance result handling error: {str(e)}")
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"error": str(e),
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 1,
				"ResultDesc": f"Error processing result: {str(e)}"
			}
	
	async def handle_transaction_status_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle transaction status result callback"""
		try:
			webhook_id = uuid7str()
			timestamp = datetime.utcnow().isoformat()
			
			# Log incoming webhook
			self.webhook_logs[webhook_id] = {
				"id": webhook_id,
				"type": "transaction_status_result",
				"timestamp": timestamp,
				"data": result_data,
				"processed": False
			}
			
			logger.info(f"Received transaction status result: {webhook_id}")
			
			# Process with MPESA service
			result = await self.mpesa_service.handle_callback({"Result": result_data})
			
			# Update log with result
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"result": result,
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 0,
				"ResultDesc": "Success"
			}
			
		except Exception as e:
			logger.error(f"Transaction status result handling error: {str(e)}")
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"error": str(e),
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 1,
				"ResultDesc": f"Error processing result: {str(e)}"
			}
	
	async def handle_transaction_reversal_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle transaction reversal result callback"""
		try:
			webhook_id = uuid7str()
			timestamp = datetime.utcnow().isoformat()
			
			# Log incoming webhook
			self.webhook_logs[webhook_id] = {
				"id": webhook_id,
				"type": "transaction_reversal_result",
				"timestamp": timestamp,
				"data": result_data,
				"processed": False
			}
			
			logger.info(f"Received transaction reversal result: {webhook_id}")
			
			# Process with MPESA service
			result = await self.mpesa_service.handle_callback({"Result": result_data})
			
			# Update log with result
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"result": result,
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 0,
				"ResultDesc": "Success"
			}
			
		except Exception as e:
			logger.error(f"Transaction reversal result handling error: {str(e)}")
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"error": str(e),
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 1,
				"ResultDesc": f"Error processing result: {str(e)}"
			}
	
	async def handle_timeout_callback(self, timeout_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle timeout callback"""
		try:
			webhook_id = uuid7str()
			timestamp = datetime.utcnow().isoformat()
			
			# Log incoming webhook
			self.webhook_logs[webhook_id] = {
				"id": webhook_id,
				"type": "timeout_callback",
				"timestamp": timestamp,
				"data": timeout_data,
				"processed": False
			}
			
			logger.info(f"Received timeout callback: {webhook_id}")
			
			# Handle timeout - mark transaction as timed out
			result = await self._handle_transaction_timeout(timeout_data)
			
			# Update log with result
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"result": result,
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 0,
				"ResultDesc": "Success"
			}
			
		except Exception as e:
			logger.error(f"Timeout callback handling error: {str(e)}")
			self.webhook_logs[webhook_id].update({
				"processed": True,
				"error": str(e),
				"processed_at": datetime.utcnow().isoformat()
			})
			
			return {
				"ResultCode": 1,
				"ResultDesc": f"Error processing timeout: {str(e)}"
			}
	
	async def _validate_c2b_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate C2B transaction"""
		try:
			# Basic validation rules
			amount = float(transaction_data.get("amount", 0))
			business_short_code = transaction_data.get("business_short_code")
			bill_ref_number = transaction_data.get("bill_ref_number")
			msisdn = transaction_data.get("msisdn")
			
			# Validation checks
			if amount <= 0:
				return {"valid": False, "reason": "Invalid amount"}
			
			if amount > 1000000:  # Max amount check
				return {"valid": False, "reason": "Amount exceeds maximum limit"}
			
			if not business_short_code or business_short_code != self.mpesa_service.config.credentials.business_short_code:
				return {"valid": False, "reason": "Invalid business short code"}
			
			if not bill_ref_number:
				return {"valid": False, "reason": "Bill reference number is required"}
			
			if not msisdn or len(msisdn) < 10:
				return {"valid": False, "reason": "Invalid phone number"}
			
			# Additional custom validation logic can be added here
			# For example, checking against customer database, blacklists, etc.
			
			logger.info(f"C2B transaction validated successfully: {bill_ref_number}")
			
			return {
				"valid": True,
				"amount": amount,
				"business_short_code": business_short_code,
				"bill_ref_number": bill_ref_number,
				"msisdn": msisdn
			}
			
		except Exception as e:
			logger.error(f"C2B validation error: {str(e)}")
			return {"valid": False, "reason": f"Validation error: {str(e)}"}
	
	async def _process_c2b_confirmation(self, confirmation_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process C2B confirmation"""
		try:
			# Extract confirmation details
			trans_id = confirmation_data.get("trans_id")
			amount = float(confirmation_data.get("amount", 0))
			bill_ref_number = confirmation_data.get("bill_ref_number")
			msisdn = confirmation_data.get("msisdn")
			customer_name = confirmation_data.get("customer_name")
			trans_time = confirmation_data.get("trans_time")
			
			# Store confirmed transaction
			confirmed_transaction = {
				"transaction_id": bill_ref_number or trans_id,
				"mpesa_receipt_number": trans_id,
				"amount": amount,
				"phone_number": msisdn,
				"customer_name": customer_name,
				"transaction_time": trans_time,
				"business_short_code": confirmation_data.get("business_short_code"),
				"bill_ref_number": bill_ref_number,
				"invoice_number": confirmation_data.get("invoice_number"),
				"org_account_balance": confirmation_data.get("org_account_balance"),
				"third_party_trans_id": confirmation_data.get("third_party_trans_id"),
				"status": "completed",
				"confirmed_at": datetime.utcnow().isoformat()
			}
			
			# Store in MPESA service completed transactions
			self.mpesa_service._completed_transactions[trans_id] = confirmed_transaction
			
			logger.info(f"C2B transaction confirmed: {trans_id} - Amount: {amount}")
			
			return {
				"success": True,
				"transaction_id": trans_id,
				"amount": amount,
				"customer_name": customer_name,
				"status": "completed"
			}
			
		except Exception as e:
			logger.error(f"C2B confirmation processing error: {str(e)}")
			return {"success": False, "error": str(e)}
	
	async def _handle_transaction_timeout(self, timeout_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle transaction timeout"""
		try:
			conversation_id = timeout_data.get("ConversationID")
			originator_conversation_id = timeout_data.get("OriginatorConversationID")
			
			# Find and mark transaction as timed out
			for tx_id, tx_data in list(self.mpesa_service._pending_transactions.items()):
				if (tx_data.get("conversation_id") == conversation_id or
					tx_data.get("originator_conversation_id") == originator_conversation_id):
					
					# Move to completed with timeout status
					timeout_tx = {
						**tx_data,
						"status": "timeout",
						"timeout_at": datetime.utcnow().isoformat(),
						"timeout_reason": "Transaction timed out"
					}
					
					self.mpesa_service._completed_transactions[tx_id] = timeout_tx
					del self.mpesa_service._pending_transactions[tx_id]
					
					logger.info(f"Transaction marked as timed out: {conversation_id}")
					break
			
			return {
				"success": True,
				"conversation_id": conversation_id,
				"status": "timeout"
			}
			
		except Exception as e:
			logger.error(f"Timeout handling error: {str(e)}")
			return {"success": False, "error": str(e)}
	
	def get_webhook_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
		"""Get recent webhook logs"""
		logs = list(self.webhook_logs.values())
		# Sort by timestamp, most recent first
		logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
		return logs[:limit]
	
	def get_webhook_stats(self) -> Dict[str, Any]:
		"""Get webhook processing statistics"""
		total_webhooks = len(self.webhook_logs)
		processed_webhooks = sum(1 for log in self.webhook_logs.values() if log.get("processed", False))
		error_webhooks = sum(1 for log in self.webhook_logs.values() if log.get("error"))
		
		# Count by type
		type_counts = {}
		for log in self.webhook_logs.values():
			webhook_type = log.get("type", "unknown")
			type_counts[webhook_type] = type_counts.get(webhook_type, 0) + 1
		
		return {
			"total_webhooks": total_webhooks,
			"processed_webhooks": processed_webhooks,
			"error_webhooks": error_webhooks,
			"success_rate": (processed_webhooks - error_webhooks) / max(1, processed_webhooks),
			"webhook_types": type_counts,
			"last_webhook": max([log.get("timestamp", "") for log in self.webhook_logs.values()]) if self.webhook_logs else None
		}

# Flask Blueprint for MPESA webhooks

def create_mpesa_webhook_blueprint(webhook_handler: MPESAWebhookHandler) -> Blueprint:
	"""Create Flask blueprint for MPESA webhooks"""
	
	mpesa_webhook_bp = Blueprint('mpesa_webhooks', __name__, url_prefix='/mpesa')
	
	@mpesa_webhook_bp.route('/stk-push-callback', methods=['POST'])
	async def stk_push_callback():
		"""STK Push callback endpoint"""
		try:
			callback_data = request.get_json()
			logger.info(f"STK Push callback received: {callback_data}")
			
			result = await webhook_handler.handle_stk_push_callback(callback_data)
			return jsonify(result)
			
		except Exception as e:
			logger.error(f"STK Push callback error: {str(e)}")
			return jsonify({"ResultCode": 1, "ResultDesc": str(e)}), 500
	
	@mpesa_webhook_bp.route('/b2b-result', methods=['POST'])
	async def b2b_result():
		"""B2B result callback endpoint"""
		try:
			result_data = request.get_json()
			logger.info(f"B2B result received: {result_data}")
			
			result = await webhook_handler.handle_b2b_result_callback(result_data)
			return jsonify(result)
			
		except Exception as e:
			logger.error(f"B2B result callback error: {str(e)}")
			return jsonify({"ResultCode": 1, "ResultDesc": str(e)}), 500
	
	@mpesa_webhook_bp.route('/b2c-result', methods=['POST'])
	async def b2c_result():
		"""B2C result callback endpoint"""
		try:
			result_data = request.get_json()
			logger.info(f"B2C result received: {result_data}")
			
			result = await webhook_handler.handle_b2c_result_callback(result_data)
			return jsonify(result)
			
		except Exception as e:
			logger.error(f"B2C result callback error: {str(e)}")
			return jsonify({"ResultCode": 1, "ResultDesc": str(e)}), 500
	
	@mpesa_webhook_bp.route('/c2b-validation', methods=['POST'])
	async def c2b_validation():
		"""C2B validation endpoint"""
		try:
			validation_data = request.get_json()
			logger.info(f"C2B validation received: {validation_data}")
			
			result = await webhook_handler.handle_c2b_validation(validation_data)
			return jsonify(result)
			
		except Exception as e:
			logger.error(f"C2B validation error: {str(e)}")
			return jsonify({"ResultCode": 1, "ResultDesc": str(e)}), 500
	
	@mpesa_webhook_bp.route('/c2b-confirmation', methods=['POST'])
	async def c2b_confirmation():
		"""C2B confirmation endpoint"""
		try:
			confirmation_data = request.get_json()
			logger.info(f"C2B confirmation received: {confirmation_data}")
			
			result = await webhook_handler.handle_c2b_confirmation(confirmation_data)
			return jsonify(result)
			
		except Exception as e:
			logger.error(f"C2B confirmation error: {str(e)}")
			return jsonify({"ResultCode": 1, "ResultDesc": str(e)}), 500
	
	@mpesa_webhook_bp.route('/account-balance-result', methods=['POST'])
	async def account_balance_result():
		"""Account balance result endpoint"""
		try:
			result_data = request.get_json()
			logger.info(f"Account balance result received: {result_data}")
			
			result = await webhook_handler.handle_account_balance_result(result_data)
			return jsonify(result)
			
		except Exception as e:
			logger.error(f"Account balance result error: {str(e)}")
			return jsonify({"ResultCode": 1, "ResultDesc": str(e)}), 500
	
	@mpesa_webhook_bp.route('/transaction-status-result', methods=['POST'])
	async def transaction_status_result():
		"""Transaction status result endpoint"""
		try:
			result_data = request.get_json()
			logger.info(f"Transaction status result received: {result_data}")
			
			result = await webhook_handler.handle_transaction_status_result(result_data)
			return jsonify(result)
			
		except Exception as e:
			logger.error(f"Transaction status result error: {str(e)}")
			return jsonify({"ResultCode": 1, "ResultDesc": str(e)}), 500
	
	@mpesa_webhook_bp.route('/transaction-reversal-result', methods=['POST'])
	async def transaction_reversal_result():
		"""Transaction reversal result endpoint"""
		try:
			result_data = request.get_json()
			logger.info(f"Transaction reversal result received: {result_data}")
			
			result = await webhook_handler.handle_transaction_reversal_result(result_data)
			return jsonify(result)
			
		except Exception as e:
			logger.error(f"Transaction reversal result error: {str(e)}")
			return jsonify({"ResultCode": 1, "ResultDesc": str(e)}), 500
	
	@mpesa_webhook_bp.route('/timeout', methods=['POST'])
	async def timeout():
		"""Timeout callback endpoint"""
		try:
			timeout_data = request.get_json()
			logger.info(f"Timeout callback received: {timeout_data}")
			
			result = await webhook_handler.handle_timeout_callback(timeout_data)
			return jsonify(result)
			
		except Exception as e:
			logger.error(f"Timeout callback error: {str(e)}")
			return jsonify({"ResultCode": 1, "ResultDesc": str(e)}), 500
	
	@mpesa_webhook_bp.route('/webhook-logs', methods=['GET'])
	def webhook_logs():
		"""Get webhook logs"""
		try:
			limit = request.args.get('limit', 100, type=int)
			logs = webhook_handler.get_webhook_logs(limit)
			return jsonify({"success": True, "logs": logs})
			
		except Exception as e:
			logger.error(f"Webhook logs error: {str(e)}")
			return jsonify({"success": False, "error": str(e)}), 500
	
	@mpesa_webhook_bp.route('/webhook-stats', methods=['GET'])
	def webhook_stats():
		"""Get webhook statistics"""
		try:
			stats = webhook_handler.get_webhook_stats()
			return jsonify({"success": True, "stats": stats})
			
		except Exception as e:
			logger.error(f"Webhook stats error: {str(e)}")
			return jsonify({"success": False, "error": str(e)}), 500
	
	return mpesa_webhook_bp

def _log_webhook_handler_module_loaded():
	"""Log webhook handler module loaded"""
	print("🔗 MPESA Webhook Handler module loaded")
	print("   - STK Push callbacks")
	print("   - B2B/B2C result callbacks")
	print("   - C2B validation and confirmation")
	print("   - Account balance result callbacks")
	print("   - Transaction status result callbacks")
	print("   - Transaction reversal result callbacks")
	print("   - Timeout handling")
	print("   - Webhook logging and statistics")

# Execute module loading log
_log_webhook_handler_module_loaded()