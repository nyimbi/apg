"""
Subscription API Endpoints - APG Payment Gateway

Flask-AppBuilder API endpoints for subscription and recurring payment management.

© 2025 Datacraft. All rights reserved.
"""

from flask import request, jsonify
from flask_appbuilder import BaseView, expose
from flask_appbuilder.security.decorators import has_access
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from .subscription_service import SubscriptionService, BillingCycle, SubscriptionStatus
from .database import get_database_service
from .auth import authenticate_api_key, require_permission


class SubscriptionAPI(BaseView):
	"""Subscription API endpoints"""
	
	route_base = "/api/v1/subscriptions"
	
	def __init__(self):
		super().__init__()
		self._subscription_service: Optional[SubscriptionService] = None
		self._loop = None
	
	def _get_subscription_service(self) -> SubscriptionService:
		"""Get or create subscription service instance"""
		if self._subscription_service is None:
			from .subscription_service import create_subscription_service
			database_service = get_database_service()
			self._subscription_service = create_subscription_service(database_service)
		return self._subscription_service
	
	def _get_event_loop(self):
		"""Get or create event loop for async operations"""
		if self._loop is None:
			try:
				self._loop = asyncio.get_event_loop()
			except RuntimeError:
				self._loop = asyncio.new_event_loop()
				asyncio.set_event_loop(self._loop)
		return self._loop
	
	def _run_async(self, coro):
		"""Run async coroutine in sync context"""
		loop = self._get_event_loop()
		return loop.run_until_complete(coro)
	
	# Plan Management Endpoints
	
	@expose('/plans', methods=['POST'])
	@has_access
	def create_plan(self):
		"""Create a new subscription plan"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "subscriptions.manage"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			data = request.get_json()
			if not data:
				return jsonify({"error": "Request body required"}), 400
			
			# Validate required fields
			required_fields = ["name", "description", "amount", "billing_cycle"]
			for field in required_fields:
				if field not in data:
					return jsonify({"error": f"Missing required field: {field}"}), 400
			
			# Validate billing cycle
			try:
				BillingCycle(data["billing_cycle"])
			except ValueError:
				return jsonify({"error": f"Invalid billing cycle: {data['billing_cycle']}"}), 400
			
			subscription_service = self._get_subscription_service()
			plan = self._run_async(subscription_service.create_plan(data))
			
			return jsonify({
				"success": True,
				"plan": plan.to_dict()
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/plans/<plan_id>', methods=['GET'])
	@has_access
	def get_plan(self, plan_id: str):
		"""Get a subscription plan by ID"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			subscription_service = self._get_subscription_service()
			plan = self._run_async(subscription_service.get_plan(plan_id))
			
			if not plan:
				return jsonify({"error": "Plan not found"}), 404
			
			return jsonify({
				"success": True,
				"plan": plan.to_dict()
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/plans', methods=['GET'])
	@has_access
	def list_plans(self):
		"""List subscription plans"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Get query parameters
			merchant_id = request.args.get('merchant_id')
			active_only = request.args.get('active_only', 'true').lower() == 'true'
			
			if not merchant_id and auth_result["user"]["role"] == "merchant":
				merchant_id = auth_result["user"]["merchant_id"]
			
			subscription_service = self._get_subscription_service()
			plans = self._run_async(subscription_service.list_plans(merchant_id, active_only))
			
			return jsonify({
				"success": True,
				"plans": [plan.to_dict() for plan in plans],
				"total": len(plans)
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/plans/<plan_id>', methods=['PUT'])
	@has_access
	def update_plan(self, plan_id: str):
		"""Update a subscription plan"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "subscriptions.manage"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			data = request.get_json()
			if not data:
				return jsonify({"error": "Request body required"}), 400
			
			subscription_service = self._get_subscription_service()
			plan = self._run_async(subscription_service.update_plan(plan_id, data))
			
			return jsonify({
				"success": True,
				"plan": plan.to_dict()
			})
		
		except ValueError as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 404
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	# Subscription Management Endpoints
	
	@expose('/', methods=['POST'])
	@has_access
	def create_subscription(self):
		"""Create a new subscription"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "subscriptions.create"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			data = request.get_json()
			if not data:
				return jsonify({"error": "Request body required"}), 400
			
			# Validate required fields
			required_fields = ["customer_id", "plan_id", "payment_method_id"]
			for field in required_fields:
				if field not in data:
					return jsonify({"error": f"Missing required field: {field}"}), 400
			
			# Set merchant_id from user context if not provided
			if "merchant_id" not in data:
				if auth_result["user"]["role"] == "merchant":
					data["merchant_id"] = auth_result["user"]["merchant_id"]
				else:
					return jsonify({"error": "merchant_id required"}), 400
			
			subscription_service = self._get_subscription_service()
			subscription = self._run_async(subscription_service.create_subscription(data))
			
			return jsonify({
				"success": True,
				"subscription": subscription.to_dict()
			})
		
		except ValueError as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 400
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/<subscription_id>', methods=['GET'])
	@has_access
	def get_subscription(self, subscription_id: str):
		"""Get a subscription by ID"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			subscription_service = self._get_subscription_service()
			subscription = self._run_async(subscription_service.get_subscription(subscription_id))
			
			if not subscription:
				return jsonify({"error": "Subscription not found"}), 404
			
			# Check access permissions for merchant users
			if (auth_result["user"]["role"] == "merchant" and 
				subscription.merchant_id != auth_result["user"]["merchant_id"]):
				return jsonify({"error": "Access denied"}), 403
			
			return jsonify({
				"success": True,
				"subscription": subscription.to_dict()
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/<subscription_id>', methods=['PUT'])
	@has_access
	def update_subscription(self, subscription_id: str):
		"""Update a subscription"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "subscriptions.manage"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			data = request.get_json()
			if not data:
				return jsonify({"error": "Request body required"}), 400
			
			subscription_service = self._get_subscription_service()
			subscription = self._run_async(subscription_service.update_subscription(subscription_id, data))
			
			return jsonify({
				"success": True,
				"subscription": subscription.to_dict()
			})
		
		except ValueError as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 404
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/<subscription_id>/cancel', methods=['POST'])
	@has_access
	def cancel_subscription(self, subscription_id: str):
		"""Cancel a subscription"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "subscriptions.manage"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			data = request.get_json() or {}
			cancel_at_period_end = data.get("cancel_at_period_end", True)
			reason = data.get("reason")
			
			subscription_service = self._get_subscription_service()
			subscription = self._run_async(subscription_service.cancel_subscription(
				subscription_id, 
				cancel_at_period_end, 
				reason
			))
			
			return jsonify({
				"success": True,
				"subscription": subscription.to_dict(),
				"message": f"Subscription cancelled {'at period end' if cancel_at_period_end else 'immediately'}"
			})
		
		except ValueError as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 404
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/<subscription_id>/pause', methods=['POST'])
	@has_access
	def pause_subscription(self, subscription_id: str):
		"""Pause a subscription"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "subscriptions.manage"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			data = request.get_json() or {}
			resume_at = None
			if "resume_at" in data:
				try:
					resume_at = datetime.fromisoformat(data["resume_at"].replace('Z', '+00:00'))
				except ValueError:
					return jsonify({"error": "Invalid resume_at format"}), 400
			
			subscription_service = self._get_subscription_service()
			subscription = self._run_async(subscription_service.pause_subscription(subscription_id, resume_at))
			
			return jsonify({
				"success": True,
				"subscription": subscription.to_dict(),
				"message": "Subscription paused"
			})
		
		except ValueError as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 404
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/<subscription_id>/resume', methods=['POST'])
	@has_access
	def resume_subscription(self, subscription_id: str):
		"""Resume a paused subscription"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "subscriptions.manage"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			subscription_service = self._get_subscription_service()
			subscription = self._run_async(subscription_service.resume_subscription(subscription_id))
			
			return jsonify({
				"success": True,
				"subscription": subscription.to_dict(),
				"message": "Subscription resumed"
			})
		
		except ValueError as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 404
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	# Invoice Management Endpoints
	
	@expose('/<subscription_id>/invoices', methods=['POST'])
	@has_access
	def create_invoice(self, subscription_id: str):
		"""Create an invoice for a subscription"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "invoices.create"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			data = request.get_json()
			if not data:
				return jsonify({"error": "Request body required"}), 400
			
			if "amount" not in data:
				return jsonify({"error": "Missing required field: amount"}), 400
			
			subscription_service = self._get_subscription_service()
			invoice = self._run_async(subscription_service.create_invoice(
				subscription_id,
				data["amount"],
				data.get("description")
			))
			
			return jsonify({
				"success": True,
				"invoice": invoice.to_dict()
			})
		
		except ValueError as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 404
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/invoices/<invoice_id>/pay', methods=['POST'])
	@has_access
	def pay_invoice(self, invoice_id: str):
		"""Process payment for an invoice"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "invoices.pay"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			subscription_service = self._get_subscription_service()
			result = self._run_async(subscription_service.process_invoice_payment(invoice_id))
			
			if result["success"]:
				return jsonify({
					"success": True,
					"message": "Invoice paid successfully",
					"transaction_id": result.get("transaction_id")
				})
			else:
				return jsonify({
					"success": False,
					"error": result["error"]
				}), 402  # Payment required
		
		except ValueError as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 404
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	# Analytics and Reporting Endpoints
	
	@expose('/analytics', methods=['GET'])
	@has_access
	def get_subscription_analytics(self):
		"""Get subscription analytics"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Get query parameters
			merchant_id = request.args.get('merchant_id')
			start_date = request.args.get('start_date')
			end_date = request.args.get('end_date')
			
			if not merchant_id and auth_result["user"]["role"] == "merchant":
				merchant_id = auth_result["user"]["merchant_id"]
			
			# Default to last 30 days if no dates provided
			if not start_date or not end_date:
				end_date = datetime.now(timezone.utc)
				start_date = end_date - timedelta(days=30)
			else:
				try:
					start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
					end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
				except ValueError:
					return jsonify({"error": "Invalid date format"}), 400
			
			# Get subscription analytics (mock data for now)
			analytics = await self._get_subscription_analytics(merchant_id, start_date, end_date)
			
			return jsonify({
				"success": True,
				"analytics": analytics
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/health', methods=['GET'])
	@has_access
	def subscription_health_check(self):
		"""Get subscription service health status"""
		try:
			subscription_service = self._get_subscription_service()
			
			health_data = {
				"service": "healthy" if subscription_service._initialized else "not_initialized",
				"billing_scheduler": "running" if subscription_service._billing_scheduler_task else "stopped",
				"dunning_processor": "running" if subscription_service._dunning_processor_task else "stopped",
				"plans_cached": len(subscription_service._plans_cache),
				"subscriptions_cached": len(subscription_service._subscriptions_cache),
				"timestamp": datetime.now(timezone.utc).isoformat()
			}
			
			return jsonify({
				"success": True,
				"health": health_data
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	# Helper methods
	
	async def _get_subscription_analytics(self, merchant_id: Optional[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Get subscription analytics data"""
		# This would query the database for real analytics
		# For now, return mock data
		return {
			"date_range": {
				"start": start_date.isoformat(),
				"end": end_date.isoformat()
			},
			"summary": {
				"total_subscriptions": 156,
				"active_subscriptions": 134,
				"cancelled_subscriptions": 22,
				"paused_subscriptions": 8,
				"trial_subscriptions": 15
			},
			"revenue": {
				"monthly_recurring_revenue": 45750.00,
				"annual_recurring_revenue": 549000.00,
				"average_revenue_per_user": 341.42,
				"total_revenue_period": 15250.00
			},
			"churn": {
				"churn_rate": 3.2,
				"retention_rate": 96.8,
				"avg_subscription_length_days": 247
			},
			"billing_cycles": {
				"monthly": 89,
				"quarterly": 34,
				"annually": 33
			},
			"top_plans": [
				{"plan_name": "Pro Plan", "subscribers": 67, "revenue": 25125.00},
				{"plan_name": "Basic Plan", "subscribers": 45, "revenue": 8950.00},
				{"plan_name": "Enterprise Plan", "subscribers": 22, "revenue": 11675.00}
			]
		}


def _log_subscription_api_module_loaded():
	"""Log subscription API module loaded"""
	print("💳 Subscription API module loaded")
	print("   - Plan management endpoints")
	print("   - Subscription lifecycle management")
	print("   - Invoice and payment processing")
	print("   - Analytics and reporting")
	print("   - Health monitoring")


# Execute module loading log
_log_subscription_api_module_loaded()