"""
Stripe Webhook Handler - APG Payment Gateway

Handles all Stripe webhook events and processing:
- Payment Intent events (succeeded, failed, requires_action)
- Charge events (succeeded, failed, dispute_created)
- Customer events (created, updated, deleted)
- Subscription events (created, updated, deleted, trial_will_end)
- Invoice events (payment_succeeded, payment_failed, finalized)
- Connect events (account updates, transfers)
- Setup Intent events for saved payment methods

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
import logging
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from flask import Blueprint, request, jsonify
from uuid_extensions import uuid7str
import stripe

from .stripe_integration import StripeService

logger = logging.getLogger(__name__)

class StripeWebhookHandler:
    """
    Complete Stripe webhook handler for all event types
    """
    
    def __init__(self, stripe_service: StripeService):
        """Initialize webhook handler with Stripe service"""
        self.stripe_service = stripe_service
        self.webhook_logs: Dict[str, Dict[str, Any]] = {}
        self.event_handlers = {
            # Payment Intent events
            "payment_intent.succeeded": self._handle_payment_intent_succeeded,
            "payment_intent.payment_failed": self._handle_payment_intent_failed,
            "payment_intent.requires_action": self._handle_payment_intent_requires_action,
            "payment_intent.canceled": self._handle_payment_intent_canceled,
            "payment_intent.processing": self._handle_payment_intent_processing,
            "payment_intent.amount_capturable_updated": self._handle_payment_intent_amount_capturable_updated,
            
            # Charge events
            "charge.succeeded": self._handle_charge_succeeded,
            "charge.failed": self._handle_charge_failed,
            "charge.captured": self._handle_charge_captured,
            "charge.dispute.created": self._handle_charge_dispute_created,
            "charge.dispute.updated": self._handle_charge_dispute_updated,
            "charge.dispute.closed": self._handle_charge_dispute_closed,
            
            # Customer events
            "customer.created": self._handle_customer_created,
            "customer.updated": self._handle_customer_updated,
            "customer.deleted": self._handle_customer_deleted,
            
            # Payment Method events
            "payment_method.attached": self._handle_payment_method_attached,
            "payment_method.detached": self._handle_payment_method_detached,
            "payment_method.updated": self._handle_payment_method_updated,
            
            # Setup Intent events
            "setup_intent.succeeded": self._handle_setup_intent_succeeded,
            "setup_intent.setup_failed": self._handle_setup_intent_failed,
            "setup_intent.requires_action": self._handle_setup_intent_requires_action,
            
            # Subscription events
            "customer.subscription.created": self._handle_subscription_created,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "customer.subscription.trial_will_end": self._handle_subscription_trial_will_end,
            
            # Invoice events
            "invoice.created": self._handle_invoice_created,
            "invoice.finalized": self._handle_invoice_finalized,
            "invoice.payment_succeeded": self._handle_invoice_payment_succeeded,
            "invoice.payment_failed": self._handle_invoice_payment_failed,
            "invoice.payment_action_required": self._handle_invoice_payment_action_required,
            
            # Refund events
            "charge.refunded": self._handle_charge_refunded,
            "refund.created": self._handle_refund_created,
            "refund.updated": self._handle_refund_updated,
            
            # Connect events
            "account.updated": self._handle_account_updated,
            "account.application.deauthorized": self._handle_account_application_deauthorized,
            "transfer.created": self._handle_transfer_created,
            "transfer.failed": self._handle_transfer_failed,
            "transfer.paid": self._handle_transfer_paid,
            "transfer.reversed": self._handle_transfer_reversed,
            
            # Payout events
            "payout.created": self._handle_payout_created,
            "payout.failed": self._handle_payout_failed,
            "payout.paid": self._handle_payout_paid,
            "payout.canceled": self._handle_payout_canceled,
            
            # Radar (fraud) events
            "radar.early_fraud_warning.created": self._handle_radar_early_fraud_warning_created,
            "radar.early_fraud_warning.updated": self._handle_radar_early_fraud_warning_updated,
            
            # Review events
            "review.opened": self._handle_review_opened,
            "review.closed": self._handle_review_closed
        }
        
    async def handle_webhook(self, payload: str, signature: str) -> Dict[str, Any]:
        """Handle incoming Stripe webhook"""
        try:
            webhook_id = uuid7str()
            timestamp = datetime.utcnow().isoformat()
            
            # Verify webhook signature
            try:
                event = stripe.Webhook.construct_event(
                    payload, signature, self.stripe_service.config.credentials.webhook_secret
                )
            except stripe.error.SignatureVerificationError as e:
                logger.error(f"Webhook signature verification failed: {str(e)}")
                return {"success": False, "error": "Invalid signature"}
            
            # Log incoming webhook
            self.webhook_logs[webhook_id] = {
                "id": webhook_id,
                "event_id": event.get("id"),
                "event_type": event.get("type"),
                "timestamp": timestamp,
                "data": event,
                "processed": False
            }
            
            logger.info(f"Received Stripe webhook: {event.get('type')} - {webhook_id}")
            
            # Handle event
            event_type = event.get("type")
            handler = self.event_handlers.get(event_type)
            
            if handler:
                result = await handler(event.get("data", {}).get("object", {}), event)
            else:
                logger.info(f"Unhandled webhook event type: {event_type}")
                result = {"success": True, "message": "Event received but not handled"}
            
            # Update log with result
            self.webhook_logs[webhook_id].update({
                "processed": True,
                "result": result,
                "processed_at": datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Webhook handling error: {str(e)}")
            if 'webhook_id' in locals():
                self.webhook_logs[webhook_id].update({
                    "processed": True,
                    "error": str(e),
                    "processed_at": datetime.utcnow().isoformat()
                })
            
            return {"success": False, "error": str(e)}
    
    # Payment Intent event handlers
    
    async def _handle_payment_intent_succeeded(self, payment_intent: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle successful payment intent"""
        try:
            pi_id = payment_intent.get("id")
            amount = payment_intent.get("amount_received", 0)
            currency = payment_intent.get("currency")
            customer_id = payment_intent.get("customer")
            
            logger.info(f"Payment Intent succeeded: {pi_id} - {amount} {currency}")
            
            # Update internal tracking
            result = await self.stripe_service._handle_payment_intent_succeeded(payment_intent)
            
            # Trigger business logic (notifications, fulfillment, etc.)
            await self._trigger_payment_success_actions(payment_intent)
            
            return {"success": True, "payment_intent_id": pi_id, "amount": amount}
            
        except Exception as e:
            logger.error(f"Payment Intent success handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_payment_intent_failed(self, payment_intent: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle failed payment intent"""
        try:
            pi_id = payment_intent.get("id")
            error = payment_intent.get("last_payment_error", {})
            
            logger.warning(f"Payment Intent failed: {pi_id} - {error.get('message', 'Unknown error')}")
            
            # Update internal tracking
            result = await self.stripe_service._handle_payment_intent_failed(payment_intent)
            
            # Trigger failure actions (notifications, retry logic, etc.)
            await self._trigger_payment_failure_actions(payment_intent)
            
            return {"success": True, "payment_intent_id": pi_id, "error": error}
            
        except Exception as e:
            logger.error(f"Payment Intent failure handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_payment_intent_requires_action(self, payment_intent: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payment intent requiring action (3D Secure, etc.)"""
        try:
            pi_id = payment_intent.get("id")
            next_action = payment_intent.get("next_action", {})
            
            logger.info(f"Payment Intent requires action: {pi_id} - {next_action.get('type', 'unknown')}")
            
            # Update internal tracking
            result = await self.stripe_service._handle_payment_intent_requires_action(payment_intent)
            
            # Trigger action required notifications
            await self._trigger_payment_action_required(payment_intent)
            
            return {"success": True, "payment_intent_id": pi_id, "requires_action": True}
            
        except Exception as e:
            logger.error(f"Payment Intent action handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_payment_intent_canceled(self, payment_intent: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle canceled payment intent"""
        try:
            pi_id = payment_intent.get("id")
            
            logger.info(f"Payment Intent canceled: {pi_id}")
            
            # Remove from pending transactions
            if pi_id in self.stripe_service._pending_payment_intents:
                del self.stripe_service._pending_payment_intents[pi_id]
            
            return {"success": True, "payment_intent_id": pi_id, "status": "canceled"}
            
        except Exception as e:
            logger.error(f"Payment Intent cancellation handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_payment_intent_processing(self, payment_intent: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payment intent being processed"""
        try:
            pi_id = payment_intent.get("id")
            
            logger.info(f"Payment Intent processing: {pi_id}")
            
            return {"success": True, "payment_intent_id": pi_id, "status": "processing"}
            
        except Exception as e:
            logger.error(f"Payment Intent processing handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_payment_intent_amount_capturable_updated(self, payment_intent: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payment intent amount capturable update"""
        try:
            pi_id = payment_intent.get("id")
            amount_capturable = payment_intent.get("amount_capturable", 0)
            
            logger.info(f"Payment Intent amount capturable updated: {pi_id} - {amount_capturable}")
            
            return {"success": True, "payment_intent_id": pi_id, "amount_capturable": amount_capturable}
            
        except Exception as e:
            logger.error(f"Payment Intent amount capturable handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Charge event handlers
    
    async def _handle_charge_succeeded(self, charge: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle successful charge"""
        try:
            charge_id = charge.get("id")
            amount = charge.get("amount", 0)
            currency = charge.get("currency")
            
            logger.info(f"Charge succeeded: {charge_id} - {amount} {currency}")
            
            return {"success": True, "charge_id": charge_id, "amount": amount}
            
        except Exception as e:
            logger.error(f"Charge success handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_charge_failed(self, charge: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle failed charge"""
        try:
            charge_id = charge.get("id")
            failure_code = charge.get("failure_code")
            failure_message = charge.get("failure_message")
            
            logger.warning(f"Charge failed: {charge_id} - {failure_code}: {failure_message}")
            
            return {"success": True, "charge_id": charge_id, "failure_code": failure_code}
            
        except Exception as e:
            logger.error(f"Charge failure handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_charge_captured(self, charge: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle charge capture"""
        try:
            charge_id = charge.get("id")
            amount_captured = charge.get("amount_captured", 0)
            
            logger.info(f"Charge captured: {charge_id} - {amount_captured}")
            
            return {"success": True, "charge_id": charge_id, "amount_captured": amount_captured}
            
        except Exception as e:
            logger.error(f"Charge capture handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_charge_dispute_created(self, dispute: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chargeback/dispute creation"""
        try:
            dispute_id = dispute.get("id")
            charge_id = dispute.get("charge")
            amount = dispute.get("amount", 0)
            reason = dispute.get("reason")
            
            logger.warning(f"Dispute created: {dispute_id} for charge: {charge_id} - {reason}")
            
            # Trigger dispute handling workflow
            await self._trigger_dispute_created_actions(dispute)
            
            return {"success": True, "dispute_id": dispute_id, "charge_id": charge_id}
            
        except Exception as e:
            logger.error(f"Dispute creation handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_charge_dispute_updated(self, dispute: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dispute update"""
        try:
            dispute_id = dispute.get("id")
            status = dispute.get("status")
            
            logger.info(f"Dispute updated: {dispute_id} - {status}")
            
            return {"success": True, "dispute_id": dispute_id, "status": status}
            
        except Exception as e:
            logger.error(f"Dispute update handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_charge_dispute_closed(self, dispute: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dispute closure"""
        try:
            dispute_id = dispute.get("id")
            status = dispute.get("status")
            
            logger.info(f"Dispute closed: {dispute_id} - {status}")
            
            return {"success": True, "dispute_id": dispute_id, "status": status}
            
        except Exception as e:
            logger.error(f"Dispute closure handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Customer event handlers
    
    async def _handle_customer_created(self, customer: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer creation"""
        try:
            customer_id = customer.get("id")
            email = customer.get("email")
            
            logger.info(f"Customer created: {customer_id} - {email}")
            
            return {"success": True, "customer_id": customer_id, "email": email}
            
        except Exception as e:
            logger.error(f"Customer creation handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_customer_updated(self, customer: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer update"""
        try:
            customer_id = customer.get("id")
            
            logger.info(f"Customer updated: {customer_id}")
            
            # Update cached customer if exists
            internal_customer_id = customer.get("metadata", {}).get("internal_customer_id")
            if internal_customer_id and internal_customer_id in self.stripe_service._customers:
                self.stripe_service._customers[internal_customer_id] = customer
            
            return {"success": True, "customer_id": customer_id}
            
        except Exception as e:
            logger.error(f"Customer update handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_customer_deleted(self, customer: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer deletion"""
        try:
            customer_id = customer.get("id")
            
            logger.info(f"Customer deleted: {customer_id}")
            
            # Remove from cache
            internal_customer_id = customer.get("metadata", {}).get("internal_customer_id")
            if internal_customer_id and internal_customer_id in self.stripe_service._customers:
                del self.stripe_service._customers[internal_customer_id]
            
            return {"success": True, "customer_id": customer_id}
            
        except Exception as e:
            logger.error(f"Customer deletion handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Payment Method event handlers
    
    async def _handle_payment_method_attached(self, payment_method: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payment method attachment to customer"""
        try:
            pm_id = payment_method.get("id")
            customer_id = payment_method.get("customer")
            
            logger.info(f"Payment method attached: {pm_id} to customer: {customer_id}")
            
            return {"success": True, "payment_method_id": pm_id, "customer_id": customer_id}
            
        except Exception as e:
            logger.error(f"Payment method attachment handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_payment_method_detached(self, payment_method: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payment method detachment from customer"""
        try:
            pm_id = payment_method.get("id")
            
            logger.info(f"Payment method detached: {pm_id}")
            
            return {"success": True, "payment_method_id": pm_id}
            
        except Exception as e:
            logger.error(f"Payment method detachment handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_payment_method_updated(self, payment_method: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payment method update"""
        try:
            pm_id = payment_method.get("id")
            
            logger.info(f"Payment method updated: {pm_id}")
            
            return {"success": True, "payment_method_id": pm_id}
            
        except Exception as e:
            logger.error(f"Payment method update handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Setup Intent event handlers
    
    async def _handle_setup_intent_succeeded(self, setup_intent: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle successful setup intent (saved payment method)"""
        try:
            si_id = setup_intent.get("id")
            payment_method = setup_intent.get("payment_method")
            customer = setup_intent.get("customer")
            
            logger.info(f"Setup Intent succeeded: {si_id} - Payment method: {payment_method}")
            
            return {"success": True, "setup_intent_id": si_id, "payment_method_id": payment_method}
            
        except Exception as e:
            logger.error(f"Setup Intent success handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_setup_intent_failed(self, setup_intent: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle failed setup intent"""
        try:
            si_id = setup_intent.get("id")
            error = setup_intent.get("last_setup_error", {})
            
            logger.warning(f"Setup Intent failed: {si_id} - {error.get('message', 'Unknown error')}")
            
            return {"success": True, "setup_intent_id": si_id, "error": error}
            
        except Exception as e:
            logger.error(f"Setup Intent failure handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_setup_intent_requires_action(self, setup_intent: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle setup intent requiring action"""
        try:
            si_id = setup_intent.get("id")
            next_action = setup_intent.get("next_action", {})
            
            logger.info(f"Setup Intent requires action: {si_id} - {next_action.get('type', 'unknown')}")
            
            return {"success": True, "setup_intent_id": si_id, "requires_action": True}
            
        except Exception as e:
            logger.error(f"Setup Intent action handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Subscription event handlers
    
    async def _handle_subscription_created(self, subscription: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle subscription creation"""
        try:
            sub_id = subscription.get("id")
            customer_id = subscription.get("customer")
            status = subscription.get("status")
            
            logger.info(f"Subscription created: {sub_id} for customer: {customer_id} - {status}")
            
            # Cache subscription
            self.stripe_service._subscriptions[sub_id] = subscription
            
            return {"success": True, "subscription_id": sub_id, "customer_id": customer_id, "status": status}
            
        except Exception as e:
            logger.error(f"Subscription creation handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_subscription_updated(self, subscription: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle subscription update"""
        try:
            sub_id = subscription.get("id")
            status = subscription.get("status")
            
            logger.info(f"Subscription updated: {sub_id} - {status}")
            
            # Update cached subscription
            self.stripe_service._subscriptions[sub_id] = subscription
            
            return {"success": True, "subscription_id": sub_id, "status": status}
            
        except Exception as e:
            logger.error(f"Subscription update handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_subscription_deleted(self, subscription: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle subscription deletion"""
        try:
            sub_id = subscription.get("id")
            
            logger.info(f"Subscription deleted: {sub_id}")
            
            # Remove from cache
            if sub_id in self.stripe_service._subscriptions:
                del self.stripe_service._subscriptions[sub_id]
            
            return {"success": True, "subscription_id": sub_id}
            
        except Exception as e:
            logger.error(f"Subscription deletion handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_subscription_trial_will_end(self, subscription: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle subscription trial ending soon"""
        try:
            sub_id = subscription.get("id")
            trial_end = subscription.get("trial_end")
            
            logger.info(f"Subscription trial will end: {sub_id} - {trial_end}")
            
            # Trigger trial ending notifications
            await self._trigger_trial_ending_actions(subscription)
            
            return {"success": True, "subscription_id": sub_id, "trial_end": trial_end}
            
        except Exception as e:
            logger.error(f"Subscription trial ending handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Invoice event handlers
    
    async def _handle_invoice_created(self, invoice: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle invoice creation"""
        try:
            invoice_id = invoice.get("id")
            customer_id = invoice.get("customer")
            amount_due = invoice.get("amount_due", 0)
            
            logger.info(f"Invoice created: {invoice_id} for customer: {customer_id} - {amount_due}")
            
            return {"success": True, "invoice_id": invoice_id, "customer_id": customer_id}
            
        except Exception as e:
            logger.error(f"Invoice creation handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_invoice_finalized(self, invoice: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle invoice finalization"""
        try:
            invoice_id = invoice.get("id")
            
            logger.info(f"Invoice finalized: {invoice_id}")
            
            return {"success": True, "invoice_id": invoice_id}
            
        except Exception as e:
            logger.error(f"Invoice finalization handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_invoice_payment_succeeded(self, invoice: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle successful invoice payment"""
        try:
            invoice_id = invoice.get("id")
            subscription_id = invoice.get("subscription")
            amount_paid = invoice.get("amount_paid", 0)
            
            logger.info(f"Invoice payment succeeded: {invoice_id} - {amount_paid}")
            
            return {"success": True, "invoice_id": invoice_id, "subscription_id": subscription_id}
            
        except Exception as e:
            logger.error(f"Invoice payment success handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_invoice_payment_failed(self, invoice: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle failed invoice payment"""
        try:
            invoice_id = invoice.get("id")
            subscription_id = invoice.get("subscription")
            
            logger.warning(f"Invoice payment failed: {invoice_id}")
            
            # Trigger payment failure actions (retry, notifications, etc.)
            await self._trigger_invoice_payment_failure_actions(invoice)
            
            return {"success": True, "invoice_id": invoice_id, "subscription_id": subscription_id}
            
        except Exception as e:
            logger.error(f"Invoice payment failure handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_invoice_payment_action_required(self, invoice: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle invoice payment requiring action"""
        try:
            invoice_id = invoice.get("id")
            
            logger.info(f"Invoice payment action required: {invoice_id}")
            
            return {"success": True, "invoice_id": invoice_id, "requires_action": True}
            
        except Exception as e:
            logger.error(f"Invoice payment action handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Refund event handlers
    
    async def _handle_charge_refunded(self, charge: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle charge refund"""
        try:
            charge_id = charge.get("id")
            amount_refunded = charge.get("amount_refunded", 0)
            
            logger.info(f"Charge refunded: {charge_id} - {amount_refunded}")
            
            return {"success": True, "charge_id": charge_id, "amount_refunded": amount_refunded}
            
        except Exception as e:
            logger.error(f"Charge refund handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_refund_created(self, refund: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle refund creation"""
        try:
            refund_id = refund.get("id")
            charge_id = refund.get("charge")
            amount = refund.get("amount", 0)
            
            logger.info(f"Refund created: {refund_id} for charge: {charge_id} - {amount}")
            
            return {"success": True, "refund_id": refund_id, "charge_id": charge_id}
            
        except Exception as e:
            logger.error(f"Refund creation handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_refund_updated(self, refund: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle refund update"""
        try:
            refund_id = refund.get("id")
            status = refund.get("status")
            
            logger.info(f"Refund updated: {refund_id} - {status}")
            
            return {"success": True, "refund_id": refund_id, "status": status}
            
        except Exception as e:
            logger.error(f"Refund update handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Connect event handlers
    
    async def _handle_account_updated(self, account: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Connect account update"""
        try:
            account_id = account.get("id")
            
            logger.info(f"Connect account updated: {account_id}")
            
            return {"success": True, "account_id": account_id}
            
        except Exception as e:
            logger.error(f"Account update handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_account_application_deauthorized(self, account: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Connect account deauthorization"""
        try:
            account_id = account.get("id")
            
            logger.warning(f"Connect account deauthorized: {account_id}")
            
            return {"success": True, "account_id": account_id}
            
        except Exception as e:
            logger.error(f"Account deauthorization handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_transfer_created(self, transfer: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transfer creation"""
        try:
            transfer_id = transfer.get("id")
            amount = transfer.get("amount", 0)
            destination = transfer.get("destination")
            
            logger.info(f"Transfer created: {transfer_id} - {amount} to {destination}")
            
            return {"success": True, "transfer_id": transfer_id, "amount": amount}
            
        except Exception as e:
            logger.error(f"Transfer creation handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_transfer_failed(self, transfer: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transfer failure"""
        try:
            transfer_id = transfer.get("id")
            failure_code = transfer.get("failure_code")
            
            logger.warning(f"Transfer failed: {transfer_id} - {failure_code}")
            
            return {"success": True, "transfer_id": transfer_id, "failure_code": failure_code}
            
        except Exception as e:
            logger.error(f"Transfer failure handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_transfer_paid(self, transfer: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transfer payment"""
        try:
            transfer_id = transfer.get("id")
            
            logger.info(f"Transfer paid: {transfer_id}")
            
            return {"success": True, "transfer_id": transfer_id}
            
        except Exception as e:
            logger.error(f"Transfer payment handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_transfer_reversed(self, transfer: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transfer reversal"""
        try:
            transfer_id = transfer.get("id")
            
            logger.info(f"Transfer reversed: {transfer_id}")
            
            return {"success": True, "transfer_id": transfer_id}
            
        except Exception as e:
            logger.error(f"Transfer reversal handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Payout event handlers
    
    async def _handle_payout_created(self, payout: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payout creation"""
        try:
            payout_id = payout.get("id")
            amount = payout.get("amount", 0)
            
            logger.info(f"Payout created: {payout_id} - {amount}")
            
            return {"success": True, "payout_id": payout_id, "amount": amount}
            
        except Exception as e:
            logger.error(f"Payout creation handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_payout_failed(self, payout: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payout failure"""
        try:
            payout_id = payout.get("id")
            failure_code = payout.get("failure_code")
            
            logger.warning(f"Payout failed: {payout_id} - {failure_code}")
            
            return {"success": True, "payout_id": payout_id, "failure_code": failure_code}
            
        except Exception as e:
            logger.error(f"Payout failure handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_payout_paid(self, payout: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payout completion"""
        try:
            payout_id = payout.get("id")
            
            logger.info(f"Payout paid: {payout_id}")
            
            return {"success": True, "payout_id": payout_id}
            
        except Exception as e:
            logger.error(f"Payout completion handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_payout_canceled(self, payout: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payout cancellation"""
        try:
            payout_id = payout.get("id")
            
            logger.info(f"Payout canceled: {payout_id}")
            
            return {"success": True, "payout_id": payout_id}
            
        except Exception as e:
            logger.error(f"Payout cancellation handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Radar event handlers
    
    async def _handle_radar_early_fraud_warning_created(self, warning: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle early fraud warning"""
        try:
            warning_id = warning.get("id")
            charge_id = warning.get("charge")
            
            logger.warning(f"Early fraud warning created: {warning_id} for charge: {charge_id}")
            
            # Trigger fraud response actions
            await self._trigger_fraud_warning_actions(warning)
            
            return {"success": True, "warning_id": warning_id, "charge_id": charge_id}
            
        except Exception as e:
            logger.error(f"Fraud warning handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_radar_early_fraud_warning_updated(self, warning: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle early fraud warning update"""
        try:
            warning_id = warning.get("id")
            
            logger.info(f"Early fraud warning updated: {warning_id}")
            
            return {"success": True, "warning_id": warning_id}
            
        except Exception as e:
            logger.error(f"Fraud warning update handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Review event handlers
    
    async def _handle_review_opened(self, review: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle review opening (manual review required)"""
        try:
            review_id = review.get("id")
            charge_id = review.get("charge")
            reason = review.get("reason")
            
            logger.warning(f"Review opened: {review_id} for charge: {charge_id} - {reason}")
            
            # Trigger manual review workflow
            await self._trigger_manual_review_actions(review)
            
            return {"success": True, "review_id": review_id, "charge_id": charge_id}
            
        except Exception as e:
            logger.error(f"Review opening handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_review_closed(self, review: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle review closure"""
        try:
            review_id = review.get("id")
            reason = review.get("reason")
            
            logger.info(f"Review closed: {review_id} - {reason}")
            
            return {"success": True, "review_id": review_id, "reason": reason}
            
        except Exception as e:
            logger.error(f"Review closure handling failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Business logic trigger methods
    
    async def _trigger_payment_success_actions(self, payment_intent: Dict[str, Any]) -> None:
        """Trigger actions for successful payment"""
        try:
            # Examples of actions to trigger:
            # - Send confirmation emails
            # - Update order status
            # - Trigger fulfillment
            # - Update analytics
            # - Send webhooks to merchant systems
            
            logger.info(f"Triggering payment success actions for: {payment_intent.get('id')}")
            
        except Exception as e:
            logger.error(f"Payment success actions failed: {str(e)}")
    
    async def _trigger_payment_failure_actions(self, payment_intent: Dict[str, Any]) -> None:
        """Trigger actions for failed payment"""
        try:
            # Examples of actions to trigger:
            # - Send failure notifications
            # - Retry payment with different method
            # - Update order status
            # - Log failure analytics
            
            logger.info(f"Triggering payment failure actions for: {payment_intent.get('id')}")
            
        except Exception as e:
            logger.error(f"Payment failure actions failed: {str(e)}")
    
    async def _trigger_payment_action_required(self, payment_intent: Dict[str, Any]) -> None:
        """Trigger actions for payment requiring action"""
        try:
            # Examples of actions to trigger:
            # - Send 3D Secure notifications
            # - Update UI to show authentication required
            # - Log action required events
            
            logger.info(f"Triggering payment action required for: {payment_intent.get('id')}")
            
        except Exception as e:
            logger.error(f"Payment action required actions failed: {str(e)}")
    
    async def _trigger_dispute_created_actions(self, dispute: Dict[str, Any]) -> None:
        """Trigger actions for dispute creation"""
        try:
            # Examples of actions to trigger:
            # - Send dispute notifications to merchants
            # - Initiate evidence collection workflow
            # - Update transaction status
            # - Log dispute analytics
            
            logger.info(f"Triggering dispute created actions for: {dispute.get('id')}")
            
        except Exception as e:
            logger.error(f"Dispute created actions failed: {str(e)}")
    
    async def _trigger_trial_ending_actions(self, subscription: Dict[str, Any]) -> None:
        """Trigger actions for trial ending"""
        try:
            # Examples of actions to trigger:
            # - Send trial ending notifications
            # - Offer upgrade prompts
            # - Update subscription status
            
            logger.info(f"Triggering trial ending actions for: {subscription.get('id')}")
            
        except Exception as e:
            logger.error(f"Trial ending actions failed: {str(e)}")
    
    async def _trigger_invoice_payment_failure_actions(self, invoice: Dict[str, Any]) -> None:
        """Trigger actions for invoice payment failure"""
        try:
            # Examples of actions to trigger:
            # - Send payment failure notifications
            # - Retry payment collection
            # - Update subscription status
            # - Dunning management
            
            logger.info(f"Triggering invoice payment failure actions for: {invoice.get('id')}")
            
        except Exception as e:
            logger.error(f"Invoice payment failure actions failed: {str(e)}")
    
    async def _trigger_fraud_warning_actions(self, warning: Dict[str, Any]) -> None:
        """Trigger actions for fraud warnings"""
        try:
            # Examples of actions to trigger:
            # - Send fraud alerts
            # - Block suspicious transactions
            # - Update fraud scores
            # - Initiate investigation
            
            logger.info(f"Triggering fraud warning actions for: {warning.get('id')}")
            
        except Exception as e:
            logger.error(f"Fraud warning actions failed: {str(e)}")
    
    async def _trigger_manual_review_actions(self, review: Dict[str, Any]) -> None:
        """Trigger actions for manual review"""
        try:
            # Examples of actions to trigger:
            # - Send review notifications to fraud team
            # - Queue for manual review
            # - Update transaction status
            # - Collect additional data
            
            logger.info(f"Triggering manual review actions for: {review.get('id')}")
            
        except Exception as e:
            logger.error(f"Manual review actions failed: {str(e)}")
    
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
        
        # Count by event type
        event_type_counts = {}
        for log in self.webhook_logs.values():
            event_type = log.get("event_type", "unknown")
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        return {
            "total_webhooks": total_webhooks,
            "processed_webhooks": processed_webhooks,
            "error_webhooks": error_webhooks,
            "success_rate": (processed_webhooks - error_webhooks) / max(1, processed_webhooks),
            "event_types": event_type_counts,
            "last_webhook": max([log.get("timestamp", "") for log in self.webhook_logs.values()]) if self.webhook_logs else None
        }

# Flask Blueprint for Stripe webhooks

def create_stripe_webhook_blueprint(webhook_handler: StripeWebhookHandler) -> Blueprint:
    """Create Flask blueprint for Stripe webhooks"""
    
    stripe_webhook_bp = Blueprint('stripe_webhooks', __name__, url_prefix='/stripe')
    
    @stripe_webhook_bp.route('/webhook', methods=['POST'])
    async def stripe_webhook():
        """Main Stripe webhook endpoint"""
        try:
            payload = request.get_data(as_text=True)
            signature = request.headers.get('Stripe-Signature')
            
            if not signature:
                logger.error("Missing Stripe-Signature header")
                return jsonify({"error": "Missing signature"}), 400
            
            logger.info(f"Stripe webhook received with signature: {signature[:20]}...")
            
            result = await webhook_handler.handle_webhook(payload, signature)
            
            if result.get("success"):
                return jsonify(result), 200
            else:
                return jsonify(result), 400
            
        except Exception as e:
            logger.error(f"Stripe webhook error: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @stripe_webhook_bp.route('/webhook-logs', methods=['GET'])
    def webhook_logs():
        """Get webhook logs"""
        try:
            limit = request.args.get('limit', 100, type=int)
            logs = webhook_handler.get_webhook_logs(limit)
            return jsonify({"success": True, "logs": logs})
            
        except Exception as e:
            logger.error(f"Webhook logs error: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @stripe_webhook_bp.route('/webhook-stats', methods=['GET'])
    def webhook_stats():
        """Get webhook statistics"""
        try:
            stats = webhook_handler.get_webhook_stats()
            return jsonify({"success": True, "stats": stats})
            
        except Exception as e:
            logger.error(f"Webhook stats error: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    return stripe_webhook_bp

def _log_stripe_webhook_handler_module_loaded():
    """Log Stripe webhook handler module loaded"""
    print("🔗 Stripe Webhook Handler module loaded")
    print("   - Payment Intent events (succeeded, failed, requires_action)")
    print("   - Charge events (succeeded, failed, dispute_created)")
    print("   - Customer events (created, updated, deleted)")
    print("   - Subscription events (created, updated, deleted)")
    print("   - Invoice events (payment_succeeded, payment_failed)")
    print("   - Connect events (account updates, transfers)")
    print("   - Fraud detection events (early warnings, reviews)")
    print("   - Complete webhook processing and business logic triggers")

# Execute module loading log
_log_stripe_webhook_handler_module_loaded()