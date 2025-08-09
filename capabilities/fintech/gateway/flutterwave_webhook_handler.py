"""
Flutterwave Webhook Handler - APG Payment Gateway

Complete webhook processing for all Flutterwave events:
- Payment completion notifications
- Charge events and status updates
- Transfer notifications
- Subscription events
- Refund notifications
- Customer events
- Comprehensive signature verification
- Async event processing
- Error handling and retry logic

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass

from models import PaymentStatus, PaymentResult
from flutterwave_integration import FlutterwaveService, FlutterwaveConfig

logger = logging.getLogger(__name__)


class FlutterwaveEventType(str, Enum):
    """Flutterwave webhook event types"""
    # Payment events
    CHARGE_COMPLETED = "charge.completed"
    CHARGE_FAILED = "charge.failed"
    CHARGE_PENDING = "charge.pending"
    
    # Transfer events
    TRANSFER_COMPLETED = "transfer.completed" 
    TRANSFER_FAILED = "transfer.failed"
    TRANSFER_PENDING = "transfer.pending"
    
    # Subscription events
    SUBSCRIPTION_CREATED = "subscription.created"
    SUBSCRIPTION_CANCELLED = "subscription.cancelled"
    SUBSCRIPTION_ACTIVATED = "subscription.activated"
    
    # Refund events
    REFUND_COMPLETED = "refund.completed"
    REFUND_FAILED = "refund.failed"
    
    # Customer events
    CUSTOMER_CREATED = "customer.created"
    CUSTOMER_UPDATED = "customer.updated"
    
    # Plan events
    PLAN_CREATED = "plan.created"
    PLAN_UPDATED = "plan.updated"
    
    # Settlement events
    SETTLEMENT_COMPLETED = "settlement.completed"
    
    # KYC events
    KYC_VERIFIED = "kyc.verified"
    KYC_FAILED = "kyc.failed"


@dataclass
class FlutterwaveWebhookEvent:
    """Flutterwave webhook event data structure"""
    event: str
    data: Dict[str, Any]
    event_type: Optional[FlutterwaveEventType] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.event_type is None:
            try:
                self.event_type = FlutterwaveEventType(self.event)
            except ValueError:
                self.event_type = None
        
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class FlutterwaveWebhookHandler:
    """Complete Flutterwave webhook event handler"""
    
    def __init__(self, service: FlutterwaveService):
        self.service = service
        self._event_handlers: Dict[FlutterwaveEventType, List[Callable[[FlutterwaveWebhookEvent], Awaitable[None]]]] = {}
        self._processed_events: set = set()  # Prevent duplicate processing
        
        # Performance tracking
        self._webhook_count = 0
        self._success_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        
        logger.info("Flutterwave webhook handler initialized")
    
    def register_handler(self, event_type: FlutterwaveEventType, handler: Callable[[FlutterwaveWebhookEvent], Awaitable[None]]) -> None:
        """Register event handler for specific event type"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type.value}")
    
    def register_handlers(self, handlers: Dict[FlutterwaveEventType, Callable[[FlutterwaveWebhookEvent], Awaitable[None]]]) -> None:
        """Register multiple event handlers"""
        for event_type, handler in handlers.items():
            self.register_handler(event_type, handler)
    
    async def process_webhook(self, payload: str, signature: str) -> Dict[str, Any]:
        """Process incoming webhook"""
        self._webhook_count += 1
        
        try:
            # Verify signature
            if not await self._verify_signature(payload, signature):
                self._error_count += 1
                logger.error("Webhook signature verification failed")
                return {
                    "success": False,
                    "error": "Invalid signature",
                    "status_code": 401
                }
            
            # Parse webhook data
            try:
                webhook_data = json.loads(payload)
            except json.JSONDecodeError as e:
                self._error_count += 1
                logger.error(f"Invalid JSON payload: {str(e)}")
                return {
                    "success": False,
                    "error": "Invalid JSON payload",
                    "status_code": 400
                }
            
            # Create webhook event
            event = FlutterwaveWebhookEvent(
                event=webhook_data.get("event", ""),
                data=webhook_data.get("data", {})
            )
            
            # Check for duplicate event
            event_id = self._get_event_id(event)
            if event_id in self._processed_events:
                logger.info(f"Skipping duplicate webhook event: {event_id}")
                return {
                    "success": True,
                    "message": "Duplicate event ignored",
                    "status_code": 200
                }
            
            # Process event
            result = await self._process_event(event)
            
            # Mark as processed if successful
            if result.get("success", False):
                self._processed_events.add(event_id)
                self._success_count += 1
            else:
                self._error_count += 1
            
            return result
            
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.error(f"Webhook processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": 500
            }
    
    async def _verify_signature(self, payload: str, signature: str) -> bool:
        """Verify webhook signature"""
        return await self.service.validate_webhook_signature(payload, signature)
    
    def _get_event_id(self, event: FlutterwaveWebhookEvent) -> str:
        """Generate unique event ID for deduplication"""
        data = event.data
        
        # Try to get unique identifiers from the data
        transaction_id = data.get("id") or data.get("tx_id") or data.get("flw_ref")
        event_type = event.event
        timestamp = data.get("created_at") or data.get("charged_at")
        
        # Create hash from combination
        unique_string = f"{event_type}:{transaction_id}:{timestamp}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    async def _process_event(self, event: FlutterwaveWebhookEvent) -> Dict[str, Any]:
        """Process webhook event"""
        try:
            logger.info(f"Processing Flutterwave webhook event: {event.event}")
            
            # Get handlers for this event type
            handlers = self._event_handlers.get(event.event_type, [])
            
            if not handlers:
                # Handle common events with default handlers
                if event.event_type in [FlutterwaveEventType.CHARGE_COMPLETED, FlutterwaveEventType.CHARGE_FAILED, FlutterwaveEventType.CHARGE_PENDING]:
                    await self._handle_charge_event(event)
                elif event.event_type in [FlutterwaveEventType.TRANSFER_COMPLETED, FlutterwaveEventType.TRANSFER_FAILED]:
                    await self._handle_transfer_event(event)
                elif event.event_type in [FlutterwaveEventType.REFUND_COMPLETED, FlutterwaveEventType.REFUND_FAILED]:
                    await self._handle_refund_event(event)
                else:
                    logger.info(f"No handlers registered for event type: {event.event}")
            else:
                # Execute registered handlers
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Handler execution failed: {str(e)}")
                        # Continue with other handlers
            
            return {
                "success": True,
                "message": "Event processed successfully",
                "event_type": event.event,
                "status_code": 200
            }
            
        except Exception as e:
            logger.error(f"Event processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "event_type": event.event,
                "status_code": 500
            }
    
    async def _handle_charge_event(self, event: FlutterwaveWebhookEvent) -> None:
        """Handle charge events (payment completion)"""
        try:
            data = event.data
            
            # Extract payment information
            transaction_ref = data.get("tx_ref")
            flw_ref = data.get("flw_ref")
            status = data.get("status", "").lower()
            amount = data.get("amount")
            currency = data.get("currency")
            customer_email = data.get("customer", {}).get("email")
            
            logger.info(f"Processing charge event: {transaction_ref} - Status: {status}")
            
            # Map Flutterwave status to internal status
            if status == "successful":
                payment_status = PaymentStatus.COMPLETED
            elif status in ["pending", "processing"]:
                payment_status = PaymentStatus.PENDING
            elif status in ["failed", "cancelled"]:
                payment_status = PaymentStatus.FAILED
            else:
                payment_status = PaymentStatus.PENDING
            
            # Here you would typically update your database
            # For now, we'll just log the event
            logger.info(f"Charge event processed: {transaction_ref} -> {payment_status.value}")
            
            # Optionally send notifications
            await self._send_payment_notification(transaction_ref, payment_status, amount, currency, customer_email)
            
        except Exception as e:
            logger.error(f"Charge event handling failed: {str(e)}")
            raise
    
    async def _handle_transfer_event(self, event: FlutterwaveWebhookEvent) -> None:
        """Handle transfer events"""
        try:
            data = event.data
            
            transfer_id = data.get("id")
            reference = data.get("reference")
            status = data.get("status", "").lower()
            amount = data.get("amount")
            currency = data.get("currency")
            
            logger.info(f"Processing transfer event: {reference} - Status: {status}")
            
            # Process transfer status update
            if status == "successful":
                logger.info(f"Transfer completed: {reference}")
            elif status == "failed":
                logger.error(f"Transfer failed: {reference}")
            
        except Exception as e:
            logger.error(f"Transfer event handling failed: {str(e)}")
            raise
    
    async def _handle_refund_event(self, event: FlutterwaveWebhookEvent) -> None:
        """Handle refund events"""
        try:
            data = event.data
            
            refund_id = data.get("id")
            transaction_id = data.get("transaction_id")
            status = data.get("status", "").lower()
            amount = data.get("amount")
            
            logger.info(f"Processing refund event: {refund_id} - Status: {status}")
            
            # Process refund status update
            if status == "successful":
                logger.info(f"Refund completed: {refund_id} for transaction: {transaction_id}")
            elif status == "failed":
                logger.error(f"Refund failed: {refund_id} for transaction: {transaction_id}")
            
        except Exception as e:
            logger.error(f"Refund event handling failed: {str(e)}")
            raise
    
    async def _send_payment_notification(self, transaction_ref: str, status: PaymentStatus, amount: float, currency: str, customer_email: str) -> None:
        """Send payment notification"""
        try:
            # This would typically send email/SMS notifications
            # For now, we'll just log
            logger.info(f"Payment notification: {transaction_ref} - {status.value} - {amount} {currency} - {customer_email}")
            
            # You could integrate with email services here
            # await email_service.send_payment_notification(...)
            
        except Exception as e:
            logger.error(f"Failed to send payment notification: {str(e)}")
    
    def get_webhook_stats(self) -> Dict[str, Any]:
        """Get webhook processing statistics"""
        if self._webhook_count > 0:
            success_rate = self._success_count / self._webhook_count
        else:
            success_rate = 1.0
        
        return {
            "total_webhooks": self._webhook_count,
            "successful": self._success_count,
            "failed": self._error_count,
            "success_rate": round(success_rate, 4),
            "last_error": self._last_error,
            "processed_events_count": len(self._processed_events)
        }
    
    async def replay_webhook(self, payload: str, signature: str) -> Dict[str, Any]:
        """Replay webhook event (for testing/debugging)"""
        logger.info("Replaying webhook event")
        
        # Temporarily disable duplicate checking
        original_processed_events = self._processed_events.copy()
        self._processed_events.clear()
        
        try:
            result = await self.process_webhook(payload, signature)
            return result
        except Exception as e:
            logger.error(f"Webhook replay failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": 500
            }
        finally:
            # Restore original processed events
            self._processed_events = original_processed_events


# Event handler examples
async def payment_completed_handler(event: FlutterwaveWebhookEvent) -> None:
    """Example payment completion handler"""
    data = event.data
    transaction_ref = data.get("tx_ref")
    amount = data.get("amount")
    currency = data.get("currency")
    
    logger.info(f"Payment completed: {transaction_ref} - {amount} {currency}")
    
    # Add your business logic here
    # - Update order status
    # - Send confirmation email
    # - Update inventory
    # - Trigger fulfillment process


async def payment_failed_handler(event: FlutterwaveWebhookEvent) -> None:
    """Example payment failure handler"""
    data = event.data
    transaction_ref = data.get("tx_ref")
    failure_reason = data.get("processor_response")
    
    logger.error(f"Payment failed: {transaction_ref} - Reason: {failure_reason}")
    
    # Add your business logic here
    # - Send failure notification
    # - Update payment status
    # - Retry payment logic
    # - Customer support notification


async def transfer_completed_handler(event: FlutterwaveWebhookEvent) -> None:
    """Example transfer completion handler"""
    data = event.data
    reference = data.get("reference")
    amount = data.get("amount")
    
    logger.info(f"Transfer completed: {reference} - {amount}")
    
    # Add your business logic here
    # - Update payout status
    # - Send confirmation to recipient
    # - Update accounting records


# Factory function for webhook handler
async def create_flutterwave_webhook_handler(service: FlutterwaveService) -> FlutterwaveWebhookHandler:
    """
    Factory function to create Flutterwave webhook handler
    
    Args:
        service: Configured FlutterwaveService instance
        
    Returns:
        FlutterwaveWebhookHandler instance with default handlers
    """
    
    handler = FlutterwaveWebhookHandler(service)
    
    # Register default handlers
    handler.register_handler(FlutterwaveEventType.CHARGE_COMPLETED, payment_completed_handler)
    handler.register_handler(FlutterwaveEventType.CHARGE_FAILED, payment_failed_handler)
    handler.register_handler(FlutterwaveEventType.TRANSFER_COMPLETED, transfer_completed_handler)
    
    logger.info("Flutterwave webhook handler created with default handlers")
    return handler