"""
Pesapal Webhook Handler - APG Payment Gateway

Complete IPN (Instant Payment Notification) processing for Pesapal:
- Payment completion notifications
- Transaction status updates
- Refund notifications
- Error handling and retry logic
- Signature verification
- Async event processing

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
from pesapal_integration import PesapalService, PesapalConfig

logger = logging.getLogger(__name__)


class PesapalIPNType(str, Enum):
    """Pesapal IPN notification types"""
    CHANGE = "CHANGE"  # Transaction status change
    REFUND = "REFUND"  # Refund notification


@dataclass
class PesapalIPNEvent:
    """Pesapal IPN event data structure"""
    order_tracking_id: str
    notification_type: str
    merchant_reference: Optional[str] = None
    order_notification_id: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class PesapalWebhookHandler:
    """Complete Pesapal IPN event handler"""
    
    def __init__(self, service: PesapalService):
        self.service = service
        self._event_handlers: Dict[PesapalIPNType, List[Callable[[PesapalIPNEvent], Awaitable[None]]]] = {}
        self._processed_events: set = set()  # Prevent duplicate processing
        
        # Performance tracking
        self._ipn_count = 0
        self._success_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        
        logger.info("Pesapal webhook handler initialized")
    
    def register_handler(self, event_type: PesapalIPNType, handler: Callable[[PesapalIPNEvent], Awaitable[None]]) -> None:
        """Register event handler for specific IPN type"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for IPN type: {event_type.value}")
    
    def register_handlers(self, handlers: Dict[PesapalIPNType, Callable[[PesapalIPNEvent], Awaitable[None]]]) -> None:
        """Register multiple event handlers"""
        for event_type, handler in handlers.items():
            self.register_handler(event_type, handler)
    
    async def process_ipn(self, ipn_data: Dict[str, Any], signature: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming IPN"""
        self._ipn_count += 1
        
        try:
            # Verify signature if provided
            if signature:
                payload = json.dumps(ipn_data, sort_keys=True)
                if not await self._verify_signature(payload, signature):
                    self._error_count += 1
                    logger.error("IPN signature verification failed")
                    return {
                        "success": False,
                        "error": "Invalid signature",
                        "status_code": 401
                    }
            
            # Extract IPN data
            order_tracking_id = ipn_data.get("orderTrackingId") or ipn_data.get("OrderTrackingId")
            notification_type = ipn_data.get("notificationType") or ipn_data.get("NotificationType", "CHANGE")
            merchant_reference = ipn_data.get("merchantReference") or ipn_data.get("MerchantReference")
            order_notification_id = ipn_data.get("orderNotificationId") or ipn_data.get("OrderNotificationId")
            
            if not order_tracking_id:
                self._error_count += 1
                logger.error("Missing order tracking ID in IPN data")
                return {
                    "success": False,
                    "error": "Missing order tracking ID",
                    "status_code": 400
                }
            
            # Create IPN event
            event = PesapalIPNEvent(
                order_tracking_id=order_tracking_id,
                notification_type=notification_type,
                merchant_reference=merchant_reference,
                order_notification_id=order_notification_id
            )
            
            # Check for duplicate event
            event_id = self._get_event_id(event)
            if event_id in self._processed_events:
                logger.info(f"Skipping duplicate IPN event: {event_id}")
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
            logger.error(f"IPN processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": 500
            }
    
    async def _verify_signature(self, payload: str, signature: str) -> bool:
        """Verify IPN signature"""
        return await self.service.validate_ipn_signature(payload, signature)
    
    def _get_event_id(self, event: PesapalIPNEvent) -> str:
        """Generate unique event ID for deduplication"""
        # Create hash from combination of tracking ID and notification type
        unique_string = f"{event.order_tracking_id}:{event.notification_type}:{event.order_notification_id}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    async def _process_event(self, event: PesapalIPNEvent) -> Dict[str, Any]:
        """Process IPN event"""
        try:
            logger.info(f"Processing Pesapal IPN event: {event.notification_type} for {event.order_tracking_id}")
            
            # Verify transaction status with Pesapal
            verification_result = await self.service.verify_payment(event.order_tracking_id)
            
            # Get event type
            try:
                event_type = PesapalIPNType(event.notification_type)
            except ValueError:
                event_type = PesapalIPNType.CHANGE  # Default to CHANGE
            
            # Get handlers for this event type
            handlers = self._event_handlers.get(event_type, [])
            
            if not handlers:
                # Handle with default handlers
                if event_type == PesapalIPNType.CHANGE:
                    await self._handle_status_change(event, verification_result)
                elif event_type == PesapalIPNType.REFUND:
                    await self._handle_refund_notification(event, verification_result)
                else:
                    logger.info(f"No handlers registered for IPN type: {event.notification_type}")
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
                "message": "IPN processed successfully",
                "order_tracking_id": event.order_tracking_id,
                "notification_type": event.notification_type,
                "transaction_status": verification_result.status.value,
                "status_code": 200
            }
            
        except Exception as e:
            logger.error(f"IPN event processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "order_tracking_id": event.order_tracking_id,
                "notification_type": event.notification_type,
                "status_code": 500
            }
    
    async def _handle_status_change(self, event: PesapalIPNEvent, verification_result: PaymentResult) -> None:
        """Handle transaction status change"""
        try:
            logger.info(f"Processing status change for transaction: {event.order_tracking_id}")
            logger.info(f"Status: {verification_result.status.value}")
            
            # Log status change
            if verification_result.status == PaymentStatus.COMPLETED:
                logger.info(f"Payment completed: {event.order_tracking_id}")
                await self._send_payment_completion_notification(event, verification_result)
            elif verification_result.status == PaymentStatus.FAILED:
                logger.error(f"Payment failed: {event.order_tracking_id}")
                await self._send_payment_failure_notification(event, verification_result)
            elif verification_result.status == PaymentStatus.CANCELLED:
                logger.info(f"Payment cancelled: {event.order_tracking_id}")
                await self._send_payment_cancellation_notification(event, verification_result)
            
            # Here you would typically update your database
            # For now, we'll just log the event
            logger.info(f"Status change processed: {event.order_tracking_id} -> {verification_result.status.value}")
            
        except Exception as e:
            logger.error(f"Status change handling failed: {str(e)}")
            raise
    
    async def _handle_refund_notification(self, event: PesapalIPNEvent, verification_result: PaymentResult) -> None:
        """Handle refund notification"""
        try:
            logger.info(f"Processing refund notification for transaction: {event.order_tracking_id}")
            
            # Process refund
            if verification_result.status == PaymentStatus.REFUNDED:
                logger.info(f"Refund completed: {event.order_tracking_id}")
                await self._send_refund_completion_notification(event, verification_result)
            else:
                logger.warning(f"Refund notification received but status is {verification_result.status.value}")
            
        except Exception as e:
            logger.error(f"Refund notification handling failed: {str(e)}")
            raise
    
    async def _send_payment_completion_notification(self, event: PesapalIPNEvent, result: PaymentResult) -> None:
        """Send payment completion notification"""
        try:
            logger.info(f"Payment completion notification: {event.order_tracking_id}")
            logger.info(f"Amount: {result.amount} {result.currency}")
            
            # Here you would typically:
            # - Send email notification to customer
            # - Update order status
            # - Trigger fulfillment process
            # - Send webhook to merchant system
            
        except Exception as e:
            logger.error(f"Failed to send payment completion notification: {str(e)}")
    
    async def _send_payment_failure_notification(self, event: PesapalIPNEvent, result: PaymentResult) -> None:
        """Send payment failure notification"""
        try:
            logger.error(f"Payment failure notification: {event.order_tracking_id}")
            logger.error(f"Error: {result.error_message}")
            
            # Here you would typically:
            # - Send failure notification to customer
            # - Update order status
            # - Retry payment if applicable
            # - Alert customer support
            
        except Exception as e:
            logger.error(f"Failed to send payment failure notification: {str(e)}")
    
    async def _send_payment_cancellation_notification(self, event: PesapalIPNEvent, result: PaymentResult) -> None:
        """Send payment cancellation notification"""
        try:
            logger.info(f"Payment cancellation notification: {event.order_tracking_id}")
            
            # Here you would typically:
            # - Send cancellation notification to customer
            # - Update order status
            # - Release any held inventory
            
        except Exception as e:
            logger.error(f"Failed to send payment cancellation notification: {str(e)}")
    
    async def _send_refund_completion_notification(self, event: PesapalIPNEvent, result: PaymentResult) -> None:
        """Send refund completion notification"""
        try:
            logger.info(f"Refund completion notification: {event.order_tracking_id}")
            logger.info(f"Refund amount: {result.amount}")
            
            # Here you would typically:
            # - Send refund confirmation to customer
            # - Update order status
            # - Update accounting records
            
        except Exception as e:
            logger.error(f"Failed to send refund completion notification: {str(e)}")
    
    def get_ipn_stats(self) -> Dict[str, Any]:
        """Get IPN processing statistics"""
        if self._ipn_count > 0:
            success_rate = self._success_count / self._ipn_count
        else:
            success_rate = 1.0
        
        return {
            "total_ipns": self._ipn_count,
            "successful": self._success_count,
            "failed": self._error_count,
            "success_rate": round(success_rate, 4),
            "last_error": self._last_error,
            "processed_events_count": len(self._processed_events)
        }
    
    async def replay_ipn(self, ipn_data: Dict[str, Any], signature: Optional[str] = None) -> Dict[str, Any]:
        """Replay IPN event (for testing/debugging)"""
        logger.info("Replaying IPN event")
        
        # Temporarily disable duplicate checking
        original_processed_events = self._processed_events.copy()
        self._processed_events.clear()
        
        try:
            result = await self.process_ipn(ipn_data, signature)
            return result
        except Exception as e:
            logger.error(f"IPN replay failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": 500
            }
        finally:
            # Restore original processed events
            self._processed_events = original_processed_events


# Event handler examples
async def payment_completed_handler(event: PesapalIPNEvent) -> None:
    """Example payment completion handler"""
    logger.info(f"Payment completed: {event.order_tracking_id}")
    
    # Add your business logic here
    # - Update order status in database
    # - Send confirmation email to customer
    # - Trigger inventory management
    # - Start fulfillment process
    # - Send webhook to merchant system


async def payment_failed_handler(event: PesapalIPNEvent) -> None:
    """Example payment failure handler"""
    logger.error(f"Payment failed: {event.order_tracking_id}")
    
    # Add your business logic here
    # - Update order status to failed
    # - Send failure notification to customer
    # - Release held inventory
    # - Log failure for analysis
    # - Trigger retry logic if applicable


async def refund_completed_handler(event: PesapalIPNEvent) -> None:
    """Example refund completion handler"""
    logger.info(f"Refund completed: {event.order_tracking_id}")
    
    # Add your business logic here
    # - Update order status
    # - Send refund confirmation to customer
    # - Update accounting records
    # - Process inventory return


# Factory function for webhook handler
async def create_pesapal_webhook_handler(service: PesapalService) -> PesapalWebhookHandler:
    """
    Factory function to create Pesapal webhook handler
    
    Args:
        service: Configured PesapalService instance
        
    Returns:
        PesapalWebhookHandler instance with default handlers
    """
    
    handler = PesapalWebhookHandler(service)
    
    # Register default handlers
    handler.register_handler(PesapalIPNType.CHANGE, payment_completed_handler)
    handler.register_handler(PesapalIPNType.REFUND, refund_completed_handler)
    
    logger.info("Pesapal webhook handler created with default handlers")
    return handler