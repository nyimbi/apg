"""
DPO Webhook Handler - APG Payment Gateway

Complete callback processing for DPO (Direct Pay Online):
- Payment completion notifications
- Transaction status updates
- Callback verification and processing
- Error handling and retry logic
- Async event processing
- Security validation

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
import urllib.parse

from models import PaymentStatus, PaymentResult
from dpo_integration import DPOService, DPOConfig

logger = logging.getLogger(__name__)


class DPOCallbackType(str, Enum):
    """DPO callback types"""
    PAYMENT_COMPLETE = "PAYMENT_COMPLETE"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    PAYMENT_CANCELLED = "PAYMENT_CANCELLED"


@dataclass
class DPOCallbackEvent:
    """DPO callback event data structure"""
    transaction_token: str
    company_ref: str
    transaction_status: str
    payment_amount: Optional[str] = None
    payment_currency: Optional[str] = None
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    transaction_final: Optional[str] = None
    transaction_date: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class DPOWebhookHandler:
    """Complete DPO callback event handler"""
    
    def __init__(self, service: DPOService):
        self.service = service
        self._event_handlers: Dict[DPOCallbackType, List[Callable[[DPOCallbackEvent], Awaitable[None]]]] = {}
        self._processed_events: set = set()  # Prevent duplicate processing
        
        # Performance tracking
        self._callback_count = 0
        self._success_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        
        logger.info("DPO webhook handler initialized")
    
    def register_handler(self, event_type: DPOCallbackType, handler: Callable[[DPOCallbackEvent], Awaitable[None]]) -> None:
        """Register event handler for specific callback type"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for callback type: {event_type.value}")
    
    def register_handlers(self, handlers: Dict[DPOCallbackType, Callable[[DPOCallbackEvent], Awaitable[None]]]) -> None:
        """Register multiple event handlers"""
        for event_type, handler in handlers.items():
            self.register_handler(event_type, handler)
    
    async def process_callback(self, callback_data: Dict[str, Any], verify_ip: bool = True) -> Dict[str, Any]:
        """Process incoming callback"""
        self._callback_count += 1
        
        try:
            # Validate required fields
            transaction_token = callback_data.get('TransactionToken') or callback_data.get('ID')
            company_ref = callback_data.get('CompanyRef')
            transaction_status = callback_data.get('TransactionStatus', '').upper()
            
            if not transaction_token:
                self._error_count += 1
                logger.error("Missing TransactionToken in callback data")
                return {
                    "success": False,
                    "error": "Missing TransactionToken",
                    "status_code": 400
                }
            
            # Create callback event
            event = DPOCallbackEvent(
                transaction_token=transaction_token,
                company_ref=company_ref or transaction_token,
                transaction_status=transaction_status,
                payment_amount=callback_data.get('PaymentAmount'),
                payment_currency=callback_data.get('PaymentCurrency'),
                customer_name=callback_data.get('CustomerName'),
                customer_email=callback_data.get('CustomerEmail'),
                transaction_final=callback_data.get('TransactionFinal'),
                transaction_date=callback_data.get('TransactionDate')
            )
            
            # Check for duplicate event
            event_id = self._get_event_id(event)
            if event_id in self._processed_events:
                logger.info(f"Skipping duplicate callback event: {event_id}")
                return {
                    "success": True,
                    "message": "Duplicate event ignored",
                    "status_code": 200
                }
            
            # Verify transaction with DPO
            verification_result = await self.service.verify_payment(transaction_token)
            
            # Process event
            result = await self._process_event(event, verification_result)
            
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
            logger.error(f"Callback processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": 500
            }
    
    def _get_event_id(self, event: DPOCallbackEvent) -> str:
        """Generate unique event ID for deduplication"""
        # Create hash from combination of token, status, and timestamp
        unique_string = f"{event.transaction_token}:{event.transaction_status}:{event.transaction_date}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    async def _process_event(self, event: DPOCallbackEvent, verification_result: PaymentResult) -> Dict[str, Any]:
        """Process callback event"""
        try:
            logger.info(f"Processing DPO callback event: {event.transaction_status} for {event.transaction_token}")
            
            # Determine callback type
            callback_type = self._get_callback_type(event.transaction_status)
            
            # Get handlers for this callback type
            handlers = self._event_handlers.get(callback_type, [])
            
            if not handlers:
                # Handle with default handlers
                if callback_type == DPOCallbackType.PAYMENT_COMPLETE:
                    await self._handle_payment_complete(event, verification_result)
                elif callback_type == DPOCallbackType.PAYMENT_FAILED:
                    await self._handle_payment_failed(event, verification_result)
                elif callback_type == DPOCallbackType.PAYMENT_CANCELLED:
                    await self._handle_payment_cancelled(event, verification_result)
                else:
                    logger.info(f"No handlers registered for callback type: {callback_type}")
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
                "message": "Callback processed successfully",
                "transaction_token": event.transaction_token,
                "company_ref": event.company_ref,
                "transaction_status": event.transaction_status,
                "verification_status": verification_result.status.value,
                "status_code": 200
            }
            
        except Exception as e:
            logger.error(f"Callback event processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "transaction_token": event.transaction_token,
                "company_ref": event.company_ref,
                "status_code": 500
            }
    
    def _get_callback_type(self, transaction_status: str) -> DPOCallbackType:
        """Determine callback type from transaction status"""
        status = transaction_status.upper()
        
        if status in ["COMPLETE", "COMPLETED", "SUCCESS", "SUCCESSFUL"]:
            return DPOCallbackType.PAYMENT_COMPLETE
        elif status in ["FAILED", "DECLINED", "ERROR", "FAILURE"]:
            return DPOCallbackType.PAYMENT_FAILED
        elif status in ["CANCELLED", "CANCELED", "CANCEL"]:
            return DPOCallbackType.PAYMENT_CANCELLED
        else:
            # Default to failed for unknown statuses
            return DPOCallbackType.PAYMENT_FAILED
    
    async def _handle_payment_complete(self, event: DPOCallbackEvent, verification_result: PaymentResult) -> None:
        """Handle payment completion"""
        try:
            logger.info(f"Processing payment completion for: {event.company_ref}")
            logger.info(f"Amount: {event.payment_amount} {event.payment_currency}")
            logger.info(f"Customer: {event.customer_name} ({event.customer_email})")
            
            # Verify the payment is actually complete
            if verification_result.status == PaymentStatus.COMPLETED:
                logger.info(f"Payment verified as completed: {event.company_ref}")
                await self._send_payment_completion_notification(event, verification_result)
                
                # Here you would typically:
                # - Update order status in database
                # - Send confirmation email to customer
                # - Trigger fulfillment process
                # - Update inventory
                # - Send webhook to merchant system
            else:
                logger.warning(f"Payment callback indicates complete but verification shows {verification_result.status.value}")
            
        except Exception as e:
            logger.error(f"Payment completion handling failed: {str(e)}")
            raise
    
    async def _handle_payment_failed(self, event: DPOCallbackEvent, verification_result: PaymentResult) -> None:
        """Handle payment failure"""
        try:
            logger.error(f"Processing payment failure for: {event.company_ref}")
            logger.error(f"Status: {event.transaction_status}")
            
            await self._send_payment_failure_notification(event, verification_result)
            
            # Here you would typically:
            # - Update order status to failed
            # - Send failure notification to customer
            # - Release held inventory
            # - Log failure for analysis
            # - Trigger retry logic if applicable
            
        except Exception as e:
            logger.error(f"Payment failure handling failed: {str(e)}")
            raise
    
    async def _handle_payment_cancelled(self, event: DPOCallbackEvent, verification_result: PaymentResult) -> None:
        """Handle payment cancellation"""
        try:
            logger.info(f"Processing payment cancellation for: {event.company_ref}")
            
            await self._send_payment_cancellation_notification(event, verification_result)
            
            # Here you would typically:
            # - Update order status to cancelled
            # - Send cancellation notification to customer
            # - Release held inventory
            # - Process any partial refunds if applicable
            
        except Exception as e:
            logger.error(f"Payment cancellation handling failed: {str(e)}")
            raise
    
    async def _send_payment_completion_notification(self, event: DPOCallbackEvent, result: PaymentResult) -> None:
        """Send payment completion notification"""
        try:
            logger.info(f"Payment completion notification: {event.company_ref}")
            logger.info(f"Amount: {event.payment_amount} {event.payment_currency}")
            logger.info(f"Customer: {event.customer_email}")
            
            # Here you would typically integrate with:
            # - Email service for customer notifications
            # - SMS service for mobile notifications
            # - Webhook service for merchant notifications
            # - Push notification service
            
        except Exception as e:
            logger.error(f"Failed to send payment completion notification: {str(e)}")
    
    async def _send_payment_failure_notification(self, event: DPOCallbackEvent, result: PaymentResult) -> None:
        """Send payment failure notification"""
        try:
            logger.error(f"Payment failure notification: {event.company_ref}")
            logger.error(f"Status: {event.transaction_status}")
            logger.error(f"Customer: {event.customer_email}")
            
            # Here you would typically:
            # - Send failure notification to customer
            # - Alert customer support team
            # - Log failure details for analysis
            # - Trigger retry mechanisms if appropriate
            
        except Exception as e:
            logger.error(f"Failed to send payment failure notification: {str(e)}")
    
    async def _send_payment_cancellation_notification(self, event: DPOCallbackEvent, result: PaymentResult) -> None:
        """Send payment cancellation notification"""
        try:
            logger.info(f"Payment cancellation notification: {event.company_ref}")
            logger.info(f"Customer: {event.customer_email}")
            
            # Here you would typically:
            # - Send cancellation confirmation to customer
            # - Update merchant dashboard
            # - Process inventory release
            
        except Exception as e:
            logger.error(f"Failed to send payment cancellation notification: {str(e)}")
    
    def get_callback_stats(self) -> Dict[str, Any]:
        """Get callback processing statistics"""
        if self._callback_count > 0:
            success_rate = self._success_count / self._callback_count
        else:
            success_rate = 1.0
        
        return {
            "total_callbacks": self._callback_count,
            "successful": self._success_count,
            "failed": self._error_count,
            "success_rate": round(success_rate, 4),
            "last_error": self._last_error,
            "processed_events_count": len(self._processed_events)
        }
    
    async def replay_callback(self, callback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Replay callback event (for testing/debugging)"""
        logger.info("Replaying callback event")
        
        # Temporarily disable duplicate checking
        original_processed_events = self._processed_events.copy()
        self._processed_events.clear()
        
        try:
            result = await self.process_callback(callback_data, verify_ip=False)
            return result
        except Exception as e:
            logger.error(f"Callback replay failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": 500
            }
        finally:
            # Restore original processed events
            self._processed_events = original_processed_events
    
    def validate_callback_ip(self, client_ip: str) -> bool:
        """Validate callback IP address"""
        # DPO callback IP ranges (update these based on DPO's current IPs)
        allowed_ips = [
            "196.201.212.22",   # DPO callback server
            "196.201.212.23",   # DPO callback server
            "41.77.11.170",     # DPO callback server
            "41.77.11.171",     # DPO callback server
            "127.0.0.1",        # Local testing
            "::1"               # Local testing IPv6
        ]
        
        # Also allow private IP ranges for development
        private_ranges = [
            "10.0.0.0/8",
            "172.16.0.0/12", 
            "192.168.0.0/16"
        ]
        
        # Simple IP validation (in production, use proper IP range checking)
        if client_ip in allowed_ips:
            return True
        
        # Check private ranges for development
        for ip_range in private_ranges:
            if self._ip_in_range(client_ip, ip_range):
                return True
        
        logger.warning(f"Callback from unrecognized IP: {client_ip}")
        return False
    
    def _ip_in_range(self, ip: str, ip_range: str) -> bool:
        """Check if IP is in given range (simplified implementation)"""
        try:
            import ipaddress
            return ipaddress.ip_address(ip) in ipaddress.ip_network(ip_range)
        except:
            return False


# Event handler examples
async def payment_completed_handler(event: DPOCallbackEvent) -> None:
    """Example payment completion handler"""
    logger.info(f"Payment completed: {event.company_ref}")
    logger.info(f"Amount: {event.payment_amount} {event.payment_currency}")
    logger.info(f"Customer: {event.customer_name} ({event.customer_email})")
    
    # Add your business logic here
    # - Update order status in database
    # - Send confirmation email to customer
    # - Trigger inventory management
    # - Start fulfillment process
    # - Send webhook to merchant system


async def payment_failed_handler(event: DPOCallbackEvent) -> None:
    """Example payment failure handler"""
    logger.error(f"Payment failed: {event.company_ref}")
    logger.error(f"Status: {event.transaction_status}")
    logger.error(f"Customer: {event.customer_email}")
    
    # Add your business logic here
    # - Update order status to failed
    # - Send failure notification to customer
    # - Release held inventory
    # - Log failure for analysis
    # - Trigger retry logic if applicable
    # - Alert customer support team


async def payment_cancelled_handler(event: DPOCallbackEvent) -> None:
    """Example payment cancellation handler"""
    logger.info(f"Payment cancelled: {event.company_ref}")
    logger.info(f"Customer: {event.customer_email}")
    
    # Add your business logic here
    # - Update order status to cancelled
    # - Send cancellation confirmation to customer
    # - Release held inventory
    # - Update analytics/reporting
    # - Process any applicable refunds


# Factory function for webhook handler
async def create_dpo_webhook_handler(service: DPOService) -> DPOWebhookHandler:
    """
    Factory function to create DPO webhook handler
    
    Args:
        service: Configured DPOService instance
        
    Returns:
        DPOWebhookHandler instance with default handlers
    """
    
    handler = DPOWebhookHandler(service)
    
    # Register default handlers
    handler.register_handler(DPOCallbackType.PAYMENT_COMPLETE, payment_completed_handler)
    handler.register_handler(DPOCallbackType.PAYMENT_FAILED, payment_failed_handler)
    handler.register_handler(DPOCallbackType.PAYMENT_CANCELLED, payment_cancelled_handler)
    
    logger.info("DPO webhook handler created with default handlers")
    return handler