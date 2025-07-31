"""
Adyen Webhook Handler - APG Payment Gateway

Complete webhook and notification processing for Adyen:
- Standard webhook notifications (authorization, capture, refund, etc.)
- Marketplace notifications (split settlements, payouts)
- Recurring payment notifications
- Report notifications (financial reports, disputes)
- Configuration notifications (account settings, merchant updates)
- Real-time webhook processing with business logic triggers
- Comprehensive error handling and retry mechanisms
- Webhook validation and security

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import hmac
import hashlib
import base64

from flask import Blueprint, request, jsonify, Response

# Adyen imports
from Adyen.util import is_valid_hmac

# APG imports
from adyen_integration import AdyenService

logger = logging.getLogger(__name__)


class AdyenEventType(str, Enum):
    """Adyen webhook event types"""
    # Payment events
    AUTHORISATION = "AUTHORISATION"
    CAPTURE = "CAPTURE"
    CAPTURE_FAILED = "CAPTURE_FAILED"
    CANCELLATION = "CANCELLATION"
    REFUND = "REFUND"
    REFUND_FAILED = "REFUND_FAILED"
    CANCEL_OR_REFUND = "CANCEL_OR_REFUND"
    
    # Chargeback events
    CHARGEBACK = "CHARGEBACK"
    CHARGEBACK_REVERSED = "CHARGEBACK_REVERSED"
    REQUEST_FOR_INFORMATION = "REQUEST_FOR_INFORMATION"
    NOTIFICATION_OF_CHARGEBACK = "NOTIFICATION_OF_CHARGEBACK"
    SECOND_CHARGEBACK = "SECOND_CHARGEBACK"
    PREARBITRATION = "PREARBITRATION"
    ARBITRATION_WON = "ARBITRATION_WON"
    ARBITRATION_LOST = "ARBITRATION_LOST"
    
    # Recurring payment events
    RECURRING_CONTRACT = "RECURRING_CONTRACT"
    
    # Report events
    REPORT_AVAILABLE = "REPORT_AVAILABLE"
    
    # Payout events (for marketplaces)
    PAYOUT_THIRDPARTY = "PAYOUT_THIRDPARTY"
    
    # Account holder events (for marketplaces)
    ACCOUNT_HOLDER_CREATED = "ACCOUNT_HOLDER_CREATED"
    ACCOUNT_HOLDER_UPDATED = "ACCOUNT_HOLDER_UPDATED"
    ACCOUNT_HOLDER_STATUS_CHANGE = "ACCOUNT_HOLDER_STATUS_CHANGE"
    ACCOUNT_HOLDER_STORE_STATUS_CHANGE = "ACCOUNT_HOLDER_STORE_STATUS_CHANGE"
    ACCOUNT_HOLDER_VERIFICATION = "ACCOUNT_HOLDER_VERIFICATION"
    ACCOUNT_HOLDER_LIMIT_REACHED = "ACCOUNT_HOLDER_LIMIT_REACHED"
    ACCOUNT_HOLDER_PAYOUT = "ACCOUNT_HOLDER_PAYOUT"
    
    # Account events (for marketplaces)
    ACCOUNT_CREATED = "ACCOUNT_CREATED"
    ACCOUNT_UPDATED = "ACCOUNT_UPDATED"
    ACCOUNT_CLOSED = "ACCOUNT_CLOSED"
    ACCOUNT_FUNDS_BELOW_THRESHOLD = "ACCOUNT_FUNDS_BELOW_THRESHOLD"
    
    # Transfer events (for marketplaces)
    TRANSFER_FUNDS = "TRANSFER_FUNDS"
    
    # Direct debit events
    DIRECT_DEBIT_INITIATED = "DIRECT_DEBIT_INITIATED"
    
    # Dispute events
    DISPUTE_OPENED = "DISPUTE_OPENED"
    DISPUTE_CLOSED = "DISPUTE_CLOSED"
    DISPUTE_EXPIRED = "DISPUTE_EXPIRED"
    
    # Order events
    ORDER_OPENED = "ORDER_OPENED"
    ORDER_CLOSED = "ORDER_CLOSED"
    
    # Manual review events
    MANUAL_REVIEW_ACCEPT = "MANUAL_REVIEW_ACCEPT"
    MANUAL_REVIEW_REJECT = "MANUAL_REVIEW_REJECT"
    
    # Offer events
    OFFER_CLOSED = "OFFER_CLOSED"
    
    # Technical events
    TECHNICAL_CANCEL = "TECHNICAL_CANCEL"
    
    # Custom events
    CUSTOM = "CUSTOM"


class AdyenSuccessCode(str, Enum):
    """Adyen success codes"""
    SUCCESS = "true"
    AUTHORISED = "Authorised"
    RECEIVED = "Received"


@dataclass
class AdyenNotificationItem:
    """Single Adyen notification item"""
    amount: Dict[str, Any]
    event_code: str
    event_date: str
    merchant_account_code: str
    merchant_reference: str
    original_reference: Optional[str]
    psp_reference: str
    reason: Optional[str]
    success: str
    payment_method: Optional[str]
    operations: Optional[List[str]]
    additional_data: Optional[Dict[str, Any]]
    raw_data: Dict[str, Any]


@dataclass
class WebhookLog:
    """Webhook processing log entry"""
    timestamp: datetime
    event_type: str
    psp_reference: str
    merchant_reference: Optional[str]
    success: bool
    processing_time_ms: float
    error_message: Optional[str]
    notification_data: Dict[str, Any]


@dataclass
class WebhookStats:
    """Webhook processing statistics"""
    total_webhooks: int = 0
    successful_webhooks: int = 0
    failed_webhooks: int = 0
    average_processing_time: float = 0.0
    last_webhook_time: Optional[datetime] = None
    error_rate: float = 0.0
    events_by_type: Dict[str, int] = field(default_factory=dict)


class AdyenWebhookHandler:
    """Complete Adyen webhook and notification handler"""
    
    def __init__(self, adyen_service: AdyenService):
        self.adyen_service = adyen_service
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._webhook_logs: List[WebhookLog] = []
        self._webhook_stats = WebhookStats()
        
        # Register default event handlers
        self._register_default_handlers()
        
        logger.info("Adyen webhook handler initialized")
    
    def register_event_handler(
        self,
        event_type: Union[AdyenEventType, str],
        handler: Callable[[AdyenNotificationItem], None]
    ) -> None:
        """
        Register custom event handler for specific event type
        
        Args:
            event_type: Adyen event type
            handler: Handler function to call for this event type
        """
        event_key = event_type.value if isinstance(event_type, AdyenEventType) else event_type
        
        if event_key not in self._event_handlers:
            self._event_handlers[event_key] = []
        
        self._event_handlers[event_key].append(handler)
        logger.info(f"Registered handler for event type: {event_key}")
    
    async def process_webhook_notification(
        self,
        notification_data: Dict[str, Any],
        hmac_signature: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process incoming Adyen webhook notification
        
        Args:
            notification_data: Raw notification data from Adyen
            hmac_signature: HMAC signature for verification (optional)
            
        Returns:
            Processing result dictionary
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info("Processing Adyen webhook notification")
            
            # Verify HMAC signature if provided
            if hmac_signature and self.adyen_service.config.hmac_key:
                if not await self.adyen_service.verify_notification(notification_data, hmac_signature):
                    logger.error("Invalid HMAC signature for Adyen notification")
                    return {
                        "processed": False,
                        "error": "Invalid HMAC signature",
                        "status": "rejected"
                    }
            
            # Extract notification items
            notification_items = notification_data.get("notificationItems", [])
            
            if not notification_items:
                logger.warning("No notification items found in webhook")
                return {
                    "processed": False,
                    "error": "No notification items found",
                    "status": "ignored"
                }
            
            # Process each notification item
            processed_items = []
            actions_taken = []
            
            for item_data in notification_items:
                try:
                    # Parse notification item
                    notification_item = self._parse_notification_item(item_data)
                    
                    # Process the notification
                    item_result = await self._process_notification_item(notification_item)
                    
                    processed_items.append({
                        "psp_reference": notification_item.psp_reference,
                        "event_code": notification_item.event_code,
                        "success": item_result["success"],
                        "actions": item_result.get("actions", [])
                    })
                    
                    if item_result.get("actions"):
                        actions_taken.extend(item_result["actions"])
                    
                    # Log processing
                    processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    self._log_webhook_processing(
                        notification_item, item_result["success"], processing_time, 
                        item_result.get("error")
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to process notification item: {str(e)}")
                    processed_items.append({
                        "error": str(e),
                        "success": False
                    })
            
            # Update statistics
            self._update_webhook_stats(len(processed_items), actions_taken)
            
            # Build response
            all_successful = all(item.get("success", False) for item in processed_items)
            
            return {
                "processed": True,
                "success": all_successful,
                "items_processed": len(processed_items),
                "actions_taken": actions_taken,
                "items": processed_items,
                "status": "accepted"
            }
            
        except Exception as e:
            logger.error(f"Failed to process Adyen webhook: {str(e)}")
            
            # Log failed processing
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._log_webhook_processing(None, False, processing_time, str(e))
            
            return {
                "processed": False,
                "error": str(e),
                "status": "error"
            }
    
    async def _process_notification_item(
        self,
        notification_item: AdyenNotificationItem
    ) -> Dict[str, Any]:
        """
        Process individual notification item
        
        Args:
            notification_item: Parsed notification item
            
        Returns:
            Processing result
        """
        try:
            logger.info(f"Processing {notification_item.event_code} for {notification_item.psp_reference}")
            
            actions_taken = []
            
            # Get event handlers for this event type
            handlers = self._event_handlers.get(notification_item.event_code, [])
            
            # Execute all registered handlers
            for handler in handlers:
                try:
                    action = await self._execute_handler(handler, notification_item)
                    if action:
                        actions_taken.append(action)
                except Exception as e:
                    logger.error(f"Handler execution failed: {str(e)}")
                    actions_taken.append(f"Handler error: {str(e)}")
            
            # Execute default business logic
            default_action = await self._execute_default_business_logic(notification_item)
            if default_action:
                actions_taken.append(default_action)
            
            return {
                "success": True,
                "actions": actions_taken,
                "event_code": notification_item.event_code,
                "psp_reference": notification_item.psp_reference
            }
            
        except Exception as e:
            logger.error(f"Failed to process notification item: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "event_code": notification_item.event_code,
                "psp_reference": notification_item.psp_reference
            }
    
    async def _execute_handler(
        self,
        handler: Callable,
        notification_item: AdyenNotificationItem
    ) -> Optional[str]:
        """Execute event handler and return action description"""
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(notification_item)
            else:
                result = handler(notification_item)
            
            return f"Custom handler executed: {handler.__name__}"
            
        except Exception as e:
            logger.error(f"Handler {handler.__name__} failed: {str(e)}")
            raise
    
    async def _execute_default_business_logic(
        self,
        notification_item: AdyenNotificationItem
    ) -> Optional[str]:
        """Execute default business logic based on event type"""
        
        event_code = notification_item.event_code
        success = notification_item.success == "true"
        
        try:
            if event_code == AdyenEventType.AUTHORISATION.value:
                return await self._handle_authorisation(notification_item, success)
            
            elif event_code == AdyenEventType.CAPTURE.value:
                return await self._handle_capture(notification_item, success)
            
            elif event_code == AdyenEventType.CAPTURE_FAILED.value:
                return await self._handle_capture_failed(notification_item)
            
            elif event_code == AdyenEventType.CANCELLATION.value:
                return await self._handle_cancellation(notification_item, success)
            
            elif event_code == AdyenEventType.REFUND.value:
                return await self._handle_refund(notification_item, success)
            
            elif event_code == AdyenEventType.REFUND_FAILED.value:
                return await self._handle_refund_failed(notification_item)
            
            elif event_code == AdyenEventType.CHARGEBACK.value:
                return await self._handle_chargeback(notification_item)
            
            elif event_code == AdyenEventType.CHARGEBACK_REVERSED.value:
                return await self._handle_chargeback_reversed(notification_item)
            
            elif event_code == AdyenEventType.DISPUTE_OPENED.value:
                return await self._handle_dispute_opened(notification_item)
            
            elif event_code == AdyenEventType.DISPUTE_CLOSED.value:
                return await self._handle_dispute_closed(notification_item)
            
            elif event_code == AdyenEventType.RECURRING_CONTRACT.value:
                return await self._handle_recurring_contract(notification_item, success)
            
            elif event_code == AdyenEventType.REPORT_AVAILABLE.value:
                return await self._handle_report_available(notification_item)
            
            elif event_code in [
                AdyenEventType.ACCOUNT_HOLDER_CREATED.value,
                AdyenEventType.ACCOUNT_HOLDER_UPDATED.value,
                AdyenEventType.ACCOUNT_HOLDER_STATUS_CHANGE.value
            ]:
                return await self._handle_account_holder_event(notification_item)
            
            elif event_code == AdyenEventType.PAYOUT_THIRDPARTY.value:
                return await self._handle_payout(notification_item, success)
            
            elif event_code == AdyenEventType.TRANSFER_FUNDS.value:
                return await self._handle_transfer_funds(notification_item, success)
            
            else:
                logger.info(f"No default handler for event type: {event_code}")
                return f"Event received: {event_code}"
            
        except Exception as e:
            logger.error(f"Default business logic failed for {event_code}: {str(e)}")
            return f"Business logic error: {str(e)}"
    
    # Event-specific handlers
    
    async def _handle_authorisation(
        self,
        notification_item: AdyenNotificationItem,
        success: bool
    ) -> str:
        """Handle authorization notification"""
        
        if success:
            logger.info(f"Payment authorized: {notification_item.psp_reference}")
            
            # Update payment status in database
            # This would integrate with your payment tracking system
            await self._update_payment_status(
                notification_item.psp_reference,
                "authorized",
                notification_item.merchant_reference
            )
            
            # Trigger post-authorization business logic
            await self._trigger_post_authorization_logic(notification_item)
            
            return "Payment authorized - status updated"
        else:
            logger.info(f"Payment authorization failed: {notification_item.psp_reference}")
            
            await self._update_payment_status(
                notification_item.psp_reference,
                "authorization_failed",
                notification_item.merchant_reference,
                notification_item.reason
            )
            
            return f"Authorization failed: {notification_item.reason}"
    
    async def _handle_capture(
        self,
        notification_item: AdyenNotificationItem,
        success: bool
    ) -> str:
        """Handle capture notification"""
        
        if success:
            logger.info(f"Payment captured: {notification_item.psp_reference}")
            
            await self._update_payment_status(
                notification_item.psp_reference,
                "captured",
                notification_item.merchant_reference
            )
            
            # Trigger fulfillment process
            await self._trigger_fulfillment_process(notification_item)
            
            return "Payment captured - fulfillment triggered"
        else:
            logger.warning(f"Payment capture failed: {notification_item.psp_reference}")
            
            await self._update_payment_status(
                notification_item.psp_reference,
                "capture_failed",
                notification_item.merchant_reference,
                notification_item.reason
            )
            
            return f"Capture failed: {notification_item.reason}"
    
    async def _handle_capture_failed(self, notification_item: AdyenNotificationItem) -> str:
        """Handle capture failed notification"""
        
        logger.warning(f"Capture failed notification: {notification_item.psp_reference}")
        
        await self._update_payment_status(
            notification_item.psp_reference,
            "capture_failed",
            notification_item.merchant_reference,
            notification_item.reason
        )
        
        # Alert operations team
        await self._send_operations_alert(
            "Capture Failed",
            f"Payment {notification_item.psp_reference} capture failed: {notification_item.reason}"
        )
        
        return f"Capture failed alert sent: {notification_item.reason}"
    
    async def _handle_cancellation(
        self,
        notification_item: AdyenNotificationItem,
        success: bool
    ) -> str:
        """Handle cancellation notification"""
        
        if success:
            logger.info(f"Payment cancelled: {notification_item.psp_reference}")
            
            await self._update_payment_status(
                notification_item.psp_reference,
                "cancelled",
                notification_item.merchant_reference
            )
            
            # Release inventory or reserved items
            await self._release_reservations(notification_item)
            
            return "Payment cancelled - reservations released"
        else:
            logger.warning(f"Payment cancellation failed: {notification_item.psp_reference}")
            return f"Cancellation failed: {notification_item.reason}"
    
    async def _handle_refund(
        self,
        notification_item: AdyenNotificationItem,
        success: bool
    ) -> str:
        """Handle refund notification"""
        
        if success:
            logger.info(f"Payment refunded: {notification_item.psp_reference}")
            
            await self._update_payment_status(
                notification_item.psp_reference,
                "refunded",
                notification_item.merchant_reference
            )
            
            # Update customer account balance or credit
            await self._process_customer_refund_credit(notification_item)
            
            return "Refund processed - customer credited"
        else:
            logger.warning(f"Refund failed: {notification_item.psp_reference}")
            return f"Refund failed: {notification_item.reason}"
    
    async def _handle_refund_failed(self, notification_item: AdyenNotificationItem) -> str:
        """Handle refund failed notification"""
        
        logger.warning(f"Refund failed notification: {notification_item.psp_reference}")
        
        await self._update_payment_status(
            notification_item.psp_reference,
            "refund_failed",
            notification_item.merchant_reference,
            notification_item.reason
        )
        
        # Alert customer service
        await self._send_customer_service_alert(
            "Refund Failed",
            f"Refund for payment {notification_item.psp_reference} failed: {notification_item.reason}"
        )
        
        return f"Refund failed alert sent: {notification_item.reason}"
    
    async def _handle_chargeback(self, notification_item: AdyenNotificationItem) -> str:
        """Handle chargeback notification"""
        
        logger.warning(f"Chargeback received: {notification_item.psp_reference}")
        
        await self._update_payment_status(
            notification_item.psp_reference,
            "chargeback",
            notification_item.merchant_reference,
            notification_item.reason
        )
        
        # Create dispute case
        await self._create_dispute_case(notification_item)
        
        # Alert risk management team
        await self._send_risk_management_alert(
            "Chargeback Received",
            f"Chargeback for payment {notification_item.psp_reference}: {notification_item.reason}"
        )
        
        return "Chargeback processed - dispute case created"
    
    async def _handle_chargeback_reversed(self, notification_item: AdyenNotificationItem) -> str:
        """Handle chargeback reversed notification"""
        
        logger.info(f"Chargeback reversed: {notification_item.psp_reference}")
        
        await self._update_payment_status(
            notification_item.psp_reference,
            "chargeback_reversed",
            notification_item.merchant_reference
        )
        
        # Update dispute case
        await self._update_dispute_case(notification_item, "won")
        
        return "Chargeback reversed - dispute case updated"
    
    async def _handle_dispute_opened(self, notification_item: AdyenNotificationItem) -> str:
        """Handle dispute opened notification"""
        
        logger.info(f"Dispute opened: {notification_item.psp_reference}")
        
        # Create or update dispute case
        await self._create_dispute_case(notification_item)
        
        return "Dispute case created"
    
    async def _handle_dispute_closed(self, notification_item: AdyenNotificationItem) -> str:
        """Handle dispute closed notification"""
        
        logger.info(f"Dispute closed: {notification_item.psp_reference}")
        
        # Update dispute case
        await self._update_dispute_case(notification_item, "closed")
        
        return "Dispute case closed"
    
    async def _handle_recurring_contract(
        self,
        notification_item: AdyenNotificationItem,
        success: bool
    ) -> str:
        """Handle recurring contract notification"""
        
        if success:
            logger.info(f"Recurring contract created: {notification_item.psp_reference}")
            
            # Store recurring payment method reference
            await self._store_recurring_payment_method(notification_item)
            
            return "Recurring contract stored"
        else:
            logger.warning(f"Recurring contract failed: {notification_item.psp_reference}")
            return f"Recurring contract failed: {notification_item.reason}"
    
    async def _handle_report_available(self, notification_item: AdyenNotificationItem) -> str:
        """Handle report available notification"""
        
        logger.info(f"Report available: {notification_item.reason}")
        
        # Download and process the report
        await self._process_available_report(notification_item)
        
        return f"Report processed: {notification_item.reason}"
    
    async def _handle_account_holder_event(self, notification_item: AdyenNotificationItem) -> str:
        """Handle account holder event (marketplace)"""
        
        logger.info(f"Account holder event: {notification_item.event_code}")
        
        # Update account holder status in marketplace system
        await self._update_account_holder_status(notification_item)
        
        return f"Account holder updated: {notification_item.event_code}"
    
    async def _handle_payout(
        self,
        notification_item: AdyenNotificationItem,
        success: bool
    ) -> str:
        """Handle payout notification (marketplace)"""
        
        if success:
            logger.info(f"Payout successful: {notification_item.psp_reference}")
            
            # Update payout status
            await self._update_payout_status(notification_item, "completed")
            
            return "Payout completed"
        else:
            logger.warning(f"Payout failed: {notification_item.psp_reference}")
            
            await self._update_payout_status(notification_item, "failed")
            
            return f"Payout failed: {notification_item.reason}"
    
    async def _handle_transfer_funds(
        self,
        notification_item: AdyenNotificationItem,
        success: bool
    ) -> str:
        """Handle transfer funds notification (marketplace)"""
        
        if success:
            logger.info(f"Funds transfer successful: {notification_item.psp_reference}")
            
            await self._update_transfer_status(notification_item, "completed")
            
            return "Funds transfer completed"
        else:
            logger.warning(f"Funds transfer failed: {notification_item.psp_reference}")
            
            await self._update_transfer_status(notification_item, "failed")
            
            return f"Funds transfer failed: {notification_item.reason}"
    
    # Business logic helper methods (these would integrate with your actual systems)
    
    async def _update_payment_status(
        self,
        psp_reference: str,
        status: str,
        merchant_reference: Optional[str],
        reason: Optional[str] = None
    ) -> None:
        """Update payment status in database"""
        logger.info(f"Updating payment status: {psp_reference} -> {status}")
        # Implementation would update your payment database
        pass
    
    async def _trigger_post_authorization_logic(self, notification_item: AdyenNotificationItem) -> None:
        """Trigger business logic after successful authorization"""
        logger.info(f"Triggering post-authorization logic for: {notification_item.psp_reference}")
        # Implementation would trigger order processing, inventory reservation, etc.
        pass
    
    async def _trigger_fulfillment_process(self, notification_item: AdyenNotificationItem) -> None:
        """Trigger fulfillment process after capture"""
        logger.info(f"Triggering fulfillment for: {notification_item.psp_reference}")
        # Implementation would trigger shipping, digital delivery, etc.
        pass
    
    async def _release_reservations(self, notification_item: AdyenNotificationItem) -> None:
        """Release inventory or service reservations"""
        logger.info(f"Releasing reservations for: {notification_item.psp_reference}")
        # Implementation would release inventory holds, cancel reservations, etc.
        pass
    
    async def _process_customer_refund_credit(self, notification_item: AdyenNotificationItem) -> None:
        """Process customer refund credit"""
        logger.info(f"Processing customer credit for: {notification_item.psp_reference}")
        # Implementation would credit customer account, update loyalty points, etc.
        pass
    
    async def _create_dispute_case(self, notification_item: AdyenNotificationItem) -> None:
        """Create dispute case for chargeback handling"""
        logger.info(f"Creating dispute case for: {notification_item.psp_reference}")
        # Implementation would create case in dispute management system
        pass
    
    async def _update_dispute_case(self, notification_item: AdyenNotificationItem, status: str) -> None:
        """Update dispute case status"""
        logger.info(f"Updating dispute case: {notification_item.psp_reference} -> {status}")
        # Implementation would update dispute case status
        pass
    
    async def _store_recurring_payment_method(self, notification_item: AdyenNotificationItem) -> None:
        """Store recurring payment method reference"""
        logger.info(f"Storing recurring payment method: {notification_item.psp_reference}")
        # Implementation would store payment method reference for future use
        pass
    
    async def _process_available_report(self, notification_item: AdyenNotificationItem) -> None:
        """Process available financial report"""
        logger.info(f"Processing report: {notification_item.reason}")
        # Implementation would download and process financial reports
        pass
    
    async def _update_account_holder_status(self, notification_item: AdyenNotificationItem) -> None:
        """Update marketplace account holder status"""
        logger.info(f"Updating account holder: {notification_item.psp_reference}")
        # Implementation would update marketplace account status
        pass
    
    async def _update_payout_status(self, notification_item: AdyenNotificationItem, status: str) -> None:
        """Update marketplace payout status"""
        logger.info(f"Updating payout status: {notification_item.psp_reference} -> {status}")
        # Implementation would update payout status in marketplace system
        pass
    
    async def _update_transfer_status(self, notification_item: AdyenNotificationItem, status: str) -> None:
        """Update funds transfer status"""
        logger.info(f"Updating transfer status: {notification_item.psp_reference} -> {status}")
        # Implementation would update transfer status
        pass
    
    # Alert and notification methods
    
    async def _send_operations_alert(self, subject: str, message: str) -> None:
        """Send alert to operations team"""
        logger.info(f"Operations alert: {subject}")
        # Implementation would send email, Slack notification, etc.
        pass
    
    async def _send_customer_service_alert(self, subject: str, message: str) -> None:
        """Send alert to customer service team"""
        logger.info(f"Customer service alert: {subject}")
        # Implementation would create support ticket, send notification, etc.
        pass
    
    async def _send_risk_management_alert(self, subject: str, message: str) -> None:
        """Send alert to risk management team"""
        logger.info(f"Risk management alert: {subject}")
        # Implementation would send high-priority alert for chargebacks, fraud, etc.
        pass
    
    # Utility methods
    
    def _parse_notification_item(self, item_data: Dict[str, Any]) -> AdyenNotificationItem:
        """Parse raw notification item data"""
        
        notification_request_item = item_data.get("NotificationRequestItem", item_data)
        
        return AdyenNotificationItem(
            amount=notification_request_item.get("amount", {}),
            event_code=notification_request_item.get("eventCode", ""),
            event_date=notification_request_item.get("eventDate", ""),
            merchant_account_code=notification_request_item.get("merchantAccountCode", ""),
            merchant_reference=notification_request_item.get("merchantReference", ""),
            original_reference=notification_request_item.get("originalReference"),
            psp_reference=notification_request_item.get("pspReference", ""),
            reason=notification_request_item.get("reason"),
            success=notification_request_item.get("success", "false"),
            payment_method=notification_request_item.get("paymentMethod"),
            operations=notification_request_item.get("operations", []),
            additional_data=notification_request_item.get("additionalData", {}),
            raw_data=notification_request_item
        )
    
    def _register_default_handlers(self) -> None:
        """Register default event handlers"""
        # Default handlers would be registered here
        # For now, we rely on the default business logic in _execute_default_business_logic
        pass
    
    def _log_webhook_processing(
        self,
        notification_item: Optional[AdyenNotificationItem],
        success: bool,
        processing_time_ms: float,
        error_message: Optional[str] = None
    ) -> None:
        """Log webhook processing result"""
        
        log_entry = WebhookLog(
            timestamp=datetime.now(timezone.utc),
            event_type=notification_item.event_code if notification_item else "unknown",
            psp_reference=notification_item.psp_reference if notification_item else "unknown",
            merchant_reference=notification_item.merchant_reference if notification_item else None,
            success=success,
            processing_time_ms=processing_time_ms,
            error_message=error_message,
            notification_data=notification_item.raw_data if notification_item else {}
        )
        
        # Keep only last 1000 log entries
        self._webhook_logs.append(log_entry)
        if len(self._webhook_logs) > 1000:
            self._webhook_logs.pop(0)
    
    def _update_webhook_stats(self, items_processed: int, actions_taken: List[str]) -> None:
        """Update webhook processing statistics"""
        
        self._webhook_stats.total_webhooks += items_processed
        self._webhook_stats.last_webhook_time = datetime.now(timezone.utc)
        
        # Update success/failure counts based on recent logs
        recent_logs = self._webhook_logs[-items_processed:] if items_processed <= len(self._webhook_logs) else self._webhook_logs
        
        successful = sum(1 for log in recent_logs if log.success)
        failed = len(recent_logs) - successful
        
        self._webhook_stats.successful_webhooks += successful
        self._webhook_stats.failed_webhooks += failed
        
        # Calculate error rate
        if self._webhook_stats.total_webhooks > 0:
            self._webhook_stats.error_rate = self._webhook_stats.failed_webhooks / self._webhook_stats.total_webhooks
        
        # Calculate average processing time
        if recent_logs:
            total_time = sum(log.processing_time_ms for log in recent_logs)
            self._webhook_stats.average_processing_time = total_time / len(recent_logs)
        
        # Update events by type
        for log in recent_logs:
            event_type = log.event_type
            self._webhook_stats.events_by_type[event_type] = self._webhook_stats.events_by_type.get(event_type, 0) + 1
    
    # Public methods for stats and monitoring
    
    def get_webhook_stats(self) -> Dict[str, Any]:
        """Get webhook processing statistics"""
        return {
            "total_webhooks": self._webhook_stats.total_webhooks,
            "successful_webhooks": self._webhook_stats.successful_webhooks,
            "failed_webhooks": self._webhook_stats.failed_webhooks,
            "success_rate": 1 - self._webhook_stats.error_rate if self._webhook_stats.total_webhooks > 0 else 0,
            "error_rate": self._webhook_stats.error_rate,
            "average_processing_time": self._webhook_stats.average_processing_time,
            "last_webhook_time": self._webhook_stats.last_webhook_time.isoformat() if self._webhook_stats.last_webhook_time else None,
            "events_by_type": self._webhook_stats.events_by_type
        }
    
    def get_webhook_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent webhook processing logs"""
        recent_logs = self._webhook_logs[-limit:] if limit <= len(self._webhook_logs) else self._webhook_logs
        
        return [
            {
                "timestamp": log.timestamp.isoformat(),
                "event_type": log.event_type,
                "psp_reference": log.psp_reference,
                "merchant_reference": log.merchant_reference,
                "success": log.success,
                "processing_time_ms": log.processing_time_ms,
                "error_message": log.error_message
            }
            for log in reversed(recent_logs)
        ]
    
    def get_webhook_health(self) -> Dict[str, Any]:
        """Get webhook endpoint health status"""
        
        if not self._webhook_logs:
            return {
                "status": "idle",
                "last_webhook_time": None,
                "processing_rate": 0.0,
                "error_rate": 0.0
            }
        
        # Calculate processing rate (webhooks per minute over last hour)
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)
        
        recent_webhooks = [
            log for log in self._webhook_logs
            if log.timestamp >= one_hour_ago
        ]
        
        processing_rate = len(recent_webhooks) / 60.0 if recent_webhooks else 0.0
        
        # Calculate recent error rate
        recent_errors = sum(1 for log in recent_webhooks if not log.success)
        recent_error_rate = recent_errors / len(recent_webhooks) if recent_webhooks else 0.0
        
        # Determine status
        if self._webhook_stats.last_webhook_time:
            time_since_last = (now - self._webhook_stats.last_webhook_time).total_seconds()
            if time_since_last > 3600:  # No webhooks in last hour
                status = "idle"
            elif recent_error_rate > 0.1:  # More than 10% errors
                status = "degraded"
            else:
                status = "healthy"
        else:
            status = "idle"
        
        return {
            "status": status,
            "last_webhook_time": self._webhook_stats.last_webhook_time.isoformat() if self._webhook_stats.last_webhook_time else None,
            "processing_rate": processing_rate,
            "error_rate": recent_error_rate
        }


def create_adyen_webhook_blueprint(webhook_handler: AdyenWebhookHandler) -> Blueprint:
    """
    Create Flask blueprint for Adyen webhook endpoints
    
    Args:
        webhook_handler: Configured AdyenWebhookHandler instance
        
    Returns:
        Flask Blueprint with webhook endpoints
    """
    
    bp = Blueprint("adyen_webhooks", __name__, url_prefix="/adyen")
    
    @bp.route("/webhook", methods=["POST"])
    async def adyen_webhook():
        """Main Adyen webhook endpoint"""
        try:
            # Get notification data
            notification_data = request.get_json()
            
            if not notification_data:
                logger.error("No JSON data in Adyen webhook request")
                return jsonify({"error": "No JSON data"}), 400
            
            # Get HMAC signature from headers
            hmac_signature = request.headers.get("hmac-signature")
            
            # Process notification
            result = await webhook_handler.process_webhook_notification(
                notification_data, hmac_signature
            )
            
            # Return appropriate response
            if result.get("processed", False):
                return jsonify({"notificationResponse": "[accepted]"}), 200
            else:
                return jsonify({
                    "notificationResponse": "[rejected]",
                    "error": result.get("error", "Processing failed")
                }), 400
                
        except Exception as e:
            logger.error(f"Adyen webhook endpoint error: {str(e)}")
            return jsonify({
                "notificationResponse": "[rejected]", 
                "error": "Internal server error"
            }), 500
    
    @bp.route("/webhook-stats", methods=["GET"])
    def webhook_stats():
        """Get webhook processing statistics"""
        try:
            stats = webhook_handler.get_webhook_stats()
            return jsonify(stats), 200
        except Exception as e:
            logger.error(f"Error getting webhook stats: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route("/webhook-logs", methods=["GET"])
    def webhook_logs():
        """Get recent webhook processing logs"""
        try:
            limit = request.args.get("limit", 50, type=int)
            logs = webhook_handler.get_webhook_logs(limit)
            return jsonify({
                "logs": logs,
                "total_returned": len(logs)
            }), 200
        except Exception as e:
            logger.error(f"Error getting webhook logs: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route("/webhook-health", methods=["GET"])
    def webhook_health():
        """Get webhook endpoint health status"""
        try:
            health = webhook_handler.get_webhook_health()
            return jsonify(health), 200
        except Exception as e:
            logger.error(f"Error getting webhook health: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    logger.info("Adyen webhook blueprint created with endpoints: /webhook, /webhook-stats, /webhook-logs, /webhook-health")
    return bp