"""
Flutterwave Client Usage Examples - APG Payment Gateway

Comprehensive examples demonstrating how to use the Flutterwave integration:
- Card payments with 3D Secure
- Mobile money payments (M-Pesa, MTN, Airtel)
- Bank transfer payments
- USSD payments
- Payment verification and status checking
- Refund processing
- Webhook handling
- Error handling and best practices

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
import logging
import os
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Import APG payment gateway components
from flutterwave_integration import (
    create_flutterwave_service, 
    FlutterwaveEnvironment,
    FlutterwaveService
)
from flutterwave_webhook_handler import create_flutterwave_webhook_handler
from models import PaymentTransaction, PaymentMethod, PaymentMethodType, PaymentStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlutterwaveClientExample:
    """Example client for Flutterwave integration"""
    
    def __init__(self):
        self.service: Optional[FlutterwaveService] = None
        self.webhook_handler = None
    
    async def initialize(self):
        """Initialize Flutterwave service"""
        try:
            # Create service (uses environment variables for credentials)
            environment = FlutterwaveEnvironment.SANDBOX  # Change to LIVE for production
            self.service = await create_flutterwave_service(environment)
            
            # Create webhook handler
            self.webhook_handler = await create_flutterwave_webhook_handler(self.service)
            
            logger.info("Flutterwave client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Flutterwave client: {str(e)}")
            raise
    
    async def card_payment_example(self):
        """Example: Process card payment"""
        logger.info("=== Card Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"CARD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("1000.00"),  # ₦1,000 
                currency="NGN",
                description="Test card payment",
                customer_email="customer@example.com",
                customer_name="John Doe"
            )
            
            # Create card payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.CARD,
                card_number="4187427415564246",  # Test card number
                expiry_month="12",
                expiry_year="2025",
                security_code="123",
                metadata={
                    'customer_email': 'customer@example.com',
                    'customer_name': 'John Doe',
                    'phone_number': '+2348012345678',
                    'redirect_url': 'https://example.com/payment/callback'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"Card payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.auth_url:
                logger.info(f"  3D Secure URL: {result.auth_url}")
                logger.info("  Customer needs to complete 3D Secure authentication")
            
            if result.error_message:
                logger.error(f"  Error: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Card payment example failed: {str(e)}")
            return None
    
    async def mobile_money_payment_example(self):
        """Example: Process M-Pesa payment"""
        logger.info("=== Mobile Money Payment Example (M-Pesa) ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"MPESA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("500.00"),  # KSh 500
                currency="KES",
                description="Test M-Pesa payment",
                customer_email="customer@example.com",
                customer_name="Jane Doe"
            )
            
            # Create M-Pesa payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.MOBILE_MONEY,
                metadata={
                    'mobile_money_type': 'mpesa',
                    'phone_number': '254708374149',  # Test phone number
                    'customer_email': 'customer@example.com',
                    'customer_name': 'Jane Doe',
                    'network': 'MPESA'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"M-Pesa payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.auth_url:
                logger.info(f"  Payment URL: {result.auth_url}")
                logger.info("  Customer will receive STK push prompt")
            
            if result.error_message:
                logger.error(f"  Error: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"M-Pesa payment example failed: {str(e)}")
            return None
    
    async def mtn_mobile_money_example(self):
        """Example: Process MTN Mobile Money payment"""
        logger.info("=== MTN Mobile Money Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"MTN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("200.00"),  # GH₵200
                currency="GHS",
                description="Test MTN Mobile Money payment",
                customer_email="customer@example.com",
                customer_name="Kwame Asante"
            )
            
            # Create MTN Mobile Money payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.MOBILE_MONEY,
                metadata={
                    'mobile_money_type': 'mobilemoney',
                    'phone_number': '233545454545',  # Test phone number
                    'customer_email': 'customer@example.com',
                    'customer_name': 'Kwame Asante',
                    'network': 'MTN'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"MTN Mobile Money payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.auth_url:
                logger.info(f"  Payment URL: {result.auth_url}")
                logger.info("  Customer will receive mobile money prompt")
            
            return result
            
        except Exception as e:
            logger.error(f"MTN Mobile Money payment example failed: {str(e)}")
            return None
    
    async def bank_transfer_example(self):
        """Example: Process bank transfer payment"""
        logger.info("=== Bank Transfer Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"BANK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("5000.00"),  # ₦5,000
                currency="NGN",
                description="Test bank transfer payment",
                customer_email="business@example.com",
                customer_name="Business Customer"
            )
            
            # Create bank transfer payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.BANK_TRANSFER,
                metadata={
                    'bank_code': '044',  # Access Bank
                    'account_number': '1234567890',
                    'customer_email': 'business@example.com',
                    'customer_name': 'Business Customer'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"Bank transfer payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.auth_url:
                logger.info(f"  Payment URL: {result.auth_url}")
                logger.info("  Customer will be redirected to complete bank transfer")
            
            return result
            
        except Exception as e:
            logger.error(f"Bank transfer payment example failed: {str(e)}")
            return None
    
    async def ussd_payment_example(self):
        """Example: Process USSD payment"""
        logger.info("=== USSD Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"USSD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("2000.00"),  # ₦2,000
                currency="NGN",
                description="Test USSD payment",
                customer_email="customer@example.com",
                customer_name="USSD Customer"
            )
            
            # Create USSD payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.USSD,
                metadata={
                    'bank_code': '058',  # GTBank
                    'customer_email': 'customer@example.com',
                    'customer_name': 'USSD Customer',
                    'phone_number': '+2348012345678'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"USSD payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.ussd_code:
                logger.info(f"  USSD Code: {result.ussd_code}")
                logger.info("  Customer should dial this USSD code to complete payment")
            
            return result
            
        except Exception as e:
            logger.error(f"USSD payment example failed: {str(e)}")
            return None
    
    async def payment_verification_example(self, transaction_id: str):
        """Example: Verify payment status"""
        logger.info(f"=== Payment Verification Example ===")
        
        try:
            # Verify payment
            result = await self.service.verify_payment(transaction_id)
            
            logger.info(f"Payment verification result:")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            logger.info(f"  Amount: {result.amount} {result.currency}")
            logger.info(f"  Success: {result.success}")
            
            if result.error_message:
                logger.error(f"  Error: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Payment verification example failed: {str(e)}")
            return None
    
    async def refund_example(self, transaction_id: str, refund_amount: Optional[Decimal] = None):
        """Example: Process refund"""
        logger.info(f"=== Refund Example ===")
        
        try:
            # Process refund
            result = await self.service.refund_payment(
                transaction_id=transaction_id,
                amount=refund_amount,
                reason="Customer requested refund"
            )
            
            logger.info(f"Refund result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            logger.info(f"  Refund Amount: {result.amount}")
            
            if result.error_message:
                logger.error(f"  Error: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Refund example failed: {str(e)}")
            return None
    
    async def get_supported_payment_methods_example(self):
        """Example: Get supported payment methods"""
        logger.info("=== Supported Payment Methods Example ===")
        
        try:
            # Get all supported methods
            all_methods = await self.service.get_supported_payment_methods()
            
            logger.info("All supported payment methods:")
            for method in all_methods:
                logger.info(f"  - {method.get('name', method.get('type'))}")
                if 'countries' in method:
                    logger.info(f"    Countries: {', '.join(method['countries'])}")
                if 'currencies' in method:
                    logger.info(f"    Currencies: {', '.join(method['currencies'])}")
            
            # Get methods for specific country
            kenya_methods = await self.service.get_supported_payment_methods(country_code="KE")
            
            logger.info("\nSupported payment methods for Kenya:")
            for method in kenya_methods:
                logger.info(f"  - {method.get('name', method.get('type'))}")
            
            return all_methods
            
        except Exception as e:
            logger.error(f"Get payment methods example failed: {str(e)}")
            return None
    
    async def account_balance_example(self):
        """Example: Get account balance"""
        logger.info("=== Account Balance Example ===")
        
        try:
            # Get all balances
            all_balances = await self.service.get_account_balance()
            
            logger.info("Account balances:")
            if 'balances' in all_balances:
                for balance in all_balances['balances']:
                    currency = balance.get('currency')
                    available = balance.get('available_balance')
                    ledger = balance.get('ledger_balance')
                    logger.info(f"  {currency}: Available={available}, Ledger={ledger}")
            else:
                logger.info(f"  Balance info: {all_balances}")
            
            # Get balance for specific currency
            ngn_balance = await self.service.get_account_balance("NGN")
            
            logger.info(f"\nNGN Balance: {ngn_balance}")
            
            return all_balances
            
        except Exception as e:
            logger.error(f"Account balance example failed: {str(e)}")
            return None
    
    async def health_check_example(self):
        """Example: Health check"""
        logger.info("=== Health Check Example ===")
        
        try:
            # Perform health check
            health = await self.service.health_check()
            
            logger.info(f"Health check result:")
            logger.info(f"  Status: {health.status.value}")
            logger.info(f"  Response time: {health.response_time_ms}ms")
            
            if health.details:
                logger.info("  Details:")
                for key, value in health.details.items():
                    logger.info(f"    {key}: {value}")
            
            if health.error_message:
                logger.error(f"  Error: {health.error_message}")
            
            return health
            
        except Exception as e:
            logger.error(f"Health check example failed: {str(e)}")
            return None
    
    async def webhook_processing_example(self):
        """Example: Process webhook"""
        logger.info("=== Webhook Processing Example ===")
        
        try:
            # Example webhook payload (this would come from Flutterwave)
            webhook_payload = json.dumps({
                "event": "charge.completed",
                "data": {
                    "id": 1234567,
                    "tx_ref": "CARD_20250131_123456",
                    "flw_ref": "FLW-MOCK-1234567890",
                    "status": "successful",
                    "amount": 1000,
                    "currency": "NGN",
                    "customer": {
                        "email": "customer@example.com",
                        "name": "John Doe"
                    },
                    "created_at": "2025-01-31T10:30:00Z"
                }
            })
            
            # Mock signature (in real scenario, this comes from Flutterwave)
            signature = "mock_signature"
            
            # Process webhook
            result = await self.webhook_handler.process_webhook(webhook_payload, signature)
            
            logger.info(f"Webhook processing result:")
            logger.info(f"  Success: {result.get('success')}")
            logger.info(f"  Message: {result.get('message')}")
            logger.info(f"  Event type: {result.get('event_type')}")
            
            if not result.get('success'):
                logger.error(f"  Error: {result.get('error')}")
            
            # Get webhook stats
            stats = self.webhook_handler.get_webhook_stats()
            logger.info(f"Webhook stats: {stats}")
            
            return result
            
        except Exception as e:
            logger.error(f"Webhook processing example failed: {str(e)}")
            return None
    
    async def transaction_fees_example(self):
        """Example: Calculate transaction fees"""
        logger.info("=== Transaction Fees Example ===")
        
        try:
            # Calculate fees for different payment methods
            amounts = [Decimal("1000"), Decimal("5000"), Decimal("10000")]
            currencies = ["NGN", "GHS", "KES"]
            payment_methods = ["card", "mobilemoney", "banktransfer"]
            
            for amount in amounts:
                for currency in currencies:
                    for method in payment_methods:
                        fees = await self.service.get_transaction_fees(amount, currency, method)
                        
                        logger.info(f"Fees for {amount} {currency} via {method}:")
                        logger.info(f"  Total fee: {fees.get('total_fee', 'N/A')}")
                        logger.info(f"  Percentage fee: {fees.get('percentage_fee', 'N/A')}")
                        logger.info(f"  Fixed fee: {fees.get('fixed_fee', 'N/A')}")
                        logger.info("")
            
        except Exception as e:
            logger.error(f"Transaction fees example failed: {str(e)}")
    
    async def run_all_examples(self):
        """Run all examples"""
        logger.info("Starting Flutterwave integration examples...")
        
        try:
            # Initialize service
            await self.initialize()
            
            # Run payment examples
            card_result = await self.card_payment_example()
            mobile_result = await self.mobile_money_payment_example()
            mtn_result = await self.mtn_mobile_money_example()
            bank_result = await self.bank_transfer_example()
            ussd_result = await self.ussd_payment_example()
            
            # Run utility examples
            await self.get_supported_payment_methods_example()
            await self.account_balance_example()
            await self.health_check_example()
            await self.transaction_fees_example()
            await self.webhook_processing_example()
            
            # Run verification and refund examples if we have successful payments
            if card_result and card_result.success:
                await self.payment_verification_example(card_result.transaction_id)
                
                # Only attempt refund if payment is completed
                if card_result.status == PaymentStatus.COMPLETED:
                    await self.refund_example(card_result.transaction_id, Decimal("500"))
            
            logger.info("All examples completed successfully!")
            
        except Exception as e:
            logger.error(f"Examples failed: {str(e)}")


# Standalone example functions
async def quick_card_payment_example():
    """Quick example: Card payment"""
    # Set environment variables first:
    # export FLUTTERWAVE_PUBLIC_KEY_SANDBOX="your_public_key"
    # export FLUTTERWAVE_SECRET_KEY_SANDBOX="your_secret_key"
    # export FLUTTERWAVE_ENCRYPTION_KEY_SANDBOX="your_encryption_key"
    
    service = await create_flutterwave_service(FlutterwaveEnvironment.SANDBOX)
    
    transaction = PaymentTransaction(
        id=f"QUICK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        amount=Decimal("100.00"),
        currency="NGN",
        description="Quick test payment",
        customer_email="test@example.com"
    )
    
    payment_method = PaymentMethod(
        method_type=PaymentMethodType.CARD,
        card_number="4187427415564246",
        expiry_month="12",
        expiry_year="2025",
        security_code="123",
        metadata={'customer_email': 'test@example.com'}
    )
    
    result = await service.process_payment(transaction, payment_method)
    print(f"Payment result: {result.success}, Status: {result.status.value}")
    
    return result


async def quick_mpesa_payment_example():
    """Quick example: M-Pesa payment"""
    service = await create_flutterwave_service(FlutterwaveEnvironment.SANDBOX)
    
    transaction = PaymentTransaction(
        id=f"MPESA_QUICK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        amount=Decimal("100.00"),
        currency="KES",
        description="Quick M-Pesa test",
        customer_email="test@example.com"
    )
    
    payment_method = PaymentMethod(
        method_type=PaymentMethodType.MOBILE_MONEY,
        metadata={
            'mobile_money_type': 'mpesa',
            'phone_number': '254708374149',
            'customer_email': 'test@example.com'
        }
    )
    
    result = await service.process_payment(transaction, payment_method)
    print(f"M-Pesa result: {result.success}, Status: {result.status.value}")
    
    return result


# Main execution
async def main():
    """Main example execution"""
    
    # Check environment variables
    required_vars = [
        'FLUTTERWAVE_PUBLIC_KEY_SANDBOX',
        'FLUTTERWAVE_SECRET_KEY_SANDBOX', 
        'FLUTTERWAVE_ENCRYPTION_KEY_SANDBOX'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        logger.info("\nPlease set these environment variables and try again.")
        logger.info("Example:")
        logger.info("  export FLUTTERWAVE_PUBLIC_KEY_SANDBOX='your_public_key'")
        logger.info("  export FLUTTERWAVE_SECRET_KEY_SANDBOX='your_secret_key'")
        logger.info("  export FLUTTERWAVE_ENCRYPTION_KEY_SANDBOX='your_encryption_key'")
        return
    
    # Run examples
    client = FlutterwaveClientExample()
    await client.run_all_examples()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())