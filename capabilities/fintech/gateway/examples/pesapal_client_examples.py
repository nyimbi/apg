"""
Pesapal Client Usage Examples - APG Payment Gateway

Comprehensive examples demonstrating how to use the Pesapal integration:
- Card payments with 3D Secure
- Mobile money payments (M-Pesa, Airtel Money)
- Bank transfer payments
- Payment verification and status checking
- Refund processing (manual process)
- IPN handling
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
from pesapal_integration import (
    create_pesapal_service, 
    PesapalEnvironment,
    PesapalService
)
from pesapal_webhook_handler import create_pesapal_webhook_handler
from models import PaymentTransaction, PaymentMethod, PaymentMethodType, PaymentStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PesapalClientExample:
    """Example client for Pesapal integration"""
    
    def __init__(self):
        self.service: Optional[PesapalService] = None
        self.webhook_handler = None
    
    async def initialize(self):
        """Initialize Pesapal service"""
        try:
            # Create service (uses environment variables for credentials)
            environment = PesapalEnvironment.SANDBOX  # Change to LIVE for production
            self.service = await create_pesapal_service(environment)
            
            # Create webhook handler
            self.webhook_handler = await create_pesapal_webhook_handler(self.service)
            
            logger.info("Pesapal client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pesapal client: {str(e)}")
            raise
    
    async def standard_payment_example(self):
        """Example: Process standard payment (all methods available)"""
        logger.info("=== Standard Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"PSP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("1000.00"),  # KSh 1,000 
                currency="KES",
                description="Test payment - Order #12345",
                customer_email="customer@example.com",
                customer_name="John Doe"
            )
            
            # Create payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.OTHER,
                metadata={
                    'customer_email': 'customer@example.com',
                    'customer_name': 'John Doe',
                    'phone_number': '+254700000000',
                    'callback_url': 'https://example.com/payment/callback',
                    'country_code': 'KE',
                    'city': 'Nairobi',
                    'address_line_1': '123 Main Street'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"Payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.payment_url:
                logger.info(f"  Payment URL: {result.payment_url}")
                logger.info("  Customer should be redirected to this URL to complete payment")
            
            if result.error_message:
                logger.error(f"  Error: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Standard payment example failed: {str(e)}")
            return None
    
    async def mobile_money_payment_example(self):
        """Example: Process M-Pesa payment"""
        logger.info("=== Mobile Money Payment Example (M-Pesa) ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"PSP_MPESA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("500.00"),  # KSh 500
                currency="KES",
                description="M-Pesa payment - Airtime purchase",
                customer_email="customer@example.com",
                customer_name="Jane Doe"
            )
            
            # Create M-Pesa payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.MOBILE_MONEY,
                metadata={
                    'phone_number': '254700000000',
                    'provider': 'MPESA',
                    'customer_email': 'customer@example.com',
                    'customer_name': 'Jane Doe',
                    'country_code': 'KE'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"M-Pesa payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.payment_url:
                logger.info(f"  Payment URL: {result.payment_url}")
                logger.info("  Customer will complete payment on Pesapal's interface")
            
            if result.error_message:
                logger.error(f"  Error: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"M-Pesa payment example failed: {str(e)}")
            return None
    
    async def airtel_money_payment_example(self):
        """Example: Process Airtel Money payment"""
        logger.info("=== Airtel Money Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"PSP_AIRTEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("300.00"),  # KSh 300
                currency="KES",
                description="Airtel Money payment - Utility bill",
                customer_email="customer@example.com",
                customer_name="Mary Wanjiku"
            )
            
            # Create Airtel Money payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.MOBILE_MONEY,
                metadata={
                    'phone_number': '254700000001',
                    'provider': 'AIRTEL_MONEY',
                    'customer_email': 'customer@example.com',
                    'customer_name': 'Mary Wanjiku',
                    'country_code': 'KE'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"Airtel Money payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.payment_url:
                logger.info(f"  Payment URL: {result.payment_url}")
            
            return result
            
        except Exception as e:
            logger.error(f"Airtel Money payment example failed: {str(e)}")
            return None
    
    async def bank_transfer_payment_example(self):
        """Example: Process bank transfer payment"""
        logger.info("=== Bank Transfer Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"PSP_BANK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("5000.00"),  # KSh 5,000
                currency="KES",
                description="Bank transfer payment - Invoice #INV-001",
                customer_email="business@example.com",
                customer_name="Business Customer Ltd"
            )
            
            # Create bank transfer payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.BANK_TRANSFER,
                metadata={
                    'customer_email': 'business@example.com',
                    'customer_name': 'Business Customer Ltd',
                    'country_code': 'KE'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"Bank transfer payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.payment_url:
                logger.info(f"  Payment URL: {result.payment_url}")
                logger.info("  Customer will complete bank transfer via Pesapal")
            
            return result
            
        except Exception as e:
            logger.error(f"Bank transfer payment example failed: {str(e)}")
            return None
    
    async def card_payment_example(self):
        """Example: Process card payment"""
        logger.info("=== Card Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"PSP_CARD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("2500.00"),  # KSh 2,500
                currency="KES",
                description="Card payment - Online shopping",
                customer_email="cardholder@example.com",
                customer_name="David Kimani"
            )
            
            # Create card payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.CARD,
                metadata={
                    'customer_email': 'cardholder@example.com',
                    'customer_name': 'David Kimani',
                    'country_code': 'KE',
                    'address_line_1': '456 Business Avenue',
                    'city': 'Nairobi'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"Card payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Provider ID: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.payment_url:
                logger.info(f"  Payment URL: {result.payment_url}")
                logger.info("  Customer will enter card details on Pesapal's secure page")
            
            return result
            
        except Exception as e:
            logger.error(f"Card payment example failed: {str(e)}")
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
        """Example: Process refund (manual process)"""
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
            
            if result.success:
                logger.info("  NOTE: Pesapal refunds require manual processing")
                logger.info("  Contact Pesapal support to complete the refund process")
            
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
    
    async def ipn_processing_example(self):
        """Example: Process IPN"""
        logger.info("=== IPN Processing Example ===")
        
        try:
            # Example IPN data (this would come from Pesapal)
            ipn_data = {
                "orderTrackingId": "PSP_20250131_123456",
                "notificationType": "CHANGE",
                "merchantReference": "ORDER-12345",
                "orderNotificationId": "123456"
            }
            
            # Mock signature (in real scenario, this comes from Pesapal)
            signature = "mock_signature"
            
            # Process IPN
            result = await self.webhook_handler.process_ipn(ipn_data, signature)
            
            logger.info(f"IPN processing result:")
            logger.info(f"  Success: {result.get('success')}")
            logger.info(f"  Message: {result.get('message')}")
            logger.info(f"  Order Tracking ID: {result.get('order_tracking_id')}")
            logger.info(f"  Notification Type: {result.get('notification_type')}")
            logger.info(f"  Transaction Status: {result.get('transaction_status')}")
            
            if not result.get('success'):
                logger.error(f"  Error: {result.get('error')}")
            
            # Get IPN stats
            stats = self.webhook_handler.get_ipn_stats()
            logger.info(f"IPN stats: {stats}")
            
            return result
            
        except Exception as e:
            logger.error(f"IPN processing example failed: {str(e)}")
            return None
    
    async def transaction_fees_example(self):
        """Example: Calculate transaction fees"""
        logger.info("=== Transaction Fees Example ===")
        
        try:
            # Calculate fees for different payment methods
            amounts = [Decimal("1000"), Decimal("5000"), Decimal("10000")]
            currencies = ["KES", "UGX", "TZS"]
            payment_methods = ["VISA", "MPESA", "AIRTEL_MONEY", "BANK_TRANSFER"]
            
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
    
    async def payment_link_example(self):
        """Example: Create payment link"""
        logger.info("=== Payment Link Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"PSP_LINK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("1500.00"),
                currency="KES",
                description="Payment link - Product purchase",
                customer_email="customer@example.com",
                customer_name="Payment Link Customer"
            )
            
            # Create payment link
            payment_url = await self.service.create_payment_link(transaction)
            
            logger.info(f"Payment link created:")
            logger.info(f"  Transaction ID: {transaction.id}")
            logger.info(f"  Payment URL: {payment_url}")
            
            if payment_url:
                logger.info("  Customer can use this URL to make payment")
            else:
                logger.error("  Failed to create payment link")
            
            return payment_url
            
        except Exception as e:
            logger.error(f"Payment link example failed: {str(e)}")
            return None
    
    async def run_all_examples(self):
        """Run all examples"""
        logger.info("Starting Pesapal integration examples...")
        
        try:
            # Initialize service
            await self.initialize()
            
            # Run payment examples
            standard_result = await self.standard_payment_example()
            mobile_result = await self.mobile_money_payment_example()
            airtel_result = await self.airtel_money_payment_example()
            bank_result = await self.bank_transfer_payment_example()
            card_result = await self.card_payment_example()
            
            # Run utility examples
            await self.get_supported_payment_methods_example()
            await self.health_check_example()
            await self.transaction_fees_example()
            await self.payment_link_example()
            await self.ipn_processing_example()
            
            # Run verification and refund examples if we have successful payments
            if standard_result and standard_result.success:
                await self.payment_verification_example(standard_result.transaction_id)
                
                # Only attempt refund if payment is completed
                if standard_result.status == PaymentStatus.COMPLETED:
                    await self.refund_example(standard_result.transaction_id, Decimal("500"))
            
            logger.info("All examples completed successfully!")
            
        except Exception as e:
            logger.error(f"Examples failed: {str(e)}")


# Standalone example functions
async def quick_pesapal_payment_example():
    """Quick example: Standard payment"""
    # Set environment variables first:
    # export PESAPAL_CONSUMER_KEY_SANDBOX="your_consumer_key"
    # export PESAPAL_CONSUMER_SECRET_SANDBOX="your_consumer_secret"
    # export PESAPAL_CALLBACK_URL="https://your-site.com/ipn"
    
    service = await create_pesapal_service(PesapalEnvironment.SANDBOX)
    
    transaction = PaymentTransaction(
        id=f"QUICK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        amount=Decimal("100.00"),
        currency="KES",
        description="Quick test payment",
        customer_email="test@example.com"
    )
    
    payment_method = PaymentMethod(
        method_type=PaymentMethodType.OTHER,
        metadata={
            'customer_email': 'test@example.com',
            'customer_name': 'Test Customer',
            'country_code': 'KE'
        }
    )
    
    result = await service.process_payment(transaction, payment_method)
    print(f"Payment result: {result.success}, Status: {result.status.value}")
    if result.payment_url:
        print(f"Payment URL: {result.payment_url}")
    
    return result


async def quick_mpesa_payment_example():
    """Quick example: M-Pesa payment"""
    service = await create_pesapal_service(PesapalEnvironment.SANDBOX)
    
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
            'phone_number': '254700000000',
            'provider': 'MPESA',
            'customer_email': 'test@example.com',
            'country_code': 'KE'
        }
    )
    
    result = await service.process_payment(transaction, payment_method)
    print(f"M-Pesa result: {result.success}, Status: {result.status.value}")
    if result.payment_url:
        print(f"Payment URL: {result.payment_url}")
    
    return result


# Main execution
async def main():
    """Main example execution"""
    
    # Check environment variables
    required_vars = [
        'PESAPAL_CONSUMER_KEY_SANDBOX',
        'PESAPAL_CONSUMER_SECRET_SANDBOX'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        logger.info("\nPlease set these environment variables and try again.")
        logger.info("Example:")
        logger.info("  export PESAPAL_CONSUMER_KEY_SANDBOX='your_consumer_key'")
        logger.info("  export PESAPAL_CONSUMER_SECRET_SANDBOX='your_consumer_secret'")
        logger.info("  export PESAPAL_CALLBACK_URL='https://your-site.com/ipn'")
        return
    
    # Run examples
    client = PesapalClientExample()
    await client.run_all_examples()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())