"""
DPO Client Usage Examples - APG Payment Gateway

Comprehensive examples demonstrating how to use the DPO integration:
- Card payments with 3D Secure
- Mobile money payments (M-Pesa, Airtel Money, MTN, Orange Money, Tigo Pesa)
- Bank transfer payments
- Payment verification and status checking
- Refund processing (manual process)
- Callback handling
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
from dpo_integration import (
    create_dpo_service, 
    DPOEnvironment,
    DPOService
)
from dpo_webhook_handler import create_dpo_webhook_handler
from models import PaymentTransaction, PaymentMethod, PaymentMethodType, PaymentStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPOClientExample:
    """Example client for DPO integration"""
    
    def __init__(self):
        self.service: Optional[DPOService] = None
        self.webhook_handler = None
    
    async def initialize(self):
        """Initialize DPO service"""
        try:
            # Create service (uses environment variables for credentials)
            environment = DPOEnvironment.SANDBOX  # Change to LIVE for production
            self.service = await create_dpo_service(environment)
            
            # Create webhook handler
            self.webhook_handler = await create_dpo_webhook_handler(self.service)
            
            logger.info("DPO client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DPO client: {str(e)}")
            raise
    
    async def standard_payment_example(self):
        """Example: Process standard payment (all methods available)"""
        logger.info("=== Standard Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"DPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("1000.00"),  # KES 1,000 
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
                    'phone': '+254700000000',
                    'address': '123 Main Street',
                    'city': 'Nairobi',
                    'country_code': 'KE',
                    'redirect_url': 'https://example.com/payment/success',
                    'back_url': 'https://example.com/payment/cancel'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"Payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  DPO Token: {result.provider_transaction_id}")
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
                id=f"DPO_MPESA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("500.00"),  # KES 500
                currency="KES",
                description="M-Pesa payment - Airtime purchase",
                customer_email="customer@example.com",
                customer_name="Jane Doe"
            )
            
            # Create M-Pesa payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.MOBILE_MONEY,
                metadata={
                    'phone': '254700000000',
                    'provider': 'MPESA',
                    'customer_email': 'customer@example.com',
                    'customer_name': 'Jane Doe',
                    'address': '456 Mobile Street',
                    'city': 'Nairobi',
                    'country_code': 'KE'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"M-Pesa payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  DPO Token: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.payment_url:
                logger.info(f"  Payment URL: {result.payment_url}")
                logger.info("  Customer will complete M-Pesa payment on DPO's interface")
            
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
                id=f"DPO_AIRTEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("300.00"),  # KES 300
                currency="KES",
                description="Airtel Money payment - Utility bill",
                customer_email="customer@example.com",
                customer_name="Mary Wanjiku"
            )
            
            # Create Airtel Money payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.MOBILE_MONEY,
                metadata={
                    'phone': '254700000001',
                    'provider': 'AIRTEL',
                    'customer_email': 'customer@example.com',
                    'customer_name': 'Mary Wanjiku',
                    'address': '789 Airtel Avenue',
                    'city': 'Nairobi',
                    'country_code': 'KE'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"Airtel Money payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  DPO Token: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.payment_url:
                logger.info(f"  Payment URL: {result.payment_url}")
            
            return result
            
        except Exception as e:
            logger.error(f"Airtel Money payment example failed: {str(e)}")
            return None
    
    async def mtn_mobile_money_example(self):
        """Example: Process MTN Mobile Money payment"""
        logger.info("=== MTN Mobile Money Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"DPO_MTN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("200.00"),  # GHS 200
                currency="GHS",
                description="MTN Mobile Money payment - Internet bundle",
                customer_email="customer@example.com",
                customer_name="Kwame Asante"
            )
            
            # Create MTN Mobile Money payment method
            payment_method = PaymentMethod(
                method_type=PaymentMethodType.MOBILE_MONEY,
                metadata={
                    'phone': '233244000000',
                    'provider': 'MTN',
                    'customer_email': 'customer@example.com',
                    'customer_name': 'Kwame Asante',
                    'address': '321 MTN Road',
                    'city': 'Accra',
                    'country_code': 'GH'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"MTN Mobile Money payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  DPO Token: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.payment_url:
                logger.info(f"  Payment URL: {result.payment_url}")
            
            return result
            
        except Exception as e:
            logger.error(f"MTN Mobile Money payment example failed: {str(e)}")
            return None
    
    async def card_payment_example(self):
        """Example: Process card payment"""
        logger.info("=== Card Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"DPO_CARD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("2500.00"),  # KES 2,500
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
                    'address': '456 Business Avenue',
                    'city': 'Nairobi',
                    'country_code': 'KE',
                    'phone': '+254700000000'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"Card payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  DPO Token: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.payment_url:
                logger.info(f"  Payment URL: {result.payment_url}")
                logger.info("  Customer will enter card details on DPO's secure page")
            
            return result
            
        except Exception as e:
            logger.error(f"Card payment example failed: {str(e)}")
            return None
    
    async def bank_transfer_example(self):
        """Example: Process bank transfer payment"""
        logger.info("=== Bank Transfer Payment Example ===")
        
        try:
            # Create transaction
            transaction = PaymentTransaction(
                id=f"DPO_BANK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                amount=Decimal("5000.00"),  # KES 5,000
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
                    'address': 'Corporate Plaza',
                    'city': 'Nairobi',
                    'country_code': 'KE',
                    'phone': '+254700000000'
                }
            )
            
            # Process payment
            result = await self.service.process_payment(transaction, payment_method)
            
            logger.info(f"Bank transfer payment result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  DPO Token: {result.provider_transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            
            if result.payment_url:
                logger.info(f"  Payment URL: {result.payment_url}")
                logger.info("  Customer will complete bank transfer via DPO")
            
            return result
            
        except Exception as e:
            logger.error(f"Bank transfer payment example failed: {str(e)}")
            return None
    
    async def payment_verification_example(self, transaction_token: str):
        """Example: Verify payment status"""
        logger.info(f"=== Payment Verification Example ===")
        
        try:
            # Verify payment
            result = await self.service.verify_payment(transaction_token)
            
            logger.info(f"Payment verification result:")
            logger.info(f"  Transaction Token: {transaction_token}")
            logger.info(f"  Transaction ID: {result.transaction_id}")
            logger.info(f"  Status: {result.status.value}")
            logger.info(f"  Amount: {result.amount} {result.currency}")
            logger.info(f"  Success: {result.success}")
            
            if result.error_message:
                logger.error(f"  Error: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Payment verification example failed: {str(e)}")
            return None
    
    async def refund_example(self, transaction_token: str, refund_amount: Optional[Decimal] = None):
        """Example: Process refund (manual process)"""
        logger.info(f"=== Refund Example ===")
        
        try:
            # Process refund
            result = await self.service.refund_payment(
                transaction_id=transaction_token,
                amount=refund_amount,
                reason="Customer requested refund"
            )
            
            logger.info(f"Refund result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Transaction Token: {transaction_token}")
            logger.info(f"  Status: {result.status.value}")
            logger.info(f"  Refund Amount: {result.amount}")
            
            if result.success:
                logger.info("  NOTE: DPO refunds require manual processing")
                logger.info("  Process refund through DPO merchant portal")
            
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
            
            # Get methods for Ghana
            ghana_methods = await self.service.get_supported_payment_methods(country_code="GH")
            
            logger.info("\nSupported payment methods for Ghana:")
            for method in ghana_methods:
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
    
    async def callback_processing_example(self):
        """Example: Process callback"""
        logger.info("=== Callback Processing Example ===")
        
        try:
            # Example callback data (this would come from DPO)
            callback_data = {
                "TransactionToken": "DPO_20250131_123456",
                "CompanyRef": "DPO_20250131_123456",
                "TransactionStatus": "COMPLETE",
                "PaymentAmount": "1000.00",
                "PaymentCurrency": "KES",
                "CustomerName": "John Doe",
                "CustomerEmail": "customer@example.com",
                "TransactionFinal": "1",
                "TransactionDate": "2025-01-31 10:30:00"
            }
            
            # Process callback
            result = await self.webhook_handler.process_callback(callback_data, verify_ip=False)
            
            logger.info(f"Callback processing result:")
            logger.info(f"  Success: {result.get('success')}")
            logger.info(f"  Message: {result.get('message')}")
            logger.info(f"  Transaction Token: {result.get('transaction_token')}")
            logger.info(f"  Company Ref: {result.get('company_ref')}")
            logger.info(f"  Transaction Status: {result.get('transaction_status')}")
            logger.info(f"  Verification Status: {result.get('verification_status')}")
            
            if not result.get('success'):
                logger.error(f"  Error: {result.get('error')}")
            
            # Get callback stats
            stats = self.webhook_handler.get_callback_stats()
            logger.info(f"Callback stats: {stats}")
            
            return result
            
        except Exception as e:
            logger.error(f"Callback processing example failed: {str(e)}")
            return None
    
    async def transaction_fees_example(self):
        """Example: Calculate transaction fees"""
        logger.info("=== Transaction Fees Example ===")
        
        try:
            # Calculate fees for different payment methods
            amounts = [Decimal("1000"), Decimal("5000"), Decimal("10000")]
            currencies = ["KES", "GHS", "TZS", "NGN", "ZAR"]
            payment_methods = ["VISA", "MASTERCARD", "MPESA", "AIRTEL", "MTN", "BANK"]
            
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
                id=f"DPO_LINK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
        logger.info("Starting DPO integration examples...")
        
        try:
            # Initialize service
            await self.initialize()
            
            # Run payment examples
            standard_result = await self.standard_payment_example()
            mobile_result = await self.mobile_money_payment_example()
            airtel_result = await self.airtel_money_payment_example()
            mtn_result = await self.mtn_mobile_money_example()
            card_result = await self.card_payment_example()
            bank_result = await self.bank_transfer_example()
            
            # Run utility examples
            await self.get_supported_payment_methods_example()
            await self.health_check_example()
            await self.transaction_fees_example()
            await self.payment_link_example()
            await self.callback_processing_example()
            
            # Run verification and refund examples if we have successful payments
            if standard_result and standard_result.success and standard_result.provider_transaction_id:
                await self.payment_verification_example(standard_result.provider_transaction_id)
                
                # Only attempt refund if payment is completed
                if standard_result.status == PaymentStatus.COMPLETED:
                    await self.refund_example(standard_result.provider_transaction_id, Decimal("500"))
            
            logger.info("All examples completed successfully!")
            
        except Exception as e:
            logger.error(f"Examples failed: {str(e)}")


# Standalone example functions
async def quick_dpo_payment_example():
    """Quick example: Standard payment"""
    # Set environment variables first:
    # export DPO_COMPANY_TOKEN_SANDBOX="your_company_token"
    # export DPO_CALLBACK_URL="https://your-site.com/callback"
    # export DPO_REDIRECT_URL="https://your-site.com/success"
    
    service = await create_dpo_service(DPOEnvironment.SANDBOX)
    
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
            'address': 'Test Address',
            'city': 'Test City',
            'country_code': 'KE',
            'phone': '+254700000000'
        }
    )
    
    result = await service.process_payment(transaction, payment_method)
    print(f"Payment result: {result.success}, Status: {result.status.value}")
    if result.payment_url:
        print(f"Payment URL: {result.payment_url}")
    
    return result


async def quick_mpesa_payment_example():
    """Quick example: M-Pesa payment"""
    service = await create_dpo_service(DPOEnvironment.SANDBOX)
    
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
            'phone': '254700000000',
            'provider': 'MPESA',
            'customer_email': 'test@example.com',
            'customer_name': 'Test Customer',
            'address': 'Test Address',
            'city': 'Test City',
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
        'DPO_COMPANY_TOKEN_SANDBOX'
    ]
    
    optional_vars = [
        'DPO_CALLBACK_URL',
        'DPO_REDIRECT_URL'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        logger.info("\nPlease set these environment variables and try again.")
        logger.info("Example:")
        logger.info("  export DPO_COMPANY_TOKEN_SANDBOX='your_company_token'")
        logger.info("  export DPO_CALLBACK_URL='https://your-site.com/callback'")
        logger.info("  export DPO_REDIRECT_URL='https://your-site.com/success'")
        return
    
    # Check optional variables
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    if missing_optional:
        logger.warning("Missing optional environment variables:")
        for var in missing_optional:
            logger.warning(f"  - {var}")
        logger.info("Some features may be limited without these variables.")
    
    # Run examples
    client = DPOClientExample()
    await client.run_all_examples()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())