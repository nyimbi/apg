"""
Complete MPESA Client Example - APG Payment Gateway

Demonstrates how to use all MPESA features:
- STK Push payments
- B2B transfers
- B2C payments
- C2B payments
- Account balance inquiry
- Transaction status queries
- Transaction reversals

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any

# Set environment variables for MPESA configuration
os.environ.update({
    "MPESA_CONSUMER_KEY": "your_consumer_key_here",
    "MPESA_CONSUMER_SECRET": "your_consumer_secret_here",
    "MPESA_BUSINESS_SHORT_CODE": "174379",  # Sandbox shortcode
    "MPESA_PASSKEY": "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919",  # Sandbox passkey
    "MPESA_INITIATOR_NAME": "testapi",
    "MPESA_SECURITY_CREDENTIAL": "your_encrypted_security_credential",
    "MPESA_CALLBACK_BASE_URL": "https://your-domain.com"
})

from mpesa_integration import create_mpesa_service, MPESAEnvironment, MPESATransactionType
from models import PaymentTransaction, PaymentMethod, PaymentMethodType
from uuid_extensions import uuid7str

async def demonstrate_stk_push():
    """Demonstrate STK Push (Customer-to-Business) payment"""
    print("🔄 Demonstrating STK Push Payment...")
    
    # Create MPESA service
    mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)
    
    # Create payment transaction
    transaction = PaymentTransaction(
        id=uuid7str(),
        merchant_id="merchant_123",
        customer_id="customer_456",
        amount=1000,  # KES 10.00 (amount in cents)
        currency="KES",
        description="Test STK Push payment",
        payment_method_type=PaymentMethodType.MPESA,
        tenant_id="tenant_001"
    )
    
    # Create payment method
    payment_method = PaymentMethod(
        id=uuid7str(),
        customer_id="customer_456",
        payment_method_type=PaymentMethodType.MPESA,
        mpesa_phone_number="254708374149",  # Test phone number
        tenant_id="tenant_001"
    )
    
    # Additional data for STK Push
    additional_data = {
        "transaction_type": MPESATransactionType.STK_PUSH.value,
        "phone_number": "254708374149",
        "account_reference": "REF123456",
    }
    
    try:
        # Process STK Push payment
        result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
        
        if result.success:
            print(f"✅ STK Push initiated successfully!")
            print(f"   Checkout Request ID: {result.processor_transaction_id}")
            print(f"   Status: {result.status.value}")
            if result.requires_action:
                print(f"   Action Required: {result.action_data.get('message')}")
            
            # Wait for customer to complete payment on their phone
            print("⏳ Waiting for customer to complete payment...")
            await asyncio.sleep(30)  # In real implementation, use callbacks
            
            # Check transaction status
            status_result = await mpesa_service.get_transaction_status(result.processor_transaction_id)
            print(f"📊 Transaction Status: {status_result.status.value}")
            
        else:
            print(f"❌ STK Push failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ STK Push error: {str(e)}")
    
    await mpesa_service.cleanup()

async def demonstrate_b2b_payment():
    """Demonstrate B2B (Business-to-Business) payment"""
    print("\n🔄 Demonstrating B2B Payment...")
    
    mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)
    
    transaction = PaymentTransaction(
        id=uuid7str(),
        merchant_id="merchant_123",
        amount=5000,  # KES 50.00
        currency="KES",
        description="B2B payment to supplier",
        payment_method_type=PaymentMethodType.MPESA,
        tenant_id="tenant_001"
    )
    
    payment_method = PaymentMethod(
        id=uuid7str(),
        customer_id="business_supplier",
        payment_method_type=PaymentMethodType.MPESA,
        tenant_id="tenant_001"
    )
    
    additional_data = {
        "transaction_type": MPESATransactionType.B2B.value,
        "receiver_party": "600000",  # Receiver business short code
        "command_id": "BusinessPayment",
        "account_reference": "SUPPLIER_001"
    }
    
    try:
        result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
        
        if result.success:
            print(f"✅ B2B payment initiated successfully!")
            print(f"   Conversation ID: {result.processor_transaction_id}")
            print(f"   Receiver: {additional_data['receiver_party']}")
            print(f"   Amount: KES {transaction.amount / 100:.2f}")
        else:
            print(f"❌ B2B payment failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ B2B payment error: {str(e)}")
    
    await mpesa_service.cleanup()

async def demonstrate_b2c_payment():
    """Demonstrate B2C (Business-to-Customer) payment"""
    print("\n🔄 Demonstrating B2C Payment...")
    
    mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)
    
    transaction = PaymentTransaction(
        id=uuid7str(),
        merchant_id="merchant_123",
        customer_id="customer_789",
        amount=2000,  # KES 20.00
        currency="KES",
        description="Salary payment to employee",
        payment_method_type=PaymentMethodType.MPESA,
        tenant_id="tenant_001"
    )
    
    payment_method = PaymentMethod(
        id=uuid7str(),
        customer_id="customer_789",
        payment_method_type=PaymentMethodType.MPESA,
        mpesa_phone_number="254708374149",
        tenant_id="tenant_001"
    )
    
    additional_data = {
        "transaction_type": MPESATransactionType.B2C.value,
        "phone_number": "254708374149",
        "command_id": "SalaryPayment",
        "occasion": "Monthly Salary"
    }
    
    try:
        result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
        
        if result.success:
            print(f"✅ B2C payment initiated successfully!")
            print(f"   Conversation ID: {result.processor_transaction_id}")
            print(f"   Recipient: {additional_data['phone_number']}")
            print(f"   Amount: KES {transaction.amount / 100:.2f}")
            print(f"   Purpose: {additional_data['occasion']}")
        else:
            print(f"❌ B2C payment failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ B2C payment error: {str(e)}")
    
    await mpesa_service.cleanup()

async def demonstrate_c2b_payment():
    """Demonstrate C2B (Customer-to-Business) payment simulation"""
    print("\n🔄 Demonstrating C2B Payment Simulation...")
    
    mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)
    
    transaction = PaymentTransaction(
        id=uuid7str(),
        merchant_id="merchant_123",
        customer_id="customer_101",
        amount=1500,  # KES 15.00
        currency="KES",
        description="Customer payment simulation",
        payment_method_type=PaymentMethodType.MPESA,
        tenant_id="tenant_001"
    )
    
    payment_method = PaymentMethod(
        id=uuid7str(),
        customer_id="customer_101",
        payment_method_type=PaymentMethodType.MPESA,
        mpesa_phone_number="254708374149",
        tenant_id="tenant_001"
    )
    
    additional_data = {
        "transaction_type": MPESATransactionType.C2B.value,
        "phone_number": "254708374149",
        "command_id": "CustomerPayBillOnline",
        "bill_ref_number": "BILL_REF_123"
    }
    
    try:
        result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
        
        if result.success:
            print(f"✅ C2B payment simulation initiated successfully!")
            print(f"   Conversation ID: {result.processor_transaction_id}")
            print(f"   Phone Number: {additional_data['phone_number']}")
            print(f"   Bill Reference: {additional_data['bill_ref_number']}")
            print(f"   Amount: KES {transaction.amount / 100:.2f}")
        else:
            print(f"❌ C2B payment failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ C2B payment error: {str(e)}")
    
    await mpesa_service.cleanup()

async def demonstrate_account_balance():
    """Demonstrate account balance inquiry"""
    print("\n🔄 Demonstrating Account Balance Inquiry...")
    
    mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)
    
    try:
        balance_result = await mpesa_service.get_account_balance()
        
        if balance_result.get("success"):
            print(f"✅ Account balance inquiry initiated successfully!")
            print(f"   Conversation ID: {balance_result.get('conversation_id')}")
            print(f"   Note: {balance_result.get('note')}")
        else:
            print(f"❌ Balance inquiry failed: {balance_result.get('error_message')}")
            
    except Exception as e:
        print(f"❌ Balance inquiry error: {str(e)}")
    
    await mpesa_service.cleanup()

async def demonstrate_transaction_status():
    """Demonstrate transaction status query"""
    print("\n🔄 Demonstrating Transaction Status Query...")
    
    mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)
    
    # Use a sample transaction ID (in real scenario, this would be from a previous transaction)
    sample_transaction_id = "sample_tx_123"
    
    try:
        status_result = await mpesa_service.get_transaction_status(sample_transaction_id)
        
        if status_result.success:
            print(f"✅ Transaction status query initiated successfully!")
            print(f"   Transaction ID: {sample_transaction_id}")
            print(f"   Status: {status_result.status.value}")
            if status_result.metadata:
                print(f"   Note: {status_result.metadata.get('note')}")
        else:
            print(f"❌ Status query failed: {status_result.error_message}")
            
    except Exception as e:
        print(f"❌ Status query error: {str(e)}")
    
    await mpesa_service.cleanup()

async def demonstrate_transaction_reversal():
    """Demonstrate transaction reversal"""
    print("\n🔄 Demonstrating Transaction Reversal...")
    
    mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)
    
    # Sample data for reversal (in real scenario, this would be from a completed transaction)
    sample_originator_conversation_id = "AG_20231201_12345678901234567890"
    sample_transaction_id = "QHJ12345678"  # MPESA receipt number
    reversal_amount = 1000  # KES 10.00
    
    try:
        reversal_result = await mpesa_service.reverse_transaction(
            originator_conversation_id=sample_originator_conversation_id,
            transaction_id=sample_transaction_id,
            amount=reversal_amount,
            reason="Customer requested refund"
        )
        
        if reversal_result.success:
            print(f"✅ Transaction reversal initiated successfully!")
            print(f"   Reversal Conversation ID: {reversal_result.processor_transaction_id}")
            print(f"   Original Transaction: {sample_transaction_id}")
            print(f"   Reversal Amount: KES {reversal_amount / 100:.2f}")
            print(f"   Status: {reversal_result.status.value}")
        else:
            print(f"❌ Transaction reversal failed: {reversal_result.error_message}")
            
    except Exception as e:
        print(f"❌ Transaction reversal error: {str(e)}")
    
    await mpesa_service.cleanup()

async def demonstrate_payment_tokenization():
    """Demonstrate payment method tokenization"""
    print("\n🔄 Demonstrating Payment Method Tokenization...")
    
    mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)
    
    payment_method_data = {
        "phone_number": "254708374149"
    }
    customer_id = "customer_tokenization_test"
    
    try:
        tokenization_result = await mpesa_service.tokenize_payment_method(
            payment_method_data, customer_id
        )
        
        if tokenization_result.get("success"):
            print(f"✅ Payment method tokenized successfully!")
            print(f"   Token: {tokenization_result.get('token')}")
            print(f"   Customer ID: {tokenization_result.get('customer_id')}")
            print(f"   Phone Last 4: {tokenization_result.get('phone_number_last_4')}")
            print(f"   Expires: {tokenization_result.get('expires_at')}")
        else:
            print(f"❌ Tokenization failed: {tokenization_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Tokenization error: {str(e)}")
    
    await mpesa_service.cleanup()

async def demonstrate_health_check():
    """Demonstrate service health check"""
    print("\n🔄 Demonstrating Service Health Check...")
    
    mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)
    
    try:
        health = await mpesa_service.health_check()
        
        print(f"📊 MPESA Service Health Status:")
        print(f"   Status: {health.status.value}")
        print(f"   Success Rate: {health.success_rate:.1%}")
        print(f"   Average Response Time: {health.average_response_time:.0f}ms")
        print(f"   Uptime: {health.uptime_percentage:.1f}%")
        print(f"   Supported Currencies: {', '.join(health.supported_currencies)}")
        print(f"   Supported Countries: {', '.join(health.supported_countries)}")
        
        if health.last_error:
            print(f"   Last Error: {health.last_error}")
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
    
    await mpesa_service.cleanup()

async def run_complete_mpesa_demo():
    """Run complete MPESA integration demonstration"""
    print("🚀 APG Payment Gateway - Complete MPESA Integration Demo")
    print("=" * 60)
    
    try:
        # Demonstrate all MPESA features
        await demonstrate_stk_push()
        await demonstrate_b2b_payment()
        await demonstrate_b2c_payment()
        await demonstrate_c2b_payment()
        await demonstrate_account_balance()
        await demonstrate_transaction_status()
        await demonstrate_transaction_reversal()
        await demonstrate_payment_tokenization()
        await demonstrate_health_check()
        
        print("\n✅ Complete MPESA integration demonstration completed!")
        print("\n📋 Summary of implemented features:")
        print("   ✅ STK Push (Customer-to-Business) payments")
        print("   ✅ B2B (Business-to-Business) transfers")
        print("   ✅ B2C (Business-to-Customer) payments")
        print("   ✅ C2B (Customer-to-Business) payments")
        print("   ✅ Account balance inquiry")
        print("   ✅ Transaction status queries")
        print("   ✅ Transaction reversals")
        print("   ✅ Payment method tokenization")
        print("   ✅ Service health monitoring")
        print("   ✅ OAuth token management")
        print("   ✅ Comprehensive error handling")
        print("   ✅ Callback/webhook processing")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the complete demonstration
    asyncio.run(run_complete_mpesa_demo())