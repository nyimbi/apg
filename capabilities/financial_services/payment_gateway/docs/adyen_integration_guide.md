# Adyen Integration Guide - APG Payment Gateway

Complete guide for integrating Adyen's enterprise-grade payment platform with the APG Payment Gateway, covering global markets, enterprise features, and advanced payment flows.

## Overview

Adyen is a global payment company that provides a unified commerce platform for businesses worldwide. This integration supports:

- **200+ payment methods** across 150+ currencies
- **Unified commerce** across online, mobile, and in-store
- **Enterprise-grade** security and compliance
- **Advanced features** like marketplace payments, recurring billing, and risk management

## Quick Start

### 1. Environment Setup

```bash
# Required environment variables
export ADYEN_API_KEY="your_live_api_key"
export ADYEN_MERCHANT_ACCOUNT="your_merchant_account"
export ADYEN_CLIENT_KEY="your_client_key"
export ADYEN_HMAC_KEY="your_hmac_key_for_webhooks"
export ADYEN_ENVIRONMENT="live"  # or "test" for sandbox
```

### 2. Basic Payment Processing

```python
from payment_gateway_service import PaymentGatewayService
from models import PaymentTransaction, PaymentMethod, PaymentMethodType
from decimal import Decimal

# Initialize service
service = PaymentGatewayService()
await service.initialize()

# Create transaction
transaction = PaymentTransaction(
    id="TXN_ADY_001",
    amount=Decimal("2500.00"),
    currency="EUR",
    description="Adyen test payment",
    customer_email="customer@example.com",
    customer_name="John Doe"
)

# Create payment method
payment_method = PaymentMethod(
    method_type=PaymentMethodType.CARD,
    metadata={
        'card_number': '4111111111111111',
        'exp_month': '03',
        'exp_year': '2026',
        'cvc': '737',
        'cardholder_name': 'John Doe'
    }
)

# Process payment
result = await service.process_payment(
    transaction=transaction,
    payment_method=payment_method,
    provider="adyen"
)

print(f"Payment result: {result.success}")
print(f"Transaction ID: {result.provider_transaction_id}")
```

## Supported Payment Methods

### Cards
- **Visa, Mastercard, American Express**: Global acceptance
- **Local schemes**: Cartes Bancaires (France), Dankort (Denmark), etc.
- **3D Secure 2.0**: Enhanced authentication for SCA compliance
- **Network tokens**: Enhanced security and higher approval rates

```python
# Card payment with 3D Secure
card_method = PaymentMethod(
    method_type=PaymentMethodType.CARD,
    metadata={
        'card_number': '4000000000003220',  # 3DS test card
        'exp_month': '03',
        'exp_year': '2026',
        'cvc': '737',
        'cardholder_name': 'John Doe',
        'enable_3ds': True
    }
)

result = await service.process_payment(transaction, card_method, provider="adyen")

if result.status == PaymentStatus.PENDING:
    # Handle 3D Secure redirect
    redirect_url = result.payment_url
    print(f"Redirect customer to: {redirect_url}")
```

### Digital Wallets
- **Apple Pay**: iOS and web integration
- **Google Pay**: Android and web integration
- **PayPal**: Global wallet solution
- **Alipay**: Chinese market
- **WeChat Pay**: Chinese market

```python
# Apple Pay payment
applepay_method = PaymentMethod(
    method_type=PaymentMethodType.DIGITAL_WALLET,
    metadata={
        'type': 'applepay',
        'applepay.token': 'base64_encoded_payment_token'
    }
)

result = await service.process_payment(transaction, applepay_method, provider="adyen")
```

### European Payment Methods
- **SEPA Direct Debit**: Euro zone bank transfers
- **iDEAL**: Netherlands online banking
- **Bancontact**: Belgium
- **Giropay**: Germany
- **SOFORT**: Germany, Austria, Belgium

```python
# iDEAL payment (Netherlands)
ideal_method = PaymentMethod(
    method_type=PaymentMethodType.BANK_TRANSFER,
    metadata={
        'type': 'ideal',
        'issuer': 'ideal_INGBNL2A'  # ING Bank
    }
)

result = await service.process_payment(transaction, ideal_method, provider="adyen")
```

### Asian Payment Methods
- **Alipay**: China's leading digital wallet
- **WeChat Pay**: Popular Chinese payment method
- **UnionPay**: Chinese card scheme
- **GrabPay**: Southeast Asia
- **Touch 'n Go**: Malaysia

```python
# Alipay payment
alipay_method = PaymentMethod(
    method_type=PaymentMethodType.DIGITAL_WALLET,
    metadata={
        'type': 'alipay'
    }
)

result = await service.process_payment(transaction, alipay_method, provider="adyen")
```

### African Payment Methods
- **M-Pesa**: Kenya, Tanzania, Uganda
- **MTN Mobile Money**: Multiple African countries
- **Orange Money**: West and Central Africa
- **Airtel Money**: Multiple African countries

```python
# M-Pesa payment via Adyen
mpesa_method = PaymentMethod(
    method_type=PaymentMethodType.MOBILE_MONEY,
    metadata={
        'type': 'mobilepay_ke',
        'telephone_number': '254700000000'
    }
)

result = await service.process_payment(transaction, mpesa_method, provider="adyen")
```

## Advanced Features

### 1. Marketplace Payments

Split payments between platform and merchants:

```python
async def process_marketplace_payment():
    transaction = PaymentTransaction(
        id="MARKETPLACE_001",
        amount=Decimal("10000.00"),
        currency="EUR",
        description="Marketplace payment"
    )
    
    payment_method = PaymentMethod(
        method_type=PaymentMethodType.CARD,
        metadata={'card_number': '4111111111111111'}
    )
    
    # Define payment splits
    splits = [
        {
            "account": "platform_account_code",
            "amount": Decimal("1000.00"),  # Platform fee
            "type": "MarketPlace",
            "description": "Platform commission"
        },
        {
            "account": "merchant_account_code", 
            "amount": Decimal("9000.00"),  # Merchant payment
            "type": "Commission",
            "description": "Merchant payment"
        }
    ]
    
    result = await service.process_marketplace_payment(
        transaction=transaction,
        payment_method=payment_method,
        splits=splits,
        provider="adyen"
    )
    
    return result
```

### 2. Recurring Payments

Set up subscription billing:

```python
async def setup_recurring_payment():
    # Initial setup transaction (can be €0)
    setup_transaction = PaymentTransaction(
        id="RECURRING_SETUP_001",
        amount=Decimal("0.00"),
        currency="EUR",
        description="Subscription setup"
    )
    
    payment_method = PaymentMethod(
        method_type=PaymentMethodType.CARD,
        metadata={
            'card_number': '4111111111111111',
            'exp_month': '03',
            'exp_year': '2026',
            'cvc': '737'
        }
    )
    
    # Setup recurring payment
    result = await service.setup_recurring_payment(
        transaction=setup_transaction,
        payment_method=payment_method,
        customer_id="customer_123",
        provider="adyen"
    )
    
    if result["success"]:
        recurring_token = result["recurring_reference"]
        
        # Process subsequent recurring payment
        recurring_transaction = PaymentTransaction(
            id="RECURRING_001",
            amount=Decimal("2999.00"),
            currency="EUR",
            description="Monthly subscription"
        )
        
        recurring_result = await service.process_recurring_payment(
            transaction=recurring_transaction,
            recurring_token=recurring_token,
            provider="adyen"
        )
        
        return recurring_result
```

### 3. Multi-currency Processing

Handle global transactions:

```python
async def process_multi_currency_payment():
    # USD transaction
    usd_transaction = PaymentTransaction(
        id="USD_001",
        amount=Decimal("100.00"),
        currency="USD",
        description="US payment"
    )
    
    # EUR transaction
    eur_transaction = PaymentTransaction(
        id="EUR_001", 
        amount=Decimal("85.00"),
        currency="EUR",
        description="European payment"
    )
    
    # GBP transaction
    gbp_transaction = PaymentTransaction(
        id="GBP_001",
        amount=Decimal("75.00"),
        currency="GBP", 
        description="UK payment"
    )
    
    # Process all currencies
    for transaction in [usd_transaction, eur_transaction, gbp_transaction]:
        result = await service.process_payment(
            transaction=transaction,
            payment_method=card_method,
            provider="adyen"
        )
        print(f"{transaction.currency} payment: {result.success}")
```

### 4. Risk Management

Configure fraud prevention:

```python
async def process_payment_with_risk_data():
    transaction = PaymentTransaction(
        id="RISK_001",
        amount=Decimal("5000.00"),
        currency="EUR",
        description="High-value payment"
    )
    
    payment_method = PaymentMethod(
        method_type=PaymentMethodType.CARD,
        metadata={
            'card_number': '4111111111111111',
            'exp_month': '03',
            'exp_year': '2026',
            'cvc': '737'
        }
    )
    
    # Add risk data
    risk_metadata = {
        'fraud_offset': '0',
        'request_id': 'risk_request_123',
        'account_score': '50',
        'delivery_address_indicator': 'shipToBillingAddress',
        'delivery_timeframe': 'electronicDelivery'
    }
    
    result = await service.process_payment(
        transaction=transaction,
        payment_method=payment_method,
        provider="adyen",
        metadata=risk_metadata
    )
    
    return result
```

## Webhook Integration

### 1. Webhook Configuration

Set up Adyen webhooks in your merchant account:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
service = PaymentGatewayService()

@app.route('/webhooks/adyen', methods=['POST'])
async def handle_adyen_webhook():
    try:
        # Get webhook payload
        payload = request.get_json()
        
        # Verify webhook signature
        is_valid = await service.validate_callback_signature(
            payload=request.get_data(),
            signature=request.headers.get('X-Adyen-Signature'),
            provider="adyen"
        )
        
        if not is_valid:
            return jsonify({'error': 'Invalid signature'}), 400
        
        # Process webhook
        result = await service.process_webhook(
            provider="adyen",
            payload=payload,
            headers=request.headers
        )
        
        if result["success"]:
            # Handle successful webhook
            event_type = result["event_type"]
            transaction_id = result["transaction_id"]
            
            if event_type == "AUTHORISATION":
                # Payment authorized
                print(f"Payment {transaction_id} authorized")
            elif event_type == "CAPTURE":
                # Payment captured
                print(f"Payment {transaction_id} captured")
            elif event_type == "REFUND":
                # Payment refunded
                print(f"Payment {transaction_id} refunded")
            
            return jsonify({'notificationResponse': '[accepted]'})
        else:
            return jsonify({'error': 'Webhook processing failed'}), 500
            
    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({'error': 'Internal server error'}), 500
```

### 2. Webhook Events

Common Adyen webhook events:

- **AUTHORISATION**: Payment authorized
- **CAPTURE**: Payment captured (settled)
- **REFUND**: Payment refunded
- **CANCELLATION**: Payment cancelled
- **CHARGEBACK**: Chargeback initiated
- **DISPUTE**: Dispute opened
- **REPORT_AVAILABLE**: Settlement report available

## Error Handling

### Common Error Scenarios

```python
async def handle_payment_errors():
    try:
        result = await service.process_payment(transaction, payment_method, provider="adyen")
        
        if not result.success:
            error_code = result.raw_response.get('errorCode')
            
            if error_code == '101':
                # Declined by issuer
                print("Payment declined - insufficient funds")
            elif error_code == '103':
                # Invalid card details
                print("Invalid card details provided")
            elif error_code == '702':
                # Invalid merchant account
                print("Merchant account configuration error")
            elif error_code == '800':
                # Contract not found
                print("Payment method not enabled")
            else:
                print(f"Payment failed: {result.error_message}")
                
    except Exception as e:
        print(f"Integration error: {e}")
```

### Retry Logic

```python
import asyncio
from typing import Optional

async def process_payment_with_retry(
    transaction: PaymentTransaction,
    payment_method: PaymentMethod,
    max_retries: int = 3
) -> Optional[PaymentResult]:
    
    for attempt in range(max_retries):
        try:
            result = await service.process_payment(
                transaction=transaction,
                payment_method=payment_method,
                provider="adyen"
            )
            
            if result.success:
                return result
            
            # Check if error is retryable
            error_code = result.raw_response.get('errorCode')
            if error_code in ['101', '103', '702']:  # Non-retryable errors
                break
                
            # Wait before retry
            await asyncio.sleep(2 ** attempt)
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    
    return None
```

## Testing

### Test Cards

Adyen provides comprehensive test cards:

```python
# Test card numbers for different scenarios
TEST_CARDS = {
    'visa_success': '4111111111111111',
    'visa_declined': '4000300011112220',
    'visa_3ds': '4000000000003220',
    'mastercard_success': '5555444433331111',
    'amex_success': '370000000000002',
    'discover_success': '6011000000000012'
}

async def test_payment_scenarios():
    for scenario, card_number in TEST_CARDS.items():
        payment_method = PaymentMethod(
            method_type=PaymentMethodType.CARD,
            metadata={
                'card_number': card_number,
                'exp_month': '03',
                'exp_year': '2026',
                'cvc': '737'
            }
        )
        
        result = await service.process_payment(
            transaction=transaction,
            payment_method=payment_method,
            provider="adyen"
        )
        
        print(f"{scenario}: {result.success}")
```

### Integration Testing

```python
import pytest

class TestAdyenIntegration:
    @pytest.fixture
    async def adyen_service(self):
        service = PaymentGatewayService()
        await service.initialize()
        return service
    
    async def test_card_payment_success(self, adyen_service):
        transaction = PaymentTransaction(
            id="test_card_001",
            amount=Decimal("100.00"),
            currency="EUR",
            description="Test card payment"
        )
        
        payment_method = PaymentMethod(
            method_type=PaymentMethodType.CARD,
            metadata={
                'card_number': '4111111111111111',
                'exp_month': '03',
                'exp_year': '2026',
                'cvc': '737'
            }
        )
        
        result = await adyen_service.process_payment(
            transaction=transaction,
            payment_method=payment_method,
            provider="adyen"
        )
        
        assert result.success is True
        assert result.status == PaymentStatus.COMPLETED
        assert result.provider_transaction_id is not None
    
    async def test_3ds_authentication(self, adyen_service):
        # Test 3D Secure flow
        payment_method = PaymentMethod(
            method_type=PaymentMethodType.CARD,
            metadata={
                'card_number': '4000000000003220',  # 3DS test card
                'exp_month': '03',
                'exp_year': '2026',
                'cvc': '737'
            }
        )
        
        result = await adyen_service.process_payment(
            transaction=transaction,
            payment_method=payment_method,
            provider="adyen"
        )
        
        assert result.success is True
        assert result.status == PaymentStatus.PENDING
        assert result.payment_url is not None
        assert "redirect" in result.raw_response.get("action", {}).get("type", "")
```

## Configuration

### Merchant Account Setup

1. **Create Adyen account**: Sign up at adyen.com
2. **Get API credentials**: Generate API key and client key
3. **Configure webhooks**: Set up notification endpoints
4. **Enable payment methods**: Configure supported payment methods
5. **Set up HMAC keys**: For webhook signature verification

### Environment Configuration

```yaml
# config/adyen.yaml
adyen:
  environment: test  # or live
  api_key: ${ADYEN_API_KEY}
  merchant_account: ${ADYEN_MERCHANT_ACCOUNT}
  client_key: ${ADYEN_CLIENT_KEY}
  hmac_key: ${ADYEN_HMAC_KEY}
  
  endpoints:
    checkout: https://checkout-test.adyen.com/v70
    management: https://management-test.adyen.com/v1
    
  timeouts:
    connection: 30
    read: 60
    
  retry:
    max_attempts: 3
    backoff_factor: 2
    
  webhook:
    verify_signature: true
    allowed_origins:
      - "*.adyen.com"
```

### Payment Method Configuration

```python
# Configure available payment methods by region
PAYMENT_METHODS_CONFIG = {
    'EUR': {
        'cards': ['visa', 'mastercard', 'amex'],
        'wallets': ['applepay', 'googlepay', 'paypal'],
        'local': ['ideal', 'sepadirectdebit', 'bancontact']
    },
    'USD': {
        'cards': ['visa', 'mastercard', 'amex', 'discover'],
        'wallets': ['applepay', 'googlepay', 'paypal']
    },
    'GBP': {
        'cards': ['visa', 'mastercard', 'amex'],
        'wallets': ['applepay', 'googlepay', 'paypal'],
        'local': ['directdebit_GB']
    },
    'KES': {
        'cards': ['visa', 'mastercard'],
        'mobile': ['mobilepay_ke'],  # M-Pesa via Adyen
        'wallets': ['googlepay']
    }
}
```

## Performance & Scaling

### Connection Pooling

```python
import aiohttp
from aiohttp import TCPConnector

async def create_adyen_client():
    connector = TCPConnector(
        limit=100,  # Total connection pool size
        limit_per_host=30,  # Per-host connection limit
        ttl_dns_cache=300,  # DNS cache TTL
        use_dns_cache=True,
        keepalive_timeout=30
    )
    
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=60)
    )
    
    return session
```

### Caching

```python
import asyncio
from typing import Dict, Any

class AdyenConfigCache:
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.cache: Dict[str, Any] = {}
        
    async def get_payment_methods(self, country_code: str, currency: str):
        cache_key = f"payment_methods_{country_code}_{currency}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Fetch from Adyen API
        methods = await self._fetch_payment_methods(country_code, currency)
        self.cache[cache_key] = methods
        
        # Schedule cache expiration
        asyncio.create_task(self._expire_cache(cache_key))
        
        return methods
    
    async def _expire_cache(self, key: str):
        await asyncio.sleep(self.ttl)
        self.cache.pop(key, None)
```

## Security Best Practices

### 1. API Key Management

```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.encryption_key = os.environ.get('ENCRYPTION_KEY')
        self.fernet = Fernet(self.encryption_key)
    
    def get_api_key(self) -> str:
        encrypted_key = os.environ.get('ADYEN_API_KEY_ENCRYPTED')
        return self.fernet.decrypt(encrypted_key.encode()).decode()
```

### 2. Webhook Signature Verification

```python
import hmac
import hashlib
import base64

def verify_adyen_webhook(payload: bytes, signature: str, hmac_key: str) -> bool:
    try:
        # Adyen uses HMAC-SHA256
        computed_signature = hmac.new(
            hmac_key.encode(),
            payload,
            hashlib.sha256
        ).digest()
        
        # Base64 encode the computed signature
        computed_signature_b64 = base64.b64encode(computed_signature).decode()
        
        return hmac.compare_digest(signature, computed_signature_b64)
    except Exception:
        return False
```

### 3. Request Logging (with PII masking)

```python
import re
import json

def mask_sensitive_data(data: dict) -> dict:
    """Mask sensitive payment data for logging"""
    masked = data.copy()
    
    # Mask card numbers
    if 'number' in masked:
        masked['number'] = mask_card_number(masked['number'])
    
    # Mask CVV
    if 'cvc' in masked:
        masked['cvc'] = '***'
    
    # Mask IBAN
    if 'iban' in masked:
        masked['iban'] = mask_iban(masked['iban'])
    
    return masked

def mask_card_number(card_number: str) -> str:
    if len(card_number) >= 8:
        return f"{card_number[:4]}****{card_number[-4:]}"
    return "****"

def mask_iban(iban: str) -> str:
    if len(iban) >= 8:
        return f"{iban[:4]}****{iban[-4:]}"
    return "****"
```

## Troubleshooting

### Common Issues

1. **Invalid Merchant Account**
   - Verify merchant account is active
   - Check API permissions
   - Ensure correct environment (test/live)

2. **3D Secure Issues**
   - Verify return URLs are accessible
   - Check SCA exemption rules
   - Test with 3DS test cards

3. **Webhook Problems**
   - Verify HMAC key configuration
   - Check endpoint accessibility
   - Review webhook event types

4. **Payment Method Not Available**
   - Check merchant account configuration
   - Verify payment method is enabled
   - Review country/currency restrictions

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add request/response logging
async def debug_payment_request(transaction, payment_method):
    print(f"Processing payment: {transaction.id}")
    print(f"Amount: {transaction.amount} {transaction.currency}")
    print(f"Payment method: {payment_method.method_type}")
    
    result = await service.process_payment(
        transaction=transaction,
        payment_method=payment_method,
        provider="adyen"
    )
    
    print(f"Result: {result.success}")
    if result.raw_response:
        print(f"Raw response: {json.dumps(result.raw_response, indent=2)}")
    
    return result
```

## Support Resources

### Documentation
- [Adyen API Documentation](https://docs.adyen.com/api-explorer/)
- [Payment Methods Guide](https://docs.adyen.com/payment-methods)
- [3D Secure Guide](https://docs.adyen.com/online-payments/3d-secure)
- [Webhooks Documentation](https://docs.adyen.com/development-resources/webhooks)

### Testing Tools
- [Test Card Numbers](https://docs.adyen.com/development-resources/test-card-numbers)
- [Webhook Testing](https://docs.adyen.com/development-resources/webhooks/webhook-testing)
- [API Explorer](https://docs.adyen.com/api-explorer/)

### Support Channels
- **Technical Support**: nyimbi@gmail.com
- **Adyen Support**: Via merchant account dashboard
- **Community**: Adyen Developer Community

---

© 2025 Datacraft. All rights reserved.

*This guide provides comprehensive coverage of Adyen integration with the APG Payment Gateway. For specific implementation questions or advanced use cases, please refer to the official Adyen documentation or contact our technical support team.*