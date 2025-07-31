# APG Payment Gateway - Complete Multi-Provider Integration

A comprehensive payment processing solution for the APG platform supporting multiple payment providers, currencies, and payment methods across Africa and globally.

## Supported Payment Providers

### 1. Stripe
- **Global leader** in online payments
- **Features**: Cards, digital wallets, subscriptions, Connect marketplace
- **Coverage**: Global, 47+ countries
- **Best for**: International businesses, subscriptions, marketplaces

### 2. Adyen
- **Enterprise-grade** payment platform
- **Features**: Unified commerce, marketplace, recurring payments
- **Coverage**: 200+ payment methods, 150+ currencies
- **Best for**: Large enterprises, omnichannel commerce

### 3. Flutterwave
- **African payment specialist**
- **Features**: Mobile money, cards, bank transfers, Barter (virtual cards)
- **Coverage**: 34+ African countries
- **Best for**: African businesses, mobile money payments

### 4. Pesapal
- **East African focused**
- **Features**: M-Pesa, Airtel Money, cards, bank transfers
- **Coverage**: Kenya, Tanzania, Uganda, Rwanda
- **Best for**: East African businesses, local payment methods

### 5. DPO (Direct Pay Online)
- **Pan-African payment gateway**
- **Features**: Mobile money, cards, bank transfers across Africa
- **Coverage**: 40+ African countries
- **Best for**: Pan-African businesses, comprehensive coverage

## Quick Start

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Set environment variables
export STRIPE_SECRET_KEY="sk_test_..."
export FLUTTERWAVE_SECRET_KEY="FLWSECK_TEST..."
export PESAPAL_CONSUMER_KEY="..."
export DPO_COMPANY_TOKEN_SANDBOX="..."
```

### Basic Usage

```python
from payment_gateway_service import PaymentGatewayService
from models import PaymentTransaction, PaymentMethod, PaymentMethodType
from decimal import Decimal

# Initialize service
service = PaymentGatewayService()
await service.initialize()

# Create transaction
transaction = PaymentTransaction(
    id="TXN_001",
    amount=Decimal("1000.00"),
    currency="KES",
    description="Test payment",
    customer_email="customer@example.com"
)

# Create payment method
payment_method = PaymentMethod(
    method_type=PaymentMethodType.MOBILE_MONEY,
    metadata={
        'phone': '254700000000',
        'provider': 'MPESA'
    }
)

# Process payment
result = await service.process_payment(transaction, payment_method, provider="flutterwave")
print(f"Payment result: {result.success}")
```

## Payment Methods by Provider

### Mobile Money Support

| Provider | M-Pesa | Airtel Money | MTN MoMo | Orange Money | Tigo Pesa |
|----------|--------|--------------|----------|--------------|-----------|
| Stripe | ❌ | ❌ | ❌ | ❌ | ❌ |
| Adyen | ✅ | ✅ | ✅ | ✅ | ✅ |
| Flutterwave | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pesapal | ✅ | ✅ | ❌ | ❌ | ❌ |
| DPO | ✅ | ✅ | ✅ | ✅ | ✅ |

### Card Support

| Provider | Visa | Mastercard | Amex | 3D Secure |
|----------|------|------------|------|-----------|
| Stripe | ✅ | ✅ | ✅ | ✅ |
| Adyen | ✅ | ✅ | ✅ | ✅ |
| Flutterwave | ✅ | ✅ | ✅ | ✅ |
| Pesapal | ✅ | ✅ | ✅ | ✅ |
| DPO | ✅ | ✅ | ✅ | ✅ |

### Bank Transfers

| Provider | Local Banks | SEPA | ACH | Instant |
|----------|-------------|------|-----|---------|
| Stripe | ✅ | ✅ | ✅ | ✅ |
| Adyen | ✅ | ✅ | ✅ | ✅ |
| Flutterwave | ✅ | ❌ | ❌ | ✅ |
| Pesapal | ✅ | ❌ | ❌ | ❌ |
| DPO | ✅ | ❌ | ❌ | ❌ |

## Country Coverage

### African Countries

| Country | Stripe | Adyen | Flutterwave | Pesapal | DPO |
|---------|--------|-------|-------------|---------|-----|
| Kenya | ✅ | ✅ | ✅ | ✅ | ✅ |
| Nigeria | ✅ | ✅ | ✅ | ❌ | ✅ |
| South Africa | ✅ | ✅ | ✅ | ❌ | ✅ |
| Ghana | ✅ | ✅ | ✅ | ❌ | ✅ |
| Tanzania | ❌ | ✅ | ✅ | ✅ | ✅ |
| Uganda | ❌ | ✅ | ✅ | ✅ | ✅ |
| Rwanda | ❌ | ✅ | ✅ | ✅ | ✅ |

## API Examples

### Process Mobile Money Payment

```python
# M-Pesa payment via Flutterwave
transaction = PaymentTransaction(
    id="MPESA_001",
    amount=Decimal("500.00"),
    currency="KES",
    description="M-Pesa payment"
)

payment_method = PaymentMethod(
    method_type=PaymentMethodType.MOBILE_MONEY,
    metadata={
        'phone': '254700000000',
        'provider': 'MPESA',
        'network': 'MPESA'
    }
)

result = await service.process_payment(transaction, payment_method, provider="flutterwave")
```

### Process Card Payment

```python
# Card payment via Stripe
payment_method = PaymentMethod(
    method_type=PaymentMethodType.CARD,
    metadata={
        'card_number': '4242424242424242',
        'exp_month': '12',
        'exp_year': '2025',
        'cvc': '123'
    }
)

result = await service.process_payment(transaction, payment_method, provider="stripe")
```

### Verify Payment

```python
# Verify payment status
result = await service.verify_payment("txn_id", provider="stripe")
print(f"Status: {result.status.value}")
```

### Process Refund

```python
# Process refund
refund_result = await service.refund_payment(
    transaction_id="txn_id", 
    amount=Decimal("100.00"),
    reason="Customer request",
    provider="stripe"
)
```

## Webhook Integration

### Flask Example

```python
from flask import Flask, request, jsonify
from payment_gateway_service import PaymentGatewayService

app = Flask(__name__)
service = PaymentGatewayService()

@app.route('/webhooks/stripe', methods=['POST'])
async def stripe_webhook():
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    
    result = await service.process_webhook(
        provider="stripe",
        payload=payload,
        headers={'Stripe-Signature': sig_header}
    )
    
    return jsonify(result)

@app.route('/webhooks/flutterwave', methods=['POST'])
async def flutterwave_webhook():
    payload = request.get_json()
    
    result = await service.process_webhook(
        provider="flutterwave",
        payload=payload,
        headers=request.headers
    )
    
    return jsonify(result)
```

## Configuration

### Environment Variables

```bash
# Stripe
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Adyen
ADYEN_API_KEY=...
ADYEN_MERCHANT_ACCOUNT=...
ADYEN_CLIENT_KEY=...
ADYEN_HMAC_KEY=...

# Flutterwave
FLUTTERWAVE_PUBLIC_KEY=FLWPUBK_TEST-...
FLUTTERWAVE_SECRET_KEY=FLWSECK_TEST-...
FLUTTERWAVE_ENCRYPTION_KEY=FLWSECK_TEST...

# Pesapal
PESAPAL_CONSUMER_KEY=...
PESAPAL_CONSUMER_SECRET=...

# DPO
DPO_COMPANY_TOKEN_SANDBOX=...
DPO_COMPANY_TOKEN_LIVE=...
DPO_CALLBACK_URL=https://yoursite.com/callbacks/dpo
```

### YAML Configuration

```yaml
# config/payment_gateway.yaml
providers:
  stripe:
    enabled: true
    environment: "sandbox"
    default_currency: "USD"
    
  flutterwave:
    enabled: true
    environment: "sandbox"
    default_currency: "KES"
    
  pesapal:
    enabled: true
    environment: "sandbox"
    default_currency: "KES"
```

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific provider tests
python -m pytest tests/test_stripe_integration.py -v
python -m pytest tests/test_flutterwave_integration.py -v
```

### Test Cards

```python
# Stripe test cards
STRIPE_TEST_CARDS = {
    'visa_success': '4242424242424242',
    'visa_declined': '4000000000000002',
    'visa_3ds': '4000000000003220'
}

# Flutterwave test cards
FLUTTERWAVE_TEST_CARDS = {
    'visa_success': '5531886652142950',
    'mastercard_success': '5438898014560229'
}
```

## Error Handling

```python
from payment_gateway_service import PaymentError, ValidationError

try:
    result = await service.process_payment(transaction, payment_method)
except ValidationError as e:
    print(f"Validation error: {e.message}")
except PaymentError as e:
    print(f"Payment error: {e.message}")
    print(f"Error code: {e.code}")
```

## Monitoring & Logging

### Health Checks

```python
# Check service health
health = await service.health_check()
print(f"Status: {health.status.value}")
print(f"Response time: {health.response_time_ms}ms")
```

### Metrics

```python
# Get service metrics
metrics = await service.get_metrics()
print(f"Total transactions: {metrics['total_transactions']}")
print(f"Success rate: {metrics['success_rate']}")
```

## Production Deployment

### Security Checklist

- [ ] Use environment variables for all secrets
- [ ] Enable webhook signature verification
- [ ] Use HTTPS for all endpoints
- [ ] Implement rate limiting
- [ ] Enable request logging
- [ ] Set up monitoring and alerting
- [ ] Configure backup payment providers
- [ ] Test disaster recovery procedures

### Performance Optimization

- [ ] Enable connection pooling
- [ ] Configure async processing
- [ ] Set up caching for static data
- [ ] Implement request batching where possible
- [ ] Monitor and optimize database queries
- [ ] Set up CDN for static assets

## Troubleshooting

### Common Issues

1. **Webhook Verification Fails**
   - Check webhook secret configuration
   - Verify payload is raw (not parsed)
   - Ensure correct signature header

2. **Mobile Money Payments Fail**
   - Verify phone number format
   - Check provider-specific requirements
   - Ensure sufficient balance for testing

3. **3D Secure Issues**
   - Test with 3DS-enabled test cards
   - Verify redirect URLs are accessible
   - Check SCA compliance settings

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed request logging
service = PaymentGatewayService(debug=True)
```

## Support

For issues and questions:

- 📧 Email: nyimbi@gmail.com
- 🌐 Website: www.datacraft.co.ke
- 📚 Documentation: /docs/
- 🐛 Issues: Create an issue in the repository

## License

© 2025 Datacraft. All rights reserved.