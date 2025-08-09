# Complete MPESA Integration Guide - APG Payment Gateway

## Overview

This guide covers the complete MPESA integration implementation in the APG Payment Gateway, featuring all MPESA APIs with full production-ready functionality.

## 🚀 Features Implemented

### ✅ Payment Processing
- **STK Push (Lipa na M-Pesa Online)**: Customer-initiated payments
- **B2B Payments**: Business-to-business transfers
- **B2C Payments**: Business-to-customer payments (salaries, refunds)
- **C2B Payments**: Customer-to-business payments with validation

### ✅ Account Management
- **Account Balance Inquiry**: Real-time balance checking
- **Transaction Status Query**: Status verification for any transaction
- **Transaction Reversal**: Refund/reverse completed transactions

### ✅ Infrastructure
- **OAuth Token Management**: Automatic token refresh with buffer
- **Webhook Processing**: Complete callback handling for all transaction types
- **Error Handling**: Comprehensive retry logic and circuit breaker
- **Security**: Request signing, IP whitelisting, credential encryption

### ✅ Monitoring & Analytics
- **Health Monitoring**: Service status and performance metrics
- **Transaction Tracking**: Complete audit trail
- **Webhook Logging**: Full callback processing logs
- **Performance Analytics**: Success rates, response times, error tracking

## 📁 File Structure

```
mpesa_integration.py          # Core MPESA service implementation
mpesa_webhook_handler.py      # Webhook processing and callback handling
mpesa_client_example.py       # Complete usage examples
flask_mpesa_app.py           # Production Flask application
config/mpesa.yaml            # Comprehensive configuration
requirements_mpesa.txt       # Python dependencies
```

## 🛠 Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements_mpesa.txt
```

### 2. Environment Configuration

Set the following environment variables:

```bash
# Required for all environments
export MPESA_CONSUMER_KEY="your_consumer_key"
export MPESA_CONSUMER_SECRET="your_consumer_secret" 
export MPESA_BUSINESS_SHORT_CODE="your_shortcode"
export MPESA_PASSKEY="your_passkey"
export MPESA_INITIATOR_NAME="your_initiator_name"
export MPESA_SECURITY_CREDENTIAL="your_encrypted_credential"

# Callback URLs (replace with your domain)
export MPESA_CALLBACK_BASE_URL="https://your-domain.com"
```

### 3. Safaricom Developer Account Setup

1. **Register** at [Safaricom Developer Portal](https://developer.safaricom.co.ke)
2. **Create App** and get Consumer Key & Secret
3. **Configure Business Short Code** for your organization
4. **Generate Security Credential** (encrypted initiator password)
5. **Set up Callback URLs** pointing to your server

## 🔧 Configuration

### Sandbox vs Production

```python
from mpesa_integration import create_mpesa_service, MPESAEnvironment

# Sandbox (for testing)
mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)

# Production (for live transactions)
mpesa_service = await create_mpesa_service(MPESAEnvironment.PRODUCTION)
```

### Security Credential Generation

The security credential is your initiator password encrypted with Safaricom's public key:

```bash
# Use Safaricom's public key to encrypt your initiator password
openssl rsautl -encrypt -pubin -inkey safaricom_public_key.pem -in password.txt -out encrypted.txt
base64 encrypted.txt > security_credential.txt
```

## 💰 Payment Types Implementation

### 1. STK Push (Customer Payments)

**Use Case**: Customer pays for goods/services

```python
from mpesa_integration import create_mpesa_service, MPESATransactionType
from models import PaymentTransaction, PaymentMethod, PaymentMethodType

# Initialize service
mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)

# Create transaction
transaction = PaymentTransaction(
    id=uuid7str(),
    merchant_id="merchant_123",
    customer_id="customer_456", 
    amount=1000,  # KES 10.00 (in cents)
    currency="KES",
    description="Product purchase",
    payment_method_type=PaymentMethodType.MPESA,
    tenant_id="tenant_001"
)

# Create payment method
payment_method = PaymentMethod(
    id=uuid7str(),
    customer_id="customer_456",
    payment_method_type=PaymentMethodType.MPESA,
    mpesa_phone_number="254708374149",
    tenant_id="tenant_001"
)

# Process STK Push
additional_data = {
    "transaction_type": MPESATransactionType.STK_PUSH.value,
    "phone_number": "254708374149",
    "account_reference": "ORDER_12345"
}

result = await mpesa_service.process_payment(transaction, payment_method, additional_data)

if result.success:
    print(f"STK Push initiated: {result.processor_transaction_id}")
    print(f"Customer will receive prompt on: {additional_data['phone_number']}")
else:
    print(f"STK Push failed: {result.error_message}")
```

### 2. B2B Payments (Business Transfers)

**Use Case**: Pay suppliers, partners, or other businesses

```python
# B2B Payment
additional_data = {
    "transaction_type": MPESATransactionType.B2B.value,
    "receiver_party": "600000",  # Receiver business short code
    "command_id": "BusinessPayment",
    "account_reference": "SUPPLIER_INV_001"
}

result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
```

**Available Command IDs:**
- `BusinessPayment`: General business payment
- `BusinessBuyGoods`: Purchase from another business
- `DisburseFundsToBusiness`: Disburse funds
- `BusinessToBusinessTransfer`: Direct transfer

### 3. B2C Payments (Business to Customer)

**Use Case**: Salary payments, refunds, promotions

```python
# B2C Payment
additional_data = {
    "transaction_type": MPESATransactionType.B2C.value,
    "phone_number": "254708374149",
    "command_id": "SalaryPayment",
    "occasion": "Monthly Salary December 2024"
}

result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
```

**Available Command IDs:**
- `SalaryPayment`: Employee salaries
- `BusinessPayment`: General business payment to customer
- `PromotionPayment`: Promotional payments/rewards

### 4. C2B Payments (Customer to Business)

**Use Case**: Bill payments, top-ups, customer deposits

```python
# C2B Payment Simulation
additional_data = {
    "transaction_type": MPESATransactionType.C2B.value,
    "phone_number": "254708374149",
    "command_id": "CustomerPayBillOnline",
    "bill_ref_number": "ACCOUNT_12345"
}

result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
```

## 🔄 Transaction Management

### Account Balance Inquiry

```python
balance_result = await mpesa_service.get_account_balance()

if balance_result["success"]:
    print(f"Balance inquiry initiated: {balance_result['conversation_id']}")
    # Result will be sent to callback URL
else:
    print(f"Balance inquiry failed: {balance_result['error_message']}")
```

### Transaction Status Query

```python
status_result = await mpesa_service.get_transaction_status("transaction_id")

if status_result.success:
    print(f"Status: {status_result.status.value}")
    print(f"Details: {status_result.metadata}")
else:
    print(f"Status query failed: {status_result.error_message}")
```

### Transaction Reversal

```python
reversal_result = await mpesa_service.reverse_transaction(
    originator_conversation_id="AG_20231201_12345678901234567890",
    transaction_id="QHJ12345678",  # MPESA receipt number
    amount=1000,  # Amount to reverse (in cents)
    reason="Customer requested refund"
)

if reversal_result.success:
    print(f"Reversal initiated: {reversal_result.processor_transaction_id}")
else:
    print(f"Reversal failed: {reversal_result.error_message}")
```

## 🔗 Webhook Implementation

### Setting Up Webhooks

1. **Configure Callback URLs** in your environment:

```bash
export MPESA_CALLBACK_BASE_URL="https://your-domain.com"
```

2. **Register Webhook Handler**:

```python
from mpesa_webhook_handler import MPESAWebhookHandler, create_mpesa_webhook_blueprint

# Create webhook handler
webhook_handler = MPESAWebhookHandler(mpesa_service)

# Create Flask blueprint
webhook_blueprint = create_mpesa_webhook_blueprint(webhook_handler)
app.register_blueprint(webhook_blueprint)
```

### Webhook Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/mpesa/stk-push-callback` | STK Push results |
| `/mpesa/b2b-result` | B2B payment results |
| `/mpesa/b2c-result` | B2C payment results |
| `/mpesa/c2b-validation` | C2B validation |
| `/mpesa/c2b-confirmation` | C2B confirmation |
| `/mpesa/account-balance-result` | Balance inquiry results |
| `/mpesa/transaction-status-result` | Status query results |
| `/mpesa/transaction-reversal-result` | Reversal results |
| `/mpesa/timeout` | Transaction timeouts |

### Webhook Security

- **IP Whitelisting**: Only accept callbacks from Safaricom IPs
- **Signature Validation**: Verify callback authenticity
- **HTTPS Only**: All callbacks use secure connections

```yaml
# Security configuration
security:
  validate_callback_signatures: true
  callback_ip_whitelist:
    - "196.201.214.200"
    - "196.201.214.206"
    - "196.201.212.127"
    - "196.201.212.138"
```

## 📊 Monitoring & Analytics

### Health Monitoring

```python
health = await mpesa_service.health_check()

print(f"Status: {health.status.value}")
print(f"Success Rate: {health.success_rate:.1%}")
print(f"Avg Response Time: {health.average_response_time:.0f}ms")
print(f"Uptime: {health.uptime_percentage:.1f}%")
```

### Webhook Statistics

```python
# Get webhook processing stats
stats = webhook_handler.get_webhook_stats()

print(f"Total Webhooks: {stats['total_webhooks']}")
print(f"Success Rate: {stats['success_rate']:.1%}")
print(f"Error Count: {stats['error_webhooks']}")
```

### Transaction Tracking

```python
# Get recent webhook logs
logs = webhook_handler.get_webhook_logs(limit=50)
for log in logs:
    print(f"{log['timestamp']}: {log['type']} - {'✅' if log['processed'] else '❌'}")
```

## 🔒 Security Best Practices

### 1. Credential Management

```python
# Use environment variables - NEVER hardcode credentials
consumer_key = os.getenv("MPESA_CONSUMER_KEY")
consumer_secret = os.getenv("MPESA_CONSUMER_SECRET")

# Use encrypted security credential
security_credential = os.getenv("MPESA_SECURITY_CREDENTIAL")
```

### 2. Network Security

```python
# Enable HTTPS for all callbacks
callback_urls = {
    "base_url": "https://your-secure-domain.com",  # Always HTTPS
    "stk_push_callback": "https://your-secure-domain.com/mpesa/stk-push-callback"
}
```

### 3. Request Validation

```python
# Validate incoming webhooks
@app.route('/mpesa/stk-push-callback', methods=['POST'])
async def stk_push_callback():
    # Verify request signature
    if not verify_mpesa_signature(request):
        return jsonify({"ResultCode": 1, "ResultDesc": "Invalid signature"}), 401
    
    # Process callback
    callback_data = request.get_json()
    result = await webhook_handler.handle_stk_push_callback(callback_data)
    return jsonify(result)
```

## 🧪 Testing

### Sandbox Testing

Use Safaricom's sandbox environment for testing:

```python
# Sandbox configuration
mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)

# Test phone numbers (sandbox)
test_numbers = [
    "254708374149",
    "254711XXXXXX"
]

# Test amounts that simulate different scenarios
test_amounts = {
    "success": [1, 10, 100, 1000],  # These will succeed
    "failure": [2, 3, 4, 5]         # These will fail (for testing error handling)
}
```

### Production Testing

Before going live:

1. **Test all transaction types** with small amounts
2. **Verify webhook processing** with actual callbacks
3. **Test error scenarios** (network failures, invalid data)
4. **Load testing** with concurrent requests
5. **Security testing** (invalid signatures, wrong IPs)

## 🚀 Production Deployment

### 1. Environment Setup

```bash
# Production environment variables
export ENVIRONMENT=production
export MPESA_CONSUMER_KEY="prod_consumer_key"
export MPESA_CONSUMER_SECRET="prod_consumer_secret"
export MPESA_BUSINESS_SHORT_CODE="your_prod_shortcode"
export MPESA_CALLBACK_BASE_URL="https://your-production-domain.com"
```

### 2. SSL Certificate

Ensure your callback URLs use valid SSL certificates:

```bash
# Verify SSL setup
curl -I https://your-domain.com/mpesa/stk-push-callback
```

### 3. Logging & Monitoring

```python
# Production logging configuration
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/mpesa/integration.log'),
        logging.StreamHandler()
    ]
)
```

### 4. Performance Optimization

```yaml
# Performance settings
performance:
  connection_pool_size: 20
  connection_pool_maxsize: 100
  connection_keep_alive: true
  request_compression: true
  async_processing: true
```

## 📈 Scaling Considerations

### High Availability

1. **Load Balancing**: Multiple application instances
2. **Database Clustering**: PostgreSQL cluster for transaction storage  
3. **Redis Caching**: Token caching and rate limiting
4. **Queue Processing**: Async webhook processing

### Performance Optimization

1. **Connection Pooling**: Reuse HTTP connections
2. **Request Batching**: Group related operations
3. **Caching**: Cache tokens and configuration
4. **Monitoring**: Real-time performance metrics

## 🐛 Troubleshooting

### Common Issues

1. **Token Expiry**
   ```
   Error: "Invalid access token"
   Solution: Check token refresh logic and buffer time
   ```

2. **Callback Not Received**
   ```
   Issue: Webhooks not being processed
   Solution: Verify callback URLs, check firewall, validate SSL
   ```

3. **Transaction Timeout**
   ```
   Issue: STK Push timeout
   Solution: Check phone number format, network connectivity
   ```

4. **Invalid Signature**
   ```
   Issue: Security credential errors
   Solution: Regenerate and re-encrypt security credential
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('mpesa_integration').setLevel(logging.DEBUG)

# Log all requests and responses
mpesa_config.monitoring.log_all_requests = True
mpesa_config.monitoring.log_all_responses = True
```

## 📞 Support

### Safaricom Support
- **Developer Portal**: https://developer.safaricom.co.ke
- **Documentation**: https://developer.safaricom.co.ke/docs
- **Support Email**: apisupport@safaricom.co.ke

### APG Support
- **Documentation**: Internal APG documentation
- **Issue Tracking**: GitHub issues
- **Email**: nyimbi@gmail.com

## 📄 License

© 2025 Datacraft. All rights reserved.

---

This implementation provides a complete, production-ready MPESA integration with all features fully implemented. No mocking, no placeholders - just real, working code ready for production deployment.