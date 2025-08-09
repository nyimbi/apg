# Payment Gateway Test Suite

Comprehensive test suite for the APG Payment Gateway with real data validation and integration testing.

## Overview

This test suite provides complete coverage of the payment gateway system, including:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interaction testing  
- **API Tests**: RESTful API endpoint testing
- **Database Tests**: Data persistence and retrieval testing
- **Authentication Tests**: Security and authorization testing
- **Processor Tests**: Payment processor integration testing

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and configuration
├── pytest.ini                 # Pytest configuration
├── test_database.py           # Database service tests
├── test_auth.py               # Authentication service tests
├── test_payment_service.py    # Payment gateway service tests
├── test_processors.py         # Payment processor tests
├── test_api.py                # API endpoint tests
├── test_integration.py        # End-to-end integration tests
└── README.md                  # This file
```

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Ensure you have the required packages
pip install uuid-extensions sqlalchemy asyncpg bcrypt
```

### Run All Tests

```bash
# From the payment gateway directory
pytest tests/

# With verbose output
pytest -v tests/

# With coverage report
pytest --cov=. --cov-report=html tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit tests/

# Integration tests only  
pytest -m integration tests/

# API tests only
pytest -m api tests/

# Database tests only
pytest tests/test_database.py

# Authentication tests only
pytest tests/test_auth.py
```

### Run Tests with Real Data

```bash
# Run with real database connections (requires setup)
pytest --real-db tests/

# Run with network access for processor tests
pytest -m "not requires_network" tests/
```

## Test Configuration

### Environment Variables

Set these environment variables for comprehensive testing:

```bash
export TEST_DATABASE_URL="postgresql://user:pass@localhost/test_db"
export TEST_MPESA_CONSUMER_KEY="your_test_key"
export TEST_MPESA_CONSUMER_SECRET="your_test_secret"
export TEST_STRIPE_API_KEY="sk_test_your_key"
export TEST_PAYPAL_CLIENT_ID="your_test_client_id"
```

### Test Markers

Use pytest markers to categorize and filter tests:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.api` - API tests
- `@pytest.mark.database` - Database tests
- `@pytest.mark.auth` - Authentication tests
- `@pytest.mark.processors` - Payment processor tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.requires_network` - Tests requiring network access

## Test Data

### Fixtures

The test suite includes comprehensive fixtures for:

- **Temporary Database**: In-memory SQLite database for testing
- **Authentication Service**: Mock authentication service
- **Payment Service**: Configured payment gateway service
- **Sample Transactions**: Realistic payment transaction data
- **Sample Payment Methods**: Various payment method types
- **Test Merchants**: Sample merchant data
- **API Keys**: Test authentication keys

### Real Data Simulation

Tests use realistic data patterns:

```python
# Sample MPESA transaction
{
    "amount": 10000,  # 100.00 KES
    "currency": "KES",
    "payment_method": {
        "type": "mpesa",
        "phone_number": "+254712345678"
    },
    "merchant_id": "test_merchant_123",
    "customer_id": "test_customer_456"
}

# Sample credit card transaction  
{
    "amount": 5000,   # $50.00
    "currency": "USD", 
    "payment_method": {
        "type": "credit_card",
        "last4": "4242",
        "brand": "visa",
        "token": "pm_test_card"
    }
}
```

## Key Test Scenarios

### Database Tests
- Transaction creation and retrieval
- Payment method storage
- Merchant analytics calculation
- Concurrent database operations
- Transaction status updates

### Authentication Tests
- API key creation and validation
- JWT token generation and verification
- Password hashing and verification
- Role-based permissions
- API key expiration and revocation

### Payment Service Tests
- MPESA payment processing
- Credit card payment processing
- Payment capture and refund operations
- Fraud detection integration
- Webhook processing
- Multi-currency support

### Processor Tests
- MPESA STK Push integration
- Stripe Payment Intents API
- PayPal Orders API
- Adyen Checkout API
- Processor health monitoring
- Concurrent processor operations

### API Tests
- Payment processing endpoints
- Authentication endpoints
- Analytics endpoints
- Health check endpoints
- Error handling scenarios
- Request validation

### Integration Tests
- End-to-end payment flows
- Multi-processor integration
- Fraud detection workflows
- Webhook processing flows
- System load testing
- Recovery and resilience testing

## Test Patterns

### Async Testing
```python
async def test_async_operation(payment_service):
    result = await payment_service.process_payment(transaction, method)
    assert result.success is True
```

### Mock Usage
```python
@pytest.fixture
def mock_processor():
    processor = AsyncMock()
    processor.process_payment.return_value = PaymentResult(success=True)
    return processor
```

### Error Testing
```python
async def test_error_handling(payment_service):
    with pytest.raises(ValidationError):
        await payment_service.process_payment(invalid_transaction)
```

### Concurrent Testing
```python
async def test_concurrent_operations():
    tasks = [process_payment(data) for data in test_data]
    results = await asyncio.gather(*tasks)
    assert all(result.success for result in results)
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test Payment Gateway
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.12
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run tests
        run: pytest --cov=. --cov-report=xml tests/
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## Performance Benchmarks

The test suite includes performance benchmarks:

- **Database Operations**: < 50ms per transaction
- **Payment Processing**: < 2000ms per payment
- **API Response Times**: < 500ms per request
- **Concurrent Load**: 100 requests/second
- **Memory Usage**: < 512MB under load

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Ensure PostgreSQL is running for integration tests
   - Check connection strings and credentials

2. **Async Test Failures**
   - Verify pytest-asyncio is installed
   - Check event loop configuration

3. **Mock Assertion Errors**
   - Verify mock setup matches actual implementation
   - Check async/sync mock usage

4. **Network Timeout Errors**
   - Skip network tests with `--no-network` flag
   - Increase timeout values for slow connections

### Debug Mode

```bash
# Run tests with debug output
pytest -s --log-cli-level=DEBUG tests/

# Run single test with full output
pytest -vvv -s tests/test_integration.py::TestFullIntegration::test_complete_mpesa_payment_flow
```

## Contributing

When adding new tests:

1. Follow existing naming conventions
2. Add appropriate test markers
3. Include realistic test data
4. Test both success and failure scenarios
5. Add integration tests for new features
6. Update this README with new test categories

## Coverage Goals

- **Unit Tests**: > 95% code coverage
- **Integration Tests**: All major workflows
- **API Tests**: All endpoints and error cases
- **Database Tests**: All CRUD operations
- **Authentication Tests**: All security scenarios
- **Processor Tests**: All payment methods

Current coverage: **Target 95%+ across all modules**