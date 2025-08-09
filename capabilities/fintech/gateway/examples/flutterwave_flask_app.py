"""
Flutterwave Flask Application Example - APG Payment Gateway

Complete Flask application demonstrating Flutterwave integration:
- Payment processing endpoints
- Webhook handling
- Health checks and monitoring
- Error handling and logging
- Card, mobile money, and bank transfer payments
- Real-time payment status updates
- Comprehensive API documentation

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from flask import Flask, request, jsonify, render_template_string
from werkzeug.exceptions import BadRequest

# Import APG payment gateway components
from flutterwave_integration import create_flutterwave_service, FlutterwaveEnvironment
from flutterwave_webhook_handler import create_flutterwave_webhook_handler
from models import PaymentTransaction, PaymentMethod, PaymentMethodType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global service instances
flutterwave_service = None
webhook_handler = None

# Initialize services
async def init_services():
    """Initialize Flutterwave services"""
    global flutterwave_service, webhook_handler
    
    try:
        # Create Flutterwave service
        environment = FlutterwaveEnvironment.SANDBOX if os.getenv("FLUTTERWAVE_ENVIRONMENT", "sandbox") == "sandbox" else FlutterwaveEnvironment.LIVE
        flutterwave_service = await create_flutterwave_service(environment)
        
        # Create webhook handler
        webhook_handler = await create_flutterwave_webhook_handler(flutterwave_service)
        
        logger.info("Flutterwave services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

# Helper function to run async code in Flask
def run_async(coro):
    """Run async coroutine in Flask context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.before_first_request
def initialize():
    """Initialize services before first request"""
    run_async(init_services())

@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flutterwave Payment Gateway - APG</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #fff; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
            .get { background-color: #61affe; }
            .post { background-color: #49cc90; }
            .put { background-color: #fca130; }
            .delete { background-color: #f93e3e; }
            code { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }
            .example { background: #e8f4fd; padding: 10px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Flutterwave Payment Gateway API</h1>
        <p>Complete Flutterwave integration for African payments</p>
        
        <h2>Payment Processing Endpoints</h2>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/process</code>
            <p>Process a payment using various Flutterwave payment methods</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "amount": 1000,
  "currency": "NGN",
  "payment_method": {
    "type": "card",
    "card_number": "4187427415564246",
    "expiry_month": "12",
    "expiry_year": "2025",
    "security_code": "123"
  },
  "customer": {
    "email": "customer@example.com",
    "name": "John Doe",
    "phone": "+2348012345678"
  },
  "description": "Payment for order #12345",
  "redirect_url": "https://yoursite.com/payment/callback"
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/mobile-money</code>
            <p>Process mobile money payments (M-Pesa, MTN, Airtel)</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "amount": 1000,
  "currency": "KES",
  "phone_number": "254708374149",
  "network": "mpesa",
  "customer": {
    "email": "customer@example.com",
    "name": "John Doe"
  },
  "description": "M-Pesa payment"
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/bank-transfer</code>
            <p>Process bank transfer payments</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "amount": 5000,
  "currency": "NGN",
  "bank_code": "044",
  "account_number": "1234567890",
  "customer": {
    "email": "customer@example.com",
    "name": "John Doe"
  },
  "description": "Bank transfer payment"
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/ussd</code>
            <p>Generate USSD payment code</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "amount": 2000,
  "currency": "NGN",
  "bank_code": "058",
  "customer": {
    "email": "customer@example.com",
    "name": "John Doe",
    "phone": "+2348012345678"
  },
  "description": "USSD payment"
}</pre>
            </div>
        </div>
        
        <h2>Payment Management Endpoints</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/payments/verify/{transaction_id}</code>
            <p>Verify payment status</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/refund</code>
            <p>Process payment refund</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "transaction_id": "TXN123456789",
  "amount": 500,
  "reason": "Customer requested refund"
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/cancel/{transaction_id}</code>
            <p>Cancel pending payment</p>
        </div>
        
        <h2>Utility Endpoints</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/payment-methods</code>
            <p>Get supported payment methods</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/balance</code>
            <p>Get account balance</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/health</code>
            <p>Health check endpoint</p>
        </div>
        
        <h2>Webhook Endpoint</h2>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/webhooks/flutterwave</code>
            <p>Flutterwave webhook endpoint for payment notifications</p>
        </div>
        
        <h2>Environment Variables Required</h2>
        <ul>
            <li><code>FLUTTERWAVE_PUBLIC_KEY_SANDBOX</code> or <code>FLUTTERWAVE_PUBLIC_KEY_LIVE</code></li>
            <li><code>FLUTTERWAVE_SECRET_KEY_SANDBOX</code> or <code>FLUTTERWAVE_SECRET_KEY_LIVE</code></li>
            <li><code>FLUTTERWAVE_ENCRYPTION_KEY_SANDBOX</code> or <code>FLUTTERWAVE_ENCRYPTION_KEY_LIVE</code></li>
            <li><code>FLUTTERWAVE_WEBHOOK_SECRET_SANDBOX</code> or <code>FLUTTERWAVE_WEBHOOK_SECRET_LIVE</code></li>
            <li><code>FLUTTERWAVE_ENVIRONMENT</code> (sandbox or live)</li>
        </ul>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/api/payments/process', methods=['POST'])
def process_payment():
    """Process standard payment"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        # Extract payment data
        amount = Decimal(str(data.get('amount', 0)))
        currency = data.get('currency', 'NGN')
        payment_method_data = data.get('payment_method', {})
        customer_data = data.get('customer', {})
        description = data.get('description', '')
        redirect_url = data.get('redirect_url', '')
        
        # Create transaction
        transaction = PaymentTransaction(
            id=f"FLW_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(data)) % 10000:04d}",
            amount=amount,
            currency=currency,
            description=description,
            customer_email=customer_data.get('email'),
            customer_name=customer_data.get('name')
        )
        
        # Create payment method
        payment_method = PaymentMethod(
            method_type=PaymentMethodType.CARD if payment_method_data.get('type') == 'card' else PaymentMethodType.OTHER,
            card_number=payment_method_data.get('card_number'),
            expiry_month=payment_method_data.get('expiry_month'),
            expiry_year=payment_method_data.get('expiry_year'),
            security_code=payment_method_data.get('security_code'),
            metadata={
                'customer_email': customer_data.get('email'),
                'customer_name': customer_data.get('name'),
                'phone_number': customer_data.get('phone'),
                'redirect_url': redirect_url
            }
        )
        
        # Process payment
        result = run_async(flutterwave_service.process_payment(transaction, payment_method))
        
        # Return result
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None,
            'currency': result.currency
        }
        
        if result.auth_url:
            response_data['auth_url'] = result.auth_url
        if result.payment_url:
            response_data['payment_url'] = result.payment_url
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        return jsonify(response_data), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Payment processing failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/payments/mobile-money', methods=['POST'])
def process_mobile_money():
    """Process mobile money payment"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        # Extract mobile money data
        amount = Decimal(str(data.get('amount', 0)))
        currency = data.get('currency', 'KES')
        phone_number = data.get('phone_number')
        network = data.get('network', 'mpesa')
        customer_data = data.get('customer', {})
        description = data.get('description', 'Mobile money payment')
        
        if not phone_number:
            return jsonify({'success': False, 'error': 'Phone number is required'}), 400
        
        # Create transaction
        transaction = PaymentTransaction(
            id=f"FLW_MM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(phone_number) % 10000:04d}",
            amount=amount,
            currency=currency,
            description=description,
            customer_email=customer_data.get('email'),
            customer_name=customer_data.get('name')
        )
        
        # Create mobile money payment method
        payment_method = PaymentMethod(
            method_type=PaymentMethodType.MOBILE_MONEY,
            metadata={
                'mobile_money_type': network,
                'phone_number': phone_number,
                'customer_email': customer_data.get('email'),
                'customer_name': customer_data.get('name'),
                'network': network.upper()
            }
        )
        
        # Process payment
        result = run_async(flutterwave_service.process_payment(transaction, payment_method))
        
        # Return result
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None,
            'currency': result.currency
        }
        
        if result.auth_url:
            response_data['auth_url'] = result.auth_url
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        return jsonify(response_data), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Mobile money payment failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/payments/bank-transfer', methods=['POST'])
def process_bank_transfer():
    """Process bank transfer payment"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        # Extract bank transfer data
        amount = Decimal(str(data.get('amount', 0)))
        currency = data.get('currency', 'NGN')
        bank_code = data.get('bank_code')
        account_number = data.get('account_number')
        customer_data = data.get('customer', {})
        description = data.get('description', 'Bank transfer payment')
        
        if not bank_code or not account_number:
            return jsonify({'success': False, 'error': 'Bank code and account number are required'}), 400
        
        # Create transaction
        transaction = PaymentTransaction(
            id=f"FLW_BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(account_number) % 10000:04d}",
            amount=amount,
            currency=currency,
            description=description,
            customer_email=customer_data.get('email'),
            customer_name=customer_data.get('name')
        )
        
        # Create bank transfer payment method
        payment_method = PaymentMethod(
            method_type=PaymentMethodType.BANK_TRANSFER,
            metadata={
                'bank_code': bank_code,
                'account_number': account_number,
                'customer_email': customer_data.get('email'),
                'customer_name': customer_data.get('name')
            }
        )
        
        # Process payment
        result = run_async(flutterwave_service.process_payment(transaction, payment_method))
        
        # Return result
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None,
            'currency': result.currency
        }
        
        if result.auth_url:
            response_data['auth_url'] = result.auth_url
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        return jsonify(response_data), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Bank transfer payment failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/payments/ussd', methods=['POST'])
def process_ussd():
    """Process USSD payment"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        # Extract USSD data
        amount = Decimal(str(data.get('amount', 0)))
        currency = data.get('currency', 'NGN')
        bank_code = data.get('bank_code', '058')  # Default to GTBank
        customer_data = data.get('customer', {})
        description = data.get('description', 'USSD payment')
        
        # Create transaction
        transaction = PaymentTransaction(
            id=f"FLW_USSD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(data)) % 10000:04d}",
            amount=amount,
            currency=currency,
            description=description,
            customer_email=customer_data.get('email'),
            customer_name=customer_data.get('name')
        )
        
        # Create USSD payment method
        payment_method = PaymentMethod(
            method_type=PaymentMethodType.USSD,
            metadata={
                'bank_code': bank_code,
                'customer_email': customer_data.get('email'),
                'customer_name': customer_data.get('name'),
                'phone_number': customer_data.get('phone')
            }
        )
        
        # Process payment
        result = run_async(flutterwave_service.process_payment(transaction, payment_method))
        
        # Return result
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None,
            'currency': result.currency
        }
        
        if result.ussd_code:
            response_data['ussd_code'] = result.ussd_code
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        return jsonify(response_data), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"USSD payment failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/payments/verify/<transaction_id>', methods=['GET'])
def verify_payment(transaction_id):
    """Verify payment status"""
    try:
        result = run_async(flutterwave_service.verify_payment(transaction_id))
        
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None,
            'currency': result.currency
        }
        
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Payment verification failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/payments/refund', methods=['POST'])
def refund_payment():
    """Process payment refund"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        transaction_id = data.get('transaction_id')
        amount = Decimal(str(data.get('amount'))) if data.get('amount') else None
        reason = data.get('reason')
        
        if not transaction_id:
            return jsonify({'success': False, 'error': 'Transaction ID is required'}), 400
        
        result = run_async(flutterwave_service.refund_payment(transaction_id, amount, reason))
        
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None
        }
        
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        return jsonify(response_data), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Refund processing failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/payments/cancel/<transaction_id>', methods=['POST'])
def cancel_payment(transaction_id):
    """Cancel payment"""
    try:
        data = request.get_json() or {}
        reason = data.get('reason')
        
        result = run_async(flutterwave_service.cancel_payment(transaction_id, reason))
        
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value
        }
        
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        return jsonify(response_data), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Payment cancellation failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/payment-methods', methods=['GET'])
def get_payment_methods():
    """Get supported payment methods"""
    try:
        country_code = request.args.get('country')
        currency = request.args.get('currency')
        
        methods = run_async(flutterwave_service.get_supported_payment_methods(country_code, currency))
        
        return jsonify({
            'success': True,
            'payment_methods': methods
        })
        
    except Exception as e:
        logger.error(f"Failed to get payment methods: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/balance', methods=['GET'])
def get_balance():
    """Get account balance"""
    try:
        currency = request.args.get('currency')
        
        balance = run_async(flutterwave_service.get_account_balance(currency))
        
        return jsonify({
            'success': True,
            'balance': balance
        })
        
    except Exception as e:
        logger.error(f"Failed to get account balance: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health = run_async(flutterwave_service.health_check())
        
        return jsonify({
            'status': health.status.value,
            'response_time_ms': health.response_time_ms,
            'details': health.details
        }), 200 if health.status.value == 'healthy' else 503
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/api/webhooks/flutterwave', methods=['POST'])
def flutterwave_webhook():
    """Handle Flutterwave webhooks"""
    try:
        # Get request data
        payload = request.get_data(as_text=True)
        signature = request.headers.get('x-flw-signature', '')
        
        # Process webhook
        result = run_async(webhook_handler.process_webhook(payload, signature))
        
        return jsonify(result), result.get('status_code', 200)
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/webhooks/stats', methods=['GET'])
def webhook_stats():
    """Get webhook processing statistics"""
    try:
        stats = webhook_handler.get_webhook_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get webhook stats: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 errors"""
    return jsonify({'success': False, 'error': 'Bad request'}), 400

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Set Flask configuration
    app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Start the application
    port = int(os.getenv('PORT', 5000))
    
    logger.info(f"Starting Flutterwave Flask application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])