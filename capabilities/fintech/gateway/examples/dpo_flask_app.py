"""
DPO Flask Application Example - APG Payment Gateway

Complete Flask application demonstrating DPO integration:
- Payment processing endpoints
- Callback handling
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
from dpo_integration import create_dpo_service, DPOEnvironment
from dpo_webhook_handler import create_dpo_webhook_handler
from models import PaymentTransaction, PaymentMethod, PaymentMethodType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global service instances
dpo_service = None
webhook_handler = None

# Initialize services
async def init_services():
    """Initialize DPO services"""
    global dpo_service, webhook_handler
    
    try:
        # Create DPO service
        environment = DPOEnvironment.SANDBOX if os.getenv("DPO_ENVIRONMENT", "sandbox") == "sandbox" else DPOEnvironment.LIVE
        dpo_service = await create_dpo_service(environment)
        
        # Create webhook handler
        webhook_handler = await create_dpo_webhook_handler(dpo_service)
        
        logger.info("DPO services initialized successfully")
        
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
        <title>DPO Payment Gateway - APG</title>
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
        <h1>DPO Payment Gateway API</h1>
        <p>Complete DPO integration for African payments</p>
        
        <h2>Payment Processing Endpoints</h2>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/process</code>
            <p>Process a payment using DPO payment methods</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "amount": 1000,
  "currency": "KES",
  "description": "Payment for order #12345",
  "customer": {
    "email": "customer@example.com",
    "name": "John Doe",
    "phone": "+254700000000",
    "address": "123 Main Street",
    "city": "Nairobi",
    "country": "KE"
  },
  "redirect_url": "https://yoursite.com/payment/success",
  "callback_url": "https://yoursite.com/dpo/callback"
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/mobile-money</code>
            <p>Process mobile money payments (M-Pesa, Airtel Money, MTN)</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "amount": 500,
  "currency": "KES",
  "phone_number": "254700000000",
  "provider": "MPESA",
  "customer": {
    "email": "customer@example.com",
    "name": "Jane Doe",
    "address": "456 Business Ave",
    "city": "Nairobi"
  },
  "description": "M-Pesa payment"
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/card</code>
            <p>Process card payments with 3D Secure</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "amount": 2500,
  "currency": "KES",
  "customer": {
    "email": "cardholder@example.com",
    "name": "David Kimani",
    "address": "789 Card Street",
    "city": "Nairobi",
    "country": "KE"
  },
  "description": "Card payment"
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
  "currency": "KES",
  "customer": {
    "email": "business@example.com",
    "name": "Business Customer Ltd",
    "address": "Corporate Plaza",
    "city": "Nairobi"
  },
  "description": "Bank transfer payment"
}</pre>
            </div>
        </div>
        
        <h2>Payment Management Endpoints</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/payments/verify/{transaction_token}</code>
            <p>Verify payment status using DPO transaction token</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/refund</code>
            <p>Request payment refund (manual processing required)</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "transaction_token": "DPO123456789",
  "amount": 500,
  "reason": "Customer requested refund"
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/cancel/{transaction_token}</code>
            <p>Cancel pending payment</p>
        </div>
        
        <h2>Utility Endpoints</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/payment-methods</code>
            <p>Get supported payment methods</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/fees/calculate</code>
            <p>Calculate transaction fees</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/health</code>
            <p>Health check endpoint</p>
        </div>
        
        <h2>Callback Endpoint</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/callbacks/dpo</code>
            <p>DPO callback endpoint for payment notifications (GET method)</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/callbacks/dpo</code>
            <p>DPO callback endpoint for payment notifications (POST method)</p>
        </div>
        
        <h2>Environment Variables Required</h2>
        <ul>
            <li><code>DPO_COMPANY_TOKEN_SANDBOX</code> or <code>DPO_COMPANY_TOKEN_LIVE</code></li>
            <li><code>DPO_SERVICE_TYPE_SANDBOX</code> or <code>DPO_SERVICE_TYPE_LIVE</code> (optional, defaults to 3854)</li>
            <li><code>DPO_CALLBACK_URL_SANDBOX</code> or <code>DPO_CALLBACK_URL_LIVE</code></li>
            <li><code>DPO_REDIRECT_URL_SANDBOX</code> or <code>DPO_REDIRECT_URL_LIVE</code></li>
            <li><code>DPO_ENVIRONMENT</code> (sandbox or live)</li>
        </ul>
        
        <h2>Payment Flow</h2>
        <ol>
            <li>Submit payment request to <code>/api/payments/process</code></li>
            <li>Redirect customer to returned payment URL</li>
            <li>Customer completes payment on DPO's secure page</li>
            <li>DPO sends callback notification to <code>/api/callbacks/dpo</code></li>
            <li>Verify payment status using <code>/api/payments/verify/{transaction_token}</code></li>
        </ol>
        
        <h2>Supported Countries</h2>
        <p>Kenya, Tanzania, Uganda, Ghana, Nigeria, South Africa, Botswana, Zambia, Malawi, Rwanda, Ethiopia, and more across Africa</p>
        
        <h2>Supported Payment Methods</h2>
        <ul>
            <li><strong>Cards:</strong> Visa, Mastercard, Amex, Diners</li>
            <li><strong>Mobile Money:</strong> M-Pesa, Airtel Money, MTN Mobile Money, Orange Money, Tigo Pesa</li>
            <li><strong>Bank Transfers:</strong> Local and international bank transfers</li>
            <li><strong>Digital Wallets:</strong> PayPal</li>
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
        currency = data.get('currency', 'KES')
        description = data.get('description', 'Payment')
        customer_data = data.get('customer', {})
        redirect_url = data.get('redirect_url', '')
        callback_url = data.get('callback_url', '')
        
        # Create transaction
        transaction = PaymentTransaction(
            id=f"DPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(data)) % 10000:04d}",
            amount=amount,
            currency=currency,
            description=description,
            customer_email=customer_data.get('email'),
            customer_name=customer_data.get('name')
        )
        
        # Create payment method
        payment_method = PaymentMethod(
            method_type=PaymentMethodType.OTHER,
            metadata={
                'customer_email': customer_data.get('email'),
                'customer_name': customer_data.get('name'),
                'phone': customer_data.get('phone'),
                'address': customer_data.get('address', 'Not Provided'),
                'city': customer_data.get('city', 'Not Provided'),
                'country_code': customer_data.get('country', 'KE'),
                'zip_code': customer_data.get('zip_code'),
                'redirect_url': redirect_url,
                'back_url': callback_url
            }
        )
        
        # Process payment
        result = run_async(dpo_service.process_payment(transaction, payment_method))
        
        # Return result
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None,
            'currency': result.currency
        }
        
        if result.payment_url:
            response_data['payment_url'] = result.payment_url
            response_data['redirect_url'] = result.payment_url  # Alias for clarity
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
        provider = data.get('provider', 'MPESA')
        customer_data = data.get('customer', {})
        description = data.get('description', 'Mobile money payment')
        
        if not phone_number:
            return jsonify({'success': False, 'error': 'Phone number is required'}), 400
        
        # Create transaction
        transaction = PaymentTransaction(
            id=f"DPO_MM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(phone_number) % 10000:04d}",
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
                'phone': phone_number,
                'provider': provider,
                'customer_email': customer_data.get('email'),
                'customer_name': customer_data.get('name'),
                'address': customer_data.get('address', 'Not Provided'),
                'city': customer_data.get('city', 'Not Provided'),
                'country_code': 'KE' if provider == 'MPESA' else 'UG'
            }
        )
        
        # Process payment
        result = run_async(dpo_service.process_payment(transaction, payment_method))
        
        # Return result
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None,
            'currency': result.currency
        }
        
        if result.payment_url:
            response_data['payment_url'] = result.payment_url
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        return jsonify(response_data), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Mobile money payment failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/payments/card', methods=['POST'])
def process_card_payment():
    """Process card payment"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        # Extract card payment data
        amount = Decimal(str(data.get('amount', 0)))
        currency = data.get('currency', 'KES')
        customer_data = data.get('customer', {})
        description = data.get('description', 'Card payment')
        
        # Create transaction
        transaction = PaymentTransaction(
            id=f"DPO_CARD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(data)) % 10000:04d}",
            amount=amount,
            currency=currency,
            description=description,
            customer_email=customer_data.get('email'),
            customer_name=customer_data.get('name')
        )
        
        # Create card payment method
        payment_method = PaymentMethod(
            method_type=PaymentMethodType.CARD,
            metadata={
                'customer_email': customer_data.get('email'),
                'customer_name': customer_data.get('name'),
                'address': customer_data.get('address', 'Not Provided'),
                'city': customer_data.get('city', 'Not Provided'),
                'country_code': customer_data.get('country', 'KE'),
                'phone': customer_data.get('phone', '+254700000000')
            }
        )
        
        # Process payment
        result = run_async(dpo_service.process_payment(transaction, payment_method))
        
        # Return result
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None,
            'currency': result.currency
        }
        
        if result.payment_url:
            response_data['payment_url'] = result.payment_url
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        return jsonify(response_data), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Card payment failed: {str(e)}")
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
        currency = data.get('currency', 'KES')
        customer_data = data.get('customer', {})
        description = data.get('description', 'Bank transfer payment')
        
        # Create transaction
        transaction = PaymentTransaction(
            id=f"DPO_BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(data)) % 10000:04d}",
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
                'customer_email': customer_data.get('email'),
                'customer_name': customer_data.get('name'),
                'address': customer_data.get('address', 'Not Provided'),
                'city': customer_data.get('city', 'Not Provided'),
                'country_code': customer_data.get('country', 'KE'),
                'phone': customer_data.get('phone', '+254700000000')
            }
        )
        
        # Process payment
        result = run_async(dpo_service.process_payment(transaction, payment_method))
        
        # Return result
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None,
            'currency': result.currency
        }
        
        if result.payment_url:
            response_data['payment_url'] = result.payment_url
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        return jsonify(response_data), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Bank transfer payment failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/payments/verify/<transaction_token>', methods=['GET'])
def verify_payment(transaction_token):
    """Verify payment status"""
    try:
        result = run_async(dpo_service.verify_payment(transaction_token))
        
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
        
        transaction_token = data.get('transaction_token')
        amount = Decimal(str(data.get('amount'))) if data.get('amount') else None
        reason = data.get('reason')
        
        if not transaction_token:
            return jsonify({'success': False, 'error': 'Transaction token is required'}), 400
        
        result = run_async(dpo_service.refund_payment(transaction_token, amount, reason))
        
        response_data = {
            'success': result.success,
            'transaction_id': result.transaction_id,
            'provider_transaction_id': result.provider_transaction_id,
            'status': result.status.value,
            'amount': str(result.amount) if result.amount else None
        }
        
        if result.error_message:
            response_data['error_message'] = result.error_message
        
        # Add note about manual processing
        response_data['note'] = 'DPO refunds require manual processing through the merchant portal.'
        
        return jsonify(response_data), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Refund processing failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/payments/cancel/<transaction_token>', methods=['POST'])
def cancel_payment(transaction_token):
    """Cancel payment"""
    try:
        data = request.get_json() or {}
        reason = data.get('reason')
        
        result = run_async(dpo_service.cancel_payment(transaction_token, reason))
        
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
        
        methods = run_async(dpo_service.get_supported_payment_methods(country_code, currency))
        
        return jsonify({
            'success': True,
            'payment_methods': methods
        })
        
    except Exception as e:
        logger.error(f"Failed to get payment methods: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/fees/calculate', methods=['POST'])
def calculate_fees():
    """Calculate transaction fees"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        amount = Decimal(str(data.get('amount', 0)))
        currency = data.get('currency', 'KES')
        payment_method = data.get('payment_method', 'VISA')
        
        fees = run_async(dpo_service.get_transaction_fees(amount, currency, payment_method))
        
        return jsonify({
            'success': True,
            'fees': fees
        })
        
    except Exception as e:
        logger.error(f"Fee calculation failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health = run_async(dpo_service.health_check())
        
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

@app.route('/api/callbacks/dpo', methods=['GET', 'POST'])
def dpo_callback():
    """Handle DPO callbacks"""
    try:
        # Get callback data
        if request.method == 'GET':
            callback_data = dict(request.args)
        else:
            callback_data = request.get_json() or dict(request.form)
        
        # Get client IP for validation
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if client_ip:
            client_ip = client_ip.split(',')[0].strip()
        
        # Validate IP if enabled
        if not webhook_handler.validate_callback_ip(client_ip):
            logger.warning(f"Callback from unauthorized IP: {client_ip}")
            return jsonify({
                'success': False,
                'error': 'Unauthorized IP address'
            }), 403
        
        # Process callback
        result = run_async(webhook_handler.process_callback(callback_data))
        
        # Log callback processing
        logger.info(f"Callback processed: {result}")
        
        return jsonify(result), result.get('status_code', 200)
        
    except Exception as e:
        logger.error(f"Callback processing failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/callbacks/stats', methods=['GET'])
def callback_stats():
    """Get callback processing statistics"""
    try:
        stats = webhook_handler.get_callback_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get callback stats: {str(e)}")
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
    
    logger.info(f"Starting DPO Flask application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])