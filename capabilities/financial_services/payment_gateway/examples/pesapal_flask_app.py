"""
Pesapal Flask Application Example - APG Payment Gateway

Complete Flask application demonstrating Pesapal integration:
- Payment processing endpoints
- IPN (Instant Payment Notification) handling
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
from pesapal_integration import create_pesapal_service, PesapalEnvironment
from pesapal_webhook_handler import create_pesapal_webhook_handler
from models import PaymentTransaction, PaymentMethod, PaymentMethodType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global service instances
pesapal_service = None
webhook_handler = None

# Initialize services
async def init_services():
    """Initialize Pesapal services"""
    global pesapal_service, webhook_handler
    
    try:
        # Create Pesapal service
        environment = PesapalEnvironment.SANDBOX if os.getenv("PESAPAL_ENVIRONMENT", "sandbox") == "sandbox" else PesapalEnvironment.LIVE
        pesapal_service = await create_pesapal_service(environment)
        
        # Create webhook handler
        webhook_handler = await create_pesapal_webhook_handler(pesapal_service)
        
        logger.info("Pesapal services initialized successfully")
        
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
        <title>Pesapal Payment Gateway - APG</title>
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
        <h1>Pesapal Payment Gateway API</h1>
        <p>Complete Pesapal integration for East African payments</p>
        
        <h2>Payment Processing Endpoints</h2>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/process</code>
            <p>Process a payment using Pesapal payment methods</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "amount": 1000,
  "currency": "KES",
  "description": "Payment for order #12345",
  "customer": {
    "email": "customer@example.com",
    "name": "John Doe",
    "phone": "+254700000000"
  },
  "callback_url": "https://yoursite.com/payment/callback",
  "billing_address": {
    "country_code": "KE",
    "city": "Nairobi",
    "line_1": "123 Main Street"
  }
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/payments/mobile-money</code>
            <p>Process mobile money payments (M-Pesa, Airtel Money)</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "amount": 500,
  "currency": "KES",
  "phone_number": "254700000000",
  "provider": "MPESA",
  "customer": {
    "email": "customer@example.com",
    "name": "Jane Doe"
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
  "currency": "KES",
  "customer": {
    "email": "customer@example.com",
    "name": "Business Customer"
  },
  "description": "Bank transfer payment"
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
            <p>Request payment refund (manual processing required)</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "transaction_id": "PSP123456789",
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
            <span class="method get">GET</span> <code>/api/health</code>
            <p>Health check endpoint</p>
        </div>
        
        <h2>IPN (Instant Payment Notification) Endpoint</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/ipn/pesapal</code>
            <p>Pesapal IPN endpoint for payment notifications</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/ipn/pesapal</code>
            <p>Alternative IPN endpoint (POST method)</p>
        </div>
        
        <h2>Environment Variables Required</h2>
        <ul>
            <li><code>PESAPAL_CONSUMER_KEY_SANDBOX</code> or <code>PESAPAL_CONSUMER_KEY_LIVE</code></li>
            <li><code>PESAPAL_CONSUMER_SECRET_SANDBOX</code> or <code>PESAPAL_CONSUMER_SECRET_LIVE</code></li>
            <li><code>PESAPAL_CALLBACK_URL_SANDBOX</code> or <code>PESAPAL_CALLBACK_URL_LIVE</code></li>
            <li><code>PESAPAL_ENVIRONMENT</code> (sandbox or live)</li>
        </ul>
        
        <h2>Payment Flow</h2>
        <ol>
            <li>Submit payment request to <code>/api/payments/process</code></li>
            <li>Redirect customer to returned payment URL</li>
            <li>Customer completes payment on Pesapal</li>
            <li>Pesapal sends IPN notification to <code>/api/ipn/pesapal</code></li>
            <li>Verify payment status using <code>/api/payments/verify/{transaction_id}</code></li>
        </ol>
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
        callback_url = data.get('callback_url', '')
        billing_address = data.get('billing_address', {})
        
        # Create transaction
        transaction = PaymentTransaction(
            id=f"PSP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(data)) % 10000:04d}",
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
                'phone_number': customer_data.get('phone'),
                'callback_url': callback_url,
                'country_code': billing_address.get('country_code', 'KE'),
                'city': billing_address.get('city', ''),
                'address_line_1': billing_address.get('line_1', ''),
                'address_line_2': billing_address.get('line_2', ''),
                'state': billing_address.get('state', ''),
                'postal_code': billing_address.get('postal_code', ''),
                'zip_code': billing_address.get('zip_code', '')
            }
        )
        
        # Process payment
        result = run_async(pesapal_service.process_payment(transaction, payment_method))
        
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
            id=f"PSP_MM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(phone_number) % 10000:04d}",
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
                'phone_number': phone_number,
                'provider': provider,
                'customer_email': customer_data.get('email'),
                'customer_name': customer_data.get('name'),
                'country_code': 'KE' if provider == 'MPESA' else 'UG'
            }
        )
        
        # Process payment
        result = run_async(pesapal_service.process_payment(transaction, payment_method))
        
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
            id=f"PSP_BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(data)) % 10000:04d}",
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
                'country_code': 'KE'
            }
        )
        
        # Process payment
        result = run_async(pesapal_service.process_payment(transaction, payment_method))
        
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

@app.route('/api/payments/verify/<transaction_id>', methods=['GET'])
def verify_payment(transaction_id):
    """Verify payment status"""
    try:
        result = run_async(pesapal_service.verify_payment(transaction_id))
        
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
        
        result = run_async(pesapal_service.refund_payment(transaction_id, amount, reason))
        
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
        response_data['note'] = 'Pesapal refunds require manual processing. Contact Pesapal support to complete the refund.'
        
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
        
        result = run_async(pesapal_service.cancel_payment(transaction_id, reason))
        
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
        
        methods = run_async(pesapal_service.get_supported_payment_methods(country_code, currency))
        
        return jsonify({
            'success': True,
            'payment_methods': methods
        })
        
    except Exception as e:
        logger.error(f"Failed to get payment methods: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health = run_async(pesapal_service.health_check())
        
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

@app.route('/api/ipn/pesapal', methods=['GET', 'POST'])
def pesapal_ipn():
    """Handle Pesapal IPN (Instant Payment Notification)"""
    try:
        # Get IPN data
        if request.method == 'GET':
            ipn_data = dict(request.args)
        else:
            ipn_data = request.get_json() or {}
        
        # Get signature if provided
        signature = request.headers.get('x-pesapal-signature')
        
        # Process IPN
        result = run_async(webhook_handler.process_ipn(ipn_data, signature))
        
        # Log IPN processing
        logger.info(f"IPN processed: {result}")
        
        return jsonify(result), result.get('status_code', 200)
        
    except Exception as e:
        logger.error(f"IPN processing failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ipn/stats', methods=['GET'])
def ipn_stats():
    """Get IPN processing statistics"""
    try:
        stats = webhook_handler.get_ipn_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get IPN stats: {str(e)}")
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
        
        fees = run_async(pesapal_service.get_transaction_fees(amount, currency, payment_method))
        
        return jsonify({
            'success': True,
            'fees': fees
        })
        
    except Exception as e:
        logger.error(f"Fee calculation failed: {str(e)}")
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
    
    logger.info(f"Starting Pesapal Flask application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])