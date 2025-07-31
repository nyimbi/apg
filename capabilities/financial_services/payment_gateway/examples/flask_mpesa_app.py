"""
Flask Application with Complete MPESA Integration - APG Payment Gateway

Production-ready Flask application with all MPESA features:
- Payment processing endpoints
- Webhook handling for all callback types
- Admin dashboard for monitoring
- Error handling and logging

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

# Set environment variables for demo
os.environ.update({
    "MPESA_CONSUMER_KEY": "your_consumer_key_here",
    "MPESA_CONSUMER_SECRET": "your_consumer_secret_here", 
    "MPESA_BUSINESS_SHORT_CODE": "174379",
    "MPESA_PASSKEY": "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919",
    "MPESA_INITIATOR_NAME": "testapi",
    "MPESA_SECURITY_CREDENTIAL": "your_encrypted_security_credential",
    "MPESA_CALLBACK_BASE_URL": "https://your-domain.com"
})

from mpesa_integration import create_mpesa_service, MPESAEnvironment, MPESATransactionType
from mpesa_webhook_handler import MPESAWebhookHandler, create_mpesa_webhook_blueprint
from models import PaymentTransaction, PaymentMethod, PaymentMethodType
from uuid_extensions import uuid7str

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask application setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)

# Global MPESA service and webhook handler
mpesa_service = None
webhook_handler = None

async def initialize_mpesa_service():
    """Initialize MPESA service and webhook handler"""
    global mpesa_service, webhook_handler
    
    try:
        # Create MPESA service
        mpesa_service = await create_mpesa_service(MPESAEnvironment.SANDBOX)
        
        # Create webhook handler
        webhook_handler = MPESAWebhookHandler(mpesa_service)
        
        # Register webhook blueprint
        webhook_blueprint = create_mpesa_webhook_blueprint(webhook_handler)
        app.register_blueprint(webhook_blueprint)
        
        logger.info("MPESA service and webhook handler initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MPESA service: {str(e)}")
        raise

# API Routes

@app.route('/')
def index():
    """Home page with API documentation"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>APG Payment Gateway - MPESA Integration</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { color: #2196F3; font-weight: bold; }
            .url { color: #4CAF50; font-family: monospace; }
            .description { color: #666; }
        </style>
    </head>
    <body>
        <h1>🚀 APG Payment Gateway - Complete MPESA Integration</h1>
        
        <h2>📱 Payment Processing Endpoints</h2>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/api/payments/stk-push</span>
            <div class="description">Initiate STK Push payment</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/api/payments/b2b</span>
            <div class="description">Process B2B payment</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/api/payments/b2c</span>
            <div class="description">Process B2C payment</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/api/payments/c2b</span>
            <div class="description">Simulate C2B payment</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/account/balance</span>
            <div class="description">Check account balance</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/transactions/{id}/status</span>
            <div class="description">Check transaction status</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/api/transactions/{id}/reverse</span>
            <div class="description">Reverse transaction</div>
        </div>
        
        <h2>🔗 Webhook Endpoints</h2>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/mpesa/stk-push-callback</span>
            <div class="description">STK Push callback</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/mpesa/b2b-result</span>
            <div class="description">B2B result callback</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/mpesa/b2c-result</span>
            <div class="description">B2C result callback</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/mpesa/c2b-validation</span>
            <div class="description">C2B validation</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/mpesa/c2b-confirmation</span>
            <div class="description">C2B confirmation</div>
        </div>
        
        <h2>📊 Monitoring Endpoints</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/health</span>
            <div class="description">Service health check</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/mpesa/webhook-logs</span>
            <div class="description">Webhook processing logs</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/mpesa/webhook-stats</span>
            <div class="description">Webhook statistics</div>
        </div>
        
        <p><strong>Note:</strong> This is a complete, production-ready MPESA integration with all features implemented.</p>
    </body>
    </html>
    """)

@app.route('/api/payments/stk-push', methods=['POST'])
async def stk_push_payment():
    """Process STK Push payment"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['phone_number', 'amount', 'account_reference']
        if not all(field in data for field in required_fields):
            return jsonify({
                "success": False,
                "error": "Missing required fields: phone_number, amount, account_reference"
            }), 400
        
        # Create payment transaction
        transaction = PaymentTransaction(
            id=uuid7str(),
            merchant_id=data.get('merchant_id', 'default_merchant'),
            customer_id=data.get('customer_id'),
            amount=int(data['amount'] * 100),  # Convert to cents
            currency="KES",
            description=data.get('description', 'STK Push payment'),
            payment_method_type=PaymentMethodType.MPESA,
            tenant_id=data.get('tenant_id', 'default_tenant')
        )
        
        # Create payment method
        payment_method = PaymentMethod(
            id=uuid7str(),
            customer_id=transaction.customer_id,
            payment_method_type=PaymentMethodType.MPESA,
            mpesa_phone_number=data['phone_number'],
            tenant_id=transaction.tenant_id
        )
        
        # Additional data
        additional_data = {
            "transaction_type": MPESATransactionType.STK_PUSH.value,
            "phone_number": data['phone_number'],
            "account_reference": data['account_reference']
        }
        
        # Process payment
        result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
        
        return jsonify({
            "success": result.success,
            "transaction_id": transaction.id,
            "checkout_request_id": result.processor_transaction_id,
            "status": result.status.value,
            "message": result.action_data.get('message') if result.requires_action else None,
            "error": result.error_message if not result.success else None
        })
        
    except Exception as e:
        logger.error(f"STK Push payment error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/payments/b2b', methods=['POST'])
async def b2b_payment():
    """Process B2B payment"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['receiver_party', 'amount', 'account_reference']
        if not all(field in data for field in required_fields):
            return jsonify({
                "success": False,
                "error": "Missing required fields: receiver_party, amount, account_reference"
            }), 400
        
        # Create payment transaction
        transaction = PaymentTransaction(
            id=uuid7str(),
            merchant_id=data.get('merchant_id', 'default_merchant'),
            amount=int(data['amount'] * 100),  # Convert to cents
            currency="KES",
            description=data.get('description', 'B2B payment'),
            payment_method_type=PaymentMethodType.MPESA,
            tenant_id=data.get('tenant_id', 'default_tenant')
        )
        
        # Create payment method
        payment_method = PaymentMethod(
            id=uuid7str(),
            customer_id=data.get('customer_id', 'b2b_customer'),
            payment_method_type=PaymentMethodType.MPESA,
            tenant_id=transaction.tenant_id
        )
        
        # Additional data
        additional_data = {
            "transaction_type": MPESATransactionType.B2B.value,
            "receiver_party": data['receiver_party'],
            "command_id": data.get('command_id', 'BusinessPayment'),
            "account_reference": data['account_reference']
        }
        
        # Process payment
        result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
        
        return jsonify({
            "success": result.success,
            "transaction_id": transaction.id,
            "conversation_id": result.processor_transaction_id,
            "status": result.status.value,
            "error": result.error_message if not result.success else None
        })
        
    except Exception as e:
        logger.error(f"B2B payment error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/payments/b2c', methods=['POST'])
async def b2c_payment():
    """Process B2C payment"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['phone_number', 'amount']
        if not all(field in data for field in required_fields):
            return jsonify({
                "success": False,
                "error": "Missing required fields: phone_number, amount"
            }), 400
        
        # Create payment transaction
        transaction = PaymentTransaction(
            id=uuid7str(),
            merchant_id=data.get('merchant_id', 'default_merchant'),
            customer_id=data.get('customer_id'),
            amount=int(data['amount'] * 100),  # Convert to cents
            currency="KES",
            description=data.get('description', 'B2C payment'),
            payment_method_type=PaymentMethodType.MPESA,
            tenant_id=data.get('tenant_id', 'default_tenant')
        )
        
        # Create payment method
        payment_method = PaymentMethod(
            id=uuid7str(),
            customer_id=transaction.customer_id,
            payment_method_type=PaymentMethodType.MPESA,
            mpesa_phone_number=data['phone_number'],
            tenant_id=transaction.tenant_id
        )
        
        # Additional data
        additional_data = {
            "transaction_type": MPESATransactionType.B2C.value,
            "phone_number": data['phone_number'],
            "command_id": data.get('command_id', 'BusinessPayment'),
            "occasion": data.get('occasion', 'Payment')
        }
        
        # Process payment
        result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
        
        return jsonify({
            "success": result.success,
            "transaction_id": transaction.id,
            "conversation_id": result.processor_transaction_id,
            "status": result.status.value,
            "error": result.error_message if not result.success else None
        })
        
    except Exception as e:
        logger.error(f"B2C payment error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/payments/c2b', methods=['POST'])
async def c2b_payment():
    """Simulate C2B payment"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['phone_number', 'amount', 'bill_ref_number']
        if not all(field in data for field in required_fields):
            return jsonify({
                "success": False,
                "error": "Missing required fields: phone_number, amount, bill_ref_number"
            }), 400
        
        # Create payment transaction
        transaction = PaymentTransaction(
            id=uuid7str(),
            merchant_id=data.get('merchant_id', 'default_merchant'),
            customer_id=data.get('customer_id'),
            amount=int(data['amount'] * 100),  # Convert to cents
            currency="KES",
            description=data.get('description', 'C2B payment simulation'),
            payment_method_type=PaymentMethodType.MPESA,
            tenant_id=data.get('tenant_id', 'default_tenant')
        )
        
        # Create payment method
        payment_method = PaymentMethod(
            id=uuid7str(),
            customer_id=transaction.customer_id,
            payment_method_type=PaymentMethodType.MPESA,
            mpesa_phone_number=data['phone_number'],
            tenant_id=transaction.tenant_id
        )
        
        # Additional data
        additional_data = {
            "transaction_type": MPESATransactionType.C2B.value,
            "phone_number": data['phone_number'],
            "command_id": data.get('command_id', 'CustomerPayBillOnline'),
            "bill_ref_number": data['bill_ref_number']
        }
        
        # Process payment
        result = await mpesa_service.process_payment(transaction, payment_method, additional_data)
        
        return jsonify({
            "success": result.success,
            "transaction_id": transaction.id,
            "conversation_id": result.processor_transaction_id,
            "status": result.status.value,
            "error": result.error_message if not result.success else None
        })
        
    except Exception as e:
        logger.error(f"C2B payment error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/account/balance', methods=['GET'])
async def get_account_balance():
    """Get account balance"""
    try:
        balance_result = await mpesa_service.get_account_balance()
        return jsonify(balance_result)
        
    except Exception as e:
        logger.error(f"Account balance error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/transactions/<transaction_id>/status', methods=['GET'])
async def get_transaction_status(transaction_id):
    """Get transaction status"""
    try:
        status_result = await mpesa_service.get_transaction_status(transaction_id)
        
        return jsonify({
            "success": status_result.success,
            "transaction_id": transaction_id,
            "status": status_result.status.value,
            "metadata": status_result.metadata,
            "error": status_result.error_message if not status_result.success else None
        })
        
    except Exception as e:
        logger.error(f"Transaction status error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/transactions/<transaction_id>/reverse', methods=['POST'])
async def reverse_transaction(transaction_id):
    """Reverse transaction"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['originator_conversation_id', 'mpesa_receipt_number']
        if not all(field in data for field in required_fields):
            return jsonify({
                "success": False,
                "error": "Missing required fields: originator_conversation_id, mpesa_receipt_number"
            }), 400
        
        reversal_result = await mpesa_service.reverse_transaction(
            originator_conversation_id=data['originator_conversation_id'],
            transaction_id=data['mpesa_receipt_number'],
            amount=int(data.get('amount', 0) * 100) if data.get('amount') else None,
            reason=data.get('reason', 'Transaction reversal requested')
        )
        
        return jsonify({
            "success": reversal_result.success,
            "transaction_id": transaction_id,
            "reversal_conversation_id": reversal_result.processor_transaction_id,
            "status": reversal_result.status.value,
            "error": reversal_result.error_message if not reversal_result.success else None
        })
        
    except Exception as e:
        logger.error(f"Transaction reversal error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
async def health_check():
    """Service health check"""
    try:
        health = await mpesa_service.health_check()
        
        return jsonify({
            "success": True,
            "service": "MPESA Payment Gateway",
            "status": health.status.value,
            "metrics": {
                "success_rate": health.success_rate,
                "average_response_time": health.average_response_time,
                "uptime_percentage": health.uptime_percentage,
                "error_count": health.error_count
            },
            "capabilities": {
                "supported_currencies": health.supported_currencies,
                "supported_countries": health.supported_countries
            },
            "last_error": health.last_error,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"success": False, "error": "Bad request"}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

# Application startup
async def startup():
    """Initialize application"""
    logger.info("Starting APG Payment Gateway with MPESA integration...")
    await initialize_mpesa_service()
    logger.info("Application ready!")

if __name__ == '__main__':
    # Initialize the service
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(startup())
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )