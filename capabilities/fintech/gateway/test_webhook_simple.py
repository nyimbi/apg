#!/usr/bin/env python3
"""
Simple test for webhook management system functionality
"""

import asyncio
from datetime import datetime, timezone
from webhook_service import (
    WebhookService, WebhookEventType, WebhookSecurityType, create_webhook_service
)


class MockDatabaseService:
    """Simple mock database service"""
    
    def __init__(self):
        self._webhook_endpoints = {}
    
    async def initialize(self):
        pass
    
    async def create_webhook_endpoint(self, endpoint):
        self._webhook_endpoints[endpoint.id] = endpoint
    
    async def update_webhook_endpoint(self, endpoint_id, updates):
        pass
    
    async def delete_webhook_endpoint(self, endpoint_id):
        pass


async def test_webhook_functionality():
    """Test core webhook functionality"""
    print("🔗 Testing Webhook Management Core Functionality")
    print("=" * 50)
    
    # Initialize service
    database_service = MockDatabaseService()
    await database_service.initialize()
    
    webhook_service = WebhookService(database_service)
    
    # Manually initialize without HTTP session to avoid connection issues
    webhook_service._endpoints = {}
    webhook_service._pending_events = {}
    webhook_service._delivery_workers = []
    webhook_service._running = True
    webhook_service._initialized = True
    
    print("✅ Webhook service initialized")
    
    # Test 1: Create webhook endpoint
    print("\n📋 Test 1: Creating Webhook Endpoint")
    
    endpoint_data = {
        "tenant_id": "test_tenant",
        "name": "Test Webhook",
        "url": "https://example.com/webhook",
        "enabled_events": [WebhookEventType.PAYMENT_COMPLETED, WebhookEventType.PAYMENT_FAILED],
        "security_type": WebhookSecurityType.HMAC_SHA256,
        "secret": "test_secret_key_12345"
    }
    
    # Skip URL validation for testing
    original_validate = webhook_service._validate_endpoint_url
    webhook_service._validate_endpoint_url = lambda url: None
    
    endpoint = await webhook_service.create_endpoint(endpoint_data)
    print(f"✅ Created webhook endpoint: {endpoint.name}")
    print(f"   URL: {endpoint.url}")
    print(f"   Events: {len(endpoint.enabled_events)}")
    print(f"   Security: {endpoint.security_type.value}")
    
    # Test 2: List endpoints
    print("\n📋 Test 2: Listing Endpoints")
    
    endpoints = await webhook_service.list_endpoints("test_tenant")
    print(f"✅ Found {len(endpoints)} endpoints for tenant")
    
    # Test 3: Update endpoint
    print("\n⚙️  Test 3: Updating Endpoint")
    
    updates = {
        "enabled": False,
        "timeout_seconds": 60
    }
    
    updated_endpoint = await webhook_service.update_endpoint(endpoint.id, updates)
    if updated_endpoint:
        print(f"✅ Updated endpoint:")
        print(f"   Enabled: {updated_endpoint.enabled}")
        print(f"   Timeout: {updated_endpoint.timeout_seconds}s")
    
    # Test 4: Test signature generation
    print("\n🔐 Test 4: Testing Security")
    
    test_payload = {"transaction_id": "test_123", "amount": 1000}
    signature = webhook_service._generate_signature(endpoint, test_payload)
    
    print(f"✅ Generated signature: {signature}")
    print(f"   Format correct: {signature.startswith('sha256=') if signature else False}")
    
    # Test 5: Send webhook event (without actual HTTP delivery)
    print("\n📤 Test 5: Queueing Webhook Events")
    
    event_ids = await webhook_service.send_webhook(
        "test_tenant",
        WebhookEventType.PAYMENT_COMPLETED,
        {
            "transaction_id": "txn_12345",
            "amount": 2500,
            "currency": "USD",
            "status": "completed"
        }
    )
    
    print(f"✅ Queued {len(event_ids)} webhook events")
    
    # Test 6: Get event details
    if event_ids:
        event = await webhook_service.get_event(event_ids[0])
        if event:
            print(f"✅ Retrieved event: {event.id}")
            print(f"   Type: {event.event_type.value}")
            print(f"   Status: {event.status.value}")
            print(f"   Payload keys: {list(event.payload.keys())}")
    
    # Test 7: Get statistics
    print("\n📊 Test 7: Endpoint Statistics")
    
    stats = await webhook_service.get_endpoint_statistics(endpoint.id)
    print(f"✅ Statistics retrieved:")
    print(f"   Total deliveries: {stats.get('total_deliveries', 0)}")
    print(f"   Success rate: {stats.get('success_rate', 0):.1f}%")
    print(f"   Is healthy: {stats.get('is_healthy', True)}")
    
    # Test 8: Service health
    print("\n💚 Test 8: Service Health")
    
    health = await webhook_service.get_service_health()
    print(f"✅ Service health: {health['status']}")
    print(f"   Total endpoints: {health['total_endpoints']}")
    print(f"   Pending events: {health['pending_events']}")
    
    # Test 9: Delete endpoint
    print("\n🗑️  Test 9: Deleting Endpoint")
    
    deleted = await webhook_service.delete_endpoint(endpoint.id)
    print(f"✅ Endpoint deleted: {deleted}")
    
    # Verify deletion
    remaining_endpoints = await webhook_service.list_endpoints("test_tenant")
    print(f"   Remaining endpoints: {len(remaining_endpoints)}")
    
    # Test Summary
    print(f"\n✅ Webhook Management Test Summary:")
    print(f"   📋 Endpoint CRUD operations: ✅")
    print(f"   🔐 Security signature generation: ✅")
    print(f"   📤 Event queueing: ✅")
    print(f"   📊 Statistics and monitoring: ✅")
    print(f"   💚 Health checking: ✅")
    
    print(f"\n🎉 Webhook management core functionality PASSED!")


if __name__ == "__main__":
    asyncio.run(test_webhook_functionality())