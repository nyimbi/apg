#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Integration Testing
Tests for APG ecosystem integration components

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid_extensions import uuid7str

from ...integrations import (
    APGIntegrationManager, EventPublisher, CacheManager,
    APGAuditLogger, ConfigurationManager, MDMEvent, APGCapability
)


class TestMDMEvent:
	"""Test MDM event model"""
	
	def test_mdm_event_creation(self):
		"""Test MDM event creation and serialization"""
		event = MDMEvent(
			event_id=uuid7str(),
			event_type="entity.created",
			entity_id=uuid7str(),
			entity_type="person",
			tenant_id="test-tenant",
			user_id="test-user",
			timestamp=datetime.utcnow(),
			event_data={"entity_name": "John Doe", "operation": "create"},
			correlation_id=uuid7str(),
			priority="normal"
		)
		
		assert event.event_id is not None
		assert event.event_type == "entity.created"
		assert event.source_service == "mdm"
		assert event.priority == "normal"
		
		# Test serialization
		event_dict = event.to_dict()
		assert isinstance(event_dict, dict)
		assert event_dict["event_type"] == "entity.created"
		assert event_dict["source_service"] == "mdm"
		assert "timestamp" in event_dict
		assert event_dict["event_data"]["entity_name"] == "John Doe"


class TestEventPublisher:
	"""Test APG event publishing integration"""
	
	async def test_event_publisher_initialization(self):
		"""Test event publisher initialization"""
		config = {
			"mqeb_url": "http://localhost:8080"
		}
		
		publisher = EventPublisher(config)
		
		assert publisher.mqeb_url == "http://localhost:8080"
		assert publisher.is_running is False
		assert len(publisher.event_routing) > 0
		
		await publisher.start()
		assert publisher.is_running is True
		assert publisher.session is not None
		
		await publisher.stop()
		assert publisher.is_running is False
	
	@patch('aiohttp.ClientSession.post')
	async def test_event_publishing_success(self, mock_post):
		"""Test successful event publishing to APG MQEB"""
		# Mock successful HTTP response
		mock_response = AsyncMock()
		mock_response.status = 200
		mock_post.return_value.__aenter__.return_value = mock_response
		
		config = {"mqeb_url": "http://test-mqeb:8080"}
		publisher = EventPublisher(config)
		await publisher.start()
		
		# Create test event
		event = MDMEvent(
			event_id=uuid7str(),
			event_type="entity.created",
			entity_id=uuid7str(),
			entity_type="person",
			tenant_id="test-tenant",
			user_id="test-user",
			timestamp=datetime.utcnow(),
			event_data={"action": "create", "entity_name": "Test Entity"}
		)
		
		# Publish event
		success = await publisher.publish_event(event)
		
		assert success is True
		
		# Wait for background processing
		await asyncio.sleep(0.1)
		await publisher.event_queue.join()
		
		# Verify HTTP call was made
		mock_post.assert_called_once()
		call_args = mock_post.call_args
		
		assert call_args[0][0].endswith("/api/v1/events/publish")
		
		# Verify payload structure
		payload = call_args.kwargs["json"]
		assert payload["topic"] == "mdm.entities"
		assert payload["priority"] == "normal"
		assert payload["event"]["event_type"] == "entity.created"
		assert payload["headers"]["source"] == "mdm"
		assert payload["headers"]["tenant-id"] == "test-tenant"
		
		await publisher.stop()
	
	@patch('aiohttp.ClientSession.post')
	async def test_event_publishing_failure(self, mock_post):
		"""Test event publishing failure handling"""
		# Mock failed HTTP response
		mock_response = AsyncMock()
		mock_response.status = 500
		mock_response.text.return_value = "Internal Server Error"
		mock_post.return_value.__aenter__.return_value = mock_response
		
		config = {"mqeb_url": "http://test-mqeb:8080"}
		publisher = EventPublisher(config)
		await publisher.start()
		
		event = MDMEvent(
			event_id=uuid7str(),
			event_type="entity.created",
			entity_id=uuid7str(),
			entity_type="person",
			tenant_id="test-tenant",
			user_id="test-user",
			timestamp=datetime.utcnow(),
			event_data={"action": "create"}
		)
		
		success = await publisher.publish_event(event)
		
		assert success is True  # Queuing succeeds
		
		# Wait for background processing
		await asyncio.sleep(0.1)
		await publisher.event_queue.join()
		
		# HTTP call should have been attempted
		mock_post.assert_called_once()
		
		await publisher.stop()
	
	def test_event_routing_configuration(self):
		"""Test event routing configuration"""
		publisher = EventPublisher()
		
		# Test known event types
		assert "entity.created" in publisher.event_routing
		assert "entity.updated" in publisher.event_routing
		assert "entity.deleted" in publisher.event_routing
		assert "quality.assessed" in publisher.event_routing
		assert "duplicates.detected" in publisher.event_routing
		
		# Test routing details
		entity_created_routing = publisher.event_routing["entity.created"]
		assert entity_created_routing["topic"] == "mdm.entities"
		assert entity_created_routing["priority"] == "normal"
		
		entity_deleted_routing = publisher.event_routing["entity.deleted"]
		assert entity_deleted_routing["priority"] == "high"
		
		quality_routing = publisher.event_routing["quality.assessed"]
		assert quality_routing["topic"] == "mdm.quality"
		assert quality_routing["priority"] == "low"


class TestCacheManager:
	"""Test APG caching integration"""
	
	@pytest.fixture
	async def mock_redis(self):
		"""Create mock Redis client"""
		mock_redis = AsyncMock()
		mock_redis.ping.return_value = "PONG"
		mock_redis.get.return_value = None
		mock_redis.setex.return_value = True
		mock_redis.delete.return_value = 1
		mock_redis.keys.return_value = []
		mock_redis.close.return_value = None
		
		with patch('redis.asyncio.from_url', return_value=mock_redis):
			yield mock_redis
	
	async def test_cache_manager_initialization(self, mock_redis):
		"""Test cache manager initialization"""
		config = {
			"redis_url": "redis://localhost:6379",
			"cache_prefix": "test_mdm",
			"default_ttl": 1800
		}
		
		cache_manager = CacheManager(config)
		await cache_manager.initialize()
		
		assert cache_manager.redis_url == "redis://localhost:6379"
		assert cache_manager.cache_prefix == "test_mdm"
		assert cache_manager.default_ttl == 1800
		assert cache_manager.redis_client is not None
		
		mock_redis.ping.assert_called_once()
		
		await cache_manager.close()
	
	async def test_entity_caching_operations(self, mock_redis):
		"""Test entity caching operations"""
		cache_manager = CacheManager()
		await cache_manager.initialize()
		
		tenant_id = "test-tenant"
		entity_id = uuid7str()
		entity_data = {
			"entity_id": entity_id,
			"entity_name": "Test Entity",
			"entity_type": "person",
			"attributes": {"first_name": "John", "last_name": "Doe"}
		}
		
		# Test SET operation
		success = await cache_manager.set_entity(tenant_id, entity_id, entity_data)
		assert success is True
		
		expected_key = f"mdm:entities:{tenant_id}:{entity_id}"
		mock_redis.setex.assert_called_once()
		call_args = mock_redis.setex.call_args[0]
		assert call_args[0] == expected_key
		assert call_args[1] == 1800  # TTL from config
		
		# Verify JSON serialization
		cached_json = call_args[2]
		cached_data = json.loads(cached_json)
		assert cached_data["entity_name"] == "Test Entity"
		
		# Test GET operation
		mock_redis.get.return_value = cached_json
		retrieved_data = await cache_manager.get_entity(tenant_id, entity_id)
		
		assert retrieved_data is not None
		assert retrieved_data["entity_name"] == "Test Entity"
		assert retrieved_data["entity_id"] == entity_id
		
		mock_redis.get.assert_called_with(expected_key)
		
		await cache_manager.close()
	
	async def test_cache_invalidation(self, mock_redis):
		"""Test cache invalidation operations"""
		cache_manager = CacheManager()
		await cache_manager.initialize()
		
		tenant_id = "test-tenant"
		entity_id = uuid7str()
		
		# Mock keys for pattern matching
		related_keys = [
			f"mdm:quality_scores:{tenant_id}:{entity_id}",
			f"mdm:duplicate_results:{tenant_id}:{entity_id}:candidate1",
			f"mdm:search_results:{tenant_id}:hash123"
		]
		mock_redis.keys.return_value = related_keys
		
		success = await cache_manager.invalidate_entity(tenant_id, entity_id)
		assert success is True
		
		# Verify main entity cache deletion
		expected_entity_key = f"mdm:entities:{tenant_id}:{entity_id}"
		delete_calls = [call[0][0] if call[0] else call[0] for call in mock_redis.delete.call_args_list]
		assert expected_entity_key in delete_calls
		
		# Verify related cache pattern deletions
		mock_redis.keys.assert_called()
		
		await cache_manager.close()
	
	async def test_quality_score_caching(self, mock_redis):
		"""Test quality score caching operations"""
		cache_manager = CacheManager()
		await cache_manager.initialize()
		
		tenant_id = "test-tenant"
		entity_id = uuid7str()
		quality_data = {
			"overall_score": 85.0,
			"quality_status": "good",
			"completeness_score": 90.0,
			"assessment_timestamp": datetime.utcnow().isoformat()
		}
		
		# Cache quality score
		success = await cache_manager.set_quality_score(tenant_id, entity_id, quality_data)
		assert success is True
		
		expected_key = f"mdm:quality_scores:{tenant_id}:{entity_id}"
		mock_redis.setex.assert_called()
		call_args = mock_redis.setex.call_args[0]
		assert call_args[0] == expected_key
		assert call_args[1] == 600  # Quality scores TTL
		
		# Retrieve quality score
		mock_redis.get.return_value = json.dumps(quality_data, default=str)
		retrieved_quality = await cache_manager.get_quality_score(tenant_id, entity_id)
		
		assert retrieved_quality is not None
		assert retrieved_quality["overall_score"] == 85.0
		assert retrieved_quality["quality_status"] == "good"
		
		await cache_manager.close()
	
	def test_search_hash_generation(self):
		"""Test search criteria hash generation"""
		cache_manager = CacheManager()
		
		search_criteria_1 = {
			"entity_type": "person",
			"entity_name": "John",
			"limit": 10,
			"offset": 0
		}
		
		search_criteria_2 = {
			"offset": 0,
			"entity_type": "person",
			"limit": 10,
			"entity_name": "John"
		}
		
		# Same criteria in different order should produce same hash
		hash_1 = cache_manager.generate_search_hash(search_criteria_1)
		hash_2 = cache_manager.generate_search_hash(search_criteria_2)
		
		assert hash_1 == hash_2
		assert len(hash_1) == 32  # MD5 hex length
		
		# Different criteria should produce different hash
		different_criteria = {
			"entity_type": "customer",
			"entity_name": "Jane",
			"limit": 20,
			"offset": 10
		}
		
		hash_3 = cache_manager.generate_search_hash(different_criteria)
		assert hash_3 != hash_1


class TestAPGAuditLogger:
	"""Test APG audit logging integration"""
	
	@patch('aiohttp.ClientSession.post')
	async def test_audit_logger_initialization(self, mock_post):
		"""Test audit logger initialization"""
		config = {
			"audl_url": "http://localhost:8081"
		}
		
		audit_logger = APGAuditLogger(config)
		
		assert audit_logger.audl_url == "http://localhost:8081"
		assert audit_logger.is_running is False
		assert len(audit_logger.audit_categories) > 0
		
		await audit_logger.start()
		assert audit_logger.is_running is True
		
		await audit_logger.stop()
		assert audit_logger.is_running is False
	
	@patch('aiohttp.ClientSession.post')
	async def test_audit_event_logging(self, mock_post):
		"""Test audit event logging"""
		# Mock successful HTTP response
		mock_response = AsyncMock()
		mock_response.status = 201
		mock_post.return_value.__aenter__.return_value = mock_response
		
		config = {"audl_url": "http://test-audl:8081"}
		audit_logger = APGAuditLogger(config)
		await audit_logger.start()
		
		# Log audit event
		success = await audit_logger.log_audit_event(
			event_type="entity.created",
			entity_id=uuid7str(),
			tenant_id="test-tenant",
			user_id="test-user",
			event_details={
				"entity_name": "Test Entity",
				"operation": "create",
				"source_system": "api"
			},
			risk_level="medium"
		)
		
		assert success is True
		
		# Wait for background processing
		await asyncio.sleep(0.1)
		await audit_logger.audit_queue.join()
		
		# Verify HTTP call was made
		mock_post.assert_called_once()
		call_args = mock_post.call_args
		
		assert call_args[0][0].endswith("/api/v1/audit/events")
		
		# Verify payload structure
		payload = call_args.kwargs["json"]
		assert payload["event_type"] == "entity.created"
		assert payload["category"] == "data_creation"
		assert payload["tenant_id"] == "test-tenant"
		assert payload["user_id"] == "test-user"
		assert payload["risk_level"] == "medium"
		assert payload["source_service"] == "mdm"
		assert "gdpr" in payload["compliance_tags"]
		assert "sox" in payload["compliance_tags"]
		
		await audit_logger.stop()
	
	def test_audit_categorization(self):
		"""Test audit event categorization"""
		audit_logger = APGAuditLogger()
		
		# Test known event categories
		assert "entity.created" in audit_logger.audit_categories
		assert "entity.updated" in audit_logger.audit_categories
		assert "entity.deleted" in audit_logger.audit_categories
		
		# Test category details
		creation_category = audit_logger.audit_categories["entity.created"]
		assert creation_category["category"] == "data_creation"
		assert creation_category["retention_years"] == 7
		
		deletion_category = audit_logger.audit_categories["entity.deleted"]
		assert deletion_category["category"] == "data_deletion"
		assert deletion_category["retention_years"] == 10


class TestConfigurationManager:
	"""Test APG configuration management integration"""
	
	@patch('aiohttp.ClientSession.get')
	async def test_configuration_loading(self, mock_get):
		"""Test configuration loading from APG CONF service"""
		# Mock successful configuration response
		config_data = {
			"config": {
				"quality_thresholds": {
					"excellent": 90.0,
					"good": 75.0,
					"fair": 60.0,
					"poor": 40.0
				},
				"matching_thresholds": {
					"exact_match": 100.0,
					"high_confidence": 85.0,
					"medium_confidence": 65.0,
					"minimum_match": 45.0
				},
				"ai_settings": {
					"enable_ai": True,
					"confidence_threshold": 0.8
				}
			}
		}
		
		mock_response = AsyncMock()
		mock_response.status = 200
		mock_response.json.return_value = config_data
		mock_get.return_value.__aenter__.return_value = mock_response
		
		config_manager = ConfigurationManager({
			"conf_url": "http://test-conf:8082"
		})
		
		await config_manager.initialize()
		
		# Verify configuration was loaded
		assert config_manager.config_cache is not None
		assert len(config_manager.config_cache) > 0
		
		# Check merged configuration
		excellent_threshold = config_manager.config_cache["quality_thresholds"]["excellent"]
		assert excellent_threshold == 90.0  # From loaded config
		
		# Test get_config method
		threshold = await config_manager.get_config("quality_thresholds.excellent")
		assert threshold == 90.0
		
		ai_enabled = await config_manager.get_config("ai_settings.enable_ai")
		assert ai_enabled is True
		
		# Test non-existent key with default
		non_existent = await config_manager.get_config("non.existent.key", "default_value")
		assert non_existent == "default_value"
		
		await config_manager.close()
	
	@patch('aiohttp.ClientSession.put')
	async def test_configuration_update(self, mock_put):
		"""Test configuration value updates"""
		# Mock successful update response
		mock_response = AsyncMock()
		mock_response.status = 200
		mock_put.return_value.__aenter__.return_value = mock_response
		
		# Mock load_configuration call triggered by update
		config_manager = ConfigurationManager()
		config_manager.config_cache = {"test": {"value": "old"}}
		config_manager.session = AsyncMock()
		
		# Update configuration
		success = await config_manager.update_config(
			"quality_thresholds.excellent", 
			95.0,
			tenant_id="test-tenant"
		)
		
		assert success is True
		
		# Verify HTTP call was made
		mock_put.assert_called_once()
		call_args = mock_put.call_args
		
		assert "quality_thresholds.excellent" in call_args[0][0]
		
		# Verify payload
		payload = call_args.kwargs["json"]
		assert payload["key"] == "quality_thresholds.excellent"
		assert payload["value"] == 95.0
		assert payload["tenant_id"] == "test-tenant"
		assert payload["updated_by"] == "mdm_service"
	
	@patch('aiohttp.ClientSession.get')
	async def test_configuration_fallback_to_defaults(self, mock_get):
		"""Test fallback to default configuration when service unavailable"""
		# Mock failed HTTP response
		mock_response = AsyncMock()
		mock_response.status = 500
		mock_get.return_value.__aenter__.return_value = mock_response
		
		config_manager = ConfigurationManager()
		await config_manager.initialize()
		
		# Should fall back to default configuration
		assert config_manager.config_cache is not None
		assert "quality_thresholds" in config_manager.config_cache
		assert "matching_thresholds" in config_manager.config_cache
		
		# Test default values
		excellent_threshold = await config_manager.get_config("quality_thresholds.excellent")
		assert excellent_threshold == 95.0  # Default value
		
		await config_manager.close()


class TestAPGIntegrationManager:
	"""Test main APG integration manager"""
	
	async def test_integration_manager_initialization(self):
		"""Test integration manager initialization"""
		config = {
			"mqeb_url": "http://test-mqeb:8080",
			"redis_url": "redis://test-redis:6379",
			"audl_url": "http://test-audl:8081",
			"conf_url": "http://test-conf:8082"
		}
		
		integration_manager = APGIntegrationManager(config)
		
		assert integration_manager.event_publisher is not None
		assert integration_manager.cache_manager is not None
		assert integration_manager.audit_logger is not None
		assert integration_manager.config_manager is not None
		assert integration_manager.is_initialized is False
		
		# Mock initialize methods
		integration_manager.event_publisher.start = AsyncMock()
		integration_manager.cache_manager.initialize = AsyncMock()
		integration_manager.cache_manager.redis_client = MagicMock()
		integration_manager.audit_logger.start = AsyncMock()
		integration_manager.config_manager.initialize = AsyncMock()
		
		result = await integration_manager.initialize()
		
		assert result["status"] == "success"
		assert integration_manager.is_initialized is True
		assert result["components"]["event_publisher"] is True
		assert result["components"]["cache_manager"] is True
		assert result["components"]["audit_logger"] is True
		assert result["components"]["config_manager"] is True
		
		# Test shutdown
		integration_manager.event_publisher.stop = AsyncMock()
		integration_manager.cache_manager.close = AsyncMock()
		integration_manager.audit_logger.stop = AsyncMock()
		integration_manager.config_manager.close = AsyncMock()
		
		await integration_manager.shutdown()
		assert integration_manager.is_initialized is False
	
	async def test_publish_entity_event_integration(self):
		"""Test integrated entity event publishing"""
		integration_manager = APGIntegrationManager()
		integration_manager.is_initialized = True
		
		# Mock component methods
		integration_manager.event_publisher.publish_event = AsyncMock(return_value=True)
		integration_manager.audit_logger.log_audit_event = AsyncMock(return_value=True)
		
		entity_id = uuid7str()
		tenant_id = "test-tenant"
		user_id = "test-user"
		
		success = await integration_manager.publish_entity_event(
			event_type="entity.created",
			entity_id=entity_id,
			entity_type="person",
			tenant_id=tenant_id,
			user_id=user_id,
			event_data={"entity_name": "Test Entity"},
			correlation_id=uuid7str()
		)
		
		assert success is True
		
		# Verify event publisher was called
		integration_manager.event_publisher.publish_event.assert_called_once()
		
		# Verify audit logger was called
		integration_manager.audit_logger.log_audit_event.assert_called_once()
		audit_call = integration_manager.audit_logger.log_audit_event.call_args[0]
		assert audit_call[0] == "entity.created"  # event_type
		assert audit_call[1] == entity_id  # entity_id
		assert audit_call[2] == tenant_id  # tenant_id
		assert audit_call[3] == user_id  # user_id
	
	async def test_cache_operations_integration(self):
		"""Test integrated cache operations"""
		integration_manager = APGIntegrationManager()
		integration_manager.is_initialized = True
		
		# Mock cache manager methods
		integration_manager.cache_manager.get_entity = AsyncMock(return_value={"test": "data"})
		integration_manager.cache_manager.set_entity = AsyncMock(return_value=True)
		integration_manager.cache_manager.invalidate_entity = AsyncMock(return_value=True)
		
		tenant_id = "test-tenant"
		entity_id = uuid7str()
		entity_data = {"entity_name": "Test Entity"}
		
		# Test cache retrieval
		cached_data = await integration_manager.get_cached_entity(tenant_id, entity_id)
		assert cached_data == {"test": "data"}
		integration_manager.cache_manager.get_entity.assert_called_once_with(tenant_id, entity_id)
		
		# Test cache storage
		success = await integration_manager.cache_entity(tenant_id, entity_id, entity_data)
		assert success is True
		integration_manager.cache_manager.set_entity.assert_called_once_with(
			tenant_id, entity_id, entity_data
		)
		
		# Test cache invalidation
		success = await integration_manager.invalidate_entity_cache(tenant_id, entity_id)
		assert success is True
		integration_manager.cache_manager.invalidate_entity.assert_called_once_with(
			tenant_id, entity_id
		)
	
	async def test_configuration_operations_integration(self):
		"""Test integrated configuration operations"""
		integration_manager = APGIntegrationManager()
		integration_manager.is_initialized = True
		
		# Mock config manager methods
		integration_manager.config_manager.get_config = AsyncMock(return_value=85.0)
		integration_manager.config_manager.update_config = AsyncMock(return_value=True)
		
		# Test configuration retrieval
		config_value = await integration_manager.get_config_value(
			"quality_thresholds.good", default=80.0
		)
		assert config_value == 85.0
		integration_manager.config_manager.get_config.assert_called_once_with(
			"quality_thresholds.good", 80.0
		)
		
		# Test configuration update
		success = await integration_manager.update_config_value(
			"quality_thresholds.good", 90.0, tenant_id="test-tenant"
		)
		assert success is True
		integration_manager.config_manager.update_config.assert_called_once_with(
			"quality_thresholds.good", 90.0, "test-tenant"
		)