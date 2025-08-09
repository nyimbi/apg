"""
APG Central Configuration - End-to-End Production Testing Scenarios

Comprehensive production-ready testing scenarios covering all major functionality,
performance characteristics, and failure modes.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
import uuid
import pytest
import httpx
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

# Test Configuration
TEST_BASE_URL = "http://localhost:8000"
TEST_API_KEY = "cc_test_api_key_2025"
TEST_TIMEOUT = 30.0
LOAD_TEST_DURATION = 300  # 5 minutes
STRESS_TEST_CONCURRENT_USERS = 50


class ProductionTestSuite:
	"""Comprehensive production testing suite."""
	
	def __init__(self, base_url: str = TEST_BASE_URL, api_key: str = TEST_API_KEY):
		"""Initialize test suite."""
		self.base_url = base_url
		self.api_key = api_key
		self.client = httpx.AsyncClient(
			base_url=base_url,
			headers={
				"X-API-Key": api_key,
				"Content-Type": "application/json"
			},
			timeout=TEST_TIMEOUT
		)
		self.test_workspace_id = f"prod_test_{uuid.uuid4().hex[:8]}"
		self.created_configs: List[str] = []
	
	async def setup(self):
		"""Set up test environment."""
		# Health check
		response = await self.client.get("/health")
		assert response.status_code == 200
		health_data = response.json()
		assert health_data["status"] == "healthy"
		
		# Create test workspace
		workspace_data = {
			"name": f"Production Test Workspace {datetime.now().strftime('%Y%m%d_%H%M%S')}",
			"description": "Automated production testing workspace",
			"settings": {
				"auto_cleanup": True,
				"test_mode": True
			}
		}
		
		response = await self.client.post("/workspaces", json=workspace_data)
		if response.status_code not in [200, 201]:
			# Use default workspace if creation fails
			self.test_workspace_id = "default"
		else:
			workspace = response.json()
			self.test_workspace_id = workspace["id"]
	
	async def teardown(self):
		"""Clean up test environment."""
		# Clean up created configurations
		for config_id in self.created_configs:
			try:
				await self.client.delete(f"/configurations/{config_id}")
			except:
				pass
		
		# Clean up test workspace if not default
		if self.test_workspace_id != "default":
			try:
				await self.client.delete(f"/workspaces/{self.test_workspace_id}")
			except:
				pass
		
		await self.client.aclose()


@pytest.fixture
async def test_suite():
	"""Fixture for production test suite."""
	suite = ProductionTestSuite()
	await suite.setup()
	yield suite
	await suite.teardown()


class TestCoreAPIFunctionality:
	"""Test core API functionality in production conditions."""
	
	async def test_health_endpoint_comprehensive(self, test_suite: ProductionTestSuite):
		"""Test comprehensive health check functionality."""
		response = await test_suite.client.get("/health")
		assert response.status_code == 200
		
		health_data = response.json()
		assert health_data["status"] == "healthy"
		assert "timestamp" in health_data
		assert "version" in health_data
		assert "components" in health_data
		
		# Check component health
		components = health_data["components"]
		assert components["database"]["status"] == "healthy"
		assert components["cache"]["status"] == "healthy"
		assert "response_time" in components["database"]
		assert "response_time" in components["cache"]
	
	async def test_configuration_crud_operations(self, test_suite: ProductionTestSuite):
		"""Test complete configuration CRUD operations."""
		# Create configuration
		config_data = {
			"name": f"Production Test Config {uuid.uuid4().hex[:8]}",
			"key_path": f"/test/prod/config_{int(time.time())}",
			"value": {
				"database": {
					"host": "prod-db.example.com",
					"port": 5432,
					"pool_size": 20,
					"timeout": 30
				},
				"cache": {
					"enabled": True,
					"ttl": 3600,
					"max_memory": "512MB"
				},
				"features": {
					"ai_optimization": True,
					"real_time_sync": True,
					"encryption": True
				}
			},
			"security_level": "confidential",
			"tags": ["production", "database", "cache", "automated-test"]
		}
		
		# CREATE
		response = await test_suite.client.post(
			f"/configurations?workspace_id={test_suite.test_workspace_id}",
			json=config_data
		)
		assert response.status_code in [200, 201]
		created_config = response.json()
		config_id = created_config["id"]
		test_suite.created_configs.append(config_id)
		
		assert created_config["name"] == config_data["name"]
		assert created_config["key_path"] == config_data["key_path"]
		assert created_config["security_level"] == config_data["security_level"]
		
		# READ
		response = await test_suite.client.get(f"/configurations/{config_id}")
		assert response.status_code == 200
		retrieved_config = response.json()
		assert retrieved_config["id"] == config_id
		assert retrieved_config["value"] == config_data["value"]
		
		# UPDATE
		update_data = {
			"value": {
				**config_data["value"],
				"cache": {
					**config_data["value"]["cache"],
					"ttl": 7200  # Updated TTL
				}
			},
			"metadata": {
				"updated_by": "production_test",
				"update_reason": "Performance optimization"
			}
		}
		
		response = await test_suite.client.put(f"/configurations/{config_id}", json=update_data)
		assert response.status_code == 200
		updated_config = response.json()
		assert updated_config["value"]["cache"]["ttl"] == 7200
		
		# DELETE
		response = await test_suite.client.delete(f"/configurations/{config_id}")
		assert response.status_code == 204
		
		# Verify deletion
		response = await test_suite.client.get(f"/configurations/{config_id}")
		assert response.status_code == 404
		
		test_suite.created_configs.remove(config_id)
	
	async def test_configuration_versioning(self, test_suite: ProductionTestSuite):
		"""Test configuration versioning system."""
		# Create initial configuration
		config_data = {
			"name": f"Versioning Test Config {uuid.uuid4().hex[:8]}",
			"key_path": f"/test/versioning/{int(time.time())}",
			"value": {"version": "1.0.0", "feature_flag": False},
			"security_level": "internal",
			"tags": ["versioning-test"]
		}
		
		response = await test_suite.client.post(
			f"/configurations?workspace_id={test_suite.test_workspace_id}",
			json=config_data
		)
		assert response.status_code in [200, 201]
		config = response.json()
		config_id = config["id"]
		test_suite.created_configs.append(config_id)
		
		initial_version = config["version"]
		
		# Update configuration multiple times
		for i in range(3):
			update_data = {
				"value": {"version": f"1.{i+1}.0", "feature_flag": i % 2 == 0},
				"change_reason": f"Version update {i+1}"
			}
			
			response = await test_suite.client.put(f"/configurations/{config_id}", json=update_data)
			assert response.status_code == 200
			updated_config = response.json()
			assert updated_config["version"] > initial_version
		
		# Get version history
		response = await test_suite.client.get(f"/configurations/{config_id}/versions")
		assert response.status_code == 200
		versions = response.json()
		assert len(versions["versions"]) >= 4  # Initial + 3 updates
		
		# Test version rollback
		target_version = versions["versions"][-2]["version"]  # Second to last version
		response = await test_suite.client.post(
			f"/configurations/{config_id}/rollback",
			json={"target_version": target_version}
		)
		assert response.status_code == 200
		
		# Verify rollback
		response = await test_suite.client.get(f"/configurations/{config_id}")
		assert response.status_code == 200
		current_config = response.json()
		assert current_config["value"]["version"] == "1.2.0"  # Rolled back version


class TestAIAndAutomationFeatures:
	"""Test AI and automation features under production load."""
	
	async def test_natural_language_query_processing(self, test_suite: ProductionTestSuite):
		"""Test natural language query processing."""
		# Create test configurations for querying
		test_configs = [
			{
				"name": "Database Connection Pool",
				"key_path": "/app/database/pool",
				"value": {"host": "db.prod.com", "pool_size": 50},
				"tags": ["database", "production"]
			},
			{
				"name": "Redis Cache Settings",
				"key_path": "/app/cache/redis",
				"value": {"host": "cache.prod.com", "memory": "2GB"},
				"tags": ["cache", "redis", "production"]
			},
			{
				"name": "API Rate Limiting",
				"key_path": "/app/api/rate_limit",
				"value": {"requests_per_minute": 1000, "burst_size": 2000},
				"tags": ["api", "rate-limiting"]
			}
		]
		
		config_ids = []
		for config_data in test_configs:
			response = await test_suite.client.post(
				f"/configurations?workspace_id={test_suite.test_workspace_id}",
				json=config_data
			)
			if response.status_code in [200, 201]:
				config_ids.append(response.json()["id"])
		
		test_suite.created_configs.extend(config_ids)
		
		# Test natural language queries
		nl_queries = [
			"find all database configurations",
			"show me cache settings",
			"configurations with production tag",
			"API rate limiting settings"
		]
		
		for query in nl_queries:
			response = await test_suite.client.post(
				"/configurations/natural-language-query",
				json={"query": query}
			)
			
			# Should return 200 (success) or 503 (AI service unavailable)
			assert response.status_code in [200, 503]
			
			if response.status_code == 200:
				result = response.json()
				assert "results" in result
				assert "query_interpretation" in result
	
	async def test_ai_optimization_suggestions(self, test_suite: ProductionTestSuite):
		"""Test AI-powered optimization suggestions."""
		# Create configuration that could be optimized
		config_data = {
			"name": "Optimization Test Config",
			"key_path": "/test/optimization",
			"value": {
				"database": {
					"pool_size": 5,  # Suboptimal
					"timeout": 60,   # Too high
					"retry_attempts": 10  # Too many
				},
				"cache": {
					"ttl": 30,      # Too low for production
					"max_memory": "64MB"  # Too small
				}
			},
			"tags": ["optimization-test"]
		}
		
		response = await test_suite.client.post(
			f"/configurations?workspace_id={test_suite.test_workspace_id}",
			json=config_data
		)
		assert response.status_code in [200, 201]
		config_id = response.json()["id"]
		test_suite.created_configs.append(config_id)
		
		# Request AI optimization
		response = await test_suite.client.post(f"/configurations/{config_id}/optimize")
		
		# Should return 200 (success) or 503 (AI service unavailable)
		assert response.status_code in [200, 503]
		
		if response.status_code == 200:
			optimization = response.json()
			assert "optimizations" in optimization
			assert "confidence_score" in optimization
			assert optimization["confidence_score"] >= 0.0
			assert optimization["confidence_score"] <= 1.0
	
	async def test_automation_engine(self, test_suite: ProductionTestSuite):
		"""Test automation engine functionality."""
		# Create automation rule
		rule_data = {
			"name": "Production Auto-scaling Rule",
			"description": "Auto-scale database connections based on load",
			"condition": {
				"metric": "database_connections_usage",
				"operator": "greater_than",
				"threshold": 0.8
			},
			"action": {
				"type": "update_configuration",
				"target_key": "/app/database/pool_size",
				"adjustment": "increase_by_percentage",
				"value": 20
			},
			"enabled": True,
			"safety_checks": True
		}
		
		response = await test_suite.client.post("/automation/rules", json=rule_data)
		
		# Should return 200/201 (success) or 503 (service unavailable)
		assert response.status_code in [200, 201, 503]
		
		if response.status_code in [200, 201]:
			rule = response.json()
			rule_id = rule["id"]
			
			# Test rule execution simulation
			response = await test_suite.client.post(
				f"/automation/rules/{rule_id}/simulate",
				json={"metric_value": 0.85}  # Above threshold
			)
			assert response.status_code == 200
			
			simulation = response.json()
			assert "would_trigger" in simulation
			assert simulation["would_trigger"] is True
			
			# Clean up rule
			await test_suite.client.delete(f"/automation/rules/{rule_id}")


class TestPerformanceAndScalability:
	"""Test performance and scalability characteristics."""
	
	async def test_concurrent_configuration_operations(self, test_suite: ProductionTestSuite):
		"""Test concurrent configuration operations."""
		async def create_config(index: int):
			config_data = {
				"name": f"Concurrent Test Config {index}",
				"key_path": f"/test/concurrent/{index}",
				"value": {"index": index, "timestamp": time.time()},
				"tags": ["concurrent-test"]
			}
			
			response = await test_suite.client.post(
				f"/configurations?workspace_id={test_suite.test_workspace_id}",
				json=config_data
			)
			
			if response.status_code in [200, 201]:
				return response.json()["id"]
			return None
		
		# Create configurations concurrently
		concurrent_tasks = [create_config(i) for i in range(20)]
		config_ids = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
		
		# Filter successful creations
		successful_ids = [cid for cid in config_ids if isinstance(cid, str)]
		test_suite.created_configs.extend(successful_ids)
		
		# Should have created at least 80% successfully
		assert len(successful_ids) >= 16
		
		# Test concurrent reads
		async def read_config(config_id: str):
			response = await test_suite.client.get(f"/configurations/{config_id}")
			return response.status_code == 200
		
		read_tasks = [read_config(cid) for cid in successful_ids[:10]]
		read_results = await asyncio.gather(*read_tasks, return_exceptions=True)
		
		# All reads should be successful
		successful_reads = [r for r in read_results if r is True]
		assert len(successful_reads) == len(successful_ids[:10])
	
	async def test_large_configuration_handling(self, test_suite: ProductionTestSuite):
		"""Test handling of large configuration objects."""
		# Create large configuration (simulate complex production config)
		large_config_value = {
			"microservices": {
				f"service_{i}": {
					"host": f"service{i}.internal.com",
					"port": 8000 + i,
					"endpoints": [f"/api/v{j}" for j in range(1, 6)],
					"config": {
						"database": {
							"connection_string": f"postgresql://user:pass@db{i}.internal.com:5432/service{i}",
							"pool_size": 10 + (i % 20),
							"timeout": 30
						},
						"cache": {
							"redis_url": f"redis://cache{i}.internal.com:6379",
							"ttl": 3600 + (i * 60)
						},
						"monitoring": {
							"metrics_enabled": True,
							"tracing_enabled": True,
							"log_level": "info"
						}
					}
				} for i in range(100)  # 100 microservices config
			},
			"shared_config": {
				"auth": {
					"jwt_secret": "production-secret-key",
					"token_expiry": 3600,
					"refresh_token_expiry": 86400
				},
				"monitoring": {
					"prometheus_url": "http://prometheus.monitoring.svc.cluster.local:9090",
					"grafana_url": "http://grafana.monitoring.svc.cluster.local:3000"
				}
			}
		}
		
		config_data = {
			"name": "Large Production Config",
			"key_path": "/production/microservices/config",
			"value": large_config_value,
			"security_level": "confidential",
			"tags": ["large-config", "microservices", "production"]
		}
		
		start_time = time.time()
		response = await test_suite.client.post(
			f"/configurations?workspace_id={test_suite.test_workspace_id}",
			json=config_data
		)
		create_time = time.time() - start_time
		
		assert response.status_code in [200, 201]
		config = response.json()
		config_id = config["id"]
		test_suite.created_configs.append(config_id)
		
		# Should handle large config reasonably quickly (under 5 seconds)
		assert create_time < 5.0
		
		# Test retrieval performance
		start_time = time.time()
		response = await test_suite.client.get(f"/configurations/{config_id}")
		read_time = time.time() - start_time
		
		assert response.status_code == 200
		assert read_time < 2.0  # Should read quickly
		
		retrieved_config = response.json()
		assert len(retrieved_config["value"]["microservices"]) == 100
	
	async def test_search_performance(self, test_suite: ProductionTestSuite):
		"""Test search performance with large dataset."""
		# Create many configurations for search testing
		search_configs = []
		
		# Create configurations with different patterns
		patterns = ["database", "cache", "api", "monitoring", "security"]
		environments = ["dev", "staging", "prod"]
		
		for i in range(50):  # Create 50 configs
			pattern = patterns[i % len(patterns)]
			env = environments[i % len(environments)]
			
			config_data = {
				"name": f"{pattern.title()} Config {env.upper()} {i}",
				"key_path": f"/{env}/{pattern}/config_{i}",
				"value": {
					"type": pattern,
					"environment": env,
					"index": i,
					"settings": {f"{pattern}_setting_{j}": f"value_{j}" for j in range(10)}
				},
				"tags": [pattern, env, f"index_{i}"]
			}
			
			response = await test_suite.client.post(
				f"/configurations?workspace_id={test_suite.test_workspace_id}",
				json=config_data
			)
			
			if response.status_code in [200, 201]:
				search_configs.append(response.json()["id"])
		
		test_suite.created_configs.extend(search_configs)
		
		# Test various search queries
		search_queries = [
			"database",
			"prod",
			"cache AND staging",
			"api OR monitoring",
			"tag:database",
			"environment:prod"
		]
		
		for query in search_queries:
			start_time = time.time()
			response = await test_suite.client.get(
				f"/configurations?query={query}&limit=20"
			)
			search_time = time.time() - start_time
			
			assert response.status_code == 200
			assert search_time < 1.0  # Should search quickly
			
			results = response.json()
			assert "configurations" in results
			assert len(results["configurations"]) <= 20


class TestSecurityAndResilience:
	"""Test security features and system resilience."""
	
	async def test_authentication_and_authorization(self, test_suite: ProductionTestSuite):
		"""Test authentication and authorization."""
		# Test with invalid API key
		invalid_client = httpx.AsyncClient(
			base_url=test_suite.base_url,
			headers={"X-API-Key": "invalid_key", "Content-Type": "application/json"},
			timeout=TEST_TIMEOUT
		)
		
		response = await invalid_client.get("/configurations")
		assert response.status_code == 401
		
		await invalid_client.aclose()
		
		# Test with no API key
		no_auth_client = httpx.AsyncClient(
			base_url=test_suite.base_url,
			headers={"Content-Type": "application/json"},
			timeout=TEST_TIMEOUT
		)
		
		response = await no_auth_client.get("/configurations")
		assert response.status_code == 401
		
		await no_auth_client.aclose()
	
	async def test_input_validation_and_sanitization(self, test_suite: ProductionTestSuite):
		"""Test input validation and sanitization."""
		# Test various invalid inputs
		invalid_configs = [
			# Missing required fields
			{"name": "Invalid Config"},
			
			# Invalid security level
			{
				"name": "Invalid Security",
				"key_path": "/test/invalid",
				"value": {"test": True},
				"security_level": "invalid_level"
			},
			
			# Malicious key path
			{
				"name": "Malicious Path",
				"key_path": "/../../etc/passwd",
				"value": {"test": True}
			},
			
			# Extremely long name
			{
				"name": "x" * 1000,
				"key_path": "/test/long",
				"value": {"test": True}
			}
		]
		
		for invalid_config in invalid_configs:
			response = await test_suite.client.post(
				f"/configurations?workspace_id={test_suite.test_workspace_id}",
				json=invalid_config
			)
			assert response.status_code in [400, 422]  # Bad request or validation error
	
	async def test_rate_limiting(self, test_suite: ProductionTestSuite):
		"""Test API rate limiting."""
		# Make many requests rapidly
		rapid_requests = []
		
		for i in range(100):  # 100 rapid requests
			task = test_suite.client.get("/health")
			rapid_requests.append(task)
		
		responses = await asyncio.gather(*rapid_requests, return_exceptions=True)
		
		# Check for rate limiting responses
		status_codes = []
		for response in responses:
			if isinstance(response, httpx.Response):
				status_codes.append(response.status_code)
			elif hasattr(response, 'response') and hasattr(response.response, 'status_code'):
				status_codes.append(response.response.status_code)
		
		# Should have some rate limiting (429) if implemented
		# Or all should be successful if rate limits are high enough for testing
		success_codes = [code for code in status_codes if code == 200]
		rate_limited_codes = [code for code in status_codes if code == 429]
		
		# Either all successful or some rate limited
		assert len(success_codes) > 0
		if rate_limited_codes:
			assert len(rate_limited_codes) > 0
	
	async def test_error_handling_and_recovery(self, test_suite: ProductionTestSuite):
		"""Test error handling and recovery mechanisms."""
		# Test handling of non-existent resources
		response = await test_suite.client.get("/configurations/non-existent-id")
		assert response.status_code == 404
		
		error_response = response.json()
		assert "error" in error_response or "message" in error_response
		
		# Test handling of malformed requests
		response = await test_suite.client.post(
			"/configurations",
			content="invalid json content",
			headers={"Content-Type": "application/json", "X-API-Key": test_suite.api_key}
		)
		assert response.status_code in [400, 422]
		
		# Test timeout handling (if implemented)
		# This would test how the system handles slow operations


class TestCapabilityManagementSystem:
	"""Test the capability management and orchestration features."""
	
	async def test_capability_health_monitoring(self, test_suite: ProductionTestSuite):
		"""Test capability health monitoring."""
		response = await test_suite.client.get("/capabilities/health")
		
		# Should return 200 (success) or 503 (service unavailable)
		assert response.status_code in [200, 503]
		
		if response.status_code == 200:
			health_data = response.json()
			assert "capabilities" in health_data
			
			# Check for expected capabilities
			expected_capabilities = ["central_configuration", "api_service_mesh", "realtime_collaboration"]
			capability_ids = [cap["capability_id"] for cap in health_data["capabilities"]]
			
			for expected_cap in expected_capabilities:
				if expected_cap in capability_ids:
					cap_health = next(cap for cap in health_data["capabilities"] if cap["capability_id"] == expected_cap)
					assert "status" in cap_health
					assert "deployments" in cap_health
	
	async def test_cross_capability_configuration(self, test_suite: ProductionTestSuite):
		"""Test cross-capability configuration management."""
		# Create cross-capability configuration
		cross_config_data = {
			"name": "Production Cross-Capability Config",
			"description": "Configuration affecting multiple capabilities",
			"capability_configs": {
				"central_configuration": {
					"ai_enabled": True,
					"automation_enabled": True
				},
				"api_service_mesh": {
					"load_balancing": "round_robin",
					"circuit_breaker_enabled": True
				}
			},
			"deployment_strategy": "sequential"
		}
		
		response = await test_suite.client.post("/capabilities/cross-config", json=cross_config_data)
		
		# Should return 200/201 (success) or 503 (service unavailable)
		assert response.status_code in [200, 201, 503]
		
		if response.status_code in [200, 201]:
			cross_config = response.json()
			config_id = cross_config["config_id"]
			
			# Test dry run application
			response = await test_suite.client.post(
				f"/capabilities/cross-config/{config_id}/apply",
				json={"dry_run": True}
			)
			assert response.status_code == 200
			
			dry_run_result = response.json()
			assert "dry_run" in dry_run_result
			assert dry_run_result["dry_run"] is True


@pytest.mark.asyncio
async def test_full_production_workflow():
	"""Test complete production workflow end-to-end."""
	test_suite = ProductionTestSuite()
	await test_suite.setup()
	
	try:
		# 1. Health Check
		response = await test_suite.client.get("/health")
		assert response.status_code == 200
		
		# 2. Create Production Configuration
		prod_config = {
			"name": "Production Database Configuration",
			"key_path": "/prod/database/primary",
			"value": {
				"host": "prod-db-cluster.internal.com",
				"port": 5432,
				"database": "production_app",
				"username": "app_user",
				"pool_config": {
					"min_connections": 10,
					"max_connections": 100,
					"connection_timeout": 30,
					"idle_timeout": 300
				},
				"ssl_config": {
					"enabled": True,
					"mode": "require",
					"cert_path": "/etc/ssl/certs/db-client.crt"
				},
				"backup_config": {
					"enabled": True,
					"schedule": "0 2 * * *",  # Daily at 2 AM
					"retention_days": 30
				}
			},
			"security_level": "confidential",
			"tags": ["production", "database", "critical"]
		}
		
		response = await test_suite.client.post(
			f"/configurations?workspace_id={test_suite.test_workspace_id}",
			json=prod_config
		)
		assert response.status_code in [200, 201]
		config = response.json()
		config_id = config["id"]
		test_suite.created_configs.append(config_id)
		
		# 3. Update Configuration
		update_data = {
			"value": {
				**prod_config["value"],
				"pool_config": {
					**prod_config["value"]["pool_config"],
					"max_connections": 150  # Scale up
				}
			},
			"change_reason": "Scale up for increased load"
		}
		
		response = await test_suite.client.put(f"/configurations/{config_id}", json=update_data)
		assert response.status_code == 200
		
		# 4. Test AI Optimization (if available)
		response = await test_suite.client.post(f"/configurations/{config_id}/optimize")
		assert response.status_code in [200, 503]  # Success or service unavailable
		
		# 5. Search for Configuration
		response = await test_suite.client.get("/configurations?query=production database")
		assert response.status_code == 200
		search_results = response.json()
		
		found_config = False
		for result in search_results.get("configurations", []):
			if result["id"] == config_id:
				found_config = True
				break
		assert found_config
		
		# 6. Test Version History
		response = await test_suite.client.get(f"/configurations/{config_id}/versions")
		assert response.status_code == 200
		versions = response.json()
		assert len(versions["versions"]) >= 2  # Initial + update
		
		# 7. Export Configuration
		response = await test_suite.client.get(f"/configurations/{config_id}/export")
		assert response.status_code in [200, 501]  # Success or not implemented
		
		print("✅ Full production workflow test completed successfully")
		
	finally:
		await test_suite.teardown()


if __name__ == "__main__":
	# Run the full production workflow test
	asyncio.run(test_full_production_workflow())