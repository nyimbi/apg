#!/usr/bin/env python3
"""
Simple Test for Enhanced Google News Crawler
============================================

Basic test to verify that the enhanced components work together properly.
This is a minimal test to ensure integration is successful.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging

# Set up basic logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_enhanced_client_creation():
	"""Test that enhanced client can be created and configured."""
	try:
		from enhanced_client import EnhancedGoogleNewsConfig, EnhancedGoogleNewsClient
		
		# Test basic configuration
		config = EnhancedGoogleNewsConfig(
			database_url="postgresql:///test_db",
			enable_database_storage=False,  # Don't require actual DB for test
			enable_rate_limiting=True,
			enable_circuit_breaker=True
		)
		
		client = EnhancedGoogleNewsClient(config)
		logger.info("✅ Enhanced client created successfully")
		
		# Test that components are properly configured
		assert client.config.enable_rate_limiting == True
		assert client.config.enable_circuit_breaker == True
		logger.info("✅ Configuration validated")
		
		return True
		
	except Exception as e:
		logger.error(f"❌ Enhanced client creation failed: {e}")
		return False

async def test_database_components():
	"""Test database component creation without actual database."""
	try:
		from database import RateLimitConfig, TokenBucketRateLimiter, GoogleNewsRecord
		
		# Test rate limiter configuration
		rate_config = RateLimitConfig(
			capacity=100,
			refill_rate=10.0,
			enable_adaptive=True
		)
		
		# Create rate limiter (doesn't require DB)
		rate_limiter = TokenBucketRateLimiter(rate_config)
		logger.info("✅ Rate limiter created successfully")
		
		# Test that we can get current tokens
		tokens = rate_limiter.get_current_tokens()
		assert tokens == 100.0  # Should start at capacity
		logger.info(f"✅ Rate limiter tokens: {tokens}")
		
		# Test GoogleNewsRecord creation
		from datetime import datetime, timezone
		record = GoogleNewsRecord(
			title="Test Article",
			content="This is test content",
			content_url="https://example.com/test",
			discovered_at=datetime.now(timezone.utc)
		)
		logger.info("✅ GoogleNewsRecord created successfully")
		
		return True
		
	except Exception as e:
		logger.error(f"❌ Database components test failed: {e}")
		return False

async def test_resilience_components():
	"""Test resilience components."""
	try:
		from resilience import (
			CircuitBreakerConfig, CircuitBreaker, 
			ErrorHandler, create_error_handler
		)
		
		# Test circuit breaker creation
		circuit_breaker = CircuitBreaker(
			CircuitBreakerConfig(failure_threshold=3),
			name="test_circuit"
		)
		logger.info("✅ Circuit breaker created successfully")
		
		# Test that circuit starts in closed state
		assert circuit_breaker.is_closed()
		logger.info("✅ Circuit breaker state validated")
		
		# Test error handler
		error_handler = create_error_handler()
		logger.info("✅ Error handler created successfully")
		
		return True
		
	except Exception as e:
		logger.error(f"❌ Resilience components test failed: {e}")
		return False

async def test_integration_layer():
	"""Test integration layer."""
	try:
		from integration import (
			IntegrationConfig, IntegratedGoogleNewsClient,
			create_basic_client
		)
		
		# Test integration config
		config = IntegrationConfig(
			enable_database=False,  # No actual DB needed for test
			enable_rate_limiting=True,
			enable_circuit_breaker=True
		)
		logger.info("✅ Integration config created successfully")
		
		# Test basic client creation (should not require DB)
		client = create_basic_client()
		logger.info("✅ Basic integrated client created successfully")
		
		return True
		
	except Exception as e:
		logger.error(f"❌ Integration layer test failed: {e}")
		return False

async def test_factory_functions():
	"""Test factory functions."""
	try:
		from enhanced_client import create_enhanced_google_news_client
		from database import create_rate_limiter
		from resilience import create_circuit_breaker, create_error_handler
		
		# Test enhanced client factory
		client = create_enhanced_google_news_client(
			database_url="postgresql:///test",
			enable_rate_limiting=True,
			enable_circuit_breaker=True
		)
		logger.info("✅ Enhanced client factory function works")
		
		# Test rate limiter factory
		rate_limiter = create_rate_limiter(capacity=50, refill_rate=5.0)
		logger.info("✅ Rate limiter factory function works")
		
		# Test circuit breaker factory
		circuit_breaker = create_circuit_breaker(failure_threshold=2)
		logger.info("✅ Circuit breaker factory function works")
		
		# Test error handler factory
		error_handler = create_error_handler()
		logger.info("✅ Error handler factory function works")
		
		return True
		
	except Exception as e:
		logger.error(f"❌ Factory functions test failed: {e}")
		return False

async def run_all_tests():
	"""Run all tests and report results."""
	logger.info("🧪 Starting Enhanced Google News Crawler Tests")
	logger.info("=" * 60)
	
	tests = [
		("Enhanced Client Creation", test_enhanced_client_creation),
		("Database Components", test_database_components),
		("Resilience Components", test_resilience_components),
		("Integration Layer", test_integration_layer),
		("Factory Functions", test_factory_functions)
	]
	
	results = []
	
	for test_name, test_func in tests:
		logger.info(f"\n🔄 Running: {test_name}")
		try:
			result = await test_func()
			results.append((test_name, result))
			if result:
				logger.info(f"✅ PASSED: {test_name}")
			else:
				logger.error(f"❌ FAILED: {test_name}")
		except Exception as e:
			logger.error(f"💥 ERROR in {test_name}: {e}")
			results.append((test_name, False))
	
	# Summary
	logger.info("\n" + "=" * 60)
	logger.info("📊 TEST SUMMARY")
	logger.info("=" * 60)
	
	passed = sum(1 for _, result in results if result)
	total = len(results)
	
	for test_name, result in results:
		status = "✅ PASS" if result else "❌ FAIL"
		logger.info(f"   {status}: {test_name}")
	
	logger.info(f"\nOverall: {passed}/{total} tests passed")
	
	if passed == total:
		logger.info("🎉 ALL TESTS PASSED! Enhanced Google News Crawler is ready for use.")
		return True
	else:
		logger.error(f"💥 {total - passed} test(s) failed. Please review the errors above.")
		return False

if __name__ == "__main__":
	"""Run the tests when script is executed directly."""
	success = asyncio.run(run_all_tests())
	exit(0 if success else 1)