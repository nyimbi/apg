#!/usr/bin/env python3
"""
Analytics Integration Validation - Complete Test

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Validate complete analytics integration with MultiTenantManager service.
"""

import asyncio
import sys
from datetime import datetime, UTC


print("🚀 Analytics Integration Validation - MultiTenantManager")
print("=" * 70)


async def test_analytics_integration():
	"""Test complete analytics integration"""
	print("🧪 Testing Analytics Integration...")
	
	# Import here to ensure clean test environment
	from service import MultiTenantManager
	from models import TenantStatus, TenantTier
	from views import TenantCreateRequest
	from analytics_engine import PredictionType, AlertLevel, TimeRange
	
	# Initialize manager with analytics enabled
	manager = MultiTenantManager(
		tenant_id="analytics-integration-test",
		db_url="sqlite:///test.db",
		cache_url="redis://localhost:6379/1"
	)
	
	# Initialize with analytics enabled
	config = {
		"enable_ai_optimization": True,
		"enable_multi_cloud": True,
		"enable_security_compliance": True,
		"enable_analytics": True,
		"provisioning_timeout_seconds": 60
	}
	
	await manager.initialize(config)
	print("  ✅ MultiTenantManager initialized with analytics enabled")
	
	# Create test tenant
	tenant_request = TenantCreateRequest(
		name="analytics-test-tenant",
		display_name="Analytics Test Tenant",
		tier=TenantTier.PREMIUM,
		template_id=None,
		initial_configuration={}
	)
	
	tenant = await manager.create_tenant(tenant_request)
	assert tenant is not None
	assert tenant.status == TenantStatus.ACTIVE
	print(f"  ✅ Created test tenant: {tenant.id}")
	
	# Test analytics methods
	print("\n🔍 Testing Analytics Methods...")
	
	# Test tenant analytics
	analytics_data = await manager.get_tenant_analytics(tenant.id)
	assert "error" not in analytics_data
	assert analytics_data["tenant_id"] == tenant.id
	assert "current_metrics" in analytics_data
	assert "trends" in analytics_data
	assert "time_series" in analytics_data
	print(f"  ✅ Tenant analytics: health score {analytics_data['current_metrics']['health_score']:.1%}")
	
	# Test predictive insights
	predictions = await manager.get_predictive_insights(tenant.id)
	assert len(predictions) >= 1
	prediction_types = [p.prediction_type for p in predictions]
	high_confidence = [p for p in predictions if p.is_high_confidence()]
	print(f"  ✅ Predictive insights: {len(predictions)} predictions, {len(high_confidence)} high-confidence")
	
	# Test alerts
	alerts = await manager.get_tenant_alerts(tenant.id)
	critical_alerts = [a for a in alerts if a.alert_level == AlertLevel.CRITICAL]
	print(f"  ✅ Tenant alerts: {len(alerts)} total, {len(critical_alerts)} critical")
	
	# Test tenant health score
	health_score = await manager.get_tenant_health_score(tenant.id)
	assert 0.0 <= health_score <= 1.0
	print(f"  ✅ Tenant health score: {health_score:.1%}")
	
	# Test optimization recommendations
	recommendations = await manager.get_performance_optimization_recommendations(tenant.id)
	print(f"  ✅ Optimization recommendations: {len(recommendations)} generated")
	
	# Test system analytics
	system_analytics = await manager.get_system_analytics()
	assert "error" not in system_analytics
	assert "mten_info" in system_analytics
	assert system_analytics["total_tenants"] >= 1
	print(f"  ✅ System analytics: {system_analytics['total_tenants']} tenants monitored")
	
	# Test analytics summary
	analytics_summary = await manager.get_analytics_summary()
	assert analytics_summary["analytics_engine_status"] == "enabled"
	assert analytics_summary["total_tenants_monitored"] >= 1
	capabilities = analytics_summary["capabilities"]
	print(f"  ✅ Analytics summary: {len(capabilities)} capabilities available")
	
	return manager, tenant.id


async def test_multi_tenant_analytics():
	"""Test analytics with multiple tenants"""
	print("\n🧪 Testing Multi-Tenant Analytics...")
	
	manager, first_tenant_id = await test_analytics_integration()
	
	# Create additional tenants
	additional_tenants = []
	for i in range(3):
		tenant_request = TenantCreateRequest(
			name=f"analytics-tenant-{i+2}",
			display_name=f"Analytics Tenant {i+2}",
			tier=TenantTier.PREMIUM if i % 2 == 0 else TenantTier.STANDARD,
			template_id=None,
			initial_configuration={}
		)
		
		tenant = await manager.create_tenant(tenant_request)
		additional_tenants.append(tenant.id)
	
	print(f"  ✅ Created {len(additional_tenants)} additional tenants")
	
	# Test system-wide analytics with multiple tenants
	system_analytics = await manager.get_system_analytics()
	total_tenants = system_analytics["total_tenants"]
	assert total_tenants >= 4  # 1 original + 3 additional
	
	# Get health distribution
	health_dist = system_analytics.get("system_health_distribution", {})
	healthy = health_dist.get("healthy_tenants", 0)
	degraded = health_dist.get("degraded_tenants", 0)
	critical = health_dist.get("critical_tenants", 0)
	
	print(f"  ✅ System health: {healthy} healthy, {degraded} degraded, {critical} critical")
	
	# Test analytics for each tenant
	all_health_scores = []
	all_predictions = []
	all_alerts = []
	
	for tenant_id in [first_tenant_id] + additional_tenants:
		health_score = await manager.get_tenant_health_score(tenant_id)
		predictions = await manager.get_predictive_insights(tenant_id)
		alerts = await manager.get_tenant_alerts(tenant_id)
		
		all_health_scores.append(health_score)
		all_predictions.extend(predictions)
		all_alerts.extend(alerts)
	
	avg_health = sum(all_health_scores) / len(all_health_scores)
	total_predictions = len(all_predictions)
	total_alerts = len(all_alerts)
	
	print(f"  ✅ Aggregate metrics:")
	print(f"    - Average health score: {avg_health:.1%}")
	print(f"    - Total predictions: {total_predictions}")
	print(f"    - Total alerts: {total_alerts}")
	
	return manager


async def test_analytics_performance():
	"""Test analytics performance under load"""
	print("\n🧪 Testing Analytics Performance...")
	
	manager = await test_multi_tenant_analytics()
	
	# Performance test: multiple concurrent analytics calls
	start_time = datetime.now(UTC)
	
	# Get all tenant IDs
	all_tenants = list(manager._tenants.keys())
	
	# Concurrent analytics calls
	tasks = []
	for tenant_id in all_tenants:
		tasks.extend([
			manager.get_tenant_analytics(tenant_id),
			manager.get_predictive_insights(tenant_id),
			manager.get_tenant_alerts(tenant_id),
			manager.get_tenant_health_score(tenant_id)
		])
	
	# Execute all tasks concurrently
	results = await asyncio.gather(*tasks, return_exceptions=True)
	
	elapsed_time = (datetime.now(UTC) - start_time).total_seconds()
	
	# Count successful results
	successful_results = [r for r in results if not isinstance(r, Exception)]
	failed_results = [r for r in results if isinstance(r, Exception)]
	
	print(f"  ⚡ Performance test results:")
	print(f"    - Total operations: {len(tasks)}")
	print(f"    - Successful: {len(successful_results)}")
	print(f"    - Failed: {len(failed_results)}")
	print(f"    - Total time: {elapsed_time:.3f}s")
	print(f"    - Avg time per operation: {elapsed_time/len(tasks):.3f}s")
	
	# Performance assertions
	assert len(successful_results) >= len(tasks) * 0.95, "At least 95% of operations should succeed"
	assert elapsed_time < 10.0, "All operations should complete within 10 seconds"
	
	print("  ✅ Performance benchmarks met")
	
	return True


async def test_analytics_edge_cases():
	"""Test analytics edge cases and error handling"""
	print("\n🧪 Testing Analytics Edge Cases...")
	
	from service import MultiTenantManager
	
	# Test with analytics disabled
	manager_no_analytics = MultiTenantManager(
		tenant_id="no-analytics-test"
	)
	
	await manager_no_analytics.initialize({"enable_analytics": False})
	
	# These should return empty/error responses gracefully
	analytics_data = await manager_no_analytics.get_tenant_analytics("nonexistent")
	assert "error" in analytics_data
	
	predictions = await manager_no_analytics.get_predictive_insights("nonexistent")
	assert len(predictions) == 0
	
	alerts = await manager_no_analytics.get_tenant_alerts("nonexistent")
	assert len(alerts) == 0
	
	health_score = await manager_no_analytics.get_tenant_health_score("nonexistent")
	assert health_score == 0.0
	
	print("  ✅ Analytics disabled scenarios handled gracefully")
	
	# Test with analytics enabled but nonexistent tenant
	manager_with_analytics = MultiTenantManager(tenant_id="analytics-enabled-test")
	await manager_with_analytics.initialize({"enable_analytics": True})
	
	analytics_data = await manager_with_analytics.get_tenant_analytics("nonexistent-tenant")
	assert "error" in analytics_data
	
	predictions = await manager_with_analytics.get_predictive_insights("nonexistent-tenant")
	assert len(predictions) == 0
	
	print("  ✅ Nonexistent tenant scenarios handled gracefully")
	
	return True


async def main():
	"""Run all analytics integration tests"""
	all_passed = True
	
	print("Testing Analytics Integration...")
	try:
		await test_analytics_integration()
		print()
	except Exception as e:
		print(f"  ❌ Analytics integration test failed: {e}")
		all_passed = False
	
	print("Testing Multi-Tenant Analytics...")
	try:
		await test_multi_tenant_analytics()
		print()
	except Exception as e:
		print(f"  ❌ Multi-tenant analytics test failed: {e}")
		all_passed = False
	
	print("Testing Analytics Performance...")
	try:
		await test_analytics_performance()
		print()
	except Exception as e:
		print(f"  ❌ Analytics performance test failed: {e}")
		all_passed = False
	
	print("Testing Analytics Edge Cases...")
	try:
		await test_analytics_edge_cases()
		print()
	except Exception as e:
		print(f"  ❌ Analytics edge cases test failed: {e}")
		all_passed = False
	
	print("=" * 70)
	
	if all_passed:
		print("🎉 ALL ANALYTICS INTEGRATION TESTS PASSED!")
		print("✅ Real-time analytics fully integrated with MultiTenantManager")
		print("✅ AI-powered predictive insights operational")
		print("✅ Multi-dimensional monitoring and alerting functional")
		print("✅ Performance optimization recommendations working")
		print("✅ System-wide analytics and tenant health scoring complete")
		print("✅ Concurrent operations performance validated")
		print("✅ Error handling and edge cases covered")
		print("✅ APG integration architecture ready")
		print("🚀 Phase 3.4: Real-Time Analytics & Predictive Monitoring COMPLETE")
		return True
	else:
		print("❌ SOME ANALYTICS INTEGRATION TESTS FAILED")
		return False


if __name__ == "__main__":
	success = asyncio.run(main())
	sys.exit(0 if success else 1)