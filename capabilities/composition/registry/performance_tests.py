"""
APG Capability Registry - Performance Tests

Comprehensive performance testing suite to validate production readiness
and identify optimization opportunities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
import pytest
from uuid_extensions import uuid7str

# =============================================================================
# Performance Test Configuration
# =============================================================================

@dataclass
class PerformanceConfig:
    """Performance testing configuration."""
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 50
    test_duration_seconds: int = 60
    ramp_up_seconds: int = 10
    think_time_ms: int = 100
    timeout_seconds: int = 30
    
    # Performance thresholds
    max_response_time_ms: int = 1000
    max_95th_percentile_ms: int = 2000
    min_throughput_rps: float = 10.0
    max_error_rate: float = 0.05  # 5%

@dataclass
class PerformanceResult:
    """Performance test result."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float]
    errors: List[str]
    start_time: datetime
    end_time: datetime
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def throughput_rps(self) -> float:
        return self.total_requests / self.duration_seconds if self.duration_seconds > 0 else 0
    
    @property
    def error_rate(self) -> float:
        return self.failed_requests / self.total_requests if self.total_requests > 0 else 0
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        return statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else 0
    
    @property
    def p99_response_time(self) -> float:
        return statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else 0

# =============================================================================
# Performance Test Framework
# =============================================================================

class PerformanceTester:
    """Performance testing framework."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.results: List[PerformanceResult] = []
    
    async def _make_request(
        self, 
        client: httpx.AsyncClient, 
        method: str, 
        url: str, 
        **kwargs
    ) -> tuple[bool, float, str]:
        """Make HTTP request and measure response time."""
        start_time = time.time()
        try:
            response = await client.request(method, url, timeout=self.config.timeout_seconds, **kwargs)
            response.raise_for_status()
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            return True, response_time, ""
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return False, response_time, str(e)
    
    async def _user_scenario(self, user_id: int, scenario_func) -> PerformanceResult:
        """Execute user scenario."""
        async with httpx.AsyncClient(base_url=self.config.base_url) as client:
            return await scenario_func(client, user_id)
    
    async def run_load_test(self, scenario_func, test_name: str) -> PerformanceResult:
        """Run load test with multiple concurrent users."""
        print(f"Starting load test: {test_name}")
        print(f"Concurrent users: {self.config.concurrent_users}")
        print(f"Duration: {self.config.test_duration_seconds}s")
        print(f"Ramp-up: {self.config.ramp_up_seconds}s")
        
        start_time = datetime.utcnow()
        
        # Create tasks for concurrent users
        tasks = []
        for user_id in range(self.config.concurrent_users):
            # Stagger user start times for ramp-up
            delay = (user_id / self.config.concurrent_users) * self.config.ramp_up_seconds
            task = asyncio.create_task(self._delayed_user_scenario(delay, user_id, scenario_func))
            tasks.append(task)
        
        # Wait for all users to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.utcnow()
        
        # Aggregate results
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        all_response_times = []
        all_errors = []
        
        for result in results:
            if isinstance(result, PerformanceResult):
                total_requests += result.total_requests
                successful_requests += result.successful_requests
                failed_requests += result.failed_requests
                all_response_times.extend(result.response_times)
                all_errors.extend(result.errors)
            else:
                print(f"Task failed with exception: {result}")
                all_errors.append(str(result))
        
        final_result = PerformanceResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            response_times=all_response_times,
            errors=all_errors,
            start_time=start_time,
            end_time=end_time
        )
        
        self.results.append(final_result)
        self._print_results(final_result)
        
        return final_result
    
    async def _delayed_user_scenario(self, delay: float, user_id: int, scenario_func) -> PerformanceResult:
        """Execute user scenario with initial delay."""
        await asyncio.sleep(delay)
        return await self._user_scenario(user_id, scenario_func)
    
    def _print_results(self, result: PerformanceResult):
        """Print test results."""
        print(f"\n=== {result.test_name} Results ===")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful: {result.successful_requests}")
        print(f"Failed: {result.failed_requests}")
        print(f"Error Rate: {result.error_rate:.2%}")
        print(f"Throughput: {result.throughput_rps:.2f} RPS")
        print(f"Avg Response Time: {result.avg_response_time:.2f}ms")
        print(f"95th Percentile: {result.p95_response_time:.2f}ms")
        print(f"99th Percentile: {result.p99_response_time:.2f}ms")
        
        # Check thresholds
        if result.error_rate > self.config.max_error_rate:
            print(f"❌ Error rate {result.error_rate:.2%} exceeds threshold {self.config.max_error_rate:.2%}")
        
        if result.p95_response_time > self.config.max_95th_percentile_ms:
            print(f"❌ 95th percentile {result.p95_response_time:.2f}ms exceeds threshold {self.config.max_95th_percentile_ms}ms")
        
        if result.throughput_rps < self.config.min_throughput_rps:
            print(f"❌ Throughput {result.throughput_rps:.2f} RPS below threshold {self.config.min_throughput_rps} RPS")
        
        if len(result.errors) > 0:
            print(f"\nFirst 5 errors:")
            for error in result.errors[:5]:
                print(f"  - {error}")

# =============================================================================
# Test Scenarios
# =============================================================================

async def capability_crud_scenario(client: httpx.AsyncClient, user_id: int) -> PerformanceResult:
    """Test scenario for capability CRUD operations."""
    response_times = []
    errors = []
    successful_requests = 0
    failed_requests = 0
    
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(seconds=30)  # Run for 30 seconds
    
    while datetime.utcnow() < end_time:
        # Create capability
        capability_data = {
            "capability_code": f"PERF_TEST_{user_id}_{uuid7str()[:8]}",
            "capability_name": f"Performance Test Capability {user_id}",
            "description": f"Test capability created by user {user_id}",
            "category": "foundation_infrastructure",
            "version": "1.0.0",
            "multi_tenant": True,
            "audit_enabled": True
        }
        
        success, response_time, error = await PerformanceTester(PerformanceConfig())._make_request(
            client, "POST", "/api/capabilities", json=capability_data
        )
        
        response_times.append(response_time)
        if success:
            successful_requests += 1
        else:
            failed_requests += 1
            errors.append(f"Create capability: {error}")
        
        # Get capabilities list
        success, response_time, error = await PerformanceTester(PerformanceConfig())._make_request(
            client, "GET", "/api/capabilities", params={"page": 1, "per_page": 10}
        )
        
        response_times.append(response_time)
        if success:
            successful_requests += 1
        else:
            failed_requests += 1
            errors.append(f"List capabilities: {error}")
        
        # Search capabilities
        success, response_time, error = await PerformanceTester(PerformanceConfig())._make_request(
            client, "GET", "/api/capabilities", params={"search": "performance", "page": 1, "per_page": 5}
        )
        
        response_times.append(response_time)
        if success:
            successful_requests += 1
        else:
            failed_requests += 1
            errors.append(f"Search capabilities: {error}")
        
        # Small delay between operations
        await asyncio.sleep(0.1)
    
    return PerformanceResult(
        test_name=f"Capability CRUD User {user_id}",
        total_requests=successful_requests + failed_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        response_times=response_times,
        errors=errors,
        start_time=start_time,
        end_time=datetime.utcnow()
    )

async def composition_validation_scenario(client: httpx.AsyncClient, user_id: int) -> PerformanceResult:
    """Test scenario for composition validation."""
    response_times = []
    errors = []
    successful_requests = 0
    failed_requests = 0
    
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(seconds=20)  # Run for 20 seconds
    
    # Create some test capabilities first
    capability_ids = []
    for i in range(3):
        capability_data = {
            "capability_code": f"COMP_TEST_{user_id}_{i}",
            "capability_name": f"Composition Test Capability {user_id}-{i}",
            "description": f"Test capability for composition by user {user_id}",
            "category": "foundation_infrastructure",
            "version": "1.0.0"
        }
        
        success, response_time, error = await PerformanceTester(PerformanceConfig())._make_request(
            client, "POST", "/api/capabilities", json=capability_data
        )
        
        if success:
            # In a real scenario, we'd extract the capability_id from response
            capability_ids.append(f"cap_{user_id}_{i}")
    
    while datetime.utcnow() < end_time:
        # Validate composition
        success, response_time, error = await PerformanceTester(PerformanceConfig())._make_request(
            client, "POST", "/api/compositions/validate", json=capability_ids
        )
        
        response_times.append(response_time)
        if success:
            successful_requests += 1
        else:
            failed_requests += 1
            errors.append(f"Validate composition: {error}")
        
        # Create composition
        composition_data = {
            "name": f"Performance Test Composition {user_id}",
            "description": f"Test composition by user {user_id}",
            "composition_type": "custom",
            "capability_ids": capability_ids,
            "business_requirements": ["testing", "performance"],
            "target_users": ["developers"]
        }
        
        success, response_time, error = await PerformanceTester(PerformanceConfig())._make_request(
            client, "POST", "/api/compositions", json=composition_data
        )
        
        response_times.append(response_time)
        if success:
            successful_requests += 1
        else:
            failed_requests += 1
            errors.append(f"Create composition: {error}")
        
        await asyncio.sleep(0.2)
    
    return PerformanceResult(
        test_name=f"Composition Validation User {user_id}",
        total_requests=successful_requests + failed_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        response_times=response_times,
        errors=errors,
        start_time=start_time,
        end_time=datetime.utcnow()
    )

async def analytics_dashboard_scenario(client: httpx.AsyncClient, user_id: int) -> PerformanceResult:
    """Test scenario for analytics dashboard."""
    response_times = []
    errors = []
    successful_requests = 0
    failed_requests = 0
    
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(seconds=15)  # Run for 15 seconds
    
    while datetime.utcnow() < end_time:
        # Dashboard data
        success, response_time, error = await PerformanceTester(PerformanceConfig())._make_request(
            client, "GET", "/api/registry/dashboard"
        )
        
        response_times.append(response_time)
        if success:
            successful_requests += 1
        else:
            failed_requests += 1
            errors.append(f"Dashboard data: {error}")
        
        # Usage analytics
        success, response_time, error = await PerformanceTester(PerformanceConfig())._make_request(
            client, "GET", "/api/analytics/usage"
        )
        
        response_times.append(response_time)
        if success:
            successful_requests += 1
        else:
            failed_requests += 1
            errors.append(f"Usage analytics: {error}")
        
        # Performance analytics
        success, response_time, error = await PerformanceTester(PerformanceConfig())._make_request(
            client, "GET", "/api/analytics/performance", params={"time_range": "7d"}
        )
        
        response_times.append(response_time)
        if success:
            successful_requests += 1
        else:
            failed_requests += 1
            errors.append(f"Performance analytics: {error}")
        
        await asyncio.sleep(0.5)
    
    return PerformanceResult(
        test_name=f"Analytics Dashboard User {user_id}",
        total_requests=successful_requests + failed_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        response_times=response_times,
        errors=errors,
        start_time=start_time,
        end_time=datetime.utcnow()
    )

# =============================================================================
# Performance Test Suite
# =============================================================================

class PerformanceTestSuite:
    """Complete performance test suite."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.tester = PerformanceTester(self.config)
    
    async def run_all_tests(self) -> Dict[str, PerformanceResult]:
        """Run all performance tests."""
        print("🚀 Starting APG Capability Registry Performance Test Suite")
        print(f"Base URL: {self.config.base_url}")
        print(f"Concurrent Users: {self.config.concurrent_users}")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Capability CRUD Operations
        print("\n📊 Test 1: Capability CRUD Operations")
        result = await self.tester.run_load_test(capability_crud_scenario, "Capability CRUD")
        results["capability_crud"] = result
        
        # Test 2: Composition Validation
        print("\n🔗 Test 2: Composition Validation")
        result = await self.tester.run_load_test(composition_validation_scenario, "Composition Validation")
        results["composition_validation"] = result
        
        # Test 3: Analytics Dashboard
        print("\n📈 Test 3: Analytics Dashboard")
        result = await self.tester.run_load_test(analytics_dashboard_scenario, "Analytics Dashboard")
        results["analytics_dashboard"] = result
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, PerformanceResult]):
        """Print performance test summary."""
        print("\n" + "=" * 60)
        print("📋 PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        
        total_requests = sum(r.total_requests for r in results.values())
        total_successful = sum(r.successful_requests for r in results.values())
        total_failed = sum(r.failed_requests for r in results.values())
        overall_error_rate = total_failed / total_requests if total_requests > 0 else 0
        
        print(f"Total Requests: {total_requests}")
        print(f"Successful: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Overall Error Rate: {overall_error_rate:.2%}")
        
        print(f"\nPer-Test Results:")
        for test_name, result in results.items():
            status = "✅" if self._meets_thresholds(result) else "❌"
            print(f"{status} {test_name}: {result.throughput_rps:.1f} RPS, "
                  f"{result.p95_response_time:.0f}ms P95, {result.error_rate:.2%} errors")
        
        # Overall assessment
        all_passed = all(self._meets_thresholds(r) for r in results.values())
        if all_passed:
            print(f"\n🎉 ALL PERFORMANCE TESTS PASSED")
        else:
            print(f"\n⚠️  SOME PERFORMANCE TESTS FAILED")
            print("Consider optimizing the application or adjusting thresholds.")
    
    def _meets_thresholds(self, result: PerformanceResult) -> bool:
        """Check if result meets performance thresholds."""
        return (
            result.error_rate <= self.config.max_error_rate and
            result.p95_response_time <= self.config.max_95th_percentile_ms and
            result.throughput_rps >= self.config.min_throughput_rps
        )

# =============================================================================
# Main Performance Test Runner
# =============================================================================

async def main():
    """Main performance test runner."""
    # Configure performance tests
    config = PerformanceConfig(
        base_url="http://localhost:8000",
        concurrent_users=20,  # Start with lower number for initial testing
        test_duration_seconds=30,
        ramp_up_seconds=5,
        max_response_time_ms=1000,
        max_95th_percentile_ms=2000,
        min_throughput_rps=5.0,
        max_error_rate=0.05
    )
    
    # Run performance test suite
    suite = PerformanceTestSuite(config)
    results = await suite.run_all_tests()
    
    # Save results to file
    import json
    with open("performance_test_results.json", "w") as f:
        json.dump({
            name: {
                "test_name": result.test_name,
                "total_requests": result.total_requests,
                "successful_requests": result.successful_requests,
                "failed_requests": result.failed_requests,
                "error_rate": result.error_rate,
                "throughput_rps": result.throughput_rps,
                "avg_response_time": result.avg_response_time,
                "p95_response_time": result.p95_response_time,
                "p99_response_time": result.p99_response_time,
                "duration_seconds": result.duration_seconds
            }
            for name, result in results.items()
        }, f, indent=2)
    
    print(f"\n💾 Results saved to performance_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())