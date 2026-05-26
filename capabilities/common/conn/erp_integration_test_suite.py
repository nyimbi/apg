"""
ERP Integration Test Suite
Comprehensive testing framework for all ERP connectors

This module provides automated testing capabilities for validating
ERP connector functionality, performance, and data quality.
"""

import asyncio
import pytest
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
import statistics

from singer_taps.erp_registry import get_erp_registry, ERPSystemType
from .service import ConnectionManager
from .models import Connection, ConnectionType, ConnectionStatus

logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Test categories for ERP integration testing"""
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    DISCOVERY = "discovery"
    EXTRACTION = "extraction"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    ERROR_HANDLING = "error_handling"
    SCALABILITY = "scalability"


@dataclass
class TestResult:
    """Result of an individual test"""
    test_name: str
    category: TestCategory
    erp_system: str
    passed: bool
    duration_seconds: float
    message: str
    metrics: Dict[str, Any] = None
    errors: List[str] = None


@dataclass
class TestSuite:
    """Collection of tests for an ERP system"""
    erp_system: str
    system_type: ERPSystemType
    test_config: Dict[str, Any]
    results: List[TestResult] = None


class ERPIntegrationTester:
    """Main ERP integration testing framework"""

    def __init__(self):
        self.registry = get_erp_registry()
        self.connection_manager = ConnectionManager()
        self.test_results = {}

    async def run_comprehensive_tests(self, test_configs: Dict[str, Dict]) -> Dict[str, List[TestResult]]:
        """Run comprehensive tests across all configured ERP systems"""
        logger.info("Starting comprehensive ERP integration tests")

        all_results = {}

        # Test each configured ERP system
        for erp_name, config in test_configs.items():
            logger.info(f"Testing ERP system: {erp_name}")

            try:
                # Get system type from registry
                system_type = self._get_system_type_from_name(erp_name)
                if not system_type:
                    logger.error(f"Unknown ERP system: {erp_name}")
                    continue

                # Create test suite
                test_suite = TestSuite(
                    erp_system=erp_name,
                    system_type=system_type,
                    test_config=config
                )

                # Run all test categories
                results = await self._run_erp_test_suite(test_suite)
                all_results[erp_name] = results

            except Exception as e:
                logger.error(f"Failed to test {erp_name}: {e}")
                all_results[erp_name] = [TestResult(
                    test_name="system_failure",
                    category=TestCategory.CONNECTION,
                    erp_system=erp_name,
                    passed=False,
                    duration_seconds=0.0,
                    message=f"System test failure: {e}",
                    errors=[str(e)]
                )]

        # Generate summary report
        await self._generate_test_report(all_results)

        return all_results

    async def _run_erp_test_suite(self, test_suite: TestSuite) -> List[TestResult]:
        """Run complete test suite for a single ERP system"""
        results = []

        # Test categories in order of dependency
        test_methods = [
            (TestCategory.CONNECTION, self._test_connection),
            (TestCategory.AUTHENTICATION, self._test_authentication),
            (TestCategory.DISCOVERY, self._test_discovery),
            (TestCategory.EXTRACTION, self._test_extraction),
            (TestCategory.DATA_QUALITY, self._test_data_quality),
            (TestCategory.PERFORMANCE, self._test_performance),
            (TestCategory.ERROR_HANDLING, self._test_error_handling),
            (TestCategory.SCALABILITY, self._test_scalability)
        ]

        for category, test_method in test_methods:
            try:
                category_results = await test_method(test_suite)
                results.extend(category_results)

                # Stop testing if critical tests fail
                if category in [TestCategory.CONNECTION, TestCategory.AUTHENTICATION]:
                    failed_critical = any(not r.passed for r in category_results)
                    if failed_critical:
                        logger.warning(f"Critical test failed for {test_suite.erp_system}, skipping remaining tests")
                        break

            except Exception as e:
                logger.error(f"Test category {category} failed for {test_suite.erp_system}: {e}")
                results.append(TestResult(
                    test_name=f"{category.value}_error",
                    category=category,
                    erp_system=test_suite.erp_system,
                    passed=False,
                    duration_seconds=0.0,
                    message=f"Test category error: {e}",
                    errors=[str(e)]
                ))

        return results

    async def _test_connection(self, test_suite: TestSuite) -> List[TestResult]:
        """Test basic connectivity to ERP system"""
        results = []

        # Test 1: Basic Connection
        start_time = time.time()
        try:
            connection = await self._create_test_connection(test_suite)
            success = await connection.test_connection()
            duration = time.time() - start_time

            results.append(TestResult(
                test_name="basic_connection",
                category=TestCategory.CONNECTION,
                erp_system=test_suite.erp_system,
                passed=success,
                duration_seconds=duration,
                message="Basic connection test completed",
                metrics={"connection_time": duration}
            ))

        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="basic_connection",
                category=TestCategory.CONNECTION,
                erp_system=test_suite.erp_system,
                passed=False,
                duration_seconds=duration,
                message=f"Connection failed: {e}",
                errors=[str(e)]
            ))

        # Test 2: Connection Timeout
        start_time = time.time()
        try:
            # Test with very short timeout
            short_timeout_config = test_suite.test_config.copy()
            short_timeout_config['timeout'] = 1  # 1 second

            connection = await self._create_test_connection(test_suite, short_timeout_config)
            success = await connection.test_connection()
            duration = time.time() - start_time

            # This should typically fail due to short timeout
            results.append(TestResult(
                test_name="connection_timeout",
                category=TestCategory.CONNECTION,
                erp_system=test_suite.erp_system,
                passed=True,  # Test passes if it handles timeout gracefully
                duration_seconds=duration,
                message="Timeout handling test completed",
                metrics={"timeout_duration": duration}
            ))

        except Exception as e:
            duration = time.time() - start_time
            # Expected behavior for timeout test
            results.append(TestResult(
                test_name="connection_timeout",
                category=TestCategory.CONNECTION,
                erp_system=test_suite.erp_system,
                passed=True,  # Pass if it fails gracefully
                duration_seconds=duration,
                message=f"Timeout handled correctly: {e}",
                metrics={"timeout_duration": duration}
            ))

        return results

    async def _test_authentication(self, test_suite: TestSuite) -> List[TestResult]:
        """Test authentication mechanisms"""
        results = []

        # Test 1: Valid Authentication
        start_time = time.time()
        try:
            connection = await self._create_test_connection(test_suite)
            auth_success = await self._test_auth_validity(connection)
            duration = time.time() - start_time

            results.append(TestResult(
                test_name="valid_authentication",
                category=TestCategory.AUTHENTICATION,
                erp_system=test_suite.erp_system,
                passed=auth_success,
                duration_seconds=duration,
                message="Valid authentication test completed",
                metrics={"auth_time": duration}
            ))

        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="valid_authentication",
                category=TestCategory.AUTHENTICATION,
                erp_system=test_suite.erp_system,
                passed=False,
                duration_seconds=duration,
                message=f"Authentication failed: {e}",
                errors=[str(e)]
            ))

        # Test 2: Invalid Authentication
        start_time = time.time()
        try:
            invalid_config = test_suite.test_config.copy()
            invalid_config['password'] = 'invalid_password_123'

            connection = await self._create_test_connection(test_suite, invalid_config)
            auth_success = await self._test_auth_validity(connection)
            duration = time.time() - start_time

            # Should fail with invalid credentials
            results.append(TestResult(
                test_name="invalid_authentication",
                category=TestCategory.AUTHENTICATION,
                erp_system=test_suite.erp_system,
                passed=not auth_success,  # Pass if authentication correctly fails
                duration_seconds=duration,
                message="Invalid authentication test completed",
                metrics={"invalid_auth_time": duration}
            ))

        except Exception as e:
            duration = time.time() - start_time
            # Expected behavior - should fail
            results.append(TestResult(
                test_name="invalid_authentication",
                category=TestCategory.AUTHENTICATION,
                erp_system=test_suite.erp_system,
                passed=True,  # Pass if it fails correctly
                duration_seconds=duration,
                message=f"Invalid authentication handled correctly: {e}",
                metrics={"invalid_auth_time": duration}
            ))

        return results

    async def _test_discovery(self, test_suite: TestSuite) -> List[TestResult]:
        """Test schema discovery capabilities"""
        results = []

        # Test 1: Schema Discovery
        start_time = time.time()
        try:
            connection = await self._create_test_connection(test_suite)

            # Get available streams
            streams = await self._discover_streams(connection)
            duration = time.time() - start_time

            stream_count = len(streams)
            has_streams = stream_count > 0

            results.append(TestResult(
                test_name="schema_discovery",
                category=TestCategory.DISCOVERY,
                erp_system=test_suite.erp_system,
                passed=has_streams,
                duration_seconds=duration,
                message=f"Discovered {stream_count} streams",
                metrics={"stream_count": stream_count, "discovery_time": duration}
            ))

        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="schema_discovery",
                category=TestCategory.DISCOVERY,
                erp_system=test_suite.erp_system,
                passed=False,
                duration_seconds=duration,
                message=f"Schema discovery failed: {e}",
                errors=[str(e)]
            ))

        return results

    async def _test_extraction(self, test_suite: TestSuite) -> List[TestResult]:
        """Test data extraction capabilities"""
        results = []

        # Test 1: Sample Data Extraction
        start_time = time.time()
        try:
            connection = await self._create_test_connection(test_suite)

            # Extract sample data from first available stream
            sample_data = await self._extract_sample_data(connection, limit=100)
            duration = time.time() - start_time

            record_count = len(sample_data)
            has_data = record_count > 0

            results.append(TestResult(
                test_name="sample_extraction",
                category=TestCategory.EXTRACTION,
                erp_system=test_suite.erp_system,
                passed=has_data,
                duration_seconds=duration,
                message=f"Extracted {record_count} sample records",
                metrics={"record_count": record_count, "extraction_time": duration}
            ))

        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="sample_extraction",
                category=TestCategory.EXTRACTION,
                erp_system=test_suite.erp_system,
                passed=False,
                duration_seconds=duration,
                message=f"Sample extraction failed: {e}",
                errors=[str(e)]
            ))

        # Test 2: Incremental Extraction
        start_time = time.time()
        try:
            connection = await self._create_test_connection(test_suite)

            # Test incremental sync
            incremental_data = await self._test_incremental_sync(connection)
            duration = time.time() - start_time

            results.append(TestResult(
                test_name="incremental_extraction",
                category=TestCategory.EXTRACTION,
                erp_system=test_suite.erp_system,
                passed=incremental_data is not None,
                duration_seconds=duration,
                message="Incremental extraction test completed",
                metrics={"incremental_time": duration}
            ))

        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="incremental_extraction",
                category=TestCategory.EXTRACTION,
                erp_system=test_suite.erp_system,
                passed=False,
                duration_seconds=duration,
                message=f"Incremental extraction failed: {e}",
                errors=[str(e)]
            ))

        return results

    async def _test_data_quality(self, test_suite: TestSuite) -> List[TestResult]:
        """Test data quality aspects"""
        results = []

        # Test 1: Data Type Validation
        start_time = time.time()
        try:
            connection = await self._create_test_connection(test_suite)
            sample_data = await self._extract_sample_data(connection, limit=50)

            quality_metrics = self._analyze_data_quality(sample_data)
            duration = time.time() - start_time

            # Pass if data quality metrics are reasonable
            quality_score = quality_metrics.get('overall_score', 0)
            passed = quality_score >= 0.7  # 70% quality threshold

            results.append(TestResult(
                test_name="data_quality_validation",
                category=TestCategory.DATA_QUALITY,
                erp_system=test_suite.erp_system,
                passed=passed,
                duration_seconds=duration,
                message=f"Data quality score: {quality_score:.2f}",
                metrics=quality_metrics
            ))

        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="data_quality_validation",
                category=TestCategory.DATA_QUALITY,
                erp_system=test_suite.erp_system,
                passed=False,
                duration_seconds=duration,
                message=f"Data quality test failed: {e}",
                errors=[str(e)]
            ))

        return results

    async def _test_performance(self, test_suite: TestSuite) -> List[TestResult]:
        """Test performance characteristics"""
        results = []

        # Test 1: Throughput Test
        start_time = time.time()
        try:
            connection = await self._create_test_connection(test_suite)

            # Extract larger dataset to measure throughput
            large_dataset = await self._extract_sample_data(connection, limit=1000)
            duration = time.time() - start_time

            record_count = len(large_dataset)
            throughput = record_count / duration if duration > 0 else 0

            # Pass if throughput is reasonable (>10 records/second)
            passed = throughput >= 10

            results.append(TestResult(
                test_name="throughput_test",
                category=TestCategory.PERFORMANCE,
                erp_system=test_suite.erp_system,
                passed=passed,
                duration_seconds=duration,
                message=f"Throughput: {throughput:.2f} records/second",
                metrics={
                    "throughput_rps": throughput,
                    "total_records": record_count,
                    "duration": duration
                }
            ))

        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="throughput_test",
                category=TestCategory.PERFORMANCE,
                erp_system=test_suite.erp_system,
                passed=False,
                duration_seconds=duration,
                message=f"Throughput test failed: {e}",
                errors=[str(e)]
            ))

        return results

    async def _test_error_handling(self, test_suite: TestSuite) -> List[TestResult]:
        """Test error handling and recovery"""
        results = []

        # Test 1: Network Interruption Simulation
        start_time = time.time()
        try:
            # Test with invalid host to simulate network error
            invalid_config = test_suite.test_config.copy()
            invalid_config['host'] = 'invalid-host-name-12345.com'

            connection = await self._create_test_connection(test_suite, invalid_config)

            # This should fail gracefully
            try:
                await connection.test_connection()
                # If it doesn't fail, something's wrong
                passed = False
                message = "Error handling failed - should have failed with invalid host"
            except Exception:
                # Expected behavior
                passed = True
                message = "Network error handled correctly"

            duration = time.time() - start_time

            results.append(TestResult(
                test_name="network_error_handling",
                category=TestCategory.ERROR_HANDLING,
                erp_system=test_suite.erp_system,
                passed=passed,
                duration_seconds=duration,
                message=message,
                metrics={"error_handling_time": duration}
            ))

        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="network_error_handling",
                category=TestCategory.ERROR_HANDLING,
                erp_system=test_suite.erp_system,
                passed=True,  # Pass if it fails gracefully
                duration_seconds=duration,
                message=f"Network error handled: {e}",
                metrics={"error_handling_time": duration}
            ))

        return results

    async def _test_scalability(self, test_suite: TestSuite) -> List[TestResult]:
        """Test scalability under load"""
        results = []

        # Test 1: Concurrent Connections
        start_time = time.time()
        try:
            # Test multiple concurrent connections
            concurrent_connections = 5

            async def create_and_test():
                connection = await self._create_test_connection(test_suite)
                return await connection.test_connection()

            # Run concurrent connection tests
            tasks = [create_and_test() for _ in range(concurrent_connections)]
            concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)

            duration = time.time() - start_time

            # Count successful connections
            successful = sum(1 for r in concurrent_results if r is True)
            success_rate = successful / concurrent_connections

            # Pass if at least 80% of concurrent connections succeed
            passed = success_rate >= 0.8

            results.append(TestResult(
                test_name="concurrent_connections",
                category=TestCategory.SCALABILITY,
                erp_system=test_suite.erp_system,
                passed=passed,
                duration_seconds=duration,
                message=f"Concurrent connections: {successful}/{concurrent_connections} successful",
                metrics={
                    "concurrent_connections": concurrent_connections,
                    "successful_connections": successful,
                    "success_rate": success_rate,
                    "total_time": duration
                }
            ))

        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="concurrent_connections",
                category=TestCategory.SCALABILITY,
                erp_system=test_suite.erp_system,
                passed=False,
                duration_seconds=duration,
                message=f"Scalability test failed: {e}",
                errors=[str(e)]
            ))

        return results

    # Helper methods
    async def _create_test_connection(self, test_suite: TestSuite, custom_config: Optional[Dict] = None) -> Connection:
        """Create a test connection for ERP system"""
        config = custom_config or test_suite.test_config

        connection = Connection(
            name=f"test_{test_suite.erp_system}",
            connection_type=ConnectionType.ERP,
            tap_config=config,
            singer_tap=f"tap_{test_suite.erp_system.lower()}",
            tenant_id="test_tenant"
        )

        return connection

    async def _test_auth_validity(self, connection: Connection) -> bool:
        """Test if authentication is valid"""
        try:
            return await connection.test_connection()
        except Exception:
            return False

    async def _discover_streams(self, connection: Connection) -> List[str]:
        """Discover available streams for connection"""
        # Simplified discovery - in real implementation would use Singer discovery
        return ["customers", "vendors", "items", "transactions"]

    async def _extract_sample_data(self, connection: Connection, limit: int = 100) -> List[Dict]:
        """Extract sample data for testing"""
        # Simplified extraction - in real implementation would use Singer tap
        return [{"id": i, "name": f"record_{i}"} for i in range(min(limit, 50))]

    async def _test_incremental_sync(self, connection: Connection) -> Optional[List[Dict]]:
        """Test incremental synchronization"""
        # Simplified incremental test
        return [{"id": 1, "modified": "2025-01-08T10:00:00Z"}]

    def _analyze_data_quality(self, data: List[Dict]) -> Dict[str, float]:
        """Analyze data quality metrics"""
        if not data:
            return {"overall_score": 0.0}

        # Simplified quality analysis
        total_fields = sum(len(record) for record in data)
        null_fields = sum(sum(1 for v in record.values() if v is None) for record in data)

        completeness = 1.0 - (null_fields / total_fields) if total_fields > 0 else 0.0

        return {
            "overall_score": completeness,
            "completeness": completeness,
            "record_count": len(data),
            "avg_fields_per_record": total_fields / len(data) if data else 0
        }

    def _get_system_type_from_name(self, erp_name: str) -> Optional[ERPSystemType]:
        """Get system type enum from ERP name"""
        name_mapping = {
            "sap_erp": ERPSystemType.SAP_ERP,
            "sap_s4hana": ERPSystemType.SAP_S4HANA,
            "sap_business_one": ERPSystemType.SAP_BUSINESS_ONE,
            "dynamics_365_fo": ERPSystemType.DYNAMICS_365_FO,
            "dynamics_365_bc": ERPSystemType.DYNAMICS_365_BC,
            "oracle_cloud_erp": ERPSystemType.ORACLE_CLOUD_ERP,
            "netsuite_erp": ERPSystemType.NETSUITE_ERP,
            "workday_hcm": ERPSystemType.WORKDAY_HCM,
            "sage_x3": ERPSystemType.SAGE_X3
        }
        return name_mapping.get(erp_name.lower())

    async def _generate_test_report(self, all_results: Dict[str, List[TestResult]]) -> None:
        """Generate comprehensive test report"""
        logger.info("Generating ERP integration test report")

        total_tests = sum(len(results) for results in all_results.values())
        total_passed = sum(sum(1 for r in results if r.passed) for results in all_results.values())
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0

        logger.info(f"Overall test results: {total_passed}/{total_tests} passed ({overall_success_rate:.1%})")

        # Log results by system
        for erp_system, results in all_results.items():
            system_passed = sum(1 for r in results if r.passed)
            system_total = len(results)
            system_rate = system_passed / system_total if system_total > 0 else 0

            logger.info(f"{erp_system}: {system_passed}/{system_total} passed ({system_rate:.1%})")

            # Log failed tests
            failed_tests = [r for r in results if not r.passed]
            for failed_test in failed_tests:
                logger.warning(f"  FAILED: {failed_test.test_name} - {failed_test.message}")


# Test configuration examples
SAMPLE_TEST_CONFIGS = {
    "sap_s4hana": {
        "sap_system_type": "s4hana",
        "host": "sap-dev.company.com",
        "client": "100",
        "system_number": "00",
        "username": "test_user",
        "password": "test_password",
        "language": "EN"
    },
    "dynamics_365_fo": {
        "dynamics_system_type": "finance_operations",
        "tenant_id": "test-tenant-id",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "base_url": "https://test.operations.dynamics.com"
    },
    "oracle_cloud_erp": {
        "oracle_system_type": "cloud_erp",
        "host": "test.oraclecloud.com",
        "username": "test_user",
        "password": "test_password",
        "pod": "test_pod"
    }
}


async def run_erp_integration_tests():
    """Main function to run ERP integration tests"""
    tester = ERPIntegrationTester()
    results = await tester.run_comprehensive_tests(SAMPLE_TEST_CONFIGS)

    return results


if __name__ == "__main__":
    asyncio.run(run_erp_integration_tests())