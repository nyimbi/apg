#!/usr/bin/env python3
"""
APG Multi-language Localization - Production Validation

Comprehensive production validation suite for the multi-language localization
capability including health checks, performance tests, and integration validation.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
import time
import logging
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
import asyncpg
import aioredis
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation test status"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"


@dataclass
class ValidationResult:
    """Single validation test result"""
    test_name: str
    status: ValidationStatus
    message: str
    duration_ms: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationConfig:
    """Validation configuration"""
    api_base_url: str
    database_url: str
    redis_url: str
    api_key: str
    timeout_seconds: int = 30
    parallel_requests: int = 10
    test_languages: List[str] = None
    
    def __post_init__(self):
        if self.test_languages is None:
            self.test_languages = ["en", "es", "fr", "de", "ja", "ar"]


class ProductionValidator:
    """Production validation orchestrator"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results: List[ValidationResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[aioredis.Redis] = None
    
    async def __aenter__(self):
        """Initialize connections"""
        # HTTP session
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "APG-Localization-Production-Validator/1.0"
        }
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        )
        
        # Database connection pool
        try:
            self.db_pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=5,
                max_size=20,
                command_timeout=30
            )
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
        
        # Redis connection
        try:
            self.redis_client = aioredis.from_url(
                self.config.redis_url,
                retry_on_timeout=True,
                socket_timeout=10,
                socket_connect_timeout=10
            )
            await self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup connections"""
        if self.session:
            await self.session.close()
        
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_client:
            await self.redis_client.close()
    
    def add_result(self, result: ValidationResult):
        """Add validation result"""
        self.results.append(result)
        status_symbol = {
            ValidationStatus.PASS: "✓",
            ValidationStatus.FAIL: "✗",
            ValidationStatus.WARNING: "⚠",
            ValidationStatus.SKIP: "○"
        }
        symbol = status_symbol.get(result.status, "?")
        logger.info(f"{symbol} {result.test_name}: {result.message} ({result.duration_ms:.1f}ms)")
    
    async def run_validation_test(self, test_name: str, test_func):
        """Run a single validation test with error handling"""
        start_time = time.time()
        try:
            await test_func()
            duration_ms = (time.time() - start_time) * 1000
            self.add_result(ValidationResult(
                test_name=test_name,
                status=ValidationStatus.PASS,
                message="Test completed successfully",
                duration_ms=duration_ms
            ))
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.add_result(ValidationResult(
                test_name=test_name,
                status=ValidationStatus.FAIL,
                message=f"Test failed: {str(e)}",
                duration_ms=duration_ms
            ))
    
    async def validate_health_endpoints(self):
        """Validate health check endpoints"""
        endpoints = [
            "/health",
            "/health/ready",
            "/health/live",
            "/status"
        ]
        
        for endpoint in endpoints:
            test_name = f"Health Check: {endpoint}"
            start_time = time.time()
            
            try:
                url = f"{self.config.api_base_url}{endpoint}"
                async with self.session.get(url) as response:
                    duration_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        if data.get("success", False):
                            self.add_result(ValidationResult(
                                test_name=test_name,
                                status=ValidationStatus.PASS,
                                message=f"Health check passed (HTTP {response.status})",
                                duration_ms=duration_ms,
                                details={"response_data": data}
                            ))
                        else:
                            self.add_result(ValidationResult(
                                test_name=test_name,
                                status=ValidationStatus.WARNING,
                                message="Health check returned unsuccessful status",
                                duration_ms=duration_ms
                            ))
                    else:
                        self.add_result(ValidationResult(
                            test_name=test_name,
                            status=ValidationStatus.FAIL,
                            message=f"Health check failed (HTTP {response.status})",
                            duration_ms=duration_ms
                        ))
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.add_result(ValidationResult(
                    test_name=test_name,
                    status=ValidationStatus.FAIL,
                    message=f"Health check error: {str(e)}",
                    duration_ms=duration_ms
                ))
    
    async def validate_database_connectivity(self):
        """Validate database connectivity and basic operations"""
        if not self.db_pool:
            self.add_result(ValidationResult(
                test_name="Database Connectivity",
                status=ValidationStatus.SKIP,
                message="Database connection not available",
                duration_ms=0
            ))
            return
        
        start_time = time.time()
        try:
            async with self.db_pool.acquire() as connection:
                # Test basic query
                result = await connection.fetch("SELECT 1 as test")
                assert result[0]['test'] == 1
                
                # Test table existence
                tables = await connection.fetch("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name LIKE 'ml_%'
                """)
                table_names = [row['table_name'] for row in tables]
                
                expected_tables = [
                    'ml_languages', 'ml_locales', 'ml_namespaces',
                    'ml_translation_keys', 'ml_translations', 'ml_user_preferences'
                ]
                
                missing_tables = [t for t in expected_tables if t not in table_names]
                if missing_tables:
                    raise Exception(f"Missing tables: {missing_tables}")
                
                # Test basic data
                lang_count = await connection.fetchval("SELECT COUNT(*) FROM ml_languages")
                if lang_count == 0:
                    logger.warning("No languages found in database")
                
                duration_ms = (time.time() - start_time) * 1000
                self.add_result(ValidationResult(
                    test_name="Database Connectivity",
                    status=ValidationStatus.PASS,
                    message=f"Database validation passed ({len(table_names)} tables, {lang_count} languages)",
                    duration_ms=duration_ms,
                    details={
                        "tables_found": table_names,
                        "language_count": lang_count
                    }
                ))
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.add_result(ValidationResult(
                test_name="Database Connectivity",
                status=ValidationStatus.FAIL,
                message=f"Database validation failed: {str(e)}",
                duration_ms=duration_ms
            ))
    
    async def validate_redis_connectivity(self):
        """Validate Redis connectivity and operations"""
        if not self.redis_client:
            self.add_result(ValidationResult(
                test_name="Redis Connectivity",
                status=ValidationStatus.SKIP,
                message="Redis connection not available",
                duration_ms=0
            ))
            return
        
        start_time = time.time()
        try:
            # Test ping
            pong = await self.redis_client.ping()
            assert pong is True
            
            # Test set/get
            test_key = f"validation_test_{int(time.time())}"
            test_value = "validation_test_value"
            
            await self.redis_client.set(test_key, test_value, ex=60)
            retrieved_value = await self.redis_client.get(test_key)
            assert retrieved_value.decode() == test_value
            
            # Cleanup
            await self.redis_client.delete(test_key)
            
            # Test info
            info = await self.redis_client.info()
            memory_usage = info.get('used_memory_human', 'unknown')
            
            duration_ms = (time.time() - start_time) * 1000
            self.add_result(ValidationResult(
                test_name="Redis Connectivity",
                status=ValidationStatus.PASS,
                message=f"Redis validation passed (memory: {memory_usage})",
                duration_ms=duration_ms,
                details={"memory_usage": memory_usage}
            ))
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.add_result(ValidationResult(
                test_name="Redis Connectivity",
                status=ValidationStatus.FAIL,
                message=f"Redis validation failed: {str(e)}",
                duration_ms=duration_ms
            ))
    
    async def validate_translation_api(self):
        """Validate core translation API functionality"""
        test_cases = [
            {
                "name": "Simple Translation",
                "key": "common.save",
                "language": "es",
                "expected_type": str
            },
            {
                "name": "Translation with Variables",
                "key": "welcome.message",
                "language": "fr",
                "variables": {"username": "TestUser"},
                "expected_type": str
            },
            {
                "name": "Missing Translation Fallback",
                "key": "nonexistent.key.test.12345",
                "language": "de",
                "expected_type": (str, type(None))
            }
        ]
        
        for case in test_cases:
            start_time = time.time()
            try:
                params = {
                    "key": case["key"],
                    "language": case["language"]
                }
                if "variables" in case:
                    params["variables"] = json.dumps(case["variables"])
                
                url = f"{self.config.api_base_url}/api/v1/translate"
                async with self.session.get(url, params=params) as response:
                    duration_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        translation = data.get("data", {}).get("translation")
                        
                        if isinstance(translation, case["expected_type"]):
                            self.add_result(ValidationResult(
                                test_name=f"Translation API: {case['name']}",
                                status=ValidationStatus.PASS,
                                message=f"Translation returned successfully",
                                duration_ms=duration_ms,
                                details={"translation": translation}
                            ))
                        else:
                            self.add_result(ValidationResult(
                                test_name=f"Translation API: {case['name']}",
                                status=ValidationStatus.FAIL,
                                message=f"Unexpected translation type: {type(translation)}",
                                duration_ms=duration_ms
                            ))
                    elif response.status == 404 and case["name"] == "Missing Translation Fallback":
                        # Expected for missing translations
                        self.add_result(ValidationResult(
                            test_name=f"Translation API: {case['name']}",
                            status=ValidationStatus.PASS,
                            message="Missing translation handled correctly (404)",
                            duration_ms=duration_ms
                        ))
                    else:
                        self.add_result(ValidationResult(
                            test_name=f"Translation API: {case['name']}",
                            status=ValidationStatus.FAIL,
                            message=f"API request failed (HTTP {response.status})",
                            duration_ms=duration_ms
                        ))
            
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.add_result(ValidationResult(
                    test_name=f"Translation API: {case['name']}",
                    status=ValidationStatus.FAIL,
                    message=f"API test error: {str(e)}",
                    duration_ms=duration_ms
                ))
    
    async def validate_batch_translation_api(self):
        """Validate batch translation API"""
        start_time = time.time()
        try:
            payload = {
                "keys": ["common.save", "common.cancel", "common.delete"],
                "language": "es",
                "namespace": "ui"
            }
            
            url = f"{self.config.api_base_url}/api/v1/translate/batch"
            async with self.session.post(url, json=payload) as response:
                duration_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    translations = data.get("data", {}).get("translations", {})
                    
                    if len(translations) > 0:
                        self.add_result(ValidationResult(
                            test_name="Batch Translation API",
                            status=ValidationStatus.PASS,
                            message=f"Batch translation returned {len(translations)} results",
                            duration_ms=duration_ms,
                            details={"translation_count": len(translations)}
                        ))
                    else:
                        self.add_result(ValidationResult(
                            test_name="Batch Translation API",
                            status=ValidationStatus.WARNING,
                            message="Batch translation returned empty results",
                            duration_ms=duration_ms
                        ))
                else:
                    self.add_result(ValidationResult(
                        test_name="Batch Translation API",
                        status=ValidationStatus.FAIL,
                        message=f"Batch API request failed (HTTP {response.status})",
                        duration_ms=duration_ms
                    ))
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.add_result(ValidationResult(
                test_name="Batch Translation API",
                status=ValidationStatus.FAIL,
                message=f"Batch API test error: {str(e)}",
                duration_ms=duration_ms
            ))
    
    async def validate_language_api(self):
        """Validate language management API"""
        start_time = time.time()
        try:
            url = f"{self.config.api_base_url}/api/v1/languages"
            async with self.session.get(url) as response:
                duration_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    languages = data.get("data", {}).get("languages", [])
                    
                    if len(languages) > 0:
                        # Check for required languages
                        language_codes = [lang.get("code") for lang in languages]
                        required_languages = ["en", "es", "fr"]
                        missing_languages = [code for code in required_languages if code not in language_codes]
                        
                        if not missing_languages:
                            self.add_result(ValidationResult(
                                test_name="Language API",
                                status=ValidationStatus.PASS,
                                message=f"Language API returned {len(languages)} languages",
                                duration_ms=duration_ms,
                                details={"language_count": len(languages), "languages": language_codes}
                            ))
                        else:
                            self.add_result(ValidationResult(
                                test_name="Language API",
                                status=ValidationStatus.WARNING,
                                message=f"Missing required languages: {missing_languages}",
                                duration_ms=duration_ms
                            ))
                    else:
                        self.add_result(ValidationResult(
                            test_name="Language API",
                            status=ValidationStatus.FAIL,
                            message="Language API returned no languages",
                            duration_ms=duration_ms
                        ))
                else:
                    self.add_result(ValidationResult(
                        test_name="Language API",
                        status=ValidationStatus.FAIL,
                        message=f"Language API request failed (HTTP {response.status})",
                        duration_ms=duration_ms
                    ))
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.add_result(ValidationResult(
                test_name="Language API",
                status=ValidationStatus.FAIL,
                message=f"Language API test error: {str(e)}",
                duration_ms=duration_ms
            ))
    
    async def validate_performance(self):
        """Validate API performance under load"""
        concurrent_requests = self.config.parallel_requests
        test_languages = self.config.test_languages
        
        async def single_request(session, language):
            """Single performance test request"""
            url = f"{self.config.api_base_url}/api/v1/translate"
            params = {"key": "common.save", "language": language}
            start_time = time.time()
            
            try:
                async with session.get(url, params=params) as response:
                    await response.read()  # Ensure full response is received
                    return time.time() - start_time, response.status
            except Exception as e:
                return time.time() - start_time, None
        
        start_time = time.time()
        try:
            # Create requests for different languages
            tasks = []
            for i in range(concurrent_requests):
                language = test_languages[i % len(test_languages)]
                tasks.append(single_request(self.session, language))
            
            # Execute concurrent requests
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            durations = []
            success_count = 0
            
            for result in results:
                if isinstance(result, tuple):
                    duration, status = result
                    durations.append(duration)
                    if status == 200:
                        success_count += 1
            
            total_duration = time.time() - start_time
            
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                success_rate = success_count / len(results)
                
                if success_rate >= 0.95 and avg_duration < 1.0:
                    status = ValidationStatus.PASS
                    message = f"Performance test passed: {success_rate:.1%} success, {avg_duration:.3f}s avg"
                elif success_rate >= 0.8:
                    status = ValidationStatus.WARNING
                    message = f"Performance degraded: {success_rate:.1%} success, {avg_duration:.3f}s avg"
                else:
                    status = ValidationStatus.FAIL
                    message = f"Performance test failed: {success_rate:.1%} success, {avg_duration:.3f}s avg"
                
                self.add_result(ValidationResult(
                    test_name="Performance Test",
                    status=status,
                    message=message,
                    duration_ms=total_duration * 1000,
                    details={
                        "concurrent_requests": concurrent_requests,
                        "success_rate": success_rate,
                        "avg_duration_s": avg_duration,
                        "max_duration_s": max_duration,
                        "total_duration_s": total_duration
                    }
                ))
            else:
                self.add_result(ValidationResult(
                    test_name="Performance Test",
                    status=ValidationStatus.FAIL,
                    message="No valid performance results",
                    duration_ms=total_duration * 1000
                ))
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.add_result(ValidationResult(
                test_name="Performance Test",
                status=ValidationStatus.FAIL,
                message=f"Performance test error: {str(e)}",
                duration_ms=duration_ms
            ))
    
    async def validate_metrics_endpoint(self):
        """Validate Prometheus metrics endpoint"""
        start_time = time.time()
        try:
            url = f"{self.config.api_base_url}/metrics"
            # Use session without auth headers for metrics
            async with aiohttp.ClientSession() as metrics_session:
                async with metrics_session.get(url) as response:
                    duration_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        content = await response.text()
                        
                        # Check for expected metrics
                        expected_metrics = [
                            "http_requests_total",
                            "http_request_duration_seconds",
                            "translation_cache_hits_total",
                            "translation_requests_total"
                        ]
                        
                        found_metrics = [metric for metric in expected_metrics if metric in content]
                        missing_metrics = [metric for metric in expected_metrics if metric not in content]
                        
                        if len(found_metrics) == len(expected_metrics):
                            self.add_result(ValidationResult(
                                test_name="Metrics Endpoint",
                                status=ValidationStatus.PASS,
                                message=f"All {len(expected_metrics)} expected metrics found",
                                duration_ms=duration_ms,
                                details={"metrics_found": found_metrics}
                            ))
                        else:
                            self.add_result(ValidationResult(
                                test_name="Metrics Endpoint",
                                status=ValidationStatus.WARNING,
                                message=f"Missing metrics: {missing_metrics}",
                                duration_ms=duration_ms,
                                details={"metrics_found": found_metrics, "missing": missing_metrics}
                            ))
                    else:
                        self.add_result(ValidationResult(
                            test_name="Metrics Endpoint",
                            status=ValidationStatus.FAIL,
                            message=f"Metrics endpoint failed (HTTP {response.status})",
                            duration_ms=duration_ms
                        ))
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.add_result(ValidationResult(
                test_name="Metrics Endpoint",
                status=ValidationStatus.FAIL,
                message=f"Metrics test error: {str(e)}",
                duration_ms=duration_ms
            ))
    
    async def run_all_validations(self):
        """Run all validation tests"""
        logger.info("Starting production validation...")
        
        validation_tests = [
            ("Health Endpoints", self.validate_health_endpoints),
            ("Database Connectivity", self.validate_database_connectivity),
            ("Redis Connectivity", self.validate_redis_connectivity),
            ("Translation API", self.validate_translation_api),
            ("Batch Translation API", self.validate_batch_translation_api),
            ("Language API", self.validate_language_api),
            ("Performance Test", self.validate_performance),
            ("Metrics Endpoint", self.validate_metrics_endpoint),
        ]
        
        for test_name, test_func in validation_tests:
            logger.info(f"Running {test_name}...")
            await self.run_validation_test(test_name, test_func)
        
        logger.info("Production validation completed")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == ValidationStatus.PASS])
        failed_tests = len([r for r in self.results if r.status == ValidationStatus.FAIL])
        warning_tests = len([r for r in self.results if r.status == ValidationStatus.WARNING])
        skipped_tests = len([r for r in self.results if r.status == ValidationStatus.SKIP])
        
        avg_duration = sum(r.duration_ms for r in self.results) / total_tests if total_tests > 0 else 0
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        overall_status = "PASS"
        if failed_tests > 0:
            overall_status = "FAIL"
        elif warning_tests > 0:
            overall_status = "WARNING"
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests,
                "skipped": skipped_tests,
                "success_rate": success_rate,
                "average_duration_ms": avg_duration
            },
            "tests": [
                {
                    "name": r.test_name,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "details": r.details
                }
                for r in self.results
            ]
        }


async def main():
    """Main validation runner"""
    parser = argparse.ArgumentParser(description="APG Localization Production Validation")
    parser.add_argument("--api-url", required=True, help="API base URL")
    parser.add_argument("--database-url", required=True, help="Database connection URL")
    parser.add_argument("--redis-url", required=True, help="Redis connection URL")
    parser.add_argument("--api-key", required=True, help="API authentication key")
    parser.add_argument("--output", help="Output report file (JSON)")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--parallel", type=int, default=10, help="Parallel requests for performance test")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "fr", "de", "ja", "ar"], help="Test languages")
    
    args = parser.parse_args()
    
    config = ValidationConfig(
        api_base_url=args.api_url,
        database_url=args.database_url,
        redis_url=args.redis_url,
        api_key=args.api_key,
        timeout_seconds=args.timeout,
        parallel_requests=args.parallel,
        test_languages=args.languages
    )
    
    async with ProductionValidator(config) as validator:
        await validator.run_all_validations()
        
        report = validator.generate_report()
        
        # Print summary
        print("\n" + "="*60)
        print("PRODUCTION VALIDATION REPORT")
        print("="*60)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Warnings: {report['summary']['warnings']}")
        print(f"Skipped: {report['summary']['skipped']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"Average Duration: {report['summary']['average_duration_ms']:.1f}ms")
        print("="*60)
        
        # Save report
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {args.output}")
        
        # Exit with appropriate code
        if report['overall_status'] == "FAIL":
            return 1
        elif report['overall_status'] == "WARNING":
            return 2
        else:
            return 0


if __name__ == "__main__":
    import sys
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)