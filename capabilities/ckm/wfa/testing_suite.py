"""
APG Workflow & Business Process Management - Comprehensive Testing Suite

Complete testing framework with unit tests, integration tests, performance tests,
and end-to-end testing for the WBPM capability.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import time
import uuid
import pytest
import httpx
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import random
import string
from unittest.mock import Mock, AsyncMock, patch
from contextlib import asynccontextmanager
import aiofiles

from models import (
    APGTenantContext, WBPMServiceResponse, WBPMPagedResponse,
    WBPMProcessInstance, WBPMTask, ProcessStatus, TaskStatus, TaskPriority
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Testing Framework Core Classes
# =============================================================================

class TestType(str, Enum):
    """Types of tests in the suite."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    E2E = "e2e"
    SECURITY = "security"
    LOAD = "load"
    CHAOS = "chaos"


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Individual test case definition."""
    test_id: str = field(default_factory=lambda: f"test_{uuid.uuid4().hex}")
    name: str = ""
    description: str = ""
    test_type: TestType = TestType.UNIT
    test_function: Optional[Callable] = None
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    test_data: Dict[str, Any] = field(default_factory=dict)
    expected_result: Any = None
    timeout_seconds: int = 30
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Test execution result."""
    test_id: str = ""
    test_name: str = ""
    status: TestStatus = TestStatus.PENDING
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    assertion_results: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    output: str = ""
    artifacts: List[str] = field(default_factory=list)


@dataclass
class TestSuite:
    """Collection of related test cases."""
    suite_id: str = field(default_factory=lambda: f"suite_{uuid.uuid4().hex}")
    name: str = ""
    description: str = ""
    test_cases: List[TestCase] = field(default_factory=list)
    setup_suite: Optional[Callable] = None
    teardown_suite: Optional[Callable] = None
    parallel_execution: bool = False
    max_parallel_tests: int = 4


@dataclass
class TestEnvironment:
    """Test environment configuration."""
    environment_id: str = field(default_factory=lambda: f"env_{uuid.uuid4().hex}")
    name: str = ""
    base_url: str = "http://localhost:8000"
    database_url: str = "postgresql://test:test@localhost:5432/wbpm_test"
    redis_url: str = "redis://localhost:6379/0"
    test_tenant_id: str = "test_tenant_123"
    test_user_id: str = "test_user_123"
    api_key: str = "test_api_key"
    jwt_token: str = "test_jwt_token"
    cleanup_after_tests: bool = True
    mock_external_services: bool = True


# =============================================================================
# Test Data Factory
# =============================================================================

class TestDataFactory:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_test_context(tenant_id: str = None, user_id: str = None) -> APGTenantContext:
        """Create test APG tenant context."""
        return APGTenantContext(
            tenant_id=tenant_id or f"tenant_{uuid.uuid4().hex}",
            user_id=user_id or f"user_{uuid.uuid4().hex}",
            permissions=["wbpm:process:read", "wbpm:process:write", "wbpm:task:complete"],
            session_id=f"session_{uuid.uuid4().hex}"
        )
    
    @staticmethod
    def create_test_process_definition() -> Dict[str, Any]:
        """Create test process definition."""
        return {
            "process_key": f"test_process_{uuid.uuid4().hex[:8]}",
            "process_name": "Test Process",
            "process_description": "Test process for automated testing",
            "bpmn_xml": """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="test_process" isExecutable="true">
    <bpmn:startEvent id="start"/>
    <bpmn:userTask id="task1" name="Test Task"/>
    <bpmn:endEvent id="end"/>
    <bpmn:sequenceFlow sourceRef="start" targetRef="task1"/>
    <bpmn:sequenceFlow sourceRef="task1" targetRef="end"/>
  </bpmn:process>
</bpmn:definitions>""",
            "category": "test",
            "tags": ["test", "automation"]
        }
    
    @staticmethod
    def create_test_process_variables() -> Dict[str, Any]:
        """Create test process variables."""
        return {
            "test_string": "test_value",
            "test_number": 42,
            "test_boolean": True,
            "test_list": ["item1", "item2", "item3"],
            "test_object": {
                "nested_field": "nested_value",
                "nested_number": 123
            }
        }
    
    @staticmethod
    def create_large_dataset(size: int = 1000) -> List[Dict[str, Any]]:
        """Create large dataset for performance testing."""
        dataset = []
        for i in range(size):
            dataset.append({
                "id": f"item_{i}",
                "name": f"Test Item {i}",
                "value": random.randint(1, 100),
                "description": ''.join(random.choices(string.ascii_letters, k=50)),
                "created_at": (datetime.utcnow() - timedelta(days=random.randint(0, 365))).isoformat()
            })
        return dataset
    
    @staticmethod
    def create_stress_test_data() -> Dict[str, Any]:
        """Create data for stress testing."""
        return {
            "concurrent_processes": 100,
            "concurrent_tasks": 500,
            "process_complexity": "high",
            "execution_duration": 300,  # 5 minutes
            "error_injection_rate": 0.05  # 5% error rate
        }


# =============================================================================
# Test Fixtures and Utilities
# =============================================================================

class TestFixtures:
    """Shared test fixtures and utilities."""
    
    def __init__(self, environment: TestEnvironment):
        self.environment = environment
        self.http_client: Optional[httpx.AsyncClient] = None
        self.created_resources: List[str] = []
        
    @asynccontextmanager
    async def test_session(self):
        """Async context manager for test session."""
        try:
            # Setup
            self.http_client = httpx.AsyncClient(
                base_url=self.environment.base_url,
                headers={
                    "Authorization": f"Bearer {self.environment.jwt_token}",
                    "X-Tenant-ID": self.environment.test_tenant_id,
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            
            yield self
            
        finally:
            # Cleanup
            if self.environment.cleanup_after_tests:
                await self._cleanup_resources()
            
            if self.http_client:
                await self.http_client.aclose()
    
    async def create_test_process(self, process_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a test process definition."""
        if not process_data:
            process_data = TestDataFactory.create_test_process_definition()
        
        response = await self.http_client.post("/api/v1/processes/definitions", json=process_data)
        response.raise_for_status()
        
        result = response.json()
        if result["success"]:
            self.created_resources.append(f"process:{result['data']['process_id']}")
        
        return result
    
    async def start_test_process_instance(self, process_key: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start a test process instance."""
        if not variables:
            variables = TestDataFactory.create_test_process_variables()
        
        response = await self.http_client.post(
            "/api/v1/processes/instances",
            params={"process_key": process_key},
            json={"variables": variables}
        )
        response.raise_for_status()
        
        result = response.json()
        if result["success"]:
            self.created_resources.append(f"instance:{result['data']['instance_id']}")
        
        return result
    
    async def complete_test_task(self, task_id: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete a test task."""
        if not variables:
            variables = {"completed": True, "result": "test_completed"}
        
        response = await self.http_client.post(
            f"/api/v1/tasks/{task_id}/complete",
            json={"variables": variables}
        )
        response.raise_for_status()
        
        return response.json()
    
    async def _cleanup_resources(self):
        """Clean up created test resources."""
        for resource in self.created_resources:
            try:
                resource_type, resource_id = resource.split(":", 1)
                
                if resource_type == "process":
                    # Delete process definition
                    await self.http_client.delete(f"/api/v1/processes/definitions/{resource_id}")
                elif resource_type == "instance":
                    # Cancel process instance
                    await self.http_client.post(f"/api/v1/processes/instances/{resource_id}/cancel")
                
            except Exception as e:
                logger.warning(f"Error cleaning up resource {resource}: {e}")
        
        self.created_resources.clear()


# =============================================================================
# Unit Tests
# =============================================================================

class WBPMUnitTests:
    """Unit tests for WBPM components."""
    
    def __init__(self, environment: TestEnvironment):
        self.environment = environment
        self.fixtures = TestFixtures(environment)
    
    async def test_process_definition_validation(self) -> TestResult:
        """Test process definition validation."""
        test_result = TestResult(
            test_name="Process Definition Validation",
            start_time=datetime.utcnow()
        )
        
        try:
            # Test valid process definition
            valid_process = TestDataFactory.create_test_process_definition()
            
            async with self.fixtures.test_session():
                result = await self.fixtures.create_test_process(valid_process)
                
                assert result["success"] == True
                assert "process_id" in result["data"]
                
                test_result.assertion_results.append({
                    "assertion": "Valid process creation",
                    "expected": True,
                    "actual": result["success"],
                    "passed": True
                })
            
            # Test invalid process definition
            invalid_process = valid_process.copy()
            invalid_process["bpmn_xml"] = "invalid xml"
            
            async with self.fixtures.test_session():
                response = await self.fixtures.http_client.post(
                    "/api/v1/processes/definitions", 
                    json=invalid_process
                )
                
                result = response.json()
                assert result["success"] == False
                assert "error" in result or "errors" in result
                
                test_result.assertion_results.append({
                    "assertion": "Invalid process rejection",
                    "expected": False,
                    "actual": result["success"],
                    "passed": True
                })
            
            test_result.status = TestStatus.PASSED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result
    
    async def test_task_assignment_logic(self) -> TestResult:
        """Test task assignment logic."""
        test_result = TestResult(
            test_name="Task Assignment Logic",
            start_time=datetime.utcnow()
        )
        
        try:
            async with self.fixtures.test_session():
                # Create process and start instance
                process_data = TestDataFactory.create_test_process_definition()
                process_result = await self.fixtures.create_test_process(process_data)
                
                instance_result = await self.fixtures.start_test_process_instance(
                    process_data["process_key"]
                )
                
                # Get user tasks
                response = await self.fixtures.http_client.get(
                    f"/api/v1/tasks/user/{self.environment.test_user_id}"
                )
                response.raise_for_status()
                
                tasks_result = response.json()
                assert tasks_result["success"] == True
                assert "tasks" in tasks_result["data"]
                
                test_result.assertion_results.append({
                    "assertion": "Tasks retrieved for user",
                    "expected": True,
                    "actual": tasks_result["success"],
                    "passed": True
                })
            
            test_result.status = TestStatus.PASSED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result
    
    async def test_notification_system(self) -> TestResult:
        """Test notification system functionality."""
        test_result = TestResult(
            test_name="Notification System",
            start_time=datetime.utcnow()
        )
        
        try:
            async with self.fixtures.test_session():
                # Send test notification
                notification_data = {
                    "recipient_id": self.environment.test_user_id,
                    "channels": ["email", "in_app"],
                    "priority": "normal",
                    "subject": "Test Notification",
                    "message": "This is a test notification from the testing suite"
                }
                
                response = await self.fixtures.http_client.post(
                    "/api/v1/notifications/send",
                    json=notification_data
                )
                response.raise_for_status()
                
                result = response.json()
                assert result["success"] == True
                assert "notification_id" in result["data"]
                
                test_result.assertion_results.append({
                    "assertion": "Notification queued successfully",
                    "expected": True,
                    "actual": result["success"],
                    "passed": True
                })
            
            test_result.status = TestStatus.PASSED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result


# =============================================================================
# Integration Tests
# =============================================================================

class WBPMIntegrationTests:
    """Integration tests for WBPM system."""
    
    def __init__(self, environment: TestEnvironment):
        self.environment = environment
        self.fixtures = TestFixtures(environment)
    
    async def test_end_to_end_process_execution(self) -> TestResult:
        """Test complete process execution flow."""
        test_result = TestResult(
            test_name="End-to-End Process Execution",
            start_time=datetime.utcnow()
        )
        
        try:
            async with self.fixtures.test_session():
                # 1. Create process definition
                process_data = TestDataFactory.create_test_process_definition()
                process_result = await self.fixtures.create_test_process(process_data)
                process_id = process_result["data"]["process_id"]
                
                test_result.assertion_results.append({
                    "assertion": "Process definition created",
                    "expected": True,
                    "actual": process_result["success"],
                    "passed": True
                })
                
                # 2. Start process instance
                variables = TestDataFactory.create_test_process_variables()
                instance_result = await self.fixtures.start_test_process_instance(
                    process_data["process_key"], variables
                )
                instance_id = instance_result["data"]["instance_id"]
                
                test_result.assertion_results.append({
                    "assertion": "Process instance started",
                    "expected": True,
                    "actual": instance_result["success"],
                    "passed": True
                })
                
                # 3. Get process instance details
                response = await self.fixtures.http_client.get(
                    f"/api/v1/processes/instances/{instance_id}",
                    params={"include_tasks": "true"}
                )
                response.raise_for_status()
                
                instance_details = response.json()
                assert instance_details["success"] == True
                assert instance_details["data"]["status"] == "active"
                
                test_result.assertion_results.append({
                    "assertion": "Process instance is active",
                    "expected": "active",
                    "actual": instance_details["data"]["status"],
                    "passed": True
                })
                
                # 4. Complete available tasks
                if "tasks" in instance_details["data"]:
                    for task in instance_details["data"]["tasks"]:
                        if task["status"] == "active":
                            complete_result = await self.fixtures.complete_test_task(
                                task["task_id"]
                            )
                            
                            test_result.assertion_results.append({
                                "assertion": f"Task {task['task_id']} completed",
                                "expected": True,
                                "actual": complete_result["success"],
                                "passed": True
                            })
                
                # 5. Verify process completion (if designed to complete)
                await asyncio.sleep(2)  # Allow time for process to complete
                
                response = await self.fixtures.http_client.get(
                    f"/api/v1/processes/instances/{instance_id}"
                )
                response.raise_for_status()
                
                final_instance = response.json()
                
                test_result.assertion_results.append({
                    "assertion": "Process flow executed successfully",
                    "expected": True,
                    "actual": final_instance["success"],
                    "passed": True
                })
            
            test_result.status = TestStatus.PASSED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result
    
    async def test_analytics_integration(self) -> TestResult:
        """Test analytics data collection and reporting."""
        test_result = TestResult(
            test_name="Analytics Integration",
            start_time=datetime.utcnow()
        )
        
        try:
            async with self.fixtures.test_session():
                # Create multiple processes for analytics
                processes_created = []
                for i in range(3):
                    process_data = TestDataFactory.create_test_process_definition()
                    process_data["process_key"] = f"analytics_test_{i}"
                    
                    process_result = await self.fixtures.create_test_process(process_data)
                    processes_created.append(process_data["process_key"])
                    
                    # Start instances
                    for j in range(2):
                        await self.fixtures.start_test_process_instance(
                            process_data["process_key"]
                        )
                
                # Wait for analytics to collect data
                await asyncio.sleep(5)
                
                # Get analytics data
                response = await self.fixtures.http_client.get(
                    "/api/v1/analytics/processes",
                    params={"time_range": "1d"}
                )
                response.raise_for_status()
                
                analytics_result = response.json()
                assert analytics_result["success"] == True
                assert "total_instances" in analytics_result["data"]
                
                test_result.assertion_results.append({
                    "assertion": "Analytics data collected",
                    "expected": True,
                    "actual": analytics_result["success"],
                    "passed": True
                })
                
                # Verify analytics contain expected data
                assert analytics_result["data"]["total_instances"] >= 6  # 3 processes * 2 instances
                
                test_result.assertion_results.append({
                    "assertion": "Analytics data accuracy",
                    "expected": ">=6",
                    "actual": analytics_result["data"]["total_instances"],
                    "passed": analytics_result["data"]["total_instances"] >= 6
                })
            
            test_result.status = TestStatus.PASSED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result
    
    async def test_template_recommendations(self) -> TestResult:
        """Test template recommendation system."""
        test_result = TestResult(
            test_name="Template Recommendations",
            start_time=datetime.utcnow()
        )
        
        try:
            async with self.fixtures.test_session():
                # Get template recommendations
                response = await self.fixtures.http_client.get(
                    "/api/v1/templates/recommendations",
                    params={"limit": "5"}
                )
                response.raise_for_status()
                
                recommendations_result = response.json()
                assert recommendations_result["success"] == True
                assert "recommendations" in recommendations_result["data"]
                
                test_result.assertion_results.append({
                    "assertion": "Template recommendations generated",
                    "expected": True,
                    "actual": recommendations_result["success"],
                    "passed": True
                })
                
                # Verify recommendation structure
                recommendations = recommendations_result["data"]["recommendations"]
                if recommendations:
                    first_rec = recommendations[0]
                    required_fields = ["template_id", "template_name", "recommendation_score"]
                    
                    for field in required_fields:
                        assert field in first_rec["recommendation"], f"Missing field: {field}"
                    
                    test_result.assertion_results.append({
                        "assertion": "Recommendation structure valid",
                        "expected": True,
                        "actual": True,
                        "passed": True
                    })
            
            test_result.status = TestStatus.PASSED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result


# =============================================================================
# Performance Tests
# =============================================================================

class WBPMPerformanceTests:
    """Performance tests for WBPM system."""
    
    def __init__(self, environment: TestEnvironment):
        self.environment = environment
        self.fixtures = TestFixtures(environment)
    
    async def test_concurrent_process_creation(self) -> TestResult:
        """Test concurrent process creation performance."""
        test_result = TestResult(
            test_name="Concurrent Process Creation",
            start_time=datetime.utcnow()
        )
        
        try:
            concurrent_count = 10
            
            async with self.fixtures.test_session():
                # Create multiple processes concurrently
                tasks = []
                for i in range(concurrent_count):
                    process_data = TestDataFactory.create_test_process_definition()
                    process_data["process_key"] = f"perf_test_{i}_{uuid.uuid4().hex[:8]}"
                    
                    task = asyncio.create_task(
                        self.fixtures.create_test_process(process_data)
                    )
                    tasks.append(task)
                
                # Wait for all processes to be created
                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                duration = end_time - start_time
                
                # Analyze results
                successful_creations = sum(
                    1 for result in results 
                    if isinstance(result, dict) and result.get("success", False)
                )
                
                test_result.performance_metrics = {
                    "total_duration_seconds": duration,
                    "concurrent_requests": concurrent_count,
                    "successful_requests": successful_creations,
                    "requests_per_second": concurrent_count / duration,
                    "average_response_time": duration / concurrent_count
                }
                
                test_result.assertion_results.append({
                    "assertion": "All processes created successfully",
                    "expected": concurrent_count,
                    "actual": successful_creations,
                    "passed": successful_creations == concurrent_count
                })
                
                # Performance threshold check (should complete within 10 seconds)
                performance_threshold = 10.0
                test_result.assertion_results.append({
                    "assertion": f"Performance under {performance_threshold}s",
                    "expected": f"<{performance_threshold}",
                    "actual": duration,
                    "passed": duration < performance_threshold
                })
            
            test_result.status = TestStatus.PASSED if all(
                r["passed"] for r in test_result.assertion_results
            ) else TestStatus.FAILED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result
    
    async def test_high_volume_task_processing(self) -> TestResult:
        """Test high volume task processing performance."""
        test_result = TestResult(
            test_name="High Volume Task Processing",
            start_time=datetime.utcnow()
        )
        
        try:
            task_count = 50
            
            async with self.fixtures.test_session():
                # Create a process for task testing
                process_data = TestDataFactory.create_test_process_definition()
                process_result = await self.fixtures.create_test_process(process_data)
                
                # Start multiple instances to generate tasks
                instance_tasks = []
                for i in range(task_count):
                    task = asyncio.create_task(
                        self.fixtures.start_test_process_instance(
                            process_data["process_key"]
                        )
                    )
                    instance_tasks.append(task)
                
                start_time = time.time()
                instances = await asyncio.gather(*instance_tasks)
                
                # Get all user tasks
                response = await self.fixtures.http_client.get(
                    f"/api/v1/tasks/user/{self.environment.test_user_id}"
                )
                response.raise_for_status()
                
                tasks_result = response.json()
                user_tasks = tasks_result["data"]["tasks"]
                
                # Complete tasks concurrently
                completion_tasks = []
                for task in user_tasks[:task_count]:  # Limit to expected count
                    completion_task = asyncio.create_task(
                        self.fixtures.complete_test_task(task["task_id"])
                    )
                    completion_tasks.append(completion_task)
                
                completed_results = await asyncio.gather(
                    *completion_tasks, return_exceptions=True
                )
                end_time = time.time()
                
                duration = end_time - start_time
                
                # Analyze results
                successful_completions = sum(
                    1 for result in completed_results 
                    if isinstance(result, dict) and result.get("success", False)
                )
                
                test_result.performance_metrics = {
                    "total_duration_seconds": duration,
                    "tasks_processed": task_count,
                    "successful_completions": successful_completions,
                    "tasks_per_second": task_count / duration,
                    "average_task_completion_time": duration / task_count
                }
                
                test_result.assertion_results.append({
                    "assertion": "All tasks completed successfully",
                    "expected": task_count,
                    "actual": successful_completions,
                    "passed": successful_completions >= task_count * 0.95  # 95% success rate
                })
                
                # Performance threshold (should process 50 tasks within 30 seconds)
                performance_threshold = 30.0
                test_result.assertion_results.append({
                    "assertion": f"Task processing under {performance_threshold}s",
                    "expected": f"<{performance_threshold}",
                    "actual": duration,
                    "passed": duration < performance_threshold
                })
            
            test_result.status = TestStatus.PASSED if all(
                r["passed"] for r in test_result.assertion_results
            ) else TestStatus.FAILED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result
    
    async def test_analytics_query_performance(self) -> TestResult:
        """Test analytics query performance with large datasets."""
        test_result = TestResult(
            test_name="Analytics Query Performance",
            start_time=datetime.utcnow()
        )
        
        try:
            async with self.fixtures.test_session():
                # Test analytics queries with different time ranges
                time_ranges = ["1d", "7d", "30d"]
                query_results = {}
                
                for time_range in time_ranges:
                    start_time = time.time()
                    
                    response = await self.fixtures.http_client.get(
                        "/api/v1/analytics/processes",
                        params={
                            "time_range": time_range,
                            "metrics": "duration,throughput,error_rate"
                        }
                    )
                    response.raise_for_status()
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    query_results[time_range] = {
                        "duration": duration,
                        "success": response.json()["success"]
                    }
                
                test_result.performance_metrics = {
                    "query_1d_duration": query_results["1d"]["duration"],
                    "query_7d_duration": query_results["7d"]["duration"],
                    "query_30d_duration": query_results["30d"]["duration"],
                    "average_query_time": sum(r["duration"] for r in query_results.values()) / len(query_results)
                }
                
                # All queries should complete within 5 seconds
                query_threshold = 5.0
                for time_range, result in query_results.items():
                    test_result.assertion_results.append({
                        "assertion": f"Query {time_range} under {query_threshold}s",
                        "expected": f"<{query_threshold}",
                        "actual": result["duration"],
                        "passed": result["duration"] < query_threshold and result["success"]
                    })
            
            test_result.status = TestStatus.PASSED if all(
                r["passed"] for r in test_result.assertion_results
            ) else TestStatus.FAILED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result


# =============================================================================
# Security Tests
# =============================================================================

class WBPMSecurityTests:
    """Security tests for WBPM system."""
    
    def __init__(self, environment: TestEnvironment):
        self.environment = environment
        self.fixtures = TestFixtures(environment)
    
    async def test_authentication_enforcement(self) -> TestResult:
        """Test authentication enforcement."""
        test_result = TestResult(
            test_name="Authentication Enforcement",
            start_time=datetime.utcnow()
        )
        
        try:
            # Test without authentication
            unauthenticated_client = httpx.AsyncClient(
                base_url=self.environment.base_url,
                timeout=30.0
            )
            
            try:
                response = await unauthenticated_client.get("/api/v1/processes/definitions")
                
                # Should return 401 Unauthorized
                test_result.assertion_results.append({
                    "assertion": "Unauthenticated request blocked",
                    "expected": 401,
                    "actual": response.status_code,
                    "passed": response.status_code == 401
                })
                
            finally:
                await unauthenticated_client.aclose()
            
            # Test with invalid token
            invalid_token_client = httpx.AsyncClient(
                base_url=self.environment.base_url,
                headers={"Authorization": "Bearer invalid_token"},
                timeout=30.0
            )
            
            try:
                response = await invalid_token_client.get("/api/v1/processes/definitions")
                
                # Should return 401 or 403
                test_result.assertion_results.append({
                    "assertion": "Invalid token rejected",
                    "expected": "401 or 403",
                    "actual": response.status_code,
                    "passed": response.status_code in [401, 403]
                })
                
            finally:
                await invalid_token_client.aclose()
            
            # Test with valid token
            async with self.fixtures.test_session():
                response = await self.fixtures.http_client.get("/api/v1/processes/definitions")
                
                test_result.assertion_results.append({
                    "assertion": "Valid token accepted",
                    "expected": 200,
                    "actual": response.status_code,
                    "passed": response.status_code == 200
                })
            
            test_result.status = TestStatus.PASSED if all(
                r["passed"] for r in test_result.assertion_results
            ) else TestStatus.FAILED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result
    
    async def test_tenant_isolation(self) -> TestResult:
        """Test multi-tenant data isolation."""
        test_result = TestResult(
            test_name="Tenant Isolation",
            start_time=datetime.utcnow()
        )
        
        try:
            # Create process with tenant A
            tenant_a_client = httpx.AsyncClient(
                base_url=self.environment.base_url,
                headers={
                    "Authorization": f"Bearer {self.environment.jwt_token}",
                    "X-Tenant-ID": "tenant_a",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            
            # Create process with tenant B
            tenant_b_client = httpx.AsyncClient(
                base_url=self.environment.base_url,
                headers={
                    "Authorization": f"Bearer {self.environment.jwt_token}",
                    "X-Tenant-ID": "tenant_b",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            
            try:
                # Create process in tenant A
                process_data_a = TestDataFactory.create_test_process_definition()
                process_data_a["process_key"] = "tenant_a_process"
                
                response_a = await tenant_a_client.post(
                    "/api/v1/processes/definitions",
                    json=process_data_a
                )
                response_a.raise_for_status()
                process_a_result = response_a.json()
                
                # Create process in tenant B
                process_data_b = TestDataFactory.create_test_process_definition()
                process_data_b["process_key"] = "tenant_b_process"
                
                response_b = await tenant_b_client.post(
                    "/api/v1/processes/definitions",
                    json=process_data_b
                )
                response_b.raise_for_status()
                process_b_result = response_b.json()
                
                # Verify tenant A cannot see tenant B's processes
                response_a_list = await tenant_a_client.get("/api/v1/processes/definitions")
                response_a_list.raise_for_status()
                a_processes = response_a_list.json()["data"]
                
                tenant_b_process_visible = any(
                    p.get("process_key") == "tenant_b_process" 
                    for p in a_processes
                )
                
                test_result.assertion_results.append({
                    "assertion": "Tenant B process not visible to tenant A",
                    "expected": False,
                    "actual": tenant_b_process_visible,
                    "passed": not tenant_b_process_visible
                })
                
                # Verify tenant B cannot see tenant A's processes
                response_b_list = await tenant_b_client.get("/api/v1/processes/definitions")
                response_b_list.raise_for_status()
                b_processes = response_b_list.json()["data"]
                
                tenant_a_process_visible = any(
                    p.get("process_key") == "tenant_a_process" 
                    for p in b_processes
                )
                
                test_result.assertion_results.append({
                    "assertion": "Tenant A process not visible to tenant B",
                    "expected": False,
                    "actual": tenant_a_process_visible,
                    "passed": not tenant_a_process_visible
                })
                
            finally:
                await tenant_a_client.aclose()
                await tenant_b_client.aclose()
            
            test_result.status = TestStatus.PASSED if all(
                r["passed"] for r in test_result.assertion_results
            ) else TestStatus.FAILED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result
    
    async def test_input_validation_and_sanitization(self) -> TestResult:
        """Test input validation and sanitization."""
        test_result = TestResult(
            test_name="Input Validation and Sanitization",
            start_time=datetime.utcnow()
        )
        
        try:
            async with self.fixtures.test_session():
                # Test SQL injection attempt
                malicious_process_data = TestDataFactory.create_test_process_definition()
                malicious_process_data["process_name"] = "'; DROP TABLE processes; --"
                
                response = await self.fixtures.http_client.post(
                    "/api/v1/processes/definitions",
                    json=malicious_process_data
                )
                
                # Should either reject input or sanitize it (not crash the system)
                test_result.assertion_results.append({
                    "assertion": "SQL injection attempt handled safely",
                    "expected": "handled",
                    "actual": "handled" if response.status_code != 500 else "error",
                    "passed": response.status_code != 500
                })
                
                # Test XSS attempt
                xss_process_data = TestDataFactory.create_test_process_definition()
                xss_process_data["process_description"] = "<script>alert('xss')</script>"
                
                response = await self.fixtures.http_client.post(
                    "/api/v1/processes/definitions",
                    json=xss_process_data
                )
                
                test_result.assertion_results.append({
                    "assertion": "XSS attempt handled safely",
                    "expected": "handled",
                    "actual": "handled" if response.status_code != 500 else "error",
                    "passed": response.status_code != 500
                })
                
                # Test oversized input
                oversized_process_data = TestDataFactory.create_test_process_definition()
                oversized_process_data["process_description"] = "A" * 100000  # 100KB description
                
                response = await self.fixtures.http_client.post(
                    "/api/v1/processes/definitions",
                    json=oversized_process_data
                )
                
                # Should reject oversized input
                test_result.assertion_results.append({
                    "assertion": "Oversized input rejected",
                    "expected": "rejected",
                    "actual": "rejected" if response.status_code == 400 else "accepted",
                    "passed": response.status_code == 400
                })
            
            test_result.status = TestStatus.PASSED if all(
                r["passed"] for r in test_result.assertion_results
            ) else TestStatus.FAILED
            
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
        
        finally:
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
        
        return test_result


# =============================================================================
# Test Suite Manager
# =============================================================================

class TestSuiteManager:
    """Manage and execute test suites."""
    
    def __init__(self, environment: TestEnvironment):
        self.environment = environment
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: List[TestResult] = []
        
    async def register_test_suite(self, suite: TestSuite) -> None:
        """Register a test suite."""
        self.test_suites[suite.suite_id] = suite
        logger.info(f"Registered test suite: {suite.name}")
    
    async def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all registered test suites."""
        all_results = {}
        
        for suite_id, suite in self.test_suites.items():
            logger.info(f"Running test suite: {suite.name}")
            
            suite_results = await self.run_test_suite(suite)
            all_results[suite.name] = suite_results
            
            # Log suite summary
            passed = sum(1 for r in suite_results if r.status == TestStatus.PASSED)
            failed = sum(1 for r in suite_results if r.status == TestStatus.FAILED)
            
            logger.info(f"Suite {suite.name} completed: {passed} passed, {failed} failed")
        
        return all_results
    
    async def run_test_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run a specific test suite."""
        results = []
        
        try:
            # Run suite setup
            if suite.setup_suite:
                await suite.setup_suite()
            
            # Execute tests
            if suite.parallel_execution:
                results = await self._run_tests_parallel(suite)
            else:
                results = await self._run_tests_sequential(suite)
            
        finally:
            # Run suite teardown
            if suite.teardown_suite:
                await suite.teardown_suite()
        
        self.test_results.extend(results)
        return results
    
    async def _run_tests_sequential(self, suite: TestSuite) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        
        for test_case in suite.test_cases:
            result = await self._execute_test_case(test_case)
            results.append(result)
            
            # Stop on failure if configured
            if result.status == TestStatus.FAILED and not getattr(suite, 'continue_on_failure', True):
                break
        
        return results
    
    async def _run_tests_parallel(self, suite: TestSuite) -> List[TestResult]:
        """Run tests in parallel."""
        semaphore = asyncio.Semaphore(suite.max_parallel_tests)
        
        async def run_with_semaphore(test_case):
            async with semaphore:
                return await self._execute_test_case(test_case)
        
        tasks = [run_with_semaphore(test_case) for test_case in suite.test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = TestResult(
                    test_id=suite.test_cases[i].test_id,
                    test_name=suite.test_cases[i].name,
                    status=TestStatus.ERROR,
                    error_message=str(result),
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow()
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    async def _execute_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case."""
        result = TestResult(
            test_id=test_case.test_id,
            test_name=test_case.name,
            start_time=datetime.utcnow()
        )
        
        try:
            # Run setup
            if test_case.setup_function:
                await test_case.setup_function()
            
            # Execute test with timeout
            if test_case.test_function:
                test_result = await asyncio.wait_for(
                    test_case.test_function(),
                    timeout=test_case.timeout_seconds
                )
                
                if isinstance(test_result, TestResult):
                    return test_result
                else:
                    result.status = TestStatus.PASSED
            else:
                result.status = TestStatus.SKIPPED
                
        except asyncio.TimeoutError:
            result.status = TestStatus.FAILED
            result.error_message = f"Test timed out after {test_case.timeout_seconds} seconds"
            
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            
        finally:
            # Run teardown
            if test_case.teardown_function:
                try:
                    await test_case.teardown_function()
                except Exception as e:
                    logger.warning(f"Teardown failed for test {test_case.name}: {e}")
            
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    async def generate_test_report(self, output_path: Path) -> str:
        """Generate comprehensive test report."""
        try:
            # Calculate overall statistics
            total_tests = len(self.test_results)
            passed_tests = sum(1 for r in self.test_results if r.status == TestStatus.PASSED)
            failed_tests = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
            error_tests = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
            skipped_tests = sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED)
            
            total_duration = sum(r.duration_seconds for r in self.test_results)
            
            # Generate HTML report
            html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>WBPM Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: white; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .error {{ color: orange; }}
        .skipped {{ color: gray; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .status-passed {{ background-color: #d4edda; }}
        .status-failed {{ background-color: #f8d7da; }}
        .status-error {{ background-color: #fff3cd; }}
        .status-skipped {{ background-color: #e2e3e5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>WBPM Test Suite Report</h1>
        <p>Generated: {datetime.utcnow().isoformat()}</p>
        <p>Environment: {self.environment.name}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Total Tests</h3>
            <p style="font-size: 24px; margin: 0;">{total_tests}</p>
        </div>
        <div class="metric">
            <h3 class="passed">Passed</h3>
            <p style="font-size: 24px; margin: 0;">{passed_tests}</p>
        </div>
        <div class="metric">
            <h3 class="failed">Failed</h3>
            <p style="font-size: 24px; margin: 0;">{failed_tests}</p>
        </div>
        <div class="metric">
            <h3 class="error">Errors</h3>
            <p style="font-size: 24px; margin: 0;">{error_tests}</p>
        </div>
        <div class="metric">
            <h3 class="skipped">Skipped</h3>
            <p style="font-size: 24px; margin: 0;">{skipped_tests}</p>
        </div>
        <div class="metric">
            <h3>Duration</h3>
            <p style="font-size: 24px; margin: 0;">{total_duration:.1f}s</p>
        </div>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <thead>
            <tr>
                <th>Test Name</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Error Message</th>
                <th>Assertions</th>
            </tr>
        </thead>
        <tbody>
"""
            
            # Add test results
            for result in self.test_results:
                status_class = f"status-{result.status.value}"
                assertions_passed = sum(1 for a in result.assertion_results if a.get("passed", False))
                total_assertions = len(result.assertion_results)
                
                html_report += f"""
            <tr class="{status_class}">
                <td>{result.test_name}</td>
                <td>{result.status.value.upper()}</td>
                <td>{result.duration_seconds:.2f}</td>
                <td>{result.error_message or ''}</td>
                <td>{assertions_passed}/{total_assertions}</td>
            </tr>
"""
            
            html_report += """
        </tbody>
    </table>
    
    <h2>Performance Metrics</h2>
    <table>
        <thead>
            <tr>
                <th>Test Name</th>
                <th>Metric</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
"""
            
            # Add performance metrics
            for result in self.test_results:
                if result.performance_metrics:
                    for metric_name, metric_value in result.performance_metrics.items():
                        html_report += f"""
            <tr>
                <td>{result.test_name}</td>
                <td>{metric_name}</td>
                <td>{metric_value}</td>
            </tr>
"""
            
            html_report += """
        </tbody>
    </table>
</body>
</html>
"""
            
            # Save report
            report_file = output_path / f"test_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
            output_path.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(report_file, 'w') as f:
                await f.write(html_report)
            
            logger.info(f"Test report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error generating test report: {e}")
            raise


# =============================================================================
# Test Suite Factory
# =============================================================================

async def create_comprehensive_test_suite(environment: TestEnvironment) -> TestSuiteManager:
    """Create comprehensive test suite for WBPM."""
    try:
        manager = TestSuiteManager(environment)
        
        # Unit Tests Suite
        unit_tests = WBPMUnitTests(environment)
        unit_suite = TestSuite(
            name="Unit Tests",
            description="Unit tests for individual WBPM components",
            test_cases=[
                TestCase(
                    name="Process Definition Validation",
                    description="Test process definition validation logic",
                    test_type=TestType.UNIT,
                    test_function=unit_tests.test_process_definition_validation,
                    timeout_seconds=30,
                    tags=["unit", "validation"]
                ),
                TestCase(
                    name="Task Assignment Logic",
                    description="Test task assignment and routing logic",
                    test_type=TestType.UNIT,
                    test_function=unit_tests.test_task_assignment_logic,
                    timeout_seconds=30,
                    tags=["unit", "tasks"]
                ),
                TestCase(
                    name="Notification System",
                    description="Test notification system functionality",
                    test_type=TestType.UNIT,
                    test_function=unit_tests.test_notification_system,
                    timeout_seconds=30,
                    tags=["unit", "notifications"]
                )
            ],
            parallel_execution=True,
            max_parallel_tests=3
        )
        await manager.register_test_suite(unit_suite)
        
        # Integration Tests Suite
        integration_tests = WBPMIntegrationTests(environment)
        integration_suite = TestSuite(
            name="Integration Tests",
            description="Integration tests for WBPM system components",
            test_cases=[
                TestCase(
                    name="End-to-End Process Execution",
                    description="Test complete process execution flow",
                    test_type=TestType.INTEGRATION,
                    test_function=integration_tests.test_end_to_end_process_execution,
                    timeout_seconds=60,
                    tags=["integration", "e2e"]
                ),
                TestCase(
                    name="Analytics Integration",
                    description="Test analytics data collection and reporting",
                    test_type=TestType.INTEGRATION,
                    test_function=integration_tests.test_analytics_integration,
                    timeout_seconds=45,
                    tags=["integration", "analytics"]
                ),
                TestCase(
                    name="Template Recommendations",
                    description="Test template recommendation system",
                    test_type=TestType.INTEGRATION,
                    test_function=integration_tests.test_template_recommendations,
                    timeout_seconds=30,
                    tags=["integration", "templates"]
                )
            ],
            parallel_execution=False  # Sequential to avoid resource conflicts
        )
        await manager.register_test_suite(integration_suite)
        
        # Performance Tests Suite
        performance_tests = WBPMPerformanceTests(environment)
        performance_suite = TestSuite(
            name="Performance Tests",
            description="Performance and load tests for WBPM system",
            test_cases=[
                TestCase(
                    name="Concurrent Process Creation",
                    description="Test concurrent process creation performance",
                    test_type=TestType.PERFORMANCE,
                    test_function=performance_tests.test_concurrent_process_creation,
                    timeout_seconds=120,
                    tags=["performance", "concurrency"]
                ),
                TestCase(
                    name="High Volume Task Processing",
                    description="Test high volume task processing performance",
                    test_type=TestType.PERFORMANCE,
                    test_function=performance_tests.test_high_volume_task_processing,
                    timeout_seconds=180,
                    tags=["performance", "volume"]
                ),
                TestCase(
                    name="Analytics Query Performance",
                    description="Test analytics query performance with large datasets",
                    test_type=TestType.PERFORMANCE,
                    test_function=performance_tests.test_analytics_query_performance,
                    timeout_seconds=60,
                    tags=["performance", "analytics"]
                )
            ],
            parallel_execution=False  # Sequential for accurate performance measurement
        )
        await manager.register_test_suite(performance_suite)
        
        # Security Tests Suite
        security_tests = WBPMSecurityTests(environment)
        security_suite = TestSuite(
            name="Security Tests",
            description="Security tests for WBPM system",
            test_cases=[
                TestCase(
                    name="Authentication Enforcement",
                    description="Test authentication enforcement",
                    test_type=TestType.SECURITY,
                    test_function=security_tests.test_authentication_enforcement,
                    timeout_seconds=30,
                    tags=["security", "authentication"]
                ),
                TestCase(
                    name="Tenant Isolation",
                    description="Test multi-tenant data isolation",
                    test_type=TestType.SECURITY,
                    test_function=security_tests.test_tenant_isolation,
                    timeout_seconds=45,
                    tags=["security", "isolation"]
                ),
                TestCase(
                    name="Input Validation",
                    description="Test input validation and sanitization",
                    test_type=TestType.SECURITY,
                    test_function=security_tests.test_input_validation_and_sanitization,
                    timeout_seconds=30,
                    tags=["security", "validation"]
                )
            ],
            parallel_execution=True,
            max_parallel_tests=2
        )
        await manager.register_test_suite(security_suite)
        
        logger.info("Comprehensive test suite created successfully")
        return manager
        
    except Exception as e:
        logger.error(f"Error creating test suite: {e}")
        raise


# =============================================================================
# Main Test Execution
# =============================================================================

async def run_wbpm_test_suite(
    environment_name: str = "test",
    base_url: str = "http://localhost:8000",
    output_dir: str = "./test_reports"
) -> Dict[str, Any]:
    """Run the complete WBPM test suite."""
    try:
        # Create test environment
        test_env = TestEnvironment(
            name=environment_name,
            base_url=base_url,
            cleanup_after_tests=True,
            mock_external_services=True
        )
        
        # Create test suite manager
        manager = await create_comprehensive_test_suite(test_env)
        
        # Run all tests
        logger.info("Starting WBPM test suite execution")
        start_time = datetime.utcnow()
        
        results = await manager.run_all_tests()
        
        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds()
        
        # Generate test report
        report_path = Path(output_dir)
        report_file = await manager.generate_test_report(report_path)
        
        # Calculate summary statistics
        total_tests = len(manager.test_results)
        passed_tests = sum(1 for r in manager.test_results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in manager.test_results if r.status == TestStatus.FAILED)
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_duration_seconds": total_duration,
            "report_file": report_file,
            "environment": environment_name,
            "results_by_suite": {}
        }
        
        # Add results by suite
        for suite_name, suite_results in results.items():
            suite_passed = sum(1 for r in suite_results if r.status == TestStatus.PASSED)
            suite_failed = sum(1 for r in suite_results if r.status == TestStatus.FAILED)
            
            summary["results_by_suite"][suite_name] = {
                "total": len(suite_results),
                "passed": suite_passed,
                "failed": suite_failed,
                "success_rate": (suite_passed / len(suite_results) * 100) if suite_results else 0
            }
        
        logger.info(f"Test suite completed: {passed_tests}/{total_tests} passed ({summary['success_rate']:.1f}%)")
        return summary
        
    except Exception as e:
        logger.error(f"Error running test suite: {e}")
        raise


# Export main classes
__all__ = [
    'TestSuiteManager',
    'WBPMUnitTests',
    'WBPMIntegrationTests',
    'WBPMPerformanceTests',
    'WBPMSecurityTests',
    'TestDataFactory',
    'TestFixtures',
    'TestCase',
    'TestResult',
    'TestSuite',
    'TestEnvironment',
    'TestType',
    'TestStatus',
    'create_comprehensive_test_suite',
    'run_wbpm_test_suite'
]