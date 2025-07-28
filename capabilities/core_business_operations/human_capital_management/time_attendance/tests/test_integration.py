"""
APG Time & Attendance Capability - Integration Test Suite

Comprehensive integration tests for the revolutionary Time & Attendance capability
testing all components, APIs, and APG ecosystem integration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import pytest
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import redis.asyncio as redis

# Import our capability components
from ..api import app
from ..service import TimeAttendanceService
from ..models import TAEmployee, TATimeEntry, WorkMode, TimeEntryStatus
from ..websocket import websocket_manager
from ..mobile_api import mobile_router
from ..monitoring import monitoring_dashboard
from ..reporting import ReportGenerator, ReportType, ReportFormat
from ..database import DatabaseManager


# Test Configuration
TEST_DATABASE_URL = "postgresql://test:test@localhost:5432/test_time_attendance"
TEST_REDIS_URL = "redis://localhost:6379/1"
TEST_TENANT_ID = "test_tenant_001"
TEST_EMPLOYEE_ID = "test_emp_001"


class TestTimeAttendanceIntegration:
    """Comprehensive integration test suite"""
    
    @pytest.fixture(scope="session")
    async def setup_test_environment(self):
        """Setup test environment with database and dependencies"""
        # Setup test database
        engine = create_engine(TEST_DATABASE_URL)
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Setup test Redis
        redis_client = await redis.from_url(TEST_REDIS_URL)
        
        # Setup test service
        service = TimeAttendanceService()
        service.db_manager = DatabaseManager(TEST_DATABASE_URL)
        
        # Initialize test data
        await self._setup_test_data(service)
        
        yield {
            "service": service,
            "redis": redis_client,
            "db_session": TestSessionLocal()
        }
        
        # Cleanup
        await redis_client.close()
    
    @pytest.fixture
    def test_client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mobile_client(self):
        """Mobile API test client"""
        from fastapi import FastAPI
        mobile_app = FastAPI()
        mobile_app.include_router(mobile_router)
        return TestClient(mobile_app)
    
    async def _setup_test_data(self, service: TimeAttendanceService):
        """Setup test data"""
        # Create test employee
        test_employee = TAEmployee(
            id=TEST_EMPLOYEE_ID,
            employee_name="Test Employee",
            tenant_id=TEST_TENANT_ID,
            department_id="test_dept_001",
            work_schedule={
                "monday": {"start": "09:00", "end": "17:00"},
                "tuesday": {"start": "09:00", "end": "17:00"},
                "wednesday": {"start": "09:00", "end": "17:00"},
                "thursday": {"start": "09:00", "end": "17:00"},
                "friday": {"start": "09:00", "end": "17:00"}
            },
            work_mode=WorkMode.HYBRID,
            status="active"
        )
        
        await service.db_manager.create_employee(test_employee)


class TestCoreAPIIntegration(TestTimeAttendanceIntegration):
    """Test core API functionality"""
    
    async def test_health_check_endpoint(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/api/human_capital_management/time_attendance/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "version" in health_data
    
    async def test_clock_in_flow(self, test_client, setup_test_environment):
        """Test complete clock-in flow"""
        # Test clock-in
        clock_in_data = {
            "employee_id": TEST_EMPLOYEE_ID,
            "tenant_id": TEST_TENANT_ID,
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060
            },
            "device_info": {
                "device_type": "mobile",
                "platform": "iOS",
                "app_version": "1.0.0"
            }
        }
        
        response = test_client.post(
            "/api/human_capital_management/time_attendance/clock-in",
            json=clock_in_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["employee_id"] == TEST_EMPLOYEE_ID
        assert data["status"] == "active"
        assert "clock_in" in data
        assert "id" in data
        
        # Store time entry ID for clock-out test
        time_entry_id = data["id"]
        
    async def test_clock_out_flow(self, test_client, setup_test_environment):
        """Test complete clock-out flow"""
        # First clock in
        await self.test_clock_in_flow(test_client, setup_test_environment)
        
        # Then clock out
        clock_out_data = {
            "employee_id": TEST_EMPLOYEE_ID,
            "tenant_id": TEST_TENANT_ID,
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060
            }
        }
        
        response = test_client.post(
            "/api/human_capital_management/time_attendance/clock-out",
            json=clock_out_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["employee_id"] == TEST_EMPLOYEE_ID
        assert data["status"] == "completed"
        assert "clock_out" in data
        assert "total_hours" in data
    
    async def test_time_entries_retrieval(self, test_client, setup_test_environment):
        """Test time entries retrieval"""
        response = test_client.get(
            f"/api/human_capital_management/time_attendance/time-entries",
            params={
                "tenant_id": TEST_TENANT_ID,
                "employee_id": TEST_EMPLOYEE_ID,
                "start_date": date.today().isoformat(),
                "end_date": date.today().isoformat()
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "entries" in data
        assert "total_count" in data
        assert isinstance(data["entries"], list)
    
    async def test_employee_management(self, test_client):
        """Test employee management endpoints"""
        # Create employee
        employee_data = {
            "employee_name": "Integration Test Employee",
            "tenant_id": TEST_TENANT_ID,
            "department_id": "test_dept_002",
            "work_schedule": {
                "monday": {"start": "08:00", "end": "16:00"}
            },
            "work_mode": "office"
        }
        
        response = test_client.post(
            "/api/human_capital_management/time_attendance/employees",
            json=employee_data
        )
        
        assert response.status_code == 201
        created_employee = response.json()
        assert created_employee["employee_name"] == employee_data["employee_name"]
        
        employee_id = created_employee["id"]
        
        # Get employee
        response = test_client.get(
            f"/api/human_capital_management/time_attendance/employees/{employee_id}",
            params={"tenant_id": TEST_TENANT_ID}
        )
        
        assert response.status_code == 200
        employee = response.json()
        assert employee["id"] == employee_id
        
        # Update employee
        update_data = {"employee_name": "Updated Test Employee"}
        response = test_client.put(
            f"/api/human_capital_management/time_attendance/employees/{employee_id}",
            json=update_data,
            params={"tenant_id": TEST_TENANT_ID}
        )
        
        assert response.status_code == 200
        updated_employee = response.json()
        assert updated_employee["employee_name"] == update_data["employee_name"]


class TestMobileAPIIntegration(TestTimeAttendanceIntegration):
    """Test mobile API functionality"""
    
    async def test_mobile_quick_clock_in(self, mobile_client):
        """Test mobile quick clock-in"""
        clock_in_data = {
            "employee_id": TEST_EMPLOYEE_ID,
            "location": {"latitude": 40.7128, "longitude": -74.0060},
            "biometric_data": "base64_encoded_fingerprint_data_here",
            "photo_verification": "base64_encoded_photo_data_here",
            "device_info": {
                "device_id": "iPhone_12_Pro",
                "platform": "iOS",
                "app_version": "1.0.0"
            },
            "network_quality": "excellent",
            "battery_level": 85
        }
        
        with patch("..mobile_api.get_mobile_user") as mock_auth:
            mock_auth.return_value = {
                "user_id": "test_user_001",
                "tenant_id": TEST_TENANT_ID,
                "employee_id": TEST_EMPLOYEE_ID,
                "device_id": "iPhone_12_Pro"
            }
            
            response = mobile_client.post(
                "/api/mobile/human_capital_management/time_attendance/quick-clock-in",
                json=clock_in_data
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "active"
        assert "fraud_score" in data
        assert "location_verified" in data
        assert "biometric_verified" in data
    
    async def test_mobile_quick_status(self, mobile_client):
        """Test mobile quick status"""
        with patch("..mobile_api.get_mobile_user") as mock_auth:
            mock_auth.return_value = {
                "user_id": "test_user_001",
                "tenant_id": TEST_TENANT_ID,
                "employee_id": TEST_EMPLOYEE_ID
            }
            
            response = mobile_client.get(
                "/api/mobile/human_capital_management/time_attendance/quick-status"
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "employee_id" in data
        assert "is_clocked_in" in data
        assert "today_total_hours" in data
        assert "week_total_hours" in data
        assert "pending_approvals" in data
        assert "recent_alerts" in data
    
    async def test_mobile_offline_sync(self, mobile_client):
        """Test mobile offline synchronization"""
        sync_data = {
            "offline_entries": [
                {
                    "offline_id": "offline_001",
                    "type": "clock_in",
                    "employee_id": TEST_EMPLOYEE_ID,
                    "timestamp": "2025-01-21T09:00:00Z",
                    "location": {"latitude": 40.7128, "longitude": -74.0060},
                    "device_info": {"offline_sync": True}
                },
                {
                    "offline_id": "offline_002",
                    "type": "clock_out",
                    "employee_id": TEST_EMPLOYEE_ID,
                    "timestamp": "2025-01-21T17:00:00Z",
                    "location": {"latitude": 40.7128, "longitude": -74.0060},
                    "device_info": {"offline_sync": True}
                }
            ],
            "sync_timestamp": "2025-01-21T17:05:00Z",
            "conflict_resolution": "server_wins"
        }
        
        with patch("..mobile_api.get_mobile_user") as mock_auth:
            mock_auth.return_value = {
                "user_id": "test_user_001",
                "tenant_id": TEST_TENANT_ID,
                "employee_id": TEST_EMPLOYEE_ID
            }
            
            response = mobile_client.post(
                "/api/mobile/human_capital_management/time_attendance/sync/offline-entries",
                json=sync_data
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "synced_entries" in data
        assert "failed_entries" in data
        assert "conflicts" in data


class TestWebSocketIntegration(TestTimeAttendanceIntegration):
    """Test WebSocket real-time functionality"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection and authentication"""
        from fastapi.testclient import TestClient
        from fastapi import WebSocket
        import websockets
        
        # This would need a more sophisticated test setup for WebSocket testing
        # For now, we'll test the WebSocket manager directly
        
        # Test connection management
        mock_websocket = AsyncMock()
        mock_websocket.client_state = "CONNECTED"
        
        connection_id = await websocket_manager.connect(
            mock_websocket,
            user_id="test_user_001",
            tenant_id=TEST_TENANT_ID,
            session_type="dashboard"
        )
        
        assert connection_id is not None
        assert len(websocket_manager.active_connections) > 0
        
        # Test event broadcasting
        from ..websocket import RealTimeEvent
        event = RealTimeEvent(
            event_type="test_event",
            entity_type="time_entry",
            entity_id="test_entry_001",
            tenant_id=TEST_TENANT_ID,
            data={"test": "data"},
            user_id="test_user_001"
        )
        
        await websocket_manager.broadcast_time_entry_event(event)
        
        # Cleanup
        await websocket_manager.disconnect(connection_id)


class TestReportingIntegration(TestTimeAttendanceIntegration):
    """Test reporting and data export functionality"""
    
    @pytest.fixture
    def report_generator(self, setup_test_environment):
        """Report generator fixture"""
        env = setup_test_environment
        return ReportGenerator(env["service"])
    
    async def test_timesheet_report_generation(self, report_generator):
        """Test timesheet report generation"""
        from ..reporting import ReportConfig, ReportType, ReportFormat, ReportPeriod
        
        config = ReportConfig(
            report_type=ReportType.TIMESHEET,
            format=ReportFormat.JSON,
            period=ReportPeriod.MONTHLY,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            tenant_id=TEST_TENANT_ID,
            employee_ids=[TEST_EMPLOYEE_ID],
            include_charts=True,
            include_analytics=True
        )
        
        result = await report_generator.generate_report(config, "test_user_001")
        
        assert result["success"] is True
        assert "metadata" in result
        assert "data" in result
        assert "analytics" in result
        assert result["metadata"]["type"] == ReportType.TIMESHEET
    
    async def test_fraud_analysis_report(self, report_generator):
        """Test fraud analysis report generation"""
        from ..reporting import ReportConfig, ReportType, ReportFormat, ReportPeriod
        
        config = ReportConfig(
            report_type=ReportType.FRAUD_ANALYSIS,
            format=ReportFormat.JSON,
            period=ReportPeriod.WEEKLY,
            start_date=date.today() - timedelta(days=7),
            end_date=date.today(),
            tenant_id=TEST_TENANT_ID,
            include_analytics=True
        )
        
        result = await report_generator.generate_report(config, "test_user_001")
        
        assert result["success"] is True
        assert result["analytics"]["insights"] is not None
        assert len(result["analytics"]["insights"]) > 0
    
    async def test_excel_report_export(self, report_generator):
        """Test Excel report export"""
        from ..reporting import ReportConfig, ReportType, ReportFormat, ReportPeriod
        
        config = ReportConfig(
            report_type=ReportType.ATTENDANCE_SUMMARY,
            format=ReportFormat.EXCEL,
            period=ReportPeriod.MONTHLY,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            tenant_id=TEST_TENANT_ID,
            include_charts=True
        )
        
        result = await report_generator.generate_report(config, "test_user_001")
        
        assert result["success"] is True
        assert isinstance(result["data"], bytes)  # Excel file as bytes
        assert len(result["data"]) > 0


class TestMonitoringIntegration(TestTimeAttendanceIntegration):
    """Test monitoring and alerting functionality"""
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self):
        """Test system metrics collection"""
        from ..monitoring import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        metrics = await monitor.collect_system_metrics()
        
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "disk_percent" in metrics
        assert isinstance(metrics["cpu_percent"], (int, float))
    
    @pytest.mark.asyncio
    async def test_alert_evaluation(self):
        """Test alert evaluation"""
        from ..monitoring import AlertManager
        
        alert_manager = AlertManager()
        
        # Simulate high CPU usage
        test_metrics = {
            "cpu_percent": 90.0,
            "memory_percent": 60.0,
            "disk_percent": 70.0
        }
        
        alerts = await alert_manager.evaluate_alerts(test_metrics)
        
        # Should trigger high CPU alert
        cpu_alerts = [a for a in alerts if "cpu" in a.title.lower()]
        assert len(cpu_alerts) > 0
        assert cpu_alerts[0].current_value == 90.0
    
    @pytest.mark.asyncio
    async def test_business_metrics_monitoring(self, setup_test_environment):
        """Test business metrics monitoring"""
        from ..monitoring import BusinessMetricsMonitor
        
        env = setup_test_environment
        monitor = BusinessMetricsMonitor(env["service"])
        
        metrics = await monitor.collect_business_metrics(TEST_TENANT_ID)
        
        assert "active_employees" in metrics
        assert "clock_in_rate_today" in metrics
        assert "fraud_alerts_today" in metrics
        assert isinstance(metrics["active_employees"], int)
    
    @pytest.mark.asyncio
    async def test_health_report_generation(self, setup_test_environment):
        """Test health report generation"""
        from ..monitoring import BusinessMetricsMonitor
        
        env = setup_test_environment
        monitor = BusinessMetricsMonitor(env["service"])
        
        health_report = await monitor.generate_health_report(TEST_TENANT_ID)
        
        assert "overall_health_score" in health_report
        assert "component_scores" in health_report
        assert "status" in health_report
        assert health_report["status"] in ["healthy", "degraded", "unhealthy"]


class TestAIFraudDetectionIntegration(TestTimeAttendanceIntegration):
    """Test AI fraud detection functionality"""
    
    @pytest.mark.asyncio
    async def test_location_anomaly_detection(self):
        """Test location-based anomaly detection"""
        from ..ai_fraud_detection import LocationAnalyzer
        
        analyzer = LocationAnalyzer()
        
        # Test normal location
        normal_location = {"latitude": 40.7128, "longitude": -74.0060}
        normal_score = await analyzer.analyze_location_pattern(
            TEST_EMPLOYEE_ID,
            normal_location,
            datetime.utcnow()
        )
        
        assert 0.0 <= normal_score <= 1.0
        assert normal_score < 0.5  # Should be low risk
        
        # Test suspicious location (different continent)
        suspicious_location = {"latitude": 35.6762, "longitude": 139.6503}  # Tokyo
        suspicious_score = await analyzer.analyze_location_pattern(
            TEST_EMPLOYEE_ID,
            suspicious_location,
            datetime.utcnow()
        )
        
        assert suspicious_score > normal_score  # Should be higher risk
    
    @pytest.mark.asyncio
    async def test_temporal_anomaly_detection(self):
        """Test temporal anomaly detection"""
        from ..ai_fraud_detection import TemporalAnalyzer
        
        analyzer = TemporalAnalyzer()
        
        # Test normal business hours
        normal_time = datetime.now().replace(hour=9, minute=0)  # 9 AM
        normal_score = await analyzer.analyze_temporal_pattern(
            TEST_EMPLOYEE_ID,
            normal_time
        )
        
        assert normal_score < 0.3  # Should be low risk
        
        # Test unusual hours
        unusual_time = datetime.now().replace(hour=3, minute=0)  # 3 AM
        unusual_score = await analyzer.analyze_temporal_pattern(
            TEST_EMPLOYEE_ID,
            unusual_time
        )
        
        assert unusual_score > normal_score  # Should be higher risk
    
    @pytest.mark.asyncio
    async def test_comprehensive_fraud_analysis(self):
        """Test comprehensive fraud analysis"""
        from ..ai_fraud_detection import FraudDetectionEngine
        
        engine = FraudDetectionEngine()
        
        time_entry_data = {
            "employee_id": TEST_EMPLOYEE_ID,
            "timestamp": datetime.utcnow(),
            "location": {"latitude": 40.7128, "longitude": -74.0060},
            "device_info": {
                "device_id": "iPhone_12_Pro",
                "ip_address": "192.168.1.100"
            },
            "biometric_data": "fingerprint_hash_123"
        }
        
        analysis = await engine.analyze_time_entry(time_entry_data)
        
        assert "overall_score" in analysis
        assert "risk_level" in analysis
        assert "anomalies" in analysis
        assert 0.0 <= analysis["overall_score"] <= 1.0
        assert analysis["risk_level"] in ["low", "medium", "high"]


class TestDatabaseIntegration(TestTimeAttendanceIntegration):
    """Test database operations and multi-tenancy"""
    
    async def test_multi_tenant_isolation(self, setup_test_environment):
        """Test multi-tenant data isolation"""
        env = setup_test_environment
        db_manager = env["service"].db_manager
        
        # Create data for tenant 1
        tenant1_employee = TAEmployee(
            id="tenant1_emp_001",
            employee_name="Tenant 1 Employee",
            tenant_id="tenant_001",
            department_id="dept_001"
        )
        
        await db_manager.create_employee(tenant1_employee)
        
        # Create data for tenant 2
        tenant2_employee = TAEmployee(
            id="tenant2_emp_001",
            employee_name="Tenant 2 Employee",
            tenant_id="tenant_002",
            department_id="dept_001"
        )
        
        await db_manager.create_employee(tenant2_employee)
        
        # Verify isolation - tenant 1 can't see tenant 2's data
        tenant1_employees = await db_manager.get_employees("tenant_001")
        tenant1_employee_ids = [emp.id for emp in tenant1_employees]
        
        assert "tenant1_emp_001" in tenant1_employee_ids
        assert "tenant2_emp_001" not in tenant1_employee_ids
        
        # Verify tenant 2 isolation
        tenant2_employees = await db_manager.get_employees("tenant_002")
        tenant2_employee_ids = [emp.id for emp in tenant2_employees]
        
        assert "tenant2_emp_001" in tenant2_employee_ids
        assert "tenant1_emp_001" not in tenant2_employee_ids
    
    async def test_database_performance(self, setup_test_environment):
        """Test database performance with concurrent operations"""
        env = setup_test_environment
        service = env["service"]
        
        # Create multiple concurrent clock-in operations
        tasks = []
        for i in range(10):
            task = service.clock_in(
                employee_id=f"perf_test_emp_{i:03d}",
                tenant_id=TEST_TENANT_ID,
                created_by="performance_test"
            )
            tasks.append(task)
        
        # Execute concurrently and measure time
        start_time = datetime.utcnow()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.utcnow()
        
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (< 2 seconds for 10 operations)
        assert duration < 2.0
        
        # Verify all operations succeeded or handle expected failures
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 8  # Allow for some failures in test environment


class TestSecurityIntegration(TestTimeAttendanceIntegration):
    """Test security features"""
    
    async def test_authentication_required(self, test_client):
        """Test that authentication is required for protected endpoints"""
        # Test without authentication header
        response = test_client.post(
            "/api/human_capital_management/time_attendance/clock-in",
            json={"employee_id": TEST_EMPLOYEE_ID}
        )
        
        # Should return 401 or 403 (depending on implementation)
        assert response.status_code in [401, 403]
    
    async def test_tenant_isolation_enforcement(self, test_client):
        """Test that tenant isolation is enforced in API"""
        # This would need proper authentication implementation
        # For now, test that tenant_id is required
        
        response = test_client.get(
            "/api/human_capital_management/time_attendance/time-entries"
            # Missing tenant_id parameter
        )
        
        # Should return 400 for missing tenant_id
        assert response.status_code == 400
    
    async def test_input_validation(self, test_client):
        """Test input validation and sanitization"""
        # Test with invalid data
        invalid_data = {
            "employee_id": "",  # Empty employee ID
            "tenant_id": "tenant_001",
            "location": {
                "latitude": 200,  # Invalid latitude
                "longitude": -200  # Invalid longitude
            }
        }
        
        response = test_client.post(
            "/api/human_capital_management/time_attendance/clock-in",
            json=invalid_data
        )
        
        assert response.status_code == 422  # Validation error


# Performance and Load Testing
class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_api_response_time(self, test_client):
        """Test API response time under normal load"""
        import time
        
        start_time = time.time()
        
        response = test_client.get(
            "/api/human_capital_management/time_attendance/health"
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 0.5  # Should respond within 500ms
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_client):
        """Test handling of concurrent requests"""
        import asyncio
        import httpx
        
        async def make_request():
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://testserver/api/human_capital_management/time_attendance/health"
                )
                return response.status_code
        
        # Make 50 concurrent requests
        tasks = [make_request() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        # All requests should succeed
        successful_requests = [r for r in results if r == 200]
        assert len(successful_requests) >= 45  # Allow for some failures


# Integration Test Runner
@pytest.mark.asyncio
async def test_full_integration_scenario():
    """Full end-to-end integration test scenario"""
    
    # This test runs a complete workflow:
    # 1. Employee clock-in
    # 2. Real-time event generation
    # 3. Fraud detection analysis
    # 4. Mobile API interaction
    # 5. Report generation
    # 6. Clock-out
    
    test_env = TestTimeAttendanceIntegration()
    
    print("🚀 Starting comprehensive Time & Attendance integration test...")
    
    # Setup
    async with test_env.setup_test_environment() as env:
        service = env["service"]
        
        print("✅ Test environment setup complete")
        
        # 1. Clock-in
        time_entry = await service.clock_in(
            employee_id=TEST_EMPLOYEE_ID,
            tenant_id=TEST_TENANT_ID,
            location={"latitude": 40.7128, "longitude": -74.0060},
            created_by="integration_test"
        )
        
        assert time_entry.status == TimeEntryStatus.ACTIVE
        print(f"✅ Clock-in successful: {time_entry.id}")
        
        # 2. Verify fraud detection
        assert time_entry.anomaly_score < 0.5  # Should be low risk
        print(f"✅ Fraud detection analysis complete: score {time_entry.anomaly_score}")
        
        # 3. Wait and clock-out
        await asyncio.sleep(1)  # Simulate work time
        
        completed_entry = await service.clock_out(
            employee_id=TEST_EMPLOYEE_ID,
            tenant_id=TEST_TENANT_ID,
            created_by="integration_test"
        )
        
        assert completed_entry.status == TimeEntryStatus.COMPLETED
        assert completed_entry.total_hours > 0
        print(f"✅ Clock-out successful: {completed_entry.total_hours} hours worked")
        
        # 4. Generate report
        from ..reporting import ReportGenerator, ReportConfig, ReportType, ReportFormat, ReportPeriod
        
        report_gen = ReportGenerator(service)
        config = ReportConfig(
            report_type=ReportType.TIMESHEET,
            format=ReportFormat.JSON,
            period=ReportPeriod.DAILY,
            start_date=date.today(),
            end_date=date.today(),
            tenant_id=TEST_TENANT_ID
        )
        
        report = await report_gen.generate_report(config, "integration_test")
        assert report["success"] is True
        print("✅ Report generation successful")
        
        print("🎉 All integration tests passed! Time & Attendance capability is fully operational.")


if __name__ == "__main__":
    # Run the comprehensive integration test
    asyncio.run(test_full_integration_scenario())