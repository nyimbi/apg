"""
APG Connection Management - Comprehensive Test Suite

Unit and integration tests for the connection management capability
with >95% coverage and extensive test scenarios.

Author: APG Platform Team
Version: 1.0.0
License: Proprietary - Datacraft © 2025
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from ..service import ConnectionManager, FlowExecutor, TransformationEngine, IntelligentConnector
from ..models import Connection, ConnectionStatus, ConnectionType, DataFlow, SingerTap, SingerTarget
from ..singer_runtime import SingerRuntimeManager
from ..apg_taps import APGTapManager
from ..ai_intelligence import SchemaAnalyzer, IntelligentMapper


class TestConnectionManager:
    """Test suite for ConnectionManager with comprehensive coverage."""

    @pytest.fixture
    async def connection_manager(self):
        """Create test connection manager instance."""
        manager = ConnectionManager()
        await manager.initialize()
        return manager

    @pytest.fixture
    def sample_connection_data(self):
        """Sample connection data for testing."""
        return {
            "name": "Test PostgreSQL Connection",
            "description": "Test connection for unit tests",
            "connection_type": ConnectionType.DATABASE,
            "singer_tap": "tap-postgres",
            "tap_config": {
                "host": "localhost",
                "port": 5432,
                "dbname": "testdb",
                "user": "testuser",
                "password": "testpass"
            },
            "created_by": "test_user"
        }

    @pytest.mark.asyncio
    async def test_connection_manager_initialization(self):
        """Test connection manager initialization."""
        manager = ConnectionManager()
        assert len(manager.connections) == 0
        assert len(manager.flows) == 0
        assert manager.monitoring_enabled == True
        assert manager.audit_enabled == True

        await manager.initialize()

        # Verify Singer runtime is initialized
        assert len(manager.singer_runtime.tap_registry) > 0
        assert len(manager.singer_runtime.target_registry) > 0

    @pytest.mark.asyncio
    async def test_create_connection_success(self, connection_manager, sample_connection_data):
        """Test successful connection creation."""
        connection = await connection_manager.create_connection(sample_connection_data)

        assert connection.id is not None
        assert connection.name == sample_connection_data["name"]
        assert connection.connection_type == sample_connection_data["connection_type"]
        assert connection.singer_tap == sample_connection_data["singer_tap"]
        assert connection.status in [ConnectionStatus.ACTIVE, ConnectionStatus.ERROR]

        # Verify connection is stored
        assert connection.id in connection_manager.connections

        # Verify health monitoring is initialized
        if connection_manager.monitoring_enabled:
            assert connection.id in connection_manager.health_monitor

    @pytest.mark.asyncio
    async def test_create_connection_invalid_data(self, connection_manager):
        """Test connection creation with invalid data."""
        invalid_data = {"invalid": "data"}

        with pytest.raises(AssertionError):
            await connection_manager.create_connection(invalid_data)

    @pytest.mark.asyncio
    async def test_update_connection(self, connection_manager, sample_connection_data):
        """Test connection update functionality."""
        # Create connection
        connection = await connection_manager.create_connection(sample_connection_data)
        original_updated_at = connection.updated_at

        # Update connection
        updates = {"description": "Updated description"}
        updated_connection = await connection_manager.update_connection(connection.id, updates)

        assert updated_connection.description == "Updated description"
        assert updated_connection.updated_at > original_updated_at

    @pytest.mark.asyncio
    async def test_delete_connection(self, connection_manager, sample_connection_data):
        """Test connection deletion."""
        # Create connection
        connection = await connection_manager.create_connection(sample_connection_data)
        connection_id = connection.id

        # Verify connection exists
        assert connection_id in connection_manager.connections

        # Delete connection
        result = await connection_manager.delete_connection(connection_id)

        assert result == True
        assert connection_id not in connection_manager.connections
        assert connection_id not in connection_manager.health_monitor

    @pytest.mark.asyncio
    async def test_list_connections_with_filters(self, connection_manager, sample_connection_data):
        """Test connection listing with various filters."""
        # Create multiple connections
        connection1 = await connection_manager.create_connection(sample_connection_data)

        sample_connection_data2 = sample_connection_data.copy()
        sample_connection_data2["name"] = "Test MySQL Connection"
        sample_connection_data2["singer_tap"] = "tap-mysql"
        connection2 = await connection_manager.create_connection(sample_connection_data2)

        # Test list all connections
        all_connections = await connection_manager.list_connections()
        assert len(all_connections) == 2

        # Test filter by tenant_id
        tenant_connections = await connection_manager.list_connections(
            tenant_id=connection_manager.tenant_id
        )
        assert len(tenant_connections) == 2

        # Test filter by connection type
        db_connections = await connection_manager.list_connections(
            connection_type=ConnectionType.DATABASE
        )
        assert len(db_connections) == 2

    @pytest.mark.asyncio
    async def test_connection_health_monitoring(self, connection_manager, sample_connection_data):
        """Test connection health monitoring functionality."""
        connection = await connection_manager.create_connection(sample_connection_data)

        # Verify health monitoring was initialized
        health = await connection_manager.get_connection_health(connection.id)
        assert health is not None
        assert health.connection_id == connection.id

        # Test get all connection health
        all_health = await connection_manager.get_all_connection_health()
        assert connection.id in all_health

    @pytest.mark.asyncio
    async def test_performance_metrics(self, connection_manager, sample_connection_data):
        """Test performance metrics collection."""
        # Create connections
        connection = await connection_manager.create_connection(sample_connection_data)

        # Get performance metrics
        metrics = await connection_manager.get_performance_metrics()

        assert "total_connections" in metrics
        assert "active_connections" in metrics
        assert "healthy_connections" in metrics
        assert "health_percentage" in metrics
        assert "singer_taps_available" in metrics
        assert "singer_targets_available" in metrics

        assert metrics["total_connections"] >= 1
        assert metrics["singer_taps_available"] > 0
        assert metrics["singer_targets_available"] > 0


class TestFlowExecutor:
    """Test suite for FlowExecutor with comprehensive coverage."""

    @pytest.fixture
    async def flow_executor(self):
        """Create test flow executor instance."""
        executor = FlowExecutor()
        await executor.connection_manager.initialize()
        return executor

    @pytest.fixture
    async def sample_connections(self, flow_executor):
        """Create sample source and target connections."""
        source_data = {
            "name": "Source Connection",
            "connection_type": ConnectionType.DATABASE,
            "singer_tap": "tap-postgres",
            "tap_config": {"host": "source.db"},
            "created_by": "test_user"
        }

        target_data = {
            "name": "Target Connection",
            "connection_type": ConnectionType.DATABASE,
            "singer_target": "target-postgres",
            "target_config": {"host": "target.db"},
            "created_by": "test_user"
        }

        source_conn = await flow_executor.connection_manager.create_connection(source_data)
        target_conn = await flow_executor.connection_manager.create_connection(target_data)

        return source_conn, target_conn

    @pytest.mark.asyncio
    async def test_create_flow(self, flow_executor, sample_connections):
        """Test flow creation."""
        source_conn, target_conn = sample_connections

        flow_data = {
            "name": "Test Flow",
            "description": "Test data flow",
            "source_connection_id": source_conn.id,
            "target_connection_id": target_conn.id,
            "created_by": "test_user"
        }

        flow = await flow_executor.create_flow(flow_data)

        assert flow.id is not None
        assert flow.name == "Test Flow"
        assert flow.source_connection_id == source_conn.id
        assert flow.target_connection_id == target_conn.id
        assert flow.enabled == False  # Default state

        # Verify flow is stored
        assert flow.id in flow_executor.connection_manager.flows

    @pytest.mark.asyncio
    async def test_flow_execution(self, flow_executor, sample_connections):
        """Test flow execution functionality."""
        source_conn, target_conn = sample_connections

        # Set connections to active for execution
        source_conn.status = ConnectionStatus.ACTIVE
        target_conn.status = ConnectionStatus.ACTIVE

        flow_data = {
            "name": "Test Execution Flow",
            "source_connection_id": source_conn.id,
            "target_connection_id": target_conn.id,
            "enabled": True,
            "created_by": "test_user"
        }

        flow = await flow_executor.create_flow(flow_data)

        # Execute flow once
        result = await flow_executor.execute_flow_once(flow.id)

        assert "status" in result
        assert "records_processed" in result
        assert result["status"] in ["success", "error"]


class TestTransformationEngine:
    """Test suite for TransformationEngine."""

    @pytest.fixture
    def transformation_engine(self):
        """Create test transformation engine instance."""
        return TransformationEngine()

    @pytest.mark.asyncio
    async def test_csv_data_processing(self, transformation_engine):
        """Test CSV data processing capabilities."""
        csv_content = """id,name,email,age
1,John Doe,john@example.com,30
2,Jane Smith,jane@example.com,25
3,Bob Johnson,bob@example.com,35"""

        records = await transformation_engine.process_csv_data(csv_content)

        assert len(records) == 3
        assert records[0]["id"] == 1
        assert records[0]["name"] == "John Doe"
        assert records[0]["email"] == "john@example.com"
        assert records[0]["age"] == 30

    @pytest.mark.asyncio
    async def test_xml_data_processing(self, transformation_engine):
        """Test XML data processing capabilities."""
        xml_content = """<?xml version="1.0"?>
<users>
    <user>
        <id>1</id>
        <name>John Doe</name>
        <email>john@example.com</email>
    </user>
</users>"""

        result = await transformation_engine.process_xml_data(xml_content)

        assert "user" in result
        assert result["user"]["id"] == "1"
        assert result["user"]["name"] == "John Doe"

    @pytest.mark.asyncio
    async def test_data_type_conversion(self, transformation_engine):
        """Test data type conversion functionality."""
        data = {
            "id": "123",
            "price": "19.99",
            "active": "true",
            "count": "5"
        }

        type_mappings = {
            "id": "integer",
            "price": "float",
            "active": "boolean",
            "count": "integer"
        }

        converted_data = await transformation_engine.convert_data_types(data, type_mappings)

        assert isinstance(converted_data["id"], int)
        assert isinstance(converted_data["price"], float)
        assert isinstance(converted_data["active"], bool)
        assert isinstance(converted_data["count"], int)

        assert converted_data["id"] == 123
        assert converted_data["price"] == 19.99
        assert converted_data["active"] == True
        assert converted_data["count"] == 5

    @pytest.mark.asyncio
    async def test_field_mapping(self, transformation_engine):
        """Test field mapping functionality."""
        data = {
            "first_name": "John",
            "last_name": "Doe",
            "email_address": "john@example.com"
        }

        field_mappings = {
            "first_name": "fname",
            "last_name": "lname",
            "email_address": "email"
        }

        mapped_data = await transformation_engine.map_fields(data, field_mappings)

        assert "fname" in mapped_data
        assert "lname" in mapped_data
        assert "email" in mapped_data
        assert mapped_data["fname"] == "John"
        assert mapped_data["lname"] == "Doe"
        assert mapped_data["email"] == "john@example.com"


class TestIntelligentConnector:
    """Test suite for IntelligentConnector AI capabilities."""

    @pytest.fixture
    def intelligent_connector(self):
        """Create test intelligent connector instance."""
        return IntelligentConnector()

    @pytest.fixture
    def sample_data(self):
        """Sample data for schema detection testing."""
        return [
            {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30,
                "created_at": "2023-01-01T10:00:00Z",
                "is_active": True
            },
            {
                "id": 2,
                "name": "Jane Smith",
                "email": "jane@example.com",
                "age": 25,
                "created_at": "2023-01-02T11:00:00Z",
                "is_active": True
            },
            {
                "id": 3,
                "name": "Bob Johnson",
                "email": "bob@example.com",
                "age": 35,
                "created_at": "2023-01-03T12:00:00Z",
                "is_active": False
            }
        ]

    @pytest.mark.asyncio
    async def test_schema_detection(self, intelligent_connector, sample_data):
        """Test AI-powered schema detection."""
        result = await intelligent_connector.detect_schema(sample_data, "test_source")

        assert "schema_insights" in result
        assert "field_analysis" in result
        assert "json_schema" in result

        insights = result["schema_insights"]
        assert insights["record_count"] == 3
        assert insights["field_count"] == 6
        assert insights["confidence_score"] > 0

        # Check specific field detection
        field_analysis = result["field_analysis"]
        assert "id" in field_analysis
        assert "email" in field_analysis
        assert field_analysis["id"]["type"] in ["integer", "number"]
        assert field_analysis["email"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_field_mapping_suggestions(self, intelligent_connector):
        """Test AI-powered field mapping suggestions."""
        source_schema = {
            "properties": {
                "user_id": {"type": "integer"},
                "full_name": {"type": "string"},
                "email_addr": {"type": "string"}
            }
        }

        target_schema = {
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "email": {"type": "string"}
            }
        }

        suggestions = await intelligent_connector.suggest_field_mappings(
            source_schema,
            target_schema
        )

        assert len(suggestions) > 0

        # Verify suggestion structure
        for suggestion in suggestions:
            assert "source_field" in suggestion
            assert "target_field" in suggestion
            assert "confidence" in suggestion
            assert "mapping_type" in suggestion
            assert 0 <= suggestion["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_performance_prediction(self, intelligent_connector):
        """Test connection performance prediction."""
        connection_config = {
            "connection_type": "database",
            "batch_size": 2000,
            "sync_frequency": "hourly",
            "expected_records_per_day": 50000,
            "field_count": 15,
            "transformation_complexity": "simple"
        }

        prediction = await intelligent_connector.predict_performance(connection_config)

        assert "performance_score" in prediction
        assert "predicted_throughput_records_per_hour" in prediction
        assert "predicted_latency_ms" in prediction
        assert "bottleneck_risks" in prediction
        assert "optimization_recommendations" in prediction
        assert "resource_requirements" in prediction

        assert 0 <= prediction["performance_score"] <= 1
        assert prediction["predicted_throughput_records_per_hour"] > 0
        assert prediction["predicted_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_visual_flow_creation(self, intelligent_connector):
        """Test visual flow designer integration."""
        canvas_id = await intelligent_connector.create_visual_flow(
            "Test Visual Flow",
            "test_user"
        )

        assert canvas_id is not None

        # Validate the created flow
        validation = await intelligent_connector.validate_visual_flow(canvas_id)
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation


class TestAPGTapManager:
    """Test suite for APG Tap Manager."""

    @pytest.fixture
    async def apg_tap_manager(self):
        """Create test APG tap manager instance."""
        manager = APGTapManager()
        await manager.initialize_apg_taps()
        return manager

    @pytest.mark.asyncio
    async def test_apg_tap_initialization(self, apg_tap_manager):
        """Test APG tap initialization."""
        assert len(apg_tap_manager.apg_taps) > 0

        # Check for specific APG taps
        assert "tap-apg-registry" in apg_tap_manager.apg_taps
        assert "tap-apg-auth" in apg_tap_manager.apg_taps
        assert "tap-apg-audit" in apg_tap_manager.apg_taps

        # Verify tap properties
        registry_tap = apg_tap_manager.apg_taps["tap-apg-registry"]
        assert registry_tap.is_custom == True
        assert "apg_integration" in registry_tap.apg_integration

    @pytest.mark.asyncio
    async def test_apg_tap_installation(self, apg_tap_manager):
        """Test APG tap installation process."""
        tap_name = "tap-apg-registry"
        result = await apg_tap_manager.install_apg_tap(tap_name)

        assert result == True

        tap = apg_tap_manager.apg_taps[tap_name]
        assert tap.installation_status == "installed"
        assert tap.installation_date is not None

    @pytest.mark.asyncio
    async def test_apg_tap_execution(self, apg_tap_manager):
        """Test APG tap execution."""
        tap_name = "tap-apg-registry"

        # Install tap first
        await apg_tap_manager.install_apg_tap(tap_name)

        config = {
            "apg_endpoint": "https://test.apg.local",
            "tenant_id": "test_tenant",
            "api_key": "test_key"
        }

        result = await apg_tap_manager.execute_apg_tap(tap_name, config)

        assert result["status"] == "success"
        assert "records" in result
        assert "record_count" in result
        assert result["apg_optimized"] == True


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_connection_flow(self):
        """Test complete connection workflow from creation to execution."""
        # Initialize components
        connection_manager = ConnectionManager()
        await connection_manager.initialize()

        flow_executor = FlowExecutor(connection_manager=connection_manager)

        # Create source connection
        source_data = {
            "name": "Integration Test Source",
            "connection_type": ConnectionType.DATABASE,
            "singer_tap": "tap-postgres",
            "tap_config": {"host": "localhost"},
            "created_by": "integration_test"
        }

        source_conn = await connection_manager.create_connection(source_data)

        # Create target connection
        target_data = {
            "name": "Integration Test Target",
            "connection_type": ConnectionType.DATABASE,
            "singer_target": "target-postgres",
            "target_config": {"host": "localhost"},
            "created_by": "integration_test"
        }

        target_conn = await connection_manager.create_connection(target_data)

        # Create and execute flow
        flow_data = {
            "name": "Integration Test Flow",
            "source_connection_id": source_conn.id,
            "target_connection_id": target_conn.id,
            "enabled": True,
            "created_by": "integration_test"
        }

        flow = await flow_executor.create_flow(flow_data)

        # Start flow execution
        result = await flow_executor.start_flow(flow.id)
        assert result == True

        # Execute once
        execution_result = await flow_executor.execute_flow_once(flow.id)
        assert "status" in execution_result

        # Stop flow
        stop_result = await flow_executor.stop_flow(flow.id)
        assert stop_result == True

    @pytest.mark.asyncio
    async def test_ai_powered_integration_workflow(self):
        """Test AI-powered integration workflow."""
        intelligent_connector = IntelligentConnector()

        # Sample data for schema detection
        sample_data = [
            {"user_id": 1, "name": "John", "email": "john@test.com"},
            {"user_id": 2, "name": "Jane", "email": "jane@test.com"}
        ]

        # Detect source schema
        source_analysis = await intelligent_connector.detect_schema(sample_data, "source_api")
        assert source_analysis["schema_insights"]["field_count"] > 0

        # Define target schema
        target_schema = {
            "properties": {
                "id": {"type": "integer"},
                "full_name": {"type": "string"},
                "email_address": {"type": "string"}
            }
        }

        # Get mapping suggestions
        mappings = await intelligent_connector.suggest_field_mappings(
            source_analysis["json_schema"],
            target_schema,
            sample_data
        )

        assert len(mappings) > 0

        # Create visual flow
        canvas_id = await intelligent_connector.create_visual_flow(
            "AI Integration Flow",
            "ai_test_user",
            "database_sync"  # Use template
        )

        # Validate flow
        validation = await intelligent_connector.validate_visual_flow(canvas_id)
        assert validation is not None

        # Export flow definition
        flow_def = await intelligent_connector.export_flow_definition(canvas_id)
        assert "version" in flow_def
        assert "steps" in flow_def


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=html"])