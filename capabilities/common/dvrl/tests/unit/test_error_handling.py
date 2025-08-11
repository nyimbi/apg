#!/usr/bin/env python3
"""
Unit Tests for DVRL Error Handling System
Tests comprehensive error handling, exception classes, and logging

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from capabilities.common.dvrl.error_handling import (
    DVRLException, ServiceUnavailableError, OperationError, RegistrationError,
    ConnectionError, QueryExecutionError, SchemaDiscoveryError, ValidationError,
    AuthenticationError, AuthorizationError, ConfigurationError,
    DVRLErrorHandler
)


class TestDVRLExceptionClasses:
    """Test suite for DVRL exception hierarchy"""
    
    def test_dvrl_exception_base_class(self):
        """Test base DVRLException class"""
        message = "Test error message"
        error_code = "TEST_ERROR"
        context = {"key": "value", "operation": "test"}
        
        exc = DVRLException(message, error_code, context)
        
        assert str(exc) == message
        assert exc.message == message
        assert exc.error_code == error_code
        assert exc.context == context
        assert isinstance(exc.timestamp, datetime)
        
    def test_dvrl_exception_defaults(self):
        """Test DVRLException with default values"""
        message = "Default error"
        
        exc = DVRLException(message)
        
        assert exc.message == message
        assert exc.error_code == "DVRLException"
        assert exc.context == {}
        assert isinstance(exc.timestamp, datetime)
    
    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError"""
        message = "Service not available"
        context = {"service": "singer_integration"}
        
        exc = ServiceUnavailableError(message, context=context)
        
        assert isinstance(exc, DVRLException)
        assert exc.message == message
        assert exc.error_code == "ServiceUnavailableError"
        assert exc.context == context
    
    def test_operation_error(self):
        """Test OperationError"""
        message = "Operation failed"
        
        exc = OperationError(message)
        
        assert isinstance(exc, DVRLException)
        assert exc.error_code == "OperationError"
    
    def test_registration_error(self):
        """Test RegistrationError"""
        message = "Data source registration failed"
        
        exc = RegistrationError(message)
        
        assert isinstance(exc, DVRLException)
        assert exc.error_code == "RegistrationError"
    
    def test_connection_error(self):
        """Test ConnectionError"""
        message = "Database connection failed"
        
        exc = ConnectionError(message)
        
        assert isinstance(exc, DVRLException)
        assert exc.error_code == "ConnectionError"
    
    def test_query_execution_error(self):
        """Test QueryExecutionError"""
        message = "Query execution failed"
        
        exc = QueryExecutionError(message)
        
        assert isinstance(exc, DVRLException)
        assert exc.error_code == "QueryExecutionError"
    
    def test_schema_discovery_error(self):
        """Test SchemaDiscoveryError"""
        message = "Schema discovery failed"
        
        exc = SchemaDiscoveryError(message)
        
        assert isinstance(exc, DVRLException)
        assert exc.error_code == "SchemaDiscoveryError"
    
    def test_validation_error(self):
        """Test ValidationError"""
        message = "Data validation failed"
        
        exc = ValidationError(message)
        
        assert isinstance(exc, DVRLException)
        assert exc.error_code == "ValidationError"
    
    def test_authentication_error(self):
        """Test AuthenticationError"""
        message = "Authentication failed"
        
        exc = AuthenticationError(message)
        
        assert isinstance(exc, DVRLException)
        assert exc.error_code == "AuthenticationError"
    
    def test_authorization_error(self):
        """Test AuthorizationError"""
        message = "Authorization failed"
        
        exc = AuthorizationError(message)
        
        assert isinstance(exc, DVRLException)
        assert exc.error_code == "AuthorizationError"
    
    def test_configuration_error(self):
        """Test ConfigurationError"""
        message = "Configuration invalid"
        
        exc = ConfigurationError(message)
        
        assert isinstance(exc, DVRLException)
        assert exc.error_code == "ConfigurationError"


class TestDVRLErrorHandler:
    """Test suite for DVRLErrorHandler class"""
    
    @pytest.fixture
    def error_handler(self):
        """Create DVRLErrorHandler instance for testing"""
        return DVRLErrorHandler('test_tenant', 'test_user')
    
    def test_error_handler_initialization(self, error_handler):
        """Test DVRLErrorHandler initialization"""
        assert error_handler.tenant_id == 'test_tenant'
        assert error_handler.user_id == 'test_user'
        assert isinstance(error_handler.error_history, list)
        assert isinstance(error_handler.error_counts, dict)
        assert len(error_handler.error_history) == 0
        assert len(error_handler.error_counts) == 0
    
    @patch('capabilities.common.dvrl.error_handling.REAL_ERROR_HANDLING_AVAILABLE', False)
    async def test_handle_error_fallback_mode(self, error_handler):
        """Test error handling in fallback mode"""
        test_error = Exception("Test error")
        context = {"operation": "test_operation", "data": "test_data"}
        
        result = await error_handler.handle_error(test_error, context, "test_op")
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'error_id' in result
        assert 'timestamp' in result
        assert 'tenant_id' in result
        assert result['tenant_id'] == 'test_tenant'
        assert result['user_id'] == 'test_user'
        
        # Check error was recorded in history
        assert len(error_handler.error_history) == 1
        assert error_handler.error_history[0]['error_type'] == 'Exception'
        assert error_handler.error_history[0]['message'] == 'Test error'
    
    async def test_handle_error_with_dvrl_exception(self, error_handler):
        """Test handling of DVRL-specific exceptions"""
        test_error = ValidationError("Invalid input data", 
                                    error_code="VALIDATION_001", 
                                    context={"field": "name"})
        context = {"operation": "data_validation"}
        
        result = await error_handler.handle_error(test_error, context, "validation")
        
        assert result['error_type'] == 'ValidationError'
        assert result['error_code'] == 'VALIDATION_001'
        assert 'field' in result['error_context']
        assert result['error_context']['field'] == 'name'
    
    async def test_error_counting(self, error_handler):
        """Test error counting functionality"""
        # Generate multiple errors of the same type
        for i in range(3):
            await error_handler.handle_error(
                ValidationError(f"Error {i}"), 
                {"iteration": i}, 
                "test"
            )
        
        # Generate different error type
        await error_handler.handle_error(
            ConnectionError("Connection failed"), 
            {}, 
            "connection_test"
        )
        
        # Check error counts
        assert error_handler.error_counts.get('ValidationError', 0) == 3
        assert error_handler.error_counts.get('ConnectionError', 0) == 1
        assert len(error_handler.error_history) == 4
    
    async def test_error_severity_handling(self, error_handler):
        """Test different error severity levels"""
        test_error = QueryExecutionError("Query timeout")
        context = {"query": "SELECT * FROM large_table"}
        
        # Test with different severity levels
        for severity in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
            result = await error_handler.handle_error(
                test_error, context, "query_test", severity
            )
            assert result['severity'] == severity
    
    async def test_error_context_enrichment(self, error_handler):
        """Test that error context is properly enriched"""
        test_error = OperationError("Operation failed")
        context = {
            "operation": "data_sync",
            "source": "postgresql_db",
            "records": 1000
        }
        
        result = await error_handler.handle_error(test_error, context, "sync")
        
        # Check that context is preserved and enriched
        assert 'operation' in result['context']
        assert 'source' in result['context']
        assert 'records' in result['context']
        assert result['context']['operation'] == 'data_sync'
        assert result['context']['source'] == 'postgresql_db'
        assert result['context']['records'] == 1000


class TestErrorHandlerIntegration:
    """Integration tests for error handling system"""
    
    async def test_exception_hierarchy_inheritance(self):
        """Test that all DVRL exceptions inherit properly"""
        exceptions = [
            ServiceUnavailableError, OperationError, RegistrationError,
            ConnectionError, QueryExecutionError, SchemaDiscoveryError,
            ValidationError, AuthenticationError, AuthorizationError,
            ConfigurationError
        ]
        
        for exc_class in exceptions:
            exc = exc_class("Test message")
            assert isinstance(exc, DVRLException)
            assert isinstance(exc, Exception)
            assert hasattr(exc, 'message')
            assert hasattr(exc, 'error_code')
            assert hasattr(exc, 'context')
            assert hasattr(exc, 'timestamp')
    
    async def test_error_serialization(self):
        """Test that errors can be properly serialized"""
        error = ValidationError(
            "Invalid data format",
            error_code="VAL_001",
            context={"field": "email", "value": "invalid_email"}
        )
        
        handler = DVRLErrorHandler('test_tenant', 'test_user')
        result = await handler.handle_error(error, {"operation": "validate"}, "test")
        
        # Should be JSON-serializable
        import json
        try:
            json_str = json.dumps(result, default=str)  # default=str for datetime
            parsed = json.loads(json_str)
            assert parsed['error_type'] == 'ValidationError'
            assert parsed['error_code'] == 'VAL_001'
        except (TypeError, ValueError) as e:
            pytest.fail(f"Error result is not JSON-serializable: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])