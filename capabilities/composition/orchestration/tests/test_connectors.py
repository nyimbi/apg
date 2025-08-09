#!/usr/bin/env python3
"""
APG Workflow Orchestration Connector Tests

Comprehensive tests for APG connectors and external system connectors.
Tests connector functionality, integration, error handling, and performance.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# APG Core imports
from ..connectors.apg_connectors import *
from ..connectors.external_connectors import *
from ..connectors.base import ConnectorBase, ConnectorConfig
from ..models import *

# Test utilities
from .conftest import TestHelpers


class TestAPGConnectors:
	"""Test APG capability connectors."""
	
	@pytest.mark.unit
	@pytest.mark.apg
	async def test_auth_rbac_connector(self):
		"""Test AuthRBACConnector functionality."""
		config = ConnectorConfig(
			connector_type="auth_rbac",
			config={
				"operation": "validate_user",
				"user_id": "test_user",
				"required_permissions": ["workflow:execute"]
			}
		)
		
		# Mock APG auth service
		with patch('..connectors.apg_connectors.APGAuthService') as mock_service:
			mock_service.return_value.validate_user_permissions = AsyncMock(
				return_value={"valid": True, "permissions": ["workflow:execute", "workflow:read"]}
			)
			
			connector = AuthRBACConnector(config)
			result = await connector.execute({})
			
			assert result["valid"] is True
			assert "workflow:execute" in result["permissions"]
			mock_service.return_value.validate_user_permissions.assert_called_once()
	
	@pytest.mark.unit
	@pytest.mark.apg 
	async def test_auth_rbac_connector_permission_denied(self):
		"""Test AuthRBACConnector with insufficient permissions."""
		config = ConnectorConfig(
			connector_type="auth_rbac",
			config={
				"operation": "validate_user",
				"user_id": "test_user",
				"required_permissions": ["admin:all"]
			}
		)
		
		# Mock APG auth service - insufficient permissions
		with patch('..connectors.apg_connectors.APGAuthService') as mock_service:
			mock_service.return_value.validate_user_permissions = AsyncMock(
				return_value={"valid": False, "permissions": ["workflow:read"], "error": "Insufficient permissions"}
			)
			
			connector = AuthRBACConnector(config)
			result = await connector.execute({})
			
			assert result["valid"] is False
			assert "error" in result
			assert "Insufficient permissions" in result["error"]
	
	@pytest.mark.unit
	@pytest.mark.apg
	async def test_audit_compliance_connector(self):
		"""Test AuditComplianceConnector functionality."""
		config = ConnectorConfig(
			connector_type="audit_compliance",
			config={
				"operation": "log_activity",
				"activity_type": "workflow_execution",
				"details": {"workflow_id": "wf_123", "user_id": "user_456"}
			}
		)
		
		# Mock APG audit service
		with patch('..connectors.apg_connectors.APGAuditService') as mock_service:
			mock_service.return_value.log_activity = AsyncMock(
				return_value={"audit_id": "audit_789", "logged_at": datetime.utcnow().isoformat()}
			)
			
			connector = AuditComplianceConnector(config)
			result = await connector.execute({})
			
			assert "audit_id" in result
			assert "logged_at" in result
			assert result["audit_id"] == "audit_789"
			mock_service.return_value.log_activity.assert_called_once()
	
	@pytest.mark.unit
	@pytest.mark.apg
	async def test_data_lake_connector_store_dataset(self):
		"""Test DataLakeConnector dataset storage."""
		config = ConnectorConfig(
			connector_type="data_lake", 
			config={
				"operation": "store_dataset",
				"dataset_name": "workflow_results",
				"data": [{"id": 1, "value": "test"}, {"id": 2, "value": "data"}],
				"format": "parquet",
				"metadata": {"source": "workflow", "version": "1.0"}
			}
		)
		
		# Mock APG data lake service
		with patch('..connectors.apg_connectors.APGDataLakeService') as mock_service:
			mock_service.return_value.store_dataset = AsyncMock(
				return_value={
					"dataset_id": "ds_123",
					"location": "s3://data-lake/workflow_results.parquet", 
					"records_stored": 2,
					"storage_size": 1024
				}
			)
			
			connector = DataLakeConnector(config)
			result = await connector.execute({})
			
			assert result["dataset_id"] == "ds_123"
			assert result["records_stored"] == 2
			assert "s3://" in result["location"]
			mock_service.return_value.store_dataset.assert_called_once()
	
	@pytest.mark.unit
	@pytest.mark.apg
	async def test_data_lake_connector_retrieve_dataset(self):
		"""Test DataLakeConnector dataset retrieval."""
		config = ConnectorConfig(
			connector_type="data_lake",
			config={
				"operation": "get_dataset",
				"dataset_id": "ds_123",
				"format": "json"
			}
		)
		
		# Mock APG data lake service
		with patch('..connectors.apg_connectors.APGDataLakeService') as mock_service:
			mock_service.return_value.get_dataset = AsyncMock(
				return_value={
					"dataset_id": "ds_123",
					"data": [{"id": 1, "value": "test"}, {"id": 2, "value": "data"}],
					"metadata": {"records": 2, "format": "json"}
				}
			)
			
			connector = DataLakeConnector(config)
			result = await connector.execute({})
			
			assert result["dataset_id"] == "ds_123"
			assert len(result["data"]) == 2
			assert result["metadata"]["records"] == 2
			mock_service.return_value.get_dataset.assert_called_once()
	
	@pytest.mark.unit
	@pytest.mark.apg
	async def test_real_time_collaboration_connector(self):
		"""Test RealTimeCollaborationConnector functionality."""
		config = ConnectorConfig(
			connector_type="real_time_collaboration",
			config={
				"operation": "broadcast_update",
				"session_id": "collab_123",
				"update_type": "workflow_changed", 
				"data": {"workflow_id": "wf_456", "changes": ["task_added"]}
			}
		)
		
		# Mock APG collaboration service
		with patch('..connectors.apg_connectors.APGCollaborationService') as mock_service:
			mock_service.return_value.broadcast_update = AsyncMock(
				return_value={
					"broadcast_id": "bc_789",
					"recipients": ["user1", "user2"],
					"delivered": True
				}
			)
			
			connector = RealTimeCollaborationConnector(config)
			result = await connector.execute({})
			
			assert result["broadcast_id"] == "bc_789"
			assert result["delivered"] is True
			assert len(result["recipients"]) == 2
			mock_service.return_value.broadcast_update.assert_called_once()
	
	@pytest.mark.unit
	@pytest.mark.apg
	async def test_natural_language_processing_connector(self):
		"""Test NaturalLanguageProcessingConnector functionality."""
		config = ConnectorConfig(
			connector_type="natural_language_processing",
			config={
				"operation": "analyze_sentiment",
				"text": "This workflow is working great!",
				"language": "en"
			}
		)
		
		# Mock APG NLP service
		with patch('..connectors.apg_connectors.APGNLPService') as mock_service:
			mock_service.return_value.analyze_sentiment = AsyncMock(
				return_value={
					"sentiment": "positive",
					"confidence": 0.95,
					"emotions": {"joy": 0.8, "satisfaction": 0.7}
				}
			)
			
			connector = NaturalLanguageProcessingConnector(config)
			result = await connector.execute({})
			
			assert result["sentiment"] == "positive"
			assert result["confidence"] == 0.95
			assert result["emotions"]["joy"] == 0.8
			mock_service.return_value.analyze_sentiment.assert_called_once()
	
	@pytest.mark.unit
	@pytest.mark.apg
	async def test_computer_vision_connector(self):
		"""Test ComputerVisionConnector functionality."""
		config = ConnectorConfig(
			connector_type="computer_vision",
			config={
				"operation": "analyze_image",
				"image_url": "https://example.com/image.jpg",
				"analysis_types": ["object_detection", "text_extraction"]
			}
		)
		
		# Mock APG computer vision service
		with patch('..connectors.apg_connectors.APGComputerVisionService') as mock_service:
			mock_service.return_value.analyze_image = AsyncMock(
				return_value={
					"objects": [{"type": "person", "confidence": 0.9}, {"type": "car", "confidence": 0.8}],
					"text": ["STOP", "Main St"],
					"analysis_id": "cv_123"
				}
			)
			
			connector = ComputerVisionConnector(config)
			result = await connector.execute({})
			
			assert len(result["objects"]) == 2
			assert "person" in [obj["type"] for obj in result["objects"]]
			assert "STOP" in result["text"]
			assert result["analysis_id"] == "cv_123"
			mock_service.return_value.analyze_image.assert_called_once()
	
	@pytest.mark.unit
	@pytest.mark.apg
	async def test_voice_processing_connector(self):
		"""Test VoiceProcessingConnector functionality."""
		config = ConnectorConfig(
			connector_type="voice_processing",
			config={
				"operation": "speech_to_text",
				"audio_url": "https://example.com/audio.wav",
				"language": "en-US"
			}
		)
		
		# Mock APG voice processing service
		with patch('..connectors.apg_connectors.APGVoiceService') as mock_service:
			mock_service.return_value.speech_to_text = AsyncMock(
				return_value={
					"transcript": "Execute workflow with high priority",
					"confidence": 0.92,
					"processing_time": 2.5
				}
			)
			
			connector = VoiceProcessingConnector(config)
			result = await connector.execute({})
			
			assert result["transcript"] == "Execute workflow with high priority"
			assert result["confidence"] == 0.92
			assert result["processing_time"] == 2.5
			mock_service.return_value.speech_to_text.assert_called_once()


class TestExternalConnectors:
	"""Test external system connectors."""
	
	@pytest.mark.unit
	async def test_rest_api_connector_get(self):
		"""Test RESTAPIConnector GET request."""
		config = ConnectorConfig(
			connector_type="rest_api",
			config={
				"url": "https://api.example.com/users/123",
				"method": "GET",
				"headers": {"Authorization": "Bearer token123"}
			}
		)
		
		# Mock httpx client
		with patch('httpx.AsyncClient') as mock_client:
			mock_response = Mock()
			mock_response.status_code = 200
			mock_response.json.return_value = {"id": 123, "name": "John Doe", "email": "john@example.com"}
			mock_response.headers = {"Content-Type": "application/json"}
			
			mock_client.return_value.__aenter__.return_value.request = AsyncMock(return_value=mock_response)
			
			connector = RESTAPIConnector(config)
			result = await connector.execute({})
			
			assert result["status_code"] == 200
			assert result["data"]["id"] == 123
			assert result["data"]["name"] == "John Doe"
	
	@pytest.mark.unit
	async def test_rest_api_connector_post(self):
		"""Test RESTAPIConnector POST request."""
		config = ConnectorConfig(
			connector_type="rest_api",
			config={
				"url": "https://api.example.com/users",
				"method": "POST",
				"headers": {"Content-Type": "application/json"},
				"json": {"name": "Jane Doe", "email": "jane@example.com"}
			}
		)
		
		# Mock httpx client
		with patch('httpx.AsyncClient') as mock_client:
			mock_response = Mock()  
			mock_response.status_code = 201
			mock_response.json.return_value = {"id": 124, "name": "Jane Doe", "email": "jane@example.com"}
			
			mock_client.return_value.__aenter__.return_value.request = AsyncMock(return_value=mock_response)
			
			connector = RESTAPIConnector(config)
			result = await connector.execute({})
			
			assert result["status_code"] == 201
			assert result["data"]["id"] == 124
			assert result["data"]["name"] == "Jane Doe"
	
	@pytest.mark.unit
	async def test_rest_api_connector_error_handling(self):
		"""Test RESTAPIConnector error handling."""
		config = ConnectorConfig(
			connector_type="rest_api",
			config={
				"url": "https://api.example.com/invalid",
				"method": "GET"
			}
		)
		
		# Mock httpx client with error
		with patch('httpx.AsyncClient') as mock_client:
			mock_response = Mock()
			mock_response.status_code = 404
			mock_response.text = "Not Found"
			mock_response.json.side_effect = ValueError("No JSON object could be decoded")
			
			mock_client.return_value.__aenter__.return_value.request = AsyncMock(return_value=mock_response)
			
			connector = RESTAPIConnector(config)
			result = await connector.execute({})
			
			assert result["status_code"] == 404
			assert result["error"] == "Not Found"
			assert "data" not in result
	
	@pytest.mark.unit
	async def test_database_connector_select(self):
		"""Test DatabaseConnector SELECT operation."""
		config = ConnectorConfig(
			connector_type="database",
			config={
				"connection_string": "postgresql://user:pass@localhost/testdb",
				"query": "SELECT id, name FROM users WHERE active = %(active)s",
				"parameters": {"active": True},
				"operation": "select"
			}
		)
		
		# Mock SQLAlchemy engine and connection
		with patch('sqlalchemy.create_engine') as mock_engine:
			mock_connection = Mock()
			mock_result = Mock()
			mock_result.fetchall.return_value = [
				(1, "John Doe"),
				(2, "Jane Smith")
			]
			mock_result.keys.return_value = ["id", "name"]
			mock_connection.execute.return_value = mock_result
			mock_engine.return_value.connect.return_value.__enter__.return_value = mock_connection
			
			connector = DatabaseConnector(config)
			result = await connector.execute({})
			
			assert len(result["rows"]) == 2
			assert result["rows"][0]["id"] == 1
			assert result["rows"][0]["name"] == "John Doe"
			assert result["rows"][1]["id"] == 2
			assert result["rows"][1]["name"] == "Jane Smith"
	
	@pytest.mark.unit
	async def test_database_connector_insert(self):
		"""Test DatabaseConnector INSERT operation."""
		config = ConnectorConfig(
			connector_type="database",
			config={
				"connection_string": "postgresql://user:pass@localhost/testdb",
				"query": "INSERT INTO users (name, email) VALUES (%(name)s, %(email)s)",
				"parameters": {"name": "Bob Wilson", "email": "bob@example.com"},
				"operation": "insert"
			}
		)
		
		# Mock SQLAlchemy engine and connection
		with patch('sqlalchemy.create_engine') as mock_engine:
			mock_connection = Mock()
			mock_result = Mock()
			mock_result.rowcount = 1
			mock_connection.execute.return_value = mock_result
			mock_engine.return_value.connect.return_value.__enter__.return_value = mock_connection
			
			connector = DatabaseConnector(config)
			result = await connector.execute({})
			
			assert result["rows_affected"] == 1
			assert result["operation"] == "insert"
	
	@pytest.mark.unit
	async def test_file_system_connector_read(self):
		"""Test FileSystemConnector read operation."""
		with tempfile.TemporaryDirectory() as temp_dir:
			test_file = Path(temp_dir) / "test.txt"
			test_content = "Hello, World!"
			test_file.write_text(test_content)
			
			config = ConnectorConfig(
				connector_type="file_system",
				config={
					"operation": "read",
					"file_path": str(test_file)
				}
			)
			
			connector = FileSystemConnector(config)
			result = await connector.execute({})
			
			assert result["content"] == test_content
			assert result["size"] == len(test_content)
			assert result["exists"] is True
	
	@pytest.mark.unit
	async def test_file_system_connector_write(self):
		"""Test FileSystemConnector write operation."""
		with tempfile.TemporaryDirectory() as temp_dir:
			test_file = Path(temp_dir) / "output.txt"
			test_content = "New file content"
			
			config = ConnectorConfig(
				connector_type="file_system",
				config={
					"operation": "write",
					"file_path": str(test_file),
					"content": test_content
				}
			)
			
			connector = FileSystemConnector(config)
			result = await connector.execute({})
			
			assert result["written"] is True
			assert result["size"] == len(test_content)
			assert test_file.read_text() == test_content
	
	@pytest.mark.unit
	async def test_file_system_connector_list(self):
		"""Test FileSystemConnector list operation."""
		with tempfile.TemporaryDirectory() as temp_dir:
			# Create test files
			(Path(temp_dir) / "file1.txt").write_text("content1")
			(Path(temp_dir) / "file2.txt").write_text("content2")
			(Path(temp_dir) / "subdir").mkdir()
			(Path(temp_dir) / "subdir" / "file3.txt").write_text("content3")
			
			config = ConnectorConfig(
				connector_type="file_system",
				config={
					"operation": "list",
					"directory_path": temp_dir,
					"recursive": True
				}
			)
			
			connector = FileSystemConnector(config)
			result = await connector.execute({})
			
			assert len(result["files"]) >= 3
			file_names = [f["name"] for f in result["files"]]
			assert "file1.txt" in file_names
			assert "file2.txt" in file_names
			assert "file3.txt" in file_names
	
	@pytest.mark.unit
	async def test_message_queue_connector_send(self):
		"""Test MessageQueueConnector send operation."""
		config = ConnectorConfig(
			connector_type="message_queue",
			config={
				"queue_url": "redis://localhost:6379",
				"queue_name": "test_queue",
				"operation": "send",
				"message": {"type": "workflow_event", "data": {"workflow_id": "wf_123"}}
			}
		)
		
		# Mock Redis client
		with patch('redis.asyncio.from_url') as mock_redis:
			mock_client = AsyncMock()
			mock_client.lpush.return_value = 1
			mock_redis.return_value = mock_client
			
			connector = MessageQueueConnector(config)
			result = await connector.execute({})
			
			assert result["sent"] is True
			assert result["queue_name"] == "test_queue"
			mock_client.lpush.assert_called_once()
	
	@pytest.mark.unit
	async def test_message_queue_connector_receive(self):
		"""Test MessageQueueConnector receive operation."""
		config = ConnectorConfig(
			connector_type="message_queue",
			config={
				"queue_url": "redis://localhost:6379",
				"queue_name": "test_queue",
				"operation": "receive",
				"timeout": 5
			}
		)
		
		# Mock Redis client
		with patch('redis.asyncio.from_url') as mock_redis:
			mock_client = AsyncMock()
			mock_client.brpop.return_value = ("test_queue", json.dumps({"type": "test_message", "data": {"id": 123}}))
			mock_redis.return_value = mock_client
			
			connector = MessageQueueConnector(config)
			result = await connector.execute({})
			
			assert result["received"] is True
			assert result["message"]["type"] == "test_message"
			assert result["message"]["data"]["id"] == 123
			mock_client.brpop.assert_called_once()
	
	@pytest.mark.unit
	async def test_cloud_storage_connector_upload(self):
		"""Test CloudStorageConnector upload operation."""
		config = ConnectorConfig(
			connector_type="cloud_storage",
			config={
				"provider": "aws_s3",
				"bucket": "test-bucket",
				"operation": "upload",
				"local_path": "/tmp/test.txt",
				"remote_path": "uploads/test.txt",
				"credentials": {"access_key": "test", "secret_key": "test"}
			}
		)
		
		# Mock boto3 client
		with patch('boto3.client') as mock_boto3:
			mock_s3 = Mock()
			mock_s3.upload_file.return_value = None
			mock_boto3.return_value = mock_s3
			
			connector = CloudStorageConnector(config)
			result = await connector.execute({})
			
			assert result["uploaded"] is True
			assert result["bucket"] == "test-bucket"
			assert result["remote_path"] == "uploads/test.txt"
			mock_s3.upload_file.assert_called_once()
	
	@pytest.mark.unit
	async def test_cloud_storage_connector_download(self):
		"""Test CloudStorageConnector download operation."""
		config = ConnectorConfig(
			connector_type="cloud_storage",
			config={
				"provider": "aws_s3",
				"bucket": "test-bucket", 
				"operation": "download",
				"remote_path": "uploads/test.txt",
				"local_path": "/tmp/downloaded.txt",
				"credentials": {"access_key": "test", "secret_key": "test"}
			}
		)
		
		# Mock boto3 client
		with patch('boto3.client') as mock_boto3:
			mock_s3 = Mock()
			mock_s3.download_file.return_value = None
			mock_boto3.return_value = mock_s3
			
			connector = CloudStorageConnector(config)
			result = await connector.execute({})
			
			assert result["downloaded"] is True
			assert result["local_path"] == "/tmp/downloaded.txt"
			mock_s3.download_file.assert_called_once()
	
	@pytest.mark.unit
	async def test_email_connector_send(self):
		"""Test EmailConnector send operation."""
		config = ConnectorConfig(
			connector_type="email",
			config={
				"smtp_server": "smtp.example.com",
				"smtp_port": 587,
				"username": "test@example.com",
				"password": "password",
				"operation": "send",
				"to": ["recipient@example.com"],
				"subject": "Test Email",
				"body": "This is a test email",
				"body_type": "text"
			}
		)
		
		# Mock smtplib
		with patch('smtplib.SMTP') as mock_smtp:
			mock_server = Mock()
			mock_smtp.return_value.__enter__.return_value = mock_server
			
			connector = EmailConnector(config)
			result = await connector.execute({})
			
			assert result["sent"] is True
			assert result["recipients"] == ["recipient@example.com"]
			mock_server.send_message.assert_called_once()


class TestConnectorFramework:
	"""Test connector framework and base functionality."""
	
	@pytest.mark.unit
	async def test_connector_base_abstract_methods(self):
		"""Test ConnectorBase abstract methods."""
		config = ConnectorConfig(connector_type="test", config={})
		
		# Should not be able to instantiate abstract base class
		with pytest.raises(TypeError):
			ConnectorBase(config)
	
	@pytest.mark.unit
	async def test_custom_connector_implementation(self):
		"""Test custom connector implementation."""
		class CustomConnector(ConnectorBase):
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				return {
					"message": "Custom connector executed",
					"context_keys": list(context.keys()),
					"config": self.config.config
				}
		
		config = ConnectorConfig(connector_type="custom", config={"param1": "value1"})
		connector = CustomConnector(config)
		
		result = await connector.execute({"input": "test"})
		
		assert result["message"] == "Custom connector executed"
		assert "input" in result["context_keys"]
		assert result["config"]["param1"] == "value1"
	
	@pytest.mark.unit 
	async def test_connector_validation(self):
		"""Test connector configuration validation."""
		# Valid configuration
		valid_config = ConnectorConfig(
			connector_type="rest_api",
			config={
				"url": "https://api.example.com",
				"method": "GET"
			}
		)
		assert valid_config.connector_type == "rest_api"
		assert valid_config.config["url"] == "https://api.example.com"
		
		# Test configuration with invalid data should use Pydantic validation
		with pytest.raises(ValueError):
			ConnectorConfig(
				connector_type="",  # Empty connector type
				config={}
			)
	
	@pytest.mark.unit
	async def test_connector_error_handling(self):
		"""Test connector error handling mechanisms."""
		class FailingConnector(ConnectorBase):
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				if context.get("should_fail"):
					raise Exception("Simulated connector failure")
				return {"success": True}
		
		config = ConnectorConfig(connector_type="failing", config={})
		connector = FailingConnector(config)
		
		# Test successful execution
		result = await connector.execute({"should_fail": False})
		assert result["success"] is True
		
		# Test failure handling
		with pytest.raises(Exception) as exc_info:
			await connector.execute({"should_fail": True})
		assert "Simulated connector failure" in str(exc_info.value)
	
	@pytest.mark.unit
	async def test_connector_timeout_handling(self):
		"""Test connector timeout handling."""
		class SlowConnector(ConnectorBase):
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				await asyncio.sleep(context.get("delay", 0.1))
				return {"completed": True}
		
		config = ConnectorConfig(
			connector_type="slow",
			config={},
			timeout_seconds=0.05  # Very short timeout
		)
		connector = SlowConnector(config)
		
		# Should complete within timeout
		result = await connector.execute({"delay": 0.01})
		assert result["completed"] is True
		
		# Should timeout
		with pytest.raises(asyncio.TimeoutError):
			await asyncio.wait_for(
				connector.execute({"delay": 0.1}),
				timeout=0.05
			)
	
	@pytest.mark.unit
	async def test_connector_retry_mechanism(self):
		"""Test connector retry mechanisms."""
		class RetryableConnector(ConnectorBase):
			def __init__(self, config):
				super().__init__(config)
				self.attempt_count = 0
			
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				self.attempt_count += 1
				if self.attempt_count < context.get("fail_attempts", 0):
					raise Exception(f"Attempt {self.attempt_count} failed")
				return {"attempt": self.attempt_count, "success": True}
		
		config = ConnectorConfig(connector_type="retryable", config={})
		connector = RetryableConnector(config)
		
		# Test successful execution after retries
		max_retries = 3
		for attempt in range(max_retries):
			try:
				result = await connector.execute({"fail_attempts": 2})
				assert result["success"] is True
				assert result["attempt"] == 2
				break
			except Exception:
				if attempt == max_retries - 1:
					raise
				continue


class TestConnectorPerformance:
	"""Test connector performance and resource usage."""
	
	@pytest.mark.performance
	async def test_concurrent_connector_execution(self):
		"""Test concurrent execution of multiple connectors."""
		class MockConnector(ConnectorBase):
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				await asyncio.sleep(0.1)  # Simulate work
				return {"id": context["id"], "completed": True}
		
		config = ConnectorConfig(connector_type="mock", config={})
		
		# Execute multiple connectors concurrently
		connectors = [MockConnector(config) for _ in range(10)]
		contexts = [{"id": i} for i in range(10)]
		
		start_time = datetime.utcnow()
		results = await asyncio.gather(*[
			connector.execute(context) 
			for connector, context in zip(connectors, contexts)
		])
		end_time = datetime.utcnow()
		
		execution_time = (end_time - start_time).total_seconds()
		
		# Should complete in approximately 0.1 seconds (concurrent), not 1.0 second (sequential)
		assert execution_time < 0.5
		assert len(results) == 10
		assert all(result["completed"] for result in results)
	
	@pytest.mark.performance
	async def test_connector_memory_usage(self):
		"""Test connector memory usage patterns."""
		class MemoryConnector(ConnectorBase):
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				# Create some data to use memory
				data = [i for i in range(context.get("data_size", 1000))]
				return {"data_length": len(data), "memory_used": True}
		
		config = ConnectorConfig(connector_type="memory", config={})
		connector = MemoryConnector(config)
		
		# Execute with different data sizes
		small_result = await connector.execute({"data_size": 100})
		large_result = await connector.execute({"data_size": 10000})
		
		assert small_result["data_length"] == 100
		assert large_result["data_length"] == 10000
		assert small_result["memory_used"] is True
		assert large_result["memory_used"] is True
	
	@pytest.mark.performance
	async def test_connector_batch_processing(self):
		"""Test connector batch processing capabilities."""
		class BatchConnector(ConnectorBase):
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				items = context.get("items", [])
				processed_items = []
				
				for item in items:
					# Simulate processing
					processed_item = {"original": item, "processed": item * 2}
					processed_items.append(processed_item)
				
				return {
					"input_count": len(items),
					"output_count": len(processed_items),
					"processed_items": processed_items
				}
		
		config = ConnectorConfig(connector_type="batch", config={})
		connector = BatchConnector(config)
		
		# Test batch processing
		items = list(range(100))
		result = await connector.execute({"items": items})
		
		assert result["input_count"] == 100
		assert result["output_count"] == 100
		assert len(result["processed_items"]) == 100
		assert result["processed_items"][0]["original"] == 0
		assert result["processed_items"][0]["processed"] == 0


class TestConnectorSecurity:
	"""Test connector security features and vulnerability prevention."""
	
	@pytest.mark.security
	async def test_connector_input_sanitization(self):
		"""Test connector input sanitization."""
		class SanitizingConnector(ConnectorBase):
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				# Simulate input sanitization
				sanitized_input = str(context.get("user_input", "")).replace("<script>", "")
				return {"sanitized_input": sanitized_input, "original_input": context.get("user_input")}
		
		config = ConnectorConfig(connector_type="sanitizing", config={})
		connector = SanitizingConnector(config)
		
		# Test malicious input
		result = await connector.execute({"user_input": "<script>alert('xss')</script>Hello"})
		
		assert result["sanitized_input"] == "alert('xss')Hello"
		assert "<script>" not in result["sanitized_input"]
	
	@pytest.mark.security
	async def test_connector_credential_handling(self):
		"""Test secure credential handling in connectors."""
		class SecureConnector(ConnectorBase):
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				# Simulate secure credential handling
				credentials = context.get("credentials", {})
				
				# Should not return actual credentials in result
				return {
					"has_credentials": bool(credentials),
					"credential_keys": list(credentials.keys()),
					"authenticated": credentials.get("token") == "valid_token"
				}
		
		config = ConnectorConfig(connector_type="secure", config={})
		connector = SecureConnector(config)
		
		result = await connector.execute({
			"credentials": {"token": "valid_token", "secret": "super_secret"}
		})
		
		assert result["has_credentials"] is True
		assert "token" in result["credential_keys"]
		assert result["authenticated"] is True
		# Ensure actual credentials are not leaked
		assert "super_secret" not in str(result)
	
	@pytest.mark.security
	async def test_connector_access_control(self):
		"""Test connector access control mechanisms."""
		class AccessControlledConnector(ConnectorBase):
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				user_role = context.get("user_role", "guest")
				required_role = self.config.config.get("required_role", "admin")
				
				if user_role != required_role:
					raise PermissionError(f"Access denied. Required role: {required_role}")
				
				return {"access_granted": True, "user_role": user_role}
		
		config = ConnectorConfig(
			connector_type="access_controlled",
			config={"required_role": "admin"}
		)
		connector = AccessControlledConnector(config)
		
		# Test authorized access
		result = await connector.execute({"user_role": "admin"})
		assert result["access_granted"] is True
		
		# Test unauthorized access
		with pytest.raises(PermissionError) as exc_info:
			await connector.execute({"user_role": "user"})
		assert "Access denied" in str(exc_info.value)
	
	@pytest.mark.security
	async def test_connector_data_encryption(self):
		"""Test connector data encryption capabilities."""
		import base64
		
		class EncryptingConnector(ConnectorBase):
			async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
				sensitive_data = context.get("sensitive_data", "")
				
				# Simple base64 encoding for demonstration (use real encryption in production)
				encrypted_data = base64.b64encode(sensitive_data.encode()).decode()
				
				return {
					"encrypted": True,
					"encrypted_data": encrypted_data,
					"data_length": len(sensitive_data)
				}
		
		config = ConnectorConfig(connector_type="encrypting", config={})
		connector = EncryptingConnector(config)
		
		result = await connector.execute({"sensitive_data": "secret information"})
		
		assert result["encrypted"] is True
		assert result["encrypted_data"] != "secret information"
		assert result["data_length"] == len("secret information")
		
		# Verify data can be decrypted
		decrypted = base64.b64decode(result["encrypted_data"]).decode()
		assert decrypted == "secret information"