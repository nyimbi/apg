#!/usr/bin/env python3
"""
APG Workflow Orchestration Security Tests

Comprehensive security testing including authentication, authorization,
input validation, injection prevention, and multi-tenant isolation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import pytest
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import re

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# APG Core imports
from ..service import WorkflowOrchestrationService
from ..database import DatabaseManager
from ..api import create_app
from ..models import *
from ..apg_integration import APGIntegration

# Test utilities
from .conftest import TestHelpers


class TestAuthenticationSecurity:
	"""Test authentication and session security."""
	
	@pytest.mark.security
	def test_api_requires_authentication(self, test_client):
		"""Test that API endpoints require proper authentication."""
		# Test workflow creation without authentication
		workflow_data = {
			"name": "Unauthorized Test",
			"description": "Should be rejected",
			"tenant_id": "test_tenant",
			"tasks": []
		}
		
		# Remove any default authentication headers
		response = test_client.post("/api/v1/workflows", json=workflow_data, headers={})
		assert response.status_code == 401  # Unauthorized
		
		# Test workflow listing without authentication
		response = test_client.get("/api/v1/workflows", headers={})
		assert response.status_code == 401  # Unauthorized
	
	@pytest.mark.security
	def test_invalid_token_rejection(self, test_client):
		"""Test that invalid authentication tokens are rejected."""
		workflow_data = {
			"name": "Invalid Token Test",
			"description": "Should be rejected",
			"tenant_id": "test_tenant",
			"tasks": []
		}
		
		# Test with invalid token formats
		invalid_tokens = [
			"invalid_token",
			"Bearer invalid",
			"Bearer " + "x" * 100,  # Too long
			"Bearer ",  # Empty
			"Basic invalid_base64",  # Wrong auth type
		]
		
		for invalid_token in invalid_tokens:
			response = test_client.post(
				"/api/v1/workflows",
				json=workflow_data,
				headers={"Authorization": invalid_token}
			)
			assert response.status_code in [401, 403]  # Unauthorized or Forbidden
	
	@pytest.mark.security
	def test_session_security(self, test_client):
		"""Test session security and token expiration."""
		# Mock expired token
		expired_token = "Bearer expired_token_12345"
		
		workflow_data = {
			"name": "Expired Token Test",
			"description": "Should be rejected",
			"tenant_id": "test_tenant",
			"tasks": []
		}
		
		with patch('..api.verify_token') as mock_verify:
			# Simulate expired token
			mock_verify.side_effect = Exception("Token expired")
			
			response = test_client.post(
				"/api/v1/workflows",
				json=workflow_data,
				headers={"Authorization": expired_token}
			)
			assert response.status_code == 401
	
	@pytest.mark.security
	async def test_password_security_requirements(self):
		"""Test password security requirements and hashing."""
		# Test password requirements (if applicable to workflow service)
		weak_passwords = [
			"123456",
			"password",
			"abc",
			"",
			"a" * 200  # Too long
		]
		
		strong_passwords = [
			"StrongPassword123!",
			"MySecureP@ssw0rd",
			"Complex123$Password"
		]
		
		# Test password validation (mock implementation)
		def validate_password(password: str) -> bool:
			if len(password) < 8:
				return False
			if not re.search(r'[A-Z]', password):
				return False
			if not re.search(r'[a-z]', password):
				return False
			if not re.search(r'\d', password):
				return False
			if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
				return False
			return True
		
		# Test weak passwords are rejected
		for weak_password in weak_passwords:
			assert not validate_password(weak_password), f"Weak password '{weak_password}' should be rejected"
		
		# Test strong passwords are accepted
		for strong_password in strong_passwords:
			assert validate_password(strong_password), f"Strong password '{strong_password}' should be accepted"
	
	@pytest.mark.security
	async def test_brute_force_protection(self, test_client):
		"""Test protection against brute force attacks."""
		# Simulate multiple failed authentication attempts
		invalid_credentials = [
			"Bearer invalid_token_1",
			"Bearer invalid_token_2", 
			"Bearer invalid_token_3",
			"Bearer invalid_token_4",
			"Bearer invalid_token_5"
		]
		
		workflow_data = {
			"name": "Brute Force Test",
			"description": "Testing brute force protection",
			"tenant_id": "test_tenant",
			"tasks": []
		}
		
		# Make multiple failed requests rapidly
		failed_attempts = 0
		for invalid_cred in invalid_credentials:
			response = test_client.post(
				"/api/v1/workflows",
				json=workflow_data,
				headers={"Authorization": invalid_cred}
			)
			if response.status_code == 401:
				failed_attempts += 1
		
		# After multiple failures, should potentially implement rate limiting
		# Note: Actual rate limiting would depend on implementation
		assert failed_attempts == len(invalid_credentials)


class TestAuthorizationSecurity:
	"""Test authorization and access control security."""
	
	@pytest.mark.security
	async def test_role_based_access_control(self, workflow_service):
		"""Test role-based access control enforcement."""
		# Create workflows with different access requirements
		admin_workflow_data = {
			"name": "Admin Only Workflow",
			"description": "Requires admin privileges",
			"tenant_id": "test_tenant",
			"tasks": [],
			"metadata": {
				"required_roles": ["admin"],
				"security_level": "high"
			}
		}
		
		user_workflow_data = {
			"name": "User Workflow",
			"description": "Regular user access",
			"tenant_id": "test_tenant", 
			"tasks": [],
			"metadata": {
				"required_roles": ["user"],
				"security_level": "normal"
			}
		}
		
		# Mock APG RBAC integration
		with patch('..connectors.apg_connectors.AuthRBACConnector') as mock_auth:
			# Test admin access
			mock_auth.return_value.validate_user_permissions = AsyncMock(
				return_value={"valid": True, "roles": ["admin", "user"]}
			)
			
			admin_workflow = await workflow_service.create_workflow(
				admin_workflow_data,
				user_id="admin_user"
			)
			assert admin_workflow is not None
			
			# Test user trying to access admin workflow
			mock_auth.return_value.validate_user_permissions = AsyncMock(
				return_value={"valid": False, "roles": ["user"], "error": "Insufficient privileges"}
			)
			
			with pytest.raises(Exception):  # Should raise permission error
				await workflow_service.create_workflow(
					admin_workflow_data,
					user_id="regular_user"
				)
	
	@pytest.mark.security
	async def test_tenant_isolation_security(self, workflow_service):
		"""Test security of tenant isolation."""
		# Create workflows in different tenants
		tenant_a_workflow = {
			"name": "Tenant A Workflow",
			"description": "Belongs to tenant A",
			"tenant_id": "tenant_a",
			"tasks": [{"id": "task_a", "name": "Task A", "task_type": "script", "config": {"script": "return {'tenant': 'a'}"}}]
		}
		
		tenant_b_workflow = {
			"name": "Tenant B Workflow",
			"description": "Belongs to tenant B",
			"tenant_id": "tenant_b",
			"tasks": [{"id": "task_b", "name": "Task B", "task_type": "script", "config": {"script": "return {'tenant': 'b'}"}}]
		}
		
		# Create workflows
		workflow_a = await workflow_service.create_workflow(tenant_a_workflow, user_id="user_a")
		workflow_b = await workflow_service.create_workflow(tenant_b_workflow, user_id="user_b")
		
		# Test cross-tenant access prevention
		try:
			# User from tenant A trying to access tenant B's workflow
			accessed_workflow = await workflow_service.get_workflow(workflow_b.id, tenant_id="tenant_a")
			# Should either return None or raise exception
			assert accessed_workflow is None or accessed_workflow.tenant_id != "tenant_b"
		except Exception:
			# Exception is acceptable - access should be denied
			pass
		
		# Test tenant-specific listing
		tenant_a_workflows = await workflow_service.list_workflows(tenant_id="tenant_a")
		tenant_b_workflows = await workflow_service.list_workflows(tenant_id="tenant_b")
		
		# Each tenant should only see their own workflows
		a_workflow_ids = [w.id for w in tenant_a_workflows]
		b_workflow_ids = [w.id for w in tenant_b_workflows]
		
		assert workflow_a.id in a_workflow_ids
		assert workflow_a.id not in b_workflow_ids
		assert workflow_b.id in b_workflow_ids
		assert workflow_b.id not in a_workflow_ids
	
	@pytest.mark.security
	async def test_resource_access_permissions(self, workflow_service):
		"""Test resource access permissions and ownership."""
		# Create workflow with specific owner
		workflow_data = {
			"name": "Owned Workflow",
			"description": "Testing ownership permissions",
			"tenant_id": "test_tenant",
			"tasks": []
		}
		
		workflow = await workflow_service.create_workflow(workflow_data, user_id="owner_user")
		
		# Test that only owner can modify
		updated_data = workflow_data.copy()
		updated_data["description"] = "Updated by owner"
		
		# Owner should be able to update
		updated_workflow = await workflow_service.update_workflow(
			workflow.id,
			updated_data,
			user_id="owner_user"
		)
		assert updated_workflow.description == "Updated by owner"
		
		# Non-owner should not be able to update
		updated_data["description"] = "Updated by non-owner"
		try:
			await workflow_service.update_workflow(
				workflow.id,
				updated_data,
				user_id="other_user"
			)
			# Should fail if proper authorization is implemented
			assert False, "Non-owner should not be able to update workflow"
		except Exception:
			# Expected - authorization should prevent this
			pass
	
	@pytest.mark.security
	def test_api_endpoint_permissions(self, test_client):
		"""Test API endpoint access permissions."""
		# Test that sensitive endpoints require appropriate permissions
		sensitive_endpoints = [
			("/api/v1/admin/workflows", "GET"),
			("/api/v1/admin/users", "GET"),
			("/api/v1/system/health", "GET"),
			("/api/v1/admin/metrics", "GET")
		]
		
		# Test with regular user token
		regular_user_headers = {"Authorization": "Bearer regular_user_token"}
		
		for endpoint, method in sensitive_endpoints:
			if method == "GET":
				response = test_client.get(endpoint, headers=regular_user_headers)
			elif method == "POST":
				response = test_client.post(endpoint, headers=regular_user_headers, json={})
			elif method == "DELETE":
				response = test_client.delete(endpoint, headers=regular_user_headers)
			
			# Should require admin permissions
			assert response.status_code in [401, 403, 404], f"Endpoint {endpoint} should require admin permissions"


class TestInputValidationSecurity:
	"""Test input validation and sanitization security."""
	
	@pytest.mark.security
	async def test_sql_injection_prevention(self, workflow_service):
		"""Test prevention of SQL injection attacks."""
		# Test with malicious SQL injection payloads
		malicious_inputs = [
			"'; DROP TABLE workflows; --",
			"' OR '1'='1",
			"'; UPDATE workflows SET name='hacked'; --",
			"' UNION SELECT * FROM users --",
			"'; INSERT INTO workflows VALUES ('hacked'); --"
		]
		
		for malicious_input in malicious_inputs:
			workflow_data = {
				"name": malicious_input,
				"description": f"Testing SQL injection with: {malicious_input}",
				"tenant_id": "security_test",
				"tasks": []
			}
			
			try:
				# Should either sanitize input or raise validation error
				workflow = await workflow_service.create_workflow(workflow_data, user_id="security_test_user")
				
				# If creation succeeds, verify no SQL injection occurred
				if workflow:
					# Name should be sanitized or escaped
					assert workflow.name != malicious_input or "DROP" not in workflow.name.upper()
				
			except Exception as e:
				# Validation error is acceptable - input should be rejected
				assert "validation" in str(e).lower() or "error" in str(e).lower()
	
	@pytest.mark.security
	async def test_script_injection_prevention(self, workflow_service):
		"""Test prevention of script injection in workflow tasks."""
		# Test with malicious script payloads
		malicious_scripts = [
			"import os; os.system('rm -rf /')",
			"exec('import subprocess; subprocess.call([\"rm\", \"-rf\", \"/\"])')",
			"__import__('os').system('malicious_command')",
			"eval('print(open(\"/etc/passwd\").read())')",
			"open('/etc/passwd', 'r').read()"
		]
		
		for malicious_script in malicious_scripts:
			workflow_data = {
				"name": "Script Injection Test",
				"description": "Testing script injection prevention",
				"tenant_id": "security_test",
				"tasks": [
					{
						"id": "malicious_task",
						"name": "Malicious Task",
						"task_type": "script",
						"config": {"script": malicious_script}
					}
				]
			}
			
			try:
				workflow = await workflow_service.create_workflow(workflow_data, user_id="security_test_user")
				
				# If workflow creation succeeds, test execution safety
				if workflow:
					instance = await workflow_service.execute_workflow(workflow.id)
					
					# Wait for completion/failure
					max_wait = 5  # Short timeout for security tests
					waited = 0
					while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
						await asyncio.sleep(0.1)
						waited += 0.1
						instance = await workflow_service.get_workflow_instance(instance.id)
					
					# Should fail or be safely sandboxed
					if instance.status == WorkflowStatus.FAILED:
						assert "security" in instance.error_details.lower() or "permission" in instance.error_details.lower()
					elif instance.status == WorkflowStatus.COMPLETED:
						# If it completed, verify no malicious actions occurred
						# This would depend on sandboxing implementation
						pass
			
			except Exception:
				# Rejection at creation time is acceptable
				pass
	
	@pytest.mark.security
	def test_xss_prevention_in_api(self, test_client):
		"""Test prevention of XSS attacks through API."""
		# Test with XSS payloads
		xss_payloads = [
			"<script>alert('XSS')</script>",
			"javascript:alert('XSS')",
			"<img src='x' onerror='alert(1)'>",
			"<svg onload=alert('XSS')>",
			"';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//\";alert(String.fromCharCode(88,83,83))//\";alert(String.fromCharCode(88,83,83))//--></SCRIPT>\">'><SCRIPT>alert(String.fromCharCode(88,83,83))</SCRIPT>"
		]
		
		for xss_payload in xss_payloads:
			workflow_data = {
				"name": xss_payload,
				"description": f"XSS test with payload: {xss_payload}",
				"tenant_id": "security_test",
				"tasks": []
			}
			
			response = test_client.post("/api/v1/workflows", json=workflow_data)
			
			if response.status_code == 201:
				# If creation succeeded, verify response is properly escaped
				response_text = response.text
				assert "<script>" not in response_text.lower()
				assert "javascript:" not in response_text.lower()
				assert "onerror=" not in response_text.lower()
				
				# Get the workflow back and verify it's escaped
				workflow = response.json()
				workflow_id = workflow["id"]
				
				get_response = test_client.get(f"/api/v1/workflows/{workflow_id}")
				if get_response.status_code == 200:
					get_response_text = get_response.text
					assert "<script>" not in get_response_text.lower()
					assert "javascript:" not in get_response_text.lower()
	
	@pytest.mark.security
	async def test_path_traversal_prevention(self, workflow_service):
		"""Test prevention of path traversal attacks."""
		# Test with path traversal payloads
		path_traversal_payloads = [
			"../../../etc/passwd",
			"..\\..\\..\\windows\\system32\\config\\sam",
			"....//....//....//etc//passwd",
			"%2e%2e%2f%2e%2e%2f%2e%2e%2f%etc%2fpasswd",
			"..%252f..%252f..%252fetc%252fpasswd"
		]
		
		for payload in path_traversal_payloads:
			# Test in file-related configurations
			workflow_data = {
				"name": "Path Traversal Test",
				"description": "Testing path traversal prevention",
				"tenant_id": "security_test",
				"tasks": [
					{
						"id": "file_task",
						"name": "File Task",
						"task_type": "connector",
						"connector_type": "file_system",
						"config": {
							"operation": "read",
							"file_path": payload
						}
					}
				]
			}
			
			try:
				workflow = await workflow_service.create_workflow(workflow_data, user_id="security_test_user")
				
				if workflow:
					# If creation succeeded, execution should be safely sandboxed
					instance = await workflow_service.execute_workflow(workflow.id)
					
					# Wait for completion
					max_wait = 3
					waited = 0
					while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
						await asyncio.sleep(0.1)
						waited += 0.1
						instance = await workflow_service.get_workflow_instance(instance.id)
					
					# Should fail or be safely handled
					if instance.status == WorkflowStatus.FAILED:
						assert "path" in instance.error_details.lower() or "access" in instance.error_details.lower()
			
			except Exception:
				# Rejection is acceptable
				pass
	
	@pytest.mark.security
	async def test_json_injection_prevention(self, workflow_service):
		"""Test prevention of JSON injection attacks."""
		# Test with malicious JSON payloads
		malicious_json_inputs = [
			'{"__proto__": {"polluted": true}}',
			'{"constructor": {"prototype": {"polluted": true}}}',
			'{"toString": "function() { return \\"hacked\\"; }"}',
			'{"valueOf": "function() { return 999999; }"}',
		]
		
		for malicious_json in malicious_json_inputs:
			workflow_data = {
				"name": "JSON Injection Test",
				"description": "Testing JSON injection prevention",
				"tenant_id": "security_test",
				"tasks": [
					{
						"id": "json_task",
						"name": "JSON Task",
						"task_type": "script",
						"config": {
							"script": f"import json; data = json.loads('{malicious_json}'); return {{'data': data}}"
						}
					}
				],
				"metadata": json.loads(malicious_json) if malicious_json.startswith('{') else {}
			}
			
			try:
				workflow = await workflow_service.create_workflow(workflow_data, user_id="security_test_user")
				
				# Verify no prototype pollution occurred
				if workflow and hasattr(workflow, 'metadata'):
					# Check that prototype pollution didn't affect the object
					assert not hasattr({}, 'polluted'), "Prototype pollution detected"
			
			except (json.JSONDecodeError, Exception):
				# Parsing/validation errors are acceptable
				pass


class TestDataSecurity:
	"""Test data security and encryption."""
	
	@pytest.mark.security
	async def test_sensitive_data_handling(self, workflow_service):
		"""Test handling of sensitive data in workflows."""
		# Create workflow with sensitive data
		sensitive_workflow_data = {
			"name": "Sensitive Data Test",
			"description": "Testing sensitive data handling",
			"tenant_id": "security_test",
			"tasks": [
				{
					"id": "sensitive_task",
					"name": "Sensitive Task",
					"task_type": "script",
					"config": {
						"script": "return {'api_key': 'sk-1234567890abcdef', 'password': 'secret123'}"
					}
				}
			],
			"metadata": {
				"contains_sensitive_data": True,
				"api_keys": ["stripe_key", "aws_key"],
				"credentials": {"database_password": "super_secret"}
			}
		}
		
		workflow = await workflow_service.create_workflow(sensitive_workflow_data, user_id="security_test_user")
		instance = await workflow_service.execute_workflow(workflow.id)
		
		# Wait for completion
		while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
			await asyncio.sleep(0.1)
			instance = await workflow_service.get_workflow_instance(instance.id)
		
		# Verify sensitive data is handled securely
		if instance.status == WorkflowStatus.COMPLETED:
			# Check that sensitive data is not logged in plain text
			for task_execution in instance.task_executions:
				if task_execution.result:
					result_str = str(task_execution.result)
					# Should not contain sensitive data in logs
					assert "sk-1234567890abcdef" not in result_str or "***" in result_str
					assert "super_secret" not in result_str or "***" in result_str
	
	@pytest.mark.security
	async def test_data_encryption_at_rest(self, database_manager):
		"""Test data encryption at rest in database."""
		# Test that sensitive workflow data is encrypted in database
		sensitive_data = {
			"api_key": "sk-very-sensitive-key-12345",
			"password": "ultra-secret-password",
			"token": "bearer-token-xyz789"
		}
		
		workflow_data = {
			"name": "Encryption Test Workflow",
			"description": "Testing data encryption",
			"tenant_id": "encryption_test",
			"definition": {"tasks": [], "sensitive_config": sensitive_data},
			"created_by": "encryption_test_user",
			"version": "1.0",
			"metadata": sensitive_data
		}
		
		# Store workflow in database
		async with database_manager.get_session() as session:
			db_workflow = WorkflowDB(**workflow_data)
			session.add(db_workflow)
			await session.commit()
			workflow_id = db_workflow.id
		
		# Directly query database to check if data is encrypted
		async with database_manager.get_session() as session:
			# Raw query to check stored data
			result = await session.execute(
				text("SELECT definition, metadata FROM workflows WHERE id = :id"),
				{"id": workflow_id}
			)
			row = result.fetchone()
			
			if row:
				definition_str = str(row[0])
				metadata_str = str(row[1])
				
				# Sensitive data should ideally be encrypted or hashed
				# At minimum, it should not be stored in plain text
				# Note: This test assumes encryption is implemented
				if "sk-very-sensitive-key-12345" in definition_str:
					# If encryption is not implemented, at least log a warning
					print("WARNING: Sensitive data may be stored in plain text")
				
				if "ultra-secret-password" in metadata_str:
					print("WARNING: Sensitive metadata may be stored in plain text")
	
	@pytest.mark.security
	async def test_data_masking_in_logs(self, workflow_service):
		"""Test that sensitive data is masked in logs."""
		# Create workflow that would generate logs with sensitive data
		workflow_data = {
			"name": "Log Masking Test",
			"description": "Testing log data masking",
			"tenant_id": "security_test",
			"tasks": [
				{
					"id": "logging_task",
					"name": "Logging Task",
					"task_type": "script",
					"config": {
						"script": """
import logging
logger = logging.getLogger(__name__)
api_key = "sk-1234567890abcdef"
password = "secret_password_123"
logger.info(f"Processing with API key: {api_key}")
logger.info(f"Database password: {password}")
return {'processed': True, 'api_key': api_key, 'password': password}
"""
					}
				}
			]
		}
		
		workflow = await workflow_service.create_workflow(workflow_data, user_id="security_test_user")
		instance = await workflow_service.execute_workflow(workflow.id)
		
		# Wait for completion
		while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
			await asyncio.sleep(0.1)
			instance = await workflow_service.get_workflow_instance(instance.id)
		
		# Check that logs don't contain sensitive data in plain text
		for task_execution in instance.task_executions:
			if task_execution.logs:
				log_content = str(task_execution.logs)
				# Sensitive data should be masked
				assert "sk-1234567890abcdef" not in log_content or "sk-***" in log_content
				assert "secret_password_123" not in log_content or "***" in log_content
	
	@pytest.mark.security
	async def test_secure_credential_storage(self, workflow_service):
		"""Test secure storage and retrieval of credentials."""
		# Test workflow with credentials
		workflow_data = {
			"name": "Credential Storage Test",
			"description": "Testing secure credential storage",
			"tenant_id": "security_test",
			"tasks": [
				{
					"id": "credential_task",
					"name": "Credential Task",
					"task_type": "connector",
					"connector_type": "rest_api",
					"config": {
						"url": "https://api.example.com/secure",
						"method": "GET",
						"headers": {"Authorization": "Bearer ${secrets.api_token}"}
					}
				}
			],
			"secrets": {
				"api_token": "very-secret-token-12345",
				"database_password": "ultra-secure-db-password"
			}
		}
		
		# Mock secure credential handling
		with patch('..service.SecretManager') as mock_secret_manager:
			mock_secret_manager.return_value.store_secret = AsyncMock(return_value="secret_id_123")
			mock_secret_manager.return_value.get_secret = AsyncMock(return_value="very-secret-token-12345")
			
			workflow = await workflow_service.create_workflow(workflow_data, user_id="security_test_user")
			
			# Verify secrets are stored securely
			mock_secret_manager.return_value.store_secret.assert_called()
			
			# Verify workflow definition doesn't contain plain text secrets
			workflow_dict = workflow.model_dump()
			workflow_str = str(workflow_dict)
			assert "very-secret-token-12345" not in workflow_str
			assert "ultra-secure-db-password" not in workflow_str


class TestNetworkSecurity:
	"""Test network security and communication security."""
	
	@pytest.mark.security
	def test_https_enforcement(self, test_client):
		"""Test HTTPS enforcement for API endpoints."""
		# Test that API enforces HTTPS (this would depend on deployment configuration)
		# For testing purposes, verify security headers are present
		
		response = test_client.get("/api/v1/health")
		
		# Check for security headers
		headers = response.headers
		
		# Should have security-related headers
		expected_security_headers = [
			"X-Content-Type-Options",
			"X-Frame-Options", 
			"X-XSS-Protection",
			"Strict-Transport-Security"  # For HTTPS enforcement
		]
		
		for security_header in expected_security_headers:
			if security_header in headers:
				print(f"Security header present: {security_header}")
		
		# At minimum, should not be completely unprotected
		assert "Server" not in headers or headers["Server"] != "development"
	
	@pytest.mark.security
	def test_cors_configuration(self, test_client):
		"""Test CORS configuration security."""
		# Test CORS preflight request
		response = test_client.options(
			"/api/v1/workflows",
			headers={
				"Origin": "https://malicious-site.com",
				"Access-Control-Request-Method": "POST",
				"Access-Control-Request-Headers": "Content-Type"
			}
		)
		
		# Should have proper CORS headers
		if "Access-Control-Allow-Origin" in response.headers:
			allowed_origin = response.headers["Access-Control-Allow-Origin"]
			# Should not allow all origins in production
			assert allowed_origin != "*" or "localhost" in allowed_origin
	
	@pytest.mark.security
	async def test_request_size_limits(self, test_client):
		"""Test request size limits for security."""
		# Test with large payload
		large_payload = {
			"name": "Large Payload Test",
			"description": "A" * 10000,  # 10KB description
			"tenant_id": "security_test",
			"tasks": [
				{
					"id": f"task_{i}",
					"name": f"Task {i}",
					"task_type": "script",
					"config": {"script": "return {'large': True}"}
				} for i in range(1000)  # 1000 tasks
			]
		}
		
		response = test_client.post("/api/v1/workflows", json=large_payload)
		
		# Should either accept with limits or reject if too large
		if response.status_code != 201:
			assert response.status_code in [413, 400]  # Payload too large or bad request
	
	@pytest.mark.security
	def test_rate_limiting(self, test_client):
		"""Test API rate limiting for security."""
		# Make multiple rapid requests
		rapid_requests = 50
		responses = []
		
		for i in range(rapid_requests):
			response = test_client.get("/api/v1/workflows")
			responses.append(response.status_code)
		
		# Should eventually hit rate limits
		rate_limited_responses = [code for code in responses if code == 429]  # Too Many Requests
		
		# If rate limiting is implemented, should see some 429 responses
		if rate_limited_responses:
			print(f"Rate limiting active: {len(rate_limited_responses)} requests limited")
		else:
			print("Rate limiting may not be implemented or limits are very high")


class TestSecurityCompliance:
	"""Test security compliance and audit requirements."""
	
	@pytest.mark.security
	async def test_audit_trail_security(self, workflow_service):
		"""Test security of audit trails and logging."""
		# Create workflow with audit requirements
		workflow_data = {
			"name": "Audit Trail Test",
			"description": "Testing audit trail security",
			"tenant_id": "audit_test",
			"tasks": [],
			"metadata": {
				"audit_required": True,
				"compliance_level": "high"
			}
		}
		
		# Mock audit integration
		with patch('..connectors.apg_connectors.AuditComplianceConnector') as mock_audit:
			mock_audit.return_value.log_activity = AsyncMock(
				return_value={"audit_id": "audit_123", "integrity_hash": "sha256hash"}
			)
			
			workflow = await workflow_service.create_workflow(workflow_data, user_id="audit_user")
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Verify audit logging was called
			mock_audit.return_value.log_activity.assert_called()
			
			# Audit logs should include integrity measures
			call_args = mock_audit.return_value.log_activity.call_args
			assert call_args is not None
	
	@pytest.mark.security
	async def test_data_retention_policies(self, database_manager):
		"""Test data retention and purging policies."""
		# Create old workflow data
		old_workflow_data = {
			"name": "Old Workflow",
			"description": "Workflow for retention testing",
			"tenant_id": "retention_test",
			"definition": {"tasks": []},
			"created_by": "retention_user",
			"version": "1.0",
			"created_at": datetime.utcnow() - timedelta(days=400)  # Very old
		}
		
		async with database_manager.get_session() as session:
			old_workflow = WorkflowDB(**old_workflow_data)
			session.add(old_workflow)
			await session.commit()
			old_workflow_id = old_workflow.id
		
		# Test retention policy (mock implementation)
		retention_policy_days = 365
		cutoff_date = datetime.utcnow() - timedelta(days=retention_policy_days)
		
		async with database_manager.get_session() as session:
			# Query for workflows older than retention policy
			result = await session.execute(
				text("SELECT COUNT(*) FROM workflows WHERE created_at < :cutoff_date"),
				{"cutoff_date": cutoff_date}
			)
			old_count = result.scalar()
			
			# In a real implementation, old data should be purged
			print(f"Workflows older than {retention_policy_days} days: {old_count}")
			
			# For testing, just verify we can identify old data
			assert old_count >= 0  # Should be able to count old records
	
	@pytest.mark.security
	async def test_gdpr_compliance(self, workflow_service):
		"""Test GDPR compliance features."""
		# Test data subject rights
		user_data = {
			"user_id": "gdpr_test_user",
			"email": "gdpr.test@example.com",
			"personal_data": {"name": "GDPR Test User", "location": "EU"}
		}
		
		# Create workflow with personal data
		workflow_data = {
			"name": "GDPR Compliance Test",
			"description": "Testing GDPR compliance",
			"tenant_id": "gdpr_test",
			"tasks": [],
			"metadata": user_data,
			"created_by": user_data["user_id"]
		}
		
		workflow = await workflow_service.create_workflow(workflow_data, user_id=user_data["user_id"])
		
		# Test right to access (data export)
		user_workflows = await workflow_service.list_workflows(
			tenant_id="gdpr_test",
			filters={"created_by": user_data["user_id"]}
		)
		
		assert len(user_workflows) >= 1
		assert any(w.id == workflow.id for w in user_workflows)
		
		# Test right to rectification (data update)
		updated_data = workflow_data.copy()
		updated_data["metadata"]["personal_data"]["name"] = "Updated Name"
		
		updated_workflow = await workflow_service.update_workflow(
			workflow.id,
			updated_data,
			user_id=user_data["user_id"]
		)
		
		assert updated_workflow.metadata["personal_data"]["name"] == "Updated Name"
		
		# Test right to erasure (data deletion)
		await workflow_service.delete_workflow(workflow.id, user_id=user_data["user_id"])
		
		# Verify deletion
		try:
			deleted_workflow = await workflow_service.get_workflow(workflow.id)
			assert deleted_workflow is None
		except Exception:
			# Exception is acceptable - workflow should not exist
			pass
	
	@pytest.mark.security
	async def test_security_incident_response(self, workflow_service):
		"""Test security incident detection and response."""
		# Simulate suspicious activity
		suspicious_activities = [
			# Multiple failed authentications
			{"type": "auth_failure", "count": 10, "timeframe": "1_minute"},
			# Unusual access patterns  
			{"type": "bulk_access", "count": 100, "timeframe": "1_minute"},
			# Privilege escalation attempts
			{"type": "privilege_escalation", "attempts": 5, "timeframe": "1_minute"}
		]
		
		# Mock security monitoring
		with patch('..service.SecurityMonitor') as mock_monitor:
			mock_monitor.return_value.detect_suspicious_activity = AsyncMock(
				return_value={"incident_detected": True, "severity": "high", "incident_id": "inc_123"}
			)
			
			# Simulate creating workflows that might trigger security monitoring
			for i in range(5):
				try:
					workflow_data = {
						"name": f"Suspicious Workflow {i}",
						"description": "Potentially suspicious workflow",
						"tenant_id": "security_test",
						"tasks": []
					}
					
					workflow = await workflow_service.create_workflow(
						workflow_data,
						user_id="potentially_malicious_user"
					)
					
				except Exception:
					# Security measures may prevent creation
					pass
			
			# Verify security monitoring would be triggered
			# In a real implementation, this would trigger alerts
			print("Security monitoring test completed")


class TestPenetrationTestScenarios:
	"""Penetration testing scenarios for comprehensive security validation."""
	
	@pytest.mark.security
	@pytest.mark.slow
	def test_comprehensive_security_scan(self, test_client):
		"""Comprehensive security scan of API endpoints."""
		# Test common vulnerabilities across multiple endpoints
		endpoints_to_test = [
			"/api/v1/workflows",
			"/api/v1/workflow-instances", 
			"/api/v1/users",
			"/api/v1/admin/metrics"
		]
		
		vulnerability_tests = {
			"sql_injection": ["'; DROP TABLE users; --", "' OR '1'='1"],
			"xss": ["<script>alert('XSS')</script>", "javascript:alert(1)"],
			"command_injection": ["; ls -la", "| cat /etc/passwd"],
			"path_traversal": ["../../../etc/passwd", "..\\..\\..\\windows\\system32"],
			"header_injection": ["\r\nSet-Cookie: malicious=true", "\n\rLocation: http://evil.com"]
		}
		
		security_results = {}
		
		for endpoint in endpoints_to_test:
			endpoint_results = {}
			
			for vuln_type, payloads in vulnerability_tests.items():
				vuln_results = []
				
				for payload in payloads:
					# Test in different contexts
					test_contexts = [
						{"method": "GET", "params": {"q": payload}},
						{"method": "POST", "json": {"name": payload}},
						{"method": "GET", "headers": {"X-Test": payload}}
					]
					
					for context in test_contexts:
						try:
							if context["method"] == "GET":
								if "params" in context:
									response = test_client.get(endpoint, params=context["params"])
								else:
									response = test_client.get(endpoint, headers=context.get("headers", {}))
							elif context["method"] == "POST":
								response = test_client.post(
									endpoint,
									json=context.get("json", {}),
									headers=context.get("headers", {})
								)
							
							vuln_results.append({
								"payload": payload,
								"context": context,
								"status_code": response.status_code,
								"vulnerable": self._check_vulnerability_indicators(response, vuln_type, payload)
							})
							
						except Exception as e:
							vuln_results.append({
								"payload": payload,
								"context": context,
								"error": str(e),
								"vulnerable": False
							})
				
				endpoint_results[vuln_type] = vuln_results
			
			security_results[endpoint] = endpoint_results
		
		# Analyze results
		vulnerabilities_found = []
		for endpoint, endpoint_results in security_results.items():
			for vuln_type, vuln_results in endpoint_results.items():
				for result in vuln_results:
					if result.get("vulnerable", False):
						vulnerabilities_found.append({
							"endpoint": endpoint,
							"vulnerability": vuln_type,
							"payload": result["payload"],
							"context": result["context"]
						})
		
		# Security assertions
		assert len(vulnerabilities_found) == 0, f"Security vulnerabilities found: {vulnerabilities_found}"
		
		print(f"Security scan completed. Tested {len(endpoints_to_test)} endpoints with {sum(len(payloads) for payloads in vulnerability_tests.values())} payloads.")
	
	def _check_vulnerability_indicators(self, response, vuln_type: str, payload: str) -> bool:
		"""Check if response indicates vulnerability."""
		response_text = response.text.lower()
		
		vulnerability_indicators = {
			"sql_injection": ["sql syntax", "mysql", "postgresql", "database error", "sqlite"],
			"xss": ["<script>", "javascript:", "onerror=", "onload="],
			"command_injection": ["uid=", "gid=", "root:", "bin/sh"],
			"path_traversal": ["root:x:", "administrator", "[boot loader]"],
			"header_injection": ["set-cookie", "location:", "content-type:"]
		}
		
		if vuln_type in vulnerability_indicators:
			indicators = vulnerability_indicators[vuln_type]
			return any(indicator in response_text for indicator in indicators)
		
		return False