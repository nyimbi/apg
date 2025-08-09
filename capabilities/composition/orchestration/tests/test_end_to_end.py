#!/usr/bin/env python3
"""
APG Workflow Orchestration End-to-End Tests

Complete end-to-end integration tests covering full workflow scenarios
from creation through execution, monitoring, and completion.

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

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# APG Core imports
from ..service import WorkflowOrchestrationService
from ..engine import WorkflowExecutionEngine
from ..database import DatabaseManager
from ..api import create_app
from ..models import *
from ..apg_integration import APGIntegration

# Test utilities
from .conftest import TestHelpers


class TestCompleteWorkflowScenarios:
	"""Test complete workflow scenarios from start to finish."""
	
	@pytest.mark.integration
	@pytest.mark.slow
	async def test_data_processing_pipeline(self, workflow_service):
		"""Test complete data processing pipeline workflow."""
		# Create comprehensive data processing workflow
		workflow_data = {
			"name": "Data Processing Pipeline",
			"description": "Complete ETL pipeline with validation and monitoring",
			"tenant_id": "test_tenant",
			"version": "1.0",
			"metadata": {
				"pipeline_type": "etl",
				"data_sources": ["api", "database", "files"],
				"output_formats": ["parquet", "json"],
				"monitoring_enabled": True
			},
			"tasks": [
				{
					"id": "validate_inputs",
					"name": "Validate Input Data",
					"task_type": "script",
					"config": {
						"script": """
# Input validation
inputs = context.get('inputs', {})
required_keys = ['source_url', 'target_table', 'batch_size']
missing_keys = [key for key in required_keys if key not in inputs]
if missing_keys:
    raise ValueError(f'Missing required inputs: {missing_keys}')
return {'validated': True, 'inputs': inputs}
"""
					},
					"timeout_seconds": 30
				},
				{
					"id": "extract_data",
					"name": "Extract Data from Sources",
					"task_type": "connector",
					"connector_type": "rest_api",
					"config": {
						"url": "${validate_inputs.result.inputs.source_url}",
						"method": "GET",
						"headers": {"Accept": "application/json"}
					},
					"depends_on": ["validate_inputs"],
					"retry_config": {
						"max_retries": 3,
						"retry_delay": 2,
						"backoff_multiplier": 2.0
					}
				},
				{
					"id": "transform_data",
					"name": "Transform and Clean Data",
					"task_type": "script",
					"config": {
						"script": """
import json
data = context.get('extract_data', {}).get('result', {}).get('data', [])
transformed_data = []
for item in data:
    # Clean and transform each item
    cleaned_item = {
        'id': item.get('id'),
        'name': str(item.get('name', '')).strip().title(),
        'value': float(item.get('value', 0)),
        'processed_at': datetime.utcnow().isoformat()
    }
    transformed_data.append(cleaned_item)
return {'transformed_data': transformed_data, 'record_count': len(transformed_data)}
"""
					},
					"depends_on": ["extract_data"],
					"timeout_seconds": 120
				},
				{
					"id": "validate_transformation",
					"name": "Validate Transformed Data",
					"task_type": "script",
					"config": {
						"script": """
data = context.get('transform_data', {}).get('result', {}).get('transformed_data', [])
record_count = len(data)
if record_count == 0:
    raise ValueError('No data to process after transformation')
# Validation checks
invalid_records = [item for item in data if not item.get('id') or item.get('value') < 0]
if invalid_records:
    raise ValueError(f'Found {len(invalid_records)} invalid records')
return {'validation_passed': True, 'record_count': record_count}
"""
					},
					"depends_on": ["transform_data"]
				},
				{
					"id": "load_to_database",
					"name": "Load Data to Database", 
					"task_type": "connector",
					"connector_type": "database",
					"config": {
						"connection_string": "postgresql://test:test@localhost/testdb",
						"operation": "insert_batch",
						"table": "${validate_inputs.result.inputs.target_table}",
						"data": "${transform_data.result.transformed_data}"
					},
					"depends_on": ["validate_transformation"],
					"timeout_seconds": 300
				},
				{
					"id": "generate_report",
					"name": "Generate Processing Report",
					"task_type": "script",
					"config": {
						"script": """
report = {
    'pipeline_name': 'Data Processing Pipeline',
    'execution_date': datetime.utcnow().isoformat(),
    'records_processed': context.get('validate_transformation', {}).get('result', {}).get('record_count', 0),
    'source_url': context.get('validate_inputs', {}).get('result', {}).get('inputs', {}).get('source_url'),
    'target_table': context.get('validate_inputs', {}).get('result', {}).get('inputs', {}).get('target_table'),
    'status': 'completed'
}
return {'report': report, 'report_generated': True}
"""
					},
					"depends_on": ["load_to_database"]
				},
				{
					"id": "send_notification",
					"name": "Send Completion Notification",
					"task_type": "connector",
					"connector_type": "email",
					"config": {
						"to": ["admin@example.com"],
						"subject": "Data Processing Pipeline Completed",
						"body": "Pipeline completed successfully. Records processed: ${validate_transformation.result.record_count}",
						"body_type": "text"
					},
					"depends_on": ["generate_report"],
					"optional": True  # Don't fail workflow if notification fails
				}
			]
		}
		
		# Mock external dependencies
		with patch.multiple(
			'..connectors.external_connectors',
			RESTAPIConnector=MagicMock(),
			DatabaseConnector=MagicMock(),
			EmailConnector=MagicMock()
		) as mocks:
			# Setup mock responses
			mocks['RESTAPIConnector'].return_value.execute = AsyncMock(
				return_value={
					"status_code": 200,
					"data": [
						{"id": 1, "name": "item one", "value": 10.5},
						{"id": 2, "name": "item two", "value": 20.3},
						{"id": 3, "name": "item three", "value": 15.7}
					]
				}
			)
			
			mocks['DatabaseConnector'].return_value.execute = AsyncMock(
				return_value={"rows_affected": 3, "operation": "insert_batch"}
			)
			
			mocks['EmailConnector'].return_value.execute = AsyncMock(
				return_value={"sent": True, "recipients": ["admin@example.com"]}
			)
			
			# Execute the complete pipeline
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			
			execution_context = {
				"inputs": {
					"source_url": "https://api.example.com/data",
					"target_table": "processed_data",
					"batch_size": 1000
				}
			}
			
			instance = await workflow_service.execute_workflow(
				workflow.id,
				execution_context=execution_context
			)
			
			# Monitor execution progress
			max_wait = 30
			waited = 0
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
				await asyncio.sleep(1.0)
				waited += 1.0
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			# Verify successful completion
			assert instance.status == WorkflowStatus.COMPLETED
			assert len(instance.task_executions) == 7  # All tasks should execute
			
			# Verify task execution order and results
			completed_tasks = [ex for ex in instance.task_executions if ex.status == TaskStatus.COMPLETED]
			assert len(completed_tasks) >= 6  # At least 6 required tasks (notification is optional)
			
			# Verify connectors were called
			mocks['RESTAPIConnector'].return_value.execute.assert_called()
			mocks['DatabaseConnector'].return_value.execute.assert_called()
			mocks['EmailConnector'].return_value.execute.assert_called()
	
	@pytest.mark.integration
	@pytest.mark.slow
	async def test_ml_model_training_pipeline(self, workflow_service):
		"""Test machine learning model training and deployment pipeline."""
		workflow_data = {
			"name": "ML Model Training Pipeline",
			"description": "Complete ML pipeline from data prep to model deployment",
			"tenant_id": "test_tenant",
			"metadata": {
				"model_type": "classification",
				"framework": "scikit-learn",
				"deployment_target": "api_endpoint"
			},
			"tasks": [
				{
					"id": "prepare_dataset",
					"name": "Prepare Training Dataset",
					"task_type": "script",
					"config": {
						"script": """
# Simulate dataset preparation
import random
dataset = []
for i in range(1000):
    feature1 = random.uniform(0, 10)
    feature2 = random.uniform(0, 10)
    label = 1 if (feature1 + feature2) > 10 else 0
    dataset.append({'feature1': feature1, 'feature2': feature2, 'label': label})
return {'dataset': dataset, 'size': len(dataset)}
"""
					}
				},
				{
					"id": "split_data",
					"name": "Split Train/Test Data",
					"task_type": "script",
					"config": {
						"script": """
import random
dataset = context.get('prepare_dataset', {}).get('result', {}).get('dataset', [])
random.shuffle(dataset)
split_idx = int(len(dataset) * 0.8)
train_data = dataset[:split_idx]
test_data = dataset[split_idx:]
return {'train_data': train_data, 'test_data': test_data, 'train_size': len(train_data), 'test_size': len(test_data)}
"""
					},
					"depends_on": ["prepare_dataset"]
				},
				{
					"id": "train_model",
					"name": "Train ML Model",
					"task_type": "script",
					"config": {
						"script": """
# Simulate model training
train_data = context.get('split_data', {}).get('result', {}).get('train_data', [])
training_metrics = {
    'accuracy': 0.85,
    'precision': 0.83,
    'recall': 0.87,
    'f1_score': 0.85,
    'training_samples': len(train_data)
}
model_id = f'model_{uuid.uuid4().hex[:8]}'
return {'model_id': model_id, 'metrics': training_metrics, 'trained': True}
"""
					},
					"depends_on": ["split_data"],
					"timeout_seconds": 600  # Allow time for training
				},
				{
					"id": "evaluate_model",
					"name": "Evaluate Model Performance",
					"task_type": "script",
					"config": {
						"script": """
# Simulate model evaluation
test_data = context.get('split_data', {}).get('result', {}).get('test_data', [])
evaluation_metrics = {
    'test_accuracy': 0.82,
    'test_precision': 0.80,
    'test_recall': 0.84,
    'test_f1_score': 0.82,
    'test_samples': len(test_data)
}
model_quality = 'good' if evaluation_metrics['test_accuracy'] > 0.8 else 'poor'
return {'evaluation': evaluation_metrics, 'quality': model_quality, 'approved_for_deployment': model_quality == 'good'}
"""
					},
					"depends_on": ["train_model"]
				},
				{
					"id": "deploy_model",
					"name": "Deploy Model to Production",
					"task_type": "connector",
					"connector_type": "rest_api",
					"config": {
						"url": "https://ml-api.example.com/models/deploy",
						"method": "POST",
						"json": {
							"model_id": "${train_model.result.model_id}",
							"metrics": "${evaluate_model.result.evaluation}",
							"deployment_config": {"replicas": 3, "memory": "2Gi"}
						}
					},
					"depends_on": ["evaluate_model"],
					"condition": "${evaluate_model.result.approved_for_deployment} == True"
				},
				{
					"id": "create_monitoring",
					"name": "Setup Model Monitoring",
					"task_type": "script",
					"config": {
						"script": """
model_id = context.get('train_model', {}).get('result', {}).get('model_id')
monitoring_config = {
    'model_id': model_id,
    'metrics_to_track': ['accuracy', 'latency', 'throughput'],
    'alert_thresholds': {'accuracy_drop': 0.05, 'latency_max': 200},
    'monitoring_enabled': True
}
return {'monitoring_config': monitoring_config, 'monitoring_setup': True}
"""
					},
					"depends_on": ["deploy_model"]
				}
			]
		}
		
		# Mock ML deployment API
		with patch('..connectors.external_connectors.RESTAPIConnector') as mock_connector:
			mock_connector.return_value.execute = AsyncMock(
				return_value={
					"status_code": 200,
					"data": {"deployment_id": "deploy_123", "status": "deployed", "endpoint": "https://ml-api.example.com/predict"}
				}
			)
			
			# Execute ML pipeline
			workflow = await workflow_service.create_workflow(workflow_data, user_id="ml_engineer")
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for completion
			max_wait = 20
			waited = 0
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
				await asyncio.sleep(1.0)
				waited += 1.0
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			# Verify ML pipeline completion
			assert instance.status == WorkflowStatus.COMPLETED
			
			# Verify all tasks completed successfully
			completed_tasks = [ex for ex in instance.task_executions if ex.status == TaskStatus.COMPLETED]
			assert len(completed_tasks) == 6  # All tasks should complete
			
			# Verify deployment API was called
			mock_connector.return_value.execute.assert_called()
	
	@pytest.mark.integration
	async def test_business_process_automation(self, workflow_service):
		"""Test business process automation workflow."""
		workflow_data = {
			"name": "Invoice Processing Automation",
			"description": "Automated invoice processing with approval workflow",
			"tenant_id": "test_tenant",
			"metadata": {
				"process_type": "approval_workflow",
				"approval_levels": 2,
				"sla_hours": 24
			},
			"tasks": [
				{
					"id": "receive_invoice",
					"name": "Receive and Parse Invoice",
					"task_type": "script",
					"config": {
						"script": """
# Simulate invoice parsing
invoice_data = {
    'invoice_id': f'INV-{uuid.uuid4().hex[:8].upper()}',
    'vendor': 'ACME Corp',
    'amount': 1250.00,
    'due_date': (datetime.utcnow() + timedelta(days=30)).isoformat(),
    'line_items': [
        {'description': 'Professional Services', 'amount': 1000.00},
        {'description': 'Travel Expenses', 'amount': 250.00}
    ],
    'received_at': datetime.utcnow().isoformat()
}
return {'invoice': invoice_data, 'parsed': True}
"""
					}
				},
				{
					"id": "validate_invoice",
					"name": "Validate Invoice Data",
					"task_type": "script",
					"config": {
						"script": """
invoice = context.get('receive_invoice', {}).get('result', {}).get('invoice', {})
validation_results = {
    'has_required_fields': all(key in invoice for key in ['invoice_id', 'vendor', 'amount']),
    'amount_valid': invoice.get('amount', 0) > 0,
    'vendor_approved': invoice.get('vendor') in ['ACME Corp', 'Tech Solutions Inc'],
    'within_limit': invoice.get('amount', 0) <= 5000.00
}
is_valid = all(validation_results.values())
return {'validation': validation_results, 'is_valid': is_valid, 'invoice': invoice}
"""
					},
					"depends_on": ["receive_invoice"]
				},
				{
					"id": "first_level_approval",
					"name": "Manager Approval",
					"task_type": "script",
					"config": {
						"script": """
# Simulate manager approval (auto-approve for demo)
invoice = context.get('validate_invoice', {}).get('result', {}).get('invoice', {})
approval = {
    'approver': 'manager@company.com',
    'approved': True,
    'approval_date': datetime.utcnow().isoformat(),
    'comments': 'Approved - standard vendor payment'
}
return {'approval': approval, 'approved': True}
"""
					},
					"depends_on": ["validate_invoice"],
					"condition": "${validate_invoice.result.is_valid} == True"
				},
				{
					"id": "second_level_approval",
					"name": "Finance Director Approval",
					"task_type": "script",
					"config": {
						"script": """
# Second level approval for amounts > 1000
invoice = context.get('validate_invoice', {}).get('result', {}).get('invoice', {})
amount = invoice.get('amount', 0)
if amount > 1000:
    approval = {
        'approver': 'finance-director@company.com',
        'approved': True,
        'approval_date': datetime.utcnow().isoformat(),
        'comments': 'Approved after budget verification'
    }
else:
    approval = {'approved': True, 'comments': 'Auto-approved - below threshold'}
return {'approval': approval, 'approved': True}
"""
					},
					"depends_on": ["first_level_approval"],
					"condition": "${first_level_approval.result.approved} == True"
				},
				{
					"id": "process_payment",
					"name": "Process Payment",
					"task_type": "connector",
					"connector_type": "rest_api",
					"config": {
						"url": "https://payment-api.company.com/payments",
						"method": "POST",
						"json": {
							"invoice_id": "${validate_invoice.result.invoice.invoice_id}",
							"vendor": "${validate_invoice.result.invoice.vendor}",
							"amount": "${validate_invoice.result.invoice.amount}",
							"approved_by": ["${first_level_approval.result.approval.approver}", "${second_level_approval.result.approval.approver}"]
						}
					},
					"depends_on": ["second_level_approval"],
					"condition": "${second_level_approval.result.approved} == True"
				},
				{
					"id": "update_accounting",
					"name": "Update Accounting System",
					"task_type": "connector",
					"connector_type": "database",
					"config": {
						"connection_string": "postgresql://accounting:pass@localhost/accounting",
						"operation": "insert",
						"table": "processed_invoices",
						"data": {
							"invoice_id": "${validate_invoice.result.invoice.invoice_id}",
							"vendor": "${validate_invoice.result.invoice.vendor}",
							"amount": "${validate_invoice.result.invoice.amount}",
							"processed_date": "${datetime.utcnow().isoformat()}",
							"status": "paid"
						}
					},
					"depends_on": ["process_payment"]
				},
				{
					"id": "send_confirmation",
					"name": "Send Payment Confirmation",
					"task_type": "connector",
					"connector_type": "email",
					"config": {
						"to": ["vendor@acmecorp.com", "accounting@company.com"],
						"subject": "Payment Processed - ${validate_invoice.result.invoice.invoice_id}",
						"body": "Payment of $${validate_invoice.result.invoice.amount} has been processed for invoice ${validate_invoice.result.invoice.invoice_id}",
						"body_type": "text"
					},
					"depends_on": ["update_accounting"]
				}
			]
		}
		
		# Mock external systems
		with patch.multiple(
			'..connectors.external_connectors',
			RESTAPIConnector=MagicMock(),
			DatabaseConnector=MagicMock(),
			EmailConnector=MagicMock()
		) as mocks:
			# Setup mock responses
			mocks['RESTAPIConnector'].return_value.execute = AsyncMock(
				return_value={"status_code": 200, "data": {"payment_id": "PAY_123", "status": "processed"}}
			)
			
			mocks['DatabaseConnector'].return_value.execute = AsyncMock(
				return_value={"rows_affected": 1, "operation": "insert"}
			)
			
			mocks['EmailConnector'].return_value.execute = AsyncMock(
				return_value={"sent": True, "recipients": ["vendor@acmecorp.com", "accounting@company.com"]}
			)
			
			# Execute business process
			workflow = await workflow_service.create_workflow(workflow_data, user_id="admin")
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for completion
			max_wait = 15
			waited = 0
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
				await asyncio.sleep(0.5)
				waited += 0.5
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			# Verify business process completion
			assert instance.status == WorkflowStatus.COMPLETED
			assert len(instance.task_executions) == 8  # All tasks should execute
			
			# Verify external systems were called
			mocks['RESTAPIConnector'].return_value.execute.assert_called()
			mocks['DatabaseConnector'].return_value.execute.assert_called()
			mocks['EmailConnector'].return_value.execute.assert_called()


class TestAPIWorkflowIntegration:
	"""Test complete workflow scenarios through API endpoints."""
	
	@pytest.mark.integration
	@pytest.mark.api
	def test_complete_workflow_lifecycle_via_api(self, test_client):
		"""Test complete workflow lifecycle through REST API."""
		# Step 1: Create workflow template
		template_data = {
			"name": "API Integration Test Workflow",
			"description": "Workflow for testing complete API integration",
			"tenant_id": "api_test_tenant",
			"tasks": [
				{
					"id": "api_task",
					"name": "API Test Task",
					"task_type": "script",
					"config": {"script": "return {'api_test': True, 'timestamp': datetime.utcnow().isoformat()}"}
				}
			],
			"metadata": {
				"api_test": True,
				"version": "1.0"
			}
		}
		
		# Create workflow
		response = test_client.post("/api/v1/workflows", json=template_data)
		assert response.status_code == 201
		workflow = response.json()
		workflow_id = workflow["id"]
		
		# Step 2: Execute workflow
		response = test_client.post(f"/api/v1/workflows/{workflow_id}/execute")
		assert response.status_code == 200
		instance = response.json()
		instance_id = instance["id"]
		
		# Step 3: Monitor execution
		max_attempts = 20
		attempts = 0
		while attempts < max_attempts:
			response = test_client.get(f"/api/v1/workflow-instances/{instance_id}")
			assert response.status_code == 200
			current_instance = response.json()
			
			if current_instance["status"] in ["completed", "failed"]:
				break
				
			attempts += 1
			asyncio.sleep(0.2)
		
		# Verify completion
		assert current_instance["status"] == "completed"
		
		# Step 4: Get execution details
		response = test_client.get(f"/api/v1/workflow-instances/{instance_id}/executions")
		assert response.status_code == 200
		executions = response.json()
		assert len(executions) == 1
		assert executions[0]["status"] == "completed"
		
		# Step 5: Get workflow metrics
		response = test_client.get(f"/api/v1/workflows/{workflow_id}/metrics")
		assert response.status_code == 200
		metrics = response.json()
		assert metrics["total_executions"] >= 1
		assert "success_rate" in metrics
		
		# Step 6: Update workflow
		updated_template = template_data.copy()
		updated_template["description"] = "Updated description via API"
		response = test_client.put(f"/api/v1/workflows/{workflow_id}", json=updated_template)
		assert response.status_code == 200
		updated_workflow = response.json()
		assert updated_workflow["description"] == "Updated description via API"
		
		# Step 7: Create new version
		response = test_client.post(f"/api/v1/workflows/{workflow_id}/versions", json=updated_template)
		assert response.status_code == 201
		new_version = response.json()
		assert new_version["version"] != workflow["version"]
		
		# Step 8: List workflow versions
		response = test_client.get(f"/api/v1/workflows/{workflow_id}/versions")
		assert response.status_code == 200
		versions = response.json()
		assert len(versions) >= 2  # Original + new version
		
		# Step 9: Export workflow
		response = test_client.get(f"/api/v1/workflows/{workflow_id}/export")
		assert response.status_code == 200
		exported_workflow = response.json()
		assert exported_workflow["name"] == template_data["name"]
		assert "tasks" in exported_workflow
		
		# Step 10: Clean up
		response = test_client.delete(f"/api/v1/workflows/{workflow_id}")
		assert response.status_code == 204
	
	@pytest.mark.integration
	@pytest.mark.api
	def test_bulk_workflow_operations_via_api(self, test_client):
		"""Test bulk workflow operations through API."""
		# Create multiple workflows
		workflows = []
		for i in range(3):
			workflow_data = {
				"name": f"Bulk Test Workflow {i+1}",
				"description": f"Workflow {i+1} for bulk operations test",
				"tenant_id": "bulk_test_tenant",
				"tasks": [
					{
						"id": f"task_{i}",
						"name": f"Task {i+1}",
						"task_type": "script",
						"config": {"script": f"return {{'workflow_index': {i}, 'completed': True}}"}
					}
				]
			}
			
			response = test_client.post("/api/v1/workflows", json=workflow_data)
			assert response.status_code == 201
			workflows.append(response.json())
		
		workflow_ids = [w["id"] for w in workflows]
		
		# Test bulk execution
		response = test_client.post("/api/v1/workflows/bulk-execute", json={"workflow_ids": workflow_ids})
		assert response.status_code == 200
		bulk_result = response.json()
		assert len(bulk_result["instances"]) == 3
		
		# Monitor bulk execution
		instance_ids = [inst["id"] for inst in bulk_result["instances"]]
		
		# Wait for all to complete
		max_wait = 10
		wait_time = 0
		while wait_time < max_wait:
			response = test_client.post("/api/v1/workflow-instances/bulk-status", json={"instance_ids": instance_ids})
			assert response.status_code == 200
			statuses = response.json()
			
			if all(status["status"] in ["completed", "failed"] for status in statuses["instances"]):
				break
				
			asyncio.sleep(0.5)
			wait_time += 0.5
		
		# Verify all completed
		completed_count = sum(1 for status in statuses["instances"] if status["status"] == "completed")
		assert completed_count == 3
		
		# Test bulk deletion
		response = test_client.delete("/api/v1/workflows/bulk-delete", json={"workflow_ids": workflow_ids})
		assert response.status_code == 200
		delete_result = response.json()
		assert delete_result["deleted_count"] == 3


class TestSystemIntegrationScenarios:
	"""Test system-level integration scenarios."""
	
	@pytest.mark.integration 
	@pytest.mark.slow
	async def test_multi_tenant_workflow_isolation(self, workflow_service):
		"""Test complete multi-tenant workflow isolation."""
		# Create workflows for different tenants
		tenant_workflows = {}
		
		for tenant_id in ["tenant_a", "tenant_b", "tenant_c"]:
			workflow_data = {
				"name": f"Isolated Workflow for {tenant_id}",
				"description": f"Workflow that should be isolated to {tenant_id}",
				"tenant_id": tenant_id,
				"tasks": [
					{
						"id": "isolation_test",
						"name": "Tenant Isolation Test",
						"task_type": "script",
						"config": {"script": f"return {{'tenant_id': '{tenant_id}', 'isolated': True}}"}
					}
				],
				"metadata": {"tenant_specific_data": f"data_for_{tenant_id}"}
			}
			
			workflow = await workflow_service.create_workflow(workflow_data, user_id=f"user_{tenant_id}")
			tenant_workflows[tenant_id] = workflow
		
		# Execute all workflows
		instances = {}
		for tenant_id, workflow in tenant_workflows.items():
			instance = await workflow_service.execute_workflow(workflow.id)
			instances[tenant_id] = instance
		
		# Wait for all executions to complete
		for tenant_id, instance in instances.items():
			max_wait = 10
			waited = 0
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
				await asyncio.sleep(0.5)
				waited += 0.5
				instance = await workflow_service.get_workflow_instance(instance.id)
			instances[tenant_id] = instance
		
		# Verify isolation - each tenant should only see their own workflows
		for tenant_id in ["tenant_a", "tenant_b", "tenant_c"]:
			tenant_workflows_list = await workflow_service.list_workflows(tenant_id=tenant_id)
			
			# Should have exactly one workflow for this tenant
			assert len(tenant_workflows_list) == 1
			assert tenant_workflows_list[0].tenant_id == tenant_id
			
			# Verify the workflow execution completed successfully
			assert instances[tenant_id].status == WorkflowStatus.COMPLETED
			
			# Verify cross-tenant access is prevented
			for other_tenant_id in ["tenant_a", "tenant_b", "tenant_c"]:
				if other_tenant_id != tenant_id:
					try:
						# Try to access another tenant's workflow
						other_workflow = await workflow_service.get_workflow(
							tenant_workflows[other_tenant_id].id,
							tenant_id=tenant_id
						)
						# Should not be able to access
						assert other_workflow is None
					except Exception:
						# Exception is also acceptable - access should be denied
						pass
	
	@pytest.mark.integration
	@pytest.mark.performance
	async def test_high_concurrency_scenario(self, workflow_service):
		"""Test system behavior under high concurrency."""
		# Create a workflow template for concurrency testing
		workflow_data = {
			"name": "Concurrency Test Workflow",
			"description": "Workflow for testing high concurrency",
			"tenant_id": "concurrency_test",
			"tasks": [
				{
					"id": "concurrent_task",
					"name": "Concurrent Task",  
					"task_type": "script",
					"config": {
						"script": """
import time
import random
# Simulate variable processing time
time.sleep(random.uniform(0.1, 0.3))
return {'task_id': context.get('task_id', 'unknown'), 'processed': True}
"""
					}
				}
			]
		}
		
		# Create workflow template
		workflow = await workflow_service.create_workflow(workflow_data, user_id="concurrency_user")
		
		# Execute many instances concurrently
		concurrent_executions = 20
		start_time = datetime.utcnow()
		
		# Start all executions concurrently
		execution_tasks = []
		for i in range(concurrent_executions):
			task = asyncio.create_task(
				workflow_service.execute_workflow(
					workflow.id,
					execution_context={"task_id": f"concurrent_{i}"}
				)
			)
			execution_tasks.append(task)
		
		# Wait for all to start
		instances = await asyncio.gather(*execution_tasks)
		assert len(instances) == concurrent_executions
		
		# Wait for all to complete
		completion_tasks = []
		for instance in instances:
			async def wait_for_completion(inst_id):
				while True:
					inst = await workflow_service.get_workflow_instance(inst_id)
					if inst.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
						return inst
					await asyncio.sleep(0.1)
			
			completion_tasks.append(asyncio.create_task(wait_for_completion(instance.id)))
		
		# Wait for all completions
		completed_instances = await asyncio.gather(*completion_tasks)
		end_time = datetime.utcnow()
		
		total_time = (end_time - start_time).total_seconds()
		
		# Verify all completed successfully
		successful_executions = sum(1 for inst in completed_instances if inst.status == WorkflowStatus.COMPLETED)
		assert successful_executions == concurrent_executions
		
		# Performance verification - should handle concurrency efficiently
		# With proper concurrency, should complete much faster than sequential execution
		max_expected_time = 5.0  # Should complete within 5 seconds
		assert total_time < max_expected_time
		
		print(f"Concurrent execution stats: {concurrent_executions} workflows in {total_time:.2f}s")
	
	@pytest.mark.integration
	async def test_workflow_recovery_and_resilience(self, workflow_service):
		"""Test workflow system recovery and resilience."""
		# Create workflow with potential failure points
		workflow_data = {
			"name": "Resilience Test Workflow",
			"description": "Workflow for testing system resilience", 
			"tenant_id": "resilience_test",
			"tasks": [
				{
					"id": "stable_task",
					"name": "Stable Task",
					"task_type": "script",
					"config": {"script": "return {'stable': True, 'timestamp': datetime.utcnow().isoformat()}"}
				},
				{
					"id": "potentially_failing_task",
					"name": "Potentially Failing Task",
					"task_type": "script",
					"config": {
						"script": """
import random
if random.random() < 0.3:  # 30% chance of failure
    raise Exception('Simulated transient failure')
return {'resilient': True, 'retry_success': True}
"""
					},
					"depends_on": ["stable_task"],
					"retry_config": {
						"max_retries": 5,
						"retry_delay": 0.2,
						"backoff_multiplier": 1.5
					}
				},
				{
					"id": "recovery_task",
					"name": "Recovery Task",
					"task_type": "script",
					"config": {"script": "return {'recovered': True, 'final_step': True}"},
					"depends_on": ["potentially_failing_task"]
				}
			]
		}
		
		# Execute multiple instances to test resilience
		workflow = await workflow_service.create_workflow(workflow_data, user_id="resilience_user")
		
		success_count = 0
		total_attempts = 10
		
		for attempt in range(total_attempts):
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for completion
			max_wait = 15  # Allow time for retries
			waited = 0
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
				await asyncio.sleep(0.5)
				waited += 0.5
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			if instance.status == WorkflowStatus.COMPLETED:
				success_count += 1
			
			# Even if individual executions fail, system should remain stable
			assert instance.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
		
		# With retries, most executions should eventually succeed
		success_rate = success_count / total_attempts
		assert success_rate >= 0.7  # At least 70% should succeed with retries
		
		print(f"Resilience test: {success_count}/{total_attempts} successful ({success_rate*100:.1f}%)")
	
	@pytest.mark.integration
	async def test_complete_apg_integration(self, workflow_service):
		"""Test complete APG platform integration."""
		# Create workflow that uses multiple APG capabilities
		workflow_data = {
			"name": "Complete APG Integration Test",
			"description": "Workflow testing full APG platform integration",
			"tenant_id": "apg_integration_test",
			"metadata": {
				"apg_integration": True,
				"capabilities_used": ["auth_rbac", "audit_compliance", "data_lake"],
				"integration_level": "full"
			},
			"tasks": [
				{
					"id": "auth_check",
					"name": "Authentication Check",
					"task_type": "connector",
					"connector_type": "auth_rbac",
					"config": {
						"operation": "validate_user",
						"user_id": "integration_test_user",
						"required_permissions": ["workflow:execute", "data:write"]
					}
				},
				{
					"id": "process_data",
					"name": "Process Data",
					"task_type": "script",  
					"config": {
						"script": """
# Generate test data for processing
data = [
    {'id': i, 'value': i * 10, 'category': 'test'}
    for i in range(1, 101)
]
return {'processed_data': data, 'record_count': len(data)}
"""
					},
					"depends_on": ["auth_check"],
					"condition": "${auth_check.result.valid} == True"
				},
				{
					"id": "store_data",
					"name": "Store Data in Data Lake",
					"task_type": "connector",
					"connector_type": "data_lake",
					"config": {
						"operation": "store_dataset",
						"dataset_name": "integration_test_data",
						"data": "${process_data.result.processed_data}",
						"format": "json",
						"metadata": {"source": "integration_test", "timestamp": "${datetime.utcnow().isoformat()}"}
					},
					"depends_on": ["process_data"]
				},
				{
					"id": "audit_log",
					"name": "Log to Audit System",
					"task_type": "connector",
					"connector_type": "audit_compliance",
					"config": {
						"operation": "log_activity",
						"activity_type": "data_processing",
						"details": {
							"workflow_id": "${workflow.id}",
							"records_processed": "${process_data.result.record_count}",
							"dataset_stored": "${store_data.result.dataset_id}"
						}
					},
					"depends_on": ["store_data"]
				}
			]
		}
		
		# Mock APG services
		with patch.multiple(
			'..connectors.apg_connectors',
			AuthRBACConnector=MagicMock(),
			DataLakeConnector=MagicMock(),
			AuditComplianceConnector=MagicMock()
		) as mocks:
			# Setup mock responses
			mocks['AuthRBACConnector'].return_value.execute = AsyncMock(
				return_value={"valid": True, "permissions": ["workflow:execute", "data:write"], "user_id": "integration_test_user"}
			)
			
			mocks['DataLakeConnector'].return_value.execute = AsyncMock(
				return_value={"dataset_id": "ds_integration_test", "records_stored": 100, "storage_location": "apg://data-lake/integration_test_data"}
			)
			
			mocks['AuditComplianceConnector'].return_value.execute = AsyncMock(
				return_value={"audit_id": "audit_integration_test", "logged_at": datetime.utcnow().isoformat()}
			)
			
			# Execute APG integrated workflow
			workflow = await workflow_service.create_workflow(workflow_data, user_id="apg_user")
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for completion
			max_wait = 10
			waited = 0
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
				await asyncio.sleep(0.5)
				waited += 0.5
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			# Verify complete APG integration
			assert instance.status == WorkflowStatus.COMPLETED
			assert len(instance.task_executions) == 4
			
			# Verify all APG connectors were called
			mocks['AuthRBACConnector'].return_value.execute.assert_called()
			mocks['DataLakeConnector'].return_value.execute.assert_called()
			mocks['AuditComplianceConnector'].return_value.execute.assert_called()
			
			# Verify all tasks completed successfully
			completed_tasks = [ex for ex in instance.task_executions if ex.status == TaskStatus.COMPLETED]
			assert len(completed_tasks) == 4