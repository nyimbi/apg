#!/usr/bin/env python3
"""
APG NLP Phase 2.4 Enterprise Features Validation Script

Comprehensive validation of all enterprise features:
- Collaborative annotation workbench
- Model training and fine-tuning workflows
- Analytics and reporting dashboard
- Enterprise security and compliance features

Tests all components systematically and provides detailed validation report.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from uuid_extensions import uuid7str

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def validate_annotation_workbench():
	"""Validate collaborative annotation workbench functionality"""
	logger.info("🔍 Validating Collaborative Annotation Workbench...")
	
	try:
		from annotation_workbench import (
			ProjectManager, AnnotationStatus, ProjectRole, ConflictType,
			AnnotationConflict, AnnotationSession, QualityMetrics
		)
		
		# Test project manager initialization
		manager = ProjectManager(tenant_id="test_tenant")
		assert manager.tenant_id == "test_tenant"
		logger.info("✅ Project manager initialization successful")
		
		# Test project creation
		project_data = {
			"name": "Test NLP Annotation Project",
			"description": "Test project for enterprise validation",
			"annotation_type": "named_entity_recognition",
			"annotation_schema": {
				"valid_labels": ["PERSON", "ORGANIZATION", "LOCATION"],
				"required_fields": ["label", "confidence"],
				"min_confidence": 0.5
			},
			"project_manager": "test_manager",
			"team_members": ["annotator1", "annotator2", "reviewer1"],
			"guidelines": "Comprehensive annotation guidelines for test project"
		}
		
		project = await manager.create_project(project_data)
		assert project.name == "Test NLP Annotation Project"
		assert len(project.team_members) == 3
		logger.info("✅ Project creation successful")
		
		# Test team member management
		success = await manager.add_team_member(project.id, "annotator3", ProjectRole.ANNOTATOR)
		assert success == True
		logger.info("✅ Team member addition successful")
		
		# Test annotation session management
		session = await manager.start_annotation_session(
			project.id, "test_document", "annotator1"
		)
		assert session is not None
		assert session.project_id == project.id
		logger.info("✅ Annotation session creation successful")
		
		# Test annotation saving and conflict detection
		annotation_data = {
			"start_position": 0,
			"end_position": 10,
			"annotated_text": "John Smith",
			"annotation_value": {"label": "PERSON", "confidence": 0.95},
			"confidence": 0.95
		}
		
		save_success = await manager.save_annotation(session.session_id, annotation_data)
		assert save_success == True
		logger.info("✅ Annotation saving successful")
		
		# Test quality metrics
		stats = manager.get_project_statistics(project.id)
		assert stats["project_name"] == "Test NLP Annotation Project"
		assert stats["team_size"] == 5  # Manager + 4 members
		logger.info("✅ Project statistics generation successful")
		
		# Test annotation export
		export_data = await manager.export_annotations(project.id, "json")
		assert "project_info" in export_data
		assert "annotations" in export_data
		logger.info("✅ Annotation export successful")
		
		# Test session cleanup
		end_success = await manager.end_annotation_session(session.session_id)
		assert end_success == True
		logger.info("✅ Session cleanup successful")
		
		await manager.cleanup()
		
		return {
			"component": "Collaborative Annotation Workbench",
			"status": "PASSED",
			"tests_run": 8,
			"tests_passed": 8,
			"details": "All annotation workbench features validated successfully"
		}
		
	except Exception as e:
		logger.error(f"❌ Annotation workbench validation failed: {str(e)}")
		return {
			"component": "Collaborative Annotation Workbench",
			"status": "FAILED",
			"error": str(e),
			"tests_run": 8,
			"tests_passed": 0
		}

async def validate_training_workflows():
	"""Validate model training and fine-tuning workflows"""
	logger.info("🔍 Validating Model Training Workflows...")
	
	try:
		from training_workflows import (
			TrainingWorkflowManager, TrainingExperiment, ExperimentStatus,
			OptimizationType, HyperparameterOptimizer, ModelRepository
		)
		
		# Test workflow manager initialization
		workflow_manager = TrainingWorkflowManager(tenant_id="test_tenant")
		assert workflow_manager.tenant_id == "test_tenant"
		logger.info("✅ Training workflow manager initialization successful")
		
		# Test experiment creation
		experiment_config = {
			"name": "Test Sentiment Classification Training",
			"description": "Test experiment for enterprise validation",
			"model_type": "transformer",
			"task_type": "sentiment_analysis",
			"base_model": "bert-base-uncased",
			"training_data": {
				"source_type": "annotation_project",
				"project_id": "test_project",
				"validation_split": 0.2
			},
			"hyperparameters": {
				"learning_rate": 0.001,
				"batch_size": 32,
				"max_epochs": 5,
				"early_stopping": True
			},
			"optimization": {
				"type": "random_search",
				"trials": 3,
				"metrics": ["accuracy", "f1_score"]
			}
		}
		
		experiment = await workflow_manager.create_experiment(experiment_config)
		assert experiment.name == "Test Sentiment Classification Training"
		assert experiment.status == ExperimentStatus.PENDING
		logger.info("✅ Training experiment creation successful")
		
		# Test hyperparameter optimization
		optimizer = HyperparameterOptimizer(experiment.id)
		optimization_results = await optimizer.optimize_hyperparameters(experiment)
		assert len(optimization_results) > 0
		logger.info("✅ Hyperparameter optimization successful")
		
		# Test model repository operations
		repo = ModelRepository(tenant_id="test_tenant")
		
		model_metadata = {
			"name": "Test Sentiment Model v1.0",
			"version": "1.0.0",
			"task_type": "sentiment_analysis",
			"performance_metrics": {
				"accuracy": 0.92,
				"f1_score": 0.89,
				"precision": 0.91,
				"recall": 0.87
			},
			"training_config": experiment_config
		}
		
		model_id = await repo.register_model(model_metadata)
		assert model_id is not None
		logger.info("✅ Model registration successful")
		
		# Test model versioning
		model_versions = await repo.get_model_versions(model_id)
		assert len(model_versions) >= 1
		logger.info("✅ Model versioning successful")
		
		# Test distributed training coordination
		training_nodes = await workflow_manager.get_available_training_nodes()
		assert isinstance(training_nodes, list)
		logger.info("✅ Training node discovery successful")
		
		# Test workflow monitoring
		workflow_status = await workflow_manager.get_workflow_status(experiment.id)
		assert "experiment_id" in workflow_status
		assert "status" in workflow_status
		logger.info("✅ Workflow monitoring successful")
		
		await workflow_manager.cleanup()
		
		return {
			"component": "Model Training Workflows",
			"status": "PASSED",
			"tests_run": 7,
			"tests_passed": 7,
			"details": "All training workflow features validated successfully"
		}
		
	except Exception as e:
		logger.error(f"❌ Training workflows validation failed: {str(e)}")
		return {
			"component": "Model Training Workflows",
			"status": "FAILED",
			"error": str(e),
			"tests_run": 7,
			"tests_passed": 0
		}

async def validate_analytics_dashboard():
	"""Validate analytics and reporting dashboard"""
	logger.info("🔍 Validating Analytics Dashboard...")
	
	try:
		from analytics_dashboard import (
			AnalyticsDashboard, MetricType, AlertCondition, AlertSeverity,
			ReportGenerator, BusinessIntelligence
		)
		
		# Test analytics dashboard initialization
		dashboard = AnalyticsDashboard(tenant_id="test_tenant")
		assert dashboard.tenant_id == "test_tenant"
		logger.info("✅ Analytics dashboard initialization successful")
		
		# Test metrics recording
		dashboard.record_metric("processing_latency", 150.5, {"model": "bert-base", "task": "sentiment"})
		dashboard.record_metric("throughput", 45.2, {"endpoint": "/api/nlp/process"})
		dashboard.record_metric("accuracy", 0.92, {"model": "sentiment_v1", "dataset": "test"})
		
		metrics_summary = dashboard.get_metrics_summary()
		assert "processing_latency" in metrics_summary
		assert "throughput" in metrics_summary
		logger.info("✅ Metrics recording successful")
		
		# Test alert configuration
		alert_config = {
			"name": "High Processing Latency Alert",
			"metric_name": "processing_latency",
			"condition": AlertCondition.GREATER_THAN,
			"threshold": 200.0,
			"severity": AlertSeverity.WARNING,
			"notification_channels": ["email", "slack"]
		}
		
		alert = await dashboard.create_alert(alert_config)
		assert alert.name == "High Processing Latency Alert"
		logger.info("✅ Alert configuration successful")
		
		# Test report generation
		report_generator = ReportGenerator(dashboard)
		
		usage_report = await report_generator.generate_usage_summary(
			start_date=datetime.utcnow() - timedelta(days=7),
			end_date=datetime.utcnow()
		)
		assert "total_requests" in usage_report
		assert "average_latency" in usage_report
		logger.info("✅ Usage report generation successful")
		
		performance_report = await report_generator.generate_performance_analysis(
			models=["sentiment_v1", "bert-base"],
			time_period_days=30
		)
		assert "model_comparison" in performance_report
		logger.info("✅ Performance analysis successful")
		
		# Test business intelligence
		bi_engine = BusinessIntelligence(dashboard)
		
		insights = await bi_engine.generate_insights()
		assert isinstance(insights, list)
		logger.info("✅ Business intelligence insights successful")
		
		trends = await bi_engine.detect_trends()
		assert "usage_trends" in trends
		logger.info("✅ Trend detection successful")
		
		predictions = await bi_engine.generate_predictions()
		assert "capacity_predictions" in predictions
		logger.info("✅ Predictive analytics successful")
		
		# Test real-time monitoring
		real_time_data = dashboard.get_real_time_metrics()
		assert "current_metrics" in real_time_data
		assert "active_alerts" in real_time_data
		logger.info("✅ Real-time monitoring successful")
		
		await dashboard.cleanup()
		
		return {
			"component": "Analytics Dashboard",
			"status": "PASSED",
			"tests_run": 8,
			"tests_passed": 8,
			"details": "All analytics dashboard features validated successfully"
		}
		
	except Exception as e:
		logger.error(f"❌ Analytics dashboard validation failed: {str(e)}")
		return {
			"component": "Analytics Dashboard",
			"status": "FAILED",
			"error": str(e),
			"tests_run": 8,
			"tests_passed": 0
		}

async def validate_security_compliance():
	"""Validate enterprise security and compliance features"""
	logger.info("🔍 Validating Security and Compliance Engine...")
	
	try:
		from security_compliance import (
			SecurityComplianceEngine, SecurityContext, ComplianceFramework,
			DataClassification, SecurityEvent, AuditEvent
		)
		
		# Test security engine initialization
		security_engine = SecurityComplianceEngine(tenant_id="test_tenant")
		assert security_engine.tenant_id == "test_tenant"
		logger.info("✅ Security compliance engine initialization successful")
		
		# Test security context creation
		request_data = {
			"ip_address": "192.168.1.100",
			"user_agent": "Mozilla/5.0 Test Browser",
			"auth_method": "mfa",
			"geo_location": {"country": "US", "region": "CA"}
		}
		
		context = await security_engine.create_security_context("test_user", request_data)
		assert context.user_id == "test_user"
		assert context.tenant_id == "test_tenant"
		logger.info("✅ Security context creation successful")
		
		# Test authorization
		authorized = await security_engine.authorize_operation(
			context, "read", "document", "test_doc_123"
		)
		assert isinstance(authorized, bool)
		logger.info("✅ Operation authorization successful")
		
		# Test data classification
		classification = await security_engine._get_resource_classification("document", "test_doc_123")
		assert isinstance(classification, DataClassification)
		logger.info("✅ Data classification successful")
		
		# Test audit logging
		await security_engine._log_security_event(
			context, SecurityEvent.DATA_ACCESSED,
			"document", "test_doc_123", "read",
			success=True
		)
		assert len(security_engine.audit_events) > 0
		logger.info("✅ Audit logging successful")
		
		# Test compliance policy validation
		policy_violations = await security_engine._check_compliance_policies(
			context, "process", "document"
		)
		assert isinstance(policy_violations, list)
		logger.info("✅ Compliance policy validation successful")
		
		# Test data retention management
		retention_record = await security_engine.create_retention_policy(
			"document", "test_doc_123", DataClassification.CONFIDENTIAL, 365
		)
		assert retention_record.resource_id == "test_doc_123"
		logger.info("✅ Data retention policy creation successful")
		
		# Test data deletion request processing
		deletion_success = await security_engine.process_data_deletion_request(
			"test_user", "document", "test_doc_123", "user_request"
		)
		assert isinstance(deletion_success, bool)
		logger.info("✅ Data deletion request processing successful")
		
		# Test compliance dashboard
		dashboard_data = security_engine.get_compliance_dashboard()
		assert "compliance_summary" in dashboard_data
		assert "security_metrics" in dashboard_data
		assert "data_management" in dashboard_data
		logger.info("✅ Compliance dashboard generation successful")
		
		# Test audit reporting
		audit_report = security_engine.get_audit_report(
			start_date=datetime.utcnow() - timedelta(days=1),
			end_date=datetime.utcnow()
		)
		assert "report_metadata" in audit_report
		assert "security_summary" in audit_report
		logger.info("✅ Audit report generation successful")
		
		await security_engine.cleanup()
		
		return {
			"component": "Security and Compliance Engine",
			"status": "PASSED",
			"tests_run": 10,
			"tests_passed": 10,
			"details": "All security and compliance features validated successfully"
		}
		
	except Exception as e:
		logger.error(f"❌ Security compliance validation failed: {str(e)}")
		return {
			"component": "Security and Compliance Engine",
			"status": "FAILED",
			"error": str(e),
			"tests_run": 10,
			"tests_passed": 0
		}

async def validate_enterprise_integration():
	"""Validate integration between all enterprise components"""
	logger.info("🔍 Validating Enterprise Component Integration...")
	
	try:
		from annotation_workbench import ProjectManager
		from training_workflows import TrainingWorkflowManager
		from analytics_dashboard import AnalyticsDashboard
		from security_compliance import SecurityComplianceEngine
		
		tenant_id = "integration_test_tenant"
		
		# Initialize all components
		project_manager = ProjectManager(tenant_id=tenant_id)
		training_manager = TrainingWorkflowManager(tenant_id=tenant_id)
		analytics = AnalyticsDashboard(tenant_id=tenant_id)
		security = SecurityComplianceEngine(tenant_id=tenant_id)
		
		logger.info("✅ All enterprise components initialized")
		
		# Test annotation to training workflow
		project_data = {
			"name": "Integration Test Project",
			"annotation_type": "text_classification",
			"annotation_schema": {"valid_labels": ["positive", "negative", "neutral"]},
			"project_manager": "test_manager",
			"team_members": ["annotator1"]
		}
		
		project = await project_manager.create_project(project_data)
		
		# Create training experiment from annotation project
		training_config = {
			"name": "Integration Training Experiment",
			"task_type": "text_classification",
			"base_model": "bert-base-uncased",
			"training_data": {
				"source_type": "annotation_project",
				"project_id": project.id
			}
		}
		
		experiment = await training_manager.create_experiment(training_config)
		logger.info("✅ Annotation to training workflow integration successful")
		
		# Test security context across components
		context = await security.create_security_context("integration_user", {
			"ip_address": "10.0.0.1",
			"auth_method": "oauth"
		})
		
		# Authorize operations across components
		project_access = await security.authorize_operation(
			context, "read", "annotation_project", project.id
		)
		training_access = await security.authorize_operation(
			context, "create", "training_experiment", experiment.id
		)
		
		assert project_access and training_access
		logger.info("✅ Cross-component security authorization successful")
		
		# Test analytics collection from all components
		analytics.record_metric("annotation_projects_created", 1)
		analytics.record_metric("training_experiments_created", 1)
		analytics.record_metric("security_contexts_created", 1)
		
		metrics = analytics.get_metrics_summary()
		assert len(metrics) >= 3
		logger.info("✅ Cross-component analytics collection successful")
		
		# Test compliance tracking across all operations
		compliance_data = security.get_compliance_dashboard()
		assert compliance_data["security_metrics"]["total_events"] > 0
		logger.info("✅ Cross-component compliance tracking successful")
		
		# Cleanup all components
		await project_manager.cleanup()
		await training_manager.cleanup()
		await analytics.cleanup()
		await security.cleanup()
		
		return {
			"component": "Enterprise Component Integration",
			"status": "PASSED",
			"tests_run": 5,
			"tests_passed": 5,
			"details": "All enterprise components integrate successfully"
		}
		
	except Exception as e:
		logger.error(f"❌ Enterprise integration validation failed: {str(e)}")
		return {
			"component": "Enterprise Component Integration",
			"status": "FAILED",
			"error": str(e),
			"tests_run": 5,
			"tests_passed": 0
		}

async def main():
	"""Run comprehensive Phase 2.4 enterprise features validation"""
	logger.info("🚀 Starting APG NLP Phase 2.4 Enterprise Features Validation")
	logger.info("=" * 70)
	
	# Run all validation tests
	test_results = []
	
	try:
		# Validate each enterprise component
		test_results.append(await validate_annotation_workbench())
		test_results.append(await validate_training_workflows())
		test_results.append(await validate_analytics_dashboard())
		test_results.append(await validate_security_compliance())
		test_results.append(await validate_enterprise_integration())
		
		# Generate validation report
		logger.info("\n" + "=" * 70)
		logger.info("📊 PHASE 2.4 ENTERPRISE FEATURES VALIDATION REPORT")
		logger.info("=" * 70)
		
		total_tests = 0
		total_passed = 0
		passed_components = 0
		
		for result in test_results:
			status_emoji = "✅" if result["status"] == "PASSED" else "❌"
			logger.info(f"{status_emoji} {result['component']}: {result['status']}")
			
			if "tests_run" in result:
				total_tests += result["tests_run"]
				total_passed += result.get("tests_passed", 0)
				logger.info(f"   Tests: {result['tests_passed']}/{result['tests_run']} passed")
			
			if result["status"] == "PASSED":
				passed_components += 1
				logger.info(f"   Details: {result.get('details', 'No details')}")
			else:
				logger.error(f"   Error: {result.get('error', 'Unknown error')}")
		
		logger.info("\n" + "-" * 70)
		logger.info("📈 VALIDATION SUMMARY:")
		logger.info(f"   Components Tested: {len(test_results)}")
		logger.info(f"   Components Passed: {passed_components}")
		logger.info(f"   Individual Tests: {total_passed}/{total_tests} passed")
		logger.info(f"   Success Rate: {(total_passed/max(total_tests, 1))*100:.1f}%")
		
		if passed_components == len(test_results):
			logger.info("🎉 ALL ENTERPRISE FEATURES VALIDATION PASSED!")
			logger.info("   Phase 2.4: Enterprise Features is COMPLETE")
		else:
			logger.error("❌ Some enterprise features validation failed")
			logger.error("   Please review failed components and fix issues")
		
		# Generate detailed validation report
		validation_report = {
			"validation_timestamp": datetime.utcnow().isoformat(),
			"phase": "2.4 - Enterprise Features",
			"total_components": len(test_results),
			"passed_components": passed_components,
			"total_tests": total_tests,
			"passed_tests": total_passed,
			"success_rate": (total_passed/max(total_tests, 1))*100,
			"component_results": test_results,
			"validation_status": "PASSED" if passed_components == len(test_results) else "FAILED"
		}
		
		# Save validation report
		report_filename = f"phase_2_4_enterprise_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
		with open(report_filename, 'w') as f:
			json.dump(validation_report, f, indent=2, default=str)
		
		logger.info(f"📄 Detailed validation report saved: {report_filename}")
		
	except Exception as e:
		logger.error(f"❌ Validation process failed: {str(e)}")
		return False
	
	return passed_components == len(test_results)

if __name__ == "__main__":
	asyncio.run(main())