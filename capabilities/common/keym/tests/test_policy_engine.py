#!/usr/bin/env python3
"""
APG Key Management - Policy Engine Tests
Comprehensive test suite for policy automation and compliance engine

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from ..policy_engine import (
	PolicyAutomationEngine, ComplianceMonitor, PolicyEvaluator,
	PolicyRule, ComplianceReport, PolicyViolation, PolicyEvaluationContext,
	PolicyEvaluationResult, create_policy_engine
)
from ..models import Key, KeyAlgorithm, KeyUsage, ComplianceFramework, create_key_spec_async


@pytest.fixture
async def policy_engine():
	"""Fixture for policy automation engine"""
	engine = PolicyAutomationEngine()
	await engine.initialize({
		'tenant_id': 'test_tenant',
		'compliance_frameworks': ['GDPR', 'HIPAA', 'PCI_DSS'],
		'policy_evaluation_interval': 3600,
		'test_mode': True
	})
	return engine


@pytest.fixture
async def sample_key():
	"""Fixture for sample key"""
	spec = await create_key_spec_async(
		tenant_id="test_tenant",
		algorithm=KeyAlgorithm.AES_256,
		usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
		name="Policy Test Key",
		created_by="test@datacraft.co.ke"
	)
	
	key = Key(
		spec=spec,
		key_material=b"test_key_material_32_bytes_long",
		key_checksum="abcd1234",
		usage_count=5000
	)
	return key


class TestPolicyAutomationEngine:
	"""Test PolicyAutomationEngine class"""
	
	@pytest.mark.asyncio
	async def test_engine_initialization(self):
		"""Test policy engine initialization"""
		engine = PolicyAutomationEngine()
		assert not engine.is_initialized
		
		config = {
			'tenant_id': 'test_tenant',
			'compliance_frameworks': ['GDPR', 'HIPAA'],
			'auto_remediation': True
		}
		await engine.initialize(config)
		
		assert engine.is_initialized
		assert engine.config == config
		assert isinstance(engine.policy_rules, dict)
		assert isinstance(engine.compliance_reports, list)
		assert isinstance(engine.policy_violations, list)
	
	@pytest.mark.asyncio
	async def test_factory_function(self):
		"""Test policy engine factory function"""
		engine = await create_policy_engine()
		assert isinstance(engine, PolicyAutomationEngine)
		assert engine.is_initialized
	
	@pytest.mark.asyncio
	async def test_evaluate_key_access_policy(self, policy_engine, sample_key):
		"""Test key access policy evaluation"""
		context = PolicyEvaluationContext(
			user_id="test@datacraft.co.ke",
			application_id="test-app",
			operation_type="decrypt",
			request_ip="192.168.1.100",
			request_time=datetime.utcnow(),
			user_roles=["key_user"],
			security_clearance="confidential"
		)
		
		result = await policy_engine.evaluate_key_access_policy(
			sample_key, 
			"decrypt", 
			context
		)
		
		assert isinstance(result, PolicyEvaluationResult)
		assert result.key_id == sample_key.spec.id
		assert result.operation == "decrypt"
		assert result.decision in ["allow", "deny", "conditional"]
		assert isinstance(result.violations, list)
		assert isinstance(result.required_actions, list)
		assert 0.0 <= result.confidence <= 1.0
	
	@pytest.mark.asyncio
	async def test_evaluate_compliance_policies(self, policy_engine, sample_key):
		"""Test compliance policy evaluation"""
		compliance_result = await policy_engine.evaluate_compliance_policies(
			sample_key, 
			[ComplianceFramework.GDPR, ComplianceFramework.HIPAA]
		)
		
		assert isinstance(compliance_result, dict)
		assert 'overall_compliance' in compliance_result
		assert 'framework_results' in compliance_result
		assert 'violations' in compliance_result
		assert 'recommendations' in compliance_result
		
		# Check framework-specific results
		assert ComplianceFramework.GDPR.value in compliance_result['framework_results']
		assert ComplianceFramework.HIPAA.value in compliance_result['framework_results']
		
		for framework, result in compliance_result['framework_results'].items():
			assert 'compliant' in result
			assert 'score' in result
			assert 'requirements_met' in result
			assert 'violations' in result
	
	@pytest.mark.asyncio
	async def test_create_policy_rule(self, policy_engine):
		"""Test policy rule creation"""
		rule = PolicyRule(
			tenant_id="test_tenant",
			rule_name="encryption_key_rotation",
			rule_type="lifecycle",
			conditions={
				"key_age_days": {"operator": ">=", "value": 90},
				"usage_count": {"operator": ">=", "value": 10000}
			},
			actions=["rotate_key", "notify_admin"],
			priority=8,
			compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
			enabled=True
		)
		
		result = await policy_engine.create_policy_rule(rule, "admin@datacraft.co.ke")
		
		assert result is True
		assert rule.rule_name in policy_engine.policy_rules
		assert policy_engine.policy_rules[rule.rule_name] == rule
	
	@pytest.mark.asyncio
	async def test_update_policy_rule(self, policy_engine):
		"""Test policy rule updates"""
		# Create initial rule
		rule = PolicyRule(
			tenant_id="test_tenant",
			rule_name="test_rule",
			rule_type="access",
			conditions={"algorithm": {"operator": "==", "value": "AES-256"}},
			actions=["log_access"],
			priority=5
		)
		
		await policy_engine.create_policy_rule(rule, "admin@datacraft.co.ke")
		
		# Update rule
		rule.priority = 9
		rule.actions.append("notify_security")
		
		result = await policy_engine.update_policy_rule(rule, "admin@datacraft.co.ke")
		
		assert result is True
		updated_rule = policy_engine.policy_rules["test_rule"]
		assert updated_rule.priority == 9
		assert "notify_security" in updated_rule.actions
	
	@pytest.mark.asyncio
	async def test_delete_policy_rule(self, policy_engine):
		"""Test policy rule deletion"""
		# Create rule first
		rule = PolicyRule(
			tenant_id="test_tenant",
			rule_name="delete_test_rule",
			rule_type="access",
			conditions={},
			actions=["deny"]
		)
		
		await policy_engine.create_policy_rule(rule, "admin@datacraft.co.ke")
		assert "delete_test_rule" in policy_engine.policy_rules
		
		# Delete rule
		result = await policy_engine.delete_policy_rule("delete_test_rule", "admin@datacraft.co.ke")
		
		assert result is True
		assert "delete_test_rule" not in policy_engine.policy_rules
	
	@pytest.mark.asyncio
	async def test_run_compliance_audit(self, policy_engine):
		"""Test compliance audit execution"""
		audit_config = {
			'frameworks': [ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
			'scope': 'all_keys',
			'include_recommendations': True,
			'generate_report': True
		}
		
		audit_result = await policy_engine.run_compliance_audit(
			audit_config, 
			"auditor@datacraft.co.ke"
		)
		
		assert audit_result is not None
		assert 'audit_id' in audit_result
		assert 'compliance_score' in audit_result
		assert 'violations_found' in audit_result
		assert 'frameworks_audited' in audit_result
		assert 'report_generated' in audit_result
		assert 'recommendations' in audit_result
	
	@pytest.mark.asyncio
	async def test_generate_compliance_report(self, policy_engine):
		"""Test compliance report generation"""
		report_config = {
			'frameworks': [ComplianceFramework.GDPR],
			'time_range': {
				'start': datetime.utcnow() - timedelta(days=30),
				'end': datetime.utcnow()
			},
			'include_violations': True,
			'include_remediation_actions': True
		}
		
		report = await policy_engine.generate_compliance_report(
			report_config, 
			"compliance@datacraft.co.ke"
		)
		
		assert isinstance(report, ComplianceReport)
		assert report.tenant_id == "test_tenant"
		assert ComplianceFramework.GDPR in report.frameworks
		assert report.generated_by == "compliance@datacraft.co.ke"
		assert isinstance(report.compliance_scores, dict)
		assert isinstance(report.violations, list)
		assert isinstance(report.recommendations, list)
	
	@pytest.mark.asyncio
	async def test_auto_remediation(self, policy_engine, sample_key):
		"""Test automated policy remediation"""
		# Create a violation scenario
		violation = PolicyViolation(
			tenant_id="test_tenant",
			key_id=sample_key.spec.id,
			rule_name="encryption_strength",
			violation_type="insufficient_key_size",
			severity="medium",
			description="Key size below recommended 256 bits",
			remediation_actions=["rotate_key_with_larger_size", "notify_admin"]
		)
		
		remediation_result = await policy_engine.execute_auto_remediation(
			violation, 
			"system"
		)
		
		assert remediation_result is not None
		assert 'remediation_id' in remediation_result
		assert 'actions_executed' in remediation_result
		assert 'success' in remediation_result
		assert 'results' in remediation_result
	
	@pytest.mark.asyncio
	async def test_policy_metrics_collection(self, policy_engine):
		"""Test policy metrics collection"""
		metrics = await policy_engine.get_policy_metrics("test_tenant")
		
		assert isinstance(metrics, dict)
		assert 'total_policies' in metrics
		assert 'active_policies' in metrics
		assert 'policy_violations_count' in metrics
		assert 'compliance_scores' in metrics
		assert 'auto_remediation_success_rate' in metrics
		assert 'policy_evaluation_count' in metrics
		assert 'average_evaluation_time_ms' in metrics
	
	@pytest.mark.asyncio
	async def test_policy_rule_validation(self, policy_engine):
		"""Test policy rule validation"""
		# Valid rule
		valid_rule = PolicyRule(
			tenant_id="test_tenant",
			rule_name="valid_rule",
			rule_type="access",
			conditions={"algorithm": {"operator": "in", "value": ["AES-256", "AES-128"]}},
			actions=["allow"],
			priority=5
		)
		
		is_valid, errors = await policy_engine.validate_policy_rule(valid_rule)
		assert is_valid is True
		assert len(errors) == 0
		
		# Invalid rule - missing required fields
		invalid_rule = PolicyRule(
			tenant_id="test_tenant",
			rule_name="",  # Empty name
			rule_type="invalid_type",  # Invalid type
			conditions={},
			actions=[],  # Empty actions
			priority=-1  # Invalid priority
		)
		
		is_valid, errors = await policy_engine.validate_policy_rule(invalid_rule)
		assert is_valid is False
		assert len(errors) > 0


class TestComplianceMonitor:
	"""Test ComplianceMonitor class"""
	
	@pytest.fixture
	def compliance_monitor(self):
		"""Fixture for compliance monitor"""
		return ComplianceMonitor()
	
	def test_gdpr_compliance_check(self, compliance_monitor, sample_key):
		"""Test GDPR compliance checking"""
		compliance_result = compliance_monitor.check_gdpr_compliance(sample_key)
		
		assert isinstance(compliance_result, dict)
		assert 'compliant' in compliance_result
		assert 'violations' in compliance_result
		assert 'requirements' in compliance_result
		
		# GDPR specific requirements
		requirements = compliance_result['requirements']
		assert 'data_minimization' in requirements
		assert 'purpose_limitation' in requirements
		assert 'storage_limitation' in requirements
		assert 'security_measures' in requirements
	
	def test_hipaa_compliance_check(self, compliance_monitor, sample_key):
		"""Test HIPAA compliance checking"""
		compliance_result = compliance_monitor.check_hipaa_compliance(sample_key)
		
		assert isinstance(compliance_result, dict)
		assert 'compliant' in compliance_result
		assert 'violations' in compliance_result
		assert 'requirements' in requirements
		
		# HIPAA specific requirements
		requirements = compliance_result['requirements']
		assert 'access_controls' in requirements
		assert 'audit_controls' in requirements
		assert 'integrity' in requirements
		assert 'transmission_security' in requirements
	
	def test_pci_dss_compliance_check(self, compliance_monitor, sample_key):
		"""Test PCI DSS compliance checking"""
		compliance_result = compliance_monitor.check_pci_dss_compliance(sample_key)
		
		assert isinstance(compliance_result, dict)
		assert 'compliant' in compliance_result
		assert 'violations' in compliance_result
		assert 'requirements' in compliance_result
		
		# PCI DSS specific requirements
		requirements = compliance_result['requirements']
		assert 'strong_cryptography' in requirements
		assert 'key_management' in requirements
		assert 'secure_transmission' in requirements
		assert 'access_restriction' in requirements


class TestPolicyEvaluator:
	"""Test PolicyEvaluator class"""
	
	@pytest.fixture
	def policy_evaluator(self):
		"""Fixture for policy evaluator"""
		return PolicyEvaluator()
	
	def test_evaluate_condition(self, policy_evaluator):
		"""Test policy condition evaluation"""
		# Test equality condition
		condition = {"operator": "==", "value": "AES-256"}
		result = policy_evaluator.evaluate_condition("AES-256", condition)
		assert result is True
		
		result = policy_evaluator.evaluate_condition("AES-128", condition)
		assert result is False
		
		# Test greater than condition
		condition = {"operator": ">", "value": 1000}
		result = policy_evaluator.evaluate_condition(1500, condition)
		assert result is True
		
		result = policy_evaluator.evaluate_condition(500, condition)
		assert result is False
		
		# Test in condition
		condition = {"operator": "in", "value": ["admin", "user", "viewer"]}
		result = policy_evaluator.evaluate_condition("admin", condition)
		assert result is True
		
		result = policy_evaluator.evaluate_condition("guest", condition)
		assert result is False
	
	def test_evaluate_complex_rule(self, policy_evaluator, sample_key):
		"""Test complex policy rule evaluation"""
		rule = PolicyRule(
			tenant_id="test_tenant",
			rule_name="complex_rule",
			rule_type="access",
			conditions={
				"algorithm": {"operator": "==", "value": "AES-256"},
				"usage_count": {"operator": "<=", "value": 10000},
				"user_role": {"operator": "in", "value": ["admin", "key_user"]}
			},
			actions=["allow"],
			priority=7
		)
		
		context = PolicyEvaluationContext(
			user_id="test@datacraft.co.ke",
			operation_type="encrypt",
			user_roles=["key_user"],
			request_time=datetime.utcnow()
		)
		
		evaluation_data = {
			"algorithm": sample_key.spec.algorithm.value,
			"usage_count": sample_key.usage_count,
			"user_role": "key_user"
		}
		
		result = policy_evaluator.evaluate_rule(rule, evaluation_data, context)
		
		assert isinstance(result, PolicyEvaluationResult)
		assert result.rule_matched is True
		assert result.decision == "allow"
	
	def test_evaluate_ml_based_rule(self, policy_evaluator):
		"""Test ML-based policy evaluation"""
		rule = PolicyRule(
			tenant_id="test_tenant",
			rule_name="ml_anomaly_rule",
			rule_type="security",
			conditions={
				"anomaly_score": {"operator": ">", "value": 0.8}
			},
			actions=["flag_for_review", "require_mfa"],
			priority=9,
			ml_enabled=True
		)
		
		context = PolicyEvaluationContext(
			user_id="suspicious@example.com",
			operation_type="decrypt",
			request_ip="10.0.0.1",
			request_time=datetime.utcnow()
		)
		
		# Mock ML prediction
		with patch.object(policy_evaluator, '_get_ml_anomaly_score', return_value=0.95):
			evaluation_data = {"anomaly_score": 0.95}
			result = policy_evaluator.evaluate_rule(rule, evaluation_data, context)
			
			assert result.rule_matched is True
			assert result.decision in ["deny", "conditional"]
			assert "require_mfa" in result.required_actions


class TestPolicyModels:
	"""Test policy data models"""
	
	def test_policy_rule_creation(self):
		"""Test PolicyRule model"""
		rule = PolicyRule(
			tenant_id="test_tenant",
			rule_name="test_rule",
			rule_type="access",
			conditions={
				"user_role": {"operator": "in", "value": ["admin", "user"]},
				"time_of_day": {"operator": "between", "value": ["09:00", "17:00"]}
			},
			actions=["allow", "log_access"],
			priority=7,
			compliance_frameworks=[ComplianceFramework.GDPR],
			enabled=True,
			created_by="admin@datacraft.co.ke"
		)
		
		assert rule.tenant_id == "test_tenant"
		assert rule.rule_name == "test_rule"
		assert rule.priority == 7
		assert len(rule.conditions) == 2
		assert len(rule.actions) == 2
		assert ComplianceFramework.GDPR in rule.compliance_frameworks
		assert rule.enabled is True
	
	def test_policy_violation_creation(self):
		"""Test PolicyViolation model"""
		violation = PolicyViolation(
			tenant_id="test_tenant",
			key_id="key_123",
			rule_name="encryption_policy",
			violation_type="weak_encryption",
			severity="high",
			description="Key uses deprecated algorithm",
			remediation_actions=["rotate_key", "update_algorithm"],
			metadata={'current_algorithm': 'DES', 'required_algorithm': 'AES-256'}
		)
		
		assert violation.tenant_id == "test_tenant"
		assert violation.key_id == "key_123"
		assert violation.severity == "high"
		assert len(violation.remediation_actions) == 2
		assert 'current_algorithm' in violation.metadata
		assert violation.status == "open"  # Default status
	
	def test_compliance_report_creation(self):
		"""Test ComplianceReport model"""
		report = ComplianceReport(
			tenant_id="test_tenant",
			frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
			compliance_scores={
				ComplianceFramework.GDPR.value: 0.85,
				ComplianceFramework.HIPAA.value: 0.92
			},
			violations=[],
			recommendations=["Implement stronger encryption", "Regular key rotation"],
			generated_by="compliance@datacraft.co.ke",
			metadata={'audit_scope': 'full', 'total_keys_evaluated': 150}
		)
		
		assert report.tenant_id == "test_tenant"
		assert len(report.frameworks) == 2
		assert report.compliance_scores[ComplianceFramework.GDPR.value] == 0.85
		assert len(report.recommendations) == 2
		assert 'audit_scope' in report.metadata


class TestIntegrationScenarios:
	"""Test integration scenarios"""
	
	@pytest.mark.asyncio
	async def test_end_to_end_policy_enforcement(self, policy_engine, sample_key):
		"""Test complete policy enforcement flow"""
		# 1. Create policy rule
		rule = PolicyRule(
			tenant_id="test_tenant",
			rule_name="business_hours_access",
			rule_type="access",
			conditions={
				"hour_of_day": {"operator": "between", "value": [9, 17]},
				"user_role": {"operator": "in", "value": ["employee", "admin"]}
			},
			actions=["allow", "log_access"],
			priority=8
		)
		
		await policy_engine.create_policy_rule(rule, "admin@datacraft.co.ke")
		
		# 2. Test policy evaluation during business hours
		business_hours_context = PolicyEvaluationContext(
			user_id="employee@company.com",
			operation_type="encrypt",
			user_roles=["employee"],
			request_time=datetime.utcnow().replace(hour=14)  # 2 PM
		)
		
		result = await policy_engine.evaluate_key_access_policy(
			sample_key, "encrypt", business_hours_context
		)
		assert result.decision == "allow"
		
		# 3. Test policy evaluation outside business hours
		after_hours_context = PolicyEvaluationContext(
			user_id="employee@company.com",
			operation_type="encrypt",
			user_roles=["employee"],
			request_time=datetime.utcnow().replace(hour=22)  # 10 PM
		)
		
		result = await policy_engine.evaluate_key_access_policy(
			sample_key, "encrypt", after_hours_context
		)
		# Should be denied or conditional based on policy
		assert result.decision in ["deny", "conditional"]
	
	@pytest.mark.asyncio
	async def test_compliance_audit_and_remediation(self, policy_engine):
		"""Test compliance audit with automatic remediation"""
		# 1. Run compliance audit
		audit_config = {
			'frameworks': [ComplianceFramework.GDPR],
			'scope': 'all_keys',
			'auto_remediation': True
		}
		
		audit_result = await policy_engine.run_compliance_audit(
			audit_config, 
			"auditor@datacraft.co.ke"
		)
		
		assert 'violations_found' in audit_result
		
		# 2. If violations found, test auto-remediation
		if audit_result['violations_found'] > 0:
			# Mock some violations for testing
			violation = PolicyViolation(
				tenant_id="test_tenant",
				key_id="test_key",
				rule_name="gdpr_encryption",
				violation_type="insufficient_encryption",
				severity="medium",
				remediation_actions=["rotate_key", "update_policy"]
			)
			
			remediation_result = await policy_engine.execute_auto_remediation(
				violation, 
				"system"
			)
			
			assert remediation_result['success'] is not None
			assert 'actions_executed' in remediation_result


if __name__ == "__main__":
	pytest.main([__file__])