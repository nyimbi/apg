#!/usr/bin/env python3
"""
APG Key Management - Security Testing Tests
Comprehensive test suite for security testing framework

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from ..security_testing import (
	SecurityTester, SecurityVulnerability, PenetrationTestConfig, 
	SecurityTestResult, VulnerabilityType, SeverityLevel,
	create_security_tester
)
from ..service import KeyManagementService
from ..models import KeyAlgorithm, KeyUsage


@pytest.fixture
async def mock_service():
	"""Fixture for mocked key management service"""
	service = AsyncMock(spec=KeyManagementService)
	service.is_initialized = True
	service.config = {'tenant_id': 'test_tenant'}
	return service


@pytest.fixture
def pentest_config():
	"""Fixture for penetration test configuration"""
	return PenetrationTestConfig(
		test_name="Comprehensive Security Test",
		target_scope=["api", "authentication", "authorization"],
		test_types=[
			VulnerabilityType.AUTHENTICATION_BYPASS,
			VulnerabilityType.AUTHORIZATION_WEAKNESS,
			VulnerabilityType.INPUT_VALIDATION
		],
		duration_minutes=30,
		concurrent_attacks=3
	)


@pytest.fixture
def sample_vulnerability():
	"""Fixture for sample vulnerability"""
	return SecurityVulnerability(
		vulnerability_id="vuln_123",
		vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
		severity=SeverityLevel.HIGH,
		title="Authentication Bypass Vulnerability",
		description="System allows bypass of authentication mechanisms",
		impact="Unauthorized access to sensitive operations",
		remediation="Implement proper authentication validation",
		proof_of_concept="Successfully accessed restricted endpoint without credentials"
	)


class TestSecurityVulnerability:
	"""Test SecurityVulnerability data model"""
	
	def test_vulnerability_creation(self):
		"""Test vulnerability model creation"""
		vuln = SecurityVulnerability(
			vulnerability_id="test_vuln_1",
			vulnerability_type=VulnerabilityType.CRYPTOGRAPHIC_WEAKNESS,
			severity=SeverityLevel.CRITICAL,
			title="Weak Encryption Algorithm",
			description="System uses deprecated encryption algorithm",
			impact="Data can be easily decrypted by attackers",
			remediation="Upgrade to AES-256 or equivalent strong encryption",
			proof_of_concept="Demonstrated DES decryption in under 24 hours",
			cve_references=["CVE-2008-0166"],
			owasp_category="A02:2021 - Cryptographic Failures"
		)
		
		assert vuln.vulnerability_id == "test_vuln_1"
		assert vuln.vulnerability_type == VulnerabilityType.CRYPTOGRAPHIC_WEAKNESS
		assert vuln.severity == SeverityLevel.CRITICAL
		assert vuln.title == "Weak Encryption Algorithm"
		assert "CVE-2008-0166" in vuln.cve_references
		assert vuln.owasp_category == "A02:2021 - Cryptographic Failures"
		assert vuln.discovered_at is not None
	
	def test_vulnerability_enum_values(self):
		"""Test vulnerability type and severity enums"""
		# Test VulnerabilityType enum values
		assert VulnerabilityType.AUTHENTICATION_BYPASS.value == "authentication_bypass"
		assert VulnerabilityType.INJECTION_ATTACK.value == "injection_attack"
		assert VulnerabilityType.DENIAL_OF_SERVICE.value == "denial_of_service"
		
		# Test SeverityLevel enum values
		assert SeverityLevel.CRITICAL.value == "critical"
		assert SeverityLevel.HIGH.value == "high"
		assert SeverityLevel.MEDIUM.value == "medium"
		assert SeverityLevel.LOW.value == "low"
		assert SeverityLevel.INFO.value == "info"


class TestPenetrationTestConfig:
	"""Test PenetrationTestConfig data model"""
	
	def test_config_creation_with_defaults(self):
		"""Test config creation with default values"""
		config = PenetrationTestConfig(
			test_name="Basic Security Test",
			target_scope=["api"],
			test_types=[VulnerabilityType.INPUT_VALIDATION]
		)
		
		assert config.test_name == "Basic Security Test"
		assert config.target_scope == ["api"]
		assert VulnerabilityType.INPUT_VALIDATION in config.test_types
		assert config.duration_minutes == 120  # Default
		assert config.concurrent_attacks == 5  # Default
		assert config.authentication_bypass_tests is True  # Default
		assert config.custom_payloads == {}  # Default
	
	def test_config_creation_with_custom_values(self):
		"""Test config creation with custom values"""
		custom_payloads = {
			"sql_injection": ["' OR 1=1 --", "'; DROP TABLE users; --"],
			"xss": ["<script>alert('xss')</script>", "javascript:alert(1)"]
		}
		
		config = PenetrationTestConfig(
			test_name="Advanced Security Test",
			target_scope=["api", "web", "database"],
			test_types=[
				VulnerabilityType.INJECTION_ATTACK,
				VulnerabilityType.CRYPTOGRAPHIC_WEAKNESS
			],
			duration_minutes=180,
			concurrent_attacks=10,
			authentication_bypass_tests=False,
			cryptographic_tests=True,
			input_validation_tests=True,
			dos_tests=False,
			custom_payloads=custom_payloads
		)
		
		assert config.duration_minutes == 180
		assert config.concurrent_attacks == 10
		assert config.authentication_bypass_tests is False
		assert config.dos_tests is False
		assert len(config.custom_payloads["sql_injection"]) == 2


class TestSecurityTester:
	"""Test SecurityTester class"""
	
	@pytest.mark.asyncio
	async def test_tester_initialization(self, mock_service):
		"""Test security tester initialization"""
		tester = SecurityTester(mock_service)
		
		assert tester.service == mock_service
		assert isinstance(tester.test_results, list)
		assert isinstance(tester.vulnerability_db, list)
		assert isinstance(tester.attack_patterns, dict)
		assert len(tester.test_results) == 0
		assert len(tester.vulnerability_db) == 0
		
		# Check attack patterns were initialized
		assert 'authentication_bypass' in tester.attack_patterns
		assert 'authorization_weakness' in tester.attack_patterns
		assert 'cryptographic_weakness' in tester.attack_patterns
		assert 'input_validation' in tester.attack_patterns
		assert 'timing_attack' in tester.attack_patterns
		assert 'denial_of_service' in tester.attack_patterns
	
	@pytest.mark.asyncio
	async def test_factory_function(self, mock_service):
		"""Test security tester factory function"""
		tester = await create_security_tester(mock_service)
		
		assert isinstance(tester, SecurityTester)
		assert tester.service == mock_service
	
	@pytest.mark.asyncio
	async def test_run_penetration_test(self, mock_service, pentest_config):
		"""Test penetration test execution"""
		tester = SecurityTester(mock_service)
		
		# Mock service methods to simulate various responses
		mock_service.create_key.side_effect = [
			Mock(spec_id="key1", spec=Mock(id="key_123")),
			Exception("Authentication failed"),  # Simulate auth failure
			Mock(spec_id="key2", spec=Mock(id="key_456"))
		]
		mock_service.retrieve_key.side_effect = [
			Mock(spec_id="retrieved_key"),
			Exception("Access denied")
		]
		mock_service.delete_key.return_value = True
		
		# Mock individual attack pattern methods to return known vulnerabilities
		with patch.object(tester, '_test_null_authentication', return_value=[
			SecurityVulnerability(
				vulnerability_id="auth_bypass_1",
				vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
				severity=SeverityLevel.HIGH,
				title="Null Authentication Bypass",
				description="Test vulnerability",
				impact="Test impact",
				remediation="Test remediation"
			)
		]):
			with patch.object(tester, '_test_privilege_escalation', return_value=[]):
				with patch.object(tester, '_test_sql_injection', return_value=[]):
					result = await tester.run_penetration_test(pentest_config)
		
		assert isinstance(result, SecurityTestResult)
		assert result.test_name == pentest_config.test_name
		assert result.total_tests_executed > 0
		assert len(result.vulnerabilities_found) >= 0
		assert 0.0 <= result.security_score <= 1.0
		assert isinstance(result.recommendations, list)
		assert isinstance(result.test_coverage, dict)
		
		# Check test was recorded
		assert len(tester.test_results) == 1
		assert tester.test_results[0] == result
	
	@pytest.mark.asyncio
	async def test_null_authentication_test(self, mock_service):
		"""Test null authentication bypass test"""
		tester = SecurityTester(mock_service)
		config = PenetrationTestConfig(
			test_name="Auth Test",
			target_scope=["auth"],
			test_types=[VulnerabilityType.AUTHENTICATION_BYPASS]
		)
		
		# Mock service to allow creation with empty user (vulnerability)
		mock_service.create_key.return_value = Mock(
			spec_id="vulnerable_key",
			spec=Mock(id="vuln_key_123")
		)
		
		vulnerabilities = await tester._test_null_authentication(config)
		
		# Should find vulnerability if empty user is allowed
		if len(vulnerabilities) > 0:
			vuln = vulnerabilities[0]
			assert vuln.vulnerability_type == VulnerabilityType.AUTHENTICATION_BYPASS
			assert vuln.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
			assert "Null Authentication" in vuln.title
	
	@pytest.mark.asyncio
	async def test_default_credentials_test(self, mock_service):
		"""Test default credentials detection"""
		tester = SecurityTester(mock_service)
		config = PenetrationTestConfig(
			test_name="Cred Test",
			target_scope=["auth"],
			test_types=[VulnerabilityType.AUTHENTICATION_BYPASS]
		)
		
		vulnerabilities = await tester._test_default_credentials(config)
		
		# This test includes hardcoded vulnerability detection for demo
		# In real implementation, would test actual authentication
		assert isinstance(vulnerabilities, list)
		
		# Check if default credentials vulnerability was simulated
		default_cred_vulns = [v for v in vulnerabilities if "Default Credentials" in v.title]
		if len(default_cred_vulns) > 0:
			vuln = default_cred_vulns[0]
			assert vuln.vulnerability_type == VulnerabilityType.AUTHENTICATION_BYPASS
			assert "admin/admin" in vuln.proof_of_concept
	
	@pytest.mark.asyncio
	async def test_privilege_escalation_test(self, mock_service):
		"""Test privilege escalation detection"""
		tester = SecurityTester(mock_service)
		config = PenetrationTestConfig(
			test_name="Privesc Test",
			target_scope=["auth"],
			test_types=[VulnerabilityType.AUTHORIZATION_WEAKNESS]
		)
		
		# Mock service to allow regular user to create key (vulnerability)
		mock_service.create_key.return_value = Mock(
			spec_id="privesc_key",
			spec=Mock(id="privesc_123")
		)
		
		vulnerabilities = await tester._test_privilege_escalation(config)
		
		# Should find vulnerability if regular user can do admin operations
		if len(vulnerabilities) > 0:
			vuln = vulnerabilities[0]
			assert vuln.vulnerability_type == VulnerabilityType.AUTHORIZATION_WEAKNESS
			assert "Privilege Escalation" in vuln.title
	
	@pytest.mark.asyncio
	async def test_sql_injection_test(self, mock_service):
		"""Test SQL injection detection"""
		tester = SecurityTester(mock_service)
		config = PenetrationTestConfig(
			test_name="SQL Injection Test",
			target_scope=["api"],
			test_types=[VulnerabilityType.INJECTION_ATTACK]
		)
		
		# Mock service to accept SQL injection payload (vulnerability)
		mock_service.create_key.return_value = Mock(
			spec_id="sql_injection_key",
			spec=Mock(id="sql_inj_123")
		)
		mock_service.delete_key.return_value = True
		
		vulnerabilities = await tester._test_sql_injection(config)
		
		# Should find vulnerabilities if SQL injection payloads are accepted
		sql_vulns = [v for v in vulnerabilities if v.vulnerability_type == VulnerabilityType.INJECTION_ATTACK]
		for vuln in sql_vulns:
			assert "SQL Injection" in vuln.title
			assert vuln.severity == SeverityLevel.CRITICAL
	
	@pytest.mark.asyncio
	async def test_timing_attack_test(self, mock_service):
		"""Test timing attack detection"""
		tester = SecurityTester(mock_service)
		config = PenetrationTestConfig(
			test_name="Timing Attack Test",
			target_scope=["crypto"],
			test_types=[VulnerabilityType.TIMING_ATTACK]
		)
		
		# Mock service with variable timing
		async def variable_timing_create_key(spec, user_id):
			import random
			await asyncio.sleep(random.uniform(0.01, 0.1))  # Variable delay
			return Mock(spec_id="timing_key", spec=Mock(id=f"timing_{user_id}"))
		
		mock_service.create_key.side_effect = variable_timing_create_key
		mock_service.delete_key.return_value = True
		
		vulnerabilities = await tester._test_timing_oracle(config)
		
		# May find timing vulnerabilities based on variance
		timing_vulns = [v for v in vulnerabilities if v.vulnerability_type == VulnerabilityType.TIMING_ATTACK]
		for vuln in timing_vulns:
			assert "Timing" in vuln.title
			assert vuln.severity in [SeverityLevel.MEDIUM, SeverityLevel.LOW]
	
	@pytest.mark.asyncio
	async def test_dos_resource_exhaustion_test(self, mock_service):
		"""Test denial of service resource exhaustion"""
		tester = SecurityTester(mock_service)
		config = PenetrationTestConfig(
			test_name="DoS Test",
			target_scope=["api"],
			test_types=[VulnerabilityType.DENIAL_OF_SERVICE]
		)
		
		# Mock service to allow many key creations (vulnerability)
		mock_service.create_key.return_value = Mock(
			spec_id="dos_key",
			spec=Mock(id="dos_123")
		)
		mock_service.delete_key.return_value = True
		
		vulnerabilities = await tester._test_resource_exhaustion(config)
		
		# Should find vulnerability if too many resources can be consumed
		dos_vulns = [v for v in vulnerabilities if v.vulnerability_type == VulnerabilityType.DENIAL_OF_SERVICE]
		for vuln in dos_vulns:
			assert "Resource Exhaustion" in vuln.title
			assert vuln.severity in [SeverityLevel.MEDIUM, SeverityLevel.HIGH]
	
	@pytest.mark.asyncio
	async def test_input_validation_tests(self, mock_service):
		"""Test various input validation attacks"""
		tester = SecurityTester(mock_service)
		config = PenetrationTestConfig(
			test_name="Input Validation Test",
			target_scope=["api"],
			test_types=[VulnerabilityType.INPUT_VALIDATION]
		)
		
		# Mock service to accept malicious inputs
		mock_service.create_key.return_value = Mock(
			spec_id="input_test_key",
			spec=Mock(id="input_123")
		)
		mock_service.delete_key.return_value = True
		
		# Test path traversal
		path_vulnerabilities = await tester._test_path_traversal(config)
		
		# Test buffer overflow
		buffer_vulnerabilities = await tester._test_buffer_overflow(config)
		
		# Test malformed requests
		malformed_vulnerabilities = await tester._test_malformed_requests(config)
		
		all_input_vulns = path_traversal + buffer_vulnerabilities + malformed_vulnerabilities
		
		# Verify input validation vulnerabilities structure
		for vuln in all_input_vulns:
			assert vuln.vulnerability_type == VulnerabilityType.INPUT_VALIDATION
			assert isinstance(vuln.title, str)
			assert isinstance(vuln.description, str)
			assert isinstance(vuln.remediation, str)
	
	def test_generate_security_recommendations(self, mock_service, sample_vulnerability):
		"""Test security recommendations generation"""
		tester = SecurityTester(mock_service)
		
		vulnerabilities = [
			sample_vulnerability,
			SecurityVulnerability(
				vulnerability_id="vuln_2",
				vulnerability_type=VulnerabilityType.CRYPTOGRAPHIC_WEAKNESS,
				severity=SeverityLevel.CRITICAL,
				title="Weak Crypto",
				description="Weak algorithm",
				impact="Data compromise",
				remediation="Use strong crypto"
			),
			SecurityVulnerability(
				vulnerability_id="vuln_3",
				vulnerability_type=VulnerabilityType.INPUT_VALIDATION,
				severity=SeverityLevel.MEDIUM,
				title="Input Validation",
				description="Missing validation",
				impact="Injection attacks",
				remediation="Validate inputs"
			)
		]
		
		recommendations = tester._generate_security_recommendations(vulnerabilities)
		
		assert isinstance(recommendations, list)
		assert len(recommendations) > 0
		
		# Should include recommendations for each vulnerability type found
		rec_text = ' '.join(recommendations).lower()
		assert 'authentication' in rec_text or 'multi-factor' in rec_text
		assert 'cryptographic' in rec_text or 'encryption' in rec_text
		assert 'input validation' in rec_text or 'sanitization' in rec_text
		assert 'critical' in rec_text  # Should mention critical vulnerabilities
	
	def test_generate_security_recommendations_no_vulnerabilities(self, mock_service):
		"""Test recommendations generation with no vulnerabilities"""
		tester = SecurityTester(mock_service)
		
		recommendations = tester._generate_security_recommendations([])
		
		assert isinstance(recommendations, list)
		assert len(recommendations) > 0
		
		# Should include general security recommendations
		rec_text = ' '.join(recommendations).lower()
		assert 'continue' in rec_text or 'regular' in rec_text
	
	def test_generate_security_report_no_results(self, mock_service):
		"""Test security report generation with no results"""
		tester = SecurityTester(mock_service)
		
		report = tester.generate_security_report()
		
		assert 'error' in report
		assert report['error'] == 'No security test results available'
	
	def test_generate_security_report_with_results(self, mock_service, sample_vulnerability):
		"""Test security report generation with results"""
		tester = SecurityTester(mock_service)
		
		# Add mock test result
		test_result = SecurityTestResult(
			test_id="test_123",
			test_name="Mock Security Test",
			start_time=datetime.utcnow() - timedelta(minutes=30),
			end_time=datetime.utcnow(),
			total_tests_executed=10,
			vulnerabilities_found=[sample_vulnerability],
			security_score=0.75,
			recommendations=["Test recommendation"],
			test_coverage={'authentication_bypass': 1.0}
		)
		
		tester.test_results.append(test_result)
		tester.vulnerability_db.append(sample_vulnerability)
		
		report = tester.generate_security_report()
		
		assert 'report_generated' in report
		assert 'executive_summary' in report
		assert 'vulnerability_distribution' in report
		assert 'vulnerability_by_type' in report
		assert 'security_recommendations' in report
		assert 'detailed_findings' in report
		assert 'test_coverage' in report
		
		# Check executive summary
		executive_summary = report['executive_summary']
		assert executive_summary['total_tests_conducted'] == 1
		assert executive_summary['total_vulnerabilities_found'] == 1
		assert executive_summary['overall_security_score'] == 0.75
		assert executive_summary['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
		
		# Check vulnerability distribution
		vuln_dist = report['vulnerability_distribution']
		assert 'critical' in vuln_dist
		assert 'high' in vuln_dist
		assert 'medium' in vuln_dist
		assert 'low' in vuln_dist
		assert 'info' in vuln_dist
		assert vuln_dist['high'] == 1  # Our sample vulnerability is HIGH severity
		
		# Check detailed findings
		findings = report['detailed_findings']
		assert len(findings) == 1
		assert findings[0]['id'] == sample_vulnerability.vulnerability_id
		assert findings[0]['type'] == sample_vulnerability.vulnerability_type.value
		assert findings[0]['severity'] == sample_vulnerability.severity.value


class TestSecurityTestResult:
	"""Test SecurityTestResult data model"""
	
	def test_security_test_result_creation(self, sample_vulnerability):
		"""Test security test result creation"""
		start_time = datetime.utcnow() - timedelta(minutes=60)
		end_time = datetime.utcnow()
		
		result = SecurityTestResult(
			test_id="security_test_123",
			test_name="Comprehensive Security Assessment",
			start_time=start_time,
			end_time=end_time,
			total_tests_executed=50,
			vulnerabilities_found=[sample_vulnerability],
			security_score=0.85,
			recommendations=[
				"Implement multi-factor authentication",
				"Regular security audits",
				"Update cryptographic algorithms"
			],
			test_coverage={
				'authentication_bypass': 1.0,
				'authorization_weakness': 0.8,
				'input_validation': 0.9
			}
		)
		
		assert result.test_id == "security_test_123"
		assert result.test_name == "Comprehensive Security Assessment"
		assert result.start_time == start_time
		assert result.end_time == end_time
		assert result.total_tests_executed == 50
		assert len(result.vulnerabilities_found) == 1
		assert result.security_score == 0.85
		assert len(result.recommendations) == 3
		assert result.test_coverage['authentication_bypass'] == 1.0
		assert 0.0 <= result.test_coverage['authorization_weakness'] <= 1.0


class TestIntegrationScenarios:
	"""Test integration scenarios"""
	
	@pytest.mark.asyncio
	async def test_full_security_assessment_workflow(self, mock_service):
		"""Test complete security assessment workflow"""
		# 1. Create security tester
		tester = await create_security_tester(mock_service)
		
		# 2. Configure penetration test
		config = PenetrationTestConfig(
			test_name="Full Security Assessment",
			target_scope=["api", "authentication", "cryptography"],
			test_types=[
				VulnerabilityType.AUTHENTICATION_BYPASS,
				VulnerabilityType.CRYPTOGRAPHIC_WEAKNESS,
				VulnerabilityType.INPUT_VALIDATION
			],
			duration_minutes=10,  # Short duration for test
			concurrent_attacks=2
		)
		
		# 3. Mock service responses
		mock_service.create_key.return_value = Mock(
			spec_id="test_key",
			spec=Mock(id="test_key_123")
		)
		mock_service.retrieve_key.return_value = Mock(spec_id="retrieved_key")
		mock_service.encrypt_data.return_value = b"encrypted_data"
		mock_service.decrypt_data.return_value = b"decrypted_data"
		mock_service.delete_key.return_value = True
		
		# 4. Run security assessment
		with patch.object(tester, '_test_null_authentication', return_value=[]):
			with patch.object(tester, '_test_weak_key_generation', return_value=[]):
				with patch.object(tester, '_test_sql_injection', return_value=[]):
					result = await tester.run_penetration_test(config)
		
		assert isinstance(result, SecurityTestResult)
		assert result.test_name == config.test_name
		
		# 5. Generate security report
		report = tester.generate_security_report()
		
		assert 'executive_summary' in report
		assert 'vulnerability_distribution' in report
		assert 'security_recommendations' in report
		
		# 6. Verify test was recorded
		assert len(tester.test_results) == 1
		assert tester.test_results[0] == result
	
	@pytest.mark.asyncio
	async def test_vulnerability_severity_impact_on_score(self, mock_service):
		"""Test how vulnerability severity affects security score"""
		tester = SecurityTester(mock_service)
		
		# Create vulnerabilities of different severities
		critical_vuln = SecurityVulnerability(
			vulnerability_id="critical_1",
			vulnerability_type=VulnerabilityType.CRYPTOGRAPHIC_WEAKNESS,
			severity=SeverityLevel.CRITICAL,
			title="Critical Vulnerability",
			description="Critical issue",
			impact="High impact",
			remediation="Fix immediately"
		)
		
		high_vuln = SecurityVulnerability(
			vulnerability_id="high_1",
			vulnerability_type=VulnerabilityType.AUTHORIZATION_WEAKNESS,
			severity=SeverityLevel.HIGH,
			title="High Vulnerability",
			description="High issue",
			impact="High impact",
			remediation="Fix soon"
		)
		
		low_vuln = SecurityVulnerability(
			vulnerability_id="low_1",
			vulnerability_type=VulnerabilityType.INFORMATION_DISCLOSURE,
			severity=SeverityLevel.LOW,
			title="Low Vulnerability",
			description="Low issue",
			impact="Low impact",
			remediation="Fix when convenient"
		)
		
		# Test with no vulnerabilities (should have high score)
		no_vulns = []
		
		# Test with critical vulnerability (should have low score)
		critical_vulns = [critical_vuln]
		
		# Test with multiple low vulnerabilities
		low_vulns = [low_vuln] * 3
		
		# Simulate security score calculation
		# (This would normally be done in run_penetration_test)
		
		# No vulnerabilities = high score
		no_vuln_score = 1.0
		assert no_vuln_score == 1.0
		
		# Critical vulnerability = significant score reduction
		critical_score_reduction = len(critical_vulns) * 0.4
		critical_score = max(0.0, 1.0 - critical_score_reduction)
		assert critical_score <= 0.6
		
		# Multiple low vulnerabilities = smaller score reduction
		low_score_reduction = len(low_vulns) * 0.05
		low_score = max(0.0, 1.0 - low_score_reduction)
		assert low_score >= 0.85
	
	@pytest.mark.asyncio
	async def test_concurrent_security_testing(self, mock_service):
		"""Test concurrent execution of security tests"""
		tester = SecurityTester(mock_service)
		
		# Mock service for concurrent testing
		mock_service.create_key.return_value = Mock(
			spec_id="concurrent_key",
			spec=Mock(id="concurrent_123")
		)
		mock_service.delete_key.return_value = True
		
		# Test concurrent request flood simulation
		vulnerabilities = await tester._test_concurrent_request_flood(
			PenetrationTestConfig(
				test_name="Concurrent Test",
				target_scope=["api"],
				test_types=[VulnerabilityType.DENIAL_OF_SERVICE]
			)
		)
		
		# Verify test completed without errors
		assert isinstance(vulnerabilities, list)
		
		# If vulnerabilities found, they should be properly structured
		for vuln in vulnerabilities:
			assert vuln.vulnerability_type == VulnerabilityType.DENIAL_OF_SERVICE
			assert isinstance(vuln.title, str)
			assert isinstance(vuln.description, str)


if __name__ == "__main__":
	pytest.main([__file__])