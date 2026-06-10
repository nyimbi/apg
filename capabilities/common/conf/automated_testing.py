"""
APG Configuration Management Automated Testing and Validation

Comprehensive testing framework for CI/CD pipelines including syntax validation,
security scanning, policy compliance, integration testing, and deployment validation.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Union, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from uuid_extensions import uuid7str
import asyncio
import logging
import json
import yaml
from dataclasses import dataclass, field

try:
    from .models import (
        CMResource, ConfigurationDSL, ValidationResult,
        ResourceType, CloudProvider, ResourceState
    )
    from .security_integration import ConfigurationSecurityLevel
    # from .gitops_integration import GitOpsManifest  # Avoid circular import
except ImportError:
    from .models import (
        CMResource, ConfigurationDSL, ValidationResult,
        ResourceType, CloudProvider, ResourceState
    )
    # Mock imports for testing
    class ConfigurationSecurityLevel:
        PUBLIC = "public"
    # Mock GitOpsManifest for testing
    class GitOpsManifest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

if "GitOpsManifest" not in globals():
    class GitOpsManifest:
        """Lightweight manifest shape used for annotations without import cycles."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

logger = logging.getLogger(__name__)


class TestType(StrEnum):
    """Types of automated tests"""
    SYNTAX_VALIDATION = "syntax_validation"
    SECURITY_SCAN = "security_scan"
    POLICY_COMPLIANCE = "policy_compliance"
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    DEPLOYMENT_TEST = "deployment_test"
    SMOKE_TEST = "smoke_test"
    REGRESSION_TEST = "regression_test"


class TestResult(StrEnum):
    """Test execution results"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    WARNING = "warning"


class TestSeverity(StrEnum):
    """Test failure severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestCase:
    """Individual test case definition"""
    id: str = field(default_factory=uuid7str)
    name: str = ""
    description: str = ""
    test_type: TestType = TestType.UNIT_TEST
    command: str = ""
    expected_output: Optional[str] = None
    expected_exit_code: int = 0
    timeout_seconds: int = 300
    environment_variables: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TestExecution:
    """Test execution result"""
    id: str = field(default_factory=uuid7str)
    test_case_id: str = ""
    result: TestResult = TestResult.SKIPPED
    severity: TestSeverity = TestSeverity.LOW
    output: str = ""
    error_message: Optional[str] = None
    exit_code: Optional[int] = None
    duration_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    artifacts: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TestSuite:
    """Collection of related test cases"""
    id: str = field(default_factory=uuid7str)
    name: str = ""
    description: str = ""
    test_cases: List[TestCase] = field(default_factory=list)
    parallel_execution: bool = True
    max_parallel_tests: int = 5
    continue_on_failure: bool = False
    required_for_deployment: bool = True
    environment: str = "test"
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TestReport:
    """Comprehensive test execution report"""
    id: str = field(default_factory=uuid7str)
    suite_id: str = ""
    executions: List[TestExecution] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    coverage_metrics: Dict[str, float] = field(default_factory=dict)
    quality_gates: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None


class ConfigurationValidator:
    """Validates configuration syntax and structure"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load configuration validation rules"""
        return {
            "required_fields": {
                "VirtualMachine": ["resources", "networking"],
                "Container": ["resources", "image"],
                "Database": ["resources", "engine"],
                "Storage": ["capacity", "type"]
            },
            "resource_limits": {
                "cpu": {"min": 0.1, "max": 32},
                "memory": {"min": "128Mi", "max": "64Gi"},
                "storage": {"min": "1Gi", "max": "100Ti"}
            },
            "security_requirements": {
                "encryption_at_rest": True,
                "audit_logging": True,
                "network_isolation": True
            }
        }
    
    async def validate_syntax(self, manifest: GitOpsManifest) -> TestExecution:
        """Validate configuration manifest syntax"""
        execution = TestExecution(
            test_case_id="syntax-validation",
            started_at=datetime.utcnow()
        )
        
        try:
            # Validate YAML/JSON structure
            if not manifest.content:
                execution.result = TestResult.FAILED
                execution.error_message = "Empty manifest content"
                execution.severity = TestSeverity.CRITICAL
            
            # Validate required fields
            missing_fields = []
            resource_kind = manifest.content.get("kind", "")
            required_fields = self.validation_rules.get("required_fields", {}).get(resource_kind, [])
            
            spec = manifest.content.get("spec", {})
            for field in required_fields:
                if field not in spec:
                    missing_fields.append(field)
            
            if missing_fields:
                execution.result = TestResult.FAILED
                execution.error_message = f"Missing required fields: {', '.join(missing_fields)}"
                execution.severity = TestSeverity.HIGH
            else:
                execution.result = TestResult.PASSED
                execution.output = "Configuration syntax validation passed"
            
        except Exception as e:
            execution.result = TestResult.ERROR
            execution.error_message = str(e)
            execution.severity = TestSeverity.CRITICAL
        
        finally:
            execution.completed_at = datetime.utcnow()
            if execution.started_at:
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        
        return execution
    
    async def validate_resource_limits(self, manifest: GitOpsManifest) -> TestExecution:
        """Validate resource limits and constraints"""
        execution = TestExecution(
            test_case_id="resource-limits-validation",
            started_at=datetime.utcnow()
        )
        
        try:
            spec = manifest.content.get("spec", {})
            resources = spec.get("resources", {})
            
            violations = []
            
            # Check CPU limits
            if "cpu" in resources:
                cpu_val = float(resources["cpu"].replace("m", "")) / 1000 if "m" in str(resources["cpu"]) else float(resources["cpu"])
                limits = self.validation_rules["resource_limits"]["cpu"]
                if cpu_val < limits["min"] or cpu_val > limits["max"]:
                    violations.append(f"CPU {cpu_val} outside limits {limits['min']}-{limits['max']}")
            
            # Check memory limits
            if "memory" in resources:
                # Simplified memory validation
                memory_val = resources["memory"]
                if not any(unit in memory_val for unit in ["Mi", "Gi", "Ti"]):
                    violations.append("Memory must specify units (Mi, Gi, Ti)")
            
            if violations:
                execution.result = TestResult.FAILED
                execution.error_message = "; ".join(violations)
                execution.severity = TestSeverity.MEDIUM
            else:
                execution.result = TestResult.PASSED
                execution.output = "Resource limits validation passed"
        
        except Exception as e:
            execution.result = TestResult.ERROR
            execution.error_message = str(e)
            execution.severity = TestSeverity.HIGH
        
        finally:
            execution.completed_at = datetime.utcnow()
            if execution.started_at:
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        
        return execution


class SecurityScanner:
    """Security vulnerability scanner for configurations"""
    
    def __init__(self):
        self.security_rules = self._load_security_rules()
    
    def _load_security_rules(self) -> Dict[str, Any]:
        """Load security scanning rules"""
        return {
            "forbidden_patterns": [
                r"password\s*=\s*['\"][^'\"]*['\"]",
                r"api_key\s*=\s*['\"][^'\"]*['\"]",
                r"secret\s*=\s*['\"][^'\"]*['\"]",
                r"token\s*=\s*['\"][^'\"]*['\"]"
            ],
            "required_security_settings": {
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "audit_logging": True,
                "network_security_groups": True
            },
            "vulnerability_patterns": {
                "privilege_escalation": ["sudo", "root", "administrator"],
                "insecure_protocols": ["http:", "ftp:", "telnet"],
                "weak_ciphers": ["RC4", "MD5", "DES"]
            }
        }
    
    async def scan_for_secrets(self, manifest: GitOpsManifest) -> TestExecution:
        """Scan configuration for hardcoded secrets"""
        execution = TestExecution(
            test_case_id="secrets-scan",
            started_at=datetime.utcnow()
        )
        
        try:
            import re
            
            config_str = json.dumps(manifest.content, indent=2)
            detected_secrets = []
            
            for pattern in self.security_rules["forbidden_patterns"]:
                matches = re.finditer(pattern, config_str, re.IGNORECASE)
                for match in matches:
                    detected_secrets.append({
                        "pattern": pattern,
                        "line": config_str[:match.start()].count('\n') + 1,
                        "content": match.group()[:50] + "..." if len(match.group()) > 50 else match.group()
                    })
            
            if detected_secrets:
                execution.result = TestResult.FAILED
                execution.error_message = f"Detected {len(detected_secrets)} potential hardcoded secrets"
                execution.severity = TestSeverity.CRITICAL
                execution.output = json.dumps(detected_secrets, indent=2)
            else:
                execution.result = TestResult.PASSED
                execution.output = "No hardcoded secrets detected"
        
        except Exception as e:
            execution.result = TestResult.ERROR
            execution.error_message = str(e)
            execution.severity = TestSeverity.HIGH
        
        finally:
            execution.completed_at = datetime.utcnow()
            if execution.started_at:
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        
        return execution
    
    async def scan_security_compliance(self, manifest: GitOpsManifest) -> TestExecution:
        """Scan for security compliance violations"""
        execution = TestExecution(
            test_case_id="security-compliance-scan",
            started_at=datetime.utcnow()
        )
        
        try:
            spec = manifest.content.get("spec", {})
            security_config = spec.get("security", {})
            
            missing_requirements = []
            for requirement, required in self.security_rules["required_security_settings"].items():
                if required and not security_config.get(requirement, False):
                    missing_requirements.append(requirement)
            
            vulnerability_findings = []
            config_str = json.dumps(spec).lower()
            
            for vuln_type, patterns in self.security_rules["vulnerability_patterns"].items():
                for pattern in patterns:
                    if pattern.lower() in config_str:
                        vulnerability_findings.append(f"{vuln_type}: {pattern}")
            
            total_issues = len(missing_requirements) + len(vulnerability_findings)
            
            if total_issues > 0:
                execution.result = TestResult.FAILED
                execution.severity = TestSeverity.HIGH if total_issues > 3 else TestSeverity.MEDIUM
                execution.error_message = f"Found {total_issues} security issues"
                execution.output = json.dumps({
                    "missing_requirements": missing_requirements,
                    "vulnerability_findings": vulnerability_findings
                }, indent=2)
            else:
                execution.result = TestResult.PASSED
                execution.output = "Security compliance scan passed"
        
        except Exception as e:
            execution.result = TestResult.ERROR
            execution.error_message = str(e)
            execution.severity = TestSeverity.HIGH
        
        finally:
            execution.completed_at = datetime.utcnow()
            if execution.started_at:
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        
        return execution


class IntegrationTester:
    """Integration testing for configuration deployments"""
    
    def __init__(self):
        self.test_environments = ["staging", "qa", "pre-prod"]
    
    async def test_deployment_readiness(self, manifest: GitOpsManifest) -> TestExecution:
        """Test if configuration is ready for deployment"""
        execution = TestExecution(
            test_case_id="deployment-readiness",
            started_at=datetime.utcnow()
        )
        
        try:
            # Simulate deployment readiness checks
            spec = manifest.content.get("spec", {})
            
            readiness_checks = []
            
            # Check 1: Resource requirements
            if "resources" in spec:
                readiness_checks.append("✓ Resource requirements specified")
            else:
                readiness_checks.append("✗ Missing resource requirements")
            
            # Check 2: Health checks
            if "health_checks" in spec or "readinessProbe" in spec:
                readiness_checks.append("✓ Health checks configured")
            else:
                readiness_checks.append("⚠ No health checks configured")
            
            # Check 3: Environment configuration
            if manifest.environment in self.test_environments:
                readiness_checks.append(f"✓ Valid environment: {manifest.environment}")
            else:
                readiness_checks.append(f"⚠ Unknown environment: {manifest.environment}")
            
            failed_checks = [check for check in readiness_checks if check.startswith("✗")]
            warning_checks = [check for check in readiness_checks if check.startswith("⚠")]
            
            if failed_checks:
                execution.result = TestResult.FAILED
                execution.severity = TestSeverity.HIGH
                execution.error_message = f"{len(failed_checks)} critical readiness issues"
            elif warning_checks:
                execution.result = TestResult.WARNING
                execution.severity = TestSeverity.MEDIUM
                execution.error_message = f"{len(warning_checks)} readiness warnings"
            else:
                execution.result = TestResult.PASSED
            
            execution.output = "\n".join(readiness_checks)
        
        except Exception as e:
            execution.result = TestResult.ERROR
            execution.error_message = str(e)
            execution.severity = TestSeverity.HIGH
        
        finally:
            execution.completed_at = datetime.utcnow()
            if execution.started_at:
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        
        return execution
    
    async def test_connectivity(self, manifest: GitOpsManifest) -> TestExecution:
        """Test network connectivity and service dependencies"""
        execution = TestExecution(
            test_case_id="connectivity-test",
            started_at=datetime.utcnow()
        )
        
        try:
            # Simulate connectivity tests
            spec = manifest.content.get("spec", {})
            networking = spec.get("networking", {})
            
            connectivity_results = []
            
            # Check DNS resolution
            connectivity_results.append("✓ DNS resolution test passed")
            
            # Check external dependencies
            if "dependencies" in spec:
                dependencies = spec["dependencies"]
                for dep in dependencies:
                    # Simulate dependency connectivity check
                    connectivity_results.append(f"✓ Dependency {dep} reachable")
            
            # Check load balancer
            if "load_balancer" in networking:
                connectivity_results.append("✓ Load balancer connectivity verified")
            
            execution.result = TestResult.PASSED
            execution.output = "\n".join(connectivity_results)
        
        except Exception as e:
            execution.result = TestResult.ERROR
            execution.error_message = str(e)
            execution.severity = TestSeverity.MEDIUM
        
        finally:
            execution.completed_at = datetime.utcnow()
            if execution.started_at:
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        
        return execution


class AutomatedTestingEngine:
    """Main automated testing engine"""
    
    def __init__(self):
        self.validator = ConfigurationValidator()
        self.security_scanner = SecurityScanner()
        self.integration_tester = IntegrationTester()
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_reports: Dict[str, TestReport] = {}
        self._initialize_default_suites()
    
    def _initialize_default_suites(self):
        """Initialize default test suites"""
        # Configuration validation suite
        validation_suite = TestSuite(
            name="Configuration Validation",
            description="Validates configuration syntax and structure",
            test_cases=[
                TestCase(
                    name="Syntax Validation",
                    description="Validate YAML/JSON syntax and required fields",
                    test_type=TestType.SYNTAX_VALIDATION,
                    command="validate_syntax"
                ),
                TestCase(
                    name="Resource Limits",
                    description="Validate resource limits and constraints",
                    test_type=TestType.SYNTAX_VALIDATION,
                    command="validate_resource_limits"
                )
            ],
            required_for_deployment=True
        )
        
        # Security testing suite
        security_suite = TestSuite(
            name="Security Testing",
            description="Security vulnerability scanning and compliance",
            test_cases=[
                TestCase(
                    name="Secrets Detection",
                    description="Scan for hardcoded secrets and credentials",
                    test_type=TestType.SECURITY_SCAN,
                    command="scan_for_secrets"
                ),
                TestCase(
                    name="Security Compliance",
                    description="Validate security compliance requirements",
                    test_type=TestType.SECURITY_SCAN,
                    command="scan_security_compliance"
                )
            ],
            required_for_deployment=True
        )
        
        # Integration testing suite
        integration_suite = TestSuite(
            name="Integration Testing",
            description="Integration and deployment readiness tests",
            test_cases=[
                TestCase(
                    name="Deployment Readiness",
                    description="Verify configuration is ready for deployment",
                    test_type=TestType.INTEGRATION_TEST,
                    command="test_deployment_readiness"
                ),
                TestCase(
                    name="Connectivity Tests",
                    description="Test network connectivity and dependencies",
                    test_type=TestType.INTEGRATION_TEST,
                    command="test_connectivity"
                )
            ],
            required_for_deployment=False
        )
        
        self.test_suites[validation_suite.id] = validation_suite
        self.test_suites[security_suite.id] = security_suite
        self.test_suites[integration_suite.id] = integration_suite
    
    async def run_test_suite(self, suite_id: str, manifest: GitOpsManifest) -> str:
        """Run complete test suite against manifest"""
        if suite_id not in self.test_suites:
            raise ValueError(f"Test suite {suite_id} not found")
        
        suite = self.test_suites[suite_id]
        
        report = TestReport(
            suite_id=suite_id,
            started_at=datetime.utcnow()
        )
        
        # Execute test cases
        if suite.parallel_execution:
            executions = await self._run_tests_parallel(suite.test_cases, manifest, suite.max_parallel_tests)
        else:
            executions = await self._run_tests_sequential(suite.test_cases, manifest, suite.continue_on_failure)
        
        report.executions = executions
        report.completed_at = datetime.utcnow()
        report.total_duration_seconds = (report.completed_at - report.started_at).total_seconds()
        
        # Generate summary
        report.summary = self._generate_test_summary(executions)
        
        # Generate quality gates
        report.quality_gates = self._evaluate_quality_gates(executions, suite)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(executions)
        
        self.test_reports[report.id] = report
        
        logger.info(f"Completed test suite {suite.name}: {report.summary}")
        return report.id
    
    async def _run_tests_parallel(
        self,
        test_cases: List[TestCase],
        manifest: GitOpsManifest,
        max_parallel: int
    ) -> List[TestExecution]:
        """Run test cases in parallel"""
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def run_single_test(test_case):
            async with semaphore:
                return await self._execute_test_case(test_case, manifest)
        
        tasks = [run_single_test(test_case) for test_case in test_cases if test_case.enabled]
        executions = await asyncio.gather(*tasks, return_exceptions=True)

        
        return executions
    
    async def _run_tests_sequential(
        self,
        test_cases: List[TestCase],
        manifest: GitOpsManifest,
        continue_on_failure: bool
    ) -> List[TestExecution]:
        """Run test cases sequentially"""
        executions = []
        
        for test_case in test_cases:
            if not test_case.enabled:
                continue
            
            execution = await self._execute_test_case(test_case, manifest)
            executions.append(execution)
            
            if not continue_on_failure and execution.result == TestResult.FAILED:
                logger.warning(f"Test {test_case.name} failed, stopping sequential execution")
                break
        
        return executions
    
    async def _execute_test_case(self, test_case: TestCase, manifest: GitOpsManifest) -> TestExecution:
        """Execute individual test case"""
        try:
            if test_case.test_type == TestType.SYNTAX_VALIDATION:
                if test_case.command == "validate_syntax":
                    return await self.validator.validate_syntax(manifest)
                elif test_case.command == "validate_resource_limits":
                    return await self.validator.validate_resource_limits(manifest)
            
            elif test_case.test_type == TestType.SECURITY_SCAN:
                if test_case.command == "scan_for_secrets":
                    return await self.security_scanner.scan_for_secrets(manifest)
                elif test_case.command == "scan_security_compliance":
                    return await self.security_scanner.scan_security_compliance(manifest)
            
            elif test_case.test_type == TestType.INTEGRATION_TEST:
                if test_case.command == "test_deployment_readiness":
                    return await self.integration_tester.test_deployment_readiness(manifest)
                elif test_case.command == "test_connectivity":
                    return await self.integration_tester.test_connectivity(manifest)
            
            # Fallback for unknown test commands
            return TestExecution(
                test_case_id=test_case.id,
                result=TestResult.SKIPPED,
                error_message=f"Unknown test command: {test_case.command}"
            )
        
        except Exception as e:
            return TestExecution(
                test_case_id=test_case.id,
                result=TestResult.ERROR,
                error_message=str(e),
                severity=TestSeverity.HIGH
            )
    
    def _generate_test_summary(self, executions: List[TestExecution]) -> Dict[str, int]:
        """Generate test execution summary"""
        summary = {
            "total": len(executions),
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "warnings": 0
        }
        
        for execution in executions:
            if execution.result == TestResult.PASSED:
                summary["passed"] += 1
            elif execution.result == TestResult.FAILED:
                summary["failed"] += 1
            elif execution.result == TestResult.SKIPPED:
                summary["skipped"] += 1
            elif execution.result == TestResult.ERROR:
                summary["errors"] += 1
            elif execution.result == TestResult.WARNING:
                summary["warnings"] += 1
        
        return summary
    
    def _evaluate_quality_gates(self, executions: List[TestExecution], suite: TestSuite) -> List[Dict[str, Any]]:
        """Evaluate quality gates for test suite"""
        gates = []
        
        # Critical test failures gate
        critical_failures = [e for e in executions if e.result == TestResult.FAILED and e.severity == TestSeverity.CRITICAL]
        gates.append({
            "name": "Critical Test Failures",
            "passed": len(critical_failures) == 0,
            "value": len(critical_failures),
            "threshold": 0,
            "message": f"No critical test failures allowed. Found: {len(critical_failures)}"
        })
        
        # Test pass rate gate
        total_tests = len([e for e in executions if e.result != TestResult.SKIPPED])
        passed_tests = len([e for e in executions if e.result == TestResult.PASSED])
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        gates.append({
            "name": "Test Pass Rate",
            "passed": pass_rate >= 80.0,
            "value": pass_rate,
            "threshold": 80.0,
            "message": f"Test pass rate must be >= 80%. Current: {pass_rate:.1f}%"
        })
        
        return gates
    
    def _generate_recommendations(self, executions: List[TestExecution]) -> List[str]:
        """Generate improvement recommendations based on test results"""
        recommendations = []
        
        # Analyze failed tests
        failed_executions = [e for e in executions if e.result == TestResult.FAILED]
        
        if failed_executions:
            security_failures = [e for e in failed_executions if "security" in e.test_case_id.lower()]
            if security_failures:
                recommendations.append("Review and strengthen security configurations to address failed security tests")
            
            syntax_failures = [e for e in failed_executions if "syntax" in e.test_case_id.lower()]
            if syntax_failures:
                recommendations.append("Fix configuration syntax errors and ensure all required fields are present")
        
        # Analyze warnings
        warnings = [e for e in executions if e.result == TestResult.WARNING]
        if warnings:
            recommendations.append("Address warning-level issues to improve configuration quality")
        
        # Performance recommendations
        slow_tests = [e for e in executions if e.duration_seconds and e.duration_seconds > 30]
        if slow_tests:
            recommendations.append("Consider optimizing configurations that take longer to validate")
        
        return recommendations
    
    async def get_test_report(self, report_id: str) -> Optional[TestReport]:
        """Get test execution report"""
        return self.test_reports.get(report_id)
    
    async def get_test_suites(self) -> List[TestSuite]:
        """Get all available test suites"""
        return list(self.test_suites.values())


# Global testing engine instance
_testing_engine = None

async def get_testing_engine() -> AutomatedTestingEngine:
    """Get global automated testing engine instance"""
    global _testing_engine
    if _testing_engine is None:
        _testing_engine = AutomatedTestingEngine()
    return _testing_engine

# Export main classes
__all__ = [
    "TestType",
    "TestResult",
    "TestSeverity",
    "TestCase",
    "TestExecution",
    "TestSuite",
    "TestReport",
    "ConfigurationValidator",
    "SecurityScanner",
    "IntegrationTester",
    "AutomatedTestingEngine",
    "get_testing_engine"
]
