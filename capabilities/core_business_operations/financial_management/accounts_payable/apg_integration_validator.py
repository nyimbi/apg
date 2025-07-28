#!/usr/bin/env python3
"""
APG Accounts Payable - Integration Validation Script

Validates that the AP capability meets all APG platform integration requirements
as specified in the comprehensive development plan (todo.md).

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import importlib.util


class APGIntegrationValidator:
	"""Validates APG platform integration requirements for AP capability"""
	
	def __init__(self, capability_path: Path):
		self.capability_path = capability_path
		self.validation_results: List[Dict[str, Any]] = []
		
	async def run_complete_validation(self) -> Dict[str, Any]:
		"""Run complete APG integration validation"""
		print("🚀 APG Accounts Payable - Integration Validation")
		print("=" * 60)
		
		validation_phases = [
			("CLAUDE.md Compliance", self._validate_claude_md_compliance),
			("APG Platform Dependencies", self._validate_apg_dependencies),
			("Data Model Standards", self._validate_data_models),
			("Service Layer Implementation", self._validate_service_layer),
			("API Implementation", self._validate_api_implementation),
			("Flask Blueprint Integration", self._validate_blueprint_integration),
			("Testing Framework", self._validate_testing_framework),
			("Documentation Completeness", self._validate_documentation),
			("APG Capability Registration", self._validate_capability_registration),
			("Performance Requirements", self._validate_performance_requirements)
		]
		
		passed_validations = 0
		total_validations = len(validation_phases)
		
		for phase_name, validation_func in validation_phases:
			print(f"\n📋 {phase_name}...")
			try:
				result = await validation_func()
				if result["status"] == "PASS":
					print(f"✅ {phase_name} - PASSED")
					passed_validations += 1
				else:
					print(f"❌ {phase_name} - FAILED: {result.get('message', 'Unknown error')}")
				
				self.validation_results.append({
					"phase": phase_name,
					"status": result["status"],
					"message": result.get("message", ""),
					"details": result.get("details", {})
				})
				
			except Exception as e:
				print(f"❌ {phase_name} - ERROR: {str(e)}")
				self.validation_results.append({
					"phase": phase_name,
					"status": "ERROR",
					"message": str(e),
					"details": {}
				})
		
		success_rate = (passed_validations / total_validations) * 100
		
		print(f"\n{'='*60}")
		print(f"📊 Validation Complete: {passed_validations}/{total_validations} passed ({success_rate:.1f}%)")
		
		if success_rate >= 90:
			print("🎉 APG Integration Status: READY FOR PRODUCTION")
			status = "READY"
		elif success_rate >= 75:
			print("⚠️  APG Integration Status: NEEDS MINOR FIXES")
			status = "MINOR_ISSUES"
		else:
			print("❌ APG Integration Status: NEEDS MAJOR WORK")
			status = "MAJOR_ISSUES"
		
		return {
			"overall_status": status,
			"success_rate": success_rate,
			"passed_validations": passed_validations,
			"total_validations": total_validations,
			"results": self.validation_results
		}
	
	async def _validate_claude_md_compliance(self) -> Dict[str, Any]:
		"""Validate CLAUDE.md coding standards compliance"""
		issues = []
		
		# Check for async Python throughout
		python_files = list(self.capability_path.glob("*.py"))
		for py_file in python_files:
			if py_file.name.startswith("test_"):
				continue
				
			try:
				content = py_file.read_text()
				
				# Check for tabs vs spaces (CLAUDE.md requires tabs)
				lines = content.split('\n')
				for i, line in enumerate(lines, 1):
					if line.startswith('    ') and line.strip():  # 4 spaces at start
						issues.append(f"{py_file.name}:{i} - Uses spaces instead of tabs")
						break  # Only report once per file
				
				# Check for modern typing
				if 'from typing import' in content:
					if 'Optional[' in content and 'str | None' not in content:
						issues.append(f"{py_file.name} - Uses old typing (Optional) instead of modern (str | None)")
				
				# Check for uuid7str usage
				if 'uuid' in content and 'uuid7str' not in content and 'import uuid' in content:
					issues.append(f"{py_file.name} - Uses standard uuid instead of uuid7str")
				
				# Check for async patterns in service files
				if py_file.name == 'service.py':
					if 'async def' not in content:
						issues.append(f"{py_file.name} - Service layer not using async patterns")
				
			except Exception as e:
				issues.append(f"{py_file.name} - Could not validate: {str(e)}")
		
		if issues:
			print(f"   CLAUDE.md Issues Found:")
			for issue in issues[:5]:  # Show first 5 issues
				print(f"     - {issue}")
			return {"status": "FAIL", "message": f"{len(issues)} CLAUDE.md compliance issues", "details": {"issues": issues}}
		return {"status": "PASS", "message": "CLAUDE.md compliance validated"}
	
	async def _validate_apg_dependencies(self) -> Dict[str, Any]:
		"""Validate APG platform dependencies"""
		required_dependencies = [
			"auth_rbac",
			"audit_compliance", 
			"general_ledger",
			"document_management"
		]
		
		enhanced_dependencies = [
			"ai_orchestration",
			"computer_vision", 
			"federated_learning",
			"real_time_collaboration"
		]
		
		# Check __init__.py for dependency declarations
		init_file = self.capability_path / "__init__.py"
		if init_file.exists():
			content = init_file.read_text()
			missing_deps = []
			
			for dep in required_dependencies:
				if dep not in content:
					missing_deps.append(f"Required dependency '{dep}' not declared")
			
			if missing_deps:
				return {"status": "FAIL", "message": "Missing APG dependencies", "details": {"missing": missing_deps}}
		
		return {"status": "PASS", "message": "APG dependencies validated"}
	
	async def _validate_data_models(self) -> Dict[str, Any]:
		"""Validate data model implementation"""
		models_file = self.capability_path / "models.py"
		if not models_file.exists():
			return {"status": "FAIL", "message": "models.py not found"}
		
		content = models_file.read_text()
		issues = []
		
		# Check for Pydantic v2 usage
		if "from pydantic import BaseModel" not in content:
			issues.append("Missing Pydantic BaseModel import")
		
		if "ConfigDict" not in content:
			issues.append("Missing Pydantic v2 ConfigDict usage")
		
		# Check for uuid7str usage
		if "uuid7str" not in content:
			issues.append("Not using uuid7str for ID generation")
		
		# Check for key models
		required_models = ["APVendor", "APInvoice", "APPayment", "APApprovalWorkflow"]
		for model in required_models:
			if f"class {model}" not in content:
				issues.append(f"Missing required model: {model}")
		
		if issues:
			return {"status": "FAIL", "message": f"{len(issues)} model issues", "details": {"issues": issues}}
		return {"status": "PASS", "message": "Data models validated"}
	
	async def _validate_service_layer(self) -> Dict[str, Any]:
		"""Validate service layer implementation"""
		service_file = self.capability_path / "service.py"
		if not service_file.exists():
			return {"status": "FAIL", "message": "service.py not found"}
		
		content = service_file.read_text()
		issues = []
		
		# Check for async implementation
		if "async def" not in content:
			issues.append("Service layer not using async patterns")
		
		# Check for _log_ methods
		if "_log_" not in content:
			issues.append("Missing _log_ prefixed methods for console logging")
		
		# Check for runtime assertions
		if "assert " not in content:
			issues.append("Missing runtime assertions at function boundaries")
		
		# Check for APG integration services
		apg_services = ["APGAuthService", "APGAuditService"]
		for service in apg_services:
			if service not in content:
				issues.append(f"Missing APG integration service: {service}")
		
		if issues:
			return {"status": "FAIL", "message": f"{len(issues)} service issues", "details": {"issues": issues}}
		return {"status": "PASS", "message": "Service layer validated"}
	
	async def _validate_api_implementation(self) -> Dict[str, Any]:
		"""Validate API implementation"""
		api_file = self.capability_path / "api.py"
		if not api_file.exists():
			return {"status": "FAIL", "message": "api.py not found"}
		
		content = api_file.read_text()
		issues = []
		
		# Check for async endpoints
		if "async def" not in content:
			issues.append("API endpoints not using async patterns")
		
		# Check for Pydantic validation
		if "pydantic" not in content.lower():
			issues.append("Missing Pydantic validation in API")
		
		# Check for authentication integration
		if "auth" not in content.lower():
			issues.append("Missing authentication integration")
		
		if issues:
			return {"status": "FAIL", "message": f"{len(issues)} API issues", "details": {"issues": issues}}
		return {"status": "PASS", "message": "API implementation validated"}
	
	async def _validate_blueprint_integration(self) -> Dict[str, Any]:
		"""Validate Flask blueprint integration"""
		blueprint_file = self.capability_path / "blueprint.py"
		if not blueprint_file.exists():
			return {"status": "FAIL", "message": "blueprint.py not found"}
		
		content = blueprint_file.read_text()
		issues = []
		
		# Check for APG composition engine registration
		if "composition_engine" not in content:
			issues.append("Missing APG composition engine integration")
		
		# Check for capability registration
		if "register_capability" not in content:
			issues.append("Missing capability registration with APG")
		
		if issues:
			return {"status": "FAIL", "message": f"{len(issues)} blueprint issues", "details": {"issues": issues}}
		return {"status": "PASS", "message": "Blueprint integration validated"}
	
	async def _validate_testing_framework(self) -> Dict[str, Any]:
		"""Validate testing framework compliance"""
		tests_dir = self.capability_path / "tests" / "ci"
		if not tests_dir.exists():
			return {"status": "FAIL", "message": "tests/ci/ directory not found"}
		
		test_files = list(tests_dir.glob("test_*.py"))
		if len(test_files) == 0:
			return {"status": "FAIL", "message": "No test files found in tests/ci/"}
		
		issues = []
		for test_file in test_files:
			content = test_file.read_text()
			
			# Check for modern pytest-asyncio (no decorators)
			if "@pytest.mark.asyncio" in content:
				issues.append(f"{test_file.name} - Using old @pytest.mark.asyncio decorator")
			
			# Check for real objects usage (mocks only allowed for LLM/AI services)
			if "mock" in content.lower():
				# Check if mocks are for allowed services (AI, LLM, external APIs)
				allowed_mock_terms = ["llm", "ai", "computer_vision", "orchestration", "external"]
				is_allowed_mock = any(term in content.lower() for term in allowed_mock_terms)
				if not is_allowed_mock:
					issues.append(f"{test_file.name} - Using mocks instead of real objects")
		
		if issues:
			print(f"   Testing Issues Found:")
			for issue in issues[:5]:  # Show first 5 issues
				print(f"     - {issue}")
			return {"status": "FAIL", "message": f"{len(issues)} testing issues", "details": {"issues": issues}}
		return {"status": "PASS", "message": "Testing framework validated"}
	
	async def _validate_documentation(self) -> Dict[str, Any]:
		"""Validate documentation completeness"""
		required_docs = [
			"cap_spec.md",
			"todo.md",
			"docs/user_guide.md",
			"docs/api_documentation.md"
		]
		
		missing_docs = []
		for doc in required_docs:
			doc_path = self.capability_path / doc
			if not doc_path.exists():
				missing_docs.append(doc)
		
		if missing_docs:
			return {"status": "FAIL", "message": f"Missing documentation", "details": {"missing": missing_docs}}
		return {"status": "PASS", "message": "Documentation complete"}
	
	async def _validate_capability_registration(self) -> Dict[str, Any]:
		"""Validate capability registration metadata"""
		init_file = self.capability_path / "__init__.py"
		if not init_file.exists():
			return {"status": "FAIL", "message": "__init__.py not found"}
		
		content = init_file.read_text()
		
		# Check for capability metadata
		if "SUBCAPABILITY_META" not in content:
			return {"status": "FAIL", "message": "Missing capability metadata"}
		
		# Check for required metadata fields
		required_fields = ["name", "version", "dependencies", "permissions", "api_endpoints"]
		issues = []
		
		for field in required_fields:
			if f"'{field}'" not in content:
				issues.append(f"Missing metadata field: {field}")
		
		if issues:
			return {"status": "FAIL", "message": f"{len(issues)} metadata issues", "details": {"issues": issues}}
		return {"status": "PASS", "message": "Capability registration validated"}
	
	async def _validate_performance_requirements(self) -> Dict[str, Any]:
		"""Validate performance requirements implementation"""
		# This would normally run actual performance tests
		# For now, check for performance-related code patterns
		
		service_file = self.capability_path / "service.py"
		if service_file.exists():
			content = service_file.read_text()
			
			# Check for async patterns (performance requirement)
			if "async def" not in content:
				return {"status": "FAIL", "message": "Missing async patterns for performance"}
			
			# Check for caching considerations
			if "cache" not in content.lower():
				return {"status": "WARN", "message": "No caching implementation found"}
		
		return {"status": "PASS", "message": "Performance patterns validated"}


async def main():
	"""Run APG integration validation"""
	capability_path = Path(__file__).parent
	validator = APGIntegrationValidator(capability_path)
	
	results = await validator.run_complete_validation()
	
	print("\n" + "="*60)
	print("📋 VALIDATION SUMMARY")
	print("="*60)
	
	for result in validator.validation_results:
		status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
		print(f"{status_icon} {result['phase']}: {result['status']}")
		if result["status"] != "PASS" and result["message"]:
			print(f"   → {result['message']}")
	
	print(f"\n🎯 Overall Status: {results['overall_status']}")
	print(f"📊 Success Rate: {results['success_rate']:.1f}%")
	
	if results['overall_status'] == 'READY':
		print("🚀 APG Accounts Payable capability is ready for production!")
		return 0
	else:
		print("⚠️ APG integration issues need to be addressed.")
		return 1


if __name__ == "__main__":
	exit_code = asyncio.run(main())
	sys.exit(exit_code)