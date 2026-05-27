"""Runtime tenant-context regressions for ENCR support modules."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENCR_PATH = REPO_ROOT / "capabilities" / "common" / "encr"
SERVICE_PATH = ENCR_PATH / "service.py"


def test_encr_global_managers_use_runtime_tenant_context():
	expectations = {
		"quality_assurance.py": [
			"security_audit_engine = SecurityAuditEngine(get_tenant_id_from_context())",
			"compliance_certification_manager = ComplianceCertificationManager(get_tenant_id_from_context())",
			"quality_metrics_engine = QualityMetricsEngine(get_tenant_id_from_context())",
		],
		"mobile_apps.py": [
			"mobile_app_manager = MobileAppManager(get_tenant_id_from_context())",
		],
		"production_features.py": [
			"backup_recovery_system = BackupRecoverySystem(get_tenant_id_from_context())",
		],
		"developer_tools.py": [
			"developer_tools_manager = DeveloperToolsManager(get_tenant_id_from_context())",
		],
	}

	for relative_path, expected_lines in expectations.items():
		source = (ENCR_PATH / relative_path).read_text(encoding="utf-8")
		assert '("default_tenant")' not in source
		assert "from ..request_context import get_tenant_id_from_context" in source
		for expected_line in expected_lines:
			assert expected_line in source


def test_encr_core_service_uses_runtime_context_for_sessions_and_proofs():
	source = SERVICE_PATH.read_text(encoding="utf-8")

	assert "user_id='mock_user'" not in source
	assert "device_id='mock_device'" not in source
	assert "tenant_id='mock_tenant'" not in source
	assert "For now, create a mock session" not in source
	assert "session = await self._get_quantum_safe_session(session_id, tenant_id, user_context)" in source
	assert "user_id=_context_value(user_context, 'user_id') or 'anonymous'" in source
	assert "device_id=_context_value(user_context, 'device_id') or 'unknown'" in source
	assert 'proof_context = {**user_context, "tenant_id": tenant_id, "session_id": operation_id}' in source
	assert "assert tenant_id, \"Tenant context required for zero-knowledge proof\"" in source
	assert "tenant_id=tenant_id" in source
	assert "session_id=session_id" in source
