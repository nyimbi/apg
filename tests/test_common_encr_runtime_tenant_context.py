"""Runtime tenant-context regressions for ENCR support modules."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENCR_PATH = REPO_ROOT / "capabilities" / "common" / "encr"


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
