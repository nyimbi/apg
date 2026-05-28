"""KEYM views should report runtime key-management state, not fixed demo data."""

from __future__ import annotations

from datetime import datetime, timedelta

from flask import Flask

from capabilities.common.keym.models import (
	AuditEvent,
	ComplianceFramework,
	Key,
	KeyAlgorithm,
	KeyMetadata,
	KeyPolicy,
	KeySpec,
	KeyState,
	KeyUsage,
	KeyUsageStats,
	SecurityLevel,
	SecurityThreat,
)
from capabilities.common.keym.service import KeyManagementService
from capabilities.common.keym.views import (
	KeyListView,
	KeyManagementDashboardView,
	api_dashboard_stats,
	api_key_health,
	api_security_alerts,
	set_key_management_service,
)


def make_key(
	key_id: str,
	name: str,
	algorithm: KeyAlgorithm,
	state: KeyState,
	frameworks: list[ComplianceFramework],
	next_rotation: datetime | None = None,
) -> Key:
	spec = KeySpec(
		id=key_id,
		tenant_id="tenant-a",
		algorithm=algorithm,
		key_size=256 if algorithm == KeyAlgorithm.AES_256 else 2048,
		usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
		metadata=KeyMetadata(name=name, project_id="runtime", environment="test"),
		policy=KeyPolicy(compliance_frameworks=frameworks, auto_rotate=True),
		security_level=SecurityLevel.CONFIDENTIAL,
		state=state,
		created_by="tester",
	)
	return Key(spec=spec, next_rotation=next_rotation)


def make_service() -> KeyManagementService:
	service = KeyManagementService()
	service.is_initialized = True
	service.config = {"tenant_id": "tenant-a"}
	soon = datetime.utcnow() + timedelta(days=7)
	service.keys = {
		"key-runtime-a": make_key(
			"key-runtime-a",
			"Runtime API Key",
			KeyAlgorithm.AES_256,
			KeyState.ACTIVE,
			[ComplianceFramework.GDPR],
			next_rotation=soon,
		),
		"key-runtime-b": make_key(
			"key-runtime-b",
			"Runtime Database Key",
			KeyAlgorithm.RSA_2048,
			KeyState.SUSPENDED,
			[ComplianceFramework.HIPAA],
		),
	}
	service.usage_stats = {
		"key-runtime-a": KeyUsageStats(
			key_id="key-runtime-a",
			tenant_id="tenant-a",
			total_operations=7,
			last_used=datetime.utcnow() - timedelta(minutes=5),
		)
	}
	service.threats = {
		"threat-a": SecurityThreat(
			threat_id="threat-a",
			tenant_id="tenant-a",
			threat_type="volume_anomaly",
			severity="high",
			confidence=0.91,
			affected_keys=["key-runtime-a"],
			detection_method="test",
		)
	}
	service.audit_events = [
		AuditEvent(
			tenant_id="tenant-a",
			event_type="policy_violation",
			resource_type="key",
			resource_id="key-runtime-b",
			action="evaluate_policy",
			outcome="violation",
			compliance_frameworks=[ComplianceFramework.HIPAA],
		)
	]
	service.hsm_configs = {
		"hsm-a": {"enabled": True},
		"hsm-b": {"enabled": False, "status": "disabled"},
	}
	return service


def test_dashboard_data_comes_from_registered_service_state() -> None:
	view = KeyManagementDashboardView.__new__(KeyManagementDashboardView)
	view._keym_service = make_service()

	dashboard = view._get_dashboard_data()

	assert dashboard["summary"]["total_keys"] == 2
	assert dashboard["summary"]["active_keys"] == 1
	assert dashboard["summary"]["pending_rotation"] == 1
	assert dashboard["summary"]["security_alerts"] == 1
	assert dashboard["summary"]["compliance_violations"] == 1
	assert dashboard["algorithm_distribution"] == {"AES-256": 1, "RSA-2048": 1}
	assert dashboard["security_metrics"]["threat_level"] == "high"
	assert dashboard["security_metrics"]["hsm_health"] == 50.0
	assert dashboard["compliance_status"]["GDPR"] == "compliant"
	assert dashboard["compliance_status"]["HIPAA"] == "violation"


def test_key_list_and_detail_use_runtime_keys() -> None:
	view = KeyListView.__new__(KeyListView)
	view._keym_service = make_service()

	list_data = view._get_keys_data("AES-256", "", "runtime api", page=1, per_page=25)
	detail = view._get_key_detail("key-runtime-a")

	assert list_data["pagination"]["total"] == 1
	assert list_data["keys"][0]["id"] == "key-runtime-a"
	assert list_data["keys"][0]["usage_count"] == 7
	assert detail is not None
	assert detail["name"] == "Runtime API Key"
	assert detail["metadata"]["project"] == "runtime"
	assert view._get_key_detail("missing") is None


def test_keym_api_helpers_reflect_registered_runtime_state() -> None:
	service = make_service()
	set_key_management_service(service)
	app = Flask(__name__)

	try:
		with app.test_request_context("/"):
			stats_payload = api_dashboard_stats().get_json()
			alerts_payload = api_security_alerts().get_json()
			health_payload = api_key_health("key-runtime-a").get_json()

		assert stats_payload["data"] == {
			"total_keys": 2,
			"active_keys": 1,
			"security_alerts": 1,
			"compliance_score": 50.0,
		}
		assert alerts_payload["data"][0]["id"] == "threat-a"
		assert alerts_payload["data"][0]["severity"] == "high"
		assert health_payload["data"]["key_id"] == "key-runtime-a"
		assert health_payload["data"]["usage_count"] == 7
	finally:
		set_key_management_service(None)
