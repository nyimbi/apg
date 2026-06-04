"""Service layer tests for APG Citizen Services Portal."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	"""Load module by path, always (re)registering deps under bare names for fallback imports."""
	# Always overwrite bare-name slots so this capability's deps win even in a multi-cap test run
	for dep in ('capability_contract', 'models'):
		dep_path = PACKAGE_DIR / f"{dep}.py"
		if dep_path.exists():
			dep_spec = importlib.util.spec_from_file_location(f"{name}__{dep}", dep_path)
			dep_mod = importlib.util.module_from_spec(dep_spec)
			sys.modules[f"{name}__{dep}"] = dep_mod
			sys.modules[dep] = dep_mod  # overwrite bare name each time
			dep_spec.loader.exec_module(dep_mod)
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_full_citizen_service_lifecycle():
	svc = _load("svc_csr", PACKAGE_DIR / "service.py").CitizenServicesService()
	svc.register_service("svc1", "t1", "certificate_issuance", "Birth Certificate", "Issue birth cert", 500, "KES", 5)
	app = svc.submit_application("app1", "t1", "svc1", "citizen-1", "web_portal", "REF-001", "doc-ref")
	assert app["status"] == "submitted"
	payment = svc.record_payment("pay1", "t1", "app1", "mpesa", 500.0, "KES", "RCP-001", "TXN-001")
	assert payment["status"] == "completed"
	verification = svc.verify_document("ver1", "t1", "app1", "identity", "id-doc-ref", "id-scan")
	assert verification["status"] == "verified"
	notification = svc.send_notification("notif1", "t1", "app1", "citizen-1", "sms", "Your certificate is ready")
	assert notification["sent"] is True
	delivery = svc.record_delivery("del1", "t1", "app1", "postal", "CERT-2025-001")
	assert delivery["certificate_reference"] == "CERT-2025-001"
	summary = svc.dashboard_summary("t1")
	assert summary["application_count"] == 1
	assert summary["payment_count"] == 1


def test_missing_citizen_id_denied():
	svc = _load("svc_csr_cid", PACKAGE_DIR / "service.py").CitizenServicesService()
	svc.register_service("svc1", "t1", "certificate_issuance", "Name", "Desc", 100, "KES", 5)
	with pytest.raises(PermissionError, match="citizen_id_required"):
		svc.submit_application("app1", "t1", "svc1", "", "web_portal", "REF-001", "ev")


def test_unsupported_payment_method_denied():
	svc = _load("svc_csr_pay", PACKAGE_DIR / "service.py").CitizenServicesService()
	svc.register_service("svc1", "t1", "certificate_issuance", "Name", "Desc", 100, "KES", 5)
	svc.submit_application("app1", "t1", "svc1", "c1", "web_portal", "REF-001", "ev")
	with pytest.raises(PermissionError, match="payment_method_not_supported"):
		svc.record_payment("pay1", "t1", "app1", "bitcoin", 100.0, "KES", "RCP-001", "TXN-001")


def test_verification_unsupported_type_denied():
	svc = _load("svc_csr_ver", PACKAGE_DIR / "service.py").CitizenServicesService()
	svc.register_service("s1", "t1", "certificate_issuance", "N", "D", 0, "KES", 1)
	svc.submit_application("a1", "t1", "s1", "c1", "web_portal", "R1", "ev")
	with pytest.raises(PermissionError, match="verification_type_not_supported"):
		svc.verify_document("v1", "t1", "a1", "unknown_check", "doc", "ev")


def test_tenant_isolation():
	svc = _load("svc_csr_iso", PACKAGE_DIR / "service.py").CitizenServicesService()
	svc.register_service("s1", "ta", "certificate_issuance", "N", "D", 0, "KES", 1)
	svc.register_service("s1", "tb", "permit_application", "N", "D", 0, "KES", 1)
	assert svc.dashboard_summary("ta")["service_count"] == 1
	assert svc.dashboard_summary("tb")["service_count"] == 1


def test_agent_registration():
	svc = _load("svc_csr_agent", PACKAGE_DIR / "service.py").CitizenServicesService()
	agent = svc.register_agent("ag1", "t1", "Status Updater", "claude_code", "status_updater", "status scope")
	assert agent["role"] == "status_updater"


def test_batch_requires_bytewax():
	svc = _load("svc_csr_batch", PACKAGE_DIR / "service.py").CitizenServicesService()
	result = svc.validate_batch("t1", 10)
	assert result["processor"] == "bytewax"
	with pytest.raises(PermissionError):
		svc.validate_batch("t1", 10, event_stream="rabbitmq")


def test_review_requires_reviewer():
	svc = _load("svc_csr_rev", PACKAGE_DIR / "service.py").CitizenServicesService()
	with pytest.raises(PermissionError, match="reviewer_required"):
		svc.record_review("r1", "t1", "ref1", "", "approved", "ev")
