"""Service tests for transport_tra (Asset Tracking)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name: str, filename: str):
	path = PACKAGE_DIR / filename
	spec = importlib.util.spec_from_file_location(mod_name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[mod_name] = mod
	spec.loader.exec_module(mod)
	return mod

_cc = _load("_contract2_tra", "capability_contract.py")
import sys as _sys
_sys.modules["capability_contract"] = _cc
_models_mod = _load("_models_tra", "models.py")
_sys.modules["models"] = _models_mod
_svc_mod = _load("_service_tra", "service.py")
AssetTrackingService = _svc_mod.AssetTrackingService

def test_register_asset():
	svc = AssetTrackingService()
	a = svc.register_asset("a1", "t1", "vehicle", "KCA001A", "fleet-1", "KCA 001A", "gps")
	assert a["asset_type"] == "vehicle"
	assert a["active"] is True


def test_asset_invalid_type():
	svc = AssetTrackingService()
	with pytest.raises(PermissionError, match="asset_type_not_supported"):
		svc.register_asset("a1", "t1", "submarine", "SUB001", "fleet-1", "", "gps")


def test_asset_missing_owner():
	svc = AssetTrackingService()
	with pytest.raises(PermissionError, match="asset_owner_required"):
		svc.register_asset("a1", "t1", "vehicle", "KCA001A", "", "KCA 001A", "gps")


def test_update_location():
	svc = AssetTrackingService()
	svc.register_asset("a1", "t1", "vehicle", "KCA001A", "fleet-1", "KCA 001A", "gps")
	u = svc.update_asset_location("u1", "t1", "a1", -1.2921, 36.8219, 60.0, 90.0, "2026-06-01T10:00:00Z", "gps")
	assert u["latitude"] == -1.2921
	assert u["speed_kmh"] == 60.0


def test_tamper_alert_blocks_update():
	svc = AssetTrackingService()
	with pytest.raises(PermissionError, match="tamper_alert_requires_escalation"):
		svc.update_asset_location("u1", "t1", "a1", -1.29, 36.82, 0.0, 0.0, "2026-06-01T10:00:00Z", "gps", tamper_detected=True)


def test_create_geofence():
	svc = AssetTrackingService()
	g = svc.create_geofence("gf1", "t1", "circle", "Depot A", '{"lat": -1.29, "lng": 36.82, "radius_m": 500}')
	assert g["geofence_type"] == "circle"


def test_raise_alert():
	svc = AssetTrackingService()
	al = svc.raise_alert("al1", "t1", "a1", "speeding", "high", "2026-06-01T10:05:00Z", "Speed 120 in 80 zone")
	assert al["alert_type"] == "speeding"
	assert al["resolved_at"] is None


def test_cold_chain_normal():
	svc = AssetTrackingService()
	cc = svc.record_cold_chain("cc1", "t1", "a1", "atp_agreement", 2.0, 8.0, 5.5, "2026-06-01T10:00:00Z")
	assert cc["breached"] is False
	assert cc["recorded_temp_c"] == 5.5


def test_cold_chain_breach():
	svc = AssetTrackingService()
	cc = svc.record_cold_chain("cc1", "t1", "a1", "atp_agreement", 2.0, 8.0, 12.0, "2026-06-01T10:00:00Z")
	assert cc["breached"] is True


def test_register_container():
	svc = AssetTrackingService()
	c = svc.register_container("con1", "t1", "MSCU1234567", "SEAL001", "owner-1", "Mombasa Port", "2026-06-01T10:00:00Z")
	assert c["iso_number"] == "MSCU1234567"
	assert c["status"] == "available"


def test_container_status_update():
	svc = AssetTrackingService()
	svc.register_container("con1", "t1", "MSCU1234567", "SEAL001", "owner-1", "Mombasa Port", "2026-06-01T10:00:00Z")
	c = svc.update_container_status("con1", "t1", "in_transit")
	assert c["status"] == "in_transit"


def test_utilisation_calculation():
	svc = AssetTrackingService()
	r = svc.record_utilisation("u1", "t1", "a1", "weekly", "2026-06-01", "2026-06-07", 1440, 8640, 2500.0)
	assert r["utilisation_pct"] == 85.71


def test_register_agent():
	svc = AssetTrackingService()
	a = svc.register_tracking_agent("a1", "t1", "Tracking Bot", "codex", "asset_tracker", "tracking scope")
	assert a["role"] == "asset_tracker"
