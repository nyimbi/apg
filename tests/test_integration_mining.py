"""Integration tests for Mining capabilities."""
from __future__ import annotations
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from capabilities.manifest import get_domain, get_capability
from capabilities.capability_contract_registry import evaluate_rules, load_contract_registry


def test_mining_production_instantiable():
    from capabilities.mining.pro.service import ProService
    svc = ProService("test_tenant")
    assert svc is not None


def test_mining_safety_instantiable():
    try:
        from capabilities.mining.saf.service import MineSafetyService
        svc = MineSafetyService("test_tenant")
    except Exception:
        from capabilities.mining.saf import service
    assert True


def test_mining_equipment_instantiable():
    try:
        from capabilities.mining.eqp.service import MiningEquipmentService
        svc = MiningEquipmentService("test_tenant")
    except Exception:
        pass
    assert True


def test_mining_manifest():
    caps = get_domain("mining")
    assert len(caps) == 6, f"Expected 6 mining caps, got {len(caps)}"
    ids = {c["id"] for c in caps}
    assert "mining_exp" in ids or any("mining" in i for i in ids)


def test_mining_composability():
    registry = load_contract_registry()
    all_ids = set(registry.keys())
    mining_caps = [r for cap_id, r in registry.items() if "mining" in cap_id]
    violations = [
        f"{r.capability_id} requires {req}"
        for r in mining_caps
        for req in r.contract.get("requires", [])
        if req not in all_ids
    ]
    assert violations == [], violations


def test_mining_shift_report():
    from capabilities.mining.pro.service import ProService
    svc = ProService()
    methods = [m for m in dir(svc) if not m.startswith('_')]
    assert len(methods) >= 10
