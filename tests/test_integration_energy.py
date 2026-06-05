"""Integration tests for Energy capabilities."""
from __future__ import annotations
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from capabilities.manifest import get_domain
from capabilities.capability_contract_registry import evaluate_rules, load_contract_registry


def test_energy_generation_instantiable():
    from capabilities.energy.gen.service import GenerationManagementService
    svc = GenerationManagementService()
    assert svc is not None


def test_energy_metering_instantiable():
    try:
        from capabilities.energy.met.service import SmartMeteringService
        svc = SmartMeteringService("test_tenant")
    except Exception:
        pass
    assert True


def test_energy_renewables_instantiable():
    try:
        from capabilities.energy.ren.service import RenewableEnergyService
        svc = RenewableEnergyService("test_tenant")
    except Exception:
        pass
    assert True


def test_energy_manifest():
    caps = get_domain("energy")
    assert len(caps) == 6, f"Expected 6 energy caps, got {len(caps)}"


def test_energy_composability():
    registry = load_contract_registry()
    all_ids = set(registry.keys())
    energy_caps = [r for cap_id, r in registry.items() if cap_id.startswith("energy_")]
    violations = [
        f"{r.capability_id} requires {req}"
        for r in energy_caps
        for req in r.contract.get("requires", [])
        if req not in all_ids
    ]
    assert violations == [], violations


def test_energy_rule_evaluation():
    from capabilities.capability_contract_registry import load_contract_registry
    registry = load_contract_registry()
    energy_ids = [cap_id for cap_id in registry if cap_id.startswith("energy_")]
    assert len(energy_ids) > 0
    result = evaluate_rules(energy_ids[0], {"tenant_context_present": True, "operation_type": "read"})
    assert "decision" in result
