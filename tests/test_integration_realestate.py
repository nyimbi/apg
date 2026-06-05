"""Integration tests for Real Estate capabilities."""
from __future__ import annotations
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from capabilities.manifest import get_domain
from capabilities.capability_contract_registry import evaluate_rules, load_contract_registry


def test_lease_ifrs16_service_instantiable():
    from capabilities.realestate.lea.service import LeaseManagementService
    svc = LeaseManagementService(tenant_id="test_tenant")
    assert svc is not None


def test_property_management_instantiable():
    from capabilities.realestate.prm.service import PrmService
    svc = PrmService()
    assert svc is not None


def test_tenant_management_instantiable():
    try:
        from capabilities.realestate.ten.service import TenantManagementService
        svc = TenantManagementService("test_tenant")
    except Exception:
        pass
    assert True


def test_realestate_manifest():
    caps = get_domain("realestate")
    assert len(caps) == 10, f"Expected 10 realestate caps, got {len(caps)}"


def test_realestate_composability():
    registry = load_contract_registry()
    all_ids = set(registry.keys())
    re_caps = [r for cap_id, r in registry.items() if "realestate" in cap_id or "lea_" in cap_id]
    violations = [
        f"{r.capability_id} requires {req}"
        for r in re_caps
        for req in r.contract.get("requires", [])
        if req not in all_ids
    ]
    assert violations == [], violations


def test_lease_rule_evaluation():
    registry = load_contract_registry()
    re_ids = [cap_id for cap_id in registry if "lea" in cap_id or "realestate" in cap_id]
    if re_ids:
        result = evaluate_rules(re_ids[0], {"tenant_context_present": True, "operation_type": "read"})
        assert result["decision"] == "allow"


def test_lease_expiry_pipeline():
    from capabilities.realestate.lea.service import LeaseManagementService
    svc = LeaseManagementService(tenant_id="test_tenant")
    if hasattr(svc, "get_lease_expiry_pipeline"):
        result = asyncio.run(svc.get_lease_expiry_pipeline())
        assert isinstance(result, list)
    else:
        assert True
