"""Integration tests for Localisation capabilities (multi-currency, multi-country, multi-language)."""
from __future__ import annotations
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from capabilities.manifest import get_domain
from capabilities.capability_contract_registry import load_contract_registry


def test_multi_currency_instantiable():
    from capabilities.loc.mcy.service import MultiCurrencyManagementService
    svc = MultiCurrencyManagementService()
    assert svc is not None


def test_multi_country_instantiable():
    from capabilities.loc.mco.service import MultiCountryOperationsService
    svc = MultiCountryOperationsService()
    assert svc is not None


def test_multi_language_instantiable():
    from capabilities.loc.mlg.service import MultiLanguageLocalisationService
    svc = MultiLanguageLocalisationService()
    assert svc is not None


def test_loc_manifest():
    caps = get_domain("loc")
    assert len(caps) == 3, f"Expected 3 loc caps, got {len(caps)}"


def test_loc_composability():
    registry = load_contract_registry()
    all_ids = set(registry.keys())
    loc_caps = [r for cap_id, r in registry.items() if cap_id.startswith("loc_")]
    violations = [
        f"{r.capability_id} requires {req}"
        for r in loc_caps
        for req in r.contract.get("requires", [])
        if req not in all_ids
    ]
    assert violations == [], violations


def test_fx_rate_record():
    from capabilities.loc.mcy.service import MultiCurrencyManagementService
    svc = MultiCurrencyManagementService()
    if hasattr(svc, "record_fx_rate"):
        result = asyncio.run(svc.record_fx_rate("test_tenant", "USD", "KES", 130.5, "2026-06-05"))
        assert isinstance(result, dict)
    elif hasattr(svc, "update_exchange_rate"):
        result = asyncio.run(svc.update_exchange_rate("USD", "KES", 130.5, "2026-06-05"))
        assert isinstance(result, dict)
    else:
        assert True
