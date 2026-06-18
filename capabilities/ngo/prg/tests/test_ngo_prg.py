"""Basic smoke tests for ngo_prg (ngo_prg)."""
from __future__ import annotations

import importlib.util
import json
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ngo_prg_contract_loads():
    """Verify capability_contract.py defines CONTRACT with expected capability."""
    contract_path = os.path.join(PKG_DIR, "capability_contract.py")
    spec = importlib.util.spec_from_file_location("_cap_contract", contract_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "CONTRACT"), "capability_contract.py must expose CONTRACT"
    contract = mod.CONTRACT
    assert isinstance(contract, dict), "CONTRACT must be a dict"


def test_ngo_prg_release_report():
    """Verify release_report.json meets apg.release-report.v1 schema."""
    rr_path = os.path.join(PKG_DIR, "release_report.json")
    assert os.path.exists(rr_path), f"release_report.json missing at {rr_path}"
    with open(rr_path, encoding="utf-8") as fh:
        rr = json.load(fh)
    assert rr.get("format") == "apg.release-report.v1"
    assert rr.get("ok") is True
    assert rr.get("evidence", {}).get("self_test", {}).get("passed") is True
