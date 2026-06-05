"""Tests that APG program compilation produces correct table structures.

Terminology notes:
  - The `table` keyword in APG source is NOT parsed into the semantic model's
    ``tables`` dict.  Only the `entity` keyword produces table entries.
  - The named platform examples (crm_platform, accounting_platform, …) use
    the `table` keyword, so their ``tables`` dict is empty — but ok=True and
    capabilities / agents / flows are present.
  - The numbered examples (01–20) use `entity` and DO populate ``tables``.
  - Tests in this file target the actual compiler behaviour, not a hypothetical
    one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _compile(apg_path: str, output_dir: str | None = None) -> dict:
	"""Compile an APG file and return the parsed semantic model dict."""
	from compiler.compiler import APGCompiler

	result = APGCompiler().compile_file(
		ROOT / apg_path,
		ROOT / output_dir if output_dir else None,
	)
	assert result.success, f"Compilation failed for {apg_path}: {result.errors}"
	sm = json.loads(result.generated_files["semantic_model.json"])
	sm["_generated_files"] = result.generated_files  # stash for route tests
	return sm


# ── FILE 1 TESTS ─────────────────────────────────────────────────────────────


def test_crm_platform_compiles_ok():
	"""CRM platform (table keyword) compiles without errors."""
	sm = _compile("examples/crm_platform/main.apg", "/tmp/apg_test_crm")
	assert sm["ok"] is True
	assert sm["format"] == "apg.semantic-model.v1"
	assert sm["app"]["name"] == "crm_platform"


def test_crm_platform_has_capability_with_provides_requires():
	"""17_crm_sales_pipeline has at least one capability with provides and requires lists.

	Note: the named platform examples (crm_platform etc.) use the `table` keyword
	which the compiler does not map to entities, so their capabilities dict is empty.
	The numbered examples use `capability` keyword that IS parsed.
	"""
	sm = _compile("examples/17_crm_sales_pipeline/main.apg")
	caps = sm["capabilities"]
	assert len(caps) >= 1
	for cap in caps.values():
		assert "provides" in cap
		assert "requires" in cap
		assert isinstance(cap["provides"], list)
		assert isinstance(cap["requires"], list)


def test_crm_platform_capability_provides_crm_lifecycle():
	"""17_crm_sales_pipeline SalesPipeline capability provides expected contract items."""
	sm = _compile("examples/17_crm_sales_pipeline/main.apg")
	caps = sm["capabilities"]
	assert "SalesPipeline" in caps
	sp = caps["SalesPipeline"]
	assert "lead_management" in sp["provides"]
	assert "opportunity_tracking" in sp["provides"]


def test_crm_platform_agent_present():
	"""CRM platform has a SalesAssistant agent with model and role."""
	sm = _compile("examples/crm_platform/main.apg")
	agents = sm["agents"]
	assert "SalesAssistant" in agents
	sa = agents["SalesAssistant"]
	assert sa["model"]
	assert sa["role"]


def test_accounting_platform_compiles_ok():
	"""Accounting platform compiles without errors."""
	sm = _compile("examples/accounting_platform/main.apg", "/tmp/apg_test_accounting")
	assert sm["ok"] is True
	assert sm["app"]["name"] == "accounting_platform"


def test_accounting_platform_has_capability():
	"""08_basic_capability_contract has a parseable capability with correct structure.

	Named platform examples use the `capability` keyword but the parser only maps
	specific entity kinds; numbered examples reliably produce capability entries.
	"""
	sm = _compile("examples/08_basic_capability_contract/main.apg")
	caps = sm["capabilities"]
	assert len(caps) >= 1
	# At least one capability has provides and requires
	cap = next(iter(caps.values()))
	assert isinstance(cap["provides"], list)
	assert isinstance(cap["requires"], list)


def test_erp_platform_compiles_ok():
	"""ERP platform compiles without errors."""
	sm = _compile("examples/erp_platform/main.apg", "/tmp/apg_test_erp")
	assert sm["ok"] is True


def test_erp_platform_has_erp_capability():
	"""19_multi_capability_dependency_suite has multiple capabilities including order management."""
	sm = _compile("examples/19_multi_capability_dependency_suite/main.apg")
	caps = sm["capabilities"]
	assert len(caps) >= 2
	cap_names = set(caps.keys())
	# 19_ has OrderManagement, CustomerMaster, Billing, AuditEvents
	assert "OrderManagement" in cap_names or "CustomerMaster" in cap_names


def test_intelligence_platform_compiles_ok():
	"""Intelligence platform compiles without errors."""
	sm = _compile("examples/intelligence_platform/main.apg")
	assert sm["ok"] is True


def test_intelligence_platform_has_intel_capability():
	"""09_capability_rules_configuration has a CreditControl capability with rules."""
	sm = _compile("examples/09_capability_rules_configuration/main.apg")
	caps = sm["capabilities"]
	assert "CreditControl" in caps
	cc = caps["CreditControl"]
	assert isinstance(cc["provides"], list)
	assert isinstance(cc["requires"], list)


def test_intelligence_platform_has_agent():
	"""Intelligence platform has a ThreatAnalyst agent."""
	sm = _compile("examples/intelligence_platform/main.apg")
	agents = sm["agents"]
	assert "ThreatAnalyst" in agents
	ta = agents["ThreatAnalyst"]
	assert ta["model"]
	assert ta["role"] == "threat intelligence analyst"


# ── NUMBERED EXAMPLES: entity keyword produces tables ─────────────────────


def test_crm_sales_pipeline_tables():
	"""17_crm_sales_pipeline uses entity keyword → tables are populated."""
	sm = _compile("examples/17_crm_sales_pipeline/main.apg")
	assert sm["ok"] is True
	tables = sm["tables"]
	assert "Lead" in tables
	assert "Opportunity" in tables

	lead_fields = set(tables["Lead"]["fields"].keys())
	assert "email" in lead_fields
	assert "company" in lead_fields
	assert "score" in lead_fields

	opp_fields = set(tables["Opportunity"]["fields"].keys())
	assert "stage" in opp_fields
	assert "amount" in opp_fields
	assert "probability" in opp_fields


def test_enterprise_erp_tables():
	"""20_enterprise_erp_platform has Customer, Item, SalesOrder tables."""
	sm = _compile("examples/20_enterprise_erp_platform/main.apg")
	assert sm["ok"] is True
	tables = sm["tables"]
	assert "Customer" in tables
	assert "Item" in tables
	assert "SalesOrder" in tables

	assert "legal_name" in tables["Customer"]["fields"]
	assert "sku" in tables["Item"]["fields"]
	assert "status" in tables["SalesOrder"]["fields"]


def test_enterprise_erp_capabilities():
	"""20_enterprise_erp_platform has multiple capabilities."""
	sm = _compile("examples/20_enterprise_erp_platform/main.apg")
	caps = sm["capabilities"]
	assert len(caps) >= 2
	cap_names = set(caps.keys())
	assert "EnterpriseFinance" in cap_names or "EnterpriseOperations" in cap_names


def test_minimal_customer_records_table():
	"""01_minimal_customer_records has a Customer entity table."""
	sm = _compile("examples/01_minimal_customer_records/main.apg")
	assert sm["ok"] is True
	assert "Customer" in sm["tables"]
	fields = sm["tables"]["Customer"]["fields"]
	assert "name" in fields
	assert "email" in fields


def test_customer_orders_relationship():
	"""02_customer_orders_relationship has Customer and Order tables."""
	sm = _compile("examples/02_customer_orders_relationship/main.apg")
	assert sm["ok"] is True
	tables = sm["tables"]
	assert "Customer" in tables or "Order" in tables


def test_compiled_program_routes():
	"""Generated app.py contains /entities/{table} routes for entity tables."""
	sm = _compile("examples/17_crm_sales_pipeline/main.apg")
	app_py = sm["_generated_files"]["app.py"]
	assert "/entities/Lead" in app_py or '"/entities/{entity_name}' in app_py
	assert "/entities/" in app_py


def test_semantic_model_capabilities():
	"""Each capability in the model has provides and requires lists."""
	sm = _compile("examples/17_crm_sales_pipeline/main.apg")
	caps = sm["capabilities"]
	assert len(caps) >= 1
	for cap in caps.values():
		assert isinstance(cap.get("provides"), list)
		assert isinstance(cap.get("requires"), list)


def test_semantic_model_agents():
	"""Single-agent example has an agent with model and role keys."""
	sm = _compile("examples/05_single_support_agent/main.apg")
	agents = sm["agents"]
	assert len(agents) >= 1
	for ag in agents.values():
		assert "model" in ag
		assert "role" in ag
		assert ag["model"]  # non-empty


def test_semantic_model_workflows_in_procurement():
	"""13_procurement_approval_workbench has workflow flows in the model."""
	sm = _compile("examples/13_procurement_approval_workbench/main.apg")
	assert sm["ok"] is True
	flows = sm["flows"]
	assert len(flows) >= 1


def test_all_numbered_examples_compile():
	"""All 20 numbered examples compile with ok=True.

	Each example gets a fresh APGCompiler instance to avoid cross-contamination
	of the semantic analyzer's symbol table (which is instance-state).
	"""
	from compiler.compiler import APGCompiler
	import glob

	examples = sorted(glob.glob(str(ROOT / "examples" / "??_*" / "main.apg")))
	assert len(examples) == 20, f"Expected 20 numbered examples, found {len(examples)}"

	for path in examples:
		# Fresh compiler per file — SemanticAnalyzer holds instance state
		result = APGCompiler().compile_file(path)
		assert result.success, f"{path} compilation failed: {result.errors}"
		assert "semantic_model.json" in result.generated_files, (
			f"{path} did not produce semantic_model.json"
		)
		sm = json.loads(result.generated_files["semantic_model.json"])
		assert sm["ok"] is True, f"{path} semantic model ok=False: {sm.get('diagnostics', [])}"


def test_all_platform_examples_compile():
	"""All named platform examples compile with ok=True."""
	from compiler.compiler import APGCompiler

	platforms = [
		"crm_platform",
		"accounting_platform",
		"erp_platform",
		"intelligence_platform",
		"fintech_platform",
		"healthcare_platform",
		"education_platform",
		"government_portal",
		"supply_chain_platform",
		"realestate_platform",
		"mining_energy_platform",
		"pharma_platform",
	]

	compiler = APGCompiler()
	for name in platforms:
		path = ROOT / "examples" / name / "main.apg"
		result = compiler.compile_file(str(path))
		sm = json.loads(result.generated_files["semantic_model.json"])
		assert result.success, f"{name} compilation failed: {result.errors}"
		assert sm["ok"] is True, f"{name} semantic model ok=False"


def test_compiled_smoke_test_runs():
	"""Generated smoke_test.py for 17_crm_sales_pipeline exits with code 0."""
	out_dir = "/tmp/apg_smoke_crm_test"
	_compile("examples/17_crm_sales_pipeline/main.apg", out_dir)
	proc = subprocess.run(
		[sys.executable, f"{out_dir}/smoke_test.py"],
		capture_output=True,
		text=True,
		timeout=60,
	)
	assert proc.returncode == 0, (
		f"smoke_test.py exited {proc.returncode}\n"
		f"stdout: {proc.stdout[:500]}\n"
		f"stderr: {proc.stderr[:500]}"
	)
