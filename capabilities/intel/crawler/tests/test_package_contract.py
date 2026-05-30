"""Intelligence crawler package contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "tested_intel_crawler"


def _load_module(name: str):
	if PACKAGE_NAME not in sys.modules:
		package = types.ModuleType(PACKAGE_NAME)
		package.__path__ = [str(PACKAGE_DIR)]
		sys.modules[PACKAGE_NAME] = package
	spec = importlib.util.spec_from_file_location(
		f"{PACKAGE_NAME}.{name}",
		PACKAGE_DIR / f"{name}.py",
	)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("capability_contract")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_crawler"
	assert "source_intelligence_registry" in contract["provides"]
	assert "crawler_agents" in contract["provides"]
	assert contract["requires"] == ["auth", "audl", "ntfy", "composition_events", "composition_config", "document_processing"]
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["stream"] == "apg.intel.crawler.lifecycle"
	assert any(route["path"] == "/intel-crawler/crawl-jobs" for route in contract["ui"]["routes"])
	assert any(route["path"] == "/intel-crawler/agents" for route in contract["ui"]["routes"])


def test_rule_engine_blocks_missing_context_and_non_bytewax_batches():
	module = _load_module("capability_contract")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	bad_stream = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "crawler_batch",
		"event_stream": "other",
	})
	bad_depth = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_crawl_job",
		"max_depth": 9,
	})

	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]
	assert bad_stream["decision"] == "deny"
	assert "crawler_batch_requires_bytewax" in bad_stream["matched_rules"]
	assert bad_depth["decision"] == "deny"
	assert "crawl_depth_within_limit" in bad_depth["matched_rules"]


def test_service_source_crawl_extraction_validation_dataset_knowledge_lifecycle():
	service_module = _load_module("service")
	service = service_module.IntelligenceCrawlerService()

	source = service.register_source(
		"news-feed",
		"tenant-test",
		"News Feed",
		"intel-team",
		"news",
		["https://example.com/feed"],
		["example.com"],
		policy_reviewed_by="policy-1",
	)
	job = service.create_crawl_job("job-1", "tenant-test", source["id"], "hourly", 2, 30)
	completed = service.complete_crawl_job("tenant-test", job["id"], fetched_count=12)
	extraction = service.record_extraction("ext-1", "tenant-test", job["id"], "article_v1", "clean article body", 0.92)
	validation = service.open_validation_session("val-1", "tenant-test", extraction["id"], "reviewer-1")
	validated = service.complete_validation_session("tenant-test", validation["id"], 0.91, "approve")
	dataset = service.publish_dataset("dataset-1", "tenant-test", extraction["id"], validation_recorded=True)
	rag = service.record_rag_plan("rag-1", "tenant-test", dataset["id"], "heading-aware", 1200, "text-embedding")
	graph = service.record_graph_projection("graph-1", "tenant-test", dataset["id"], "entities_v1", "sentence-links")
	summary = service.dashboard_summary("tenant-test")

	assert completed["status"] == "completed"
	assert extraction["status"] == "recorded"
	assert validated["status"] == "validated"
	assert dataset["status"] == "published"
	assert rag["status"] == "ready"
	assert graph["status"] == "ready"
	assert summary["dataset_count"] == 1
	assert summary["audit_event_count"] >= 7
	assert service.audit_events("tenant-test")[-1]["processor"] == "bytewax"


def test_service_enforces_crawler_guardrails():
	service_module = _load_module("service")
	service = service_module.IntelligenceCrawlerService()

	try:
		service.register_source("bad", "", "Bad", "owner", "news", ["https://example.com"], ["example.com"], "reviewer")
	except PermissionError as exc:
		assert "tenant_context_required" in str(exc)
	else:
		raise AssertionError("expected tenant guardrail")

	try:
		service.register_source("bad", "tenant-test", "Bad", "", "news", ["https://example.com"], ["example.com"], "reviewer")
	except PermissionError as exc:
		assert "source_requires_owner" in str(exc)
	else:
		raise AssertionError("expected source owner guardrail")

	try:
		service.register_source("review", "tenant-test", "Needs Review", "owner", "news", ["https://example.com"], ["example.com"])
	except PermissionError as exc:
		assert "source_requires_policy_review" in str(exc)
	else:
		raise AssertionError("expected policy review guardrail")

	source = service.register_source("source", "tenant-test", "Source", "owner", "news", ["https://example.com"], ["example.com"], "reviewer")
	try:
		service.create_crawl_job("deep", "tenant-test", source["id"], "daily", 9, 30)
	except PermissionError as exc:
		assert "crawl_depth_within_limit" in str(exc)
	else:
		raise AssertionError("expected depth guardrail")

	try:
		service.create_crawl_job("risk", "tenant-test", source["id"], "daily", 2, 30, high_risk=True)
	except PermissionError as exc:
		assert "high_risk_crawl_requires_approval" in str(exc)
	else:
		raise AssertionError("expected high-risk approval guardrail")

	job = service.create_crawl_job("job", "tenant-test", source["id"], "daily", 2, 30)
	try:
		service.record_extraction("low", "tenant-test", job["id"], "article", "body", 0.5)
	except PermissionError as exc:
		assert "extraction_quality_minimum" in str(exc)
	else:
		raise AssertionError("expected quality review guardrail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("service")
	api_module = _load_module("api")
	views_module = _load_module("views")
	app_module = _load_module("app")

	service = service_module.IntelligenceCrawlerService()
	source = service.register_source("source", "tenant-test", "Source", "owner", "news", ["https://example.com"], ["example.com"], "reviewer")
	agent = service.register_crawler_agent(
		"tenant-test",
		"Crawl Review",
		"codex",
		"crawl_policy_reviewer",
		"Review crawl policy.",
	)
	agent_result = service.validate_agent_crawler_action(
		"tenant-test",
		agent["id"],
		"recommend_crawl",
		privileged_scope=True,
		human_approval_recorded=True,
	)
	batch = service.validate_batch_ingest("tenant-test", 3)
	dashboard = views_module.dashboard_model(service, "tenant-test")
	source_view = views_module.source_model(service, "tenant-test")
	agent_view = views_module.agent_workbench_model(service, "tenant-test")
	api_record = api_module.create_record({"source_id": "api-source", "tenant_id": "tenant-api"})
	status = api_module.capability_status("tenant-api")
	self_test = app_module.self_test()
	model = app_module.semantic_model()

	assert source["status"] == "active"
	assert agent_result["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["crawler_agent_count"] == 1
	assert source_view["records"][0]["source_id"] == "source"
	assert agent_view["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert api_record["id"].startswith("crawler_source_")
	assert status["streaming"]["processor"] == "bytewax"
	assert self_test["passed"] is True
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_crawler"]["streaming"]["processor"] == "bytewax"
