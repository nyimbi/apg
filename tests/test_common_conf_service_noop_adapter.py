"""Configuration manager fallback adapter regressions."""

from __future__ import annotations

from capabilities.common.conf.service import _NoopAIModelAdapter


def test_noop_ai_model_adapter_records_dependency_bindings():
	adapter = _NoopAIModelAdapter()
	manager = object()
	nlp_service = object()

	adapter.set_config_manager(manager)
	adapter.set_gitops_manager(None)
	adapter.set_nlp_service(nlp_service)

	description = adapter.describe_runtime()

	assert adapter.config_manager is manager
	assert adapter.gitops_manager is None
	assert adapter.nlp_service is nlp_service
	assert description["adapter"] == "noop"
	assert description["config_manager_attached"] is True
	assert description["gitops_manager_attached"] is False
	assert description["nlp_service_attached"] is True
	assert [binding["component"] for binding in description["bindings"]] == [
		"config_manager",
		"gitops_manager",
		"nlp_service",
	]
	assert [binding["attached"] for binding in description["bindings"]] == [True, False, True]
