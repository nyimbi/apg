"""Gateway API runtime-state regressions for executable app flows."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from uuid_extensions import uuid7str


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "composition" / "gateway" / "api.py"


def _runtime_helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("gateway_runtime_state:")
	end = source.index("\n@asynccontextmanager")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"Optional": Optional,
		"datetime": __import__("datetime").datetime,
		"timezone": __import__("datetime").timezone,
		"uuid7str": uuid7str,
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


class _Algorithm(Enum):
	ROUND_ROBIN = "round_robin"


def test_gateway_runtime_state_stores_and_filters_api_items():
	helpers = _runtime_helpers()
	store = helpers["_store_runtime_item"]
	list_items = helpers["_list_runtime_items"]

	route = store(
		"tenant-a",
		"routes",
		"route-1",
		{
			"route_id": "route-1",
			"route_name": "orders",
			"priority": 10,
			"destination_services": [{"service_id": "orders", "weight": 100}],
		},
		"user-a",
	)
	policy = store(
		"tenant-a",
		"policies",
		"policy-1",
		{
			"policy_id": "policy-1",
			"policy_name": "orders-rate-limit",
			"policy_type": "rate_limit",
			"configuration": {"requests": 100},
		},
		"user-a",
	)
	store(
		"tenant-a",
		"policies",
		"policy-2",
		{
			"policy_id": "policy-2",
			"policy_name": "orders-auth",
			"policy_type": "auth",
			"configuration": {"required": True},
		},
		"user-a",
	)

	routes = list_items("tenant-a", "routes")
	rate_limit_policies = list_items("tenant-a", "policies", policy_type="rate_limit")

	assert route["created_by"] == "user-a"
	assert route["destination_services"] == [{"service_id": "orders", "weight": 100}]
	assert routes["total"] == 1
	assert routes["items"][0]["route_id"] == "route-1"
	assert policy["policy_type"] == "rate_limit"
	assert rate_limit_policies["total"] == 1
	assert rate_limit_policies["items"][0]["policy_id"] == "policy-1"


def test_gateway_runtime_state_serializes_enums_and_health_checks():
	helpers = _runtime_helpers()
	store = helpers["_store_runtime_item"]
	record_health = helpers["_record_runtime_health_check"]
	state = helpers["gateway_runtime_state"]

	load_balancer = store(
		"tenant-b",
		"load_balancers",
		"lb-1",
		{
			"load_balancer_id": "lb-1",
			"load_balancer_name": "agent-lb",
			"algorithm": _Algorithm.ROUND_ROBIN,
		},
		"user-b",
	)
	health_check = record_health(
		"tenant-b",
		"agent-service",
		"queued",
		{"force_check": True},
	)

	assert load_balancer["algorithm"] == "round_robin"
	assert health_check["service_id"] == "agent-service"
	assert health_check["status"] == "queued"
	assert state["tenant-b"]["health_checks"][health_check["health_check_id"]] == health_check


def test_gateway_api_placeholders_replaced_by_runtime_state_calls():
	source = API_PATH.read_text(encoding="utf-8")

	assert "routes = []  # Placeholder" not in source
	assert "load_balancers = []  # Placeholder" not in source
	assert "policies = []  # Placeholder" not in source
	assert "# Implementation would trigger health check" not in source
	assert "_store_runtime_item(" in source
	assert "_list_runtime_items(" in source
	assert "_record_runtime_health_check(" in source
