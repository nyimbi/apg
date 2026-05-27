"""Regressions for AUTH ABAC policy applicability."""

from __future__ import annotations

import pytest

from capabilities.common.auth import ABACEngine, AccessRequest, Policy


@pytest.mark.asyncio
async def test_abac_policy_applies_to_canonical_request_resource_and_action():
	engine = ABACEngine()
	engine.add_policy(Policy(
		name="Allow invoice reads",
		priority=10,
		effect="allow",
		resource_conditions=[
			{"attribute": "resource", "operator": "equals", "value": "invoice"}
		],
		action_conditions=[
			{"attribute": "action", "operator": "equals", "value": "read"}
		],
	))

	decision = await engine.evaluate_access(AccessRequest(
		subject_id="user-a",
		resource="invoice",
		action="read",
		tenant_id="tenant-a",
	))

	assert decision.decision == "allow"
	assert decision.reason == "Policy Allow invoice reads evaluated to allow"
	assert len(decision.policies_evaluated) == 1


@pytest.mark.asyncio
async def test_abac_policy_applicability_skips_unrelated_policies_before_first_match():
	engine = ABACEngine()
	engine.add_policy(Policy(
		name="Allow invoice reads only",
		priority=1,
		effect="allow",
		resource_conditions=[
			{"attribute": "resource", "operator": "equals", "value": "invoice"}
		],
		action_conditions=[
			{"attribute": "action", "operator": "equals", "value": "read"}
		],
	))
	engine.add_policy(Policy(
		name="Deny payroll deletes",
		priority=20,
		effect="deny",
		resource_conditions=[
			{"attribute": "resource", "operator": "equals", "value": "payroll"}
		],
		action_conditions=[
			{"attribute": "action", "operator": "equals", "value": "delete"}
		],
	))

	decision = await engine.evaluate_access(AccessRequest(
		subject_id="user-a",
		resource="payroll",
		action="delete",
		tenant_id="tenant-a",
	))

	assert decision.decision == "deny"
	assert decision.reason == "Policy Deny payroll deletes evaluated to deny"
	assert len(decision.policies_evaluated) == 1


@pytest.mark.asyncio
async def test_abac_policy_applicability_matches_subject_and_environment_context():
	engine = ABACEngine()
	engine.set_attributes("user-a", {"department": "finance"})
	engine.add_policy(Policy(
		name="Allow finance workstation exports",
		priority=10,
		effect="allow",
		subject_conditions=[
			{"attribute": "subject_id", "operator": "equals", "value": "user-a"},
			{"attribute": "department", "operator": "equals", "value": "finance"},
		],
		environment_conditions=[
			{"attribute": "ip_address", "operator": "starts_with", "value": "10.0."}
		],
		action_conditions=[
			{"attribute": "action", "operator": "equals", "value": "export"}
		],
	))

	decision = await engine.evaluate_access(AccessRequest(
		subject_id="user-a",
		resource="ledger",
		action="export",
		tenant_id="tenant-a",
		ip_address="10.0.5.9",
	))

	assert decision.decision == "allow"
