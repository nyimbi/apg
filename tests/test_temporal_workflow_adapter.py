"""Tests for Temporal durable workflow adapter.

Tests validate adapter interface, factory routing, workflow input construction,
and task completion signal — without requiring a live Temporal server.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── TemporalWorkflowAdapter ───────────────────────────────────────────────────

async def test_temporal_adapter_starts_workflow_returns_instance_id():
	"""start_workflow() returns a dict with instance_id and status."""
	from capabilities.common.temporal.temporal_adapter import TemporalWorkflowAdapter

	mock_handle = MagicMock()
	mock_handle.result_run_id = "run-abc-123"
	mock_client = AsyncMock()
	mock_client.start_workflow = AsyncMock(return_value=mock_handle)

	adapter = TemporalWorkflowAdapter(host="localhost:7233")
	adapter._client = mock_client

	result = await adapter.start_workflow(
		"PayRunProcess",
		{
			"tenant_id": "tenant1",
			"actor_id": "admin",
			"_workflow_declaration": {
				"states": ["draft", "calculated", "approved", "closed"],
				"transitions": [
					{"source": "draft", "target": "calculated"},
					{"source": "calculated", "target": "approved"},
					{"source": "approved", "target": "closed"},
				],
				"human_tasks": ["approved"],
				"guards": {},
				"timers": {},
				"assignments": {},
			},
		},
	)

	assert "instance_id" in result
	assert result["status"] == "running"
	assert result["run_id"] == "run-abc-123"
	mock_client.start_workflow.assert_called_once()


async def test_temporal_adapter_complete_task_sends_signal():
	"""complete_task() sends complete_human_task signal to the workflow."""
	from capabilities.common.temporal.temporal_adapter import TemporalWorkflowAdapter

	mock_handle = AsyncMock()
	mock_client = MagicMock()
	mock_client.get_workflow_handle = MagicMock(return_value=mock_handle)

	adapter = TemporalWorkflowAdapter(host="localhost:7233")
	adapter._client = mock_client

	await adapter.complete_task(
		"tenant1-PayRunProcess-abc::approved",
		"approved",
		{"signed_by": "manager"},
	)

	mock_handle.signal.assert_called_once_with(
		"complete_human_task", "approved", {"signed_by": "manager"}
	)


async def test_temporal_adapter_complete_task_invalid_id_does_not_raise():
	"""complete_task() with malformed task_id logs an error but does not raise."""
	from capabilities.common.temporal.temporal_adapter import TemporalWorkflowAdapter

	adapter = TemporalWorkflowAdapter(host="localhost:7233")
	adapter._client = MagicMock()

	# Should not raise
	await adapter.complete_task("bad-task-id-no-separator", "approved", {})


async def test_temporal_adapter_get_workflow_status():
	"""get_workflow_status() returns status dict."""
	from capabilities.common.temporal.temporal_adapter import TemporalWorkflowAdapter
	from unittest.mock import MagicMock
	import datetime

	mock_desc = MagicMock()
	mock_desc.status.name = "RUNNING"
	mock_desc.workflow_type = "APGStateMachineWorkflow"
	mock_desc.start_time = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)

	mock_handle = AsyncMock()
	mock_handle.describe = AsyncMock(return_value=mock_desc)

	mock_client = MagicMock()
	mock_client.get_workflow_handle = MagicMock(return_value=mock_handle)

	adapter = TemporalWorkflowAdapter(host="localhost:7233")
	adapter._client = mock_client

	status = await adapter.get_workflow_status("wf-123")
	assert status["instance_id"] == "wf-123"
	assert "running" in status["status"].lower()


async def test_temporal_adapter_cancel_workflow():
	"""cancel_workflow() calls handle.cancel() and returns True."""
	from capabilities.common.temporal.temporal_adapter import TemporalWorkflowAdapter

	mock_handle = AsyncMock()
	mock_client = MagicMock()
	mock_client.get_workflow_handle = MagicMock(return_value=mock_handle)

	adapter = TemporalWorkflowAdapter(host="localhost:7233")
	adapter._client = mock_client

	result = await adapter.cancel_workflow("wf-123")
	assert result is True
	mock_handle.cancel.assert_called_once()


# ── Factory routing ───────────────────────────────────────────────────────────

def test_get_temporal_workflow_adapter_returns_none_without_env(monkeypatch):
	"""Returns None when TEMPORAL_HOST is not configured."""
	monkeypatch.delenv("TEMPORAL_HOST", raising=False)
	from capabilities.common.temporal.temporal_adapter import get_temporal_workflow_adapter
	assert get_temporal_workflow_adapter() is None


def test_get_temporal_workflow_adapter_returns_adapter_with_env(monkeypatch):
	"""Returns TemporalWorkflowAdapter when TEMPORAL_HOST is set."""
	monkeypatch.setenv("TEMPORAL_HOST", "localhost:7233")
	from capabilities.common.temporal.temporal_adapter import (
		TemporalWorkflowAdapter,
		get_temporal_workflow_adapter,
	)
	adapter = get_temporal_workflow_adapter()
	assert isinstance(adapter, TemporalWorkflowAdapter)
	assert adapter._host == "localhost:7233"


def test_get_workflow_adapter_uses_temporal_when_host_set(monkeypatch):
	"""get_workflow_adapter() returns TemporalWorkflowAdapter when TEMPORAL_HOST set."""
	monkeypatch.setenv("TEMPORAL_HOST", "localhost:7233")
	from capabilities.ckm.wfa.domain.adapters import get_workflow_adapter
	from capabilities.common.temporal.temporal_adapter import TemporalWorkflowAdapter
	adapter = get_workflow_adapter()
	assert isinstance(adapter, TemporalWorkflowAdapter)


def test_get_workflow_adapter_falls_back_to_null_without_temporal(monkeypatch):
	"""get_workflow_adapter() falls back to NullWorkflowAdapter without env."""
	monkeypatch.delenv("TEMPORAL_HOST", raising=False)
	from capabilities.ckm.wfa.domain.adapters import NullWorkflowAdapter, get_workflow_adapter
	adapter = get_workflow_adapter()
	assert isinstance(adapter, NullWorkflowAdapter)


# ── APGWorkflowInput ──────────────────────────────────────────────────────────

def test_apg_workflow_input_construction():
	"""APGWorkflowInput can be constructed with required fields."""
	from capabilities.common.temporal.apg_workflow import APGWorkflowInput
	wf_input = APGWorkflowInput(
		workflow_id="wf-001",
		definition_id="PayRunProcess",
		tenant_id="tenant1",
		actor_id="admin",
		initial_state="draft",
		states=["draft", "calculated", "approved"],
		transitions=[
			{"source": "draft", "target": "calculated"},
			{"source": "calculated", "target": "approved"},
		],
		human_tasks=["approved"],
	)
	assert wf_input.workflow_id == "wf-001"
	assert wf_input.initial_state == "draft"
	assert len(wf_input.transitions) == 2
	assert "approved" in wf_input.human_tasks


# ── Guard evaluation activity ─────────────────────────────────────────────────

async def test_evaluate_guard_true_expression():
	"""evaluate_guard returns True for a true condition."""
	from capabilities.common.temporal.apg_activities import evaluate_guard
	result = await evaluate_guard(
		"total_hours > 0",
		{"total_hours": 9.0},
		"tenant1",
	)
	assert result is True


async def test_evaluate_guard_false_expression():
	"""evaluate_guard returns False for a false condition."""
	from capabilities.common.temporal.apg_activities import evaluate_guard
	result = await evaluate_guard(
		"all_payslips_generated == True",
		{"all_payslips_generated": False},
		"tenant1",
	)
	assert result is False


async def test_evaluate_guard_invalid_expression_returns_true():
	"""evaluate_guard returns True (non-blocking) on invalid expression."""
	from capabilities.common.temporal.apg_activities import evaluate_guard
	result = await evaluate_guard("this is not valid python !!!!", {}, "tenant1")
	assert result is True  # non-blocking on error
