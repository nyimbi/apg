"""Executable GitOps integration coverage for CONF."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from capabilities.common.conf.gitops_integration import (
	CIPipeline,
	CIPipelineEngine,
	GitOpsRepository,
	GitRepository,
	PipelineStatus,
)


@pytest.mark.asyncio
async def test_gitops_repository_commits_manifests_and_records_pr_evidence(tmp_path: Path):
	repository = GitRepository(
		name="local-conf-gitops",
		branch="main",
		local_path=str(tmp_path / "repo"),
		sync_enabled=False,
	)
	gitops_repo = GitOpsRepository(repository)

	assert await gitops_repo.clone_or_pull() is True
	assert await gitops_repo.write_manifest_file(
		"environments/dev/resources/api.yaml",
		{"kind": "Configuration", "spec": {"resources": {"replicas": 2}}},
	) is True
	assert await gitops_repo.commit_and_push(
		["environments/dev/resources/api.yaml"],
		"Add API manifest",
	) is True

	commit_sha = await gitops_repo.get_latest_commit_sha()
	assert commit_sha is not None
	assert len(commit_sha) == 40

	assert await gitops_repo.create_branch("feature/config-review", "main") is True
	pr_id = await gitops_repo.create_pull_request(
		"feature/config-review",
		"main",
		"Promote API configuration",
		"Review local GitOps evidence before deployment.",
	)

	assert pr_id is not None
	evidence = tmp_path / "repo" / ".apg" / "pull_requests" / f"{pr_id}.json"
	assert evidence.exists()
	assert await gitops_repo.get_latest_commit_sha() != commit_sha


@pytest.mark.asyncio
async def test_pipeline_test_and_deploy_stages_use_trigger_context():
	engine = CIPipelineEngine()
	pipeline = CIPipeline(
		name="context-backed-deploy",
		stages=[
			{"name": "context_checks", "type": "test", "checks": ["commit_sha", "branch", "manifest"]},
			{"name": "prepare_deploy", "type": "deploy"},
		],
	)

	execution_id = await engine.execute_pipeline(
		pipeline,
		{
			"event": "push",
			"commit_sha": "a" * 40,
			"branch": "main",
			"author": "conf-bot",
			"environment": "dev",
			"deployment_target": "local-runtime",
			"manifest": {"kind": "Configuration", "spec": {"resources": {"replicas": 3}}},
		},
	)

	for _ in range(20):
		execution = await engine.get_execution_status(execution_id)
		if execution and execution.status != PipelineStatus.RUNNING:
			break
		await asyncio.sleep(0.01)

	assert execution is not None
	assert execution.status == PipelineStatus.SUCCESS
	assert any(artifact.get("type") == "test_result" for artifact in execution.artifacts)
	deployment = next(artifact for artifact in execution.artifacts if artifact.get("type") == "deployment_evidence")
	assert deployment["environment"] == "dev"
	assert deployment["target"] == "local-runtime"
	assert deployment["manifest_present"] is True


@pytest.mark.asyncio
async def test_pipeline_deploy_stage_requires_environment():
	engine = CIPipelineEngine()
	pipeline = CIPipeline(name="missing-env", stages=[{"name": "deploy", "type": "deploy"}])

	execution_id = await engine.execute_pipeline(
		pipeline,
		{"event": "push", "commit_sha": "b" * 40, "branch": "main", "author": "conf-bot"},
	)

	for _ in range(20):
		execution = await engine.get_execution_status(execution_id)
		if execution and execution.status != PipelineStatus.RUNNING:
			break
		await asyncio.sleep(0.01)

	assert execution is not None
	assert execution.status == PipelineStatus.FAILED
	assert any("requires an environment" in log["message"] for log in execution.logs)
