"""Focused AUDP API job-status contract tests."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from capabilities.common.audp import api


class _WorkflowOrchestrator:
	async def process_complete_workflow(self, **kwargs):
		return {
			"workflow_id": "workflow_status_001",
			"status": "completed",
			"total_processing_time": 1.25,
			"results": {
				"transcription": {"job_id": "transcription_001", "status": "completed"},
				"steps_completed": ["transcription"]
			}
		}

	async def get_workflow_status(self, workflow_id):
		return None


def _client(monkeypatch):
	api.JOB_STATUS_REGISTRY.clear()
	monkeypatch.setattr(api, "create_workflow_orchestrator", lambda: _WorkflowOrchestrator())
	app = FastAPI()
	app.include_router(api.router)
	return TestClient(app)


def test_workflow_execution_records_job_status(monkeypatch):
	client = _client(monkeypatch)
	request_data = {
		"audio_source": {"file_path": "/tmp/status.wav", "format": "wav"},
		"workflow_type": "transcribe_analyze_enhance",
		"parameters": {"enhance_first": False}
	}

	response = client.post(
		"/api/v1/audio/workflows/execute?tenant_id=tenant-a&user_id=user-1",
		json=request_data
	)
	assert response.status_code == 201
	assert response.json()["workflow_id"] == "workflow_status_001"

	status_response = client.get("/api/v1/audio/jobs/workflow_status_001?tenant_id=tenant-a")
	assert status_response.status_code == 200
	status = status_response.json()
	assert status["job_id"] == "workflow_status_001"
	assert status["status"] == "completed"
	assert status["tenant_id"] == "tenant-a"
	assert status["user_id"] == "user-1"
	assert status["workflow_type"] == "transcribe_analyze_enhance"
	assert status["steps_completed"] == ["transcription"]
	assert status["total_processing_time"] == 1.25
	assert status["created_at"]
	assert status["updated_at"]


def test_job_status_is_tenant_scoped(monkeypatch):
	client = _client(monkeypatch)
	request_data = {
		"audio_source": {"file_path": "/tmp/status.wav", "format": "wav"},
		"workflow_type": "transcribe_analyze_enhance"
	}

	client.post(
		"/api/v1/audio/workflows/execute?tenant_id=tenant-a",
		json=request_data
	)

	response = client.get("/api/v1/audio/jobs/workflow_status_001?tenant_id=tenant-b")
	assert response.status_code == 404
	assert response.json()["detail"] == "Job workflow_status_001 not found"


def test_workflow_status_reads_recorded_execution(monkeypatch):
	client = _client(monkeypatch)
	request_data = {
		"audio_source": {"file_path": "/tmp/status.wav", "format": "wav"},
		"workflow_type": "transcribe_analyze_enhance"
	}
	client.post(
		"/api/v1/audio/workflows/execute?tenant_id=tenant-a",
		json=request_data
	)

	response = client.get("/api/v1/audio/workflows/workflow_status_001/status?tenant_id=tenant-a")
	assert response.status_code == 200
	assert response.json()["workflow_id"] == "workflow_status_001"
