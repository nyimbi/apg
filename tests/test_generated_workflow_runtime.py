"""Generated workflow execution runtime coverage."""

from __future__ import annotations

from compiler.compiler import APGCompiler


WORKFLOW_SOURCE = """
module procurement_flow version 1.0.0 {
    description: "Workflow runtime coverage";
}

workflow ProcurementApproval {
    steps: str = "draft -> budget_review -> procurement_review -> finance_approval -> approved";
    stages: str = "draft, review, approval, complete";
    process: () -> bool = {
        return true;
    };
}
"""


ADVANCED_WORKFLOW_SOURCE = """
module governed_procurement_flow version 1.0.0 {
    description: "Governed workflow runtime coverage";
}

workflow GovernedProcurement {
    steps: str = "draft -> budget_review -> procurement_review -> finance_approval -> approved";
    guards: dict = {"budget_review": "amount <= budget_limit", "finance_approval": "amount > finance_threshold"};
    assignments: dict = {"budget_review": "budget_owner", "finance_approval": "finance_controller"};
    human_tasks: str = "budget_review, finance_approval";
    timers: dict = {"finance_approval": "PT24H"};
    waits: dict = {"finance_approval": "finance_packet_ready"};
    retry_policy: dict = {"procurement_review": "3"};
    compensation: dict = {"procurement_review": "release_budget_hold"};
    process: () -> bool = {
        return true;
    };
}
"""


def test_generated_app_executes_declared_workflow_steps():
    result = APGCompiler().compile_string(WORKFLOW_SOURCE, "procurement_flow.apg")
    assert result.success is True, result.errors

    namespace: dict[str, object] = {}
    exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

    assert namespace["list_workflows"]() == ["ProcurementApproval"]

    workflow = namespace["describe_workflow"]("ProcurementApproval")
    assert workflow["steps"] == [
        "draft",
        "budget_review",
        "procurement_review",
        "finance_approval",
        "approved",
    ]
    assert workflow["transitions"][0] == {"from": "draft", "to": "budget_review"}

    run = namespace["run_workflow"]("ProcurementApproval", {"request_id": "PR-100"})
    assert run["status"] == "completed"
    assert run["completed_at"] == "approved"
    assert [step["step"] for step in run["trace"]] == workflow["steps"]
    assert namespace["list_events"]("ProcurementApproval")[-1]["action"] == "workflow.run"

    status, payload = namespace["_route_payload"]("/workflows/ProcurementApproval")
    assert status == 200
    assert payload["name"] == "ProcurementApproval"

    status, payload = namespace["_post_payload"](
        "/workflows/ProcurementApproval/run",
        {"payload": {"request_id": "PR-101", "start_at": "procurement_review"}},
    )
    assert status == 200
    assert payload["steps"] == ["procurement_review", "finance_approval", "approved"]

    openapi = namespace["openapi_document"]()
    assert "/workflows" in openapi["paths"]
    assert "/workflows/ProcurementApproval" in openapi["paths"]
    assert "/workflows/ProcurementApproval/run" in openapi["paths"]
    assert namespace["component_manifest"]()["workflows"]["ProcurementApproval"]["steps"][0] == "draft"
    assert namespace["validate_application"]()["checks"]["workflows"]["errors"] == []


def test_generated_app_persists_and_resumes_workflow_runs(tmp_path, monkeypatch):
    result = APGCompiler().compile_string(WORKFLOW_SOURCE, "procurement_flow.apg")
    assert result.success is True, result.errors

    data_file = tmp_path / "apg-data.json"
    monkeypatch.setenv("APG_DATA_FILE", str(data_file))

    namespace: dict[str, object] = {}
    exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

    run = namespace["run_workflow"](
        "ProcurementApproval",
        {"request_id": "PR-200", "pause_at": "budget_review"},
    )
    assert run["id"] == "workflow-run-1"
    assert run["status"] == "paused"
    assert run["current_step"] == "budget_review"
    assert run["completed_at"] is None
    assert run["pending_steps"] == ["procurement_review", "finance_approval", "approved"]
    assert data_file.exists()

    assert namespace["get_workflow_run"]("workflow-run-1")["status"] == "paused"
    assert namespace["list_workflow_runs"]("ProcurementApproval")[0]["id"] == "workflow-run-1"

    status, payload = namespace["_route_payload"]("/workflows/runs")
    assert status == 200
    assert payload["runs"][0]["id"] == "workflow-run-1"

    status, payload = namespace["_route_payload"]("/workflows/runs/workflow-run-1")
    assert status == 200
    assert payload["current_step"] == "budget_review"

    status, resumed = namespace["_post_payload"](
        "/workflows/runs/workflow-run-1/resume",
        {"payload": {"reviewer": "finance"}, "pause_at": "finance_approval"},
    )
    assert status == 200
    assert resumed["status"] == "paused"
    assert resumed["current_step"] == "finance_approval"
    assert resumed["pending_steps"] == ["approved"]

    completed = namespace["resume_workflow"]("workflow-run-1")
    assert completed["status"] == "completed"
    assert completed["completed_at"] == "approved"
    assert completed["completed_steps"] == [
        "draft",
        "budget_review",
        "procurement_review",
        "finance_approval",
        "approved",
    ]
    assert [event["action"] for event in namespace["list_events"]("ProcurementApproval")] == [
        "workflow.run",
        "workflow.resume",
        "workflow.resume",
    ]
    assert namespace["storage_status"](include_records=True)["workflow_runs"][0]["status"] == "completed"

    reloaded: dict[str, object] = {}
    exec(compile(result.generated_files["app.py"], "app.py", "exec"), reloaded)
    assert reloaded["get_workflow_run"]("workflow-run-1")["status"] == "completed"
    assert reloaded["list_events"]("ProcurementApproval")[-1]["action"] == "workflow.resume"

    openapi = namespace["openapi_document"]()
    assert "/workflows/runs" in openapi["paths"]
    assert "/workflows/runs/{id}" in openapi["paths"]
    assert "/workflows/runs/{id}/resume" in openapi["paths"]
    assert "resume_workflow" in namespace["component_manifest"]()["interfaces"]["python"]["exports"]


def test_generated_app_executes_workflow_guards_and_task_metadata():
    result = APGCompiler().compile_string(ADVANCED_WORKFLOW_SOURCE, "governed_procurement_flow.apg")
    assert result.success is True, result.errors

    namespace: dict[str, object] = {}
    exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

    workflow = namespace["describe_workflow"]("GovernedProcurement")
    assert workflow["guards"]["budget_review"] == "amount <= budget_limit"
    assert workflow["assignments"]["finance_approval"] == "finance_controller"
    assert workflow["human_tasks"] == ["budget_review", "finance_approval"]
    assert workflow["timers"]["finance_approval"] == "PT24H"
    assert workflow["waits"]["finance_approval"] == "finance_packet_ready"
    assert workflow["retry_policy"]["procurement_review"] == "3"
    assert workflow["compensation"]["procurement_review"] == "release_budget_hold"
    assert workflow["transitions"][0]["guard"] == "amount <= budget_limit"

    blocked = namespace["run_workflow"](
        "GovernedProcurement",
        {"amount": 7500, "budget_limit": 5000, "finance_threshold": 1000},
    )
    assert blocked["status"] == "blocked"
    assert blocked["blocked_at"] == "budget_review"
    assert blocked["pending_steps"][0] == "budget_review"
    assert blocked["compensations"] == []
    assert blocked["trace"][-1]["guard_passed"] is False
    assert blocked["trace"][-1]["assignee"] == "budget_owner"
    assert blocked["trace"][-1]["task_type"] == "human"

    waiting = namespace["run_workflow"](
        "GovernedProcurement",
        {"amount": 4500, "budget_limit": 5000, "finance_threshold": 1000},
    )
    assert waiting["status"] == "waiting"
    assert waiting["waiting_at"] == "finance_approval"
    assert waiting["waiting_for"] == "finance_packet_ready"
    assert waiting["pending_steps"][0] == "finance_approval"
    assert waiting["trace"][-1]["status"] == "waiting"

    failed = namespace["run_workflow"](
        "GovernedProcurement",
        {
            "amount": 4500,
            "budget_limit": 5000,
            "finance_threshold": 1000,
            "events": ["finance_packet_ready"],
            "step_failures": {"procurement_review": 3},
        },
    )
    assert failed["status"] == "failed"
    assert failed["failed_at"] == "procurement_review"
    assert [attempt["status"] for attempt in failed["attempts"]] == ["failed", "failed", "failed"]
    assert failed["compensations"] == []

    compensated = namespace["run_workflow"](
        "GovernedProcurement",
        {
            "amount": 4500,
            "budget_limit": 5000,
            "finance_threshold": 1000,
            "events": ["finance_packet_ready"],
            "fail_steps": ["finance_approval"],
        },
    )
    assert compensated["status"] == "failed"
    assert compensated["failed_at"] == "finance_approval"
    assert compensated["compensations"] == [
        {"step": "procurement_review", "action": "release_budget_hold"}
    ]

    completed = namespace["run_workflow"](
        "GovernedProcurement",
        {
            "amount": 4500,
            "budget_limit": 5000,
            "finance_threshold": 1000,
            "events": ["finance_packet_ready"],
            "step_failures": {"procurement_review": 2},
        },
    )
    assert completed["status"] == "completed"
    assert completed["completed_at"] == "approved"
    finance_step = [step for step in completed["trace"] if step["step"] == "finance_approval"][0]
    assert finance_step["guard_passed"] is True
    assert finance_step["assignee"] == "finance_controller"
    assert finance_step["timer"] == "PT24H"
    assert finance_step["event_received"] == "finance_packet_ready"
    procurement_step = [step for step in completed["trace"] if step["step"] == "procurement_review"][0]
    assert procurement_step["retry_policy"] == "3"
    assert procurement_step["compensation"] == "release_budget_hold"
    assert [attempt["status"] for attempt in procurement_step["attempts"]] == [
        "failed",
        "failed",
        "completed",
    ]
    assert completed["compensations"] == []

    validation = namespace["validate_application"]()
    assert validation["checks"]["workflows"]["errors"] == []
