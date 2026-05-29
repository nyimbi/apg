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
