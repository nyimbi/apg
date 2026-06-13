"""ComplianceReportingSaga — grc_pol → grc_aud → grc_rcm → fin_rpt.

Triggered on period close. Collects policy status, audit findings,
risk control matrix, and generates the financial compliance report.
Human approval gate: the CFO/compliance officer must signal approval
before the report is marked final and published.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
import logging

_log = logging.getLogger(__name__)

try:
    from temporalio import workflow, activity
    from temporalio.common import RetryPolicy
    _TEMPORAL_AVAILABLE = True
except ImportError:
    _TEMPORAL_AVAILABLE = False


@dataclass
class ComplianceSagaInput:
    period_id: str          # e.g. "2025-Q4"
    tenant_id: str
    report_type: str = "quarterly"     # "quarterly" | "annual" | "regulatory"
    frameworks: list[str] = field(default_factory=lambda: ["IFRS", "SOC2"])
    approver_user_id: str = ""


@dataclass
class ComplianceSagaResult:
    status: str             # "approved" | "pending_approval" | "failed"
    period_id: str
    report_id: str | None = None
    policy_gaps: int = 0
    audit_findings: int = 0
    risk_items: int = 0
    approved_by: str | None = None


async def _collect_policy_status(period_id: str, tenant_id: str, frameworks: list[str]) -> dict[str, Any]:
    _log.info("Collecting policy status for period %s", period_id)
    return {"policy_gaps": 2, "compliant_policies": 47, "total_policies": 49, "frameworks": frameworks}


async def _run_audit_collection(period_id: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Collecting audit findings for period %s", period_id)
    return {"findings": 3, "critical": 0, "high": 1, "medium": 2, "low": 0}


async def _assess_risk_controls(period_id: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Assessing risk control matrix for period %s", period_id)
    return {"risk_items": 5, "mitigated": 4, "residual_risk": "low"}


async def _generate_report(period_id: str, policy: dict, audit: dict, risk: dict, report_type: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Generating compliance report for period %s", period_id)
    return {"report_id": f"rpt-{period_id}-{report_type}", "status": "draft", "pages": 24}


async def _publish_report(report_id: str, approved_by: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Publishing report %s approved by %s", report_id, approved_by)
    return {"published": True, "report_id": report_id}


if _TEMPORAL_AVAILABLE:
    _policy_act = activity.defn(name="apg_collect_policy_status")(_collect_policy_status)
    _audit_act = activity.defn(name="apg_run_audit_collection")(_run_audit_collection)
    _risk_act = activity.defn(name="apg_assess_risk_controls")(_assess_risk_controls)
    _report_act = activity.defn(name="apg_generate_compliance_report")(_generate_report)
    _publish_act = activity.defn(name="apg_publish_compliance_report")(_publish_report)
    compliance_activities = [_policy_act, _audit_act, _risk_act, _report_act, _publish_act]
    _OPTS = {"retry_policy": RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=2)), "start_to_close_timeout": timedelta(seconds=120)}

    @workflow.defn(name="ComplianceReportingSaga")
    class ComplianceReportingSaga:
        def __init__(self) -> None:
            self._approval_signal: str | None = None

        @workflow.signal(name="approve_report")
        def approve_report(self, approver_user_id: str) -> None:
            self._approval_signal = approver_user_id

        @workflow.run
        async def run(self, inp: ComplianceSagaInput) -> ComplianceSagaResult:
            import asyncio as _asyncio
            policy, audit, risk = await _asyncio.gather(
                workflow.execute_activity(_policy_act, inp.period_id, inp.tenant_id, inp.frameworks, **_OPTS),
                workflow.execute_activity(_audit_act, inp.period_id, inp.tenant_id, **_OPTS),
                workflow.execute_activity(_risk_act, inp.period_id, inp.tenant_id, **_OPTS),
            )
            report = await workflow.execute_activity(_report_act, inp.period_id, policy, audit, risk, inp.report_type, inp.tenant_id, **_OPTS)

            # Human approval gate — wait up to 7 days for CFO/compliance officer signal
            approved = await workflow.wait_condition(lambda: self._approval_signal is not None, timeout=timedelta(days=7))
            if not approved or not self._approval_signal:
                return ComplianceSagaResult(status="pending_approval", period_id=inp.period_id, report_id=report.get("report_id"), policy_gaps=policy.get("policy_gaps", 0), audit_findings=audit.get("findings", 0), risk_items=risk.get("risk_items", 0))

            await workflow.execute_activity(_publish_act, report["report_id"], self._approval_signal, inp.tenant_id, **_OPTS)
            return ComplianceSagaResult(status="approved", period_id=inp.period_id, report_id=report.get("report_id"), policy_gaps=policy.get("policy_gaps", 0), audit_findings=audit.get("findings", 0), risk_items=risk.get("risk_items", 0), approved_by=self._approval_signal)
else:
    compliance_activities = []
    class ComplianceReportingSaga:  # type: ignore[no-redef]
        async def run(self, inp: ComplianceSagaInput) -> ComplianceSagaResult:
            raise RuntimeError("temporalio not installed")
