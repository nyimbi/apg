"""IncidentResponseSaga — intel_alerts → intel_correlation → intel_threats → ntfy."""
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
class IncidentSagaInput:
    alert_id: str
    tenant_id: str
    severity: str           # "critical" | "high" | "medium" | "low"
    alert_type: str
    source_capability: str
    payload: dict = field(default_factory=dict)
    notify_channels: list[str] = field(default_factory=list)


@dataclass
class IncidentSagaResult:
    status: str             # "resolved" | "escalated" | "false_positive"
    alert_id: str
    incident_id: str | None = None
    correlation_id: str | None = None
    threat_level: str = "unknown"
    notifications_sent: int = 0
    assigned_to: str | None = None


async def _correlate_alert(alert_id: str, alert_type: str, tenant_id: str, payload: dict) -> dict[str, Any]:
    _log.info("Correlating alert %s", alert_id)
    return {"correlation_id": f"corr-{alert_id}", "related_alerts": [], "is_new_incident": True}


async def _assess_threat(alert_id: str, correlation_id: str, alert_type: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Assessing threat for alert %s", alert_id)
    return {"threat_level": "medium", "confidence": 0.72, "recommended_action": "investigate"}


async def _create_incident(alert_id: str, correlation_id: str, threat_level: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Creating incident for alert %s", alert_id)
    return {"incident_id": f"inc-{alert_id}", "assigned_to": "soc-team"}


async def _send_notifications(incident_id: str, severity: str, channels: list[str], tenant_id: str) -> dict[str, Any]:
    _log.info("Sending notifications for incident %s via %s", incident_id, channels)
    return {"notifications_sent": len(channels), "channels": channels}


async def _close_false_positive(alert_id: str, tenant_id: str) -> None:
    _log.info("Closing false positive alert %s", alert_id)


if _TEMPORAL_AVAILABLE:
    _correlate_act = activity.defn(name="apg_correlate_alert")(_correlate_alert)
    _assess_act = activity.defn(name="apg_assess_threat")(_assess_threat)
    _create_inc_act = activity.defn(name="apg_create_incident")(_create_incident)
    _notify_act = activity.defn(name="apg_send_incident_notifications")(_send_notifications)
    _close_fp_act = activity.defn(name="apg_close_false_positive")(_close_false_positive)
    incident_activities = [_correlate_act, _assess_act, _create_inc_act, _notify_act, _close_fp_act]
    _OPTS = {"retry_policy": RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=1)), "start_to_close_timeout": timedelta(seconds=30)}

    @workflow.defn(name="IncidentResponseSaga")
    class IncidentResponseSaga:
        @workflow.run
        async def run(self, inp: IncidentSagaInput) -> IncidentSagaResult:
            corr = await workflow.execute_activity(_correlate_act, inp.alert_id, inp.alert_type, inp.tenant_id, inp.payload, **_OPTS)

            if not corr.get("is_new_incident"):
                return IncidentSagaResult(status="false_positive", alert_id=inp.alert_id, correlation_id=corr.get("correlation_id"))

            threat = await workflow.execute_activity(_assess_act, inp.alert_id, corr["correlation_id"], inp.alert_type, inp.tenant_id, **_OPTS)

            if threat.get("confidence", 0) < 0.3:
                await workflow.execute_activity(_close_fp_act, inp.alert_id, inp.tenant_id, **_OPTS)
                return IncidentSagaResult(status="false_positive", alert_id=inp.alert_id, threat_level=threat.get("threat_level", "unknown"))

            inc = await workflow.execute_activity(_create_inc_act, inp.alert_id, corr["correlation_id"], threat["threat_level"], inp.tenant_id, **_OPTS)
            channels = inp.notify_channels or (["pager"] if inp.severity == "critical" else ["slack"])
            notifs = await workflow.execute_activity(_notify_act, inc["incident_id"], inp.severity, channels, inp.tenant_id, **_OPTS)

            return IncidentSagaResult(
                status="escalated" if inp.severity in ("critical", "high") else "resolved",
                alert_id=inp.alert_id,
                incident_id=inc.get("incident_id"),
                correlation_id=corr.get("correlation_id"),
                threat_level=threat.get("threat_level", "medium"),
                notifications_sent=notifs.get("notifications_sent", 0),
                assigned_to=inc.get("assigned_to"),
            )
else:
    incident_activities = []
    class IncidentResponseSaga:  # type: ignore[no-redef]
        async def run(self, inp: IncidentSagaInput) -> IncidentSagaResult:
            raise RuntimeError("temporalio not installed")
