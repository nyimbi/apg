"""DeviceEnrolmentSaga — mob_mdm → auth → mten → ntfy."""
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
class DeviceEnrolmentSagaInput:
    device_id: str
    tenant_id: str
    user_id: str
    device_type: str        # "mobile" | "desktop" | "iot"
    platform: str           # "android" | "ios" | "windows" | "linux"
    serial_number: str = ""
    policy_group: str = "standard"


@dataclass
class DeviceEnrolmentSagaResult:
    status: str             # "enrolled" | "rejected" | "pending_approval"
    device_id: str
    mdm_device_id: str | None = None
    auth_token: str | None = None
    tenant_profile_id: str | None = None
    notification_sent: bool = False
    rejection_reason: str | None = None


async def _register_device_mdm(device_id: str, user_id: str, device_type: str, platform: str, serial: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Registering device %s in MDM", device_id)
    return {"mdm_device_id": f"mdm-{device_id}", "compliant": True, "policy_applied": True}


async def _issue_device_certificate(device_id: str, user_id: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Issuing device certificate for %s", device_id)
    return {"auth_token": f"tok-{device_id}", "certificate_id": f"cert-{device_id}", "expires_at": "2026-12-31"}


async def _apply_tenant_profile(device_id: str, user_id: str, policy_group: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Applying tenant profile to device %s (group=%s)", device_id, policy_group)
    return {"tenant_profile_id": f"prof-{device_id}", "apps_pushed": 5, "restrictions_applied": True}


async def _send_enrolment_notification(user_id: str, device_id: str, status: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Sending enrolment notification to user %s for device %s", user_id, device_id)
    return {"sent": True, "channel": "email"}


async def _revoke_device(device_id: str, tenant_id: str, reason: str) -> None:
    _log.warning("Revoking device %s — %s", device_id, reason)


if _TEMPORAL_AVAILABLE:
    _mdm_act = activity.defn(name="apg_register_device_mdm")(_register_device_mdm)
    _cert_act = activity.defn(name="apg_issue_device_certificate")(_issue_device_certificate)
    _profile_act = activity.defn(name="apg_apply_tenant_profile")(_apply_tenant_profile)
    _notify_act = activity.defn(name="apg_enrolment_notification")(_send_enrolment_notification)
    _revoke_act = activity.defn(name="apg_revoke_device")(_revoke_device)
    device_activities = [_mdm_act, _cert_act, _profile_act, _notify_act, _revoke_act]
    _OPTS = {"retry_policy": RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=2)), "start_to_close_timeout": timedelta(seconds=60)}

    @workflow.defn(name="DeviceEnrolmentSaga")
    class DeviceEnrolmentSaga:
        @workflow.run
        async def run(self, inp: DeviceEnrolmentSagaInput) -> DeviceEnrolmentSagaResult:
            mdm_registered = False
            try:
                mdm = await workflow.execute_activity(_mdm_act, inp.device_id, inp.user_id, inp.device_type, inp.platform, inp.serial_number, inp.tenant_id, **_OPTS)
                mdm_registered = True

                if not mdm.get("compliant"):
                    await workflow.execute_activity(_notify_act, inp.user_id, inp.device_id, "rejected", inp.tenant_id, **_OPTS)
                    return DeviceEnrolmentSagaResult(status="rejected", device_id=inp.device_id, mdm_device_id=mdm.get("mdm_device_id"), rejection_reason="Device non-compliant", notification_sent=True)

                import asyncio as _asyncio
                cert, profile = await _asyncio.gather(
                    workflow.execute_activity(_cert_act, inp.device_id, inp.user_id, inp.tenant_id, **_OPTS),
                    workflow.execute_activity(_profile_act, inp.device_id, inp.user_id, inp.policy_group, inp.tenant_id, **_OPTS),
                )
                notif = await workflow.execute_activity(_notify_act, inp.user_id, inp.device_id, "enrolled", inp.tenant_id, **_OPTS)
                return DeviceEnrolmentSagaResult(status="enrolled", device_id=inp.device_id, mdm_device_id=mdm.get("mdm_device_id"), auth_token=cert.get("auth_token"), tenant_profile_id=profile.get("tenant_profile_id"), notification_sent=notif.get("sent", False))
            except Exception as exc:
                if mdm_registered:
                    await workflow.execute_activity(_revoke_act, inp.device_id, inp.tenant_id, str(exc), **_OPTS)
                return DeviceEnrolmentSagaResult(status="rejected", device_id=inp.device_id, rejection_reason=str(exc))
else:
    device_activities = []
    class DeviceEnrolmentSaga:  # type: ignore[no-redef]
        async def run(self, inp: DeviceEnrolmentSagaInput) -> DeviceEnrolmentSagaResult:
            raise RuntimeError("temporalio not installed")
