"""CustomerOnboardingSaga — crm_adv → fintech_kyc → fintech_aml → auth."""
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
class OnboardingSagaInput:
    customer_id: str
    tenant_id: str
    full_name: str
    email: str
    phone: str
    id_number: str
    country: str = "KE"
    metadata: dict = field(default_factory=dict)


@dataclass
class OnboardingSagaResult:
    status: str             # "completed" | "pending_review" | "rejected"
    customer_id: str
    canonical_entity_id: str | None = None
    kyc_status: str = "pending"
    aml_status: str = "pending"
    auth_user_id: str | None = None
    review_reasons: list[str] = field(default_factory=list)


async def _create_crm_contact(customer_id: str, full_name: str, email: str, phone: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Creating CRM contact %s", customer_id)
    return {"contact_id": customer_id, "canonical_entity_id": f"canonical-{customer_id}"}


async def _submit_kyc(customer_id: str, id_number: str, country: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Submitting KYC for %s", customer_id)
    return {"kyc_status": "cleared", "risk_band": "low", "customer_id": customer_id}


async def _run_aml_check(customer_id: str, full_name: str, country: str, tenant_id: str) -> dict[str, Any]:
    _log.info("AML check for %s", customer_id)
    return {"aml_cleared": True, "watchlist_hits": 0}


async def _provision_auth_user(customer_id: str, email: str, tenant_id: str) -> dict[str, Any]:
    _log.info("Provisioning auth user for %s", customer_id)
    return {"auth_user_id": f"usr-{customer_id}", "provisioned": True}


async def _deactivate_crm_contact(customer_id: str, tenant_id: str, reason: str) -> None:
    _log.warning("Deactivating CRM contact %s — %s", customer_id, reason)


if _TEMPORAL_AVAILABLE:
    _create_contact_act = activity.defn(name="apg_create_crm_contact")(_create_crm_contact)
    _submit_kyc_act = activity.defn(name="apg_submit_kyc")(_submit_kyc)
    _run_aml_act = activity.defn(name="apg_onboard_aml_check")(_run_aml_check)
    _provision_auth_act = activity.defn(name="apg_provision_auth_user")(_provision_auth_user)
    _deactivate_act = activity.defn(name="apg_deactivate_crm_contact")(_deactivate_crm_contact)
    onboarding_activities = [_create_contact_act, _submit_kyc_act, _run_aml_act, _provision_auth_act, _deactivate_act]
    _OPTS = {"retry_policy": RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=2)), "start_to_close_timeout": timedelta(seconds=60)}

    @workflow.defn(name="CustomerOnboardingSaga")
    class CustomerOnboardingSaga:
        @workflow.run
        async def run(self, inp: OnboardingSagaInput) -> OnboardingSagaResult:
            contact_created = False
            try:
                crm = await workflow.execute_activity(_create_contact_act, inp.customer_id, inp.full_name, inp.email, inp.phone, inp.tenant_id, **_OPTS)
                contact_created = True

                kyc, aml = await __import__("asyncio").gather(
                    workflow.execute_activity(_submit_kyc_act, inp.customer_id, inp.id_number, inp.country, inp.tenant_id, **_OPTS),
                    workflow.execute_activity(_run_aml_act, inp.customer_id, inp.full_name, inp.country, inp.tenant_id, **_OPTS),
                )

                reasons = []
                if kyc.get("kyc_status") not in ("cleared", "verified"):
                    reasons.append(f"KYC: {kyc.get('kyc_status')}")
                if not aml.get("aml_cleared"):
                    reasons.append("AML: watchlist match")

                if reasons:
                    return OnboardingSagaResult(
                        status="pending_review", customer_id=inp.customer_id,
                        canonical_entity_id=crm.get("canonical_entity_id"),
                        kyc_status=kyc.get("kyc_status", "pending"),
                        review_reasons=reasons,
                    )

                auth = await workflow.execute_activity(_provision_auth_act, inp.customer_id, inp.email, inp.tenant_id, **_OPTS)
                return OnboardingSagaResult(
                    status="completed", customer_id=inp.customer_id,
                    canonical_entity_id=crm.get("canonical_entity_id"),
                    kyc_status=kyc.get("kyc_status", "cleared"),
                    aml_status="cleared",
                    auth_user_id=auth.get("auth_user_id"),
                )
            except Exception as exc:
                if contact_created:
                    await workflow.execute_activity(_deactivate_act, inp.customer_id, inp.tenant_id, str(exc), **_OPTS)
                return OnboardingSagaResult(status="rejected", customer_id=inp.customer_id, review_reasons=[str(exc)])
else:
    onboarding_activities = []
    class CustomerOnboardingSaga:  # type: ignore[no-redef]
        async def run(self, inp: OnboardingSagaInput) -> OnboardingSagaResult:
            raise RuntimeError("temporalio not installed")
