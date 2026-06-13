"""APG cross-capability Temporal saga workflows.

Each saga coordinates activities across multiple capabilities with:
- Durable execution (survives process restarts)
- Automatic retries with exponential backoff
- Compensation (rollback) on failure
- Human approval gates via signals

Available sagas:
    PaymentProcessingSaga   — gateway → kyc → aml → fraud → ledger
    CustomerOnboardingSaga  — crm_adv → kyc → aml → auth
    IncidentResponseSaga    — alerts → correlation → threats → ntfy
    ComplianceReportingSaga — grc_pol → grc_aud → grc_rcm → fin_rpt
    DeviceEnrolmentSaga     — mob_mdm → auth → mten → ntfy

Usage::

    from capabilities.composition.workflows import get_workflow_client

    client = await get_workflow_client()
    handle = await client.start_workflow(
        PaymentProcessingSaga.run,
        PaymentSagaInput(payment_id="pay-123", tenant_id="acme", amount=5000.0),
        id=f"payment-{payment_id}",
        task_queue="apg-sagas",
    )
    result = await handle.result()
"""
from .payment_processing import PaymentProcessingSaga, PaymentSagaInput
from .customer_onboarding import CustomerOnboardingSaga, OnboardingSagaInput
from .incident_response import IncidentResponseSaga, IncidentSagaInput
from .compliance_reporting import ComplianceReportingSaga, ComplianceSagaInput
from .device_enrolment import DeviceEnrolmentSaga, DeviceEnrolmentSagaInput
from .client import get_workflow_client, run_saga_worker

__all__ = [
    "PaymentProcessingSaga", "PaymentSagaInput",
    "CustomerOnboardingSaga", "OnboardingSagaInput",
    "IncidentResponseSaga", "IncidentSagaInput",
    "ComplianceReportingSaga", "ComplianceSagaInput",
    "DeviceEnrolmentSaga", "DeviceEnrolmentSagaInput",
    "get_workflow_client", "run_saga_worker",
]
