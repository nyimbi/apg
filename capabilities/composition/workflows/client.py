"""Temporal client + worker bootstrap for APG saga workflows."""
from __future__ import annotations

import os
import logging
from typing import Any

_log = logging.getLogger(__name__)

TEMPORAL_HOST = os.environ.get("TEMPORAL_HOST", "localhost:7233")
APG_SAGAS_QUEUE = "apg-sagas"


async def get_workflow_client() -> Any:
    """Return a connected Temporal client. Raises if TEMPORAL_HOST unreachable."""
    try:
        from temporalio.client import Client
        client = await Client.connect(TEMPORAL_HOST)
        _log.info("Connected to Temporal at %s", TEMPORAL_HOST)
        return client
    except ImportError:
        raise RuntimeError(
            "temporalio package not installed. "
            "Add 'temporalio' to your project dependencies."
        )


async def run_saga_worker() -> None:
    """Start a Temporal worker that handles all APG saga workflows and activities."""
    try:
        from temporalio.worker import Worker
        from .payment_processing import PaymentProcessingSaga, payment_activities
        from .customer_onboarding import CustomerOnboardingSaga, onboarding_activities
        from .incident_response import IncidentResponseSaga, incident_activities
        from .compliance_reporting import ComplianceReportingSaga, compliance_activities
        from .device_enrolment import DeviceEnrolmentSaga, device_activities
    except ImportError:
        raise RuntimeError("temporalio package not installed.")

    client = await get_workflow_client()
    worker = Worker(
        client,
        task_queue=APG_SAGAS_QUEUE,
        workflows=[
            PaymentProcessingSaga,
            CustomerOnboardingSaga,
            IncidentResponseSaga,
            ComplianceReportingSaga,
            DeviceEnrolmentSaga,
        ],
        activities=(
            payment_activities
            + onboarding_activities
            + incident_activities
            + compliance_activities
            + device_activities
        ),
    )
    _log.info("APG saga worker starting on queue '%s'", APG_SAGAS_QUEUE)
    await worker.run()
