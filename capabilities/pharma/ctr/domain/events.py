"""Domain events for Clinical Trials Management.

Events are emitted to the capability event stream via Bytewax whenever state
changes occur. Subscribe to these events for integration, auditing, and
downstream capability composition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DomainEvent:
	"""Base class for all Clinical Trials Management domain events."""
	event_type: str
	tenant_id: str
	actor_id: str
	timestamp: datetime = field(default_factory=datetime.utcnow)
	payload: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"event_type": self.event_type,
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"timestamp": self.timestamp.isoformat(),
			"payload": self.payload,
			"capability_id": "pharma_ctr",
		}


@dataclass(frozen=True)
class TrialCreated(DomainEvent):
	event_type: str = "trial_created"


@dataclass(frozen=True)
class TrialActivated(DomainEvent):
	event_type: str = "trial_activated"


@dataclass(frozen=True)
class TrialStatusChanged(DomainEvent):
	event_type: str = "trial_status_changed"


@dataclass(frozen=True)
class ProtocolCreated(DomainEvent):
	event_type: str = "protocol_created"


@dataclass(frozen=True)
class ProtocolApproved(DomainEvent):
	event_type: str = "protocol_approved"


@dataclass(frozen=True)
class AmendmentCreated(DomainEvent):
	event_type: str = "amendment_created"


@dataclass(frozen=True)
class AmendmentApproved(DomainEvent):
	event_type: str = "amendment_approved"


@dataclass(frozen=True)
class SiteSelected(DomainEvent):
	event_type: str = "site_selected"


@dataclass(frozen=True)
class SiteInitiated(DomainEvent):
	event_type: str = "site_initiated"


@dataclass(frozen=True)
class SiteClosed(DomainEvent):
	event_type: str = "site_closed"


@dataclass(frozen=True)
class PatientEnrolled(DomainEvent):
	event_type: str = "patient_enrolled"


@dataclass(frozen=True)
class PatientRandomised(DomainEvent):
	event_type: str = "patient_randomised"


@dataclass(frozen=True)
class PatientWithdrawn(DomainEvent):
	event_type: str = "patient_withdrawn"


@dataclass(frozen=True)
class InformedConsentRecorded(DomainEvent):
	event_type: str = "informed_consent_recorded"


@dataclass(frozen=True)
class CrfDataCollected(DomainEvent):
	event_type: str = "crf_data_collected"


@dataclass(frozen=True)
class CrfValidated(DomainEvent):
	event_type: str = "crf_validated"


@dataclass(frozen=True)
class DataQueryRaised(DomainEvent):
	event_type: str = "data_query_raised"


@dataclass(frozen=True)
class DataQueryClosed(DomainEvent):
	event_type: str = "data_query_closed"


@dataclass(frozen=True)
class DatabaseLocked(DomainEvent):
	event_type: str = "data_locked"


@dataclass(frozen=True)
class AdverseEventReported(DomainEvent):
	event_type: str = "adverse_event_reported"


@dataclass(frozen=True)
class SusarReported(DomainEvent):
	event_type: str = "susar_reported"


@dataclass(frozen=True)
class SarFiled(DomainEvent):
	event_type: str = "sar_filed"


@dataclass(frozen=True)
class MonitoringVisitCompleted(DomainEvent):
	event_type: str = "monitoring_visit_completed"


@dataclass(frozen=True)
class InspectionCreated(DomainEvent):
	event_type: str = "inspection_created"


@dataclass(frozen=True)
class TmfDocumentFiled(DomainEvent):
	event_type: str = "tmf_document_filed"


@dataclass(frozen=True)
class RegulatorySubmissionFiled(DomainEvent):
	event_type: str = "submission_filed"


@dataclass(frozen=True)
class IrbDecisionRecorded(DomainEvent):
	event_type: str = "irb_decision_recorded"


@dataclass(frozen=True)
class InterimAnalysisCompleted(DomainEvent):
	event_type: str = "interim_analysis_completed"


@dataclass(frozen=True)
class TrialClosed(DomainEvent):
	event_type: str = "trial_closed"


@dataclass(frozen=True)
class ProtocolDeviationRecorded(DomainEvent):
	event_type: str = "protocol_deviation_recorded"
