"""Flask-AppBuilder compatible views and Pydantic schema re-exports for ESS."""
from __future__ import annotations

from typing import Any

from flask_appbuilder import ModelView
from flask_appbuilder.models.sqla.interface import SQLAInterface

# Re-export Pydantic schemas
from .models import (
	ESSLeaveRequestCreate,
	ESSLeaveRequestUpdate,
	ESSLeaveRequestResponse,
	ESSLeaveRequestList,
	ESSLeaveRequestFilter,
	ESSPayslipResponse,
	ESSExpenseClaimCreate,
	ESSExpenseClaimUpdate,
	ESSExpenseClaimResponse,
	ESSBenefitEnrolmentCreate,
	ESSBenefitEnrolmentUpdate,
	ESSBenefitEnrolmentResponse,
	ESSTrainingRegistrationCreate,
	ESSTrainingRegistrationUpdate,
	ESSTrainingRegistrationResponse,
	ESSPersonalDataUpdate,
	ESSPersonalDataResponse,
	ESSAuditEvent,
)

__all__ = [
	"ESSLeaveRequestCreate",
	"ESSLeaveRequestUpdate",
	"ESSLeaveRequestResponse",
	"ESSLeaveRequestList",
	"ESSLeaveRequestFilter",
	"ESSPayslipResponse",
	"ESSExpenseClaimCreate",
	"ESSExpenseClaimUpdate",
	"ESSExpenseClaimResponse",
	"ESSBenefitEnrolmentCreate",
	"ESSBenefitEnrolmentUpdate",
	"ESSBenefitEnrolmentResponse",
	"ESSTrainingRegistrationCreate",
	"ESSTrainingRegistrationUpdate",
	"ESSTrainingRegistrationResponse",
	"ESSPersonalDataUpdate",
	"ESSPersonalDataResponse",
	"ESSAuditEvent",
]


def serialize_record(record: dict[str, Any]) -> dict[str, Any]:
	"""Sanitise a record dict for JSON serialisation."""
	return {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v for k, v in record.items()}
