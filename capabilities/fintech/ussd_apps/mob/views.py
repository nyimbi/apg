"""Flask-AppBuilder compatible views and Pydantic schema re-exports for Mobile Banking USSD."""
from __future__ import annotations

from typing import Any

from flask_appbuilder import ModelView
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .models import (
	MobAccountCreate,
	MobAccountUpdate,
	MobAccountResponse,
	MobAccountListResponse,
	MobAccountFilter,
	MobTransferCreate,
	MobTransferUpdate,
	MobTransferResponse,
	MobStatementEntry,
	MobMiniStatementResponse,
	MobStandingOrderCreate,
	MobStandingOrderUpdate,
	MobStandingOrderResponse,
	MobPinChangeRequest,
	MobPinResetRequest,
	MobAuditEvent,
	MobUssdSessionCreate,
	MobUssdSessionResponse,
)

# Re-export all models for external consumers
__all__ = [
	"MobAccountCreate",
	"MobAccountUpdate",
	"MobAccountResponse",
	"MobAccountListResponse",
	"MobAccountFilter",
	"MobTransferCreate",
	"MobTransferUpdate",
	"MobTransferResponse",
	"MobStatementEntry",
	"MobMiniStatementResponse",
	"MobStandingOrderCreate",
	"MobStandingOrderUpdate",
	"MobStandingOrderResponse",
	"MobPinChangeRequest",
	"MobPinResetRequest",
	"MobAuditEvent",
	"MobUssdSessionCreate",
	"MobUssdSessionResponse",
	"MobAccountModelView",
	"MobTransferModelView",
	"MobStandingOrderModelView",
]


class MobAccountModelView(ModelView):
	"""Flask-AppBuilder view for mobile banking accounts."""
	datamodel = SQLAInterface(None)  # type: ignore[arg-type]  # replaced at runtime
	list_columns = ["id", "account_number", "customer_name", "phone_number", "account_type", "currency", "balance", "status"]
	show_columns = [
		"id", "account_number", "customer_name", "phone_number", "account_type",
		"currency", "balance", "available_balance", "daily_limit", "status", "created_at",
	]
	search_columns = ["account_number", "customer_name", "phone_number", "status"]
	label_columns = {
		"account_number": "Account Number",
		"customer_name": "Customer Name",
		"phone_number": "Phone Number",
		"account_type": "Account Type",
		"balance": "Balance",
		"available_balance": "Available Balance",
		"daily_limit": "Daily Limit",
		"status": "Status",
	}


class MobTransferModelView(ModelView):
	"""Flask-AppBuilder view for fund transfers."""
	datamodel = SQLAInterface(None)  # type: ignore[arg-type]
	list_columns = ["id", "from_account", "to_account", "amount", "currency", "status", "created_at"]
	show_columns = ["id", "from_account", "to_account", "amount", "currency", "narration", "reference", "status", "created_at", "settled_at"]
	search_columns = ["from_account", "to_account", "status", "reference"]
	label_columns = {
		"from_account": "From Account",
		"to_account": "To Account",
		"amount": "Amount",
		"currency": "Currency",
		"narration": "Narration",
		"reference": "Reference",
		"status": "Status",
	}


class MobStandingOrderModelView(ModelView):
	"""Flask-AppBuilder view for standing orders."""
	datamodel = SQLAInterface(None)  # type: ignore[arg-type]
	list_columns = ["id", "from_account", "to_account", "amount", "frequency", "status", "next_execution_date"]
	show_columns = [
		"id", "from_account", "to_account", "amount", "frequency",
		"start_date", "end_date", "next_execution_date", "executions_count", "status",
	]
	search_columns = ["from_account", "to_account", "frequency", "status"]
	label_columns = {
		"from_account": "From Account",
		"to_account": "To Account",
		"amount": "Amount",
		"frequency": "Frequency",
		"start_date": "Start Date",
		"end_date": "End Date",
		"next_execution_date": "Next Execution",
		"executions_count": "Executions",
		"status": "Status",
	}


def get_account_response_schema() -> dict[str, Any]:
	"""Return JSON schema for account response."""
	return MobAccountResponse.model_json_schema()


def get_transfer_response_schema() -> dict[str, Any]:
	"""Return JSON schema for transfer response."""
	return MobTransferResponse.model_json_schema()


def get_standing_order_response_schema() -> dict[str, Any]:
	"""Return JSON schema for standing order response."""
	return MobStandingOrderResponse.model_json_schema()
