"""Flask-AppBuilder compatible views and Pydantic schema re-exports for Payment USSD App."""
from __future__ import annotations

from typing import Any

from flask_appbuilder import ModelView
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .models import (
	PayBillCreate,
	PayBillUpdate,
	PayBillResponse,
	PayMerchantCreate,
	PayMerchantResponse,
	PayAirtimeCreate,
	PayAirtimeResponse,
	PayUtilityCreate,
	PayUtilityResponse,
	PaySendMoneyCreate,
	PaySendMoneyConfirmation,
	PaySendMoneyResponse,
	PayBillerCreate,
	PayBillerResponse,
	PayPaymentFilter,
	PayAuditEvent,
	PayUssdSessionCreate,
	PayUssdSessionResponse,
)

# Re-export all models
__all__ = [
	"PayBillCreate",
	"PayBillUpdate",
	"PayBillResponse",
	"PayMerchantCreate",
	"PayMerchantResponse",
	"PayAirtimeCreate",
	"PayAirtimeResponse",
	"PayUtilityCreate",
	"PayUtilityResponse",
	"PaySendMoneyCreate",
	"PaySendMoneyConfirmation",
	"PaySendMoneyResponse",
	"PayBillerCreate",
	"PayBillerResponse",
	"PayPaymentFilter",
	"PayAuditEvent",
	"PayUssdSessionCreate",
	"PayUssdSessionResponse",
	"PayBillModelView",
	"PayMerchantModelView",
	"PayAirtimeModelView",
	"PayUtilityModelView",
	"PaySendMoneyModelView",
	"PayBillerModelView",
]


class PayBillModelView(ModelView):
	"""Flask-AppBuilder view for bill payments."""
	datamodel = SQLAInterface(None)  # type: ignore[arg-type]
	list_columns = ["id", "phone_number", "biller_name", "account_reference", "amount", "status", "created_at"]
	show_columns = ["id", "phone_number", "biller_code", "biller_name", "paybill_number",
					"account_reference", "amount", "currency", "narration", "receipt_number", "status", "created_at"]
	search_columns = ["phone_number", "biller_code", "biller_name", "account_reference", "status"]
	label_columns = {
		"phone_number": "Phone Number",
		"biller_name": "Biller Name",
		"account_reference": "Account Reference",
		"amount": "Amount",
		"receipt_number": "Receipt Number",
		"status": "Status",
	}


class PayMerchantModelView(ModelView):
	"""Flask-AppBuilder view for merchant payments."""
	datamodel = SQLAInterface(None)  # type: ignore[arg-type]
	list_columns = ["id", "phone_number", "merchant_till", "merchant_name", "amount", "status", "created_at"]
	show_columns = ["id", "phone_number", "merchant_till", "merchant_name", "amount", "currency",
					"narration", "receipt_number", "status", "created_at"]
	search_columns = ["phone_number", "merchant_till", "merchant_name", "status"]
	label_columns = {
		"phone_number": "Phone Number",
		"merchant_till": "Merchant Till",
		"merchant_name": "Merchant Name",
		"amount": "Amount",
		"receipt_number": "Receipt Number",
	}


class PayAirtimeModelView(ModelView):
	"""Flask-AppBuilder view for airtime top-ups."""
	datamodel = SQLAInterface(None)  # type: ignore[arg-type]
	list_columns = ["id", "phone_number", "recipient_phone", "telco", "amount", "status", "created_at"]
	show_columns = ["id", "phone_number", "recipient_phone", "telco", "amount", "currency",
					"receipt_number", "status", "created_at"]
	search_columns = ["phone_number", "recipient_phone", "telco", "status"]
	label_columns = {
		"phone_number": "Buyer Phone",
		"recipient_phone": "Recipient Phone",
		"telco": "Telco",
		"amount": "Amount",
	}


class PayUtilityModelView(ModelView):
	"""Flask-AppBuilder view for utility payments."""
	datamodel = SQLAInterface(None)  # type: ignore[arg-type]
	list_columns = ["id", "phone_number", "utility_name", "meter_number", "amount", "units_purchased", "status", "created_at"]
	show_columns = ["id", "phone_number", "utility_code", "utility_name", "meter_number",
					"amount", "currency", "units_purchased", "token", "receipt_number", "status", "created_at"]
	search_columns = ["phone_number", "utility_code", "meter_number", "status"]
	label_columns = {
		"phone_number": "Phone Number",
		"utility_name": "Utility",
		"meter_number": "Meter Number",
		"units_purchased": "Units",
		"token": "Prepaid Token",
	}


class PaySendMoneyModelView(ModelView):
	"""Flask-AppBuilder view for send money transactions."""
	datamodel = SQLAInterface(None)  # type: ignore[arg-type]
	list_columns = ["id", "from_phone", "to_phone", "amount", "status", "requires_confirmation", "created_at"]
	show_columns = ["id", "from_phone", "to_phone", "amount", "currency",
					"narration", "receipt_number", "status", "requires_confirmation", "created_at", "confirmed_at"]
	search_columns = ["from_phone", "to_phone", "status"]
	label_columns = {
		"from_phone": "From Phone",
		"to_phone": "To Phone",
		"amount": "Amount",
		"requires_confirmation": "Requires Confirmation",
		"confirmed_at": "Confirmed At",
	}


class PayBillerModelView(ModelView):
	"""Flask-AppBuilder view for biller registry."""
	datamodel = SQLAInterface(None)  # type: ignore[arg-type]
	list_columns = ["id", "biller_code", "biller_name", "category", "paybill_number", "status"]
	show_columns = ["id", "biller_code", "biller_name", "category", "paybill_number",
					"account_mask", "min_amount", "max_amount", "status", "created_at"]
	search_columns = ["biller_code", "biller_name", "category", "paybill_number", "status"]
	label_columns = {
		"biller_code": "Biller Code",
		"biller_name": "Biller Name",
		"category": "Category",
		"paybill_number": "Paybill Number",
		"min_amount": "Min Amount",
		"max_amount": "Max Amount",
	}


def get_bill_response_schema() -> dict[str, Any]:
	"""Return JSON schema for bill payment response."""
	return PayBillResponse.model_json_schema()


def get_send_money_response_schema() -> dict[str, Any]:
	"""Return JSON schema for send money response."""
	return PaySendMoneyResponse.model_json_schema()
