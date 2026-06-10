"""Flask-AppBuilder views and re-exported Pydantic schemas for ussd_eng."""

from __future__ import annotations

from .models import (
	UssdAuditEvent,
	UssdCallbackRequest,
	UssdCallbackResponse,
	UssdGatewayCreate,
	UssdGatewayResponse,
	UssdGatewayUpdate,
	UssdMenuCreate,
	UssdMenuResponse,
	UssdMenuUpdate,
	UssdSessionCreate,
	UssdSessionFilter,
	UssdSessionList,
	UssdSessionResponse,
	UssdSessionUpdate,
)

__all__ = [
	"UssdSessionCreate",
	"UssdSessionUpdate",
	"UssdSessionResponse",
	"UssdSessionList",
	"UssdSessionFilter",
	"UssdMenuCreate",
	"UssdMenuUpdate",
	"UssdMenuResponse",
	"UssdGatewayCreate",
	"UssdGatewayUpdate",
	"UssdGatewayResponse",
	"UssdCallbackRequest",
	"UssdCallbackResponse",
	"UssdAuditEvent",
	"UssdEngModelView",
	"UssdSessionView",
	"UssdMenuView",
]

try:
	from flask_appbuilder import ModelView
	from flask_appbuilder.models.sqla.interface import SQLAInterface

	class UssdEngModelView(ModelView):
		"""Base FAB view for USSD Engine entities."""
		datamodel = None  # Set at registration time
		list_title = "USSD Engine"
		show_title = "USSD Engine Detail"
		add_title = "Add USSD Entry"
		edit_title = "Edit USSD Entry"

	class UssdSessionView(UssdEngModelView):
		list_title = "USSD Sessions"
		list_columns = ["id", "tenant_id", "phone_number", "service_code", "session_state", "hop_count", "created_at"]
		show_columns = list_columns + ["variables", "input_history", "menu_history", "ended_at"]
		search_columns = ["phone_number", "service_code", "session_state"]
		label_columns = {
			"id": "Session ID",
			"tenant_id": "Tenant",
			"phone_number": "Phone Number",
			"service_code": "Service Code",
			"session_state": "State",
			"hop_count": "Hops",
			"created_at": "Created",
		}

	class UssdMenuView(UssdEngModelView):
		list_title = "USSD Menus"
		list_columns = ["menu_id", "title", "service_code", "language", "is_end_screen", "created_at"]
		show_columns = list_columns + ["body", "items", "timeout_seconds"]
		search_columns = ["menu_id", "service_code", "language"]
		label_columns = {
			"menu_id": "Menu ID",
			"title": "Title",
			"service_code": "Service Code",
			"language": "Language",
			"is_end_screen": "End Screen",
			"created_at": "Created",
		}

except ImportError:
	# flask_appbuilder not installed — provide minimal stubs so imports don't break
	class UssdEngModelView:  # type: ignore[no-redef]
		pass

	class UssdSessionView(UssdEngModelView):  # type: ignore[no-redef]
		pass

	class UssdMenuView(UssdEngModelView):  # type: ignore[no-redef]
		pass
