"""Flask-AppBuilder views and re-exported Pydantic schemas for ussd_flo."""

from __future__ import annotations

from .models import (
	AbTestCreate,
	AbTestResponse,
	AbTestUpdate,
	FloAuditEvent,
	FlowCreate,
	FlowEdgeCreate,
	FlowEdgeResponse,
	FlowFilter,
	FlowList,
	FlowNodeCreate,
	FlowNodeResponse,
	FlowNodeUpdate,
	FlowResponse,
	FlowTranslationCreate,
	FlowTranslationResponse,
	FlowUpdate,
)

__all__ = [
	"FlowCreate",
	"FlowUpdate",
	"FlowResponse",
	"FlowList",
	"FlowFilter",
	"FlowNodeCreate",
	"FlowNodeUpdate",
	"FlowNodeResponse",
	"FlowEdgeCreate",
	"FlowEdgeResponse",
	"FlowTranslationCreate",
	"FlowTranslationResponse",
	"AbTestCreate",
	"AbTestUpdate",
	"AbTestResponse",
	"FloAuditEvent",
	"UssdFloModelView",
	"UssdFlowView",
	"UssdAbTestView",
]

try:
	from flask_appbuilder import ModelView

	class UssdFloModelView(ModelView):
		"""Base FAB view for USSD Flow Designer entities."""
		datamodel = None
		list_title = "USSD Flow Designer"
		show_title = "Flow Detail"
		add_title = "Add Flow"
		edit_title = "Edit Flow"

	class UssdFlowView(UssdFloModelView):
		list_title = "USSD Flows"
		list_columns = ["id", "name", "service_code", "status", "node_count", "edge_count", "created_at"]
		show_columns = list_columns + ["description", "languages", "tags", "root_node_id"]
		search_columns = ["name", "service_code", "status"]
		label_columns = {
			"id": "Flow ID",
			"name": "Flow Name",
			"service_code": "Service Code",
			"status": "Status",
			"node_count": "Nodes",
			"edge_count": "Edges",
			"created_at": "Created",
		}

	class UssdAbTestView(UssdFloModelView):
		list_title = "A/B Tests"
		list_columns = ["id", "name", "service_code", "status", "split_percentage", "control_sessions", "variant_sessions"]
		show_columns = list_columns + ["control_flow_id", "variant_flow_id", "control_completions", "variant_completions"]
		search_columns = ["name", "service_code", "status"]
		label_columns = {
			"name": "Test Name",
			"service_code": "Service Code",
			"status": "Status",
			"split_percentage": "Variant %",
			"control_sessions": "Control Sessions",
			"variant_sessions": "Variant Sessions",
		}

except ImportError:
	class UssdFloModelView:  # type: ignore[no-redef]
		pass

	class UssdFlowView(UssdFloModelView):  # type: ignore[no-redef]
		pass

	class UssdAbTestView(UssdFloModelView):  # type: ignore[no-redef]
		pass
