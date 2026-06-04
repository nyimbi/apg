"""Flask-AppBuilder views for Sustainability and ESG Management."""

from __future__ import annotations

import logging
from typing import Any

from flask_appbuilder import BaseView, ModelView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .context import get_current_user_id, get_tenant_id_from_request

logger = logging.getLogger(__name__)


class ESGExecutiveDashboardView(BaseView):
	"""Executive dashboard for ESG KPIs and AI insights."""

	route_base = "/esg/executive"
	default_view = "dashboard"

	@expose("/dashboard")
	@has_access
	def dashboard(self) -> Any:
		from .api import dashboard_summary
		tenant_id = get_tenant_id_from_request()
		user_id = get_current_user_id(self.appbuilder)
		summary = dashboard_summary(tenant_id)
		return self.render_template(
			"esg/executive_dashboard.html",
			tenant_id=tenant_id,
			user_id=user_id,
			summary=summary,
		)


class ESGMetricsView(BaseView):
	"""ESG metrics management with AI predictions."""

	route_base = "/esg/metrics"
	default_view = "list"

	@expose("/list")
	@has_access
	def list(self) -> Any:
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		svc = ESGManagementLifecycleService()
		records = svc.list_records(tenant_id, "metric")
		return self.render_template("esg/metrics_list.html", tenant_id=tenant_id, records=records)

	@expose("/add", methods=["GET", "POST"])
	@has_access
	def add(self) -> Any:
		from flask import request as flask_request
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		user_id = get_current_user_id(self.appbuilder)
		svc = ESGManagementLifecycleService()
		if flask_request.method == "POST":
			payload = flask_request.get_json(silent=True) or {}
			payload.setdefault("tenant_id", tenant_id)
			payload.setdefault("owner_id", user_id or "")
			result = svc.define_metric(
				payload.get("id", ""),
				tenant_id,
				payload.get("profile_id", ""),
				payload.get("pillar", "environmental"),
				payload.get("metric_type", "emissions"),
				payload.get("unit", "tco2e"),
				payload.get("name", ""),
				payload.get("owner_id", ""),
			)
			return self.render_template("esg/metric_detail.html", record=result)
		return self.render_template("esg/metric_form.html", tenant_id=tenant_id)


class ESGTargetsView(BaseView):
	"""ESG targets with achievement prediction."""

	route_base = "/esg/targets"
	default_view = "list"

	@expose("/list")
	@has_access
	def list(self) -> Any:
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		svc = ESGManagementLifecycleService()
		records = svc.list_records(tenant_id, "target")
		return self.render_template("esg/targets_list.html", tenant_id=tenant_id, records=records)

	@expose("/add", methods=["GET", "POST"])
	@has_access
	def add(self) -> Any:
		from flask import request as flask_request
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		user_id = get_current_user_id(self.appbuilder)
		svc = ESGManagementLifecycleService()
		if flask_request.method == "POST":
			payload = flask_request.get_json(silent=True) or {}
			result = svc.set_target(
				payload.get("id", ""),
				tenant_id,
				payload.get("metric_id", ""),
				payload.get("target_type", "absolute"),
				payload.get("baseline_value"),
				payload.get("target_value"),
				payload.get("due_date", ""),
				user_id or "",
			)
			return self.render_template("esg/target_detail.html", record=result)
		return self.render_template("esg/target_form.html", tenant_id=tenant_id)


class ESGStakeholdersView(BaseView):
	"""Stakeholder engagement management."""

	route_base = "/esg/stakeholders"
	default_view = "list"

	@expose("/list")
	@has_access
	def list(self) -> Any:
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		svc = ESGManagementLifecycleService()
		records = svc.list_records(tenant_id, "stakeholder")
		return self.render_template("esg/stakeholders_list.html", tenant_id=tenant_id, records=records)

	@expose("/add", methods=["GET", "POST"])
	@has_access
	def add(self) -> Any:
		from flask import request as flask_request
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		user_id = get_current_user_id(self.appbuilder)
		svc = ESGManagementLifecycleService()
		if flask_request.method == "POST":
			payload = flask_request.get_json(silent=True) or {}
			result = svc.register_stakeholder(
				payload.get("id", ""),
				tenant_id,
				payload.get("profile_id", ""),
				payload.get("stakeholder_type", "investor"),
				payload.get("name", ""),
				payload.get("channel", ""),
				payload.get("consent_recorded", False),
			)
			return self.render_template("esg/stakeholder_detail.html", record=result)
		return self.render_template("esg/stakeholder_form.html", tenant_id=tenant_id)


class ESGSuppliersView(BaseView):
	"""Supply chain ESG assessment management."""

	route_base = "/esg/suppliers"
	default_view = "list"

	@expose("/list")
	@has_access
	def list(self) -> Any:
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		svc = ESGManagementLifecycleService()
		records = svc.list_records(tenant_id, "supplier")
		return self.render_template("esg/suppliers_list.html", tenant_id=tenant_id, records=records)


class ESGInitiativesView(BaseView):
	"""ESG initiatives and programmes."""

	route_base = "/esg/initiatives"
	default_view = "list"

	@expose("/list")
	@has_access
	def list(self) -> Any:
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		svc = ESGManagementLifecycleService()
		records = svc.list_records(tenant_id, "initiative")
		return self.render_template("esg/initiatives_list.html", tenant_id=tenant_id, records=records)


class ESGReportsView(BaseView):
	"""ESG reporting — generation and audit trail."""

	route_base = "/esg/reports"
	default_view = "list"

	@expose("/list")
	@has_access
	def list(self) -> Any:
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		svc = ESGManagementLifecycleService()
		records = svc.list_records(tenant_id, "report")
		return self.render_template("esg/reports_list.html", tenant_id=tenant_id, records=records)

	@expose("/generate", methods=["POST"])
	@has_access
	def generate(self) -> Any:
		from flask import request as flask_request
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		user_id = get_current_user_id(self.appbuilder)
		svc = ESGManagementLifecycleService()
		payload = flask_request.get_json(silent=True) or {}
		result = svc.create_report(
			payload.get("id", ""),
			tenant_id,
			payload.get("profile_id", ""),
			payload.get("report_type", "annual"),
			payload.get("period", ""),
			payload.get("framework_ids", []),
			payload.get("measurement_ids", []),
			user_id or "",
		)
		return self.render_template("esg/report_detail.html", record=result)


class ESGStakeholderPortalView(BaseView):
	"""Public-facing stakeholder engagement portal."""

	route_base = "/esg/portal"
	default_view = "dashboard"

	@expose("/dashboard")
	def dashboard(self) -> Any:
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		svc = ESGManagementLifecycleService()
		records = svc.list_records(tenant_id, "stakeholder")
		return self.render_template("esg/portal_dashboard.html", tenant_id=tenant_id, records=records)


class ESGMetricsChartView(BaseView):
	"""ESG metrics analytics and visualisation."""

	route_base = "/esg/analytics/metrics"
	default_view = "chart"

	@expose("/chart")
	@has_access
	def chart(self) -> Any:
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		svc = ESGManagementLifecycleService()
		records = svc.list_records(tenant_id, "metric")
		return self.render_template("esg/metrics_chart.html", tenant_id=tenant_id, records=records)


class ESGTargetsProgressChartView(BaseView):
	"""ESG targets progress analytics."""

	route_base = "/esg/analytics/targets"
	default_view = "chart"

	@expose("/chart")
	@has_access
	def chart(self) -> Any:
		from .service import ESGManagementLifecycleService
		tenant_id = get_tenant_id_from_request()
		svc = ESGManagementLifecycleService()
		records = svc.list_records(tenant_id, "target")
		return self.render_template("esg/targets_chart.html", tenant_id=tenant_id, records=records)


# ── Legacy functional helpers (kept for backward-compat with tests) ──────────

def dashboard_model(service: Any, tenant_id: str) -> dict[str, Any]:
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Sustainability and ESG",
		"tenant_id": tenant_id,
		"cards": [
			{"label": "Profiles", "value": summary["profile_count"], "icon": "building-2"},
			{"label": "Metrics", "value": summary["metric_count"], "icon": "ruler"},
			{"label": "Measurements", "value": summary["measurement_count"], "icon": "database"},
			{"label": "Targets", "value": summary["target_count"], "icon": "target"},
			{"label": "Reports", "value": summary["report_count"], "icon": "file-text"},
			{"label": "Agents", "value": summary["agent_count"], "icon": "bot"},
		],
		"streaming": summary["streaming"],
	}


def profile_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "ESG Profiles", "records": service.list_records(tenant_id, "profile")}


def framework_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Frameworks", "records": service.list_records(tenant_id, "framework")}


def metric_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Metrics", "records": service.list_records(tenant_id, "metric")}


def measurement_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Measurements", "records": service.list_records(tenant_id, "measurement")}


def target_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Targets", "records": service.list_records(tenant_id, "target")}


def report_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Reports", "records": service.list_records(tenant_id, "report")}


def stakeholder_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Stakeholders", "records": service.list_records(tenant_id, "stakeholder")}


def agent_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {
		"name": "ESG Agents",
		"records": service.list_records(tenant_id, "agent"),
		"policy": {"max_autonomous_scope": "inspect_prepare_and_recommend", "human_approval_required": True},
	}


def rules_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {"name": "ESG Rules", "rule_count": len(contract["rule_engine"]["rules"]), "rules": contract["rule_engine"]["rules"]}


def settings_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {"name": "ESG Settings", "configuration": contract["configuration"], "theme": contract["theme"], "routes": contract["ui"]["routes"]}
