"""Executable capability contract for APG Store Intelligence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "retail_sin"
CAPABILITY_NAME = "Store Intelligence"
CAPABILITY_VERSION = "1.0.0"
SIN_EVENT_STREAM = "apg.retail.sin.lifecycle"

SUPPORTED_TRAFFIC_SENSOR_TYPES = ["infrared_beam", "stereo_camera", "wifi_probe", "bluetooth_beacon", "thermal_camera", "lidar", "people_counter"]
SUPPORTED_ZONE_TYPES = ["entrance", "aisle", "department", "checkout", "display", "fitting_room", "service_desk", "window", "external"]
SUPPORTED_PLANOGRAM_STATUSES = ["compliant", "minor_deviation", "major_deviation", "out_of_stock", "facing_issue", "price_label_missing", "unchecked"]
SUPPORTED_SHELF_ALERT_TYPES = ["out_of_stock", "low_stock", "misplace", "price_label_missing", "planogram_deviation", "expiry_risk", "overstock"]
SUPPORTED_CONVERSION_METRICS = ["entry_to_dwell", "dwell_to_browse", "browse_to_basket", "basket_to_purchase", "overall_conversion"]
SUPPORTED_BENCHMARK_TYPES = ["peer_group", "region", "national", "year_on_year", "rolling_4_week", "target"]
SUPPORTED_KPI_CATEGORIES = ["traffic", "conversion", "dwell_time", "basket_size", "revenue_per_sqm", "staff_productivity", "shrinkage", "availability"]
SUPPORTED_HEATMAP_RESOLUTIONS = ["1m", "2m", "5m", "10m"]
SUPPORTED_ALERT_SEVERITIES = ["info", "warning", "critical"]
SUPPORTED_REPORT_FREQUENCIES = ["real_time", "hourly", "daily", "weekly", "monthly", "quarterly"]
SUPPORTED_AGENT_ROLES = ["traffic_analyst", "planogram_auditor", "availability_monitor", "conversion_optimizer", "benchmark_analyst"]
SUPPORTED_STORE_FORMATS = ["hypermarket", "supermarket", "convenience", "fashion", "electronics", "pharmacy", "diy", "specialty", "outlet", "pop_up"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"traffic": {
		"supported_sensor_types": SUPPORTED_TRAFFIC_SENSOR_TYPES,
		"supported_zone_types": SUPPORTED_ZONE_TYPES,
		"counting_interval_seconds": 60,
		"anonymisation_required": True,
	},
	"planogram": {
		"supported_statuses": SUPPORTED_PLANOGRAM_STATUSES,
		"audit_frequency_hours": 24,
		"image_capture_enabled": True,
		"ai_compliance_check_enabled": True,
	},
	"shelf": {
		"supported_alert_types": SUPPORTED_SHELF_ALERT_TYPES,
		"oos_replenishment_sla_minutes": 30,
		"low_stock_threshold_pct": 20,
		"alert_deduplication_window_minutes": 15,
	},
	"conversion": {
		"supported_metrics": SUPPORTED_CONVERSION_METRICS,
		"attribution_window_minutes": 60,
		"journey_stitching_enabled": True,
	},
	"benchmarking": {
		"supported_types": SUPPORTED_BENCHMARK_TYPES,
		"kpi_categories": SUPPORTED_KPI_CATEGORIES,
		"peer_group_min_stores": 5,
	},
	"heatmaps": {"supported_resolutions": SUPPORTED_HEATMAP_RESOLUTIONS, "retention_days": 90, "pii_masking_required": True},
	"reporting": {"supported_frequencies": SUPPORTED_REPORT_FREQUENCIES, "export_formats": ["pdf", "csv", "json"]},
	"stores": {"supported_formats": SUPPORTED_STORE_FORMATS, "location_required": True, "sqm_required": True},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"pii_data_anonymisation_required": True,
		"cross_tenant_access_denied": True,
		"raw_video_storage_denied": True,
		"biometric_id_denied": True,
	},
	"observability": {"event_stream": SIN_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_traffic": True, "enable_planogram": True, "enable_shelf": True, "enable_conversion": True, "enable_benchmarking": True},
	"theme": {"default_theme": "retail_sin_insights", "allow_tenant_overrides": True},
}

PROVIDES = [
	"store_foot_traffic_analytics",
	"planogram_compliance_monitoring",
	"shelf_availability_alerting",
	"store_conversion_optimisation",
	"store_performance_benchmarking",
	"zone_heatmap_analytics",
	"store_kpi_reporting",
	"replenishment_triggering",
	"shopper_journey_analytics",
	"store_format_benchmarking",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "mqeb", "moni", "nlpc", "schd", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/retail-sin/dashboard", "component": "SinDashboard", "permission": "retail_sin:view", "nav_group": "Overview"},
	{"name": "traffic", "path": "/retail-sin/traffic", "component": "SinTrafficAnalytics", "permission": "retail_sin:view", "nav_group": "Traffic"},
	{"name": "heatmaps", "path": "/retail-sin/heatmaps", "component": "SinHeatmapViewer", "permission": "retail_sin:view", "nav_group": "Traffic"},
	{"name": "planogram", "path": "/retail-sin/planogram", "component": "SinPlanogramAudit", "permission": "retail_sin:view", "nav_group": "Compliance"},
	{"name": "planogram_detail", "path": "/retail-sin/planogram/<id>", "component": "SinPlanogramDetail", "permission": "retail_sin:view", "nav_group": "Compliance"},
	{"name": "shelf_alerts", "path": "/retail-sin/shelf-alerts", "component": "SinShelfAlertList", "permission": "retail_sin:view", "nav_group": "Availability"},
	{"name": "conversion", "path": "/retail-sin/conversion", "component": "SinConversionFunnel", "permission": "retail_sin:view", "nav_group": "Performance"},
	{"name": "journey", "path": "/retail-sin/journey", "component": "SinShopperJourney", "permission": "retail_sin:view", "nav_group": "Performance"},
	{"name": "benchmarking", "path": "/retail-sin/benchmarking", "component": "SinBenchmarkReport", "permission": "retail_sin:view", "nav_group": "Benchmarking"},
	{"name": "kpis", "path": "/retail-sin/kpis", "component": "SinKpiScorecard", "permission": "retail_sin:view", "nav_group": "Benchmarking"},
	{"name": "stores", "path": "/retail-sin/stores", "component": "SinStoreList", "permission": "retail_sin:admin", "nav_group": "Configuration"},
	{"name": "sensors", "path": "/retail-sin/sensors", "component": "SinSensorManager", "permission": "retail_sin:admin", "nav_group": "Configuration"},
	{"name": "reports", "path": "/retail-sin/reports", "component": "SinReports", "permission": "retail_sin:view", "nav_group": "Analytics"},
	{"name": "settings", "path": "/retail-sin/settings", "component": "SinSettings", "permission": "retail_sin:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "retail_sin_insights",
	"tokens": {
		"color.primary": "#0F766E",
		"color.accent": "#0EA5E9",
		"color.success": "#16A34A",
		"color.warning": "#D97706",
		"color.danger": "#DC2626",
		"surface.canvas": "#F0FDFA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#134E4A",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"store": {"icon": "store", "status_indicator": "store-format-chip"},
		"zone": {"icon": "grid", "status_indicator": "zone-type-chip"},
		"sensor": {"icon": "radio", "status_indicator": "sensor-status-chip"},
		"planogram": {"icon": "layout", "status_indicator": "compliance-chip"},
		"shelf_alert": {"icon": "alert-triangle", "status_indicator": "alert-severity-chip"},
		"conversion": {"icon": "funnel", "status_indicator": "conversion-trend-chip"},
		"benchmark": {"icon": "bar-chart", "status_indicator": "benchmark-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": SIN_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"traffic_count_recorded",
		"zone_dwell_recorded",
		"planogram_audit_completed",
		"planogram_deviation_detected",
		"shelf_alert_raised",
		"shelf_alert_resolved",
		"oos_replenishment_triggered",
		"conversion_event_recorded",
		"kpi_snapshot_published",
		"benchmark_updated",
		"heatmap_generated",
	],
	"guardrails": [
		"pii_anonymisation_required",
		"raw_video_storage_denied",
		"biometric_id_denied",
		"cross_tenant_access_denied",
		"sensor_data_requires_consent",
		"batch_traffic_requires_bytewax",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_sin_policy"}},
	{"name": "pii_anonymisation_required", "condition": {"operation": "ingest_sensor_data", "pii_anonymised": False}, "effect": {"decision": "deny", "reason": "pii_anonymisation_required", "required_action": "anonymise_sensor_data"}},
	{"name": "raw_video_storage_denied", "condition": {"operation": "store_sensor_data", "data_type": "raw_video"}, "effect": {"decision": "deny", "reason": "raw_video_storage_not_permitted", "required_action": "use_anonymised_counts_only"}},
	{"name": "biometric_id_denied", "condition": {"operation": "ingest_sensor_data", "biometric_id_present": True}, "effect": {"decision": "deny", "reason": "biometric_identification_not_permitted", "required_action": "remove_biometric_identifiers"}},
	{"name": "sensor_type_supported", "condition": {"operation": "register_sensor", "sensor_type_supported": False}, "effect": {"decision": "deny", "reason": "sensor_type_not_supported", "required_action": "select_supported_sensor_type"}},
	{"name": "zone_type_supported", "condition": {"operation": "create_zone", "zone_type_supported": False}, "effect": {"decision": "deny", "reason": "zone_type_not_supported", "required_action": "select_supported_zone_type"}},
	{"name": "store_location_required", "condition": {"operation": "create_store", "location_present": False}, "effect": {"decision": "deny", "reason": "store_location_required", "required_action": "set_store_location"}},
	{"name": "store_sqm_required", "condition": {"operation": "create_store", "sqm_present": False}, "effect": {"decision": "deny", "reason": "store_sqm_required", "required_action": "set_store_sqm"}},
	{"name": "store_format_supported", "condition": {"operation": "create_store", "store_format_supported": False}, "effect": {"decision": "deny", "reason": "store_format_not_supported", "required_action": "select_supported_store_format"}},
	{"name": "planogram_audit_frequency_required", "condition": {"operation": "schedule_planogram_audit", "audit_frequency_set": False}, "effect": {"decision": "deny", "reason": "audit_frequency_required", "required_action": "set_audit_frequency"}},
	{"name": "shelf_alert_type_supported", "condition": {"operation": "raise_shelf_alert", "alert_type_supported": False}, "effect": {"decision": "deny", "reason": "shelf_alert_type_not_supported", "required_action": "select_supported_alert_type"}},
	{"name": "shelf_alert_severity_supported", "condition": {"operation": "raise_shelf_alert", "alert_severity_supported": False}, "effect": {"decision": "deny", "reason": "alert_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "heatmap_resolution_supported", "condition": {"operation": "generate_heatmap", "heatmap_resolution_supported": False}, "effect": {"decision": "deny", "reason": "heatmap_resolution_not_supported", "required_action": "select_supported_resolution"}},
	{"name": "heatmap_pii_masking_required", "condition": {"operation": "generate_heatmap", "pii_masked": False}, "effect": {"decision": "deny", "reason": "heatmap_pii_masking_required", "required_action": "enable_pii_masking"}},
	{"name": "benchmark_type_supported", "condition": {"operation": "run_benchmark", "benchmark_type_supported": False}, "effect": {"decision": "deny", "reason": "benchmark_type_not_supported", "required_action": "select_supported_benchmark_type"}},
	{"name": "benchmark_min_peer_stores", "condition": {"operation": "run_peer_benchmark", "peer_store_count_sufficient": False}, "effect": {"decision": "deny", "reason": "insufficient_peer_stores_for_benchmark", "required_action": "expand_peer_group"}},
	{"name": "kpi_category_supported", "condition": {"operation": "record_kpi", "kpi_category_supported": False}, "effect": {"decision": "deny", "reason": "kpi_category_not_supported", "required_action": "select_supported_kpi_category"}},
	{"name": "conversion_metric_supported", "condition": {"operation": "record_conversion", "conversion_metric_supported": False}, "effect": {"decision": "deny", "reason": "conversion_metric_not_supported", "required_action": "select_supported_conversion_metric"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "report_frequency_supported", "condition": {"operation": "schedule_report", "report_frequency_supported": False}, "effect": {"decision": "deny", "reason": "report_frequency_not_supported", "required_action": "select_supported_report_frequency"}},
	{"name": "batch_traffic_requires_bytewax", "condition": {"operation": "batch_ingest_traffic", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "batch_traffic_requires_bytewax", "required_action": "route_batch_to_bytewax"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {k: {"type": "object"} for k in configuration if k != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/retail-sin/api/v1",
			"requires_theme": True,
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
