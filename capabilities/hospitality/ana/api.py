"""Flask Blueprint — REST API for Hospitality Analytics."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import ANAService

_log = logging.getLogger(__name__)

ana_bp = Blueprint("hos_ana", __name__, url_prefix="/api/hospitality/ana")
_svc = ANAService()


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@ana_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@ana_bp.get("/kpi-snapshots")
def list_kpi_snapshots():
	date_from = request.args.get("date_from")
	date_to = request.args.get("date_to")
	return jsonify(_run(_svc.list_kpi_snapshots(_tenant(), date_from=date_from, date_to=date_to)))


@ana_bp.get("/kpi-snapshots/<snapshot_id>")
def get_kpi_snapshot(snapshot_id: str):
	try:
		return jsonify(_run(_svc.get_kpi_snapshot(snapshot_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@ana_bp.post("/kpi-snapshots")
def record_kpi_snapshot():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.record_kpi_snapshot(
			date=data["date"],
			total_rooms=int(data["total_rooms"]),
			occupied_rooms=int(data["occupied_rooms"]),
			total_revenue=float(data["total_revenue"]),
			room_revenue=float(data["room_revenue"]),
			ancillary_revenue=float(data.get("ancillary_revenue", 0)),
			goppar=float(data["goppar"]) if "goppar" in data else None,
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@ana_bp.put("/kpi-snapshots/<snapshot_id>")
def update_kpi_snapshot(snapshot_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_kpi_snapshot(snapshot_id, data, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@ana_bp.delete("/kpi-snapshots/<snapshot_id>")
def delete_kpi_snapshot(snapshot_id: str):
	try:
		return jsonify(_run(_svc.delete_kpi_snapshot(snapshot_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@ana_bp.get("/kpi-summary")
def kpi_period_summary():
	date_from = request.args.get("date_from", "")
	date_to = request.args.get("date_to", "9999-12-31")
	return jsonify(_run(_svc.kpi_period_summary(date_from, date_to, _tenant())))


@ana_bp.get("/segment-reports")
def list_segment_reports():
	period = request.args.get("period")
	segment = request.args.get("segment")
	return jsonify(_run(_svc.list_segment_reports(_tenant(), period=period, segment=segment)))


@ana_bp.post("/segment-reports")
def record_segment_report():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.record_segment_report(
			period=data["period"],
			segment=data["segment"],
			room_nights=int(data["room_nights"]),
			revenue=float(data["revenue"]),
			total_room_nights=int(data.get("total_room_nights", data["room_nights"])),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@ana_bp.get("/segment-mix")
def segment_mix():
	period = request.args.get("period", "")
	return jsonify(_run(_svc.segment_mix_report(period, _tenant())))


@ana_bp.get("/pace-reports")
def list_pace_reports():
	future_date = request.args.get("future_date")
	return jsonify(_run(_svc.list_pace_reports(_tenant(), future_date=future_date)))


@ana_bp.post("/pace-reports")
def record_pace_report():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.record_pace_report(
			report_date=data["report_date"],
			future_date=data["future_date"],
			booked_rooms=int(data["booked_rooms"]),
			booked_revenue=float(data["booked_revenue"]),
			on_the_books_adr=float(data["on_the_books_adr"]),
			pickup_last_7_days=int(data.get("pickup_last_7_days", 0)),
			vs_last_year_pct=float(data["vs_last_year_pct"]) if "vs_last_year_pct" in data else None,
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@ana_bp.get("/pace-comparison")
def pace_comparison():
	future_date = request.args.get("future_date", "")
	days_out = int(request.args.get("days_out", 30))
	return jsonify(_run(_svc.pace_comparison(future_date, days_out, _tenant())))


@ana_bp.get("/satisfaction-surveys")
def list_satisfaction_surveys():
	date_from = request.args.get("date_from")
	nps_category = request.args.get("nps_category")
	return jsonify(_run(_svc.list_satisfaction_surveys(_tenant(), date_from=date_from, nps_category=nps_category)))


@ana_bp.get("/satisfaction-surveys/<survey_id>")
def get_satisfaction_survey(survey_id: str):
	try:
		return jsonify(_run(_svc.get_satisfaction_survey(survey_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@ana_bp.post("/satisfaction-surveys")
def record_satisfaction_survey():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.record_satisfaction_survey(
			reservation_id=data.get("reservation_id", ""),
			guest_name=data.get("guest_name", ""),
			overall_score=float(data["overall_score"]),
			room_score=float(data["room_score"]) if "room_score" in data else None,
			service_score=float(data["service_score"]) if "service_score" in data else None,
			food_score=float(data["food_score"]) if "food_score" in data else None,
			cleanliness_score=float(data["cleanliness_score"]) if "cleanliness_score" in data else None,
			comments=data.get("comments"),
			channel=data.get("channel", "post_stay_email"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@ana_bp.delete("/satisfaction-surveys/<survey_id>")
def delete_satisfaction_survey(survey_id: str):
	try:
		return jsonify(_run(_svc.delete_satisfaction_survey(survey_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@ana_bp.get("/satisfaction-summary")
def satisfaction_summary():
	date_from = request.args.get("date_from")
	date_to = request.args.get("date_to")
	return jsonify(_run(_svc.satisfaction_summary(date_from, date_to, _tenant())))


@ana_bp.post("/benchmarks")
def record_benchmark():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.record_benchmark(
			period=data["period"],
			metric=data["metric"],
			our_value=float(data["our_value"]),
			market_avg=float(data["market_avg"]),
			competitive_set_avg=float(data["competitive_set_avg"]) if "competitive_set_avg" in data else None,
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@ana_bp.get("/benchmarks")
def list_benchmarks():
	metric = request.args.get("metric")
	return jsonify(_run(_svc.list_benchmarks(_tenant(), metric=metric)))


@ana_bp.post("/competitive-sets")
def create_competitive_set():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_competitive_set(
			name=data["name"],
			properties=data.get("properties", []),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@ana_bp.get("/competitive-sets")
def list_competitive_sets():
	return jsonify(_run(_svc.list_competitive_sets(_tenant())))


@ana_bp.post("/channel-revenue")
def record_channel_revenue():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.record_channel_revenue(
			period=data["period"],
			channel=data["channel"],
			bookings=int(data["bookings"]),
			revenue=float(data["revenue"]),
			commission=float(data.get("commission", 0)),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@ana_bp.get("/channel-mix")
def channel_mix():
	period = request.args.get("period", "")
	return jsonify(_run(_svc.channel_mix_report(period, _tenant())))


@ana_bp.get("/executive-dashboard")
def executive_dashboard():
	date_from = request.args.get("date_from", "")
	date_to = request.args.get("date_to", "9999-12-31")
	return jsonify(_run(_svc.executive_dashboard(date_from, date_to, _tenant())))


@ana_bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


@ana_bp.get("/audit-events")
def audit_events():
	return jsonify(_run(_svc.get_audit_events(_tenant())))
