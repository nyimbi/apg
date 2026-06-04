"""Flask blueprint helpers for the POSE capability.

Context resolution utilities are extracted first so they can be exec'd in tests
without importing Flask or the full model runtime.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from flask import Blueprint, g, has_request_context, request, session

from .capability_contract import get_capability_contract


def _clean_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _object_value(source: Any, name: str) -> Any:
	if source is None:
		return None
	if isinstance(source, dict):
		return source.get(name)
	return getattr(source, name, None)


def _mapping_value(source: Any, name: str) -> Any:
	if source is None:
		return None
	getter = getattr(source, "get", None)
	return getter(name) if getter else None


def _resolve_pose_request_context(payload: Any = None, view: Any = None) -> Dict[str, Any]:
	"""Resolve tenant and user from APG runtime request sources.

	Priority: request.current_user > g.current_user > headers > args > session > env
	"""
	default_user = os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "anonymous"))
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))

	if not has_request_context():
		return {
			"user_id": default_user,
			"tenant_id": default_tenant,
		}

	# 1. request.current_user dict
	request_user = getattr(request, "current_user", None)
	if isinstance(request_user, dict) and request_user.get("user_id"):
		return {
			"user_id": request_user["user_id"],
			"tenant_id": request_user.get("tenant_id") or default_tenant,
		}

	# 2. g.current_user dict
	g_user = (
		getattr(g, "current_user", None)
		or getattr(g, "user", None)
		or getattr(g, "auth_user", None)
	)
	if isinstance(g_user, dict) and g_user.get("user_id"):
		return {
			"user_id": g_user["user_id"],
			"tenant_id": g_user.get("tenant_id") or default_tenant,
		}

	# 3. Headers
	headers = getattr(request, "headers", {})
	header_user = _mapping_value(headers, "X-APG-User-ID") or _mapping_value(headers, "X-User-ID")
	header_tenant = _mapping_value(headers, "X-APG-Tenant-ID") or _mapping_value(headers, "X-Tenant-ID")
	if header_user:
		return {
			"user_id": header_user,
			"tenant_id": header_tenant or default_tenant,
		}

	# 4. Query params
	try:
		args = getattr(request, "args", {}) or {}
		q_user = _mapping_value(args, "user_id")
		q_tenant = _mapping_value(args, "tenant_id") or _mapping_value(args, "tenant")
		if q_user:
			return {
				"user_id": q_user,
				"tenant_id": q_tenant or default_tenant,
			}
	except Exception:
		pass

	# 5. Session
	sess_user = _mapping_value(session, "user_id") or _mapping_value(session, "username")
	sess_tenant = _mapping_value(session, "tenant_id")
	if sess_user:
		return {
			"user_id": sess_user,
			"tenant_id": sess_tenant or default_tenant,
		}

	# 6. Env fallback
	return {
		"user_id": default_user,
		"tenant_id": default_tenant,
	}


# APG Blueprint Registration

pose_bp = Blueprint("pose", __name__, template_folder="templates", static_folder="static")


def blueprint_manifest(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"name": "pose",
		"display_name": contract["display_name"],
		"api_prefix": contract["ui"]["api_prefix"],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"]["name"],
	}


@pose_bp.route("/pose/session", methods=["POST"])
def create_pose_session() -> Any:
	"""Create a new POSE session."""
	from flask import jsonify
	session_data = request.get_json(silent=True) or {}
	ctx = _resolve_pose_request_context(session_data, self)
	session_data.setdefault("tenant_id", ctx["tenant_id"])
	session_data.setdefault("created_by", ctx["user_id"])
	return jsonify({"status": "ok", "context": ctx})


@pose_bp.route("/pose/track", methods=["POST"])
def create_pose_tracking() -> Any:
	"""Submit POSE tracking data."""
	from flask import jsonify
	tracking_data = request.get_json(silent=True) or {}
	ctx = _resolve_pose_request_context(tracking_data, self)
	tracking_data.setdefault("tenant_id", ctx["tenant_id"])
	tracking_data.setdefault("created_by", ctx["user_id"])
	return jsonify({"status": "ok", "context": ctx})


def register_pose_views(appbuilder, pose_service: Any) -> None:
	"""Register POSE views with Flask-AppBuilder."""
	appbuilder.app.config["POSE_SERVICE"] = pose_service
	appbuilder.app.register_blueprint(pose_bp)


__all__ = [
	"pose_bp",
	"blueprint_manifest",
	"register_pose_views",
	"_resolve_pose_request_context",
]
