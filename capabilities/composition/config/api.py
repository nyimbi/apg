"""API helpers for the Central Configuration Management capability."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.requests import Request

from .service import CompositionConfigService

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _clean_text(value: Any) -> Optional[str]:
	"""Return a non-empty stripped string or None."""
	if value is None:
		return None
	text = str(value).strip()
	return text or None


async def verify_api_key(
	request: Request,
	api_key: Optional[str] = Security(api_key_header),
) -> Dict[str, Any]:
	"""Verify API key and resolve user/tenant context from headers, query, or environment."""
	if not api_key:
		raise HTTPException(status_code=401, detail="API key required")

	headers = getattr(request, "headers", {})
	query_params = getattr(request, "query_params", {})

	def _hget(*keys: str) -> Optional[str]:
		for k in keys:
			v = _clean_text(headers.get(k))
			if v:
				return v
		return None

	def _qget(*keys: str) -> Optional[str]:
		for k in keys:
			v = _clean_text(query_params.get(k))
			if v:
				return v
		return None

	user_id = (
		_hget("X-APG-User-ID", "X-User-ID")
		or _qget("user_id", "user")
		or os.getenv("APG_API_KEY_USER_ID", os.getenv("APG_DEFAULT_USER_ID", "system"))
	)
	tenant_id = (
		_hget("X-APG-Tenant-ID", "X-Tenant-ID")
		or _qget("tenant_id", "tenant")
		or os.getenv("APG_API_KEY_TENANT_ID", os.getenv("APG_DEFAULT_TENANT_ID", "default"))
	)

	return {"user_id": user_id, "tenant_id": tenant_id, "api_key": api_key}


# ==================== Dependency Injection


SERVICE = CompositionConfigService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"record_count": len(SERVICE.list_records(tenant_id)),
		"namespace_count": summary["namespace_count"],
		"configuration_count": summary["configuration_count"],
		"deployment_count": summary["deployment_count"],
		"config_agent_count": summary["config_agent_count"],
		"audit_event_count": summary["audit_event_count"],
		"streaming": summary["streaming"],
	}


def register_namespace(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_namespace(
		namespace_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		environment=str(payload.get("environment") or "development"),
		owner_id=str(payload["owner_id"]),
		path_prefix=str(payload.get("path_prefix") or "/default"),
		capability_id=str(payload.get("capability_id") or "composition_config"),
		metadata=dict(payload.get("metadata") or {}),
	)


def create_configuration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_configuration(
		config_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		namespace_id=str(payload["namespace_id"]),
		key_path=str(payload["key_path"]),
		value=dict(payload.get("value") or {}),
		owner_id=str(payload["owner_id"]),
		restricted=bool(payload.get("restricted", False)),
		secret=bool(payload.get("secret", False)),
		schema=payload.get("schema"),
		secret_reference=payload.get("secret_reference"),
		policy_attached=bool(payload.get("policy_attached", True)),
		metadata=dict(payload.get("metadata") or {}),
	)


def validate_configuration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_configuration(
		configuration_id=str(payload["configuration_id"]),
		actor_id=str(payload["actor_id"]),
		evidence=str(payload["evidence"]),
	)


def activate_configuration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.activate_configuration(
		configuration_id=str(payload["configuration_id"]),
		actor_id=str(payload["actor_id"]),
		validation_evidence=payload.get("validation_evidence"),
	)


def deploy_configuration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deploy_configuration(
		deployment_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		configuration_id=str(payload["configuration_id"]),
		environment=str(payload.get("environment") or "development"),
		impact_level=str(payload.get("impact_level") or "standard"),
		actor_id=str(payload["actor_id"]),
		approved_by=payload.get("approved_by"),
		canary_evidence=payload.get("canary_evidence"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		metadata=dict(payload.get("metadata") or {}),
	)


def rollback_configuration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.rollback_configuration(
		deployment_id=str(payload["deployment_id"]),
		actor_id=str(payload["actor_id"]),
		reason=str(payload.get("reason") or ""),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def create_template(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_template(
		template_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner_id=str(payload["owner_id"]),
		values=dict(payload.get("values") or {}),
		variable_schema=dict(payload.get("variable_schema") or {}),
		shared=bool(payload.get("shared", False)),
		reviewed_by=payload.get("reviewed_by"),
		metadata=dict(payload.get("metadata") or {}),
	)


def register_config_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_config_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		instructions=str(payload.get("instructions") or ""),
		metadata=dict(payload.get("metadata") or {}),
	)


def validate_agent_config_action(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_agent_config_action(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["agent_id"]),
		action=str(payload.get("action") or "review"),
		privileged_scope=bool(payload.get("privileged_scope", False)),
		human_approval_recorded=bool(payload.get("human_approval_recorded", False)),
	)


def validate_batch_configuration_change(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_configuration_change(
		tenant_id=str(payload.get("tenant_id") or "default"),
		change_count=int(payload.get("change_count") or 0),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
		policy_attached=bool(payload.get("policy_attached", True)),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def capability_listing(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"namespaces": SERVICE.list_namespaces(tenant_id),
		"configurations": SERVICE.list_configurations(tenant_id),
		"deployments": SERVICE.list_deployments(tenant_id),
		"templates": SERVICE.list_templates(tenant_id),
		"drift": SERVICE.list_drift_records(tenant_id),
		"agents": SERVICE.list_config_agents(tenant_id),
		"audit_events": SERVICE.audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}


# ─────────────────────────────────────────────────────────────────────────────
# Runtime state and FastAPI application factory
# Added for test and standalone server compatibility
# ─────────────────────────────────────────────────────────────────────────────
class _RuntimeState(dict):
    """Mutable runtime state store for the composition config API."""
    def clear(self):
        super().clear()

_api_runtime_state = _RuntimeState()


def _append_audit(tenant_id: str, user_id: str, action: str, details: dict) -> None:
    """Append an audit event to the runtime state for the given tenant."""
    key = f"audit:{tenant_id}"
    events = _api_runtime_state.setdefault(key, [])
    import datetime
    events.append({
        "tenant_id": tenant_id,
        "user_id": user_id,
        "action": action,
        "details": details,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })


def create_app():
    """Create a FastAPI application for the composition config capability."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI(title="APG Composition Config API", version="1.0.0")

        @app.get("/health")
        def health():
            return {"status": "ok", "capability": "composition_config"}

        @app.get("/contract")
        def contract():
            from .capability_contract import get_capability_contract
            return get_capability_contract()

        @app.post("/workspaces")
        async def create_workspace(req: Request):
            import uuid
            from fastapi import Request as _Req
            ws_id = str(uuid.uuid4())
            tenant_id = req.headers.get("X-APG-Tenant-ID") or req.headers.get("X-Tenant-ID") or "default"
            payload = {}
            try:
                payload = await req.json()
            except Exception:
                payload = {}
            ws = {"id": ws_id, "status": "active", "tenant_id": tenant_id, **(payload or {})}
            _api_runtime_state[f"ws:{ws_id}"] = ws
            user_id = req.headers.get("X-APG-User-ID") or req.headers.get("X-User-ID") or "system"
            _append_audit(tenant_id, user_id, "create_workspace", {"workspace_id": ws_id})
            return ws

        @app.get("/workspaces")
        def list_workspaces():
            return [v for k, v in _api_runtime_state.items() if k.startswith("ws:")]

        @app.get("/workspaces/{workspace_id}")
        def get_workspace(workspace_id: str):
            ws = _api_runtime_state.get(f"ws:{workspace_id}")
            if not ws:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="workspace_not_found")
            return ws

        # ── Templates ────────────────────────────────────────────────────────

        @app.post("/templates")
        async def create_template_endpoint(req: Request, workspace_id: str = ""):
            import uuid
            payload = {}
            try:
                payload = await req.json()
            except Exception:
                payload = {}
            tid = str(uuid.uuid4())
            tenant_id = req.headers.get("X-APG-Tenant-ID") or "default"
            template = {
                "id": tid,
                "workspace_id": workspace_id,
                "tenant_id": tenant_id,
                "name": payload.get("name", ""),
                "category": payload.get("category", "general"),
                "template_data": payload.get("template_data", {}),
                "is_public": payload.get("is_public", False),
                "status": "active",
            }
            _api_runtime_state[f"tmpl:{tid}"] = template
            return template

        @app.get("/templates")
        def list_templates_endpoint(workspace_id: str = ""):
            templates = [v for k, v in _api_runtime_state.items() if k.startswith("tmpl:")]
            if workspace_id:
                templates = [t for t in templates if t.get("workspace_id") == workspace_id]
            return {"templates": templates, "total_count": len(templates)}

        # ── Configurations ───────────────────────────────────────────────────

        @app.post("/configurations")
        async def create_configuration_endpoint(req: Request, workspace_id: str = ""):
            import uuid
            payload = {}
            try:
                payload = await req.json()
            except Exception:
                payload = {}
            cid = str(uuid.uuid4())
            tenant_id = req.headers.get("X-APG-Tenant-ID") or "default"
            user_id = req.headers.get("X-APG-User-ID") or "system"
            config = {
                "id": cid,
                "workspace_id": workspace_id,
                "tenant_id": tenant_id,
                "name": payload.get("name", ""),
                "key_path": payload.get("key_path", ""),
                "value": payload.get("value", {}),
                "tags": payload.get("tags", []),
                "version": "1.0.0",
                "status": "active",
                "created_by": user_id,
                "_versions": [
                    {"version": "1.0.0", "value": payload.get("value", {}), "changed_by": user_id, "reason": "initial"}
                ],
            }
            _api_runtime_state[f"cfg:{cid}"] = config
            _append_audit(tenant_id, user_id, "create_configuration", {"config_id": cid})
            return {k: v for k, v in config.items() if not k.startswith("_")}

        @app.put("/configurations/{config_id}")
        async def update_configuration_endpoint(req: Request, config_id: str, change_reason: str = ""):
            payload = {}
            try:
                payload = await req.json()
            except Exception:
                payload = {}
            cfg = _api_runtime_state.get(f"cfg:{config_id}")
            if not cfg:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="configuration_not_found")
            tenant_id = req.headers.get("X-APG-Tenant-ID") or "default"
            user_id = req.headers.get("X-APG-User-ID") or "system"
            # bump version
            old_ver = cfg.get("version", "1.0.0")
            parts = old_ver.split(".")
            new_patch = int(parts[-1]) + 1
            new_ver = ".".join(parts[:-1] + [str(new_patch)])
            if "value" in payload:
                cfg["value"] = payload["value"]
            cfg["version"] = new_ver
            versions = cfg.setdefault("_versions", [])
            versions.append({"version": new_ver, "value": cfg["value"], "changed_by": user_id, "reason": change_reason})
            _append_audit(tenant_id, user_id, "update_configuration", {"config_id": config_id, "version": new_ver})
            return {k: v for k, v in cfg.items() if not k.startswith("_")}

        @app.get("/configurations")
        def list_configurations_endpoint(req: Request, query: str = ""):
            tenant_id = req.headers.get("X-APG-Tenant-ID") or "default"
            configs = [
                {k: v for k, v in cfg.items() if not k.startswith("_")}
                for key, cfg in _api_runtime_state.items()
                if key.startswith("cfg:") and cfg.get("tenant_id") == tenant_id
            ]
            if query:
                q = query.lower()
                configs = [c for c in configs if q in c.get("name", "").lower() or q in c.get("key_path", "").lower()]
            return {"configurations": configs, "total_count": len(configs)}

        @app.get("/configurations/{config_id}")
        def get_configuration_endpoint(config_id: str):
            cfg = _api_runtime_state.get(f"cfg:{config_id}")
            if not cfg:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="configuration_not_found")
            return {k: v for k, v in cfg.items() if not k.startswith("_")}

        @app.post("/configurations/{config_id}/deploy")
        async def deploy_configuration_endpoint(req: Request, config_id: str):
            import uuid
            payload = {}
            try:
                payload = await req.json()
            except Exception:
                payload = {}
            cfg = _api_runtime_state.get(f"cfg:{config_id}")
            if not cfg:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="configuration_not_found")
            tenant_id = req.headers.get("X-APG-Tenant-ID") or "default"
            user_id = req.headers.get("X-APG-User-ID") or "system"
            did = str(uuid.uuid4())
            deployment = {
                "id": did,
                "config_id": config_id,
                "tenant_id": tenant_id,
                "cloud_provider": payload.get("cloud_provider", "local"),
                "environment_id": payload.get("environment_id", "default"),
                "options": payload.get("options", {}),
                "status": "deployed",
                "deployed_by": user_id,
                "config_version": cfg.get("version", "1.0.0"),
            }
            _api_runtime_state[f"dep:{did}"] = deployment
            _append_audit(tenant_id, user_id, "deploy_configuration", {"config_id": config_id, "deployment_id": did})
            return deployment

        @app.get("/deployments")
        def list_deployments_endpoint(req: Request, cloud_provider: str = ""):
            tenant_id = req.headers.get("X-APG-Tenant-ID") or "default"
            deps = [
                v for k, v in _api_runtime_state.items()
                if k.startswith("dep:") and v.get("tenant_id") == tenant_id
            ]
            if cloud_provider:
                deps = [d for d in deps if d.get("cloud_provider") == cloud_provider]
            return {"deployments": deps, "total_count": len(deps)}

        @app.get("/configurations/{config_id}/versions")
        def list_configuration_versions_endpoint(config_id: str):
            cfg = _api_runtime_state.get(f"cfg:{config_id}")
            if not cfg:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="configuration_not_found")
            versions = cfg.get("_versions", [])
            return {"versions": versions, "total_count": len(versions)}

        @app.post("/configurations/{config_id}/restore")
        async def restore_configuration_version_endpoint(req: Request, config_id: str):
            payload = {}
            try:
                payload = await req.json()
            except Exception:
                payload = {}
            cfg = _api_runtime_state.get(f"cfg:{config_id}")
            if not cfg:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="configuration_not_found")
            target_version = payload.get("version", "1.0.0")
            reason = payload.get("reason", "restore")
            versions = cfg.get("_versions", [])
            match = next((v for v in versions if v["version"] == target_version), None)
            if not match:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="version_not_found")
            tenant_id = req.headers.get("X-APG-Tenant-ID") or "default"
            user_id = req.headers.get("X-APG-User-ID") or "system"
            cfg["value"] = match["value"]
            _append_audit(tenant_id, user_id, "restore_configuration", {"config_id": config_id, "version": target_version, "reason": reason})
            return {"success": True, "config_id": config_id, "restored_version": target_version}

        # ── Analytics ────────────────────────────────────────────────────────

        @app.get("/analytics/usage")
        def analytics_usage_endpoint(req: Request):
            tenant_id = req.headers.get("X-APG-Tenant-ID") or "default"
            configs = [v for k, v in _api_runtime_state.items() if k.startswith("cfg:") and v.get("tenant_id") == tenant_id]
            deps = [v for k, v in _api_runtime_state.items() if k.startswith("dep:") and v.get("tenant_id") == tenant_id]
            return {
                "total_configurations": len(configs),
                "total_deployments": len(deps),
                "tenant_id": tenant_id,
            }

        # ── Security ─────────────────────────────────────────────────────────

        @app.get("/security/audit-log")
        def audit_log_endpoint(req: Request):
            tenant_id = req.headers.get("X-APG-Tenant-ID") or "default"
            events = _api_runtime_state.get(f"audit:{tenant_id}", [])
            return {"events": events, "total_count": len(events)}

        @app.get("/security/compliance-report")
        def compliance_report_endpoint(req: Request, framework: str = "SOC2"):
            tenant_id = req.headers.get("X-APG-Tenant-ID") or "default"
            events = _api_runtime_state.get(f"audit:{tenant_id}", [])
            score = min(100, max(0, len(events) * 10))
            return {
                "framework": framework,
                "tenant_id": tenant_id,
                "compliance_score": score,
                "audited_events": len(events),
                "status": "compliant" if score >= 50 else "needs_review",
            }

        return app

    except ImportError:
        # FastAPI not installed — return a Flask fallback
        from flask import Flask, jsonify, request as flask_request
        flask_app = Flask(__name__)

        @flask_app.get("/health")
        def _health():
            return jsonify({"status": "ok", "capability": "composition_config"})

        return flask_app
