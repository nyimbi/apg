"""Auth hub REST API — provider-agnostic authentication endpoints."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from .service import AuthHubService
from .protocols import AuthenticationError, AuthorizationError, ProviderNotImplementedError

_log = logging.getLogger(__name__)

auth_hub_api = Blueprint("auth_hub_api", __name__, url_prefix="/api/auth")


def _svc() -> AuthHubService:
    return AuthHubService(tenant_id=request.headers.get("X-Tenant-Id", "default"))


def _err(msg: str, code: str = "error", status: int = 400):
    return jsonify({"error": msg, "code": code}), status


# ── Health & info ─────────────────────────────────────────────────

@auth_hub_api.get("/health")
async def health():
    return jsonify(await _svc().health_check())


@auth_hub_api.get("/info")
async def info():
    return jsonify(await _svc().describe())


# ── Authentication ────────────────────────────────────────────────

@auth_hub_api.post("/sign-in")
async def sign_in():
    body = request.get_json(force=True) or {}
    svc = _svc()
    try:
        result = await svc.authenticate(body)
    except AuthenticationError as exc:
        return _err(str(exc), exc.code, 401)
    except ProviderNotImplementedError as exc:
        return _err(str(exc), "not_implemented", 501)
    return jsonify({
        "user": {
            "id": result.user.id,
            "email": result.user.email,
            "username": result.user.username,
            "first_name": result.user.first_name,
            "last_name": result.user.last_name,
            "roles": result.user.roles,
            "is_active": result.user.is_active,
        },
        "tokens": {
            "access_token": result.tokens.access_token,
            "refresh_token": result.tokens.refresh_token,
            "token_type": result.tokens.token_type,
            "expires_in": result.tokens.expires_in,
        },
        "mfa_required": result.mfa_required,
        "mfa_session_token": result.mfa_session_token,
    })


@auth_hub_api.post("/sign-out")
async def sign_out():
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
    body = request.get_json(force=True) or {}
    if not token:
        return _err("No token provided", "missing_token", 400)
    await _svc().logout(token, body.get("refresh_token"))
    return jsonify({"signed_out": True})


@auth_hub_api.post("/token/refresh")
async def refresh_token():
    body = request.get_json(force=True) or {}
    refresh = body.get("refresh_token", "")
    if not refresh:
        return _err("refresh_token required", "missing_refresh_token", 400)
    try:
        tokens = await _svc().refresh_token(refresh)
    except AuthenticationError as exc:
        return _err(str(exc), exc.code, 401)
    return jsonify({
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_in": tokens.expires_in,
    })


@auth_hub_api.post("/token/validate")
async def validate_token():
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
    if not token:
        token = (request.get_json(force=True) or {}).get("token", "")
    if not token:
        return _err("token required", "missing_token", 400)
    try:
        payload = await _svc().validate_token(token)
    except AuthenticationError as exc:
        return _err(str(exc), exc.code, 401)
    return jsonify({
        "valid": True,
        "user_id": payload.user_id,
        "email": payload.email,
        "roles": payload.roles,
        "tenant_id": payload.tenant_id,
        "expires_at": payload.expires_at.isoformat() if payload.expires_at else None,
    })


# ── Users ─────────────────────────────────────────────────────────

@auth_hub_api.get("/users")
async def list_users():
    svc = _svc()
    search = request.args.get("search")
    limit = int(request.args.get("limit", "50"))
    page = int(request.args.get("page", "1"))
    result = await svc.list_users(search=search, limit=limit, page=page)
    return jsonify({
        "users": [{"id": u.id, "email": u.email, "username": u.username,
                   "first_name": u.first_name, "last_name": u.last_name,
                   "roles": u.roles, "is_active": u.is_active} for u in result.users],
        "total": result.total, "page": result.page, "limit": result.limit,
    })


@auth_hub_api.post("/users")
async def create_user():
    body = request.get_json(force=True) or {}
    try:
        user = await _svc().create_user(body)
    except ValueError as exc:
        return _err(str(exc), "validation_error", 400)
    return jsonify({"id": user.id, "email": user.email, "roles": user.roles}), 201


@auth_hub_api.get("/users/<user_id>")
async def get_user(user_id: str):
    try:
        user = await _svc().get_user(user_id)
    except KeyError as exc:
        return _err(str(exc), "not_found", 404)
    return jsonify({"id": user.id, "email": user.email, "username": user.username,
                    "first_name": user.first_name, "last_name": user.last_name,
                    "roles": user.roles, "is_active": user.is_active,
                    "mfa_enabled": user.mfa_enabled})


@auth_hub_api.patch("/users/<user_id>")
async def update_user(user_id: str):
    body = request.get_json(force=True) or {}
    try:
        user = await _svc().update_user(user_id, body)
    except KeyError as exc:
        return _err(str(exc), "not_found", 404)
    return jsonify({"id": user.id, "email": user.email, "roles": user.roles})


@auth_hub_api.delete("/users/<user_id>")
async def delete_user(user_id: str):
    try:
        await _svc().delete_user(user_id)
    except KeyError as exc:
        return _err(str(exc), "not_found", 404)
    return jsonify({"deleted": True})


# ── Password / Magic Link ─────────────────────────────────────────

@auth_hub_api.post("/password/reset-request")
async def request_password_reset():
    body = request.get_json(force=True) or {}
    email = body.get("email", "")
    if not email:
        return _err("email required", "missing_email", 400)
    try:
        await _svc().send_password_reset(email)
    except ProviderNotImplementedError as exc:
        return _err(str(exc), "not_implemented", 501)
    return jsonify({"sent": True})


@auth_hub_api.post("/password/reset")
async def reset_password():
    body = request.get_json(force=True) or {}
    try:
        await _svc().reset_password(body.get("token", ""), body.get("new_password", ""))
    except (ValueError, ProviderNotImplementedError) as exc:
        return _err(str(exc), "reset_failed", 400)
    return jsonify({"reset": True})


@auth_hub_api.post("/magic-link/send")
async def send_magic_link():
    body = request.get_json(force=True) or {}
    try:
        await _svc().send_magic_link(body.get("email", ""), body.get("redirect_url", "/"))
    except ProviderNotImplementedError as exc:
        return _err(str(exc), "not_implemented", 501)
    return jsonify({"sent": True})


@auth_hub_api.post("/magic-link/verify")
async def verify_magic_link():
    body = request.get_json(force=True) or {}
    try:
        result = await _svc().verify_magic_link(body.get("token", ""))
    except (AuthenticationError, ProviderNotImplementedError) as exc:
        return _err(str(exc), "verification_failed", 400)
    return jsonify({
        "user": {"id": result.user.id, "email": result.user.email},
        "tokens": {"access_token": result.tokens.access_token, "expires_in": result.tokens.expires_in},
    })


# ── OAuth ─────────────────────────────────────────────────────────

@auth_hub_api.get("/oauth/authorize")
async def oauth_authorize():
    provider = request.args.get("provider", "")
    redirect_uri = request.args.get("redirect_uri", "")
    state = request.args.get("state", "")
    if not provider or not redirect_uri:
        return _err("provider and redirect_uri required", "missing_params", 400)
    try:
        url = await _svc().get_oauth_url(provider, redirect_uri, state)
    except ProviderNotImplementedError as exc:
        return _err(str(exc), "not_implemented", 501)
    return jsonify({"authorization_url": url})


@auth_hub_api.post("/oauth/callback")
async def oauth_callback():
    body = request.get_json(force=True) or {}
    try:
        result = await _svc().exchange_oauth_code(
            body.get("code", ""), body.get("state", ""),
            body.get("redirect_uri", ""), body.get("provider", ""),
        )
    except (AuthenticationError, ProviderNotImplementedError) as exc:
        return _err(str(exc), "oauth_failed", 400)
    return jsonify({
        "user": {"id": result.user.id, "email": result.user.email},
        "tokens": {"access_token": result.tokens.access_token,
                   "refresh_token": result.tokens.refresh_token,
                   "expires_in": result.tokens.expires_in},
    })


# ── MFA ────────────────────────────────────────────────────────────

@auth_hub_api.post("/users/<user_id>/mfa/setup")
async def setup_mfa(user_id: str):
    body = request.get_json(force=True) or {}
    try:
        setup = await _svc().setup_mfa(user_id, body.get("mfa_type", "totp"))
    except ProviderNotImplementedError as exc:
        return _err(str(exc), "not_implemented", 501)
    return jsonify({
        "mfa_type": setup.mfa_type,
        "secret": setup.secret,
        "qr_code_url": setup.qr_code_url,
        "backup_codes": setup.backup_codes,
    })


@auth_hub_api.post("/mfa/verify")
async def verify_mfa():
    body = request.get_json(force=True) or {}
    try:
        result = await _svc().verify_mfa(
            body.get("user_id", ""), body.get("code", ""), body.get("session_token", "")
        )
    except (AuthenticationError, ProviderNotImplementedError) as exc:
        return _err(str(exc), "mfa_failed", 400)
    return jsonify({
        "tokens": {"access_token": result.tokens.access_token, "expires_in": result.tokens.expires_in}
    })


# ── Roles & Permissions ───────────────────────────────────────────

@auth_hub_api.get("/roles")
async def list_roles():
    return jsonify({"roles": await _svc().list_roles()})


@auth_hub_api.post("/roles")
async def create_role():
    body = request.get_json(force=True) or {}
    role = await _svc().create_role(
        body.get("role", ""), body.get("permissions", []),
        description=body.get("description", ""),
    )
    return jsonify(role), 201


@auth_hub_api.delete("/roles/<role>")
async def delete_role(role: str):
    await _svc().delete_role(role)
    return jsonify({"deleted": True})


@auth_hub_api.get("/users/<user_id>/roles")
async def get_user_roles(user_id: str):
    roles = await _svc().get_user_roles(user_id)
    return jsonify({"user_id": user_id, "roles": roles})


@auth_hub_api.post("/users/<user_id>/roles")
async def assign_role(user_id: str):
    body = request.get_json(force=True) or {}
    role = body.get("role", "")
    if not role:
        return _err("role required", "missing_role", 400)
    await _svc().assign_role(user_id, role, granted_by=request.headers.get("X-Actor-Id", "api"))
    return jsonify({"assigned": True, "role": role})


@auth_hub_api.delete("/users/<user_id>/roles/<role>")
async def revoke_role(user_id: str, role: str):
    await _svc().revoke_role(user_id, role, revoked_by=request.headers.get("X-Actor-Id", "api"))
    return jsonify({"revoked": True})


@auth_hub_api.post("/permissions/check")
async def check_permission():
    body = request.get_json(force=True) or {}
    allowed = await _svc().check_permission(
        user_id=body.get("user_id", ""),
        permission=body.get("permission", ""),
        resource_id=body.get("resource_id"),
        resource_type=body.get("resource_type"),
    )
    return jsonify({"allowed": allowed, "permission": body.get("permission")})


@auth_hub_api.post("/permissions/bulk-check")
async def bulk_check():
    body = request.get_json(force=True) or {}
    results = await _svc().bulk_check_permissions(
        user_id=body.get("user_id", ""),
        checks=body.get("checks", []),
    )
    return jsonify({"results": results})


@auth_hub_api.post("/relationships")
async def write_relationship():
    body = request.get_json(force=True) or {}
    await _svc().write_relationship(
        body.get("resource_type", ""), body.get("resource_id", ""),
        body.get("relation", ""), body.get("subject_type", ""), body.get("subject_id", ""),
    )
    return jsonify({"written": True}), 201


@auth_hub_api.delete("/relationships")
async def delete_relationship():
    body = request.get_json(force=True) or {}
    await _svc().delete_relationship(
        body.get("resource_type", ""), body.get("resource_id", ""),
        body.get("relation", ""), body.get("subject_type", ""), body.get("subject_id", ""),
    )
    return jsonify({"deleted": True})
