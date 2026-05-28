from flask import Flask, g, jsonify
import pytest

from capabilities.common.conf.api import require_auth, require_permission


def _app() -> Flask:
	app = Flask(__name__)
	app.config["APG_CONF_API_KEYS"] = {
		"test-key": {
			"user_id": "api-client",
			"tenant_id": "tenant-1234",
			"permissions": ["config.read"],
		}
	}
	return app


@pytest.mark.asyncio
async def test_require_auth_rejects_missing_principal() -> None:
	app = _app()

	@require_auth
	async def protected():
		return jsonify({"ok": True})

	with app.test_request_context("/config"):
		response, status = await protected()

	assert status == 401
	assert response.get_json()["success"] is False
	assert "Authentication required" in response.get_json()["message"]


@pytest.mark.asyncio
async def test_require_auth_accepts_configured_api_key() -> None:
	app = _app()

	@require_auth
	async def protected():
		return jsonify({
			"user_id": g.current_user["user_id"],
			"tenant_id": g.current_user["tenant_id"],
			"auth_method": g.current_user["auth_method"],
		})

	with app.test_request_context("/config", headers={"X-API-Key": "test-key"}):
		response = await protected()

	assert response.get_json() == {
		"user_id": "api-client",
		"tenant_id": "tenant-1234",
		"auth_method": "api_key",
	}


@pytest.mark.asyncio
async def test_require_auth_accepts_env_configured_api_key(monkeypatch) -> None:
	app = Flask(__name__)
	monkeypatch.setenv("APG_CONF_API_KEY", "env-key")
	monkeypatch.setenv("APG_CONF_API_KEY_USER", "env-client")
	monkeypatch.setenv("APG_CONF_API_KEY_TENANT", "tenant-env")
	monkeypatch.setenv("APG_CONF_API_KEY_PERMISSIONS", "config.read, config.deploy")

	@require_auth
	async def protected():
		return jsonify({
			"user_id": g.current_user["user_id"],
			"tenant_id": g.current_user["tenant_id"],
			"permissions": sorted(g.current_user["permissions"]),
		})

	with app.test_request_context("/config", headers={"X-API-Key": "env-key"}):
		response = await protected()

	assert response.get_json() == {
		"user_id": "env-client",
		"tenant_id": "tenant-env",
		"permissions": ["config.deploy", "config.read"],
	}


@pytest.mark.asyncio
async def test_require_permission_enforces_declared_permission() -> None:
	app = _app()

	@require_auth
	@require_permission("config.write")
	async def protected():
		return jsonify({"ok": True})

	with app.test_request_context("/config", headers={"X-API-Key": "test-key"}):
		response, status = await protected()

	assert status == 403
	assert response.get_json()["errors"] == ["Missing permission: config.write"]

	with app.test_request_context(
		"/config",
		headers={
			"X-User-ID": "alice",
			"X-Tenant-ID": "tenant-5678",
			"X-APG-Permissions": "config.read, config.write",
		},
	):
		response = await protected()
		assert g.current_user["user_id"] == "alice"
		assert g.current_user["tenant_id"] == "tenant-5678"

	assert response.get_json() == {"ok": True}
