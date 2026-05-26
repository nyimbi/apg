"""Context resolution regressions for Facial Recognition APIs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = REPO_ROOT / "capabilities" / "common" / "frec"
CONTEXT_PATH = CAPABILITY_PATH / "context.py"
API_PATH = CAPABILITY_PATH / "api.py"


def _context_helpers() -> dict[str, Any]:
	source = CONTEXT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Iterable": Iterable,
		"Optional": Optional,
		"os": __import__("os"),
	}
	exec(compile(source[start:], str(CONTEXT_PATH), "exec"), namespace)
	return namespace


def _request(headers: dict[str, str] | None = None, args: dict[str, str] | None = None):
	return SimpleNamespace(headers=headers or {}, args=args or {})


def test_frec_api_delegates_tenant_context_resolution():
	source = API_PATH.read_text(encoding="utf-8")

	assert "from .context import resolve_tenant_id" in source
	assert "return resolve_tenant_id(request=request)" in source
	assert "request.headers.get('X-Tenant-ID', 'default_tenant')" not in source


def test_frec_context_resolves_tenant_precedence(monkeypatch):
	helpers = _context_helpers()
	resolve = helpers["resolve_tenant_id"]

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "frec-env-tenant")
	assert resolve(_request()) == "frec-env-tenant"

	assert resolve(
		_request(headers={"X-Tenant-ID": "header-tenant"}),
		g=SimpleNamespace(tenant_id="g-tenant"),
	) == "g-tenant"
	assert resolve(_request(headers={"X-APG-Tenant-ID": "header-tenant"})) == "header-tenant"
	assert resolve(_request(args={"tenant": "query-tenant"})) == "query-tenant"
