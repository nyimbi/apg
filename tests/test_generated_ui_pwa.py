"""Generated UI PWA and budget regressions."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from click.testing import CliRunner

from cli.main import cli
from compiler.compiler import compile_apg_file


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_20 = REPO_ROOT / "examples" / "20_enterprise_erp_platform" / "main.apg"


def test_generated_ui_emits_pwa_assets_and_page_hooks():
	result = compile_apg_file(str(EXAMPLE_20))

	assert result.success, result.errors
	manifest = json.loads(result.generated_files["static/manifest.webmanifest"])
	sw = result.generated_files["static/sw.js"]
	app_py = result.generated_files["app.py"]

	assert manifest["name"] == "Enterprise Erp"
	assert manifest["id"] == "/ui"
	assert manifest["start_url"] == "/ui"
	assert manifest["orientation"] == "any"
	assert {"business", "productivity"}.issubset(set(manifest["categories"]))
	assert [shortcut["url"] for shortcut in manifest["shortcuts"]] == [
		"/ui",
		"/ui/workflows",
		"/ui/marketplace",
	]
	assert manifest["theme_color"].startswith("#")
	assert manifest["icons"][0]["src"] == "/static/icon.svg"
	assert "static/icon.svg" in result.generated_files
	assert "APG_STATIC" in sw
	assert "apg-ui-v2" in sw
	assert "SKIP_WAITING" in sw
	assert "/static/apg.css" in sw
	assert "/ui" in sw
	assert '<link rel="manifest" href="/static/manifest.webmanifest">' in app_py
	assert '<meta name="theme-color" content="#1E5B5A">' in app_py
	assert "apg-offline-banner" in app_py
	assert 'navigator.serviceWorker.register("/static/sw.js")' in app_py


def test_generated_shell_chrome_exposes_power_user_controls():
	result = compile_apg_file(str(EXAMPLE_20))
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	with namespace["_flask_app"].test_client() as client:
		response = client.get("/ui/entities/Vendor")

	assert response.status_code == 200
	html = response.get_data(as_text=True)
	assert 'class="apg-btn apg-btn-secondary apg-command-trigger"' in html
	assert 'aria-haspopup="dialog"' in html
	assert 'role="dialog" aria-modal="true" aria-label="Command palette"' in html
	assert "apg-notification-panel" in html
	assert "apg-notification-list" in html
	assert "apg-install-btn" in html
	assert "apg-update-btn" in html
	assert 'href="/ui/entities/Vendor" aria-current="page"' in html
	assert "apgApplyUpdate" in html
	assert "beforeinstallprompt" in html


def test_generated_ui_static_asset_budgets():
	result = compile_apg_file(str(EXAMPLE_20))

	assert result.success, result.errors
	css_gzip_bytes = len(gzip.compress(result.generated_files["static/apg.css"].encode("utf-8")))
	js_gzip_bytes = sum(
		len(gzip.compress(content.encode("utf-8")))
		for path, content in result.generated_files.items()
		if path.startswith("static/") and path.endswith(".js")
	)

	assert css_gzip_bytes <= 60 * 1024
	assert js_gzip_bytes <= 120 * 1024


def test_cli_baseline_refresh_alias_is_wired():
	result = CliRunner().invoke(cli, ["baseline", "--help"])

	assert result.exit_code == 0, result.output
	assert "--refresh" in result.output
	assert "--refresh-outputs" in result.output
