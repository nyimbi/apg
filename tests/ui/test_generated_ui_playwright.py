"""Opt-in Playwright smoke harness for generated UI examples.

Run with APG_RUN_PLAYWRIGHT=1 after installing Playwright and browser binaries.
The regular pytest gate skips this file when browser tooling is unavailable.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from compiler.compiler import compile_apg_file


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_IDS = ("01_basic_todo", "10_workflow_with_database", "20_enterprise_erp_platform")
VIEWPORTS = ((375, 812), (768, 1024), (1440, 900))


pytestmark = pytest.mark.skipif(
	os.environ.get("APG_RUN_PLAYWRIGHT") != "1",
	reason="set APG_RUN_PLAYWRIGHT=1 to run browser UI smoke tests",
)


def _free_port() -> int:
	with socket.socket() as sock:
		sock.bind(("127.0.0.1", 0))
		return int(sock.getsockname()[1])


def _compile_example(example_id: str, output_dir: Path) -> None:
	source = REPO_ROOT / "examples" / example_id / "main.apg"
	result = compile_apg_file(str(source))
	assert result.success, result.errors
	for path, content in result.generated_files.items():
		target = output_dir / path
		target.parent.mkdir(parents=True, exist_ok=True)
		target.write_text(content, encoding="utf-8")


def _start_app(output_dir: Path, port: int) -> subprocess.Popen[str]:
	code = (
		"import importlib.util;"
		"spec=importlib.util.spec_from_file_location('generated_app','app.py');"
		"app=importlib.util.module_from_spec(spec);"
		"spec.loader.exec_module(app);"
		f"app._flask_app.run(host='127.0.0.1',port={port},debug=False,use_reloader=False)"
	)
	return subprocess.Popen(
		[sys.executable, "-c", code],
		cwd=output_dir,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)


def _wait_for(url: str, timeout: float = 15.0) -> None:
	import urllib.request

	deadline = time.time() + timeout
	last_error: Exception | None = None
	while time.time() < deadline:
		try:
			with urllib.request.urlopen(url, timeout=1) as response:
				if response.status < 500:
					return
		except Exception as exc:  # pragma: no cover - diagnostic path
			last_error = exc
		time.sleep(0.2)
	raise AssertionError(f"generated app did not start at {url}: {last_error}")


def test_generated_examples_browser_smoke(tmp_path: Path):
	playwright_mod = pytest.importorskip("playwright.sync_api")

	for example_id in EXAMPLE_IDS:
		output_dir = tmp_path / example_id
		_compile_example(example_id, output_dir)
		port = _free_port()
		process = _start_app(output_dir, port)
		try:
			base_url = f"http://127.0.0.1:{port}"
			_wait_for(f"{base_url}/ui")
			with playwright_mod.sync_playwright() as p:
				browser = p.chromium.launch()
				page = browser.new_page()
				for theme in ("light", "dark"):
					for width, height in VIEWPORTS:
						page.set_viewport_size({"width": width, "height": height})
						page.goto(f"{base_url}/ui", wait_until="networkidle")
						page.evaluate("localStorage.setItem('apg-theme', arguments[0])", theme)
						page.reload(wait_until="networkidle")
						page.locator("#content").wait_for()
						page.screenshot(path=tmp_path / f"{example_id}-{theme}-{width}.png", full_page=True)
				page.keyboard.press("Control+K")
				page.locator("#apg-cmd-input").fill("a")
				browser.close()
		finally:
			process.terminate()
			try:
				process.wait(timeout=5)
			except subprocess.TimeoutExpired:
				process.kill()
