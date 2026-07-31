"""Integration test harness: spawns generated app as a subprocess for real HTTP tests."""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from compiler.ast_builder import (
    EntityDeclaration,
    EntityType,
    ModuleDeclaration,
    PropertyDeclaration,
    TypeAnnotation,
)
from compiler.code_generator import CodeGenConfig, PythonCodeGenerator


APG_TEST_PORT = int(os.environ.get("APG_TEST_PORT", "15432"))


def _minimal_product_module() -> ModuleDeclaration:
	return ModuleDeclaration(
		name="apg_integration",
		entities=[
			EntityDeclaration(
				entity_type=EntityType.FORM,
				name="Product",
				properties=[
					PropertyDeclaration("name", TypeAnnotation("str")),
					PropertyDeclaration("price", TypeAnnotation("float")),
				],
			)
		],
	)


def _wait_port(host: str, port: int, timeout: float = 10.0) -> bool:
	deadline = time.time() + timeout
	while time.time() < deadline:
		try:
			with socket.create_connection((host, port), timeout=0.5):
				return True
		except OSError:
			time.sleep(0.2)
	return False


@pytest.fixture(scope="session")
def running_app(tmp_path_factory):
	"""Compile the minimal Product APG source, run it, yield base URL, tear down."""
	work = tmp_path_factory.mktemp("apg_int_app")
	module = _minimal_product_module()
	files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(module)
	app_py = work / "app.py"
	app_py.write_text(files["app.py"], encoding="utf-8")
	env = os.environ.copy()
	env["APG_PORT"] = str(APG_TEST_PORT)
	env["APG_SECRET_KEY"] = secrets.token_hex(32)
	env["APG_DATABASE_URL"] = f"sqlite:///{work / 'apg_data.db'}"
	proc = subprocess.Popen(
		[sys.executable, "app.py"],
		cwd=str(work),
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
	)
	# Give it a moment to bind, then wait for the socket.
	time.sleep(2.0)
	if not _wait_port("127.0.0.1", APG_TEST_PORT, timeout=10.0):
		proc.terminate()
		out, err = proc.communicate(timeout=3)
		raise RuntimeError(
			f"generated app failed to listen on {APG_TEST_PORT}\nSTDOUT:\n{out.decode(errors='replace')}\nSTDERR:\n{err.decode(errors='replace')}"
		)
	base_url = f"http://127.0.0.1:{APG_TEST_PORT}"
	try:
		yield base_url
	finally:
		proc.terminate()
		try:
			proc.wait(timeout=5)
		except subprocess.TimeoutExpired:
			proc.kill()
			proc.wait(timeout=5)
