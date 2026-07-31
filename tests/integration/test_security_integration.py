"""Security integration tests for the generated app."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from compiler.ast_builder import (
	EntityDeclaration,
	EntityType,
	ModuleDeclaration,
	PropertyDeclaration,
	TypeAnnotation,
)
from compiler.code_generator import CodeGenConfig, PythonCodeGenerator


def test_rate_limiting_real_http(running_app):
	saw_429 = False
	for _ in range(102):
		r = requests.post(
			f"{running_app}/records/Product",
			json={"name": "RateProd", "price": 1.0},
			timeout=5,
		)
		if r.status_code == 429:
			saw_429 = True
			break
	assert saw_429, "expected at least one 429 within 102 rapid POSTs"


def test_production_rejects_default_key(tmp_path):
	module = ModuleDeclaration(
		name="apg_prod_probe",
		entities=[
			EntityDeclaration(
				entity_type=EntityType.FORM,
				name="Product",
				properties=[PropertyDeclaration("name", TypeAnnotation("str"))],
			)
		],
	)
	files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(module)
	app_py = tmp_path / "app.py"
	app_py.write_text(files["app.py"], encoding="utf-8")
	env = os.environ.copy()
	env["APG_PRODUCTION"] = "1"
	env.pop("APG_SECRET_KEY", None)
	env.pop("APG_SESSION_SECRET", None)
	env.pop("APG_JWT_SECRET", None)
	env["APG_PORT"] = "15499"
	proc = subprocess.Popen(
		[sys.executable, "app.py"],
		cwd=str(tmp_path),
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
	)
	try:
		rc = proc.wait(timeout=3)
	except subprocess.TimeoutExpired:
		proc.terminate()
		proc.wait(timeout=3)
		pytest.fail("production app did not exit when APG_SECRET_KEY was unset")
	assert rc != 0, f"expected non-zero exit, got {rc}"
