"""Boot a single APG capability behind an honest health endpoint.

Container entrypoint for ``Dockerfile.capability``. The capability is named by
``--capability`` or ``$APG_CAPABILITY``; the image refuses to report healthy
unless that module imports and its app factory succeeds.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os

from flask import Flask, jsonify

_log = logging.getLogger("apg.capability_runner")


def load_app(module_path: str) -> Flask:
	"""Import `module_path` and build its Flask app via `create_app()`."""
	module = importlib.import_module(module_path)

	factory = getattr(module, "create_app", None)
	if factory is None:
		raise RuntimeError(f"{module_path} exposes no create_app() factory")

	app = factory()
	if not isinstance(app, Flask):
		raise RuntimeError(f"{module_path}.create_app() returned {type(app)!r}, expected Flask")

	return app


def register_health(app: Flask, module_path: str) -> Flask:
	"""Attach a root-level /health so HEALTHCHECK reflects real boot success."""

	@app.get("/health")
	def health() -> tuple[dict[str, str], int]:
		return {"status": "ok", "capability": module_path}, 200

	return app


def main() -> None:
	logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--capability",
		default=os.environ.get("APG_CAPABILITY", ""),
		help="importable module exposing create_app(), e.g. capabilities.agriculture.mkt.app",
	)
	parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
	args = parser.parse_args()

	if not args.capability:
		raise SystemExit("no capability specified: pass --capability or set $APG_CAPABILITY")

	app = register_health(load_app(args.capability), args.capability)
	_log.info("serving capability %s on port %d", args.capability, args.port)
	app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
	main()