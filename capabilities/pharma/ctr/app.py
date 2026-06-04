"""Standalone APG Clinical Trials Management application.

Run directly::

    python -m apg_pharma_ctr
    apg-pharma-ctr --port 8080

Or as part of the APG platform via the capability registry.
"""
from __future__ import annotations

import argparse
from flask import Flask, jsonify

from .capability_contract import (
    CAPABILITY_ID,
    CAPABILITY_NAME,
    CAPABILITY_VERSION,
    get_capability_contract,
    evaluate_capability_rules,
)


def create_app(config: dict | None = None) -> Flask:
    """Create and configure the standalone Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    app.config["CAPABILITY_ID"] = CAPABILITY_ID
    app.config["CAPABILITY_VERSION"] = CAPABILITY_VERSION
    if config:
        app.config.update(config)

    # Register the API blueprint
    try:
        from .api import blueprint as api_bp
        app.register_blueprint(api_bp, url_prefix="/api/v1")
    except (ImportError, AttributeError):
        pass

    # Register the views blueprint
    try:
        from .views import blueprint as views_bp
        app.register_blueprint(views_bp)
    except (ImportError, AttributeError):
        pass

    @app.get("/health")
    def health():
        """Liveness probe."""
        return jsonify({
            "status": "ok",
            "capability": CAPABILITY_ID,
            "version": CAPABILITY_VERSION,
        })

    @app.get("/contract")
    def contract():
        """Return the full capability contract."""
        return jsonify(get_capability_contract())

    @app.post("/evaluate")
    def evaluate():
        """Evaluate capability rules against a context payload."""
        from flask import request
        ctx = request.get_json(force=True, silent=True) or {}
        return jsonify(evaluate_capability_rules(ctx))

    @app.get("/openapi.json")
    def openapi():
        """Minimal OpenAPI 3 description."""
        return jsonify({
            "openapi": "3.1.0",
            "info": {
                "title": CAPABILITY_NAME,
                "version": CAPABILITY_VERSION,
                "description": f"APG {CAPABILITY_NAME} standalone API",
            },
            "paths": {
                "/health": {"get": {"summary": "Liveness probe", "responses": {"200": {"description": "ok"}}}},
                "/contract": {"get": {"summary": "Capability contract", "responses": {"200": {"description": "contract"}}}},
            },
        })

    return app


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — run the standalone capability server."""
    parser = argparse.ArgumentParser(
        prog="apg-pharma-ctr",
        description=f"APG Clinical Trials Management capability server",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--tenant", default="default", help="Default tenant ID")
    args = parser.parse_args(argv)

    app = create_app({"DEFAULT_TENANT": args.tenant})
    print(f"APG {CAPABILITY_NAME} v{CAPABILITY_VERSION} starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
