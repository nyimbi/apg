"""Standalone APG Patient Management application.

Run directly::

    python -m apg_healthcare_pmt
    apg-healthcare-pmt --port 8080

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


def semantic_model() -> dict:
    """APG semantic model for this capability."""
    contract = get_capability_contract()
    return {
        "format": "apg.semantic-model.v1",
        "ok": True,
        "app": {"name": CAPABILITY_ID, "version": CAPABILITY_VERSION, "description": "Patient Management package-backed APG capability", "entity_count": 15},
        "capabilities": {
            CAPABILITY_ID: {
                "name": contract.get("display_name", CAPABILITY_NAME),
                "version": contract["version"],
                "provides": contract["provides"],
                "requires": contract["requires"],
                "configuration": contract["configuration"],
                "rules": contract["rule_engine"]["rules"],
                "rule_engine": contract["rule_engine"],
                "ui": contract["ui"],
                "screens": {r["name"]: {"route": r["path"], "component": r["component"], "permission": r["permission"]} for r in contract["ui"]["routes"]},
                "theme": contract["theme"],
                "streaming": contract["streaming"],
                "runtime": {"entrypoint": "app.py", "service": "service.py", "api": "api.py", "views": "views.py"},
            }
        },
        "contracts": {CAPABILITY_ID: {"id": CAPABILITY_ID, "provides": contract["provides"], "requires": contract["requires"], "configuration": contract["configuration"]}},
        "composition": {"capability_dependencies": {CAPABILITY_ID: contract["requires"]}, "agent_teams": {}, "applications": {}},
        "packages": {CAPABILITY_ID: {"entrypoint": "app.py", "profile": "capability"}},
        "deployment": {"source": "capability_contract.py", "target": "python"},
        "graphs": {"capability": {"kind": "capability", "nodes": 1, "edges": len(contract["requires"])}},
        "diagnostics": [],
    }


def component_manifest() -> dict:
    """APG component manifest for this capability."""
    return {
        "format": "apg.component-manifest.v1",
        "kind": "apg.generated_application",
        "name": CAPABILITY_ID,
        "display_name": CAPABILITY_NAME,
        "target": "python",
        "interfaces": {"health": "/health", "self_test": "/self-test", "semantic_model": "/semantic-model.json"},
        "capabilities": [CAPABILITY_ID],
    }


def self_test() -> dict:
    """Run capability self-test checks."""
    model = semantic_model()
    manifest = component_manifest()
    capability = model.get("capabilities", {}).get(CAPABILITY_ID, {})
    errors: list[str] = []
    if model.get("format") != "apg.semantic-model.v1":
        errors.append("semantic model format mismatch")
    if not capability:
        errors.append("capability missing from semantic model")
    if "patient_registration" not in capability.get("provides", []):
        errors.append("patient_registration provide missing")
    if "dashboard" not in capability.get("screens", {}):
        errors.append("dashboard screen missing")
    if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
        errors.append("component manifest semantic model interface mismatch")
    return {
        "passed": not errors,
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
        "capability": CAPABILITY_ID,
    }


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

    @app.get("/semantic-model.json")
    def semantic_model_route():
        return jsonify(semantic_model())

    @app.get("/self-test")
    def self_test_route():
        return jsonify(self_test())

    @app.get("/component.json")
    def component_route():
        return jsonify(component_manifest())

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
        prog="apg-healthcare-pmt",
        description=f"APG Patient Management capability server",
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
