"""Standalone smoke test for an APG generated Python application."""

from __future__ import annotations

import json

import app


def main() -> int:
    report = app.self_test()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        return 1
    validation = report["checks"]["validation"]["checks"]
    for check_name in ("openapi_contract", "component_manifest", "route_dispatch"):
        errors = validation.get(check_name, {}).get("errors", [])
        if errors:
            print(json.dumps({"contract": check_name, "errors": errors}, indent=2, sort_keys=True))
            return 1
    capability_health = report["checks"].get("capability_health")
    if capability_health is not None and capability_health.get("healthy") is not True:
        print(json.dumps({"capability_health": capability_health}, indent=2, sort_keys=True))
        return 1
    component = app.component_manifest()
    required_routes = {"/health", "/self-test", "/component.json", "/openapi.json"}
    missing_routes = sorted(required_routes.difference(component["interfaces"]["http"]["paths"]))
    if missing_routes:
        print(json.dumps({"missing_routes": missing_routes}, indent=2, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
