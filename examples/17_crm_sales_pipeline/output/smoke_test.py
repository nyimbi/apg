"""Standalone smoke test for an APG generated Python application."""

from __future__ import annotations

import json

import app


def main() -> int:
    report = app.self_test()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
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
