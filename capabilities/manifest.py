"""APG Capability Manifest — programmatic access.

Usage::

    from capabilities.manifest import get_capability, find_capabilities, get_by_path

    # Code to description
    cap = get_capability("intel_alerts")
    print(cap["display_name"], cap["description"])

    # Path to description
    cap = get_by_path("capabilities/intel/alerts")

    # Package to description
    cap = get_by_package("apg-intel-alerts")

    # Search by keyword
    results = find_capabilities("alerts")

    # All in a domain
    intel_caps = get_domain("intel")

    # Get all method names for a capability
    methods = get_capability("intel_alerts")["service_methods"]
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_MANIFEST_PATH = Path(__file__).parent / "MANIFEST.json"


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def get_manifest() -> dict[str, Any]:
    """Return the full manifest dict."""
    return _load()


def get_capability(capability_id: str) -> dict[str, Any] | None:
    """Return a capability entry by its ID.  Returns None if not found."""
    return _load()["capabilities"].get(capability_id)


def get_by_path(path: str) -> dict[str, Any] | None:
    """Return a capability entry given its filesystem path (e.g. capabilities/intel/alerts)."""
    cap_id = _load()["by_path"].get(path.rstrip("/"))
    return _load()["capabilities"].get(cap_id) if cap_id else None


def get_by_package(package_name: str) -> dict[str, Any] | None:
    """Return a capability entry given its PyPI package name (e.g. apg-intel-alerts)."""
    cap_id = _load()["by_package"].get(package_name)
    return _load()["capabilities"].get(cap_id) if cap_id else None


def get_domain(domain: str) -> list[dict[str, Any]]:
    """Return all capability entries in a domain."""
    cap_ids = _load()["by_domain"].get(domain, [])
    caps = _load()["capabilities"]
    return [caps[c] for c in sorted(cap_ids) if c in caps]


def list_domains() -> list[str]:
    """Return sorted list of all domains."""
    return sorted(_load()["by_domain"].keys())


def find_capabilities(keyword: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search capabilities by keyword across id, name, description, provides, and methods.

    Returns up to *limit* matching capability entries, ordered by relevance.
    """
    keyword = keyword.lower().strip()
    if not keyword:
        return []

    caps = _load()["capabilities"]
    index = _load()["search_index"]
    scored: list[tuple[int, str]] = []

    for entry in index:
        score = 0
        cap_id = entry["id"]
        text = entry["text"]
        keywords = entry["keywords"]

        # Exact ID match
        if keyword == cap_id:
            score += 100
        # ID contains keyword
        elif keyword in cap_id:
            score += 50
        # Keyword exact match in keywords list
        elif keyword in keywords:
            score += 40
        # Full text match
        elif keyword in text:
            score += 10 + text.count(keyword) * 2

        if score > 0:
            scored.append((score, cap_id))

    scored.sort(reverse=True)
    return [caps[cap_id] for _, cap_id in scored[:limit] if cap_id in caps]


def all_capabilities() -> list[dict[str, Any]]:
    """Return all capabilities sorted by domain then id."""
    caps = _load()["capabilities"]
    return [caps[k] for k in sorted(caps)]


def capability_count() -> int:
    """Return total number of capabilities."""
    return _load()["capability_count"]


# ── Convenience: path-to-id lookup ──────────────────────────────────────────
def path_to_id(path: str) -> str | None:
    """Convert a filesystem path to a capability ID."""
    return _load()["by_path"].get(path.rstrip("/"))


def id_to_path(capability_id: str) -> str | None:
    """Convert a capability ID to its filesystem path."""
    cap = _load()["capabilities"].get(capability_id)
    return cap["path"] if cap else None


def id_to_package(capability_id: str) -> str | None:
    """Convert a capability ID to its PyPI package name."""
    cap = _load()["capabilities"].get(capability_id)
    return cap["package"] if cap else None


def package_to_id(package_name: str) -> str | None:
    """Convert a PyPI package name to a capability ID."""
    return _load()["by_package"].get(package_name)
