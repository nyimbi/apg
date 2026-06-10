"""APG Studio — service layer (thin facade over compiler and MANIFEST)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent.parent.parent.parent / "capabilities" / "MANIFEST.json"


class StudioService:
    """Exposes MANIFEST data and compiler integration for the Studio UI."""

    def __init__(self, tenant_id: str = "default") -> None:
        self._tenant_id = tenant_id
        self._manifest: dict[str, Any] | None = None

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            with open(_MANIFEST_PATH) as f:
                self._manifest = json.load(f)
        return self._manifest

    async def list_capabilities(self) -> list[dict[str, Any]]:
        m = self._load_manifest()
        return [
            {
                "id": c["id"],
                "display_name": c.get("display_name", c["id"]),
                "domain": c.get("domain", ""),
                "description": c.get("description", "")[:120],
                "provides": c.get("provides", []),
                "requires": c.get("requires", []),
                "service_method_count": c.get("service_method_count", 0),
            }
            for c in m["capabilities"].values()
        ]

    async def get_capability(self, cap_id: str) -> dict[str, Any] | None:
        m = self._load_manifest()
        return m["capabilities"].get(cap_id)

    async def get_stats(self) -> dict[str, Any]:
        m = self._load_manifest()
        caps = list(m["capabilities"].values())
        domains: dict[str, int] = {}
        for c in caps:
            d = c.get("domain", "other")
            domains[d] = domains.get(d, 0) + 1
        return {
            "capabilities": len(caps),
            "domains": len(domains),
            "domain_breakdown": domains,
            "avg_service_methods": sum(c.get("service_method_count", 0) for c in caps) // max(len(caps), 1),
        }

    async def compile_source(self, source: str, filename: str = "untitled.apg") -> dict[str, Any]:
        try:
            import sys
            repo_root = Path(__file__).parent.parent.parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from compiler.compiler import APGCompiler  # type: ignore[import]
            result = APGCompiler().compile_string(source, filename)
            return {
                "success": result.success,
                "files": dict(result.generated_files) if result.success else {},
                "errors": [str(e) for e in (result.errors or [])],
                "warnings": [str(w) for w in (result.warnings or [])],
            }
        except Exception as exc:
            return {"success": False, "files": {}, "errors": [str(exc)], "warnings": []}

    async def health_check(self) -> dict[str, Any]:
        try:
            stats = await self.get_stats()
            return {"status": "ok", **stats}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
