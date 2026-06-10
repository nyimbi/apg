"""Deep health check framework for APG capabilities.

Provides dependency-graph health probing suitable for:
- Kubernetes liveness + readiness probes
- Monitoring dashboards
- Pre-flight checks before processing critical operations

Usage:
    checker = DeepHealthCheck("fintech_gwy")
    checker.add_dependency("postgresql", check_db_connection)
    checker.add_dependency("nats", check_nats_connection)
    checker.add_dependency("vault", check_vault_service)
    checker.add_dependency("mpesa_api", check_mpesa_connectivity)

    status = await checker.run()
    # Returns HealthStatus with per-component results

    @app.get("/health/ready")
    async def readiness():
        s = await checker.run()
        return jsonify(s.to_dict()), 200 if s.ready else 503
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_log = logging.getLogger(__name__)


class HealthLevel(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    name: str
    level: HealthLevel
    latency_ms: float
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    required: bool = True  # If True, failure makes the whole system UNHEALTHY


@dataclass
class HealthStatus:
    capability_id: str
    level: HealthLevel
    components: list[ComponentHealth]
    checked_at: float = field(default_factory=time.time)
    total_latency_ms: float = 0.0
    version: str = "1.0.0"

    @property
    def ready(self) -> bool:
        """True if all required components are healthy or degraded."""
        for c in self.components:
            if c.required and c.level == HealthLevel.UNHEALTHY:
                return False
        return True

    @property
    def alive(self) -> bool:
        """True if the process is running (always True if we can respond)."""
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "status": self.level.value,
            "ready": self.ready,
            "alive": self.alive,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "checked_at": self.checked_at,
            "version": self.version,
            "components": [
                {
                    "name": c.name,
                    "status": c.level.value,
                    "latency_ms": round(c.latency_ms, 2),
                    "message": c.message,
                    "required": c.required,
                    **({"details": c.details} if c.details else {}),
                }
                for c in self.components
            ],
        }


class DeepHealthCheck:
    """Orchestrates health checks for a capability and its dependencies.

    All checks run concurrently with individual timeouts. A slow dependency
    never blocks health reporting.

    Args:
        capability_id: The capability being checked.
        check_timeout: Maximum seconds to wait for any single check.
        version: Capability version string for /health response.
    """

    def __init__(
        self,
        capability_id: str,
        check_timeout: float = 5.0,
        version: str = "1.0.0",
    ) -> None:
        self._capability_id = capability_id
        self._check_timeout = check_timeout
        self._version = version
        self._checks: list[tuple[str, Any, bool]] = []  # (name, fn, required)

    def add_dependency(
        self,
        name: str,
        check_fn: Any,
        *,
        required: bool = True,
    ) -> "DeepHealthCheck":
        """Register a health check function.

        check_fn must be an async callable returning:
          - True / "ok" / {"status": "ok"} for healthy
          - A dict {"status": "degraded", "message": "..."} for degraded
          - Raise any exception for unhealthy

        Args:
            name: Component name (shown in /health response).
            check_fn: Async callable.
            required: If True, failure makes the capability UNHEALTHY (not DEGRADED).
        """
        self._checks.append((name, check_fn, required))
        return self

    def add_simple_check(
        self,
        name: str,
        check_fn: Any,
        *,
        required: bool = True,
    ) -> "DeepHealthCheck":
        """Alias for add_dependency."""
        return self.add_dependency(name, check_fn, required=required)

    async def run(self) -> HealthStatus:
        """Run all health checks concurrently and return a HealthStatus."""
        t_start = time.monotonic()

        async def _run_one(name: str, fn: Any, required: bool) -> ComponentHealth:
            t0 = time.monotonic()
            try:
                result = await asyncio.wait_for(fn(), timeout=self._check_timeout)
                latency_ms = (time.monotonic() - t0) * 1000

                if result is True or result == "ok":
                    return ComponentHealth(name, HealthLevel.HEALTHY, latency_ms, required=required)

                if isinstance(result, dict):
                    status = result.get("status", "ok")
                    msg = result.get("message", "")
                    details = {k: v for k, v in result.items() if k not in ("status", "message")}
                    level = {
                        "healthy": HealthLevel.HEALTHY,
                        "ok": HealthLevel.HEALTHY,
                        "degraded": HealthLevel.DEGRADED,
                        "warning": HealthLevel.DEGRADED,
                        "unhealthy": HealthLevel.UNHEALTHY,
                        "error": HealthLevel.UNHEALTHY,
                    }.get(str(status).lower(), HealthLevel.HEALTHY)
                    return ComponentHealth(name, level, latency_ms, msg, details, required=required)

                return ComponentHealth(name, HealthLevel.HEALTHY, latency_ms, required=required)

            except asyncio.TimeoutError:
                latency_ms = (time.monotonic() - t0) * 1000
                _log.warning("Health check timeout: %s (%.0fms)", name, latency_ms)
                return ComponentHealth(
                    name, HealthLevel.UNHEALTHY, latency_ms,
                    f"Timed out after {self._check_timeout:.1f}s",
                    required=required,
                )
            except Exception as exc:
                latency_ms = (time.monotonic() - t0) * 1000
                _log.warning("Health check failed: %s — %s: %s", name, type(exc).__name__, exc)
                return ComponentHealth(
                    name, HealthLevel.UNHEALTHY, latency_ms,
                    f"{type(exc).__name__}: {exc}",
                    required=required,
                )

        components = await asyncio.gather(
            *[_run_one(name, fn, req) for name, fn, req in self._checks],
            return_exceptions=False,
        )

        total_ms = (time.monotonic() - t_start) * 1000

        # Determine overall level
        has_required_unhealthy = any(
            c.level == HealthLevel.UNHEALTHY and c.required for c in components
        )
        has_degraded = any(c.level == HealthLevel.DEGRADED for c in components)
        has_optional_unhealthy = any(
            c.level == HealthLevel.UNHEALTHY and not c.required for c in components
        )

        if has_required_unhealthy:
            overall = HealthLevel.UNHEALTHY
        elif has_degraded or has_optional_unhealthy:
            overall = HealthLevel.DEGRADED
        else:
            overall = HealthLevel.HEALTHY

        return HealthStatus(
            capability_id=self._capability_id,
            level=overall,
            components=list(components),
            total_latency_ms=total_ms,
            version=self._version,
        )


# ── Common check functions ────────────────────────────────────────

async def check_postgresql(dsn: str) -> dict[str, Any]:
    """Check PostgreSQL connectivity and basic query."""
    try:
        import asyncpg  # type: ignore[import]
        conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=3.0)
        await conn.execute("SELECT 1")
        await conn.close()
        return {"status": "healthy"}
    except ImportError:
        return {"status": "degraded", "message": "asyncpg not installed"}
    except Exception as exc:
        return {"status": "unhealthy", "message": str(exc)}


async def check_nats(url: str) -> dict[str, Any]:
    """Check NATS connectivity."""
    try:
        import nats  # type: ignore[import]
        nc = await asyncio.wait_for(nats.connect(url), timeout=3.0)
        await nc.close()
        return {"status": "healthy"}
    except ImportError:
        return {"status": "degraded", "message": "nats-py not installed"}
    except Exception as exc:
        return {"status": "unhealthy", "message": str(exc)}


async def check_http_endpoint(url: str, expected_status: int = 200) -> dict[str, Any]:
    """Check an HTTP endpoint responds within timeout."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(url)
        if r.status_code == expected_status:
            return {"status": "healthy", "http_status": r.status_code}
        return {"status": "degraded", "message": f"HTTP {r.status_code}", "http_status": r.status_code}
    except Exception as exc:
        return {"status": "unhealthy", "message": str(exc)}


async def check_ollama(base_url: str, model: str = "mistral:7b") -> dict[str, Any]:
    """Check Ollama is running and model is available."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base_url}/api/tags")
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        if any(m.startswith(model.split(":")[0]) for m in models):
            return {"status": "healthy", "models": models[:5]}
        return {"status": "degraded", "message": f"Model {model!r} not pulled", "models": models[:5]}
    except Exception as exc:
        return {"status": "unhealthy", "message": str(exc)}
