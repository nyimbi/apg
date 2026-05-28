from datetime import datetime

import pytest

from capabilities.common.cach.models import CacheEntry, SecurityLevel
from capabilities.common.cach.quantum_security import QuantumSecurityEngine, ThreatLevel


def _cache_entry(**overrides):
	data = {
		"key": "secret:payment-token",
		"value": b"payment token credential",
		"size_bytes": len(b"payment token credential"),
		"original_size_bytes": len(b"payment token credential"),
		"tenant_id": "tenant_runtime",
		"encrypted": False,
		"security_level": SecurityLevel.QUANTUM_SAFE,
		"semantic_tags": ["payment", "credential"],
		"access_count": 140,
		"hit_count": 0,
		"miss_count": 40,
	}
	data.update(overrides)
	return CacheEntry(**data)


@pytest.mark.asyncio
async def test_cache_quantum_security_detects_entry_and_access_threats():
	engine = QuantumSecurityEngine({"behavioral_analysis": True, "adaptive_policies": True})
	await engine.initialize()
	engine.threat_intelligence["203.0.113.10"] = ThreatLevel.HIGH

	analysis = await engine.analyze_threat_patterns(
		{"secret:payment-token": _cache_entry()},
		[
			{
				"user_id": "user_runtime",
				"timestamp": datetime(2026, 5, 28, 2, 15),
				"source_ip": "203.0.113.10",
				"user_agent": "headless-bot",
				"authentication_method": "unknown",
				"status": "failed",
				"attempt_count": 9,
				"bytes_returned": 12 * 1024 * 1024,
				"key": "secret:payment-token",
			},
			{"user_id": "user_runtime", "timestamp": datetime(2026, 5, 28, 10, 0), "source_ip": "203.0.113.10"},
			{"user_id": "user_runtime", "timestamp": datetime(2026, 5, 28, 10, 5), "source_ip": "203.0.113.10"},
		],
	)

	threat_types = {threat["type"] for threat in analysis["threats_detected"]}
	assert "unencrypted_sensitive_entry" in threat_types
	assert "quantum_safe_entry_unencrypted" in threat_types
	assert analysis["anomalies_found"]
	assert analysis["risk_score"] >= engine.risk_threshold
	assert "require_quantum_safe_keys" in analysis["recommended_actions"]
	assert "rate_limit_suspicious_sources" in analysis["recommended_actions"]
	assert analysis["behavior_changes"] == [{"user_id": "user_runtime", "change": "baseline_established"}]

	adaptation = await engine.adapt_security_policies(analysis)

	assert adaptation["policies_updated"]
	assert adaptation["new_policies"]
	assert adaptation["risk_reduction"] > 0
	assert "threat_quantum_safe_entry_unencrypted" in engine.adaptive_policies

	await engine.shutdown()


@pytest.mark.asyncio
async def test_cache_quantum_transition_reports_readiness_from_key_state():
	engine = QuantumSecurityEngine({"quantum_transition_phase": 2})
	await engine.initialize()

	status = await engine.prepare_quantum_transition()

	assert status["current_phase"] == 2
	assert status["readiness_score"] > 0
	assert status["hybrid_keys_deployed"] >= 2
	assert status["migration_progress"] > 0

	await engine.shutdown()
