"""Executable contextual-risk coverage for AUTH."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from capabilities.common.auth.contextual_risk import (
	AuthContext,
	AuthRequirement,
	ContextualRiskEngine,
	DeviceRisk,
	LocationRisk,
	NetworkRisk,
	RiskLevel,
	TimeRisk,
)


@pytest.mark.asyncio
async def test_contextual_risk_uses_configured_ip_intelligence():
	engine = ContextualRiskEngine()
	engine.configure_risk_intelligence({
		"vpn_cidrs": ["203.0.113.0/25"],
		"tor_exit_cidrs": ["198.51.100.10/32"],
		"datacenter_cidrs": ["203.0.113.0/24"],
		"public_wifi_cidrs": ["203.0.113.128/25"],
		"threat_ip_scores": {"198.51.100.10": 0.93},
		"ip_reputation_scores": {"198.51.100.10": 0.07},
	})

	location = await engine.assess_location_risk(
		"user-1",
		{"ip_address": "198.51.100.10", "country": "KE", "city": "Nairobi"},
	)
	network = await engine.assess_network_risk(
		"user-1",
		{"ip_address": "203.0.113.140", "connection_count": 1},
	)

	assert location.is_tor is True
	assert location.ip_reputation_score == 0.07
	assert network.is_datacenter is True
	assert network.is_public_wifi is True
	assert network.is_residential is False
	assert await engine._get_threat_intelligence("198.51.100.10") == 0.93


@pytest.mark.asyncio
async def test_contextual_risk_uses_device_and_calendar_evidence():
	engine = ContextualRiskEngine()
	engine.configure_risk_intelligence({
		"trusted_device_ids": ["device-ok"],
		"blocked_device_ids": ["device-bad"],
		"holiday_dates": ["2026-06-01"],
	})

	trusted_reputation = await engine._get_device_reputation("device-ok", "Mozilla/5.0 Chrome Safari")
	blocked_reputation = await engine._get_device_reputation("device-bad", "Mozilla/5.0 Chrome Safari")
	headless_integrity = await engine._assess_browser_integrity("HeadlessChrome Selenium WebDriver")
	device = await engine.assess_device_risk(
		"user-1",
		{
			"device_id": "device-bad",
			"user_agent": "HeadlessChrome Selenium WebDriver",
			"security_indicators": ["root", "keylogger"],
		},
	)
	time_risk = await engine.assess_time_risk(
		"user-1",
		{"timestamp": "2026-06-01T10:00:00Z", "timezone": "Africa/Nairobi"},
	)

	assert trusted_reputation > 0.9
	assert blocked_reputation < 0.1
	assert headless_integrity < 0.4
	assert device.is_jailbroken is True
	assert device.has_malware_indicators is True
	assert device.device_reputation_score < 0.1
	assert time_risk.is_holiday is True


@pytest.mark.asyncio
async def test_contextual_risk_scores_impossible_travel_and_requires_step_up():
	engine = ContextualRiskEngine()
	engine.configure_risk_intelligence({
		"vpn_cidrs": ["203.0.113.0/24"],
		"high_risk_countries": ["ZZ"],
	})

	old_context_time = datetime.utcnow() - timedelta(hours=1)
	engine._user_locations["user-1"] = [
		LocationRisk(country="KE", city="Nairobi", ip_address="10.0.0.1")
	]
	engine._user_time_patterns["user-1"] = [old_context_time]

	location = await engine.assess_location_risk(
		"user-1",
		{"ip_address": "203.0.113.25", "country": "US", "city": "Chicago"},
	)
	device = DeviceRisk(
		device_id=None,
		user_agent="HeadlessChrome Selenium WebDriver",
		is_known_device=False,
		browser_integrity_score=0.25,
		device_reputation_score=0.35,
	)
	time_risk = await engine.assess_time_risk(
		"user-1",
		{
			"timestamp": datetime.utcnow(),
			"timezone": "UTC",
			"location": {"country": "US", "city": "Chicago"},
		},
	)
	network = NetworkRisk(
		ip_address="203.0.113.25",
		is_residential=False,
		is_datacenter=True,
		is_public_wifi=False,
		threat_intel_score=0.60,
		connection_count=9,
	)

	context = AuthContext(
		user_id="user-1",
		location_risk=location,
		device_risk=device,
		time_risk=time_risk,
		network_risk=network,
	)

	assessment = await engine.calculate_auth_requirements(context)

	assert time_risk.velocity_risk_score > 0.5
	assert assessment.risk_level in {RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL}
	assert AuthRequirement.MFA_REQUIRED in assessment.required_auth_methods
	assert assessment.risk_reasons
