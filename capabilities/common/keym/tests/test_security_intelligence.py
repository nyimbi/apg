#!/usr/bin/env python3
"""
APG Key Management - Security Intelligence Tests
Comprehensive test suite for behavioral analytics and security intelligence

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from ..security_intelligence import (
	SecurityIntelligenceEngine, BehavioralAnalytics, ThreatCorrelationEngine,
	SecurityAlert, BehavioralPattern, ThreatIndicator, SecurityMetrics,
	create_security_intelligence_engine
)
from ..models import KeyOperation, SecurityThreat


@pytest.fixture
async def security_engine():
	"""Fixture for security intelligence engine"""
	engine = SecurityIntelligenceEngine()
	await engine.initialize({
		'tenant_id': 'test_tenant',
		'ml_model_path': '/tmp/test_models',
		'anomaly_threshold': 0.75,
		'test_mode': True
	})
	return engine


@pytest.fixture
def sample_operations():
	"""Fixture for sample key operations"""
	operations = []
	base_time = datetime.utcnow()
	
	for i in range(50):
		operation = KeyOperation(
			key_id=f"key_{i % 5}",  # 5 different keys
			operation_type="encrypt" if i % 2 == 0 else "decrypt",
			user_id=f"user_{i % 3}@datacraft.co.ke",  # 3 different users
			application_id=f"app_{i % 2}",  # 2 different apps
			request_ip=f"192.168.1.{100 + (i % 50)}",
			timestamp=base_time - timedelta(minutes=i * 5)
		)
		operations.append(operation)
	
	return operations


class TestSecurityIntelligenceEngine:
	"""Test SecurityIntelligenceEngine class"""
	
	@pytest.mark.asyncio
	async def test_engine_initialization(self):
		"""Test security intelligence engine initialization"""
		engine = SecurityIntelligenceEngine()
		assert not engine.is_initialized
		
		config = {
			'tenant_id': 'test_tenant',
			'anomaly_threshold': 0.8,
			'correlation_window_minutes': 30
		}
		await engine.initialize(config)
		
		assert engine.is_initialized
		assert engine.config == config
		assert isinstance(engine.behavioral_patterns, dict)
		assert isinstance(engine.threat_indicators, list)
		assert isinstance(engine.security_alerts, list)
	
	@pytest.mark.asyncio
	async def test_factory_function(self):
		"""Test security engine factory function"""
		engine = await create_security_intelligence_engine()
		assert isinstance(engine, SecurityIntelligenceEngine)
		assert engine.is_initialized
	
	@pytest.mark.asyncio
	async def test_analyze_key_operation(self, security_engine):
		"""Test key operation analysis"""
		operation = KeyOperation(
			key_id="test_key_123",
			operation_type="encrypt",
			user_id="test@datacraft.co.ke",
			application_id="test-app",
			request_ip="192.168.1.100",
			session_id="session_123"
		)
		
		analysis = await security_engine.analyze_key_operation(operation)
		
		assert analysis is not None
		assert 'risk_score' in analysis
		assert 'anomaly_indicators' in analysis
		assert 'behavioral_deviations' in analysis
		assert 'threat_level' in analysis
		assert 'recommended_actions' in analysis
		
		# Validate risk score
		assert 0.0 <= analysis['risk_score'] <= 1.0
		
		# Validate threat level
		assert analysis['threat_level'] in ['low', 'medium', 'high', 'critical']
	
	@pytest.mark.asyncio
	async def test_detect_behavioral_anomalies(self, security_engine, sample_operations):
		"""Test behavioral anomaly detection"""
		# Build behavioral baseline
		await security_engine.build_behavioral_baseline(sample_operations)
		
		# Create anomalous operation
		anomalous_operation = KeyOperation(
			key_id="key_0",
			operation_type="encrypt",
			user_id="unknown_user@example.com",  # Unknown user
			application_id="suspicious_app",  # Unknown app
			request_ip="10.0.0.1",  # Different IP range
			timestamp=datetime.utcnow()
		)
		
		anomalies = await security_engine.detect_behavioral_anomalies(
			[anomalous_operation], 
			"test@datacraft.co.ke"
		)
		
		assert isinstance(anomalies, list)
		for anomaly in anomalies:
			assert 'operation_id' in anomaly
			assert 'anomaly_type' in anomaly
			assert 'severity' in anomaly
			assert 'confidence' in anomaly
			assert 'description' in anomaly
			
			# Validate confidence
			assert 0.0 <= anomaly['confidence'] <= 1.0
	
	@pytest.mark.asyncio
	async def test_correlate_threat_patterns(self, security_engine):
		"""Test threat pattern correlation"""
		# Create threat indicators
		threat_indicators = [
			ThreatIndicator(
				indicator_type="unusual_access_pattern",
				severity="medium",
				confidence=0.85,
				source="behavioral_analysis",
				metadata={'user_id': 'test@datacraft.co.ke', 'deviation_score': 0.75}
			),
			ThreatIndicator(
				indicator_type="suspicious_ip",
				severity="high",
				confidence=0.92,
				source="ip_intelligence",
				metadata={'ip_address': '192.168.1.100', 'threat_score': 0.9}
			)
		]
		
		correlations = await security_engine.correlate_threat_patterns(
			threat_indicators, 
			"test@datacraft.co.ke"
		)
		
		assert isinstance(correlations, list)
		for correlation in correlations:
			assert 'correlation_id' in correlation
			assert 'threat_level' in correlation
			assert 'confidence' in correlation
			assert 'related_indicators' in correlation
			assert 'attack_scenario' in correlation
			assert 'recommended_actions' in correlation
	
	@pytest.mark.asyncio
	async def test_generate_security_alerts(self, security_engine):
		"""Test security alert generation"""
		# Create high-risk scenario
		threat_data = {
			'threat_type': 'credential_stuffing_attack',
			'severity': 'high',
			'confidence': 0.95,
			'affected_keys': ['key_001', 'key_002', 'key_003'],
			'source_ips': ['192.168.1.100', '192.168.1.101'],
			'attack_timeline': {
				'start': datetime.utcnow() - timedelta(minutes=30),
				'end': datetime.utcnow()
			}
		}
		
		alert = await security_engine.generate_security_alert(
			threat_data, 
			"admin@datacraft.co.ke"
		)
		
		assert isinstance(alert, SecurityAlert)
		assert alert.tenant_id == "test_tenant"
		assert alert.alert_type == "credential_stuffing_attack"
		assert alert.severity == "high"
		assert alert.confidence == 0.95
		assert len(alert.affected_resources) == 3
		assert alert.status == "new"
		assert alert.created_by == "admin@datacraft.co.ke"
	
	@pytest.mark.asyncio
	async def test_real_time_monitoring(self, security_engine):
		"""Test real-time security monitoring"""
		# Start monitoring
		monitoring_config = {
			'check_interval_seconds': 5,
			'batch_size': 100,
			'alert_threshold': 0.8
		}
		
		result = await security_engine.start_real_time_monitoring(
			monitoring_config, 
			"admin@datacraft.co.ke"
		)
		
		assert result is not None
		assert result['monitoring_enabled'] is True
		assert 'monitoring_id' in result
		assert 'next_check' in result
	
	@pytest.mark.asyncio
	async def test_get_security_dashboard_data(self, security_engine):
		"""Test security dashboard data retrieval"""
		dashboard_data = await security_engine.get_security_dashboard_data("test_tenant")
		
		assert isinstance(dashboard_data, dict)
		assert 'threat_summary' in dashboard_data
		assert 'alert_counts' in dashboard_data
		assert 'risk_distribution' in dashboard_data
		assert 'behavioral_insights' in dashboard_data
		assert 'top_threats' in dashboard_data
		assert 'security_trends' in dashboard_data
		assert 'system_health' in dashboard_data
	
	@pytest.mark.asyncio
	async def test_investigate_security_incident(self, security_engine):
		"""Test security incident investigation"""
		incident_id = "incident_123"
		investigation_params = {
			'time_range': {
				'start': datetime.utcnow() - timedelta(hours=2),
				'end': datetime.utcnow()
			},
			'affected_keys': ['key_001'],
			'include_related_events': True
		}
		
		investigation = await security_engine.investigate_security_incident(
			incident_id,
			investigation_params,
			"admin@datacraft.co.ke"
		)
		
		assert investigation is not None
		assert 'incident_id' in investigation
		assert 'timeline' in investigation
		assert 'affected_resources' in investigation
		assert 'threat_analysis' in investigation
		assert 'forensic_data' in investigation
		assert 'recommendations' in investigation


class TestBehavioralAnalytics:
	"""Test BehavioralAnalytics class"""
	
	@pytest.fixture
	def behavioral_analytics(self):
		"""Fixture for behavioral analytics"""
		return BehavioralAnalytics()
	
	def test_build_user_profile(self, behavioral_analytics, sample_operations):
		"""Test user profile building"""
		user_operations = [op for op in sample_operations if op.user_id == "user_0@datacraft.co.ke"]
		
		profile = behavioral_analytics.build_user_profile("user_0@datacraft.co.ke", user_operations)
		
		assert isinstance(profile, BehavioralPattern)
		assert profile.pattern_type == "user_behavior"
		assert profile.entity_id == "user_0@datacraft.co.ke"
		assert 'typical_hours' in profile.pattern_data
		assert 'common_operations' in profile.pattern_data
		assert 'preferred_keys' in profile.pattern_data
		assert 'ip_ranges' in profile.pattern_data
	
	def test_analyze_access_patterns(self, behavioral_analytics, sample_operations):
		"""Test access pattern analysis"""
		patterns = behavioral_analytics.analyze_access_patterns(sample_operations)
		
		assert isinstance(patterns, dict)
		assert 'temporal_patterns' in patterns
		assert 'geographic_patterns' in patterns
		assert 'operational_patterns' in patterns
		assert 'anomaly_scores' in patterns
	
	def test_calculate_deviation_score(self, behavioral_analytics):
		"""Test deviation score calculation"""
		baseline_pattern = BehavioralPattern(
			pattern_type="user_behavior",
			entity_id="test_user",
			pattern_data={
				'typical_hours': [9, 10, 11, 14, 15, 16, 17],
				'common_operations': {'encrypt': 0.6, 'decrypt': 0.4},
				'ip_ranges': ['192.168.1.0/24']
			}
		)
		
		current_operation = KeyOperation(
			key_id="test_key",
			operation_type="encrypt",
			user_id="test_user",
			request_ip="10.0.0.1",  # Different IP range
			timestamp=datetime.utcnow().replace(hour=22)  # Unusual hour
		)
		
		deviation = behavioral_analytics.calculate_deviation_score(
			baseline_pattern, 
			current_operation
		)
		
		assert isinstance(deviation, float)
		assert 0.0 <= deviation <= 1.0
		assert deviation > 0.5  # Should be high due to IP and time anomalies


class TestThreatCorrelationEngine:
	"""Test ThreatCorrelationEngine class"""
	
	@pytest.fixture
	def correlation_engine(self):
		"""Fixture for threat correlation engine"""
		return ThreatCorrelationEngine()
	
	def test_correlate_indicators(self, correlation_engine):
		"""Test threat indicator correlation"""
		indicators = [
			ThreatIndicator(
				indicator_type="brute_force_attempt",
				severity="high",
				confidence=0.9,
				source="failed_auth_detector",
				metadata={'failed_attempts': 50, 'time_window': 300}
			),
			ThreatIndicator(
				indicator_type="suspicious_ip",
				severity="medium",
				confidence=0.8,
				source="ip_intelligence",
				metadata={'ip_reputation': 0.2, 'geographic_anomaly': True}
			)
		]
		
		correlations = correlation_engine.correlate_indicators(indicators)
		
		assert isinstance(correlations, list)
		for correlation in correlations:
			assert 'correlation_strength' in correlation
			assert 'attack_scenario' in correlation
			assert 'combined_threat_level' in correlation
			assert 'indicators' in correlation
	
	def test_identify_attack_patterns(self, correlation_engine):
		"""Test attack pattern identification"""
		security_events = [
			{
				'event_type': 'failed_authentication',
				'timestamp': datetime.utcnow() - timedelta(minutes=10),
				'source_ip': '192.168.1.100',
				'user_id': 'admin@company.com'
			},
			{
				'event_type': 'failed_authentication',
				'timestamp': datetime.utcnow() - timedelta(minutes=8),
				'source_ip': '192.168.1.100', 
				'user_id': 'admin@company.com'
			},
			{
				'event_type': 'successful_authentication',
				'timestamp': datetime.utcnow() - timedelta(minutes=5),
				'source_ip': '192.168.1.100',
				'user_id': 'admin@company.com'
			}
		]
		
		patterns = correlation_engine.identify_attack_patterns(security_events)
		
		assert isinstance(patterns, list)
		for pattern in patterns:
			assert 'pattern_type' in pattern
			assert 'confidence' in pattern
			assert 'timeline' in pattern
			assert 'indicators' in pattern


class TestSecurityModels:
	"""Test security data models"""
	
	def test_security_alert_creation(self):
		"""Test SecurityAlert model"""
		alert = SecurityAlert(
			tenant_id="test_tenant",
			alert_type="suspicious_activity",
			severity="medium",
			confidence=0.85,
			description="Unusual access pattern detected",
			affected_resources=["key_001", "key_002"],
			threat_indicators=['unusual_time', 'new_ip_address'],
			created_by="system",
			metadata={
				'detection_method': 'behavioral_analysis',
				'risk_score': 0.75
			}
		)
		
		assert alert.tenant_id == "test_tenant"
		assert alert.alert_type == "suspicious_activity"
		assert alert.severity == "medium"
		assert alert.confidence == 0.85
		assert len(alert.affected_resources) == 2
		assert alert.status == "new"  # Default status
		assert alert.metadata['risk_score'] == 0.75
	
	def test_behavioral_pattern_validation(self):
		"""Test BehavioralPattern validation"""
		# Valid pattern
		pattern = BehavioralPattern(
			pattern_type="user_behavior",
			entity_id="user_123",
			pattern_data={'common_hours': [9, 10, 11]},
			confidence_level=0.9
		)
		
		assert pattern.confidence_level == 0.9
		
		# Invalid confidence level
		with pytest.raises(ValueError):
			BehavioralPattern(
				pattern_type="user_behavior",
				entity_id="user_123",
				pattern_data={},
				confidence_level=1.5  # > 1.0
			)
	
	def test_threat_indicator_creation(self):
		"""Test ThreatIndicator model"""
		indicator = ThreatIndicator(
			indicator_type="anomalous_behavior",
			severity="high",
			confidence=0.92,
			source="ml_detector",
			description="User accessing keys outside normal hours",
			metadata={
				'user_id': 'test@datacraft.co.ke',
				'anomaly_score': 0.88,
				'detection_time': datetime.utcnow().isoformat()
			}
		)
		
		assert indicator.indicator_type == "anomalous_behavior"
		assert indicator.severity == "high"
		assert indicator.confidence == 0.92
		assert indicator.source == "ml_detector"
		assert 'user_id' in indicator.metadata
	
	def test_security_metrics_calculation(self):
		"""Test SecurityMetrics model"""
		metrics = SecurityMetrics(
			tenant_id="test_tenant",
			total_operations=10000,
			suspicious_operations=50,
			blocked_operations=5,
			alert_count=12,
			threat_level_distribution={
				'low': 8000,
				'medium': 1950,
				'high': 45,
				'critical': 5
			}
		)
		
		# Test calculated properties
		assert metrics.threat_detection_rate == 0.005  # 50/10000
		assert metrics.block_rate == 0.0005  # 5/10000
		assert metrics.high_risk_operations == 50  # high + critical


class TestIntegrationScenarios:
	"""Test integration scenarios"""
	
	@pytest.mark.asyncio
	async def test_complete_threat_detection_flow(self, security_engine):
		"""Test complete threat detection and response flow"""
		# 1. Create suspicious operations
		suspicious_operations = [
			KeyOperation(
				key_id="critical_key_001",
				operation_type="decrypt",
				user_id="attacker@evil.com",
				application_id="unknown_app",
				request_ip="10.0.0.1",
				timestamp=datetime.utcnow() - timedelta(minutes=i)
			)
			for i in range(20)  # Rapid fire attempts
		]
		
		# 2. Analyze operations
		threat_results = []
		for operation in suspicious_operations:
			analysis = await security_engine.analyze_key_operation(operation)
			threat_results.append(analysis)
		
		# Should detect high threat levels
		high_threat_count = sum(1 for r in threat_results if r['threat_level'] in ['high', 'critical'])
		assert high_threat_count > 0
		
		# 3. Generate security alert
		threat_data = {
			'threat_type': 'potential_data_breach',
			'severity': 'critical',
			'confidence': 0.95,
			'affected_keys': ['critical_key_001'],
			'source_ips': ['10.0.0.1']
		}
		
		alert = await security_engine.generate_security_alert(threat_data, "system")
		assert alert.severity == "critical"
		
		# 4. Investigate incident
		investigation = await security_engine.investigate_security_incident(
			alert.id,
			{'include_related_events': True},
			"admin@datacraft.co.ke"
		)
		assert investigation is not None
	
	@pytest.mark.asyncio
	async def test_behavioral_baseline_and_detection(self, security_engine):
		"""Test behavioral baseline building and anomaly detection"""
		# Create normal operation pattern
		normal_operations = []
		base_time = datetime.utcnow()
		
		for day in range(7):  # One week of data
			for hour in [9, 10, 11, 14, 15, 16, 17]:  # Business hours
				for i in range(10):  # 10 operations per hour
					operation = KeyOperation(
						key_id="business_key_001",
						operation_type="encrypt" if i % 2 == 0 else "decrypt",
						user_id="employee@company.com",
						application_id="business_app",
						request_ip="192.168.1.50",
						timestamp=base_time - timedelta(days=day, hours=(24-hour))
					)
					normal_operations.append(operation)
		
		# Build baseline
		await security_engine.build_behavioral_baseline(normal_operations)
		
		# Create anomalous operation
		anomalous_operation = KeyOperation(
			key_id="business_key_001",
			operation_type="decrypt",
			user_id="employee@company.com",
			application_id="business_app",
			request_ip="192.168.1.50",
			timestamp=datetime.utcnow().replace(hour=2)  # 2 AM - unusual
		)
		
		# Detect anomaly
		anomalies = await security_engine.detect_behavioral_anomalies(
			[anomalous_operation],
			"admin@datacraft.co.ke"
		)
		
		assert len(anomalies) > 0
		time_anomaly = next((a for a in anomalies if 'time' in a['anomaly_type']), None)
		assert time_anomaly is not None


if __name__ == "__main__":
	pytest.main([__file__])