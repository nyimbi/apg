#!/usr/bin/env python3
"""
APG Key Management - AI Lifecycle Tests
Comprehensive test suite for AI-powered key lifecycle management

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from ..ai_lifecycle import (
	AutonomousKeyLifecycleManager, PredictiveKeyAnalytics, 
	LifecycleEvent, LifecyclePolicy, LifecycleRecommendation,
	create_lifecycle_manager
)
from ..models import KeyAlgorithm, KeyUsage, KeyState, Key, create_key_spec_async


@pytest.fixture
async def lifecycle_manager():
	"""Fixture for lifecycle manager"""
	manager = AutonomousKeyLifecycleManager()
	await manager.initialize({
		'tenant_id': 'test_tenant',
		'ml_model_path': '/tmp/test_models',
		'test_mode': True
	})
	return manager


@pytest.fixture
async def sample_key():
	"""Fixture for sample key"""
	spec = await create_key_spec_async(
		tenant_id="test_tenant",
		algorithm=KeyAlgorithm.AES_256,
		usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
		name="Test Key",
		created_by="test@datacraft.co.ke"
	)
	
	key = Key(
		spec=spec,
		key_material=b"test_key_material_32_bytes_long",
		key_checksum="abcd1234",
		usage_count=100
	)
	return key


class TestAutonomousKeyLifecycleManager:
	"""Test AutonomousKeyLifecycleManager class"""
	
	@pytest.mark.asyncio
	async def test_manager_initialization(self):
		"""Test lifecycle manager initialization"""
		manager = AutonomousKeyLifecycleManager()
		assert not manager.is_initialized
		
		config = {
			'tenant_id': 'test_tenant',
			'ml_model_path': '/tmp/models',
			'prediction_interval_hours': 24
		}
		await manager.initialize(config)
		
		assert manager.is_initialized
		assert manager.config == config
		assert isinstance(manager.lifecycle_events, list)
		assert isinstance(manager.policies, dict)
		assert isinstance(manager.recommendations, list)
	
	@pytest.mark.asyncio
	async def test_factory_function(self):
		"""Test lifecycle manager factory function"""
		manager = await create_lifecycle_manager()
		assert isinstance(manager, AutonomousKeyLifecycleManager)
		assert manager.is_initialized
	
	@pytest.mark.asyncio
	async def test_analyze_key_lifecycle(self, lifecycle_manager, sample_key):
		"""Test key lifecycle analysis"""
		analysis = await lifecycle_manager.analyze_key_lifecycle(
			sample_key, 
			"test@datacraft.co.ke"
		)
		
		assert analysis is not None
		assert 'health_score' in analysis
		assert 'risk_factors' in analysis
		assert 'lifecycle_stage' in analysis
		assert 'recommendations' in analysis
		assert 'predicted_events' in analysis
		
		# Validate health score range
		assert 0.0 <= analysis['health_score'] <= 1.0
		
		# Validate lifecycle stage
		assert analysis['lifecycle_stage'] in [
			'creation', 'active', 'rotation_due', 'deprecation', 'retirement'
		]
	
	@pytest.mark.asyncio
	async def test_predict_lifecycle_events(self, lifecycle_manager, sample_key):
		"""Test lifecycle event prediction"""
		predictions = await lifecycle_manager.predict_lifecycle_events(
			sample_key, 
			days_ahead=30
		)
		
		assert isinstance(predictions, list)
		
		for prediction in predictions:
			assert 'event_type' in prediction
			assert 'predicted_date' in prediction
			assert 'confidence' in prediction
			assert 'impact_level' in prediction
			assert 'recommended_actions' in prediction
			
			# Validate confidence
			assert 0.0 <= prediction['confidence'] <= 1.0
			
			# Validate predicted date is in future
			predicted_date = datetime.fromisoformat(prediction['predicted_date'])
			assert predicted_date > datetime.utcnow()
	
	@pytest.mark.asyncio
	async def test_generate_recommendations(self, lifecycle_manager, sample_key):
		"""Test lifecycle recommendations generation"""
		recommendations = await lifecycle_manager.generate_recommendations(
			sample_key, 
			"test@datacraft.co.ke"
		)
		
		assert isinstance(recommendations, list)
		
		for rec in recommendations:
			assert isinstance(rec, LifecycleRecommendation)
			assert rec.key_id == sample_key.spec.id
			assert rec.tenant_id == "test_tenant"
			assert rec.recommendation_type in [
				'rotate_key', 'update_policy', 'increase_security', 
				'optimize_usage', 'schedule_maintenance'
			]
			assert 0.0 <= rec.priority_score <= 1.0
			assert rec.rationale is not None
			assert isinstance(rec.actions, list)
	
	@pytest.mark.asyncio
	async def test_execute_autonomous_actions(self, lifecycle_manager, sample_key):
		"""Test autonomous action execution"""
		# Mock key service for testing
		mock_key_service = AsyncMock()
		lifecycle_manager.key_service = mock_key_service
		
		actions = [
			{
				'action_type': 'rotate_key',
				'key_id': sample_key.spec.id,
				'parameters': {'reason': 'scheduled_rotation'}
			}
		]
		
		results = await lifecycle_manager.execute_autonomous_actions(
			actions, 
			"test@datacraft.co.ke"
		)
		
		assert isinstance(results, list)
		assert len(results) == len(actions)
		
		for result in results:
			assert 'action_type' in result
			assert 'success' in result
			assert 'result' in result or 'error' in result
	
	@pytest.mark.asyncio
	async def test_schedule_lifecycle_maintenance(self, lifecycle_manager):
		"""Test lifecycle maintenance scheduling"""
		schedule_config = {
			'check_interval_hours': 6,
			'maintenance_window': '02:00-04:00',
			'timezone': 'UTC',
			'enabled_actions': ['rotate_expired_keys', 'update_policies']
		}
		
		result = await lifecycle_manager.schedule_lifecycle_maintenance(
			schedule_config, 
			"admin@datacraft.co.ke"
		)
		
		assert result is not None
		assert result['scheduled'] is True
		assert 'next_run' in result
		assert 'maintenance_id' in result
	
	@pytest.mark.asyncio
	async def test_get_lifecycle_metrics(self, lifecycle_manager):
		"""Test lifecycle metrics retrieval"""
		metrics = await lifecycle_manager.get_lifecycle_metrics("test_tenant")
		
		assert isinstance(metrics, dict)
		assert 'total_keys' in metrics
		assert 'keys_by_stage' in metrics
		assert 'rotation_rate' in metrics
		assert 'health_score_distribution' in metrics
		assert 'prediction_accuracy' in metrics
		assert 'autonomous_actions_taken' in metrics
		assert 'recommendations_generated' in metrics
	
	@pytest.mark.asyncio
	async def test_update_lifecycle_policy(self, lifecycle_manager):
		"""Test lifecycle policy updates"""
		policy = LifecyclePolicy(
			tenant_id="test_tenant",
			policy_name="test_policy",
			rotation_triggers=['usage_threshold', 'time_based'],
			rotation_threshold_days=90,
			usage_threshold_count=10000,
			auto_rotation_enabled=True,
			risk_threshold=0.7,
			compliance_requirements=['GDPR', 'HIPAA']
		)
		
		result = await lifecycle_manager.update_lifecycle_policy(
			policy, 
			"admin@datacraft.co.ke"
		)
		
		assert result is True
		assert "test_policy" in lifecycle_manager.policies
		assert lifecycle_manager.policies["test_policy"] == policy
	
	@pytest.mark.asyncio
	async def test_key_health_assessment(self, lifecycle_manager, sample_key):
		"""Test key health assessment"""
		# Set key to have some usage history
		sample_key.usage_count = 5000
		sample_key.last_used = datetime.utcnow() - timedelta(days=5)
		
		health = await lifecycle_manager._assess_key_health(sample_key)
		
		assert isinstance(health, dict)
		assert 'overall_score' in health
		assert 'security_score' in health
		assert 'performance_score' in health
		assert 'compliance_score' in health
		assert 'risk_factors' in health
		
		# Validate score ranges
		for score_key in ['overall_score', 'security_score', 'performance_score', 'compliance_score']:
			assert 0.0 <= health[score_key] <= 1.0
	
	@pytest.mark.asyncio
	async def test_ml_prediction_engine(self, lifecycle_manager, sample_key):
		"""Test ML-based prediction engine"""
		historical_data = {
			'usage_patterns': [100, 150, 200, 180, 220],
			'error_rates': [0.01, 0.02, 0.015, 0.018, 0.012],
			'security_events': [0, 1, 0, 0, 2],
			'performance_metrics': [25.5, 26.1, 24.8, 25.2, 26.8]
		}
		
		predictions = await lifecycle_manager._run_ml_predictions(
			sample_key, 
			historical_data, 
			30
		)
		
		assert isinstance(predictions, dict)
		assert 'rotation_probability' in predictions
		assert 'failure_probability' in predictions
		assert 'security_risk_level' in predictions
		assert 'performance_degradation' in predictions
		
		# Validate probability ranges
		for prob_key in ['rotation_probability', 'failure_probability']:
			assert 0.0 <= predictions[prob_key] <= 1.0
	
	@pytest.mark.asyncio
	async def test_error_handling_invalid_key(self, lifecycle_manager):
		"""Test error handling with invalid key"""
		with pytest.raises(AssertionError, match="Key required"):
			await lifecycle_manager.analyze_key_lifecycle(None, "test@datacraft.co.ke")
	
	@pytest.mark.asyncio
	async def test_error_handling_not_initialized(self, sample_key):
		"""Test error handling when not initialized"""
		manager = AutonomousKeyLifecycleManager()
		
		with pytest.raises(AssertionError, match="Manager not initialized"):
			await manager.analyze_key_lifecycle(sample_key, "test@datacraft.co.ke")


class TestPredictiveKeyAnalytics:
	"""Test PredictiveKeyAnalytics class"""
	
	@pytest.fixture
	def analytics_engine(self):
		"""Fixture for analytics engine"""
		return PredictiveKeyAnalytics()
	
	def test_usage_pattern_analysis(self, analytics_engine):
		"""Test usage pattern analysis"""
		usage_data = [100, 120, 110, 150, 140, 160, 180, 200, 190, 220]
		
		patterns = analytics_engine.analyze_usage_patterns(usage_data)
		
		assert isinstance(patterns, dict)
		assert 'trend' in patterns
		assert 'seasonality' in patterns
		assert 'anomalies' in patterns
		assert 'predicted_next' in patterns
		
		assert patterns['trend'] in ['increasing', 'decreasing', 'stable']
	
	def test_security_risk_assessment(self, analytics_engine):
		"""Test security risk assessment"""
		security_events = [
			{'type': 'failed_access', 'count': 5},
			{'type': 'unusual_location', 'count': 2},
			{'type': 'suspicious_pattern', 'count': 1}
		]
		
		risk_score = analytics_engine.calculate_security_risk(security_events)
		
		assert isinstance(risk_score, float)
		assert 0.0 <= risk_score <= 1.0
	
	def test_performance_prediction(self, analytics_engine):
		"""Test performance prediction"""
		performance_history = [25.0, 26.5, 24.8, 27.2, 25.9, 28.1, 26.7]
		
		prediction = analytics_engine.predict_performance(performance_history, days_ahead=7)
		
		assert isinstance(prediction, dict)
		assert 'predicted_latency' in prediction
		assert 'confidence_interval' in prediction
		assert 'degradation_probability' in prediction


class TestLifecycleModels:
	"""Test lifecycle data models"""
	
	def test_lifecycle_event_creation(self):
		"""Test LifecycleEvent model"""
		event = LifecycleEvent(
			tenant_id="test_tenant",
			key_id="key_123",
			event_type="key_rotated",
			event_data={'reason': 'scheduled', 'old_version': 1},
			user_id="admin@datacraft.co.ke"
		)
		
		assert event.tenant_id == "test_tenant"
		assert event.key_id == "key_123"
		assert event.event_type == "key_rotated"
		assert event.event_data['reason'] == "scheduled"
		assert event.user_id == "admin@datacraft.co.ke"
		assert event.timestamp is not None
	
	def test_lifecycle_policy_validation(self):
		"""Test LifecyclePolicy validation"""
		# Valid policy
		policy = LifecyclePolicy(
			tenant_id="test_tenant",
			policy_name="valid_policy",
			rotation_threshold_days=90,
			usage_threshold_count=10000,
			risk_threshold=0.8
		)
		
		assert policy.rotation_threshold_days == 90
		assert policy.usage_threshold_count == 10000
		assert policy.risk_threshold == 0.8
		
		# Invalid risk threshold
		with pytest.raises(ValueError):
			LifecyclePolicy(
				tenant_id="test_tenant",
				policy_name="invalid_policy",
				risk_threshold=1.5  # > 1.0
			)
	
	def test_lifecycle_recommendation_creation(self):
		"""Test LifecycleRecommendation model"""
		recommendation = LifecycleRecommendation(
			tenant_id="test_tenant",
			key_id="key_123",
			recommendation_type="rotate_key",
			priority_score=0.85,
			rationale="Key has exceeded usage threshold",
			actions=["schedule_rotation", "notify_admin"],
			estimated_impact="medium"
		)
		
		assert recommendation.tenant_id == "test_tenant"
		assert recommendation.key_id == "key_123"
		assert recommendation.recommendation_type == "rotate_key"
		assert recommendation.priority_score == 0.85
		assert len(recommendation.actions) == 2
		assert recommendation.estimated_impact == "medium"


class TestIntegrationScenarios:
	"""Test integration scenarios"""
	
	@pytest.mark.asyncio
	async def test_full_lifecycle_automation(self, lifecycle_manager):
		"""Test complete lifecycle automation flow"""
		# Create test key
		spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.AES_256,
			usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
			name="Automation Test Key",
			created_by="test@datacraft.co.ke"
		)
		
		key = Key(
			spec=spec,
			key_material=b"test_key_material_32_bytes_long",
			key_checksum="abcd1234",
			usage_count=9500  # Near threshold
		)
		
		# Mock key service
		mock_key_service = AsyncMock()
		lifecycle_manager.key_service = mock_key_service
		
		# 1. Analyze lifecycle
		analysis = await lifecycle_manager.analyze_key_lifecycle(key, "test@datacraft.co.ke")
		assert analysis['health_score'] < 0.8  # Should indicate issues
		
		# 2. Generate recommendations
		recommendations = await lifecycle_manager.generate_recommendations(key, "test@datacraft.co.ke")
		assert len(recommendations) > 0
		
		# 3. Execute autonomous actions
		actions = [
			{
				'action_type': 'rotate_key',
				'key_id': key.spec.id,
				'parameters': {'reason': 'usage_threshold'}
			}
		]
		results = await lifecycle_manager.execute_autonomous_actions(actions, "test@datacraft.co.ke")
		assert len(results) == 1
	
	@pytest.mark.asyncio
	async def test_policy_driven_lifecycle(self, lifecycle_manager):
		"""Test policy-driven lifecycle management"""
		# Create strict policy
		policy = LifecyclePolicy(
			tenant_id="test_tenant",
			policy_name="strict_policy",
			rotation_threshold_days=30,  # Aggressive rotation
			usage_threshold_count=5000,
			auto_rotation_enabled=True,
			risk_threshold=0.5  # Low risk tolerance
		)
		
		# Update policy
		await lifecycle_manager.update_lifecycle_policy(policy, "admin@datacraft.co.ke")
		
		# Test key that exceeds policy
		spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.AES_256,
			usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
			name="Policy Test Key",
			created_by="test@datacraft.co.ke"
		)
		
		key = Key(
			spec=spec,
			key_material=b"test_key_material_32_bytes_long",
			key_checksum="abcd1234",
			usage_count=6000  # Exceeds policy threshold
		)
		key.spec.created_at = datetime.utcnow() - timedelta(days=35)  # Exceeds time threshold
		
		# Analyze with policy
		analysis = await lifecycle_manager.analyze_key_lifecycle(key, "test@datacraft.co.ke")
		
		# Should recommend rotation due to policy violations
		recommendations = await lifecycle_manager.generate_recommendations(key, "test@datacraft.co.ke")
		rotation_recs = [r for r in recommendations if r.recommendation_type == 'rotate_key']
		assert len(rotation_recs) > 0


if __name__ == "__main__":
	pytest.main([__file__])