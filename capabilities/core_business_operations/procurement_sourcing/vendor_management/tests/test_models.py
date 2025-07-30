"""
APG Vendor Management - Model Tests
Comprehensive tests for Pydantic models and data validation

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import pytest
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4
from pydantic import ValidationError

from ..models import (
	VMVendor, VMPerformance, VMRisk, VMIntelligence, VMContract,
	VMCommunication, VMBenchmark, VMPortalUser, VMAuditLog, VMCompliance,
	VendorStatus, VendorType, RiskSeverity, StrategicImportance,
	VendorSizeClassification, LifecycleStage
)


# ============================================================================
# VENDOR MODEL TESTS
# ============================================================================

@pytest.mark.unit
class TestVMVendor:
	"""Test VMVendor model validation and functionality"""
	
	def test_vendor_creation_valid_data(self, test_tenant_id):
		"""Test creating vendor with valid data"""
		
		vendor_data = {
			'tenant_id': test_tenant_id,
			'vendor_code': 'TEST001',
			'name': 'Test Vendor Inc.',
			'vendor_type': VendorType.SUPPLIER,
			'category': 'technology',
			'strategic_importance': StrategicImportance.STANDARD
		}
		
		vendor = VMVendor(**vendor_data)
		
		assert vendor.vendor_code == 'TEST001'
		assert vendor.name == 'Test Vendor Inc.'
		assert vendor.vendor_type == VendorType.SUPPLIER
		assert vendor.status == VendorStatus.ACTIVE  # default
		assert vendor.performance_score == Decimal('85.00')  # default
		assert isinstance(vendor.id, str)
		assert isinstance(vendor.created_at, datetime)
	
	def test_vendor_validation_required_fields(self, test_tenant_id):
		"""Test vendor validation with missing required fields"""
		
		with pytest.raises(ValidationError) as exc_info:
			VMVendor(tenant_id=test_tenant_id)
		
		errors = exc_info.value.errors()
		required_fields = {'vendor_code', 'name', 'category'}
		missing_fields = {error['loc'][0] for error in errors if error['type'] == 'missing'}
		
		assert required_fields.intersection(missing_fields)
	
	def test_vendor_score_validation(self, test_tenant_id):
		"""Test vendor score validation (0-100 range)"""
		
		base_data = {
			'tenant_id': test_tenant_id,
			'vendor_code': 'TEST001',
			'name': 'Test Vendor',
			'category': 'technology'
		}
		
		# Valid scores
		vendor = VMVendor(**base_data, performance_score=Decimal('95.50'))
		assert vendor.performance_score == Decimal('95.50')
		
		# Invalid scores - too low
		with pytest.raises(ValidationError):
			VMVendor(**base_data, performance_score=Decimal('-5.00'))
		
		# Invalid scores - too high
		with pytest.raises(ValidationError):
			VMVendor(**base_data, performance_score=Decimal('105.00'))
	
	def test_vendor_email_validation(self, test_tenant_id):
		"""Test vendor email validation"""
		
		base_data = {
			'tenant_id': test_tenant_id,
			'vendor_code': 'TEST001',
			'name': 'Test Vendor',
			'category': 'technology'
		}
		
		# Valid email
		vendor = VMVendor(**base_data, email='valid@example.com')
		assert vendor.email == 'valid@example.com'
		
		# Invalid email
		with pytest.raises(ValidationError):
			VMVendor(**base_data, email='invalid-email')
	
	def test_vendor_enum_validation(self, test_tenant_id):
		"""Test vendor enum field validation"""
		
		base_data = {
			'tenant_id': test_tenant_id,
			'vendor_code': 'TEST001',
			'name': 'Test Vendor',
			'category': 'technology'
		}
		
		# Valid enums
		vendor = VMVendor(
			**base_data,
			vendor_type=VendorType.STRATEGIC_PARTNER,
			status=VendorStatus.ACTIVE,
			strategic_importance=StrategicImportance.CRITICAL
		)
		
		assert vendor.vendor_type == VendorType.STRATEGIC_PARTNER
		assert vendor.status == VendorStatus.ACTIVE
		assert vendor.strategic_importance == StrategicImportance.CRITICAL
	
	@pytest.mark.ai
	def test_vendor_ai_fields(self, test_tenant_id):
		"""Test vendor AI-related fields"""
		
		ai_insights = {
			'patterns': ['responsive_communication', 'consistent_delivery'],
			'predictions': {
				'next_quarter_performance': 88.5,
				'risk_probability': 0.15
			}
		}
		
		vendor = VMVendor(
			tenant_id=test_tenant_id,
			vendor_code='AI001',
			name='AI Test Vendor',
			category='technology',
			ai_insights=ai_insights,
			predicted_performance={'q1_2025': 87.5},
			optimization_recommendations=['increase_order_frequency']
		)
		
		assert vendor.ai_insights == ai_insights
		assert vendor.predicted_performance == {'q1_2025': 87.5}
		assert 'increase_order_frequency' in vendor.optimization_recommendations


# ============================================================================
# PERFORMANCE MODEL TESTS
# ============================================================================

@pytest.mark.unit
class TestVMPerformance:
	"""Test VMPerformance model validation and functionality"""
	
	def test_performance_creation_valid_data(self, test_tenant_id):
		"""Test creating performance record with valid data"""
		
		performance_data = {
			'tenant_id': test_tenant_id,
			'vendor_id': str(uuid4()),
			'measurement_period': 'quarterly',
			'overall_score': Decimal('85.50'),
			'quality_score': Decimal('90.00'),
			'delivery_score': Decimal('82.00'),
			'cost_score': Decimal('88.00'),
			'service_score': Decimal('85.00')
		}
		
		performance = VMPerformance(**performance_data)
		
		assert performance.measurement_period == 'quarterly'
		assert performance.overall_score == Decimal('85.50')
		assert performance.quality_score == Decimal('90.00')
		assert isinstance(performance.start_date, datetime)
		assert isinstance(performance.end_date, datetime)
	
	def test_performance_score_validation(self, test_tenant_id):
		"""Test performance score validation"""
		
		base_data = {
			'tenant_id': test_tenant_id,
			'vendor_id': str(uuid4()),
			'measurement_period': 'monthly'
		}
		
		# Valid scores
		performance = VMPerformance(
			**base_data,
			overall_score=Decimal('95.75'),
			quality_score=Decimal('0.00'),  # minimum
			delivery_score=Decimal('100.00')  # maximum
		)
		
		assert performance.overall_score == Decimal('95.75')
		
		# Invalid scores
		with pytest.raises(ValidationError):
			VMPerformance(**base_data, overall_score=Decimal('-1.00'))
		
		with pytest.raises(ValidationError):
			VMPerformance(**base_data, quality_score=Decimal('101.00'))
	
	def test_performance_metrics_validation(self, test_tenant_id):
		"""Test performance metrics validation"""
		
		performance = VMPerformance(
			tenant_id=test_tenant_id,
			vendor_id=str(uuid4()),
			measurement_period='quarterly',
			overall_score=Decimal('85.00'),
			order_volume=Decimal('1000000.50'),
			order_count=50,
			total_spend=Decimal('950000.75'),
			on_time_delivery_rate=Decimal('95.25')
		)
		
		assert performance.order_volume == Decimal('1000000.50')
		assert performance.order_count == 50
		assert performance.on_time_delivery_rate == Decimal('95.25')


# ============================================================================
# RISK MODEL TESTS
# ============================================================================

@pytest.mark.unit
class TestVMRisk:
	"""Test VMRisk model validation and functionality"""
	
	def test_risk_creation_valid_data(self, test_tenant_id):
		"""Test creating risk record with valid data"""
		
		risk_data = {
			'tenant_id': test_tenant_id,
			'vendor_id': str(uuid4()),
			'risk_type': 'operational',
			'risk_category': 'delivery',
			'severity': RiskSeverity.MEDIUM,
			'title': 'Delivery Risk',
			'description': 'Risk of delivery delays',
			'overall_risk_score': Decimal('65.00')
		}
		
		risk = VMRisk(**risk_data)
		
		assert risk.risk_type == 'operational'
		assert risk.severity == RiskSeverity.MEDIUM
		assert risk.overall_risk_score == Decimal('65.00')
		assert isinstance(risk.identified_date, datetime)
	
	def test_risk_severity_validation(self, test_tenant_id):
		"""Test risk severity enum validation"""
		
		base_data = {
			'tenant_id': test_tenant_id,
			'vendor_id': str(uuid4()),
			'risk_type': 'financial',
			'risk_category': 'credit',
			'title': 'Credit Risk',
			'description': 'Risk description',
			'overall_risk_score': Decimal('45.00')
		}
		
		# Valid severities
		for severity in RiskSeverity:
			risk = VMRisk(**base_data, severity=severity)
			assert risk.severity == severity
	
	def test_risk_impact_validation(self, test_tenant_id):
		"""Test risk impact score validation"""
		
		risk = VMRisk(
			tenant_id=test_tenant_id,
			vendor_id=str(uuid4()),
			risk_type='operational',
			risk_category='delivery',
			severity=RiskSeverity.HIGH,
			title='High Risk',
			description='High risk description',
			overall_risk_score=Decimal('85.00'),
			operational_impact=8,
			reputational_impact=6,
			financial_impact=Decimal('100000.00')
		)
		
		assert risk.operational_impact == 8
		assert risk.reputational_impact == 6
		assert risk.financial_impact == Decimal('100000.00')
		
		# Invalid impact scores
		with pytest.raises(ValidationError):
			VMRisk(
				tenant_id=test_tenant_id,
				vendor_id=str(uuid4()),
				risk_type='test',
				risk_category='test',
				severity=RiskSeverity.LOW,
				title='Test',
				description='Test',
				overall_risk_score=Decimal('30.00'),
				operational_impact=11  # > 10
			)


# ============================================================================
# INTELLIGENCE MODEL TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.ai
class TestVMIntelligence:
	"""Test VMIntelligence model validation and functionality"""
	
	def test_intelligence_creation_valid_data(self, test_tenant_id):
		"""Test creating intelligence record with valid data"""
		
		behavior_patterns = [
			{
				'pattern_type': 'communication',
				'pattern_name': 'responsive',
				'confidence': 0.85,
				'description': 'Quick response to communications'
			}
		]
		
		predictive_insights = [
			{
				'insight_type': 'performance',
				'prediction': 'improvement',
				'confidence': 0.75,
				'time_horizon': 90
			}
		]
		
		intelligence = VMIntelligence(
			tenant_id=test_tenant_id,
			vendor_id=str(uuid4()),
			model_version='v1.0',
			confidence_score=Decimal('0.82'),
			behavior_patterns=behavior_patterns,
			predictive_insights=predictive_insights
		)
		
		assert intelligence.model_version == 'v1.0'
		assert intelligence.confidence_score == Decimal('0.82')
		assert len(intelligence.behavior_patterns) == 1
		assert len(intelligence.predictive_insights) == 1
	
	def test_intelligence_confidence_validation(self, test_tenant_id):
		"""Test intelligence confidence score validation"""
		
		base_data = {
			'tenant_id': test_tenant_id,
			'vendor_id': str(uuid4()),
			'model_version': 'v1.0',
			'behavior_patterns': [],
			'predictive_insights': []
		}
		
		# Valid confidence scores
		intelligence = VMIntelligence(**base_data, confidence_score=Decimal('0.95'))
		assert intelligence.confidence_score == Decimal('0.95')
		
		# Invalid confidence scores
		with pytest.raises(ValidationError):
			VMIntelligence(**base_data, confidence_score=Decimal('1.5'))
		
		with pytest.raises(ValidationError):
			VMIntelligence(**base_data, confidence_score=Decimal('-0.1'))
	
	@pytest.mark.ai
	def test_intelligence_json_fields(self, test_tenant_id):
		"""Test intelligence JSON field validation"""
		
		performance_forecasts = {
			'next_quarter': {
				'overall_score': 88.5,
				'confidence': 0.78
			},
			'next_year': {
				'overall_score': 90.0,
				'confidence': 0.65
			}
		}
		
		risk_assessments = {
			'delivery_risk': {
				'probability': 0.25,
				'impact': 'medium'
			}
		}
		
		intelligence = VMIntelligence(
			tenant_id=test_tenant_id,
			vendor_id=str(uuid4()),
			model_version='v1.0',
			confidence_score=Decimal('0.80'),
			behavior_patterns=[],
			predictive_insights=[],
			performance_forecasts=performance_forecasts,
			risk_assessments=risk_assessments
		)
		
		assert intelligence.performance_forecasts == performance_forecasts
		assert intelligence.risk_assessments == risk_assessments


# ============================================================================
# VALIDATION EDGE CASES
# ============================================================================

@pytest.mark.unit
class TestModelValidationEdgeCases:
	"""Test model validation edge cases and boundary conditions"""
	
	def test_empty_string_validation(self, test_tenant_id):
		"""Test validation with empty strings"""
		
		with pytest.raises(ValidationError):
			VMVendor(
				tenant_id=test_tenant_id,
				vendor_code='',  # empty string
				name='Test Vendor',
				category='technology'
			)
	
	def test_none_vs_missing_fields(self, test_tenant_id):
		"""Test difference between None and missing optional fields"""
		
		# None values for optional fields should be accepted
		vendor = VMVendor(
			tenant_id=test_tenant_id,
			vendor_code='TEST001',
			name='Test Vendor',
			category='technology',
			legal_name=None,
			email=None,
			phone=None
		)
		
		assert vendor.legal_name is None
		assert vendor.email is None
		assert vendor.phone is None
	
	def test_uuid_validation(self, test_tenant_id):
		"""Test UUID field validation"""
		
		valid_uuid = str(uuid4())
		
		vendor = VMVendor(
			tenant_id=test_tenant_id,
			vendor_id=valid_uuid,
			vendor_code='UUID001',
			name='UUID Test Vendor',
			category='technology'
		)
		
		# Test invalid UUID strings
		with pytest.raises(ValidationError):
			VMVendor(
				tenant_id='invalid-uuid',
				vendor_code='INVALID001',
				name='Invalid UUID Vendor',
				category='technology'
			)
	
	@pytest.mark.performance
	def test_large_json_fields(self, test_tenant_id):
		"""Test performance with large JSON fields"""
		
		# Create large JSON data
		large_capabilities = [f"capability_{i}" for i in range(1000)]
		large_ai_insights = {
			f"insight_{i}": f"value_{i}" for i in range(500)
		}
		
		vendor = VMVendor(
			tenant_id=test_tenant_id,
			vendor_code='LARGE001',
			name='Large Data Vendor',
			category='technology',
			capabilities=large_capabilities,
			ai_insights=large_ai_insights
		)
		
		assert len(vendor.capabilities) == 1000
		assert len(vendor.ai_insights) == 500


# ============================================================================
# MODEL SERIALIZATION TESTS
# ============================================================================

@pytest.mark.unit
class TestModelSerialization:
	"""Test model serialization and deserialization"""
	
	def test_vendor_model_dump(self, sample_vendor):
		"""Test vendor model serialization"""
		
		vendor_dict = sample_vendor.model_dump()
		
		assert isinstance(vendor_dict, dict)
		assert 'id' in vendor_dict
		assert 'vendor_code' in vendor_dict
		assert 'created_at' in vendor_dict
		
		# Test that datetime is serialized properly
		assert isinstance(vendor_dict['created_at'], datetime)
	
	def test_vendor_json_serialization(self, sample_vendor):
		"""Test vendor JSON serialization"""
		
		json_str = sample_vendor.model_dump_json()
		
		assert isinstance(json_str, str)
		assert 'TEST001' in json_str
		assert 'Test Vendor Inc.' in json_str
	
	def test_model_reconstruction(self, sample_vendor):
		"""Test model reconstruction from dict"""
		
		vendor_dict = sample_vendor.model_dump()
		reconstructed_vendor = VMVendor(**vendor_dict)
		
		assert reconstructed_vendor.id == sample_vendor.id
		assert reconstructed_vendor.vendor_code == sample_vendor.vendor_code
		assert reconstructed_vendor.name == sample_vendor.name