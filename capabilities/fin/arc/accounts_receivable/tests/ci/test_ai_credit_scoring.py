"""
APG Accounts Receivable - AI Credit Scoring Tests
Unit tests for AI-powered credit scoring and risk assessment

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List

from uuid_extensions import uuid7str

from ..ai_credit_scoring import (
	CreditScoringFeatures, CreditScoringResult, CreditScoringConfig,
	APGCreditScoringService, CreditScoringModelTrainer,
	create_credit_scoring_service
)
from ..models import (
	ARCustomer, ARCreditAssessment, ARInvoice, ARPayment,
	ARCreditRating, ARCustomerStatus, ARCustomerType, ARInvoiceStatus, ARPaymentStatus
)


class TestCreditScoringFeatures:
	"""Test credit scoring feature extraction and validation."""
	
	def test_credit_scoring_features_creation(self):
		"""Test creating credit scoring features with valid data."""
		features = CreditScoringFeatures(
			customer_age_months=24,
			customer_type="corporation",
			total_invoices=50,
			paid_invoices=47,
			late_payments=3,
			disputed_invoices=1,
			avg_payment_days=28.5,
			current_outstanding=Decimal('15000.00'),
			average_invoice_amount=Decimal('2500.00'),
			credit_utilization=0.75,
			payment_consistency_score=0.85,
			dispute_resolution_rate=0.95,
			communication_responsiveness=0.8
		)
		
		assert features.customer_age_months == 24
		assert features.customer_type == "corporation"
		assert features.total_invoices == 50
		assert features.paid_invoices == 47
		assert features.current_outstanding == Decimal('15000.00')
		assert features.payment_consistency_score == 0.85
	
	def test_credit_scoring_features_validation(self):
		"""Test credit scoring features validation rules."""
		
		# Test paid invoices cannot exceed total invoices
		with pytest.raises(ValueError, match="Paid invoices cannot exceed total invoices"):
			CreditScoringFeatures(
				customer_age_months=12,
				customer_type="individual",
				total_invoices=10,
				paid_invoices=15  # Invalid: more than total
			)
		
		# Test negative values validation
		with pytest.raises(ValueError):
			CreditScoringFeatures(
				customer_age_months=-5,  # Invalid: negative
				customer_type="individual"
			)
	
	def test_credit_scoring_features_defaults(self):
		"""Test default values for optional fields."""
		features = CreditScoringFeatures(
			customer_age_months=12,
			customer_type="individual"
		)
		
		assert features.total_invoices == 0
		assert features.paid_invoices == 0
		assert features.late_payments == 0
		assert features.current_outstanding == Decimal('0.00')
		assert features.payment_consistency_score == 0.0
		assert features.dispute_resolution_rate == 1.0


class TestCreditScoringResult:
	"""Test credit scoring result model and validation."""
	
	def test_credit_scoring_result_creation(self):
		"""Test creating credit scoring result with valid data."""
		result = CreditScoringResult(
			customer_id=uuid7str(),
			model_version="2.1.0",
			credit_score=720,
			risk_rating=ARCreditRating.AA,
			default_probability=0.12,
			confidence_score=0.89,
			recommended_credit_limit=Decimal('50000.00'),
			payment_terms_days=30,
			next_review_date=date.today() + timedelta(days=180)
		)
		
		assert 300 <= result.credit_score <= 850
		assert result.risk_rating == ARCreditRating.AA
		assert 0 <= result.default_probability <= 1
		assert 0 <= result.confidence_score <= 1
		assert result.recommended_credit_limit > 0
		assert isinstance(result.next_review_date, date)
	
	def test_credit_score_range_validation(self):
		"""Test credit score must be within valid range."""
		
		# Test credit score too low
		with pytest.raises(ValueError):
			CreditScoringResult(
				customer_id=uuid7str(),
				model_version="2.1.0",
				credit_score=250,  # Invalid: below 300
				risk_rating=ARCreditRating.D,
				default_probability=0.5,
				confidence_score=0.8,
				recommended_credit_limit=Decimal('1000.00'),
				payment_terms_days=0,
				next_review_date=date.today() + timedelta(days=30)
			)
		
		# Test credit score too high
		with pytest.raises(ValueError):
			CreditScoringResult(
				customer_id=uuid7str(),
				model_version="2.1.0",
				credit_score=900,  # Invalid: above 850
				risk_rating=ARCreditRating.AAA,
				default_probability=0.05,
				confidence_score=0.95,
				recommended_credit_limit=Decimal('100000.00'),
				payment_terms_days=45,
				next_review_date=date.today() + timedelta(days=365)
			)


class TestCreditScoringConfig:
	"""Test credit scoring configuration."""
	
	def test_config_creation_with_defaults(self):
		"""Test creating config with default values."""
		config = CreditScoringConfig(
			federated_learning_endpoint="https://fl.apg.company.com/v1"
		)
		
		assert config.federated_learning_endpoint == "https://fl.apg.company.com/v1"
		assert config.model_name == "ar_credit_scoring_v2"
		assert config.model_version == "2.1.0"
		assert config.min_confidence_threshold == 0.85
		assert config.default_credit_limit == Decimal('10000.00')
	
	def test_config_custom_values(self):
		"""Test creating config with custom values."""
		config = CreditScoringConfig(
			federated_learning_endpoint="https://test.fl.com/v1",
			model_name="test_model",
			model_version="1.0.0",
			min_confidence_threshold=0.9,
			default_credit_limit=Decimal('5000.00')
		)
		
		assert config.federated_learning_endpoint == "https://test.fl.com/v1"
		assert config.model_name == "test_model"
		assert config.model_version == "1.0.0"
		assert config.min_confidence_threshold == 0.9
		assert config.default_credit_limit == Decimal('5000.00')


class TestAPGCreditScoringService:
	"""Test APG credit scoring service functionality."""
	
	@pytest.fixture
	def sample_customer(self):
		"""Create sample customer for testing."""
		return ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="TEST001",
			legal_name="Test Corporation",
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('20000.00'),
			payment_terms_days=30,
			total_outstanding=Decimal('8500.00'),
			overdue_amount=Decimal('1200.00'),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
	
	@pytest.fixture
	def sample_invoices(self, sample_customer):
		"""Create sample invoices for testing."""
		invoices = []
		for i in range(10):
			invoice = ARInvoice(
				id=uuid7str(),
				tenant_id=sample_customer.tenant_id,
				customer_id=sample_customer.id,
				invoice_number=f"INV-2025-{i+1:03d}",
				invoice_date=date.today() - timedelta(days=30*(i+1)),
				due_date=date.today() - timedelta(days=30*(i+1)-30),
				total_amount=Decimal('2500.00'),
				paid_amount=Decimal('2500.00') if i < 8 else Decimal('1300.00'),
				currency_code="USD",
				status=ARInvoiceStatus.PAID if i < 8 else ARInvoiceStatus.OVERDUE,
				payment_status="paid" if i < 8 else "overdue",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
			invoices.append(invoice)
		return invoices
	
	@pytest.fixture
	def sample_payments(self, sample_customer):
		"""Create sample payments for testing."""
		payments = []
		for i in range(8):  # 8 payments for 8 paid invoices
			payment = ARPayment(
				id=uuid7str(),
				tenant_id=sample_customer.tenant_id,
				customer_id=sample_customer.id,
				payment_number=f"PAY-2025-{i+1:03d}",
				payment_date=date.today() - timedelta(days=25*(i+1)),
				payment_amount=Decimal('2500.00'),
				payment_method="ach",
				currency_code="USD",
				status=ARPaymentStatus.CLEARED,
				applied_amount=Decimal('2500.00'),
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
			payments.append(payment)
		return payments
	
	@pytest.fixture
	def credit_config(self):
		"""Create credit scoring configuration for testing."""
		return CreditScoringConfig(
			federated_learning_endpoint="https://test.fl.com/v1",
			model_name="test_credit_model",
			model_version="1.0.0",
			min_confidence_threshold=0.8,
			manual_review_threshold=0.6,
			default_credit_limit=Decimal('5000.00')
		)
	
	@pytest.fixture
	def credit_service(self, credit_config):
		"""Create credit scoring service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return APGCreditScoringService(tenant_id, user_id, credit_config)
	
	async def test_extract_customer_features(self, credit_service, sample_customer, 
											sample_invoices, sample_payments):
		"""Test feature extraction from customer data."""
		features = await credit_service._extract_customer_features(
			sample_customer, sample_invoices, sample_payments
		)
		
		assert isinstance(features, CreditScoringFeatures)
		assert features.customer_age_months >= 0
		assert features.customer_type == "corporation"
		assert features.total_invoices == 10
		assert features.paid_invoices == 8
		assert features.late_payments == 2  # 2 overdue invoices
		assert features.current_outstanding == Decimal('8500.00')
		assert features.payment_consistency_score == 0.8  # 8/10 = 0.8
	
	async def test_assess_customer_credit(self, credit_service, sample_customer, 
										 sample_invoices, sample_payments):
		"""Test comprehensive credit assessment."""
		result = await credit_service.assess_customer_credit(
			sample_customer, sample_invoices, sample_payments
		)
		
		assert isinstance(result, CreditScoringResult)
		assert result.customer_id == sample_customer.id
		assert 300 <= result.credit_score <= 850
		assert isinstance(result.risk_rating, ARCreditRating)
		assert 0 <= result.default_probability <= 1
		assert 0 <= result.confidence_score <= 1
		assert result.recommended_credit_limit > 0
		assert result.payment_terms_days >= 0
		assert isinstance(result.next_review_date, date)
		assert result.next_review_date > date.today()
	
	async def test_assess_customer_credit_minimal_data(self, credit_service, sample_customer):
		"""Test credit assessment with minimal customer data."""
		# Test with no invoices or payments
		result = await credit_service.assess_customer_credit(sample_customer, [], [])
		
		assert isinstance(result, CreditScoringResult)
		assert result.customer_id == sample_customer.id
		assert result.requires_manual_review  # Should require review due to lack of data
		assert 'limited_payment_history' in result.risk_factors or 'insufficient_data' in result.risk_factors
	
	async def test_batch_assess_customers(self, credit_service, sample_customer):
		"""Test batch assessment of multiple customers."""
		# Create multiple customers
		customers = []
		for i in range(3):
			customer = ARCustomer(
				id=uuid7str(),
				tenant_id=sample_customer.tenant_id,
				customer_code=f"BATCH{i+1:03d}",
				legal_name=f"Batch Customer {i+1}",
				customer_type=ARCustomerType.CORPORATION,
				status=ARCustomerStatus.ACTIVE,
				credit_limit=Decimal('10000.00'),
				payment_terms_days=30,
				total_outstanding=Decimal('5000.00'),
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
			customers.append(customer)
		
		results = await credit_service.batch_assess_customers(customers)
		
		assert len(results) == 3
		for result in results:
			assert isinstance(result, CreditScoringResult)
			assert 300 <= result.credit_score <= 850
	
	async def test_create_credit_assessment_record(self, credit_service, sample_customer):
		"""Test creating credit assessment database record."""
		# First assess the customer
		scoring_result = await credit_service.assess_customer_credit(sample_customer, [], [])
		
		# Create assessment record
		assessment = await credit_service.create_credit_assessment_record(scoring_result)
		
		assert isinstance(assessment, ARCreditAssessment)
		assert assessment.tenant_id == credit_service.tenant_id
		assert assessment.customer_id == scoring_result.customer_id
		assert assessment.assessment_type == 'ai_automated'
		assert assessment.credit_score == scoring_result.credit_score
		assert assessment.risk_rating == scoring_result.risk_rating
		assert assessment.ai_model_version == scoring_result.model_version
		assert assessment.ai_confidence_score == scoring_result.confidence_score
	
	async def test_update_customer_credit_info(self, credit_service, sample_customer):
		"""Test updating customer with new credit information."""
		# Get scoring result
		scoring_result = await credit_service.assess_customer_credit(sample_customer, [], [])
		
		# Force high confidence and no manual review for testing
		scoring_result.confidence_score = 0.95
		scoring_result.requires_manual_review = False
		
		# Update customer
		updated_customer = await credit_service.update_customer_credit_info(
			sample_customer, scoring_result
		)
		
		assert updated_customer.credit_limit == scoring_result.recommended_credit_limit
		assert updated_customer.credit_rating == scoring_result.risk_rating
		assert updated_customer.payment_terms_days == scoring_result.payment_terms_days
	
	def test_determine_risk_rating(self, credit_service):
		"""Test credit score to risk rating mapping."""
		test_cases = [
			(800, ARCreditRating.AAA),
			(720, ARCreditRating.AA),
			(680, ARCreditRating.A),
			(620, ARCreditRating.BBB),
			(580, ARCreditRating.BB),
			(520, ARCreditRating.B),
			(480, ARCreditRating.CCC),
			(420, ARCreditRating.CC),
			(380, ARCreditRating.C),
			(320, ARCreditRating.D)
		]
		
		for score, expected_rating in test_cases:
			rating = credit_service._determine_risk_rating(score)
			assert rating == expected_rating
	
	def test_identify_risk_factors(self, credit_service):
		"""Test risk factor identification logic."""
		# High-risk features
		high_risk_features = CreditScoringFeatures(
			customer_age_months=6,  # Limited history
			customer_type="individual",
			total_invoices=10,
			paid_invoices=7,  # 30% late payment rate
			late_payments=3,
			current_outstanding=Decimal('8000.00'),
			credit_utilization=0.9,  # High utilization
			payment_consistency_score=0.6,  # Low consistency
			disputed_invoices=3,  # Frequent disputes
			economic_sector_risk=0.7  # High industry risk
		)
		
		risk_factors = credit_service._identify_risk_factors(high_risk_features, 450)
		
		assert 'high_late_payment_rate' in risk_factors
		assert 'high_credit_utilization' in risk_factors
		assert 'limited_payment_history' in risk_factors
		assert 'inconsistent_payment_behavior' in risk_factors
		assert 'frequent_disputes' in risk_factors
		assert 'high_industry_risk' in risk_factors
		assert 'poor_credit_score' in risk_factors
	
	def test_identify_positive_factors(self, credit_service):
		"""Test positive factor identification logic."""
		# Low-risk features
		low_risk_features = CreditScoringFeatures(
			customer_age_months=36,  # Established relationship
			customer_type="corporation",
			total_invoices=50,
			paid_invoices=49,  # 98% payment rate
			late_payments=1,
			current_outstanding=Decimal('5000.00'),
			credit_utilization=0.25,  # Low utilization
			payment_consistency_score=0.95,  # High consistency
			avg_payment_days=20.0,  # Fast payment
			dispute_resolution_rate=0.95  # Good dispute resolution
		)
		
		positive_factors = credit_service._identify_positive_factors(low_risk_features, 750)
		
		assert 'excellent_payment_consistency' in positive_factors
		assert 'established_relationship' in positive_factors
		assert 'low_credit_utilization' in positive_factors
		assert 'excellent_payment_history' in positive_factors
		assert 'fast_payment_processing' in positive_factors
		assert 'good_dispute_resolution' in positive_factors
		assert 'high_credit_score' in positive_factors


class TestCreditScoringModelTrainer:
	"""Test credit scoring model training integration."""
	
	@pytest.fixture
	def model_trainer(self, credit_config):
		"""Create model trainer for testing."""
		tenant_id = uuid7str()
		return CreditScoringModelTrainer(tenant_id, credit_config)
	
	async def test_prepare_training_data(self, model_trainer):
		"""Test training data preparation."""
		training_data = await model_trainer.prepare_training_data()
		
		assert isinstance(training_data, dict)
		assert 'tenant_id' in training_data
		assert 'model_name' in training_data
		assert 'training_samples' in training_data
		assert 'feature_definitions' in training_data
		assert 'privacy_parameters' in training_data
		
		# Check privacy parameters
		privacy_params = training_data['privacy_parameters']
		assert privacy_params['differential_privacy'] is True
		assert 'epsilon' in privacy_params
	
	async def test_submit_training_job(self, model_trainer):
		"""Test submitting training job to federated learning."""
		job_id = await model_trainer.submit_training_job()
		
		assert isinstance(job_id, str)
		assert len(job_id) > 0
	
	async def test_check_training_status(self, model_trainer):
		"""Test checking training job status."""
		job_id = uuid7str()
		status = await model_trainer.check_training_status(job_id)
		
		assert isinstance(status, dict)
		assert 'job_id' in status
		assert 'status' in status
		assert 'model_version' in status
		assert 'accuracy_metrics' in status
		
		# Check accuracy metrics
		metrics = status['accuracy_metrics']
		assert 'precision' in metrics
		assert 'recall' in metrics
		assert 'f1_score' in metrics


class TestServiceFactory:
	"""Test credit scoring service factory functions."""
	
	async def test_create_credit_scoring_service_default_config(self):
		"""Test creating service with default configuration."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		
		service = await create_credit_scoring_service(tenant_id, user_id)
		
		assert isinstance(service, APGCreditScoringService)
		assert service.tenant_id == tenant_id
		assert service.user_id == user_id
		assert service.config.model_name == "ar_credit_scoring_v2"
		assert service.config.model_version == "2.1.0"
	
	async def test_create_credit_scoring_service_custom_config(self):
		"""Test creating service with custom configuration."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		
		custom_config = CreditScoringConfig(
			federated_learning_endpoint="https://custom.fl.com/v1",
			model_name="custom_model",
			model_version="3.0.0"
		)
		
		service = await create_credit_scoring_service(tenant_id, user_id, custom_config)
		
		assert isinstance(service, APGCreditScoringService)
		assert service.config.model_name == "custom_model"
		assert service.config.model_version == "3.0.0"


class TestIntegrationScenarios:
	"""Test realistic integration scenarios."""
	
	@pytest.fixture
	def established_customer(self):
		"""Customer with established payment history."""
		return ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="ESTABLISHED001",
			legal_name="Established Corp",
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('50000.00'),
			payment_terms_days=30,
			total_outstanding=Decimal('12000.00'),
			overdue_amount=Decimal('0.00'),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
	
	@pytest.fixture
	def risky_customer(self):
		"""Customer with risky payment patterns."""
		return ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="RISKY001",
			legal_name="Risky Ventures LLC",
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('10000.00'),
			payment_terms_days=30,
			total_outstanding=Decimal('9500.00'),
			overdue_amount=Decimal('4500.00'),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
	
	async def test_established_customer_assessment(self, established_customer):
		"""Test assessment of established customer with good history."""
		service = await create_credit_scoring_service(uuid7str(), uuid7str())
		
		# Create good payment history
		invoices = []
		for i in range(20):
			invoice = ARInvoice(
				id=uuid7str(),
				tenant_id=established_customer.tenant_id,
				customer_id=established_customer.id,
				invoice_number=f"EST-{i+1:03d}",
				invoice_date=date.today() - timedelta(days=30*i),
				due_date=date.today() - timedelta(days=30*i-30),
				total_amount=Decimal('3000.00'),
				paid_amount=Decimal('3000.00'),
				currency_code="USD",
				status=ARInvoiceStatus.PAID,
				payment_status="paid",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
			invoices.append(invoice)
		
		result = await service.assess_customer_credit(established_customer, invoices, [])
		
		# Should get good rating due to consistent payment history
		assert result.credit_score >= 600
		assert result.risk_rating in [ARCreditRating.AAA, ARCreditRating.AA, ARCreditRating.A, ARCreditRating.BBB]
		assert not result.requires_manual_review or result.confidence_score > 0.8
		assert 'excellent_payment_history' in result.positive_factors or 'excellent_payment_consistency' in result.positive_factors
	
	async def test_risky_customer_assessment(self, risky_customer):
		"""Test assessment of risky customer with poor history."""
		service = await create_credit_scoring_service(uuid7str(), uuid7str())
		
		# Create problematic payment history
		invoices = []
		for i in range(10):
			# 60% of invoices overdue
			is_overdue = i < 6
			invoice = ARInvoice(
				id=uuid7str(),
				tenant_id=risky_customer.tenant_id,
				customer_id=risky_customer.id,
				invoice_number=f"RISK-{i+1:03d}",
				invoice_date=date.today() - timedelta(days=30*i),
				due_date=date.today() - timedelta(days=30*i-30),
				total_amount=Decimal('1500.00'),
				paid_amount=Decimal('0.00') if is_overdue else Decimal('1500.00'),
				currency_code="USD",
				status=ARInvoiceStatus.OVERDUE if is_overdue else ARInvoiceStatus.PAID,
				payment_status="overdue" if is_overdue else "paid",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
			invoices.append(invoice)
		
		result = await service.assess_customer_credit(risky_customer, invoices, [])
		
		# Should get poor rating due to payment issues
		assert result.credit_score <= 600
		assert result.requires_manual_review
		assert 'high_late_payment_rate' in result.risk_factors
		assert 'high_credit_utilization' in result.risk_factors


# Run tests
if __name__ == "__main__":
	pytest.main([__file__, "-v"])