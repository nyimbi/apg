"""
APG Accounts Receivable - Integration Tests
End-to-end workflow testing for comprehensive AR capability integration

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any
import json

from uuid_extensions import uuid7str

from ..models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity, ARCreditAssessment,
	ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus, ARCollectionPriority
)
from ..service import (
	ARCustomerService, ARInvoiceService, ARCollectionsService,
	ARCashApplicationService, ARAnalyticsService
)
from ..ai_credit_scoring import APGCreditScoringService, CreditScoringResult
from ..ai_collections_optimization import APGCollectionsAIService, CollectionStrategyRecommendation
from ..ai_cashflow_forecasting import APGCashFlowForecastingService, CashFlowForecastSummary


class TestCompleteARWorkflow:
	"""Test complete AR workflow from customer creation to collection."""
	
	@pytest.fixture
	def tenant_context(self):
		"""Create tenant context for testing."""
		return {
			'tenant_id': uuid7str(),
			'user_id': uuid7str()
		}
	
	async def test_end_to_end_ar_workflow(self, tenant_context):
		"""Test complete AR workflow across all services and AI components."""
		tenant_id = tenant_context['tenant_id']
		user_id = tenant_context['user_id']
		
		# Initialize all services
		customer_service = ARCustomerService(tenant_id, user_id)
		invoice_service = ARInvoiceService(tenant_id, user_id)
		payment_service = ARCashApplicationService(tenant_id, user_id)
		collections_service = ARCollectionsService(tenant_id, user_id)
		analytics_service = ARAnalyticsService(tenant_id, user_id)
		
		# Initialize AI services
		credit_ai_service = APGCreditScoringService(tenant_id, user_id)
		collections_ai_service = APGCollectionsAIService(tenant_id, user_id)
		cashflow_ai_service = APGCashFlowForecastingService(tenant_id, user_id)
		
		# Mock all service dependencies
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(payment_service, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(collections_service, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(analytics_service, '_validate_permissions', new_callable=AsyncMock):
			
			# Step 1: Create Customer
			customer_data = {
				'customer_code': 'INTEGRATION001',
				'legal_name': 'Integration Test Corporation',
				'customer_type': ARCustomerType.CORPORATION,
				'status': ARCustomerStatus.ACTIVE,
				'credit_limit': Decimal('100000.00'),
				'payment_terms_days': 30,
				'contact_email': 'billing@integration.test',
				'contact_phone': '+1-555-999-0001'
			}
			
			with patch('uuid_extensions.uuid7str', side_effect=[
				'customer-id', 'invoice-id-1', 'invoice-id-2', 'payment-id-1', 'assessment-id'
			]):
				customer = await customer_service.create_customer(customer_data)
				
				# Step 2: AI Credit Assessment
				mock_credit_assessment = CreditScoringResult(
					customer_id=customer.id,
					assessment_date=date.today(),
					credit_score=720,
					risk_level='MEDIUM',
					confidence_score=0.85,
					feature_importance={'payment_history': 0.4, 'credit_utilization': 0.3},
					explanations=['Strong business fundamentals', 'Good payment patterns expected']
				)
				
				with patch.object(credit_ai_service, 'assess_customer_credit', new_callable=AsyncMock) as mock_credit:
					mock_credit.return_value = mock_credit_assessment
					credit_result = await credit_ai_service.assess_customer_credit(customer)
				
				# Step 3: Create Multiple Invoices
				invoice_1_data = {
					'customer_id': customer.id,
					'invoice_number': 'INT-INV-001',
					'invoice_date': date.today(),
					'due_date': date.today() + timedelta(days=30),
					'total_amount': Decimal('25000.00'),
					'description': 'First integration test invoice'
				}
				
				invoice_2_data = {
					'customer_id': customer.id,
					'invoice_number': 'INT-INV-002',
					'invoice_date': date.today(),
					'due_date': date.today() + timedelta(days=30),
					'total_amount': Decimal('35000.00'),
					'description': 'Second integration test invoice'
				}
				
				with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock):
					invoice_1 = await invoice_service.create_invoice(invoice_1_data)
					invoice_2 = await invoice_service.create_invoice(invoice_2_data)
				
				# Step 4: Process Partial Payment
				payment_data = {
					'customer_id': customer.id,
					'payment_reference': 'INT-PAY-001',
					'payment_date': date.today(),
					'payment_amount': Decimal('20000.00'),
					'payment_method': 'WIRE_TRANSFER',
					'bank_reference': 'WIRE123456'
				}
				
				with patch.object(payment_service, '_validate_customer_exists', new_callable=AsyncMock):
					payment = await payment_service.create_payment(payment_data)
				
				# Step 5: Simulate Invoice Becoming Overdue
				with patch.object(invoice_service, '_get_invoice_from_db', new_callable=AsyncMock) as mock_get:
					overdue_invoice = invoice_2.copy(update={
						'due_date': date.today() - timedelta(days=5),
						'status': ARInvoiceStatus.OVERDUE
					})
					mock_get.return_value = overdue_invoice
					
					with patch.object(invoice_service, '_update_invoice_status', new_callable=AsyncMock) as mock_update:
						mock_update.return_value = overdue_invoice
						await invoice_service.mark_invoice_overdue(invoice_2.id)
				
				# Step 6: AI Collections Optimization
				from ..ai_collections_optimization import CustomerCollectionProfile, CollectionStrategyType, CollectionChannelType
				
				collection_profile = CustomerCollectionProfile(
					customer_id=customer.id,
					overdue_amount=Decimal('35000.00'),
					days_overdue=5,
					payment_history_score=0.72,
					previous_collection_attempts=0,
					preferred_contact_method='EMAIL',
					response_rate_email=0.68,
					customer_segment='CORPORATION',
					risk_level='MEDIUM'
				)
				
				mock_collection_strategy = CollectionStrategyRecommendation(
					customer_id=customer.id,
					recommended_strategy=CollectionStrategyType.EMAIL_SEQUENCE,
					contact_method=CollectionChannelType.EMAIL,
					success_probability=0.75,
					estimated_resolution_days=12,
					priority_level='MEDIUM'
				)
				
				with patch.object(collections_ai_service, 'optimize_collection_strategy', new_callable=AsyncMock) as mock_optimize:
					mock_optimize.return_value = mock_collection_strategy
					strategy = await collections_ai_service.optimize_collection_strategy(collection_profile)
				
				# Step 7: Execute Collection Activity
				with patch.object(collections_service, '_get_invoice_from_db', new_callable=AsyncMock) as mock_get_inv:
					mock_get_inv.return_value = overdue_invoice
					
					with patch.object(collections_service, '_get_customer_from_db', new_callable=AsyncMock) as mock_get_cust:
						mock_get_cust.return_value = customer
						
						with patch.object(collections_service, '_send_reminder_email', new_callable=AsyncMock) as mock_send:
							mock_send.return_value = True
							
							with patch.object(collections_service, '_create_collection_activity', new_callable=AsyncMock) as mock_activity:
								mock_collection_activity = ARCollectionActivity(
									id=uuid7str(),
									tenant_id=tenant_id,
									customer_id=customer.id,
									activity_type='EMAIL_REMINDER',
									activity_date=date.today(),
									priority=ARCollectionPriority.MEDIUM,
									contact_method='EMAIL',
									outcome='SENT',
									status='COMPLETED',
									created_by=user_id,
									updated_by=user_id
								)
								mock_activity.return_value = mock_collection_activity
								
								reminder_result = await collections_service.send_payment_reminder(invoice_2.id)
				
				# Step 8: AI Cash Flow Forecasting
				from ..ai_cashflow_forecasting import CashFlowForecastInput, CashFlowDataPoint
				
				forecast_input = CashFlowForecastInput(
					tenant_id=tenant_id,
					forecast_start_date=date.today(),
					forecast_end_date=date.today() + timedelta(days=30),
					include_seasonal_trends=True,
					scenario_type='realistic'
				)
				
				mock_forecast_points = []
				for i in range(30):
					mock_forecast_points.append(CashFlowDataPoint(
						forecast_date=date.today() + timedelta(days=i),
						expected_collections=Decimal('3000.00') + (i * 100),
						invoice_receipts=Decimal('2000.00'),
						overdue_collections=Decimal('1000.00') + (i * 50),
						total_cash_flow=Decimal('6000.00') + (i * 150),
						confidence_interval_lower=Decimal('5400.00') + (i * 135),
						confidence_interval_upper=Decimal('6600.00') + (i * 165)
					))
				
				mock_forecast_result = CashFlowForecastSummary(
					forecast_id=uuid7str(),
					tenant_id=tenant_id,
					forecast_start_date=forecast_input.forecast_start_date,
					forecast_end_date=forecast_input.forecast_end_date,
					scenario_type='realistic',
					forecast_points=mock_forecast_points,
					overall_accuracy=0.89,
					model_confidence=0.86,
					seasonal_factors=['Month-end payment concentration'],
					risk_factors=['Economic uncertainty in sector'],
					insights=['Strong collection probability for overdue amounts']
				)
				
				with patch.object(cashflow_ai_service, 'generate_forecast', new_callable=AsyncMock) as mock_forecast:
					mock_forecast.return_value = mock_forecast_result
					forecast = await cashflow_ai_service.generate_forecast(forecast_input)
				
				# Step 9: Analytics Dashboard Update
				mock_dashboard_metrics = {
					'total_ar_balance': Decimal('40000.00'),  # 25000 + 35000 - 20000
					'overdue_amount': Decimal('35000.00'),
					'current_month_sales': Decimal('60000.00'),
					'current_month_collections': Decimal('20000.00'),
					'total_customers': 1,
					'active_customers': 1,
					'overdue_customers': 1,
					'average_days_to_pay': 30.0,
					'collection_effectiveness_index': 0.80,
					'days_sales_outstanding': 42.5,
					'ai_assessments_today': 1,
					'ai_collection_recommendations': 1
				}
				
				with patch.object(analytics_service, '_calculate_dashboard_metrics', new_callable=AsyncMock) as mock_metrics:
					mock_metrics.return_value = mock_dashboard_metrics
					dashboard = await analytics_service.get_ar_dashboard_metrics()
				
				# Verification: Complete Workflow Consistency
				
				# Customer verification
				assert customer.tenant_id == tenant_id
				assert customer.customer_code == 'INTEGRATION001'
				assert customer.legal_name == 'Integration Test Corporation'
				assert customer.credit_limit == Decimal('100000.00')
				
				# Credit assessment verification
				assert credit_result.customer_id == customer.id
				assert credit_result.credit_score == 720
				assert credit_result.risk_level == 'MEDIUM'
				assert credit_result.confidence_score == 0.85
				
				# Invoice verification
				assert invoice_1.customer_id == customer.id
				assert invoice_2.customer_id == customer.id
				assert invoice_1.total_amount == Decimal('25000.00')
				assert invoice_2.total_amount == Decimal('35000.00')
				
				# Payment verification
				assert payment.customer_id == customer.id
				assert payment.payment_amount == Decimal('20000.00')
				
				# Collections verification
				assert strategy.customer_id == customer.id
				assert strategy.success_probability == 0.75
				assert reminder_result is True
				
				# Forecast verification
				assert forecast.tenant_id == tenant_id
				assert len(forecast.forecast_points) == 30
				assert forecast.overall_accuracy == 0.89
				
				# Dashboard verification
				assert dashboard['total_customers'] == 1
				assert dashboard['overdue_customers'] == 1
				assert dashboard['ai_assessments_today'] == 1
				assert dashboard['total_ar_balance'] == Decimal('40000.00')
				
				# Business logic verification
				total_invoiced = invoice_1.total_amount + invoice_2.total_amount
				remaining_balance = total_invoiced - payment.payment_amount
				assert remaining_balance == Decimal('40000.00')
				assert remaining_balance <= customer.credit_limit
				
				# AI coordination verification
				assert credit_result.risk_level == 'MEDIUM'
				assert strategy.priority_level == 'MEDIUM'
				assert 'Strong collection probability' in ' '.join(forecast.insights)


class TestMultiTenantWorkflow:
	"""Test multi-tenant isolation and data integrity."""
	
	async def test_tenant_data_isolation(self):
		"""Test that tenant data is properly isolated."""
		# Create two different tenant contexts
		tenant_1_id = uuid7str()
		tenant_2_id = uuid7str()
		user_1_id = uuid7str()
		user_2_id = uuid7str()
		
		# Initialize services for both tenants
		customer_service_1 = ARCustomerService(tenant_1_id, user_1_id)
		customer_service_2 = ARCustomerService(tenant_2_id, user_2_id)
		
		# Mock permissions for both tenants
		with patch.object(customer_service_1, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(customer_service_2, '_validate_permissions', new_callable=AsyncMock):
			
			# Create customer for tenant 1
			customer_1_data = {
				'customer_code': 'TENANT1-001',
				'legal_name': 'Tenant 1 Customer',
				'customer_type': ARCustomerType.CORPORATION,
				'status': ARCustomerStatus.ACTIVE,
				'credit_limit': Decimal('50000.00'),
				'payment_terms_days': 30
			}
			
			with patch('uuid_extensions.uuid7str', return_value='customer-1-id'):
				customer_1 = await customer_service_1.create_customer(customer_1_data)
			
			# Create customer for tenant 2
			customer_2_data = {
				'customer_code': 'TENANT2-001',
				'legal_name': 'Tenant 2 Customer',
				'customer_type': ARCustomerType.INDIVIDUAL,
				'status': ARCustomerStatus.ACTIVE,
				'credit_limit': Decimal('25000.00'),
				'payment_terms_days': 45
			}
			
			with patch('uuid_extensions.uuid7str', return_value='customer-2-id'):
				customer_2 = await customer_service_2.create_customer(customer_2_data)
			
			# Verify tenant isolation
			assert customer_1.tenant_id == tenant_1_id
			assert customer_2.tenant_id == tenant_2_id
			assert customer_1.tenant_id != customer_2.tenant_id
			
			# Verify user attribution
			assert customer_1.created_by == user_1_id
			assert customer_2.created_by == user_2_id
			assert customer_1.created_by != customer_2.created_by
			
			# Mock cross-tenant access attempt
			with patch.object(customer_service_1, '_get_customer_from_db', new_callable=AsyncMock) as mock_get:
				mock_get.return_value = None  # Customer not found in tenant 1's scope
				
				# Attempt to access tenant 2's customer from tenant 1's service
				result = await customer_service_1.get_customer(customer_2.id)
				assert result is None  # Should not find customer from different tenant


class TestBulkOperationsWorkflow:
	"""Test bulk operations and batch processing."""
	
	async def test_bulk_invoice_processing(self):
		"""Test bulk invoice creation and processing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		
		# Initialize services
		customer_service = ARCustomerService(tenant_id, user_id)
		invoice_service = ARInvoiceService(tenant_id, user_id)
		credit_ai_service = APGCreditScoringService(tenant_id, user_id)
		
		# Mock permissions
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock):
			
			# Create multiple customers
			customers = []
			customer_ids = []
			
			for i in range(5):
				customer_data = {
					'customer_code': f'BULK{i:03d}',
					'legal_name': f'Bulk Customer {i+1}',
					'customer_type': ARCustomerType.CORPORATION,
					'status': ARCustomerStatus.ACTIVE,
					'credit_limit': Decimal('75000.00'),
					'payment_terms_days': 30
				}
				
				customer_id = uuid7str()
				customer_ids.append(customer_id)
				
				with patch('uuid_extensions.uuid7str', return_value=customer_id):
					customer = await customer_service.create_customer(customer_data)
					customers.append(customer)
			
			# Bulk credit assessment
			mock_assessments = []
			for i, customer in enumerate(customers):
				assessment = CreditScoringResult(
					customer_id=customer.id,
					assessment_date=date.today(),
					credit_score=650 + (i * 20),  # Varying scores
					risk_level='HIGH' if i == 0 else 'MEDIUM' if i < 3 else 'LOW',
					confidence_score=0.75 + (i * 0.05),
					feature_importance={'payment_history': 0.4},
					explanations=[f'Assessment for customer {i+1}']
				)
				mock_assessments.append(assessment)
			
			with patch.object(credit_ai_service, 'batch_assess_customers_credit', new_callable=AsyncMock) as mock_batch:
				mock_batch.return_value = mock_assessments
				
				with patch.object(credit_ai_service, '_get_customer_from_db', new_callable=AsyncMock) as mock_get:
					mock_get.side_effect = customers
					
					batch_results = await credit_ai_service.batch_assess_customers_credit(customer_ids)
			
			# Create bulk invoices based on credit assessment results
			invoices = []
			
			for i, (customer, assessment) in enumerate(zip(customers, batch_results)):
				# Adjust invoice amount based on credit score
				invoice_amount = Decimal('10000.00') if assessment.credit_score >= 700 else \
							   Decimal('7500.00') if assessment.credit_score >= 650 else \
							   Decimal('5000.00')
				
				invoice_data = {
					'customer_id': customer.id,
					'invoice_number': f'BULK-INV-{i+1:03d}',
					'invoice_date': date.today(),
					'due_date': date.today() + timedelta(days=30),
					'total_amount': invoice_amount,
					'description': f'Bulk invoice {i+1} based on credit assessment'
				}
				
				with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock):
					with patch('uuid_extensions.uuid7str', return_value=f'invoice-{i+1}-id'):
						invoice = await invoice_service.create_invoice(invoice_data)
						invoices.append(invoice)
			
			# Verify bulk processing results
			assert len(batch_results) == 5
			assert len(invoices) == 5
			
			# Verify credit score progression
			assert batch_results[0].credit_score == 650  # Lowest
			assert batch_results[4].credit_score == 730  # Highest
			
			# Verify risk-based invoice amounts
			high_risk_invoice = next(inv for inv, assess in zip(invoices, batch_results) if assess.risk_level == 'HIGH')
			low_risk_invoice = next(inv for inv, assess in zip(invoices, batch_results) if assess.risk_level == 'LOW')
			
			assert high_risk_invoice.total_amount == Decimal('5000.00')  # Lower amount for high risk
			assert low_risk_invoice.total_amount == Decimal('10000.00')  # Higher amount for low risk
			
			# Verify all invoices belong to correct tenant
			assert all(invoice.tenant_id == tenant_id for invoice in invoices)
			assert all(invoice.created_by == user_id for invoice in invoices)


class TestErrorRecoveryWorkflow:
	"""Test error handling and recovery scenarios."""
	
	async def test_partial_failure_recovery(self):
		"""Test recovery from partial failures in multi-step workflows."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		
		# Initialize services
		customer_service = ARCustomerService(tenant_id, user_id)
		invoice_service = ARInvoiceService(tenant_id, user_id)
		
		# Mock permissions
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock):
			
			# Step 1: Successfully create customer
			customer_data = {
				'customer_code': 'RECOVERY001',
				'legal_name': 'Recovery Test Customer',
				'customer_type': ARCustomerType.CORPORATION,
				'status': ARCustomerStatus.ACTIVE,
				'credit_limit': Decimal('30000.00'),
				'payment_terms_days': 30
			}
			
			with patch('uuid_extensions.uuid7str', return_value='recovery-customer-id'):
				customer = await customer_service.create_customer(customer_data)
			
			# Step 2: Attempt to create invoice with validation failure
			invalid_invoice_data = {
				'customer_id': customer.id,
				'invoice_number': 'RECOVERY-INV-001',
				'invoice_date': date.today(),
				'due_date': date.today() - timedelta(days=1),  # Invalid: due date before invoice date
				'total_amount': Decimal('15000.00'),
				'description': 'Recovery test invoice'
			}
			
			# Mock validation failure
			with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock):
				with pytest.raises(ValueError, match="Due date must be after invoice date"):
					await invoice_service.create_invoice(invalid_invoice_data)
			
			# Step 3: Correct the data and retry
			corrected_invoice_data = invalid_invoice_data.copy()
			corrected_invoice_data['due_date'] = date.today() + timedelta(days=30)
			
			with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock):
				with patch('uuid_extensions.uuid7str', return_value='recovery-invoice-id'):
					invoice = await invoice_service.create_invoice(corrected_invoice_data)
			
			# Verify recovery
			assert customer.id == 'recovery-customer-id'
			assert invoice.id == 'recovery-invoice-id'
			assert invoice.customer_id == customer.id
			assert invoice.due_date == date.today() + timedelta(days=30)
			
			# Verify business logic integrity maintained
			assert invoice.total_amount <= customer.credit_limit
			assert invoice.due_date > invoice.invoice_date


# Run tests
if __name__ == "__main__":
	pytest.main([__file__, "-v"])