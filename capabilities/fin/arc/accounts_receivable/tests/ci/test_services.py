"""
APG Accounts Receivable - Services Tests
Unit tests for business logic services with APG integration

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from uuid_extensions import uuid7str

from ..service import (
	ARCustomerService, ARInvoiceService, ARCollectionsService,
	ARCashApplicationService, ARAnalyticsService
)
from ..models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity,
	ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus
)


class TestARCustomerService:
	"""Test AR customer service functionality."""
	
	@pytest.fixture
	def customer_service(self):
		"""Create customer service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return ARCustomerService(tenant_id, user_id)
	
	@pytest.fixture
	def sample_customer_data(self):
		"""Create sample customer data for testing."""
		return {
			'customer_code': 'TEST001',
			'legal_name': 'Test Customer Corp',
			'customer_type': ARCustomerType.CORPORATION,
			'status': ARCustomerStatus.ACTIVE,
			'credit_limit': Decimal('25000.00'),
			'payment_terms_days': 30,
			'contact_email': 'billing@testcorp.com',
			'contact_phone': '+1-555-123-4567'
		}
	
	async def test_create_customer_success(self, customer_service, sample_customer_data):
		"""Test successful customer creation."""
		
		# Mock permission validation
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock) as mock_validate:
			mock_validate.return_value = None
			
			# Mock database operations
			with patch('uuid_extensions.uuid7str', return_value='test-customer-id'):
				customer = await customer_service.create_customer(sample_customer_data)
				
				assert customer.id == 'test-customer-id'
				assert customer.tenant_id == customer_service.tenant_id
				assert customer.customer_code == 'TEST001'
				assert customer.legal_name == 'Test Customer Corp'
				assert customer.customer_type == ARCustomerType.CORPORATION
				assert customer.status == ARCustomerStatus.ACTIVE
				assert customer.credit_limit == Decimal('25000.00')
				assert customer.payment_terms_days == 30
				assert customer.contact_email == 'billing@testcorp.com'
				assert customer.created_by == customer_service.user_id
				assert customer.updated_by == customer_service.user_id
	
	async def test_create_customer_duplicate_code(self, customer_service, sample_customer_data):
		"""Test customer creation with duplicate customer code."""
		
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(customer_service, '_check_customer_code_unique', new_callable=AsyncMock) as mock_check:
				mock_check.side_effect = ValueError("Customer code already exists")
				
				with pytest.raises(ValueError, match="Customer code already exists"):
					await customer_service.create_customer(sample_customer_data)
	
	async def test_get_customer_success(self, customer_service):
		"""Test successful customer retrieval."""
		customer_id = uuid7str()
		
		mock_customer = ARCustomer(
			id=customer_id,
			tenant_id=customer_service.tenant_id,
			customer_code='EXISTING001',
			legal_name='Existing Customer',
			customer_type=ARCustomerType.INDIVIDUAL,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('10000.00'),
			payment_terms_days=30,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(customer_service, '_get_customer_from_db', new_callable=AsyncMock) as mock_get:
				mock_get.return_value = mock_customer
				
				customer = await customer_service.get_customer(customer_id)
				
				assert customer.id == customer_id
				assert customer.customer_code == 'EXISTING001'
				assert customer.legal_name == 'Existing Customer'
				mock_get.assert_called_once_with(customer_id)
	
	async def test_get_customer_not_found(self, customer_service):
		"""Test customer retrieval when customer not found."""
		customer_id = uuid7str()
		
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(customer_service, '_get_customer_from_db', new_callable=AsyncMock) as mock_get:
				mock_get.return_value = None
				
				customer = await customer_service.get_customer(customer_id)
				
				assert customer is None
				mock_get.assert_called_once_with(customer_id)
	
	async def test_update_customer_success(self, customer_service):
		"""Test successful customer update."""
		customer_id = uuid7str()
		
		existing_customer = ARCustomer(
			id=customer_id,
			tenant_id=customer_service.tenant_id,
			customer_code='UPDATE001',
			legal_name='Original Name',
			customer_type=ARCustomerType.INDIVIDUAL,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('15000.00'),
			payment_terms_days=30,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		update_data = {
			'legal_name': 'Updated Name',
			'credit_limit': Decimal('20000.00'),
			'contact_email': 'updated@example.com'
		}
		
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(customer_service, '_get_customer_from_db', new_callable=AsyncMock) as mock_get:
				mock_get.return_value = existing_customer
				
				with patch.object(customer_service, '_update_customer_in_db', new_callable=AsyncMock) as mock_update:
					updated_customer = existing_customer.copy(update=update_data)
					updated_customer.updated_by = customer_service.user_id
					updated_customer.updated_at = datetime.utcnow()
					mock_update.return_value = updated_customer
					
					result = await customer_service.update_customer(customer_id, update_data)
					
					assert result.legal_name == 'Updated Name'
					assert result.credit_limit == Decimal('20000.00')
					assert result.contact_email == 'updated@example.com'
					assert result.updated_by == customer_service.user_id
	
	async def test_get_customers_filtered(self, customer_service):
		"""Test filtered customer retrieval."""
		
		filter_params = {
			'customer_type': ARCustomerType.CORPORATION,
			'status': ARCustomerStatus.ACTIVE,
			'min_outstanding': Decimal('1000.00'),
			'page': 1,
			'per_page': 20
		}
		
		mock_customers = [
			ARCustomer(
				id=uuid7str(),
				tenant_id=customer_service.tenant_id,
				customer_code=f'CORP{i:03d}',
				legal_name=f'Corporation {i}',
				customer_type=ARCustomerType.CORPORATION,
				status=ARCustomerStatus.ACTIVE,
				credit_limit=Decimal('50000.00'),
				total_outstanding=Decimal('5000.00'),
				payment_terms_days=30,
				created_by=uuid7str(),
				updated_by=uuid7str()
			) for i in range(1, 6)
		]
		
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(customer_service, '_get_customers_filtered_from_db', new_callable=AsyncMock) as mock_get:
				mock_get.return_value = mock_customers
				
				customers = await customer_service.get_customers_filtered(**filter_params)
				
				assert len(customers) == 5
				assert all(c.customer_type == ARCustomerType.CORPORATION for c in customers)
				assert all(c.status == ARCustomerStatus.ACTIVE for c in customers)
				mock_get.assert_called_once()
	
	async def test_get_customer_summary(self, customer_service):
		"""Test customer summary generation."""
		customer_id = uuid7str()
		
		mock_customer = ARCustomer(
			id=customer_id,
			tenant_id=customer_service.tenant_id,
			customer_code='SUMMARY001',
			legal_name='Summary Test Customer',
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('30000.00'),
			total_outstanding=Decimal('12000.00'),
			overdue_amount=Decimal('3000.00'),
			payment_terms_days=30,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		mock_summary_data = {
			'total_invoices': 15,
			'total_payments': 12,
			'average_payment_days': 28.5,
			'last_payment_date': date.today() - timedelta(days=5),
			'last_payment_amount': Decimal('2500.00'),
			'credit_utilization': 0.4,  # 12000 / 30000
			'payment_history_rating': 'GOOD'
		}
		
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(customer_service, '_get_customer_from_db', new_callable=AsyncMock) as mock_get:
				mock_get.return_value = mock_customer
				
				with patch.object(customer_service, '_calculate_customer_summary', new_callable=AsyncMock) as mock_calc:
					mock_calc.return_value = mock_summary_data
					
					summary = await customer_service.get_customer_summary(customer_id)
					
					assert summary['customer'] == mock_customer
					assert summary['total_invoices'] == 15
					assert summary['total_payments'] == 12
					assert summary['average_payment_days'] == 28.5
					assert summary['credit_utilization'] == 0.4
					assert summary['payment_history_rating'] == 'GOOD'


class TestARInvoiceService:
	"""Test AR invoice service functionality."""
	
	@pytest.fixture
	def invoice_service(self):
		"""Create invoice service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return ARInvoiceService(tenant_id, user_id)
	
	@pytest.fixture
	def sample_invoice_data(self):
		"""Create sample invoice data for testing."""
		return {
			'customer_id': uuid7str(),
			'invoice_number': 'INV-TEST-001',
			'invoice_date': date.today(),
			'due_date': date.today() + timedelta(days=30),
			'total_amount': Decimal('5000.00'),
			'currency_code': 'USD',
			'description': 'Test invoice for services'
		}
	
	async def test_create_invoice_success(self, invoice_service, sample_invoice_data):
		"""Test successful invoice creation."""
		
		with patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock):
				with patch('uuid_extensions.uuid7str', return_value='test-invoice-id'):
					invoice = await invoice_service.create_invoice(sample_invoice_data)
					
					assert invoice.id == 'test-invoice-id'
					assert invoice.tenant_id == invoice_service.tenant_id
					assert invoice.customer_id == sample_invoice_data['customer_id']
					assert invoice.invoice_number == 'INV-TEST-001'
					assert invoice.total_amount == Decimal('5000.00')
					assert invoice.outstanding_amount == Decimal('5000.00')
					assert invoice.status == ARInvoiceStatus.DRAFT
					assert invoice.created_by == invoice_service.user_id
	
	async def test_create_invoice_invalid_customer(self, invoice_service, sample_invoice_data):
		"""Test invoice creation with invalid customer."""
		
		with patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock) as mock_validate:
				mock_validate.side_effect = ValueError("Customer not found")
				
				with pytest.raises(ValueError, match="Customer not found"):
					await invoice_service.create_invoice(sample_invoice_data)
	
	async def test_mark_invoice_overdue(self, invoice_service):
		"""Test marking invoice as overdue."""
		invoice_id = uuid7str()
		
		mock_invoice = ARInvoice(
			id=invoice_id,
			tenant_id=invoice_service.tenant_id,
			customer_id=uuid7str(),
			invoice_number='OVERDUE-001',
			invoice_date=date.today() - timedelta(days=45),
			due_date=date.today() - timedelta(days=15),
			total_amount=Decimal('3000.00'),
			outstanding_amount=Decimal('3000.00'),
			status=ARInvoiceStatus.SENT,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		with patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(invoice_service, '_get_invoice_from_db', new_callable=AsyncMock) as mock_get:
				mock_get.return_value = mock_invoice
				
				with patch.object(invoice_service, '_update_invoice_status', new_callable=AsyncMock) as mock_update:
					updated_invoice = mock_invoice.copy(update={'status': ARInvoiceStatus.OVERDUE})
					mock_update.return_value = updated_invoice
					
					result = await invoice_service.mark_invoice_overdue(invoice_id)
					
					assert result.status == ARInvoiceStatus.OVERDUE
					mock_update.assert_called_once_with(invoice_id, ARInvoiceStatus.OVERDUE)
	
	async def test_predict_payment_date(self, invoice_service):
		"""Test AI payment date prediction."""
		invoice_id = uuid7str()
		
		mock_invoice = ARInvoice(
			id=invoice_id,
			tenant_id=invoice_service.tenant_id,
			customer_id=uuid7str(),
			invoice_number='PREDICT-001',
			invoice_date=date.today(),
			due_date=date.today() + timedelta(days=30),
			total_amount=Decimal('2500.00'),
			outstanding_amount=Decimal('2500.00'),
			status=ARInvoiceStatus.SENT,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		mock_prediction = {
			'predicted_payment_date': date.today() + timedelta(days=32),
			'confidence_level': 0.85,
			'factors': [
				'Customer has good payment history',
				'Invoice amount within normal range',
				'No seasonal payment delays expected'
			]
		}
		
		with patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(invoice_service, '_get_invoice_from_db', new_callable=AsyncMock) as mock_get:
				mock_get.return_value = mock_invoice
				
				with patch.object(invoice_service, '_predict_payment_with_ai', new_callable=AsyncMock) as mock_predict:
					mock_predict.return_value = mock_prediction
					
					prediction = await invoice_service.predict_payment_date(invoice_id)
					
					assert prediction['predicted_payment_date'] == date.today() + timedelta(days=32)
					assert prediction['confidence_level'] == 0.85
					assert len(prediction['factors']) == 3
					mock_predict.assert_called_once_with(mock_invoice)


class TestARCollectionsService:
	"""Test AR collections service functionality."""
	
	@pytest.fixture
	def collections_service(self):
		"""Create collections service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return ARCollectionsService(tenant_id, user_id)
	
	async def test_send_payment_reminder(self, collections_service):
		"""Test sending payment reminder."""
		invoice_id = uuid7str()
		customer_id = uuid7str()
		
		mock_invoice = ARInvoice(
			id=invoice_id,
			tenant_id=collections_service.tenant_id,
			customer_id=customer_id,
			invoice_number='REMINDER-001',
			invoice_date=date.today() - timedelta(days=35),
			due_date=date.today() - timedelta(days=5),
			total_amount=Decimal('1500.00'),
			outstanding_amount=Decimal('1500.00'),
			status=ARInvoiceStatus.OVERDUE,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		mock_customer = ARCustomer(
			id=customer_id,
			tenant_id=collections_service.tenant_id,
			customer_code='REMIND001',
			legal_name='Reminder Test Customer',
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			contact_email='billing@remindtest.com',
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		with patch.object(collections_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(collections_service, '_get_invoice_from_db', new_callable=AsyncMock) as mock_get_invoice:
				mock_get_invoice.return_value = mock_invoice
				
				with patch.object(collections_service, '_get_customer_from_db', new_callable=AsyncMock) as mock_get_customer:
					mock_get_customer.return_value = mock_customer
					
					with patch.object(collections_service, '_send_reminder_email', new_callable=AsyncMock) as mock_send:
						mock_send.return_value = True
						
						with patch.object(collections_service, '_create_collection_activity', new_callable=AsyncMock) as mock_create:
							mock_activity = ARCollectionActivity(
								id=uuid7str(),
								tenant_id=collections_service.tenant_id,
								customer_id=customer_id,
								activity_type='EMAIL_REMINDER',
								activity_date=date.today(),
								contact_method='EMAIL',
								outcome='SENT',
								status='COMPLETED',
								created_by=collections_service.user_id,
								updated_by=collections_service.user_id
							)
							mock_create.return_value = mock_activity
							
							result = await collections_service.send_payment_reminder(invoice_id)
							
							assert result is True
							mock_send.assert_called_once()
							mock_create.assert_called_once()
	
	async def test_get_collections_metrics(self, collections_service):
		"""Test collections performance metrics calculation."""
		
		mock_metrics = {
			'total_activities': 150,
			'successful_collections': 105,
			'success_rate': 0.7,
			'average_resolution_days': 12.5,
			'total_collected_amount': Decimal('125000.00'),
			'overdue_amount_recovered': Decimal('85000.00'),
			'current_month_collections': Decimal('32000.00'),
			'previous_month_collections': Decimal('28000.00'),
			'month_over_month_growth': 0.143  # (32000 - 28000) / 28000
		}
		
		with patch.object(collections_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(collections_service, '_calculate_collections_metrics', new_callable=AsyncMock) as mock_calc:
				mock_calc.return_value = mock_metrics
				
				metrics = await collections_service.get_collections_metrics()
				
				assert metrics['total_activities'] == 150
				assert metrics['successful_collections'] == 105
				assert metrics['success_rate'] == 0.7
				assert metrics['average_resolution_days'] == 12.5
				assert metrics['total_collected_amount'] == Decimal('125000.00')
				assert metrics['month_over_month_growth'] == 0.143


class TestARCashApplicationService:
	"""Test AR cash application service functionality."""
	
	@pytest.fixture
	def cash_service(self):
		"""Create cash application service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return ARCashApplicationService(tenant_id, user_id)
	
	@pytest.fixture
	def sample_payment_data(self):
		"""Create sample payment data for testing."""
		return {
			'customer_id': uuid7str(),
			'payment_reference': 'PAY-TEST-001',
			'payment_date': date.today(),
			'payment_amount': Decimal('2500.00'),
			'payment_method': 'WIRE_TRANSFER',
			'bank_reference': 'WIRE123456789',
			'currency_code': 'USD'
		}
	
	async def test_create_payment_success(self, cash_service, sample_payment_data):
		"""Test successful payment creation."""
		
		with patch.object(cash_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(cash_service, '_validate_customer_exists', new_callable=AsyncMock):
				with patch('uuid_extensions.uuid7str', return_value='test-payment-id'):
					payment = await cash_service.create_payment(sample_payment_data)
					
					assert payment.id == 'test-payment-id'
					assert payment.tenant_id == cash_service.tenant_id
					assert payment.customer_id == sample_payment_data['customer_id']
					assert payment.payment_reference == 'PAY-TEST-001'
					assert payment.payment_amount == Decimal('2500.00')
					assert payment.payment_method == 'WIRE_TRANSFER'
					assert payment.status == ARPaymentStatus.PENDING
					assert payment.created_by == cash_service.user_id
	
	async def test_process_payment(self, cash_service):
		"""Test payment processing."""
		payment_id = uuid7str()
		
		mock_payment = ARPayment(
			id=payment_id,
			tenant_id=cash_service.tenant_id,
			customer_id=uuid7str(),
			payment_reference='PROCESS-001',
			payment_date=date.today(),
			payment_amount=Decimal('3000.00'),
			payment_method='ACH',
			status=ARPaymentStatus.PENDING,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		with patch.object(cash_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(cash_service, '_get_payment_from_db', new_callable=AsyncMock) as mock_get:
				mock_get.return_value = mock_payment
				
				with patch.object(cash_service, '_process_payment_transaction', new_callable=AsyncMock) as mock_process:
					mock_process.return_value = True
					
					with patch.object(cash_service, '_update_payment_status', new_callable=AsyncMock) as mock_update:
						processed_payment = mock_payment.copy(update={
							'status': ARPaymentStatus.PROCESSED,
							'processed_at': datetime.utcnow()
						})
						mock_update.return_value = processed_payment
						
						result = await cash_service.process_payment(payment_id)
						
						assert result.status == ARPaymentStatus.PROCESSED
						assert result.processed_at is not None
						mock_process.assert_called_once_with(mock_payment)
						mock_update.assert_called_once()


class TestARAnalyticsService:
	"""Test AR analytics service functionality."""
	
	@pytest.fixture
	def analytics_service(self):
		"""Create analytics service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return ARAnalyticsService(tenant_id, user_id)
	
	async def test_get_ar_dashboard_metrics(self, analytics_service):
		"""Test AR dashboard metrics calculation."""
		
		mock_metrics = {
			'total_ar_balance': Decimal('450000.00'),
			'overdue_amount': Decimal('125000.00'),
			'current_month_sales': Decimal('85000.00'),
			'current_month_collections': Decimal('92000.00'),
			'total_customers': 156,
			'active_customers': 142,
			'overdue_customers': 28,
			'average_days_to_pay': 31.5,
			'collection_effectiveness_index': 0.85,
			'days_sales_outstanding': 38.2,
			'ai_assessments_today': 8,
			'ai_collection_recommendations': 15
		}
		
		with patch.object(analytics_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(analytics_service, '_calculate_dashboard_metrics', new_callable=AsyncMock) as mock_calc:
				mock_calc.return_value = mock_metrics
				
				metrics = await analytics_service.get_ar_dashboard_metrics()
				
				assert metrics['total_ar_balance'] == Decimal('450000.00')
				assert metrics['overdue_amount'] == Decimal('125000.00')
				assert metrics['total_customers'] == 156
				assert metrics['active_customers'] == 142
				assert metrics['overdue_customers'] == 28
				assert metrics['collection_effectiveness_index'] == 0.85
				assert metrics['ai_assessments_today'] == 8
	
	async def test_get_aging_analysis(self, analytics_service):
		"""Test aging analysis calculation."""
		as_of_date = date.today()
		
		mock_aging_data = {
			'as_of_date': as_of_date,
			'current': Decimal('185000.00'),
			'days_1_30': Decimal('95000.00'),
			'days_31_60': Decimal('65000.00'),
			'days_61_90': Decimal('45000.00'),
			'days_90_plus': Decimal('60000.00'),
			'total_outstanding': Decimal('450000.00'),
			'aging_buckets': [
				{
					'bucket': 'Current',
					'amount': Decimal('185000.00'),
					'percentage': 41.1,
					'customer_count': 89
				},
				{
					'bucket': '1-30 Days',
					'amount': Decimal('95000.00'),
					'percentage': 21.1,
					'customer_count': 34
				},
				{
					'bucket': '31-60 Days',
					'amount': Decimal('65000.00'),
					'percentage': 14.4,
					'customer_count': 18
				},
				{
					'bucket': '61-90 Days',
					'amount': Decimal('45000.00'),
					'percentage': 10.0,
					'customer_count': 9
				},
				{
					'bucket': '90+ Days',
					'amount': Decimal('60000.00'),
					'percentage': 13.3,
					'customer_count': 6
				}
			]
		}
		
		with patch.object(analytics_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(analytics_service, '_calculate_aging_analysis', new_callable=AsyncMock) as mock_calc:
				mock_calc.return_value = mock_aging_data
				
				aging = await analytics_service.get_aging_analysis(as_of_date)
				
				assert aging['as_of_date'] == as_of_date
				assert aging['total_outstanding'] == Decimal('450000.00')
				assert aging['current'] == Decimal('185000.00')
				assert aging['days_90_plus'] == Decimal('60000.00')
				assert len(aging['aging_buckets']) == 5
				assert aging['aging_buckets'][0]['bucket'] == 'Current'
				assert aging['aging_buckets'][0]['percentage'] == 41.1
				assert aging['aging_buckets'][0]['customer_count'] == 89
	
	async def test_get_collection_performance_metrics(self, analytics_service):
		"""Test collection performance metrics."""
		
		mock_performance = {
			'total_collection_activities': 245,
			'successful_activities': 168,
			'success_rate': 0.686,
			'average_resolution_days': 14.2,
			'total_amount_collected': Decimal('285000.00'),
			'collection_by_method': {
				'EMAIL': {'count': 98, 'success_rate': 0.72, 'amount': Decimal('95000.00')},
				'PHONE': {'count': 85, 'success_rate': 0.68, 'amount': Decimal('125000.00')},
				'LETTER': {'count': 42, 'success_rate': 0.64, 'amount': Decimal('45000.00')},
				'LEGAL': {'count': 20, 'success_rate': 0.55, 'amount': Decimal('20000.00')}
			},
			'monthly_trend': [
				{'month': 'Jan 2025', 'collected': Decimal('45000.00'), 'activities': 52},
				{'month': 'Feb 2025', 'collected': Decimal('52000.00'), 'activities': 58},
				{'month': 'Mar 2025', 'collected': Decimal('48000.00'), 'activities': 55}
			],
			'top_collectors': [
				{'collector_id': uuid7str(), 'name': 'Jane Smith', 'success_rate': 0.82, 'amount': Decimal('85000.00')},
				{'collector_id': uuid7str(), 'name': 'Bob Johnson', 'success_rate': 0.76, 'amount': Decimal('72000.00')}
			]
		}
		
		with patch.object(analytics_service, '_validate_permissions', new_callable=AsyncMock):
			with patch.object(analytics_service, '_calculate_collection_performance', new_callable=AsyncMock) as mock_calc:
				mock_calc.return_value = mock_performance
				
				performance = await analytics_service.get_collection_performance_metrics()
				
				assert performance['total_collection_activities'] == 245
				assert performance['successful_activities'] == 168
				assert performance['success_rate'] == 0.686
				assert performance['total_amount_collected'] == Decimal('285000.00')
				assert len(performance['collection_by_method']) == 4
				assert performance['collection_by_method']['EMAIL']['success_rate'] == 0.72
				assert len(performance['monthly_trend']) == 3
				assert len(performance['top_collectors']) == 2


class TestServiceIntegration:
	"""Test service integration scenarios."""
	
	async def test_complete_ar_workflow(self):
		"""Test complete AR workflow across all services."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		
		# Initialize services
		customer_service = ARCustomerService(tenant_id, user_id)
		invoice_service = ARInvoiceService(tenant_id, user_id)
		cash_service = ARCashApplicationService(tenant_id, user_id)
		collections_service = ARCollectionsService(tenant_id, user_id)
		analytics_service = ARAnalyticsService(tenant_id, user_id)
		
		# Mock all service dependencies
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(cash_service, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(collections_service, '_validate_permissions', new_callable=AsyncMock), \
			 patch.object(analytics_service, '_validate_permissions', new_callable=AsyncMock):
			
			# 1. Create customer
			customer_data = {
				'customer_code': 'WORKFLOW001',
				'legal_name': 'Workflow Test Corp',
				'customer_type': ARCustomerType.CORPORATION,
				'status': ARCustomerStatus.ACTIVE,
				'credit_limit': Decimal('50000.00'),
				'payment_terms_days': 30
			}
			
			with patch('uuid_extensions.uuid7str', side_effect=['customer-id', 'invoice-id', 'payment-id']):
				customer = await customer_service.create_customer(customer_data)
				
				# 2. Create invoice
				invoice_data = {
					'customer_id': customer.id,
					'invoice_number': 'WF-INV-001',
					'invoice_date': date.today(),
					'due_date': date.today() + timedelta(days=30),
					'total_amount': Decimal('10000.00'),
					'description': 'Workflow test invoice'
				}
				
				with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock):
					invoice = await invoice_service.create_invoice(invoice_data)
				
				# 3. Create payment
				payment_data = {
					'customer_id': customer.id,
					'payment_reference': 'WF-PAY-001',
					'payment_date': date.today(),
					'payment_amount': Decimal('6000.00'),
					'payment_method': 'WIRE_TRANSFER'
				}
				
				with patch.object(cash_service, '_validate_customer_exists', new_callable=AsyncMock):
					payment = await cash_service.create_payment(payment_data)
				
				# Verify workflow consistency
				assert customer.tenant_id == tenant_id
				assert invoice.tenant_id == tenant_id
				assert payment.tenant_id == tenant_id
				
				assert invoice.customer_id == customer.id
				assert payment.customer_id == customer.id
				
				assert customer.created_by == user_id
				assert invoice.created_by == user_id
				assert payment.created_by == user_id
				
				# Verify business logic
				remaining_balance = invoice.total_amount - payment.payment_amount
				assert remaining_balance == Decimal('4000.00')
				assert remaining_balance <= customer.credit_limit


# Run tests
if __name__ == "__main__":
	pytest.main([__file__, "-v"])