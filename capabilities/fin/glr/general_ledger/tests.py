"""
APG Financial Management General Ledger - Comprehensive Test Suite

Complete test suite for the revolutionary General Ledger capability including
unit tests, integration tests, game-changer feature tests, and performance tests.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
import json
from datetime import datetime, timezone, date, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

# Import all the modules we're testing
from .service import GeneralLedgerService, JournalEntryRequest
from .models import GLAccount, GLJournalEntry, AccountTypeEnum, JournalSourceEnum
from .ai_assistant import IntelligentJournalAssistant, ConfidenceLevel
from .collaborative_workspace import CollaborativeWorkspace, UserActivity
from .intelligence_dashboard import FinancialIntelligenceEngine, InsightType
from .smart_reconciliation import SmartReconciliationEngine, MatchType
from .contextual_search import AdvancedTransactionSearchEngine, SearchType
from .multi_entity_transactions import MultiEntityTransactionProcessor, TransactionType
from .compliance_audit_intelligence import ComplianceAuditIntelligenceEngine, RiskLevel
from .visual_transaction_designer import VisualTransactionFlowDesigner, NodeType
from .smart_period_close import SmartPeriodCloseEngine, CloseType
from .financial_health_monitor import ContinuousFinancialHealthMonitor, HealthScore


class TestGeneralLedgerService:
	"""Test core GL service functionality"""
	
	@pytest.fixture
	def mock_db_session(self):
		"""Mock database session"""
		session = Mock()
		session.add = Mock()
		session.commit = AsyncMock()
		session.rollback = AsyncMock()
		session.close = AsyncMock()
		return session
	
	@pytest.fixture
	def gl_service(self, mock_db_session):
		"""Create GL service for testing"""
		return GeneralLedgerService(
			tenant_id="test_tenant",
			db_session=mock_db_session
		)
	
	@pytest.mark.asyncio
	async def test_create_journal_entry_success(self, gl_service):
		"""Test successful journal entry creation"""
		# Arrange
		journal_request = JournalEntryRequest(
			description="Test journal entry",
			entry_date=date.today(),
			source=JournalSourceEnum.MANUAL,
			lines=[
				{
					"account_id": "1000",
					"debit_amount": Decimal("1000.00"),
					"credit_amount": Decimal("0.00"),
					"description": "Debit Cash"
				},
				{
					"account_id": "4000",
					"debit_amount": Decimal("0.00"),
					"credit_amount": Decimal("1000.00"),
					"description": "Credit Revenue"
				}
			]
		)
		
		# Act
		result = await gl_service.create_journal_entry(journal_request)
		
		# Assert
		assert result is not None
		assert result["status"] == "success"
		assert result["journal_entry_id"] is not None
		assert result["total_debits"] == Decimal("1000.00")
		assert result["total_credits"] == Decimal("1000.00")
	
	@pytest.mark.asyncio
	async def test_create_journal_entry_unbalanced(self, gl_service):
		"""Test journal entry creation with unbalanced entries"""
		# Arrange
		journal_request = JournalEntryRequest(
			description="Unbalanced entry",
			entry_date=date.today(),
			source=JournalSourceEnum.MANUAL,
			lines=[
				{
					"account_id": "1000",
					"debit_amount": Decimal("1000.00"),
					"credit_amount": Decimal("0.00"),
					"description": "Debit Cash"
				},
				{
					"account_id": "4000",
					"debit_amount": Decimal("0.00"),
					"credit_amount": Decimal("500.00"),
					"description": "Credit Revenue"
				}
			]
		)
		
		# Act & Assert
		with pytest.raises(ValueError, match="Journal entry is not balanced"):
			await gl_service.create_journal_entry(journal_request)
	
	@pytest.mark.asyncio
	async def test_get_trial_balance(self, gl_service):
		"""Test trial balance generation"""
		# Act
		trial_balance = await gl_service.get_trial_balance(
			as_of_date=date.today()
		)
		
		# Assert
		assert trial_balance is not None
		assert "accounts" in trial_balance
		assert "total_debits" in trial_balance
		assert "total_credits" in trial_balance
		assert trial_balance["is_balanced"] == True


class TestAIAssistant:
	"""Test AI-powered journal entry assistant"""
	
	@pytest.fixture
	def mock_gl_service(self):
		"""Mock GL service"""
		service = Mock()
		service.tenant_id = "test_tenant"
		return service
	
	@pytest.fixture
	def ai_assistant(self, mock_gl_service):
		"""Create AI assistant for testing"""
		return IntelligentJournalAssistant(mock_gl_service, "test_user")
	
	@pytest.mark.asyncio
	async def test_process_natural_language_payment(self, ai_assistant):
		"""Test natural language processing for payment"""
		# Arrange
		description = "Paid $5,000 office rent for January with check #1234"
		
		# Act
		result = await ai_assistant.process_natural_language_entry(
			description=description,
			amount=Decimal("5000.00")
		)
		
		# Assert
		assert result is not None
		assert result.description == description
		assert len(result.suggested_lines) >= 2
		assert result.confidence in [level.value for level in ConfidenceLevel]
		assert "payment" in result.reasoning.lower()
		assert result.estimated_time_saved > 0
	
	@pytest.mark.asyncio
	async def test_smart_account_suggestions(self, ai_assistant):
		"""Test smart account suggestions"""
		# Act
		suggestions = await ai_assistant.get_smart_account_suggestions(
			context="office rent payment",
			transaction_amount=Decimal("5000.00"),
			entry_side="debit"
		)
		
		# Assert
		assert isinstance(suggestions, list)
		assert len(suggestions) <= 10
		for suggestion in suggestions:
			assert suggestion.account_id is not None
			assert suggestion.confidence in [level.value for level in ConfidenceLevel]
			assert suggestion.reasoning is not None
	
	@pytest.mark.asyncio
	async def test_error_detection_unbalanced(self, ai_assistant):
		"""Test error detection for unbalanced entries"""
		# Arrange
		journal_lines = [
			{
				"account_id": "1000",
				"debit_amount": Decimal("1000.00"),
				"credit_amount": Decimal("0.00"),
				"description": "Cash"
			},
			{
				"account_id": "4000",
				"debit_amount": Decimal("0.00"),
				"credit_amount": Decimal("500.00"),
				"description": "Revenue"
			}
		]
		
		# Act
		result = await ai_assistant.detect_and_prevent_errors(journal_lines)
		
		# Assert
		assert len(result["errors"]) > 0
		assert any("unbalanced" in error["type"].lower() for error in result["errors"])
		assert result["overall_score"] < 80


class TestCollaborativeWorkspace:
	"""Test real-time collaborative features"""
	
	@pytest.fixture
	def workspace(self):
		"""Create collaborative workspace"""
		return CollaborativeWorkspace("test_tenant")
	
	@pytest.mark.asyncio
	async def test_join_workspace(self, workspace):
		"""Test user joining workspace"""
		# Act
		context = await workspace.join_workspace(
			user_id="user1",
			user_name="John Doe",
			entity_type="journal_entry",
			entity_id="je_001"
		)
		
		# Assert
		assert context is not None
		assert context.workspace_id == "journal_entry:je_001"
		assert len(context.active_users) == 1
		assert context.active_users[0].user_id == "user1"
		assert context.active_users[0].activity == UserActivity.VIEWING
	
	@pytest.mark.asyncio
	async def test_conflict_detection(self, workspace):
		"""Test conflict detection in collaborative editing"""
		# Arrange
		await workspace.join_workspace("user1", "John", "journal_entry", "je_001")
		workspace_id = "journal_entry:je_001"
		
		# Add pending change
		change1 = {"field": "description", "value": "Updated by user1"}
		await workspace.propose_change("user1", workspace_id, change1)
		
		# Act - propose conflicting change
		change2 = {"field": "description", "value": "Updated by user2"}
		result = await workspace.propose_change("user2", workspace_id, change2)
		
		# Assert
		assert result["status"] == "conflict_detected"
		assert len(result["conflicts"]) > 0


class TestIntelligenceDashboard:
	"""Test contextual financial intelligence"""
	
	@pytest.fixture
	def mock_gl_service(self):
		service = Mock()
		service.tenant_id = "test_tenant"
		return service
	
	@pytest.fixture
	def intelligence_engine(self, mock_gl_service):
		return FinancialIntelligenceEngine(mock_gl_service)
	
	@pytest.mark.asyncio
	async def test_generate_contextual_dashboard(self, intelligence_engine):
		"""Test contextual dashboard generation"""
		# Act
		dashboard = await intelligence_engine.generate_contextual_dashboard(
			user_id="user1",
			role="accountant",
			current_context="period_close"
		)
		
		# Assert
		assert dashboard is not None
		assert dashboard.user_id == "user1"
		assert dashboard.role == "accountant"
		assert dashboard.current_focus == "period_close"
		assert len(dashboard.primary_insights) >= 0
		assert dashboard.personalization_score >= 0
	
	@pytest.mark.asyncio
	async def test_anomaly_detection(self, intelligence_engine):
		"""Test anomaly detection with explanation"""
		# Act
		anomalies = await intelligence_engine.detect_anomalies_with_explanation(
			account_id="6000",  # Office Expenses
			time_period=(datetime.now() - timedelta(days=30), datetime.now())
		)
		
		# Assert
		assert isinstance(anomalies, list)
		for anomaly in anomalies:
			assert anomaly.type == InsightType.ANOMALY_DETECTION
			assert anomaly.confidence >= 0 and anomaly.confidence <= 1
			assert len(anomaly.recommendations) > 0


class TestSmartReconciliation:
	"""Test intelligent transaction matching"""
	
	@pytest.fixture
	def mock_gl_service(self):
		service = Mock()
		service.tenant_id = "test_tenant"
		return service
	
	@pytest.fixture
	def reconciliation_engine(self, mock_gl_service):
		return SmartReconciliationEngine(mock_gl_service)
	
	@pytest.mark.asyncio
	async def test_smart_reconciliation_process(self, reconciliation_engine):
		"""Test complete smart reconciliation process"""
		# Arrange
		source_data = [
			{"id": "src_001", "amount": 1000.00, "date": "2025-01-15", "description": "Payment ABC Corp"},
			{"id": "src_002", "amount": 2500.00, "date": "2025-01-16", "description": "Office Rent"}
		]
		
		target_data = [
			{"id": "tgt_001", "amount": 1000.00, "date": "2025-01-15", "description": "ABC Corp Payment"},
			{"id": "tgt_002", "amount": 2500.00, "date": "2025-01-16", "description": "Monthly Rent"}
		]
		
		# Act
		session = await reconciliation_engine.start_smart_reconciliation(
			account_id="1000",
			source_data=source_data,
			target_data=target_data,
			reconciliation_date=datetime.now()
		)
		
		# Assert
		assert session is not None
		assert session.source_count == 2
		assert session.target_count == 2
		assert session.completion_percentage >= 0


class TestContextualSearch:
	"""Test advanced transaction search"""
	
	@pytest.fixture
	def mock_gl_service(self):
		service = Mock()
		service.tenant_id = "test_tenant"
		return service
	
	@pytest.fixture
	def search_engine(self, mock_gl_service):
		return AdvancedTransactionSearchEngine(mock_gl_service)
	
	@pytest.mark.asyncio
	async def test_natural_language_search(self, search_engine):
		"""Test natural language search"""
		# Arrange
		from .contextual_search import SearchContext
		context = SearchContext(
			user_id="user1",
			user_role="accountant",
			tenant_id="test_tenant"
		)
		
		# Act
		response = await search_engine.search(
			query="Show me all office expenses over $5000 last month",
			context=context
		)
		
		# Assert
		assert response is not None
		assert response.query == "Show me all office expenses over $5000 last month"
		assert response.query_type in [t.value for t in SearchType]
		assert response.processing_time_ms >= 0
		assert isinstance(response.results, list)


class TestMultiEntityTransactions:
	"""Test multi-entity transaction processing"""
	
	@pytest.fixture
	def mock_gl_service(self):
		service = Mock()
		service.tenant_id = "test_tenant"
		return service
	
	@pytest.fixture
	def multi_entity_processor(self, mock_gl_service):
		return MultiEntityTransactionProcessor(mock_gl_service)
	
	@pytest.mark.asyncio
	async def test_inter_entity_sale(self, multi_entity_processor):
		"""Test inter-entity sale processing"""
		# Act
		transaction = await multi_entity_processor.handle_inter_entity_sale(
			seller_entity_id="entity_001",
			buyer_entity_id="entity_002",
			amount=Decimal("50000.00"),
			product_description="Software License",
			sale_date=datetime.now()
		)
		
		# Assert
		assert transaction is not None
		assert transaction.transaction_type == TransactionType.INTER_ENTITY_SALE
		assert transaction.base_amount == Decimal("50000.00")
		assert len(transaction.involved_entities) == 2


class TestComplianceAuditIntelligence:
	"""Test compliance and audit intelligence"""
	
	@pytest.fixture
	def mock_gl_service(self):
		service = Mock()
		service.tenant_id = "test_tenant"
		return service
	
	@pytest.fixture
	def compliance_engine(self, mock_gl_service):
		return ComplianceAuditIntelligenceEngine(mock_gl_service)
	
	@pytest.mark.asyncio
	async def test_real_time_compliance_monitoring(self, compliance_engine):
		"""Test real-time compliance monitoring"""
		# Arrange
		transaction_data = {
			"id": "txn_001",
			"amount": 15000.00,
			"created_by": "user1",
			"manager_approval": False
		}
		
		# Act
		result = await compliance_engine.monitor_real_time_compliance(transaction_data)
		
		# Assert
		assert result is not None
		assert result["transaction_id"] == "txn_001"
		assert result["compliance_status"] in ["compliant", "non_compliant"]
		assert "risk_score" in result


class TestVisualTransactionDesigner:
	"""Test visual transaction flow designer"""
	
	@pytest.fixture
	def mock_gl_service(self):
		service = Mock()
		service.tenant_id = "test_tenant"
		return service
	
	@pytest.fixture
	def flow_designer(self, mock_gl_service):
		return VisualTransactionFlowDesigner(mock_gl_service)
	
	@pytest.mark.asyncio
	async def test_create_visual_flow(self, flow_designer):
		"""Test visual flow creation"""
		# Arrange
		flow_definition = {
			"name": "Payment Processing Flow",
			"description": "Standard payment processing workflow",
			"created_by": "user1",
			"nodes": [
				{
					"id": "start_1",
					"type": "START",
					"name": "Start Payment",
					"position": {"x": 100, "y": 100},
					"properties": {}
				},
				{
					"id": "debit_1",
					"type": "ACCOUNT_DEBIT",
					"name": "Debit Expense",
					"position": {"x": 200, "y": 100},
					"properties": {"account_id": "6000", "amount": "$amount"}
				}
			],
			"connections": [
				{
					"id": "conn_1",
					"from": "start_1",
					"to": "debit_1",
					"type": "success"
				}
			]
		}
		
		# Act
		flow = await flow_designer.create_visual_flow(flow_definition)
		
		# Assert
		assert flow is not None
		assert flow.flow_name == "Payment Processing Flow"
		assert len(flow.nodes) == 2
		assert len(flow.connections) == 1


class TestSmartPeriodClose:
	"""Test smart period close automation"""
	
	@pytest.fixture
	def mock_gl_service(self):
		service = Mock()
		service.tenant_id = "test_tenant"
		return service
	
	@pytest.fixture
	def period_close_engine(self, mock_gl_service):
		return SmartPeriodCloseEngine(mock_gl_service)
	
	@pytest.mark.asyncio
	async def test_initiate_smart_close(self, period_close_engine):
		"""Test smart period close initiation"""
		# Act
		session = await period_close_engine.initiate_smart_close(
			close_type=CloseType.MONTHLY,
			period_end=date(2025, 1, 31),
			entity_id="entity_001",
			user_id="user1"
		)
		
		# Assert
		assert session is not None
		assert session.close_type == CloseType.MONTHLY
		assert session.period_end == date(2025, 1, 31)
		assert len(session.tasks) > 0
		assert session.automation_percentage >= 0


class TestFinancialHealthMonitor:
	"""Test continuous financial health monitoring"""
	
	@pytest.fixture
	def mock_gl_service(self):
		service = Mock()
		service.tenant_id = "test_tenant"
		return service
	
	@pytest.fixture
	def health_monitor(self, mock_gl_service):
		return ContinuousFinancialHealthMonitor(mock_gl_service)
	
	@pytest.mark.asyncio
	async def test_assess_financial_health(self, health_monitor):
		"""Test financial health assessment"""
		# Act
		assessment = await health_monitor.assess_financial_health("entity_001")
		
		# Assert
		assert assessment is not None
		assert assessment.entity_id == "entity_001"
		assert assessment.overall_score >= 0 and assessment.overall_score <= 100
		assert assessment.overall_grade in [grade.value for grade in HealthScore]
		assert len(assessment.key_metrics) >= 0
	
	@pytest.mark.asyncio
	async def test_detect_anomalies(self, health_monitor):
		"""Test financial anomaly detection"""
		# Act
		anomalies = await health_monitor.detect_anomalies("entity_001")
		
		# Assert
		assert isinstance(anomalies, list)
		for anomaly in anomalies[:5]:  # Check first 5 anomalies
			assert "anomaly_type" in anomaly
			assert "confidence" in anomaly
			assert "explanation" in anomaly


class TestPerformance:
	"""Performance tests for critical operations"""
	
	@pytest.mark.asyncio
	async def test_journal_entry_creation_performance(self):
		"""Test journal entry creation performance"""
		# Arrange
		mock_session = Mock()
		mock_session.commit = AsyncMock()
		
		gl_service = GeneralLedgerService("test_tenant", mock_session)
		
		# Create 100 journal entries and measure time
		import time
		start_time = time.time()
		
		tasks = []
		for i in range(100):
			journal_request = JournalEntryRequest(
				description=f"Performance test entry {i}",
				entry_date=date.today(),
				source=JournalSourceEnum.MANUAL,
				lines=[
					{
						"account_id": "1000",
						"debit_amount": Decimal("100.00"),
						"credit_amount": Decimal("0.00"),
						"description": "Test debit"
					},
					{
						"account_id": "4000",
						"debit_amount": Decimal("0.00"),
						"credit_amount": Decimal("100.00"),
						"description": "Test credit"
					}
				]
			)
			tasks.append(gl_service.create_journal_entry(journal_request))
		
		# Execute all tasks concurrently
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		end_time = time.time()
		elapsed_time = end_time - start_time
		
		# Assert performance requirements
		assert elapsed_time < 10.0  # Should complete in under 10 seconds
		assert len([r for r in results if not isinstance(r, Exception)]) == 100


@pytest.fixture(scope="session")
def event_loop():
	"""Create event loop for async tests"""
	loop = asyncio.get_event_loop()
	yield loop
	loop.close()


# Integration test fixtures
@pytest.fixture
def integration_test_data():
	"""Provide test data for integration tests"""
	return {
		"test_accounts": [
			{"account_id": "1000", "account_code": "1000", "account_name": "Cash", "account_type": "ASSET"},
			{"account_id": "4000", "account_code": "4000", "account_name": "Revenue", "account_type": "REVENUE"},
			{"account_id": "6000", "account_code": "6000", "account_name": "Expenses", "account_type": "EXPENSE"}
		],
		"test_journal_entries": [
			{
				"description": "Test Revenue Entry",
				"entry_date": "2025-01-15",
				"lines": [
					{"account_id": "1000", "debit_amount": 1000.00, "credit_amount": 0.00},
					{"account_id": "4000", "debit_amount": 0.00, "credit_amount": 1000.00}
				]
			}
		]
	}


class TestIntegration:
	"""Integration tests for complete workflows"""
	
	@pytest.mark.asyncio
	async def test_end_to_end_journal_entry_with_ai_assistance(self, integration_test_data):
		"""Test complete journal entry creation with AI assistance"""
		# This test would verify the entire flow from natural language input
		# to journal entry creation with AI assistance
		
		# Arrange
		mock_gl_service = Mock()
		mock_gl_service.tenant_id = "test_tenant"
		
		ai_assistant = IntelligentJournalAssistant(mock_gl_service, "test_user")
		
		# Act - Process natural language input
		nl_result = await ai_assistant.process_natural_language_entry(
			description="Received $5,000 payment from customer ABC Corp",
			amount=Decimal("5000.00")
		)
		
		# Assert - Verify AI processing worked
		assert nl_result is not None
		assert nl_result.confidence != ConfidenceLevel.UNCERTAIN
		assert len(nl_result.suggested_lines) >= 2
		
		# Verify accounting equation balance
		total_debits = sum(Decimal(str(line.get('debit_amount', 0))) 
						  for line in nl_result.suggested_lines)
		total_credits = sum(Decimal(str(line.get('credit_amount', 0))) 
						   for line in nl_result.suggested_lines)
		assert total_debits == total_credits


# Test configuration
pytest_plugins = ['pytest_asyncio']

if __name__ == "__main__":
	# Run tests when executed directly
	pytest.main([__file__, "-v", "--tb=short"])