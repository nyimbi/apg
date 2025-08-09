"""
APG Core Financials - Analytics Service Unit Tests

CLAUDE.md compliant unit tests with APG AI integration validation,
federated learning, cash flow forecasting, and business intelligence.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from ...service import APAnalyticsService
from .conftest import (
	assert_valid_uuid, assert_decimal_equals, assert_apg_compliance
)


# Cash Flow Analytics Tests

async def test_generate_cash_flow_forecast(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test AI-powered cash flow forecasting"""
	# Setup
	historical_data = [
		{
			"date": "2025-01-01",
			"amount": "50000.00",
			"type": "payment",
			"vendor_category": "utilities"
		},
		{
			"date": "2025-01-02", 
			"amount": "25000.00",
			"type": "payment",
			"vendor_category": "supplies"
		}
	]
	
	pending_payments = [
		{
			"due_date": "2025-01-15",
			"amount": "15000.00",
			"vendor_id": "vendor_123",
			"priority": "high"
		}
	]
	
	# Execute
	forecast = await analytics_service.generate_cash_flow_forecast(
		historical_data=historical_data,
		pending_payments=pending_payments,
		forecast_horizon_days=30,
		tenant_context=tenant_context
	)
	
	# Verify
	assert forecast is not None, "Forecast should be generated"
	assert "daily_projections" in forecast, "Should have daily projections"
	assert "confidence_score" in forecast, "Should have confidence score"
	assert forecast["confidence_score"] >= 0.85, "Confidence should be >= 85%"
	assert "model_version" in forecast, "Should include model version"
	assert "feature_importance" in forecast, "Should include feature importance"
	
	# Verify forecast structure
	assert isinstance(forecast["daily_projections"], list), "Projections should be list"
	assert len(forecast["daily_projections"]) == 30, "Should have 30 days of projections"
	
	# Verify feature importance
	importance = forecast["feature_importance"]
	assert "seasonal_patterns" in importance, "Should analyze seasonal patterns"
	assert "vendor_payment_history" in importance, "Should analyze vendor history"


async def test_cash_flow_forecast_with_seasonality(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test cash flow forecasting with seasonal adjustments"""
	# Setup - historical data spanning multiple seasons
	historical_data = []
	for month in range(1, 13):  # Full year of data
		for day in range(1, 8):  # Weekly samples
			historical_data.append({
				"date": f"2024-{month:02d}-{day:02d}",
				"amount": str(30000 + (month * 1000)),  # Seasonal variation
				"type": "payment",
				"vendor_category": "operations"
			})
	
	# Execute
	forecast = await analytics_service.generate_cash_flow_forecast(
		historical_data=historical_data,
		pending_payments=[],
		forecast_horizon_days=90,  # Quarterly forecast
		tenant_context=tenant_context
	)
	
	# Verify seasonal analysis
	assert forecast["feature_importance"]["seasonal_patterns"] > 0.2, "Should detect seasonality"
	
	# Verify quarterly projections have variation
	daily_amounts = [proj["projected_amount"] for proj in forecast["daily_projections"]]
	amount_variance = max(daily_amounts) - min(daily_amounts)
	assert amount_variance > 5000, "Should show seasonal variation in projections"


# Spending Analytics Tests

async def test_generate_spending_analysis(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test comprehensive spending analysis"""
	# Setup
	spending_data = [
		{
			"vendor_id": "vendor_001",
			"vendor_name": "Office Supplies Co",
			"category": "office_supplies",
			"amount": "5000.00",
			"date": "2025-01-01"
		},
		{
			"vendor_id": "vendor_002", 
			"vendor_name": "Tech Services Inc",
			"category": "technology",
			"amount": "25000.00",
			"date": "2025-01-02"
		}
	]
	
	# Execute
	analysis = await analytics_service.generate_spending_analysis(
		spending_data=spending_data,
		analysis_period_days=30,
		tenant_context=tenant_context
	)
	
	# Verify
	assert analysis is not None, "Analysis should be generated"
	assert "category_breakdown" in analysis, "Should have category breakdown"
	assert "vendor_rankings" in analysis, "Should have vendor rankings"
	assert "spending_trends" in analysis, "Should have spending trends"
	assert "cost_optimization_opportunities" in analysis, "Should identify optimization"
	
	# Verify category analysis
	categories = analysis["category_breakdown"]
	assert len(categories) > 0, "Should have spending categories"
	tech_category = next((c for c in categories if c["category"] == "technology"), None)
	assert tech_category is not None, "Should include technology category"
	assert tech_category["total_amount"] == 25000.00, "Should calculate category totals"
	
	# Verify vendor rankings
	vendors = analysis["vendor_rankings"]
	assert len(vendors) > 0, "Should have vendor rankings"
	top_vendor = vendors[0]
	assert top_vendor["vendor_name"] == "Tech Services Inc", "Should rank by spending"


async def test_spending_trend_analysis(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test spending trend analysis over time"""
	# Setup - spending data with clear trend
	spending_data = []
	base_amount = 10000
	for week in range(12):  # 12 weeks of increasing spending
		spending_data.append({
			"vendor_id": f"vendor_{week}",
			"vendor_name": f"Vendor {week}",
			"category": "services",
			"amount": str(base_amount + (week * 1000)),  # Increasing trend
			"date": (date.today() - timedelta(days=week*7)).isoformat()
		})
	
	# Execute
	analysis = await analytics_service.generate_spending_analysis(
		spending_data=spending_data,
		analysis_period_days=90,
		tenant_context=tenant_context
	)
	
	# Verify trend detection
	trends = analysis["spending_trends"]
	assert "weekly_trend" in trends, "Should analyze weekly trends"
	assert trends["weekly_trend"]["direction"] == "increasing", "Should detect increasing trend"
	assert trends["weekly_trend"]["rate"] > 0, "Should quantify trend rate"


# Vendor Performance Analytics Tests

async def test_generate_vendor_performance_analysis(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test vendor performance analytics"""
	# Setup
	vendor_data = [
		{
			"vendor_id": "vendor_001",
			"vendor_name": "Reliable Vendor",
			"payment_history": [
				{"due_date": "2025-01-01", "paid_date": "2025-01-01", "amount": "1000.00"},
				{"due_date": "2025-01-15", "paid_date": "2025-01-14", "amount": "1500.00"}
			],
			"invoice_accuracy_score": 95,
			"delivery_performance_score": 98
		},
		{
			"vendor_id": "vendor_002",
			"vendor_name": "Problematic Vendor", 
			"payment_history": [
				{"due_date": "2025-01-01", "paid_date": "2025-01-10", "amount": "2000.00"},  # Late
				{"due_date": "2025-01-15", "paid_date": "2025-01-20", "amount": "1000.00"}   # Late
			],
			"invoice_accuracy_score": 75,
			"delivery_performance_score": 60
		}
	]
	
	# Execute
	analysis = await analytics_service.generate_vendor_performance_analysis(
		vendor_data=vendor_data,
		tenant_context=tenant_context
	)
	
	# Verify
	assert analysis is not None, "Analysis should be generated"
	assert "vendor_rankings" in analysis, "Should have vendor rankings"
	assert "performance_metrics" in analysis, "Should have performance metrics"
	assert "risk_assessment" in analysis, "Should assess vendor risk"
	
	# Verify vendor rankings
	rankings = analysis["vendor_rankings"]
	top_vendor = rankings[0]
	assert top_vendor["vendor_name"] == "Reliable Vendor", "Should rank by performance"
	assert top_vendor["overall_score"] > 90, "Top vendor should have high score"
	
	# Verify risk assessment
	risk_assessment = analysis["risk_assessment"]
	high_risk_vendors = risk_assessment["high_risk_vendors"]
	assert len(high_risk_vendors) > 0, "Should identify high-risk vendors"
	assert "Problematic Vendor" in [v["vendor_name"] for v in high_risk_vendors]


async def test_vendor_payment_pattern_analysis(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test vendor payment pattern analysis"""
	# Setup
	payment_patterns = [
		{
			"vendor_id": "vendor_seasonal",
			"payments": [
				{"month": 1, "amount": "10000.00"},
				{"month": 2, "amount": "12000.00"},
				{"month": 3, "amount": "8000.00"},
				{"month": 4, "amount": "15000.00"}  # Spike in Q1
			]
		}
	]
	
	# Execute
	analysis = await analytics_service.analyze_payment_patterns(
		payment_patterns=payment_patterns,
		tenant_context=tenant_context
	)
	
	# Verify
	assert analysis is not None, "Pattern analysis should be generated"
	assert "seasonal_patterns" in analysis, "Should detect seasonal patterns"
	assert "anomaly_detection" in analysis, "Should detect payment anomalies"
	assert "forecasted_patterns" in analysis, "Should forecast future patterns"


# Fraud Detection Analytics Tests

async def test_fraud_risk_analysis(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test AI-powered fraud risk analysis"""
	# Setup
	transaction_data = [
		{
			"transaction_id": "txn_001",
			"vendor_id": "vendor_001",
			"amount": "50000.00",  # High amount
			"payment_method": "wire",
			"created_at": datetime.now().isoformat(),
			"approver_id": "approver_001",
			"submission_time": "23:45"  # Unusual time
		},
		{
			"transaction_id": "txn_002",
			"vendor_id": "vendor_new",  # New vendor
			"amount": "100000.00",     # Very high amount
			"payment_method": "ach",
			"created_at": datetime.now().isoformat(),
			"bank_account_changed": True  # Risk factor
		}
	]
	
	# Execute
	analysis = await analytics_service.analyze_fraud_risk(
		transaction_data=transaction_data,
		tenant_context=tenant_context
	)
	
	# Verify
	assert analysis is not None, "Fraud analysis should be generated"
	assert "risk_scores" in analysis, "Should have risk scores"
	assert "high_risk_transactions" in analysis, "Should identify high-risk transactions"
	assert "risk_factors" in analysis, "Should identify specific risk factors"
	
	# Verify risk scoring
	risk_scores = analysis["risk_scores"]
	high_risk_txn = next((t for t in risk_scores if t["transaction_id"] == "txn_002"), None)
	assert high_risk_txn is not None, "Should identify high-risk transaction"
	assert high_risk_txn["risk_score"] > 0.8, "High-risk transaction should have high score"
	
	# Verify risk factors
	risk_factors = analysis["risk_factors"]
	assert "new_vendor" in risk_factors, "Should identify new vendor risk"
	assert "high_amount" in risk_factors, "Should identify high amount risk"


async def test_duplicate_payment_detection(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test duplicate payment detection using ML"""
	# Setup
	payment_data = [
		{
			"payment_id": "pay_001",
			"vendor_id": "vendor_123",
			"amount": "1500.00",
			"invoice_number": "INV-001",
			"payment_date": "2025-01-01"
		},
		{
			"payment_id": "pay_002", 
			"vendor_id": "vendor_123",
			"amount": "1500.00",  # Same amount
			"invoice_number": "INV-001",  # Same invoice
			"payment_date": "2025-01-02"  # Next day - suspicious
		}
	]
	
	# Execute
	analysis = await analytics_service.detect_duplicate_payments(
		payment_data=payment_data,
		tenant_context=tenant_context
	)
	
	# Verify
	assert analysis is not None, "Duplicate analysis should be generated"
	assert "potential_duplicates" in analysis, "Should identify potential duplicates"
	assert "similarity_scores" in analysis, "Should calculate similarity scores"
	
	# Verify duplicate detection
	duplicates = analysis["potential_duplicates"]
	assert len(duplicates) > 0, "Should identify potential duplicate pair"
	duplicate_pair = duplicates[0]
	assert duplicate_pair["similarity_score"] > 0.9, "Should have high similarity score"


# Performance Analytics Tests

async def test_ap_process_performance_analysis(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test accounts payable process performance analysis"""
	# Setup
	process_data = [
		{
			"invoice_id": "inv_001",
			"received_date": "2025-01-01T09:00:00",
			"processed_date": "2025-01-01T11:30:00",  # 2.5 hours
			"approved_date": "2025-01-02T14:00:00",   # 1 day 5 hours
			"paid_date": "2025-01-15T16:00:00",       # Net 30 terms
			"process_stage_times": {
				"ocr_processing": 300,      # 5 minutes
				"validation": 1800,         # 30 minutes
				"approval_workflow": 86400, # 1 day
				"payment_processing": 600   # 10 minutes
			}
		}
	]
	
	# Execute
	analysis = await analytics_service.analyze_process_performance(
		process_data=process_data,
		tenant_context=tenant_context
	)
	
	# Verify
	assert analysis is not None, "Performance analysis should be generated"
	assert "average_processing_times" in analysis, "Should have average times"
	assert "bottleneck_analysis" in analysis, "Should identify bottlenecks"
	assert "efficiency_metrics" in analysis, "Should calculate efficiency"
	assert "improvement_recommendations" in analysis, "Should suggest improvements"
	
	# Verify bottleneck identification
	bottlenecks = analysis["bottleneck_analysis"]
	assert "approval_workflow" in bottlenecks, "Should identify approval as bottleneck"
	assert bottlenecks["approval_workflow"]["avg_time"] > 80000, "Should measure long approval times"


async def test_sla_compliance_analysis(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test SLA compliance analysis and reporting"""
	# Setup
	sla_data = [
		{
			"metric": "invoice_processing_time",
			"target": 120,  # 2 hours
			"actual_times": [90, 150, 180, 60, 240],  # Mixed performance
			"sla_threshold": 120
		},
		{
			"metric": "payment_processing_time", 
			"target": 1440,  # 24 hours
			"actual_times": [800, 1200, 1600, 900, 1100],  # Good performance
			"sla_threshold": 1440
		}
	]
	
	# Execute
	analysis = await analytics_service.analyze_sla_compliance(
		sla_data=sla_data,
		tenant_context=tenant_context
	)
	
	# Verify
	assert analysis is not None, "SLA analysis should be generated"
	assert "compliance_rates" in analysis, "Should calculate compliance rates"
	assert "sla_violations" in analysis, "Should identify violations"
	assert "trend_analysis" in analysis, "Should analyze compliance trends"
	
	# Verify compliance calculation
	compliance_rates = analysis["compliance_rates"]
	invoice_compliance = next((c for c in compliance_rates if c["metric"] == "invoice_processing_time"), None)
	assert invoice_compliance is not None, "Should have invoice processing compliance"
	assert invoice_compliance["compliance_rate"] == 0.6, "Should calculate 60% compliance (3/5)"


# Predictive Analytics Tests

async def test_payment_default_prediction(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test payment default risk prediction using ML"""
	# Setup
	vendor_financial_data = [
		{
			"vendor_id": "vendor_stable",
			"credit_score": 750,
			"debt_to_equity": 0.3,
			"current_ratio": 2.1,
			"payment_history_score": 95,
			"industry_risk_factor": "low"
		},
		{
			"vendor_id": "vendor_risky",
			"credit_score": 580,  # Low credit
			"debt_to_equity": 1.8,  # High debt
			"current_ratio": 0.8,   # Poor liquidity
			"payment_history_score": 65,
			"industry_risk_factor": "high"
		}
	]
	
	# Execute
	prediction = await analytics_service.predict_payment_defaults(
		vendor_financial_data=vendor_financial_data,
		prediction_horizon_days=90,
		tenant_context=tenant_context
	)
	
	# Verify
	assert prediction is not None, "Prediction should be generated"
	assert "vendor_risk_scores" in prediction, "Should have vendor risk scores"
	assert "high_risk_vendors" in prediction, "Should identify high-risk vendors"
	assert "model_confidence" in prediction, "Should include model confidence"
	
	# Verify risk scoring
	risk_scores = prediction["vendor_risk_scores"]
	risky_vendor = next((v for v in risk_scores if v["vendor_id"] == "vendor_risky"), None)
	stable_vendor = next((v for v in risk_scores if v["vendor_id"] == "vendor_stable"), None)
	
	assert risky_vendor["default_probability"] > stable_vendor["default_probability"], "Should rank risk correctly"
	assert risky_vendor["default_probability"] > 0.5, "High-risk vendor should have high probability"


# Integration Tests with APG AI Capabilities

async def test_analytics_federated_learning_integration(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test integration with APG federated learning service"""
	# Verify federated learning integration exists
	assert hasattr(analytics_service, 'federated_learning_service'), "Should have federated learning integration"
	
	# Verify required ML methods exist
	assert hasattr(analytics_service.federated_learning_service, 'train_model'), "Should have model training"
	assert hasattr(analytics_service.federated_learning_service, 'predict'), "Should have prediction capability"
	assert hasattr(analytics_service.federated_learning_service, 'get_model_metrics'), "Should have model metrics"


async def test_analytics_ai_orchestration_integration(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test integration with APG AI orchestration service"""
	# Verify AI orchestration integration exists  
	assert hasattr(analytics_service, 'ai_orchestration_service'), "Should have AI orchestration integration"
	
	# Verify required AI methods exist
	assert hasattr(analytics_service.ai_orchestration_service, 'analyze_patterns'), "Should have pattern analysis"
	assert hasattr(analytics_service.ai_orchestration_service, 'generate_insights'), "Should have insight generation"


# Performance Tests

async def test_analytics_processing_performance(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test analytics processing performance under load"""
	import time
	
	# Setup large dataset
	large_dataset = []
	for i in range(1000):  # 1000 data points
		large_dataset.append({
			"transaction_id": f"txn_{i:04d}",
			"amount": str(100 + (i % 500)),
			"date": (date.today() - timedelta(days=i % 365)).isoformat(),
			"vendor_id": f"vendor_{i % 50}"  # 50 different vendors
		})
	
	# Execute performance test
	start_time = time.time()
	
	analysis = await analytics_service.generate_spending_analysis(
		spending_data=large_dataset,
		analysis_period_days=365,
		tenant_context=tenant_context
	)
	
	end_time = time.time()
	duration = end_time - start_time
	
	# Verify performance
	assert duration < 10.0, f"Large dataset analysis took {duration:.2f}s, should be < 10s"
	assert analysis is not None, "Should handle large datasets successfully"


# Error Handling Tests

async def test_analytics_service_error_handling(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test analytics service error handling and recovery"""
	# Test with None data
	with pytest.raises(AssertionError):
		await analytics_service.generate_cash_flow_forecast(
			None, [], 30, tenant_context
		)
	
	# Test with None context
	with pytest.raises(AssertionError):
		await analytics_service.generate_spending_analysis([], 30, None)
	
	# Test with invalid forecast horizon
	with pytest.raises(ValueError):
		await analytics_service.generate_cash_flow_forecast(
			[], [], -1, tenant_context  # Negative horizon
		)


# Model Performance Tests

async def test_analytics_model_accuracy_validation(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test analytics model accuracy and validation"""
	# Setup test data with known outcomes
	historical_payments = [
		{"amount": "1000.00", "date": "2024-12-01", "category": "utilities"},
		{"amount": "1500.00", "date": "2024-12-02", "category": "supplies"},
		{"amount": "2000.00", "date": "2024-12-03", "category": "services"}
	]
	
	# Execute forecast
	forecast = await analytics_service.generate_cash_flow_forecast(
		historical_data=historical_payments,
		pending_payments=[],
		forecast_horizon_days=7,
		tenant_context=tenant_context
	)
	
	# Verify model performance metrics
	assert forecast["confidence_score"] >= 0.85, "Model confidence should be >= 85%"
	
	# Verify model versioning and tracking
	assert "model_version" in forecast, "Should track model version"
	assert forecast["model_version"].startswith("cash_flow_v"), "Should have semantic versioning"


# Export test functions for discovery
__all__ = [
	"test_generate_cash_flow_forecast",
	"test_cash_flow_forecast_with_seasonality",
	"test_generate_spending_analysis",
	"test_spending_trend_analysis",
	"test_generate_vendor_performance_analysis",
	"test_vendor_payment_pattern_analysis",
	"test_fraud_risk_analysis",
	"test_duplicate_payment_detection",
	"test_ap_process_performance_analysis",
	"test_sla_compliance_analysis",
	"test_payment_default_prediction",
	"test_analytics_federated_learning_integration",
	"test_analytics_ai_orchestration_integration",
	"test_analytics_processing_performance",
	"test_analytics_model_accuracy_validation"
]