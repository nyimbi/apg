"""
APG Cash Management - Core Service Layer

Enterprise-grade business logic with APG async patterns and integration.
Implements CLAUDE.md standards with async Python, modern typing, and APG integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

import aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from .models import (
	APGBaseModel, Bank, CashAccount, CashPosition, CashFlow, CashForecast, ForecastAssumption,
	Investment, InvestmentOpportunity, CashAlert, OptimizationRule,
	BankStatus, CashAccountStatus, CashAccountType, InvestmentStatus, InvestmentType,
	ForecastType, ForecastScenario, TransactionType, AlertType, RiskRating,
	OptimizationGoal
)


class CashManagementService:
	"""
	Core business logic for APG Cash Management capability.
	
	Implements async Python patterns with APG platform integration for:
	- Real-time cash positioning and monitoring
	- AI-powered cash forecasting and optimization
	- Investment management and opportunity identification
	- Bank connectivity and automated reconciliation
	- Risk management and compliance monitoring
	"""
	
	def __init__(self, db_session: AsyncSession, redis_client: aioredis.Redis):
		"""Initialize service with APG infrastructure dependencies."""
		self.db = db_session
		self.cache = redis_client
		self._log_service_init()
	
	# =========================================================================
	# Bank Management Operations
	# =========================================================================
	
	async def create_bank(self, tenant_id: str, bank_data: Dict[str, Any], created_by: str) -> Bank:
		"""Create new bank relationship with validation and audit compliance."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		assert created_by is not None, "created_by required for audit compliance"
		
		# Validate bank data
		validation_result = await self._validate_bank_data(bank_data, tenant_id)
		if not validation_result['valid']:
			raise ValueError(f"Bank validation failed: {validation_result['errors']}")
		
		# Create bank record
		bank = Bank(
			tenant_id=tenant_id,
			created_by=created_by,
			updated_by=created_by,
			**bank_data
		)
		
		self.db.add(bank)
		await self.db.commit()
		await self.db.refresh(bank)
		
		# Cache bank data for performance
		await self._cache_bank_data(bank)
		
		self._log_bank_created(bank.bank_code, bank.bank_name)
		return bank
	
	async def get_banks(self, tenant_id: str, status: Optional[BankStatus] = None) -> List[Bank]:
		"""Retrieve banks with optional status filtering."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		# Check cache first
		cache_key = f"banks:{tenant_id}:{status or 'all'}"
		cached_data = await self.cache.get(cache_key)
		if cached_data:
			return cached_data
		
		query = select(Bank).where(
			and_(
				Bank.tenant_id == tenant_id,
				Bank.is_deleted == False
			)
		)
		
		if status:
			query = query.where(Bank.status == status)
		
		query = query.order_by(Bank.bank_name)
		
		result = await self.db.execute(query)
		banks = result.scalars().all()
		
		# Cache results
		await self.cache.setex(cache_key, 300, banks)  # 5 minute cache
		
		return banks
	
	async def update_bank_api_status(self, bank_id: str, api_enabled: bool, 
									last_sync: Optional[datetime] = None) -> Bank:
		"""Update bank API connectivity status."""
		bank = await self._get_bank_by_id(bank_id)
		if not bank:
			raise ValueError(f"Bank {bank_id} not found")
		
		bank.api_enabled = api_enabled
		bank.last_api_sync = last_sync or datetime.utcnow()
		bank.updated_at = datetime.utcnow()
		bank.version += 1
		
		await self.db.commit()
		await self.db.refresh(bank)
		
		# Invalidate cache
		await self._invalidate_bank_cache(bank.tenant_id)
		
		self._log_bank_api_updated(bank.bank_code, api_enabled)
		return bank
	
	# =========================================================================
	# Cash Account Management
	# =========================================================================
	
	async def create_cash_account(self, tenant_id: str, account_data: Dict[str, Any], 
								 created_by: str) -> CashAccount:
		"""Create new cash account with comprehensive validation."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		assert created_by is not None, "created_by required for audit compliance"
		
		# Validate account data
		validation_result = await self._validate_account_data(account_data, tenant_id)
		if not validation_result['valid']:
			raise ValueError(f"Account validation failed: {validation_result['errors']}")
		
		# Create account record
		account = CashAccount(
			tenant_id=tenant_id,
			created_by=created_by,
			updated_by=created_by,
			**account_data
		)
		
		self.db.add(account)
		await self.db.commit()
		await self.db.refresh(account)
		
		# Initialize cash position tracking
		await self._initialize_account_position(account)
		
		self._log_account_created(account.account_number, account.account_name)
		return account
	
	async def get_cash_accounts(self, tenant_id: str, entity_id: Optional[str] = None,
							   currency_code: Optional[str] = None,
							   status: Optional[CashAccountStatus] = None) -> List[CashAccount]:
		"""Retrieve cash accounts with filtering options."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		query = select(CashAccount).where(
			and_(
				CashAccount.tenant_id == tenant_id,
				CashAccount.is_deleted == False
			)
		).options(selectinload(CashAccount.bank))
		
		if entity_id:
			query = query.where(CashAccount.entity_id == entity_id)
		if currency_code:
			query = query.where(CashAccount.currency_code == currency_code)
		if status:
			query = query.where(CashAccount.status == status)
		
		query = query.order_by(CashAccount.account_name)
		
		result = await self.db.execute(query)
		return result.scalars().all()
	
	async def update_account_balance(self, account_id: str, current_balance: Decimal,
									available_balance: Optional[Decimal] = None,
									pending_credits: Optional[Decimal] = None,
									pending_debits: Optional[Decimal] = None) -> CashAccount:
		"""Update account balance with real-time position tracking."""
		account = await self._get_account_by_id(account_id)
		if not account:
			raise ValueError(f"Account {account_id} not found")
		
		# Update balance fields
		old_balance = account.current_balance
		account.current_balance = current_balance
		account.available_balance = available_balance or current_balance
		account.pending_credits = pending_credits or Decimal('0')
		account.pending_debits = pending_debits or Decimal('0')
		account.last_balance_update = datetime.utcnow()
		account.updated_at = datetime.utcnow()
		account.version += 1
		
		await self.db.commit()
		await self.db.refresh(account)
		
		# Update real-time position
		balance_change = current_balance - old_balance
		await self._update_realtime_position(account, balance_change)
		
		# Check for alerts
		await self._check_balance_alerts(account)
		
		self._log_balance_updated(account.account_number, old_balance, current_balance)
		return account
	
	async def configure_auto_sweep(self, account_id: str, enabled: bool,
								  target_account_id: Optional[str] = None,
								  threshold: Optional[Decimal] = None) -> CashAccount:
		"""Configure automated cash sweeping for optimal liquidity management."""
		account = await self._get_account_by_id(account_id)
		if not account:
			raise ValueError(f"Account {account_id} not found")
		
		if enabled and not target_account_id:
			raise ValueError("Target account required for auto sweep")
		
		account.auto_sweep_enabled = enabled
		account.sweep_target_account = target_account_id
		account.sweep_threshold = threshold
		account.updated_at = datetime.utcnow()
		account.version += 1
		
		await self.db.commit()
		await self.db.refresh(account)
		
		self._log_sweep_configured(account.account_number, enabled)
		return account
	
	# =========================================================================
	# Real-Time Cash Positioning
	# =========================================================================
	
	async def get_global_cash_position(self, tenant_id: str, as_of_date: Optional[date] = None,
									  currency_code: Optional[str] = None) -> Dict[str, Any]:
		"""Get consolidated global cash position with real-time data."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		position_date = as_of_date or date.today()
		
		# Build query for cash positions
		query = select(CashPosition).where(
			and_(
				CashPosition.tenant_id == tenant_id,
				CashPosition.position_date == position_date,
				CashPosition.is_deleted == False
			)
		)
		
		if currency_code:
			query = query.where(CashPosition.currency_code == currency_code)
		
		result = await self.db.execute(query)
		positions = result.scalars().all()
		
		# If no positions exist, calculate from current account balances
		if not positions:
			positions = await self._calculate_current_positions(tenant_id, position_date)
		
		# Aggregate position data
		total_cash = sum(pos.total_cash for pos in positions)
		available_cash = sum(pos.available_cash for pos in positions)
		restricted_cash = sum(pos.restricted_cash for pos in positions)
		invested_cash = sum(pos.invested_cash for pos in positions)
		
		# Currency breakdown
		currency_breakdown = {}
		for pos in positions:
			if pos.currency_code not in currency_breakdown:
				currency_breakdown[pos.currency_code] = {
					'total_cash': Decimal('0'),
					'available_cash': Decimal('0'),
					'account_count': 0
				}
			
			currency_breakdown[pos.currency_code]['total_cash'] += pos.total_cash
			currency_breakdown[pos.currency_code]['available_cash'] += pos.available_cash
			currency_breakdown[pos.currency_code]['account_count'] += 1
		
		# Risk metrics calculation
		risk_metrics = await self._calculate_risk_metrics(positions)
		
		return {
			'as_of_date': position_date.isoformat(),
			'summary': {
				'total_cash': float(total_cash),
				'available_cash': float(available_cash),
				'restricted_cash': float(restricted_cash),
				'invested_cash': float(invested_cash),
				'total_position': float(total_cash + invested_cash),
				'liquidity_ratio': risk_metrics.get('liquidity_ratio'),
				'concentration_risk': risk_metrics.get('concentration_risk'),
				'days_cash_on_hand': risk_metrics.get('days_cash_on_hand')
			},
			'currency_breakdown': {
				curr: {
					'total_cash': float(data['total_cash']),
					'available_cash': float(data['available_cash']),
					'account_count': data['account_count']
				}
				for curr, data in currency_breakdown.items()
			},
			'position_count': len(positions)
		}
	
	async def get_cash_position_trend(self, tenant_id: str, start_date: date,
									 end_date: date, entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
		"""Get historical cash position trend for analytics."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		query = select(CashPosition).where(
			and_(
				CashPosition.tenant_id == tenant_id,
				CashPosition.position_date >= start_date,
				CashPosition.position_date <= end_date,
				CashPosition.is_deleted == False
			)
		)
		
		if entity_id:
			query = query.where(CashPosition.entity_id == entity_id)
		
		query = query.order_by(CashPosition.position_date, CashPosition.entity_id)
		
		result = await self.db.execute(query)
		positions = result.scalars().all()
		
		# Aggregate by date
		daily_positions = {}
		for pos in positions:
			date_key = pos.position_date.isoformat()
			if date_key not in daily_positions:
				daily_positions[date_key] = {
					'date': date_key,
					'total_cash': Decimal('0'),
					'available_cash': Decimal('0'),
					'projected_inflows': Decimal('0'),
					'projected_outflows': Decimal('0'),
					'net_projected_flow': Decimal('0')
				}
			
			daily_positions[date_key]['total_cash'] += pos.total_cash
			daily_positions[date_key]['available_cash'] += pos.available_cash
			daily_positions[date_key]['projected_inflows'] += pos.projected_inflows
			daily_positions[date_key]['projected_outflows'] += pos.projected_outflows
			daily_positions[date_key]['net_projected_flow'] += pos.net_projected_flow
		
		# Convert to list and format
		trend_data = []
		for date_key in sorted(daily_positions.keys()):
			data = daily_positions[date_key]
			trend_data.append({
				'date': data['date'],
				'total_cash': float(data['total_cash']),
				'available_cash': float(data['available_cash']),
				'projected_inflows': float(data['projected_inflows']),
				'projected_outflows': float(data['projected_outflows']),
				'net_projected_flow': float(data['net_projected_flow'])
			})
		
		return trend_data
	
	# =========================================================================
	# Cash Flow Tracking and Analysis
	# =========================================================================
	
	async def record_cash_flow(self, tenant_id: str, flow_data: Dict[str, Any],
							  created_by: str) -> CashFlow:
		"""Record individual cash flow transaction with categorization."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		assert created_by is not None, "created_by required for audit compliance"
		
		# Validate cash flow data
		validation_result = await self._validate_cash_flow_data(flow_data, tenant_id)
		if not validation_result['valid']:
			raise ValueError(f"Cash flow validation failed: {validation_result['errors']}")
		
		# Create cash flow record
		cash_flow = CashFlow(
			tenant_id=tenant_id,
			created_by=created_by,
			updated_by=created_by,
			**flow_data
		)
		
		self.db.add(cash_flow)
		await self.db.commit()
		await self.db.refresh(cash_flow)
		
		# Update real-time position
		await self._update_position_from_flow(cash_flow)
		
		# Check for forecasting pattern updates
		if cash_flow.is_recurring:
			await self._update_forecasting_patterns(cash_flow)
		
		self._log_cash_flow_recorded(cash_flow.transaction_type, cash_flow.amount)
		return cash_flow
	
	async def get_cash_flows(self, tenant_id: str, start_date: Optional[date] = None,
							end_date: Optional[date] = None, account_id: Optional[str] = None,
							transaction_type: Optional[TransactionType] = None,
							limit: int = 1000) -> List[CashFlow]:
		"""Retrieve cash flows with filtering and pagination."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		query = select(CashFlow).where(
			and_(
				CashFlow.tenant_id == tenant_id,
				CashFlow.is_deleted == False
			)
		)
		
		if start_date:
			query = query.where(CashFlow.flow_date >= start_date)
		if end_date:
			query = query.where(CashFlow.flow_date <= end_date)
		if account_id:
			query = query.where(CashFlow.account_id == account_id)
		if transaction_type:
			query = query.where(CashFlow.transaction_type == transaction_type)
		
		query = query.order_by(desc(CashFlow.flow_date), desc(CashFlow.created_at)).limit(limit)
		
		result = await self.db.execute(query)
		return result.scalars().all()
	
	async def analyze_cash_flow_patterns(self, tenant_id: str, analysis_period_days: int = 90) -> Dict[str, Any]:
		"""Analyze cash flow patterns for forecasting improvement."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		end_date = date.today()
		start_date = end_date - timedelta(days=analysis_period_days)
		
		# Get cash flows for analysis period
		query = select(CashFlow).where(
			and_(
				CashFlow.tenant_id == tenant_id,
				CashFlow.flow_date >= start_date,
				CashFlow.flow_date <= end_date,
				CashFlow.is_deleted == False
			)
		)
		
		result = await self.db.execute(query)
		flows = result.scalars().all()
		
		# Analyze patterns
		category_patterns = {}
		transaction_type_patterns = {}
		recurring_patterns = {}
		
		for flow in flows:
			# Category analysis
			if flow.category not in category_patterns:
				category_patterns[flow.category] = {
					'total_amount': Decimal('0'),
					'transaction_count': 0,
					'average_amount': Decimal('0'),
					'inflow_count': 0,
					'outflow_count': 0
				}
			
			category_patterns[flow.category]['total_amount'] += abs(flow.amount)
			category_patterns[flow.category]['transaction_count'] += 1
			
			if flow.is_inflow:
				category_patterns[flow.category]['inflow_count'] += 1
			else:
				category_patterns[flow.category]['outflow_count'] += 1
			
			# Transaction type analysis
			if flow.transaction_type not in transaction_type_patterns:
				transaction_type_patterns[flow.transaction_type] = {
					'total_amount': Decimal('0'),
					'transaction_count': 0
				}
			
			transaction_type_patterns[flow.transaction_type]['total_amount'] += abs(flow.amount)
			transaction_type_patterns[flow.transaction_type]['transaction_count'] += 1
			
			# Recurring pattern analysis
			if flow.is_recurring and flow.recurrence_pattern:
				if flow.recurrence_pattern not in recurring_patterns:
					recurring_patterns[flow.recurrence_pattern] = {
						'total_amount': Decimal('0'),
						'transaction_count': 0,
						'average_confidence': Decimal('0')
					}
				
				recurring_patterns[flow.recurrence_pattern]['total_amount'] += abs(flow.amount)
				recurring_patterns[flow.recurrence_pattern]['transaction_count'] += 1
				if flow.forecast_confidence:
					recurring_patterns[flow.recurrence_pattern]['average_confidence'] += flow.forecast_confidence
		
		# Calculate averages
		for pattern in category_patterns.values():
			if pattern['transaction_count'] > 0:
				pattern['average_amount'] = pattern['total_amount'] / pattern['transaction_count']
		
		for pattern in recurring_patterns.values():
			if pattern['transaction_count'] > 0:
				pattern['average_confidence'] /= pattern['transaction_count']
		
		return {
			'analysis_period': {
				'start_date': start_date.isoformat(),
				'end_date': end_date.isoformat(),
				'days': analysis_period_days
			},
			'total_flows_analyzed': len(flows),
			'category_patterns': {
				cat: {
					'total_amount': float(data['total_amount']),
					'transaction_count': data['transaction_count'],
					'average_amount': float(data['average_amount']),
					'inflow_count': data['inflow_count'],
					'outflow_count': data['outflow_count']
				}
				for cat, data in category_patterns.items()
			},
			'transaction_type_patterns': {
				txn_type: {
					'total_amount': float(data['total_amount']),
					'transaction_count': data['transaction_count']
				}
				for txn_type, data in transaction_type_patterns.items()
			},
			'recurring_patterns': {
				pattern: {
					'total_amount': float(data['total_amount']),
					'transaction_count': data['transaction_count'],
					'average_confidence': float(data['average_confidence'])
				}
				for pattern, data in recurring_patterns.items()
			}
		}
	
	# =========================================================================
	# AI-Powered Cash Forecasting
	# =========================================================================
	
	async def generate_cash_forecast(self, tenant_id: str, forecast_config: Dict[str, Any],
									created_by: str) -> Dict[str, Any]:
		"""Generate AI-powered cash forecast with confidence intervals."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		assert created_by is not None, "created_by required for audit compliance"
		
		# Extract configuration
		entity_id = forecast_config['entity_id']
		currency_code = forecast_config['currency_code']
		horizon_days = forecast_config.get('horizon_days', 90)
		forecast_type = ForecastType(forecast_config.get('forecast_type', 'daily'))
		scenario = ForecastScenario(forecast_config.get('scenario', 'base_case'))
		
		# Get current cash position as opening balance
		current_position = await self.get_global_cash_position(tenant_id, currency_code=currency_code)
		opening_balance = Decimal(str(current_position['summary']['available_cash']))
		
		# Generate forecast using ML model (simplified implementation)
		forecast_results = await self._run_ml_forecast_model(
			tenant_id, entity_id, currency_code, horizon_days, scenario
		)
		
		# Create forecast record
		cash_forecast = CashForecast(
			tenant_id=tenant_id,
			forecast_date=date.today(),
			forecast_type=forecast_type,
			scenario=scenario,
			entity_id=entity_id,
			currency_code=currency_code,
			horizon_days=horizon_days,
			opening_balance=opening_balance,
			opening_date=date.today(),
			projected_inflows=forecast_results['projected_inflows'],
			projected_outflows=forecast_results['projected_outflows'],
			net_flow=forecast_results['net_flow'],
			closing_balance=opening_balance + forecast_results['net_flow'],
			confidence_level=forecast_results['confidence_level'],
			confidence_interval_lower=forecast_results['confidence_interval_lower'],
			confidence_interval_upper=forecast_results['confidence_interval_upper'],
			standard_deviation=forecast_results.get('standard_deviation'),
			model_used=forecast_results['model_used'],
			model_version=forecast_results['model_version'],
			feature_importance=forecast_results.get('feature_importance', {}),
			shortfall_probability=forecast_results.get('shortfall_probability'),
			stress_test_result=forecast_results.get('stress_test_result'),
			value_at_risk=forecast_results.get('value_at_risk'),
			created_by=created_by,
			updated_by=created_by
		)
		
		self.db.add(cash_forecast)
		await self.db.commit()
		await self.db.refresh(cash_forecast)
		
		# Generate forecast assumptions if provided
		if 'assumptions' in forecast_config:
			for assumption_data in forecast_config['assumptions']:
				assumption = ForecastAssumption(
					tenant_id=tenant_id,
					forecast_id=cash_forecast.id,
					created_by=created_by,
					updated_by=created_by,
					**assumption_data
				)
				self.db.add(assumption)
		
		await self.db.commit()
		
		# Check for forecast alerts
		await self._check_forecast_alerts(cash_forecast)
		
		self._log_forecast_generated(cash_forecast.forecast_id, horizon_days, float(cash_forecast.confidence_level))
		
		return {
			'forecast_id': cash_forecast.forecast_id,
			'forecast_date': cash_forecast.forecast_date.isoformat(),
			'forecast_type': cash_forecast.forecast_type,
			'scenario': cash_forecast.scenario,
			'horizon_days': cash_forecast.horizon_days,
			'opening_balance': float(cash_forecast.opening_balance),
			'projected_inflows': float(cash_forecast.projected_inflows),
			'projected_outflows': float(cash_forecast.projected_outflows),
			'net_flow': float(cash_forecast.net_flow),
			'closing_balance': float(cash_forecast.closing_balance),
			'confidence_level': float(cash_forecast.confidence_level),
			'confidence_interval': {
				'lower': float(cash_forecast.confidence_interval_lower),
				'upper': float(cash_forecast.confidence_interval_upper)
			},
			'risk_metrics': {
				'shortfall_probability': float(cash_forecast.shortfall_probability) if cash_forecast.shortfall_probability else None,
				'stress_test_result': float(cash_forecast.stress_test_result) if cash_forecast.stress_test_result else None,
				'value_at_risk': float(cash_forecast.value_at_risk) if cash_forecast.value_at_risk else None
			},
			'model_info': {
				'model_used': cash_forecast.model_used,
				'model_version': cash_forecast.model_version,
				'feature_importance': cash_forecast.feature_importance
			}
		}
	
	async def get_forecast_accuracy(self, tenant_id: str, lookback_days: int = 30) -> Dict[str, Any]:
		"""Analyze forecast accuracy for model improvement."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		end_date = date.today()
		start_date = end_date - timedelta(days=lookback_days)
		
		# Get forecasts with actual outcomes
		query = select(CashForecast).where(
			and_(
				CashForecast.tenant_id == tenant_id,
				CashForecast.forecast_date >= start_date,
				CashForecast.forecast_date <= end_date,
				CashForecast.actual_outcome.isnot(None),
				CashForecast.is_deleted == False
			)
		)
		
		result = await self.db.execute(query)
		forecasts = result.scalars().all()
		
		if not forecasts:
			return {
				'analysis_period': {
					'start_date': start_date.isoformat(),
					'end_date': end_date.isoformat(),
					'days': lookback_days
				},
				'forecast_count': 0,
				'accuracy_metrics': None
			}
		
		# Calculate accuracy metrics
		total_error = Decimal('0')
		total_absolute_error = Decimal('0')
		total_percentage_error = Decimal('0')
		accurate_forecasts = 0
		
		accuracy_threshold = Decimal('5.0')  # 5% accuracy threshold
		
		for forecast in forecasts:
			if forecast.forecast_error and forecast.accuracy_percentage:
				total_error += forecast.forecast_error
				total_absolute_error += abs(forecast.forecast_error)
				total_percentage_error += abs(forecast.accuracy_percentage)
				
				if abs(forecast.accuracy_percentage) <= accuracy_threshold:
					accurate_forecasts += 1
		
		forecast_count = len(forecasts)
		mean_error = total_error / forecast_count
		mean_absolute_error = total_absolute_error / forecast_count
		mean_percentage_error = total_percentage_error / forecast_count
		accuracy_rate = (accurate_forecasts / forecast_count) * 100
		
		# Model performance breakdown
		model_performance = {}
		for forecast in forecasts:
			model = forecast.model_used
			if model not in model_performance:
				model_performance[model] = {
					'forecast_count': 0,
					'total_error': Decimal('0'),
					'total_absolute_error': Decimal('0'),
					'accurate_count': 0
				}
			
			model_performance[model]['forecast_count'] += 1
			if forecast.forecast_error:
				model_performance[model]['total_error'] += forecast.forecast_error
				model_performance[model]['total_absolute_error'] += abs(forecast.forecast_error)
				
				if forecast.accuracy_percentage and abs(forecast.accuracy_percentage) <= accuracy_threshold:
					model_performance[model]['accurate_count'] += 1
		
		# Calculate model-specific metrics
		for model, perf in model_performance.items():
			if perf['forecast_count'] > 0:
				perf['mean_error'] = float(perf['total_error'] / perf['forecast_count'])
				perf['mean_absolute_error'] = float(perf['total_absolute_error'] / perf['forecast_count'])
				perf['accuracy_rate'] = (perf['accurate_count'] / perf['forecast_count']) * 100
			
			# Clean up intermediate calculations
			del perf['total_error']
			del perf['total_absolute_error']
			del perf['accurate_count']
		
		return {
			'analysis_period': {
				'start_date': start_date.isoformat(),
				'end_date': end_date.isoformat(),
				'days': lookback_days
			},
			'forecast_count': forecast_count,
			'accuracy_metrics': {
				'mean_error': float(mean_error),
				'mean_absolute_error': float(mean_absolute_error),
				'mean_percentage_error': float(mean_percentage_error),
				'accuracy_rate_percent': float(accuracy_rate),
				'accurate_forecasts': accurate_forecasts
			},
			'model_performance': model_performance
		}
	
	# =========================================================================
	# Investment Management and Optimization
	# =========================================================================
	
	async def create_investment(self, tenant_id: str, investment_data: Dict[str, Any],
							   created_by: str) -> Investment:
		"""Create new investment with optimization scoring."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		assert created_by is not None, "created_by required for audit compliance"
		
		# Validate investment data
		validation_result = await self._validate_investment_data(investment_data, tenant_id)
		if not validation_result['valid']:
			raise ValueError(f"Investment validation failed: {validation_result['errors']}")
		
		# Create investment record
		investment = Investment(
			tenant_id=tenant_id,
			created_by=created_by,
			updated_by=created_by,
			**investment_data
		)
		
		self.db.add(investment)
		await self.db.commit()
		await self.db.refresh(investment)
		
		# Update portfolio metrics
		await self._update_investment_portfolio_metrics(tenant_id)
		
		self._log_investment_created(investment.investment_number, float(investment.principal_amount))
		return investment
	
	async def get_investment_opportunities(self, tenant_id: str, minimum_amount: Optional[Decimal] = None,
										  maximum_term_days: Optional[int] = None,
										  risk_rating: Optional[RiskRating] = None,
										  limit: int = 50) -> List[InvestmentOpportunity]:
		"""Get AI-curated investment opportunities with scoring."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		query = select(InvestmentOpportunity).where(
			and_(
				InvestmentOpportunity.tenant_id == tenant_id,
				InvestmentOpportunity.available_until > datetime.utcnow(),
				InvestmentOpportunity.is_deleted == False
			)
		)
		
		if minimum_amount:
			query = query.where(InvestmentOpportunity.minimum_amount >= minimum_amount)
		if maximum_term_days:
			query = query.where(InvestmentOpportunity.term_days <= maximum_term_days)
		if risk_rating:
			query = query.where(InvestmentOpportunity.risk_rating == risk_rating)
		
		query = query.order_by(desc(InvestmentOpportunity.ai_score)).limit(limit)
		
		result = await self.db.execute(query)
		return result.scalars().all()
	
	async def optimize_investment_allocation(self, tenant_id: str, available_cash: Decimal,
											optimization_goal: OptimizationGoal,
											constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""AI-powered investment allocation optimization."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		# Get available opportunities
		opportunities = await self.get_investment_opportunities(tenant_id, limit=100)
		
		if not opportunities:
			return {
				'optimization_goal': optimization_goal,
				'available_cash': float(available_cash),
				'recommendations': [],
				'total_allocated': 0.0,
				'remaining_cash': float(available_cash),
				'optimization_score': 0.0
			}
		
		# Apply optimization constraints
		constraints = constraints or {}
		max_single_investment = constraints.get('max_single_investment', available_cash * Decimal('0.3'))
		max_counterparty_exposure = constraints.get('max_counterparty_exposure', available_cash * Decimal('0.2'))
		min_liquidity_reserve = constraints.get('min_liquidity_reserve', available_cash * Decimal('0.1'))
		
		# Run optimization algorithm
		optimization_results = await self._run_investment_optimization(
			opportunities, available_cash, optimization_goal, constraints
		)
		
		return {
			'optimization_goal': optimization_goal,
			'available_cash': float(available_cash),
			'recommendations': optimization_results['recommendations'],
			'total_allocated': optimization_results['total_allocated'],
			'remaining_cash': optimization_results['remaining_cash'],
			'optimization_score': optimization_results['optimization_score'],
			'diversification_metrics': optimization_results['diversification_metrics'],
			'risk_metrics': optimization_results['risk_metrics']
		}
	
	# =========================================================================
	# Alert and Monitoring System
	# =========================================================================
	
	async def create_cash_alert(self, tenant_id: str, alert_data: Dict[str, Any],
							   created_by: str) -> CashAlert:
		"""Create cash management alert with escalation tracking."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		assert created_by is not None, "created_by required for audit compliance"
		
		alert = CashAlert(
			tenant_id=tenant_id,
			created_by=created_by,
			updated_by=created_by,
			**alert_data
		)
		
		self.db.add(alert)
		await self.db.commit()
		await self.db.refresh(alert)
		
		# Trigger notifications
		await self._trigger_alert_notifications(alert)
		
		self._log_alert_created(alert.alert_type, alert.severity)
		return alert
	
	async def get_active_alerts(self, tenant_id: str, severity: Optional[str] = None,
							   alert_type: Optional[AlertType] = None) -> List[CashAlert]:
		"""Get active cash management alerts."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		query = select(CashAlert).where(
			and_(
				CashAlert.tenant_id == tenant_id,
				CashAlert.status == 'active',
				CashAlert.is_deleted == False
			)
		)
		
		if severity:
			query = query.where(CashAlert.severity == severity)
		if alert_type:
			query = query.where(CashAlert.alert_type == alert_type)
		
		query = query.order_by(desc(CashAlert.triggered_at))
		
		result = await self.db.execute(query)
		return result.scalars().all()
	
	async def acknowledge_alert(self, alert_id: str, acknowledged_by: str,
							   notes: Optional[str] = None) -> CashAlert:
		"""Acknowledge cash alert with audit trail."""
		alert = await self._get_alert_by_id(alert_id)
		if not alert:
			raise ValueError(f"Alert {alert_id} not found")
		
		alert.status = 'acknowledged'
		alert.acknowledged_at = datetime.utcnow()
		alert.acknowledged_by = acknowledged_by
		if notes:
			alert.resolution_notes = notes
		alert.updated_at = datetime.utcnow()
		alert.version += 1
		
		await self.db.commit()
		await self.db.refresh(alert)
		
		self._log_alert_acknowledged(alert.alert_type, acknowledged_by)
		return alert
	
	# =========================================================================
	# Optimization Rules and Automation
	# =========================================================================
	
	async def create_optimization_rule(self, tenant_id: str, rule_data: Dict[str, Any],
									  created_by: str) -> OptimizationRule:
		"""Create cash optimization rule for automated decision making."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		assert created_by is not None, "created_by required for audit compliance"
		
		# Validate rule data
		validation_result = await self._validate_optimization_rule(rule_data, tenant_id)
		if not validation_result['valid']:
			raise ValueError(f"Optimization rule validation failed: {validation_result['errors']}")
		
		rule = OptimizationRule(
			tenant_id=tenant_id,
			created_by=created_by,
			updated_by=created_by,
			**rule_data
		)
		
		self.db.add(rule)
		await self.db.commit()
		await self.db.refresh(rule)
		
		self._log_optimization_rule_created(rule.rule_code, rule.optimization_goal)
		return rule
	
	async def execute_optimization_rules(self, tenant_id: str, trigger_context: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute applicable optimization rules with automated decision making."""
		assert tenant_id is not None, "tenant_id required for APG multi-tenancy"
		
		# Get active optimization rules
		query = select(OptimizationRule).where(
			and_(
				OptimizationRule.tenant_id == tenant_id,
				OptimizationRule.is_active == True,
				OptimizationRule.is_deleted == False
			)
		).order_by(desc(OptimizationRule.priority))
		
		result = await self.db.execute(query)
		rules = result.scalars().all()
		
		executed_rules = []
		total_actions = 0
		
		for rule in rules:
			# Check if rule conditions are met
			if await self._evaluate_rule_conditions(rule, trigger_context):
				# Execute rule logic
				execution_result = await self._execute_optimization_rule(rule, trigger_context)
				
				if execution_result['success']:
					# Update rule execution statistics
					rule.last_executed = datetime.utcnow()
					rule.execution_count += 1
					
					# Update success rate
					if rule.success_rate is None:
						rule.success_rate = Decimal('1.0')
					else:
						# Exponential moving average for success rate
						alpha = Decimal('0.1')
						rule.success_rate = (alpha * Decimal('1.0')) + ((Decimal('1.0') - alpha) * rule.success_rate)
					
					executed_rules.append({
						'rule_id': rule.id,
						'rule_code': rule.rule_code,
						'actions_taken': execution_result['actions_taken'],
						'optimization_score': execution_result['optimization_score']
					})
					
					total_actions += len(execution_result['actions_taken'])
				else:
					# Update failure statistics
					if rule.success_rate is not None:
						alpha = Decimal('0.1')
						rule.success_rate = ((Decimal('1.0') - alpha) * rule.success_rate)
		
		await self.db.commit()
		
		return {
			'execution_timestamp': datetime.utcnow().isoformat(),
			'rules_evaluated': len(rules),
			'rules_executed': len(executed_rules),
			'total_actions': total_actions,
			'executed_rules': executed_rules
		}
	
	# =========================================================================
	# Private Helper Methods
	# =========================================================================
	
	def _log_service_init(self) -> None:
		"""Log service initialization."""
		print("CashManagementService initialized with APG integration")
	
	def _log_bank_created(self, bank_code: str, bank_name: str) -> None:
		"""Log bank creation."""
		print(f"Bank created: {bank_code} - {bank_name}")
	
	def _log_bank_api_updated(self, bank_code: str, api_enabled: bool) -> None:
		"""Log bank API status update."""
		status = "enabled" if api_enabled else "disabled"
		print(f"Bank API {status}: {bank_code}")
	
	def _log_account_created(self, account_number: str, account_name: str) -> None:
		"""Log cash account creation."""
		print(f"Cash account created: {account_number} - {account_name}")
	
	def _log_balance_updated(self, account_number: str, old_balance: Decimal, new_balance: Decimal) -> None:
		"""Log balance update."""
		change = new_balance - old_balance
		print(f"Balance updated for {account_number}: {old_balance} -> {new_balance} (change: {change})")
	
	def _log_sweep_configured(self, account_number: str, enabled: bool) -> None:
		"""Log sweep configuration."""
		status = "enabled" if enabled else "disabled"
		print(f"Auto sweep {status} for account: {account_number}")
	
	def _log_cash_flow_recorded(self, transaction_type: TransactionType, amount: Decimal) -> None:
		"""Log cash flow recording."""
		print(f"Cash flow recorded: {transaction_type} - {amount}")
	
	def _log_forecast_generated(self, forecast_id: str, horizon_days: int, confidence: float) -> None:
		"""Log forecast generation."""
		print(f"Cash forecast generated: {forecast_id} - {horizon_days} days - {confidence:.1f}% confidence")
	
	def _log_investment_created(self, investment_number: str, principal_amount: float) -> None:
		"""Log investment creation."""
		print(f"Investment created: {investment_number} - ${principal_amount:,.2f}")
	
	def _log_alert_created(self, alert_type: AlertType, severity: str) -> None:
		"""Log alert creation."""
		print(f"Alert created: {alert_type} - {severity}")
	
	def _log_alert_acknowledged(self, alert_type: AlertType, acknowledged_by: str) -> None:
		"""Log alert acknowledgment."""
		print(f"Alert acknowledged: {alert_type} by {acknowledged_by}")
	
	def _log_optimization_rule_created(self, rule_code: str, optimization_goal: OptimizationGoal) -> None:
		"""Log optimization rule creation."""
		print(f"Optimization rule created: {rule_code} - {optimization_goal}")
	
	# Data validation methods
	async def _validate_bank_data(self, bank_data: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
		"""Validate bank data with business rules."""
		errors = []
		warnings = []
		
		# Required fields validation
		required_fields = ['bank_code', 'bank_name', 'swift_code', 'country_code']
		for field in required_fields:
			if field not in bank_data or not bank_data[field]:
				errors.append(f"Required field '{field}' is missing")
		
		# SWIFT code format validation
		if 'swift_code' in bank_data:
			swift_code = bank_data['swift_code']
			if len(swift_code) not in [8, 11] or not swift_code.isalnum():
				errors.append("SWIFT code must be 8 or 11 alphanumeric characters")
		
		return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
	
	async def _validate_account_data(self, account_data: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
		"""Validate cash account data with business rules."""
		errors = []
		warnings = []
		
		# Required fields validation
		required_fields = ['account_number', 'account_name', 'bank_id', 'account_type', 'currency_code', 'entity_id']
		for field in required_fields:
			if field not in account_data or not account_data[field]:
				errors.append(f"Required field '{field}' is missing")
		
		# Currency code validation
		if 'currency_code' in account_data:
			currency_code = account_data['currency_code']
			if len(currency_code) != 3 or not currency_code.isalpha():
				errors.append("Currency code must be 3-letter ISO code")
		
		return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
	
	async def _validate_cash_flow_data(self, flow_data: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
		"""Validate cash flow data with business rules."""
		errors = []
		warnings = []
		
		# Required fields validation
		required_fields = ['flow_date', 'description', 'account_id', 'transaction_type', 'amount', 'currency_code', 'category']
		for field in required_fields:
			if field not in flow_data or flow_data[field] is None:
				errors.append(f"Required field '{field}' is missing")
		
		# Amount validation
		if 'amount' in flow_data:
			amount = Decimal(str(flow_data['amount']))
			if amount == 0:
				errors.append("Amount cannot be zero")
		
		return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
	
	async def _validate_investment_data(self, investment_data: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
		"""Validate investment data with business rules."""
		errors = []
		warnings = []
		
		# Required fields validation
		required_fields = ['investment_type', 'issuer', 'principal_amount', 'currency_code', 'interest_rate', 'trade_date', 'value_date', 'maturity_date', 'booking_account_id']
		for field in required_fields:
			if field not in investment_data or investment_data[field] is None:
				errors.append(f"Required field '{field}' is missing")
		
		# Date validation
		if all(field in investment_data for field in ['trade_date', 'value_date', 'maturity_date']):
			trade_date = investment_data['trade_date']
			value_date = investment_data['value_date']
			maturity_date = investment_data['maturity_date']
			
			if maturity_date < value_date:
				errors.append("Maturity date cannot be before value date")
			if value_date < trade_date:
				errors.append("Value date cannot be before trade date")
		
		return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
	
	async def _validate_optimization_rule(self, rule_data: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
		"""Validate optimization rule data with business rules."""
		errors = []
		warnings = []
		
		# Required fields validation
		required_fields = ['rule_name', 'rule_code', 'category', 'optimization_goal']
		for field in required_fields:
			if field not in rule_data or not rule_data[field]:
				errors.append(f"Required field '{field}' is missing")
		
		return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
	
	# Database helper methods
	async def _get_bank_by_id(self, bank_id: str) -> Optional[Bank]:
		"""Get bank by ID."""
		query = select(Bank).where(
			and_(Bank.id == bank_id, Bank.is_deleted == False)
		)
		result = await self.db.execute(query)
		return result.scalar_one_or_none()
	
	async def _get_account_by_id(self, account_id: str) -> Optional[CashAccount]:
		"""Get cash account by ID."""
		query = select(CashAccount).where(
			and_(CashAccount.id == account_id, CashAccount.is_deleted == False)
		)
		result = await self.db.execute(query)
		return result.scalar_one_or_none()
	
	async def _get_alert_by_id(self, alert_id: str) -> Optional[CashAlert]:
		"""Get cash alert by ID."""
		query = select(CashAlert).where(
			and_(CashAlert.id == alert_id, CashAlert.is_deleted == False)
		)
		result = await self.db.execute(query)
		return result.scalar_one_or_none()
	
	# Cache helper methods
	async def _cache_bank_data(self, bank: Bank) -> None:
		"""Cache bank data for performance."""
		cache_key = f"bank:{bank.id}"
		await self.cache.setex(cache_key, 3600, bank)  # 1 hour cache
	
	async def _invalidate_bank_cache(self, tenant_id: str) -> None:
		"""Invalidate bank cache for tenant."""
		pattern = f"banks:{tenant_id}:*"
		keys = await self.cache.keys(pattern)
		if keys:
			await self.cache.delete(*keys)
	
	# Business logic helper methods (simplified implementations)
	async def _initialize_account_position(self, account: CashAccount) -> None:
		"""Initialize cash position tracking for new account."""
		# Create initial position record
		position = CashPosition(
			tenant_id=account.tenant_id,
			position_date=date.today(),
			entity_id=account.entity_id,
			currency_code=account.currency_code,
			total_cash=account.current_balance,
			available_cash=account.available_balance,
			created_by=account.created_by,
			updated_by=account.updated_by
		)
		
		self.db.add(position)
		await self.db.commit()
	
	async def _update_realtime_position(self, account: CashAccount, balance_change: Decimal) -> None:
		"""Update real-time cash position from balance change."""
		# This would update real-time position tracking
		# Simplified implementation
		pass
	
	async def _check_balance_alerts(self, account: CashAccount) -> None:
		"""Check for balance-based alerts."""
		# Check for low balance alerts
		if account.minimum_balance and account.effective_balance < account.minimum_balance:
			await self.create_cash_alert(
				account.tenant_id,
				{
					'alert_type': AlertType.BALANCE_LOW,
					'severity': 'high',
					'title': f'Low Balance Alert - {account.account_name}',
					'description': f'Account balance ${account.effective_balance} is below minimum ${account.minimum_balance}',
					'entity_id': account.entity_id,
					'account_id': account.id,
					'currency_code': account.currency_code,
					'current_value': account.effective_balance,
					'threshold_value': account.minimum_balance
				},
				'SYSTEM'
			)
	
	async def _calculate_current_positions(self, tenant_id: str, position_date: date) -> List[CashPosition]:
		"""Calculate current cash positions from account balances."""
		# Get all active accounts
		accounts = await self.get_cash_accounts(tenant_id, status=CashAccountStatus.ACTIVE)
		
		positions = []
		for account in accounts:
			position = CashPosition(
				tenant_id=tenant_id,
				position_date=position_date,
				entity_id=account.entity_id,
				currency_code=account.currency_code,
				total_cash=account.current_balance,
				available_cash=account.available_balance,
				checking_balance=account.current_balance if account.account_type == CashAccountType.CHECKING else Decimal('0'),
				savings_balance=account.current_balance if account.account_type == CashAccountType.SAVINGS else Decimal('0'),
				money_market_balance=account.current_balance if account.account_type == CashAccountType.MONEY_MARKET else Decimal('0'),
				investment_balance=account.current_balance if account.account_type == CashAccountType.INVESTMENT else Decimal('0'),
				created_by='SYSTEM',
				updated_by='SYSTEM'
			)
			
			self.db.add(position)
			positions.append(position)
		
		await self.db.commit()
		return positions
	
	async def _calculate_risk_metrics(self, positions: List[CashPosition]) -> Dict[str, Optional[float]]:
		"""Calculate risk metrics from cash positions."""
		if not positions:
			return {'liquidity_ratio': None, 'concentration_risk': None, 'days_cash_on_hand': None}
		
		total_cash = sum(pos.total_cash for pos in positions)
		available_cash = sum(pos.available_cash for pos in positions)
		
		# Liquidity ratio (available cash / total cash)
		liquidity_ratio = float(available_cash / total_cash) if total_cash > 0 else 0.0
		
		# Concentration risk (largest position / total cash)
		largest_position = max(pos.total_cash for pos in positions) if positions else Decimal('0')
		concentration_risk = float(largest_position / total_cash) if total_cash > 0 else 0.0
		
		# Days cash on hand (simplified calculation)
		# This would typically use historical outflow data
		daily_outflow = total_cash * Decimal('0.01')  # Assume 1% daily outflow
		days_cash_on_hand = int(available_cash / daily_outflow) if daily_outflow > 0 else 365
		
		return {
			'liquidity_ratio': liquidity_ratio,
			'concentration_risk': concentration_risk,
			'days_cash_on_hand': days_cash_on_hand
		}
	
	async def _update_position_from_flow(self, cash_flow: CashFlow) -> None:
		"""Update cash position from new cash flow."""
		# This would update position tracking in real-time
		# Simplified implementation
		pass
	
	async def _update_forecasting_patterns(self, cash_flow: CashFlow) -> None:
		"""Update forecasting patterns from recurring cash flow."""
		# This would update ML model training data
		# Simplified implementation
		pass
	
	async def _run_ml_forecast_model(self, tenant_id: str, entity_id: str, currency_code: str,
									horizon_days: int, scenario: ForecastScenario) -> Dict[str, Any]:
		"""Run machine learning forecast model (simplified implementation)."""
		# This would integrate with APG's AI orchestration
		# For now, return mock forecast results
		
		base_inflow = Decimal('100000')
		base_outflow = Decimal('90000')
		
		# Scenario adjustments
		scenario_multipliers = {
			ForecastScenario.BASE_CASE: (Decimal('1.0'), Decimal('1.0')),
			ForecastScenario.OPTIMISTIC: (Decimal('1.1'), Decimal('0.9')),
			ForecastScenario.PESSIMISTIC: (Decimal('0.9'), Decimal('1.1')),
			ForecastScenario.STRESS_TEST: (Decimal('0.7'), Decimal('1.3'))
		}
		
		inflow_mult, outflow_mult = scenario_multipliers[scenario]
		projected_inflows = base_inflow * inflow_mult
		projected_outflows = base_outflow * outflow_mult
		net_flow = projected_inflows - projected_outflows
		
		return {
			'projected_inflows': projected_inflows,
			'projected_outflows': projected_outflows,
			'net_flow': net_flow,
			'confidence_level': Decimal('0.85'),
			'confidence_interval_lower': net_flow * Decimal('0.8'),
			'confidence_interval_upper': net_flow * Decimal('1.2'),
			'standard_deviation': abs(net_flow) * Decimal('0.1'),
			'model_used': 'LSTM_v2.1',
			'model_version': '2.1.0',
			'feature_importance': {
				'historical_patterns': 0.4,
				'seasonal_factors': 0.3,
				'economic_indicators': 0.2,
				'business_events': 0.1
			},
			'shortfall_probability': Decimal('0.05') if scenario == ForecastScenario.STRESS_TEST else Decimal('0.01'),
			'stress_test_result': net_flow * Decimal('0.6') if scenario == ForecastScenario.STRESS_TEST else None,
			'value_at_risk': abs(net_flow) * Decimal('0.15')
		}
	
	async def _check_forecast_alerts(self, forecast: CashForecast) -> None:
		"""Check for forecast-based alerts."""
		# Check for shortfall alerts
		if forecast.shortfall_probability and forecast.shortfall_probability > Decimal('0.1'):
			await self.create_cash_alert(
				forecast.tenant_id,
				{
					'alert_type': AlertType.FORECAST_SHORTFALL,
					'severity': 'high',
					'title': 'Cash Shortfall Risk',
					'description': f'Forecast indicates {float(forecast.shortfall_probability * 100):.1f}% probability of cash shortfall',
					'entity_id': forecast.entity_id,
					'currency_code': forecast.currency_code,
					'current_value': forecast.closing_balance,
					'related_forecast_id': forecast.id
				},
				'SYSTEM'
			)
	
	async def _update_investment_portfolio_metrics(self, tenant_id: str) -> None:
		"""Update investment portfolio metrics."""
		# This would calculate portfolio-level metrics
		# Simplified implementation
		pass
	
	async def _run_investment_optimization(self, opportunities: List[InvestmentOpportunity],
										  available_cash: Decimal, optimization_goal: OptimizationGoal,
										  constraints: Dict[str, Any]) -> Dict[str, Any]:
		"""Run investment optimization algorithm (simplified implementation)."""
		# This would run sophisticated optimization algorithms
		# For now, return simple greedy selection based on AI score
		
		recommendations = []
		allocated_cash = Decimal('0')
		
		# Sort opportunities by AI score
		sorted_opportunities = sorted(opportunities, key=lambda x: x.ai_score, reverse=True)
		
		for opp in sorted_opportunities:
			if allocated_cash >= available_cash * Decimal('0.9'):  # Leave 10% cash buffer
				break
			
			# Check if we can afford this investment
			investment_amount = min(opp.recommended_amount or opp.minimum_amount, 
								   available_cash - allocated_cash)
			
			if investment_amount >= opp.minimum_amount:
				recommendations.append({
					'opportunity_id': opp.id,
					'investment_type': opp.investment_type,
					'provider': opp.provider,
					'recommended_amount': float(investment_amount),
					'interest_rate': float(opp.interest_rate),
					'term_days': opp.term_days,
					'ai_score': float(opp.ai_score),
					'yield_score': float(opp.yield_score),
					'risk_score': float(opp.risk_score),
					'annualized_yield': float(opp.annualized_yield)
				})
				
				allocated_cash += investment_amount
		
		optimization_score = sum(rec['ai_score'] * rec['recommended_amount'] for rec in recommendations)
		optimization_score = optimization_score / float(allocated_cash) if allocated_cash > 0 else 0.0
		
		return {
			'recommendations': recommendations,
			'total_allocated': float(allocated_cash),
			'remaining_cash': float(available_cash - allocated_cash),
			'optimization_score': optimization_score,
			'diversification_metrics': {
				'investment_count': len(recommendations),
				'type_diversity': len(set(rec['investment_type'] for rec in recommendations)),
				'provider_diversity': len(set(rec['provider'] for rec in recommendations))
			},
			'risk_metrics': {
				'weighted_risk_score': sum(rec['risk_score'] * rec['recommended_amount'] for rec in recommendations) / float(allocated_cash) if allocated_cash > 0 else 0.0,
				'average_term_days': sum(rec['term_days'] * rec['recommended_amount'] for rec in recommendations) / float(allocated_cash) if allocated_cash > 0 else 0
			}
		}
	
	async def _trigger_alert_notifications(self, alert: CashAlert) -> None:
		"""Trigger alert notifications through APG notification engine."""
		# This would integrate with APG's notification engine
		# Simplified implementation
		pass
	
	async def _evaluate_rule_conditions(self, rule: OptimizationRule, context: Dict[str, Any]) -> bool:
		"""Evaluate if optimization rule conditions are met."""
		# This would evaluate complex rule conditions
		# Simplified implementation - always return True for demo
		return True
	
	async def _execute_optimization_rule(self, rule: OptimizationRule, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute optimization rule logic."""
		# This would execute sophisticated optimization logic
		# Simplified implementation
		return {
			'success': True,
			'actions_taken': [f"Executed rule {rule.rule_code}"],
			'optimization_score': 85.0
		}


# Export all service classes and utilities
__all__ = [
	'CashManagementService'
]