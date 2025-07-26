"""
Cash Management Service

Business logic for Cash Management operations including bank reconciliation,
cash forecasting, foreign exchange management, and liquidity optimization.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func, case
import json
import logging

from .models import (
	CFCMBankAccount, CFCMBankTransaction, CFCMReconciliation, CFCMReconciliationItem,
	CFCMCashForecast, CFCMCashPosition, CFCMInvestment, CFCMCurrencyRate,
	CFCMCashTransfer, CFCMDeposit, CFCMCheckRegister
)

logger = logging.getLogger(__name__)


class CashManagementService:
	"""
	Core service class for cash management operations.
	
	Provides business logic for bank reconciliation, cash forecasting,
	liquidity management, and treasury operations.
	"""
	
	def __init__(self, db_session: Session):
		self.db = db_session
	
	# Bank Account Management
	
	def create_bank_account(self, tenant_id: str, account_data: Dict[str, Any]) -> CFCMBankAccount:
		"""Create new bank account"""
		account = CFCMBankAccount(
			tenant_id=tenant_id,
			**account_data
		)
		
		self.db.add(account)
		self.db.commit()
		
		self._log_activity(f"Bank account created: {account.account_number}")
		return account
	
	def get_bank_account_summary(self, tenant_id: str, account_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get bank account summary with balances and activity"""
		query = self.db.query(CFCMBankAccount).filter(CFCMBankAccount.tenant_id == tenant_id)
		
		if account_id:
			query = query.filter(CFCMBankAccount.bank_account_id == account_id)
		
		accounts = query.filter(CFCMBankAccount.is_active == True).all()
		
		summary = {
			'total_accounts': len(accounts),
			'total_balance': sum(acc.current_balance for acc in accounts),
			'total_available': sum(acc.get_available_balance() for acc in accounts),
			'accounts': []
		}
		
		for account in accounts:
			account_data = {
				'account_id': account.bank_account_id,
				'account_number': account.account_number,
				'account_name': account.account_name,
				'bank_name': account.bank_name,
				'current_balance': float(account.current_balance),
				'available_balance': float(account.get_available_balance()),
				'currency_code': account.currency_code,
				'last_reconciliation_date': account.last_reconciliation_date.isoformat() if account.last_reconciliation_date else None,
				'requires_reconciliation': account.requires_reconciliation,
				'is_overdrawn': account.is_overdrawn()
			}
			summary['accounts'].append(account_data)
		
		return summary
	
	# Transaction Management
	
	def import_bank_transactions(self, account_id: str, transactions: List[Dict[str, Any]], 
								 import_batch_id: Optional[str] = None) -> Dict[str, Any]:
		"""Import bank transactions from statement or API"""
		account = self.db.query(CFCMBankAccount).filter(
			CFCMBankAccount.bank_account_id == account_id
		).first()
		
		if not account:
			raise ValueError(f"Bank account {account_id} not found")
		
		imported_count = 0
		duplicate_count = 0
		error_count = 0
		errors = []
		
		for txn_data in transactions:
			try:
				# Check for duplicates based on bank reference and date
				existing = self.db.query(CFCMBankTransaction).filter(
					and_(
						CFCMBankTransaction.bank_account_id == account_id,
						CFCMBankTransaction.bank_reference == txn_data.get('bank_reference'),
						CFCMBankTransaction.transaction_date == txn_data.get('transaction_date')
					)
				).first()
				
				if existing:
					duplicate_count += 1
					continue
				
				# Create transaction
				transaction = CFCMBankTransaction(
					tenant_id=account.tenant_id,
					bank_account_id=account_id,
					imported=True,
					import_date=datetime.utcnow(),
					import_batch_id=import_batch_id,
					**txn_data
				)
				
				self.db.add(transaction)
				imported_count += 1
				
			except Exception as e:
				error_count += 1
				errors.append(f"Transaction {txn_data.get('bank_reference', 'Unknown')}: {str(e)}")
		
		# Update account last import date
		account.last_import_date = datetime.utcnow()
		
		self.db.commit()
		
		result = {
			'imported_count': imported_count,
			'duplicate_count': duplicate_count,
			'error_count': error_count,
			'errors': errors
		}
		
		self._log_activity(f"Imported {imported_count} transactions for account {account.account_number}")
		return result
	
	def get_transaction_history(self, account_id: str, start_date: Optional[date] = None,
							   end_date: Optional[date] = None, limit: int = 100) -> List[Dict[str, Any]]:
		"""Get transaction history for account"""
		query = self.db.query(CFCMBankTransaction).filter(
			CFCMBankTransaction.bank_account_id == account_id
		)
		
		if start_date:
			query = query.filter(CFCMBankTransaction.transaction_date >= start_date)
		if end_date:
			query = query.filter(CFCMBankTransaction.transaction_date <= end_date)
		
		transactions = query.order_by(desc(CFCMBankTransaction.transaction_date)).limit(limit).all()
		
		return [self._format_transaction(txn) for txn in transactions]
	
	# Bank Reconciliation
	
	def create_reconciliation(self, account_id: str, statement_data: Dict[str, Any]) -> CFCMReconciliation:
		"""Create new bank reconciliation"""
		account = self.db.query(CFCMBankAccount).filter(
			CFCMBankAccount.bank_account_id == account_id
		).first()
		
		if not account:
			raise ValueError(f"Bank account {account_id} not found")
		
		# Generate reconciliation number
		recon_number = self._generate_reconciliation_number(account.tenant_id)
		
		reconciliation = CFCMReconciliation(
			tenant_id=account.tenant_id,
			bank_account_id=account_id,
			reconciliation_number=recon_number,
			book_balance=account.ledger_balance,
			adjusted_book_balance=account.ledger_balance,
			adjusted_bank_balance=statement_data['statement_ending_balance'],
			**statement_data
		)
		
		self.db.add(reconciliation)
		self.db.commit()
		
		self._log_activity(f"Reconciliation created: {recon_number}")
		return reconciliation
	
	def perform_auto_matching(self, reconciliation_id: str, 
							 matching_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Perform automatic transaction matching"""
		reconciliation = self.db.query(CFCMReconciliation).filter(
			CFCMReconciliation.reconciliation_id == reconciliation_id
		).first()
		
		if not reconciliation:
			raise ValueError(f"Reconciliation {reconciliation_id} not found")
		
		# Default matching rules
		if not matching_rules:
			matching_rules = {
				'exact_match_tolerance': Decimal('0.00'),
				'amount_tolerance': Decimal('5.00'),
				'date_tolerance_days': 3,
				'min_confidence': Decimal('80.00')
			}
		
		# Get unreconciled bank transactions
		bank_transactions = self.db.query(CFCMBankTransaction).filter(
			and_(
				CFCMBankTransaction.bank_account_id == reconciliation.bank_account_id,
				CFCMBankTransaction.transaction_date >= reconciliation.period_start_date,
				CFCMBankTransaction.transaction_date <= reconciliation.period_end_date,
				CFCMBankTransaction.is_reconciled == False
			)
		).all()
		
		# Get GL transactions (simplified - would integrate with GL module)
		gl_transactions = self._get_gl_transactions_for_reconciliation(
			reconciliation.bank_account_id,
			reconciliation.period_start_date,
			reconciliation.period_end_date
		)
		
		matched_count = 0
		auto_matched_items = []
		
		for bank_txn in bank_transactions:
			best_match = None
			best_confidence = Decimal('0.00')
			
			for gl_txn in gl_transactions:
				confidence = self._calculate_match_confidence(bank_txn, gl_txn, matching_rules)
				
				if confidence > best_confidence and confidence >= matching_rules['min_confidence']:
					best_match = gl_txn
					best_confidence = confidence
			
			if best_match:
				# Create reconciliation item
				item = CFCMReconciliationItem(
					tenant_id=reconciliation.tenant_id,
					reconciliation_id=reconciliation_id,
					line_number=len(reconciliation.reconciliation_items) + 1,
					item_type='MATCHED',
					description=f"Auto-matched: {bank_txn.description}",
					bank_transaction_id=bank_txn.transaction_id,
					gl_transaction_id=best_match['id'],
					amount=bank_txn.amount,
					is_debit=bank_txn.is_debit,
					is_matched=True,
					match_confidence=best_confidence,
					match_method='AUTO'
				)
				
				self.db.add(item)
				
				# Mark bank transaction as reconciled
				bank_txn.reconcile(reconciliation_id)
				
				matched_count += 1
				auto_matched_items.append({
					'bank_transaction_id': bank_txn.transaction_id,
					'gl_transaction_id': best_match['id'],
					'confidence': float(best_confidence),
					'amount': float(bank_txn.amount)
				})
		
		# Update reconciliation statistics
		reconciliation.matched_transactions = matched_count
		reconciliation.calculate_variance()
		
		self.db.commit()
		
		result = {
			'matched_count': matched_count,
			'auto_matched_items': auto_matched_items,
			'remaining_variance': float(reconciliation.variance_amount)
		}
		
		self._log_activity(f"Auto-matching completed for reconciliation {reconciliation.reconciliation_number}: {matched_count} matches")
		return result
	
	def complete_reconciliation(self, reconciliation_id: str, user_id: str) -> bool:
		"""Complete bank reconciliation"""
		reconciliation = self.db.query(CFCMReconciliation).filter(
			CFCMReconciliation.reconciliation_id == reconciliation_id
		).first()
		
		if not reconciliation:
			raise ValueError(f"Reconciliation {reconciliation_id} not found")
		
		if reconciliation.can_complete():
			reconciliation.complete_reconciliation(user_id)
			self.db.commit()
			
			self._log_activity(f"Reconciliation completed: {reconciliation.reconciliation_number}")
			return True
		
		return False
	
	# Cash Forecasting
	
	def generate_cash_forecast(self, tenant_id: str, forecast_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate cash flow forecast"""
		horizon_days = forecast_config.get('horizon_days', 90)
		include_categories = forecast_config.get('categories', [])
		forecast_method = forecast_config.get('method', 'HISTORICAL')
		
		forecast_date = date.today()
		period_end = forecast_date + timedelta(days=horizon_days)
		
		# Clear existing forecasts for this period if requested
		if forecast_config.get('replace_existing', False):
			self.db.query(CFCMCashForecast).filter(
				and_(
					CFCMCashForecast.tenant_id == tenant_id,
					CFCMCashForecast.forecast_date >= forecast_date,
					CFCMCashForecast.forecast_date <= period_end
				)
			).delete()
		
		forecasts_created = 0
		forecast_items = []
		
		# Generate forecasts by category
		categories = self._get_forecast_categories(tenant_id, include_categories)
		
		for category in categories:
			category_forecasts = self._generate_category_forecast(
				tenant_id, category, forecast_date, period_end, forecast_method
			)
			
			for forecast_data in category_forecasts:
				forecast = CFCMCashForecast(
					tenant_id=tenant_id,
					forecast_date=forecast_data['forecast_date'],
					category_code=category['code'],
					category_name=category['name'],
					category_type=category['category_type'],
					forecast_amount=forecast_data['amount'],
					confidence_level=forecast_data['confidence'],
					forecast_method=forecast_method,
					period_start_date=forecast_date,
					period_end_date=period_end,
					**forecast_data.get('additional_fields', {})
				)
				
				self.db.add(forecast)
				forecasts_created += 1
				
				forecast_items.append({
					'date': forecast_data['forecast_date'].isoformat(),
					'category': category['name'],
					'type': category['category_type'],
					'amount': float(forecast_data['amount']),
					'confidence': float(forecast_data['confidence'])
				})
		
		self.db.commit()
		
		# Calculate summary statistics
		total_inflows = sum(item['amount'] for item in forecast_items if item['type'] == 'INFLOW')
		total_outflows = sum(item['amount'] for item in forecast_items if item['type'] == 'OUTFLOW')
		net_cash_flow = total_inflows - total_outflows
		
		result = {
			'forecast_period': {
				'start_date': forecast_date.isoformat(),
				'end_date': period_end.isoformat(),
				'horizon_days': horizon_days
			},
			'summary': {
				'total_inflows': total_inflows,
				'total_outflows': total_outflows,
				'net_cash_flow': net_cash_flow,
				'forecasts_created': forecasts_created
			},
			'forecast_items': forecast_items
		}
		
		self._log_activity(f"Cash forecast generated: {forecasts_created} items for {horizon_days} days")
		return result
	
	def get_cash_position_summary(self, tenant_id: str, as_of_date: Optional[date] = None) -> Dict[str, Any]:
		"""Get consolidated cash position summary"""
		if not as_of_date:
			as_of_date = date.today()
		
		# Get cash positions for all accounts
		positions = self.db.query(CFCMCashPosition).filter(
			and_(
				CFCMCashPosition.tenant_id == tenant_id,
				CFCMCashPosition.position_date == as_of_date
			)
		).all()
		
		# If no positions exist for the date, calculate from account balances
		if not positions:
			positions = self._calculate_cash_positions(tenant_id, as_of_date)
		
		# Calculate summary
		total_cash = sum(pos.closing_balance for pos in positions)
		total_inflows = sum(pos.total_inflows for pos in positions)
		total_outflows = sum(pos.total_outflows for pos in positions)
		net_change = sum(pos.net_change for pos in positions)
		
		# Get account breakdowns
		account_positions = []
		for pos in positions:
			account_positions.append({
				'account_id': pos.bank_account_id,
				'account_name': pos.bank_account.account_name,
				'currency': pos.currency_code,
				'opening_balance': float(pos.opening_balance),
				'closing_balance': float(pos.closing_balance),
				'net_change': float(pos.net_change),
				'transaction_count': pos.transaction_count
			})
		
		# Get weekly trend
		weekly_trend = self._get_cash_position_trend(tenant_id, as_of_date, 7)
		
		summary = {
			'as_of_date': as_of_date.isoformat(),
			'total_cash': float(total_cash),
			'total_inflows': float(total_inflows),
			'total_outflows': float(total_outflows),
			'net_change': float(net_change),
			'account_count': len(positions),
			'account_positions': account_positions,
			'weekly_trend': weekly_trend
		}
		
		return summary
	
	# Investment Management
	
	def create_investment(self, tenant_id: str, investment_data: Dict[str, Any]) -> CFCMInvestment:
		"""Create new investment record"""
		# Generate investment number if not provided
		if 'investment_number' not in investment_data:
			investment_data['investment_number'] = self._generate_investment_number(tenant_id)
		
		investment = CFCMInvestment(
			tenant_id=tenant_id,
			current_value=investment_data['purchase_amount'],  # Initial value
			**investment_data
		)
		
		self.db.add(investment)
		self.db.commit()
		
		self._log_activity(f"Investment created: {investment.investment_number}")
		return investment
	
	def update_investment_values(self, tenant_id: str, as_of_date: Optional[date] = None) -> Dict[str, Any]:
		"""Update current values for all investments"""
		if not as_of_date:
			as_of_date = date.today()
		
		investments = self.db.query(CFCMInvestment).filter(
			and_(
				CFCMInvestment.tenant_id == tenant_id,
				CFCMInvestment.status == 'Active'
			)
		).all()
		
		updated_count = 0
		total_value = Decimal('0.00')
		
		for investment in investments:
			# Calculate accrued interest
			investment.calculate_accrued_interest(as_of_date)
			
			# Update current value (simplified - would integrate with market data)
			if investment.investment_type in ['CD', 'BOND']:
				# For fixed income, value = face value + accrued interest
				investment.current_value = investment.face_value + investment.accrued_interest
			else:
				# For other investments, would need market data
				pass
			
			# Calculate unrealized gain/loss
			investment.unrealized_gain_loss = investment.current_value - investment.purchase_amount
			
			total_value += investment.current_value
			updated_count += 1
		
		self.db.commit()
		
		result = {
			'updated_count': updated_count,
			'total_portfolio_value': float(total_value),
			'as_of_date': as_of_date.isoformat()
		}
		
		self._log_activity(f"Updated {updated_count} investment values")
		return result
	
	def get_maturing_investments(self, tenant_id: str, days_ahead: int = 30) -> List[Dict[str, Any]]:
		"""Get investments maturing within specified days"""
		cutoff_date = date.today() + timedelta(days=days_ahead)
		
		investments = self.db.query(CFCMInvestment).filter(
			and_(
				CFCMInvestment.tenant_id == tenant_id,
				CFCMInvestment.status == 'Active',
				CFCMInvestment.maturity_date <= cutoff_date,
				CFCMInvestment.maturity_date >= date.today()
			)
		).order_by(CFCMInvestment.maturity_date).all()
		
		maturing_investments = []
		for inv in investments:
			days_to_maturity = inv.calculate_days_to_maturity()
			maturing_investments.append({
				'investment_id': inv.investment_id,
				'investment_number': inv.investment_number,
				'investment_name': inv.investment_name,
				'investment_type': inv.investment_type,
				'maturity_date': inv.maturity_date.isoformat(),
				'days_to_maturity': days_to_maturity,
				'current_value': float(inv.current_value),
				'auto_rollover': inv.auto_rollover,
				'bank_account': inv.bank_account.account_name
			})
		
		return maturing_investments
	
	# Foreign Exchange Management
	
	def update_fx_rates(self, tenant_id: str, rate_source: str = 'System', 
					   currency_pairs: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
		"""Update foreign exchange rates"""
		if not currency_pairs:
			# Get all currency pairs used in the system
			currency_pairs = self._get_active_currency_pairs(tenant_id)
		
		updated_count = 0
		errors = []
		rate_date = date.today()
		
		for from_currency, to_currency in currency_pairs:
			try:
				# Get rate from external source (simplified)
				rate_data = self._fetch_fx_rate(from_currency, to_currency, rate_source)
				
				if rate_data:
					# Check if rate already exists for today
					existing_rate = self.db.query(CFCMCurrencyRate).filter(
						and_(
							CFCMCurrencyRate.tenant_id == tenant_id,
							CFCMCurrencyRate.from_currency == from_currency,
							CFCMCurrencyRate.to_currency == to_currency,
							CFCMCurrencyRate.rate_date == rate_date,
							CFCMCurrencyRate.rate_type == 'SPOT'
						)
					).first()
					
					if existing_rate:
						# Update existing rate
						existing_rate.previous_rate = existing_rate.exchange_rate
						existing_rate.exchange_rate = rate_data['rate']
						existing_rate.calculate_inverse_rate()
						existing_rate.calculate_rate_change()
						existing_rate.last_updated = datetime.utcnow()
						existing_rate.rate_source = rate_source
					else:
						# Create new rate
						fx_rate = CFCMCurrencyRate(
							tenant_id=tenant_id,
							from_currency=from_currency,
							to_currency=to_currency,
							rate_date=rate_date,
							exchange_rate=rate_data['rate'],
							rate_source=rate_source,
							**rate_data.get('additional_fields', {})
						)
						fx_rate.calculate_inverse_rate()
						self.db.add(fx_rate)
					
					updated_count += 1
			
			except Exception as e:
				errors.append(f"{from_currency}/{to_currency}: {str(e)}")
		
		self.db.commit()
		
		result = {
			'updated_count': updated_count,
			'error_count': len(errors),
			'errors': errors,
			'rate_date': rate_date.isoformat()
		}
		
		self._log_activity(f"Updated {updated_count} FX rates")
		return result
	
	def convert_currency(self, amount: Decimal, from_currency: str, to_currency: str,
						as_of_date: Optional[date] = None) -> Dict[str, Any]:
		"""Convert currency amount using latest rates"""
		if from_currency == to_currency:
			return {
				'original_amount': float(amount),
				'converted_amount': float(amount),
				'exchange_rate': 1.0,
				'rate_date': (as_of_date or date.today()).isoformat()
			}
		
		if not as_of_date:
			as_of_date = date.today()
		
		# Find most recent rate
		fx_rate = self.db.query(CFCMCurrencyRate).filter(
			and_(
				CFCMCurrencyRate.from_currency == from_currency,
				CFCMCurrencyRate.to_currency == to_currency,
				CFCMCurrencyRate.rate_date <= as_of_date,
				CFCMCurrencyRate.is_active == True
			)
		).order_by(desc(CFCMCurrencyRate.rate_date)).first()
		
		if not fx_rate:
			# Try inverse rate
			fx_rate = self.db.query(CFCMCurrencyRate).filter(
				and_(
					CFCMCurrencyRate.from_currency == to_currency,
					CFCMCurrencyRate.to_currency == from_currency,
					CFCMCurrencyRate.rate_date <= as_of_date,
					CFCMCurrencyRate.is_active == True
				)
			).order_by(desc(CFCMCurrencyRate.rate_date)).first()
			
			if fx_rate:
				converted_amount = fx_rate.convert_amount(amount, reverse=True)
				exchange_rate = fx_rate.inverse_rate
			else:
				raise ValueError(f"No exchange rate found for {from_currency}/{to_currency}")
		else:
			converted_amount = fx_rate.convert_amount(amount)
			exchange_rate = fx_rate.exchange_rate
		
		return {
			'original_amount': float(amount),
			'converted_amount': float(converted_amount),
			'exchange_rate': float(exchange_rate),
			'rate_date': fx_rate.rate_date.isoformat(),
			'rate_source': fx_rate.rate_source
		}
	
	# Cash Transfer Management
	
	def create_cash_transfer(self, tenant_id: str, transfer_data: Dict[str, Any]) -> CFCMCashTransfer:
		"""Create new cash transfer"""
		# Generate transfer number if not provided
		if 'transfer_number' not in transfer_data:
			transfer_data['transfer_number'] = self._generate_transfer_number(tenant_id)
		
		transfer = CFCMCashTransfer(
			tenant_id=tenant_id,
			**transfer_data
		)
		
		# Calculate total amount
		transfer.calculate_total_amount()
		
		self.db.add(transfer)
		self.db.commit()
		
		self._log_activity(f"Cash transfer created: {transfer.transfer_number}")
		return transfer
	
	def process_transfer_batch(self, tenant_id: str, transfer_ids: List[str]) -> Dict[str, Any]:
		"""Process multiple transfers as a batch"""
		transfers = self.db.query(CFCMCashTransfer).filter(
			and_(
				CFCMCashTransfer.tenant_id == tenant_id,
				CFCMCashTransfer.transfer_id.in_(transfer_ids),
				CFCMCashTransfer.status == 'Approved'
			)
		).all()
		
		processed_count = 0
		failed_count = 0
		errors = []
		
		for transfer in transfers:
			try:
				if transfer.can_submit():
					transfer.submit_transfer('SYSTEM')
					processed_count += 1
				else:
					failed_count += 1
					errors.append(f"Transfer {transfer.transfer_number}: Cannot submit")
			
			except Exception as e:
				failed_count += 1
				errors.append(f"Transfer {transfer.transfer_number}: {str(e)}")
		
		self.db.commit()
		
		result = {
			'processed_count': processed_count,
			'failed_count': failed_count,
			'errors': errors
		}
		
		self._log_activity(f"Processed transfer batch: {processed_count} successful, {failed_count} failed")
		return result
	
	# Reporting and Analytics
	
	def generate_cash_flow_report(self, tenant_id: str, start_date: date, end_date: date,
								 account_ids: Optional[List[str]] = None) -> Dict[str, Any]:
		"""Generate comprehensive cash flow report"""
		# Base query for transactions
		query = self.db.query(CFCMBankTransaction).filter(
			and_(
				CFCMBankTransaction.tenant_id == tenant_id,
				CFCMBankTransaction.transaction_date >= start_date,
				CFCMBankTransaction.transaction_date <= end_date,
				CFCMBankTransaction.status == 'Posted'
			)
		)
		
		if account_ids:
			query = query.filter(CFCMBankTransaction.bank_account_id.in_(account_ids))
		
		transactions = query.all()
		
		# Categorize cash flows
		operating_inflows = Decimal('0.00')
		operating_outflows = Decimal('0.00')
		investing_inflows = Decimal('0.00')
		investing_outflows = Decimal('0.00')
		financing_inflows = Decimal('0.00')
		financing_outflows = Decimal('0.00')
		
		daily_positions = {}
		
		for txn in transactions:
			# Categorize transaction (simplified categorization)
			category = self._categorize_cash_flow(txn)
			
			if txn.is_debit:
				if category == 'OPERATING':
					operating_outflows += txn.amount
				elif category == 'INVESTING':
					investing_outflows += txn.amount
				elif category == 'FINANCING':
					financing_outflows += txn.amount
			else:
				if category == 'OPERATING':
					operating_inflows += txn.amount
				elif category == 'INVESTING':
					investing_inflows += txn.amount
				elif category == 'FINANCING':
					financing_inflows += txn.amount
			
			# Track daily positions
			txn_date = txn.transaction_date.isoformat()
			if txn_date not in daily_positions:
				daily_positions[txn_date] = {'inflows': 0.0, 'outflows': 0.0, 'net': 0.0}
			
			if txn.is_debit:
				daily_positions[txn_date]['outflows'] += float(txn.amount)
			else:
				daily_positions[txn_date]['inflows'] += float(txn.amount)
			
			daily_positions[txn_date]['net'] = (
				daily_positions[txn_date]['inflows'] - daily_positions[txn_date]['outflows']
			)
		
		# Calculate totals
		total_inflows = operating_inflows + investing_inflows + financing_inflows
		total_outflows = operating_outflows + investing_outflows + financing_outflows
		net_cash_flow = total_inflows - total_outflows
		
		report = {
			'report_period': {
				'start_date': start_date.isoformat(),
				'end_date': end_date.isoformat()
			},
			'cash_flow_summary': {
				'operating': {
					'inflows': float(operating_inflows),
					'outflows': float(operating_outflows),
					'net': float(operating_inflows - operating_outflows)
				},
				'investing': {
					'inflows': float(investing_inflows),
					'outflows': float(investing_outflows),
					'net': float(investing_inflows - investing_outflows)
				},
				'financing': {
					'inflows': float(financing_inflows),
					'outflows': float(financing_outflows),
					'net': float(financing_inflows - financing_outflows)
				},
				'total': {
					'inflows': float(total_inflows),
					'outflows': float(total_outflows),
					'net': float(net_cash_flow)
				}
			},
			'daily_positions': daily_positions,
			'transaction_count': len(transactions)
		}
		
		return report
	
	def get_reconciliation_status(self, tenant_id: str) -> Dict[str, Any]:
		"""Get reconciliation status across all accounts"""
		accounts = self.db.query(CFCMBankAccount).filter(
			and_(
				CFCMBankAccount.tenant_id == tenant_id,
				CFCMBankAccount.is_active == True,
				CFCMBankAccount.requires_reconciliation == True
			)
		).all()
		
		status_summary = {
			'total_accounts': len(accounts),
			'reconciled_current_month': 0,
			'pending_reconciliation': 0,
			'overdue_reconciliation': 0,
			'accounts': []
		}
		
		current_month_start = date.today().replace(day=1)
		
		for account in accounts:
			# Get latest reconciliation
			latest_recon = self.db.query(CFCMReconciliation).filter(
				CFCMReconciliation.bank_account_id == account.bank_account_id
			).order_by(desc(CFCMReconciliation.statement_date)).first()
			
			account_status = {
				'account_id': account.bank_account_id,
				'account_name': account.account_name,
				'last_reconciliation_date': account.last_reconciliation_date.isoformat() if account.last_reconciliation_date else None,
				'status': 'Never Reconciled'
			}
			
			if latest_recon:
				if latest_recon.statement_date >= current_month_start:
					account_status['status'] = 'Current'
					status_summary['reconciled_current_month'] += 1
				elif latest_recon.statement_date >= (current_month_start - timedelta(days=45)):
					account_status['status'] = 'Pending'
					status_summary['pending_reconciliation'] += 1
				else:
					account_status['status'] = 'Overdue'
					status_summary['overdue_reconciliation'] += 1
				
				account_status['last_variance'] = float(latest_recon.variance_amount)
				account_status['last_status'] = latest_recon.status
			else:
				status_summary['overdue_reconciliation'] += 1
			
			status_summary['accounts'].append(account_status)
		
		return status_summary
	
	# Helper Methods
	
	def _log_activity(self, message: str):
		"""Log cash management activity"""
		logger.info(f"CashManagement: {message}")
	
	def _generate_reconciliation_number(self, tenant_id: str) -> str:
		"""Generate unique reconciliation number"""
		count = self.db.query(CFCMReconciliation).filter(
			CFCMReconciliation.tenant_id == tenant_id
		).count()
		return f"RECON-{date.today().strftime('%Y%m')}-{count + 1:04d}"
	
	def _generate_investment_number(self, tenant_id: str) -> str:
		"""Generate unique investment number"""
		count = self.db.query(CFCMInvestment).filter(
			CFCMInvestment.tenant_id == tenant_id
		).count()
		return f"INV-{date.today().strftime('%Y%m')}-{count + 1:04d}"
	
	def _generate_transfer_number(self, tenant_id: str) -> str:
		"""Generate unique transfer number"""
		count = self.db.query(CFCMCashTransfer).filter(
			CFCMCashTransfer.tenant_id == tenant_id
		).count()
		return f"XFER-{date.today().strftime('%Y%m')}-{count + 1:04d}"
	
	def _format_transaction(self, transaction: CFCMBankTransaction) -> Dict[str, Any]:
		"""Format transaction for API response"""
		return {
			'transaction_id': transaction.transaction_id,
			'transaction_date': transaction.transaction_date.isoformat(),
			'description': transaction.description,
			'amount': float(transaction.amount),
			'is_debit': transaction.is_debit,
			'transaction_type': transaction.transaction_type,
			'status': transaction.status,
			'is_reconciled': transaction.is_reconciled,
			'counterparty_name': transaction.counterparty_name,
			'check_number': transaction.check_number,
			'running_balance': float(transaction.running_balance) if transaction.running_balance else None
		}
	
	def _calculate_match_confidence(self, bank_txn: CFCMBankTransaction, 
									gl_txn: Dict[str, Any], rules: Dict[str, Any]) -> Decimal:
		"""Calculate matching confidence between bank and GL transactions"""
		confidence = Decimal('0.00')
		
		# Exact amount match
		amount_diff = abs(bank_txn.amount - Decimal(str(gl_txn['amount'])))
		if amount_diff <= rules['exact_match_tolerance']:
			confidence += Decimal('60.00')
		elif amount_diff <= rules['amount_tolerance']:
			confidence += Decimal('30.00')
		
		# Date proximity
		date_diff = abs((bank_txn.transaction_date - gl_txn['date']).days)
		if date_diff == 0:
			confidence += Decimal('30.00')
		elif date_diff <= rules['date_tolerance_days']:
			confidence += Decimal('15.00')
		
		# Description similarity (simplified)
		if bank_txn.description and gl_txn.get('description'):
			if bank_txn.description.lower() in gl_txn['description'].lower():
				confidence += Decimal('10.00')
		
		return min(confidence, Decimal('100.00'))
	
	def _get_gl_transactions_for_reconciliation(self, account_id: str, start_date: date, 
											   end_date: date) -> List[Dict[str, Any]]:
		"""Get GL transactions for reconciliation (simplified)"""
		# This would integrate with the GL module to get actual GL transactions
		# For now, return empty list
		return []
	
	def _get_forecast_categories(self, tenant_id: str, include_categories: List[str]) -> List[Dict[str, Any]]:
		"""Get forecast categories configuration"""
		from . import get_cash_forecast_categories
		
		all_categories = get_cash_forecast_categories()
		
		if include_categories:
			return [cat for cat in all_categories if cat['code'] in include_categories]
		
		return all_categories
	
	def _generate_category_forecast(self, tenant_id: str, category: Dict[str, Any], 
								   start_date: date, end_date: date, method: str) -> List[Dict[str, Any]]:
		"""Generate forecast for specific category"""
		forecasts = []
		
		# Simplified forecast generation - would be more sophisticated in practice
		if method == 'HISTORICAL':
			# Use historical data for forecasting
			base_amount = self._get_historical_average(tenant_id, category['code'])
			
			current_date = start_date
			while current_date <= end_date:
				# Vary amount based on day of week, month, etc.
				variance = Decimal('0.9') + (Decimal('0.2') * Decimal(str(hash(current_date) % 100)) / Decimal('100'))
				amount = base_amount * variance
				
				forecasts.append({
					'forecast_date': current_date,
					'amount': amount,
					'confidence': Decimal('75.00')
				})
				
				current_date += timedelta(days=7)  # Weekly forecasts
		
		return forecasts
	
	def _get_historical_average(self, tenant_id: str, category_code: str) -> Decimal:
		"""Get historical average for category (simplified)"""
		# This would analyze historical data
		# For now, return a default amount
		category_defaults = {
			'AR_COLLECTIONS': Decimal('10000.00'),
			'AP_PAYMENTS': Decimal('8000.00'),
			'PAYROLL': Decimal('15000.00'),
			'SALES_RECEIPTS': Decimal('5000.00')
		}
		
		return category_defaults.get(category_code, Decimal('1000.00'))
	
	def _calculate_cash_positions(self, tenant_id: str, position_date: date) -> List[CFCMCashPosition]:
		"""Calculate cash positions for all accounts"""
		accounts = self.db.query(CFCMBankAccount).filter(
			and_(
				CFCMBankAccount.tenant_id == tenant_id,
				CFCMBankAccount.is_active == True
			)
		).all()
		
		positions = []
		
		for account in accounts:
			# Get transactions for the day
			transactions = self.db.query(CFCMBankTransaction).filter(
				and_(
					CFCMBankTransaction.bank_account_id == account.bank_account_id,
					CFCMBankTransaction.transaction_date == position_date,
					CFCMBankTransaction.status == 'Posted'
				)
			).all()
			
			total_inflows = sum(txn.amount for txn in transactions if not txn.is_debit)
			total_outflows = sum(txn.amount for txn in transactions if txn.is_debit)
			
			position = CFCMCashPosition(
				tenant_id=tenant_id,
				position_date=position_date,
				bank_account_id=account.bank_account_id,
				opening_balance=account.current_balance - (total_inflows - total_outflows),
				closing_balance=account.current_balance,
				total_inflows=total_inflows,
				total_outflows=total_outflows,
				transaction_count=len(transactions)
			)
			
			position.calculate_net_change()
			position.calculate_average_balance()
			
			self.db.add(position)
			positions.append(position)
		
		self.db.commit()
		return positions
	
	def _get_cash_position_trend(self, tenant_id: str, end_date: date, days: int) -> List[Dict[str, Any]]:
		"""Get cash position trend over specified days"""
		start_date = end_date - timedelta(days=days - 1)
		
		# Get daily totals
		daily_totals = self.db.query(
			CFCMCashPosition.position_date,
			func.sum(CFCMCashPosition.closing_balance).label('total_cash'),
			func.sum(CFCMCashPosition.net_change).label('net_change')
		).filter(
			and_(
				CFCMCashPosition.tenant_id == tenant_id,
				CFCMCashPosition.position_date >= start_date,
				CFCMCashPosition.position_date <= end_date
			)
		).group_by(CFCMCashPosition.position_date).all()
		
		trend = []
		for daily_total in daily_totals:
			trend.append({
				'date': daily_total.position_date.isoformat(),
				'total_cash': float(daily_total.total_cash or 0),
				'net_change': float(daily_total.net_change or 0)
			})
		
		return trend
	
	def _get_active_currency_pairs(self, tenant_id: str) -> List[Tuple[str, str]]:
		"""Get active currency pairs used in the system"""
		# Get all currencies used in bank accounts
		currencies = self.db.query(CFCMBankAccount.currency_code).filter(
			CFCMBankAccount.tenant_id == tenant_id
		).distinct().all()
		
		currency_list = [curr[0] for curr in currencies if curr[0] != 'USD']
		
		# Create pairs with USD as base
		pairs = []
		for currency in currency_list:
			pairs.append(('USD', currency))
			pairs.append((currency, 'USD'))
		
		return pairs
	
	def _fetch_fx_rate(self, from_currency: str, to_currency: str, source: str) -> Optional[Dict[str, Any]]:
		"""Fetch FX rate from external source (simplified)"""
		# This would integrate with actual FX rate providers
		# For now, return mock data
		mock_rates = {
			('USD', 'EUR'): Decimal('0.85'),
			('EUR', 'USD'): Decimal('1.18'),
			('USD', 'GBP'): Decimal('0.73'),
			('GBP', 'USD'): Decimal('1.37'),
			('USD', 'CAD'): Decimal('1.25'),
			('CAD', 'USD'): Decimal('0.80')
		}
		
		rate = mock_rates.get((from_currency, to_currency))
		if rate:
			return {
				'rate': rate,
				'bid_rate': rate * Decimal('0.999'),
				'ask_rate': rate * Decimal('1.001'),
				'mid_rate': rate
			}
		
		return None
	
	def _categorize_cash_flow(self, transaction: CFCMBankTransaction) -> str:
		"""Categorize transaction for cash flow reporting"""
		# Simplified categorization based on transaction type
		operating_types = ['DEPOSIT', 'ACH_IN', 'ACH_OUT', 'CHECK', 'FEE', 'INTEREST']
		investing_types = ['WIRE_IN', 'WIRE_OUT']
		financing_types = ['TRANSFER_IN', 'TRANSFER_OUT']
		
		if transaction.transaction_type in operating_types:
			return 'OPERATING'
		elif transaction.transaction_type in investing_types:
			return 'INVESTING'
		elif transaction.transaction_type in financing_types:
			return 'FINANCING'
		else:
			return 'OPERATING'  # Default