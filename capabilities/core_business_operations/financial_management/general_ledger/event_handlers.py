"""
APG Financial Management General Ledger - Event Handlers

Event handlers for processing platform and business events, enabling
seamless integration with other APG capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
import json

from .service import GeneralLedgerService, AccountCreationRequest, JournalEntryRequest
from .models import AccountTypeEnum, JournalSourceEnum, CurrencyEnum
from .integration import GLEventPublisher, IntegrationEventType

# Configure logging
logger = logging.getLogger(__name__)


class GLEventHandler:
	"""Handles incoming platform and business events for General Ledger"""
	
	def __init__(self, gl_service: GeneralLedgerService, event_publisher: GLEventPublisher):
		self.gl_service = gl_service
		self.event_publisher = event_publisher
		self.tenant_id = gl_service.tenant_id
		
		# Event handler registry
		self.handlers = {
			# Authentication events
			'authentication.user_logged_in': self.handle_user_login,
			'authentication.tenant_switched': self.handle_tenant_switch,
			'authentication.user_created': self.handle_user_created,
			
			# System events
			'system.period_end_approaching': self.handle_period_end_approaching,
			'system.backup_completed': self.handle_backup_completed,
			'system.maintenance_scheduled': self.handle_maintenance_scheduled,
			
			# Integration events
			'integration.api_rate_limit_exceeded': self.handle_rate_limit_exceeded,
			'integration.service_degraded': self.handle_service_degraded,
			
			# Business capability events
			'crm.customer_created': self.handle_customer_created,
			'crm.customer_updated': self.handle_customer_updated,
			'inventory.item_created': self.handle_inventory_item_created,
			'sales.order_completed': self.handle_sales_order_completed,
			'purchasing.invoice_received': self.handle_purchase_invoice_received,
			'hr.employee_hired': self.handle_employee_hired,
			'hr.payroll_processed': self.handle_payroll_processed,
			
			# Financial events from other modules
			'accounts_payable.invoice_approved': self.handle_ap_invoice_approved,
			'accounts_receivable.payment_received': self.handle_ar_payment_received,
			'fixed_assets.asset_acquired': self.handle_asset_acquired,
			'fixed_assets.depreciation_calculated': self.handle_depreciation_calculated,
			
			# Localization events
			'localization.currency_rate_updated': self.handle_currency_rate_updated,
			'localization.chart_template_updated': self.handle_chart_template_updated,
		}
		
		logger.info(f"GL Event Handler initialized for tenant {self.tenant_id}")
	
	async def handle_event(self, event_data: Dict[str, Any]) -> bool:
		"""Main event dispatcher"""
		try:
			event_type = event_data.get('event_type')
			
			if not event_type:
				logger.warning("Received event without event_type")
				return False
			
			# Get handler for event type
			handler = self.handlers.get(event_type)
			if not handler:
				logger.debug(f"No handler registered for event type: {event_type}")
				return False
			
			# Execute handler
			logger.info(f"Processing event: {event_type}")
			await handler(event_data)
			
			logger.debug(f"Successfully processed event: {event_type}")
			return True
			
		except Exception as e:
			logger.error(f"Error handling event {event_data.get('event_type', 'unknown')}: {e}")
			return False
	
	# =====================================
	# AUTHENTICATION EVENT HANDLERS
	# =====================================
	
	async def handle_user_login(self, event_data: Dict[str, Any]):
		"""Handle user login event"""
		try:
			payload = event_data.get('payload', {})
			user_id = payload.get('user_id')
			tenant_id = payload.get('tenant_id')
			
			if tenant_id != self.tenant_id:
				return  # Not for this tenant
			
			logger.info(f"User {user_id} logged in to tenant {tenant_id}")
			
			# Optionally pre-load user-specific GL data
			# Could cache frequently accessed accounts, recent transactions, etc.
			
		except Exception as e:
			logger.error(f"Error handling user login: {e}")
	
	async def handle_tenant_switch(self, event_data: Dict[str, Any]):
		"""Handle tenant switch event"""
		try:
			payload = event_data.get('payload', {})
			user_id = payload.get('user_id')
			old_tenant_id = payload.get('old_tenant_id')
			new_tenant_id = payload.get('new_tenant_id')
			
			logger.info(f"User {user_id} switched from tenant {old_tenant_id} to {new_tenant_id}")
			
			# Clear any cached data for the old tenant
			# Validate user access to new tenant's GL data
			
		except Exception as e:
			logger.error(f"Error handling tenant switch: {e}")
	
	async def handle_user_created(self, event_data: Dict[str, Any]):
		"""Handle new user creation"""
		try:
			payload = event_data.get('payload', {})
			user_id = payload.get('user_id')
			tenant_id = payload.get('tenant_id')
			user_role = payload.get('role')
			
			if tenant_id != self.tenant_id:
				return
			
			logger.info(f"New user {user_id} created with role {user_role} in tenant {tenant_id}")
			
			# Could set up default GL permissions, create user-specific accounts, etc.
			
		except Exception as e:
			logger.error(f"Error handling user creation: {e}")
	
	# =====================================
	# SYSTEM EVENT HANDLERS
	# =====================================
	
	async def handle_period_end_approaching(self, event_data: Dict[str, Any]):
		"""Handle period end approaching notification"""
		try:
			payload = event_data.get('payload', {})
			period_end_date = payload.get('period_end_date')
			days_remaining = payload.get('days_remaining')
			
			logger.info(f"Period end approaching: {period_end_date} ({days_remaining} days remaining)")
			
			# Perform pre-closing validations
			await self._perform_period_end_validations()
			
			# Notify users about pending period close
			await self._notify_period_end_approaching(period_end_date, days_remaining)
			
		except Exception as e:
			logger.error(f"Error handling period end approaching: {e}")
	
	async def handle_backup_completed(self, event_data: Dict[str, Any]):
		"""Handle backup completion notification"""
		try:
			payload = event_data.get('payload', {})
			backup_id = payload.get('backup_id')
			backup_type = payload.get('backup_type')
			success = payload.get('success', False)
			
			logger.info(f"Backup {backup_id} ({backup_type}) completed: {'✓' if success else '✗'}")
			
			# Could trigger GL-specific backup verification or cleanup
			
		except Exception as e:
			logger.error(f"Error handling backup completion: {e}")
	
	async def handle_maintenance_scheduled(self, event_data: Dict[str, Any]):
		"""Handle scheduled maintenance notification"""
		try:
			payload = event_data.get('payload', {})
			maintenance_start = payload.get('start_time')
			maintenance_end = payload.get('end_time')
			affected_services = payload.get('affected_services', [])
			
			if 'general_ledger' in affected_services or 'database' in affected_services:
				logger.info(f"Maintenance scheduled: {maintenance_start} to {maintenance_end}")
				
				# Could prepare for maintenance (finish pending operations, notify users, etc.)
			
		except Exception as e:
			logger.error(f"Error handling maintenance scheduling: {e}")
	
	# =====================================
	# BUSINESS EVENT HANDLERS
	# =====================================
	
	async def handle_customer_created(self, event_data: Dict[str, Any]):
		"""Handle new customer creation from CRM"""
		try:
			payload = event_data.get('payload', {})
			customer_id = payload.get('customer_id')
			customer_name = payload.get('customer_name')
			customer_type = payload.get('customer_type', 'standard')
			
			logger.info(f"New customer created: {customer_name} ({customer_id})")
			
			# Auto-create customer accounts if configured
			if self._should_auto_create_customer_accounts():
				await self._create_customer_accounts(customer_id, customer_name, customer_type)
			
		except Exception as e:
			logger.error(f"Error handling customer creation: {e}")
	
	async def handle_inventory_item_created(self, event_data: Dict[str, Any]):
		"""Handle new inventory item creation"""
		try:
			payload = event_data.get('payload', {})
			item_id = payload.get('item_id')
			item_name = payload.get('item_name')
			item_category = payload.get('category')
			
			logger.info(f"New inventory item created: {item_name} ({item_id})")
			
			# Auto-create inventory accounts if configured
			if self._should_auto_create_inventory_accounts():
				await self._create_inventory_accounts(item_id, item_name, item_category)
			
		except Exception as e:
			logger.error(f"Error handling inventory item creation: {e}")
	
	async def handle_sales_order_completed(self, event_data: Dict[str, Any]):
		"""Handle completed sales order"""
		try:
			payload = event_data.get('payload', {})
			order_id = payload.get('order_id')
			customer_id = payload.get('customer_id')
			total_amount = Decimal(str(payload.get('total_amount', 0)))
			currency = payload.get('currency', 'USD')
			order_date = payload.get('order_date')
			
			logger.info(f"Sales order completed: {order_id} for {total_amount} {currency}")
			
			# Create sales journal entry
			await self._create_sales_journal_entry(
				order_id, customer_id, total_amount, currency, order_date
			)
			
		except Exception as e:
			logger.error(f"Error handling sales order completion: {e}")
	
	async def handle_purchase_invoice_received(self, event_data: Dict[str, Any]):
		"""Handle received purchase invoice"""
		try:
			payload = event_data.get('payload', {})
			invoice_id = payload.get('invoice_id')
			vendor_id = payload.get('vendor_id')
			total_amount = Decimal(str(payload.get('total_amount', 0)))
			currency = payload.get('currency', 'USD')
			invoice_date = payload.get('invoice_date')
			
			logger.info(f"Purchase invoice received: {invoice_id} for {total_amount} {currency}")
			
			# Create purchase journal entry
			await self._create_purchase_journal_entry(
				invoice_id, vendor_id, total_amount, currency, invoice_date
			)
			
		except Exception as e:
			logger.error(f"Error handling purchase invoice: {e}")
	
	async def handle_payroll_processed(self, event_data: Dict[str, Any]):
		"""Handle payroll processing completion"""
		try:
			payload = event_data.get('payload', {})
			payroll_id = payload.get('payroll_id')
			pay_period = payload.get('pay_period')
			total_gross = Decimal(str(payload.get('total_gross', 0)))
			total_net = Decimal(str(payload.get('total_net', 0)))
			total_taxes = Decimal(str(payload.get('total_taxes', 0)))
			
			logger.info(f"Payroll processed: {payroll_id} for period {pay_period}")
			
			# Create payroll journal entries
			await self._create_payroll_journal_entries(
				payroll_id, pay_period, total_gross, total_net, total_taxes
			)
			
		except Exception as e:
			logger.error(f"Error handling payroll processing: {e}")
	
	# =====================================
	# FINANCIAL MODULE EVENT HANDLERS
	# =====================================
	
	async def handle_ap_invoice_approved(self, event_data: Dict[str, Any]):
		"""Handle accounts payable invoice approval"""
		try:
			payload = event_data.get('payload', {})
			invoice_id = payload.get('invoice_id')
			vendor_id = payload.get('vendor_id')
			amount = Decimal(str(payload.get('amount', 0)))
			
			logger.info(f"AP Invoice approved: {invoice_id} for {amount}")
			
			# Create AP journal entry
			await self._create_ap_journal_entry(invoice_id, vendor_id, amount)
			
		except Exception as e:
			logger.error(f"Error handling AP invoice approval: {e}")
	
	async def handle_ar_payment_received(self, event_data: Dict[str, Any]):
		"""Handle accounts receivable payment"""
		try:
			payload = event_data.get('payload', {})
			payment_id = payload.get('payment_id')
			customer_id = payload.get('customer_id')
			amount = Decimal(str(payload.get('amount', 0)))
			
			logger.info(f"AR Payment received: {payment_id} for {amount}")
			
			# Create AR payment journal entry
			await self._create_ar_payment_journal_entry(payment_id, customer_id, amount)
			
		except Exception as e:
			logger.error(f"Error handling AR payment: {e}")
	
	async def handle_asset_acquired(self, event_data: Dict[str, Any]):
		"""Handle fixed asset acquisition"""
		try:
			payload = event_data.get('payload', {})
			asset_id = payload.get('asset_id')
			asset_cost = Decimal(str(payload.get('cost', 0)))
			asset_category = payload.get('category')
			
			logger.info(f"Fixed asset acquired: {asset_id} for {asset_cost}")
			
			# Create asset acquisition journal entry
			await self._create_asset_acquisition_journal_entry(asset_id, asset_cost, asset_category)
			
		except Exception as e:
			logger.error(f"Error handling asset acquisition: {e}")
	
	async def handle_depreciation_calculated(self, event_data: Dict[str, Any]):
		"""Handle depreciation calculation"""
		try:
			payload = event_data.get('payload', {})
			asset_id = payload.get('asset_id')
			depreciation_amount = Decimal(str(payload.get('depreciation_amount', 0)))
			depreciation_period = payload.get('period')
			
			logger.info(f"Depreciation calculated for asset {asset_id}: {depreciation_amount}")
			
			# Create depreciation journal entry
			await self._create_depreciation_journal_entry(asset_id, depreciation_amount, depreciation_period)
			
		except Exception as e:
			logger.error(f"Error handling depreciation calculation: {e}")
	
	# =====================================
	# LOCALIZATION EVENT HANDLERS
	# =====================================
	
	async def handle_currency_rate_updated(self, event_data: Dict[str, Any]):
		"""Handle currency exchange rate updates"""
		try:
			payload = event_data.get('payload', {})
			from_currency = payload.get('from_currency')
			to_currency = payload.get('to_currency')
			new_rate = Decimal(str(payload.get('rate', 0)))
			effective_date = payload.get('effective_date')
			
			logger.info(f"Currency rate updated: {from_currency}/{to_currency} = {new_rate}")
			
			# Update currency rates in GL
			await self._update_currency_rate(from_currency, to_currency, new_rate, effective_date)
			
		except Exception as e:
			logger.error(f"Error handling currency rate update: {e}")
	
	# =====================================
	# HELPER METHODS
	# =====================================
	
	async def _perform_period_end_validations(self):
		"""Perform period end validations"""
		try:
			# Check for unposted journal entries
			unposted_entries = await self.gl_service.get_unposted_journal_entries()
			if unposted_entries:
				logger.warning(f"Found {len(unposted_entries)} unposted journal entries")
			
			# Check for out-of-balance entries
			unbalanced_entries = await self.gl_service.get_unbalanced_journal_entries()
			if unbalanced_entries:
				logger.error(f"Found {len(unbalanced_entries)} unbalanced journal entries")
			
			# Generate pre-close reports
			trial_balance = await self.gl_service.generate_trial_balance()
			if not trial_balance.metadata.get('balanced', False):
				logger.error("Trial balance is not balanced - period cannot be closed")
			
		except Exception as e:
			logger.error(f"Error in period end validations: {e}")
	
	async def _notify_period_end_approaching(self, period_end_date: str, days_remaining: int):
		"""Notify relevant users about approaching period end"""
		# This would typically send notifications through the notification service
		logger.info(f"Period end notification sent: {days_remaining} days remaining until {period_end_date}")
	
	def _should_auto_create_customer_accounts(self) -> bool:
		"""Check if customer accounts should be auto-created"""
		# This would check configuration settings
		return True  # Mock
	
	async def _create_customer_accounts(self, customer_id: str, customer_name: str, customer_type: str):
		"""Auto-create accounts for new customer"""
		try:
			# Create accounts receivable account for customer
			ar_account = AccountCreationRequest(
				account_code=f"1200-{customer_id[:8]}",
				account_name=f"AR - {customer_name}",
				account_type_id="accounts_receivable",
				description=f"Accounts Receivable for customer {customer_name}",
				currency=CurrencyEnum.USD
			)
			
			await self.gl_service.create_account(ar_account)
			logger.info(f"Created AR account for customer {customer_name}")
			
		except Exception as e:
			logger.error(f"Error creating customer accounts: {e}")
	
	async def _create_sales_journal_entry(self, order_id: str, customer_id: str, 
										  total_amount: Decimal, currency: str, order_date: str):
		"""Create journal entry for sales order"""
		try:
			# Create sales journal entry
			journal_request = JournalEntryRequest(
				description=f"Sales Order {order_id}",
				reference=order_id,
				entry_date=datetime.fromisoformat(order_date).date(),
				posting_date=datetime.now().date(),
				source=JournalSourceEnum.SALES,
				lines=[
					{
						"line_number": 1,
						"account_id": "1200-001",  # AR account
						"debit_amount": total_amount,
						"credit_amount": 0,
						"description": f"AR for customer {customer_id}"
					},
					{
						"line_number": 2,
						"account_id": "4000-001",  # Sales revenue account
						"debit_amount": 0,
						"credit_amount": total_amount,
						"description": f"Sales revenue for order {order_id}"
					}
				]
			)
			
			journal = await self.gl_service.create_journal_entry(journal_request)
			await self.event_publisher.journal_entry_posted(journal, self.tenant_id)
			
		except Exception as e:
			logger.error(f"Error creating sales journal entry: {e}")
	
	async def _update_currency_rate(self, from_currency: str, to_currency: str, 
									new_rate: Decimal, effective_date: str):
		"""Update currency exchange rate"""
		try:
			# Update currency rate in GL service
			await self.gl_service.update_currency_rate(
				from_currency=CurrencyEnum(from_currency),
				to_currency=CurrencyEnum(to_currency),
				rate=new_rate,
				effective_date=datetime.fromisoformat(effective_date).date()
			)
			
		except Exception as e:
			logger.error(f"Error updating currency rate: {e}")


# Export the event handler class
__all__ = ['GLEventHandler']