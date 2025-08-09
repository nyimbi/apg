"""
Fixed Asset Management Service

Business logic for Fixed Asset Management operations including asset lifecycle
management, depreciation calculations, maintenance scheduling, and reporting.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, extract
from dateutil.relativedelta import relativedelta
import json

from .models import (
	CFAMAsset, CFAMAssetCategory, CFAMDepreciationMethod, CFAMDepreciation,
	CFAMAssetAcquisition, CFAMAssetDisposal, CFAMAssetTransfer, CFAMAssetMaintenance,
	CFAMAssetInsurance, CFAMAssetValuation, CFAMAssetLease
)
from ..general_ledger.models import CFGLJournalEntry, CFGLJournalLine, CFGLAccount, CFGLPeriod
from ...auth_rbac.models import db


class FixedAssetManagementService:
	"""Service class for Fixed Asset Management operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	# Asset Management
	
	def create_asset(self, asset_data: Dict[str, Any]) -> CFAMAsset:
		"""Create a new fixed asset"""
		asset = CFAMAsset(
			tenant_id=self.tenant_id,
			asset_number=asset_data['asset_number'],
			asset_name=asset_data['asset_name'],
			description=asset_data.get('description'),
			category_id=asset_data['category_id'],
			acquisition_cost=Decimal(str(asset_data['acquisition_cost'])),
			salvage_value=Decimal(str(asset_data.get('salvage_value', 0.00))),
			acquisition_date=asset_data['acquisition_date'],
			placed_in_service_date=asset_data.get('placed_in_service_date'),
			location=asset_data.get('location'),
			department=asset_data.get('department'),
			cost_center=asset_data.get('cost_center'),
			custodian=asset_data.get('custodian'),
			manufacturer=asset_data.get('manufacturer'),
			model=asset_data.get('model'),
			serial_number=asset_data.get('serial_number'),
			useful_life_years=asset_data.get('useful_life_years'),
			useful_life_months=asset_data.get('useful_life_months'),
			depreciation_method_id=asset_data.get('depreciation_method_id'),
			is_depreciable=asset_data.get('is_depreciable', True),
			status=asset_data.get('status', 'Active'),
			currency_code=asset_data.get('currency_code', 'USD')
		)
		
		# Set current book value to acquisition cost initially
		asset.current_book_value = asset.acquisition_cost
		
		# Apply category defaults if not specified
		if asset.category:
			if not asset.useful_life_years:
				asset.useful_life_years = asset.category.default_useful_life_years
			if not asset.depreciation_method_id:
				asset.depreciation_method_id = asset.category.default_depreciation_method_id
			if not asset.salvage_value and asset.category.default_salvage_percent:
				asset.salvage_value = asset.acquisition_cost * (asset.category.default_salvage_percent / 100)
		
		db.session.add(asset)
		db.session.commit()
		return asset
	
	def get_asset(self, asset_id: str) -> Optional[CFAMAsset]:
		"""Get asset by ID"""
		return CFAMAsset.query.filter_by(
			tenant_id=self.tenant_id,
			asset_id=asset_id
		).first()
	
	def get_asset_by_number(self, asset_number: str) -> Optional[CFAMAsset]:
		"""Get asset by asset number"""
		return CFAMAsset.query.filter_by(
			tenant_id=self.tenant_id,
			asset_number=asset_number
		).first()
	
	def get_assets(self, status: str = None, category_id: str = None, 
				   location: str = None, include_disposed: bool = False) -> List[CFAMAsset]:
		"""Get assets with optional filters"""
		query = CFAMAsset.query.filter_by(tenant_id=self.tenant_id)
		
		if status:
			query = query.filter_by(status=status)
		elif not include_disposed:
			query = query.filter(CFAMAsset.status != 'Disposed')
		
		if category_id:
			query = query.filter_by(category_id=category_id)
		
		if location:
			query = query.filter_by(location=location)
		
		return query.order_by(CFAMAsset.asset_number).all()
	
	def update_asset(self, asset_id: str, update_data: Dict[str, Any]) -> Optional[CFAMAsset]:
		"""Update asset information"""
		asset = self.get_asset(asset_id)
		if not asset:
			return None
		
		# Update allowed fields
		for field, value in update_data.items():
			if hasattr(asset, field) and field not in ['asset_id', 'tenant_id', 'created_on']:
				setattr(asset, field, value)
		
		# Recalculate book value if depreciation changed
		if 'accumulated_depreciation' in update_data:
			asset.current_book_value = asset.acquisition_cost - asset.accumulated_depreciation
		
		db.session.commit()
		return asset
	
	def transfer_asset(self, asset_id: str, transfer_data: Dict[str, Any]) -> CFAMAssetTransfer:
		"""Create asset transfer record"""
		asset = self.get_asset(asset_id)
		if not asset:
			raise ValueError("Asset not found")
		
		# Create transfer record
		transfer = CFAMAssetTransfer(
			tenant_id=self.tenant_id,
			asset_id=asset_id,
			transfer_number=transfer_data.get('transfer_number', self._generate_transfer_number()),
			transfer_date=transfer_data['transfer_date'],
			transfer_type=transfer_data.get('transfer_type', 'Location'),
			reason=transfer_data.get('reason'),
			from_location=asset.location,
			from_department=asset.department,
			from_cost_center=asset.cost_center,
			from_custodian=asset.custodian,
			to_location=transfer_data.get('to_location'),
			to_department=transfer_data.get('to_department'),
			to_cost_center=transfer_data.get('to_cost_center'),
			to_custodian=transfer_data.get('to_custodian'),
			effective_date=transfer_data.get('effective_date', transfer_data['transfer_date']),
			transfer_cost=Decimal(str(transfer_data.get('transfer_cost', 0.00))),
			description=transfer_data.get('description'),
			notes=transfer_data.get('notes')
		)
		
		db.session.add(transfer)
		
		# Update asset with new location/assignment
		if transfer.to_location:
			asset.location = transfer.to_location
		if transfer.to_department:
			asset.department = transfer.to_department
		if transfer.to_cost_center:
			asset.cost_center = transfer.to_cost_center
		if transfer.to_custodian:
			asset.custodian = transfer.to_custodian
		
		db.session.commit()
		return transfer
	
	# Depreciation Management
	
	def calculate_monthly_depreciation(self, as_of_date: date = None) -> List[CFAMDepreciation]:
		"""Calculate monthly depreciation for all eligible assets"""
		if not as_of_date:
			as_of_date = date.today()
		
		# Get assets eligible for depreciation
		assets = CFAMAsset.query.filter(
			CFAMAsset.tenant_id == self.tenant_id,
			CFAMAsset.is_depreciable == True,
			CFAMAsset.is_fully_depreciated == False,
			CFAMAsset.status.in_(['Active', 'Maintenance']),
			or_(
				CFAMAsset.placed_in_service_date <= as_of_date,
				and_(
					CFAMAsset.placed_in_service_date.is_(None),
					CFAMAsset.acquisition_date <= as_of_date
				)
			)
		).all()
		
		depreciation_records = []
		
		for asset in assets:
			# Check if depreciation already calculated for this period
			period_name = as_of_date.strftime('%Y-%m')
			existing = CFAMDepreciation.query.filter_by(
				tenant_id=self.tenant_id,
				asset_id=asset.asset_id,
				period_name=period_name
			).first()
			
			if existing:
				continue  # Skip if already calculated
			
			# Calculate depreciation
			depreciation_record = self._calculate_asset_depreciation(asset, as_of_date)
			if depreciation_record and depreciation_record.depreciation_amount > 0:
				depreciation_records.append(depreciation_record)
		
		db.session.commit()
		return depreciation_records
	
	def _calculate_asset_depreciation(self, asset: CFAMAsset, as_of_date: date) -> Optional[CFAMDepreciation]:
		"""Calculate depreciation for a single asset"""
		if not asset.depreciation_method:
			return None
		
		# Determine service start date
		service_date = asset.placed_in_service_date or asset.acquisition_date
		if service_date > as_of_date:
			return None  # Not yet in service
		
		# Calculate months in service
		months_in_service = self._calculate_months_between(service_date, as_of_date)
		if months_in_service <= 0:
			return None
		
		# Get useful life
		useful_life_months = (asset.useful_life_years or 5) * 12
		if asset.useful_life_months:
			useful_life_months += asset.useful_life_months
		
		# Check if fully depreciated
		if months_in_service >= useful_life_months:
			remaining_value = asset.acquisition_cost - asset.salvage_value - asset.accumulated_depreciation
			if remaining_value <= 0.01:  # Consider fully depreciated if less than 1 cent
				asset.is_fully_depreciated = True
				return None
		
		# Calculate depreciation amount
		depreciable_base = asset.acquisition_cost - asset.salvage_value
		
		if asset.depreciation_method.formula == 'straight_line':
			monthly_depreciation = depreciable_base / useful_life_months
		elif asset.depreciation_method.formula == 'declining_balance':
			rate = asset.depreciation_method.depreciation_rate / 100 if asset.depreciation_method.depreciation_rate else (1 / (asset.useful_life_years or 5))
			monthly_rate = rate / 12
			current_book_value = asset.acquisition_cost - asset.accumulated_depreciation
			monthly_depreciation = current_book_value * monthly_rate
		else:
			# Default to straight line
			monthly_depreciation = depreciable_base / useful_life_months
		
		# Apply half-year convention for first year if applicable
		if asset.depreciation_method.convention == 'half_year':
			if months_in_service == 1:  # First month
				monthly_depreciation = monthly_depreciation / 2
		
		# Ensure we don't depreciate below salvage value
		remaining_depreciable = asset.acquisition_cost - asset.salvage_value - asset.accumulated_depreciation
		if monthly_depreciation > remaining_depreciable:
			monthly_depreciation = remaining_depreciable
		
		# Round to 2 decimal places
		monthly_depreciation = monthly_depreciation.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
		
		if monthly_depreciation <= 0:
			return None
		
		# Create depreciation record
		depreciation_record = CFAMDepreciation(
			tenant_id=self.tenant_id,
			asset_id=asset.asset_id,
			method_id=asset.depreciation_method_id,
			depreciation_date=as_of_date,
			fiscal_year=as_of_date.year,
			fiscal_period=as_of_date.month,
			period_name=as_of_date.strftime('%Y-%m'),
			beginning_book_value=asset.current_book_value,
			depreciation_amount=monthly_depreciation,
			accumulated_depreciation_before=asset.accumulated_depreciation,
			accumulated_depreciation_after=asset.accumulated_depreciation + monthly_depreciation,
			ending_book_value=asset.current_book_value - monthly_depreciation,
			depreciable_base=depreciable_base,
			salvage_value=asset.salvage_value,
			useful_life_remaining_months=useful_life_months - months_in_service,
			depreciation_rate=monthly_depreciation / asset.acquisition_cost if asset.acquisition_cost > 0 else Decimal('0.00')
		)
		
		# Update asset balances
		asset.accumulated_depreciation = depreciation_record.accumulated_depreciation_after
		asset.current_book_value = depreciation_record.ending_book_value
		asset.last_depreciation_date = as_of_date
		
		# Check if now fully depreciated
		if depreciation_record.ending_book_value <= asset.salvage_value + Decimal('0.01'):
			asset.is_fully_depreciated = True
		
		db.session.add(depreciation_record)
		return depreciation_record
	
	def post_depreciation_to_gl(self, depreciation_records: List[CFAMDepreciation]) -> List[str]:
		"""Post depreciation entries to General Ledger"""
		journal_entry_ids = []
		
		# Group by period for batch posting
		periods = {}
		for record in depreciation_records:
			period_key = record.period_name
			if period_key not in periods:
				periods[period_key] = []
			periods[period_key].append(record)
		
		for period_name, period_records in periods.items():
			# Get or create GL period
			year, month = period_name.split('-')
			gl_period = CFGLPeriod.query.filter_by(
				tenant_id=self.tenant_id,
				fiscal_year=int(year),
				period_number=int(month)
			).first()
			
			if not gl_period or not gl_period.can_post():
				continue  # Skip if period not found or closed
			
			# Create journal entry
			journal_entry = CFGLJournalEntry(
				tenant_id=self.tenant_id,
				journal_number=self._generate_journal_number(),
				description=f'Monthly depreciation - {period_name}',
				entry_date=date(int(year), int(month), 1),
				posting_date=date.today(),
				period_id=gl_period.period_id,
				source='FAM_DEPRECIATION',
				status='Draft'
			)
			
			db.session.add(journal_entry)
			db.session.flush()  # Get journal_entry.journal_id
			
			# Create journal lines
			line_number = 1
			total_depreciation = Decimal('0.00')
			
			# Group by GL accounts for summary posting
			account_totals = {}
			
			for record in period_records:
				asset = record.asset
				
				# Get GL accounts
				expense_account_id = (asset.gl_expense_account_id or 
									 asset.category.gl_expense_account_id)
				depreciation_account_id = (asset.gl_depreciation_account_id or 
										  asset.category.gl_depreciation_account_id)
				
				if not expense_account_id or not depreciation_account_id:
					continue  # Skip if GL accounts not configured
				
				# Accumulate by account
				if expense_account_id not in account_totals:
					account_totals[expense_account_id] = {'type': 'expense', 'amount': Decimal('0.00')}
				if depreciation_account_id not in account_totals:
					account_totals[depreciation_account_id] = {'type': 'depreciation', 'amount': Decimal('0.00')}
				
				account_totals[expense_account_id]['amount'] += record.depreciation_amount
				account_totals[depreciation_account_id]['amount'] += record.depreciation_amount
				total_depreciation += record.depreciation_amount
			
			# Create journal lines for each account
			for account_id, account_data in account_totals.items():
				if account_data['type'] == 'expense':
					# Depreciation Expense (Debit)
					line = CFGLJournalLine(
						journal_id=journal_entry.journal_id,
						tenant_id=self.tenant_id,
						line_number=line_number,
						account_id=account_id,
						debit_amount=account_data['amount'],
						credit_amount=Decimal('0.00'),
						description=f'Depreciation expense - {period_name}'
					)
				else:
					# Accumulated Depreciation (Credit)
					line = CFGLJournalLine(
						journal_id=journal_entry.journal_id,
						tenant_id=self.tenant_id,
						line_number=line_number,
						account_id=account_id,
						debit_amount=Decimal('0.00'),
						credit_amount=account_data['amount'],
						description=f'Accumulated depreciation - {period_name}'
					)
				
				db.session.add(line)
				line_number += 1
			
			# Update journal entry totals
			journal_entry.calculate_totals()
			
			# Post journal entry if balanced
			if journal_entry.validate_balance():
				journal_entry.post_entry('system')  # TODO: Use actual user ID
				journal_entry_ids.append(journal_entry.journal_id)
				
				# Update depreciation records with posting info
				for record in period_records:
					record.is_posted = True
					record.posted_date = datetime.utcnow()
					record.journal_entry_id = journal_entry.journal_id
					record.posted_by = 'system'  # TODO: Use actual user ID
		
		db.session.commit()
		return journal_entry_ids
	
	# Asset Acquisition
	
	def create_acquisition(self, acquisition_data: Dict[str, Any]) -> CFAMAssetAcquisition:
		"""Create asset acquisition record"""
		acquisition = CFAMAssetAcquisition(
			tenant_id=self.tenant_id,
			asset_id=acquisition_data['asset_id'],
			acquisition_number=acquisition_data.get('acquisition_number', self._generate_acquisition_number()),
			acquisition_type=acquisition_data.get('acquisition_type', 'Purchase'),
			acquisition_date=acquisition_data['acquisition_date'],
			gross_cost=Decimal(str(acquisition_data['gross_cost'])),
			freight_cost=Decimal(str(acquisition_data.get('freight_cost', 0.00))),
			installation_cost=Decimal(str(acquisition_data.get('installation_cost', 0.00))),
			other_costs=Decimal(str(acquisition_data.get('other_costs', 0.00))),
			vendor_name=acquisition_data.get('vendor_name'),
			vendor_id=acquisition_data.get('vendor_id'),
			purchase_order_number=acquisition_data.get('purchase_order_number'),
			invoice_number=acquisition_data.get('invoice_number'),
			invoice_date=acquisition_data.get('invoice_date'),
			funding_source=acquisition_data.get('funding_source'),
			project_id=acquisition_data.get('project_id'),
			department=acquisition_data.get('department'),
			cost_center=acquisition_data.get('cost_center'),
			description=acquisition_data.get('description'),
			notes=acquisition_data.get('notes')
		)
		
		acquisition.calculate_total_cost()
		db.session.add(acquisition)
		db.session.commit()
		return acquisition
	
	# Asset Disposal
	
	def create_disposal(self, disposal_data: Dict[str, Any]) -> CFAMAssetDisposal:
		"""Create asset disposal record"""
		asset = self.get_asset(disposal_data['asset_id'])
		if not asset:
			raise ValueError("Asset not found")
		
		if not asset.is_disposal_eligible():
			raise ValueError("Asset is not eligible for disposal")
		
		disposal = CFAMAssetDisposal(
			tenant_id=self.tenant_id,
			asset_id=disposal_data['asset_id'],
			disposal_number=disposal_data.get('disposal_number', self._generate_disposal_number()),
			disposal_date=disposal_data['disposal_date'],
			disposal_method=disposal_data['disposal_method'],
			disposal_reason=disposal_data.get('disposal_reason'),
			book_value_at_disposal=asset.current_book_value,
			accumulated_depreciation_at_disposal=asset.accumulated_depreciation,
			disposal_proceeds=Decimal(str(disposal_data.get('disposal_proceeds', 0.00))),
			disposal_costs=Decimal(str(disposal_data.get('disposal_costs', 0.00))),
			purchaser_name=disposal_data.get('purchaser_name'),
			purchaser_contact=disposal_data.get('purchaser_contact'),
			description=disposal_data.get('description'),
			notes=disposal_data.get('notes')
		)
		
		disposal.calculate_gain_loss()
		
		# Update asset status
		asset.status = 'Disposed'
		asset.disposal_date = disposal.disposal_date
		
		db.session.add(disposal)
		db.session.commit()
		return disposal
	
	# Maintenance Management
	
	def schedule_maintenance(self, maintenance_data: Dict[str, Any]) -> CFAMAssetMaintenance:
		"""Schedule asset maintenance"""
		maintenance = CFAMAssetMaintenance(
			tenant_id=self.tenant_id,
			asset_id=maintenance_data['asset_id'],
			maintenance_number=maintenance_data.get('maintenance_number', self._generate_maintenance_number()),
			maintenance_type=maintenance_data['maintenance_type'],
			scheduled_date=maintenance_data['scheduled_date'],
			maintenance_date=maintenance_data.get('maintenance_date', maintenance_data['scheduled_date']),
			service_provider=maintenance_data.get('service_provider'),
			technician_name=maintenance_data.get('technician_name'),
			description=maintenance_data['description'],
			priority=maintenance_data.get('priority', 'Normal'),
			urgency=maintenance_data.get('urgency', 'Routine'),
			status='Scheduled',
			recurring=maintenance_data.get('recurring', False),
			maintenance_interval_days=maintenance_data.get('maintenance_interval_days'),
			notes=maintenance_data.get('notes')
		)
		
		db.session.add(maintenance)
		
		# Update asset next maintenance date
		if maintenance.recurring and maintenance.maintenance_interval_days:
			asset = self.get_asset(maintenance_data['asset_id'])
			if asset:
				asset.next_maintenance_date = (
					maintenance.scheduled_date + 
					timedelta(days=maintenance.maintenance_interval_days)
				)
		
		db.session.commit()
		return maintenance
	
	def complete_maintenance(self, maintenance_id: str, completion_data: Dict[str, Any]) -> CFAMAssetMaintenance:
		"""Complete scheduled maintenance"""
		maintenance = CFAMAssetMaintenance.query.filter_by(
			tenant_id=self.tenant_id,
			maintenance_id=maintenance_id
		).first()
		
		if not maintenance:
			raise ValueError("Maintenance record not found")
		
		# Update maintenance record
		maintenance.status = 'Completed'
		maintenance.completed_date = completion_data.get('completed_date', date.today())
		maintenance.work_performed = completion_data.get('work_performed')
		maintenance.parts_replaced = completion_data.get('parts_replaced')
		maintenance.findings = completion_data.get('findings')
		maintenance.recommendations = completion_data.get('recommendations')
		
		# Update costs
		if 'labor_hours' in completion_data:
			maintenance.labor_hours = Decimal(str(completion_data['labor_hours']))
		if 'labor_rate' in completion_data:
			maintenance.labor_rate = Decimal(str(completion_data['labor_rate']))
		if 'parts_cost' in completion_data:
			maintenance.parts_cost = Decimal(str(completion_data['parts_cost']))
		if 'other_costs' in completion_data:
			maintenance.other_costs = Decimal(str(completion_data['other_costs']))
		
		maintenance.calculate_total_cost()
		
		# Update asset maintenance info
		asset = maintenance.asset
		asset.last_maintenance_date = maintenance.completed_date
		asset.maintenance_cost_ytd += maintenance.total_cost
		
		# Schedule next maintenance if recurring
		if maintenance.recurring and maintenance.maintenance_interval_days:
			next_date = maintenance.completed_date + timedelta(days=maintenance.maintenance_interval_days)
			asset.next_maintenance_date = next_date
			
			# Create next maintenance record
			next_maintenance = CFAMAssetMaintenance(
				tenant_id=self.tenant_id,
				asset_id=asset.asset_id,
				maintenance_number=self._generate_maintenance_number(),
				maintenance_type=maintenance.maintenance_type,
				scheduled_date=next_date,
				maintenance_date=next_date,
				description=maintenance.description,
				priority=maintenance.priority,
				status='Scheduled',
				recurring=True,
				maintenance_interval_days=maintenance.maintenance_interval_days
			)
			db.session.add(next_maintenance)
		
		db.session.commit()
		return maintenance
	
	# Insurance Management
	
	def add_insurance_policy(self, insurance_data: Dict[str, Any]) -> CFAMAssetInsurance:
		"""Add insurance policy for asset"""
		insurance = CFAMAssetInsurance(
			tenant_id=self.tenant_id,
			asset_id=insurance_data['asset_id'],
			policy_number=insurance_data['policy_number'],
			insurance_company=insurance_data['insurance_company'],
			policy_type=insurance_data['policy_type'],
			coverage_type=insurance_data.get('coverage_type'),
			coverage_amount=Decimal(str(insurance_data['coverage_amount'])),
			deductible_amount=Decimal(str(insurance_data.get('deductible_amount', 0.00))),
			premium_amount=Decimal(str(insurance_data['premium_amount'])),
			policy_start_date=insurance_data['policy_start_date'],
			policy_end_date=insurance_data['policy_end_date'],
			agent_name=insurance_data.get('agent_name'),
			agent_phone=insurance_data.get('agent_phone'),
			agent_email=insurance_data.get('agent_email'),
			auto_renew=insurance_data.get('auto_renew', False),
			notes=insurance_data.get('notes')
		)
		
		# Update asset insurance info
		asset = self.get_asset(insurance_data['asset_id'])
		if asset:
			asset.is_insured = True
			asset.insurance_value = insurance.coverage_amount
		
		db.session.add(insurance)
		db.session.commit()
		return insurance
	
	def get_insurance_renewals_due(self, days_ahead: int = 30) -> List[CFAMAssetInsurance]:
		"""Get insurance policies due for renewal"""
		cutoff_date = date.today() + timedelta(days=days_ahead)
		
		return CFAMAssetInsurance.query.filter(
			CFAMAssetInsurance.tenant_id == self.tenant_id,
			CFAMAssetInsurance.is_active == True,
			CFAMAssetInsurance.policy_end_date <= cutoff_date
		).order_by(CFAMAssetInsurance.policy_end_date).all()
	
	# Reporting and Analytics
	
	def get_asset_summary(self) -> Dict[str, Any]:
		"""Get asset summary statistics"""
		assets = CFAMAsset.query.filter_by(tenant_id=self.tenant_id).all()
		
		total_assets = len(assets)
		total_cost = sum(asset.acquisition_cost for asset in assets)
		total_accumulated_depreciation = sum(asset.accumulated_depreciation for asset in assets)
		total_book_value = sum(asset.current_book_value for asset in assets)
		
		# Status breakdown
		status_counts = {}
		for asset in assets:
			status_counts[asset.status] = status_counts.get(asset.status, 0) + 1
		
		# Category breakdown
		category_totals = {}
		for asset in assets:
			category_name = asset.category.category_name if asset.category else 'Uncategorized'
			if category_name not in category_totals:
				category_totals[category_name] = {'count': 0, 'cost': Decimal('0.00'), 'book_value': Decimal('0.00')}
			category_totals[category_name]['count'] += 1
			category_totals[category_name]['cost'] += asset.acquisition_cost
			category_totals[category_name]['book_value'] += asset.current_book_value
		
		return {
			'total_assets': total_assets,
			'total_cost': float(total_cost),
			'total_accumulated_depreciation': float(total_accumulated_depreciation),
			'total_book_value': float(total_book_value),
			'depreciation_percentage': float((total_accumulated_depreciation / total_cost * 100) if total_cost > 0 else 0),
			'status_breakdown': status_counts,
			'category_breakdown': category_totals
		}
	
	def get_depreciation_schedule(self, asset_id: str = None, months_ahead: int = 12) -> List[Dict[str, Any]]:
		"""Get depreciation schedule for assets"""
		query = CFAMAsset.query.filter_by(tenant_id=self.tenant_id, is_depreciable=True, is_fully_depreciated=False)
		
		if asset_id:
			query = query.filter_by(asset_id=asset_id)
		
		assets = query.all()
		schedule = []
		
		for asset in assets:
			for month in range(1, months_ahead + 1):
				future_date = date.today() + relativedelta(months=month)
				monthly_depreciation = asset.get_monthly_depreciation()
				
				if monthly_depreciation > 0:
					schedule.append({
						'asset_id': asset.asset_id,
						'asset_number': asset.asset_number,
						'asset_name': asset.asset_name,
						'month': future_date.strftime('%Y-%m'),
						'depreciation_amount': float(monthly_depreciation),
						'accumulated_depreciation': float(asset.accumulated_depreciation + (monthly_depreciation * month)),
						'book_value': float(asset.current_book_value - (monthly_depreciation * month))
					})
		
		return sorted(schedule, key=lambda x: (x['month'], x['asset_number']))
	
	def get_maintenance_schedule(self, days_ahead: int = 90) -> List[CFAMAssetMaintenance]:
		"""Get upcoming maintenance schedule"""
		cutoff_date = date.today() + timedelta(days=days_ahead)
		
		return CFAMAssetMaintenance.query.filter(
			CFAMAssetMaintenance.tenant_id == self.tenant_id,
			CFAMAssetMaintenance.status == 'Scheduled',
			CFAMAssetMaintenance.scheduled_date <= cutoff_date
		).order_by(CFAMAssetMaintenance.scheduled_date).all()
	
	def get_assets_by_location(self) -> Dict[str, List[Dict[str, Any]]]:
		"""Get assets grouped by location"""
		assets = CFAMAsset.query.filter(
			CFAMAsset.tenant_id == self.tenant_id,
			CFAMAsset.status != 'Disposed'
		).order_by(CFAMAsset.location, CFAMAsset.asset_number).all()
		
		locations = {}
		for asset in assets:
			location = asset.location or 'Unassigned'
			if location not in locations:
				locations[location] = []
			
			locations[location].append({
				'asset_id': asset.asset_id,
				'asset_number': asset.asset_number,
				'asset_name': asset.asset_name,
				'category': asset.category.category_name if asset.category else None,
				'custodian': asset.custodian,
				'acquisition_cost': float(asset.acquisition_cost),
				'current_book_value': float(asset.current_book_value),
				'status': asset.status
			})
		
		return locations
	
	# Utility Methods
	
	def _calculate_months_between(self, start_date: date, end_date: date) -> int:
		"""Calculate months between two dates"""
		return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
	
	def _generate_asset_number(self) -> str:
		"""Generate next asset number"""
		last_asset = CFAMAsset.query.filter_by(tenant_id=self.tenant_id)\
			.order_by(CFAMAsset.asset_number.desc()).first()
		
		if last_asset and last_asset.asset_number.isdigit():
			next_number = int(last_asset.asset_number) + 1
			return f"{next_number:06d}"
		
		return "000001"
	
	def _generate_acquisition_number(self) -> str:
		"""Generate acquisition number"""
		today = date.today()
		prefix = f"ACQ{today.strftime('%Y%m')}"
		
		last_acquisition = CFAMAssetAcquisition.query.filter(
			CFAMAssetAcquisition.tenant_id == self.tenant_id,
			CFAMAssetAcquisition.acquisition_number.like(f"{prefix}%")
		).order_by(CFAMAssetAcquisition.acquisition_number.desc()).first()
		
		if last_acquisition:
			last_seq = int(last_acquisition.acquisition_number[-4:])
			return f"{prefix}{last_seq + 1:04d}"
		
		return f"{prefix}0001"
	
	def _generate_disposal_number(self) -> str:
		"""Generate disposal number"""
		today = date.today()
		prefix = f"DSP{today.strftime('%Y%m')}"
		
		last_disposal = CFAMAssetDisposal.query.filter(
			CFAMAssetDisposal.tenant_id == self.tenant_id,
			CFAMAssetDisposal.disposal_number.like(f"{prefix}%")
		).order_by(CFAMAssetDisposal.disposal_number.desc()).first()
		
		if last_disposal:
			last_seq = int(last_disposal.disposal_number[-4:])
			return f"{prefix}{last_seq + 1:04d}"
		
		return f"{prefix}0001"
	
	def _generate_transfer_number(self) -> str:
		"""Generate transfer number"""
		today = date.today()
		prefix = f"TRF{today.strftime('%Y%m')}"
		
		last_transfer = CFAMAssetTransfer.query.filter(
			CFAMAssetTransfer.tenant_id == self.tenant_id,
			CFAMAssetTransfer.transfer_number.like(f"{prefix}%")
		).order_by(CFAMAssetTransfer.transfer_number.desc()).first()
		
		if last_transfer:
			last_seq = int(last_transfer.transfer_number[-4:])
			return f"{prefix}{last_seq + 1:04d}"
		
		return f"{prefix}0001"
	
	def _generate_maintenance_number(self) -> str:
		"""Generate maintenance number"""
		today = date.today()
		prefix = f"MNT{today.strftime('%Y%m')}"
		
		last_maintenance = CFAMAssetMaintenance.query.filter(
			CFAMAssetMaintenance.tenant_id == self.tenant_id,
			CFAMAssetMaintenance.maintenance_number.like(f"{prefix}%")
		).order_by(CFAMAssetMaintenance.maintenance_number.desc()).first()
		
		if last_maintenance:
			last_seq = int(last_maintenance.maintenance_number[-4:])
			return f"{prefix}{last_seq + 1:04d}"
		
		return f"{prefix}0001"
	
	def _generate_journal_number(self) -> str:
		"""Generate GL journal entry number"""
		today = date.today()
		prefix = f"FAM{today.strftime('%Y%m')}"
		
		last_entry = CFGLJournalEntry.query.filter(
			CFGLJournalEntry.tenant_id == self.tenant_id,
			CFGLJournalEntry.journal_number.like(f"{prefix}%")
		).order_by(CFGLJournalEntry.journal_number.desc()).first()
		
		if last_entry:
			last_seq = int(last_entry.journal_number[-4:])
			return f"{prefix}{last_seq + 1:04d}"
		
		return f"{prefix}0001"
	
	def _log_asset_activity(self, asset_id: str, activity_type: str, description: str, user_id: str = None):
		"""Log asset activity for audit trail"""
		# This would integrate with an audit logging system
		# For now, we'll add it to the asset notes
		asset = self.get_asset(asset_id)
		if asset:
			timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
			log_entry = f"[{timestamp}] {activity_type}: {description}"
			if user_id:
				log_entry += f" (User: {user_id})"
			
			if asset.notes:
				asset.notes += f"\n{log_entry}"
			else:
				asset.notes = log_entry
			
			db.session.commit()