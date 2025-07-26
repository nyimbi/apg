"""
Cost Accounting Service

Business logic for Cost Accounting operations including cost allocation,
activity-based costing, job costing, standard costing, and variance analysis.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, extract
import json
from dataclasses import dataclass

from .models import (
	CFCACostCenter, CFCACostCategory, CFCACostDriver, CFCACostAllocation,
	CFCACostPool, CFCAActivity, CFCAActivityCost, CFCAProductCost,
	CFCAJobCost, CFCAStandardCost, CFCAVarianceAnalysis
)
from ..general_ledger.models import CFGLAccount, CFGLPosting
from ...auth_rbac.models import db


@dataclass
class CostAllocationRequest:
	"""Cost allocation request parameters"""
	source_center_id: str
	allocation_method: str
	cost_amount: Decimal
	period: str
	allocation_basis: Optional[str] = None
	target_centers: Optional[List[str]] = None
	cost_driver_id: Optional[str] = None


@dataclass
class JobCostSummary:
	"""Job cost summary data"""
	job_number: str
	job_name: str
	total_cost: Decimal
	budgeted_cost: Decimal
	percent_complete: Decimal
	profitability: Dict[str, Any]


@dataclass
class VarianceReport:
	"""Variance analysis report data"""
	cost_object: str
	period: str
	total_variance: Decimal
	variance_percent: Decimal
	is_significant: bool
	primary_variances: List[Dict[str, Any]]


class CostAccountingService:
	"""Service class for Cost Accounting operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
		"""Log cost accounting operations for audit trail"""
		print(f"[Cost Accounting] {operation}: {details}")
	
	# Cost Center Management
	
	def create_cost_center(self, center_data: Dict[str, Any]) -> CFCACostCenter:
		"""Create a new cost center"""
		assert 'center_code' in center_data, "Center code is required"
		assert 'center_name' in center_data, "Center name is required"
		assert 'center_type' in center_data, "Center type is required"
		
		cost_center = CFCACostCenter(
			tenant_id=self.tenant_id,
			center_code=center_data['center_code'],
			center_name=center_data['center_name'],
			description=center_data.get('description'),
			parent_center_id=center_data.get('parent_center_id'),
			center_type=center_data['center_type'],
			responsibility_type=center_data.get('responsibility_type', 'Cost'),
			manager_name=center_data.get('manager_name'),
			manager_email=center_data.get('manager_email'),
			department=center_data.get('department'),
			location=center_data.get('location'),
			annual_budget=center_data.get('annual_budget', Decimal('0')),
			effective_date=center_data.get('effective_date', date.today()),
			allow_cost_allocation=center_data.get('allow_cost_allocation', True),
			require_job_number=center_data.get('require_job_number', False)
		)
		
		# Calculate hierarchy level and path
		if cost_center.parent_center_id:
			parent = self.get_cost_center(cost_center.parent_center_id)
			if parent:
				cost_center.level = parent.level + 1
				cost_center.path = f"{parent.path}/{center_data['center_code']}"
		else:
			cost_center.level = 0
			cost_center.path = center_data['center_code']
		
		db.session.add(cost_center)
		db.session.commit()
		
		self._log_operation('create_cost_center', {
			'center_code': cost_center.center_code,
			'center_name': cost_center.center_name,
			'center_type': cost_center.center_type
		})
		
		return cost_center
	
	def get_cost_center(self, center_id: str) -> Optional[CFCACostCenter]:
		"""Get cost center by ID"""
		return CFCACostCenter.query.filter_by(
			tenant_id=self.tenant_id,
			cost_center_id=center_id,
			is_active=True
		).first()
	
	def get_cost_centers_hierarchy(self) -> List[Dict[str, Any]]:
		"""Get cost centers in hierarchical structure"""
		centers = CFCACostCenter.query.filter_by(
			tenant_id=self.tenant_id,
			is_active=True
		).order_by(CFCACostCenter.level, CFCACostCenter.center_code).all()
		
		def build_hierarchy(parent_id=None, level=0):
			hierarchy = []
			for center in centers:
				if center.parent_center_id == parent_id:
					center_dict = {
						'cost_center_id': center.cost_center_id,
						'center_code': center.center_code,
						'center_name': center.center_name,
						'center_type': center.center_type,
						'level': level,
						'budget_variance': center.calculate_budget_variance(),
						'children': build_hierarchy(center.cost_center_id, level + 1)
					}
					hierarchy.append(center_dict)
			return hierarchy
		
		return build_hierarchy()
	
	# Cost Allocation
	
	def create_cost_allocation_rule(self, allocation_data: Dict[str, Any]) -> CFCACostAllocation:
		"""Create a cost allocation rule"""
		assert 'allocation_code' in allocation_data, "Allocation code is required"
		assert 'source_center_id' in allocation_data, "Source center is required"
		assert 'allocation_method' in allocation_data, "Allocation method is required"
		
		allocation = CFCACostAllocation(
			tenant_id=self.tenant_id,
			allocation_code=allocation_data['allocation_code'],
			allocation_name=allocation_data['allocation_name'],
			description=allocation_data.get('description'),
			source_center_id=allocation_data['source_center_id'],
			target_center_id=allocation_data.get('target_center_id'),
			cost_category_id=allocation_data.get('cost_category_id'),
			allocation_method=allocation_data['allocation_method'],
			cost_driver_id=allocation_data.get('cost_driver_id'),
			allocation_basis=allocation_data.get('allocation_basis'),
			allocation_percent=allocation_data.get('allocation_percent'),
			allocation_formula=allocation_data.get('allocation_formula'),
			allocation_frequency=allocation_data.get('allocation_frequency', 'Monthly'),
			effective_date=allocation_data.get('effective_date', date.today()),
			is_automatic=allocation_data.get('is_automatic', False),
			requires_approval=allocation_data.get('requires_approval', True)
		)
		
		db.session.add(allocation)
		db.session.commit()
		
		self._log_operation('create_allocation_rule', {
			'allocation_code': allocation.allocation_code,
			'method': allocation.allocation_method,
			'source_center': allocation.source_center_id
		})
		
		return allocation
	
	def execute_cost_allocation(self, request: CostAllocationRequest) -> Dict[str, Any]:
		"""Execute cost allocation based on rules and parameters"""
		assert request.cost_amount > 0, "Cost amount must be positive"
		
		allocation_results = []
		total_allocated = Decimal('0')
		
		if request.allocation_method == 'Direct':
			# Direct allocation to specific targets
			if request.target_centers:
				amount_per_center = request.cost_amount / len(request.target_centers)
				for target_id in request.target_centers:
					allocation_results.append({
						'target_center_id': target_id,
						'allocated_amount': amount_per_center,
						'allocation_basis': 'Equal Distribution'
					})
					total_allocated += amount_per_center
		
		elif request.allocation_method == 'Driver_Based':
			# Allocate based on cost driver quantities
			if request.cost_driver_id:
				driver_data = self._get_cost_driver_data(request.cost_driver_id, request.period)
				total_driver_quantity = sum(d['quantity'] for d in driver_data)
				
				if total_driver_quantity > 0:
					for data in driver_data:
						allocation_ratio = data['quantity'] / total_driver_quantity
						allocated_amount = request.cost_amount * allocation_ratio
						allocation_results.append({
							'target_center_id': data['center_id'],
							'allocated_amount': allocated_amount,
							'allocation_basis': f"{data['quantity']} / {total_driver_quantity} units",
							'driver_quantity': data['quantity']
						})
						total_allocated += allocated_amount
		
		elif request.allocation_method == 'Percentage':
			# Allocate based on predefined percentages
			allocation_rules = self._get_allocation_rules(request.source_center_id, request.allocation_method)
			for rule in allocation_rules:
				if rule.allocation_percent:
					allocated_amount = request.cost_amount * (rule.allocation_percent / 100)
					allocation_results.append({
						'target_center_id': rule.target_center_id,
						'allocated_amount': allocated_amount,
						'allocation_basis': f"{rule.allocation_percent}%",
						'rule_id': rule.allocation_id
					})
					total_allocated += allocated_amount
		
		# Create allocation entries (simplified - would integrate with GL)
		self._create_allocation_entries(request, allocation_results)
		
		result = {
			'allocation_id': f"ALLOC_{request.period}_{datetime.now().strftime('%H%M%S')}",
			'source_center_id': request.source_center_id,
			'total_cost': request.cost_amount,
			'total_allocated': total_allocated,
			'unallocated_amount': request.cost_amount - total_allocated,
			'allocation_method': request.allocation_method,
			'period': request.period,
			'allocations': allocation_results,
			'execution_date': datetime.now().isoformat()
		}
		
		self._log_operation('execute_cost_allocation', {
			'method': request.allocation_method,
			'amount': request.cost_amount,
			'targets': len(allocation_results)
		})
		
		return result
	
	def _get_cost_driver_data(self, driver_id: str, period: str) -> List[Dict[str, Any]]:
		"""Get cost driver data for allocation period"""
		# Simplified - would query actual driver data from operational systems
		return [
			{'center_id': 'CC001', 'quantity': Decimal('100')},
			{'center_id': 'CC002', 'quantity': Decimal('150')},
			{'center_id': 'CC003', 'quantity': Decimal('75')}
		]
	
	def _get_allocation_rules(self, source_center_id: str, method: str) -> List[CFCACostAllocation]:
		"""Get allocation rules for source center and method"""
		return CFCACostAllocation.query.filter_by(
			tenant_id=self.tenant_id,
			source_center_id=source_center_id,
			allocation_method=method,
			is_active=True
		).all()
	
	def _create_allocation_entries(self, request: CostAllocationRequest, allocations: List[Dict[str, Any]]) -> None:
		"""Create allocation entries (would integrate with GL postings)"""
		# Simplified implementation - would create actual GL postings
		for allocation in allocations:
			print(f"Allocation Entry: {allocation}")
	
	# Activity-Based Costing (ABC)
	
	def create_activity(self, activity_data: Dict[str, Any]) -> CFCAActivity:
		"""Create an activity for ABC"""
		assert 'activity_code' in activity_data, "Activity code is required"
		assert 'activity_name' in activity_data, "Activity name is required"
		assert 'activity_type' in activity_data, "Activity type is required"
		
		activity = CFCAActivity(
			tenant_id=self.tenant_id,
			activity_code=activity_data['activity_code'],
			activity_name=activity_data['activity_name'],
			description=activity_data.get('description'),
			activity_type=activity_data['activity_type'],
			value_category=activity_data.get('value_category', 'Value-Added'),
			cost_pool_id=activity_data.get('cost_pool_id'),
			cost_center_id=activity_data.get('cost_center_id'),
			primary_driver_id=activity_data.get('primary_driver_id'),
			capacity_measure=activity_data.get('capacity_measure'),
			practical_capacity=activity_data.get('practical_capacity'),
			current_capacity=activity_data.get('current_capacity'),
			estimated_cost_per_unit=activity_data.get('estimated_cost_per_unit'),
			setup_time_minutes=activity_data.get('setup_time_minutes'),
			processing_time_minutes=activity_data.get('processing_time_minutes'),
			quality_rating=activity_data.get('quality_rating'),
			efficiency_rating=activity_data.get('efficiency_rating'),
			cycle_time_minutes=activity_data.get('cycle_time_minutes'),
			effective_date=activity_data.get('effective_date', date.today()),
			is_value_added=activity_data.get('is_value_added', True)
		)
		
		db.session.add(activity)
		db.session.commit()
		
		self._log_operation('create_activity', {
			'activity_code': activity.activity_code,
			'activity_name': activity.activity_name,
			'activity_type': activity.activity_type
		})
		
		return activity
	
	def calculate_activity_costs(self, period: str) -> List[Dict[str, Any]]:
		"""Calculate activity costs for ABC"""
		activities = CFCAActivity.query.filter_by(
			tenant_id=self.tenant_id,
			is_active=True
		).all()
		
		activity_costs = []
		
		for activity in activities:
			# Get resource costs consumed by this activity
			resource_costs = self._get_activity_resource_consumption(activity.activity_id, period)
			
			# Calculate activity rate
			total_cost = sum(rc['cost'] for rc in resource_costs)
			total_activity_volume = self._get_activity_volume(activity.activity_id, period)
			activity_rate = activity.calculate_activity_rate(total_cost, total_activity_volume)
			
			activity_cost_data = {
				'activity_id': activity.activity_id,
				'activity_code': activity.activity_code,
				'activity_name': activity.activity_name,
				'period': period,
				'total_cost': total_cost,
				'activity_volume': total_activity_volume,
				'activity_rate': activity_rate,
				'resource_costs': resource_costs,
				'capacity_utilization': activity.get_capacity_utilization(total_activity_volume) if activity.practical_capacity else None
			}
			
			activity_costs.append(activity_cost_data)
		
		self._log_operation('calculate_activity_costs', {
			'period': period,
			'activities_processed': len(activity_costs)
		})
		
		return activity_costs
	
	def _get_activity_resource_consumption(self, activity_id: str, period: str) -> List[Dict[str, Any]]:
		"""Get resource costs consumed by activity"""
		# Simplified - would query actual resource consumption data
		return [
			{'resource_type': 'Labor', 'cost': Decimal('5000')},
			{'resource_type': 'Materials', 'cost': Decimal('2000')},
			{'resource_type': 'Equipment', 'cost': Decimal('1500')}
		]
	
	def _get_activity_volume(self, activity_id: str, period: str) -> Decimal:
		"""Get activity volume for period"""
		# Simplified - would query actual activity volume data
		return Decimal('100')
	
	def allocate_activity_costs_to_products(self, period: str) -> Dict[str, Any]:
		"""Allocate activity costs to products using ABC"""
		activity_costs = self.calculate_activity_costs(period)
		products = self._get_products_for_period(period)
		
		product_allocations = {}
		
		for product in products:
			product_code = product['code']
			product_allocations[product_code] = {
				'product_code': product_code,
				'product_name': product['name'],
				'total_allocated_cost': Decimal('0'),
				'activity_allocations': []
			}
			
			for activity_cost in activity_costs:
				# Get product's consumption of this activity
				activity_consumption = self._get_product_activity_consumption(
					product_code, activity_cost['activity_id'], period
				)
				
				if activity_consumption > 0:
					allocated_cost = activity_cost['activity_rate'] * activity_consumption
					
					product_allocations[product_code]['activity_allocations'].append({
						'activity_code': activity_cost['activity_code'],
						'activity_name': activity_cost['activity_name'],
						'consumption': activity_consumption,
						'activity_rate': activity_cost['activity_rate'],
						'allocated_cost': allocated_cost
					})
					
					product_allocations[product_code]['total_allocated_cost'] += allocated_cost
		
		result = {
			'period': period,
			'total_activity_costs': sum(ac['total_cost'] for ac in activity_costs),
			'products_costed': len(product_allocations),
			'product_allocations': list(product_allocations.values())
		}
		
		self._log_operation('allocate_activity_costs', {
			'period': period,
			'products': len(product_allocations),
			'total_cost': result['total_activity_costs']
		})
		
		return result
	
	def _get_products_for_period(self, period: str) -> List[Dict[str, Any]]:
		"""Get products produced in period"""
		# Simplified - would query actual production data
		return [
			{'code': 'PROD001', 'name': 'Product A', 'quantity': 100},
			{'code': 'PROD002', 'name': 'Product B', 'quantity': 75},
			{'code': 'PROD003', 'name': 'Product C', 'quantity': 50}
		]
	
	def _get_product_activity_consumption(self, product_code: str, activity_id: str, period: str) -> Decimal:
		"""Get product's consumption of activity"""
		# Simplified - would query actual consumption data
		consumption_map = {
			('PROD001', 'activity_id'): Decimal('25'),
			('PROD002', 'activity_id'): Decimal('30'),
			('PROD003', 'activity_id'): Decimal('20')
		}
		return consumption_map.get((product_code, activity_id), Decimal('0'))
	
	# Job Costing
	
	def create_job_cost(self, job_data: Dict[str, Any]) -> CFCAJobCost:
		"""Create a new job cost record"""
		assert 'job_number' in job_data, "Job number is required"
		assert 'cost_center_id' in job_data, "Cost center is required"
		assert 'cost_category_id' in job_data, "Cost category is required"
		
		job_cost = CFCAJobCost(
			tenant_id=self.tenant_id,
			job_number=job_data['job_number'],
			job_name=job_data['job_name'],
			job_description=job_data.get('job_description'),
			cost_center_id=job_data['cost_center_id'],
			cost_category_id=job_data['cost_category_id'],
			customer_code=job_data.get('customer_code'),
			customer_name=job_data.get('customer_name'),
			project_code=job_data.get('project_code'),
			contract_number=job_data.get('contract_number'),
			start_date=job_data['start_date'],
			planned_completion_date=job_data.get('planned_completion_date'),
			budgeted_cost=job_data.get('budgeted_cost', Decimal('0')),
			budgeted_hours=job_data.get('budgeted_hours', Decimal('0')),
			contract_value=job_data.get('contract_value'),
			billing_method=job_data.get('billing_method'),
			job_status=job_data.get('job_status', 'Active'),
			is_billable=job_data.get('is_billable', True)
		)
		
		db.session.add(job_cost)
		db.session.commit()
		
		self._log_operation('create_job_cost', {
			'job_number': job_cost.job_number,
			'job_name': job_cost.job_name,
			'budgeted_cost': job_cost.budgeted_cost
		})
		
		return job_cost
	
	def update_job_costs(self, job_number: str, cost_updates: Dict[str, Any]) -> List[CFCAJobCost]:
		"""Update job costs with actual expenses"""
		job_costs = CFCAJobCost.query.filter_by(
			tenant_id=self.tenant_id,
			job_number=job_number
		).all()
		
		updated_jobs = []
		
		for job_cost in job_costs:
			if job_cost.cost_category.category_code in cost_updates:
				updates = cost_updates[job_cost.cost_category.category_code]
				
				# Update actual costs
				if 'material_cost' in updates:
					job_cost.actual_material_cost += updates['material_cost']
				if 'labor_cost' in updates:
					job_cost.actual_labor_cost += updates['labor_cost']
				if 'overhead_cost' in updates:
					job_cost.actual_overhead_cost += updates['overhead_cost']
				if 'other_cost' in updates:
					job_cost.actual_other_cost += updates['other_cost']
				
				# Update hours
				if 'labor_hours' in updates:
					job_cost.actual_labor_hours += updates['labor_hours']
				if 'machine_hours' in updates:
					job_cost.actual_machine_hours += updates['machine_hours']
				
				# Update committed costs
				if 'committed_material' in updates:
					job_cost.committed_material_cost += updates['committed_material']
				if 'committed_labor' in updates:
					job_cost.committed_labor_cost += updates['committed_labor']
				if 'committed_other' in updates:
					job_cost.committed_other_cost += updates['committed_other']
				
				updated_jobs.append(job_cost)
		
		db.session.commit()
		
		# Recalculate percent complete if requested
		if 'percent_complete' in cost_updates:
			self._update_job_percent_complete(job_number, cost_updates['percent_complete'])
		
		self._log_operation('update_job_costs', {
			'job_number': job_number,
			'categories_updated': len(updated_jobs)
		})
		
		return updated_jobs
	
	def _update_job_percent_complete(self, job_number: str, percent_complete: Decimal) -> None:
		"""Update job percent complete"""
		CFCAJobCost.query.filter_by(
			tenant_id=self.tenant_id,
			job_number=job_number
		).update({'percent_complete': percent_complete})
		db.session.commit()
	
	def get_job_cost_summary(self, job_number: str) -> JobCostSummary:
		"""Get comprehensive job cost summary"""
		job_costs = CFCAJobCost.query.filter_by(
			tenant_id=self.tenant_id,
			job_number=job_number
		).all()
		
		if not job_costs:
			raise ValueError(f"Job {job_number} not found")
		
		# Aggregate costs across all categories
		total_actual = sum(jc.calculate_total_actual_cost() for jc in job_costs)
		total_budgeted = sum(jc.budgeted_cost for jc in job_costs)
		
		# Get profitability from first job cost record
		profitability = job_costs[0].get_job_profitability()
		percent_complete = job_costs[0].percent_complete
		
		return JobCostSummary(
			job_number=job_number,
			job_name=job_costs[0].job_name,
			total_cost=total_actual,
			budgeted_cost=total_budgeted,
			percent_complete=percent_complete,
			profitability=profitability
		)
	
	def get_jobs_by_status(self, status: str = None) -> List[Dict[str, Any]]:
		"""Get jobs filtered by status"""
		query = CFCAJobCost.query.filter_by(tenant_id=self.tenant_id)
		
		if status:
			query = query.filter_by(job_status=status)
		
		jobs = query.distinct(CFCAJobCost.job_number).all()
		
		job_summaries = []
		for job in jobs:
			try:
				summary = self.get_job_cost_summary(job.job_number)
				job_summaries.append({
					'job_number': summary.job_number,
					'job_name': summary.job_name,
					'status': job.job_status,
					'total_cost': summary.total_cost,
					'budgeted_cost': summary.budgeted_cost,
					'percent_complete': summary.percent_complete,
					'is_profitable': summary.profitability['is_profitable'],
					'is_over_budget': summary.profitability['is_over_budget'],
					'profit_margin': summary.profitability['profit_margin_percent']
				})
			except ValueError:
				continue
		
		return job_summaries
	
	# Standard Costing and Variance Analysis
	
	def create_standard_cost(self, standard_data: Dict[str, Any]) -> CFCAStandardCost:
		"""Create a standard cost"""
		assert 'cost_object_type' in standard_data, "Cost object type is required"
		assert 'cost_object_code' in standard_data, "Cost object code is required"
		assert 'cost_category_id' in standard_data, "Cost category is required"
		assert 'standard_cost_per_unit' in standard_data, "Standard cost per unit is required"
		
		standard_cost = CFCAStandardCost(
			tenant_id=self.tenant_id,
			cost_object_type=standard_data['cost_object_type'],
			cost_object_code=standard_data['cost_object_code'],
			cost_object_name=standard_data['cost_object_name'],
			cost_category_id=standard_data['cost_category_id'],
			cost_center_id=standard_data.get('cost_center_id'),
			standard_cost_per_unit=standard_data['standard_cost_per_unit'],
			standard_quantity_per_unit=standard_data.get('standard_quantity_per_unit', Decimal('1')),
			standard_rate_per_quantity=standard_data['standard_rate_per_quantity'],
			unit_of_measure=standard_data['unit_of_measure'],
			quantity_unit_of_measure=standard_data.get('quantity_unit_of_measure'),
			effective_date=standard_data.get('effective_date', date.today()),
			fiscal_year=standard_data.get('fiscal_year', date.today().year),
			version=standard_data.get('version', '1.0'),
			standard_type=standard_data.get('standard_type', 'Attainable'),
			revision_reason=standard_data.get('revision_reason'),
			favorable_variance_threshold=standard_data.get('favorable_variance_threshold', Decimal('5')),
			unfavorable_variance_threshold=standard_data.get('unfavorable_variance_threshold', Decimal('5')),
			is_approved=standard_data.get('is_approved', False),
			approved_by=standard_data.get('approved_by'),
			approved_date=standard_data.get('approved_date')
		)
		
		db.session.add(standard_cost)
		db.session.commit()
		
		self._log_operation('create_standard_cost', {
			'cost_object': standard_cost.cost_object_code,
			'cost_per_unit': standard_cost.standard_cost_per_unit,
			'effective_date': standard_cost.effective_date
		})
		
		return standard_cost
	
	def perform_variance_analysis(self, analysis_data: Dict[str, Any]) -> CFCAVarianceAnalysis:
		"""Perform variance analysis comparing actual to standard costs"""
		assert 'standard_cost_id' in analysis_data, "Standard cost ID is required"
		assert 'actual_cost' in analysis_data, "Actual cost is required"
		assert 'actual_quantity' in analysis_data, "Actual quantity is required"
		assert 'analysis_period' in analysis_data, "Analysis period is required"
		
		# Get standard cost
		standard_cost = CFCAStandardCost.query.filter_by(
			tenant_id=self.tenant_id,
			standard_cost_id=analysis_data['standard_cost_id']
		).first()
		
		if not standard_cost:
			raise ValueError("Standard cost not found")
		
		# Calculate variances
		actual_cost = analysis_data['actual_cost']
		actual_quantity = analysis_data['actual_quantity']
		actual_rate = actual_cost / actual_quantity if actual_quantity > 0 else Decimal('0')
		
		standard_quantity = actual_quantity  # Assuming standard quantity equals actual for this analysis
		standard_cost_total = standard_cost.calculate_standard_cost(actual_quantity)
		standard_rate = standard_cost.standard_rate_per_quantity
		
		total_variance = actual_cost - standard_cost_total
		
		# Price/Rate Variance: (Actual Rate - Standard Rate) × Actual Quantity
		price_rate_variance = (actual_rate - standard_rate) * actual_quantity
		
		# For quantity variance, we'd need budgeted vs actual quantity
		# Simplified here as zero since we're using actual quantity as standard
		quantity_efficiency_variance = Decimal('0')
		
		# Volume variance (for overhead) - simplified
		volume_variance = Decimal('0')
		
		variance_percent = (total_variance / standard_cost_total * 100) if standard_cost_total > 0 else Decimal('0')
		is_favorable = total_variance < 0
		is_significant = abs(variance_percent) > standard_cost.unfavorable_variance_threshold
		
		variance_analysis = CFCAVarianceAnalysis(
			tenant_id=self.tenant_id,
			standard_cost_id=analysis_data['standard_cost_id'],
			cost_center_id=analysis_data.get('cost_center_id'),
			analysis_period=analysis_data['analysis_period'],
			fiscal_year=int(analysis_data['analysis_period'][:4]),
			fiscal_period=int(analysis_data['analysis_period'][5:7]),
			analysis_date=analysis_data.get('analysis_date', date.today()),
			cost_object_type=standard_cost.cost_object_type,
			cost_object_code=standard_cost.cost_object_code,
			cost_object_name=standard_cost.cost_object_name,
			actual_quantity=actual_quantity,
			actual_cost=actual_cost,
			actual_rate=actual_rate,
			standard_quantity=standard_quantity,
			standard_cost=standard_cost_total,
			standard_rate=standard_rate,
			total_variance=total_variance,
			price_rate_variance=price_rate_variance,
			quantity_efficiency_variance=quantity_efficiency_variance,
			volume_variance=volume_variance,
			is_favorable=is_favorable,
			is_significant=is_significant,
			variance_percent=variance_percent,
			variance_type=analysis_data.get('variance_type', 'Cost'),
			primary_cause=analysis_data.get('primary_cause'),
			requires_action=is_significant,
			resolution_status=analysis_data.get('resolution_status', 'Open')
		)
		
		db.session.add(variance_analysis)
		db.session.commit()
		
		self._log_operation('perform_variance_analysis', {
			'cost_object': variance_analysis.cost_object_code,
			'period': variance_analysis.analysis_period,
			'total_variance': total_variance,
			'is_significant': is_significant
		})
		
		return variance_analysis
	
	def get_variance_report(self, period: str, cost_object_type: str = None) -> List[VarianceReport]:
		"""Get variance analysis report for period"""
		query = CFCAVarianceAnalysis.query.filter_by(
			tenant_id=self.tenant_id,
			analysis_period=period
		)
		
		if cost_object_type:
			query = query.filter_by(cost_object_type=cost_object_type)
		
		variances = query.all()
		
		# Group by cost object
		variance_groups = {}
		for variance in variances:
			key = variance.cost_object_code
			if key not in variance_groups:
				variance_groups[key] = {
					'cost_object': variance.cost_object_name,
					'variances': [],
					'total_variance': Decimal('0'),
					'total_standard': Decimal('0')
				}
			
			variance_groups[key]['variances'].append(variance)
			variance_groups[key]['total_variance'] += variance.total_variance
			variance_groups[key]['total_standard'] += variance.standard_cost
		
		# Create variance reports
		reports = []
		for cost_object, data in variance_groups.items():
			total_variance_percent = (data['total_variance'] / data['total_standard'] * 100) if data['total_standard'] > 0 else Decimal('0')
			
			primary_variances = []
			for variance in data['variances']:
				if variance.is_significant:
					primary_variances.append({
						'variance_type': variance.variance_type,
						'variance_amount': variance.total_variance,
						'variance_percent': variance.variance_percent,
						'is_favorable': variance.is_favorable,
						'primary_cause': variance.primary_cause,
						'requires_action': variance.requires_action
					})
			
			report = VarianceReport(
				cost_object=data['cost_object'],
				period=period,
				total_variance=data['total_variance'],
				variance_percent=total_variance_percent,
				is_significant=abs(total_variance_percent) > 5,
				primary_variances=primary_variances
			)
			
			reports.append(report)
		
		return reports
	
	# Reporting and Analytics
	
	def get_cost_center_performance(self, center_id: str, period: str) -> Dict[str, Any]:
		"""Get comprehensive cost center performance metrics"""
		cost_center = self.get_cost_center(center_id)
		if not cost_center:
			raise ValueError("Cost center not found")
		
		# Get budget variance
		budget_variance = cost_center.calculate_budget_variance()
		
		# Get job costs for this center and period
		job_costs = CFCAJobCost.query.filter_by(
			tenant_id=self.tenant_id,
			cost_center_id=center_id
		).filter(
			CFCAJobCost.start_date <= datetime.strptime(f"{period}-01", "%Y-%m-%d").date()
		).all()
		
		# Calculate job performance metrics
		active_jobs = [jc for jc in job_costs if jc.job_status == 'Active']
		completed_jobs = [jc for jc in job_costs if jc.job_status == 'Complete']
		
		total_job_cost = sum(jc.calculate_total_actual_cost() for jc in job_costs)
		total_job_budget = sum(jc.budgeted_cost for jc in job_costs)
		
		performance_data = {
			'cost_center': {
				'center_code': cost_center.center_code,
				'center_name': cost_center.center_name,
				'center_type': cost_center.center_type,
				'manager': cost_center.manager_name
			},
			'budget_performance': budget_variance,
			'job_performance': {
				'active_jobs': len(active_jobs),
				'completed_jobs': len(completed_jobs),
				'total_jobs': len(job_costs),
				'total_actual_cost': total_job_cost,
				'total_budgeted_cost': total_job_budget,
				'cost_variance': total_job_cost - total_job_budget,
				'average_job_size': total_job_cost / len(job_costs) if job_costs else 0
			},
			'period': period,
			'analysis_date': date.today().isoformat()
		}
		
		return performance_data
	
	def get_abc_profitability_analysis(self, period: str) -> Dict[str, Any]:
		"""Get product profitability analysis using ABC"""
		abc_allocation = self.allocate_activity_costs_to_products(period)
		
		profitability_analysis = {
			'period': period,
			'total_activity_costs': abc_allocation['total_activity_costs'],
			'products': []
		}
		
		for product_alloc in abc_allocation['product_allocations']:
			# Would need revenue data to calculate true profitability
			# Simplified here to show cost structure
			product_analysis = {
				'product_code': product_alloc['product_code'],
				'product_name': product_alloc['product_name'],
				'total_abc_cost': product_alloc['total_allocated_cost'],
				'activity_breakdown': product_alloc['activity_allocations'],
				'cost_per_unit': 0,  # Would calculate with production volume
				'margin_analysis': {
					'revenue': 0,  # Would get from sales data
					'gross_margin': 0,
					'margin_percent': 0
				}
			}
			
			profitability_analysis['products'].append(product_analysis)
		
		return profitability_analysis
	
	def generate_cost_dashboard_data(self, period: str) -> Dict[str, Any]:
		"""Generate comprehensive cost accounting dashboard data"""
		# Get key metrics
		total_cost_centers = CFCACostCenter.query.filter_by(
			tenant_id=self.tenant_id,
			is_active=True
		).count()
		
		active_jobs = CFCAJobCost.query.filter_by(
			tenant_id=self.tenant_id,
			job_status='Active'
		).count()
		
		significant_variances = CFCAVarianceAnalysis.query.filter_by(
			tenant_id=self.tenant_id,
			analysis_period=period,
			is_significant=True
		).count()
		
		# Cost center budget performance
		cost_centers = CFCACostCenter.query.filter_by(
			tenant_id=self.tenant_id,
			is_active=True
		).all()
		
		budget_performance = []
		for center in cost_centers[:10]:  # Top 10 for dashboard
			variance = center.calculate_budget_variance()
			budget_performance.append({
				'center_code': center.center_code,
				'center_name': center.center_name,
				'budget_variance': variance['variance_amount'],
				'variance_percent': variance['variance_percent'],
				'is_favorable': variance['is_favorable']
			})
		
		dashboard_data = {
			'period': period,
			'summary_metrics': {
				'total_cost_centers': total_cost_centers,
				'active_jobs': active_jobs,
				'significant_variances': significant_variances,
				'analysis_date': date.today().isoformat()
			},
			'budget_performance': sorted(budget_performance, 
										key=lambda x: abs(x['variance_percent']), 
										reverse=True),
			'top_variances': self._get_top_variances(period),
			'activity_utilization': self._get_activity_utilization(period),
			'job_status_summary': self._get_job_status_summary()
		}
		
		return dashboard_data
	
	def _get_top_variances(self, period: str) -> List[Dict[str, Any]]:
		"""Get top variances for dashboard"""
		variances = CFCAVarianceAnalysis.query.filter_by(
			tenant_id=self.tenant_id,
			analysis_period=period
		).order_by(desc(func.abs(CFCAVarianceAnalysis.variance_percent))).limit(5).all()
		
		return [{
			'cost_object': v.cost_object_name,
			'variance_type': v.variance_type,
			'variance_amount': v.total_variance,
			'variance_percent': v.variance_percent,
			'is_favorable': v.is_favorable,
			'is_significant': v.is_significant
		} for v in variances]
	
	def _get_activity_utilization(self, period: str) -> List[Dict[str, Any]]:
		"""Get activity utilization metrics"""
		activities = CFCAActivity.query.filter_by(
			tenant_id=self.tenant_id,
			is_active=True
		).limit(5).all()
		
		utilization_data = []
		for activity in activities:
			# Simplified utilization calculation
			actual_volume = self._get_activity_volume(activity.activity_id, period)
			utilization = activity.get_capacity_utilization(actual_volume)
			
			utilization_data.append({
				'activity_code': activity.activity_code,
				'activity_name': activity.activity_name,
				'utilization_percent': utilization.get('utilization_percent', 0),
				'is_bottleneck': utilization.get('is_over_capacity', False),
				'efficiency_rating': utilization.get('efficiency_rating', 'Unknown')
			})
		
		return utilization_data
	
	def _get_job_status_summary(self) -> Dict[str, int]:
		"""Get job status summary"""
		job_statuses = db.session.query(
			CFCAJobCost.job_status,
			func.count(func.distinct(CFCAJobCost.job_number))
		).filter_by(
			tenant_id=self.tenant_id
		).group_by(CFCAJobCost.job_status).all()
		
		return {status: count for status, count in job_statuses}