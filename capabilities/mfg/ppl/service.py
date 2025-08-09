"""
Production Planning Service

Business logic for production planning including master production scheduling,
production orders, demand forecasting, and resource capacity planning.
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from .models import (
	MFPMasterProductionSchedule, MFPProductionOrder, MFPDemandForecast, MFPResourceCapacity,
	MasterProductionScheduleCreate, ProductionOrderCreate, DemandForecastCreate, ResourceCapacityCreate,
	ProductionOrderStatus, SchedulingPriority, PlanningHorizon
)

class ProductionPlanningService:
	"""Service for production planning operations"""

	def __init__(self, db_session: AsyncSession):
		self.db = db_session

	async def create_master_schedule(
		self, 
		tenant_id: str, 
		schedule_data: MasterProductionScheduleCreate,
		user_id: str
	) -> MFPMasterProductionSchedule:
		"""Create a new master production schedule entry"""
		
		# Validate date logic
		if schedule_data.planned_end_date <= schedule_data.planned_start_date:
			raise ValueError("Planned end date must be after planned start date")
		
		# Calculate capacity utilization if both values provided
		capacity_utilization_pct = None
		if schedule_data.available_capacity and schedule_data.available_capacity > 0:
			capacity_utilization_pct = (schedule_data.planned_quantity / schedule_data.available_capacity) * 100
		
		schedule = MFPMasterProductionSchedule(
			tenant_id=tenant_id,
			schedule_name=schedule_data.schedule_name,
			planning_period=schedule_data.planning_period,
			planning_horizon=schedule_data.planning_horizon.value,
			product_id=schedule_data.product_id,
			facility_id=schedule_data.facility_id,
			production_line_id=schedule_data.production_line_id,
			planned_quantity=schedule_data.planned_quantity,
			planned_start_date=schedule_data.planned_start_date,
			planned_end_date=schedule_data.planned_end_date,
			forecast_demand=schedule_data.forecast_demand,
			available_capacity=schedule_data.available_capacity,
			capacity_utilization_pct=capacity_utilization_pct,
			priority=schedule_data.priority.value,
			safety_stock_days=schedule_data.safety_stock_days,
			lead_time_days=schedule_data.lead_time_days,
			batch_size_min=schedule_data.batch_size_min,
			batch_size_max=schedule_data.batch_size_max,
			created_by=user_id
		)
		
		self.db.add(schedule)
		await self.db.commit()
		await self.db.refresh(schedule)
		
		await self._log_schedule_creation(schedule)
		return schedule

	async def create_production_order(
		self,
		tenant_id: str,
		order_data: ProductionOrderCreate,
		user_id: str
	) -> MFPProductionOrder:
		"""Create a new production order"""
		
		# Validate date logic
		if order_data.scheduled_end_date <= order_data.scheduled_start_date:
			raise ValueError("Scheduled end date must be after scheduled start date")
		
		# Check for duplicate order number
		existing_order = await self.db.execute(
			select(MFPProductionOrder).where(
				and_(
					MFPProductionOrder.tenant_id == tenant_id,
					MFPProductionOrder.order_number == order_data.order_number
				)
			)
		)
		if existing_order.scalar_one_or_none():
			raise ValueError(f"Production order number {order_data.order_number} already exists")
		
		order = MFPProductionOrder(
			tenant_id=tenant_id,
			order_number=order_data.order_number,
			order_type=order_data.order_type,
			master_schedule_id=order_data.master_schedule_id,
			product_id=order_data.product_id,
			product_sku=order_data.product_sku,
			product_name=order_data.product_name,
			facility_id=order_data.facility_id,
			production_line_id=order_data.production_line_id,
			work_center_id=order_data.work_center_id,
			ordered_quantity=order_data.ordered_quantity,
			scheduled_start_date=order_data.scheduled_start_date,
			scheduled_end_date=order_data.scheduled_end_date,
			priority=order_data.priority.value,
			estimated_labor_hours=order_data.estimated_labor_hours,
			estimated_machine_hours=order_data.estimated_machine_hours,
			bom_id=order_data.bom_id,
			routing_id=order_data.routing_id,
			production_notes=order_data.production_notes,
			special_instructions=order_data.special_instructions,
			quality_requirements=order_data.quality_requirements,
			created_by=user_id
		)
		
		self.db.add(order)
		await self.db.commit()
		await self.db.refresh(order)
		
		await self._log_order_creation(order)
		return order

	async def update_production_order_status(
		self,
		tenant_id: str,
		order_id: str,
		new_status: ProductionOrderStatus,
		user_id: str,
		actual_start_date: Optional[datetime] = None,
		actual_end_date: Optional[datetime] = None
	) -> MFPProductionOrder:
		"""Update production order status and dates"""
		
		order = await self.get_production_order(tenant_id, order_id)
		if not order:
			raise ValueError(f"Production order {order_id} not found")
		
		old_status = order.status
		order.status = new_status.value
		order.updated_by = user_id
		order.updated_at = datetime.utcnow()
		
		# Update actual dates based on status
		if new_status == ProductionOrderStatus.IN_PROGRESS and actual_start_date:
			order.actual_start_date = actual_start_date
		elif new_status == ProductionOrderStatus.COMPLETED and actual_end_date:
			order.actual_end_date = actual_end_date
		
		await self.db.commit()
		await self.db.refresh(order)
		
		await self._log_status_change(order, old_status, new_status.value)
		return order

	async def create_demand_forecast(
		self,
		tenant_id: str,
		forecast_data: DemandForecastCreate,
		user_id: str
	) -> MFPDemandForecast:
		"""Create a new demand forecast"""
		
		# Validate date logic
		if forecast_data.period_end_date <= forecast_data.period_start_date:
			raise ValueError("Period end date must be after period start date")
		
		forecast = MFPDemandForecast(
			tenant_id=tenant_id,
			forecast_name=forecast_data.forecast_name,
			forecast_period=forecast_data.forecast_period,
			forecast_type=forecast_data.forecast_type,
			product_id=forecast_data.product_id,
			facility_id=forecast_data.facility_id,
			customer_id=forecast_data.customer_id,
			forecast_quantity=forecast_data.forecast_quantity,
			forecast_value=forecast_data.forecast_value,
			period_start_date=forecast_data.period_start_date,
			period_end_date=forecast_data.period_end_date,
			forecast_method=forecast_data.forecast_method,
			confidence_level=forecast_data.confidence_level,
			seasonality_factor=forecast_data.seasonality_factor,
			trend_factor=forecast_data.trend_factor,
			created_by=user_id
		)
		
		self.db.add(forecast)
		await self.db.commit()
		await self.db.refresh(forecast)
		
		await self._log_forecast_creation(forecast)
		return forecast

	async def create_resource_capacity(
		self,
		tenant_id: str,
		capacity_data: ResourceCapacityCreate,
		user_id: str
	) -> MFPResourceCapacity:
		"""Create or update resource capacity planning"""
		
		# Validate date logic
		if capacity_data.period_end_date <= capacity_data.period_start_date:
			raise ValueError("Period end date must be after period start date")
		
		# Check for existing capacity record for same resource and period
		existing = await self.db.execute(
			select(MFPResourceCapacity).where(
				and_(
					MFPResourceCapacity.tenant_id == tenant_id,
					MFPResourceCapacity.resource_id == capacity_data.resource_id,
					MFPResourceCapacity.planning_period == capacity_data.planning_period
				)
			)
		)
		existing_capacity = existing.scalar_one_or_none()
		
		if existing_capacity:
			# Update existing record
			existing_capacity.available_capacity = capacity_data.available_capacity
			existing_capacity.period_start_date = capacity_data.period_start_date
			existing_capacity.period_end_date = capacity_data.period_end_date
			existing_capacity.shifts_per_day = capacity_data.shifts_per_day
			existing_capacity.hours_per_shift = capacity_data.hours_per_shift
			existing_capacity.working_days_per_week = capacity_data.working_days_per_week
			existing_capacity.max_capacity = capacity_data.max_capacity
			existing_capacity.min_capacity = capacity_data.min_capacity
			existing_capacity.setup_time_hours = capacity_data.setup_time_hours
			existing_capacity.maintenance_time_hours = capacity_data.maintenance_time_hours
			existing_capacity.updated_by = user_id
			existing_capacity.updated_at = datetime.utcnow()
			
			capacity = existing_capacity
		else:
			# Create new record
			capacity = MFPResourceCapacity(
				tenant_id=tenant_id,
				resource_type=capacity_data.resource_type,
				resource_id=capacity_data.resource_id,
				resource_name=capacity_data.resource_name,
				facility_id=capacity_data.facility_id,
				work_center_id=capacity_data.work_center_id,
				planning_period=capacity_data.planning_period,
				capacity_unit=capacity_data.capacity_unit,
				available_capacity=capacity_data.available_capacity,
				period_start_date=capacity_data.period_start_date,
				period_end_date=capacity_data.period_end_date,
				shifts_per_day=capacity_data.shifts_per_day,
				hours_per_shift=capacity_data.hours_per_shift,
				working_days_per_week=capacity_data.working_days_per_week,
				max_capacity=capacity_data.max_capacity,
				min_capacity=capacity_data.min_capacity,
				setup_time_hours=capacity_data.setup_time_hours,
				maintenance_time_hours=capacity_data.maintenance_time_hours,
				created_by=user_id
			)
			self.db.add(capacity)
		
		await self.db.commit()
		await self.db.refresh(capacity)
		
		# Calculate utilization if planned capacity exists
		await self._calculate_capacity_utilization(capacity)
		return capacity

	async def get_production_orders_by_facility(
		self,
		tenant_id: str,
		facility_id: str,
		status_filter: Optional[List[ProductionOrderStatus]] = None,
		start_date: Optional[date] = None,
		end_date: Optional[date] = None
	) -> List[MFPProductionOrder]:
		"""Get production orders for a facility with optional filters"""
		
		query = select(MFPProductionOrder).where(
			and_(
				MFPProductionOrder.tenant_id == tenant_id,
				MFPProductionOrder.facility_id == facility_id
			)
		)
		
		if status_filter:
			status_values = [status.value for status in status_filter]
			query = query.where(MFPProductionOrder.status.in_(status_values))
		
		if start_date:
			query = query.where(MFPProductionOrder.scheduled_start_date >= start_date)
		
		if end_date:
			query = query.where(MFPProductionOrder.scheduled_end_date <= end_date)
		
		query = query.order_by(MFPProductionOrder.scheduled_start_date)
		
		result = await self.db.execute(query)
		return list(result.scalars().all())

	async def get_production_order(self, tenant_id: str, order_id: str) -> Optional[MFPProductionOrder]:
		"""Get a production order by ID"""
		
		result = await self.db.execute(
			select(MFPProductionOrder)
			.options(selectinload(MFPProductionOrder.master_schedule))
			.where(
				and_(
					MFPProductionOrder.tenant_id == tenant_id,
					MFPProductionOrder.id == order_id
				)
			)
		)
		return result.scalar_one_or_none()

	async def get_capacity_utilization_report(
		self,
		tenant_id: str,
		facility_id: str,
		planning_period: str
	) -> Dict[str, Any]:
		"""Generate capacity utilization report for a facility and period"""
		
		# Get all resource capacities for the period
		capacity_query = select(MFPResourceCapacity).where(
			and_(
				MFPResourceCapacity.tenant_id == tenant_id,
				MFPResourceCapacity.facility_id == facility_id,
				MFPResourceCapacity.planning_period == planning_period
			)
		)
		
		capacity_result = await self.db.execute(capacity_query)
		capacities = list(capacity_result.scalars().all())
		
		# Calculate overall utilization metrics
		total_available = sum(c.available_capacity for c in capacities if c.available_capacity)
		total_planned = sum(c.planned_capacity or 0 for c in capacities)
		
		overall_utilization = (total_planned / total_available * 100) if total_available > 0 else 0
		
		# Group by resource type
		by_resource_type = {}
		for capacity in capacities:
			resource_type = capacity.resource_type
			if resource_type not in by_resource_type:
				by_resource_type[resource_type] = {
					'total_available': 0,
					'total_planned': 0,
					'resources': []
				}
			
			by_resource_type[resource_type]['total_available'] += capacity.available_capacity or 0
			by_resource_type[resource_type]['total_planned'] += capacity.planned_capacity or 0
			by_resource_type[resource_type]['resources'].append({
				'resource_id': capacity.resource_id,
				'resource_name': capacity.resource_name,
				'available_capacity': float(capacity.available_capacity or 0),
				'planned_capacity': float(capacity.planned_capacity or 0),
				'utilization_pct': float(capacity.capacity_utilization_pct or 0)
			})
		
		# Calculate utilization by resource type
		for resource_type, data in by_resource_type.items():
			if data['total_available'] > 0:
				data['utilization_pct'] = (data['total_planned'] / data['total_available']) * 100
			else:
				data['utilization_pct'] = 0
		
		return {
			'facility_id': facility_id,
			'planning_period': planning_period,
			'overall_utilization_pct': round(overall_utilization, 2),
			'total_available_capacity': float(total_available),
			'total_planned_capacity': float(total_planned),
			'by_resource_type': by_resource_type,
			'total_resources': len(capacities)
		}

	async def optimize_production_schedule(
		self,
		tenant_id: str,
		facility_id: str,
		planning_horizon: PlanningHorizon,
		optimization_criteria: str = "minimize_makespan"
	) -> Dict[str, Any]:
		"""Optimize production schedule using basic heuristics"""
		
		# Get production orders for optimization
		orders = await self.get_production_orders_by_facility(
			tenant_id, 
			facility_id,
			status_filter=[ProductionOrderStatus.PLANNED, ProductionOrderStatus.RELEASED]
		)
		
		if not orders:
			return {"message": "No orders to optimize", "optimized_orders": []}
		
		# Simple priority-based scheduling optimization
		if optimization_criteria == "minimize_makespan":
			# Sort by priority and then by due date
			orders.sort(key=lambda x: (
				self._get_priority_weight(x.priority),
				x.scheduled_end_date
			))
		elif optimization_criteria == "minimize_lateness":
			# Sort by due date first
			orders.sort(key=lambda x: x.scheduled_end_date)
		
		# Reschedule orders to avoid conflicts
		optimized_orders = []
		current_time = datetime.utcnow()
		
		for order in orders:
			# Simple scheduling: ensure no overlaps
			if optimized_orders:
				last_order = optimized_orders[-1]
				if order.scheduled_start_date < last_order['scheduled_end_date']:
					# Reschedule to start after last order
					duration = order.scheduled_end_date - order.scheduled_start_date
					new_start = last_order['scheduled_end_date']
					new_end = new_start + duration
					
					# Update order in database if significant change
					if abs((new_start - order.scheduled_start_date).total_seconds()) > 3600:  # > 1 hour
						order.scheduled_start_date = new_start
						order.scheduled_end_date = new_end
						await self.db.commit()
			
			optimized_orders.append({
				'order_id': order.id,
				'order_number': order.order_number,
				'product_name': order.product_name,
				'scheduled_start_date': order.scheduled_start_date.isoformat(),
				'scheduled_end_date': order.scheduled_end_date.isoformat(),
				'priority': order.priority,
				'quantity': float(order.ordered_quantity)
			})
		
		return {
			'optimization_criteria': optimization_criteria,
			'total_orders_optimized': len(optimized_orders),
			'optimized_orders': optimized_orders,
			'optimization_timestamp': datetime.utcnow().isoformat()
		}

	# Private helper methods

	def _get_priority_weight(self, priority: str) -> int:
		"""Convert priority to numeric weight for sorting"""
		priority_weights = {
			'critical': 1,
			'urgent': 2, 
			'high': 3,
			'normal': 4,
			'low': 5
		}
		return priority_weights.get(priority, 4)

	async def _calculate_capacity_utilization(self, capacity: MFPResourceCapacity):
		"""Calculate and update capacity utilization percentage"""
		if capacity.available_capacity and capacity.available_capacity > 0 and capacity.planned_capacity:
			utilization_pct = (capacity.planned_capacity / capacity.available_capacity) * 100
			capacity.capacity_utilization_pct = min(utilization_pct, 100)  # Cap at 100%
			await self.db.commit()

	async def _log_schedule_creation(self, schedule: MFPMasterProductionSchedule):
		"""Log master schedule creation"""
		print(f"Created master production schedule: {schedule.schedule_name} "
			  f"for product {schedule.product_id} at facility {schedule.facility_id}")

	async def _log_order_creation(self, order: MFPProductionOrder):
		"""Log production order creation"""
		print(f"Created production order: {order.order_number} "
			  f"for {order.product_name} quantity {order.ordered_quantity}")

	async def _log_status_change(self, order: MFPProductionOrder, old_status: str, new_status: str):
		"""Log production order status change"""
		print(f"Production order {order.order_number} status changed from {old_status} to {new_status}")

	async def _log_forecast_creation(self, forecast: MFPDemandForecast):
		"""Log demand forecast creation"""
		print(f"Created demand forecast: {forecast.forecast_name} "
			  f"for product {forecast.product_id} quantity {forecast.forecast_quantity}")

__all__ = ["ProductionPlanningService"]