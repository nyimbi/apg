"""
Order Processing Service

Business logic for order workflow management including fulfillment,
task management, shipping, and order completion.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from .models import (
	SOPOrderStatus, SOPFulfillmentTask, SOPLineTask, SOPShipment,
	SOPShipmentPackage, SOPTrackingEvent, SOPOrderWorkflow
)


class OrderProcessingService:
	"""Service class for order processing operations"""
	
	def __init__(self, db_session: Session):
		self.db = db_session
	
	def create_fulfillment_workflow(self, tenant_id: str, order_id: str, workflow_type: str = 'STANDARD') -> List[SOPFulfillmentTask]:
		"""Create fulfillment tasks for an order based on workflow"""
		workflow = self._get_workflow_template(tenant_id, workflow_type)
		
		tasks = []
		for step in workflow.get_workflow_steps():
			task = SOPFulfillmentTask(
				tenant_id=tenant_id,
				task_type=step['task_type'],
				task_name=step['task_name'],
				description=step.get('description'),
				order_id=order_id,
				priority=step.get('priority', 'NORMAL'),
				estimated_duration=step.get('estimated_duration'),
				quality_check_required=step.get('quality_check_required', False),
				instructions=step.get('instructions')
			)
			
			# Set dependencies
			if step.get('depends_on'):
				task.set_dependencies(step['depends_on'])
			
			self.db.add(task)
			tasks.append(task)
		
		self.db.commit()
		return tasks
	
	def assign_task(self, tenant_id: str, task_id: str, user_id: str, workstation_id: str = None) -> SOPFulfillmentTask:
		"""Assign task to user"""
		task = self.db.query(SOPFulfillmentTask).filter(
			and_(
				SOPFulfillmentTask.tenant_id == tenant_id,
				SOPFulfillmentTask.task_id == task_id
			)
		).first()
		
		if not task:
			raise ValueError("Task not found")
		
		if task.status != 'PENDING':
			raise ValueError("Task must be pending to assign")
		
		task.assigned_to = user_id
		task.assigned_date = datetime.utcnow()
		task.workstation_id = workstation_id
		task.status = 'ASSIGNED'
		
		self.db.commit()
		return task
	
	def start_task(self, tenant_id: str, task_id: str, user_id: str) -> SOPFulfillmentTask:
		"""Start a task"""
		task = self.db.query(SOPFulfillmentTask).filter(
			and_(
				SOPFulfillmentTask.tenant_id == tenant_id,
				SOPFulfillmentTask.task_id == task_id
			)
		).first()
		
		if not task:
			raise ValueError("Task not found")
		
		task.start_task(user_id)
		self.db.commit()
		return task
	
	def complete_task(self, tenant_id: str, task_id: str, user_id: str, completion_data: Dict[str, Any] = None) -> SOPFulfillmentTask:
		"""Complete a task"""
		task = self.db.query(SOPFulfillmentTask).filter(
			and_(
				SOPFulfillmentTask.tenant_id == tenant_id,
				SOPFulfillmentTask.task_id == task_id
			)
		).first()
		
		if not task:
			raise ValueError("Task not found")
		
		# Process completion data based on task type
		if task.task_type == 'PICK' and completion_data:
			self._process_picking_completion(task, completion_data)
		elif task.task_type == 'PACK' and completion_data:
			self._process_packing_completion(task, completion_data)
		
		task.complete_task(user_id, completion_data.get('notes') if completion_data else None)
		self.db.commit()
		
		# Check if this completes the order
		self._check_order_completion(task.order_id)
		
		return task
	
	def create_shipment(self, tenant_id: str, shipment_data: Dict[str, Any], user_id: str) -> SOPShipment:
		"""Create a shipment"""
		shipment_number = self._generate_shipment_number(tenant_id)
		
		shipment = SOPShipment(
			tenant_id=tenant_id,
			shipment_number=shipment_number,
			shipment_type=shipment_data.get('shipment_type', 'STANDARD'),
			customer_id=shipment_data['customer_id'],
			customer_name=shipment_data['customer_name'],
			ship_to_address_line1=shipment_data['ship_to_address_line1'],
			ship_to_city=shipment_data['ship_to_city'],
			ship_to_state_province=shipment_data['ship_to_state_province'],
			ship_to_postal_code=shipment_data['ship_to_postal_code'],
			ship_to_country=shipment_data['ship_to_country'],
			carrier=shipment_data['carrier'],
			service_level=shipment_data['service_level'],
			ship_date=shipment_data.get('ship_date', date.today()),
			shipping_cost=shipment_data.get('shipping_cost', 0),
			created_by_user_id=user_id
		)
		
		shipment.set_order_ids(shipment_data['order_ids'])
		
		self.db.add(shipment)
		self.db.flush()
		
		# Create packages
		if 'packages' in shipment_data:
			for pkg_num, pkg_data in enumerate(shipment_data['packages'], 1):
				package = SOPShipmentPackage(
					shipment_id=shipment.shipment_id,
					tenant_id=tenant_id,
					package_number=pkg_num,
					package_type=pkg_data.get('package_type', 'BOX'),
					weight=pkg_data.get('weight'),
					declared_value=pkg_data.get('declared_value', 0),
					created_by_user_id=user_id
				)
				
				if 'contents' in pkg_data:
					package.set_contents(pkg_data['contents'])
				
				self.db.add(package)
		
		self.db.commit()
		return shipment
	
	def ship_shipment(self, tenant_id: str, shipment_id: str, tracking_number: str, user_id: str) -> SOPShipment:
		"""Ship a shipment"""
		shipment = self.db.query(SOPShipment).filter(
			and_(
				SOPShipment.tenant_id == tenant_id,
				SOPShipment.shipment_id == shipment_id
			)
		).first()
		
		if not shipment:
			raise ValueError("Shipment not found")
		
		shipment.ship_shipment(user_id, tracking_number)
		
		# Create initial tracking event
		tracking_event = SOPTrackingEvent(
			shipment_id=shipment.shipment_id,
			tenant_id=tenant_id,
			event_type='PICKUP',
			description='Package picked up by carrier',
			event_date=date.today(),
			event_time=datetime.utcnow(),
			event_source='INTERNAL',
			created_by_user_id=user_id
		)
		self.db.add(tracking_event)
		
		self.db.commit()
		return shipment
	
	def add_tracking_event(self, tenant_id: str, shipment_id: str, event_data: Dict[str, Any]) -> SOPTrackingEvent:
		"""Add tracking event to shipment"""
		event = SOPTrackingEvent(
			shipment_id=shipment_id,
			tenant_id=tenant_id,
			event_type=event_data['event_type'],
			description=event_data['description'],
			event_date=event_data.get('event_date', date.today()),
			event_time=event_data.get('event_time', datetime.utcnow()),
			location_city=event_data.get('location_city'),
			location_state=event_data.get('location_state'),
			event_source=event_data.get('event_source', 'CARRIER'),
			carrier_reference=event_data.get('carrier_reference'),
			is_exception=event_data.get('is_exception', False),
			exception_type=event_data.get('exception_type')
		)
		
		self.db.add(event)
		self.db.commit()
		
		# Update shipment status based on event
		self._update_shipment_status(shipment_id, event_data['event_type'])
		
		return event
	
	def get_order_tasks(self, tenant_id: str, order_id: str) -> List[SOPFulfillmentTask]:
		"""Get all tasks for an order"""
		return self.db.query(SOPFulfillmentTask).filter(
			and_(
				SOPFulfillmentTask.tenant_id == tenant_id,
				SOPFulfillmentTask.order_id == order_id
			)
		).order_by(SOPFulfillmentTask.created_date).all()
	
	def get_user_tasks(self, tenant_id: str, user_id: str, status: str = None) -> List[SOPFulfillmentTask]:
		"""Get tasks assigned to user"""
		query = self.db.query(SOPFulfillmentTask).filter(
			and_(
				SOPFulfillmentTask.tenant_id == tenant_id,
				SOPFulfillmentTask.assigned_to == user_id
			)
		)
		
		if status:
			query = query.filter(SOPFulfillmentTask.status == status)
		
		return query.order_by(SOPFulfillmentTask.scheduled_start).all()
	
	def _get_workflow_template(self, tenant_id: str, workflow_type: str) -> SOPOrderWorkflow:
		"""Get workflow template"""
		workflow = self.db.query(SOPOrderWorkflow).filter(
			and_(
				SOPOrderWorkflow.tenant_id == tenant_id,
				SOPOrderWorkflow.workflow_type == workflow_type,
				SOPOrderWorkflow.is_active == True
			)
		).first()
		
		if not workflow:
			# Return default workflow
			return self._create_default_workflow(tenant_id, workflow_type)
		
		return workflow
	
	def _create_default_workflow(self, tenant_id: str, workflow_type: str) -> SOPOrderWorkflow:
		"""Create default workflow template"""
		default_steps = [
			{
				'task_type': 'PICK',
				'task_name': 'Pick Items',
				'description': 'Pick all items from inventory',
				'estimated_duration': 30,
				'quality_check_required': True
			},
			{
				'task_type': 'PACK',
				'task_name': 'Pack Order',
				'description': 'Pack items for shipping',
				'estimated_duration': 15,
				'depends_on': ['PICK']
			},
			{
				'task_type': 'SHIP',
				'task_name': 'Ship Order',
				'description': 'Create shipment and generate labels',
				'estimated_duration': 10,
				'depends_on': ['PACK']
			}
		]
		
		workflow = SOPOrderWorkflow(
			tenant_id=tenant_id,
			workflow_name=f"Default {workflow_type} Workflow",
			workflow_type=workflow_type,
			is_default=True
		)
		workflow.set_workflow_steps(default_steps)
		
		self.db.add(workflow)
		self.db.commit()
		return workflow
	
	def _process_picking_completion(self, task: SOPFulfillmentTask, completion_data: Dict[str, Any]):
		"""Process picking task completion"""
		for line_data in completion_data.get('lines', []):
			line_task = self.db.query(SOPLineTask).filter(
				and_(
					SOPLineTask.task_id == task.task_id,
					SOPLineTask.order_line_id == line_data['order_line_id']
				)
			).first()
			
			if line_task:
				line_task.pick_quantity(
					Decimal(str(line_data['quantity_picked'])),
					completion_data['user_id'],
					line_data.get('location')
				)
	
	def _process_packing_completion(self, task: SOPFulfillmentTask, completion_data: Dict[str, Any]):
		"""Process packing task completion"""
		for line_data in completion_data.get('lines', []):
			line_task = self.db.query(SOPLineTask).filter(
				and_(
					SOPLineTask.task_id == task.task_id,
					SOPLineTask.order_line_id == line_data['order_line_id']
				)
			).first()
			
			if line_task:
				line_task.pack_quantity(
					Decimal(str(line_data['quantity_packed'])),
					completion_data['user_id']
				)
	
	def _check_order_completion(self, order_id: str):
		"""Check if order is complete and update status"""
		# Logic to check if all tasks are complete and update order status
		pass
	
	def _generate_shipment_number(self, tenant_id: str) -> str:
		"""Generate next shipment number"""
		# Simple implementation - in production would use sequence table
		count = self.db.query(SOPShipment).filter(SOPShipment.tenant_id == tenant_id).count()
		return f"SHIP-{count + 1:06d}"
	
	def _update_shipment_status(self, shipment_id: str, event_type: str):
		"""Update shipment status based on tracking event"""
		shipment = self.db.query(SOPShipment).filter(SOPShipment.shipment_id == shipment_id).first()
		if not shipment:
			return
		
		status_mapping = {
			'PICKUP': 'SHIPPED',
			'IN_TRANSIT': 'IN_TRANSIT',
			'OUT_FOR_DELIVERY': 'IN_TRANSIT',
			'DELIVERED': 'DELIVERED',
			'EXCEPTION': 'IN_TRANSIT'
		}
		
		new_status = status_mapping.get(event_type)
		if new_status and shipment.shipment_status != 'DELIVERED':
			shipment.shipment_status = new_status
			
			if new_status == 'DELIVERED':
				shipment.actual_delivery_date = date.today()
		
		self.db.commit()