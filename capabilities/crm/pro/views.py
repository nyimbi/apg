"""
Order Processing Views

Flask-AppBuilder views for order processing including fulfillment tasks,
workflow management, and shipment tracking.
"""

from flask_appbuilder import ModelView
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .models import (
	SOPFulfillmentTask, SOPShipment, SOPOrderStatus, SOPOrderWorkflow
)


class SOPFulfillmentTaskView(ModelView):
	"""Fulfillment task management view"""
	
	datamodel = SQLAInterface(SOPFulfillmentTask)
	
	list_columns = [
		'task_type', 'task_name', 'order_number', 'status', 
		'assigned_to', 'priority', 'scheduled_start'
	]
	search_columns = ['task_name', 'order_number', 'description']
	list_filters = ['task_type', 'status', 'priority']
	base_order = ('scheduled_start', 'asc')


class SOPShipmentView(ModelView):
	"""Shipment management view"""
	
	datamodel = SQLAInterface(SOPShipment)
	
	list_columns = [
		'shipment_number', 'customer_name', 'carrier', 
		'tracking_number', 'ship_date', 'shipment_status'
	]
	search_columns = ['shipment_number', 'customer_name', 'tracking_number']
	list_filters = ['carrier', 'shipment_status', 'ship_date']
	base_order = ('ship_date', 'desc')


class SOPOrderStatusView(ModelView):
	"""Order status configuration view"""
	
	datamodel = SQLAInterface(SOPOrderStatus)
	
	list_columns = [
		'status_code', 'status_name', 'sequence_number', 
		'is_active', 'requires_approval'
	]
	base_order = ('sequence_number', 'asc')


class SOPOrderWorkflowView(ModelView):
	"""Order workflow configuration view"""
	
	datamodel = SQLAInterface(SOPOrderWorkflow)
	
	list_columns = [
		'workflow_name', 'workflow_type', 'total_sla_hours',
		'is_active', 'is_default'
	]
	base_order = ('workflow_name', 'asc')