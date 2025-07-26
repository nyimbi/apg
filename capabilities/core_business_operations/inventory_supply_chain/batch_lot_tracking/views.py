"""
Batch & Lot Tracking Views

Flask-AppBuilder views for batch/lot management, quality control,
recall management, and genealogy tracking.
"""

from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.actions import action
from datetime import datetime, timedelta

from .models import (
	IMBLTBatch, IMBLTBatchTransaction, IMBLTQualityTest,
	IMBLTRecallEvent, IMBLTRecallAction, IMBLTGenealogyTrace
)
from .service import BatchLotService


class IMBLTBatchView(ModelView):
	"""View for managing batches/lots"""
	
	datamodel = SQLAInterface(IMBLTBatch)
	
	list_columns = ['batch_number', 'lot_number', 'item_id', 'production_date', 
					'expiry_date', 'current_quantity', 'batch_status', 'quality_status']
	show_columns = ['batch_number', 'lot_number', 'item_id', 'production_date', 'production_line',
					'expiry_date', 'best_before_date', 'shelf_life_days', 'original_quantity',
					'current_quantity', 'available_quantity', 'quarantine_quantity',
					'batch_status', 'quality_status', 'recall_status', 'quality_grade',
					'supplier_id', 'warehouse_id', 'storage_conditions']
	edit_columns = ['batch_number', 'lot_number', 'item_id', 'production_date', 'production_line',
					'expiry_date', 'best_before_date', 'shelf_life_days', 'batch_status',
					'quality_status', 'quality_grade', 'supplier_id', 'warehouse_id',
					'storage_conditions', 'quality_notes']
	add_columns = edit_columns + ['original_quantity', 'current_quantity', 'available_quantity']
	
	base_order = ('batch_number', 'asc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'batch_number': 'Batch Number',
		'lot_number': 'Lot Number',
		'item_id': 'Item ID',
		'production_date': 'Production Date',
		'production_line': 'Production Line',
		'expiry_date': 'Expiry Date',
		'best_before_date': 'Best Before Date',
		'shelf_life_days': 'Shelf Life (Days)',
		'original_quantity': 'Original Quantity',
		'current_quantity': 'Current Quantity',
		'available_quantity': 'Available Quantity',
		'quarantine_quantity': 'Quarantine Quantity',
		'batch_status': 'Batch Status',
		'quality_status': 'Quality Status',
		'recall_status': 'Recall Status',
		'quality_grade': 'Quality Grade',
		'supplier_id': 'Supplier ID',
		'warehouse_id': 'Warehouse ID',
		'storage_conditions': 'Storage Conditions',
		'quality_notes': 'Quality Notes'
	}
	
	@action("quarantine", "Quarantine", "Quarantine selected batches", "fa-ban", multiple=True)
	def quarantine_action(self, batches):
		"""Quarantine selected batches"""
		count = 0
		for batch in batches:
			if batch.quality_status != 'Quarantine':
				batch.quality_status = 'Quarantine'
				batch.quarantine_quantity = batch.available_quantity
				batch.available_quantity = 0
				count += 1
		
		if count > 0:
			self.datamodel.session.commit()
			flash(f'{count} batches quarantined', 'success')
		else:
			flash('No batches to quarantine', 'warning')
		
		return redirect(self.get_redirect())
	
	def get_tenant_id(self):
		return "default_tenant"


class IMBLTQualityTestView(ModelView):
	"""View for managing quality tests"""
	
	datamodel = SQLAInterface(IMBLTQualityTest)
	
	list_columns = ['test_date', 'batch.batch_number', 'test_type', 'parameter_name', 
					'test_result', 'pass_fail']
	show_columns = ['test_date', 'test_type', 'test_method', 'batch', 'parameter_name',
					'target_value', 'actual_value', 'unit_of_measure', 'tolerance_range',
					'test_result', 'pass_fail', 'specification_met', 'lab_name',
					'technician_name', 'critical_parameter', 'affects_release', 'status']
	edit_columns = ['test_type', 'test_method', 'batch_id', 'parameter_name', 'target_value',
					'actual_value', 'unit_of_measure', 'tolerance_range', 'test_result',
					'pass_fail', 'lab_name', 'technician_name', 'critical_parameter',
					'affects_release', 'notes']
	add_columns = edit_columns
	
	base_order = ('test_date', 'desc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'test_date': 'Test Date',
		'test_type': 'Test Type',
		'test_method': 'Test Method',
		'batch': 'Batch',
		'batch_id': 'Batch ID',
		'parameter_name': 'Parameter',
		'target_value': 'Target Value',
		'actual_value': 'Actual Value',
		'unit_of_measure': 'Unit',
		'tolerance_range': 'Tolerance',
		'test_result': 'Result',
		'pass_fail': 'Pass/Fail',
		'specification_met': 'Spec Met',
		'lab_name': 'Lab Name',
		'technician_name': 'Technician',
		'critical_parameter': 'Critical',
		'affects_release': 'Affects Release'
	}
	
	def get_tenant_id(self):
		return "default_tenant"


class IMBLTRecallEventView(ModelView):
	"""View for managing recall events"""
	
	datamodel = SQLAInterface(IMBLTRecallEvent)
	
	list_columns = ['recall_number', 'recall_title', 'recall_type', 'severity_level',
					'initiated_date', 'status']
	show_columns = ['recall_number', 'recall_title', 'recall_reason', 'recall_type',
					'severity_level', 'initiated_date', 'notification_date', 'completion_date',
					'affected_batch', 'geographical_scope', 'status', 'total_quantity_affected',
					'quantity_recovered', 'recovery_percentage', 'regulatory_authority']
	edit_columns = ['recall_number', 'recall_title', 'recall_reason', 'recall_type',
					'severity_level', 'affected_batch_id', 'geographical_scope', 'status',
					'regulatory_authority', 'health_hazard_evaluation']
	add_columns = edit_columns
	
	base_order = ('initiated_date', 'desc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'recall_number': 'Recall Number',
		'recall_title': 'Recall Title',
		'recall_reason': 'Recall Reason',
		'recall_type': 'Recall Type',
		'severity_level': 'Severity Level',
		'initiated_date': 'Initiated Date',
		'notification_date': 'Notification Date',
		'completion_date': 'Completion Date',
		'affected_batch': 'Affected Batch',
		'affected_batch_id': 'Affected Batch ID',
		'geographical_scope': 'Geographical Scope',
		'total_quantity_affected': 'Total Quantity Affected',
		'quantity_recovered': 'Quantity Recovered',
		'recovery_percentage': 'Recovery %',
		'regulatory_authority': 'Regulatory Authority',
		'health_hazard_evaluation': 'Health Hazard Evaluation'
	}
	
	def get_tenant_id(self):
		return "default_tenant"


class BatchLotDashboardView(BaseView):
	"""Batch & Lot Tracking Dashboard"""
	
	route_base = "/batch_lot/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Display Batch & Lot Tracking dashboard"""
		
		try:
			with BatchLotService(self.get_tenant_id()) as service:
				dashboard_data = {
					'active_batches': service.get_active_batches_count(),
					'quarantined_lots': service.get_quarantined_lots_count(),
					'recall_eligible': service.get_recall_eligible_lots_count(),
					'recent_quality_tests': self._get_recent_quality_tests(),
					'expiring_batches': self._get_expiring_batches()
				}
				
				return self.render_template(
					'batch_lot_dashboard.html',
					dashboard_data=dashboard_data,
					title="Batch & Lot Tracking Dashboard"
				)
		
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return self.render_template('error.html', error=str(e))
	
	def _get_recent_quality_tests(self) -> list:
		"""Get recent quality tests"""
		return []  # Simplified for now
	
	def _get_expiring_batches(self) -> list:
		"""Get batches expiring soon"""
		return []  # Simplified for now
	
	def get_tenant_id(self):
		return "default_tenant"