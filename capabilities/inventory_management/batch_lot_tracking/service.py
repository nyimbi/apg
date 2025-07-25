"""
Batch & Lot Tracking Service

Business logic for batch/lot management, genealogy tracking,
quality control, and recall management.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy import and_, or_, func, text
import json

from ....auth_rbac.models import get_session
from .models import (
	IMBLTBatch, IMBLTBatchTransaction, IMBLTQualityTest,
	IMBLTRecallEvent, IMBLTRecallAction, IMBLTGenealogyTrace
)


class BatchLotService:
	"""Service for batch and lot tracking operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.session = get_session()
	
	def __enter__(self):
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		if exc_type:
			self.session.rollback()
		else:
			self.session.commit()
		self.session.close()
	
	# Batch Management
	
	def create_batch(self, batch_data: Dict[str, Any]) -> IMBLTBatch:
		"""Create a new batch/lot record"""
		
		required_fields = ['batch_number', 'item_id', 'original_quantity']
		for field in required_fields:
			if field not in batch_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Check for duplicate batch number for same item
		existing_batch = self.session.query(IMBLTBatch).filter(
			and_(
				IMBLTBatch.tenant_id == self.tenant_id,
				IMBLTBatch.batch_number == batch_data['batch_number'],
				IMBLTBatch.item_id == batch_data['item_id']
			)
		).first()
		
		if existing_batch:
			raise ValueError(f"Batch number '{batch_data['batch_number']}' already exists for this item")
		
		# Set current and available quantities to original if not specified
		if 'current_quantity' not in batch_data:
			batch_data['current_quantity'] = batch_data['original_quantity']
		if 'available_quantity' not in batch_data:
			batch_data['available_quantity'] = batch_data['original_quantity']
		
		batch = IMBLTBatch(
			tenant_id=self.tenant_id,
			**batch_data
		)
		
		self.session.add(batch)
		self.session.flush()
		
		self._log_batch_creation(batch)
		return batch
	
	def get_batch_by_id(self, batch_id: str) -> Optional[IMBLTBatch]:
		"""Get batch by ID"""
		return self.session.query(IMBLTBatch).filter(
			and_(
				IMBLTBatch.tenant_id == self.tenant_id,
				IMBLTBatch.batch_id == batch_id
			)
		).first()
	
	def get_batch_by_number(self, batch_number: str, item_id: str = None) -> Optional[IMBLTBatch]:
		"""Get batch by batch number"""
		query = self.session.query(IMBLTBatch).filter(
			and_(
				IMBLTBatch.tenant_id == self.tenant_id,
				IMBLTBatch.batch_number == batch_number
			)
		)
		
		if item_id:
			query = query.filter(IMBLTBatch.item_id == item_id)
		
		return query.first()
	
	# Quality Testing
	
	def record_quality_test(self, test_data: Dict[str, Any]) -> IMBLTQualityTest:
		"""Record a quality test result"""
		
		required_fields = ['batch_id', 'test_type', 'parameter_name', 'test_result']
		for field in required_fields:
			if field not in test_data:
				raise ValueError(f"Missing required field: {field}")
		
		test = IMBLTQualityTest(
			tenant_id=self.tenant_id,
			**test_data
		)
		
		self.session.add(test)
		self.session.flush()
		
		# Update batch quality status if test fails
		if test.test_result == 'Fail':
			batch = self.get_batch_by_id(test.batch_id)
			if batch and test.affects_release:
				batch.quality_status = 'Quarantine'
		
		self._log_quality_test(test)
		return test
	
	# Recall Management
	
	def initiate_recall(self, recall_data: Dict[str, Any]) -> IMBLTRecallEvent:
		"""Initiate a recall event"""
		
		required_fields = ['recall_number', 'recall_title', 'recall_reason', 'recall_type', 'severity_level']
		for field in required_fields:
			if field not in recall_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Check for duplicate recall number
		existing_recall = self.session.query(IMBLTRecallEvent).filter(
			and_(
				IMBLTRecallEvent.tenant_id == self.tenant_id,
				IMBLTRecallEvent.recall_number == recall_data['recall_number']
			)
		).first()
		
		if existing_recall:
			raise ValueError(f"Recall number '{recall_data['recall_number']}' already exists")
		
		recall = IMBLTRecallEvent(
			tenant_id=self.tenant_id,
			**recall_data
		)
		
		self.session.add(recall)
		self.session.flush()
		
		# Update affected batch status
		if recall.affected_batch_id:
			batch = self.get_batch_by_id(recall.affected_batch_id)
			if batch:
				batch.recall_status = 'Recall'
				batch.batch_status = 'Recalled'
		
		self._log_recall_initiation(recall)
		return recall
	
	# Genealogy Tracking
	
	def create_genealogy_link(self, parent_batch_id: str, child_batch_id: str, 
							  relationship_type: str, **kwargs) -> IMBLTGenealogyTrace:
		"""Create genealogy link between parent and child batches"""
		
		trace = IMBLTGenealogyTrace(
			tenant_id=self.tenant_id,
			parent_batch_id=parent_batch_id,
			child_batch_id=child_batch_id,
			relationship_type=relationship_type,
			**kwargs
		)
		
		self.session.add(trace)
		self.session.flush()
		
		# Update batch genealogy references
		parent_batch = self.get_batch_by_id(parent_batch_id)
		child_batch = self.get_batch_by_id(child_batch_id)
		
		if parent_batch:
			child_ids = parent_batch.get_child_batch_ids()
			if child_batch_id not in child_ids:
				child_ids.append(child_batch_id)
				parent_batch.set_child_batch_ids(child_ids)
		
		if child_batch:
			parent_ids = child_batch.get_parent_batch_ids()
			if parent_batch_id not in parent_ids:
				parent_ids.append(parent_batch_id)
				child_batch.set_parent_batch_ids(parent_ids)
		
		return trace
	
	# Dashboard Methods
	
	def get_active_batches_count(self) -> int:
		"""Get count of active batches"""
		return self.session.query(IMBLTBatch).filter(
			and_(
				IMBLTBatch.tenant_id == self.tenant_id,
				IMBLTBatch.batch_status == 'Active'
			)
		).count()
	
	def get_quarantined_lots_count(self) -> int:
		"""Get count of quarantined lots"""
		return self.session.query(IMBLTBatch).filter(
			and_(
				IMBLTBatch.tenant_id == self.tenant_id,
				IMBLTBatch.quality_status == 'Quarantine'
			)
		).count()
	
	def get_recall_eligible_lots_count(self) -> int:
		"""Get count of lots eligible for recall"""
		return self.session.query(IMBLTBatch).filter(
			and_(
				IMBLTBatch.tenant_id == self.tenant_id,
				IMBLTBatch.recall_status.in_(['Investigation', 'Recall'])
			)
		).count()
	
	# Logging Methods
	
	def _log_batch_creation(self, batch: IMBLTBatch):
		"""Log batch creation"""
		print(f"Batch created: {batch.batch_number}")
	
	def _log_quality_test(self, test: IMBLTQualityTest):
		"""Log quality test"""
		print(f"Quality test recorded: {test.test_type} - {test.test_result}")
	
	def _log_recall_initiation(self, recall: IMBLTRecallEvent):
		"""Log recall initiation"""
		print(f"Recall initiated: {recall.recall_number}")