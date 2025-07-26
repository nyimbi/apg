"""
Sourcing & Supplier Selection Service

Business logic for RFQ/RFP management and supplier evaluation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from .models import PPSRFQHeader, PPSBid, PPSSupplierEvaluation, PPSAwardRecommendation
from ...auth_rbac.models import get_db_session


class SourcingSupplierSelectionService:
	"""Service class for sourcing and supplier selection operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.db: Session = get_db_session()
	
	def get_active_rfqs(self, limit: int = 100) -> List[PPSRFQHeader]:
		"""Get active RFQs"""
		return self.db.query(PPSRFQHeader).filter(
			and_(
				PPSRFQHeader.tenant_id == self.tenant_id,
				PPSRFQHeader.status.in_(['Draft', 'Issued'])
			)
		).order_by(desc(PPSRFQHeader.issue_date)).limit(limit).all()
	
	def get_pending_evaluations(self) -> List[PPSRFQHeader]:
		"""Get RFQs pending evaluation"""
		return self.db.query(PPSRFQHeader).filter(
			and_(
				PPSRFQHeader.tenant_id == self.tenant_id,
				PPSRFQHeader.status == 'Closed',
				PPSRFQHeader.evaluation_complete_date.is_(None)
			)
		).all()
	
	def get_total_sourcing_value_ytd(self) -> Decimal:
		"""Get total sourcing value year-to-date"""
		from sqlalchemy import func, extract
		
		current_year = datetime.now().year
		
		result = self.db.query(func.sum(PPSRFQHeader.estimated_value)).filter(
			and_(
				PPSRFQHeader.tenant_id == self.tenant_id,
				extract('year', PPSRFQHeader.issue_date) == current_year,
				PPSRFQHeader.status.in_(['Issued', 'Closed', 'Awarded'])
			)
		).scalar()
		
		return result or Decimal('0.00')
	
	def get_avg_bid_count_per_rfq(self) -> float:
		"""Get average number of bids per RFQ"""
		from sqlalchemy import func
		
		result = self.db.query(func.avg(
			self.db.query(func.count(PPSBid.bid_id)).filter(
				PPSBid.rfq_id == PPSRFQHeader.rfq_id
			).scalar_subquery()
		)).filter(
			and_(
				PPSRFQHeader.tenant_id == self.tenant_id,
				PPSRFQHeader.status.in_(['Closed', 'Awarded'])
			)
		).scalar()
		
		return float(result) if result else 0.0
	
	def get_avg_evaluation_time(self) -> float:
		"""Get average evaluation time in days"""
		# Placeholder implementation
		return 5.0
	
	def get_cost_savings_ytd(self) -> Decimal:
		"""Get cost savings year-to-date"""
		from sqlalchemy import func, extract
		
		current_year = datetime.now().year
		
		result = self.db.query(func.sum(PPSAwardRecommendation.estimated_savings)).filter(
			and_(
				PPSAwardRecommendation.tenant_id == self.tenant_id,
				extract('year', PPSAwardRecommendation.recommendation_date) == current_year,
				PPSAwardRecommendation.status == 'Approved'
			)
		).scalar()
		
		return result or Decimal('0.00')
	
	def __del__(self):
		"""Cleanup database session"""
		if hasattr(self, 'db'):
			self.db.close()