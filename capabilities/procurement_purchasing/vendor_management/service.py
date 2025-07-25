"""
Vendor Management Service

Business logic for vendor management operations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from .models import PPVVendor, PPVVendorContact, PPVVendorPerformance, PPVVendorCategory
from ...auth_rbac.models import get_db_session


class VendorManagementService:
	"""Service class for vendor management operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.db: Session = get_db_session()
	
	def get_active_vendor_count(self) -> int:
		"""Get count of active vendors"""
		return self.db.query(PPVVendor).filter(
			and_(
				PPVVendor.tenant_id == self.tenant_id,
				PPVVendor.is_active == True
			)
		).count()
	
	def get_top_vendors_by_spend(self, limit: int = 5) -> List[Dict[str, Any]]:
		"""Get top vendors by spend"""
		# This would typically query purchase order data
		# Placeholder implementation
		return [
			{'vendor_id': 'v1', 'vendor_name': 'Vendor 1', 'spend': 100000},
			{'vendor_id': 'v2', 'vendor_name': 'Vendor 2', 'spend': 75000}
		]
	
	def get_average_performance_score(self) -> float:
		"""Get average vendor performance score"""
		result = self.db.query(func.avg(PPVVendor.overall_rating)).filter(
			and_(
				PPVVendor.tenant_id == self.tenant_id,
				PPVVendor.is_active == True,
				PPVVendor.overall_rating > 0
			)
		).scalar()
		
		return float(result) if result else 0.0
	
	def get_new_vendor_count(self, days: int = 30) -> int:
		"""Get count of new vendors in last N days"""
		from datetime import timedelta
		
		cutoff_date = datetime.utcnow() - timedelta(days=days)
		
		return self.db.query(PPVVendor).filter(
			and_(
				PPVVendor.tenant_id == self.tenant_id,
				PPVVendor.created_date >= cutoff_date
			)
		).count()
	
	def get_poor_performing_vendors(self, threshold: float = 2.0) -> List[PPVVendor]:
		"""Get vendors with poor performance ratings"""
		return self.db.query(PPVVendor).filter(
			and_(
				PPVVendor.tenant_id == self.tenant_id,
				PPVVendor.is_active == True,
				PPVVendor.overall_rating < threshold,
				PPVVendor.overall_rating > 0
			)
		).all()
	
	def __del__(self):
		"""Cleanup database session"""
		if hasattr(self, 'db'):
			self.db.close()