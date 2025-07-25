"""
Contract Management Service

Business logic for contract management operations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from .models import PPCContract, PPCContractAmendment, PPCContractRenewal, PPCContractMilestone
from ...auth_rbac.models import get_db_session


class ContractManagementService:
	"""Service class for contract management operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.db: Session = get_db_session()
	
	def get_active_contract_count(self) -> int:
		"""Get count of active contracts"""
		return self.db.query(PPCContract).filter(
			and_(
				PPCContract.tenant_id == self.tenant_id,
				PPCContract.status == 'Active'
			)
		).count()
	
	def get_contracts_expiring_soon(self, days: int = 90) -> List[PPCContract]:
		"""Get contracts expiring within specified days"""
		cutoff_date = date.today() + timedelta(days=days)
		
		return self.db.query(PPCContract).filter(
			and_(
				PPCContract.tenant_id == self.tenant_id,
				PPCContract.status == 'Active',
				PPCContract.expiration_date <= cutoff_date
			)
		).order_by(PPCContract.expiration_date).all()
	
	def get_total_contract_value(self) -> Decimal:
		"""Get total value of active contracts"""
		result = self.db.query(func.sum(PPCContract.contract_value)).filter(
			and_(
				PPCContract.tenant_id == self.tenant_id,
				PPCContract.status == 'Active'
			)
		).scalar()
		
		return result or Decimal('0.00')
	
	def get_renewal_rate(self) -> float:
		"""Get contract renewal rate percentage"""
		# Get contracts that had renewal decisions in the last year
		one_year_ago = date.today() - timedelta(days=365)
		
		total_renewals = self.db.query(PPCContractRenewal).filter(
			and_(
				PPCContractRenewal.tenant_id == self.tenant_id,
				PPCContractRenewal.decision_date >= one_year_ago,
				PPCContractRenewal.renewal_status.in_(['Approved', 'Declined'])
			)
		).count()
		
		approved_renewals = self.db.query(PPCContractRenewal).filter(
			and_(
				PPCContractRenewal.tenant_id == self.tenant_id,
				PPCContractRenewal.decision_date >= one_year_ago,
				PPCContractRenewal.renewal_status == 'Approved'
			)
		).count()
		
		if total_renewals == 0:
			return 0.0
		
		return (approved_renewals / total_renewals) * 100
	
	def get_overdue_milestones(self) -> List[PPCContractMilestone]:
		"""Get overdue contract milestones"""
		today = date.today()
		
		return self.db.query(PPCContractMilestone).filter(
			and_(
				PPCContractMilestone.tenant_id == self.tenant_id,
				PPCContractMilestone.status.in_(['Planned', 'In Progress']),
				PPCContractMilestone.planned_date < today
			)
		).all()
	
	def get_contract_utilization_summary(self) -> Dict[str, Any]:
		"""Get contract utilization summary"""
		active_contracts = self.db.query(PPCContract).filter(
			and_(
				PPCContract.tenant_id == self.tenant_id,
				PPCContract.status == 'Active'
			)
		).all()
		
		total_value = sum(contract.contract_value for contract in active_contracts)
		total_spend = sum(contract.actual_spend for contract in active_contracts)
		
		utilization_rate = (total_spend / total_value * 100) if total_value > 0 else 0
		
		return {
			'total_contracts': len(active_contracts),
			'total_value': float(total_value),
			'total_spend': float(total_spend),
			'utilization_rate': utilization_rate,
			'underutilized_contracts': len([c for c in active_contracts if c.get_utilization_percentage() < 50])
		}
	
	def __del__(self):
		"""Cleanup database session"""
		if hasattr(self, 'db'):
			self.db.close()