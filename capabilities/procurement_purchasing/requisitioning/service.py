"""
Requisitioning Service

Business logic and service layer for requisition management including
approval workflows, budget checking, and requisition-to-PO conversion.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from .models import PPRRequisition, PPRRequisitionLine, PPRApprovalWorkflow, PPRRequisitionComment
from ...auth_rbac.models import get_db_session


class RequisitioningService:
	"""Service class for requisition management operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.db: Session = get_db_session()
	
	# Requisition Management
	
	def create_requisition(self, requisition_data: Dict[str, Any], user_id: str) -> PPRRequisition:
		"""Create a new requisition"""
		
		# Generate requisition number
		requisition_number = self._generate_requisition_number()
		
		# Create requisition
		requisition = PPRRequisition(
			tenant_id=self.tenant_id,
			requisition_number=requisition_number,
			title=requisition_data['title'],
			description=requisition_data.get('description'),
			business_justification=requisition_data.get('business_justification'),
			requestor_id=user_id,
			requestor_name=requisition_data['requestor_name'],
			requestor_email=requisition_data.get('requestor_email'),
			department=requisition_data.get('department'),
			cost_center=requisition_data.get('cost_center'),
			required_date=requisition_data['required_date'],
			delivery_location=requisition_data.get('delivery_location'),
			priority=requisition_data.get('priority', 'Normal'),
			currency_code=requisition_data.get('currency_code', 'USD'),
			budget_account_id=requisition_data.get('budget_account_id'),
			project_id=requisition_data.get('project_id'),
			project_name=requisition_data.get('project_name'),
			rush_order=requisition_data.get('rush_order', False),
			drop_ship=requisition_data.get('drop_ship', False),
			special_instructions=requisition_data.get('special_instructions'),
			notes=requisition_data.get('notes')
		)
		
		self.db.add(requisition)
		self.db.flush()  # Get the ID
		
		# Add requisition lines
		if 'lines' in requisition_data:
			for line_data in requisition_data['lines']:
				self._create_requisition_line(requisition, line_data)
		
		# Calculate totals
		requisition.calculate_totals()
		
		# Check budget if configured
		if requisition.budget_account_id:
			budget_check = requisition.check_budget_availability()
			requisition.budget_checked = True
			requisition.budget_available = budget_check['budget_available']
		
		self.db.commit()
		
		# Log activity
		self._log_requisition_activity(
			requisition.requisition_id,
			'Created',
			f"Requisition {requisition.requisition_number} created by {requisition.requestor_name}",
			user_id
		)
		
		return requisition
	
	def update_requisition(self, requisition_id: str, updates: Dict[str, Any], user_id: str) -> PPRRequisition:
		"""Update an existing requisition"""
		
		requisition = self.get_requisition_by_id(requisition_id)
		if not requisition:
			raise ValueError(f"Requisition {requisition_id} not found")
		
		if not requisition.can_edit(user_id):
			raise ValueError("User not authorized to edit this requisition")
		
		# Update fields
		for field, value in updates.items():
			if hasattr(requisition, field):
				setattr(requisition, field, value)
		
		# Update lines if provided
		if 'lines' in updates:
			# Remove existing lines
			for line in requisition.lines:
				self.db.delete(line)
			
			# Add new lines
			for line_data in updates['lines']:
				self._create_requisition_line(requisition, line_data)
		
		# Recalculate totals
		requisition.calculate_totals()
		
		# Re-check budget if amount changed
		if requisition.budget_account_id and 'lines' in updates:
			budget_check = requisition.check_budget_availability()
			requisition.budget_available = budget_check['budget_available']
		
		self.db.commit()
		
		# Log activity
		self._log_requisition_activity(
			requisition_id,
			'Updated',
			f"Requisition updated by {user_id}",
			user_id
		)
		
		return requisition
	
	def submit_requisition(self, requisition_id: str, user_id: str) -> PPRRequisition:
		"""Submit requisition for approval"""
		
		requisition = self.get_requisition_by_id(requisition_id)
		if not requisition:
			raise ValueError(f"Requisition {requisition_id} not found")
		
		if not requisition.can_submit():
			raise ValueError("Requisition cannot be submitted")
		
		if requisition.requestor_id != user_id:
			raise ValueError("Only the requestor can submit the requisition")
		
		# Submit requisition
		requisition.submit_requisition()
		self.db.commit()
		
		# Send notifications to approvers
		self._send_approval_notifications(requisition)
		
		# Log activity
		self._log_requisition_activity(
			requisition_id,
			'Submitted',
			f"Requisition submitted for approval",
			user_id
		)
		
		return requisition
	
	def approve_requisition(self, requisition_id: str, user_id: str, comments: str = None) -> PPRRequisition:
		"""Approve requisition at current approval level"""
		
		requisition = self.get_requisition_by_id(requisition_id)
		if not requisition:
			raise ValueError(f"Requisition {requisition_id} not found")
		
		if not requisition.can_approve(user_id):
			raise ValueError("User not authorized to approve this requisition")
		
		# Get approver information
		approver_info = self._get_user_info(user_id)
		
		# Approve requisition
		requisition.approve_requisition(user_id, comments)
		self.db.commit()
		
		# Add comment if provided
		if comments:
			self.add_requisition_comment(
				requisition_id,
				comments,
				user_id,
				comment_type='Approval'
			)
		
		# Send notifications
		if requisition.status == 'Approved':
			# Final approval - notify requestor
			self._send_approval_notification(requisition, 'final_approved')
		else:
			# Intermediate approval - notify next approver
			self._send_approval_notifications(requisition)
		
		# Log activity
		self._log_requisition_activity(
			requisition_id,
			'Approved',
			f"Approved by {approver_info.get('name', user_id)} at level {requisition.approval_level}",
			user_id
		)
		
		return requisition
	
	def reject_requisition(self, requisition_id: str, user_id: str, reason: str) -> PPRRequisition:
		"""Reject requisition"""
		
		requisition = self.get_requisition_by_id(requisition_id)
		if not requisition:
			raise ValueError(f"Requisition {requisition_id} not found")
		
		if not requisition.can_reject(user_id):
			raise ValueError("User not authorized to reject this requisition")
		
		# Get approver information
		approver_info = self._get_user_info(user_id)
		
		# Reject requisition
		requisition.reject_requisition(user_id, reason)
		self.db.commit()
		
		# Add rejection comment
		self.add_requisition_comment(
			requisition_id,
			reason,
			user_id,
			comment_type='Rejection'
		)
		
		# Send notification to requestor
		self._send_approval_notification(requisition, 'rejected')
		
		# Log activity
		self._log_requisition_activity(
			requisition_id,
			'Rejected',
			f"Rejected by {approver_info.get('name', user_id)}: {reason}",
			user_id
		)
		
		return requisition
	
	def cancel_requisition(self, requisition_id: str, user_id: str, reason: str = None) -> PPRRequisition:
		"""Cancel requisition"""
		
		requisition = self.get_requisition_by_id(requisition_id)
		if not requisition:
			raise ValueError(f"Requisition {requisition_id} not found")
		
		if not requisition.can_cancel(user_id):
			raise ValueError("User not authorized to cancel this requisition")
		
		# Cancel requisition
		requisition.status = 'Cancelled'
		self.db.commit()
		
		# Add cancellation comment
		if reason:
			self.add_requisition_comment(
				requisition_id,
				f"Requisition cancelled: {reason}",
				user_id,
				comment_type='General'
			)
		
		# Log activity
		self._log_requisition_activity(
			requisition_id,
			'Cancelled',
			f"Requisition cancelled by {user_id}",
			user_id
		)
		
		return requisition
	
	# Query Methods
	
	def get_requisition_by_id(self, requisition_id: str) -> Optional[PPRRequisition]:
		"""Get requisition by ID"""
		return self.db.query(PPRRequisition).filter(
			and_(
				PPRRequisition.tenant_id == self.tenant_id,
				PPRRequisition.requisition_id == requisition_id
			)
		).first()
	
	def get_requisition_by_number(self, requisition_number: str) -> Optional[PPRRequisition]:
		"""Get requisition by number"""
		return self.db.query(PPRRequisition).filter(
			and_(
				PPRRequisition.tenant_id == self.tenant_id,
				PPRRequisition.requisition_number == requisition_number
			)
		).first()
	
	def get_requisitions_by_status(self, status: str, limit: int = 100) -> List[PPRRequisition]:
		"""Get requisitions by status"""
		return self.db.query(PPRRequisition).filter(
			and_(
				PPRRequisition.tenant_id == self.tenant_id,
				PPRRequisition.status == status
			)
		).order_by(desc(PPRRequisition.request_date)).limit(limit).all()
	
	def get_requisitions_by_requestor(self, requestor_id: str, limit: int = 100) -> List[PPRRequisition]:
		"""Get requisitions by requestor"""
		return self.db.query(PPRRequisition).filter(
			and_(
				PPRRequisition.tenant_id == self.tenant_id,
				PPRRequisition.requestor_id == requestor_id
			)
		).order_by(desc(PPRRequisition.request_date)).limit(limit).all()
	
	def get_requisitions_for_approval(self, approver_id: str) -> List[PPRRequisition]:
		"""Get requisitions pending approval by user"""
		return self.db.query(PPRRequisition).filter(
			and_(
				PPRRequisition.tenant_id == self.tenant_id,
				PPRRequisition.status == 'Submitted',
				PPRRequisition.current_approver_id == approver_id
			)
		).order_by(asc(PPRRequisition.submitted_date)).all()
	
	def get_overdue_approvals(self, days_overdue: int = 2) -> List[PPRRequisition]:
		"""Get requisitions with overdue approvals"""
		cutoff_date = datetime.utcnow() - timedelta(days=days_overdue)
		
		return self.db.query(PPRRequisition).filter(
			and_(
				PPRRequisition.tenant_id == self.tenant_id,
				PPRRequisition.status == 'Submitted',
				PPRRequisition.submitted_date < cutoff_date
			)
		).order_by(asc(PPRRequisition.submitted_date)).all()
	
	def search_requisitions(self, search_params: Dict[str, Any], limit: int = 100) -> List[PPRRequisition]:
		"""Search requisitions with various filters"""
		
		query = self.db.query(PPRRequisition).filter(PPRRequisition.tenant_id == self.tenant_id)
		
		# Apply filters
		if 'status' in search_params:
			query = query.filter(PPRRequisition.status == search_params['status'])
		
		if 'requestor_id' in search_params:
			query = query.filter(PPRRequisition.requestor_id == search_params['requestor_id'])
		
		if 'department' in search_params:
			query = query.filter(PPRRequisition.department == search_params['department'])
		
		if 'date_from' in search_params:
			query = query.filter(PPRRequisition.request_date >= search_params['date_from'])
		
		if 'date_to' in search_params:
			query = query.filter(PPRRequisition.request_date <= search_params['date_to'])
		
		if 'min_amount' in search_params:
			query = query.filter(PPRRequisition.total_amount >= search_params['min_amount'])
		
		if 'max_amount' in search_params:
			query = query.filter(PPRRequisition.total_amount <= search_params['max_amount'])
		
		if 'search_text' in search_params:
			search_text = f"%{search_params['search_text']}%"
			query = query.filter(
				or_(
					PPRRequisition.title.ilike(search_text),
					PPRRequisition.description.ilike(search_text),
					PPRRequisition.requisition_number.ilike(search_text)
				)
			)
		
		return query.order_by(desc(PPRRequisition.request_date)).limit(limit).all()
	
	# Comment Management
	
	def add_requisition_comment(self, requisition_id: str, comment_text: str, user_id: str, 
								comment_type: str = 'General', is_internal: bool = True) -> PPRRequisitionComment:
		"""Add comment to requisition"""
		
		user_info = self._get_user_info(user_id)
		
		comment = PPRRequisitionComment(
			requisition_id=requisition_id,
			tenant_id=self.tenant_id,
			comment_text=comment_text,
			comment_type=comment_type,
			is_internal=is_internal,
			author_id=user_id,
			author_name=user_info.get('name', 'Unknown'),
			author_role=user_info.get('role')
		)
		
		self.db.add(comment)
		self.db.commit()
		
		return comment
	
	def get_requisition_comments(self, requisition_id: str, include_internal: bool = True) -> List[PPRRequisitionComment]:
		"""Get comments for requisition"""
		
		query = self.db.query(PPRRequisitionComment).filter(
			and_(
				PPRRequisitionComment.tenant_id == self.tenant_id,
				PPRRequisitionComment.requisition_id == requisition_id,
				PPRRequisitionComment.comment_status == 'Active'
			)
		)
		
		if not include_internal:
			query = query.filter(PPRRequisitionComment.is_internal == False)
		
		return query.order_by(asc(PPRRequisitionComment.comment_date)).all()
	
	# Approval Workflow Management
	
	def get_approval_workflow_config(self, amount: Decimal, department: str = None, requestor_id: str = None) -> Dict[str, Any]:
		"""Get approval workflow configuration based on business rules"""
		
		# TODO: This should be configurable per tenant
		# For now, using hard-coded rules
		
		steps = []
		
		# Manager approval for amounts > $1,000
		if amount > Decimal('1000.00'):
			manager_info = self._get_manager_info(requestor_id, department)
			if manager_info:
				steps.append({
					'order': 1,
					'name': 'Manager Approval',
					'approver_id': manager_info['id'],
					'approver_name': manager_info['name'],
					'role': 'Manager',
					'required': True,
					'limit': Decimal('25000.00')
				})
		
		# Department Head approval for amounts > $5,000
		if amount > Decimal('5000.00'):
			dept_head_info = self._get_department_head_info(department)
			if dept_head_info:
				steps.append({
					'order': 2,
					'name': 'Department Head Approval',
					'approver_id': dept_head_info['id'],
					'approver_name': dept_head_info['name'],
					'role': 'Department Head',
					'required': True,
					'limit': Decimal('100000.00')
				})
		
		# Finance approval for amounts > $25,000
		if amount > Decimal('25000.00'):
			finance_info = self._get_finance_approver_info()
			if finance_info:
				steps.append({
					'order': 3,
					'name': 'Finance Approval',
					'approver_id': finance_info['id'],
					'approver_name': finance_info['name'],
					'role': 'Finance',
					'required': True,
					'limit': Decimal('500000.00')
				})
		
		# Executive approval for amounts > $100,000
		if amount > Decimal('100000.00'):
			exec_info = self._get_executive_approver_info()
			if exec_info:
				steps.append({
					'order': 4,
					'name': 'Executive Approval',
					'approver_id': exec_info['id'],
					'approver_name': exec_info['name'],
					'role': 'Executive',
					'required': True,
					'limit': None
				})
		
		return {
			'steps': steps,
			'parallel_approval': False,
			'escalation_hours': 48 if amount > Decimal('50000.00') else 72
		}
	
	# Analytics and Reporting
	
	def get_avg_approval_time(self, days: int = 30) -> float:
		"""Get average approval time in hours"""
		
		cutoff_date = datetime.utcnow() - timedelta(days=days)
		
		approved_reqs = self.db.query(PPRRequisition).filter(
			and_(
				PPRRequisition.tenant_id == self.tenant_id,
				PPRRequisition.status == 'Approved',
				PPRRequisition.approved_date >= cutoff_date,
				PPRRequisition.submitted_date.isnot(None)
			)
		).all()
		
		if not approved_reqs:
			return 0.0
		
		total_hours = sum(
			(req.approved_date - req.submitted_date).total_seconds() / 3600
			for req in approved_reqs
		)
		
		return total_hours / len(approved_reqs)
	
	def get_requisition_metrics(self, date_from: date = None, date_to: date = None) -> Dict[str, Any]:
		"""Get requisition metrics for dashboard"""
		
		if not date_from:
			date_from = date.today() - timedelta(days=30)
		if not date_to:
			date_to = date.today()
		
		base_query = self.db.query(PPRRequisition).filter(
			and_(
				PPRRequisition.tenant_id == self.tenant_id,
				PPRRequisition.request_date >= date_from,
				PPRRequisition.request_date <= date_to
			)
		)
		
		# Status counts
		status_counts = {}
		for status in ['Draft', 'Submitted', 'Approved', 'Rejected', 'Cancelled', 'Converted']:
			count = base_query.filter(PPRRequisition.status == status).count()
			status_counts[status.lower()] = count
		
		# Amount totals
		total_amount = base_query.with_entities(func.sum(PPRRequisition.total_amount)).scalar() or 0
		approved_amount = base_query.filter(PPRRequisition.status == 'Approved').with_entities(
			func.sum(PPRRequisition.total_amount)
		).scalar() or 0
		
		# Average amounts
		avg_amount = base_query.with_entities(func.avg(PPRRequisition.total_amount)).scalar() or 0
		
		return {
			'period': {'from': date_from, 'to': date_to},
			'counts': status_counts,
			'amounts': {
				'total_requested': float(total_amount),
				'total_approved': float(approved_amount),
				'average_amount': float(avg_amount)
			},
			'approval_rate': (status_counts['approved'] / max(1, sum(status_counts.values()))) * 100,
			'avg_approval_time_hours': self.get_avg_approval_time()
		}
	
	# Helper Methods
	
	def _create_requisition_line(self, requisition: PPRRequisition, line_data: Dict[str, Any]) -> PPRRequisitionLine:
		"""Create a requisition line"""
		
		line = PPRRequisitionLine(
			requisition_id=requisition.requisition_id,
			tenant_id=self.tenant_id,
			line_number=line_data['line_number'],
			description=line_data['description'],
			detailed_specification=line_data.get('detailed_specification'),
			item_code=line_data.get('item_code'),
			item_description=line_data.get('item_description'),
			item_category=line_data.get('item_category'),
			manufacturer=line_data.get('manufacturer'),
			model_number=line_data.get('model_number'),
			part_number=line_data.get('part_number'),
			quantity_requested=line_data.get('quantity_requested', 1),
			unit_of_measure=line_data.get('unit_of_measure', 'EA'),
			unit_price=line_data.get('unit_price', 0),
			tax_code=line_data.get('tax_code'),
			tax_rate=line_data.get('tax_rate', 0),
			is_tax_inclusive=line_data.get('is_tax_inclusive', False),
			gl_account_id=line_data['gl_account_id'],
			cost_center=line_data.get('cost_center'),
			department=line_data.get('department'),
			project_id=line_data.get('project_id'),
			activity_code=line_data.get('activity_code'),
			required_date=line_data.get('required_date'),
			delivery_location=line_data.get('delivery_location'),
			special_instructions=line_data.get('special_instructions'),
			preferred_vendor_id=line_data.get('preferred_vendor_id'),
			preferred_vendor_name=line_data.get('preferred_vendor_name'),
			vendor_part_number=line_data.get('vendor_part_number'),
			is_asset=line_data.get('is_asset', False),
			asset_category=line_data.get('asset_category'),
			useful_life_years=line_data.get('useful_life_years'),
			is_service=line_data.get('is_service', False),
			service_period_start=line_data.get('service_period_start'),
			service_period_end=line_data.get('service_period_end'),
			warranty_required=line_data.get('warranty_required', False),
			warranty_period=line_data.get('warranty_period'),
			technical_specs=line_data.get('technical_specs'),
			notes=line_data.get('notes')
		)
		
		# Calculate amounts
		line.calculate_line_amount()
		
		requisition.lines.append(line)
		return line
	
	def _generate_requisition_number(self) -> str:
		"""Generate unique requisition number"""
		
		# Get current year and month
		now = datetime.now()
		year = now.year
		month = now.month
		
		# Get next sequence number for this month
		prefix = f"REQ-{year:04d}{month:02d}-"
		
		latest = self.db.query(PPRRequisition).filter(
			and_(
				PPRRequisition.tenant_id == self.tenant_id,
				PPRRequisition.requisition_number.like(f"{prefix}%")
			)
		).order_by(desc(PPRRequisition.requisition_number)).first()
		
		if latest:
			# Extract sequence number and increment
			sequence = int(latest.requisition_number.split('-')[-1]) + 1
		else:
			sequence = 1
		
		return f"{prefix}{sequence:04d}"
	
	def _get_user_info(self, user_id: str) -> Dict[str, Any]:
		"""Get user information"""
		# TODO: Integrate with user management system
		return {
			'id': user_id,
			'name': f'User {user_id}',
			'email': f'{user_id}@company.com',
			'role': 'Employee'
		}
	
	def _get_manager_info(self, user_id: str, department: str = None) -> Optional[Dict[str, Any]]:
		"""Get manager information for user"""
		# TODO: Integrate with HR system
		return {
			'id': f'mgr_{user_id}',
			'name': f'Manager of {user_id}',
			'email': f'mgr_{user_id}@company.com'
		}
	
	def _get_department_head_info(self, department: str) -> Optional[Dict[str, Any]]:
		"""Get department head information"""
		# TODO: Integrate with organizational structure
		return {
			'id': f'head_{department}',
			'name': f'Head of {department}',
			'email': f'head_{department}@company.com'
		}
	
	def _get_finance_approver_info(self) -> Optional[Dict[str, Any]]:
		"""Get finance approver information"""
		# TODO: Get from configuration
		return {
			'id': 'finance_approver',
			'name': 'Finance Manager',
			'email': 'finance@company.com'
		}
	
	def _get_executive_approver_info(self) -> Optional[Dict[str, Any]]:
		"""Get executive approver information"""
		# TODO: Get from configuration
		return {
			'id': 'executive_approver',
			'name': 'Executive Approver',
			'email': 'executive@company.com'
		}
	
	def _send_approval_notifications(self, requisition: PPRRequisition):
		"""Send notifications to current approver"""
		# TODO: Integrate with notification system
		print(f"Notification: Requisition {requisition.requisition_number} needs approval from {requisition.current_approver_name}")
	
	def _send_approval_notification(self, requisition: PPRRequisition, notification_type: str):
		"""Send notification for approval status change"""
		# TODO: Integrate with notification system
		print(f"Notification: Requisition {requisition.requisition_number} status: {notification_type}")
	
	def _log_requisition_activity(self, requisition_id: str, activity_type: str, description: str, user_id: str):
		"""Log requisition activity"""
		# TODO: Integrate with audit logging system
		print(f"Activity Log: {activity_type} - {description} by {user_id}")
	
	def __del__(self):
		"""Cleanup database session"""
		if hasattr(self, 'db'):
			self.db.close()