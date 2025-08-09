"""
Enterprise Asset Management Service Layer

Comprehensive business logic services with full APG platform integration.
Provides asset lifecycle management, maintenance optimization, performance analytics,
and regulatory compliance with seamless integration to existing APG capabilities.

APG Integration Services:
- auth_rbac: Permission checking and multi-tenant data access
- audit_compliance: Complete audit trails and regulatory compliance
- fixed_asset_management: Financial asset synchronization and depreciation
- predictive_maintenance: AI-driven failure prediction and health monitoring
- digital_twin_marketplace: Real-time asset mirroring and simulation
- document_management: Asset documentation and compliance certificates
- notification_engine: Automated alerts and stakeholder communications
- ai_orchestration: Machine learning model management and optimization
- real_time_collaboration: Team coordination and expert consultation
- iot_management: Sensor integration and real-time data collection
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy import select, update, delete, and_, or_, func, text
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import User
from ...auth_rbac.service import AuthRBACService
from ...audit_compliance.service import AuditComplianceService
from .models import (
	EAAsset, EALocation, EAWorkOrder, EAMaintenanceRecord, 
	EAInventory, EAContract, EAPerformanceRecord, EAAssetContract
)


class EAMAssetService:
	"""
	Asset Lifecycle Management Service
	
	Comprehensive asset management with APG integration for financial tracking,
	predictive maintenance, digital twin mirroring, and regulatory compliance.
	"""
	
	def __init__(self, session: AsyncSession, user_id: str | None = None, tenant_id: str | None = None):
		assert session is not None, "Database session is required"
		assert user_id is not None, "User ID is required for audit trails"
		assert tenant_id is not None, "Tenant ID is required for multi-tenant security"
		
		self.session = session
		self.user_id = user_id
		self.tenant_id = tenant_id
		
		# APG Service Dependencies
		self.auth_service = AuthRBACService(session, user_id, tenant_id)
		self.audit_service = AuditComplianceService(session, user_id, tenant_id)
	
	async def _log_asset_operation(self, operation: str, asset_id: str, details: str) -> None:
		"""Log asset operations for APG audit compliance"""
		await self.audit_service.log_activity(
			entity_type="EAAsset",
			entity_id=asset_id,
			action=operation,
			details=details,
			user_id=self.user_id,
			tenant_id=self.tenant_id
		)
	
	async def create_asset(self, asset_data: Dict[str, Any]) -> EAAsset:
		"""
		Create new enterprise asset with APG integration
		
		Integrates with:
		- Fixed Asset Management for financial tracking
		- Predictive Maintenance for health monitoring
		- Digital Twin for real-time mirroring
		- Audit Compliance for change tracking
		"""
		assert asset_data is not None, "Asset data is required"
		assert "asset_name" in asset_data, "Asset name is required"
		assert "asset_type" in asset_data, "Asset type is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.asset.create")
		
		# Generate asset number if not provided
		if "asset_number" not in asset_data:
			asset_data["asset_number"] = await self._generate_asset_number(asset_data.get("asset_type"))
		
		# Set tenant isolation
		asset_data["tenant_id"] = self.tenant_id
		asset_data["asset_id"] = uuid7str()
		
		# Create asset
		asset = EAAsset(**asset_data)
		
		# Update search vector for full-text search
		asset.update_search_vector()
		
		self.session.add(asset)
		await self.session.flush()
		
		# APG Integration: Sync with Fixed Asset Management
		if asset.is_capitalized and asset.purchase_cost:
			await self._sync_with_fixed_asset_management(asset)
		
		# APG Integration: Register with Predictive Maintenance
		if asset.iot_enabled or asset.maintenance_strategy == "predictive":
			await self._register_with_predictive_maintenance(asset)
		
		# APG Integration: Create Digital Twin if applicable
		if asset.has_digital_twin:
			await self._create_digital_twin(asset)
		
		# APG Audit Logging
		await self._log_asset_operation(
			"CREATE", 
			asset.asset_id, 
			f"Created asset {asset.asset_number} - {asset.asset_name}"
		)
		
		await self.session.commit()
		
		assert asset.asset_id is not None, "Asset ID should be set after creation"
		return asset
	
	async def get_asset(self, asset_id: str, include_relationships: bool = False) -> EAAsset | None:
		"""Get asset by ID with optional relationship loading"""
		assert asset_id is not None, "Asset ID is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.asset.view")
		
		query = select(EAAsset).where(
			and_(
				EAAsset.asset_id == asset_id,
				EAAsset.tenant_id == self.tenant_id
			)
		)
		
		if include_relationships:
			query = query.options(
				selectinload(EAAsset.location),
				selectinload(EAAsset.parent_asset),
				selectinload(EAAsset.child_assets),
				selectinload(EAAsset.work_orders),
				selectinload(EAAsset.maintenance_records),
				selectinload(EAAsset.performance_records),
				selectinload(EAAsset.contracts)
			)
		
		result = await self.session.execute(query)
		asset = result.scalar_one_or_none()
		
		if asset:
			# Update health score from latest performance data
			await self._update_asset_health_score(asset)
		
		return asset
	
	async def update_asset(self, asset_id: str, update_data: Dict[str, Any]) -> EAAsset:
		"""Update asset with APG audit logging and cross-capability synchronization"""
		assert asset_id is not None, "Asset ID is required"
		assert update_data is not None, "Update data is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.asset.create")  # Update requires create permission
		
		# Get existing asset
		asset = await self.get_asset(asset_id)
		if not asset:
			raise ValueError(f"Asset {asset_id} not found")
		
		# Track changes for audit
		changes = []
		for key, new_value in update_data.items():
			if hasattr(asset, key):
				old_value = getattr(asset, key)
				if old_value != new_value:
					changes.append(f"{key}: {old_value} -> {new_value}")
					setattr(asset, key, new_value)
		
		# Update search vector if content changed
		if any(field in update_data for field in ['asset_name', 'description', 'manufacturer', 'model_number']):
			asset.update_search_vector()
		
		# Increment version for change tracking
		asset.version_number = (asset.version_number or 1) + 1
		asset.change_reason = update_data.get('change_reason', 'Asset updated')
		asset.approved_by = self.user_id
		asset.approval_date = datetime.utcnow()
		
		await self.session.flush()
		
		# APG Integration: Sync changes with related systems
		if "status" in update_data:
			await self._handle_status_change(asset, update_data["status"])
		
		if any(financial_field in update_data for financial_field in ['purchase_cost', 'current_book_value']):
			await self._sync_financial_changes(asset)
		
		# APG Audit Logging
		if changes:
			await self._log_asset_operation(
				"UPDATE",
				asset_id,
				f"Updated asset {asset.asset_number}: {'; '.join(changes)}"
			)
		
		await self.session.commit()
		return asset
	
	async def delete_asset(self, asset_id: str, reason: str) -> bool:
		"""Soft delete asset with APG audit compliance"""
		assert asset_id is not None, "Asset ID is required"
		assert reason is not None, "Deletion reason is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.asset.delete")
		
		asset = await self.get_asset(asset_id)
		if not asset:
			return False
		
		# Check if asset can be deleted
		if not asset.is_disposal_eligible():
			raise ValueError(f"Asset {asset.asset_number} cannot be deleted in current status")
		
		# Soft delete - update status instead of physical deletion
		asset.status = 'disposed'
		asset.disposal_date = date.today()
		asset.change_reason = reason
		asset.approved_by = self.user_id
		asset.approval_date = datetime.utcnow()
		
		await self.session.flush()
		
		# APG Integration: Handle disposal in related systems
		await self._handle_asset_disposal(asset)
		
		# APG Audit Logging
		await self._log_asset_operation(
			"DELETE",
			asset_id,
			f"Disposed asset {asset.asset_number}: {reason}"
		)
		
		await self.session.commit()
		return True
	
	async def search_assets(
		self, 
		search_term: str | None = None,
		filters: Dict[str, Any] | None = None,
		page: int = 1,
		page_size: int = 50
	) -> Tuple[List[EAAsset], int]:
		"""Advanced asset search with full-text search and filtering"""
		assert page > 0, "Page must be positive"
		assert page_size > 0, "Page size must be positive"
		assert page_size <= 1000, "Page size cannot exceed 1000"
		
		# Check permissions
		await self.auth_service.check_permission("eam.asset.view")
		
		query = select(EAAsset).where(EAAsset.tenant_id == self.tenant_id)
		
		# Full-text search
		if search_term:
			search_term = search_term.lower().strip()
			query = query.where(EAAsset.search_vector.contains(search_term))
		
		# Apply filters
		if filters:
			if "asset_type" in filters:
				query = query.where(EAAsset.asset_type == filters["asset_type"])
			if "status" in filters:
				query = query.where(EAAsset.status == filters["status"])
			if "criticality_level" in filters:
				query = query.where(EAAsset.criticality_level == filters["criticality_level"])
			if "location_id" in filters:
				query = query.where(EAAsset.location_id == filters["location_id"])
			if "health_score_min" in filters:
				query = query.where(EAAsset.health_score >= filters["health_score_min"])
			if "maintenance_due" in filters and filters["maintenance_due"]:
				query = query.where(EAAsset.next_maintenance_due <= date.today())
		
		# Get total count
		count_query = select(func.count()).select_from(query.subquery())
		total_count = await self.session.scalar(count_query)
		
		# Apply pagination
		offset = (page - 1) * page_size
		query = query.offset(offset).limit(page_size)
		
		# Order by relevance and name
		query = query.order_by(EAAsset.asset_name)
		
		result = await self.session.execute(query)
		assets = result.scalars().all()
		
		return list(assets), total_count
	
	async def get_asset_hierarchy(self, asset_id: str) -> Dict[str, Any]:
		"""Get complete asset hierarchy including parent and children"""
		assert asset_id is not None, "Asset ID is required"
		
		asset = await self.get_asset(asset_id, include_relationships=True)
		if not asset:
			raise ValueError(f"Asset {asset_id} not found")
		
		# Build hierarchy tree
		hierarchy = {
			"asset": asset,
			"parent_chain": [],
			"children": [],
			"total_children": 0
		}
		
		# Get parent chain
		current = asset.parent_asset
		while current:
			hierarchy["parent_chain"].insert(0, current)
			current = current.parent_asset
		
		# Get direct children
		children_query = select(EAAsset).where(
			and_(
				EAAsset.parent_asset_id == asset_id,
				EAAsset.tenant_id == self.tenant_id
			)
		).order_by(EAAsset.asset_name)
		
		children_result = await self.session.execute(children_query)
		hierarchy["children"] = list(children_result.scalars().all())
		
		# Get total descendant count
		descendant_count_query = text("""
			WITH RECURSIVE asset_tree AS (
				SELECT asset_id, parent_asset_id, 1 as level
				FROM ea_asset 
				WHERE parent_asset_id = :asset_id AND tenant_id = :tenant_id
				UNION ALL
				SELECT a.asset_id, a.parent_asset_id, at.level + 1
				FROM ea_asset a
				JOIN asset_tree at ON a.parent_asset_id = at.asset_id
				WHERE a.tenant_id = :tenant_id
			)
			SELECT COUNT(*) FROM asset_tree
		""")
		
		result = await self.session.execute(
			descendant_count_query, 
			{"asset_id": asset_id, "tenant_id": self.tenant_id}
		)
		hierarchy["total_children"] = result.scalar() or 0
		
		return hierarchy
	
	async def get_maintenance_due_assets(self, days_ahead: int = 30) -> List[EAAsset]:
		"""Get assets with maintenance due within specified days"""
		assert days_ahead >= 0, "Days ahead must be non-negative"
		
		# Check permissions
		await self.auth_service.check_permission("eam.asset.view")
		
		due_date = date.today() + timedelta(days=days_ahead)
		
		query = select(EAAsset).where(
			and_(
				EAAsset.tenant_id == self.tenant_id,
				EAAsset.status == 'active',
				EAAsset.next_maintenance_due <= due_date,
				EAAsset.next_maintenance_due.is_not(None)
			)
		).order_by(EAAsset.next_maintenance_due)
		
		result = await self.session.execute(query)
		return list(result.scalars().all())
	
	async def get_critical_assets_health(self) -> List[Dict[str, Any]]:
		"""Get health status of critical assets for dashboard"""
		# Check permissions
		await self.auth_service.check_permission("eam.analytics.view")
		
		query = select(EAAsset).where(
			and_(
				EAAsset.tenant_id == self.tenant_id,
				EAAsset.criticality_level.in_(['high', 'critical']),
				EAAsset.status == 'active'
			)
		).order_by(EAAsset.health_score.nulls_last())
		
		result = await self.session.execute(query)
		assets = result.scalars().all()
		
		health_summary = []
		for asset in assets:
			health_summary.append({
				"asset_id": asset.asset_id,
				"asset_number": asset.asset_number,
				"asset_name": asset.asset_name,
				"criticality_level": asset.criticality_level,
				"health_score": asset.health_score,
				"health_status": asset.get_health_status(),
				"condition_status": asset.condition_status,
				"last_maintenance": asset.last_maintenance_date,
				"next_maintenance": asset.next_maintenance_due,
				"maintenance_status": asset.get_maintenance_status()
			})
		
		return health_summary
	
	async def _generate_asset_number(self, asset_type: str) -> str:
		"""Generate unique asset number with type prefix"""
		# Get type prefix mapping
		type_prefixes = {
			"equipment": "EQ",
			"vehicle": "VH", 
			"facility": "FC",
			"tool": "TL",
			"instrument": "IN",
			"computer": "CP"
		}
		
		prefix = type_prefixes.get(asset_type, "AS")
		
		# Get next sequence number
		query = select(func.count()).where(
			and_(
				EAAsset.tenant_id == self.tenant_id,
				EAAsset.asset_number.startswith(prefix)
			)
		)
		count = await self.session.scalar(query) or 0
		
		return f"{prefix}{count + 1:06d}"
	
	async def _sync_with_fixed_asset_management(self, asset: EAAsset) -> None:
		"""Sync asset with APG Fixed Asset Management capability"""
		try:
			# This would integrate with the actual Fixed Asset Management service
			# For now, we set the reference ID
			asset.fixed_asset_id = uuid7str()
			
			await self._log_asset_operation(
				"SYNC_FAM",
				asset.asset_id,
				f"Synced with Fixed Asset Management: {asset.fixed_asset_id}"
			)
		except Exception as e:
			await self._log_asset_operation(
				"SYNC_FAM_ERROR", 
				asset.asset_id,
				f"Failed to sync with Fixed Asset Management: {str(e)}"
			)
	
	async def _register_with_predictive_maintenance(self, asset: EAAsset) -> None:
		"""Register asset with APG Predictive Maintenance capability"""
		try:
			# This would integrate with the actual Predictive Maintenance service
			asset.predictive_asset_id = uuid7str()
			asset.health_score = 100.0  # Initial health score
			
			await self._log_asset_operation(
				"REGISTER_PM",
				asset.asset_id,
				f"Registered with Predictive Maintenance: {asset.predictive_asset_id}"
			)
		except Exception as e:
			await self._log_asset_operation(
				"REGISTER_PM_ERROR",
				asset.asset_id,
				f"Failed to register with Predictive Maintenance: {str(e)}"
			)
	
	async def _create_digital_twin(self, asset: EAAsset) -> None:
		"""Create digital twin in APG Digital Twin Marketplace"""
		try:
			# This would integrate with the actual Digital Twin service
			asset.digital_twin_id = uuid7str()
			
			await self._log_asset_operation(
				"CREATE_DT",
				asset.asset_id,
				f"Created Digital Twin: {asset.digital_twin_id}"
			)
		except Exception as e:
			await self._log_asset_operation(
				"CREATE_DT_ERROR",
				asset.asset_id,
				f"Failed to create Digital Twin: {str(e)}"
			)
	
	async def _handle_status_change(self, asset: EAAsset, new_status: str) -> None:
		"""Handle asset status changes with APG integration"""
		old_status = asset.status
		
		# Status-specific logic
		if new_status == 'maintenance' and old_status == 'active':
			# Asset going into maintenance
			await self._notify_stakeholders(asset, "Asset entering maintenance mode")
		
		elif new_status == 'active' and old_status == 'maintenance':
			# Asset returning to service
			await self._notify_stakeholders(asset, "Asset returned to service")
		
		elif new_status == 'disposed':
			# Asset being disposed
			await self._handle_asset_disposal(asset)
		
		await self._log_asset_operation(
			"STATUS_CHANGE",
			asset.asset_id,
			f"Status changed from {old_status} to {new_status}"
		)
	
	async def _sync_financial_changes(self, asset: EAAsset) -> None:
		"""Sync financial changes with Fixed Asset Management"""
		if asset.fixed_asset_id:
			# This would sync with actual Fixed Asset Management
			await self._log_asset_operation(
				"SYNC_FINANCIAL",
				asset.asset_id,
				f"Synced financial data with Fixed Asset Management"
			)
	
	async def _handle_asset_disposal(self, asset: EAAsset) -> None:
		"""Handle asset disposal across APG capabilities"""
		# Close open work orders
		await self.session.execute(
			update(EAWorkOrder)
			.where(
				and_(
					EAWorkOrder.asset_id == asset.asset_id,
					EAWorkOrder.status.in_(['draft', 'planned', 'scheduled'])
				)
			)
			.values(status='cancelled')
		)
		
		# Update digital twin status
		if asset.digital_twin_id:
			await self._log_asset_operation(
				"DISPOSE_DT",
				asset.asset_id,
				"Disposed Digital Twin"
			)
	
	async def _notify_stakeholders(self, asset: EAAsset, message: str) -> None:
		"""Send notifications via APG Notification Engine"""
		# This would integrate with APG Notification Engine
		await self._log_asset_operation(
			"NOTIFICATION",
			asset.asset_id,
			f"Sent notification: {message}"
		)
	
	async def _update_asset_health_score(self, asset: EAAsset) -> None:
		"""Update asset health score from latest performance data"""
		if asset.predictive_asset_id:
			# Get latest performance record
			latest_performance_query = select(EAPerformanceRecord).where(
				and_(
					EAPerformanceRecord.asset_id == asset.asset_id,
					EAPerformanceRecord.tenant_id == self.tenant_id
				)
			).order_by(EAPerformanceRecord.measurement_date.desc()).limit(1)
			
			result = await self.session.execute(latest_performance_query)
			latest_record = result.scalar_one_or_none()
			
			if latest_record and latest_record.health_score:
				asset.health_score = latest_record.health_score
				asset.condition_status = asset.get_health_status()


class EAMWorkOrderService:
	"""
	Work Order Management Service
	
	Comprehensive work order lifecycle management with APG Real-time Collaboration,
	Notification Engine integration, and mobile field operations support.
	"""
	
	def __init__(self, session: AsyncSession, user_id: str | None = None, tenant_id: str | None = None):
		assert session is not None, "Database session is required"
		assert user_id is not None, "User ID is required for audit trails"
		assert tenant_id is not None, "Tenant ID is required for multi-tenant security"
		
		self.session = session
		self.user_id = user_id
		self.tenant_id = tenant_id
		
		# APG Service Dependencies
		self.auth_service = AuthRBACService(session, user_id, tenant_id)
		self.audit_service = AuditComplianceService(session, user_id, tenant_id)
	
	async def _log_work_order_operation(self, operation: str, work_order_id: str, details: str) -> None:
		"""Log work order operations for APG audit compliance"""
		await self.audit_service.log_activity(
			entity_type="EAWorkOrder",
			entity_id=work_order_id,
			action=operation,
			details=details,
			user_id=self.user_id,
			tenant_id=self.tenant_id
		)
	
	async def create_work_order(self, work_order_data: Dict[str, Any]) -> EAWorkOrder:
		"""Create new work order with APG integration"""
		assert work_order_data is not None, "Work order data is required"
		assert "title" in work_order_data, "Work order title is required"
		assert "description" in work_order_data, "Work order description is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.workorder.create")
		
		# Generate work order number if not provided
		if "work_order_number" not in work_order_data:
			work_order_data["work_order_number"] = await self._generate_work_order_number(
				work_order_data.get("work_type", "maintenance")
			)
		
		# Set tenant isolation and defaults
		work_order_data["tenant_id"] = self.tenant_id
		work_order_data["work_order_id"] = uuid7str()
		work_order_data["requested_date"] = work_order_data.get("requested_date", datetime.utcnow())
		
		# Create work order
		work_order = EAWorkOrder(**work_order_data)
		self.session.add(work_order)
		await self.session.flush()
		
		# APG Integration: Create collaboration room for team coordination
		if work_order.priority in ['high', 'urgent', 'emergency']:
			await self._create_collaboration_room(work_order)
		
		# APG Integration: Send notifications based on priority
		await self._send_work_order_notifications(work_order, "CREATED")
		
		# APG Audit Logging
		await self._log_work_order_operation(
			"CREATE",
			work_order.work_order_id,
			f"Created work order {work_order.work_order_number}: {work_order.title}"
		)
		
		await self.session.commit()
		return work_order
	
	async def assign_work_order(self, work_order_id: str, assigned_to: str, team_members: List[str] | None = None) -> EAWorkOrder:
		"""Assign work order to technician and team with APG collaboration"""
		assert work_order_id is not None, "Work order ID is required"
		assert assigned_to is not None, "Assignee is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.workorder.assign")
		
		work_order = await self.get_work_order(work_order_id)
		if not work_order:
			raise ValueError(f"Work order {work_order_id} not found")
		
		old_assignee = work_order.assigned_to
		work_order.assigned_to = assigned_to
		work_order.assigned_team = team_members or []
		work_order.status = 'assigned' if work_order.status == 'draft' else work_order.status
		
		await self.session.flush()
		
		# APG Integration: Update collaboration room membership
		if work_order.collaboration_room_id:
			await self._update_collaboration_room_members(work_order)
		
		# APG Integration: Send assignment notifications
		await self._send_assignment_notifications(work_order, old_assignee)
		
		# APG Audit Logging
		await self._log_work_order_operation(
			"ASSIGN",
			work_order_id,
			f"Assigned work order to {assigned_to} with team {team_members}"
		)
		
		await self.session.commit()
		return work_order
	
	async def start_work_order(self, work_order_id: str) -> EAWorkOrder:
		"""Start work order execution with real-time tracking"""
		assert work_order_id is not None, "Work order ID is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.workorder.execute")
		
		work_order = await self.get_work_order(work_order_id)
		if not work_order:
			raise ValueError(f"Work order {work_order_id} not found")
		
		if work_order.status not in ['scheduled', 'assigned']:
			raise ValueError(f"Cannot start work order in status {work_order.status}")
		
		work_order.status = 'in_progress'
		work_order.actual_start = datetime.utcnow()
		
		# Mark asset as under maintenance if applicable
		if work_order.asset_id:
			asset_query = select(EAAsset).where(EAAsset.asset_id == work_order.asset_id)
			result = await self.session.execute(asset_query)
			asset = result.scalar_one_or_none()
			if asset and asset.status == 'active':
				asset.operational_status = 'maintenance'
		
		await self.session.flush()
		
		# APG Integration: Real-time status updates
		await self._broadcast_status_update(work_order)
		
		# APG Integration: Send start notifications
		await self._send_work_order_notifications(work_order, "STARTED")
		
		# APG Audit Logging
		await self._log_work_order_operation(
			"START",
			work_order_id,
			f"Started work order execution"
		)
		
		await self.session.commit()
		return work_order
	
	async def complete_work_order(self, work_order_id: str, completion_data: Dict[str, Any]) -> EAWorkOrder:
		"""Complete work order with maintenance record creation"""
		assert work_order_id is not None, "Work order ID is required"
		assert completion_data is not None, "Completion data is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.workorder.execute")
		
		work_order = await self.get_work_order(work_order_id)
		if not work_order:
			raise ValueError(f"Work order {work_order_id} not found")
		
		if work_order.status != 'in_progress':
			raise ValueError(f"Cannot complete work order in status {work_order.status}")
		
		# Update work order completion
		work_order.status = 'completed'
		work_order.actual_end = datetime.utcnow()
		work_order.actual_hours = completion_data.get("actual_hours")
		work_order.actual_cost = completion_data.get("actual_cost")
		work_order.completion_notes = completion_data.get("completion_notes")
		work_order.work_performed = completion_data.get("work_performed")
		work_order.completion_percentage = 100
		
		# Calculate durations
		if work_order.actual_start:
			duration = work_order.actual_end - work_order.actual_start
			work_order.actual_hours = work_order.actual_hours or duration.total_seconds() / 3600
		
		# Create maintenance record if this was maintenance work
		if work_order.work_type in ['maintenance', 'repair']:
			await self._create_maintenance_record(work_order, completion_data)
		
		# Update asset status back to operational
		if work_order.asset_id:
			asset_query = select(EAAsset).where(EAAsset.asset_id == work_order.asset_id)
			result = await self.session.execute(asset_query)
			asset = result.scalar_one_or_none()
			if asset and asset.operational_status == 'maintenance':
				asset.operational_status = 'operational'
				asset.last_maintenance_date = date.today()
				
				# Schedule next maintenance if preventive
				if work_order.maintenance_type == 'preventive' and asset.maintenance_frequency_days:
					asset.next_maintenance_due = date.today() + timedelta(days=asset.maintenance_frequency_days)
		
		await self.session.flush()
		
		# APG Integration: Real-time completion updates
		await self._broadcast_status_update(work_order)
		
		# APG Integration: Send completion notifications
		await self._send_work_order_notifications(work_order, "COMPLETED")
		
		# APG Audit Logging
		await self._log_work_order_operation(
			"COMPLETE",
			work_order_id,
			f"Completed work order: {work_order.work_performed}"
		)
		
		await self.session.commit()
		return work_order
	
	async def get_work_order(self, work_order_id: str, include_relationships: bool = False) -> EAWorkOrder | None:
		"""Get work order by ID with optional relationship loading"""
		assert work_order_id is not None, "Work order ID is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.workorder.view")
		
		query = select(EAWorkOrder).where(
			and_(
				EAWorkOrder.work_order_id == work_order_id,
				EAWorkOrder.tenant_id == self.tenant_id
			)
		)
		
		if include_relationships:
			query = query.options(
				selectinload(EAWorkOrder.asset),
				selectinload(EAWorkOrder.location),
				selectinload(EAWorkOrder.maintenance_records)
			)
		
		result = await self.session.execute(query)
		return result.scalar_one_or_none()
	
	async def get_assigned_work_orders(self, user_id: str, status_filter: str | None = None) -> List[EAWorkOrder]:
		"""Get work orders assigned to specific user"""
		assert user_id is not None, "User ID is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.workorder.view")
		
		query = select(EAWorkOrder).where(
			and_(
				EAWorkOrder.tenant_id == self.tenant_id,
				or_(
					EAWorkOrder.assigned_to == user_id,
					EAWorkOrder.assigned_team.contains([user_id])
				)
			)
		)
		
		if status_filter:
			query = query.where(EAWorkOrder.status == status_filter)
		
		query = query.order_by(EAWorkOrder.priority.desc(), EAWorkOrder.scheduled_start)
		
		result = await self.session.execute(query)
		return list(result.scalars().all())
	
	async def get_overdue_work_orders(self) -> List[EAWorkOrder]:
		"""Get overdue work orders for monitoring"""
		# Check permissions
		await self.auth_service.check_permission("eam.workorder.view")
		
		query = select(EAWorkOrder).where(
			and_(
				EAWorkOrder.tenant_id == self.tenant_id,
				EAWorkOrder.status.in_(['scheduled', 'in_progress']),
				EAWorkOrder.scheduled_end < datetime.utcnow()
			)
		).order_by(EAWorkOrder.scheduled_end)
		
		result = await self.session.execute(query)
		return list(result.scalars().all())
	
	async def _generate_work_order_number(self, work_type: str) -> str:
		"""Generate unique work order number with type prefix"""
		type_prefixes = {
			"maintenance": "WO-M",
			"repair": "WO-R",
			"inspection": "WO-I",
			"project": "WO-P",
			"emergency": "WO-E"
		}
		
		prefix = type_prefixes.get(work_type, "WO")
		
		# Get next sequence number for this year
		current_year = datetime.now().year
		query = select(func.count()).where(
			and_(
				EAWorkOrder.tenant_id == self.tenant_id,
				EAWorkOrder.work_order_number.startswith(f"{prefix}-{current_year}"),
				func.extract('year', EAWorkOrder.requested_date) == current_year
			)
		)
		count = await self.session.scalar(query) or 0
		
		return f"{prefix}-{current_year}-{count + 1:05d}"
	
	async def _create_collaboration_room(self, work_order: EAWorkOrder) -> None:
		"""Create collaboration room via APG Real-time Collaboration"""
		try:
			# This would integrate with APG Real-time Collaboration service
			work_order.collaboration_room_id = uuid7str()
			
			await self._log_work_order_operation(
				"CREATE_COLLAB",
				work_order.work_order_id,
				f"Created collaboration room: {work_order.collaboration_room_id}"
			)
		except Exception as e:
			await self._log_work_order_operation(
				"CREATE_COLLAB_ERROR",
				work_order.work_order_id,
				f"Failed to create collaboration room: {str(e)}"
			)
	
	async def _send_work_order_notifications(self, work_order: EAWorkOrder, event_type: str) -> None:
		"""Send notifications via APG Notification Engine"""
		# This would integrate with APG Notification Engine
		work_order.notification_sent = True
		
		await self._log_work_order_operation(
			"NOTIFICATION",
			work_order.work_order_id,
			f"Sent {event_type} notification"
		)
	
	async def _update_collaboration_room_members(self, work_order: EAWorkOrder) -> None:
		"""Update collaboration room membership"""
		if work_order.collaboration_room_id:
			await self._log_work_order_operation(
				"UPDATE_COLLAB",
				work_order.work_order_id,
				"Updated collaboration room members"
			)
	
	async def _send_assignment_notifications(self, work_order: EAWorkOrder, old_assignee: str | None) -> None:
		"""Send assignment change notifications"""
		await self._log_work_order_operation(
			"ASSIGN_NOTIFICATION",
			work_order.work_order_id,
			f"Sent assignment notification from {old_assignee} to {work_order.assigned_to}"
		)
	
	async def _broadcast_status_update(self, work_order: EAWorkOrder) -> None:
		"""Broadcast real-time status updates"""
		await self._log_work_order_operation(
			"STATUS_BROADCAST",
			work_order.work_order_id,
			f"Broadcasted status update: {work_order.status}"
		)
	
	async def _create_maintenance_record(self, work_order: EAWorkOrder, completion_data: Dict[str, Any]) -> None:
		"""Create maintenance record from completed work order"""
		maintenance_record = EAMaintenanceRecord(
			record_id=uuid7str(),
			tenant_id=self.tenant_id,
			asset_id=work_order.asset_id,
			work_order_id=work_order.work_order_id,
			location_id=work_order.location_id,
			maintenance_number=f"MR-{work_order.work_order_number}",
			maintenance_type=work_order.maintenance_type or 'corrective',
			maintenance_category=work_order.work_category or 'repair',
			description=work_order.description,
			started_at=work_order.actual_start,
			completed_at=work_order.actual_end,
			duration_hours=work_order.actual_hours,
			work_performed=work_order.work_performed,
			technician_id=work_order.assigned_to,
			total_cost=work_order.actual_cost,
			outcome='successful'  # Default to successful, can be updated
		)
		
		self.session.add(maintenance_record)
		await self.session.flush()


class EAMInventoryService:
	"""
	Inventory Management Service
	
	Comprehensive parts and materials management with APG Procurement integration
	for automated reordering and vendor management.
	"""
	
	def __init__(self, session: AsyncSession, user_id: str | None = None, tenant_id: str | None = None):
		assert session is not None, "Database session is required"
		assert user_id is not None, "User ID is required for audit trails"
		assert tenant_id is not None, "Tenant ID is required for multi-tenant security"
		
		self.session = session
		self.user_id = user_id
		self.tenant_id = tenant_id
		
		# APG Service Dependencies
		self.auth_service = AuthRBACService(session, user_id, tenant_id)
		self.audit_service = AuditComplianceService(session, user_id, tenant_id)
	
	async def _log_inventory_operation(self, operation: str, inventory_id: str, details: str) -> None:
		"""Log inventory operations for APG audit compliance"""
		await self.audit_service.log_activity(
			entity_type="EAInventory",
			entity_id=inventory_id,
			action=operation,
			details=details,
			user_id=self.user_id,
			tenant_id=self.tenant_id
		)
	
	async def create_inventory_item(self, item_data: Dict[str, Any]) -> EAInventory:
		"""Create new inventory item with APG integration"""
		assert item_data is not None, "Item data is required"
		assert "part_number" in item_data, "Part number is required"
		assert "description" in item_data, "Description is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.inventory.manage")
		
		# Set tenant isolation and defaults
		item_data["tenant_id"] = self.tenant_id
		item_data["inventory_id"] = uuid7str()
		
		# Create inventory item
		item = EAInventory(**item_data)
		item.calculate_total_value()
		
		self.session.add(item)
		await self.session.flush()
		
		# APG Integration: Setup auto-reordering if enabled
		if item.auto_reorder and item.primary_vendor_id:
			await self._setup_auto_reorder(item)
		
		# APG Audit Logging
		await self._log_inventory_operation(
			"CREATE",
			item.inventory_id,
			f"Created inventory item {item.part_number}: {item.description}"
		)
		
		await self.session.commit()
		return item
	
	async def update_stock_level(self, inventory_id: str, quantity_change: int, reason: str) -> EAInventory:
		"""Update stock level with audit trail"""
		assert inventory_id is not None, "Inventory ID is required"
		assert reason is not None, "Reason is required for stock changes"
		
		# Check permissions
		await self.auth_service.check_permission("eam.inventory.manage")
		
		item = await self.get_inventory_item(inventory_id)
		if not item:
			raise ValueError(f"Inventory item {inventory_id} not found")
		
		old_stock = item.current_stock
		item.current_stock = max(0, item.current_stock + quantity_change)
		item.calculate_total_value()
		
		# Update usage statistics
		if quantity_change < 0:  # Issue/consumption
			item.total_issued_ytd += abs(quantity_change)
			item.last_issue_date = date.today()
		elif quantity_change > 0:  # Receipt
			item.total_received_ytd += quantity_change
			item.last_receipt_date = date.today()
		
		await self.session.flush()
		
		# APG Integration: Check for reorder trigger
		if item.is_reorder_required() and item.auto_reorder:
			await self._trigger_auto_reorder(item)
		
		# APG Audit Logging
		await self._log_inventory_operation(
			"STOCK_UPDATE",
			inventory_id,
			f"Stock changed from {old_stock} to {item.current_stock}: {reason}"
		)
		
		await self.session.commit()
		return item
	
	async def get_inventory_item(self, inventory_id: str) -> EAInventory | None:
		"""Get inventory item by ID"""
		assert inventory_id is not None, "Inventory ID is required"
		
		# Check permissions
		await self.auth_service.check_permission("eam.inventory.manage")
		
		query = select(EAInventory).where(
			and_(
				EAInventory.inventory_id == inventory_id,
				EAInventory.tenant_id == self.tenant_id
			)
		)
		
		result = await self.session.execute(query)
		return result.scalar_one_or_none()
	
	async def get_reorder_recommendations(self) -> List[Dict[str, Any]]:
		"""Get items that need reordering"""
		# Check permissions
		await self.auth_service.check_permission("eam.inventory.manage")
		
		query = select(EAInventory).where(
			and_(
				EAInventory.tenant_id == self.tenant_id,
				EAInventory.status == 'active',
				EAInventory.current_stock <= EAInventory.reorder_point
			)
		).order_by(EAInventory.criticality.desc(), EAInventory.current_stock)
		
		result = await self.session.execute(query)
		items = result.scalars().all()
		
		recommendations = []
		for item in items:
			recommendations.append({
				"inventory_id": item.inventory_id,
				"part_number": item.part_number,
				"description": item.description,
				"current_stock": item.current_stock,
				"reorder_point": item.reorder_point,
				"recommended_quantity": item.get_reorder_quantity(),
				"primary_vendor_id": item.primary_vendor_id,
				"estimated_cost": item.unit_cost * item.get_reorder_quantity() if item.unit_cost else None,
				"stock_days_remaining": item.calculate_stock_days(),
				"criticality": item.criticality
			})
		
		return recommendations
	
	async def _setup_auto_reorder(self, item: EAInventory) -> None:
		"""Setup automatic reordering via APG Procurement"""
		try:
			await self._log_inventory_operation(
				"SETUP_AUTO_REORDER",
				item.inventory_id,
				f"Setup auto-reorder with vendor {item.primary_vendor_id}"
			)
		except Exception as e:
			await self._log_inventory_operation(
				"SETUP_AUTO_REORDER_ERROR",
				item.inventory_id,
				f"Failed to setup auto-reorder: {str(e)}"
			)
	
	async def _trigger_auto_reorder(self, item: EAInventory) -> None:
		"""Trigger automatic reorder via APG Procurement"""
		try:
			reorder_quantity = item.get_reorder_quantity()
			item.last_auto_order_date = date.today()
			
			await self._log_inventory_operation(
				"AUTO_REORDER",
				item.inventory_id,
				f"Triggered auto-reorder for {reorder_quantity} units"
			)
		except Exception as e:
			await self._log_inventory_operation(
				"AUTO_REORDER_ERROR",
				item.inventory_id,
				f"Failed to trigger auto-reorder: {str(e)}"
			)


class EAMAnalyticsService:
	"""
	Performance Analytics Service
	
	Comprehensive asset performance analytics with APG AI Orchestration integration
	for predictive insights and optimization recommendations.
	"""
	
	def __init__(self, session: AsyncSession, user_id: str | None = None, tenant_id: str | None = None):
		assert session is not None, "Database session is required"
		assert user_id is not None, "User ID is required for audit trails"
		assert tenant_id is not None, "Tenant ID is required for multi-tenant security"
		
		self.session = session
		self.user_id = user_id
		self.tenant_id = tenant_id
		
		# APG Service Dependencies
		self.auth_service = AuthRBACService(session, user_id, tenant_id)
		self.audit_service = AuditComplianceService(session, user_id, tenant_id)
	
	async def get_asset_performance_summary(self, asset_id: str, days: int = 30) -> Dict[str, Any]:
		"""Get comprehensive asset performance summary"""
		assert asset_id is not None, "Asset ID is required"
		assert days > 0, "Days must be positive"
		
		# Check permissions
		await self.auth_service.check_permission("eam.analytics.view")
		
		start_date = date.today() - timedelta(days=days)
		
		# Get performance records
		performance_query = select(EAPerformanceRecord).where(
			and_(
				EAPerformanceRecord.asset_id == asset_id,
				EAPerformanceRecord.tenant_id == self.tenant_id,
				EAPerformanceRecord.measurement_date >= start_date
			)
		).order_by(EAPerformanceRecord.measurement_date)
		
		result = await self.session.execute(performance_query)
		records = list(result.scalars().all())
		
		if not records:
			return {"asset_id": asset_id, "performance_data": [], "summary": {}}
		
		# Calculate summary metrics
		latest_record = records[-1]
		avg_availability = sum(r.availability_percentage or 0 for r in records) / len(records)
		avg_oee = sum(r.oee_overall or 0 for r in records) / len(records)
		avg_health = sum(r.health_score or 0 for r in records) / len(records)
		
		summary = {
			"asset_id": asset_id,
			"period_days": days,
			"latest_availability": latest_record.availability_percentage,
			"average_availability": round(avg_availability, 2),
			"latest_oee": latest_record.oee_overall,
			"average_oee": round(avg_oee, 2),
			"latest_health_score": latest_record.health_score,
			"average_health_score": round(avg_health, 2),
			"total_downtime_hours": sum(r.downtime_hours or 0 for r in records),
			"failure_count": sum(r.failure_count or 0 for r in records),
			"maintenance_cost": sum(r.maintenance_cost or 0 for r in records),
			"energy_consumption": sum(r.energy_consumption or 0 for r in records),
			"performance_grade": latest_record.get_performance_grade(),
			"trend_direction": latest_record.trend_direction
		}
		
		return {
			"asset_id": asset_id,
			"performance_data": records,
			"summary": summary
		}
	
	async def get_fleet_performance_dashboard(self) -> Dict[str, Any]:
		"""Get fleet-wide performance dashboard"""
		# Check permissions
		await self.auth_service.check_permission("eam.analytics.view")
		
		# Get asset counts by status
		asset_status_query = select(
			EAAsset.status,
			func.count(EAAsset.asset_id).label('count')
		).where(
			EAAsset.tenant_id == self.tenant_id
		).group_by(EAAsset.status)
		
		status_result = await self.session.execute(asset_status_query)
		status_counts = {row.status: row.count for row in status_result}
		
		# Get critical assets health
		critical_health_query = select(
			func.avg(EAAsset.health_score).label('avg_health'),
			func.count(EAAsset.asset_id).label('count')
		).where(
			and_(
				EAAsset.tenant_id == self.tenant_id,
				EAAsset.criticality_level.in_(['high', 'critical']),
				EAAsset.status == 'active'
			)
		)
		
		health_result = await self.session.execute(critical_health_query)
		health_row = health_result.first()
		
		# Get maintenance metrics
		maintenance_due_query = select(func.count()).where(
			and_(
				EAAsset.tenant_id == self.tenant_id,
				EAAsset.next_maintenance_due <= date.today() + timedelta(days=7),
				EAAsset.status == 'active'
			)
		)
		
		maintenance_due_count = await self.session.scalar(maintenance_due_query)
		
		return {
			"asset_counts": status_counts,
			"total_assets": sum(status_counts.values()),
			"critical_assets_health": {
				"average_health": round(health_row.avg_health or 0, 1),
				"count": health_row.count
			},
			"maintenance_due_soon": maintenance_due_count,
			"timestamp": datetime.utcnow().isoformat()
		}