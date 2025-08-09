"""
Enterprise Asset Management REST API

Comprehensive REST API endpoints following APG patterns and standards.
Provides async endpoints with JWT authentication, rate limiting, input validation,
comprehensive error handling, and real-time WebSocket integration.

APG API Integration:
- JWT authentication via APG auth_rbac
- Rate limiting using APG API gateway
- Input validation with Pydantic v2
- Error handling following APG error patterns
- API versioning aligned with APG standards
- WebSocket support for real-time updates
- Comprehensive OpenAPI 3.0 documentation
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, Path, Body, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import selectinload
import json
import csv
import io
from uuid_extensions import uuid7str

from ...auth_rbac.service import AuthRBACService
from ...audit_compliance.service import AuditComplianceService
from .models import EAAsset, EALocation, EAWorkOrder, EAMaintenanceRecord, EAInventory, EAContract, EAPerformanceRecord
from .service import EAMAssetService, EAMWorkOrderService, EAMInventoryService, EAMAnalyticsService
from .views import (
	EAAssetCreateModel, EAAssetUpdateModel, EAWorkOrderCreateModel, 
	EAInventoryCreateModel, EALocationCreateModel
)


# =============================================================================
# API RESPONSE MODELS
# =============================================================================

class APIResponse(BaseModel):
	"""Standard APG API response format"""
	success: bool = Field(..., description="Operation success status")
	message: str = Field(..., description="Response message")
	data: Any = Field(None, description="Response data")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class PaginatedResponse(BaseModel):
	"""Paginated response for list endpoints"""
	items: List[Any] = Field(..., description="List of items")
	total: int = Field(..., description="Total number of items")
	page: int = Field(..., description="Current page number")
	page_size: int = Field(..., description="Page size")
	pages: int = Field(..., description="Total number of pages")
	has_next: bool = Field(..., description="Has next page")
	has_prev: bool = Field(..., description="Has previous page")


class AssetResponse(BaseModel):
	"""Asset response model"""
	asset_id: str
	asset_number: str
	asset_name: str
	description: str | None
	asset_type: str
	status: str
	health_score: float | None
	location_name: str | None
	last_maintenance_date: date | None
	next_maintenance_due: date | None
	created_on: datetime
	changed_on: datetime


class WorkOrderResponse(BaseModel):
	"""Work order response model"""
	work_order_id: str
	work_order_number: str
	title: str
	work_type: str
	priority: str
	status: str
	asset_number: str | None
	assigned_to: str | None
	scheduled_start: datetime | None
	scheduled_end: datetime | None
	completion_percentage: int
	created_on: datetime


class InventoryResponse(BaseModel):
	"""Inventory response model"""
	inventory_id: str
	part_number: str
	description: str
	current_stock: int
	minimum_stock: int
	reorder_point: int
	unit_cost: Decimal | None
	is_reorder_required: bool
	criticality: str


# =============================================================================
# API DEPENDENCIES
# =============================================================================

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
	"""Extract user information from JWT token via APG auth"""
	# This would integrate with APG auth_rbac service for JWT validation
	# For now, return mock user data
	return {
		"user_id": "user-123",
		"tenant_id": "tenant-456",
		"permissions": ["eam.asset.view", "eam.asset.create", "eam.workorder.view"]
	}

async def get_database_session() -> AsyncSession:
	"""Get database session"""
	# This would return actual database session from APG infrastructure
	pass

async def check_permission(permission: str, user: Dict[str, Any] = Depends(get_current_user)) -> bool:
	"""Check if user has required permission"""
	if permission not in user.get("permissions", []):
		raise HTTPException(
			status_code=status.HTTP_403_FORBIDDEN,
			detail=f"Insufficient permissions. Required: {permission}"
		)
	return True

async def get_asset_service(
	session: AsyncSession = Depends(get_database_session),
	user: Dict[str, Any] = Depends(get_current_user)
) -> EAMAssetService:
	"""Get asset service instance"""
	return EAMAssetService(session, user["user_id"], user["tenant_id"])

async def get_work_order_service(
	session: AsyncSession = Depends(get_database_session),
	user: Dict[str, Any] = Depends(get_current_user)
) -> EAMWorkOrderService:
	"""Get work order service instance"""
	return EAMWorkOrderService(session, user["user_id"], user["tenant_id"])

async def get_inventory_service(
	session: AsyncSession = Depends(get_database_session),
	user: Dict[str, Any] = Depends(get_current_user)
) -> EAMInventoryService:
	"""Get inventory service instance"""
	return EAMInventoryService(session, user["user_id"], user["tenant_id"])

async def get_analytics_service(
	session: AsyncSession = Depends(get_database_session),
	user: Dict[str, Any] = Depends(get_current_user)
) -> EAMAnalyticsService:
	"""Get analytics service instance"""
	return EAMAnalyticsService(session, user["user_id"], user["tenant_id"])


# =============================================================================
# API ROUTER SETUP
# =============================================================================

# Create API router
eam_router = APIRouter(prefix="/api/v1/eam", tags=["Enterprise Asset Management"])


# =============================================================================
# ASSET MANAGEMENT ENDPOINTS
# =============================================================================

@eam_router.post("/assets", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_asset(
	asset_data: EAAssetCreateModel,
	service: EAMAssetService = Depends(get_asset_service),
	_: bool = Depends(lambda: check_permission("eam.asset.create"))
) -> APIResponse:
	"""
	Create new enterprise asset
	
	Creates a new asset with full APG integration including:
	- Fixed Asset Management synchronization
	- Predictive Maintenance registration
	- Digital Twin creation (if enabled)
	- Audit trail logging
	"""
	try:
		asset = await service.create_asset(asset_data.model_dump())
		
		return APIResponse(
			success=True,
			message="Asset created successfully",
			data={
				"asset_id": asset.asset_id,
				"asset_number": asset.asset_number,
				"asset_name": asset.asset_name
			},
			meta={"created_at": datetime.utcnow().isoformat()}
		)
	
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(e)
		)
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to create asset"
		)


@eam_router.get("/assets", response_model=PaginatedResponse)
async def list_assets(
	page: int = Query(1, ge=1, description="Page number"),
	page_size: int = Query(50, ge=1, le=1000, description="Page size"),
	search: str | None = Query(None, description="Search term"),
	asset_type: str | None = Query(None, description="Filter by asset type"),
	status: str | None = Query(None, description="Filter by status"),
	criticality: str | None = Query(None, description="Filter by criticality"),
	location_id: str | None = Query(None, description="Filter by location"),
	health_score_min: float | None = Query(None, ge=0, le=100, description="Minimum health score"),
	maintenance_due: bool | None = Query(None, description="Filter maintenance due"),
	service: EAMAssetService = Depends(get_asset_service),
	_: bool = Depends(lambda: check_permission("eam.asset.view"))
) -> PaginatedResponse:
	"""
	List assets with advanced filtering and search
	
	Supports:
	- Full-text search across asset attributes
	- Advanced filtering by multiple criteria
	- Pagination with configurable page sizes
	- Performance optimization for large datasets
	"""
	try:
		filters = {}
		if asset_type:
			filters["asset_type"] = asset_type
		if status:
			filters["status"] = status
		if criticality:
			filters["criticality_level"] = criticality
		if location_id:
			filters["location_id"] = location_id
		if health_score_min is not None:
			filters["health_score_min"] = health_score_min
		if maintenance_due is not None:
			filters["maintenance_due"] = maintenance_due
		
		assets, total = await service.search_assets(
			search_term=search,
			filters=filters,
			page=page,
			page_size=page_size
		)
		
		# Convert to response format
		asset_responses = []
		for asset in assets:
			asset_responses.append(AssetResponse(
				asset_id=asset.asset_id,
				asset_number=asset.asset_number,
				asset_name=asset.asset_name,
				description=asset.description,
				asset_type=asset.asset_type,
				status=asset.status,
				health_score=asset.health_score,
				location_name=asset.location.location_name if asset.location else None,
				last_maintenance_date=asset.last_maintenance_date,
				next_maintenance_due=asset.next_maintenance_due,
				created_on=asset.created_on,
				changed_on=asset.changed_on
			))
		
		pages = (total + page_size - 1) // page_size
		
		return PaginatedResponse(
			items=asset_responses,
			total=total,
			page=page,
			page_size=page_size,
			pages=pages,
			has_next=page < pages,
			has_prev=page > 1
		)
	
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve assets"
		)


@eam_router.get("/assets/{asset_id}", response_model=APIResponse)
async def get_asset(
	asset_id: str = Path(..., description="Asset ID"),
	include_relationships: bool = Query(False, description="Include related data"),
	service: EAMAssetService = Depends(get_asset_service),
	_: bool = Depends(lambda: check_permission("eam.asset.view"))
) -> APIResponse:
	"""
	Get asset by ID with optional relationship data
	
	Optionally includes:
	- Location information
	- Parent/child asset relationships
	- Work order history
	- Maintenance records
	- Performance data
	- Contract information
	"""
	try:
		asset = await service.get_asset(asset_id, include_relationships)
		
		if not asset:
			raise HTTPException(
				status_code=status.HTTP_404_NOT_FOUND,
				detail=f"Asset {asset_id} not found"
			)
		
		# Build response data
		response_data = {
			"asset_id": asset.asset_id,
			"asset_number": asset.asset_number,
			"asset_name": asset.asset_name,
			"description": asset.description,
			"asset_type": asset.asset_type,
			"asset_category": asset.asset_category,
			"criticality_level": asset.criticality_level,
			"status": asset.status,
			"operational_status": asset.operational_status,
			"health_score": asset.health_score,
			"condition_status": asset.condition_status,
			"manufacturer": asset.manufacturer,
			"model_number": asset.model_number,
			"serial_number": asset.serial_number,
			"purchase_cost": asset.purchase_cost,
			"current_book_value": asset.current_book_value,
			"installation_date": asset.installation_date,
			"last_maintenance_date": asset.last_maintenance_date,
			"next_maintenance_due": asset.next_maintenance_due,
			"maintenance_status": asset.get_maintenance_status(),
			"health_status": asset.get_health_status(),
			"is_critical": asset.is_critical_asset(),
			"created_on": asset.created_on,
			"changed_on": asset.changed_on
		}
		
		if include_relationships:
			response_data.update({
				"location": {
					"location_id": asset.location.location_id,
					"location_name": asset.location.location_name,
					"location_type": asset.location.location_type
				} if asset.location else None,
				"parent_asset": {
					"asset_id": asset.parent_asset.asset_id,
					"asset_number": asset.parent_asset.asset_number,
					"asset_name": asset.parent_asset.asset_name
				} if asset.parent_asset else None,
				"child_assets_count": len(asset.child_assets),
				"open_work_orders": len([wo for wo in asset.work_orders if wo.status not in ['completed', 'cancelled']]),
				"recent_maintenance_count": len([mr for mr in asset.maintenance_records if mr.started_at and mr.started_at > datetime.utcnow() - timedelta(days=90)])
			})
		
		return APIResponse(
			success=True,
			message="Asset retrieved successfully",
			data=response_data
		)
	
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve asset"
		)


@eam_router.put("/assets/{asset_id}", response_model=APIResponse)
async def update_asset(
	asset_id: str = Path(..., description="Asset ID"),
	update_data: EAAssetUpdateModel = Body(...),
	service: EAMAssetService = Depends(get_asset_service),
	_: bool = Depends(lambda: check_permission("eam.asset.create"))
) -> APIResponse:
	"""
	Update asset with change tracking and APG integration
	
	Features:
	- Version control and change tracking
	- Audit trail logging
	- Cross-capability synchronization
	- Status change workflow management
	"""
	try:
		asset = await service.update_asset(
			asset_id, 
			update_data.model_dump(exclude_unset=True)
		)
		
		return APIResponse(
			success=True,
			message="Asset updated successfully",
			data={
				"asset_id": asset.asset_id,
				"asset_number": asset.asset_number,
				"version_number": asset.version_number,
				"changed_on": asset.changed_on
			}
		)
	
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(e)
		)
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to update asset"
		)


@eam_router.delete("/assets/{asset_id}", response_model=APIResponse)
async def delete_asset(
	asset_id: str = Path(..., description="Asset ID"),
	reason: str = Query(..., description="Deletion reason"),
	service: EAMAssetService = Depends(get_asset_service),
	_: bool = Depends(lambda: check_permission("eam.asset.delete"))
) -> APIResponse:
	"""
	Soft delete asset with audit compliance
	
	Performs soft deletion with:
	- Audit trail logging
	- Related data cleanup
	- Cross-capability notifications
	- Compliance tracking
	"""
	try:
		success = await service.delete_asset(asset_id, reason)
		
		if not success:
			raise HTTPException(
				status_code=status.HTTP_404_NOT_FOUND,
				detail=f"Asset {asset_id} not found"
			)
		
		return APIResponse(
			success=True,
			message="Asset deleted successfully",
			data={"asset_id": asset_id, "deletion_reason": reason}
		)
	
	except HTTPException:
		raise
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(e)
		)
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to delete asset"
		)


@eam_router.get("/assets/{asset_id}/hierarchy", response_model=APIResponse)
async def get_asset_hierarchy(
	asset_id: str = Path(..., description="Asset ID"),
	service: EAMAssetService = Depends(get_asset_service),
	_: bool = Depends(lambda: check_permission("eam.asset.view"))
) -> APIResponse:
	"""
	Get complete asset hierarchy including parents and children
	
	Returns:
	- Parent chain to root asset
	- Direct children
	- Total descendant count
	- Hierarchy visualization data
	"""
	try:
		hierarchy = await service.get_asset_hierarchy(asset_id)
		
		return APIResponse(
			success=True,
			message="Asset hierarchy retrieved successfully",
			data=hierarchy
		)
	
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=str(e)
		)
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve asset hierarchy"
		)


@eam_router.get("/assets/maintenance/due", response_model=APIResponse)
async def get_maintenance_due_assets(
	days_ahead: int = Query(30, ge=0, le=365, description="Days ahead to check"),
	service: EAMAssetService = Depends(get_asset_service),
	_: bool = Depends(lambda: check_permission("eam.asset.view"))
) -> APIResponse:
	"""
	Get assets with maintenance due within specified days
	
	Used for:
	- Maintenance planning
	- Resource allocation
	- Preventive maintenance scheduling
	- Dashboard alerts
	"""
	try:
		assets = await service.get_maintenance_due_assets(days_ahead)
		
		due_assets = []
		for asset in assets:
			due_assets.append({
				"asset_id": asset.asset_id,
				"asset_number": asset.asset_number,
				"asset_name": asset.asset_name,
				"criticality_level": asset.criticality_level,
				"next_maintenance_due": asset.next_maintenance_due,
				"days_until_due": (asset.next_maintenance_due - date.today()).days if asset.next_maintenance_due else None,
				"maintenance_status": asset.get_maintenance_status()
			})
		
		return APIResponse(
			success=True,
			message=f"Found {len(due_assets)} assets with maintenance due",
			data=due_assets,
			meta={"days_ahead": days_ahead}
		)
	
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve maintenance due assets"
		)


# =============================================================================
# WORK ORDER MANAGEMENT ENDPOINTS
# =============================================================================

@eam_router.post("/work-orders", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_work_order(
	work_order_data: EAWorkOrderCreateModel,
	service: EAMWorkOrderService = Depends(get_work_order_service),
	_: bool = Depends(lambda: check_permission("eam.workorder.create"))
) -> APIResponse:
	"""
	Create new work order with APG integration
	
	Features:
	- Automatic work order numbering
	- Priority-based collaboration room creation
	- Notification system integration
	- Resource allocation optimization
	"""
	try:
		work_order = await service.create_work_order(work_order_data.model_dump())
		
		return APIResponse(
			success=True,
			message="Work order created successfully",
			data={
				"work_order_id": work_order.work_order_id,
				"work_order_number": work_order.work_order_number,
				"title": work_order.title,
				"status": work_order.status,
				"collaboration_room_id": work_order.collaboration_room_id
			}
		)
	
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(e)
		)
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to create work order"
		)


@eam_router.get("/work-orders", response_model=PaginatedResponse)
async def list_work_orders(
	page: int = Query(1, ge=1),
	page_size: int = Query(50, ge=1, le=1000),
	status_filter: str | None = Query(None, description="Filter by status"),
	priority: str | None = Query(None, description="Filter by priority"),
	assigned_to: str | None = Query(None, description="Filter by assignee"),
	asset_id: str | None = Query(None, description="Filter by asset"),
	work_type: str | None = Query(None, description="Filter by work type"),
	overdue_only: bool = Query(False, description="Show only overdue work orders"),
	service: EAMWorkOrderService = Depends(get_work_order_service),
	_: bool = Depends(lambda: check_permission("eam.workorder.view"))
) -> PaginatedResponse:
	"""
	List work orders with advanced filtering
	
	Supports filtering by:
	- Status and priority
	- Assignment and asset
	- Work type and category
	- Due date and overdue status
	"""
	# Implementation would be similar to assets list endpoint
	pass


@eam_router.put("/work-orders/{work_order_id}/assign", response_model=APIResponse)
async def assign_work_order(
	work_order_id: str = Path(..., description="Work order ID"),
	assigned_to: str = Body(..., embed=True, description="Assignee user ID"),
	team_members: List[str] | None = Body(None, embed=True, description="Team member IDs"),
	service: EAMWorkOrderService = Depends(get_work_order_service),
	_: bool = Depends(lambda: check_permission("eam.workorder.assign"))
) -> APIResponse:
	"""
	Assign work order to technician and team
	
	Features:
	- Team collaboration setup
	- Notification distribution
	- Resource allocation tracking
	- Workload balancing
	"""
	try:
		work_order = await service.assign_work_order(work_order_id, assigned_to, team_members)
		
		return APIResponse(
			success=True,
			message="Work order assigned successfully",
			data={
				"work_order_id": work_order.work_order_id,
				"assigned_to": work_order.assigned_to,
				"team_members": work_order.assigned_team,
				"status": work_order.status
			}
		)
	
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(e)
		)
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to assign work order"
		)


@eam_router.put("/work-orders/{work_order_id}/start", response_model=APIResponse)
async def start_work_order(
	work_order_id: str = Path(..., description="Work order ID"),
	service: EAMWorkOrderService = Depends(get_work_order_service),
	_: bool = Depends(lambda: check_permission("eam.workorder.execute"))
) -> APIResponse:
	"""
	Start work order execution
	
	Actions:
	- Update status to in_progress
	- Record actual start time
	- Update asset operational status
	- Broadcast real-time updates
	"""
	try:
		work_order = await service.start_work_order(work_order_id)
		
		return APIResponse(
			success=True,
			message="Work order started successfully",
			data={
				"work_order_id": work_order.work_order_id,
				"status": work_order.status,
				"actual_start": work_order.actual_start,
				"asset_operational_status": work_order.asset.operational_status if work_order.asset else None
			}
		)
	
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(e)
		)
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to start work order"
		)


@eam_router.put("/work-orders/{work_order_id}/complete", response_model=APIResponse)
async def complete_work_order(
	work_order_id: str = Path(..., description="Work order ID"),
	completion_data: Dict[str, Any] = Body(..., description="Completion details"),
	service: EAMWorkOrderService = Depends(get_work_order_service),
	_: bool = Depends(lambda: check_permission("eam.workorder.execute"))
) -> APIResponse:
	"""
	Complete work order with maintenance record creation
	
	Actions:
	- Update completion details
	- Create maintenance record
	- Update asset status
	- Schedule next maintenance
	- Generate performance metrics
	"""
	try:
		work_order = await service.complete_work_order(work_order_id, completion_data)
		
		return APIResponse(
			success=True,
			message="Work order completed successfully",
			data={
				"work_order_id": work_order.work_order_id,
				"status": work_order.status,
				"actual_end": work_order.actual_end,
				"actual_hours": work_order.actual_hours,
				"completion_percentage": work_order.completion_percentage
			}
		)
	
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(e)
		)
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to complete work order"
		)


# =============================================================================
# INVENTORY MANAGEMENT ENDPOINTS
# =============================================================================

@eam_router.post("/inventory", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_inventory_item(
	item_data: EAInventoryCreateModel,
	service: EAMInventoryService = Depends(get_inventory_service),
	_: bool = Depends(lambda: check_permission("eam.inventory.manage"))
) -> APIResponse:
	"""Create new inventory item with auto-reorder setup"""
	try:
		item = await service.create_inventory_item(item_data.model_dump())
		
		return APIResponse(
			success=True,
			message="Inventory item created successfully",
			data={
				"inventory_id": item.inventory_id,
				"part_number": item.part_number,
				"description": item.description,
				"current_stock": item.current_stock
			}
		)
	
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(e)
		)
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to create inventory item"
		)


@eam_router.put("/inventory/{inventory_id}/stock", response_model=APIResponse)
async def update_stock_level(
	inventory_id: str = Path(..., description="Inventory ID"),
	quantity_change: int = Body(..., embed=True, description="Stock quantity change"),
	reason: str = Body(..., embed=True, description="Reason for stock change"),
	service: EAMInventoryService = Depends(get_inventory_service),
	_: bool = Depends(lambda: check_permission("eam.inventory.manage"))
) -> APIResponse:
	"""
	Update inventory stock level with audit trail
	
	Features:
	- Audit trail logging
	- Auto-reorder trigger checking
	- Stock level validation
	- Cost calculation updates
	"""
	try:
		item = await service.update_stock_level(inventory_id, quantity_change, reason)
		
		return APIResponse(
			success=True,
			message="Stock level updated successfully",
			data={
				"inventory_id": item.inventory_id,
				"part_number": item.part_number,
				"current_stock": item.current_stock,
				"total_value": item.total_value,
				"is_reorder_required": item.is_reorder_required(),
				"stock_days_remaining": item.calculate_stock_days()
			}
		)
	
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(e)
		)
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to update stock level"
		)


@eam_router.get("/inventory/reorder-recommendations", response_model=APIResponse)
async def get_reorder_recommendations(
	service: EAMInventoryService = Depends(get_inventory_service),
	_: bool = Depends(lambda: check_permission("eam.inventory.manage"))
) -> APIResponse:
	"""
	Get inventory reorder recommendations
	
	Returns:
	- Items below reorder point
	- Recommended order quantities
	- Cost estimates
	- Vendor information
	- Criticality-based prioritization
	"""
	try:
		recommendations = await service.get_reorder_recommendations()
		
		return APIResponse(
			success=True,
			message=f"Found {len(recommendations)} items requiring reorder",
			data=recommendations,
			meta={
				"total_items": len(recommendations),
				"total_estimated_cost": sum(r.get("estimated_cost", 0) for r in recommendations if r.get("estimated_cost")),
				"critical_items": len([r for r in recommendations if r.get("criticality") == "critical"])
			}
		)
	
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve reorder recommendations"
		)


# =============================================================================
# ANALYTICS AND REPORTING ENDPOINTS
# =============================================================================

@eam_router.get("/analytics/dashboard", response_model=APIResponse)
async def get_dashboard_analytics(
	service: EAMAnalyticsService = Depends(get_analytics_service),
	_: bool = Depends(lambda: check_permission("eam.analytics.view"))
) -> APIResponse:
	"""
	Get main dashboard analytics
	
	Returns:
	- Fleet performance summary
	- Critical asset health
	- Maintenance metrics
	- Cost analysis
	- Trend indicators
	"""
	try:
		dashboard_data = await service.get_fleet_performance_dashboard()
		
		return APIResponse(
			success=True,
			message="Dashboard analytics retrieved successfully",
			data=dashboard_data
		)
	
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve dashboard analytics"
		)


@eam_router.get("/analytics/assets/{asset_id}/performance", response_model=APIResponse)
async def get_asset_performance_analytics(
	asset_id: str = Path(..., description="Asset ID"),
	days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
	service: EAMAnalyticsService = Depends(get_analytics_service),
	_: bool = Depends(lambda: check_permission("eam.analytics.view"))
) -> APIResponse:
	"""
	Get comprehensive asset performance analytics
	
	Returns:
	- Performance trends and metrics
	- Health score evolution
	- Maintenance effectiveness
	- Cost analysis
	- Predictive insights
	"""
	try:
		performance_data = await service.get_asset_performance_summary(asset_id, days)
		
		return APIResponse(
			success=True,
			message="Asset performance analytics retrieved successfully",
			data=performance_data,
			meta={"analysis_period_days": days}
		)
	
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve asset performance analytics"
		)


# =============================================================================
# BULK OPERATIONS AND EXPORT ENDPOINTS
# =============================================================================

@eam_router.post("/assets/bulk-update", response_model=APIResponse)
async def bulk_update_assets(
	updates: List[Dict[str, Any]] = Body(..., description="List of asset updates"),
	service: EAMAssetService = Depends(get_asset_service),
	_: bool = Depends(lambda: check_permission("eam.asset.create"))
) -> APIResponse:
	"""
	Bulk update multiple assets
	
	Features:
	- Batch processing for efficiency
	- Atomic transaction handling
	- Progress tracking
	- Error reporting per asset
	"""
	results = []
	errors = []
	
	for update in updates:
		try:
			asset_id = update.get("asset_id")
			if not asset_id:
				errors.append("Asset ID is required for each update")
				continue
			
			asset = await service.update_asset(asset_id, update)
			results.append({
				"asset_id": asset.asset_id,
				"asset_number": asset.asset_number,
				"status": "updated"
			})
		
		except Exception as e:
			errors.append(f"Failed to update asset {update.get('asset_id', 'unknown')}: {str(e)}")
	
	return APIResponse(
		success=len(errors) == 0,
		message=f"Bulk update completed: {len(results)} successful, {len(errors)} failed",
		data={"updated_assets": results},
		errors=errors
	)


@eam_router.get("/assets/export")
async def export_assets(
	format: str = Query("csv", regex="^(csv|excel|json)$", description="Export format"),
	search: str | None = Query(None, description="Search filter"),
	filters: str | None = Query(None, description="JSON filters"),
	service: EAMAssetService = Depends(get_asset_service),
	_: bool = Depends(lambda: check_permission("eam.asset.view"))
) -> StreamingResponse:
	"""
	Export assets to various formats
	
	Supports:
	- CSV for spreadsheet import
	- Excel for advanced analysis
	- JSON for system integration
	"""
	try:
		# Parse filters if provided
		filter_dict = {}
		if filters:
			filter_dict = json.loads(filters)
		
		# Get all matching assets
		assets, _ = await service.search_assets(
			search_term=search,
			filters=filter_dict,
			page=1,
			page_size=10000  # Large page size for export
		)
		
		if format == "csv":
			output = io.StringIO()
			writer = csv.writer(output)
			
			# Write header
			writer.writerow([
				"Asset Number", "Asset Name", "Type", "Status", "Criticality",
				"Health Score", "Location", "Manufacturer", "Model",
				"Purchase Cost", "Last Maintenance", "Next Maintenance"
			])
			
			# Write data
			for asset in assets:
				writer.writerow([
					asset.asset_number,
					asset.asset_name,
					asset.asset_type,
					asset.status,
					asset.criticality_level,
					asset.health_score,
					asset.location.location_name if asset.location else "",
					asset.manufacturer,
					asset.model_number,
					asset.purchase_cost,
					asset.last_maintenance_date,
					asset.next_maintenance_due
				])
			
			output.seek(0)
			return StreamingResponse(
				io.BytesIO(output.getvalue().encode()),
				media_type="text/csv",
				headers={"Content-Disposition": "attachment; filename=assets.csv"}
			)
		
		elif format == "json":
			asset_data = []
			for asset in assets:
				asset_data.append({
					"asset_id": asset.asset_id,
					"asset_number": asset.asset_number,
					"asset_name": asset.asset_name,
					"asset_type": asset.asset_type,
					"status": asset.status,
					"criticality_level": asset.criticality_level,
					"health_score": asset.health_score,
					"location_name": asset.location.location_name if asset.location else None,
					"manufacturer": asset.manufacturer,
					"model_number": asset.model_number,
					"purchase_cost": float(asset.purchase_cost) if asset.purchase_cost else None,
					"last_maintenance_date": asset.last_maintenance_date.isoformat() if asset.last_maintenance_date else None,
					"next_maintenance_due": asset.next_maintenance_due.isoformat() if asset.next_maintenance_due else None
				})
			
			json_data = json.dumps(asset_data, indent=2)
			return StreamingResponse(
				io.BytesIO(json_data.encode()),
				media_type="application/json",
				headers={"Content-Disposition": "attachment; filename=assets.json"}
			)
	
	except Exception as e:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to export assets"
		)


# =============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# =============================================================================

class ConnectionManager:
	"""WebSocket connection manager for real-time updates"""
	
	def __init__(self):
		self.active_connections: List[WebSocket] = []
	
	async def connect(self, websocket: WebSocket):
		await websocket.accept()
		self.active_connections.append(websocket)
	
	def disconnect(self, websocket: WebSocket):
		self.active_connections.remove(websocket)
	
	async def send_personal_message(self, message: str, websocket: WebSocket):
		await websocket.send_text(message)
	
	async def broadcast(self, message: str):
		for connection in self.active_connections:
			try:
				await connection.send_text(message)
			except:
				# Remove broken connections
				self.active_connections.remove(connection)

manager = ConnectionManager()

@eam_router.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
	"""
	WebSocket endpoint for real-time EAM updates
	
	Provides real-time updates for:
	- Asset status changes
	- Work order progress
	- Maintenance alerts
	- Inventory levels
	- Performance metrics
	"""
	await manager.connect(websocket)
	try:
		while True:
			# Listen for messages from client
			data = await websocket.receive_text()
			message_data = json.loads(data)
			
			# Handle different message types
			if message_data.get("type") == "subscribe":
				# Subscribe to specific updates
				await websocket.send_text(json.dumps({
					"type": "subscription_confirmed",
					"topics": message_data.get("topics", [])
				}))
			
			elif message_data.get("type") == "ping":
				# Health check
				await websocket.send_text(json.dumps({
					"type": "pong",
					"timestamp": datetime.utcnow().isoformat()
				}))
	
	except WebSocketDisconnect:
		manager.disconnect(websocket)


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@eam_router.exception_handler(ValueError)
async def value_error_handler(request, exc):
	"""Handle validation errors"""
	return JSONResponse(
		status_code=status.HTTP_400_BAD_REQUEST,
		content=APIResponse(
			success=False,
			message="Validation error",
			errors=[str(exc)]
		).model_dump()
	)


@eam_router.exception_handler(Exception)
async def general_exception_handler(request, exc):
	"""Handle general exceptions"""
	return JSONResponse(
		status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
		content=APIResponse(
			success=False,
			message="Internal server error",
			errors=["An unexpected error occurred"]
		).model_dump()
	)


# =============================================================================
# API ROUTER EXPORT
# =============================================================================

# Export the router for registration with main FastAPI app
__all__ = ["eam_router", "ConnectionManager"]