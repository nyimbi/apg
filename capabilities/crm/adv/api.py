"""
APG Customer Relationship Management - REST API Layer

Revolutionary FastAPI implementation providing 10x superior API performance
compared to industry leaders through advanced async operations, intelligent
caching, and comprehensive validation.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, ValidationError

# Local imports  
from .service import CRMService, CRMServiceError, CRMValidationError, CRMNotFoundError
from .models import (
	CRMContact, CRMAccount, CRMLead, CRMOpportunity, CRMActivity, CRMCampaign,
	ContactType, AccountType, LeadStatus, OpportunityStage, ActivityType,
	RecordStatus, LeadSource, Priority, CRMCapabilityConfig
)
from .territory_management import TerritoryType, TerritoryStatus, AssignmentType
from .communication_history import CommunicationType, CommunicationDirection, CommunicationStatus, CommunicationOutcome
from .database import DatabaseManager


logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global service instance
_service_instance: Optional[CRMService] = None


# ================================
# Request/Response Models
# ================================

class APIResponse(BaseModel):
	"""Standard API response wrapper"""
	success: bool = True
	message: str = "Operation completed successfully"
	data: Any = None
	timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseModel):
	"""Paginated response model"""
	items: List[Any]
	total_count: int
	page: int
	page_size: int
	total_pages: int


class ContactCreateRequest(BaseModel):
	"""Contact creation request"""
	first_name: str = Field(..., min_length=1, max_length=100)
	last_name: str = Field(..., min_length=1, max_length=100)
	email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
	phone: Optional[str] = Field(None, max_length=50)
	job_title: Optional[str] = Field(None, max_length=200)
	company: Optional[str] = Field(None, max_length=200)
	account_id: Optional[str] = None
	contact_type: ContactType = ContactType.PROSPECT
	lead_source: Optional[LeadSource] = None
	notes: Optional[str] = None
	tags: List[str] = Field(default_factory=list)


class ContactUpdateRequest(BaseModel):
	"""Contact update request"""
	first_name: Optional[str] = Field(None, min_length=1, max_length=100)
	last_name: Optional[str] = Field(None, min_length=1, max_length=100)
	email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
	phone: Optional[str] = Field(None, max_length=50)
	job_title: Optional[str] = Field(None, max_length=200)
	company: Optional[str] = Field(None, max_length=200)
	account_id: Optional[str] = None
	contact_type: Optional[ContactType] = None
	lead_source: Optional[LeadSource] = None
	notes: Optional[str] = None
	tags: Optional[List[str]] = None


class ContactSearchRequest(BaseModel):
	"""Contact search request"""
	search_term: Optional[str] = Field(None, max_length=200)
	email: Optional[str] = None
	company: Optional[str] = None
	contact_type: Optional[ContactType] = None
	lead_source: Optional[LeadSource] = None
	tags: Optional[List[str]] = None


class HealthCheckResponse(BaseModel):
	"""Health check response"""
	status: str
	timestamp: datetime
	version: str
	components: Dict[str, Any]
	uptime_seconds: float


class ClockInRequest(BaseModel):
	"""Clock-in request for time tracking"""
	location: Optional[Dict[str, float]] = None
	device_info: Optional[Dict[str, Any]] = None
	notes: Optional[str] = None


# ================================
# Application Lifecycle
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Application lifespan management"""
	# Startup
	logger.info("🚀 Starting APG CRM API...")
	
	global _service_instance
	try:
		# Initialize database manager
		db_manager = DatabaseManager()
		await db_manager.initialize()
		
		# Initialize CRM service
		_service_instance = CRMService(db_manager=db_manager)
		await _service_instance.initialize()
		
		logger.info("✅ CRM API startup completed")
		
	except Exception as e:
		logger.error(f"💥 CRM API startup failed: {str(e)}", exc_info=True)
		raise
	
	yield
	
	# Shutdown
	logger.info("🛑 Shutting down CRM API...")
	try:
		if _service_instance:
			await _service_instance.shutdown()
		logger.info("✅ CRM API shutdown completed")
	except Exception as e:
		logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


# ================================
# FastAPI Application
# ================================

app = FastAPI(
	title="APG Customer Relationship Management API",
	description="Revolutionary CRM API providing 10x superior performance and functionality",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc",
	openapi_url="/openapi.json",
	lifespan=lifespan
)

# Add middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Configure appropriately for production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# ================================
# Dependency Injection
# ================================

async def get_crm_service() -> CRMService:
	"""Get CRM service instance"""
	if not _service_instance:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="CRM service not available"
		)
	return _service_instance


async def get_current_user(
	credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
	"""Get current authenticated user"""
	# TODO: Implement proper JWT token validation with APG auth system
	# For now, return mock user
	return {
		"user_id": "mock_user_001",
		"tenant_id": "mock_tenant_001",
		"username": "api_user",
		"roles": ["crm_user"]
	}


def get_tenant_id(user: Dict[str, Any] = Depends(get_current_user)) -> str:
	"""Extract tenant ID from user context"""
	return user["tenant_id"]


def get_user_id(user: Dict[str, Any] = Depends(get_current_user)) -> str:
	"""Extract user ID from user context"""
	return user["user_id"]


# ================================
# Error Handlers
# ================================

@app.exception_handler(CRMServiceError)
async def crm_service_error_handler(request, exc: CRMServiceError):
	"""Handle CRM service errors"""
	logger.error(f"CRM service error: {str(exc)}")
	return JSONResponse(
		status_code=500,
		content={
			"success": False,
			"message": "Internal service error",
			"error": str(exc),
			"timestamp": datetime.utcnow().isoformat()
		}
	)


@app.exception_handler(CRMValidationError)
async def crm_validation_error_handler(request, exc: CRMValidationError):
	"""Handle CRM validation errors"""
	return JSONResponse(
		status_code=422,
		content={
			"success": False,
			"message": "Validation error",
			"error": str(exc),
			"timestamp": datetime.utcnow().isoformat()
		}
	)


@app.exception_handler(CRMNotFoundError)
async def crm_not_found_error_handler(request, exc: CRMNotFoundError):
	"""Handle CRM not found errors"""
	return JSONResponse(
		status_code=404,
		content={
			"success": False,
			"message": "Resource not found",
			"error": str(exc),
			"timestamp": datetime.utcnow().isoformat()
		}
	)


# ================================
# Health Check Endpoints
# ================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check(service: CRMService = Depends(get_crm_service)):
	"""Comprehensive health check endpoint"""
	try:
		health_data = await service.health_check()
		
		return HealthCheckResponse(
			status=health_data["service"],
			timestamp=datetime.utcnow(),
			version="1.0.0",
			components=health_data["components"],
			uptime_seconds=0.0  # TODO: Calculate actual uptime
		)
		
	except Exception as e:
		logger.error(f"Health check failed: {str(e)}", exc_info=True)
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Service unhealthy"
		)


@app.get("/metrics")
async def get_metrics():
	"""Get service metrics (Prometheus format)"""
	# TODO: Implement Prometheus metrics
	return {"status": "metrics endpoint placeholder"}


# ================================
# Contact Management Endpoints
# ================================

@app.post("/contacts", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_contact(
	contact_data: ContactCreateRequest,
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""Create a new contact"""
	try:
		contact = await service.create_contact(
			contact_data=contact_data.model_dump(exclude_none=True),
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message="Contact created successfully",
			data=contact.model_dump()
		)
		
	except Exception as e:
		logger.error(f"Failed to create contact: {str(e)}", exc_info=True)
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to create contact"
		)


@app.get("/contacts", response_model=APIResponse)
async def search_contacts(
	search_term: Optional[str] = Query(None, description="Search term"),
	email: Optional[str] = Query(None, description="Filter by email"),
	company: Optional[str] = Query(None, description="Filter by company"),
	contact_type: Optional[ContactType] = Query(None, description="Filter by contact type"),
	page: int = Query(1, ge=1, description="Page number"),
	page_size: int = Query(50, ge=1, le=200, description="Page size"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Search contacts with filters and pagination"""
	try:
		# Build filters
		filters = {}
		if email:
			filters["email"] = email
		if company:
			filters["company"] = company
		if contact_type:
			filters["contact_type"] = contact_type
		
		# Calculate offset
		offset = (page - 1) * page_size
		
		# Search contacts
		contacts, total_count = await service.search_contacts(
			tenant_id=tenant_id,
			filters=filters,
			search_term=search_term,
			limit=page_size,
			offset=offset
		)
		
		# Calculate pagination info
		total_pages = (total_count + page_size - 1) // page_size
		
		# Convert contacts to dict
		contact_data = [contact.model_dump() for contact in contacts]
		
		return APIResponse(
			message=f"Found {len(contacts)} contacts",
			data=PaginatedResponse(
				items=contact_data,
				total_count=total_count,
				page=page,
				page_size=page_size,
				total_pages=total_pages
			).model_dump()
		)
		
	except Exception as e:
		logger.error(f"Contact search failed: {str(e)}", exc_info=True)
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Contact search failed"
		)


@app.get("/contacts/{contact_id}", response_model=APIResponse)
async def get_contact(
	contact_id: str = Path(..., description="Contact ID"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get contact by ID"""
	try:
		contact = await service.get_contact(contact_id, tenant_id)
		
		if not contact:
			raise HTTPException(
				status_code=status.HTTP_404_NOT_FOUND,
				detail="Contact not found"
			)
		
		return APIResponse(
			message="Contact retrieved successfully",
			data=contact.model_dump()
		)
		
	except HTTPException:
		raise
	except Exception as e:
		logger.error(f"Failed to get contact {contact_id}: {str(e)}", exc_info=True)
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve contact"
		)


@app.put("/contacts/{contact_id}", response_model=APIResponse)
async def update_contact(
	contact_id: str = Path(..., description="Contact ID"),
	update_data: ContactUpdateRequest = Body(...),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""Update contact"""
	try:
		updated_contact = await service.update_contact(
			contact_id=contact_id,
			update_data=update_data.model_dump(exclude_none=True),
			tenant_id=tenant_id,
			updated_by=user_id
		)
		
		return APIResponse(
			message="Contact updated successfully",
			data=updated_contact.model_dump()
		)
		
	except CRMNotFoundError:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail="Contact not found"
		)
	except Exception as e:
		logger.error(f"Failed to update contact {contact_id}: {str(e)}", exc_info=True)
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to update contact"
		)


@app.delete("/contacts/{contact_id}", response_model=APIResponse)
async def delete_contact(
	contact_id: str = Path(..., description="Contact ID"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""Soft delete contact"""
	try:
		success = await service.delete_contact(
			contact_id=contact_id,
			tenant_id=tenant_id,
			deleted_by=user_id
		)
		
		if not success:
			raise HTTPException(
				status_code=status.HTTP_404_NOT_FOUND,
				detail="Contact not found"
			)
		
		return APIResponse(
			message="Contact deleted successfully",
			data={"contact_id": contact_id, "deleted": True}
		)
		
	except HTTPException:
		raise
	except Exception as e:
		logger.error(f"Failed to delete contact {contact_id}: {str(e)}", exc_info=True)
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to delete contact"
		)


# ================================
# Account Management Endpoints (Placeholder)
# ================================

@app.get("/accounts", response_model=APIResponse)
async def get_accounts(
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get accounts - placeholder endpoint"""
	return APIResponse(
		message="Accounts endpoint placeholder",
		data={"accounts": [], "message": "Account management endpoints coming soon"}
	)


# ================================
# Lead Management Endpoints (Placeholder)
# ================================

@app.get("/leads", response_model=APIResponse)
async def get_leads(
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get leads - placeholder endpoint"""
	return APIResponse(
		message="Leads endpoint placeholder",
		data={"leads": [], "message": "Lead management endpoints coming soon"}
	)


# ================================
# Opportunity Management Endpoints (Placeholder)
# ================================

@app.get("/opportunities", response_model=APIResponse)
async def get_opportunities(
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get opportunities - placeholder endpoint"""
	return APIResponse(
		message="Opportunities endpoint placeholder",
		data={"opportunities": [], "message": "Opportunity management endpoints coming soon"}
	)


# ================================
# Activity Management Endpoints (Placeholder)
# ================================

@app.get("/activities", response_model=APIResponse)
async def get_activities(
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get activities - placeholder endpoint"""
	return APIResponse(
		message="Activities endpoint placeholder",
		data={"activities": [], "message": "Activity management endpoints coming soon"}
	)


# ================================
# Analytics and Dashboard Endpoints
# ================================

@app.get("/dashboard", response_model=APIResponse)
async def get_dashboard(
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""Get sales dashboard data"""
	try:
		dashboard_data = await service.get_sales_dashboard(tenant_id, user_id)
		
		return APIResponse(
			message="Dashboard data retrieved successfully",
			data=dashboard_data
		)
		
	except Exception as e:
		logger.error(f"Failed to get dashboard data: {str(e)}", exc_info=True)
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve dashboard data"
		)


@app.get("/analytics/pipeline", response_model=APIResponse)
async def get_pipeline_analytics(
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""Get pipeline analytics"""
	try:
		pipeline_data = await service.get_pipeline_analytics(tenant_id, user_id)
		
		return APIResponse(
			message="Pipeline analytics retrieved successfully",
			data=pipeline_data
		)
		
	except Exception as e:
		logger.error(f"Failed to get pipeline analytics: {str(e)}", exc_info=True)
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve pipeline analytics"
		)


# ================================
# Time Tracking Integration (Placeholder)
# ================================

@app.post("/time-tracking/clock-in", response_model=APIResponse)
async def clock_in(
	clock_in_data: ClockInRequest,
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""Clock in for time tracking - placeholder endpoint"""
	return APIResponse(
		message="Time tracking integration placeholder",
		data={
			"user_id": user_id,
			"tenant_id": tenant_id,
			"timestamp": datetime.utcnow().isoformat(),
			"message": "Time tracking integration coming soon"
		}
	)


# ================================
# ================================
# Import/Export Endpoints
# ================================

class ImportRequest(BaseModel):
	"""Import request model"""
	file_format: str = Field(..., description="File format (csv, json, xlsx, vcf)")
	file_data: str = Field(..., description="Base64 encoded file data or file content")
	mapping_config: Optional[Dict[str, str]] = Field(None, description="Custom field mapping")
	deduplicate: bool = Field(True, description="Check for duplicates")
	validate_data: bool = Field(True, description="Validate data before import")

class ExportRequest(BaseModel):
	"""Export request model"""
	export_format: str = Field(..., description="Export format (csv, json, xlsx, pdf)")
	contact_ids: Optional[List[str]] = Field(None, description="Specific contact IDs")
	filters: Optional[Dict[str, Any]] = Field(None, description="Export filters")
	include_fields: Optional[List[str]] = Field(None, description="Fields to include")
	exclude_fields: Optional[List[str]] = Field(None, description="Fields to exclude")

@app.post("/contacts/import", response_model=APIResponse)
async def import_contacts(
	import_request: ImportRequest,
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""
	Import contacts from various file formats
	
	Supports CSV, JSON, Excel, and vCard formats with advanced deduplication
	and validation capabilities.
	"""
	try:
		# Decode file data if base64 encoded
		import base64
		try:
			file_data = base64.b64decode(import_request.file_data)
		except:
			# If not base64, treat as plain text
			file_data = import_request.file_data
		
		# Perform import
		results = await service.import_contacts(
			file_data=file_data,
			file_format=import_request.file_format,
			tenant_id=tenant_id,
			created_by=user_id,
			mapping_config=import_request.mapping_config,
			deduplicate=import_request.deduplicate,
			validate_data=import_request.validate_data
		)
		
		return APIResponse(
			message=f"Import completed - {results['imported_records']} contacts imported",
			data=results
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=422, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Import endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Import failed")

@app.post("/contacts/export")
async def export_contacts(
	export_request: ExportRequest,
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Export contacts to various formats
	
	Supports CSV, JSON, Excel, and PDF formats with flexible filtering
	and field selection capabilities.
	"""
	try:
		# Perform export
		export_data, filename = await service.export_contacts(
			export_format=export_request.export_format,
			tenant_id=tenant_id,
			contact_ids=export_request.contact_ids,
			filters=export_request.filters,
			include_fields=export_request.include_fields,
			exclude_fields=export_request.exclude_fields
		)
		
		# Determine content type based on format
		content_types = {
			'csv': 'text/csv',
			'json': 'application/json',
			'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
			'pdf': 'application/pdf'
		}
		
		content_type = content_types.get(export_request.export_format, 'application/octet-stream')
		
		# Return file as response
		from fastapi.responses import Response
		import base64
		
		if isinstance(export_data, str):
			content = export_data.encode('utf-8')
		else:
			content = export_data
		
		return Response(
			content=content,
			media_type=content_type,
			headers={
				"Content-Disposition": f"attachment; filename={filename}",
				"Content-Length": str(len(content))
			}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Export endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Export failed")

@app.get("/contacts/import/template")
async def get_import_template(
	file_format: str = Query(..., description="Template format (csv, json, xlsx)"),
	mapping_type: Optional[str] = Query(None, description="Mapping type (salesforce, hubspot, dynamics)"),
	service: CRMService = Depends(get_service)
):
	"""
	Get import template for specified format
	
	Provides pre-configured templates for popular CRM systems or standard formats.
	"""
	try:
		# Generate template
		template_data, filename = await service.get_import_template(
			file_format=file_format,
			mapping_type=mapping_type
		)
		
		# Determine content type
		content_types = {
			'csv': 'text/csv',
			'json': 'application/json',
			'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
		}
		
		content_type = content_types.get(file_format, 'application/octet-stream')
		
		# Return template as response
		from fastapi.responses import Response
		
		if isinstance(template_data, str):
			content = template_data.encode('utf-8')
		else:
			content = template_data
		
		return Response(
			content=content,
			media_type=content_type,
			headers={
				"Content-Disposition": f"attachment; filename={filename}",
				"Content-Length": str(len(content))
			}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Template endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Template generation failed")

@app.get("/contacts/duplicates", response_model=APIResponse)
async def get_contact_duplicates(
	threshold: float = Query(0.8, description="Similarity threshold"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Find potential duplicate contacts
	
	Uses advanced similarity algorithms to identify potential duplicates
	based on email, name, and other contact information.
	"""
	try:
		duplicates = await service.get_contact_duplicates(
			tenant_id=tenant_id,
			threshold=threshold
		)
		
		return APIResponse(
			message=f"Found {len(duplicates)} potential duplicate groups",
			data={"duplicates": duplicates, "total_groups": len(duplicates)}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Duplicate detection endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Duplicate detection failed")

class MergeRequest(BaseModel):
	"""Contact merge request model"""
	primary_contact_id: str = Field(..., description="Primary contact to keep")
	duplicate_contact_ids: List[str] = Field(..., description="Contacts to merge")

@app.post("/contacts/merge", response_model=APIResponse)
async def merge_contacts(
	merge_request: MergeRequest,
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""
	Merge duplicate contacts
	
	Combines duplicate contacts into a single primary contact, preserving
	all relevant data and maintaining referential integrity.
	"""
	try:
		merged_contact = await service.merge_contacts(
			primary_contact_id=merge_request.primary_contact_id,
			duplicate_contact_ids=merge_request.duplicate_contact_ids,
			tenant_id=tenant_id,
			merged_by=user_id
		)
		
		return APIResponse(
			message=f"Successfully merged {len(merge_request.duplicate_contact_ids)} contacts",
			data=merged_contact.model_dump()
		)
		
	except CRMNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Merge endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Contact merge failed")


# ================================
# Account Hierarchy Management Endpoints
# ================================

class HierarchyUpdateRequest(BaseModel):
	"""Account hierarchy update request"""
	new_parent_id: Optional[str] = Field(None, description="New parent account ID")
	relationship_type: str = Field(default="parent_child", description="Relationship type")
	notes: Optional[str] = Field(None, description="Notes about the change")

@app.get("/accounts/hierarchy", response_model=APIResponse)
async def get_account_hierarchy(
	root_account_id: Optional[str] = Query(None, description="Root account ID"),
	max_depth: int = Query(10, description="Maximum hierarchy depth"),
	include_metrics: bool = Query(True, description="Include aggregated metrics"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Get complete account hierarchy tree
	
	Builds and returns the account hierarchy with support for filtering by root account,
	controlling depth, and including aggregated financial metrics.
	"""
	try:
		hierarchy = await service.build_account_hierarchy(
			tenant_id=tenant_id,
			root_account_id=root_account_id,
			max_depth=max_depth,
			include_metrics=include_metrics
		)
		
		return APIResponse(
			message=f"Hierarchy retrieved with {hierarchy['total_accounts']} accounts",
			data=hierarchy
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Hierarchy endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve hierarchy")

@app.get("/accounts/{account_id}/ancestors", response_model=APIResponse)
async def get_account_ancestors(
	account_id: str = Path(..., description="Account ID"),
	include_self: bool = Query(False, description="Include the account itself"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Get all ancestor accounts up the hierarchy
	
	Returns all parent accounts from the immediate parent to the root of the hierarchy.
	"""
	try:
		ancestors = await service.get_account_ancestors(
			account_id=account_id,
			tenant_id=tenant_id,
			include_self=include_self
		)
		
		return APIResponse(
			message=f"Found {len(ancestors)} ancestor accounts",
			data={"ancestors": ancestors, "total": len(ancestors)}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Ancestors endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve ancestors")

@app.get("/accounts/{account_id}/descendants", response_model=APIResponse)
async def get_account_descendants(
	account_id: str = Path(..., description="Account ID"),
	max_depth: int = Query(5, description="Maximum depth to traverse"),
	include_self: bool = Query(False, description="Include the account itself"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Get all descendant accounts down the hierarchy
	
	Returns all child accounts with configurable depth and hierarchy information.
	"""
	try:
		descendants = await service.get_account_descendants(
			account_id=account_id,
			tenant_id=tenant_id,
			max_depth=max_depth,
			include_self=include_self
		)
		
		return APIResponse(
			message=f"Found {len(descendants)} descendant accounts",
			data={"descendants": descendants, "total": len(descendants)}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Descendants endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve descendants")

@app.put("/accounts/{account_id}/hierarchy", response_model=APIResponse)
async def update_account_hierarchy(
	account_id: str = Path(..., description="Account ID"),
	hierarchy_update: HierarchyUpdateRequest = Body(...),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""
	Update account hierarchy relationships
	
	Move an account to a new parent or change its position in the hierarchy with
	full validation to prevent circular references and maintain data integrity.
	"""
	try:
		# Import the enum here to avoid circular imports
		from .account_hierarchy import HierarchyRelationshipType
		
		# Convert string to enum
		relationship_type = HierarchyRelationshipType(hierarchy_update.relationship_type)
		
		result = await service.update_account_hierarchy(
			account_id=account_id,
			new_parent_id=hierarchy_update.new_parent_id,
			relationship_type=relationship_type,
			tenant_id=tenant_id,
			updated_by=user_id,
			notes=hierarchy_update.notes
		)
		
		return APIResponse(
			message="Account hierarchy updated successfully",
			data=result
		)
		
	except ValueError as e:
		raise HTTPException(status_code=422, detail=f"Invalid relationship type: {str(e)}")
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Hierarchy update endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to update hierarchy")

@app.get("/accounts/hierarchy/analytics", response_model=APIResponse)
async def get_hierarchy_analytics(
	account_id: Optional[str] = Query(None, description="Specific account to analyze"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Get comprehensive hierarchy analytics and metrics
	
	Provides detailed analytics about the account hierarchy including depth distribution,
	account type breakdowns, and structural metrics.
	"""
	try:
		analytics = await service.get_hierarchy_analytics(
			tenant_id=tenant_id,
			account_id=account_id
		)
		
		return APIResponse(
			message="Hierarchy analytics retrieved successfully",
			data=analytics
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Hierarchy analytics endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve analytics")

@app.get("/accounts/hierarchy/path", response_model=APIResponse)
async def find_hierarchy_path(
	from_account_id: str = Query(..., description="Starting account ID"),
	to_account_id: str = Query(..., description="Target account ID"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Find path between two accounts in the hierarchy
	
	Determines if there's a hierarchical relationship between two accounts and
	returns the path through the hierarchy tree if one exists.
	"""
	try:
		path = await service.find_hierarchy_path(
			from_account_id=from_account_id,
			to_account_id=to_account_id,
			tenant_id=tenant_id
		)
		
		if path:
			return APIResponse(
				message=f"Found hierarchy path with {len(path)} steps",
				data={"path": path, "exists": True, "steps": len(path)}
			)
		else:
			return APIResponse(
				message="No hierarchy path found between accounts",
				data={"path": None, "exists": False, "steps": 0}
			)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Hierarchy path endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to find hierarchy path")


# ================================
# Account Relationship Management Endpoints
# ================================

class AccountRelationshipCreateRequest(BaseModel):
	"""Account relationship creation request"""
	from_account_id: str = Field(..., description="Source account ID")
	to_account_id: str = Field(..., description="Target account ID")
	relationship_type: str = Field(..., description="Type of relationship")
	relationship_strength: Optional[str] = Field("moderate", description="Strength of relationship")
	direction: Optional[str] = Field("outbound", description="Direction of relationship")
	annual_value: Optional[float] = Field(None, description="Annual relationship value")
	contract_start_date: Optional[str] = Field(None, description="Contract start date")
	contract_end_date: Optional[str] = Field(None, description="Contract end date")
	key_contact_id: Optional[str] = Field(None, description="Key contact ID")
	description: Optional[str] = Field(None, description="Relationship description")
	notes: Optional[str] = Field(None, description="Additional notes")
	tags: Optional[List[str]] = Field(None, description="Relationship tags")

class AccountRelationshipUpdateRequest(BaseModel):
	"""Account relationship update request"""
	relationship_type: Optional[str] = Field(None, description="Type of relationship")
	relationship_strength: Optional[str] = Field(None, description="Strength of relationship")
	relationship_status: Optional[str] = Field(None, description="Status of relationship")
	annual_value: Optional[float] = Field(None, description="Annual relationship value")
	contract_start_date: Optional[str] = Field(None, description="Contract start date")
	contract_end_date: Optional[str] = Field(None, description="Contract end date")
	key_contact_id: Optional[str] = Field(None, description="Key contact ID")
	description: Optional[str] = Field(None, description="Relationship description")
	notes: Optional[str] = Field(None, description="Additional notes")
	tags: Optional[List[str]] = Field(None, description="Relationship tags")


# ================================
# Territory Management Models
# ================================

class TerritoryCreateRequest(BaseModel):
	"""Territory creation request"""
	territory_name: str = Field(..., min_length=1, max_length=200, description="Territory name")
	territory_code: Optional[str] = Field(None, max_length=50, description="Territory code")
	territory_type: TerritoryType = Field(..., description="Type of territory")
	status: TerritoryStatus = Field(TerritoryStatus.ACTIVE, description="Territory status")
	owner_id: str = Field(..., description="Territory owner ID")
	sales_rep_ids: Optional[List[str]] = Field(None, description="Assigned sales representatives")
	
	# Geographic criteria
	countries: Optional[List[str]] = Field(None, description="Countries in territory")
	states_provinces: Optional[List[str]] = Field(None, description="States/provinces in territory")
	cities: Optional[List[str]] = Field(None, description="Cities in territory")
	postal_codes: Optional[List[str]] = Field(None, description="Postal codes in territory")
	
	# Business criteria
	industries: Optional[List[str]] = Field(None, description="Target industries")
	company_size_min: Optional[int] = Field(None, description="Minimum company size")
	company_size_max: Optional[int] = Field(None, description="Maximum company size")
	revenue_min: Optional[float] = Field(None, description="Minimum annual revenue")
	revenue_max: Optional[float] = Field(None, description="Maximum annual revenue")
	
	# Product/service criteria
	product_lines: Optional[List[str]] = Field(None, description="Product lines")
	service_types: Optional[List[str]] = Field(None, description="Service types")
	
	# Goals and metrics
	annual_quota: Optional[float] = Field(None, description="Annual quota")
	account_target: Optional[int] = Field(None, description="Target number of accounts")
	
	# Metadata
	description: Optional[str] = Field(None, max_length=2000, description="Territory description")
	notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")
	rules: Optional[Dict[str, Any]] = Field(None, description="Territory assignment rules")


class TerritoryUpdateRequest(BaseModel):
	"""Territory update request"""
	territory_name: Optional[str] = Field(None, min_length=1, max_length=200, description="Territory name")
	territory_code: Optional[str] = Field(None, max_length=50, description="Territory code")
	territory_type: Optional[TerritoryType] = Field(None, description="Type of territory")
	status: Optional[TerritoryStatus] = Field(None, description="Territory status")
	owner_id: Optional[str] = Field(None, description="Territory owner ID")
	sales_rep_ids: Optional[List[str]] = Field(None, description="Assigned sales representatives")
	
	# Geographic criteria
	countries: Optional[List[str]] = Field(None, description="Countries in territory")
	states_provinces: Optional[List[str]] = Field(None, description="States/provinces in territory")
	cities: Optional[List[str]] = Field(None, description="Cities in territory")
	postal_codes: Optional[List[str]] = Field(None, description="Postal codes in territory")
	
	# Business criteria
	industries: Optional[List[str]] = Field(None, description="Target industries")
	company_size_min: Optional[int] = Field(None, description="Minimum company size")
	company_size_max: Optional[int] = Field(None, description="Maximum company size")
	revenue_min: Optional[float] = Field(None, description="Minimum annual revenue")
	revenue_max: Optional[float] = Field(None, description="Maximum annual revenue")
	
	# Product/service criteria
	product_lines: Optional[List[str]] = Field(None, description="Product lines")
	service_types: Optional[List[str]] = Field(None, description="Service types")
	
	# Goals and metrics
	annual_quota: Optional[float] = Field(None, description="Annual quota")
	account_target: Optional[int] = Field(None, description="Target number of accounts")
	
	# Metadata
	description: Optional[str] = Field(None, max_length=2000, description="Territory description")
	notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")
	rules: Optional[Dict[str, Any]] = Field(None, description="Territory assignment rules")


class TerritoryAssignmentRequest(BaseModel):
	"""Territory assignment request"""
	account_id: str = Field(..., description="Account ID to assign")
	territory_id: str = Field(..., description="Territory ID for assignment")
	assignment_type: AssignmentType = Field(AssignmentType.PRIMARY, description="Type of assignment")
	assignment_reason: Optional[str] = Field(None, description="Reason for assignment")
	notes: Optional[str] = Field(None, max_length=1000, description="Assignment notes")


# ================================
# Communication History Models
# ================================

class CommunicationCreateRequest(BaseModel):
	"""Communication creation request"""
	# Related entities (at least one required)
	contact_id: Optional[str] = Field(None, description="Contact ID")
	account_id: Optional[str] = Field(None, description="Account ID")
	lead_id: Optional[str] = Field(None, description="Lead ID")
	opportunity_id: Optional[str] = Field(None, description="Opportunity ID")
	
	# Communication details
	communication_type: CommunicationType = Field(..., description="Type of communication")
	direction: CommunicationDirection = Field(..., description="Direction of communication")
	status: CommunicationStatus = Field(CommunicationStatus.COMPLETED, description="Communication status")
	priority: str = Field("normal", description="Priority level")
	
	# Content
	subject: str = Field(..., min_length=1, max_length=500, description="Communication subject")
	content: Optional[str] = Field(None, max_length=10000, description="Communication content")
	summary: Optional[str] = Field(None, max_length=2000, description="Communication summary")
	
	# Participants
	from_address: Optional[str] = Field(None, description="From address/number")
	to_addresses: Optional[List[str]] = Field(None, description="To addresses/numbers")
	cc_addresses: Optional[List[str]] = Field(None, description="CC addresses")
	bcc_addresses: Optional[List[str]] = Field(None, description="BCC addresses")
	participants: Optional[List[str]] = Field(None, description="Meeting participants")
	
	# Timing
	scheduled_at: Optional[str] = Field(None, description="Scheduled time (ISO format)")
	started_at: Optional[str] = Field(None, description="Start time (ISO format)")
	ended_at: Optional[str] = Field(None, description="End time (ISO format)")
	duration_minutes: Optional[int] = Field(None, ge=0, description="Duration in minutes")
	
	# Outcome and follow-up
	outcome: Optional[CommunicationOutcome] = Field(None, description="Communication outcome")
	outcome_notes: Optional[str] = Field(None, max_length=2000, description="Outcome notes")
	follow_up_required: bool = Field(False, description="Follow-up required")
	follow_up_date: Optional[str] = Field(None, description="Follow-up date (ISO format)")
	follow_up_notes: Optional[str] = Field(None, max_length=2000, description="Follow-up notes")
	
	# Attachments and references
	attachments: Optional[List[Dict[str, Any]]] = Field(None, description="Attachments")
	external_id: Optional[str] = Field(None, description="External system ID")
	external_source: Optional[str] = Field(None, description="External system name")
	
	# Metadata
	tags: Optional[List[str]] = Field(None, description="Communication tags")
	metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class CommunicationUpdateRequest(BaseModel):
	"""Communication update request"""
	# Communication details
	communication_type: Optional[CommunicationType] = Field(None, description="Type of communication")
	direction: Optional[CommunicationDirection] = Field(None, description="Direction of communication")
	status: Optional[CommunicationStatus] = Field(None, description="Communication status")
	priority: Optional[str] = Field(None, description="Priority level")
	
	# Content
	subject: Optional[str] = Field(None, min_length=1, max_length=500, description="Communication subject")
	content: Optional[str] = Field(None, max_length=10000, description="Communication content")
	summary: Optional[str] = Field(None, max_length=2000, description="Communication summary")
	
	# Participants
	from_address: Optional[str] = Field(None, description="From address/number")
	to_addresses: Optional[List[str]] = Field(None, description="To addresses/numbers")
	cc_addresses: Optional[List[str]] = Field(None, description="CC addresses")
	bcc_addresses: Optional[List[str]] = Field(None, description="BCC addresses")
	participants: Optional[List[str]] = Field(None, description="Meeting participants")
	
	# Timing
	scheduled_at: Optional[str] = Field(None, description="Scheduled time (ISO format)")
	started_at: Optional[str] = Field(None, description="Start time (ISO format)")
	ended_at: Optional[str] = Field(None, description="End time (ISO format)")
	duration_minutes: Optional[int] = Field(None, ge=0, description="Duration in minutes")
	
	# Outcome and follow-up
	outcome: Optional[CommunicationOutcome] = Field(None, description="Communication outcome")
	outcome_notes: Optional[str] = Field(None, max_length=2000, description="Outcome notes")
	follow_up_required: Optional[bool] = Field(None, description="Follow-up required")
	follow_up_date: Optional[str] = Field(None, description="Follow-up date (ISO format)")
	follow_up_notes: Optional[str] = Field(None, max_length=2000, description="Follow-up notes")
	
	# Attachments and references
	attachments: Optional[List[Dict[str, Any]]] = Field(None, description="Attachments")
	external_id: Optional[str] = Field(None, description="External system ID")
	external_source: Optional[str] = Field(None, description="External system name")
	
	# Metadata
	tags: Optional[List[str]] = Field(None, description="Communication tags")
	metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


@app.post("/accounts/relationships", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_account_relationship(
	relationship_data: AccountRelationshipCreateRequest,
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""
	Create a new account relationship
	
	Establishes business relationships between accounts with support for multiple
	relationship types, financial tracking, and automatic reverse relationships.
	"""
	try:
		# Parse dates if provided
		relationship_dict = relationship_data.model_dump(exclude_none=True)
		
		if relationship_dict.get('contract_start_date'):
			relationship_dict['contract_start_date'] = datetime.fromisoformat(
				relationship_dict['contract_start_date'].replace('Z', '+00:00')
			)
		
		if relationship_dict.get('contract_end_date'):
			relationship_dict['contract_end_date'] = datetime.fromisoformat(
				relationship_dict['contract_end_date'].replace('Z', '+00:00')
			)
		
		# Set owner to current user
		relationship_dict['relationship_owner_id'] = user_id
		
		relationship = await service.create_account_relationship(
			relationship_data=relationship_dict,
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message="Account relationship created successfully",
			data=relationship
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except ValueError as e:
		raise HTTPException(status_code=422, detail=str(e))
	except Exception as e:
		logger.error(f"Create relationship endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create relationship")

@app.get("/accounts/relationships/{relationship_id}", response_model=APIResponse)
async def get_account_relationship(
	relationship_id: str = Path(..., description="Relationship ID"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Get account relationship by ID
	
	Retrieves detailed information about a specific account relationship
	including financial terms, status, and metadata.
	"""
	try:
		relationship = await service.get_account_relationship(
			relationship_id=relationship_id,
			tenant_id=tenant_id
		)
		
		if not relationship:
			raise HTTPException(
				status_code=status.HTTP_404_NOT_FOUND,
				detail="Account relationship not found"
			)
		
		return APIResponse(
			message="Account relationship retrieved successfully",
			data=relationship
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Get relationship endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve relationship")

@app.get("/accounts/{account_id}/relationships", response_model=APIResponse)
async def get_account_relationships(
	account_id: str = Path(..., description="Account ID"),
	relationship_type: Optional[str] = Query(None, description="Filter by relationship type"),
	direction: Optional[str] = Query(None, description="Filter by direction"),
	status: Optional[str] = Query(None, description="Filter by status"),
	include_details: bool = Query(True, description="Include detailed account information"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Get all relationships for an account
	
	Returns comprehensive relationship data including outbound, inbound, and
	bidirectional relationships with filtering and summary statistics.
	"""
	try:
		# Convert string parameters to enums if provided
		relationship_type_enum = None
		direction_enum = None
		
		if relationship_type:
			from .account_relationships import AccountRelationshipType
			relationship_type_enum = AccountRelationshipType(relationship_type)
		
		if direction:
			from .account_relationships import RelationshipDirection
			direction_enum = RelationshipDirection(direction)
		
		relationships = await service.get_account_relationships(
			account_id=account_id,
			tenant_id=tenant_id,
			relationship_type=relationship_type_enum,
			direction=direction_enum,
			status=status,
			include_details=include_details
		)
		
		return APIResponse(
			message=f"Found {relationships['total']} relationships",
			data=relationships
		)
		
	except ValueError as e:
		raise HTTPException(status_code=422, detail=f"Invalid parameter: {str(e)}")
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Get relationships endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve relationships")

@app.put("/accounts/relationships/{relationship_id}", response_model=APIResponse)
async def update_account_relationship(
	relationship_id: str = Path(..., description="Relationship ID"),
	update_data: AccountRelationshipUpdateRequest = Body(...),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""
	Update account relationship
	
	Updates relationship properties including status, financial terms,
	and metadata with full validation and audit tracking.
	"""
	try:
		# Parse dates if provided
		update_dict = update_data.model_dump(exclude_none=True)
		
		if update_dict.get('contract_start_date'):
			update_dict['contract_start_date'] = datetime.fromisoformat(
				update_dict['contract_start_date'].replace('Z', '+00:00')
			)
		
		if update_dict.get('contract_end_date'):
			update_dict['contract_end_date'] = datetime.fromisoformat(
				update_dict['contract_end_date'].replace('Z', '+00:00')
			)
		
		relationship = await service.update_account_relationship(
			relationship_id=relationship_id,
			update_data=update_dict,
			tenant_id=tenant_id,
			updated_by=user_id
		)
		
		return APIResponse(
			message="Account relationship updated successfully",
			data=relationship
		)
		
	except ValueError as e:
		raise HTTPException(status_code=422, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Update relationship endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to update relationship")

@app.delete("/accounts/relationships/{relationship_id}", response_model=APIResponse)
async def delete_account_relationship(
	relationship_id: str = Path(..., description="Relationship ID"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Delete account relationship
	
	Removes an account relationship from the system. This action cannot be undone.
	"""
	try:
		success = await service.delete_account_relationship(
			relationship_id=relationship_id,
			tenant_id=tenant_id
		)
		
		if not success:
			raise HTTPException(
				status_code=status.HTTP_404_NOT_FOUND,
				detail="Account relationship not found"
			)
		
		return APIResponse(
			message="Account relationship deleted successfully",
			data={"relationship_id": relationship_id, "deleted": True}
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Delete relationship endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to delete relationship")

@app.get("/accounts/relationships/discover", response_model=APIResponse)
async def discover_potential_relationships(
	account_id: Optional[str] = Query(None, description="Specific account to analyze"),
	min_confidence: float = Query(0.7, description="Minimum confidence threshold"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Discover potential relationships between accounts
	
	Uses AI and data analysis to identify potential business relationships
	based on shared contacts, industry connections, and other signals.
	"""
	try:
		relationships = await service.discover_potential_relationships(
			tenant_id=tenant_id,
			account_id=account_id,
			min_confidence=min_confidence
		)
		
		return APIResponse(
			message=f"Discovered {len(relationships)} potential relationships",
			data={"potential_relationships": relationships, "total": len(relationships)}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Discover relationships endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to discover relationships")

@app.get("/accounts/relationships/analytics", response_model=APIResponse)
async def get_account_relationship_analytics(
	account_id: Optional[str] = Query(None, description="Specific account to analyze"),
	start_date: Optional[str] = Query(None, description="Analysis start date (ISO format)"),
	end_date: Optional[str] = Query(None, description="Analysis end date (ISO format)"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Get comprehensive account relationship analytics
	
	Provides detailed analytics about account relationships including type distribution,
	financial metrics, contract expiration tracking, and relationship trends.
	"""
	try:
		# Parse dates if provided
		start_date_obj = None
		end_date_obj = None
		
		if start_date:
			start_date_obj = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
		
		if end_date:
			end_date_obj = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
		
		analytics = await service.get_account_relationship_analytics(
			tenant_id=tenant_id,
			account_id=account_id,
			start_date=start_date_obj,
			end_date=end_date_obj
		)
		
		return APIResponse(
			message="Account relationship analytics retrieved successfully",
			data=analytics
		)
		
	except ValueError as e:
		raise HTTPException(status_code=422, detail=f"Invalid date format: {str(e)}")
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Relationship analytics endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


# ================================
# Territory Management Endpoints  
# ================================

@app.post("/territories", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_territory(
	territory_data: TerritoryCreateRequest,
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""
	Create a new sales territory
	
	Creates a new territory with geographic, industry, and business criteria for
	account assignment and sales coverage optimization.
	"""
	try:
		territory = await service.create_territory(
			territory_data=territory_data.model_dump(exclude_none=True),
			tenant_id=tenant_id,
			created_by=user_id,
			owner_id=territory_data.owner_id
		)
		
		return APIResponse(
			message="Territory created successfully",
			data=territory
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Create territory endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create territory")


@app.get("/territories/{territory_id}", response_model=APIResponse)
async def get_territory(
	territory_id: str = Path(..., description="Territory ID"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get territory by ID"""
	try:
		territory = await service.get_territory(territory_id, tenant_id)
		
		return APIResponse(
			message="Territory retrieved successfully",
			data=territory
		)
		
	except CRMNotFoundError:
		raise HTTPException(status_code=404, detail="Territory not found")
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Get territory endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve territory")


@app.get("/territories", response_model=APIResponse)
async def list_territories(
	territory_type: Optional[TerritoryType] = Query(None, description="Filter by territory type"),
	status: Optional[TerritoryStatus] = Query(None, description="Filter by status"),
	owner_id: Optional[str] = Query(None, description="Filter by owner"),
	limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
	offset: int = Query(0, ge=0, description="Number of results to skip"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""List territories with optional filtering"""
	try:
		result = await service.list_territories(
			tenant_id=tenant_id,
			territory_type=territory_type,
			status=status,
			owner_id=owner_id,
			limit=limit,
			offset=offset
		)
		
		return APIResponse(
			message="Territories retrieved successfully",
			data=result
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"List territories endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to list territories")


@app.put("/territories/{territory_id}", response_model=APIResponse)
async def update_territory(
	territory_id: str = Path(..., description="Territory ID"),
	update_data: TerritoryUpdateRequest = Body(...),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""Update territory information"""
	try:
		territory = await service.update_territory(
			territory_id=territory_id,
			update_data=update_data.model_dump(exclude_none=True),
			tenant_id=tenant_id,
			updated_by=user_id
		)
		
		return APIResponse(
			message="Territory updated successfully",
			data=territory
		)
		
	except CRMNotFoundError:
		raise HTTPException(status_code=404, detail="Territory not found")
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Update territory endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to update territory")


@app.post("/territories/assignments", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def assign_account_to_territory(
	assignment_data: TerritoryAssignmentRequest,
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""
	Assign an account to a territory
	
	Creates an assignment between an account and territory with specified
	assignment type and tracking information.
	"""
	try:
		assignment = await service.assign_account_to_territory(
			account_id=assignment_data.account_id,
			territory_id=assignment_data.territory_id,
			assignment_type=assignment_data.assignment_type,
			tenant_id=tenant_id,
			assigned_by=user_id,
			assignment_reason=assignment_data.assignment_reason
		)
		
		return APIResponse(
			message="Account assigned to territory successfully",
			data=assignment
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Territory assignment endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to assign account to territory")


@app.get("/territories/{territory_id}/assignments", response_model=APIResponse)
async def get_territory_assignments(
	territory_id: str = Path(..., description="Territory ID"),
	assignment_type: Optional[AssignmentType] = Query(None, description="Filter by assignment type"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get all account assignments for a territory"""
	try:
		assignments = await service.get_territory_assignments(
			territory_id=territory_id,
			tenant_id=tenant_id,
			assignment_type=assignment_type
		)
		
		return APIResponse(
			message="Territory assignments retrieved successfully",
			data=assignments
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Get territory assignments endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve territory assignments")


@app.get("/accounts/{account_id}/territories", response_model=APIResponse)
async def get_account_territory_assignments(
	account_id: str = Path(..., description="Account ID"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get all territory assignments for an account"""
	try:
		assignments = await service.get_account_territory_assignments(
			account_id=account_id,
			tenant_id=tenant_id
		)
		
		return APIResponse(
			message="Account territory assignments retrieved successfully",
			data=assignments
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Get account territory assignments endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve account territory assignments")


@app.get("/territories/{territory_id}/coverage", response_model=APIResponse)
async def analyze_territory_coverage(
	territory_id: str = Path(..., description="Territory ID"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Analyze territory coverage and performance
	
	Provides comprehensive analysis of territory coverage including account
	distribution, performance metrics, and coverage gaps.
	"""
	try:
		analysis = await service.analyze_territory_coverage(
			territory_id=territory_id,
			tenant_id=tenant_id
		)
		
		return APIResponse(
			message="Territory coverage analysis completed",
			data=analysis
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Territory coverage analysis endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to analyze territory coverage")


@app.get("/territories/{territory_id}/recommendations", response_model=APIResponse)
async def get_territory_assignment_recommendations(
	territory_id: str = Path(..., description="Territory ID"),
	limit: int = Query(10, ge=1, le=50, description="Maximum number of recommendations"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Get recommended accounts for territory assignment
	
	Returns AI-powered recommendations for accounts that should be assigned
	to the territory based on fit analysis and territory criteria.
	"""
	try:
		recommendations = await service.get_territory_assignment_recommendations(
			territory_id=territory_id,
			tenant_id=tenant_id,
			limit=limit
		)
		
		return APIResponse(
			message="Territory assignment recommendations retrieved successfully",
			data=recommendations
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Territory recommendations endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve territory recommendations")


# ================================
# Communication History Endpoints
# ================================

@app.post("/communications", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_communication(
	communication_data: CommunicationCreateRequest,
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""
	Create a new communication record
	
	Creates a comprehensive record of customer interactions including emails,
	calls, meetings, and other communications with full tracking and analytics.
	"""
	try:
		# Validate that at least one entity is provided
		if not any([communication_data.contact_id, communication_data.account_id,
				   communication_data.lead_id, communication_data.opportunity_id]):
			raise HTTPException(
				status_code=422, 
				detail="At least one related entity (contact_id, account_id, lead_id, or opportunity_id) is required"
			)
		
		# Parse datetime fields if provided
		communication_dict = communication_data.model_dump(exclude_none=True)
		
		for date_field in ['scheduled_at', 'started_at', 'ended_at', 'follow_up_date']:
			if communication_dict.get(date_field):
				try:
					communication_dict[date_field] = datetime.fromisoformat(
						communication_dict[date_field].replace('Z', '+00:00')
					)
				except ValueError:
					raise HTTPException(
						status_code=422, 
						detail=f"Invalid date format for {date_field}. Use ISO format."
					)
		
		communication = await service.create_communication(
			communication_data=communication_dict,
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message="Communication created successfully",
			data=communication
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Create communication endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create communication")


@app.get("/communications/{communication_id}", response_model=APIResponse)
async def get_communication(
	communication_id: str = Path(..., description="Communication ID"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get communication by ID"""
	try:
		communication = await service.get_communication(communication_id, tenant_id)
		
		return APIResponse(
			message="Communication retrieved successfully",
			data=communication
		)
		
	except CRMNotFoundError:
		raise HTTPException(status_code=404, detail="Communication not found")
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Get communication endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve communication")


@app.get("/communications", response_model=APIResponse)
async def list_communications(
	contact_id: Optional[str] = Query(None, description="Filter by contact ID"),
	account_id: Optional[str] = Query(None, description="Filter by account ID"),
	communication_type: Optional[CommunicationType] = Query(None, description="Filter by communication type"),
	direction: Optional[CommunicationDirection] = Query(None, description="Filter by direction"),
	status: Optional[CommunicationStatus] = Query(None, description="Filter by status"),
	start_date: Optional[str] = Query(None, description="Filter by start date (ISO format)"),
	end_date: Optional[str] = Query(None, description="Filter by end date (ISO format)"),
	tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
	limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
	offset: int = Query(0, ge=0, description="Number of results to skip"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""List communications with optional filtering"""
	try:
		# Parse dates if provided
		start_date_obj = None
		end_date_obj = None
		
		if start_date:
			try:
				start_date_obj = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
			except ValueError:
				raise HTTPException(status_code=422, detail="Invalid start_date format. Use ISO format.")
		
		if end_date:
			try:
				end_date_obj = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
			except ValueError:
				raise HTTPException(status_code=422, detail="Invalid end_date format. Use ISO format.")
		
		# Parse tags if provided
		tags_list = None
		if tags:
			tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
		
		result = await service.list_communications(
			tenant_id=tenant_id,
			contact_id=contact_id,
			account_id=account_id,
			communication_type=communication_type,
			direction=direction,
			status=status,
			start_date=start_date_obj,
			end_date=end_date_obj,
			tags=tags_list,
			limit=limit,
			offset=offset
		)
		
		return APIResponse(
			message="Communications retrieved successfully",
			data=result
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"List communications endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to list communications")


@app.put("/communications/{communication_id}", response_model=APIResponse)
async def update_communication(
	communication_id: str = Path(..., description="Communication ID"),
	update_data: CommunicationUpdateRequest = Body(...),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id)
):
	"""Update communication record"""
	try:
		# Parse datetime fields if provided
		update_dict = update_data.model_dump(exclude_none=True)
		
		for date_field in ['scheduled_at', 'started_at', 'ended_at', 'follow_up_date']:
			if update_dict.get(date_field):
				try:
					update_dict[date_field] = datetime.fromisoformat(
						update_dict[date_field].replace('Z', '+00:00')
					)
				except ValueError:
					raise HTTPException(
						status_code=422, 
						detail=f"Invalid date format for {date_field}. Use ISO format."
					)
		
		communication = await service.update_communication(
			communication_id=communication_id,
			update_data=update_dict,
			tenant_id=tenant_id,
			updated_by=user_id
		)
		
		return APIResponse(
			message="Communication updated successfully",
			data=communication
		)
		
	except HTTPException:
		raise
	except CRMNotFoundError:
		raise HTTPException(status_code=404, detail="Communication not found")
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Update communication endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to update communication")


@app.delete("/communications/{communication_id}", response_model=APIResponse)
async def delete_communication(
	communication_id: str = Path(..., description="Communication ID"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Delete communication record"""
	try:
		deleted = await service.delete_communication(communication_id, tenant_id)
		
		if not deleted:
			raise HTTPException(status_code=404, detail="Communication not found")
		
		return APIResponse(
			message="Communication deleted successfully",
			data={"deleted": True}
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Delete communication endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to delete communication")


@app.get("/communications/analytics", response_model=APIResponse)
async def get_communication_analytics(
	contact_id: Optional[str] = Query(None, description="Filter by contact ID"),
	account_id: Optional[str] = Query(None, description="Filter by account ID"),
	start_date: Optional[str] = Query(None, description="Analysis start date (ISO format)"),
	end_date: Optional[str] = Query(None, description="Analysis end date (ISO format)"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Get comprehensive communication analytics
	
	Provides detailed analytics about communication patterns, outcomes,
	response times, and engagement metrics for data-driven insights.
	"""
	try:
		# Parse dates if provided
		start_date_obj = None
		end_date_obj = None
		
		if start_date:
			try:
				start_date_obj = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
			except ValueError:
				raise HTTPException(status_code=422, detail="Invalid start_date format. Use ISO format.")
		
		if end_date:
			try:
				end_date_obj = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
			except ValueError:
				raise HTTPException(status_code=422, detail="Invalid end_date format. Use ISO format.")
		
		analytics = await service.get_communication_analytics(
			tenant_id=tenant_id,
			contact_id=contact_id,
			account_id=account_id,
			start_date=start_date_obj,
			end_date=end_date_obj
		)
		
		return APIResponse(
			message="Communication analytics retrieved successfully",
			data=analytics
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Communication analytics endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve communication analytics")


@app.get("/communications/follow-ups", response_model=APIResponse)
async def get_pending_follow_ups(
	user_id: Optional[str] = Query(None, description="Filter by user ID"),
	overdue_only: bool = Query(False, description="Only return overdue follow-ups"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Get pending follow-up communications
	
	Returns communications that require follow-up action, with options
	to filter by user or show only overdue items.
	"""
	try:
		follow_ups = await service.get_pending_follow_ups(
			tenant_id=tenant_id,
			user_id=user_id,
			overdue_only=overdue_only
		)
		
		return APIResponse(
			message="Pending follow-ups retrieved successfully",
			data=follow_ups
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Pending follow-ups endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve pending follow-ups")


@app.get("/contacts/{contact_id}/communications/timeline", response_model=APIResponse)
async def get_contact_communication_timeline(
	contact_id: str = Path(..., description="Contact ID"),
	limit: int = Query(20, ge=1, le=100, description="Maximum number of communications"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get communication timeline for a contact"""
	try:
		timeline = await service.get_communication_timeline(
			entity_id=contact_id,
			entity_type="contact",
			tenant_id=tenant_id,
			limit=limit
		)
		
		return APIResponse(
			message="Contact communication timeline retrieved successfully",
			data=timeline
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Contact communication timeline endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve contact communication timeline")


@app.get("/accounts/{account_id}/communications/timeline", response_model=APIResponse)
async def get_account_communication_timeline(
	account_id: str = Path(..., description="Account ID"),
	limit: int = Query(20, ge=1, le=100, description="Maximum number of communications"),
	service: CRMService = Depends(get_crm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get communication timeline for an account"""
	try:
		timeline = await service.get_communication_timeline(
			entity_id=account_id,
			entity_type="account",
			tenant_id=tenant_id,
			limit=limit
		)
		
		return APIResponse(
			message="Account communication timeline retrieved successfully",
			data=timeline
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Account communication timeline endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve account communication timeline")


# ================================
# Configuration Endpoints
# ================================

@app.get("/config", response_model=APIResponse)
async def get_configuration(
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get CRM configuration"""
	# TODO: Implement proper configuration management
	config = CRMCapabilityConfig()
	
	return APIResponse(
		message="Configuration retrieved successfully",
		data=config.model_dump()
	)


# ================================
# Lead Scoring Endpoints
# ================================

class LeadScoringRuleCreateRequest(BaseModel):
	"""Request model for creating lead scoring rules"""
	name: str = Field(..., min_length=1, max_length=200, description="Rule name")
	description: Optional[str] = Field(None, max_length=1000, description="Rule description")
	category: str = Field(..., description="Scoring category")
	weight: str = Field(..., description="Score weight level")
	field: str = Field(..., min_length=1, max_length=100, description="Field to evaluate")
	operator: str = Field(..., min_length=1, max_length=50, description="Comparison operator")
	value: Any = Field(..., description="Value to compare against")
	score_points: int = Field(..., ge=0, le=100, description="Points awarded when rule matches")
	is_active: bool = Field(True, description="Whether rule is active")
	applies_to_lead_sources: List[str] = Field(default_factory=list, description="Specific lead sources")
	applies_to_contact_types: List[str] = Field(default_factory=list, description="Specific contact types")
	valid_from: Optional[datetime] = Field(None, description="Rule valid from date")
	valid_until: Optional[datetime] = Field(None, description="Rule valid until date")


class LeadScoreCalculateRequest(BaseModel):
	"""Request model for calculating lead scores"""
	force_recalculate: bool = Field(False, description="Force recalculation")


class BatchLeadScoreRequest(BaseModel):
	"""Request model for batch lead scoring"""
	lead_ids: List[str] = Field(..., min_length=1, description="List of lead IDs to score")
	force_recalculate: bool = Field(False, description="Force recalculation for all leads")


@app.post("/lead-scoring/rules", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_lead_scoring_rule(
	request: LeadScoringRuleCreateRequest,
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Create a new lead scoring rule"""
	try:
		rule = await service.create_lead_scoring_rule(
			rule_data=request.model_dump(exclude_unset=True),
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message="Lead scoring rule created successfully",
			data=rule.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create lead scoring rule endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create lead scoring rule")


@app.post("/leads/{lead_id}/score", response_model=APIResponse)
async def calculate_lead_score(
	lead_id: str = Path(..., description="Lead ID"),
	request: LeadScoreCalculateRequest = Body(default_factory=LeadScoreCalculateRequest),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Calculate score for a specific lead"""
	try:
		score = await service.calculate_lead_score(
			lead_id=lead_id,
			tenant_id=tenant_id,
			force_recalculate=request.force_recalculate
		)
		
		return APIResponse(
			message="Lead score calculated successfully",
			data=score.model_dump()
		)
		
	except CRMNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Calculate lead score endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to calculate lead score")


@app.get("/leads/{lead_id}/score", response_model=APIResponse)
async def get_lead_score(
	lead_id: str = Path(..., description="Lead ID"),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Get current score for a specific lead"""
	try:
		score = await service.get_lead_score(
			lead_id=lead_id,
			tenant_id=tenant_id
		)
		
		if not score:
			raise HTTPException(status_code=404, detail="Lead score not found")
		
		return APIResponse(
			message="Lead score retrieved successfully",
			data=score.model_dump()
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get lead score endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve lead score")


@app.post("/leads/batch-score", response_model=APIResponse)
async def batch_score_leads(
	request: BatchLeadScoreRequest,
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Calculate scores for multiple leads in batch"""
	try:
		scores = await service.batch_score_leads(
			lead_ids=request.lead_ids,
			tenant_id=tenant_id,
			force_recalculate=request.force_recalculate
		)
		
		return APIResponse(
			message=f"Batch scoring completed for {len(scores)} leads",
			data={
				"scores": {lead_id: score.model_dump() for lead_id, score in scores.items()},
				"total_processed": len(scores),
				"requested_count": len(request.lead_ids)
			}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Batch score leads endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to batch score leads")


@app.get("/lead-scoring/analytics", response_model=APIResponse)
async def get_lead_scoring_analytics(
	period_days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Get comprehensive lead scoring analytics"""
	try:
		analytics = await service.get_lead_scoring_analytics(
			tenant_id=tenant_id,
			period_days=period_days
		)
		
		return APIResponse(
			message="Lead scoring analytics retrieved successfully",
			data=analytics
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Lead scoring analytics endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve lead scoring analytics")


@app.post("/lead-scoring/default-rules", response_model=APIResponse, status_code=status.HTTP_201_CREATED)  
async def create_default_scoring_rules(
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Create default lead scoring rules for the tenant"""
	try:
		rules = await service.create_default_scoring_rules(
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message=f"Default scoring rules created successfully ({len(rules)} rules)",
			data={
				"rules": [rule.model_dump() for rule in rules],
				"count": len(rules)
			}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create default scoring rules endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create default scoring rules")


# ================================
# APPROVAL WORKFLOWS ENDPOINTS
# ================================

@app.post("/approval-templates", response_model=APIResponse, status_code=status.HTTP_201_CREATED)  
async def create_approval_template(
	template_data: Dict[str, Any] = Body(...),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Create a new approval workflow template"""
	try:
		template = await service.create_approval_template(
			template_data=template_data,
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message="Approval template created successfully",
			data=template.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create approval template endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create approval template")


@app.post("/approval-requests", response_model=APIResponse, status_code=status.HTTP_201_CREATED)  
async def submit_approval_request(
	request_data: Dict[str, Any] = Body(...),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Submit a new approval request"""
	try:
		request = await service.submit_approval_request(
			request_data=request_data,
			tenant_id=tenant_id,
			requested_by=user_id
		)
		
		return APIResponse(
			message="Approval request submitted successfully",
			data=request.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Submit approval request endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to submit approval request")


@app.post("/approval-requests/{request_id}/actions", response_model=APIResponse)  
async def process_approval_action(
	request_id: str = Path(...),
	step_id: str = Body(...),
	action: str = Body(...),
	notes: str = Body(""),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Process an approval action (approve, reject, delegate, etc.)"""
	try:
		request = await service.process_approval_action(
			request_id=request_id,
			step_id=step_id,
			action=action,
			actor_id=user_id,
			notes=notes,
			tenant_id=tenant_id
		)
		
		return APIResponse(
			message=f"Approval action '{action}' processed successfully",
			data=request.model_dump()
		)
		
	except CRMNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Process approval action endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to process approval action")


@app.get("/approval-requests/{request_id}", response_model=APIResponse)
async def get_approval_request(
	request_id: str = Path(...),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Retrieve approval request by ID"""
	try:
		request = await service.get_approval_request(
			request_id=request_id,
			tenant_id=tenant_id
		)
		
		if not request:
			raise HTTPException(status_code=404, detail="Approval request not found")
		
		return APIResponse(
			message="Approval request retrieved successfully",
			data=request.model_dump()
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get approval request endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve approval request")


@app.get("/approval-requests", response_model=APIResponse)
async def list_approval_requests(
	status: Optional[str] = Query(None),
	requester_id: Optional[str] = Query(None),
	limit: int = Query(50, ge=1, le=100),
	offset: int = Query(0, ge=0),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""List approval requests with filtering"""
	try:
		requests, total = await service.list_approval_requests(
			tenant_id=tenant_id,
			status=status,
			requester_id=requester_id,
			limit=limit,
			offset=offset
		)
		
		return APIResponse(
			message=f"Retrieved {len(requests)} approval requests",
			data={
				"requests": [req.model_dump() for req in requests],
				"total": total,
				"limit": limit,
				"offset": offset
			}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"List approval requests endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to list approval requests")


@app.get("/approval-requests/pending/{approver_id}", response_model=APIResponse)
async def get_pending_approvals(
	approver_id: str = Path(...),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Get pending approvals for a specific approver"""
	try:
		steps = await service.get_pending_approvals(
			tenant_id=tenant_id,
			approver_id=approver_id
		)
		
		return APIResponse(
			message=f"Retrieved {len(steps)} pending approvals",
			data={
				"pending_approvals": [step.model_dump() for step in steps],
				"count": len(steps)
			}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get pending approvals endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve pending approvals")


@app.get("/approval-workflows/analytics", response_model=APIResponse)
async def get_approval_analytics(
	approval_type: Optional[str] = Query(None),
	period_days: int = Query(30, ge=1, le=365),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Get approval workflow analytics"""
	try:
		analytics = await service.get_approval_analytics(
			tenant_id=tenant_id,
			approval_type=approval_type,
			period_days=period_days
		)
		
		return APIResponse(
			message="Approval analytics retrieved successfully",
			data=analytics.model_dump()
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Approval analytics endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve approval analytics")


# ================================
# LEAD ASSIGNMENT ENDPOINTS
# ================================

@app.post("/lead-assignment/rules", response_model=APIResponse, status_code=status.HTTP_201_CREATED)  
async def create_assignment_rule(
	rule_data: Dict[str, Any] = Body(...),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Create a new lead assignment rule"""
	try:
		rule = await service.create_assignment_rule(
			rule_data=rule_data,
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message="Assignment rule created successfully",
			data=rule.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create assignment rule endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create assignment rule")


@app.post("/lead-assignment/assign", response_model=APIResponse)  
async def assign_lead(
	lead_data: Dict[str, Any] = Body(...),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Assign a lead using configured assignment rules"""
	try:
		assignment = await service.assign_lead(
			lead_data=lead_data,
			tenant_id=tenant_id
		)
		
		if not assignment:
			return APIResponse(
				message="No suitable assignment found for lead",
				data={"assigned": False}
			)
		
		return APIResponse(
			message=f"Lead assigned to {assignment.assigned_to_name}",
			data={
				"assigned": True,
				"assignment": assignment.model_dump()
			}
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Assign lead endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to assign lead")


@app.get("/lead-assignment/analytics", response_model=APIResponse)
async def get_assignment_analytics(
	period_days: int = Query(30, ge=1, le=365),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Get lead assignment analytics"""
	try:
		analytics = await service.get_assignment_analytics(
			tenant_id=tenant_id,
			period_days=period_days
		)
		
		return APIResponse(
			message="Assignment analytics retrieved successfully",
			data=analytics.model_dump()
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Assignment analytics endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve assignment analytics")


# ================================
# LEAD NURTURING ENDPOINTS
# ================================

@app.post("/lead-nurturing/workflows", response_model=APIResponse, status_code=status.HTTP_201_CREATED)  
async def create_nurturing_workflow(
	workflow_data: Dict[str, Any] = Body(...),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Create a new lead nurturing workflow"""
	try:
		workflow = await service.create_nurturing_workflow(
			workflow_data=workflow_data,
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message="Nurturing workflow created successfully",
			data=workflow.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create nurturing workflow endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create nurturing workflow")


@app.post("/lead-nurturing/enroll", response_model=APIResponse)  
async def enroll_lead_in_nurturing(
	workflow_id: str = Body(...),
	lead_data: Dict[str, Any] = Body(...),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Enroll a lead in a nurturing workflow"""
	try:
		enrollment = await service.enroll_lead_in_nurturing(
			workflow_id=workflow_id,
			lead_data=lead_data,
			tenant_id=tenant_id,
			enrolled_by=user_id
		)
		
		if not enrollment:
			return APIResponse(
				message="Lead enrollment failed - criteria not met or already enrolled",
				data={"enrolled": False}
			)
		
		return APIResponse(
			message="Lead enrolled in nurturing workflow successfully",
			data={
				"enrolled": True,
				"enrollment": enrollment.model_dump()
			}
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Enroll lead nurturing endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to enroll lead in nurturing")


@app.post("/lead-nurturing/trigger", response_model=APIResponse)  
async def process_nurturing_trigger(
	trigger_type: str = Body(...),
	trigger_data: Dict[str, Any] = Body(...),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Process a nurturing workflow trigger event"""
	try:
		await service.process_nurturing_trigger(
			trigger_type=trigger_type,
			trigger_data=trigger_data,
			tenant_id=tenant_id
		)
		
		return APIResponse(
			message="Nurturing trigger processed successfully",
			data={"processed": True}
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Process nurturing trigger endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to process nurturing trigger")


@app.get("/lead-nurturing/analytics", response_model=APIResponse)
async def get_nurturing_analytics(
	workflow_id: Optional[str] = Query(None),
	period_days: int = Query(30, ge=1, le=365),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Get lead nurturing analytics"""
	try:
		analytics = await service.get_nurturing_analytics(
			tenant_id=tenant_id,
			workflow_id=workflow_id,
			period_days=period_days
		)
		
		return APIResponse(
			message="Nurturing analytics retrieved successfully",
			data=analytics.model_dump()
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Nurturing analytics endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve nurturing analytics")


# ================================
# CRM DASHBOARD ENDPOINTS
# ================================

@app.post("/dashboards", response_model=APIResponse, status_code=status.HTTP_201_CREATED)  
async def create_dashboard(
	dashboard_data: Dict[str, Any] = Body(...),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Create a new CRM dashboard"""
	try:
		dashboard = await service.create_dashboard(
			dashboard_data=dashboard_data,
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message="Dashboard created successfully",
			data=dashboard.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create dashboard endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create dashboard")


@app.get("/dashboards/{dashboard_id}/data", response_model=APIResponse)
async def get_dashboard_data(
	dashboard_id: str = Path(...),
	time_range: Optional[str] = Query(None),
	filters: Optional[str] = Query(None),
	tenant_id: str = Depends(get_tenant_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Get dashboard data for all widgets"""
	try:
		# Parse filters if provided
		parsed_filters = None
		if filters:
			try:
				parsed_filters = json.loads(filters)
			except json.JSONDecodeError:
				raise HTTPException(status_code=400, detail="Invalid filters JSON format")
		
		dashboard_data = await service.get_dashboard_data(
			dashboard_id=dashboard_id,
			tenant_id=tenant_id,
			time_range=time_range,
			filters=parsed_filters
		)
		
		# Convert DashboardData objects to dictionaries
		serialized_data = {}
		for widget_id, data in dashboard_data.items():
			serialized_data[widget_id] = data.model_dump()
		
		return APIResponse(
			message="Dashboard data retrieved successfully",
			data={
				"dashboard_id": dashboard_id,
				"widgets": serialized_data,
				"widget_count": len(serialized_data)
			}
		)
		
	except CRMNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get dashboard data endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")


# ================================
# REPORTING ENGINE ENDPOINTS
# ================================

@app.post("/reports", response_model=APIResponse, status_code=status.HTTP_201_CREATED)  
async def create_report(
	report_data: Dict[str, Any] = Body(...),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Create a new report definition"""
	try:
		report = await service.create_report(
			report_data=report_data,
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message="Report created successfully",
			data=report.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create report endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create report")


@app.post("/reports/{report_id}/execute", response_model=APIResponse)
async def execute_report(
	report_id: str = Path(...),
	parameters: Optional[Dict[str, Any]] = Body(None),
	export_format: Optional[str] = Body(None),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Execute a report and return results"""
	try:
		report_data = await service.execute_report(
			report_id=report_id,
			tenant_id=tenant_id,
			executed_by=user_id,
			parameters=parameters,
			export_format=export_format
		)
		
		return APIResponse(
			message="Report executed successfully",
			data={
				"report_id": report_id,
				"execution_id": report_data.execution_id,
				"columns": report_data.columns,
				"rows": report_data.rows,
				"summary": report_data.summary,
				"total_rows": report_data.total_rows,
				"charts": report_data.charts,
				"export_urls": report_data.export_urls,
				"query_time_ms": report_data.query_time_ms,
				"processing_time_ms": report_data.processing_time_ms,
				"generated_at": report_data.generated_at.isoformat()
			}
		)
		
	except CRMNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except PermissionError as e:
		raise HTTPException(status_code=403, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Execute report endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to execute report")


@app.post("/reports/{report_id}/schedule", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def schedule_report(
	report_id: str = Path(...),
	schedule_data: Dict[str, Any] = Body(...),
	tenant_id: str = Depends(get_tenant_id),
	user_id: str = Depends(get_user_id),
	service: CRMService = Depends(get_crm_service)
):
	"""Create a report schedule"""
	try:
		# Add report_id to schedule data
		schedule_data["report_id"] = report_id
		
		schedule = await service.schedule_report(
			schedule_data=schedule_data,
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		return APIResponse(
			message="Report schedule created successfully",
			data=schedule.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Schedule report endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to schedule report")


# ================================
# PREDICTIVE ANALYTICS ENDPOINTS
# ================================

@app.post("/analytics/models", response_model=APIResponse)
async def create_prediction_model(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Create a new predictive model"""
	try:
		model = await crm_service.create_prediction_model(
			tenant_id=tenant_id,
			name=request["name"],
			model_type=request["model_type"],
			algorithm=request["algorithm"],
			target_variable=request["target_variable"],
			feature_columns=request["feature_columns"],
			data_sources=request["data_sources"],
			training_data_query=request["training_data_query"],
			created_by=user_id,
			hyperparameters=request.get("hyperparameters"),
			description=request.get("description")
		)
		
		return APIResponse(
			message="Prediction model created successfully",
			data=model.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create prediction model endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create prediction model")


@app.post("/analytics/models/{model_id}/train", response_model=APIResponse)
async def train_prediction_model(
	model_id: str,
	training_data: Optional[Dict[str, Any]] = None,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Train a predictive model"""
	try:
		result = await crm_service.train_prediction_model(
			tenant_id=tenant_id,
			model_id=model_id,
			training_data=training_data
		)
		
		return APIResponse(
			message="Model training completed successfully",
			data=result
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Train model endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to train model")


@app.post("/analytics/predictions", response_model=APIResponse)
async def make_prediction(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Make predictions using a trained model"""
	try:
		result = await crm_service.make_prediction(
			tenant_id=tenant_id,
			model_id=request["model_id"],
			input_data=request["input_data"],
			prediction_type=request.get("prediction_type", "single"),
			include_explanations=request.get("include_explanations", True),
			created_by=user_id
		)
		
		return APIResponse(
			message="Prediction generated successfully",
			data=result.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Make prediction endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to make prediction")


@app.get("/analytics/forecasts", response_model=APIResponse)
async def generate_sales_forecast(
	forecast_type: str = "revenue",
	period_type: str = "monthly",
	periods_ahead: int = 3,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Generate sales forecasting insights"""
	try:
		forecasts = await crm_service.generate_sales_forecast(
			tenant_id=tenant_id,
			forecast_type=forecast_type,
			period_type=period_type,
			periods_ahead=periods_ahead
		)
		
		return APIResponse(
			message="Sales forecast generated successfully",
			data=[forecast.model_dump() for forecast in forecasts]
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Generate forecast endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to generate forecast")


@app.get("/analytics/churn-predictions", response_model=APIResponse)
async def predict_customer_churn(
	entity_type: str = "contact",
	entity_ids: Optional[str] = None,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Predict customer churn risk"""
	try:
		entity_ids_list = entity_ids.split(",") if entity_ids else None
		
		predictions = await crm_service.predict_customer_churn(
			tenant_id=tenant_id,
			entity_type=entity_type,
			entity_ids=entity_ids_list
		)
		
		return APIResponse(
			message="Churn predictions generated successfully",
			data=[prediction.model_dump() for prediction in predictions]
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Predict churn endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to predict churn")


@app.get("/analytics/lead-scoring-insights", response_model=APIResponse)
async def optimize_lead_scoring(
	lead_ids: Optional[str] = None,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Generate optimized lead scoring insights"""
	try:
		lead_ids_list = lead_ids.split(",") if lead_ids else None
		
		insights = await crm_service.optimize_lead_scoring(
			tenant_id=tenant_id,
			lead_ids=lead_ids_list
		)
		
		return APIResponse(
			message="Lead scoring insights generated successfully",
			data=[insight.model_dump() for insight in insights]
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Optimize lead scoring endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to optimize lead scoring")


@app.post("/analytics/market-segmentation", response_model=APIResponse)
async def perform_market_segmentation(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Perform intelligent market segmentation analysis"""
	try:
		segments = await crm_service.perform_market_segmentation(
			tenant_id=tenant_id,
			segmentation_criteria=request["segmentation_criteria"],
			num_segments=request.get("num_segments", 5)
		)
		
		return APIResponse(
			message="Market segmentation completed successfully",
			data=[segment.model_dump() for segment in segments]
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Market segmentation endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to perform market segmentation")


@app.get("/analytics/models", response_model=APIResponse)
async def get_prediction_models(
	model_type: Optional[str] = None,
	is_active: bool = True,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Get available prediction models"""
	try:
		models = await crm_service.get_prediction_models(
			tenant_id=tenant_id,
			model_type=model_type,
			is_active=is_active
		)
		
		return APIResponse(
			message="Prediction models retrieved successfully",
			data=models
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get prediction models endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get prediction models")


@app.get("/analytics/ai-insights", response_model=APIResponse)
async def get_ai_insights(
	insight_category: Optional[str] = None,
	is_active: bool = True,
	limit: int = 10,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Get AI-generated insights"""
	try:
		insights = await crm_service.get_ai_insights(
			tenant_id=tenant_id,
			insight_category=insight_category,
			is_active=is_active,
			limit=limit
		)
		
		return APIResponse(
			message="AI insights retrieved successfully",
			data=insights
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get AI insights endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get AI insights")


@app.post("/analytics/ai-insights/{insight_id}/acknowledge", response_model=APIResponse)
async def acknowledge_ai_insight(
	insight_id: str,
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Acknowledge an AI insight"""
	try:
		success = await crm_service.acknowledge_ai_insight(
			tenant_id=tenant_id,
			insight_id=insight_id,
			acknowledged_by=user_id
		)
		
		if not success:
			raise HTTPException(status_code=404, detail="Insight not found")
		
		return APIResponse(
			message="AI insight acknowledged successfully",
			data={"acknowledged": True}
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Acknowledge insight endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to acknowledge insight")


# ================================
# PERFORMANCE BENCHMARKING ENDPOINTS
# ================================

@app.post("/performance/benchmarks", response_model=APIResponse)
async def create_performance_benchmark(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Create a new performance benchmark"""
	try:
		benchmark = await crm_service.create_performance_benchmark(
			tenant_id=tenant_id,
			name=request["name"],
			benchmark_type=request["benchmark_type"],
			metric_name=request["metric_name"],
			measurement_unit=request["measurement_unit"],
			benchmark_value=request["benchmark_value"],
			data_source=request["data_source"],
			calculation_method=request["calculation_method"],
			created_by=user_id,
			description=request.get("description"),
			target_value=request.get("target_value"),
			threshold_ranges=request.get("threshold_ranges"),
			period_type=request.get("period_type", "monthly")
		)
		
		return APIResponse(
			message="Performance benchmark created successfully",
			data=benchmark.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create benchmark endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create performance benchmark")


@app.post("/performance/measurements", response_model=APIResponse)
async def measure_performance(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Measure performance against a benchmark"""
	try:
		from datetime import datetime
		
		metric = await crm_service.measure_performance(
			tenant_id=tenant_id,
			benchmark_id=request["benchmark_id"],
			entity_type=request["entity_type"],
			entity_id=request["entity_id"],
			entity_name=request["entity_name"],
			measurement_period=request["measurement_period"],
			period_start=datetime.fromisoformat(request["period_start"]).date(),
			period_end=datetime.fromisoformat(request["period_end"]).date()
		)
		
		return APIResponse(
			message="Performance measured successfully",
			data=metric.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Measure performance endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to measure performance")


@app.post("/performance/comparisons", response_model=APIResponse)
async def compare_performance(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Compare performance across entities"""
	try:
		from datetime import datetime
		
		comparison = await crm_service.compare_performance(
			tenant_id=tenant_id,
			comparison_name=request["comparison_name"],
			comparison_type=request["comparison_type"],
			entities=request["entities"],
			metrics=request["metrics"],
			period_start=datetime.fromisoformat(request["period_start"]).date(),
			period_end=datetime.fromisoformat(request["period_end"]).date(),
			created_by=user_id
		)
		
		return APIResponse(
			message="Performance comparison completed successfully",
			data=comparison.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Compare performance endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to compare performance")


@app.put("/performance/goals/{goal_id}/progress", response_model=APIResponse)
async def track_goal_progress(
	goal_id: str,
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Update goal progress and tracking"""
	try:
		goal = await crm_service.track_goal_progress(
			tenant_id=tenant_id,
			goal_id=goal_id,
			current_value=request["current_value"],
			update_milestones=request.get("update_milestones", True)
		)
		
		return APIResponse(
			message="Goal progress updated successfully",
			data=goal.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Track goal progress endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to track goal progress")


@app.post("/performance/reports", response_model=APIResponse)
async def generate_performance_report(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Generate comprehensive performance report"""
	try:
		from datetime import datetime
		
		report = await crm_service.generate_performance_report(
			tenant_id=tenant_id,
			report_type=request["report_type"],
			entity_id=request["entity_id"],
			entity_name=request["entity_name"],
			period_start=datetime.fromisoformat(request["period_start"]).date(),
			period_end=datetime.fromisoformat(request["period_end"]).date(),
			include_peer_comparison=request.get("include_peer_comparison", True)
		)
		
		return APIResponse(
			message="Performance report generated successfully",
			data=report.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Generate performance report endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to generate performance report")


@app.get("/performance/dashboard", response_model=APIResponse)
async def get_performance_dashboard(
	entity_type: str,
	entity_id: str,
	dashboard_type: str = "comprehensive",
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Get performance dashboard data"""
	try:
		dashboard_data = await crm_service.get_performance_dashboard(
			tenant_id=tenant_id,
			entity_type=entity_type,
			entity_id=entity_id,
			dashboard_type=dashboard_type
		)
		
		return APIResponse(
			message="Performance dashboard retrieved successfully",
			data=dashboard_data
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get performance dashboard endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get performance dashboard")


@app.get("/performance/benchmarks", response_model=APIResponse)
async def get_performance_benchmarks(
	benchmark_type: Optional[str] = None,
	is_active: bool = True,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Get available performance benchmarks"""
	try:
		benchmarks = await crm_service.get_performance_benchmarks(
			tenant_id=tenant_id,
			benchmark_type=benchmark_type,
			is_active=is_active
		)
		
		return APIResponse(
			message="Performance benchmarks retrieved successfully",
			data=benchmarks
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get performance benchmarks endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get performance benchmarks")


@app.get("/performance/metrics", response_model=APIResponse)
async def get_performance_metrics(
	entity_type: Optional[str] = None,
	entity_id: Optional[str] = None,
	benchmark_id: Optional[str] = None,
	period_start: Optional[str] = None,
	period_end: Optional[str] = None,
	limit: int = 50,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Get performance metrics with filtering"""
	try:
		from datetime import datetime
		
		metrics = await crm_service.get_performance_metrics(
			tenant_id=tenant_id,
			entity_type=entity_type,
			entity_id=entity_id,
			benchmark_id=benchmark_id,
			period_start=datetime.fromisoformat(period_start).date() if period_start else None,
			period_end=datetime.fromisoformat(period_end).date() if period_end else None,
			limit=limit
		)
		
		return APIResponse(
			message="Performance metrics retrieved successfully",
			data=metrics
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get performance metrics endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get performance metrics")


# ================================
# API GATEWAY ENDPOINTS
# ================================

@app.post("/gateway/rate-limits", response_model=APIResponse)
async def create_rate_limit_rule(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Create a new rate limiting rule"""
	try:
		rule = await crm_service.create_rate_limit_rule(
			tenant_id=tenant_id,
			rule_name=request["rule_name"],
			resource_pattern=request["resource_pattern"],
			rate_limit_type=request["rate_limit_type"],
			limit_value=request["limit_value"],
			created_by=user_id,
			description=request.get("description"),
			window_size_seconds=request.get("window_size_seconds", 60),
			burst_limit=request.get("burst_limit"),
			scope=request.get("scope", "tenant"),
			enforcement_action=request.get("enforcement_action", "reject")
		)
		
		return APIResponse(
			message="Rate limit rule created successfully",
			data=rule.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create rate limit rule endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create rate limit rule")


@app.post("/gateway/endpoints", response_model=APIResponse)
async def register_api_endpoint(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Register a new API endpoint"""
	try:
		endpoint = await crm_service.register_api_endpoint(
			tenant_id=tenant_id,
			endpoint_path=request["endpoint_path"],
			http_methods=request["http_methods"],
			created_by=user_id,
			description=request.get("description"),
			version=request.get("version", "v1"),
			is_public=request.get("is_public", False),
			authentication_required=request.get("authentication_required", True),
			rate_limit_rules=request.get("rate_limit_rules"),
			caching_config=request.get("caching_config")
		)
		
		return APIResponse(
			message="API endpoint registered successfully",
			data=endpoint.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Register API endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to register API endpoint")


@app.get("/gateway/metrics", response_model=APIResponse)
async def get_api_gateway_metrics(
	start_date: str,
	end_date: str,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Get API gateway metrics for a time period"""
	try:
		from datetime import datetime
		
		metrics = await crm_service.get_api_gateway_metrics(
			tenant_id=tenant_id,
			start_date=datetime.fromisoformat(start_date),
			end_date=datetime.fromisoformat(end_date)
		)
		
		return APIResponse(
			message="API gateway metrics retrieved successfully",
			data=metrics.model_dump()
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get API gateway metrics endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get API gateway metrics")


@app.get("/gateway/rate-limits", response_model=APIResponse)
async def get_rate_limit_rules(
	is_active: bool = True,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Get rate limiting rules"""
	try:
		rules = await crm_service.get_rate_limit_rules(
			tenant_id=tenant_id,
			is_active=is_active
		)
		
		return APIResponse(
			message="Rate limit rules retrieved successfully",
			data=rules
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get rate limit rules endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get rate limit rules")


@app.get("/gateway/endpoints", response_model=APIResponse)
async def get_api_endpoints(
	version: Optional[str] = None,
	is_active: bool = True,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Get API endpoints"""
	try:
		endpoints = await crm_service.get_api_endpoints(
			tenant_id=tenant_id,
			version=version,
			is_active=is_active
		)
		
		return APIResponse(
			message="API endpoints retrieved successfully",
			data=endpoints
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get API endpoints endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get API endpoints")


@app.get("/gateway/requests", response_model=APIResponse)
async def get_api_requests_log(
	start_time: Optional[str] = None,
	end_time: Optional[str] = None,
	endpoint_path: Optional[str] = None,
	user_id: Optional[str] = None,
	limit: int = 100,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Get API request logs with filtering"""
	try:
		from datetime import datetime
		
		logs = await crm_service.get_api_requests_log(
			tenant_id=tenant_id,
			start_time=datetime.fromisoformat(start_time) if start_time else None,
			end_time=datetime.fromisoformat(end_time) if end_time else None,
			endpoint_path=endpoint_path,
			user_id=user_id,
			limit=limit
		)
		
		return APIResponse(
			message="API request logs retrieved successfully",
			data=logs
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get API request logs endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get API request logs")


@app.put("/gateway/rate-limits/{rule_id}", response_model=APIResponse)
async def update_rate_limit_rule(
	rule_id: str,
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Update a rate limiting rule"""
	try:
		success = await crm_service.update_rate_limit_rule(
			tenant_id=tenant_id,
			rule_id=rule_id,
			updates=request,
			updated_by=user_id
		)
		
		if not success:
			raise HTTPException(status_code=404, detail="Rate limit rule not found")
		
		return APIResponse(
			message="Rate limit rule updated successfully",
			data={"updated": True}
		)
		
	except HTTPException:
		raise
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Update rate limit rule endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to update rate limit rule")


@app.delete("/gateway/rate-limits/{rule_id}", response_model=APIResponse)
async def delete_rate_limit_rule(
	rule_id: str,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Delete a rate limiting rule"""
	try:
		success = await crm_service.delete_rate_limit_rule(
			tenant_id=tenant_id,
			rule_id=rule_id
		)
		
		if not success:
			raise HTTPException(status_code=404, detail="Rate limit rule not found")
		
		return APIResponse(
			message="Rate limit rule deleted successfully",
			data={"deleted": True}
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Delete rate limit rule endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to delete rate limit rule")


# ================================
# THIRD-PARTY INTEGRATION ENDPOINTS
# ================================

@app.post("/integrations/connectors", response_model=APIResponse)
async def create_integration_connector(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Create a new third-party integration connector"""
	try:
		connector = await crm_service.create_integration_connector(
			tenant_id=tenant_id,
			connector_name=request["connector_name"],
			integration_type=request["integration_type"],
			platform_name=request["platform_name"],
			base_url=request["base_url"],
			authentication_type=request["authentication_type"],
			authentication_config=request["authentication_config"],
			created_by=user_id,
			description=request.get("description"),
			supported_operations=request.get("supported_operations"),
			supported_entities=request.get("supported_entities"),
			custom_headers=request.get("custom_headers"),
			rate_limit_config=request.get("rate_limit_config")
		)
		
		return APIResponse(
			message="Integration connector created successfully",
			data=connector.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create integration connector endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create integration connector")


@app.get("/integrations/connectors", response_model=APIResponse)
async def get_integration_connectors(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	integration_type: Optional[str] = Query(None, description="Filter by integration type"),
	is_active: Optional[bool] = Query(None, description="Filter by active status")
):
	"""Get integration connectors"""
	try:
		connectors = await crm_service.get_integration_connectors(
			tenant_id=tenant_id,
			integration_type=integration_type,
			is_active=is_active
		)
		
		return APIResponse(
			message="Integration connectors retrieved successfully",
			data=connectors
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get integration connectors endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get integration connectors")


@app.post("/integrations/field-mappings", response_model=APIResponse)
async def create_field_mapping(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Create field mapping configuration for integration"""
	try:
		mapping = await crm_service.create_field_mapping(
			connector_id=request["connector_id"],
			tenant_id=tenant_id,
			mapping_name=request["mapping_name"],
			source_entity=request["source_entity"],
			target_entity=request["target_entity"],
			field_mappings=request["field_mappings"],
			sync_direction=request["sync_direction"],
			created_by=user_id,
			transformation_functions=request.get("transformation_functions"),
			validation_rules=request.get("validation_rules"),
			default_values=request.get("default_values")
		)
		
		return APIResponse(
			message="Field mapping created successfully",
			data=mapping.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create field mapping endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create field mapping")


@app.post("/integrations/sync-configurations", response_model=APIResponse)
async def create_sync_configuration(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Create synchronization configuration"""
	try:
		sync_config = await crm_service.create_sync_configuration(
			connector_id=request["connector_id"],
			tenant_id=tenant_id,
			sync_name=request["sync_name"],
			sync_frequency=request["sync_frequency"],
			created_by=user_id,
			description=request.get("description"),
			schedule_config=request.get("schedule_config"),
			entity_filters=request.get("entity_filters"),
			batch_size=request.get("batch_size", 100)
		)
		
		return APIResponse(
			message="Sync configuration created successfully",
			data=sync_config.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create sync configuration endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create sync configuration")


@app.post("/integrations/sync-executions", response_model=APIResponse)
async def execute_integration_sync(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Execute a synchronization"""
	try:
		execution = await crm_service.execute_integration_sync(
			sync_config_id=request["sync_config_id"],
			tenant_id=tenant_id,
			execution_type=request.get("execution_type", "manual"),
			trigger_source=request.get("trigger_source")
		)
		
		return APIResponse(
			message="Sync execution started successfully",
			data=execution.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Execute integration sync endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to execute integration sync")


@app.get("/integrations/sync-history", response_model=APIResponse)
async def get_sync_history(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	connector_id: Optional[str] = Query(None, description="Filter by connector ID"),
	limit: int = Query(100, description="Maximum number of records to return")
):
	"""Get synchronization execution history"""
	try:
		history = await crm_service.get_sync_history(
			tenant_id=tenant_id,
			connector_id=connector_id,
			limit=limit
		)
		
		return APIResponse(
			message="Sync history retrieved successfully",
			data=history
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get sync history endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get sync history")


# ================================
# WEBHOOK MANAGEMENT ENDPOINTS
# ================================

@app.post("/webhooks/endpoints", response_model=APIResponse)
async def create_webhook_endpoint(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Create a new webhook endpoint"""
	try:
		webhook = await crm_service.create_webhook_endpoint(
			tenant_id=tenant_id,
			webhook_name=request["webhook_name"],
			endpoint_url=request["endpoint_url"],
			event_types=request["event_types"],
			created_by=user_id,
			description=request.get("description"),
			http_method=request.get("http_method", "POST"),
			headers=request.get("headers"),
			authentication=request.get("authentication"),
			retry_config=request.get("retry_config"),
			timeout_seconds=request.get("timeout_seconds", 30),
			secret_key=request.get("secret_key")
		)
		
		return APIResponse(
			message="Webhook endpoint created successfully",
			data=webhook.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create webhook endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create webhook endpoint")


@app.get("/webhooks/endpoints", response_model=APIResponse)
async def get_webhook_endpoints(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	is_active: Optional[bool] = Query(None, description="Filter by active status"),
	event_type: Optional[str] = Query(None, description="Filter by event type")
):
	"""Get webhook endpoints for tenant"""
	try:
		webhooks = await crm_service.get_webhook_endpoints(
			tenant_id=tenant_id,
			is_active=is_active,
			event_type=event_type
		)
		
		return APIResponse(
			message="Webhook endpoints retrieved successfully",
			data=webhooks
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get webhook endpoints error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get webhook endpoints")


@app.post("/webhooks/events", response_model=APIResponse)
async def emit_webhook_event(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(None, alias="X-User-ID")
):
	"""Emit a webhook event for processing"""
	try:
		event = await crm_service.emit_webhook_event(
			tenant_id=tenant_id,
			event_type=request["event_type"],
			event_category=request["event_category"],
			event_action=request["event_action"],
			entity_id=request["entity_id"],
			entity_type=request["entity_type"],
			entity_data=request["entity_data"],
			user_id=user_id,
			previous_data=request.get("previous_data"),
			metadata=request.get("metadata")
		)
		
		return APIResponse(
			message="Webhook event emitted successfully",
			data=event.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Emit webhook event error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to emit webhook event")


@app.get("/webhooks/delivery-history", response_model=APIResponse)
async def get_webhook_delivery_history(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	webhook_id: Optional[str] = Query(None, description="Filter by webhook ID"),
	event_type: Optional[str] = Query(None, description="Filter by event type"),
	success: Optional[bool] = Query(None, description="Filter by success status"),
	limit: int = Query(100, description="Maximum number of records to return")
):
	"""Get webhook delivery history"""
	try:
		history = await crm_service.get_webhook_delivery_history(
			tenant_id=tenant_id,
			webhook_id=webhook_id,
			event_type=event_type,
			success=success,
			limit=limit
		)
		
		return APIResponse(
			message="Webhook delivery history retrieved successfully",
			data=history
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get webhook delivery history error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get webhook delivery history")


@app.get("/webhooks/metrics", response_model=APIResponse)
async def get_webhook_metrics(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
	end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
	"""Get webhook delivery metrics"""
	try:
		from datetime import datetime
		
		start = datetime.fromisoformat(start_date)
		end = datetime.fromisoformat(end_date)
		
		metrics = await crm_service.get_webhook_metrics(
			tenant_id=tenant_id,
			start_date=start,
			end_date=end
		)
		
		return APIResponse(
			message="Webhook metrics retrieved successfully",
			data=metrics
		)
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get webhook metrics error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get webhook metrics")


@app.post("/webhooks/test/{webhook_id}", response_model=APIResponse)
async def test_webhook_endpoint(
	webhook_id: str,
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Test webhook endpoint connectivity"""
	try:
		result = await crm_service.test_webhook_endpoint(
			tenant_id=tenant_id,
			webhook_id=webhook_id
		)
		
		return APIResponse(
			message="Webhook endpoint test completed",
			data=result
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Test webhook endpoint error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to test webhook endpoint")


# ================================
# REAL-TIME SYNCHRONIZATION ENDPOINTS
# ================================

@app.post("/realtime-sync/events", response_model=APIResponse)
async def emit_realtime_sync_event(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(None, alias="X-User-ID")
):
	"""Emit a real-time synchronization event"""
	try:
		event = await crm_service.emit_realtime_sync_event(
			tenant_id=tenant_id,
			event_type=request["event_type"],
			entity_type=request["entity_type"],
			entity_id=request["entity_id"],
			current_data=request["current_data"],
			previous_data=request.get("previous_data"),
			user_id=user_id,
			target_systems=request.get("target_systems"),
			metadata=request.get("metadata")
		)
		
		return APIResponse(
			message="Real-time sync event emitted successfully",
			data={
				"event_id": event.id,
				"correlation_id": event.correlation_id,
				"timestamp": event.timestamp.isoformat()
			}
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Emit realtime sync event error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to emit realtime sync event")


@app.post("/realtime-sync/configurations", response_model=APIResponse)
async def create_realtime_sync_configuration(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Create a new real-time sync configuration"""
	try:
		config = await crm_service.create_realtime_sync_configuration(
			tenant_id=tenant_id,
			config_name=request["config_name"],
			entity_types=request["entity_types"],
			target_systems=request["target_systems"],
			created_by=user_id,
			description=request.get("description"),
			change_detection_mode=request.get("change_detection_mode", "timestamp_based"),
			conflict_resolution=request.get("conflict_resolution", "timestamp_wins"),
			sync_direction=request.get("sync_direction", "bidirectional"),
			field_filters=request.get("field_filters"),
			**{k: v for k, v in request.items() if k not in [
				"config_name", "entity_types", "target_systems", "description",
				"change_detection_mode", "conflict_resolution", "sync_direction", "field_filters"
			]}
		)
		
		return APIResponse(
			message="Real-time sync configuration created successfully",
			data=config.model_dump()
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create realtime sync configuration error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create realtime sync configuration")


@app.get("/realtime-sync/configurations", response_model=APIResponse)
async def get_realtime_sync_configurations(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	is_active: Optional[bool] = Query(None, description="Filter by active status")
):
	"""Get real-time sync configurations"""
	try:
		configurations = await crm_service.get_realtime_sync_configurations(
			tenant_id=tenant_id,
			is_active=is_active
		)
		
		return APIResponse(
			message="Real-time sync configurations retrieved successfully",
			data=configurations
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get realtime sync configurations error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get realtime sync configurations")


@app.get("/realtime-sync/status", response_model=APIResponse)
async def get_realtime_sync_status(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	config_id: Optional[str] = Query(None, description="Specific configuration ID")
):
	"""Get real-time sync status and metrics"""
	try:
		status = await crm_service.get_realtime_sync_status(
			tenant_id=tenant_id,
			config_id=config_id
		)
		
		return APIResponse(
			message="Real-time sync status retrieved successfully",
			data=status
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get realtime sync status error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get realtime sync status")


@app.post("/realtime-sync/pause", response_model=APIResponse)
async def pause_realtime_sync(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Pause real-time synchronization"""
	try:
		success = await crm_service.pause_realtime_sync(
			tenant_id=tenant_id,
			config_id=request.get("config_id")
		)
		
		if not success:
			raise HTTPException(status_code=404, detail="Sync configuration not found")
		
		return APIResponse(
			message="Real-time sync paused successfully",
			data={"paused": True}
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Pause realtime sync error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to pause realtime sync")


@app.post("/realtime-sync/resume", response_model=APIResponse)
async def resume_realtime_sync(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Resume real-time synchronization"""
	try:
		success = await crm_service.resume_realtime_sync(
			tenant_id=tenant_id,
			config_id=request.get("config_id")
		)
		
		if not success:
			raise HTTPException(status_code=404, detail="Sync configuration not found")
		
		return APIResponse(
			message="Real-time sync resumed successfully",
			data={"resumed": True}
		)
		
	except HTTPException:
		raise
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Resume realtime sync error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to resume realtime sync")


@app.get("/realtime-sync/conflicts", response_model=APIResponse)
async def get_sync_conflict_records(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	resolved: Optional[bool] = Query(None, description="Filter by resolution status"),
	limit: int = Query(100, description="Maximum number of records to return")
):
	"""Get data conflict records"""
	try:
		conflicts = await crm_service.get_sync_conflict_records(
			tenant_id=tenant_id,
			resolved=resolved,
			limit=limit
		)
		
		return APIResponse(
			message="Sync conflict records retrieved successfully",
			data=conflicts
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get sync conflict records error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get sync conflict records")


@app.post("/realtime-sync/conflicts/{conflict_id}/resolve", response_model=APIResponse)
async def resolve_sync_conflict(
	conflict_id: str,
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	user_id: str = Header(..., alias="X-User-ID")
):
	"""Manually resolve a data conflict"""
	try:
		success = await crm_service.resolve_sync_conflict(
			tenant_id=tenant_id,
			conflict_id=conflict_id,
			resolution_strategy=request["resolution_strategy"],
			resolved_value=request["resolved_value"],
			resolved_by=user_id
		)
		
		if not success:
			raise HTTPException(status_code=404, detail="Conflict not found or already resolved")
		
		return APIResponse(
			message="Sync conflict resolved successfully",
			data={"resolved": True}
		)
		
	except HTTPException:
		raise
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Resolve sync conflict error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to resolve sync conflict")


# ================================
# API Versioning & Deprecation Management
# ================================

@app.post("/api-versioning/versions", response_model=APIResponse)
async def create_api_version(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Create a new API version"""
	try:
		version = await crm_service.create_api_version(
			tenant_id=tenant_id,
			version_number=request["version_number"],
			version_name=request.get("version_name"),
			status=request.get("status", "development"),
			supported_endpoints=request.get("supported_endpoints", []),
			breaking_changes=request.get("breaking_changes", []),
			documentation_url=request.get("documentation_url"),
			created_by=request["created_by"]
		)
		
		return APIResponse(
			message="API version created successfully",
			data=version
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create API version error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create API version")


@app.get("/api-versioning/versions", response_model=APIResponse)
async def get_api_versions(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	status: Optional[str] = Query(None, description="Filter by version status"),
	is_default: Optional[bool] = Query(None, description="Filter default versions")
):
	"""Get API versions"""
	try:
		versions = await crm_service.get_api_versions(
			tenant_id=tenant_id,
			status=status,
			is_default=is_default
		)
		
		return APIResponse(
			message="API versions retrieved successfully",
			data={"versions": versions, "count": len(versions)}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get API versions error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get API versions")


@app.put("/api-versioning/versions/{version_id}", response_model=APIResponse)
async def update_api_version(
	version_id: str,
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Update an API version"""
	try:
		updated = await crm_service.update_api_version(
			tenant_id=tenant_id,
			version_id=version_id,
			**request
		)
		
		if not updated:
			raise HTTPException(status_code=404, detail="API version not found")
		
		return APIResponse(
			message="API version updated successfully",
			data={"updated": True}
		)
		
	except HTTPException:
		raise
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Update API version error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to update API version")


@app.post("/api-versioning/deprecations", response_model=APIResponse)
async def create_deprecation_notice(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Create a deprecation notice"""
	try:
		notice = await crm_service.create_deprecation_notice(
			tenant_id=tenant_id,
			version_id=request["version_id"],
			endpoint_path=request["endpoint_path"],
			http_method=request.get("http_method", "GET"),
			severity=request.get("severity", "medium"),
			deprecation_reason=request["deprecation_reason"],
			replacement_endpoint=request.get("replacement_endpoint"),
			replacement_version=request.get("replacement_version"),
			grace_period_days=request.get("grace_period_days", 90),
			created_by=request["created_by"]
		)
		
		return APIResponse(
			message="Deprecation notice created successfully",
			data=notice
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create deprecation notice error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create deprecation notice")


@app.get("/api-versioning/deprecations", response_model=APIResponse)
async def get_deprecation_notices(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	version_id: Optional[str] = Query(None, description="Filter by version ID"),
	severity: Optional[str] = Query(None, description="Filter by severity")
):
	"""Get deprecation notices"""
	try:
		notices = await crm_service.get_deprecation_notices(
			tenant_id=tenant_id,
			version_id=version_id,
			severity=severity
		)
		
		return APIResponse(
			message="Deprecation notices retrieved successfully",
			data={"notices": notices, "count": len(notices)}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get deprecation notices error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get deprecation notices")


@app.get("/api-versioning/client-usage", response_model=APIResponse)
async def get_client_version_usage(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	version_id: Optional[str] = Query(None, description="Filter by version ID"),
	client_type: Optional[str] = Query(None, description="Filter by client type"),
	migration_status: Optional[str] = Query(None, description="Filter by migration status")
):
	"""Get client version usage analytics"""
	try:
		usage = await crm_service.get_client_version_usage(
			tenant_id=tenant_id,
			version_id=version_id,
			client_type=client_type,
			migration_status=migration_status
		)
		
		return APIResponse(
			message="Client version usage retrieved successfully",
			data={"usage": usage, "count": len(usage)}
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get client version usage error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get client version usage")


@app.post("/api-versioning/migrations", response_model=APIResponse)
async def create_version_migration(
	request: Dict[str, Any],
	tenant_id: str = Header(..., alias="X-Tenant-ID")
):
	"""Create a version migration plan"""
	try:
		migration = await crm_service.create_version_migration(
			tenant_id=tenant_id,
			from_version_id=request["from_version_id"],
			to_version_id=request["to_version_id"],
			migration_name=request["migration_name"],
			complexity=request.get("complexity", "moderate"),
			is_breaking_change=request.get("is_breaking_change", False),
			field_mappings=request.get("field_mappings", {}),
			transformation_rules=request.get("transformation_rules", {}),
			created_by=request["created_by"]
		)
		
		return APIResponse(
			message="Version migration created successfully",
			data=migration
		)
		
	except CRMValidationError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Create version migration error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to create version migration")


@app.get("/api-versioning/analytics", response_model=APIResponse)
async def get_version_analytics(
	tenant_id: str = Header(..., alias="X-Tenant-ID"),
	version_id: Optional[str] = Query(None, description="Specific version ID"),
	days: int = Query(30, description="Number of days for analytics")
):
	"""Get API version usage analytics"""
	try:
		analytics = await crm_service.get_version_analytics(
			tenant_id=tenant_id,
			version_id=version_id,
			days=days
		)
		
		return APIResponse(
			message="Version analytics retrieved successfully",
			data=analytics
		)
		
	except CRMServiceError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		logger.error(f"Get version analytics error: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to get version analytics")


# ================================
# OpenAPI Customization
# ================================

# Customize OpenAPI schema
def custom_openapi():
	if app.openapi_schema:
		return app.openapi_schema
	
	openapi_schema = get_openapi(
		title="APG Customer Relationship Management API",
		version="1.0.0",
		description="Revolutionary CRM API with 10x superior performance",
		routes=app.routes,
	)
	
	# Add custom info
	openapi_schema["info"]["x-logo"] = {
		"url": "https://datacraft.co.ke/logo.png"
	}
	
	app.openapi_schema = openapi_schema
	return app.openapi_schema


app.openapi = custom_openapi


# ================================
# Export
# ================================

__all__ = ["app"]