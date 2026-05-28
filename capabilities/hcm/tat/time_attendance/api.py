"""
Time & Attendance Capability API

FastAPI-based REST API providing comprehensive endpoints for the revolutionary
APG Time & Attendance capability with full support for traditional, remote, and AI workers.

Copyright � 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, Path, Body, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .service import TimeAttendanceService
from .views import (
	# Request models
	ClockInRequest, ClockOutRequest, TimeEntryUpdateRequest,
	RemoteWorkSessionRequest, ProductivityTrackingRequest,
	AIAgentRegistrationRequest, AIAgentWorkTrackingRequest,
	HybridCollaborationRequest, BiometricAuthenticationRequest,
	LeaveRequestSubmission, ScheduleOptimizationRequest,
	BulkTimeEntryUpdate, BulkApprovalRequest,
	
	# Response models
	SuccessResponse, ErrorResponse, TimeEntryResponse,
	RemoteWorkerResponse, AIAgentResponse, HybridCollaborationResponse,
	WorkforcePredictionsResponse, ProductivityAnalysisResponse,
	ConfigurationResponse, HealthCheckResponse,
	TimeEntriesListResponse, RemoteWorkersListResponse, AIAgentsListResponse
)
from .models import WorkMode, AIAgentType, ProductivityMetric
from .config import get_config
from .context import resolve_current_user_context

# Initialize logger
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Create FastAPI router
router = APIRouter(
	prefix="/api/human_capital_management/time_attendance",
	tags=["Time & Attendance"],
	responses={
		400: {"model": ErrorResponse, "description": "Bad Request"},
		401: {"model": ErrorResponse, "description": "Unauthorized"},
		403: {"model": ErrorResponse, "description": "Forbidden"},
		404: {"model": ErrorResponse, "description": "Not Found"},
		500: {"model": ErrorResponse, "description": "Internal Server Error"}
	}
)


# Dependency functions
async def get_current_user(
	request: Request,
	credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
	"""Get current authenticated user"""
	return resolve_current_user_context(request)


async def get_tenant_id(current_user: Dict[str, Any] = Depends(get_current_user)) -> str:
	"""Extract tenant ID from current user"""
	return current_user.get("tenant_id", "default")


async def get_service() -> TimeAttendanceService:
	"""Get Time Attendance service instance"""
	return TimeAttendanceService()


# Helper functions
def create_success_response(data: Any = None, message: str = "Success") -> SuccessResponse:
	"""Create standardized success response"""
	return SuccessResponse(
		success=True,
		message=message,
		data=data,
		timestamp=datetime.utcnow()
	)


def create_error_response(message: str, error_code: int = 400, details: Any = None) -> ErrorResponse:
	"""Create standardized error response"""
	return ErrorResponse(
		success=False,
		message=message,
		error_code=error_code,
		details=details,
		timestamp=datetime.utcnow()
	)


def _pagination(total: int, page: int, per_page: int) -> Dict[str, int]:
	"""Create pagination metadata."""
	return {
		"page": page,
		"per_page": per_page,
		"total": total,
		"pages": (total + per_page - 1) // per_page if total > 0 else 0
	}


def _page_items(items: List[Any], page: int, per_page: int) -> List[Any]:
	"""Return one page from an already-filtered list."""
	start = (page - 1) * per_page
	return items[start:start + per_page]


def _time_entry_response(time_entry: Any) -> TimeEntryResponse:
	"""Map a service time-entry model to an API response model."""
	return TimeEntryResponse(
		time_entry_id=time_entry.id,
		employee_id=time_entry.employee_id,
		entry_date=time_entry.entry_date,
		clock_in=time_entry.clock_in,
		clock_out=time_entry.clock_out,
		total_hours=time_entry.total_hours,
		regular_hours=time_entry.regular_hours,
		overtime_hours=time_entry.overtime_hours,
		entry_type=time_entry.entry_type,
		status=time_entry.status,
		fraud_score=time_entry.anomaly_score,
		requires_approval=time_entry.requires_approval,
		verification_confidence=time_entry.verification_confidence,
	)


def _remote_worker_response(remote_worker: Any) -> RemoteWorkerResponse:
	"""Map a service remote-worker model to an API response model."""
	return RemoteWorkerResponse(
		remote_worker_id=remote_worker.id,
		employee_id=remote_worker.employee_id,
		workspace_id=remote_worker.workspace_id,
		work_mode=remote_worker.work_mode,
		current_activity=remote_worker.current_activity,
		productivity_score=remote_worker.overall_productivity_score,
		work_life_balance_score=remote_worker.work_life_balance_score,
		active_hours_today=Decimal(str(remote_worker.home_office_setup.get("active_hours_today", 0))),
		focus_time_blocks=len(remote_worker.focus_time_blocks),
	)


def _ai_agent_response(ai_agent: Any) -> AIAgentResponse:
	"""Map a service AI-agent model to an API response model."""
	return AIAgentResponse(
		ai_agent_id=ai_agent.id,
		agent_name=ai_agent.agent_name,
		agent_type=ai_agent.agent_type,
		capabilities=ai_agent.capabilities,
		health_status=ai_agent.health_status,
		performance_score=ai_agent.overall_performance_score,
		cost_efficiency_score=ai_agent.cost_efficiency_score,
		tasks_completed=ai_agent.tasks_completed,
		total_operational_cost=ai_agent.total_operational_cost,
		uptime_percentage=ai_agent.uptime_percentage,
	)


# Core Time Tracking Endpoints
@router.post("/clock-in", response_model=SuccessResponse, status_code=201)
async def clock_in(
	request: ClockInRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Process employee clock-in with AI validation and fraud detection
	
	- **employee_id**: Employee identifier
	- **device_info**: Device information for validation
	- **location**: Optional GPS coordinates
	- **biometric_data**: Optional biometric authentication data
	- **notes**: Optional notes for the entry
	"""
	try:
		time_entry = await service.clock_in(
			employee_id=request.employee_id,
			tenant_id=current_user["tenant_id"],
			device_info=request.device_info,
			location=request.location,
			biometric_data=request.biometric_data,
			created_by=current_user["user_id"]
		)
		
		response_data = {
			"time_entry_id": time_entry.id,
			"clock_in_time": time_entry.clock_in.isoformat() if time_entry.clock_in else None,
			"status": time_entry.status.value,
			"fraud_score": time_entry.anomaly_score,
			"requires_approval": time_entry.requires_approval,
			"verification_confidence": time_entry.verification_confidence
		}
		
		return create_success_response(response_data, "Clock-in processed successfully")
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Error processing clock-in: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/clock-out", response_model=SuccessResponse, status_code=200)
async def clock_out(
	request: ClockOutRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Process employee clock-out with automatic calculations and validation
	
	- **employee_id**: Employee identifier
	- **device_info**: Device information for validation
	- **location**: Optional GPS coordinates
	- **biometric_data**: Optional biometric authentication data
	- **notes**: Optional notes for the entry
	"""
	try:
		time_entry = await service.clock_out(
			employee_id=request.employee_id,
			tenant_id=current_user["tenant_id"],
			device_info=request.device_info,
			location=request.location,
			biometric_data=request.biometric_data,
			created_by=current_user["user_id"]
		)
		
		response_data = {
			"time_entry_id": time_entry.id,
			"clock_out_time": time_entry.clock_out.isoformat() if time_entry.clock_out else None,
			"total_hours": float(time_entry.total_hours) if time_entry.total_hours else 0,
			"regular_hours": float(time_entry.regular_hours) if time_entry.regular_hours else 0,
			"overtime_hours": float(time_entry.overtime_hours) if time_entry.overtime_hours else 0,
			"status": time_entry.status.value,
			"fraud_score": time_entry.anomaly_score
		}
		
		return create_success_response(response_data, "Clock-out processed successfully")
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Error processing clock-out: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/time-entries", response_model=TimeEntriesListResponse)
async def get_time_entries(
	employee_id: Optional[str] = Query(None, description="Filter by employee ID"),
	start_date: Optional[date] = Query(None, description="Start date filter"),
	end_date: Optional[date] = Query(None, description="End date filter"),
	status: Optional[str] = Query(None, description="Status filter"),
	page: int = Query(1, ge=1, description="Page number"),
	per_page: int = Query(50, ge=1, le=100, description="Items per page"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Get time entries with filtering and pagination
	
	Query parameters:
	- **employee_id**: Filter by specific employee
	- **start_date**: Filter entries from this date
	- **end_date**: Filter entries until this date
	- **status**: Filter by entry status
	- **page**: Page number for pagination
	- **per_page**: Number of items per page
	"""
	try:
		time_entries = await service.list_time_entries(
			tenant_id=current_user["tenant_id"],
			employee_id=employee_id,
			start_date=start_date,
			end_date=end_date,
			status=status,
		)
		total = len(time_entries)
		page_data = [_time_entry_response(entry) for entry in _page_items(time_entries, page, per_page)]
		
		return TimeEntriesListResponse(
			success=True,
			message="Time entries retrieved successfully",
			data=page_data,
			pagination=_pagination(total, page, per_page),
			timestamp=datetime.utcnow()
		)
		
	except Exception as e:
		logger.error(f"Error retrieving time entries: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


# Remote Worker Endpoints
@router.post("/remote-work/start-session", response_model=SuccessResponse, status_code=201)
async def start_remote_work_session(
	request: RemoteWorkSessionRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Start intelligent remote work session with productivity tracking
	
	- **employee_id**: Employee identifier
	- **work_mode**: Work mode classification (remote_only, hybrid, etc.)
	- **workspace_config**: Home office setup and configuration
	- **timezone**: Worker timezone
	- **collaboration_platforms**: Active collaboration platforms
	"""
	try:
		remote_worker = await service.start_remote_work_session(
			employee_id=request.employee_id,
			tenant_id=current_user["tenant_id"],
			workspace_config=request.workspace_config,
			work_mode=request.work_mode,
			created_by=current_user["user_id"]
		)
		
		response_data = {
			"remote_worker_id": remote_worker.id,
			"workspace_id": remote_worker.workspace_id,
			"work_mode": remote_worker.work_mode.value,
			"current_activity": remote_worker.current_activity.value,
			"productivity_score": remote_worker.overall_productivity_score,
			"setup_complete": True
		}
		
		return create_success_response(response_data, "Remote work session started successfully")
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Error starting remote work session: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/remote-work/track-productivity", response_model=ProductivityAnalysisResponse)
async def track_remote_productivity(
	request: ProductivityTrackingRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Track and analyze remote worker productivity with AI insights
	
	- **employee_id**: Employee identifier
	- **activity_data**: Productivity and activity data
	- **metric_type**: Type of productivity measurement
	- **timestamp**: Activity timestamp
	"""
	try:
		productivity_analysis = await service.track_remote_productivity(
			employee_id=request.employee_id,
			tenant_id=current_user["tenant_id"],
			activity_data=request.activity_data,
			metric_type=request.metric_type
		)
		
		return ProductivityAnalysisResponse(
			productivity_score=productivity_analysis["productivity_score"],
			insights=productivity_analysis["insights"],
			recommendations=productivity_analysis["recommendations"],
			burnout_risk=productivity_analysis["burnout_risk"],
			work_life_balance=productivity_analysis["work_life_balance"],
			trend_analysis=productivity_analysis.get("trend_analysis", {})
		)
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Error tracking remote productivity: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/remote-workers", response_model=RemoteWorkersListResponse)
async def get_remote_workers(
	department_id: Optional[str] = Query(None, description="Filter by department"),
	work_mode: Optional[str] = Query(None, description="Filter by work mode"),
	active_only: bool = Query(True, description="Show only active workers"),
	page: int = Query(1, ge=1, description="Page number"),
	per_page: int = Query(50, ge=1, le=100, description="Items per page"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Get remote workers with status and productivity metrics
	
	Query parameters:
	- **department_id**: Filter by department
	- **work_mode**: Filter by work mode
	- **active_only**: Show only currently active workers
	- **page**: Page number for pagination
	- **per_page**: Number of items per page
	"""
	try:
		remote_workers = await service.list_remote_workers(
			tenant_id=current_user["tenant_id"],
			department_id=department_id,
			work_mode=work_mode,
			active_only=active_only,
		)
		total = len(remote_workers)
		page_data = [
			_remote_worker_response(worker)
			for worker in _page_items(remote_workers, page, per_page)
		]
		active_count = len([worker for worker in remote_workers if worker.is_actively_working])
		average_productivity = (
			sum(worker.overall_productivity_score for worker in remote_workers) / total
			if total else 0.0
		)
		average_balance = (
			sum(worker.work_life_balance_score for worker in remote_workers) / total
			if total else 0.8
		)
		
		summary = {
			"total_remote_workers": total,
			"active_count": active_count,
			"average_productivity": round(average_productivity, 4),
			"average_work_life_balance": round(average_balance, 4)
		}
		
		return RemoteWorkersListResponse(
			success=True,
			message="Remote workers retrieved successfully",
			data=page_data,
			summary=summary,
			pagination=_pagination(total, page, per_page),
			timestamp=datetime.utcnow()
		)
		
	except Exception as e:
		logger.error(f"Error retrieving remote workers: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


# AI Agent Endpoints
@router.post("/ai-agents/register", response_model=SuccessResponse, status_code=201)
async def register_ai_agent(
	request: AIAgentRegistrationRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Register AI agent in the workforce management system
	
	- **agent_name**: Human-readable agent name
	- **agent_type**: Type of AI agent (conversational_ai, automation_bot, etc.)
	- **capabilities**: List of agent capabilities and skills
	- **configuration**: Agent configuration including API endpoints and resource limits
	- **version**: Agent version identifier
	- **environment**: Deployment environment
	- **cost_per_hour**: Optional hourly operational cost
	"""
	try:
		ai_agent = await service.register_ai_agent(
			agent_name=request.agent_name,
			agent_type=request.agent_type,
			capabilities=request.capabilities,
			tenant_id=current_user["tenant_id"],
			configuration=request.configuration,
			created_by=current_user["user_id"]
		)
		
		response_data = {
			"ai_agent_id": ai_agent.id,
			"agent_name": ai_agent.agent_name,
			"agent_type": ai_agent.agent_type.value,
			"capabilities": ai_agent.capabilities,
			"health_status": ai_agent.health_status,
			"performance_score": ai_agent.overall_performance_score,
			"registration_complete": True
		}
		
		return create_success_response(response_data, "AI agent registered successfully")
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Error registering AI agent: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai-agents/{agent_id}/track-work", response_model=SuccessResponse)
async def track_ai_agent_work(
	agent_id: str = Path(..., description="AI agent identifier"),
	request: AIAgentWorkTrackingRequest = Body(...),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Track AI agent work completion and resource consumption
	
	- **agent_id**: AI agent identifier from path
	- **task_data**: Task completion information
	- **resource_consumption**: Resource usage data (CPU, GPU, memory, API calls)
	- **timestamp**: Work completion timestamp
	"""
	try:
		work_analysis = await service.track_ai_agent_work(
			agent_id=agent_id,
			tenant_id=current_user["tenant_id"],
			task_data=request.task_data,
			resource_consumption=request.resource_consumption
		)
		
		return create_success_response(work_analysis, "AI agent work tracked successfully")
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Error tracking AI agent work: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/ai-agents", response_model=AIAgentsListResponse)
async def get_ai_agents(
	agent_type: Optional[str] = Query(None, description="Filter by agent type"),
	active_only: bool = Query(True, description="Show only active agents"),
	page: int = Query(1, ge=1, description="Page number"),
	per_page: int = Query(50, ge=1, le=100, description="Items per page"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Get AI agents with performance metrics
	
	Query parameters:
	- **agent_type**: Filter by agent type
	- **active_only**: Show only active agents
	- **page**: Page number for pagination
	- **per_page**: Number of items per page
	"""
	try:
		ai_agents = await service.list_ai_agents(
			tenant_id=current_user["tenant_id"],
			agent_type=agent_type,
			active_only=active_only,
		)
		total = len(ai_agents)
		page_data = [_ai_agent_response(agent) for agent in _page_items(ai_agents, page, per_page)]
		average_performance = (
			sum(agent.overall_performance_score for agent in ai_agents) / total
			if total else 0.0
		)
		total_operational_cost = sum(float(agent.total_operational_cost) for agent in ai_agents)
		cost_efficiency_average = (
			sum(agent.cost_efficiency_score for agent in ai_agents) / total
			if total else 0.0
		)
		
		summary = {
			"total_ai_agents": total,
			"active_count": len([agent for agent in ai_agents if agent.is_active]),
			"average_performance": round(average_performance, 4),
			"total_operational_cost": round(total_operational_cost, 6),
			"cost_efficiency_average": round(cost_efficiency_average, 4)
		}
		
		return AIAgentsListResponse(
			success=True,
			message="AI agents retrieved successfully",
			data=page_data,
			summary=summary,
			pagination=_pagination(total, page, per_page),
			timestamp=datetime.utcnow()
		)
		
	except Exception as e:
		logger.error(f"Error retrieving AI agents: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


# Hybrid Collaboration Endpoints
@router.post("/collaboration/start-session", response_model=SuccessResponse, status_code=201)
async def start_hybrid_collaboration(
	request: HybridCollaborationRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Start hybrid collaboration session between humans and AI agents
	
	- **session_name**: Collaboration session name
	- **project_id**: Associated project identifier
	- **human_participants**: List of human employee IDs
	- **ai_participants**: List of AI agent IDs
	- **session_type**: Type of collaboration session
	- **planned_duration_minutes**: Planned session duration
	- **objectives**: Optional session objectives
	"""
	try:
		collaboration = await service.start_hybrid_collaboration(
			session_name=request.session_name,
			project_id=request.project_id,
			human_participants=request.human_participants,
			ai_participants=request.ai_participants,
			tenant_id=current_user["tenant_id"],
			session_type=request.session_type,
			planned_duration_minutes=request.planned_duration_minutes,
			created_by=current_user["user_id"]
		)
		
		response_data = {
			"collaboration_id": collaboration.id,
			"session_name": collaboration.session_name,
			"human_participants": collaboration.human_participants,
			"ai_participants": collaboration.ai_participants,
			"start_time": collaboration.start_time.isoformat(),
			"session_lead": collaboration.session_lead,
			"session_active": True
		}
		
		return create_success_response(response_data, "Hybrid collaboration session started successfully")
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Error starting hybrid collaboration: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


# Leave Management Endpoints
@router.post("/leave-requests", response_model=SuccessResponse, status_code=201)
async def submit_leave_request(
	request: LeaveRequestSubmission,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Submit intelligent leave request with AI-powered approval prediction
	
	- **employee_id**: Employee requesting leave
	- **leave_type**: Type of leave (vacation, sick, personal, etc.)
	- **start_date**: Leave start date
	- **end_date**: Leave end date
	- **reason**: Optional reason for leave
	- **is_emergency**: Emergency leave flag
	- **supporting_documents**: Optional document attachments
	"""
	try:
		leave_request = await service.process_leave_request(
			employee_id=request.employee_id,
			tenant_id=current_user["tenant_id"],
			leave_type=request.leave_type,
			start_date=request.start_date,
			end_date=request.end_date,
			reason=request.reason,
			is_emergency=request.is_emergency,
			created_by=current_user["user_id"]
		)
		
		response_data = {
			"leave_request_id": leave_request.id,
			"status": leave_request.status.value,
			"approval_probability": leave_request.approval_probability,
			"total_days": float(leave_request.total_days),
			"leave_balance_after": float(leave_request.leave_balance_after) if leave_request.leave_balance_after else None,
			"current_approver": leave_request.current_approver,
			"conflicts_detected": len(leave_request.conflicts_detected),
			"workload_impact": leave_request.workload_impact
		}
		
		return create_success_response(response_data, "Leave request submitted successfully")
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Error submitting leave request: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/leave-requests", response_model=SuccessResponse)
async def get_leave_requests(
	employee_id: Optional[str] = Query(None, description="Filter by employee ID"),
	status: Optional[str] = Query(None, description="Filter by status"),
	leave_type: Optional[str] = Query(None, description="Filter by leave type"),
	start_date: Optional[date] = Query(None, description="Filter from start date"),
	end_date: Optional[date] = Query(None, description="Filter until end date"),
	page: int = Query(1, ge=1, description="Page number"),
	per_page: int = Query(50, ge=1, le=100, description="Items per page"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Get leave requests with filtering and pagination
	
	Query parameters:
	- **employee_id**: Filter by specific employee
	- **status**: Filter by approval status
	- **leave_type**: Filter by type of leave
	- **start_date**: Filter requests from this date
	- **end_date**: Filter requests until this date
	- **page**: Page number for pagination
	- **per_page**: Number of items per page
	"""
	try:
		leave_requests = await service.list_leave_requests(
			tenant_id=current_user["tenant_id"],
			employee_id=employee_id,
			status=status,
			leave_type=leave_type,
			start_date=start_date,
			end_date=end_date,
		)
		total = len(leave_requests)
		page_data = [
			request.model_dump(mode="json")
			for request in _page_items(leave_requests, page, per_page)
		]
		
		return create_success_response({
			"leave_requests": page_data,
			"pagination": _pagination(total, page, per_page)
		}, "Leave requests retrieved successfully")
		
	except Exception as e:
		logger.error(f"Error retrieving leave requests: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


# Schedule Management Endpoints
@router.post("/schedules", response_model=SuccessResponse, status_code=201)
async def create_intelligent_schedule(
	request: ScheduleOptimizationRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Create AI-optimized work schedule with predictive staffing
	
	- **schedule_name**: Schedule name
	- **schedule_patterns**: Weekly schedule patterns
	- **assigned_employees**: Employee IDs to assign
	- **optimization_goals**: Optimization objectives
	- **effective_date**: When schedule becomes effective
	- **skill_requirements**: Required skills and competencies
	"""
	try:
		schedule = await service.create_intelligent_schedule(
			schedule_name=request.schedule_name,
			tenant_id=current_user["tenant_id"],
			schedule_patterns=request.schedule_patterns,
			assigned_employees=request.assigned_employees,
			optimization_goals=request.optimization_goals,
			created_by=current_user["user_id"]
		)
		
		response_data = {
			"schedule_id": schedule.id,
			"schedule_name": schedule.schedule_name,
			"total_weekly_hours": schedule.total_weekly_hours,
			"assigned_employees": len(schedule.assigned_employees),
			"optimization_enabled": schedule.optimization_enabled,
			"status": schedule.status.value,
			"effective_date": schedule.effective_date.isoformat()
		}
		
		return create_success_response(response_data, "Intelligent schedule created successfully")
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Error creating intelligent schedule: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/schedules", response_model=SuccessResponse)
async def get_schedules(
	employee_id: Optional[str] = Query(None, description="Filter by employee ID"),
	department_id: Optional[str] = Query(None, description="Filter by department"),
	status: Optional[str] = Query(None, description="Filter by status"),
	effective_date: Optional[date] = Query(None, description="Filter by effective date"),
	page: int = Query(1, ge=1, description="Page number"),
	per_page: int = Query(50, ge=1, le=100, description="Items per page"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Get work schedules with filtering and pagination
	"""
	try:
		schedules = await service.list_schedules(
			tenant_id=current_user["tenant_id"],
			employee_id=employee_id,
			department_id=department_id,
			status=status,
			effective_date=effective_date,
		)
		total = len(schedules)
		page_data = [
			schedule.model_dump(mode="json")
			for schedule in _page_items(schedules, page, per_page)
		]
		
		return create_success_response({
			"schedules": page_data,
			"pagination": _pagination(total, page, per_page)
		}, "Schedules retrieved successfully")
		
	except Exception as e:
		logger.error(f"Error retrieving schedules: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


# Fraud Detection and Compliance Endpoints
@router.post("/fraud-detection/analyze", response_model=SuccessResponse)
async def detect_time_fraud(
	employee_ids: Optional[List[str]] = Body(None, description="Specific employees to analyze"),
	start_date: Optional[date] = Body(None, description="Start date for analysis"),
	end_date: Optional[date] = Body(None, description="End date for analysis"),
	background_tasks: BackgroundTasks = BackgroundTasks(),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Run advanced AI-powered fraud detection across workforce
	
	- **employee_ids**: Optional list of specific employees to analyze
	- **start_date**: Optional start date for analysis period
	- **end_date**: Optional end date for analysis period
	
	Returns fraud detection results including:
	- Detected fraud cases
	- Severity levels
	- Financial impact estimates
	- Prevention recommendations
	"""
	try:
		date_range = None
		if start_date and end_date:
			date_range = {
				"start_date": datetime.combine(start_date, datetime.min.time()),
				"end_date": datetime.combine(end_date, datetime.max.time())
			}
		
		fraud_detections = await service.detect_time_fraud(
			tenant_id=current_user["tenant_id"],
			employee_ids=employee_ids,
			date_range=date_range
		)
		
		response_data = {
			"analysis_id": f"fraud_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
			"fraud_cases_detected": len(fraud_detections),
			"fraud_cases": [
				{
					"detection_id": detection.id,
					"employee_id": detection.employee_id,
					"fraud_types": [ft.value for ft in detection.fraud_types],
					"severity_level": detection.severity_level,
					"confidence_score": detection.confidence_score,
					"financial_impact": float(detection.financial_impact) if detection.financial_impact else None,
					"requires_immediate_action": detection.requires_immediate_action
				}
				for detection in fraud_detections
			],
			"summary": {
				"high_severity_cases": len([d for d in fraud_detections if d.severity_level in ["HIGH", "CRITICAL"]]),
				"total_estimated_impact": sum(float(d.financial_impact) for d in fraud_detections if d.financial_impact),
				"immediate_action_required": len([d for d in fraud_detections if d.requires_immediate_action])
			}
		}
		
		return create_success_response(response_data, f"Fraud detection completed. Found {len(fraud_detections)} cases")
		
	except Exception as e:
		logger.error(f"Error running fraud detection: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/compliance/enforce", response_model=SuccessResponse)
async def enforce_compliance_rules(
	rule_types: Optional[List[str]] = Body(None, description="Specific rule types to enforce"),
	background_tasks: BackgroundTasks = BackgroundTasks(),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Enforce compliance rules with automated violation detection
	
	- **rule_types**: Optional list of specific rule types to enforce
	
	Returns compliance enforcement results including:
	- Violations detected
	- Automatic corrections applied
	- Compliance score
	- Recommendations
	"""
	try:
		enforcement_results = await service.enforce_compliance_rules(
			tenant_id=current_user["tenant_id"],
			rule_types=rule_types
		)
		
		return create_success_response(enforcement_results, "Compliance rules enforced successfully")
		
	except Exception as e:
		logger.error(f"Error enforcing compliance rules: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


# Bulk Operations Endpoints
@router.post("/time-entries/bulk-update", response_model=SuccessResponse)
async def bulk_update_time_entries(
	request: BulkTimeEntryUpdate,
	background_tasks: BackgroundTasks = BackgroundTasks(),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Bulk update multiple time entries
	
	- **time_entry_ids**: List of time entry IDs to update
	- **updates**: Updates to apply to all entries
	- **update_reason**: Reason for bulk update
	"""
	try:
		result = await service.bulk_update_time_entries(
			tenant_id=current_user["tenant_id"],
			time_entry_ids=request.time_entry_ids,
			updates=request.updates,
			updated_by=current_user["user_id"],
		)
		response_data = {
			"updated_count": len(result["updated_ids"]),
			"updated_ids": result["updated_ids"],
			"failed_updates": result["failed_updates"],
			"batch_id": f"bulk_update_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
		}
		
		return create_success_response(response_data, f"Bulk update completed for {len(result['updated_ids'])} entries")
		
	except Exception as e:
		logger.error(f"Error in bulk update: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/approvals/bulk-approve", response_model=SuccessResponse)
async def bulk_approve_entries(
	request: BulkApprovalRequest,
	background_tasks: BackgroundTasks = BackgroundTasks(),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Bulk approve multiple time entries or leave requests
	
	- **time_entry_ids**: List of entry IDs to approve or reject
	- **entry_type**: Type of entries (time_entry, leave_request)
	- **comments**: Optional approval notes
	"""
	try:
		result = await service.bulk_approve_entries(
			tenant_id=current_user["tenant_id"],
			entry_ids=request.time_entry_ids,
			entry_type=request.entry_type,
			approved_by=current_user["user_id"],
			action=request.action,
			approval_notes=request.comments,
		)
		response_data = {
			"processed_count": len(result["processed_ids"]),
			"processed_ids": result["processed_ids"],
			"action": result["action"],
			"failed_approvals": result["failed_approvals"],
			"batch_id": f"bulk_approval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
		}
		
		return create_success_response(response_data, f"Bulk approval completed for {len(result['processed_ids'])} entries")
		
	except Exception as e:
		logger.error(f"Error in bulk approval: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


# Analytics and Reporting Endpoints
@router.post("/analytics/workforce-predictions", response_model=WorkforcePredictionsResponse)
async def generate_workforce_predictions(
	prediction_period_days: int = Body(30, ge=1, le=365, description="Prediction period in days"),
	departments: Optional[List[str]] = Body(None, description="Department IDs to analyze"),
	background_tasks: BackgroundTasks = BackgroundTasks(),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Generate AI-powered workforce predictions and optimization recommendations
	
	- **prediction_period_days**: Number of days to predict (1-365)
	- **departments**: Optional list of department IDs to include in analysis
	
	Returns comprehensive workforce analytics including:
	- Staffing requirement predictions
	- Cost optimization recommendations
	- Projected savings
	- Actionable insights
	- Strategic recommendations
	"""
	try:
		analytics = await service.generate_workforce_predictions(
			tenant_id=current_user["tenant_id"],
			prediction_period_days=prediction_period_days,
			departments=departments
		)
		
		return WorkforcePredictionsResponse(
			analytics_id=analytics.id,
			analysis_type=analytics.analysis_type,
			prediction_period_days=prediction_period_days,
			model_confidence=analytics.model_confidence,
			staffing_predictions=analytics.staffing_predictions,
			cost_optimization=analytics.cost_optimization,
			projected_savings=analytics.projected_savings,
			actionable_insights=analytics.actionable_insights,
			strategic_recommendations=analytics.strategic_recommendations
		)
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Error generating workforce predictions: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/dashboard", response_model=SuccessResponse)
async def get_analytics_dashboard(
	date_range_days: int = Query(30, ge=1, le=365, description="Date range for analytics"),
	department_id: Optional[str] = Query(None, description="Filter by department"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: TimeAttendanceService = Depends(get_service)
):
	"""
	Get comprehensive analytics dashboard data
	
	Returns real-time analytics including:
	- Attendance rates
	- Productivity metrics
	- Cost analysis
	- Fraud indicators
	- Compliance status
	"""
	try:
		dashboard_data = await service.get_analytics_dashboard(
			tenant_id=current_user["tenant_id"],
			date_range_days=date_range_days,
			department_id=department_id,
		)
		
		return create_success_response(dashboard_data, "Analytics dashboard data retrieved successfully")
		
	except Exception as e:
		logger.error(f"Error retrieving analytics dashboard: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


# Configuration and Health Endpoints
@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
	"""
	Health check endpoint for service monitoring
	
	Returns service health status, version, and feature availability.
	Used by load balancers and monitoring systems.
	"""
	try:
		config = get_config()
		
		return HealthCheckResponse(
			status="healthy",
			timestamp=datetime.utcnow(),
			version="1.0.0",
			environment=config.environment.value,
			features={
				"ai_fraud_detection": config.is_feature_enabled("ai_fraud_detection"),
				"biometric_authentication": config.is_feature_enabled("biometric_authentication"),
				"remote_work_tracking": config.is_feature_enabled("remote_work_tracking"),
				"ai_agent_management": config.is_feature_enabled("ai_agent_management"),
				"hybrid_collaboration": config.is_feature_enabled("hybrid_collaboration")
			}
		)
		
	except Exception as e:
		logger.error(f"Health check failed: {str(e)}")
		return HealthCheckResponse(
			status="unhealthy",
			timestamp=datetime.utcnow(),
			version="1.0.0",
			environment="unknown",
			features={}
		)


@router.get("/config", response_model=ConfigurationResponse)
async def get_configuration(
	current_user: Dict[str, Any] = Depends(get_current_user)
):
	"""
	Get sanitized configuration for client applications
	
	Returns configuration settings that are safe to expose to client applications.
	Excludes sensitive information like API keys and database credentials.
	"""
	try:
		config = get_config()
		
		return ConfigurationResponse(
			environment=config.environment.value,
			tracking_mode=config.tracking_mode.value,
			features=config.features,
			performance={
				"target_response_time_ms": config.performance.target_response_time_ms,
				"target_availability_percent": config.performance.target_availability_percent
			},
			compliance={
				"flsa_enabled": config.compliance.flsa_compliance_enabled,
				"gdpr_enabled": config.compliance.gdpr_compliance_enabled,
				"overtime_threshold": config.compliance.overtime_threshold_hours
			}
		)
		
	except Exception as e:
		logger.error(f"Error retrieving configuration: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


# Exception handlers
async def validation_exception_handler(request, exc: ValidationError):
	"""Handle Pydantic validation errors"""
	return JSONResponse(
		status_code=422,
		content=create_error_response(
			"Validation failed",
			422,
			exc.errors()
		).dict()
	)


async def value_error_handler(request, exc: ValueError):
	"""Handle value errors"""
	return JSONResponse(
		status_code=400,
		content=create_error_response(str(exc), 400).dict()
	)


# Create FastAPI app
def create_app() -> FastAPI:
	"""Create and configure FastAPI application"""
	app = FastAPI(
		title="APG Time & Attendance API",
		description="Revolutionary workforce management API supporting traditional, remote, and AI workers",
		version="1.0.0",
		docs_url="/api/human_capital_management/time_attendance/docs",
		redoc_url="/api/human_capital_management/time_attendance/redoc"
	)
	
	# Add CORS middleware
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],  # Configure appropriately for production
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)
	
	# Include router
	app.include_router(router)
	app.add_exception_handler(ValidationError, validation_exception_handler)
	app.add_exception_handler(ValueError, value_error_handler)
	
	return app


# Export router and app factory
__all__ = ["router", "create_app"]
