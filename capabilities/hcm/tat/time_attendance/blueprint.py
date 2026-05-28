"""
Time & Attendance Capability Blueprint

Flask-AppBuilder blueprint providing comprehensive API endpoints for the revolutionary
APG Time & Attendance capability with full support for traditional, remote, and AI workers.

Copyright 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from flask import Blueprint, request, jsonify, current_app
from flask_appbuilder.security.decorators import protect
from marshmallow import Schema, fields, validate, post_load
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import asyncio
import logging
from functools import wraps

from .service import TimeAttendanceService
from .models import (
	TAEmployee, TATimeEntry, TASchedule, TALeaveRequest, TAFraudDetection,
	TABiometricAuthentication, TAPredictiveAnalytics, TAComplianceRule,
	TARemoteWorker, TAAIAgent, TAHybridCollaboration,
	TimeEntryStatus, TimeEntryType, WorkforceType, WorkMode, AIAgentType,
	ProductivityMetric, RemoteWorkStatus
)
from .config import get_config

# Initialize logger
logger = logging.getLogger(__name__)


def protect():
	"""Function-route compatible access wrapper for this blueprint."""
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			return func(*args, **kwargs)
		return wrapper
	return decorator

# Create blueprint
time_attendance_bp = Blueprint(
	"time_attendance",
	__name__,
	url_prefix="/api/human_capital_management/time_attendance"
)


# Marshmallow Schemas for API Validation
class TimeEntrySchema(Schema):
	"""Schema for time entry validation"""
	employee_id = fields.Str(required=True, validate=validate.Length(min=1))
	entry_date = fields.Date(required=True)
	clock_in = fields.DateTime(allow_none=True)
	clock_out = fields.DateTime(allow_none=True)
	entry_type = fields.Str(validate=validate.OneOf([e.value for e in TimeEntryType]))
	location = fields.Dict(allow_none=True)
	device_info = fields.Dict(load_default=dict)
	notes = fields.Str(allow_none=True, validate=validate.Length(max=1000))
	

class RemoteWorkerSchema(Schema):
	"""Schema for remote worker validation"""
	employee_id = fields.Str(required=True, validate=validate.Length(min=1))
	work_mode = fields.Str(required=True, validate=validate.OneOf([e.value for e in WorkMode]))
	workspace_config = fields.Dict(required=True)
	timezone = fields.Str(load_default="UTC")
	

class AIAgentSchema(Schema):
	"""Schema for AI agent validation"""
	agent_name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
	agent_type = fields.Str(required=True, validate=validate.OneOf([e.value for e in AIAgentType]))
	capabilities = fields.List(fields.Str(), required=True)
	configuration = fields.Dict(required=True)
	

class HybridCollaborationSchema(Schema):
	"""Schema for hybrid collaboration validation"""
	session_name = fields.Str(required=True, validate=validate.Length(min=1, max=200))
	project_id = fields.Str(required=True)
	human_participants = fields.List(fields.Str(), required=True)
	ai_participants = fields.List(fields.Str(), required=True)
	session_type = fields.Str(load_default="collaborative_work")
	planned_duration_minutes = fields.Int(load_default=60, validate=validate.Range(min=1))


# API Response Helpers
def success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
	"""Standard success response format"""
	return {
		"success": True,
		"message": message,
		"data": data,
		"timestamp": datetime.utcnow().isoformat()
	}


def error_response(message: str, error_code: int = 400, details: Any = None) -> Dict[str, Any]:
	"""Standard error response format"""
	return {
		"success": False,
		"message": message,
		"error_code": error_code,
		"details": details,
		"timestamp": datetime.utcnow().isoformat()
	}


def run_async(coro):
	"""Helper to run async functions in Flask context"""
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	
	return loop.run_until_complete(coro)


def _current_tenant_id() -> str:
	"""Resolve tenant from Flask security context."""
	user = getattr(getattr(current_app, "sm", None), "user", None)
	return str(getattr(user, "tenant_id", "default"))


def _current_user_id() -> str:
	"""Resolve user from Flask security context."""
	user = getattr(getattr(current_app, "sm", None), "user", None)
	return str(getattr(user, "id", "system"))


def _paginate(items: List[Any], page: int, per_page: int) -> Dict[str, Any]:
	"""Create a stable paginated payload."""
	total = len(items)
	start = (page - 1) * per_page
	return {
		"items": items[start:start + per_page],
		"total": total,
		"page": page,
		"per_page": per_page,
		"pages": (total + per_page - 1) // per_page if total else 0,
	}


def _time_entry_payload(time_entry: TATimeEntry) -> Dict[str, Any]:
	"""Serialize a time entry for Flask blueprint responses."""
	return {
		"time_entry_id": time_entry.id,
		"employee_id": time_entry.employee_id,
		"entry_date": time_entry.entry_date.isoformat(),
		"clock_in": time_entry.clock_in.isoformat() if time_entry.clock_in else None,
		"clock_out": time_entry.clock_out.isoformat() if time_entry.clock_out else None,
		"total_hours": float(time_entry.total_hours) if time_entry.total_hours else 0.0,
		"regular_hours": float(time_entry.regular_hours) if time_entry.regular_hours else 0.0,
		"overtime_hours": float(time_entry.overtime_hours) if time_entry.overtime_hours else 0.0,
		"entry_type": time_entry.entry_type.value,
		"status": time_entry.status.value,
		"fraud_score": time_entry.anomaly_score,
		"requires_approval": time_entry.requires_approval,
	}


def _remote_worker_payload(remote_worker: TARemoteWorker) -> Dict[str, Any]:
	"""Serialize a remote worker for Flask blueprint responses."""
	return {
		"remote_worker_id": remote_worker.id,
		"employee_id": remote_worker.employee_id,
		"workspace_id": remote_worker.workspace_id,
		"work_mode": remote_worker.work_mode.value,
		"current_activity": remote_worker.current_activity.value,
		"productivity_score": remote_worker.overall_productivity_score,
		"work_life_balance_score": remote_worker.work_life_balance_score,
		"active": remote_worker.is_actively_working,
	}


def _ai_agent_payload(ai_agent: TAAIAgent) -> Dict[str, Any]:
	"""Serialize an AI agent for Flask blueprint responses."""
	return {
		"ai_agent_id": ai_agent.id,
		"agent_name": ai_agent.agent_name,
		"agent_type": ai_agent.agent_type.value,
		"capabilities": ai_agent.capabilities,
		"health_status": ai_agent.health_status,
		"performance_score": ai_agent.overall_performance_score,
		"cost_efficiency_score": ai_agent.cost_efficiency_score,
		"tasks_completed": ai_agent.tasks_completed,
		"total_operational_cost": float(ai_agent.total_operational_cost),
		"active": ai_agent.is_active,
	}


# Core Time Tracking API Endpoints
@time_attendance_bp.route("/clock-in", methods=["POST"])
@protect()
def clock_in():
	"""
	Process employee clock-in with AI validation
	
	POST /api/human_capital_management/time_attendance/clock-in
	"""
	try:
		# Validate request data
		schema = TimeEntrySchema()
		data = schema.load(request.json)
		
		# Get tenant and user info from security context
		tenant_id = _current_tenant_id()
		created_by = _current_user_id()
		
		# Initialize service
		service = TimeAttendanceService()
		
		# Process clock-in
		time_entry = run_async(service.clock_in(
			employee_id=data["employee_id"],
			tenant_id=tenant_id,
			device_info=data.get("device_info", {}),
			location=data.get("location"),
			biometric_data=request.json.get("biometric_data"),
			created_by=created_by
		))
		
		logger.info(f"Clock-in processed for employee {data['employee_id']}")
		
		return jsonify(success_response({
			"time_entry_id": time_entry.id,
			"clock_in_time": time_entry.clock_in.isoformat() if time_entry.clock_in else None,
			"status": time_entry.status.value,
			"fraud_score": time_entry.anomaly_score,
			"requires_approval": time_entry.requires_approval
		}, "Clock-in processed successfully"))
		
	except Exception as e:
		logger.error(f"Error processing clock-in: {str(e)}")
		return jsonify(error_response(f"Clock-in failed: {str(e)}")), 400


@time_attendance_bp.route("/clock-out", methods=["POST"])
@protect()
def clock_out():
	"""
	Process employee clock-out with automatic calculations
	
	POST /api/human_capital_management/time_attendance/clock-out
	"""
	try:
		# Validate request data
		employee_id = request.json.get("employee_id")
		if not employee_id:
			return jsonify(error_response("employee_id is required")), 400
		
		# Get tenant and user info
		tenant_id = _current_tenant_id()
		created_by = _current_user_id()
		
		# Initialize service
		service = TimeAttendanceService()
		
		# Process clock-out
		time_entry = run_async(service.clock_out(
			employee_id=employee_id,
			tenant_id=tenant_id,
			device_info=request.json.get("device_info", {}),
			location=request.json.get("location"),
			biometric_data=request.json.get("biometric_data"),
			created_by=created_by
		))
		
		logger.info(f"Clock-out processed for employee {employee_id}")
		
		return jsonify(success_response({
			"time_entry_id": time_entry.id,
			"clock_out_time": time_entry.clock_out.isoformat() if time_entry.clock_out else None,
			"total_hours": float(time_entry.total_hours) if time_entry.total_hours else 0,
			"regular_hours": float(time_entry.regular_hours) if time_entry.regular_hours else 0,
			"overtime_hours": float(time_entry.overtime_hours) if time_entry.overtime_hours else 0,
			"status": time_entry.status.value,
			"fraud_score": time_entry.anomaly_score
		}, "Clock-out processed successfully"))
		
	except Exception as e:
		logger.error(f"Error processing clock-out: {str(e)}")
		return jsonify(error_response(f"Clock-out failed: {str(e)}")), 400


# Remote Worker API Endpoints
@time_attendance_bp.route("/remote-work/start-session", methods=["POST"])
@protect()
def start_remote_work_session():
	"""
	Start intelligent remote work session
	
	POST /api/human_capital_management/time_attendance/remote-work/start-session
	"""
	try:
		# Validate request data
		schema = RemoteWorkerSchema()
		data = schema.load(request.json)
		
		# Get tenant and user info
		tenant_id = _current_tenant_id()
		created_by = _current_user_id()
		
		# Initialize service
		service = TimeAttendanceService()
		
		# Start remote work session
		remote_worker = run_async(service.start_remote_work_session(
			employee_id=data["employee_id"],
			tenant_id=tenant_id,
			workspace_config=data["workspace_config"],
			work_mode=WorkMode(data["work_mode"]),
			created_by=created_by
		))
		
		logger.info(f"Remote work session started for employee {data['employee_id']}")
		
		return jsonify(success_response({
			"remote_worker_id": remote_worker.id,
			"workspace_id": remote_worker.workspace_id,
			"work_mode": remote_worker.work_mode.value,
			"current_activity": remote_worker.current_activity.value,
			"productivity_score": remote_worker.overall_productivity_score
		}, "Remote work session started successfully"))
		
	except Exception as e:
		logger.error(f"Error starting remote work session: {str(e)}")
		return jsonify(error_response(f"Failed to start remote work session: {str(e)}")), 400


@time_attendance_bp.route("/remote-work/track-productivity", methods=["POST"])
@protect()
def track_remote_productivity():
	"""
	Track remote worker productivity with AI insights
	
	POST /api/human_capital_management/time_attendance/remote-work/track-productivity
	"""
	try:
		# Validate required fields
		employee_id = request.json.get("employee_id")
		activity_data = request.json.get("activity_data", {})
		metric_type = request.json.get("metric_type", "task_completion")
		
		if not employee_id:
			return jsonify(error_response("employee_id is required")), 400
		
		# Get tenant info
		tenant_id = _current_tenant_id()
		
		# Initialize service
		service = TimeAttendanceService()
		
		# Track productivity
		productivity_analysis = run_async(service.track_remote_productivity(
			employee_id=employee_id,
			tenant_id=tenant_id,
			activity_data=activity_data,
			metric_type=ProductivityMetric(metric_type)
		))
		
		logger.info(f"Remote productivity tracked for employee {employee_id}")
		
		return jsonify(success_response(productivity_analysis, "Productivity tracked successfully"))
		
	except Exception as e:
		logger.error(f"Error tracking remote productivity: {str(e)}")
		return jsonify(error_response(f"Failed to track productivity: {str(e)}")), 400


# AI Agent API Endpoints
@time_attendance_bp.route("/ai-agents/register", methods=["POST"])
@protect()
def register_ai_agent():
	"""
	Register AI agent in workforce management system
	
	POST /api/human_capital_management/time_attendance/ai-agents/register
	"""
	try:
		# Validate request data
		schema = AIAgentSchema()
		data = schema.load(request.json)
		
		# Get tenant and user info
		tenant_id = _current_tenant_id()
		created_by = _current_user_id()
		
		# Initialize service
		service = TimeAttendanceService()
		
		# Register AI agent
		ai_agent = run_async(service.register_ai_agent(
			agent_name=data["agent_name"],
			agent_type=AIAgentType(data["agent_type"]),
			capabilities=data["capabilities"],
			tenant_id=tenant_id,
			configuration=data["configuration"],
			created_by=created_by
		))
		
		logger.info(f"AI agent registered: {data['agent_name']}")
		
		return jsonify(success_response({
			"ai_agent_id": ai_agent.id,
			"agent_name": ai_agent.agent_name,
			"agent_type": ai_agent.agent_type.value,
			"capabilities": ai_agent.capabilities,
			"health_status": ai_agent.health_status,
			"performance_score": ai_agent.overall_performance_score
		}, "AI agent registered successfully"))
		
	except Exception as e:
		logger.error(f"Error registering AI agent: {str(e)}")
		return jsonify(error_response(f"Failed to register AI agent: {str(e)}")), 400


@time_attendance_bp.route("/ai-agents/<agent_id>/track-work", methods=["POST"])
@protect()
def track_ai_agent_work(agent_id: str):
	"""
	Track AI agent work completion and resource consumption
	
	POST /api/human_capital_management/time_attendance/ai-agents/{agent_id}/track-work
	"""
	try:
		# Validate request data
		task_data = request.json.get("task_data", {})
		resource_consumption = request.json.get("resource_consumption", {})
		
		# Get tenant info
		tenant_id = _current_tenant_id()
		
		# Initialize service
		service = TimeAttendanceService()
		
		# Track AI agent work
		work_analysis = run_async(service.track_ai_agent_work(
			agent_id=agent_id,
			tenant_id=tenant_id,
			task_data=task_data,
			resource_consumption=resource_consumption
		))
		
		logger.info(f"Work tracked for AI agent {agent_id}")
		
		return jsonify(success_response(work_analysis, "AI agent work tracked successfully"))
		
	except Exception as e:
		logger.error(f"Error tracking AI agent work: {str(e)}")
		return jsonify(error_response(f"Failed to track AI agent work: {str(e)}")), 400


# Hybrid Collaboration API Endpoints
@time_attendance_bp.route("/collaboration/start-session", methods=["POST"])
@protect()
def start_hybrid_collaboration():
	"""
	Start hybrid collaboration session between humans and AI agents
	
	POST /api/human_capital_management/time_attendance/collaboration/start-session
	"""
	try:
		# Validate request data
		schema = HybridCollaborationSchema()
		data = schema.load(request.json)
		
		# Get tenant and user info
		tenant_id = _current_tenant_id()
		created_by = _current_user_id()
		
		# Initialize service
		service = TimeAttendanceService()
		
		# Start collaboration session
		collaboration = run_async(service.start_hybrid_collaboration(
			session_name=data["session_name"],
			project_id=data["project_id"],
			human_participants=data["human_participants"],
			ai_participants=data["ai_participants"],
			tenant_id=tenant_id,
			session_type=data["session_type"],
			planned_duration_minutes=data["planned_duration_minutes"],
			created_by=created_by
		))
		
		logger.info(f"Hybrid collaboration session started: {data['session_name']}")
		
		return jsonify(success_response({
			"collaboration_id": collaboration.id,
			"session_name": collaboration.session_name,
			"human_participants": collaboration.human_participants,
			"ai_participants": collaboration.ai_participants,
			"start_time": collaboration.start_time.isoformat(),
			"session_lead": collaboration.session_lead
		}, "Hybrid collaboration session started successfully"))
		
	except Exception as e:
		logger.error(f"Error starting hybrid collaboration: {str(e)}")
		return jsonify(error_response(f"Failed to start collaboration: {str(e)}")), 400


# Analytics and Reporting API Endpoints
@time_attendance_bp.route("/analytics/workforce-predictions", methods=["POST"])
@protect()
def generate_workforce_predictions():
	"""
	Generate AI-powered workforce predictions
	
	POST /api/human_capital_management/time_attendance/analytics/workforce-predictions
	"""
	try:
		# Get request parameters
		prediction_period_days = request.json.get("prediction_period_days", 30)
		departments = request.json.get("departments")
		
		# Get tenant info
		tenant_id = _current_tenant_id()
		
		# Initialize service
		service = TimeAttendanceService()
		
		# Generate predictions
		analytics = run_async(service.generate_workforce_predictions(
			tenant_id=tenant_id,
			prediction_period_days=prediction_period_days,
			departments=departments
		))
		
		logger.info(f"Workforce predictions generated for tenant {tenant_id}")
		
		return jsonify(success_response({
			"analytics_id": analytics.id,
			"analysis_type": analytics.analysis_type,
			"model_confidence": analytics.model_confidence,
			"projected_savings": float(analytics.projected_savings) if analytics.projected_savings else None,
			"insights_count": len(analytics.actionable_insights),
			"recommendations_count": len(analytics.strategic_recommendations)
		}, "Workforce predictions generated successfully"))
		
	except Exception as e:
		logger.error(f"Error generating workforce predictions: {str(e)}")
		return jsonify(error_response(f"Failed to generate predictions: {str(e)}")), 400


# Data Query API Endpoints
@time_attendance_bp.route("/time-entries", methods=["GET"])
@protect()
def get_time_entries():
	"""
	Get time entries with filtering and pagination
	
	GET /api/human_capital_management/time_attendance/time-entries
	"""
	try:
		# Get query parameters
		employee_id = request.args.get("employee_id")
		start_date = request.args.get("start_date")
		end_date = request.args.get("end_date")
		status = request.args.get("status")
		page = int(request.args.get("page", 1))
		per_page = int(request.args.get("per_page", 50))
		
		tenant_id = _current_tenant_id()
		service = TimeAttendanceService()
		start = date.fromisoformat(start_date) if start_date else None
		end = date.fromisoformat(end_date) if end_date else None
		entries = run_async(service.list_time_entries(
			tenant_id=tenant_id,
			employee_id=employee_id,
			start_date=start,
			end_date=end,
			status=status,
		))
		payload = _paginate([_time_entry_payload(entry) for entry in entries], page, per_page)
		
		return jsonify(success_response({
			"time_entries": payload["items"],
			"total": payload["total"],
			"page": payload["page"],
			"per_page": payload["per_page"],
			"pages": payload["pages"]
		}, "Time entries retrieved successfully"))
		
	except Exception as e:
		logger.error(f"Error retrieving time entries: {str(e)}")
		return jsonify(error_response(f"Failed to retrieve time entries: {str(e)}")), 400


@time_attendance_bp.route("/remote-workers", methods=["GET"])
@protect()
def get_remote_workers():
	"""
	Get remote workers with status and productivity metrics
	
	GET /api/human_capital_management/time_attendance/remote-workers
	"""
	try:
		# Get query parameters
		department_id = request.args.get("department_id")
		work_mode = request.args.get("work_mode")
		active_only = request.args.get("active_only", "true").lower() == "true"
		
		tenant_id = _current_tenant_id()
		service = TimeAttendanceService()
		workers = run_async(service.list_remote_workers(
			tenant_id=tenant_id,
			department_id=department_id,
			work_mode=work_mode,
			active_only=active_only,
		))
		payload = [_remote_worker_payload(worker) for worker in workers]
		average_productivity = (
			sum(worker["productivity_score"] for worker in payload) / len(payload)
			if payload else 0.0
		)
		
		return jsonify(success_response({
			"remote_workers": payload,
			"total": len(payload),
			"active_count": len([worker for worker in payload if worker["active"]]),
			"average_productivity": round(average_productivity, 4)
		}, "Remote workers retrieved successfully"))
		
	except Exception as e:
		logger.error(f"Error retrieving remote workers: {str(e)}")
		return jsonify(error_response(f"Failed to retrieve remote workers: {str(e)}")), 400


@time_attendance_bp.route("/ai-agents", methods=["GET"])
@protect()
def get_ai_agents():
	"""
	Get AI agents with performance metrics
	
	GET /api/human_capital_management/time_attendance/ai-agents
	"""
	try:
		# Get query parameters
		agent_type = request.args.get("agent_type")
		active_only = request.args.get("active_only", "true").lower() == "true"
		
		tenant_id = _current_tenant_id()
		service = TimeAttendanceService()
		agents = run_async(service.list_ai_agents(
			tenant_id=tenant_id,
			agent_type=agent_type,
			active_only=active_only,
		))
		payload = [_ai_agent_payload(agent) for agent in agents]
		average_performance = (
			sum(agent["performance_score"] for agent in payload) / len(payload)
			if payload else 0.0
		)
		total_cost = sum(agent["total_operational_cost"] for agent in payload)
		
		return jsonify(success_response({
			"ai_agents": payload,
			"total": len(payload),
			"active_count": len([agent for agent in payload if agent["active"]]),
			"average_performance": round(average_performance, 4),
			"total_cost": round(total_cost, 6)
		}, "AI agents retrieved successfully"))
		
	except Exception as e:
		logger.error(f"Error retrieving AI agents: {str(e)}")
		return jsonify(error_response(f"Failed to retrieve AI agents: {str(e)}")), 400


# Health Check and Status Endpoints
@time_attendance_bp.route("/health", methods=["GET"])
def health_check():
	"""
	Health check endpoint for monitoring
	
	GET /api/human_capital_management/time_attendance/health
	"""
	try:
		config = get_config()
		
		return jsonify({
			"status": "healthy",
			"timestamp": datetime.utcnow().isoformat(),
			"version": "1.0.0",
			"environment": config.environment.value,
			"features": {
				"ai_fraud_detection": config.is_feature_enabled("ai_fraud_detection"),
				"biometric_authentication": config.is_feature_enabled("biometric_authentication"),
				"remote_work_tracking": config.is_feature_enabled("remote_work_tracking"),
				"ai_agent_management": config.is_feature_enabled("ai_agent_management"),
				"hybrid_collaboration": config.is_feature_enabled("hybrid_collaboration")
			}
		})
		
	except Exception as e:
		logger.error(f"Health check failed: {str(e)}")
		return jsonify({
			"status": "unhealthy",
			"timestamp": datetime.utcnow().isoformat(),
			"error": str(e)
		}), 500


@time_attendance_bp.route("/config", methods=["GET"])
@protect()
def get_configuration():
	"""
	Get sanitized configuration for client applications
	
	GET /api/human_capital_management/time_attendance/config
	"""
	try:
		config = get_config()
		
		# Return sanitized configuration (no secrets)
		return jsonify(success_response({
			"environment": config.environment.value,
			"tracking_mode": config.tracking_mode.value,
			"features": config.features,
			"performance": {
				"target_response_time_ms": config.performance.target_response_time_ms,
				"target_availability_percent": config.performance.target_availability_percent
			},
			"compliance": {
				"flsa_enabled": config.compliance.flsa_compliance_enabled,
				"gdpr_enabled": config.compliance.gdpr_compliance_enabled,
				"overtime_threshold": config.compliance.overtime_threshold_hours
			}
		}, "Configuration retrieved successfully"))
		
	except Exception as e:
		logger.error(f"Error retrieving configuration: {str(e)}")
		return jsonify(error_response(f"Failed to retrieve configuration: {str(e)}")), 400


# Error Handlers
@time_attendance_bp.errorhandler(400)
def bad_request(error):
	"""Handle bad request errors"""
	return jsonify(error_response("Bad request", 400, str(error))), 400


@time_attendance_bp.errorhandler(401)
def unauthorized(error):
	"""Handle unauthorized errors"""
	return jsonify(error_response("Unauthorized", 401, str(error))), 401


@time_attendance_bp.errorhandler(403)
def forbidden(error):
	"""Handle forbidden errors"""
	return jsonify(error_response("Forbidden", 403, str(error))), 403


@time_attendance_bp.errorhandler(404)
def not_found(error):
	"""Handle not found errors"""
	return jsonify(error_response("Not found", 404, str(error))), 404


@time_attendance_bp.errorhandler(500)
def internal_error(error):
	"""Handle internal server errors"""
	logger.error(f"Internal server error: {str(error)}")
	return jsonify(error_response("Internal server error", 500)), 500


# Blueprint registration function
def register_blueprint(app):
	"""Register the time attendance blueprint with the Flask app"""
	app.register_blueprint(time_attendance_bp)
	logger.info("Time & Attendance blueprint registered successfully")


# Export blueprint and registration function
__all__ = ["time_attendance_bp", "register_blueprint"]
