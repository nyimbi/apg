"""
Time & Attendance Mobile API

Mobile-optimized endpoints with push notifications, offline support,
and enhanced user experience for the governed APG Time & Attendance capability.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict

from .service import TimeAttendanceService
from .websocket import websocket_manager, RealTimeEvent
from .ai_fraud_detection import fraud_engine
from .models import WorkMode, AIAgentType, ProductivityMetric, RemoteWorkStatus
from .config import get_config
from .context import decode_bearer_claims, resolve_current_user_context


logger = logging.getLogger(__name__)
security = HTTPBearer()
_mobile_runtime_state: Dict[str, Any] = {
	"notifications": {},
	"photo_verifications": {},
	"work_summaries": {},
	"push_tokens": {},
	"sync_conflicts": [],
}

# Create mobile-specific router
mobile_router = APIRouter(
	prefix="/api/mobile/human_capital_management/time_attendance",
	tags=["Mobile Time & Attendance"],
	responses={
		400: {"description": "Bad Request"},
		401: {"description": "Unauthorized"},
		403: {"description": "Forbidden"},
		404: {"description": "Not Found"},
		500: {"description": "Internal Server Error"}
	}
)


# Mobile-specific models for optimized payloads
class MobileClockInRequest(BaseModel):
	"""Mobile-optimized clock-in request"""
	model_config = ConfigDict(extra='forbid')
	
	employee_id: str = Field(..., description="Employee identifier")
	location: Optional[Dict[str, float]] = Field(None, description="GPS coordinates")
	biometric_data: Optional[str] = Field(None, description="Base64 encoded biometric data") 
	photo_verification: Optional[str] = Field(None, description="Base64 encoded photo")
	device_info: Dict[str, Any] = Field(default_factory=dict, description="Mobile device info")
	offline_timestamp: Optional[datetime] = Field(None, description="Timestamp when offline")
	network_quality: Optional[str] = Field(None, description="Network quality indicator")
	battery_level: Optional[int] = Field(None, ge=0, le=100, description="Battery percentage")


class MobileClockOutRequest(BaseModel):
	"""Mobile-optimized clock-out request"""
	model_config = ConfigDict(extra='forbid')
	
	employee_id: str = Field(..., description="Employee identifier")
	location: Optional[Dict[str, float]] = Field(None, description="GPS coordinates")
	biometric_data: Optional[str] = Field(None, description="Base64 encoded biometric data")
	photo_verification: Optional[str] = Field(None, description="Base64 encoded photo")
	work_summary: Optional[str] = Field(None, max_length=500, description="Brief work summary")
	offline_timestamp: Optional[datetime] = Field(None, description="Timestamp when offline")
	productivity_rating: Optional[int] = Field(None, ge=1, le=5, description="Self-reported productivity")


class MobileTimeEntryResponse(BaseModel):
	"""Mobile-optimized time entry response"""
	id: str
	status: str
	clock_in: Optional[datetime]
	clock_out: Optional[datetime]
	total_hours: Optional[float]
	fraud_score: float
	requires_approval: bool
	location_verified: bool
	biometric_verified: bool


class QuickStatusResponse(BaseModel):
	"""Quick status for mobile dashboard"""
	employee_id: str
	is_clocked_in: bool
	current_session_hours: Optional[float]
	today_total_hours: float
	week_total_hours: float
	pending_approvals: int
	recent_alerts: List[Dict[str, Any]]


class PushNotificationRequest(BaseModel):
	"""Push notification configuration"""
	device_token: str = Field(..., description="FCM/APNS device token")
	platform: str = Field(..., description="ios or android")
	notification_preferences: Dict[str, bool] = Field(default_factory=dict)


class OfflineSyncRequest(BaseModel):
	"""Offline data synchronization"""
	offline_entries: List[Dict[str, Any]] = Field(..., description="Offline time entries")
	sync_timestamp: datetime = Field(..., description="Last sync timestamp")
	conflict_resolution: str = Field(default="server_wins", description="Conflict resolution strategy")


# Mobile Authentication Helper
def _clean_mobile_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _first_mobile_text(candidates: List[Any], fallback: str) -> str:
	for candidate in candidates:
		text = _clean_mobile_text(candidate)
		if text:
			return text
	return fallback


def _request_state_value(request: Request, name: str) -> Any:
	state = getattr(request, "state", None)
	if state is None:
		return None
	current_user = getattr(state, "current_user", None)
	if isinstance(current_user, dict) and current_user.get(name) is not None:
		return current_user.get(name)
	if current_user is not None and getattr(current_user, name, None) is not None:
		return getattr(current_user, name)
	return getattr(state, name, None)


async def get_mobile_user(
	request: Request,
	credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
	"""Get mobile user with enhanced context"""
	claims = decode_bearer_claims(credentials)
	context = resolve_current_user_context(request, credentials=credentials, roles=["employee"])
	headers = request.headers or {}
	query_params = request.query_params or {}

	context["employee_id"] = _first_mobile_text([
		_request_state_value(request, "employee_id"),
		claims.get("employee_id"),
		claims.get("employee"),
		headers.get("X-Employee-ID"),
		headers.get("X-APG-Employee-ID"),
		query_params.get("employee_id"),
		query_params.get("employee"),
	], context["user_id"])
	context["device_id"] = _first_mobile_text([
		_request_state_value(request, "device_id"),
		claims.get("device_id"),
		claims.get("device"),
		headers.get("X-Device-ID"),
		headers.get("X-APG-Device-ID"),
		query_params.get("device_id"),
		query_params.get("device"),
	], f"mobile:{context['user_id']}")
	context["mobile_verified"] = bool(context["user_id"] and context["tenant_id"])
	return context


async def get_mobile_service() -> TimeAttendanceService:
	"""Get service instance for mobile"""
	return TimeAttendanceService()


# Quick Actions - Optimized for mobile usage
@mobile_router.post("/quick-clock-in", response_model=MobileTimeEntryResponse)
async def quick_clock_in(
	request: MobileClockInRequest,
	background_tasks: BackgroundTasks,
	current_user: Dict[str, Any] = Depends(get_mobile_user),
	service: TimeAttendanceService = Depends(get_mobile_service)
):
	"""
	Quick clock-in optimized for mobile devices
	
	Features:
	- Reduced payload size
	- Enhanced location verification
	- Biometric and photo verification
	- Offline support with sync
	- Real-time fraud detection
	"""
	try:
		# Enhanced mobile device info
		mobile_device_info = {
			**request.device_info,
			"device_id": current_user["device_id"],
			"platform": "mobile",
			"network_quality": request.network_quality,
			"battery_level": request.battery_level,
			"app_version": request.device_info.get("app_version", "1.0.0")
		}
		
		# Process clock-in with mobile optimizations
		time_entry = await service.clock_in(
			employee_id=request.employee_id,
			tenant_id=current_user["tenant_id"],
			device_info=mobile_device_info,
			location=request.location,
			biometric_data={"mobile_biometric": request.biometric_data} if request.biometric_data else None,
			created_by=current_user["user_id"]
		)
		
		# Enhanced mobile validations
		location_verified = await _verify_mobile_location(request.location, request.employee_id)
		biometric_verified = await _verify_mobile_biometric(request.biometric_data, request.employee_id)
		
		# Add photo verification if provided
		if request.photo_verification:
			background_tasks.add_task(
				_process_photo_verification,
				request.photo_verification,
				request.employee_id,
				time_entry.id
			)
		
		# Real-time notification
		await _send_mobile_notification(
			current_user["device_id"],
			"Clock-in Successful",
			f"Clocked in at {time_entry.clock_in.strftime('%H:%M')}",
			{"type": "clock_in_success", "entry_id": time_entry.id}
		)
		
		# Broadcast real-time event
		event = RealTimeEvent(
			event_type="mobile_clock_in",
			entity_type="time_entry",
			entity_id=time_entry.id,
			tenant_id=current_user["tenant_id"],
			data={
				"employee_id": request.employee_id,
				"timestamp": time_entry.clock_in.isoformat(),
				"location_verified": location_verified,
				"biometric_verified": biometric_verified,
				"fraud_score": time_entry.anomaly_score
			},
			user_id=current_user["user_id"]
		)
		await websocket_manager.broadcast_time_entry_event(event)
		
		return MobileTimeEntryResponse(
			id=time_entry.id,
			status=time_entry.status.value,
			clock_in=time_entry.clock_in,
			clock_out=time_entry.clock_out,
			total_hours=float(time_entry.total_hours) if time_entry.total_hours else None,
			fraud_score=time_entry.anomaly_score,
			requires_approval=time_entry.requires_approval,
			location_verified=location_verified,
			biometric_verified=biometric_verified
		)
		
	except ValueError as e:
		# Send error notification
		await _send_mobile_notification(
			current_user["device_id"],
			"Clock-in Failed", 
			str(e),
			{"type": "clock_in_error", "error": str(e)}
		)
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Mobile clock-in error: {str(e)}")
		raise HTTPException(status_code=500, detail="Clock-in failed")


@mobile_router.post("/quick-clock-out", response_model=MobileTimeEntryResponse)
async def quick_clock_out(
	request: MobileClockOutRequest,
	background_tasks: BackgroundTasks,
	current_user: Dict[str, Any] = Depends(get_mobile_user),
	service: TimeAttendanceService = Depends(get_mobile_service)
):
	"""
	Quick clock-out optimized for mobile devices
	
	Features:
	- Work summary capture
	- Self-reported productivity
	- Enhanced verification
	- Automatic overtime calculation
	"""
	try:
		# Enhanced mobile device info
		mobile_device_info = {
			"device_id": current_user["device_id"],
			"platform": "mobile",
			"work_summary": request.work_summary,
			"productivity_rating": request.productivity_rating
		}
		
		# Process clock-out
		time_entry = await service.clock_out(
			employee_id=request.employee_id,
			tenant_id=current_user["tenant_id"],
			device_info=mobile_device_info,
			location=request.location,
			biometric_data={"mobile_biometric": request.biometric_data} if request.biometric_data else None,
			created_by=current_user["user_id"]
		)
		
		# Calculate session summary
		session_hours = float(time_entry.total_hours) if time_entry.total_hours else 0
		overtime_hours = float(time_entry.overtime_hours) if time_entry.overtime_hours else 0
		
		# Enhanced verification
		location_verified = await _verify_mobile_location(request.location, request.employee_id)
		biometric_verified = await _verify_mobile_biometric(request.biometric_data, request.employee_id)
		
		# Send completion notification with summary
		notification_message = f"Worked {session_hours:.1f} hours"
		if overtime_hours > 0:
			notification_message += f" ({overtime_hours:.1f} overtime)"
		
		await _send_mobile_notification(
			current_user["device_id"],
			"Clock-out Complete",
			notification_message,
			{
				"type": "clock_out_success",
				"entry_id": time_entry.id,
				"total_hours": session_hours,
				"overtime_hours": overtime_hours
			}
		)
		
		# Process work summary in background
		if request.work_summary:
			background_tasks.add_task(
				_process_work_summary,
				request.work_summary,
				time_entry.id,
				request.productivity_rating
			)
		
		return MobileTimeEntryResponse(
			id=time_entry.id,
			status=time_entry.status.value,
			clock_in=time_entry.clock_in,
			clock_out=time_entry.clock_out,
			total_hours=session_hours,
			fraud_score=time_entry.anomaly_score,
			requires_approval=time_entry.requires_approval,
			location_verified=location_verified,
			biometric_verified=biometric_verified
		)
		
	except ValueError as e:
		await _send_mobile_notification(
			current_user["device_id"],
			"Clock-out Failed",
			str(e),
			{"type": "clock_out_error", "error": str(e)}
		)
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Mobile clock-out error: {str(e)}")
		raise HTTPException(status_code=500, detail="Clock-out failed")


@mobile_router.get("/quick-status", response_model=QuickStatusResponse)
async def get_quick_status(
	current_user: Dict[str, Any] = Depends(get_mobile_user),
	service: TimeAttendanceService = Depends(get_mobile_service)
):
	"""
	Get quick status for mobile dashboard
	
	Optimized payload with essential information:
	- Current clock status
	- Today's hours
	- Week's hours  
	- Pending items
	"""
	try:
		employee_id = current_user["employee_id"]
		tenant_id = current_user["tenant_id"]
		
		# Get current clock status
		active_entry = await service._get_active_time_entry(employee_id, tenant_id)
		is_clocked_in = active_entry is not None
		
		# Calculate current session hours
		current_session_hours = None
		if active_entry and active_entry.clock_in:
			session_duration = datetime.utcnow() - active_entry.clock_in
			current_session_hours = session_duration.total_seconds() / 3600
		
		entries = await service.list_time_entries(tenant_id, employee_id=employee_id)
		today = date.today()
		week_start = today - timedelta(days=today.weekday())
		today_total_hours = sum(
			float(entry.total_hours or 0)
			for entry in entries
			if entry.entry_date == today
		)
		week_total_hours = sum(
			float(entry.total_hours or 0)
			for entry in entries
			if entry.entry_date >= week_start
		)
		pending_approvals = len([entry for entry in entries if entry.requires_approval])
		recent_alerts = []
		for entry in entries[:5]:
			if entry.anomaly_score >= 0.5:
				recent_alerts.append({
					"type": "fraud_alert",
					"message": "Time entry requires fraud review",
					"timestamp": entry.updated_at.isoformat(),
					"severity": "warning"
				})
			if entry.requires_approval:
				recent_alerts.append({
					"type": "approval_required",
					"message": "Time entry requires approval",
					"timestamp": entry.updated_at.isoformat(),
					"severity": "info"
				})
		
		return QuickStatusResponse(
			employee_id=employee_id,
			is_clocked_in=is_clocked_in,
			current_session_hours=current_session_hours,
			today_total_hours=round(today_total_hours, 2),
			week_total_hours=round(week_total_hours, 2),
			pending_approvals=pending_approvals,
			recent_alerts=recent_alerts
		)
		
	except Exception as e:
		logger.error(f"Error getting quick status: {str(e)}")
		raise HTTPException(status_code=500, detail="Failed to get status")


# Push Notification Management
@mobile_router.post("/notifications/register")
async def register_for_notifications(
	request: PushNotificationRequest,
	current_user: Dict[str, Any] = Depends(get_mobile_user)
):
	"""Register device for push notifications"""
	try:
		# Store device token and preferences
		await _register_push_token(
			current_user["user_id"],
			current_user["device_id"],
			request.device_token,
			request.platform,
			request.notification_preferences
		)
		
		# Send welcome notification
		await _send_mobile_notification(
			current_user["device_id"],
			"Notifications Enabled",
			"You'll receive important time tracking updates",
			{"type": "registration_success"}
		)
		
		return {"success": True, "message": "Push notifications registered"}
		
	except Exception as e:
		logger.error(f"Error registering push notifications: {str(e)}")
		raise HTTPException(status_code=500, detail="Registration failed")


@mobile_router.put("/notifications/preferences")
async def update_notification_preferences(
	preferences: Dict[str, bool],
	current_user: Dict[str, Any] = Depends(get_mobile_user)
):
	"""Update notification preferences"""
	try:
		await _update_notification_preferences(
			current_user["user_id"],
			preferences
		)
		
		return {"success": True, "message": "Preferences updated"}
		
	except Exception as e:
		logger.error(f"Error updating notification preferences: {str(e)}")
		raise HTTPException(status_code=500, detail="Update failed")


# Offline Synchronization
@mobile_router.post("/sync/offline-entries")
async def sync_offline_entries(
	request: OfflineSyncRequest,
	background_tasks: BackgroundTasks,
	current_user: Dict[str, Any] = Depends(get_mobile_user),
	service: TimeAttendanceService = Depends(get_mobile_service)
):
	"""
	Synchronize offline time entries
	
	Handles conflict resolution and ensures data integrity
	when mobile app reconnects after being offline.
	"""
	try:
		sync_results = []
		conflicts = []
		
		for offline_entry in request.offline_entries:
			try:
				# Check for conflicts with server data
				conflict = await _check_sync_conflict(offline_entry, current_user["tenant_id"])
				
				if conflict:
					conflicts.append({
						"offline_entry": offline_entry,
						"server_entry": conflict,
						"resolution": request.conflict_resolution
					})
					
					if request.conflict_resolution == "server_wins":
						continue  # Skip this entry
				
				# Process offline entry
				if offline_entry["type"] == "clock_in":
					result = await service.clock_in(
						employee_id=offline_entry["employee_id"],
						tenant_id=current_user["tenant_id"],
						device_info={**offline_entry["device_info"], "offline_sync": True},
						location=offline_entry.get("location"),
						created_by=current_user["user_id"]
					)
				elif offline_entry["type"] == "clock_out":
					result = await service.clock_out(
						employee_id=offline_entry["employee_id"],
						tenant_id=current_user["tenant_id"],
						device_info={**offline_entry["device_info"], "offline_sync": True},
						location=offline_entry.get("location"),
						created_by=current_user["user_id"]
					)
				
				sync_results.append({
					"offline_id": offline_entry["offline_id"],
					"server_id": result.id,
					"status": "synced",
					"timestamp": result.created_at.isoformat()
				})
				
			except Exception as e:
				sync_results.append({
					"offline_id": offline_entry.get("offline_id"),
					"status": "error",
					"error": str(e)
				})
		
		# Process conflicts in background
		if conflicts:
			background_tasks.add_task(_resolve_sync_conflicts, conflicts)
		
		return {
			"success": True,
			"synced_entries": len([r for r in sync_results if r["status"] == "synced"]),
			"failed_entries": len([r for r in sync_results if r["status"] == "error"]),
			"conflicts": len(conflicts),
			"results": sync_results
		}
		
	except Exception as e:
		logger.error(f"Error syncing offline entries: {str(e)}")
		raise HTTPException(status_code=500, detail="Sync failed")


# Location Services for Mobile
@mobile_router.post("/location/verify")
async def verify_work_location(
	location: Dict[str, float],
	current_user: Dict[str, Any] = Depends(get_mobile_user)
):
	"""Verify if current location is valid for work"""
	try:
		is_valid = await _verify_mobile_location(location, current_user["employee_id"])
		
		# Get distance to nearest work location
		distance_info = await _get_location_distance_info(location, current_user["employee_id"])
		
		return {
			"is_valid": is_valid,
			"location": location,
			"distance_to_work": distance_info["distance_meters"],
			"nearest_work_location": distance_info["nearest_location"],
			"within_geofence": distance_info["within_geofence"]
		}
		
	except Exception as e:
		logger.error(f"Error verifying location: {str(e)}")
		raise HTTPException(status_code=500, detail="Location verification failed")


# Mobile Analytics
@mobile_router.get("/analytics/personal")
async def get_personal_analytics(
	days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
	current_user: Dict[str, Any] = Depends(get_mobile_user),
	service: TimeAttendanceService = Depends(get_mobile_service)
):
	"""Get personal time tracking analytics for mobile"""
	try:
		start_date = date.today() - timedelta(days=days - 1)
		entries = await service.list_time_entries(
			current_user["tenant_id"],
			employee_id=current_user["employee_id"],
			start_date=start_date,
		)
		total_hours = sum(float(entry.total_hours or 0) for entry in entries)
		overtime_hours = sum(float(entry.overtime_hours or 0) for entry in entries)
		worked_days = len({entry.entry_date for entry in entries})
		punctual_entries = [
			entry for entry in entries
			if entry.clock_in and entry.clock_in.time().hour <= 9
		]
		daily_breakdown = []
		for offset in range(days):
			entry_date = start_date + timedelta(days=offset)
			day_entries = [entry for entry in entries if entry.entry_date == entry_date]
			day_hours = sum(float(entry.total_hours or 0) for entry in day_entries)
			day_overtime = sum(float(entry.overtime_hours or 0) for entry in day_entries)
			daily_breakdown.append({
				"date": entry_date.isoformat(),
				"hours": round(day_hours, 2),
				"overtime": round(day_overtime, 2),
				"punctuality": 1.0 if any(entry.clock_in and entry.clock_in.time().hour <= 9 for entry in day_entries) else 0.0
			})
		achievements = []
		if worked_days and len(punctual_entries) == len(entries):
			achievements.append({
				"type": "punctuality",
				"title": "On-time period",
				"description": "All tracked starts were on time",
				"earned_date": date.today().isoformat()
			})
		analytics = {
			"period_days": days,
			"total_hours": round(total_hours, 2),
			"average_daily_hours": round(total_hours / days, 2),
			"punctuality_score": round(len(punctual_entries) / len(entries), 4) if entries else 0.0,
			"overtime_hours": round(overtime_hours, 2),
			"productivity_trend": "improving" if total_hours else "stable",
			"daily_breakdown": daily_breakdown,
			"achievements": achievements
		}
		
		return analytics
		
	except Exception as e:
		logger.error(f"Error getting personal analytics: {str(e)}")
		raise HTTPException(status_code=500, detail="Analytics unavailable")


# Helper Functions
async def _verify_mobile_location(location: Optional[Dict[str, float]], employee_id: str) -> bool:
	"""Verify mobile location against allowed work locations"""
	if not location:
		return False
	
	# Mock verification - would check against actual geofences
	# For demo, accept locations within reasonable bounds
	lat, lon = location.get("latitude", 0), location.get("longitude", 0)
	return -90 <= lat <= 90 and -180 <= lon <= 180


async def _verify_mobile_biometric(biometric_data: Optional[str], employee_id: str) -> bool:
	"""Verify mobile biometric data"""
	if not biometric_data:
		return False
	
	# Mock verification - would use actual biometric matching
	return len(biometric_data) > 50  # Simple length check for demo


async def _send_mobile_notification(
	device_id: str,
	title: str,
	message: str,
	data: Dict[str, Any]
):
	"""Send push notification to mobile device"""
	logger.info(f"Sending notification to {device_id}: {title} - {message}")
	notifications = _mobile_runtime_state["notifications"].setdefault(device_id, [])
	notifications.append({
		"title": title,
		"message": message,
		"data": data,
		"sent_at": datetime.utcnow().isoformat(),
	})


async def _process_photo_verification(photo_data: str, employee_id: str, time_entry_id: str):
	"""Process photo verification in background"""
	logger.info(f"Processing photo verification for entry {time_entry_id}")
	_mobile_runtime_state["photo_verifications"][time_entry_id] = {
		"employee_id": employee_id,
		"photo_size": len(photo_data),
		"verified": bool(photo_data),
		"processed_at": datetime.utcnow().isoformat(),
	}


async def _process_work_summary(summary: str, time_entry_id: str, productivity_rating: Optional[int]):
	"""Process work summary and productivity rating"""
	logger.info(f"Processing work summary for entry {time_entry_id}: {summary}")
	_mobile_runtime_state["work_summaries"][time_entry_id] = {
		"summary": summary,
		"productivity_rating": productivity_rating,
		"processed_at": datetime.utcnow().isoformat(),
	}


async def _register_push_token(
	user_id: str,
	device_id: str,
	token: str,
	platform: str,
	preferences: Dict[str, bool]
):
	"""Register push notification token"""
	logger.info(f"Registering push token for user {user_id}, device {device_id}")
	_mobile_runtime_state["push_tokens"][device_id] = {
		"user_id": user_id,
		"token": token,
		"platform": platform,
		"preferences": preferences,
		"registered_at": datetime.utcnow().isoformat(),
	}


async def _update_notification_preferences(user_id: str, preferences: Dict[str, bool]):
	"""Update notification preferences"""
	logger.info(f"Updating notification preferences for user {user_id}")
	for token_record in _mobile_runtime_state["push_tokens"].values():
		if token_record["user_id"] == user_id:
			token_record["preferences"] = preferences
			token_record["updated_at"] = datetime.utcnow().isoformat()


async def _check_sync_conflict(offline_entry: Dict[str, Any], tenant_id: str) -> Optional[Dict[str, Any]]:
	"""Check for synchronization conflicts"""
	if offline_entry.get("server_id"):
		return {
			"id": offline_entry["server_id"],
			"tenant_id": tenant_id,
			"reason": "server_record_exists",
		}
	return None


async def _resolve_sync_conflicts(conflicts: List[Dict[str, Any]]):
	"""Resolve synchronization conflicts"""
	logger.info(f"Resolving {len(conflicts)} sync conflicts")
	_mobile_runtime_state["sync_conflicts"].extend({
		**conflict,
		"resolved_at": datetime.utcnow().isoformat(),
	} for conflict in conflicts)


async def _get_location_distance_info(location: Dict[str, float], employee_id: str) -> Dict[str, Any]:
	"""Get distance information for location"""
	# Mock distance calculation - would use actual work locations
	return {
		"distance_meters": 150,
		"nearest_location": {"name": "Main Office", "latitude": 40.7128, "longitude": -74.0060},
		"within_geofence": True
	}


# Export mobile router
__all__ = ["mobile_router"]
