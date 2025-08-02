"""
Mobile API Endpoints

Provides REST API endpoints for mobile applications:
- Device registration and management
- Mobile-optimized workflow APIs
- Push notification management
- Offline synchronization
- Touch-friendly interfaces
- Native app integration

© 2025 Datacraft
Author: Nyimbi Odero
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from flask_appbuilder import BaseView, expose, has_access
from pydantic import ValidationError
import structlog

from .mobile_app_service import (
	mobile_app_service, MobilePlatform, MobileDeviceType,
	NotificationType, OfflineMode, TouchGesture
)

logger = structlog.get_logger(__name__)

# Create Flask Blueprint
mobile_bp = Blueprint(
	'mobile',
	__name__,
	url_prefix='/api/mobile'
)


# =============================================================================
# Device Registration and Management
# =============================================================================

@mobile_bp.route('/devices/register', methods=['POST'])
async def register_mobile_device():
	"""Register mobile device"""
	try:
		data = request.get_json()
		
		# Validate required fields
		required_fields = ['user_id', 'platform', 'model', 'os_version', 'app_version']
		for field in required_fields:
			if field not in data:
				return jsonify({
					"success": False,
					"error": f"Missing required field: {field}"
				}), 400
		
		device_id = await mobile_app_service.register_device(data)
		
		return jsonify({
			"success": True,
			"data": {
				"device_id": device_id,
				"registration_time": datetime.utcnow().isoformat()
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Device registration error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/devices/<device_id>/update', methods=['PUT'])
async def update_mobile_device(device_id: str):
	"""Update mobile device information"""
	try:
		data = request.get_json()
		
		# Update device info
		device = mobile_app_service.devices.get(device_id)
		if not device:
			return jsonify({
				"success": False,
				"error": "Device not found"
			}), 404
		
		# Update allowed fields
		if 'push_token' in data:
			device.push_token = data['push_token']
		if 'app_version' in data:
			device.app_version = data['app_version']
		if 'os_version' in data:
			device.os_version = data['os_version']
		if 'capabilities' in data:
			device.capabilities = data['capabilities']
		
		device.last_seen = datetime.utcnow()
		
		# Store updated info
		await mobile_app_service._store_device_info(device)
		
		return jsonify({
			"success": True,
			"data": {
				"device_id": device_id,
				"updated_at": device.last_seen.isoformat()
			}
		})
		
	except Exception as e:
		logger.error(f"Device update error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/sessions/create', methods=['POST'])
async def create_mobile_session():
	"""Create mobile app session"""
	try:
		data = request.get_json()
		device_id = data.get('device_id')
		
		if not device_id:
			return jsonify({
				"success": False,
				"error": "device_id is required"
			}), 400
		
		config = data.get('config', {})
		session_id = await mobile_app_service.create_mobile_session(device_id, config)
		
		return jsonify({
			"success": True,
			"data": {
				"session_id": session_id,
				"device_id": device_id,
				"offline_mode": config.get('offline_mode', 'read_only')
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Mobile session creation error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/sessions/<session_id>/heartbeat', methods=['POST'])
async def session_heartbeat(session_id: str):
	"""Update session activity"""
	try:
		session_obj = mobile_app_service.sessions.get(session_id)
		if not session_obj:
			return jsonify({
				"success": False,
				"error": "Session not found"
			}), 404
		
		data = request.get_json() or {}
		
		# Update session info
		session_obj.last_activity = datetime.utcnow()
		session_obj.is_foreground = data.get('is_foreground', True)
		
		# Update device last seen
		if session_obj.device:
			session_obj.device.last_seen = datetime.utcnow()
		
		return jsonify({
			"success": True,
			"data": {
				"session_id": session_id,
				"last_activity": session_obj.last_activity.isoformat(),
				"is_foreground": session_obj.is_foreground
			}
		})
		
	except Exception as e:
		logger.error(f"Session heartbeat error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Mobile Workflow APIs
# =============================================================================

@mobile_bp.route('/workflows', methods=['GET'])
async def get_mobile_workflows():
	"""Get workflows optimized for mobile"""
	try:
		session_id = request.headers.get('X-Mobile-Session-ID')
		if not session_id:
			return jsonify({
				"success": False,
				"error": "Mobile session ID required"
			}), 400
		
		# Build filters from query parameters
		filters = {
			"limit": int(request.args.get('limit', 20)),
			"offset": int(request.args.get('offset', 0)),
			"status": request.args.get('status'),
			"category": request.args.get('category'),
			"offline_capable": request.args.get('offline_capable') == 'true',
			"complexity": request.args.get('complexity'),  # simple, medium, complex
			"estimated_time": request.args.get('estimated_time')  # quick, medium, long
		}
		
		workflows = await mobile_app_service.get_mobile_workflows(session_id, filters)
		
		return jsonify({
			"success": True,
			"data": {
				"workflows": workflows,
				"total": len(workflows),
				"filters": filters
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Get mobile workflows error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/workflows/<workflow_id>/details', methods=['GET'])
async def get_mobile_workflow_details(workflow_id: str):
	"""Get detailed workflow information for mobile"""
	try:
		session_id = request.headers.get('X-Mobile-Session-ID')
		if not session_id:
			return jsonify({
				"success": False,
				"error": "Mobile session ID required"
			}), 400
		
		session_obj = mobile_app_service.sessions.get(session_id)
		if not session_obj:
			return jsonify({
				"success": False,
				"error": "Session not found"
			}), 404
		
		# Get workflow details with mobile optimizations
		from .database import get_async_db_session
		from sqlalchemy import text
		
		async with get_async_db_session() as db_session:
			result = await db_session.execute(
				text("""
				SELECT 
					w.id,
					w.name,
					w.description,
					w.definition,
					w.mobile_config,
					w.estimated_duration,
					w.complexity_level,
					w.offline_capable,
					COUNT(wn.id) as node_count,
					AVG(we.duration_seconds) as avg_duration
				FROM workflow_definitions w
				LEFT JOIN workflow_nodes wn ON w.id = wn.workflow_id
				LEFT JOIN workflow_executions we ON w.id = we.workflow_id AND we.status = 'completed'
				WHERE w.id = :workflow_id
				GROUP BY w.id, w.name, w.description, w.definition, w.mobile_config,
						 w.estimated_duration, w.complexity_level, w.offline_capable
				"""),
				{"workflow_id": workflow_id}
			)
			
			row = result.first()
			if not row:
				return jsonify({
					"success": False,
					"error": "Workflow not found"
				}), 404
			
			mobile_config = json.loads(row.mobile_config) if row.mobile_config else {}
			definition = json.loads(row.definition) if row.definition else {}
			
			# Create mobile-optimized response
			workflow_details = {
				"id": row.id,
				"name": row.name,
				"description": row.description,
				"node_count": row.node_count,
				"estimated_duration": row.estimated_duration or int(row.avg_duration or 300),
				"complexity": row.complexity_level or "medium",
				"offline_capable": row.offline_capable or False,
				"mobile_layout": mobile_config.get("layout", "vertical"),
				"touch_optimized": mobile_config.get("touch_optimized", True),
				"requires_approval": mobile_config.get("requires_approval", False),
				"prerequisites": mobile_config.get("prerequisites", []),
				"expected_outputs": mobile_config.get("expected_outputs", []),
				"mobile_steps": []
			}
			
			# Convert nodes to mobile-friendly steps
			nodes = definition.get("nodes", [])
			for i, node in enumerate(nodes):
				mobile_step = {
					"step_number": i + 1,
					"id": node.get("id"),
					"name": node.get("name", f"Step {i + 1}"),
					"type": node.get("type", "task"),
					"description": node.get("description", ""),
					"required_input": node.get("required_input", []),
					"estimated_time": node.get("estimated_time", 30),
					"touch_interactions": node.get("touch_interactions", ["tap"]),
					"can_skip": node.get("can_skip", False),
					"offline_capable": node.get("offline_capable", False)
				}
				workflow_details["mobile_steps"].append(mobile_step)
		
		return jsonify({
			"success": True,
			"data": workflow_details
		})
		
	except Exception as e:
		logger.error(f"Get mobile workflow details error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/workflows/<workflow_id>/execute', methods=['POST'])
async def execute_mobile_workflow(workflow_id: str):
	"""Execute workflow from mobile app"""
	try:
		session_id = request.headers.get('X-Mobile-Session-ID')
		if not session_id:
			return jsonify({
				"success": False,
				"error": "Mobile session ID required"
			}), 400
		
		data = request.get_json() or {}
		params = data.get('parameters', {})
		
		execution_id = await mobile_app_service.execute_mobile_workflow(session_id, workflow_id, params)
		
		return jsonify({
			"success": True,
			"data": {
				"execution_id": execution_id,
				"workflow_id": workflow_id,
				"started_at": datetime.utcnow().isoformat(),
				"status": "running"
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Mobile workflow execution error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/executions/<execution_id>/status', methods=['GET'])
async def get_mobile_execution_status(execution_id: str):
	"""Get workflow execution status for mobile"""
	try:
		session_id = request.headers.get('X-Mobile-Session-ID')
		if not session_id:
			return jsonify({
				"success": False,
				"error": "Mobile session ID required"
			}), 400
		
		status = await mobile_app_service.get_mobile_execution_status(session_id, execution_id)
		
		return jsonify({
			"success": True,
			"data": status
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Get mobile execution status error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/executions/<execution_id>/cancel', methods=['POST'])
async def cancel_mobile_execution(execution_id: str):
	"""Cancel workflow execution from mobile"""
	try:
		session_id = request.headers.get('X-Mobile-Session-ID')
		if not session_id:
			return jsonify({
				"success": False,
				"error": "Mobile session ID required"
			}), 400
		
		# Update execution status
		from .database import get_async_db_session
		from sqlalchemy import text
		
		async with get_async_db_session() as db_session:
			await db_session.execute(
				text("""
				UPDATE workflow_executions 
				SET status = 'cancelled', completed_at = :completed_at
				WHERE id = :execution_id
				AND mobile_session_id = :session_id
				AND status IN ('running', 'paused')
				"""),
				{
					"execution_id": execution_id,
					"session_id": session_id,
					"completed_at": datetime.utcnow()
				}
			)
			await db_session.commit()
		
		return jsonify({
			"success": True,
			"data": {
				"execution_id": execution_id,
				"status": "cancelled",
				"cancelled_at": datetime.utcnow().isoformat()
			}
		})
		
	except Exception as e:
		logger.error(f"Cancel mobile execution error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Push Notification APIs
# =============================================================================

@mobile_bp.route('/notifications/send', methods=['POST'])
async def send_push_notification():
	"""Send push notification to mobile device"""
	try:
		data = request.get_json()
		
		required_fields = ['device_id', 'notification_type', 'title', 'message']
		for field in required_fields:
			if field not in data:
				return jsonify({
					"success": False,
					"error": f"Missing required field: {field}"
				}), 400
		
		notification_id = await mobile_app_service.send_push_notification(
			data['device_id'],
			NotificationType(data['notification_type']),
			data['title'],
			data['message'],
			data.get('data', {})
		)
		
		return jsonify({
			"success": True,
			"data": {
				"notification_id": notification_id,
				"sent_at": datetime.utcnow().isoformat()
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Send push notification error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/notifications/<device_id>/preferences', methods=['GET'])
async def get_notification_preferences(device_id: str):
	"""Get notification preferences for device"""
	try:
		# Get notification preferences from database
		from .database import get_async_db_session
		from sqlalchemy import text
		
		async with get_async_db_session() as db_session:
			result = await db_session.execute(
				text("""
				SELECT notification_preferences 
				FROM mobile_devices 
				WHERE device_id = :device_id
				"""),
				{"device_id": device_id}
			)
			
			row = result.first()
			if not row:
				return jsonify({
					"success": False,
					"error": "Device not found"
				}), 404
			
			preferences = json.loads(row.notification_preferences) if row.notification_preferences else {
				"workflow_started": True,
				"workflow_completed": True,
				"workflow_failed": True,
				"approval_request": True,
				"task_assigned": True,
				"deadline_approaching": True,
				"system_alert": False,
				"quiet_hours": {
					"enabled": False,
					"start_time": "22:00",
					"end_time": "08:00"
				}
			}
		
		return jsonify({
			"success": True,
			"data": {
				"device_id": device_id,
				"preferences": preferences
			}
		})
		
	except Exception as e:
		logger.error(f"Get notification preferences error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/notifications/<device_id>/preferences', methods=['PUT'])
async def update_notification_preferences(device_id: str):
	"""Update notification preferences for device"""
	try:
		data = request.get_json()
		preferences = data.get('preferences', {})
		
		# Update preferences in database
		from .database import get_async_db_session
		from sqlalchemy import text
		
		async with get_async_db_session() as db_session:
			await db_session.execute(
				text("""
				UPDATE mobile_devices 
				SET notification_preferences = :preferences
				WHERE device_id = :device_id
				"""),
				{
					"device_id": device_id,
					"preferences": json.dumps(preferences)
				}
			)
			await db_session.commit()
		
		return jsonify({
			"success": True,
			"data": {
				"device_id": device_id,
				"updated_at": datetime.utcnow().isoformat()
			}
		})
		
	except Exception as e:
		logger.error(f"Update notification preferences error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Offline Synchronization APIs
# =============================================================================

@mobile_bp.route('/sync/status', methods=['GET'])
async def get_sync_status():
	"""Get offline synchronization status"""
	try:
		session_id = request.headers.get('X-Mobile-Session-ID')
		if not session_id:
			return jsonify({
				"success": False,
				"error": "Mobile session ID required"
			}), 400
		
		session_obj = mobile_app_service.sessions.get(session_id)
		if not session_obj:
			return jsonify({
				"success": False,
				"error": "Session not found"
			}), 404
		
		cache_info = mobile_app_service.offline_cache.get(session_id, {})
		
		sync_status = {
			"session_id": session_id,
			"offline_mode": session_obj.offline_mode.value,
			"cached_workflows": len(session_obj.cached_workflows),
			"pending_sync_items": len(session_obj.pending_sync),
			"background_tasks": len(session_obj.background_tasks),
			"last_sync": cache_info.get("last_sync"),
			"cache_size_mb": len(str(cache_info)) / (1024 * 1024),  # Approximate
			"sync_needed": len(session_obj.pending_sync) > 0
		}
		
		return jsonify({
			"success": True,
			"data": sync_status
		})
		
	except Exception as e:
		logger.error(f"Get sync status error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/sync/execute', methods=['POST'])
async def execute_sync():
	"""Execute offline data synchronization"""
	try:
		session_id = request.headers.get('X-Mobile-Session-ID')
		if not session_id:
			return jsonify({
				"success": False,
				"error": "Mobile session ID required"
			}), 400
		
		sync_results = await mobile_app_service.sync_offline_data(session_id)
		
		return jsonify({
			"success": True,
			"data": sync_results
		})
		
	except Exception as e:
		logger.error(f"Execute sync error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/cache/clear', methods=['POST'])
async def clear_offline_cache():
	"""Clear offline cache"""
	try:
		session_id = request.headers.get('X-Mobile-Session-ID')
		if not session_id:
			return jsonify({
				"success": False,
				"error": "Mobile session ID required"
			}), 400
		
		session_obj = mobile_app_service.sessions.get(session_id)
		if not session_obj:
			return jsonify({
				"success": False,
				"error": "Session not found"
			}), 404
		
		# Clear cache
		if session_id in mobile_app_service.offline_cache:
			del mobile_app_service.offline_cache[session_id]
		
		session_obj.cached_workflows.clear()
		session_obj.pending_sync.clear()
		
		return jsonify({
			"success": True,
			"data": {
				"session_id": session_id,
				"cache_cleared": True,
				"cleared_at": datetime.utcnow().isoformat()
			}
		})
		
	except Exception as e:
		logger.error(f"Clear cache error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Mobile-Specific Feature APIs
# =============================================================================

@mobile_bp.route('/gestures/register', methods=['POST'])
async def register_gesture_interaction():
	"""Register touch gesture interaction"""
	try:
		data = request.get_json()
		session_id = request.headers.get('X-Mobile-Session-ID')
		
		gesture_data = {
			"session_id": session_id,
			"gesture_type": data.get('gesture_type'),
			"target_element": data.get('target_element'),
			"coordinates": data.get('coordinates', {}),
			"timestamp": datetime.utcnow().isoformat(),
			"duration": data.get('duration', 0),
			"force": data.get('force', 1.0)  # 3D Touch support
		}
		
		# Log gesture for analytics
		logger.info(f"Gesture interaction: {gesture_data}")
		
		return jsonify({
			"success": True,
			"data": {
				"gesture_registered": True,
				"timestamp": gesture_data["timestamp"]
			}
		})
		
	except Exception as e:
		logger.error(f"Register gesture error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/layout/adaptive', methods=['GET'])
async def get_adaptive_layout():
	"""Get adaptive layout for mobile device"""
	try:
		session_id = request.headers.get('X-Mobile-Session-ID')
		device_id = request.headers.get('X-Device-ID')
		
		if not session_id or not device_id:
			return jsonify({
				"success": False,
				"error": "Session ID and Device ID required"
			}), 400
		
		device = mobile_app_service.devices.get(device_id)
		if not device:
			return jsonify({
				"success": False,
				"error": "Device not found"
			}), 404
		
		# Generate adaptive layout based on device characteristics
		layout_config = {
			"device_id": device_id,
			"screen_size": {
				"width": device.screen_width,
				"height": device.screen_height,
				"density": device.screen_density
			},
			"layout_type": "compact" if device.device_type == MobileDeviceType.PHONE else "expanded",
			"touch_targets": {
				"minimum_size": 44 if device.platform == MobilePlatform.IOS else 48,
				"spacing": 8,
				"edge_margin": 16
			},
			"navigation": {
				"type": "bottom_tabs" if device.device_type == MobileDeviceType.PHONE else "side_drawer",
				"gesture_navigation": device.platform in [MobilePlatform.IOS, MobilePlatform.ANDROID]
			},
			"typography": {
				"base_size": 16 if device.screen_density >= 2.0 else 14,
				"line_height": 1.4,
				"scale_factor": device.screen_density
			},
			"colors": {
				"theme": "auto",  # Light/dark based on system
				"accent": "#007AFF" if device.platform == MobilePlatform.IOS else "#2196F3"
			}
		}
		
		return jsonify({
			"success": True,
			"data": layout_config
		})
		
	except Exception as e:
		logger.error(f"Get adaptive layout error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@mobile_bp.route('/accessibility/features', methods=['GET'])
async def get_accessibility_features():
	"""Get accessibility features for mobile app"""
	try:
		device_id = request.headers.get('X-Device-ID')
		
		if not device_id:
			return jsonify({
				"success": False,
				"error": "Device ID required"
			}), 400
		
		device = mobile_app_service.devices.get(device_id)
		if not device:
			return jsonify({
				"success": False,
				"error": "Device not found"
			}), 404
		
		accessibility_features = {
			"screen_reader": {
				"supported": True,
				"labels": "comprehensive",
				"navigation_hints": True
			},
			"voice_control": {
				"supported": device.platform in [MobilePlatform.IOS, MobilePlatform.ANDROID],
				"commands": ["start workflow", "cancel", "next step", "previous step"]
			},
			"high_contrast": {
				"supported": True,
				"auto_detect": True
			},
			"font_scaling": {
				"supported": True,
				"min_scale": 0.8,
				"max_scale": 2.0
			},
			"reduced_motion": {
				"supported": True,
				"auto_detect": True
			},
			"haptic_feedback": {
				"supported": device.platform in [MobilePlatform.IOS, MobilePlatform.ANDROID],
				"patterns": ["success", "warning", "error", "selection"]
			}
		}
		
		return jsonify({
			"success": True,
			"data": accessibility_features
		})
		
	except Exception as e:
		logger.error(f"Get accessibility features error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Health and Performance APIs
# =============================================================================

@mobile_bp.route('/health', methods=['GET'])
async def mobile_health_check():
	"""Mobile service health check"""
	try:
		health_status = {
			"status": "healthy",
			"timestamp": datetime.utcnow().isoformat(),
			"services": {
				"mobile_app_service": "healthy",
				"push_notifications": "healthy",
				"offline_sync": "healthy",
				"device_management": "healthy"
			},
			"metrics": {
				"registered_devices": len(mobile_app_service.devices),
				"active_sessions": len(mobile_app_service.sessions),
				"cache_entries": len(mobile_app_service.offline_cache)
			}
		}
		
		return jsonify(health_status)
		
	except Exception as e:
		return jsonify({
			"status": "unhealthy",
			"error": str(e),
			"timestamp": datetime.utcnow().isoformat()
		}), 500


@mobile_bp.route('/performance/metrics', methods=['GET'])
async def get_performance_metrics():
	"""Get mobile app performance metrics"""
	try:
		session_id = request.headers.get('X-Mobile-Session-ID')
		
		# Calculate performance metrics
		metrics = {
			"timestamp": datetime.utcnow().isoformat(),
			"global_metrics": {
				"total_devices": len(mobile_app_service.devices),
				"active_sessions": len(mobile_app_service.sessions),
				"total_cache_size_mb": sum(
					len(str(cache)) / (1024 * 1024) 
					for cache in mobile_app_service.offline_cache.values()
				)
			}
		}
		
		if session_id and session_id in mobile_app_service.sessions:
			session_obj = mobile_app_service.sessions[session_id]
			cache = mobile_app_service.offline_cache.get(session_id, {})
			
			metrics["session_metrics"] = {
				"session_id": session_id,
				"session_duration": (datetime.utcnow() - session_obj.started_at).total_seconds(),
				"cached_workflows": len(session_obj.cached_workflows),
				"pending_sync": len(session_obj.pending_sync),
				"background_tasks": len(session_obj.background_tasks),
				"cache_size_mb": len(str(cache)) / (1024 * 1024)
			}
		
		return jsonify({
			"success": True,
			"data": metrics
		})
		
	except Exception as e:
		logger.error(f"Get performance metrics error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Flask-AppBuilder Views
# =============================================================================

class MobileAppView(BaseView):
	"""Mobile application management view"""
	
	route_base = "/mobile"
	
	@expose("/dashboard")
	@has_access
	def dashboard(self):
		"""Mobile app dashboard"""
		return self.render_template(
			"mobile/dashboard.html",
			title="Mobile App Management"
		)
	
	@expose("/devices")
	@has_access
	def device_management(self):
		"""Device management interface"""
		return self.render_template(
			"mobile/devices.html",
			title="Mobile Devices"
		)
	
	@expose("/notifications")
	@has_access
	def notification_management(self):
		"""Notification management interface"""
		return self.render_template(
			"mobile/notifications.html",
			title="Push Notifications"
		)
	
	@expose("/analytics")
	@has_access
	def mobile_analytics(self):
		"""Mobile app analytics"""
		return self.render_template(
			"mobile/analytics.html",
			title="Mobile Analytics"
		)


def register_mobile_views(appbuilder):
	"""Register mobile views with Flask-AppBuilder"""
	appbuilder.add_view(
		MobileAppView,
		"Mobile Dashboard",
		icon="fa-mobile",
		category="Mobile",
		category_icon="fa-mobile-alt"
	)


def register_mobile_routes(app):
	"""Register mobile routes with Flask app"""
	app.register_blueprint(mobile_bp)