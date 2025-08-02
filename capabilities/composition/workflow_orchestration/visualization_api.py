"""
Advanced Visualization API Endpoints

Provides REST API endpoints for advanced visualization features:
- 3D workflow rendering
- VR/AR session management
- Real-time collaboration
- Spatial computing interfaces
- Performance optimization

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

from .visualization_3d import (
	visualization_3d, LayoutAlgorithm, VisualizationMode, AnimationType,
	Vector3D, Node3D, Edge3D
)
from .vr_ar_interface import (
	vr_ar_interface, VRPlatform, ARPlatform, InteractionMode
)
from .advanced_collaboration import (
	advanced_collaboration, CollaborationMode, ConflictResolutionStrategy,
	PresenceStatus, InteractionType
)

logger = structlog.get_logger(__name__)

# Create Flask Blueprint
visualization_bp = Blueprint(
	'visualization',
	__name__,
	url_prefix='/api/visualization'
)


# =============================================================================
# 3D Visualization Endpoints
# =============================================================================

@visualization_bp.route('/3d/render/<workflow_id>', methods=['POST'])
async def render_workflow_3d(workflow_id: str):
	"""Render workflow in 3D"""
	try:
		data = request.get_json() or {}
		
		mode = VisualizationMode(data.get('mode', 'standard_3d'))
		layout = LayoutAlgorithm(data.get('layout', 'force_directed'))
		
		# Render 3D visualization
		render_data = await visualization_3d.render_workflow_3d(workflow_id, mode, layout)
		
		return jsonify({
			"success": True,
			"data": render_data
		})
		
	except Exception as e:
		logger.error(f"3D rendering error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/3d/layouts', methods=['GET'])
async def get_available_layouts():
	"""Get available 3D layout algorithms"""
	try:
		layouts = [
			{
				"id": layout.value,
				"name": layout.value.replace("_", " ").title(),
				"description": f"{layout.value.replace('_', ' ').title()} layout algorithm"
			}
			for layout in LayoutAlgorithm
		]
		
		return jsonify({
			"success": True,
			"data": layouts
		})
		
	except Exception as e:
		logger.error(f"Get layouts error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/3d/animate/<workflow_id>/execution/<execution_id>', methods=['POST'])
async def animate_workflow_execution(workflow_id: str, execution_id: str):
	"""Animate workflow execution in 3D"""
	try:
		animation_id = await visualization_3d.animate_workflow_execution(workflow_id, execution_id)
		
		return jsonify({
			"success": True,
			"data": {
				"animation_id": animation_id
			}
		})
		
	except Exception as e:
		logger.error(f"Animation error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/3d/optimize', methods=['POST'])
async def optimize_rendering():
	"""Optimize 3D rendering performance"""
	try:
		data = request.get_json()
		node_count = data.get('node_count', 0)
		edge_count = data.get('edge_count', 0)
		
		visualization_3d.optimize_performance(node_count, edge_count)
		
		return jsonify({
			"success": True,
			"data": {
				"performance_mode": visualization_3d.performance_mode,
				"shadows_enabled": visualization_3d.shadows_enabled,
				"post_processing": visualization_3d.post_processing
			}
		})
		
	except Exception as e:
		logger.error(f"Optimization error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# VR/AR Interface Endpoints
# =============================================================================

@visualization_bp.route('/vr/session/start', methods=['POST'])
async def start_vr_session():
	"""Start VR session"""
	try:
		data = request.get_json()
		user_id = data.get('user_id') or session.get('user_id')
		platform = VRPlatform(data.get('platform', 'webxr'))
		config = data.get('config', {})
		
		if not user_id:
			return jsonify({
				"success": False,
				"error": "User ID required"
			}), 400
		
		session_id = await vr_ar_interface.start_vr_session(user_id, platform, config)
		
		return jsonify({
			"success": True,
			"data": {
				"session_id": session_id,
				"platform": platform.value
			}
		})
		
	except Exception as e:
		logger.error(f"VR session start error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/ar/session/start', methods=['POST'])
async def start_ar_session():
	"""Start AR session"""
	try:
		data = request.get_json()
		user_id = data.get('user_id') or session.get('user_id')
		platform = ARPlatform(data.get('platform', 'webxr_ar'))
		config = data.get('config', {})
		
		if not user_id:
			return jsonify({
				"success": False,
				"error": "User ID required"
			}), 400
		
		session_id = await vr_ar_interface.start_ar_session(user_id, platform, config)
		
		return jsonify({
			"success": True,
			"data": {
				"session_id": session_id,
				"platform": platform.value
			}
		})
		
	except Exception as e:
		logger.error(f"AR session start error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/vr/<session_id>/render/<workflow_id>', methods=['GET'])
async def render_workflow_vr(session_id: str, workflow_id: str):
	"""Render workflow in VR"""
	try:
		render_data = await vr_ar_interface.render_workflow_vr(session_id, workflow_id)
		
		return jsonify({
			"success": True,
			"data": render_data
		})
		
	except Exception as e:
		logger.error(f"VR rendering error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/ar/<session_id>/render/<workflow_id>', methods=['GET'])
async def render_workflow_ar(session_id: str, workflow_id: str):
	"""Render workflow in AR"""
	try:
		render_data = await vr_ar_interface.render_workflow_ar(session_id, workflow_id)
		
		return jsonify({
			"success": True,
			"data": render_data
		})
		
	except Exception as e:
		logger.error(f"AR rendering error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/vr/<session_id>/interact', methods=['POST'])
async def process_vr_interaction(session_id: str):
	"""Process VR interaction"""
	try:
		interaction_data = request.get_json()
		
		result = await vr_ar_interface.process_vr_interaction(session_id, interaction_data)
		
		return jsonify({
			"success": True,
			"data": result
		})
		
	except Exception as e:
		logger.error(f"VR interaction error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/voice/command', methods=['POST'])
async def process_voice_command():
	"""Process voice command"""
	try:
		# Get audio data (would normally be binary)
		data = request.get_json()
		audio_data = data.get('audio_data', b'')  # Mock audio data
		session_context = data.get('context', {})
		
		result = await vr_ar_interface.voice_processor.process_voice_command(
			audio_data, session_context
		)
		
		return jsonify({
			"success": True,
			"data": result
		})
		
	except Exception as e:
		logger.error(f"Voice command error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/haptic/trigger', methods=['POST'])
async def trigger_haptic_feedback():
	"""Trigger haptic feedback"""
	try:
		data = request.get_json()
		device_id = data.get('device_id')
		pattern_name = data.get('pattern')
		intensity = data.get('intensity', 1.0)
		
		success = await vr_ar_interface.haptic_system.trigger_haptic_feedback(
			device_id, pattern_name, intensity
		)
		
		return jsonify({
			"success": success,
			"data": {
				"device_id": device_id,
				"pattern": pattern_name,
				"triggered": success
			}
		})
		
	except Exception as e:
		logger.error(f"Haptic feedback error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Collaboration Endpoints
# =============================================================================

@visualization_bp.route('/collaboration/session/create', methods=['POST'])
async def create_collaboration_session():
	"""Create collaboration session"""
	try:
		data = request.get_json()
		workflow_id = data.get('workflow_id')
		user_id = data.get('user_id') or session.get('user_id')
		config = data.get('config', {})
		
		if not workflow_id or not user_id:
			return jsonify({
				"success": False,
				"error": "workflow_id and user_id are required"
			}), 400
		
		session_id = await advanced_collaboration.create_collaboration_session(
			workflow_id, user_id, config
		)
		
		return jsonify({
			"success": True,
			"data": {
				"session_id": session_id,
				"workflow_id": workflow_id
			}
		})
		
	except Exception as e:
		logger.error(f"Create collaboration session error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/collaboration/session/<session_id>/join', methods=['POST'])
async def join_collaboration_session(session_id: str):
	"""Join collaboration session"""
	try:
		data = request.get_json()
		user_id = data.get('user_id') or session.get('user_id')
		user_info = data.get('user_info', {})
		
		if not user_id:
			return jsonify({
				"success": False,
				"error": "user_id is required"
			}), 400
		
		success = await advanced_collaboration.join_collaboration_session(
			session_id, user_id, user_info
		)
		
		return jsonify({
			"success": success,
			"data": {
				"session_id": session_id,
				"user_id": user_id,
				"joined": success
			}
		})
		
	except Exception as e:
		logger.error(f"Join collaboration session error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/collaboration/session/<session_id>/leave', methods=['POST'])
async def leave_collaboration_session(session_id: str):
	"""Leave collaboration session"""
	try:
		data = request.get_json()
		user_id = data.get('user_id') or session.get('user_id')
		
		if not user_id:
			return jsonify({
				"success": False,
				"error": "user_id is required"
			}), 400
		
		success = await advanced_collaboration.leave_collaboration_session(
			session_id, user_id
		)
		
		return jsonify({
			"success": success,
			"data": {
				"session_id": session_id,
				"user_id": user_id,
				"left": success
			}
		})
		
	except Exception as e:
		logger.error(f"Leave collaboration session error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/collaboration/session/<session_id>/presence', methods=['POST'])
async def update_user_presence(session_id: str):
	"""Update user presence"""
	try:
		data = request.get_json()
		user_id = data.get('user_id') or session.get('user_id')
		presence_data = data.get('presence', {})
		
		if not user_id:
			return jsonify({
				"success": False,
				"error": "user_id is required"
			}), 400
		
		success = await advanced_collaboration.update_user_presence(
			session_id, user_id, presence_data
		)
		
		return jsonify({
			"success": success,
			"data": {
				"session_id": session_id,
				"user_id": user_id,
				"updated": success
			}
		})
		
	except Exception as e:
		logger.error(f"Update presence error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/collaboration/session/<session_id>/operate', methods=['POST'])
async def apply_collaborative_operation():
	"""Apply collaborative operation"""
	try:
		data = request.get_json()
		session_id = request.view_args['session_id']
		user_id = data.get('user_id') or session.get('user_id')
		operation_data = data.get('operation', {})
		
		if not user_id:
			return jsonify({
				"success": False,
				"error": "user_id is required"
			}), 400
		
		success, conflicts = await advanced_collaboration.apply_collaborative_change(
			session_id, user_id, operation_data
		)
		
		return jsonify({
			"success": success,
			"data": {
				"session_id": session_id,
				"operation_applied": success,
				"conflicts": conflicts
			}
		})
		
	except Exception as e:
		logger.error(f"Collaborative operation error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/collaboration/session/<session_id>/status', methods=['GET'])
async def get_session_status(session_id: str):
	"""Get collaboration session status"""
	try:
		status = await advanced_collaboration.get_session_status(session_id)
		
		if status:
			return jsonify({
				"success": True,
				"data": status
			})
		else:
			return jsonify({
				"success": False,
				"error": "Session not found"
			}), 404
		
	except Exception as e:
		logger.error(f"Get session status error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/collaboration/session/<session_id>/voice/start', methods=['POST'])
async def start_voice_channel(session_id: str):
	"""Start voice communication channel"""
	try:
		data = request.get_json()
		user_id = data.get('user_id') or session.get('user_id')
		config = data.get('config', {})
		
		if not user_id:
			return jsonify({
				"success": False,
				"error": "user_id is required"
			}), 400
		
		channel_id = await advanced_collaboration.start_voice_channel(
			session_id, user_id, config
		)
		
		return jsonify({
			"success": True,
			"data": {
				"channel_id": channel_id,
				"session_id": session_id
			}
		})
		
	except Exception as e:
		logger.error(f"Start voice channel error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/collaboration/session/<session_id>/screen-share/start', methods=['POST'])
async def start_screen_sharing(session_id: str):
	"""Start screen sharing"""
	try:
		data = request.get_json()
		user_id = data.get('user_id') or session.get('user_id')
		config = data.get('config', {})
		
		if not user_id:
			return jsonify({
				"success": False,
				"error": "user_id is required"
			}), 400
		
		sharing_id = await advanced_collaboration.start_screen_sharing(
			session_id, user_id, config
		)
		
		return jsonify({
			"success": True,
			"data": {
				"sharing_id": sharing_id,
				"session_id": session_id
			}
		})
		
	except Exception as e:
		logger.error(f"Start screen sharing error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/collaboration/user/<user_id>/sessions', methods=['GET'])
async def get_user_sessions(user_id: str):
	"""Get user's collaboration sessions"""
	try:
		sessions = await advanced_collaboration.get_user_sessions(user_id)
		
		return jsonify({
			"success": True,
			"data": {
				"user_id": user_id,
				"sessions": sessions
			}
		})
		
	except Exception as e:
		logger.error(f"Get user sessions error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Spatial Computing Endpoints
# =============================================================================

@visualization_bp.route('/spatial/workspace/create', methods=['POST'])
async def create_spatial_workspace():
	"""Create spatial computing workspace"""
	try:
		data = request.get_json()
		workflow_id = data.get('workflow_id')
		config = data.get('config', {})
		
		if not workflow_id:
			return jsonify({
				"success": False,
				"error": "workflow_id is required"
			}), 400
		
		# Create spatial workspace configuration
		workspace = {
			"id": str(uuid7str()),
			"workflow_id": workflow_id,
			"type": "spatial_computing",
			"dimensions": config.get("dimensions", {"width": 100, "height": 100, "depth": 100}),
			"origin": config.get("origin", {"x": 0, "y": 0, "z": 0}),
			"tracking_space": config.get("tracking_space", "room_scale"),
			"interaction_modes": config.get("interaction_modes", ["hand_tracking", "voice", "gaze"]),
			"physics_enabled": config.get("physics_enabled", True),
			"created_at": datetime.utcnow().isoformat()
		}
		
		return jsonify({
			"success": True,
			"data": workspace
		})
		
	except Exception as e:
		logger.error(f"Create spatial workspace error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/spatial/gesture/recognize', methods=['POST'])
async def recognize_gesture():
	"""Recognize spatial gesture"""
	try:
		data = request.get_json()
		hand_data = data.get('hand_data', {})
		
		gestures = await vr_ar_interface.gesture_recognizer.recognize_gesture(hand_data)
		
		return jsonify({
			"success": True,
			"data": {
				"gestures": gestures,
				"timestamp": datetime.utcnow().isoformat()
			}
		})
		
	except Exception as e:
		logger.error(f"Gesture recognition error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Performance and Analytics Endpoints
# =============================================================================

@visualization_bp.route('/analytics/performance', methods=['GET'])
async def get_performance_analytics():
	"""Get visualization performance analytics"""
	try:
		analytics = {
			"rendering": {
				"engine": visualization_3d.rendering_engine.value,
				"performance_mode": visualization_3d.performance_mode,
				"shadows_enabled": visualization_3d.shadows_enabled,
				"post_processing": visualization_3d.post_processing,
				"physics_enabled": visualization_3d.physics_enabled
			},
			"collaboration": {
				"active_sessions": len(advanced_collaboration.sessions),
				"total_users": sum(len(session.users) for session in advanced_collaboration.sessions.values()),
				"voice_channels": len(advanced_collaboration.voice_channels),
				"screen_sharing": len(advanced_collaboration.screen_sharing_sessions)
			},
			"vr_ar": {
				"vr_sessions": len(vr_ar_interface.vr_sessions),
				"ar_sessions": len(vr_ar_interface.ar_sessions),
				"hand_tracking_enabled": vr_ar_interface.hand_tracking_enabled,
				"voice_commands_enabled": vr_ar_interface.voice_commands_enabled,
				"haptic_feedback_enabled": vr_ar_interface.haptic_feedback_enabled
			},
			"timestamp": datetime.utcnow().isoformat()
		}
		
		return jsonify({
			"success": True,
			"data": analytics
		})
		
	except Exception as e:
		logger.error(f"Performance analytics error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@visualization_bp.route('/health', methods=['GET'])
async def health_check():
	"""Advanced visualization health check"""
	try:
		health_status = {
			"status": "healthy",
			"timestamp": datetime.utcnow().isoformat(),
			"services": {
				"3d_visualization": "healthy",
				"vr_interface": "healthy",
				"ar_interface": "healthy",
				"collaboration": "healthy",
				"spatial_computing": "healthy"
			},
			"metrics": {
				"active_sessions": len(advanced_collaboration.sessions),
				"vr_sessions": len(vr_ar_interface.vr_sessions),
				"ar_sessions": len(vr_ar_interface.ar_sessions)
			}
		}
		
		return jsonify(health_status)
		
	except Exception as e:
		return jsonify({
			"status": "unhealthy",
			"error": str(e),
			"timestamp": datetime.utcnow().isoformat()
		}), 500


# =============================================================================
# Flask-AppBuilder Views
# =============================================================================

class AdvancedVisualizationView(BaseView):
	"""Advanced visualization management view"""
	
	route_base = "/visualization"
	
	@expose("/3d-viewer")
	@has_access
	def viewer_3d(self):
		"""3D workflow viewer"""
		return self.render_template(
			"visualization/3d_viewer.html",
			title="3D Workflow Viewer"
		)
	
	@expose("/vr-interface")
	@has_access
	def vr_interface(self):
		"""VR interface"""
		return self.render_template(
			"visualization/vr_interface.html",
			title="VR Workflow Interface"
		)
	
	@expose("/ar-interface")
	@has_access
	def ar_interface(self):
		"""AR interface"""
		return self.render_template(
			"visualization/ar_interface.html",
			title="AR Workflow Interface"
		)
	
	@expose("/collaboration")
	@has_access
	def collaboration_hub(self):
		"""Collaboration hub"""
		return self.render_template(
			"visualization/collaboration.html",
			title="Collaboration Hub"
		)
	
	@expose("/spatial-computing")
	@has_access
	def spatial_computing(self):
		"""Spatial computing interface"""
		return self.render_template(
			"visualization/spatial_computing.html",
			title="Spatial Computing"
		)


def register_visualization_views(appbuilder):
	"""Register visualization views with Flask-AppBuilder"""
	appbuilder.add_view(
		AdvancedVisualizationView,
		"3D Viewer",
		icon="fa-cube",
		category="Visualization",
		category_icon="fa-eye"
	)


def register_visualization_routes(app):
	"""Register visualization routes with Flask app"""
	app.register_blueprint(visualization_bp)