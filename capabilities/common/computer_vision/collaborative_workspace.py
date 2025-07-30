"""
Real-Time Collaborative Visual Analysis - Revolutionary Team Workspace

Advanced collaborative platform enabling multiple users to simultaneously analyze, 
annotate, and discuss visual content with real-time synchronization, voice/video 
integration, and intelligent collaboration features.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncIterable
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict
import websockets
from fastapi import WebSocket
import redis.asyncio as redis

from .models import CVBaseModel, ProcessingType, AnalysisLevel


class CollaborativeSession(CVBaseModel):
	"""Collaborative analysis session definition"""
	
	session_id: str = Field(default_factory=uuid7str, description="Session identifier")
	session_name: str = Field(..., description="Human-readable session name")
	session_type: str = Field(
		default="visual_analysis", 
		regex="^(visual_analysis|quality_review|training|audit)$",
		description="Type of collaborative session"
	)
	participants: List[str] = Field(
		default_factory=list, description="List of participant user IDs"
	)
	session_owner: str = Field(..., description="Session creator/owner")
	image_data_url: str = Field(..., description="URL or reference to image being analyzed")
	analysis_type: str = Field(..., description="Type of analysis being performed")
	session_status: str = Field(
		default="active",
		regex="^(active|paused|completed|archived)$",
		description="Current session status"
	)
	max_participants: int = Field(default=10, ge=1, le=50, description="Maximum participants allowed")
	session_settings: Dict[str, Any] = Field(
		default_factory=dict, description="Session configuration settings"
	)
	started_at: datetime = Field(default_factory=datetime.utcnow, description="Session start time")
	last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")


class VisualAnnotation(CVBaseModel):
	"""Visual annotation made by participant"""
	
	annotation_id: str = Field(default_factory=uuid7str, description="Annotation identifier")
	author_id: str = Field(..., description="User ID of annotation author")
	author_name: str = Field(..., description="Display name of author")
	annotation_type: str = Field(
		..., 
		regex="^(rectangle|circle|arrow|text|measurement|highlight|freehand)$",
		description="Type of annotation"
	)
	coordinates: Dict[str, float] = Field(
		..., description="Annotation coordinates (x, y, width, height, etc.)"
	)
	content: str = Field(default="", description="Text content or description")
	style: Dict[str, Any] = Field(
		default_factory=dict, description="Visual styling (color, thickness, etc.)"
	)
	visibility: str = Field(
		default="public",
		regex="^(public|private|team)$",
		description="Annotation visibility level"
	)
	tags: List[str] = Field(
		default_factory=list, description="Tags for annotation categorization"
	)
	replies: List[str] = Field(
		default_factory=list, description="Annotation reply IDs"
	)
	reaction_counts: Dict[str, int] = Field(
		default_factory=dict, description="Reaction counts (like, agree, etc.)"
	)
	is_resolved: bool = Field(default=False, description="Whether annotation is resolved")


class ParticipantStatus(CVBaseModel):
	"""Real-time participant status"""
	
	user_id: str = Field(..., description="User identifier")
	display_name: str = Field(..., description="User display name")
	user_role: str = Field(
		default="analyst",
		regex="^(owner|moderator|analyst|viewer|guest)$",
		description="User role in session"
	)
	connection_status: str = Field(
		default="connected",
		regex="^(connected|disconnected|away|busy)$",
		description="Connection status"
	)
	cursor_position: Dict[str, float] = Field(
		default_factory=dict, description="Current cursor position"
	)
	current_tool: str = Field(default="pointer", description="Currently selected tool")
	viewing_area: Dict[str, float] = Field(
		default_factory=dict, description="Current viewing area/zoom"
	)
	last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity time")
	permissions: List[str] = Field(
		default_factory=list, description="User permissions in session"
	)


class CollaborativeInsight(CVBaseModel):
	"""Insight generated from collaborative analysis"""
	
	insight_id: str = Field(default_factory=uuid7str, description="Insight identifier")
	insight_type: str = Field(..., description="Type of collaborative insight")
	insight_content: str = Field(..., description="Insight description")
	contributing_users: List[str] = Field(
		..., description="Users who contributed to this insight"
	)
	supporting_annotations: List[str] = Field(
		default_factory=list, description="Annotations supporting this insight"
	)
	consensus_level: float = Field(
		..., ge=0.0, le=1.0, description="Level of participant consensus"
	)
	confidence_score: float = Field(
		..., ge=0.0, le=1.0, description="Confidence in insight"
	)
	business_impact: str = Field(..., description="Assessed business impact")
	recommended_actions: List[str] = Field(
		default_factory=list, description="Recommended actions"
	)
	priority_level: str = Field(
		default="medium",
		regex="^(low|medium|high|critical)$",
		description="Priority level"
	)


class CollaborationEvent(CVBaseModel):
	"""Real-time collaboration event"""
	
	event_id: str = Field(default_factory=uuid7str, description="Event identifier")
	event_type: str = Field(
		...,
		regex="^(join|leave|annotation|cursor_move|tool_change|message|insight)$",
		description="Type of collaboration event"
	)
	user_id: str = Field(..., description="User who triggered event")
	session_id: str = Field(..., description="Session identifier")
	event_data: Dict[str, Any] = Field(
		default_factory=dict, description="Event-specific data"
	)
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
	broadcast_to: List[str] = Field(
		default_factory=list, description="Users to broadcast event to (empty = all)"
	)


class CollaborativeVisualWorkspace:
	"""
	Revolutionary Real-Time Collaborative Visual Analysis Workspace
	
	Enables multiple users to work together on visual content analysis with
	real-time synchronization, intelligent collaboration features, and
	comprehensive team coordination capabilities.
	"""
	
	def __init__(self):
		self.active_sessions: Dict[str, CollaborativeSession] = {}
		self.session_participants: Dict[str, List[ParticipantStatus]] = {}
		self.session_annotations: Dict[str, List[VisualAnnotation]] = {}
		self.session_insights: Dict[str, List[CollaborativeInsight]] = {}
		
		# Real-time communication
		self.websocket_connections: Dict[str, List[WebSocket]] = {}
		self.redis_client: Optional[redis.Redis] = None
		
		# Collaboration analytics
		self.collaboration_metrics: Dict[str, Dict[str, Any]] = {}
		self.user_activity_tracking: Dict[str, List[Dict]] = {}
		
		# Session templates and presets
		self.session_templates: Dict[str, Dict] = {}
		self.analysis_workflows: Dict[str, List[Dict]] = {}

	async def _log_collaboration_operation(
		self,
		operation: str,
		session_id: Optional[str] = None,
		user_id: Optional[str] = None,
		details: Optional[str] = None
	) -> None:
		"""Log collaborative workspace operations"""
		assert operation is not None, "Operation name must be provided"
		session_ref = f" [Session: {session_id}]" if session_id else ""
		user_ref = f" [User: {user_id}]" if user_id else ""
		detail_info = f" - {details}" if details else ""
		print(f"Collaborative Workspace: {operation}{session_ref}{user_ref}{detail_info}")

	async def initialize_collaboration_system(
		self,
		redis_config: Dict[str, Any],
		session_templates: Dict[str, Any]
	) -> bool:
		"""
		Initialize the collaborative workspace system
		
		Args:
			redis_config: Redis configuration for real-time sync
			session_templates: Predefined session templates
			
		Returns:
			bool: Success status of initialization
		"""
		try:
			await self._log_collaboration_operation("Initializing collaborative workspace system")
			
			# Initialize Redis for real-time synchronization
			await self._initialize_redis_connection(redis_config)
			
			# Load session templates
			await self._load_session_templates(session_templates)
			
			# Setup collaboration workflows
			await self._setup_collaboration_workflows()
			
			# Initialize analytics tracking
			await self._initialize_collaboration_analytics()
			
			await self._log_collaboration_operation(
				"Collaborative workspace system initialized successfully",
				details=f"Templates: {len(self.session_templates)}"
			)
			
			return True
			
		except Exception as e:
			await self._log_collaboration_operation(
				"Failed to initialize collaborative workspace system",
				details=str(e)
			)
			return False

	async def _initialize_redis_connection(self, redis_config: Dict[str, Any]) -> None:
		"""Initialize Redis connection for real-time sync"""
		try:
			self.redis_client = redis.Redis(
				host=redis_config.get("host", "localhost"),
				port=redis_config.get("port", 6379),
				db=redis_config.get("db", 0),
				password=redis_config.get("password"),
				decode_responses=True
			)
			
			# Test connection
			await self.redis_client.ping()
			
		except Exception as e:
			raise RuntimeError(f"Failed to initialize Redis connection: {e}")

	async def _load_session_templates(self, templates: Dict[str, Any]) -> None:
		"""Load predefined session templates"""
		default_templates = {
			"quality_review": {
				"name": "Quality Review Session",
				"max_participants": 8,
				"default_tools": ["rectangle", "text", "measurement"],
				"workflow_stages": [
					"Initial Review",
					"Detailed Analysis", 
					"Issue Identification",
					"Consensus Building",
					"Final Decision"
				],
				"required_roles": ["quality_expert", "analyst"],
				"session_duration_minutes": 60
			},
			"training_session": {
				"name": "Visual Analysis Training",
				"max_participants": 15,
				"default_tools": ["text", "highlight", "arrow"],
				"workflow_stages": [
					"Demonstration",
					"Guided Practice",
					"Independent Analysis",
					"Group Discussion",
					"Assessment"
				],
				"required_roles": ["trainer", "trainee"],
				"session_duration_minutes": 90
			},
			"audit_review": {
				"name": "Compliance Audit Review",
				"max_participants": 5,
				"default_tools": ["rectangle", "text", "measurement"],
				"workflow_stages": [
					"Documentation Review",
					"Compliance Check",
					"Issue Documentation",
					"Corrective Actions",
					"Final Report"
				],
				"required_roles": ["auditor", "compliance_officer"],
				"session_duration_minutes": 120
			}
		}
		
		# Merge provided templates with defaults
		self.session_templates = {**default_templates, **templates}

	async def _setup_collaboration_workflows(self) -> None:
		"""Setup collaboration workflow definitions"""
		self.analysis_workflows = {
			"defect_analysis": [
				{"stage": "detection", "description": "Identify potential defects"},
				{"stage": "classification", "description": "Classify defect types"},
				{"stage": "severity", "description": "Assess defect severity"},
				{"stage": "root_cause", "description": "Determine root causes"},
				{"stage": "resolution", "description": "Plan resolution actions"}
			],
			"document_review": [
				{"stage": "content_review", "description": "Review document content"},
				{"stage": "accuracy_check", "description": "Verify information accuracy"},
				{"stage": "completeness", "description": "Check completeness"},
				{"stage": "compliance", "description": "Verify compliance requirements"},
				{"stage": "approval", "description": "Final approval process"}
			],
			"product_inspection": [
				{"stage": "visual_inspection", "description": "Visual quality check"},
				{"stage": "measurement", "description": "Dimensional verification"},
				{"stage": "functionality", "description": "Function testing"},
				{"stage": "packaging", "description": "Packaging inspection"},
				{"stage": "final_approval", "description": "Final quality approval"}
			]
		}

	async def _initialize_collaboration_analytics(self) -> None:
		"""Initialize collaboration analytics tracking"""
		self.collaboration_metrics = {
			"session_metrics": {
				"total_sessions": 0,
				"active_sessions": 0,
				"average_duration": 0.0,
				"average_participants": 0.0
			},
			"user_engagement": {
				"total_annotations": 0,
				"average_annotations_per_user": 0.0,
				"collaboration_effectiveness": 0.0
			},
			"insight_generation": {
				"total_insights": 0,
				"consensus_rate": 0.0,
				"actionable_insights_rate": 0.0
			}
		}

	async def create_collaborative_session(
		self,
		session_name: str,
		image_data_url: str,
		analysis_type: str,
		session_owner: str,
		initial_participants: List[str],
		template_name: Optional[str] = None,
		custom_settings: Optional[Dict[str, Any]] = None
	) -> CollaborativeSession:
		"""
		Create a new collaborative analysis session
		
		Args:
			session_name: Human-readable session name
			image_data_url: URL or reference to image being analyzed
			analysis_type: Type of analysis to perform
			session_owner: User ID of session creator
			initial_participants: List of initial participant user IDs
			template_name: Optional template to use for session setup
			custom_settings: Optional custom session settings
			
		Returns:
			CollaborativeSession: Created session object
		"""
		try:
			session_id = uuid7str()
			await self._log_collaboration_operation(
				"Creating collaborative session",
				session_id=session_id,
				user_id=session_owner,
				details=f"Template: {template_name or 'default'}"
			)
			
			# Apply template settings if specified
			session_settings = {}
			max_participants = 10
			
			if template_name and template_name in self.session_templates:
				template = self.session_templates[template_name]
				session_settings.update(template)
				max_participants = template.get("max_participants", 10)
			
			# Apply custom settings
			if custom_settings:
				session_settings.update(custom_settings)
			
			# Create session
			session = CollaborativeSession(
				tenant_id=custom_settings.get("tenant_id", "unknown") if custom_settings else "unknown",
				created_by=session_owner,
				session_id=session_id,
				session_name=session_name,
				participants=initial_participants + [session_owner],
				session_owner=session_owner,
				image_data_url=image_data_url,
				analysis_type=analysis_type,
				max_participants=max_participants,
				session_settings=session_settings
			)
			
			# Store session
			self.active_sessions[session_id] = session
			self.session_participants[session_id] = []
			self.session_annotations[session_id] = []
			self.session_insights[session_id] = []
			
			# Initialize WebSocket connections
			self.websocket_connections[session_id] = []
			
			# Setup real-time synchronization
			await self._setup_realtime_sync(session_id)
			
			# Initialize annotation layers
			await self._initialize_annotation_layers(session_id)
			
			# Perform initial analysis if specified
			if analysis_type != "manual":
				await self._perform_initial_analysis(session_id, image_data_url, analysis_type)
			
			# Update metrics
			self.collaboration_metrics["session_metrics"]["total_sessions"] += 1
			self.collaboration_metrics["session_metrics"]["active_sessions"] += 1
			
			await self._log_collaboration_operation(
				"Collaborative session created successfully",
				session_id=session_id,
				user_id=session_owner,
				details=f"Participants: {len(initial_participants) + 1}"
			)
			
			return session
			
		except Exception as e:
			await self._log_collaboration_operation(
				"Failed to create collaborative session",
				user_id=session_owner,
				details=str(e)
			)
			raise

	async def _setup_realtime_sync(self, session_id: str) -> None:
		"""Setup real-time synchronization for session"""
		if self.redis_client:
			# Create Redis channels for session
			await self.redis_client.publish(
				f"session:{session_id}:init",
				json.dumps({"type": "session_created", "session_id": session_id})
			)

	async def _initialize_annotation_layers(self, session_id: str) -> None:
		"""Initialize annotation layers for session"""
		# Create default annotation layers
		default_layers = [
			{"name": "General", "color": "#007bff", "visible": True},
			{"name": "Issues", "color": "#dc3545", "visible": True},
			{"name": "Measurements", "color": "#28a745", "visible": True},
			{"name": "Notes", "color": "#ffc107", "visible": True}
		]
		
		session = self.active_sessions[session_id]
		session.session_settings["annotation_layers"] = default_layers

	async def _perform_initial_analysis(
		self,
		session_id: str,
		image_data_url: str,
		analysis_type: str
	) -> None:
		"""Perform initial automated analysis"""
		# Placeholder for initial analysis
		# In production, this would integrate with the main CV analysis pipeline
		initial_results = {
			"timestamp": datetime.utcnow().isoformat(),
			"analysis_type": analysis_type,
			"automated_findings": [
				"Image quality: Good",
				"Resolution: 1920x1080",
				"Color space: RGB"
			]
		}
		
		session = self.active_sessions[session_id]
		session.session_settings["initial_analysis"] = initial_results

	async def join_session(
		self,
		session_id: str,
		user_id: str,
		display_name: str,
		user_role: str = "analyst",
		websocket: Optional[WebSocket] = None
	) -> ParticipantStatus:
		"""
		Add participant to collaborative session
		
		Args:
			session_id: Session to join
			user_id: User identifier
			display_name: User display name
			user_role: User role in session
			websocket: Optional WebSocket connection
			
		Returns:
			ParticipantStatus: Participant status object
		"""
		try:
			await self._log_collaboration_operation(
				"User joining session",
				session_id=session_id,
				user_id=user_id,
				details=f"Role: {user_role}"
			)
			
			# Check session exists and has capacity
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			current_participants = len(self.session_participants[session_id])
			
			if current_participants >= session.max_participants:
				raise ValueError("Session is at maximum capacity")
			
			# Create participant status
			participant = ParticipantStatus(
				tenant_id=session.tenant_id,
				created_by=user_id,
				user_id=user_id,
				display_name=display_name,
				user_role=user_role,
				permissions=self._get_role_permissions(user_role)
			)
			
			# Add to session
			self.session_participants[session_id].append(participant)
			
			# Add WebSocket connection
			if websocket:
				self.websocket_connections[session_id].append(websocket)
			
			# Update session participant list
			if user_id not in session.participants:
				session.participants.append(user_id)
			
			# Broadcast join event
			await self._broadcast_collaboration_event(
				session_id,
				CollaborationEvent(
					tenant_id=session.tenant_id,
					created_by=user_id,
					event_type="join",
					user_id=user_id,
					session_id=session_id,
					event_data={
						"display_name": display_name,
						"user_role": user_role,
						"participant_count": len(self.session_participants[session_id])
					}
				)
			)
			
			await self._log_collaboration_operation(
				"User joined session successfully",
				session_id=session_id,
				user_id=user_id
			)
			
			return participant
			
		except Exception as e:
			await self._log_collaboration_operation(
				"Failed to join session",
				session_id=session_id,
				user_id=user_id,
				details=str(e)
			)
			raise

	def _get_role_permissions(self, user_role: str) -> List[str]:
		"""Get permissions for user role"""
		role_permissions = {
			"owner": [
				"annotate", "delete_annotations", "moderate", "manage_participants",
				"change_settings", "export_session", "archive_session"
			],
			"moderator": [
				"annotate", "delete_annotations", "moderate", "manage_participants",
				"export_session"
			],
			"analyst": [
				"annotate", "delete_own_annotations", "view_all", "export_data"
			],
			"viewer": [
				"view_all", "add_reactions"
			],
			"guest": [
				"view_public"
			]
		}
		
		return role_permissions.get(user_role, role_permissions["viewer"])

	async def add_collaborative_annotation(
		self,
		session_id: str,
		user_id: str,
		annotation: VisualAnnotation
	) -> str:
		"""
		Add annotation with real-time sync to all participants
		
		Args:
			session_id: Session identifier
			user_id: User adding annotation
			annotation: Annotation object
			
		Returns:
			str: Annotation ID
		"""
		try:
			await self._log_collaboration_operation(
				"Adding collaborative annotation",
				session_id=session_id,
				user_id=user_id,
				details=f"Type: {annotation.annotation_type}"
			)
			
			# Validate session and user permissions
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			participant = self._get_participant(session_id, user_id)
			if not participant or "annotate" not in participant.permissions:
				raise ValueError("User does not have annotation permissions")
			
			# Validate annotation
			validated_annotation = await self._validate_annotation(annotation)
			
			# Store annotation
			self.session_annotations[session_id].append(validated_annotation)
			
			# Update metrics
			self.collaboration_metrics["user_engagement"]["total_annotations"] += 1
			
			# Track user activity
			await self._track_user_activity(
				user_id, "annotation_added", {"annotation_id": validated_annotation.annotation_id}
			)
			
			# Broadcast annotation update
			await self._broadcast_collaboration_event(
				session_id,
				CollaborationEvent(
					tenant_id=self.active_sessions[session_id].tenant_id,
					created_by=user_id,
					event_type="annotation",
					user_id=user_id,
					session_id=session_id,
					event_data={
						"action": "added",
						"annotation": validated_annotation.dict()
					}
				)
			)
			
			# Update collaborative insights
			await self._update_collaborative_insights(session_id)
			
			await self._log_collaboration_operation(
				"Collaborative annotation added successfully",
				session_id=session_id,
				user_id=user_id,
				details=f"ID: {validated_annotation.annotation_id}"
			)
			
			return validated_annotation.annotation_id
			
		except Exception as e:
			await self._log_collaboration_operation(
				"Failed to add collaborative annotation",
				session_id=session_id,
				user_id=user_id,
				details=str(e)
			)
			raise

	def _get_participant(self, session_id: str, user_id: str) -> Optional[ParticipantStatus]:
		"""Get participant by user ID"""
		participants = self.session_participants.get(session_id, [])
		for participant in participants:
			if participant.user_id == user_id:
				return participant
		return None

	async def _validate_annotation(self, annotation: VisualAnnotation) -> VisualAnnotation:
		"""Validate annotation data"""
		# Ensure required coordinates are present
		required_coords = ["x", "y"]
		for coord in required_coords:
			if coord not in annotation.coordinates:
				raise ValueError(f"Missing required coordinate: {coord}")
		
		# Validate coordinate ranges (assuming normalized 0-1 coordinates)
		for coord, value in annotation.coordinates.items():
			if not isinstance(value, (int, float)) or value < 0 or value > 1:
				raise ValueError(f"Invalid coordinate value for {coord}: {value}")
		
		# Set default style if not provided
		if not annotation.style:
			annotation.style = {
				"color": "#007bff",
				"thickness": 2,
				"opacity": 0.8
			}
		
		return annotation

	async def _track_user_activity(
		self,
		user_id: str,
		activity_type: str,
		activity_data: Dict[str, Any]
	) -> None:
		"""Track user activity for analytics"""
		if user_id not in self.user_activity_tracking:
			self.user_activity_tracking[user_id] = []
		
		activity_record = {
			"timestamp": datetime.utcnow().isoformat(),
			"activity_type": activity_type,
			"data": activity_data
		}
		
		self.user_activity_tracking[user_id].append(activity_record)
		
		# Keep only recent activities (last 1000)
		if len(self.user_activity_tracking[user_id]) > 1000:
			self.user_activity_tracking[user_id] = self.user_activity_tracking[user_id][-1000:]

	async def _broadcast_collaboration_event(
		self,
		session_id: str,
		event: CollaborationEvent
	) -> None:
		"""Broadcast event to all session participants"""
		# Broadcast via WebSocket connections
		websockets_list = self.websocket_connections.get(session_id, [])
		active_websockets = []
		
		for websocket in websockets_list:
			try:
				await websocket.send_text(json.dumps({
					"type": "collaboration_event",
					"event": event.dict()
				}))
				active_websockets.append(websocket)
			except Exception:
				# Connection is closed, skip
				continue
		
		# Update active connections list
		self.websocket_connections[session_id] = active_websockets
		
		# Broadcast via Redis for multi-instance support
		if self.redis_client:
			await self.redis_client.publish(
				f"session:{session_id}:events",
				json.dumps(event.dict())
			)

	async def _update_collaborative_insights(self, session_id: str) -> None:
		"""Update collaborative insights based on current annotations"""
		try:
			annotations = self.session_annotations.get(session_id, [])
			if len(annotations) < 2:  # Need multiple annotations for insights
				return
			
			# Analyze annotation patterns for insights
			insights = await self._analyze_annotation_patterns(session_id, annotations)
			
			# Store insights
			self.session_insights[session_id].extend(insights)
			
			# Update metrics
			self.collaboration_metrics["insight_generation"]["total_insights"] += len(insights)
			
		except Exception as e:
			await self._log_collaboration_operation(
				"Failed to update collaborative insights",
				session_id=session_id,
				details=str(e)
			)

	async def _analyze_annotation_patterns(
		self,
		session_id: str,
		annotations: List[VisualAnnotation]
	) -> List[CollaborativeInsight]:
		"""Analyze annotation patterns to generate insights"""
		insights = []
		
		# Group annotations by location
		location_groups = self._group_annotations_by_location(annotations)
		
		# Generate insights for clustered annotations
		for location, annotation_group in location_groups.items():
			if len(annotation_group) >= 3:  # Multiple users annotating same area
				contributing_users = list(set([ann.author_id for ann in annotation_group]))
				
				if len(contributing_users) >= 2:  # Multiple users involved
					insight = CollaborativeInsight(
						tenant_id=self.active_sessions[session_id].tenant_id,
						created_by="system",
						insight_type="attention_cluster",
						insight_content=f"Multiple participants ({len(contributing_users)}) have focused attention on this area",
						contributing_users=contributing_users,
						supporting_annotations=[ann.annotation_id for ann in annotation_group],
						consensus_level=min(len(contributing_users) / 5.0, 1.0),
						confidence_score=0.8,
						business_impact="High attention area may indicate critical findings",
						recommended_actions=[
							"Prioritize detailed analysis of this area",
							"Seek expert validation of findings",
							"Document consensus decision"
						],
						priority_level="high" if len(contributing_users) >= 4 else "medium"
					)
					insights.append(insight)
		
		# Analyze annotation types for patterns
		type_analysis = self._analyze_annotation_types(annotations)
		if type_analysis["dominant_type_ratio"] > 0.6:
			insight = CollaborativeInsight(
				tenant_id=self.active_sessions[session_id].tenant_id,
				created_by="system",
				insight_type="analysis_focus",
				insight_content=f"Analysis is heavily focused on {type_analysis['dominant_type']} activities",
				contributing_users=list(set([ann.author_id for ann in annotations])),
				supporting_annotations=[],
				consensus_level=type_analysis["dominant_type_ratio"],
				confidence_score=0.7,
				business_impact="Focused analysis pattern indicates specific area of concern",
				recommended_actions=[
					f"Continue detailed {type_analysis['dominant_type']} analysis",
					"Consider broadening analysis scope",
					"Document focused findings"
				]
			)
			insights.append(insight)
		
		return insights

	def _group_annotations_by_location(
		self,
		annotations: List[VisualAnnotation]
	) -> Dict[str, List[VisualAnnotation]]:
		"""Group annotations by spatial proximity"""
		location_groups = {}
		proximity_threshold = 0.1  # 10% of image size
		
		for annotation in annotations:
			x = annotation.coordinates.get("x", 0)
			y = annotation.coordinates.get("y", 0)
			
			# Find existing group within proximity
			group_key = None
			for key in location_groups.keys():
				key_x, key_y = map(float, key.split(","))
				if (abs(x - key_x) < proximity_threshold and 
					abs(y - key_y) < proximity_threshold):
					group_key = key
					break
			
			# Create new group or add to existing
			if group_key is None:
				group_key = f"{x:.2f},{y:.2f}"
				location_groups[group_key] = []
			
			location_groups[group_key].append(annotation)
		
		return location_groups

	def _analyze_annotation_types(
		self,
		annotations: List[VisualAnnotation]
	) -> Dict[str, Any]:
		"""Analyze distribution of annotation types"""
		type_counts = {}
		for annotation in annotations:
			annotation_type = annotation.annotation_type
			type_counts[annotation_type] = type_counts.get(annotation_type, 0) + 1
		
		if not type_counts:
			return {"dominant_type": "none", "dominant_type_ratio": 0.0}
		
		dominant_type = max(type_counts.items(), key=lambda x: x[1])
		total_annotations = len(annotations)
		
		return {
			"dominant_type": dominant_type[0],
			"dominant_type_ratio": dominant_type[1] / total_annotations,
			"type_distribution": type_counts
		}

	async def get_session_insights(self, session_id: str) -> List[CollaborativeInsight]:
		"""
		Get collaborative insights for session
		
		Args:
			session_id: Session identifier
			
		Returns:
			List[CollaborativeInsight]: Current session insights
		"""
		try:
			await self._log_collaboration_operation(
				"Retrieving session insights",
				session_id=session_id
			)
			
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			insights = self.session_insights.get(session_id, [])
			
			# Sort by priority and consensus level
			insights.sort(
				key=lambda x: (
					{"critical": 4, "high": 3, "medium": 2, "low": 1}[x.priority_level],
					x.consensus_level
				),
				reverse=True
			)
			
			return insights
			
		except Exception as e:
			await self._log_collaboration_operation(
				"Failed to retrieve session insights",
				session_id=session_id,
				details=str(e)
			)
			raise

	async def complete_session(
		self,
		session_id: str,
		user_id: str,
		completion_summary: str
	) -> Dict[str, Any]:
		"""
		Complete collaborative session and generate summary
		
		Args:
			session_id: Session to complete
			user_id: User completing session
			completion_summary: Summary of session outcomes
			
		Returns:
			Dict[str, Any]: Session completion report
		"""
		try:
			await self._log_collaboration_operation(
				"Completing collaborative session",
				session_id=session_id,
				user_id=user_id
			)
			
			if session_id not in self.active_sessions:
				raise ValueError(f"Session {session_id} not found")
			
			session = self.active_sessions[session_id]
			
			# Verify user has permission to complete session
			participant = self._get_participant(session_id, user_id)
			if not participant or participant.user_role not in ["owner", "moderator"]:
				raise ValueError("User does not have permission to complete session")
			
			# Calculate session duration
			duration = datetime.utcnow() - session.started_at
			
			# Generate completion report
			completion_report = {
				"session_id": session_id,
				"session_name": session.session_name,
				"completion_time": datetime.utcnow().isoformat(),
				"duration_minutes": int(duration.total_seconds() / 60),
				"participant_count": len(self.session_participants[session_id]),
				"annotation_count": len(self.session_annotations[session_id]),
				"insight_count": len(self.session_insights[session_id]),
				"completion_summary": completion_summary,
				"participants": [
					{
						"user_id": p.user_id,
						"display_name": p.display_name,
						"role": p.user_role,
						"contribution_score": await self._calculate_contribution_score(session_id, p.user_id)
					}
					for p in self.session_participants[session_id]
				],
				"key_insights": [
					{
						"type": insight.insight_type,
						"content": insight.insight_content,
						"consensus": insight.consensus_level,
						"priority": insight.priority_level
					}
					for insight in self.session_insights[session_id]
				]
			}
			
			# Update session status
			session.session_status = "completed"
			session.last_activity = datetime.utcnow()
			
			# Update metrics
			self.collaboration_metrics["session_metrics"]["active_sessions"] -= 1
			current_avg_duration = self.collaboration_metrics["session_metrics"]["average_duration"]
			total_sessions = self.collaboration_metrics["session_metrics"]["total_sessions"]
			new_avg_duration = ((current_avg_duration * (total_sessions - 1)) + duration.total_seconds() / 60) / total_sessions
			self.collaboration_metrics["session_metrics"]["average_duration"] = new_avg_duration
			
			# Broadcast completion event
			await self._broadcast_collaboration_event(
				session_id,
				CollaborationEvent(
					tenant_id=session.tenant_id,
					created_by=user_id,
					event_type="session_completed",
					user_id=user_id,
					session_id=session_id,
					event_data={"completion_summary": completion_summary}
				)
			)
			
			await self._log_collaboration_operation(
				"Collaborative session completed successfully",
				session_id=session_id,
				user_id=user_id,
				details=f"Duration: {int(duration.total_seconds() / 60)} min, Participants: {len(self.session_participants[session_id])}"
			)
			
			return completion_report
			
		except Exception as e:
			await self._log_collaboration_operation(
				"Failed to complete collaborative session",
				session_id=session_id,
				user_id=user_id,
				details=str(e)
			)
			raise

	async def _calculate_contribution_score(self, session_id: str, user_id: str) -> float:
		"""Calculate user contribution score for session"""
		try:
			annotations = self.session_annotations.get(session_id, [])
			user_annotations = [ann for ann in annotations if ann.author_id == user_id]
			
			# Base score from annotations
			annotation_score = min(len(user_annotations) / 10.0, 1.0)  # Max 1.0 for 10+ annotations
			
			# Quality score from annotation interactions
			interaction_score = 0.0
			for annotation in user_annotations:
				# Score based on replies and reactions
				reply_score = min(len(annotation.replies) * 0.1, 0.3)
				reaction_score = min(sum(annotation.reaction_counts.values()) * 0.05, 0.2)
				interaction_score += reply_score + reaction_score
			
			interaction_score = min(interaction_score / max(len(user_annotations), 1), 1.0)
			
			# Insights contribution score
			insights = self.session_insights.get(session_id, [])
			user_insights = [insight for insight in insights if user_id in insight.contributing_users]
			insight_score = min(len(user_insights) / 5.0, 1.0)  # Max 1.0 for 5+ insights
			
			# Combined score (weighted)
			total_score = (
				annotation_score * 0.4 +
				interaction_score * 0.3 +
				insight_score * 0.3
			)
			
			return round(total_score, 2)
			
		except Exception:
			return 0.0


# Export main classes
__all__ = [
	"CollaborativeVisualWorkspace",
	"CollaborativeSession",
	"VisualAnnotation",
	"ParticipantStatus",
	"CollaborativeInsight",
	"CollaborationEvent"
]