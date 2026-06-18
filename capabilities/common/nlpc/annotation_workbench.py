"""
APG NLP Collaborative Annotation Workbench

Enterprise-grade collaborative annotation system with real-time editing,
conflict resolution, quality assurance, and advanced annotation workflows.

Features:
- Real-time collaborative annotation with conflict resolution
- Multi-user annotation projects with role-based access
- Quality assurance workflows with inter-annotator agreement
- Advanced annotation schemas with validation
- Export capabilities for training data generation
- Integration with model training pipelines
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Callable, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
from contextlib import asynccontextmanager
from uuid_extensions import uuid7str

from .models import (
	AnnotationProject, TextAnnotation, NLPTaskType, LanguageCode,
	ProcessingRequest, ProcessingResult, TextDocument
)

# Configure logging
logger = logging.getLogger(__name__)

class AnnotationStatus(str, Enum):
	"""Annotation workflow status"""
	DRAFT = "draft"
	SUBMITTED = "submitted"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	REJECTED = "rejected"
	CONFLICTED = "conflicted"
	RESOLVED = "resolved"

class ProjectRole(str, Enum):
	"""Project role types"""
	PROJECT_MANAGER = "project_manager"
	SENIOR_ANNOTATOR = "senior_annotator"
	ANNOTATOR = "annotator"
	REVIEWER = "reviewer"
	QUALITY_ANALYST = "quality_analyst"
	OBSERVER = "observer"

class ConflictType(str, Enum):
	"""Types of annotation conflicts"""
	OVERLAPPING_ENTITIES = "overlapping_entities"
	DIFFERENT_LABELS = "different_labels"
	BOUNDARY_MISMATCH = "boundary_mismatch"
	MISSING_ANNOTATION = "missing_annotation"
	EXTRA_ANNOTATION = "extra_annotation"

@dataclass
class AnnotationConflict:
	"""Annotation conflict information"""
	conflict_id: str = field(default_factory=uuid7str)
	project_id: str = ""
	document_id: str = ""
	conflict_type: ConflictType = ConflictType.DIFFERENT_LABELS
	annotator_1: str = ""
	annotator_2: str = ""
	annotation_1: Optional[Dict[str, Any]] = None
	annotation_2: Optional[Dict[str, Any]] = None
	severity: str = "medium"  # low, medium, high, critical
	created_at: datetime = field(default_factory=datetime.utcnow)
	resolved_at: Optional[datetime] = None
	resolved_by: Optional[str] = None
	resolution: Optional[Dict[str, Any]] = None

@dataclass
class AnnotationSession:
	"""Real-time annotation session"""
	session_id: str = field(default_factory=uuid7str)
	project_id: str = ""
	document_id: str = ""
	annotator_id: str = ""
	started_at: datetime = field(default_factory=datetime.utcnow)
	last_activity: datetime = field(default_factory=datetime.utcnow)
	annotations_made: int = 0
	time_spent_seconds: int = 0
	active: bool = True

@dataclass
class QualityMetrics:
	"""Annotation quality metrics"""
	project_id: str
	annotator_id: str
	total_annotations: int = 0
	approved_annotations: int = 0
	rejected_annotations: int = 0
	conflicted_annotations: int = 0
	average_confidence: float = 0.0
	inter_annotator_agreement: float = 0.0
	consistency_score: float = 0.0
	speed_annotations_per_hour: float = 0.0
	last_updated: datetime = field(default_factory=datetime.utcnow)
	
	@property
	def approval_rate(self) -> float:
		"""Calculate approval rate percentage"""
		if self.total_annotations == 0:
			return 0.0
		return (self.approved_annotations / self.total_annotations) * 100

	@property
	def quality_score(self) -> float:
		"""Calculate overall quality score"""
		weights = {
			'approval_rate': 0.4,
			'agreement': 0.3,
			'consistency': 0.2,
			'confidence': 0.1
		}
		
		score = (
			weights['approval_rate'] * (self.approval_rate / 100) +
			weights['agreement'] * self.inter_annotator_agreement +
			weights['consistency'] * self.consistency_score +
			weights['confidence'] * self.average_confidence
		)
		
		return min(score, 1.0)

class ProjectManager:
	"""Collaborative annotation project manager"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		assert tenant_id, "Tenant ID is required for project manager"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Project state
		self.active_projects: Dict[str, AnnotationProject] = {}
		self.project_members: Dict[str, Dict[str, ProjectRole]] = defaultdict(dict)
		self.active_sessions: Dict[str, AnnotationSession] = {}
		self.annotation_conflicts: Dict[str, List[AnnotationConflict]] = defaultdict(list)
		self.quality_metrics: Dict[str, QualityMetrics] = {}
		
		# Real-time collaboration
		self.document_locks: Dict[str, Dict[str, datetime]] = defaultdict(dict)
		self.annotation_cache: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
		self.change_streams: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
		
		self._setup_project_config()
		self._log_manager_initialized()
	
	def _setup_project_config(self) -> None:
		"""Setup project management configuration"""
		self.max_concurrent_annotators = self.config.get("max_concurrent_annotators", 20)
		self.document_lock_timeout = self.config.get("document_lock_timeout", 300)  # 5 minutes
		self.quality_check_interval = self.config.get("quality_check_interval", 3600)  # 1 hour
		self.conflict_detection_enabled = self.config.get("conflict_detection_enabled", True)
		self.auto_save_interval = self.config.get("auto_save_interval", 30)  # 30 seconds
	
	def _log_manager_initialized(self) -> None:
		"""Log manager initialization"""
		logger.info(f"Annotation project manager initialized for tenant: {self.tenant_id}")
	
	async def create_project(self, project_data: Dict[str, Any]) -> AnnotationProject:
		"""Create new annotation project"""
		# Validate project data
		required_fields = ["name", "annotation_type", "annotation_schema", "project_manager"]
		for field in required_fields:
			if field not in project_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Create project
		project = AnnotationProject(
			tenant_id=self.tenant_id,
			name=project_data["name"],
			description=project_data.get("description"),
			annotation_type=NLPTaskType(project_data["annotation_type"]),
			annotation_schema=project_data["annotation_schema"],
			guidelines=project_data.get("guidelines"),
			team_members=project_data.get("team_members", []),
			project_manager=project_data["project_manager"],
			quality_requirements=project_data.get("quality_requirements", {}),
			created_by=project_data["project_manager"]
		)
		
		# Store project
		self.active_projects[project.id] = project
		
		# Setup project manager role
		self.project_members[project.id][project.project_manager] = ProjectRole.PROJECT_MANAGER
		
		# Initialize project quality metrics
		for member_id in project.team_members + [project.project_manager]:
			metrics_key = f"{project.id}_{member_id}"
			self.quality_metrics[metrics_key] = QualityMetrics(
				project_id=project.id,
				annotator_id=member_id
			)
		
		self._log_project_created(project.id, project.name)
		
		return project
	
	def _log_project_created(self, project_id: str, project_name: str) -> None:
		"""Log project creation"""
		logger.info(f"Annotation project created: {project_id} ({project_name})")
	
	async def add_team_member(self, project_id: str, user_id: str, role: ProjectRole) -> bool:
		"""Add team member to project"""
		if project_id not in self.active_projects:
			return False
		
		project = self.active_projects[project_id]
		
		# Add to project team
		if user_id not in project.team_members:
			project.team_members.append(user_id)
		
		# Set role
		self.project_members[project_id][user_id] = role
		
		# Initialize quality metrics
		metrics_key = f"{project_id}_{user_id}"
		if metrics_key not in self.quality_metrics:
			self.quality_metrics[metrics_key] = QualityMetrics(
				project_id=project_id,
				annotator_id=user_id
			)
		
		self._log_member_added(project_id, user_id, role)
		
		return True
	
	def _log_member_added(self, project_id: str, user_id: str, role: ProjectRole) -> None:
		"""Log team member addition"""
		logger.info(f"Team member added to project {project_id}: {user_id} ({role})")
	
	async def start_annotation_session(self, project_id: str, document_id: str, 
									   annotator_id: str) -> Optional[AnnotationSession]:
		"""Start annotation session for document"""
		if project_id not in self.active_projects:
			return None
		
		# Check if user has permission
		if annotator_id not in self.project_members[project_id]:
			return None
		
		# Check document lock
		if await self._is_document_locked(document_id, annotator_id):
			return None
		
		# Create session
		session = AnnotationSession(
			project_id=project_id,
			document_id=document_id,
			annotator_id=annotator_id
		)
		
		# Lock document
		await self._lock_document(document_id, annotator_id)
		
		# Store session
		self.active_sessions[session.session_id] = session
		
		self._log_session_started(session.session_id, annotator_id, document_id)
		
		return session
	
	def _log_session_started(self, session_id: str, annotator_id: str, document_id: str) -> None:
		"""Log annotation session start"""
		logger.info(f"Annotation session started: {session_id} (annotator: {annotator_id}, doc: {document_id})")
	
	async def _is_document_locked(self, document_id: str, annotator_id: str) -> bool:
		"""Check if document is locked by another user"""
		if document_id not in self.document_locks:
			return False
		
		locks = self.document_locks[document_id]
		current_time = datetime.utcnow()
		
		# Clean expired locks
		expired_locks = [
			user_id for user_id, lock_time in locks.items()
			if current_time - lock_time > timedelta(seconds=self.document_lock_timeout)
		]
		
		for user_id in expired_locks:
			del locks[user_id]
		
		# Check if locked by someone else
		return bool([user_id for user_id in locks if user_id != annotator_id])
	
	async def _lock_document(self, document_id: str, annotator_id: str) -> None:
		"""Lock document for annotation"""
		self.document_locks[document_id][annotator_id] = datetime.utcnow()
	
	async def _unlock_document(self, document_id: str, annotator_id: str) -> None:
		"""Unlock document"""
		if document_id in self.document_locks:
			self.document_locks[document_id].pop(annotator_id, None)
			if not self.document_locks[document_id]:
				del self.document_locks[document_id]
	
	async def save_annotation(self, session_id: str, annotation_data: Dict[str, Any]) -> bool:
		"""Save annotation with real-time collaboration support"""
		if session_id not in self.active_sessions:
			return False
		
		session = self.active_sessions[session_id]
		project = self.active_projects[session.project_id]
		
		# Create annotation
		annotation = TextAnnotation(
			project_id=session.project_id,
			document_id=session.document_id,
			annotator_id=session.annotator_id,
			start_position=annotation_data["start_position"],
			end_position=annotation_data["end_position"],
			annotated_text=annotation_data["annotated_text"],
			annotation_value=annotation_data["annotation_value"],
			confidence=annotation_data.get("confidence", 1.0),
			metadata=annotation_data.get("metadata", {})
		)
		
		# Validate annotation
		if not await self._validate_annotation(annotation, project):
			return False
		
		# Check for conflicts
		if self.conflict_detection_enabled:
			conflicts = await self._detect_conflicts(annotation, session.project_id)
			if conflicts:
				for conflict in conflicts:
					self.annotation_conflicts[session.project_id].append(conflict)
		
		# Store annotation
		document_key = f"{session.project_id}_{session.document_id}"
		self.annotation_cache[document_key].append({
			"annotation_id": annotation.id,
			"annotator_id": annotation.annotator_id,
			"data": annotation_data,
			"timestamp": datetime.utcnow(),
			"status": AnnotationStatus.DRAFT
		})
		
		# Update session metrics
		session.annotations_made += 1
		session.last_activity = datetime.utcnow()
		
		# Broadcast change to other users
		await self._broadcast_annotation_change(session.project_id, session.document_id, {
			"action": "annotation_added",
			"annotation_id": annotation.id,
			"annotator_id": annotation.annotator_id,
			"data": annotation_data
		})
		
		self._log_annotation_saved(annotation.id, session.annotator_id)
		
		return True
	
	def _log_annotation_saved(self, annotation_id: str, annotator_id: str) -> None:
		"""Log annotation save"""
		logger.info(f"Annotation saved: {annotation_id} by {annotator_id}")
	
	async def _validate_annotation(self, annotation: TextAnnotation, project: AnnotationProject) -> bool:
		"""Validate annotation against project schema"""
		schema = project.annotation_schema
		
		# Basic validation
		if annotation.start_position >= annotation.end_position:
			return False
		
		if annotation.start_position < 0:
			return False
		
		# Schema validation
		if "required_fields" in schema:
			for field in schema["required_fields"]:
				if field not in annotation.annotation_value:
					return False
		
		# Label validation
		if "valid_labels" in schema:
			annotation_label = annotation.annotation_value.get("label")
			if annotation_label not in schema["valid_labels"]:
				return False
		
		# Confidence threshold
		min_confidence = schema.get("min_confidence", 0.0)
		if annotation.confidence < min_confidence:
			return False
		
		return True
	
	async def _detect_conflicts(self, new_annotation: TextAnnotation, 
								project_id: str) -> List[AnnotationConflict]:
		"""Detect conflicts with existing annotations"""
		conflicts = []
		document_key = f"{project_id}_{new_annotation.document_id}"
		existing_annotations = self.annotation_cache.get(document_key, [])
		
		for existing in existing_annotations:
			# Skip annotations from same annotator
			if existing["annotator_id"] == new_annotation.annotator_id:
				continue
			
			existing_data = existing["data"]
			
			# Check for overlapping entities
			if (new_annotation.start_position < existing_data["end_position"] and
				new_annotation.end_position > existing_data["start_position"]):
				
				# Determine conflict type
				if (new_annotation.start_position == existing_data["start_position"] and
					new_annotation.end_position == existing_data["end_position"]):
					# Same span, different labels
					if (new_annotation.annotation_value.get("label") != 
						existing_data["annotation_value"].get("label")):
						conflict_type = ConflictType.DIFFERENT_LABELS
					else:
						continue  # No conflict
				else:
					# Overlapping boundaries
					conflict_type = ConflictType.OVERLAPPING_ENTITIES
				
				conflict = AnnotationConflict(
					project_id=project_id,
					document_id=new_annotation.document_id,
					conflict_type=conflict_type,
					annotator_1=new_annotation.annotator_id,
					annotator_2=existing["annotator_id"],
					annotation_1={
						"start": new_annotation.start_position,
						"end": new_annotation.end_position,
						"value": new_annotation.annotation_value
					},
					annotation_2={
						"start": existing_data["start_position"],
						"end": existing_data["end_position"], 
						"value": existing_data["annotation_value"]
					}
				)
				
				conflicts.append(conflict)
		
		return conflicts
	
	async def _broadcast_annotation_change(self, project_id: str, document_id: str, 
										   change_data: Dict[str, Any]) -> None:
		"""Broadcast annotation changes to all active sessions"""
		change_message = {
			"project_id": project_id,
			"document_id": document_id,
			"timestamp": datetime.utcnow().isoformat(),
			"change": change_data
		}
		
		# Add to change streams for active sessions
		for session in self.active_sessions.values():
			if (session.project_id == project_id and 
				session.document_id == document_id and
				session.active):
				
				try:
					await self.change_streams[session.session_id].put(change_message)
				except asyncio.QueueFull:
					logger.warning(f"Change stream full for session: {session.session_id}")
	
	async def get_annotation_changes(self, session_id: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
		"""Get real-time annotation changes for session"""
		if session_id not in self.change_streams:
			return None
		
		try:
			change = await asyncio.wait_for(
				self.change_streams[session_id].get(),
				timeout=timeout
			)
			return change
		except asyncio.TimeoutError:
			return None
	
	async def end_annotation_session(self, session_id: str) -> bool:
		"""End annotation session and cleanup resources"""
		if session_id not in self.active_sessions:
			return False
		
		session = self.active_sessions[session_id]
		session.active = False
		
		# Update session time
		session.time_spent_seconds = int(
			(datetime.utcnow() - session.started_at).total_seconds()
		)
		
		# Unlock document
		await self._unlock_document(session.document_id, session.annotator_id)
		
		# Update quality metrics
		await self._update_quality_metrics(session)
		
		# Cleanup
		del self.active_sessions[session_id]
		if session_id in self.change_streams:
			del self.change_streams[session_id]
		
		self._log_session_ended(session_id, session.time_spent_seconds)
		
		return True
	
	def _log_session_ended(self, session_id: str, duration_seconds: int) -> None:
		"""Log session end"""
		logger.info(f"Annotation session ended: {session_id} (duration: {duration_seconds}s)")
	
	async def _update_quality_metrics(self, session: AnnotationSession) -> None:
		"""Update quality metrics for annotator"""
		metrics_key = f"{session.project_id}_{session.annotator_id}"
		
		if metrics_key not in self.quality_metrics:
			return
		
		metrics = self.quality_metrics[metrics_key]
		metrics.total_annotations += session.annotations_made
		
		# Update speed metric
		if session.time_spent_seconds > 0:
			annotations_per_hour = (session.annotations_made / session.time_spent_seconds) * 3600
			# Exponential moving average
			if metrics.speed_annotations_per_hour == 0:
				metrics.speed_annotations_per_hour = annotations_per_hour
			else:
				alpha = 0.2  # Smoothing factor
				metrics.speed_annotations_per_hour = (
					alpha * annotations_per_hour + 
					(1 - alpha) * metrics.speed_annotations_per_hour
				)
		
		metrics.last_updated = datetime.utcnow()
	
	async def resolve_conflict(self, conflict_id: str, resolution: Dict[str, Any], 
							   resolver_id: str) -> bool:
		"""Resolve annotation conflict"""
		# Find conflict
		conflict = None
		project_id = None
		
		for pid, conflicts in self.annotation_conflicts.items():
			for c in conflicts:
				if c.conflict_id == conflict_id:
					conflict = c
					project_id = pid
					break
			if conflict:
				break
		
		if not conflict:
			return False
		
		# Validate resolver permission
		if resolver_id not in self.project_members[project_id]:
			return False
		
		role = self.project_members[project_id][resolver_id]
		if role not in [ProjectRole.PROJECT_MANAGER, ProjectRole.SENIOR_ANNOTATOR, ProjectRole.REVIEWER]:
			return False
		
		# Apply resolution
		conflict.resolved_at = datetime.utcnow()
		conflict.resolved_by = resolver_id
		conflict.resolution = resolution
		
		# Broadcast resolution
		await self._broadcast_annotation_change(project_id, conflict.document_id, {
			"action": "conflict_resolved",
			"conflict_id": conflict_id,
			"resolved_by": resolver_id,
			"resolution": resolution
		})
		
		self._log_conflict_resolved(conflict_id, resolver_id)
		
		return True
	
	def _log_conflict_resolved(self, conflict_id: str, resolver_id: str) -> None:
		"""Log conflict resolution"""
		logger.info(f"Annotation conflict resolved: {conflict_id} by {resolver_id}")
	
	def get_project_statistics(self, project_id: str) -> Dict[str, Any]:
		"""Get comprehensive project statistics"""
		if project_id not in self.active_projects:
			return {}
		
		project = self.active_projects[project_id]
		
		# Collect metrics for all team members
		team_metrics = []
		total_annotations = 0
		total_conflicts = 0
		
		for member_id in project.team_members + [project.project_manager]:
			metrics_key = f"{project_id}_{member_id}"
			if metrics_key in self.quality_metrics:
				metrics = self.quality_metrics[metrics_key]
				team_metrics.append({
					"annotator_id": member_id,
					"role": self.project_members[project_id].get(member_id, "unknown"),
					"total_annotations": metrics.total_annotations,
					"approval_rate": metrics.approval_rate,
					"quality_score": metrics.quality_score,
					"speed": metrics.speed_annotations_per_hour,
					"agreement": metrics.inter_annotator_agreement
				})
				total_annotations += metrics.total_annotations
		
		# Count conflicts
		total_conflicts = len(self.annotation_conflicts.get(project_id, []))
		resolved_conflicts = len([
			c for c in self.annotation_conflicts.get(project_id, [])
			if c.resolved_at is not None
		])
		
		# Active sessions
		active_sessions = [
			s for s in self.active_sessions.values()
			if s.project_id == project_id and s.active
		]
		
		return {
			"project_id": project_id,
			"project_name": project.name,
			"team_size": len(project.team_members) + 1,  # +1 for manager
			"total_annotations": total_annotations,
			"total_conflicts": total_conflicts,
			"resolved_conflicts": resolved_conflicts,
			"conflict_resolution_rate": (resolved_conflicts / max(total_conflicts, 1)) * 100,
			"active_sessions": len(active_sessions),
			"team_metrics": team_metrics,
			"project_created": project.created_at.isoformat(),
			"annotation_type": project.annotation_type.value
		}
	
	def get_annotator_performance(self, project_id: str, annotator_id: str) -> Dict[str, Any]:
		"""Get detailed performance metrics for specific annotator"""
		metrics_key = f"{project_id}_{annotator_id}"
		
		if metrics_key not in self.quality_metrics:
			return {}
		
		metrics = self.quality_metrics[metrics_key]
		role = self.project_members[project_id].get(annotator_id, "unknown")
		
		# Find conflicts involving this annotator
		annotator_conflicts = [
			c for c in self.annotation_conflicts.get(project_id, [])
			if c.annotator_1 == annotator_id or c.annotator_2 == annotator_id
		]
		
		# Active session info
		active_session = None
		for session in self.active_sessions.values():
			if (session.project_id == project_id and 
				session.annotator_id == annotator_id and 
				session.active):
				active_session = {
					"session_id": session.session_id,
					"document_id": session.document_id,
					"duration_minutes": (datetime.utcnow() - session.started_at).total_seconds() / 60,
					"annotations_this_session": session.annotations_made
				}
				break
		
		return {
			"annotator_id": annotator_id,
			"project_id": project_id,
			"role": role,
			"performance_metrics": {
				"total_annotations": metrics.total_annotations,
				"approved_annotations": metrics.approved_annotations,
				"approval_rate": round(metrics.approval_rate, 2),
				"quality_score": round(metrics.quality_score, 3),
				"average_confidence": round(metrics.average_confidence, 3),
				"inter_annotator_agreement": round(metrics.inter_annotator_agreement, 3),
				"consistency_score": round(metrics.consistency_score, 3),
				"speed_annotations_per_hour": round(metrics.speed_annotations_per_hour, 2)
			},
			"conflict_metrics": {
				"total_conflicts": len(annotator_conflicts),
				"resolved_conflicts": len([c for c in annotator_conflicts if c.resolved_at]),
				"conflict_types": {
					conflict_type.value: len([
						c for c in annotator_conflicts 
						if c.conflict_type == conflict_type
					])
					for conflict_type in ConflictType
				}
			},
			"current_session": active_session,
			"last_updated": metrics.last_updated.isoformat()
		}
	
	async def export_annotations(self, project_id: str, export_format: str = "json") -> Dict[str, Any]:
		"""Export project annotations for training data generation"""
		if project_id not in self.active_projects:
			return {}
		
		project = self.active_projects[project_id]
		
		# Collect all annotations for the project
		project_annotations = []
		
		for document_key, annotations in self.annotation_cache.items():
			if document_key.startswith(f"{project_id}_"):
				for annotation in annotations:
					# Only export approved annotations
					if annotation.get("status") == AnnotationStatus.APPROVED:
						project_annotations.append({
							"document_id": document_key.split("_", 1)[1],
							"annotation_id": annotation["annotation_id"],
							"annotator_id": annotation["annotator_id"],
							"start_position": annotation["data"]["start_position"],
							"end_position": annotation["data"]["end_position"],
							"text": annotation["data"]["annotated_text"],
							"label": annotation["data"]["annotation_value"],
							"confidence": annotation["data"].get("confidence", 1.0),
							"timestamp": annotation["timestamp"].isoformat()
						})
		
		export_data = {
			"project_info": {
				"project_id": project_id,
				"project_name": project.name,
				"annotation_type": project.annotation_type.value,
				"schema": project.annotation_schema,
				"exported_at": datetime.utcnow().isoformat(),
				"export_format": export_format
			},
			"annotations": project_annotations,
			"statistics": {
				"total_annotations": len(project_annotations),
				"unique_documents": len(set(ann["document_id"] for ann in project_annotations)),
				"unique_annotators": len(set(ann["annotator_id"] for ann in project_annotations)),
				"label_distribution": self._calculate_label_distribution(project_annotations)
			}
		}
		
		self._log_export_completed(project_id, len(project_annotations))
		
		return export_data
	
	def _calculate_label_distribution(self, annotations: List[Dict[str, Any]]) -> Dict[str, int]:
		"""Calculate label distribution in annotations"""
		distribution = defaultdict(int)
		
		for annotation in annotations:
			label = annotation["label"].get("label", "unknown")
			distribution[label] += 1
		
		return dict(distribution)
	
	def _log_export_completed(self, project_id: str, annotation_count: int) -> None:
		"""Log export completion"""
		logger.info(f"Annotations exported for project {project_id}: {annotation_count} annotations")
	
	async def cleanup(self) -> None:
		"""Cleanup project manager resources"""
		# End all active sessions
		session_ids = list(self.active_sessions.keys())
		for session_id in session_ids:
			await self.end_annotation_session(session_id)
		
		# Clear all caches
		self.active_projects.clear()
		self.project_members.clear()
		self.annotation_conflicts.clear()
		self.quality_metrics.clear()
		self.document_locks.clear()
		self.annotation_cache.clear()
		self.change_streams.clear()
		
		logger.info(f"Project manager cleanup completed for tenant: {self.tenant_id}")

# Export main classes
__all__ = [
	"ProjectManager", "AnnotationConflict", "AnnotationSession", "QualityMetrics",
	"AnnotationStatus", "ProjectRole", "ConflictType"
]