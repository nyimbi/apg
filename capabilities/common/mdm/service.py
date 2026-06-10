#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Core Service Layer
Advanced multi-tenant master data management with real-time AI/ML processing

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from uuid_extensions import uuid7str

from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field, ConfigDict, validator

from .models import (
	MdEntity, MdEntityVersion, MdGoldenRecord, MdCrossReference,
	MdDataQualityAssessment, MdMatchRule, MdSurvivorshipRule, MdAuditLog, MdDataLineage,
	EntityType, EntityStatus, DataQualityStatus, MatchConfidence, SurvivorshipRule,
	MdEntityCreate, MdEntityUpdate, MdDataQualityScore, MdDuplicateDetectionResult, MdMatchCandidate
)
try:
	from .database import MDMDatabaseManager
	_DATABASE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
	MDMDatabaseManager = None
	_DATABASE_IMPORT_ERROR = exc
from .capability_contract import (
	PRIVILEGED_MDM_AGENT_ROLES,
	SUPPORTED_MDM_AGENT_ROLES,
	SUPPORTED_MDM_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)


class MDMOperationType(str, Enum):
	"""MDM operation types for audit and event tracking"""
	CREATE_ENTITY = "create_entity"
	UPDATE_ENTITY = "update_entity"
	DELETE_ENTITY = "delete_entity"
	MERGE_ENTITIES = "merge_entities"
	SPLIT_ENTITY = "split_entity"
	ASSESS_QUALITY = "assess_quality"
	DETECT_DUPLICATES = "detect_duplicates"
	CREATE_GOLDEN_RECORD = "create_golden_record"
	UPDATE_GOLDEN_RECORD = "update_golden_record"


@dataclass
class MDMOperationContext:
	"""Context for MDM operations"""
	tenant_id: str
	user_id: str
	operation_type: MDMOperationType
	entity_id: Optional[str] = None
	entity_type: Optional[EntityType] = None
	operation_id: str = Field(default_factory=uuid7str)
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	source_system: Optional[str] = None
	client_ip: Optional[str] = None
	user_agent: Optional[str] = None


@dataclass
class MdmEntityRecord:
	"""Dependency-light entity lifecycle record for generated applications."""

	record_id: str
	tenant_id: str
	entity_id: str
	entity_type: str
	name: str
	business_key: str
	source_system: str
	data_owner: str | None
	classification: str
	attributes: dict[str, Any] = field(default_factory=dict)
	status: str = "active"
	decision: str = "allow"
	quality_score: float | None = None
	latest_quality_assessment_id: str | None = None
	duplicate_status: str = "not_checked"
	golden_record_id: str | None = None
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MdmQualityRecord:
	"""Quality assessment evidence used by publish and stewardship gates."""

	assessment_id: str
	tenant_id: str
	entity_id: str
	overall_score: float
	dimensions: dict[str, float]
	assessor: str
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	issues: list[dict[str, Any]] = field(default_factory=list)
	recommendations: list[str] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MdmDuplicateCandidateRecord:
	"""Duplicate candidate and stewardship review record."""

	candidate_id: str
	tenant_id: str
	entity_id: str
	candidate_entity_id: str
	confidence: float
	reason: str
	decision: str
	status: str
	steward_review_recorded: bool = False
	steward: str | None = None
	review_notes: str | None = None
	review_decision: str | None = None
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	reviewed_at: datetime | None = None


@dataclass
class MdmGoldenRecord:
	"""Golden-record composition state."""

	golden_record_id: str
	tenant_id: str
	entity_type: str
	survivorship_policy: str
	source_entity_ids: list[str]
	status: str
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	attributes: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MdmMergeRequestRecord:
	"""Golden-record merge decision state."""

	merge_id: str
	tenant_id: str
	golden_record_id: str
	source_entity_ids: list[str]
	survivorship_policy: str | None
	conflict_present: bool
	independent_steward: str | None
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	review_notes: str | None = None
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MdmCrossReferenceRecord:
	"""Source-system identifier mapping evidence."""

	cross_reference_id: str
	tenant_id: str
	entity_id: str
	source_system: str
	source_identifier: str
	evidence_reference: str | None
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MdmPublishRecord:
	"""Publish readiness and release decision."""

	publish_id: str
	tenant_id: str
	entity_id: str
	channel: str
	decision: str
	status: str
	quality_score: float | None
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MdmDataAgentRecord:
	"""First-class master-data governance agent registration."""

	agent_id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool
	human_approval_required: bool
	status: str = "active"
	policy_decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MdmLifecycleBatchRecord:
	"""Bytewax lifecycle-batch validation evidence."""

	batch_id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	required_processor: str = "bytewax"
	status: str = "accepted"
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MdmAuditEventRecord:
	"""Dependency-light MDM audit event."""

	event_id: str
	tenant_id: str
	event_type: str
	subject: str
	actor: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	details: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


class EntityService:
	"""Core entity management service with advanced CRUD operations"""
	
	def __init__(self, db_manager: MDMDatabaseManager):
		self.db_manager = db_manager
		self.quality_service = None  # Will be injected
		self.matching_service = None  # Will be injected
		self.audit_service = None  # Will be injected
		self._background_tasks: set[asyncio.Task] = set()
	
	async def create_entity(self, entity_data: MdEntityCreate, 
						   context: MDMOperationContext) -> Dict[str, Any]:
		"""Create new master data entity with comprehensive processing"""
		try:
			async with self.db_manager.get_session(context.tenant_id) as session:
				# Generate unique entity ID
				entity_id = uuid7str()
				
				# Create entity record
				entity = MdEntity(
					id=entity_id,
					tenant_id=context.tenant_id,
					entity_type=entity_data.entity_type.value,
					entity_name=entity_data.entity_name,
					entity_description=entity_data.entity_description,
					business_key=entity_data.business_key,
					source_system=entity_data.source_system,
					status=entity_data.status.value,
					attributes=entity_data.attributes,
					tags=entity_data.tags,
					data_classification=entity_data.data_classification,
					created_by=context.user_id,
					updated_by=context.user_id,
					quality_score=0.0  # Will be calculated
				)
				
				session.add(entity)
				await session.flush()  # Get the ID
				
				# Create initial version
				await self._create_entity_version(session, entity, context, "create")
				
				# Perform initial quality assessment
				if self.quality_service:
					quality_result = await self.quality_service.assess_quality(
						entity_id, context.tenant_id, entity_data.attributes
					)
					entity.quality_score = quality_result.get('overall_score', 0.0)
					entity.last_quality_check = datetime.utcnow()
				
				# Log creation event
				if self.audit_service:
					await self.audit_service.log_event(
						context, entity_id, entity_data.entity_type.value,
						"Entity created", {"entity_data": entity_data.dict()}
					)
				
				await session.commit()
				
				# Trigger duplicate detection asynchronously
				if self.matching_service:
					try:
						from capabilities.common.reliability import create_tracked_task
						create_tracked_task(
							self.matching_service.detect_duplicates(entity_id, context.tenant_id),
							task_set=self._background_tasks,
							name='detect_duplicates',
						)
					except Exception as exc:
						import logging; logging.getLogger(__name__).warning('create_tracked_task failed: %s', exc)
				
				return {
					'status': 'success',
					'entity_id': entity_id,
					'quality_score': entity.quality_score,
					'created_at': entity.created_at.isoformat(),
					'message': 'Entity created successfully'
				}
				
		except Exception as e:
			return {
				'status': 'error',
				'message': f'Failed to create entity: {str(e)}',
				'timestamp': datetime.utcnow().isoformat()
			}
	
	async def update_entity(self, entity_id: str, entity_updates: MdEntityUpdate,
						   context: MDMOperationContext) -> Dict[str, Any]:
		"""Update existing master data entity with version tracking"""
		try:
			async with self.db_manager.get_session(context.tenant_id) as session:
				# Fetch existing entity
				result = await session.execute(
					select(MdEntity).where(
						and_(MdEntity.id == entity_id, MdEntity.tenant_id == context.tenant_id)
					)
				)
				entity = result.scalar_one_or_none()
				
				if not entity:
					return {
						'status': 'error',
						'message': 'Entity not found',
						'entity_id': entity_id
					}
				
				# Store previous values for audit
				previous_values = {
					'entity_name': entity.entity_name,
					'entity_description': entity.entity_description,
					'status': entity.status,
					'attributes': entity.attributes,
					'tags': entity.tags,
					'data_classification': entity.data_classification
				}
				
				# Apply updates
				changed_fields = []
				if entity_updates.entity_name is not None:
					entity.entity_name = entity_updates.entity_name
					changed_fields.append('entity_name')
				
				if entity_updates.entity_description is not None:
					entity.entity_description = entity_updates.entity_description
					changed_fields.append('entity_description')
				
				if entity_updates.status is not None:
					entity.status = entity_updates.status.value
					changed_fields.append('status')
				
				if entity_updates.attributes is not None:
					# Merge attributes instead of replacing
					entity.attributes = {**entity.attributes, **entity_updates.attributes}
					changed_fields.append('attributes')
				
				if entity_updates.tags is not None:
					entity.tags = entity_updates.tags
					changed_fields.append('tags')
				
				if entity_updates.data_classification is not None:
					entity.data_classification = entity_updates.data_classification
					changed_fields.append('data_classification')
				
				# Update metadata
				entity.updated_by = context.user_id
				entity.updated_at = datetime.utcnow()
				
				# Create version record
				await self._create_entity_version(
					session, entity, context, "update", 
					changed_fields=changed_fields, previous_values=previous_values
				)
				
				# Re-assess quality if attributes changed
				quality_reassess = False
				if 'attributes' in changed_fields and self.quality_service:
					quality_result = await self.quality_service.assess_quality(
						entity_id, context.tenant_id, entity.attributes
					)
					entity.quality_score = quality_result.get('overall_score', entity.quality_score)
					entity.last_quality_check = datetime.utcnow()
					quality_reassess = True
				
				# Log update event
				if self.audit_service:
					await self.audit_service.log_event(
						context, entity_id, entity.entity_type,
						"Entity updated", {
							"changed_fields": changed_fields,
							"previous_values": previous_values,
							"new_values": entity_updates.dict(exclude_unset=True)
						}
					)
				
				await session.commit()
				
				# Trigger duplicate re-detection if significant changes
				if 'attributes' in changed_fields and self.matching_service:
					try:
						from capabilities.common.reliability import create_tracked_task
						create_tracked_task(
							self.matching_service.detect_duplicates(entity_id, context.tenant_id),
							task_set=self._background_tasks,
							name='detect_duplicates',
						)
					except Exception as exc:
						import logging; logging.getLogger(__name__).warning('create_tracked_task failed: %s', exc)
				
				return {
					'status': 'success',
					'entity_id': entity_id,
					'changed_fields': changed_fields,
					'quality_score': entity.quality_score,
					'quality_reassessed': quality_reassess,
					'updated_at': entity.updated_at.isoformat(),
					'message': 'Entity updated successfully'
				}
				
		except Exception as e:
			return {
				'status': 'error',
				'message': f'Failed to update entity: {str(e)}',
				'entity_id': entity_id,
				'timestamp': datetime.utcnow().isoformat()
			}
	
	async def get_entity(self, entity_id: str, tenant_id: str,
						include_versions: bool = False, include_quality: bool = False,
						include_cross_refs: bool = False) -> Dict[str, Any]:
		"""Retrieve entity with optional related data"""
		try:
			async with self.db_manager.get_session(tenant_id) as session:
				# Build query with optional eager loading
				query = select(MdEntity).where(
					and_(MdEntity.id == entity_id, MdEntity.tenant_id == tenant_id)
				)
				
				if include_versions:
					query = query.options(selectinload(MdEntity.versions))
				if include_quality:
					query = query.options(selectinload(MdEntity.quality_assessments))
				if include_cross_refs:
					query = query.options(selectinload(MdEntity.cross_references))
				
				result = await session.execute(query)
				entity = result.scalar_one_or_none()
				
				if not entity:
					return {
						'status': 'error',
						'message': 'Entity not found',
						'entity_id': entity_id
					}
				
				# Build response
				entity_data = {
					'entity_id': entity.id,
					'tenant_id': entity.tenant_id,
					'entity_type': entity.entity_type,
					'entity_name': entity.entity_name,
					'entity_description': entity.entity_description,
					'business_key': entity.business_key,
					'source_system': entity.source_system,
					'status': entity.status,
					'attributes': entity.attributes,
					'tags': entity.tags,
					'data_classification': entity.data_classification,
					'quality_score': entity.quality_score,
					'last_quality_check': entity.last_quality_check.isoformat() if entity.last_quality_check else None,
					'is_golden_record': entity.is_golden_record,
					'golden_record_id': entity.golden_record_id,
					'created_at': entity.created_at.isoformat(),
					'updated_at': entity.updated_at.isoformat(),
					'created_by': entity.created_by,
					'updated_by': entity.updated_by
				}
				
				# Add optional data
				if include_versions and entity.versions:
					entity_data['versions'] = [
						{
							'version_number': v.version_number,
							'version_timestamp': v.version_timestamp.isoformat(),
							'version_type': v.version_type,
							'created_by': v.created_by,
							'change_description': v.change_description,
							'changed_fields': v.changed_fields
						}
						for v in sorted(entity.versions, key=lambda x: x.version_number, reverse=True)
					]
				
				if include_quality and entity.quality_assessments:
					latest_quality = max(entity.quality_assessments, 
										key=lambda x: x.assessment_timestamp)
					entity_data['latest_quality_assessment'] = {
						'overall_score': latest_quality.overall_score,
						'quality_status': latest_quality.quality_status,
						'assessment_timestamp': latest_quality.assessment_timestamp.isoformat(),
						'completeness_score': latest_quality.completeness_score,
						'accuracy_score': latest_quality.accuracy_score,
						'consistency_score': latest_quality.consistency_score
					}
				
				if include_cross_refs and entity.cross_references:
					entity_data['cross_references'] = [
						{
							'source_system': cr.source_system,
							'source_entity_id': cr.source_entity_id,
							'source_entity_type': cr.source_entity_type,
							'confidence_score': cr.confidence_score,
							'is_primary_reference': cr.is_primary_reference,
							'last_verified': cr.last_verified.isoformat() if cr.last_verified else None
						}
						for cr in entity.cross_references if cr.is_active
					]
				
				return {
					'status': 'success',
					'entity': entity_data
				}
				
		except Exception as e:
			return {
				'status': 'error',
				'message': f'Failed to retrieve entity: {str(e)}',
				'entity_id': entity_id,
				'timestamp': datetime.utcnow().isoformat()
			}
	
	async def search_entities(self, tenant_id: str, search_criteria: Dict[str, Any],
							 limit: int = 50, offset: int = 0) -> Dict[str, Any]:
		"""Advanced entity search with filtering and sorting"""
		try:
			async with self.db_manager.get_session(tenant_id) as session:
				# Build base query
				query = select(MdEntity).where(MdEntity.tenant_id == tenant_id)
				
				# Apply filters
				if 'entity_type' in search_criteria:
					query = query.where(MdEntity.entity_type == search_criteria['entity_type'])
				
				if 'status' in search_criteria:
					query = query.where(MdEntity.status == search_criteria['status'])
				
				if 'entity_name' in search_criteria:
					search_term = f"%{search_criteria['entity_name']}%"
					query = query.where(MdEntity.entity_name.ilike(search_term))
				
				if 'business_key' in search_criteria:
					query = query.where(MdEntity.business_key == search_criteria['business_key'])
				
				if 'source_system' in search_criteria:
					query = query.where(MdEntity.source_system == search_criteria['source_system'])
				
				if 'min_quality_score' in search_criteria:
					query = query.where(MdEntity.quality_score >= search_criteria['min_quality_score'])
				
				if 'is_golden_record' in search_criteria:
					query = query.where(MdEntity.is_golden_record == search_criteria['is_golden_record'])
				
				if 'data_classification' in search_criteria:
					query = query.where(MdEntity.data_classification == search_criteria['data_classification'])
				
				# Apply date filters
				if 'created_after' in search_criteria:
					query = query.where(MdEntity.created_at >= search_criteria['created_after'])
				
				if 'updated_after' in search_criteria:
					query = query.where(MdEntity.updated_at >= search_criteria['updated_after'])
				
				# Attribute search (JSONB queries)
				if 'attributes' in search_criteria:
					for key, value in search_criteria['attributes'].items():
						query = query.where(MdEntity.attributes[key].astext == str(value))
				
				# Tag search
				if 'tags' in search_criteria:
					for tag in search_criteria['tags']:
						query = query.where(MdEntity.tags.contains([tag]))
				
				# Apply sorting
				sort_by = search_criteria.get('sort_by', 'updated_at')
				sort_order = search_criteria.get('sort_order', 'desc')
				
				if hasattr(MdEntity, sort_by):
					sort_column = getattr(MdEntity, sort_by)
					if sort_order.lower() == 'desc':
						query = query.order_by(sort_column.desc())
					else:
						query = query.order_by(sort_column.asc())
				
				# Get total count for pagination
				count_query = select(func.count()).select_from(query.subquery())
				total_result = await session.execute(count_query)
				total_count = total_result.scalar()
				
				# Apply pagination
				query = query.offset(offset).limit(limit)
				
				# Execute query
				result = await session.execute(query)
				entities = result.scalars().all()
				
				# Format results
				entity_list = []
				for entity in entities:
					entity_list.append({
						'entity_id': entity.id,
						'entity_type': entity.entity_type,
						'entity_name': entity.entity_name,
						'business_key': entity.business_key,
						'source_system': entity.source_system,
						'status': entity.status,
						'quality_score': entity.quality_score,
						'is_golden_record': entity.is_golden_record,
						'data_classification': entity.data_classification,
						'created_at': entity.created_at.isoformat(),
						'updated_at': entity.updated_at.isoformat(),
						'tags': entity.tags[:5] if entity.tags else []  # Limit tags for performance
					})
				
				return {
					'status': 'success',
					'entities': entity_list,
					'pagination': {
						'total_count': total_count,
						'offset': offset,
						'limit': limit,
						'has_next': offset + limit < total_count,
						'has_previous': offset > 0
					},
					'search_criteria': search_criteria
				}
				
		except Exception as e:
			return {
				'status': 'error',
				'message': f'Entity search failed: {str(e)}',
				'timestamp': datetime.utcnow().isoformat()
			}
	
	async def delete_entity(self, entity_id: str, context: MDMOperationContext,
						   soft_delete: bool = True) -> Dict[str, Any]:
		"""Delete entity with soft/hard delete options"""
		try:
			async with self.db_manager.get_session(context.tenant_id) as session:
				result = await session.execute(
					select(MdEntity).where(
						and_(MdEntity.id == entity_id, MdEntity.tenant_id == context.tenant_id)
					)
				)
				entity = result.scalar_one_or_none()
				
				if not entity:
					return {
						'status': 'error',
						'message': 'Entity not found',
						'entity_id': entity_id
					}
				
				if soft_delete:
					# Soft delete - mark as deleted
					entity.status = EntityStatus.DELETED.value
					entity.updated_by = context.user_id
					entity.updated_at = datetime.utcnow()
					
					# Create version record
					await self._create_entity_version(
						session, entity, context, "delete", 
						change_description="Entity soft deleted"
					)
					
					delete_type = "soft_delete"
				else:
					# Hard delete - remove from database
					await session.delete(entity)
					delete_type = "hard_delete"
				
				# Log deletion event
				if self.audit_service:
					await self.audit_service.log_event(
						context, entity_id, entity.entity_type,
						f"Entity {delete_type}", {"delete_type": delete_type}
					)
				
				await session.commit()
				
				return {
					'status': 'success',
					'entity_id': entity_id,
					'delete_type': delete_type,
					'message': 'Entity deleted successfully',
					'timestamp': datetime.utcnow().isoformat()
				}
				
		except Exception as e:
			return {
				'status': 'error',
				'message': f'Failed to delete entity: {str(e)}',
				'entity_id': entity_id,
				'timestamp': datetime.utcnow().isoformat()
			}
	
	async def _create_entity_version(self, session, entity: MdEntity, 
									context: MDMOperationContext, version_type: str,
									changed_fields: List[str] = None,
									previous_values: Dict[str, Any] = None) -> None:
		"""Create entity version record for audit trail"""
		# Get next version number
		result = await session.execute(
			select(func.max(MdEntityVersion.version_number)).where(
				MdEntityVersion.entity_id == entity.id
			)
		)
		max_version = result.scalar() or 0
		next_version = max_version + 1
		
		# Create entity snapshot
		entity_snapshot = {
			'id': entity.id,
			'entity_type': entity.entity_type,
			'entity_name': entity.entity_name,
			'entity_description': entity.entity_description,
			'business_key': entity.business_key,
			'source_system': entity.source_system,
			'status': entity.status,
			'quality_score': entity.quality_score,
			'is_golden_record': entity.is_golden_record,
			'data_classification': entity.data_classification,
			'created_by': entity.created_by,
			'updated_by': entity.updated_by,
			'created_at': entity.created_at.isoformat(),
			'updated_at': entity.updated_at.isoformat()
		}
		
		# Create version record
		version = MdEntityVersion(
			id=uuid7str(),
			entity_id=entity.id,
			tenant_id=context.tenant_id,
			version_number=next_version,
			version_timestamp=datetime.utcnow(),
			version_type=version_type,
			created_by=context.user_id,
			change_description=f"{version_type.title()} operation",
			entity_snapshot=entity_snapshot,
			attributes_snapshot=entity.attributes,
			quality_score_snapshot=entity.quality_score,
			changed_fields=changed_fields or [],
			previous_values=previous_values or {},
			change_source=context.source_system or "mdm_api"
		)
		
		session.add(version)


class QualityService:
	"""AI-enhanced data quality assessment service"""
	
	def __init__(self, db_manager: MDMDatabaseManager, ollama_client=None):
		self.db_manager = db_manager
		self.ollama_client = ollama_client
		if ollama_client:
			from .ai_engines import DataQualityEngine, AnomalyDetectionEngine
			self.quality_engine = DataQualityEngine(ollama_client)
			self.anomaly_engine = AnomalyDetectionEngine(ollama_client)
		else:
			self.quality_engine = None
			self.anomaly_engine = None
		self.quality_rules = self._initialize_quality_rules()
	
	def _initialize_quality_rules(self) -> Dict[str, Any]:
		"""Initialize comprehensive quality assessment rules"""
		return {
			'completeness': {
				'required_fields': {
					'customer': ['name', 'email', 'phone'],
					'product': ['name', 'sku', 'category'],
					'supplier': ['name', 'contact_email'],
					'default': ['name']
				},
				'weight': 0.25
			},
			'accuracy': {
				'validation_rules': {
					'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
					'phone': r'^\+?[\d\s\-\(\)]{10,}$',
					'url': r'^https?://[^\s]+$'
				},
				'weight': 0.20
			},
			'consistency': {
				'format_rules': {
					'phone': 'normalize_phone',
					'name': 'normalize_name',
					'address': 'normalize_address'
				},
				'weight': 0.15
			},
			'validity': {
				'domain_rules': {
					'age': {'min': 0, 'max': 150},
					'price': {'min': 0},
					'quantity': {'min': 0}
				},
				'weight': 0.15
			},
			'uniqueness': {
				'check_duplicates': True,
				'weight': 0.15
			},
			'timeliness': {
				'freshness_threshold_days': 30,
				'weight': 0.10
			}
		}
	
	async def assess_quality(self, entity_id: str, tenant_id: str, 
							attributes: Dict[str, Any], entity_type: str = None) -> Dict[str, Any]:
		"""Comprehensive AI-enhanced quality assessment"""
		try:
			start_time = datetime.utcnow()
			
			# Prepare entity data for AI assessment
			entity_data = {
				'entity_id': entity_id,
				'attributes': attributes,
				'entity_type': entity_type
			}
			
			# Try AI-enhanced assessment first
			if self.quality_engine:
				try:
					ai_result = await self.quality_engine.assess_data_quality_with_ai(
						entity_data, entity_type
					)
					
					if ai_result.get('overall_score', 0) > 0:
						# Use AI results
						quality_scores = {
							'completeness_score': ai_result['dimension_results'].get('completeness', {}).get('score', 75.0),
							'accuracy_score': ai_result['dimension_results'].get('accuracy', {}).get('score', 75.0),
							'consistency_score': ai_result['dimension_results'].get('consistency', {}).get('score', 75.0),
							'validity_score': 75.0,  # Not directly assessed by AI
							'uniqueness_score': 0.0,  # Will be calculated below
							'timeliness_score': 75.0  # Default
						}
						
						# Extract quality issues from AI results
						quality_issues = []
						for dimension, result in ai_result['dimension_results'].items():
							if 'details' in result and isinstance(result['details'], dict):
								if 'missing_fields' in result['details']:
									for field in result['details']['missing_fields']:
										quality_issues.append({
											'type': 'completeness',
											'field': field,
											'severity': 'high',
											'message': f'Required field {field} is missing'
										})
								if 'issues' in result['details']:
									for issue in result['details']['issues']:
										quality_issues.append({
											'type': dimension,
											'field': 'multiple',
											'severity': 'medium',
											'message': str(issue)
										})
						
						overall_score = ai_result['overall_score']
					else:
						# Fallback to rule-based assessment
						quality_scores, quality_issues, overall_score = await self._fallback_assessment(
							entity_id, tenant_id, attributes, entity_type
						)
						
				except Exception as e:
					print(f"[MDM-Quality] AI assessment failed: {str(e)}")
					# Fallback to rule-based assessment
					quality_scores, quality_issues, overall_score = await self._fallback_assessment(
						entity_id, tenant_id, attributes, entity_type
					)
			else:
				# No AI engine available, use rule-based assessment
				quality_scores, quality_issues, overall_score = await self._fallback_assessment(
					entity_id, tenant_id, attributes, entity_type
				)
			
			# Always calculate uniqueness (requires database lookup)
			uniqueness_result = await self._assess_uniqueness(
				entity_id, tenant_id, attributes, quality_issues
			)
			quality_scores['uniqueness_score'] = uniqueness_result
			
			# Recalculate overall score with uniqueness
			overall_score = self._calculate_overall_score(quality_scores)
			quality_status = self._determine_quality_status(overall_score)
			
			# Anomaly detection if AI engine available
			anomalies = []
			if self.anomaly_engine:
				try:
					anomaly_result = await self.anomaly_engine.detect_anomalies(
						entity_data, entity_type=entity_type
					)
					anomalies = anomaly_result.get('anomalies', [])
					
					# Add anomalies as quality issues
					for anomaly in anomalies:
						quality_issues.append({
							'type': 'anomaly',
							'field': anomaly.get('attribute', 'unknown'),
							'severity': anomaly.get('severity', 'medium'),
							'message': f"Anomaly detected: {anomaly.get('description', 'Unknown anomaly')}"
						})
				except Exception as e:
					print(f"[MDM-Quality] Anomaly detection failed: {str(e)}")
			
			# Calculate processing time
			end_time = datetime.utcnow()
			duration_ms = (end_time - start_time).total_seconds() * 1000
			
			# Store assessment result
			assessment_result = {
				'entity_id': entity_id,
				'tenant_id': tenant_id,
				'overall_score': overall_score,
				'quality_status': quality_status,
				'quality_issues': quality_issues,
				'anomalies_detected': len(anomalies),
				'assessment_duration_ms': duration_ms,
				**quality_scores
			}
			
			await self._store_quality_assessment(assessment_result)
			
			return assessment_result
			
		except Exception as e:
			return {
				'entity_id': entity_id,
				'error': f'Quality assessment failed: {str(e)}',
				'overall_score': 0.0,
				'quality_status': DataQualityStatus.CRITICAL.value
			}
	
	async def _fallback_assessment(self, entity_id: str, tenant_id: str,
								  attributes: Dict[str, Any], entity_type: str = None) -> Tuple[Dict[str, float], List[Dict], float]:
		"""Fallback rule-based quality assessment"""
		# Initialize quality scores
		quality_scores = {
			'completeness_score': 0.0,
			'accuracy_score': 0.0,
			'consistency_score': 0.0,
			'validity_score': 0.0,
			'uniqueness_score': 0.0,
			'timeliness_score': 0.0
		}
		
		quality_issues = []
		
		# Completeness assessment
		completeness_result = await self._assess_completeness(
			attributes, entity_type, quality_issues
		)
		quality_scores['completeness_score'] = completeness_result
		
		# Accuracy assessment
		accuracy_result = await self._assess_accuracy(
			attributes, quality_issues
		)
		quality_scores['accuracy_score'] = accuracy_result
		
		# Consistency assessment
		consistency_result = await self._assess_consistency(
			attributes, quality_issues
		)
		quality_scores['consistency_score'] = consistency_result
		
		# Validity assessment
		validity_result = await self._assess_validity(
			attributes, quality_issues
		)
		quality_scores['validity_score'] = validity_result
		
		# Timeliness assessment
		timeliness_result = await self._assess_timeliness(
			attributes, quality_issues
		)
		quality_scores['timeliness_score'] = timeliness_result
		
		# Calculate overall score (excluding uniqueness for now)
		temp_scores = {k: v for k, v in quality_scores.items() if k != 'uniqueness_score'}
		overall_score = self._calculate_overall_score(temp_scores)
		
		return quality_scores, quality_issues, overall_score
	
	async def _assess_completeness(self, attributes: Dict[str, Any], 
								  entity_type: str, issues: List[Dict]) -> float:
		"""Assess data completeness based on required fields"""
		entity_type = entity_type or 'default'
		required_fields = self.quality_rules['completeness']['required_fields'].get(
			entity_type, self.quality_rules['completeness']['required_fields']['default']
		)
		
		total_fields = len(required_fields)
		completed_fields = 0
		
		for field in required_fields:
			value = attributes.get(field)
			if value is not None and str(value).strip() != '':
				completed_fields += 1
			else:
				issues.append({
					'type': 'completeness',
					'field': field,
					'severity': 'high',
					'message': f'Required field {field} is missing or empty'
				})
		
		return (completed_fields / total_fields) * 100 if total_fields > 0 else 100
	
	async def _assess_accuracy(self, attributes: Dict[str, Any], 
							  issues: List[Dict]) -> float:
		"""Assess data accuracy using validation rules"""
		import re
		
		validation_rules = self.quality_rules['accuracy']['validation_rules']
		total_validations = 0
		passed_validations = 0
		
		for field, pattern in validation_rules.items():
			if field in attributes:
				value = attributes[field]
				if value is not None:
					total_validations += 1
					if isinstance(value, str) and re.match(pattern, value.strip()):
						passed_validations += 1
					else:
						issues.append({
							'type': 'accuracy',
							'field': field,
							'severity': 'medium',
							'message': f'Field {field} does not match expected format'
						})
		
		return (passed_validations / total_validations) * 100 if total_validations > 0 else 100
	
	async def _assess_consistency(self, attributes: Dict[str, Any], 
								 issues: List[Dict]) -> float:
		"""Assess data consistency and format standardization"""
		consistency_score = 100.0
		
		# Check for consistent formatting
		format_issues = 0
		
		# Phone number consistency
		if 'phone' in attributes:
			phone = str(attributes['phone'])
			if phone and not self._is_consistent_phone_format(phone):
				format_issues += 1
				issues.append({
					'type': 'consistency',
					'field': 'phone',
					'severity': 'low',
					'message': 'Phone number format is not standardized'
				})
		
		# Name consistency
		if 'name' in attributes:
			name = str(attributes['name'])
			if name and not self._is_consistent_name_format(name):
				format_issues += 1
				issues.append({
					'type': 'consistency',
					'field': 'name',
					'severity': 'low',
					'message': 'Name format could be standardized'
				})
		
		# Adjust score based on format issues
		if format_issues > 0:
			consistency_score = max(0, 100 - (format_issues * 20))
		
		return consistency_score
	
	async def _assess_validity(self, attributes: Dict[str, Any], 
							  issues: List[Dict]) -> float:
		"""Assess data validity against domain rules"""
		domain_rules = self.quality_rules['validity']['domain_rules']
		total_checks = 0
		passed_checks = 0
		
		for field, rules in domain_rules.items():
			if field in attributes:
				value = attributes[field]
				if value is not None:
					total_checks += 1
					valid = True
					
					try:
						numeric_value = float(value)
						
						if 'min' in rules and numeric_value < rules['min']:
							valid = False
							issues.append({
								'type': 'validity',
								'field': field,
								'severity': 'high',
								'message': f'{field} value {numeric_value} is below minimum {rules["min"]}'
							})
						
						if 'max' in rules and numeric_value > rules['max']:
							valid = False
							issues.append({
								'type': 'validity',
								'field': field,
								'severity': 'high',
								'message': f'{field} value {numeric_value} exceeds maximum {rules["max"]}'
							})
						
						if valid:
							passed_checks += 1
							
					except (ValueError, TypeError):
						issues.append({
							'type': 'validity',
							'field': field,
							'severity': 'medium',
							'message': f'{field} value is not numeric as expected'
						})
		
		return (passed_checks / total_checks) * 100 if total_checks > 0 else 100
	
	async def _assess_uniqueness(self, entity_id: str, tenant_id: str,
								attributes: Dict[str, Any], issues: List[Dict]) -> float:
		"""Assess uniqueness by checking for potential duplicates"""
		try:
			# This would integrate with the matching service
			# For now, we'll do a basic check
			uniqueness_score = 95.0  # Assume mostly unique
			
			# Check for common duplicate indicators
			if 'email' in attributes or 'phone' in attributes:
				async with self.db_manager.get_session(tenant_id) as session:
					duplicate_check_conditions = []
					
					if 'email' in attributes and attributes['email']:
						duplicate_check_conditions.append(
							MdEntity.attributes['email'].astext == str(attributes['email'])
						)
					
					if 'phone' in attributes and attributes['phone']:
						duplicate_check_conditions.append(
							MdEntity.attributes['phone'].astext == str(attributes['phone'])
						)
					
					if duplicate_check_conditions:
						query = select(func.count()).where(
							and_(
								MdEntity.tenant_id == tenant_id,
								MdEntity.id != entity_id,  # Exclude self
								MdEntity.status != EntityStatus.DELETED.value,
								or_(*duplicate_check_conditions)
							)
						)
						
						result = await session.execute(query)
						duplicate_count = result.scalar()
						
						if duplicate_count > 0:
							uniqueness_score = max(0, 100 - (duplicate_count * 30))
							issues.append({
								'type': 'uniqueness',
								'field': 'multiple',
								'severity': 'high' if duplicate_count > 1 else 'medium',
								'message': f'Found {duplicate_count} potential duplicate(s)'
							})
			
			return uniqueness_score
			
		except Exception as e:
			print(f"[MDM-Quality] Uniqueness assessment error: {str(e)}")
			return 90.0  # Default to high uniqueness if check fails
	
	async def _assess_timeliness(self, attributes: Dict[str, Any], 
								issues: List[Dict]) -> float:
		"""Assess data timeliness and freshness"""
		timeliness_score = 100.0
		threshold_days = self.quality_rules['timeliness']['freshness_threshold_days']
		
		# Check for date fields that might indicate staleness
		date_fields = ['last_updated', 'modified_date', 'created_date']
		
		for field in date_fields:
			if field in attributes:
				try:
					if isinstance(attributes[field], str):
						field_date = datetime.fromisoformat(attributes[field].replace('Z', '+00:00'))
					else:
						field_date = attributes[field]
					
					days_old = (datetime.utcnow() - field_date).days
					
					if days_old > threshold_days:
						timeliness_score = max(0, 100 - ((days_old - threshold_days) * 2))
						issues.append({
							'type': 'timeliness',
							'field': field,
							'severity': 'medium' if days_old < threshold_days * 2 else 'high',
							'message': f'Data is {days_old} days old, exceeding freshness threshold'
						})
						break  # Only report one timeliness issue
						
				except (ValueError, TypeError, AttributeError):
					# Skip invalid date fields
					continue
		
		return timeliness_score
	
	def _calculate_overall_score(self, quality_scores: Dict[str, float]) -> float:
		"""Calculate weighted overall quality score"""
		total_score = 0.0
		
		for dimension, weight in [
			('completeness_score', self.quality_rules['completeness']['weight']),
			('accuracy_score', self.quality_rules['accuracy']['weight']),
			('consistency_score', self.quality_rules['consistency']['weight']),
			('validity_score', self.quality_rules['validity']['weight']),
			('uniqueness_score', self.quality_rules['uniqueness']['weight']),
			('timeliness_score', self.quality_rules['timeliness']['weight'])
		]:
			total_score += quality_scores[dimension] * weight
		
		return round(total_score, 2)
	
	def _determine_quality_status(self, overall_score: float) -> str:
		"""Determine quality status based on overall score"""
		if overall_score >= 95:
			return DataQualityStatus.EXCELLENT.value
		elif overall_score >= 80:
			return DataQualityStatus.GOOD.value
		elif overall_score >= 60:
			return DataQualityStatus.FAIR.value
		elif overall_score >= 40:
			return DataQualityStatus.POOR.value
		else:
			return DataQualityStatus.CRITICAL.value
	
	def _is_consistent_phone_format(self, phone: str) -> bool:
		"""Check if phone number follows consistent format"""
		# Remove all non-digit characters
		digits_only = ''.join(filter(str.isdigit, phone))
		# Check if it's a reasonable length
		return 10 <= len(digits_only) <= 15
	
	def _is_consistent_name_format(self, name: str) -> bool:
		"""Check if name follows consistent format"""
		# Basic checks: proper capitalization, no excessive spaces
		name = name.strip()
		if not name:
			return False
		
		# Check for proper capitalization (first letter of each word)
		words = name.split()
		for word in words:
			if word and not word[0].isupper():
				return False
		
		return True
	
	async def _store_quality_assessment(self, assessment_result: Dict[str, Any]) -> None:
		"""Store quality assessment result in database"""
		try:
			async with self.db_manager.get_session(assessment_result['tenant_id']) as session:
				assessment = MdDataQualityAssessment(
					id=uuid7str(),
					entity_id=assessment_result['entity_id'],
					tenant_id=assessment_result['tenant_id'],
					overall_score=assessment_result['overall_score'],
					completeness_score=assessment_result['completeness_score'],
					accuracy_score=assessment_result['accuracy_score'],
					consistency_score=assessment_result['consistency_score'],
					validity_score=assessment_result['validity_score'],
					uniqueness_score=assessment_result['uniqueness_score'],
					timeliness_score=assessment_result['timeliness_score'],
					quality_status=assessment_result['quality_status'],
					quality_issues=assessment_result['quality_issues'],
					assessment_duration_ms=assessment_result.get('assessment_duration_ms'),
					assessment_algorithm='ai_enhanced',
					algorithm_version='1.0.0'
				)
				
				session.add(assessment)
				await session.commit()
				
		except Exception as e:
			print(f"[MDM-Quality] Failed to store quality assessment: {str(e)}")


class MatchingService:
	"""AI-powered entity matching and duplicate detection"""
	
	def __init__(self, db_manager: MDMDatabaseManager, ollama_client=None):
		self.db_manager = db_manager
		self.ollama_client = ollama_client
		if ollama_client:
			from .ai_engines import EntityMatchingEngine
			self.matching_engine = EntityMatchingEngine(ollama_client)
		else:
			self.matching_engine = None
	
	async def detect_duplicates(self, entity_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Detect potential duplicate entities using AI-enhanced matching"""
		try:
			async with self.db_manager.get_session(tenant_id) as session:
				# Get the target entity
				target_result = await session.execute(
					select(MdEntity).where(
						and_(MdEntity.id == entity_id, MdEntity.tenant_id == tenant_id)
					)
				)
				target_entity = target_result.scalar_one_or_none()
				
				if not target_entity:
					return {
						'status': 'error',
						'message': 'Entity not found',
						'entity_id': entity_id
					}
				
				# Get candidate entities of the same type
				candidates_result = await session.execute(
					select(MdEntity).where(
						and_(
							MdEntity.tenant_id == tenant_id,
							MdEntity.entity_type == target_entity.entity_type,
							MdEntity.id != entity_id,
							MdEntity.status != EntityStatus.DELETED.value
						)
					).limit(100)  # Limit for performance
				)
				candidate_entities = candidates_result.scalars().all()
				
				if not candidate_entities:
					return {
						'status': 'success',
						'entity_id': entity_id,
						'total_candidates': 0,
						'matches': []
					}
				
				# Prepare data for matching engine
				target_data = {
					'entity_id': target_entity.id,
					'entity_name': target_entity.entity_name,
					'entity_description': target_entity.entity_description,
					'business_key': target_entity.business_key,
					'source_system': target_entity.source_system,
					'attributes': target_entity.attributes
				}
				
				candidate_data = [
					{
						'entity_id': candidate.id,
						'entity_name': candidate.entity_name,
						'entity_description': candidate.entity_description,
						'business_key': candidate.business_key,
						'source_system': candidate.source_system,
						'attributes': candidate.attributes
					}
					for candidate in candidate_entities
				]
				
				# Use AI matching engine if available
				if self.matching_engine:
					matches = await self.matching_engine.find_duplicate_candidates(
						target_data, candidate_data, target_entity.entity_type
					)
				else:
					# Fallback to simple matching
					matches = await self._simple_matching(target_data, candidate_data)
				
				# Categorize matches by confidence
				high_confidence = len([m for m in matches if m['confidence'] in ['exact', 'high']])
				medium_confidence = len([m for m in matches if m['confidence'] == 'medium'])
				low_confidence = len([m for m in matches if m['confidence'] == 'low'])
				
				return {
					'status': 'success',
					'entity_id': entity_id,
					'entity_name': target_entity.entity_name,
					'tenant_id': tenant_id,
					'total_candidates': len(matches),
					'high_confidence_matches': high_confidence,
					'medium_confidence_matches': medium_confidence,
					'low_confidence_matches': low_confidence,
					'matches': matches,
					'detection_timestamp': datetime.utcnow().isoformat(),
					'algorithm_version': '1.0.0'
				}
				
		except Exception as e:
			return {
				'status': 'error',
				'message': f'Duplicate detection failed: {str(e)}',
				'entity_id': entity_id,
				'timestamp': datetime.utcnow().isoformat()
			}
	
	async def _simple_matching(self, target_data: Dict[str, Any], 
							  candidate_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Simple fallback matching when AI engine is not available"""
		matches = []
		
		for candidate in candidate_data:
			score = 0.0
			
			# Name similarity
			target_name = target_data.get('entity_name', '').lower()
			candidate_name = candidate.get('entity_name', '').lower()
			if target_name and candidate_name:
				from difflib import SequenceMatcher
				name_sim = SequenceMatcher(None, target_name, candidate_name).ratio()
				score += name_sim * 0.4
			
			# Business key similarity
			target_key = target_data.get('business_key', '').lower()
			candidate_key = candidate.get('business_key', '').lower()
			if target_key and candidate_key:
				if target_key == candidate_key:
					score += 0.6
				else:
					key_sim = SequenceMatcher(None, target_key, candidate_key).ratio()
					score += key_sim * 0.3
			
			# Convert to percentage
			match_score = score * 100
			
			if match_score >= 50:  # Only include reasonable matches
				confidence = 'high' if match_score >= 80 else ('medium' if match_score >= 60 else 'low')
				action = 'merge' if match_score >= 85 else ('review' if match_score >= 60 else 'ignore')
				
				matches.append({
					'candidate_id': candidate['entity_id'],
					'candidate_name': candidate['entity_name'],
					'candidate_business_key': candidate['business_key'],
					'candidate_source_system': candidate['source_system'],
					'match_score': round(match_score, 2),
					'confidence': confidence,
					'matching_attributes': ['name', 'business_key'],
					'similarity_details': {'overall': match_score},
					'recommended_action': action,
					'match_explanation': f'Simple matching with {match_score:.1f}% similarity'
				})
		
		return sorted(matches, key=lambda x: x['match_score'], reverse=True)


class AuditService:
	"""Comprehensive audit logging service"""
	
	def __init__(self, db_manager: MDMDatabaseManager):
		self.db_manager = db_manager
	
	async def log_event(self, context: MDMOperationContext, entity_id: str,
					   entity_type: str, description: str, details: Dict[str, Any] = None) -> None:
		"""Log audit event with comprehensive context"""
		try:
			async with self.db_manager.get_session(context.tenant_id) as session:
				audit_log = MdAuditLog(
					id=uuid7str(),
					tenant_id=context.tenant_id,
					event_type=context.operation_type.value,
					entity_id=entity_id,
					entity_type=entity_type,
					event_timestamp=context.timestamp,
					event_description=description,
					event_details=details or {},
					user_id=context.user_id,
					source_system=context.source_system,
					client_ip=context.client_ip,
					user_agent=context.user_agent,
					operation_id=context.operation_id,
					data_sensitivity='internal',  # Default classification
					compliance_tags=['mdm', 'data_governance']
				)
				
				session.add(audit_log)
				await session.commit()
				
		except Exception as e:
			print(f"[MDM-Audit] Failed to log event: {str(e)}")


class MDMService:
	"""Main MDM service orchestrator with AI/ML integration"""
	
	def __init__(self, database_url: str = None, config: Dict[str, Any] = None):
		if MDMDatabaseManager is None:
			raise ModuleNotFoundError(
				"MDM database runtime requires optional dependency asyncpg"
			) from _DATABASE_IMPORT_ERROR
		self.config = config or {}
		self.db_manager = MDMDatabaseManager(database_url, config)
		
		# Initialize Ollama client for AI/ML capabilities
		self.ollama_client = None
		if self.config.get('enable_ai', True):
			try:
				from .ai_engines import OllamaClient
				ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
				self.ollama_client = OllamaClient(ollama_url, config)
				print(f"[MDM-Service] AI/ML capabilities enabled with Ollama at {ollama_url}")
			except Exception as e:
				print(f"[MDM-Service] AI/ML capabilities disabled: {str(e)}")
		
		# Initialize sub-services with AI integration
		self.entity_service = EntityService(self.db_manager)
		self.quality_service = QualityService(self.db_manager, self.ollama_client)
		self.matching_service = MatchingService(self.db_manager, self.ollama_client)
		self.audit_service = AuditService(self.db_manager)
		
		# Inject dependencies
		self.entity_service.quality_service = self.quality_service
		self.entity_service.matching_service = self.matching_service
		self.entity_service.audit_service = self.audit_service
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize MDM service and database"""
		return await self.db_manager.initialize_database()
	
	async def health_check(self) -> Dict[str, Any]:
		"""Comprehensive health check"""
		return await self.db_manager.health_check()
	
	def create_operation_context(self, tenant_id: str, user_id: str,
								operation_type: MDMOperationType, **kwargs) -> MDMOperationContext:
		"""Create operation context for request tracking"""
		return MDMOperationContext(
			tenant_id=tenant_id,
			user_id=user_id,
			operation_type=operation_type,
			**kwargs
		)


class MdmService:
	"""Dependency-light MDM lifecycle and guardrail control plane.

	This service is intentionally separate from the database-backed ``MDMService``.
	Generated APG applications use it to compose MDM workflows, evaluate
	guardrails, and build UI state without requiring PostgreSQL, Redis, AI
	engines, or event-stream adapters to be running.
	"""

	def __init__(self, tenant_id: str = "default"):
		self.tenant_id = tenant_id
		self.contract = get_capability_contract(tenant_id)
		self._agent_runtimes = set(SUPPORTED_MDM_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_MDM_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_MDM_AGENT_ROLES)
		self.entities: dict[str, MdmEntityRecord] = {}
		self.quality_assessments: dict[str, MdmQualityRecord] = {}
		self.duplicate_candidates: dict[str, MdmDuplicateCandidateRecord] = {}
		self.golden_records: dict[str, MdmGoldenRecord] = {}
		self.merge_requests: dict[str, MdmMergeRequestRecord] = {}
		self.cross_references: dict[str, MdmCrossReferenceRecord] = {}
		self.publish_records: dict[str, MdmPublishRecord] = {}
		self.data_agents: dict[str, MdmDataAgentRecord] = {}
		self.lifecycle_batches: dict[str, MdmLifecycleBatchRecord] = {}
		self.audit_events: list[MdmAuditEventRecord] = []
		self.records: dict[str, dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the current executable MDM contract."""
		return get_capability_contract(tenant_id)

	def create_record(
		self,
		*,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper for older generated package tests."""
		record_id = self._require_text(record_id, "record_id")
		tenant_id = self._require_text(tenant_id, "tenant_id")
		record = {
			"id": record_id,
			"tenant_id": tenant_id,
			"metadata": dict(metadata or {}),
			"status": status,
			"created_at": datetime.utcnow().isoformat(),
		}
		self.records[f"{tenant_id}:{record_id}"] = record
		self._audit(tenant_id, "record.created", record_id, "system", _allow_result(), record)
		return record

	def register_entity(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		entity_type: str,
		name: str,
		business_key: str,
		source_system: str,
		data_owner: str | None,
		classification: str = "internal",
		attributes: dict[str, Any] | None = None,
		audit_evidence: str | None = None,
		classification_evidence: str | None = None,
	) -> MdmEntityRecord:
		"""Register an entity after tenant, type, key, and restricted-data guardrails."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		entity_type = self._require_text(entity_type, "entity_type")
		attributes = dict(attributes or {})
		restricted = classification in {"restricted", "confidential", "sensitive"}
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_entity",
			"unsupported_entity_type": entity_type not in self._supported_entity_types(tenant_id),
			"business_key_present": bool(str(business_key or "").strip()),
			"data_owner_assigned": bool(data_owner),
			"entity_classification": "restricted" if restricted else classification,
			"audit_evidence_present": bool(audit_evidence),
			"restricted_attributes_present": restricted,
			"classification_evidence_present": bool(classification_evidence),
		}
		decision = evaluate_capability_rules(context)
		record = MdmEntityRecord(
			record_id=uuid7str(),
			tenant_id=tenant_id,
			entity_id=self._require_text(entity_id, "entity_id"),
			entity_type=entity_type,
			name=self._require_text(name, "name"),
			business_key=str(business_key or "").strip(),
			source_system=self._require_text(source_system, "source_system"),
			data_owner=data_owner.strip() if isinstance(data_owner, str) and data_owner.strip() else None,
			classification=classification,
			attributes=attributes,
			status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			decision=decision["decision"],
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.entities[self._entity_key(record.tenant_id, record.entity_id)] = record
		self._audit(tenant_id, "entity.registered", record.entity_id, record.data_owner or "system", decision, context)
		return record

	def assess_quality(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		overall_score: float,
		dimensions: dict[str, float],
		assessor: str,
		issues: list[dict[str, Any]] | None = None,
		recommendations: list[str] | None = None,
	) -> MdmQualityRecord:
		"""Record quality evidence and update entity publish readiness."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		entity = self._require_entity(tenant_id, entity_id)
		invalid = overall_score < 0.0 or overall_score > 100.0 or any(
			score < 0.0 or score > 100.0 for score in dimensions.values()
		)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "assess_quality",
			"quality_score_invalid": invalid,
		}
		decision = evaluate_capability_rules(context)
		record = MdmQualityRecord(
			assessment_id=uuid7str(),
			tenant_id=tenant_id,
			entity_id=entity.entity_id,
			overall_score=overall_score,
			dimensions=dict(dimensions),
			assessor=self._require_text(assessor, "assessor"),
			decision=decision["decision"],
			status="accepted" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
			issues=list(issues or []),
			recommendations=list(recommendations or []),
		)
		self.quality_assessments[record.assessment_id] = record
		if decision["decision"] == "allow":
			entity.quality_score = overall_score
			entity.latest_quality_assessment_id = record.assessment_id
			entity.updated_at = datetime.utcnow()
		self._audit(tenant_id, "quality.assessed", entity.entity_id, record.assessor, decision, context)
		return record

	def create_duplicate_candidate(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		candidate_entity_id: str,
		confidence: float,
		reason: str,
		steward_review_recorded: bool = False,
	) -> MdmDuplicateCandidateRecord:
		"""Create a duplicate candidate and route likely matches to stewardship."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		entity = self._require_entity(tenant_id, entity_id)
		candidate = self._require_entity(tenant_id, candidate_entity_id)
		if confidence < 0.0 or confidence > 100.0:
			raise ValueError("confidence must be between 0 and 100")
		decision = evaluate_capability_rules({
			"tenant_context_present": bool(tenant_id),
			"duplicate_confidence": confidence,
			"steward_review_recorded": steward_review_recorded,
		})
		record = MdmDuplicateCandidateRecord(
			candidate_id=uuid7str(),
			tenant_id=tenant_id,
			entity_id=entity.entity_id,
			candidate_entity_id=candidate.entity_id,
			confidence=confidence,
			reason=self._require_text(reason, "reason"),
			decision=decision["decision"],
			status="review_required" if decision["decision"] == "require_review" else "accepted",
			steward_review_recorded=steward_review_recorded,
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision, review_recorded=steward_review_recorded),
		)
		self.duplicate_candidates[record.candidate_id] = record
		entity.duplicate_status = record.status
		self._audit(tenant_id, "duplicate.candidate.created", record.candidate_id, "system", decision, asdict(record))
		return record

	def review_duplicate_candidate(
		self,
		*,
		candidate_id: str,
		steward: str,
		review_decision: str,
		review_notes: str,
	) -> MdmDuplicateCandidateRecord:
		"""Record a stewardship decision for a duplicate candidate."""
		if candidate_id not in self.duplicate_candidates:
			raise KeyError(f"Duplicate candidate {candidate_id} not found")
		record = self.duplicate_candidates[candidate_id]
		review_decision = self._require_choice(review_decision, "review_decision", {"merge", "keep_separate", "defer"})
		context = {
			"tenant_context_present": bool(record.tenant_id),
			"operation": "review",
			"review_notes_present": bool(str(review_notes or "").strip()),
		}
		decision = evaluate_capability_rules(context)
		record.steward = self._require_text(steward, "steward")
		record.review_notes = str(review_notes or "").strip() or None
		record.review_decision = review_decision
		record.steward_review_recorded = decision["decision"] == "allow"
		record.decision = review_decision if decision["decision"] == "allow" else decision["decision"]
		record.status = "reviewed" if decision["decision"] == "allow" else "review_denied"
		record.matched_rules = decision["matched_rules"]
		record.policy_decision = decision["decision"]
		record.review_reasons = self._reasons(decision)
		record.review_evidence = self._review_evidence(decision, review_recorded=decision["decision"] == "allow")
		record.reviewed_at = datetime.utcnow()
		self._audit(record.tenant_id, "duplicate.candidate.reviewed", record.candidate_id, record.steward, decision, context)
		return record

	def create_golden_record(
		self,
		*,
		tenant_id: str,
		entity_type: str,
		source_entity_ids: list[str],
		survivorship_policy: str,
		attributes: dict[str, Any] | None = None,
	) -> MdmGoldenRecord:
		"""Create a golden record shell from governed source entities."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		entity_type = self._require_text(entity_type, "entity_type")
		survivorship_policy = self._require_choice(
			survivorship_policy,
			"survivorship_policy",
			set(self.describe(tenant_id)["configuration"]["survivorship"]["supported_policies"]),
		)
		sources = [self._require_entity(tenant_id, entity_id) for entity_id in source_entity_ids]
		record = MdmGoldenRecord(
			golden_record_id=uuid7str(),
			tenant_id=tenant_id,
			entity_type=entity_type,
			survivorship_policy=survivorship_policy,
			source_entity_ids=[source.entity_id for source in sources],
			status="active",
			attributes=dict(attributes or {}),
		)
		self.golden_records[record.golden_record_id] = record
		for source in sources:
			source.golden_record_id = record.golden_record_id
			source.updated_at = datetime.utcnow()
		self._audit(tenant_id, "golden_record.created", record.golden_record_id, "system", _allow_result(), asdict(record))
		return record

	def merge_golden_record(
		self,
		*,
		tenant_id: str,
		golden_record_id: str,
		source_entity_ids: list[str],
		survivorship_policy: str | None,
		conflict_present: bool = False,
		independent_steward: str | None = None,
		review_notes: str | None = None,
	) -> MdmMergeRequestRecord:
		"""Evaluate and record a golden-record merge request."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		if golden_record_id not in self.golden_records:
			raise KeyError(f"Golden record {golden_record_id} not found")
		for entity_id in source_entity_ids:
			self._require_entity(tenant_id, entity_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "merge_golden_record",
			"survivorship_policy_present": bool(survivorship_policy),
			"conflict_present": conflict_present,
			"independent_steward_present": bool(independent_steward),
		}
		decision = evaluate_capability_rules(context)
		status = "merged" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"])
		record = MdmMergeRequestRecord(
			merge_id=uuid7str(),
			tenant_id=tenant_id,
			golden_record_id=golden_record_id,
			source_entity_ids=list(source_entity_ids),
			survivorship_policy=survivorship_policy,
			conflict_present=conflict_present,
			independent_steward=independent_steward,
			review_notes=review_notes,
			decision=decision["decision"],
			status=status,
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision, review_recorded=bool(independent_steward)),
		)
		self.merge_requests[record.merge_id] = record
		if decision["decision"] == "allow":
			golden_record = self.golden_records[golden_record_id]
			golden_record.source_entity_ids = list(dict.fromkeys(golden_record.source_entity_ids + source_entity_ids))
			golden_record.survivorship_policy = survivorship_policy or golden_record.survivorship_policy
			golden_record.updated_at = datetime.utcnow()
		self._audit(tenant_id, "golden_record.merge_requested", record.merge_id, independent_steward or "system", decision, context)
		return record

	def update_cross_reference(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		source_system: str,
		source_identifier: str,
		evidence_reference: str | None,
	) -> MdmCrossReferenceRecord:
		"""Attach a source-system identifier mapping with evidence."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		entity = self._require_entity(tenant_id, entity_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "update_cross_reference",
			"source_system_evidence_present": bool(evidence_reference),
		}
		decision = evaluate_capability_rules(context)
		record = MdmCrossReferenceRecord(
			cross_reference_id=uuid7str(),
			tenant_id=tenant_id,
			entity_id=entity.entity_id,
			source_system=self._require_text(source_system, "source_system"),
			source_identifier=self._require_text(source_identifier, "source_identifier"),
			evidence_reference=evidence_reference,
			decision=decision["decision"],
			status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision, review_recorded=bool(evidence_reference)),
		)
		self.cross_references[record.cross_reference_id] = record
		self._audit(tenant_id, "cross_reference.updated", record.cross_reference_id, source_system, decision, context)
		return record

	def retire_entity(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		lineage_evidence: str | None,
		actor: str,
	) -> MdmEntityRecord:
		"""Retire an entity only when lineage evidence is present."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		entity = self._require_entity(tenant_id, entity_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "retire_entity",
			"lineage_evidence_present": bool(lineage_evidence),
		}
		decision = evaluate_capability_rules(context)
		entity.decision = decision["decision"]
		entity.matched_rules = decision["matched_rules"]
		entity.policy_decision = decision["decision"]
		entity.review_reasons = self._reasons(decision)
		entity.review_evidence = self._review_evidence(decision, review_recorded=bool(lineage_evidence))
		if decision["decision"] == "allow":
			entity.status = "retired"
			entity.updated_at = datetime.utcnow()
		self._audit(tenant_id, "entity.retired", entity.entity_id, self._require_text(actor, "actor"), decision, context)
		return entity

	def publish_entity(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		channel: str,
	) -> MdmPublishRecord:
		"""Evaluate publish readiness for a mastered entity."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		entity = self._require_entity(tenant_id, entity_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_entity",
			"data_owner_assigned": bool(entity.data_owner),
			"latest_quality_assessment_present": bool(entity.latest_quality_assessment_id),
			"quality_score": entity.quality_score if entity.quality_score is not None else 0.0,
		}
		decision = evaluate_capability_rules(context)
		record = MdmPublishRecord(
			publish_id=uuid7str(),
			tenant_id=tenant_id,
			entity_id=entity.entity_id,
			channel=self._require_text(channel, "channel"),
			decision=decision["decision"],
			status="published" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			quality_score=entity.quality_score,
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.publish_records[record.publish_id] = record
		if decision["decision"] == "allow":
			entity.status = "published"
			entity.updated_at = datetime.utcnow()
		self._audit(tenant_id, "entity.publish_evaluated", record.publish_id, entity.data_owner or "system", decision, context)
		return record

	def register_data_agent(
		self,
		*,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> MdmDataAgentRecord:
		"""Register a first-class MDM data agent with guardrail evidence."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		agent_id = self._require_text(agent_id, "agent_id")
		name = self._require_text(name, "name")
		runtime_value = self._normalize_agent_token(runtime)
		role_value = self._normalize_agent_token(role)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_data_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"agent_scope_present": bool(str(scope or "").strip()),
			"agent_owner_present": bool(str(owner or "").strip()),
			"agent_purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_agent_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		}
		rule_decision = evaluate_capability_rules(context)
		if rule_decision["decision"] == "deny":
			self._audit(
				tenant_id,
				"agent.registration_denied",
				agent_id,
				str(owner or "system").strip() or "system",
				rule_decision,
				context,
			)
			raise PermissionError(self._first_reason(rule_decision))
		record_key = self._agent_key(tenant_id, agent_id)
		if record_key in self.data_agents:
			raise ValueError(f"data_agent_already_exists:{agent_id}")
		record = MdmDataAgentRecord(
			agent_id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=self._require_text(scope, "scope"),
			owner=self._require_text(owner, "owner"),
			purpose=self._require_text(purpose, "purpose"),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status="pending_review" if rule_decision["decision"] == "require_review" else "active",
			policy_decision=rule_decision["decision"],
			matched_rules=list(rule_decision["matched_rules"]),
			review_reasons=self._reasons(rule_decision),
			review_evidence=self._review_evidence(rule_decision, review_recorded=bool(human_approval_required)),
		)
		self.data_agents[record_key] = record
		self._audit(tenant_id, "agent.registered", agent_id, record.owner, rule_decision, asdict(record))
		return record

	def validate_mdm_lifecycle_batch(
		self,
		*,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> MdmLifecycleBatchRecord:
		"""Validate that MDM lifecycle mutation batches flow through Bytewax."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("mdm_lifecycle_batch_empty")
		stream_value = self._normalize_agent_token(event_stream)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_mdm_lifecycle_batch",
			"event_stream": stream_value,
		}
		rule_decision = evaluate_capability_rules(context)
		accepted = rule_decision["decision"] == "allow"
		record = MdmLifecycleBatchRecord(
			batch_id=uuid7str(),
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			accepted=accepted,
			decision=rule_decision["decision"],
			matched_rules=list(rule_decision["matched_rules"]),
			policy_decision=rule_decision["decision"],
			review_reasons=self._reasons(rule_decision),
			review_evidence=self._review_evidence(rule_decision),
			status="accepted" if accepted else "denied",
		)
		self.lifecycle_batches[record.batch_id] = record
		self._audit(tenant_id, f"lifecycle_batch.{record.status}", stream_value, "mdm", rule_decision, asdict(record))
		if not accepted:
			raise PermissionError(self._first_reason(rule_decision))
		return record

	def golden_record_create(
		self,
		*,
		tenant_id: str,
		entity_type: str,
		source_entity_ids: list[str],
		survivorship_policy: str,
		attributes: dict[str, Any] | None = None,
	) -> MdmGoldenRecord:
		"""Create a golden record from governed source entities (alias for create_golden_record)."""
		return self.create_golden_record(
			tenant_id=tenant_id,
			entity_type=entity_type,
			source_entity_ids=source_entity_ids,
			survivorship_policy=survivorship_policy,
			attributes=attributes,
		)

	def match_score(
		self,
		*,
		tenant_id: str,
		entity_id_a: str,
		entity_id_b: str,
	) -> dict[str, Any]:
		"""Compute a lightweight match score between two entities based on shared attributes."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		a = self._require_entity(tenant_id, entity_id_a)
		b = self._require_entity(tenant_id, entity_id_b)
		common_keys = set(a.attributes) & set(b.attributes)
		matching = sum(1 for k in common_keys if str(a.attributes[k]).lower() == str(b.attributes[k]).lower())
		score = round(matching / max(len(common_keys), 1) * 100, 2)
		return {
			"entity_id_a": entity_id_a,
			"entity_id_b": entity_id_b,
			"tenant_id": tenant_id,
			"common_attribute_count": len(common_keys),
			"matching_attribute_count": matching,
			"match_score": score,
			"confidence": "high" if score >= 80 else ("medium" if score >= 50 else "low"),
		}

	def merge_records(
		self,
		*,
		tenant_id: str,
		golden_record_id: str,
		source_entity_ids: list[str],
		survivorship_policy: str | None = None,
		conflict_present: bool = False,
		independent_steward: str | None = None,
		review_notes: str | None = None,
	) -> MdmMergeRequestRecord:
		"""Merge source entities into an existing golden record (alias for merge_golden_record)."""
		return self.merge_golden_record(
			tenant_id=tenant_id,
			golden_record_id=golden_record_id,
			source_entity_ids=source_entity_ids,
			survivorship_policy=survivorship_policy,
			conflict_present=conflict_present,
			independent_steward=independent_steward,
			review_notes=review_notes,
		)

	def split_record(
		self,
		*,
		tenant_id: str,
		golden_record_id: str,
		split_entity_ids: list[str],
		reason: str,
		actor: str,
	) -> dict[str, Any]:
		"""Split a golden record by removing specified source entities from its composition."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		if golden_record_id not in self.golden_records:
			raise KeyError(f"Golden record {golden_record_id} not found")
		gr = self.golden_records[golden_record_id]
		removed = [e for e in split_entity_ids if e in gr.source_entity_ids]
		gr.source_entity_ids = [e for e in gr.source_entity_ids if e not in split_entity_ids]
		gr.updated_at = datetime.utcnow()
		self._audit(tenant_id, "golden_record.split", golden_record_id, actor, _allow_result(), {"removed": removed, "reason": reason})
		return {"golden_record_id": golden_record_id, "removed_entity_ids": removed, "remaining_count": len(gr.source_entity_ids), "reason": reason}

	def survivorship_rule(
		self,
		*,
		tenant_id: str,
		rule_id: str,
		entity_type: str,
		field: str,
		strategy: str,
		priority: int = 1,
		owner: str = "system",
	) -> dict[str, Any]:
		"""Define a survivorship rule for a field determining which source value wins in merges."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		supported = set(self.describe(tenant_id)["configuration"]["survivorship"]["supported_policies"])
		strategy = self._require_choice(strategy, "strategy", supported)
		record = {
			"rule_id": rule_id,
			"tenant_id": tenant_id,
			"entity_type": entity_type,
			"field": field,
			"strategy": strategy,
			"priority": priority,
			"owner": owner,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "survivorship_rule.defined", rule_id, owner, _allow_result(), record)
		return record

	def workflow_approve(
		self,
		*,
		tenant_id: str,
		candidate_id: str,
		steward: str,
		review_decision: str,
		review_notes: str,
	) -> MdmDuplicateCandidateRecord:
		"""Approve or reject a duplicate candidate via stewardship workflow (alias for review_duplicate_candidate)."""
		return self.review_duplicate_candidate(
			candidate_id=candidate_id,
			steward=steward,
			review_decision=review_decision,
			review_notes=review_notes,
		)

	def steward_assign(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		steward_id: str,
		role: str = "data_steward",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Assign a data steward to an entity."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		entity = self._require_entity(tenant_id, entity_id)
		entity.attributes["assigned_steward"] = steward_id
		entity.attributes["steward_role"] = role
		entity.updated_at = datetime.utcnow()
		record = {"entity_id": entity_id, "tenant_id": tenant_id, "steward_id": steward_id, "role": role, "assigned_by": actor, "assigned_at": datetime.utcnow().isoformat()}
		self._audit(tenant_id, "steward.assigned", entity_id, actor, _allow_result(), record)
		return record

	def domain_publish(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		channel: str,
	) -> MdmPublishRecord:
		"""Publish a mastered entity to a domain channel (alias for publish_entity)."""
		return self.publish_entity(tenant_id=tenant_id, entity_id=entity_id, channel=channel)

	def subscription_notify(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		event_type: str,
		subscriber_ids: list[str],
		payload: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Notify downstream subscribers of an MDM entity change event."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		entity = self._require_entity(tenant_id, entity_id)
		record = {
			"entity_id": entity_id,
			"tenant_id": tenant_id,
			"event_type": event_type,
			"subscriber_ids": list(subscriber_ids),
			"subscriber_count": len(subscriber_ids),
			"payload": dict(payload or {}),
			"notified_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "subscription.notified", entity_id, "system", _allow_result(), record)
		return record

	def data_quality_score(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		overall_score: float,
		dimensions: dict[str, float],
		assessor: str,
	) -> MdmQualityRecord:
		"""Record a quality score for an entity (alias for assess_quality)."""
		return self.assess_quality(
			tenant_id=tenant_id,
			entity_id=entity_id,
			overall_score=overall_score,
			dimensions=dimensions,
			assessor=assessor,
		)

	def entity_search(
		self,
		*,
		tenant_id: str,
		entity_type: str | None = None,
		status: str | None = None,
		min_quality_score: float | None = None,
		limit: int = 50,
	) -> list[dict[str, Any]]:
		"""Search entities by type, status, or minimum quality score."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		results = []
		for entity in self.entities.values():
			if entity.tenant_id != tenant_id:
				continue
			if entity_type and entity.entity_type != entity_type:
				continue
			if status and entity.status != status:
				continue
			if min_quality_score is not None and (entity.quality_score is None or entity.quality_score < min_quality_score):
				continue
			results.append(asdict(entity))
			if len(results) >= limit:
				break
		return results

	def entity_bulk_register(
		self,
		*,
		tenant_id: str,
		entities: list[dict[str, Any]],
		data_owner: str | None = None,
	) -> list[dict[str, Any]]:
		"""Register multiple entities in a single call, returning per-entity outcomes."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		outcomes: list[dict[str, Any]] = []
		for e in entities:
			try:
				rec = self.register_entity(
					tenant_id=tenant_id,
					entity_id=e["entity_id"],
					entity_type=e["entity_type"],
					name=e["name"],
					business_key=e.get("business_key", e["entity_id"]),
					source_system=e.get("source_system", "bulk"),
					data_owner=e.get("data_owner", data_owner),
					classification=e.get("classification", "internal"),
					attributes=e.get("attributes"),
				)
				outcomes.append({"status": "registered", "entity_id": e["entity_id"], "record_id": rec.record_id})
			except Exception as exc:
				outcomes.append({"status": "error", "entity_id": e.get("entity_id", "unknown"), "error": str(exc)})
		return outcomes

	def data_lineage(
		self,
		*,
		tenant_id: str,
		entity_id: str,
		lineage_direction: str = "upstream",
	) -> dict[str, Any]:
		"""Return data lineage graph for an entity (upstream sources or downstream consumers)."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		entity = self._require_entity(tenant_id, entity_id)
		assert lineage_direction in {"upstream", "downstream", "both"}, f"invalid direction: {lineage_direction}"
		xrefs = [asdict(xr) for xr in self.cross_references.values() if xr.tenant_id == tenant_id and xr.entity_id == entity_id]
		golden = self.golden_records.get(entity.golden_record_id) if entity.golden_record_id else None
		return {
			"entity_id": entity_id,
			"tenant_id": tenant_id,
			"lineage_direction": lineage_direction,
			"source_system": entity.source_system,
			"cross_references": xrefs,
			"golden_record_id": entity.golden_record_id,
			"golden_record_sources": golden.source_entity_ids if golden else [],
			"generated_at": datetime.utcnow().isoformat(),
		}

	def list_records(self, tenant_id: str | None = None, record_type: str | None = None) -> list[dict[str, Any]]:
		"""List generated-app records for a tenant."""
		tenant_id = tenant_id or self.tenant_id
		collections: dict[str, Any] = {
			"entities": self.entities.values(),
			"quality_assessments": self.quality_assessments.values(),
			"duplicate_candidates": self.duplicate_candidates.values(),
			"golden_records": self.golden_records.values(),
			"merge_requests": self.merge_requests.values(),
			"cross_references": self.cross_references.values(),
			"publish_records": self.publish_records.values(),
			"data_agents": self.data_agents.values(),
			"lifecycle_batches": self.lifecycle_batches.values(),
			"audit_events": self.audit_events,
			"records": self.records.values(),
		}
		if record_type:
			if record_type not in collections:
				raise ValueError(f"Unsupported record_type {record_type}")
			values = collections[record_type]
		else:
			values = []
			for collection in collections.values():
				values.extend(collection)
		return [
			dict(record) if isinstance(record, dict) else asdict(record)
			for record in values
			if (record.get("tenant_id") if isinstance(record, dict) else getattr(record, "tenant_id", None)) == tenant_id
		]

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return summary metrics for generated MDM dashboards."""
		tenant_id = tenant_id or self.tenant_id
		return {
			"tenant_id": tenant_id,
			"entity_count": len(self.list_records(tenant_id, "entities")),
			"quality_assessment_count": len(self.list_records(tenant_id, "quality_assessments")),
			"duplicate_review_count": sum(1 for row in self.list_records(tenant_id, "duplicate_candidates") if row["status"] == "review_required"),
			"golden_record_count": len(self.list_records(tenant_id, "golden_records")),
			"pending_merge_count": sum(1 for row in self.list_records(tenant_id, "merge_requests") if row["status"] == "pending_review"),
			"published_entity_count": sum(1 for row in self.list_records(tenant_id, "entities") if row["status"] == "published"),
			"data_agent_count": len(self.list_records(tenant_id, "data_agents")),
			"pending_data_agent_review_count": sum(1 for row in self.list_records(tenant_id, "data_agents") if row["status"] == "pending_review"),
			"lifecycle_batch_count": len(self.list_records(tenant_id, "lifecycle_batches")),
			"denied_lifecycle_batch_count": sum(1 for row in self.list_records(tenant_id, "lifecycle_batches") if not row["accepted"]),
			"pending_review_count": len(self.list_pending_reviews(tenant_id)),
			"audit_event_count": len(self.list_records(tenant_id, "audit_events")),
		}

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return all MDM records awaiting steward or human review."""
		tenant_id = tenant_id or self.tenant_id
		items = (
			self.list_records(tenant_id, "entities")
			+ self.list_records(tenant_id, "quality_assessments")
			+ self.list_records(tenant_id, "duplicate_candidates")
			+ self.list_records(tenant_id, "merge_requests")
			+ self.list_records(tenant_id, "cross_references")
			+ self.list_records(tenant_id, "publish_records")
			+ self.list_records(tenant_id, "data_agents")
			+ self.list_records(tenant_id, "lifecycle_batches")
		)
		return [
			item
			for item in items
			if item.get("status") in {"pending", "pending_review", "review_required"}
		]

	def _audit(
		self,
		tenant_id: str,
		event_type: str,
		subject: str,
		actor: str,
		policy_result: dict[str, Any],
		details: dict[str, Any],
	) -> None:
		policy_result = policy_result or _allow_result()
		self.audit_events.append(MdmAuditEventRecord(
			event_id=uuid7str(),
			tenant_id=tenant_id,
			event_type=event_type,
			subject=subject,
			actor=actor,
			decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			policy_decision=policy_result["decision"],
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result),
			details=details,
		))

	def _reasons(self, result: dict[str, Any]) -> list[str]:
		return list(dict.fromkeys(
			str(action["reason"])
			for action in result.get("actions", [])
			if action.get("reason")
		))

	def _review_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": list(dict.fromkeys(
				str(action.get("required_action"))
				for action in result.get("actions", [])
				if action.get("required_action")
			)),
			"reasons": self._reasons(result),
			"review_recorded": bool(review_recorded),
		}

	def _supported_entity_types(self, tenant_id: str) -> set[str]:
		return set(self.describe(tenant_id)["configuration"]["entities"]["supported_entity_types"])

	def _require_entity(self, tenant_id: str, entity_id: str) -> MdmEntityRecord:
		entity_id = self._require_text(entity_id, "entity_id")
		entity = self.entities.get(self._entity_key(tenant_id, entity_id))
		if entity is None:
			raise KeyError(f"Entity {entity_id} not found for tenant {tenant_id}")
		if entity.status == "denied":
			raise ValueError(f"Entity {entity_id} is denied and cannot continue lifecycle operations")
		return entity

	@staticmethod
	def _status_for_decision(decision: str) -> str:
		if decision == "require_review":
			return "pending_review"
		if decision == "deny":
			return "denied"
		return "active"

	@staticmethod
	def _require_text(value: str, field_name: str) -> str:
		if not isinstance(value, str) or not value.strip():
			raise ValueError(f"{field_name} is required")
		return value.strip()

	@staticmethod
	def _require_choice(value: str, field_name: str, allowed: set[str]) -> str:
		text = MdmService._require_text(value, field_name)
		if text not in allowed:
			raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
		return text

	@staticmethod
	def _entity_key(tenant_id: str, entity_id: str) -> str:
		return f"{tenant_id}:{entity_id}"

	@staticmethod
	def _agent_key(tenant_id: str, agent_id: str) -> str:
		return f"{tenant_id}:{agent_id}"

	@staticmethod
	def _normalize_agent_token(value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	@staticmethod
	def _first_reason(result: dict[str, Any]) -> str:
		for action in result.get("actions", []):
			if action.get("reason"):
				return str(action["reason"])
		return "mdm_operation_denied"


def _allow_result() -> dict[str, Any]:
	return {"decision": "allow", "matched_rules": [], "actions": []}


# Export main classes
__all__ = [
	'MDMService', 'MdmService', 'EntityService', 'QualityService', 'MatchingService', 'AuditService',
	'MDMOperationType', 'MDMOperationContext', 'MdmEntityRecord', 'MdmQualityRecord',
	'MdmDuplicateCandidateRecord', 'MdmGoldenRecord', 'MdmMergeRequestRecord',
	'MdmCrossReferenceRecord', 'MdmPublishRecord', 'MdmDataAgentRecord',
	'MdmLifecycleBatchRecord', 'MdmAuditEventRecord'
]
