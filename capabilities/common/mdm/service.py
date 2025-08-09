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
from dataclasses import dataclass
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
from .database import MDMDatabaseManager


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


class EntityService:
	"""Core entity management service with advanced CRUD operations"""
	
	def __init__(self, db_manager: MDMDatabaseManager):
		self.db_manager = db_manager
		self.quality_service = None  # Will be injected
		self.matching_service = None  # Will be injected
		self.audit_service = None  # Will be injected
	
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
					asyncio.create_task(
						self.matching_service.detect_duplicates(entity_id, context.tenant_id)
					)
				
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
					asyncio.create_task(
						self.matching_service.detect_duplicates(entity_id, context.tenant_id)
					)
				
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


# Export main classes
__all__ = [
	'MDMService', 'EntityService', 'QualityService', 'MatchingService', 'AuditService',
	'MDMOperationType', 'MDMOperationContext'
]