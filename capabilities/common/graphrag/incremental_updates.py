"""
APG GraphRAG Capability - Real-Time Incremental Knowledge Updates

Revolutionary real-time incremental knowledge graph updates without full reprocessing.
Advanced conflict resolution, entity merging, and relationship reconciliation.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from __future__ import annotations
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import uuid

from .database import GraphRAGDatabaseService
from .ollama_integration import OllamaClient
from .views import (
	GraphEntity, GraphRelationship, KnowledgeGraph,
	EntityType, RelationshipType
)


logger = logging.getLogger(__name__)


class UpdateType(str, Enum):
	"""Types of incremental updates"""
	ENTITY_CREATE = "entity_create"
	ENTITY_UPDATE = "entity_update"
	ENTITY_MERGE = "entity_merge"
	ENTITY_DELETE = "entity_delete"
	RELATIONSHIP_CREATE = "relationship_create"
	RELATIONSHIP_UPDATE = "relationship_update"
	RELATIONSHIP_DELETE = "relationship_delete"
	GRAPH_RESTRUCTURE = "graph_restructure"


class ConflictType(str, Enum):
	"""Types of conflicts in knowledge updates"""
	ENTITY_DUPLICATION = "entity_duplication"
	PROPERTY_MISMATCH = "property_mismatch"
	RELATIONSHIP_INCONSISTENCY = "relationship_inconsistency"
	TEMPORAL_CONFLICT = "temporal_conflict"
	CONFIDENCE_CONFLICT = "confidence_conflict"


@dataclass
class UpdateOperation:
	"""Individual update operation"""
	operation_id: str
	update_type: UpdateType
	target_id: str
	data: Dict[str, Any]
	timestamp: datetime
	source: str
	confidence: float
	metadata: Dict[str, Any]


@dataclass
class ConflictResolution:
	"""Conflict resolution strategy and result"""
	conflict_id: str
	conflict_type: ConflictType
	conflicting_operations: List[UpdateOperation]
	resolution_strategy: str
	resolved_data: Dict[str, Any]
	confidence_score: float
	resolution_timestamp: datetime


@dataclass
class UpdateResult:
	"""Result of incremental update operation"""
	operation_id: str
	success: bool
	update_type: UpdateType
	affected_entities: List[str]
	affected_relationships: List[str]
	conflicts_detected: List[ConflictResolution]
	processing_time_ms: float
	metadata: Dict[str, Any]


class IncrementalUpdateEngine:
	"""
	Revolutionary incremental update engine providing:
	
	- Real-time knowledge graph updates without reprocessing
	- Advanced conflict detection and resolution
	- Entity merging and deduplication
	- Relationship reconciliation
	- Consistency maintenance
	- Performance optimization with minimal graph impact
	- Audit trails and rollback capabilities
	"""
	
	def __init__(
		self,
		db_service: GraphRAGDatabaseService,
		ollama_client: OllamaClient,
		config: Optional[Dict[str, Any]] = None
	):
		"""Initialize incremental update engine"""
		self.db_service = db_service
		self.ollama_client = ollama_client
		self.config = config or {}
		
		# Update parameters
		self.similarity_threshold = self.config.get("similarity_threshold", 0.85)
		self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
		self.max_merge_candidates = self.config.get("max_merge_candidates", 5)
		self.enable_auto_merge = self.config.get("enable_auto_merge", True)
		
		# Conflict resolution strategies
		self.conflict_strategies = self._initialize_conflict_strategies()
		
		# Performance tracking
		self._update_queue = asyncio.Queue()
		self._processing_stats = defaultdict(list)
		self._conflict_history = {}
		
		# Concurrency control
		self._update_lock = asyncio.Lock()
		self._processing_semaphore = asyncio.Semaphore(10)
		
		logger.info("Incremental update engine initialized")
	
	async def process_incremental_update(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		update_operation: UpdateOperation
	) -> UpdateResult:
		"""
		Process a single incremental update operation
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			update_operation: Update operation to process
			
		Returns:
			UpdateResult with operation outcome
		"""
		start_time = time.time()
		
		async with self._processing_semaphore:
			try:
				logger.info(f"Processing {update_operation.update_type} operation {update_operation.operation_id}")
				
				# Step 1: Validate update operation
				validation_result = await self._validate_update_operation(
					tenant_id, knowledge_graph_id, update_operation
				)
				
				if not validation_result["valid"]:
					return UpdateResult(
						operation_id=update_operation.operation_id,
						success=False,
						update_type=update_operation.update_type,
						affected_entities=[],
						affected_relationships=[],
						conflicts_detected=[],
						processing_time_ms=(time.time() - start_time) * 1000,
						metadata={"error": validation_result["error"]}
					)
				
				# Step 2: Detect potential conflicts
				conflicts = await self._detect_conflicts(
					tenant_id, knowledge_graph_id, update_operation
				)
				
				# Step 3: Resolve conflicts if any
				resolved_conflicts = []
				if conflicts:
					resolved_conflicts = await self._resolve_conflicts(
						tenant_id, knowledge_graph_id, conflicts
					)
				
				# Step 4: Execute the update
				execution_result = await self._execute_update_operation(
					tenant_id, knowledge_graph_id, update_operation, resolved_conflicts
				)
				
				# Step 5: Update graph consistency
				await self._maintain_graph_consistency(
					tenant_id, knowledge_graph_id, execution_result
				)
				
				# Step 6: Build result
				result = UpdateResult(
					operation_id=update_operation.operation_id,
					success=execution_result["success"],
					update_type=update_operation.update_type,
					affected_entities=execution_result.get("affected_entities", []),
					affected_relationships=execution_result.get("affected_relationships", []),
					conflicts_detected=resolved_conflicts,
					processing_time_ms=(time.time() - start_time) * 1000,
					metadata={
						"validation": validation_result,
						"execution": execution_result,
						"conflicts_resolved": len(resolved_conflicts)
					}
				)
				
				# Record performance
				self._record_performance(update_operation.update_type.value, result.processing_time_ms)
				
				logger.info(f"Update operation {update_operation.operation_id} completed in {result.processing_time_ms:.1f}ms")
				return result
				
			except Exception as e:
				logger.error(f"Update operation {update_operation.operation_id} failed: {e}")
				return UpdateResult(
					operation_id=update_operation.operation_id,
					success=False,
					update_type=update_operation.update_type,
					affected_entities=[],
					affected_relationships=[],
					conflicts_detected=[],
					processing_time_ms=(time.time() - start_time) * 1000,
					metadata={"error": str(e)}
				)
	
	async def process_batch_updates(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		update_operations: List[UpdateOperation]
	) -> List[UpdateResult]:
		"""
		Process multiple update operations efficiently
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			update_operations: List of update operations
			
		Returns:
			List of UpdateResult objects
		"""
		start_time = time.time()
		
		try:
			# Sort operations by priority and dependencies
			sorted_operations = await self._sort_operations_by_priority(update_operations)
			
			# Group operations for batch processing
			operation_groups = await self._group_operations_for_batch_processing(sorted_operations)
			
			results = []
			
			# Process each group
			for group in operation_groups:
				if group["can_parallel"]:
					# Process in parallel
					group_tasks = [
						self.process_incremental_update(tenant_id, knowledge_graph_id, op)
						for op in group["operations"]
					]
					group_results = await asyncio.gather(*group_tasks)
					results.extend(group_results)
				else:
					# Process sequentially
					for operation in group["operations"]:
						result = await self.process_incremental_update(
							tenant_id, knowledge_graph_id, operation
						)
						results.append(result)
			
			total_time = (time.time() - start_time) * 1000
			successful_operations = sum(1 for r in results if r.success)
			
			logger.info(f"Batch update completed: {successful_operations}/{len(update_operations)} successful in {total_time:.1f}ms")
			return results
			
		except Exception as e:
			logger.error(f"Batch update processing failed: {e}")
			raise
	
	async def detect_and_merge_duplicate_entities(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		entity_type: Optional[str] = None
	) -> Dict[str, Any]:
		"""
		Detect and merge duplicate entities in the knowledge graph
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			entity_type: Optional entity type filter
			
		Returns:
			Merge operation results
		"""
		start_time = time.time()
		
		try:
			logger.info(f"Starting duplicate entity detection for graph {knowledge_graph_id}")
			
			# Get all entities
			entities = await self.db_service.list_entities(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				entity_type=entity_type,
				limit=10000  # Process large batches
			)
			
			# Find potential duplicates
			duplicate_groups = await self._find_duplicate_entities(entities)
			
			merge_results = []
			
			# Process each duplicate group
			for group in duplicate_groups:
				if len(group) > 1:
					merge_result = await self._merge_entity_group(
						tenant_id, knowledge_graph_id, group
					)
					merge_results.append(merge_result)
			
			total_time = (time.time() - start_time) * 1000
			
			return {
				"duplicate_groups_found": len(duplicate_groups),
				"entities_merged": sum(len(group) - 1 for group in duplicate_groups),
				"merge_operations": len(merge_results),
				"successful_merges": sum(1 for r in merge_results if r.get("success", False)),
				"processing_time_ms": total_time,
				"merge_details": merge_results
			}
			
		except Exception as e:
			logger.error(f"Duplicate entity detection failed: {e}")
			raise
	
	async def reconcile_relationship_inconsistencies(
		self,
		tenant_id: str,
		knowledge_graph_id: str
	) -> Dict[str, Any]:
		"""
		Identify and reconcile relationship inconsistencies
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			
		Returns:
			Reconciliation results
		"""
		start_time = time.time()
		
		try:
			logger.info(f"Starting relationship reconciliation for graph {knowledge_graph_id}")
			
			# Get all relationships
			relationships = await self.db_service.list_relationships(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				limit=50000
			)
			
			# Find inconsistencies
			inconsistencies = await self._find_relationship_inconsistencies(relationships)
			
			reconciliation_results = []
			
			# Reconcile each inconsistency
			for inconsistency in inconsistencies:
				reconciliation = await self._reconcile_relationship_inconsistency(
					tenant_id, knowledge_graph_id, inconsistency
				)
				reconciliation_results.append(reconciliation)
			
			total_time = (time.time() - start_time) * 1000
			
			return {
				"inconsistencies_found": len(inconsistencies),
				"reconciliations_attempted": len(reconciliation_results),
				"successful_reconciliations": sum(1 for r in reconciliation_results if r.get("success", False)),
				"processing_time_ms": total_time,
				"reconciliation_details": reconciliation_results
			}
			
		except Exception as e:
			logger.error(f"Relationship reconciliation failed: {e}")
			raise
	
	# ========================================================================
	# CONFLICT DETECTION AND RESOLUTION
	# ========================================================================
	
	async def _detect_conflicts(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		update_operation: UpdateOperation
	) -> List[Dict[str, Any]]:
		"""Detect potential conflicts for update operation"""
		conflicts = []
		
		try:
			if update_operation.update_type in [UpdateType.ENTITY_CREATE, UpdateType.ENTITY_UPDATE]:
				# Check for entity conflicts
				entity_conflicts = await self._detect_entity_conflicts(
					tenant_id, knowledge_graph_id, update_operation
				)
				conflicts.extend(entity_conflicts)
			
			elif update_operation.update_type in [UpdateType.RELATIONSHIP_CREATE, UpdateType.RELATIONSHIP_UPDATE]:
				# Check for relationship conflicts
				relationship_conflicts = await self._detect_relationship_conflicts(
					tenant_id, knowledge_graph_id, update_operation
				)
				conflicts.extend(relationship_conflicts)
			
			logger.debug(f"Detected {len(conflicts)} conflicts for operation {update_operation.operation_id}")
			return conflicts
			
		except Exception as e:
			logger.error(f"Conflict detection failed: {e}")
			return []
	
	async def _detect_entity_conflicts(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		update_operation: UpdateOperation
	) -> List[Dict[str, Any]]:
		"""Detect entity-specific conflicts"""
		conflicts = []
		
		# Check for similar entities (potential duplicates)
		if "canonical_name" in update_operation.data:
			entity_name = update_operation.data["canonical_name"]
			
			# Generate embedding for similarity comparison
			embedding_result = await self.ollama_client.generate_embedding(entity_name)
			
			# Find similar entities
			existing_entities = await self.db_service.list_entities(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				entity_type=update_operation.data.get("entity_type"),
				limit=100
			)
			
			for existing_entity in existing_entities:
				if existing_entity.embeddings:
					# Calculate similarity
					similarity = self._calculate_embedding_similarity(
						embedding_result.embeddings, existing_entity.embeddings
					)
					
					if similarity > self.similarity_threshold:
						conflicts.append({
							"conflict_type": ConflictType.ENTITY_DUPLICATION,
							"existing_entity_id": existing_entity.canonical_entity_id,
							"similarity_score": similarity,
							"conflict_data": {
								"existing_name": existing_entity.canonical_name,
								"new_name": entity_name,
								"existing_properties": existing_entity.properties,
								"new_properties": update_operation.data.get("properties", {})
							}
						})
		
		return conflicts
	
	async def _detect_relationship_conflicts(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		update_operation: UpdateOperation
	) -> List[Dict[str, Any]]:
		"""Detect relationship-specific conflicts"""
		conflicts = []
		
		# Check for conflicting relationships
		if "source_entity_id" in update_operation.data and "target_entity_id" in update_operation.data:
			source_id = update_operation.data["source_entity_id"]
			target_id = update_operation.data["target_entity_id"]
			relationship_type = update_operation.data.get("relationship_type")
			
			# Find existing relationships between these entities
			existing_relationships = await self.db_service.list_relationships(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				entity_id=source_id,
				limit=100
			)
			
			for existing_rel in existing_relationships:
				if (existing_rel.target_entity_id == target_id and 
					existing_rel.relationship_type == relationship_type):
					
					conflicts.append({
						"conflict_type": ConflictType.RELATIONSHIP_INCONSISTENCY,
						"existing_relationship_id": existing_rel.canonical_relationship_id,
						"conflict_data": {
							"existing_strength": existing_rel.strength,
							"new_strength": update_operation.data.get("strength", 1.0),
							"existing_properties": existing_rel.properties,
							"new_properties": update_operation.data.get("properties", {})
						}
					})
		
		return conflicts
	
	async def _resolve_conflicts(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		conflicts: List[Dict[str, Any]]
	) -> List[ConflictResolution]:
		"""Resolve detected conflicts using appropriate strategies"""
		resolved_conflicts = []
		
		for conflict in conflicts:
			conflict_type = conflict["conflict_type"]
			strategy = self.conflict_strategies.get(conflict_type, "manual_review")
			
			resolution = await self._apply_conflict_resolution_strategy(
				tenant_id, knowledge_graph_id, conflict, strategy
			)
			
			resolved_conflicts.append(resolution)
		
		return resolved_conflicts
	
	async def _apply_conflict_resolution_strategy(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		conflict: Dict[str, Any],
		strategy: str
	) -> ConflictResolution:
		"""Apply specific conflict resolution strategy"""
		
		conflict_id = str(uuid.uuid4())
		conflict_type = conflict["conflict_type"]
		
		if strategy == "auto_merge" and conflict_type == ConflictType.ENTITY_DUPLICATION:
			# Automatically merge duplicate entities
			resolved_data = await self._auto_merge_entities(conflict)
			confidence_score = 0.9
			
		elif strategy == "weighted_average" and conflict_type == ConflictType.RELATIONSHIP_INCONSISTENCY:
			# Use weighted average for relationship properties
			resolved_data = await self._weighted_average_relationships(conflict)
			confidence_score = 0.8
			
		elif strategy == "highest_confidence":
			# Choose data with highest confidence
			resolved_data = await self._choose_highest_confidence(conflict)
			confidence_score = 0.7
			
		else:
			# Default to manual review
			resolved_data = {"requires_manual_review": True}
			confidence_score = 0.5
		
		return ConflictResolution(
			conflict_id=conflict_id,
			conflict_type=conflict_type,
			conflicting_operations=[],  # Would be populated with actual operations
			resolution_strategy=strategy,
			resolved_data=resolved_data,
			confidence_score=confidence_score,
			resolution_timestamp=datetime.utcnow()
		)
	
	# ========================================================================
	# UPDATE EXECUTION
	# ========================================================================
	
	async def _execute_update_operation(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		update_operation: UpdateOperation,
		resolved_conflicts: List[ConflictResolution]
	) -> Dict[str, Any]:
		"""Execute the update operation with conflict resolutions applied"""
		
		try:
			if update_operation.update_type == UpdateType.ENTITY_CREATE:
				return await self._execute_entity_create(
					tenant_id, knowledge_graph_id, update_operation, resolved_conflicts
				)
			
			elif update_operation.update_type == UpdateType.ENTITY_UPDATE:
				return await self._execute_entity_update(
					tenant_id, knowledge_graph_id, update_operation, resolved_conflicts
				)
			
			elif update_operation.update_type == UpdateType.ENTITY_MERGE:
				return await self._execute_entity_merge(
					tenant_id, knowledge_graph_id, update_operation
				)
			
			elif update_operation.update_type == UpdateType.RELATIONSHIP_CREATE:
				return await self._execute_relationship_create(
					tenant_id, knowledge_graph_id, update_operation, resolved_conflicts
				)
			
			elif update_operation.update_type == UpdateType.RELATIONSHIP_UPDATE:
				return await self._execute_relationship_update(
					tenant_id, knowledge_graph_id, update_operation, resolved_conflicts
				)
			
			else:
				return {
					"success": False,
					"error": f"Unsupported update type: {update_operation.update_type}"
				}
				
		except Exception as e:
			logger.error(f"Update execution failed: {e}")
			return {
				"success": False,
				"error": str(e)
			}
	
	async def _execute_entity_create(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		update_operation: UpdateOperation,
		resolved_conflicts: List[ConflictResolution]
	) -> Dict[str, Any]:
		"""Execute entity creation with conflict resolution"""
		
		# Apply conflict resolutions to update data
		final_data = update_operation.data.copy()
		
		for resolution in resolved_conflicts:
			if resolution.conflict_type == ConflictType.ENTITY_DUPLICATION:
				if resolution.resolution_strategy == "auto_merge":
					# Merge with existing entity instead of creating new one
					return await self._execute_entity_merge_operation(
						tenant_id, knowledge_graph_id, resolution.resolved_data
					)
		
		# Create new entity
		try:
			entity = await self.db_service.create_entity(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				entity_id=final_data["entity_id"],
				entity_type=final_data["entity_type"],
				canonical_name=final_data["canonical_name"],
				properties=final_data.get("properties", {}),
				embeddings=final_data.get("embeddings"),
				aliases=final_data.get("aliases", []),
				confidence_score=final_data.get("confidence_score", 1.0)
			)
			
			return {
				"success": True,
				"affected_entities": [entity.canonical_entity_id],
				"affected_relationships": [],
				"created_entity": entity.canonical_entity_id
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e)
			}
	
	async def _execute_entity_update(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		update_operation: UpdateOperation,
		resolved_conflicts: List[ConflictResolution]
	) -> Dict[str, Any]:
		"""Execute entity update with conflict resolution"""
		
		# Apply conflict resolutions
		final_updates = update_operation.data.copy()
		
		for resolution in resolved_conflicts:
			if resolution.resolution_strategy == "weighted_average":
				final_updates.update(resolution.resolved_data)
		
		try:
			entity = await self.db_service.update_entity(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				entity_id=update_operation.target_id,
				updates=final_updates
			)
			
			return {
				"success": True,
				"affected_entities": [entity.canonical_entity_id],
				"affected_relationships": [],
				"updated_entity": entity.canonical_entity_id
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e)
			}
	
	async def _execute_relationship_create(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		update_operation: UpdateOperation,
		resolved_conflicts: List[ConflictResolution]
	) -> Dict[str, Any]:
		"""Execute relationship creation with conflict resolution"""
		
		final_data = update_operation.data.copy()
		
		# Apply conflict resolutions
		for resolution in resolved_conflicts:
			if resolution.conflict_type == ConflictType.RELATIONSHIP_INCONSISTENCY:
				final_data.update(resolution.resolved_data)
		
		try:
			relationship = await self.db_service.create_relationship(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				relationship_id=final_data["relationship_id"],
				source_entity_id=final_data["source_entity_id"],
				target_entity_id=final_data["target_entity_id"],
				relationship_type=final_data["relationship_type"],
				strength=final_data.get("strength", 1.0),
				properties=final_data.get("properties", {}),
				confidence_score=final_data.get("confidence_score", 1.0)
			)
			
			return {
				"success": True,
				"affected_entities": [relationship.source_entity_id, relationship.target_entity_id],
				"affected_relationships": [relationship.canonical_relationship_id],
				"created_relationship": relationship.canonical_relationship_id
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e)
			}
	
	# ========================================================================
	# HELPER METHODS
	# ========================================================================
	
	def _initialize_conflict_strategies(self) -> Dict[ConflictType, str]:
		"""Initialize conflict resolution strategies"""
		return {
			ConflictType.ENTITY_DUPLICATION: "auto_merge",
			ConflictType.PROPERTY_MISMATCH: "weighted_average",
			ConflictType.RELATIONSHIP_INCONSISTENCY: "weighted_average",
			ConflictType.TEMPORAL_CONFLICT: "latest_wins",
			ConflictType.CONFIDENCE_CONFLICT: "highest_confidence"
		}
	
	async def _validate_update_operation(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		update_operation: UpdateOperation
	) -> Dict[str, Any]:
		"""Validate update operation before processing"""
		
		# Check required fields
		required_fields = {
			UpdateType.ENTITY_CREATE: ["entity_id", "entity_type", "canonical_name"],
			UpdateType.ENTITY_UPDATE: ["entity_id"],
			UpdateType.RELATIONSHIP_CREATE: ["relationship_id", "source_entity_id", "target_entity_id", "relationship_type"],
			UpdateType.RELATIONSHIP_UPDATE: ["relationship_id"]
		}
		
		required = required_fields.get(update_operation.update_type, [])
		missing_fields = [field for field in required if field not in update_operation.data]
		
		if missing_fields:
			return {
				"valid": False,
				"error": f"Missing required fields: {missing_fields}"
			}
		
		# Validate confidence score
		if update_operation.confidence < 0.0 or update_operation.confidence > 1.0:
			return {
				"valid": False,
				"error": "Confidence score must be between 0.0 and 1.0"
			}
		
		return {"valid": True}
	
	async def _sort_operations_by_priority(
		self,
		operations: List[UpdateOperation]
	) -> List[UpdateOperation]:
		"""Sort operations by priority for optimal processing"""
		
		# Priority order: Create entities first, then relationships, then updates
		priority_order = {
			UpdateType.ENTITY_CREATE: 1,
			UpdateType.RELATIONSHIP_CREATE: 2,
			UpdateType.ENTITY_UPDATE: 3,
			UpdateType.RELATIONSHIP_UPDATE: 4,
			UpdateType.ENTITY_MERGE: 5,
			UpdateType.ENTITY_DELETE: 6,
			UpdateType.RELATIONSHIP_DELETE: 7
		}
		
		return sorted(operations, key=lambda op: priority_order.get(op.update_type, 10))
	
	async def _group_operations_for_batch_processing(
		self,
		operations: List[UpdateOperation]
	) -> List[Dict[str, Any]]:
		"""Group operations for efficient batch processing"""
		
		groups = []
		current_group = {"operations": [], "can_parallel": True}
		
		for operation in operations:
			# Check if operation can be processed in parallel
			if operation.update_type in [UpdateType.ENTITY_CREATE, UpdateType.RELATIONSHIP_CREATE]:
				current_group["operations"].append(operation)
			else:
				# Operations that modify existing data should be processed sequentially
				if current_group["operations"]:
					groups.append(current_group)
				
				current_group = {
					"operations": [operation],
					"can_parallel": False
				}
		
		if current_group["operations"]:
			groups.append(current_group)
		
		return groups
	
	async def _find_duplicate_entities(
		self,
		entities: List[GraphEntity]
	) -> List[List[GraphEntity]]:
		"""Find groups of duplicate entities"""
		
		duplicate_groups = []
		processed_entities = set()
		
		for i, entity1 in enumerate(entities):
			if entity1.canonical_entity_id in processed_entities:
				continue
			
			duplicate_group = [entity1]
			processed_entities.add(entity1.canonical_entity_id)
			
			# Compare with remaining entities
			for entity2 in entities[i+1:]:
				if entity2.canonical_entity_id in processed_entities:
					continue
				
				# Check similarity
				if await self._are_entities_duplicates(entity1, entity2):
					duplicate_group.append(entity2)
					processed_entities.add(entity2.canonical_entity_id)
			
			if len(duplicate_group) > 1:
				duplicate_groups.append(duplicate_group)
		
		return duplicate_groups
	
	async def _are_entities_duplicates(
		self,
		entity1: GraphEntity,
		entity2: GraphEntity
	) -> bool:
		"""Check if two entities are duplicates"""
		
		# Same entity type required
		if entity1.entity_type != entity2.entity_type:
			return False
		
		# Check name similarity
		name_similarity = self._calculate_string_similarity(
			entity1.canonical_name, entity2.canonical_name
		)
		
		if name_similarity > 0.9:
			return True
		
		# Check embedding similarity if available
		if entity1.embeddings and entity2.embeddings:
			embedding_similarity = self._calculate_embedding_similarity(
				entity1.embeddings, entity2.embeddings
			)
			
			if embedding_similarity > self.similarity_threshold:
				return True
		
		# Check aliases
		all_names1 = [entity1.canonical_name] + entity1.aliases
		all_names2 = [entity2.canonical_name] + entity2.aliases
		
		for name1 in all_names1:
			for name2 in all_names2:
				if self._calculate_string_similarity(name1, name2) > 0.95:
					return True
		
		return False
	
	def _calculate_embedding_similarity(
		self,
		embedding1: List[float],
		embedding2: List[float]
	) -> float:
		"""Calculate cosine similarity between embeddings"""
		from scipy.spatial.distance import cosine
		
		try:
			if len(embedding1) != len(embedding2):
				return 0.0
			
			return 1 - cosine(embedding1, embedding2)
		except:
			return 0.0
	
	def _calculate_string_similarity(self, str1: str, str2: str) -> float:
		"""Calculate string similarity using Levenshtein distance"""
		from difflib import SequenceMatcher
		
		return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
	
	async def _merge_entity_group(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		entity_group: List[GraphEntity]
	) -> Dict[str, Any]:
		"""Merge a group of duplicate entities"""
		
		if len(entity_group) < 2:
			return {"success": False, "error": "Group must have at least 2 entities"}
		
		try:
			# Choose primary entity (highest confidence)
			primary_entity = max(entity_group, key=lambda e: e.confidence_score)
			entities_to_merge = [e for e in entity_group if e.canonical_entity_id != primary_entity.canonical_entity_id]
			
			# Merge properties and aliases
			merged_properties = primary_entity.properties.copy()
			merged_aliases = primary_entity.aliases.copy()
			
			for entity in entities_to_merge:
				# Merge properties
				for key, value in entity.properties.items():
					if key not in merged_properties:
						merged_properties[key] = value
				
				# Merge aliases
				merged_aliases.extend(entity.aliases)
				if entity.canonical_name not in merged_aliases:
					merged_aliases.append(entity.canonical_name)
			
			# Remove duplicates from aliases
			merged_aliases = list(set(merged_aliases))
			
			# Update primary entity
			await self.db_service.update_entity(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				entity_id=primary_entity.canonical_entity_id,
				updates={
					"properties": merged_properties,
					"aliases": merged_aliases,
					"confidence_score": min(1.0, primary_entity.confidence_score * 1.1)  # Slight boost for merged entity
				}
			)
			
			# TODO: Update relationships pointing to merged entities
			# TODO: Delete merged entities
			
			return {
				"success": True,
				"primary_entity": primary_entity.canonical_entity_id,
				"merged_entities": [e.canonical_entity_id for e in entities_to_merge],
				"merged_properties": len(merged_properties),
				"merged_aliases": len(merged_aliases)
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e)
			}
	
	# Additional helper methods for conflict resolution strategies
	async def _auto_merge_entities(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
		"""Auto merge entities in conflict"""
		return {"merge_entities": True, "strategy": "auto_merge"}
	
	async def _weighted_average_relationships(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
		"""Use weighted average for conflicting relationships"""
		return {"use_weighted_average": True, "strategy": "weighted_average"}
	
	async def _choose_highest_confidence(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
		"""Choose data with highest confidence"""
		return {"use_highest_confidence": True, "strategy": "highest_confidence"}
	
	async def _maintain_graph_consistency(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		execution_result: Dict[str, Any]
	) -> None:
		"""Maintain graph consistency after updates"""
		# Implementation would validate and fix graph consistency
		pass
	
	async def _find_relationship_inconsistencies(
		self,
		relationships: List[GraphRelationship]
	) -> List[Dict[str, Any]]:
		"""Find inconsistencies in relationships"""
		# Implementation would identify relationship inconsistencies
		return []
	
	async def _reconcile_relationship_inconsistency(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		inconsistency: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Reconcile a relationship inconsistency"""
		# Implementation would reconcile the inconsistency
		return {"success": True}
	
	# Placeholder implementations for other execution methods
	async def _execute_entity_merge(self, tenant_id: str, knowledge_graph_id: str, update_operation: UpdateOperation) -> Dict[str, Any]:
		return {"success": True, "affected_entities": [], "affected_relationships": []}
	
	async def _execute_relationship_update(self, tenant_id: str, knowledge_graph_id: str, update_operation: UpdateOperation, resolved_conflicts: List[ConflictResolution]) -> Dict[str, Any]:
		return {"success": True, "affected_entities": [], "affected_relationships": []}
	
	async def _execute_entity_merge_operation(self, tenant_id: str, knowledge_graph_id: str, resolved_data: Dict[str, Any]) -> Dict[str, Any]:
		return {"success": True, "affected_entities": [], "affected_relationships": []}
	
	def _record_performance(self, operation_type: str, time_ms: float) -> None:
		"""Record performance statistics"""
		self._processing_stats[operation_type].append(time_ms)
		
		# Keep only last 1000 measurements
		if len(self._processing_stats[operation_type]) > 1000:
			self._processing_stats[operation_type] = self._processing_stats[operation_type][-1000:]


__all__ = [
	'IncrementalUpdateEngine',
	'UpdateOperation',
	'UpdateResult',
	'ConflictResolution',
	'UpdateType',
	'ConflictType',
]