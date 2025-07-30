"""
APG RAG Conversation Management System

Sophisticated conversation management with persistent context, memory consolidation,
and intelligent turn-by-turn RAG processing with PostgreSQL storage.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str
from collections import deque, defaultdict

# Database imports
import asyncpg
from asyncpg import Pool

# APG imports
from .models import (
	Conversation, ConversationCreate, ConversationUpdate,
	ConversationTurn, ConversationTurnCreate, ConversationStatus,
	TurnType, RetrievalRequest, RetrievalMethod, GenerationRequest,
	APGBaseModel
)
from .retrieval_engine import IntelligentRetrievalEngine, RetrievalContext
from .generation_engine import RAGGenerationEngine, GenerationContext
from .ollama_integration import RequestPriority

class MemoryStrategy(str, Enum):
	"""Memory management strategies"""
	FULL_RETENTION = "full_retention"
	SLIDING_WINDOW = "sliding_window"
	IMPORTANCE_BASED = "importance_based"
	TOPIC_BASED = "topic_based"
	HYBRID = "hybrid"

class ContextType(str, Enum):
	"""Types of conversation context"""
	FACTUAL = "factual"
	PROCEDURAL = "procedural"
	ANALYTICAL = "analytical"
	CREATIVE = "creative"
	MIXED = "mixed"

@dataclass
class ConversationConfig:
	"""Configuration for conversation management"""
	# Memory management
	memory_strategy: MemoryStrategy = MemoryStrategy.HYBRID
	max_turns_in_memory: int = 20
	max_context_tokens: int = 6000  # Reserve 2k for new turn
	memory_consolidation_threshold: int = 10
	
	# Context management
	context_window_turns: int = 5
	context_relevance_threshold: float = 0.7
	enable_context_summarization: bool = True
	
	# Turn processing
	auto_retrieve_threshold: float = 0.8
	max_retrievals_per_turn: int = 10
	enable_follow_up_detection: bool = True
	
	# Quality control
	enable_coherence_checking: bool = True
	enable_context_validation: bool = True
	min_response_quality: float = 0.6
	
	# Performance
	turn_timeout_seconds: float = 30.0
	enable_async_processing: bool = True

@dataclass
class ConversationMemory:
	"""Structured conversation memory"""
	key_facts: List[str] = field(default_factory=list)
	important_entities: Dict[str, int] = field(default_factory=dict)  # entity -> mention count
	topic_progression: List[str] = field(default_factory=list)
	user_preferences: Dict[str, Any] = field(default_factory=dict)
	context_summary: str = ""
	last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class TurnContext:
	"""Context for processing a single turn"""
	conversation: Conversation
	previous_turns: List[ConversationTurn]
	memory: ConversationMemory
	user_context: Dict[str, Any] = field(default_factory=dict)
	retrieval_context: Optional[RetrievalContext] = None

class MemoryManager:
	"""Manages conversation memory and context consolidation"""
	
	def __init__(self, config: ConversationConfig):
		self.config = config
		self.logger = logging.getLogger(__name__)
	
	async def update_memory(self, conversation: Conversation, turns: List[ConversationTurn]) -> ConversationMemory:
		"""Update conversation memory based on turns"""
		
		# Get existing memory or create new
		existing_memory = self._deserialize_memory(conversation.context_summary) if conversation.context_summary else ConversationMemory()
		
		# Process recent turns for memory updates
		recent_turns = turns[-self.config.memory_consolidation_threshold:] if len(turns) > self.config.memory_consolidation_threshold else turns
		
		# Update key facts
		new_facts = await self._extract_key_facts(recent_turns)
		existing_memory.key_facts.extend(new_facts)
		existing_memory.key_facts = self._deduplicate_facts(existing_memory.key_facts)
		
		# Update entities
		new_entities = self._extract_entities(recent_turns)
		for entity, count in new_entities.items():
			existing_memory.important_entities[entity] = existing_memory.important_entities.get(entity, 0) + count
		
		# Update topic progression
		new_topics = self._extract_topics(recent_turns)
		existing_memory.topic_progression.extend(new_topics)
		
		# Update context summary
		if self.config.enable_context_summarization:
			existing_memory.context_summary = await self._generate_context_summary(existing_memory, recent_turns)
		
		existing_memory.last_updated = datetime.now()
		
		# Consolidate memory if it gets too large
		if len(existing_memory.key_facts) > 50:
			existing_memory = await self._consolidate_memory(existing_memory)
		
		return existing_memory
	
	async def _extract_key_facts(self, turns: List[ConversationTurn]) -> List[str]:
		"""Extract key facts from recent turns"""
		facts = []
		
		for turn in turns:
			if turn.turn_type == TurnType.ASSISTANT:
				# Extract factual statements (simple approach)
				content = turn.content
				sentences = content.split('.')
				
				for sentence in sentences:
					sentence = sentence.strip()
					if len(sentence) > 20 and any(indicator in sentence.lower() for indicator in ['is', 'are', 'was', 'were', 'has', 'have']):
						facts.append(sentence)
		
		return facts[:10]  # Limit to 10 new facts per update
	
	def _extract_entities(self, turns: List[ConversationTurn]) -> Dict[str, int]:
		"""Extract and count entities from turns"""
		entities = defaultdict(int)
		
		for turn in turns:
			# Simple entity extraction (capitalize words)
			words = turn.content.split()
			for word in words:
				if word[0].isupper() and len(word) > 2 and word.isalpha():
					entities[word] += 1
		
		return dict(entities)
	
	def _extract_topics(self, turns: List[ConversationTurn]) -> List[str]:
		"""Extract topics from turns"""
		topics = []
		
		# Simple topic extraction based on question content
		for turn in turns:
			if turn.turn_type == TurnType.USER and '?' in turn.content:
				# Extract main topic from question
				content = turn.content.lower()
				if 'about' in content:
					about_index = content.find('about')
					topic_part = content[about_index+5:about_index+50]
					topic = topic_part.split()[0] if topic_part.split() else ""
					if topic and len(topic) > 3:
						topics.append(topic)
		
		return topics
	
	async def _generate_context_summary(self, memory: ConversationMemory, recent_turns: List[ConversationTurn]) -> str:
		"""Generate a concise context summary"""
		summary_parts = []
		
		# Key facts summary
		if memory.key_facts:
			summary_parts.append(f"Key facts: {'; '.join(memory.key_facts[-5:])}")
		
		# Entity summary
		if memory.important_entities:
			top_entities = sorted(memory.important_entities.items(), key=lambda x: x[1], reverse=True)[:5]
			entity_summary = ", ".join([f"{entity}({count})" for entity, count in top_entities])
			summary_parts.append(f"Important entities: {entity_summary}")
		
		# Topic progression
		if memory.topic_progression:
			topic_summary = " → ".join(memory.topic_progression[-3:])
			summary_parts.append(f"Topic flow: {topic_summary}")
		
		return " | ".join(summary_parts)
	
	async def _consolidate_memory(self, memory: ConversationMemory) -> ConversationMemory:
		"""Consolidate memory when it becomes too large"""
		# Keep only most important facts (those mentioned multiple times or recently)
		consolidated_facts = memory.key_facts[-25:]  # Keep most recent 25 facts
		
		# Keep only top entities
		top_entities = dict(sorted(memory.important_entities.items(), key=lambda x: x[1], reverse=True)[:20])
		
		# Keep recent topics
		recent_topics = memory.topic_progression[-10:]
		
		return ConversationMemory(
			key_facts=consolidated_facts,
			important_entities=top_entities,
			topic_progression=recent_topics,
			user_preferences=memory.user_preferences,
			context_summary=memory.context_summary,
			last_updated=datetime.now()
		)
	
	def _serialize_memory(self, memory: ConversationMemory) -> str:
		"""Serialize memory to JSON string"""
		memory_dict = {
			'key_facts': memory.key_facts,
			'important_entities': memory.important_entities,
			'topic_progression': memory.topic_progression,
			'user_preferences': memory.user_preferences,
			'context_summary': memory.context_summary,
			'last_updated': memory.last_updated.isoformat()
		}
		return json.dumps(memory_dict)
	
	def _deserialize_memory(self, memory_json: str) -> ConversationMemory:
		"""Deserialize memory from JSON string"""
		try:
			memory_dict = json.loads(memory_json) if memory_json else {}
			return ConversationMemory(
				key_facts=memory_dict.get('key_facts', []),
				important_entities=memory_dict.get('important_entities', {}),
				topic_progression=memory_dict.get('topic_progression', []),
				user_preferences=memory_dict.get('user_preferences', {}),
				context_summary=memory_dict.get('context_summary', ''),
				last_updated=datetime.fromisoformat(memory_dict.get('last_updated', datetime.now().isoformat()))
			)
		except (json.JSONDecodeError, ValueError):
			return ConversationMemory()

class TurnProcessor:
	"""Processes individual conversation turns with RAG integration"""
	
	def __init__(self,
	             config: ConversationConfig,
	             retrieval_engine: IntelligentRetrievalEngine,
	             generation_engine: RAGGenerationEngine):
		self.config = config
		self.retrieval_engine = retrieval_engine
		self.generation_engine = generation_engine
		self.logger = logging.getLogger(__name__)
	
	async def process_user_turn(self, user_input: str, turn_context: TurnContext) -> ConversationTurn:
		"""Process user turn and generate appropriate response"""
		
		# Create user turn
		user_turn = ConversationTurn(
			tenant_id=turn_context.conversation.tenant_id,
			conversation_id=turn_context.conversation.id,
			turn_number=len(turn_context.previous_turns) + 1,
			turn_type=TurnType.USER,
			content=user_input,
			content_tokens=len(user_input.split())
		)
		
		# Determine if retrieval is needed
		needs_retrieval = await self._should_retrieve(user_input, turn_context)
		
		retrieved_chunks = []
		retrieval_scores = []
		retrieval_result = None
		
		if needs_retrieval and turn_context.conversation.knowledge_base_id:
			# Perform retrieval
			retrieval_request = RetrievalRequest(
				query_text=user_input,
				knowledge_base_id=turn_context.conversation.knowledge_base_id,
				k_retrievals=self.config.max_retrievals_per_turn,
				similarity_threshold=self.config.auto_retrieve_threshold,
				retrieval_method=RetrievalMethod.HYBRID_SEARCH
			)
			
			retrieval_result = await self.retrieval_engine.retrieve(
				retrieval_request, 
				turn_context.retrieval_context
			)
			
			retrieved_chunks = retrieval_result.retrieved_chunk_ids
			retrieval_scores = retrieval_result.similarity_scores
		
		# Update user turn with retrieval info
		user_turn.retrieved_chunks = retrieved_chunks
		user_turn.retrieval_scores = retrieval_scores
		
		return user_turn
	
	async def generate_assistant_response(self, user_turn: ConversationTurn, turn_context: TurnContext) -> ConversationTurn:
		"""Generate assistant response using RAG"""
		
		start_time = time.time()
		
		# Prepare generation request
		generation_request = GenerationRequest(
			prompt=user_turn.content,
			conversation_id=turn_context.conversation.id,
			model=turn_context.conversation.generation_model,
			max_tokens=turn_context.conversation.max_context_tokens // 4,  # Reserve space for context
			temperature=turn_context.conversation.temperature
		)
		
		# Get retrieval result if chunks were retrieved
		retrieval_result = None
		if user_turn.retrieved_chunks:
			retrieval_result = await self._create_retrieval_result_from_chunks(
				user_turn.content, user_turn.retrieved_chunks, user_turn.retrieval_scores, turn_context
			)
		
		# Generate response
		generation_result = await self.generation_engine.generate_response(
			generation_request,
			retrieval_result,
			turn_context.previous_turns[-self.config.context_window_turns:]
		)
		
		# Create assistant turn
		assistant_turn = ConversationTurn(
			tenant_id=turn_context.conversation.tenant_id,
			conversation_id=turn_context.conversation.id,
			turn_number=user_turn.turn_number + 1,
			turn_type=TurnType.ASSISTANT,
			content=generation_result.response_text,
			content_tokens=generation_result.token_count,
			model_used=generation_result.generation_model,
			generation_time_ms=int(generation_result.generation_time_ms),
			generation_tokens=generation_result.token_count,
			confidence_score=generation_result.confidence_score,
			context_used=json.dumps({
				'sources_count': len(generation_result.sources_used),
				'factual_accuracy': generation_result.factual_accuracy_score,
				'citation_coverage': generation_result.citation_coverage
			})
		)
		
		processing_time_ms = (time.time() - start_time) * 1000
		self.logger.info(f"Generated assistant response in {processing_time_ms:.1f}ms")
		
		return assistant_turn
	
	async def _should_retrieve(self, user_input: str, turn_context: TurnContext) -> bool:
		"""Determine if retrieval is needed for this turn"""
		
		# Always retrieve if no knowledge base
		if not turn_context.conversation.knowledge_base_id:
			return False
		
		# Check for question indicators
		has_question = '?' in user_input or any(
			word in user_input.lower() 
			for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']
		)
		
		# Check for factual request indicators  
		has_factual_request = any(
			word in user_input.lower()
			for word in ['explain', 'describe', 'tell me', 'show me', 'define']
		)
		
		# Check for follow-up indicators
		is_followup = self.config.enable_follow_up_detection and any(
			word in user_input.lower()
			for word in ['more', 'also', 'additionally', 'furthermore', 'what about']
		)
		
		# Check conversation context
		has_knowledge_context = (
			turn_context.memory.important_entities or 
			turn_context.memory.key_facts or
			any(turn.retrieved_chunks for turn in turn_context.previous_turns[-3:])
		)
		
		# Decision logic
		return (has_question or has_factual_request or is_followup) and (
			len(turn_context.previous_turns) == 0 or  # First turn
			has_knowledge_context or  # Context suggests knowledge needed
			len(user_input.split()) > 10  # Complex question
		)
	
	async def _create_retrieval_result_from_chunks(self,
	                                              query: str,
	                                              chunk_ids: List[str],
	                                              scores: List[float],
	                                              turn_context: TurnContext):
		"""Create retrieval result from existing chunks"""
		# This would typically reconstruct the retrieval result
		# For now, return None and let generation work without explicit retrieval result
		return None

class ConversationManager:
	"""Main conversation management system"""
	
	def __init__(self,
	             config: ConversationConfig,
	             db_pool: Pool,
	             retrieval_engine: IntelligentRetrievalEngine,
	             generation_engine: RAGGenerationEngine,
	             tenant_id: str,
	             capability_id: str = "rag"):
		
		self.config = config
		self.db_pool = db_pool
		self.retrieval_engine = retrieval_engine
		self.generation_engine = generation_engine
		self.tenant_id = tenant_id
		self.capability_id = capability_id
		
		# Core components
		self.memory_manager = MemoryManager(config)
		self.turn_processor = TurnProcessor(config, retrieval_engine, generation_engine)
		
		# Active conversations cache
		self.active_conversations = {}
		self.conversation_locks = defaultdict(asyncio.Lock)
		
		# Statistics
		self.stats = {
			'total_conversations': 0,
			'active_conversations': 0,
			'total_turns': 0,
			'average_turns_per_conversation': 0.0,
			'average_response_time_ms': 0.0,
			'retrieval_rate': 0.0
		}
		
		self.logger = logging.getLogger(__name__)
	
	async def create_conversation(self, conversation_create: ConversationCreate) -> Conversation:
		"""Create a new conversation"""
		
		conversation = Conversation(
			tenant_id=self.tenant_id,
			knowledge_base_id=conversation_create.knowledge_base_id,
			title=conversation_create.title,
			description=conversation_create.description,
			generation_model=conversation_create.generation_model,
			max_context_tokens=conversation_create.max_context_tokens,
			temperature=conversation_create.temperature,
			user_id=conversation_create.user_id,
			session_id=conversation_create.session_id
		)
		
		# Store in database
		async with self.db_pool.acquire() as conn:
			await conn.execute("""
				INSERT INTO apg_rag_conversations (
					id, tenant_id, knowledge_base_id, title, description, 
					generation_model, max_context_tokens, temperature,
					status, turn_count, total_tokens_used, user_id, session_id,
					created_at, updated_at, created_by, updated_by
				) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
			""", conversation.id, conversation.tenant_id, conversation.knowledge_base_id,
			     conversation.title, conversation.description, conversation.generation_model,
			     conversation.max_context_tokens, conversation.temperature,
			     conversation.status.value, conversation.turn_count, conversation.total_tokens_used,
			     conversation.user_id, conversation.session_id,
			     conversation.created_at, conversation.updated_at,
			     conversation.created_by, conversation.updated_by)
		
		# Add to active conversations
		self.active_conversations[conversation.id] = conversation
		self.stats['total_conversations'] += 1
		self.stats['active_conversations'] += 1
		
		self.logger.info(f"Created conversation {conversation.id}")
		return conversation
	
	async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
		"""Get conversation by ID"""
		
		# Check cache first
		if conversation_id in self.active_conversations:
			return self.active_conversations[conversation_id]
		
		# Load from database
		async with self.db_pool.acquire() as conn:
			row = await conn.fetchrow("""
				SELECT * FROM apg_rag_conversations 
				WHERE id = $1 AND tenant_id = $2
			""", conversation_id, self.tenant_id)
			
			if row:
				conversation = Conversation(**dict(row))
				self.active_conversations[conversation_id] = conversation
				return conversation
		
		return None
	
	async def process_user_message(self,
	                              conversation_id: str,
	                              user_message: str,
	                              user_context: Dict[str, Any] = None) -> Tuple[ConversationTurn, ConversationTurn]:
		"""Process user message and generate response"""
		
		async with self.conversation_locks[conversation_id]:
			start_time = time.time()
			
			# Get conversation
			conversation = await self.get_conversation(conversation_id)
			if not conversation:
				raise ValueError(f"Conversation {conversation_id} not found")
			
			# Get conversation history
			previous_turns = await self._get_conversation_turns(conversation_id)
			
			# Get/update memory
			memory = await self.memory_manager.update_memory(conversation, previous_turns)
			
			# Create turn context
			turn_context = TurnContext(
				conversation=conversation,
				previous_turns=previous_turns,
				memory=memory,
				user_context=user_context or {},
				retrieval_context=RetrievalContext(
					user_id=conversation.user_id,
					session_id=conversation.session_id,
					conversation_history=[turn.content for turn in previous_turns[-5:]],
					user_preferences=memory.user_preferences
				)
			)
			
			# Process user turn
			user_turn = await self.turn_processor.process_user_turn(user_message, turn_context)
			
			# Generate assistant response
			assistant_turn = await self.turn_processor.generate_assistant_response(user_turn, turn_context)
			
			# Store turns in database
			await self._store_conversation_turn(user_turn)
			await self._store_conversation_turn(assistant_turn)
			
			# Update conversation metadata
			await self._update_conversation_metadata(conversation, [user_turn, assistant_turn], memory)
			
			# Update statistics
			processing_time_ms = (time.time() - start_time) * 1000
			self._update_stats(processing_time_ms, bool(user_turn.retrieved_chunks))
			
			self.logger.info(f"Processed conversation turn for {conversation_id} in {processing_time_ms:.1f}ms")
			
			return user_turn, assistant_turn
	
	async def _get_conversation_turns(self, conversation_id: str) -> List[ConversationTurn]:
		"""Get all turns for a conversation"""
		
		async with self.db_pool.acquire() as conn:
			rows = await conn.fetch("""
				SELECT * FROM apg_rag_conversation_turns 
				WHERE conversation_id = $1 AND tenant_id = $2
				ORDER BY turn_number ASC
			""", conversation_id, self.tenant_id)
			
			turns = []
			for row in rows:
				turn_dict = dict(row)
				# Handle JSON fields
				if turn_dict.get('query_embedding'):
					turn_dict['query_embedding'] = list(turn_dict['query_embedding'])
				if turn_dict.get('retrieved_chunks'):
					turn_dict['retrieved_chunks'] = list(turn_dict['retrieved_chunks'])
				if turn_dict.get('retrieval_scores'):
					turn_dict['retrieval_scores'] = list(turn_dict['retrieval_scores'])
				
				turn = ConversationTurn(**turn_dict)
				turns.append(turn)
			
			return turns
	
	async def _store_conversation_turn(self, turn: ConversationTurn) -> None:
		"""Store conversation turn in database"""
		
		async with self.db_pool.acquire() as conn:
			await conn.execute("""
				INSERT INTO apg_rag_conversation_turns (
					id, tenant_id, conversation_id, turn_number, turn_type,
					content, content_tokens, query_embedding, retrieved_chunks, retrieval_scores,
					model_used, generation_time_ms, generation_tokens, confidence_score,
					context_used, memory_summary, created_at
				) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
			""", turn.id, turn.tenant_id, turn.conversation_id, turn.turn_number, turn.turn_type.value,
			     turn.content, turn.content_tokens, turn.query_embedding, turn.retrieved_chunks, turn.retrieval_scores,
			     turn.model_used, turn.generation_time_ms, turn.generation_tokens, turn.confidence_score,
			     turn.context_used, turn.memory_summary, turn.created_at)
	
	async def _update_conversation_metadata(self,
	                                       conversation: Conversation,
	                                       new_turns: List[ConversationTurn],
	                                       memory: ConversationMemory) -> None:
		"""Update conversation metadata after new turns"""
		
		# Update turn count and token usage
		conversation.turn_count += len(new_turns)
		conversation.total_tokens_used += sum(turn.content_tokens or 0 for turn in new_turns)
		conversation.context_summary = self.memory_manager._serialize_memory(memory)
		conversation.updated_at = datetime.now()
		
		# Store updates
		async with self.db_pool.acquire() as conn:
			await conn.execute("""
				UPDATE apg_rag_conversations 
				SET turn_count = $1, total_tokens_used = $2, context_summary = $3, updated_at = $4
				WHERE id = $5 AND tenant_id = $6
			""", conversation.turn_count, conversation.total_tokens_used, conversation.context_summary,
			     conversation.updated_at, conversation.id, conversation.tenant_id)
		
		# Update cache
		self.active_conversations[conversation.id] = conversation
	
	async def update_conversation(self, conversation_id: str, updates: ConversationUpdate) -> Optional[Conversation]:
		"""Update conversation details"""
		
		conversation = await self.get_conversation(conversation_id)
		if not conversation:
			return None
		
		# Apply updates
		update_fields = []
		update_values = []
		param_count = 0
		
		if updates.title is not None:
			param_count += 1
			update_fields.append(f"title = ${param_count}")
			update_values.append(updates.title)
			conversation.title = updates.title
		
		if updates.description is not None:
			param_count += 1
			update_fields.append(f"description = ${param_count}")
			update_values.append(updates.description)
			conversation.description = updates.description
		
		if updates.status is not None:
			param_count += 1
			update_fields.append(f"status = ${param_count}")
			update_values.append(updates.status.value)
			conversation.status = updates.status
		
		if updates.context_summary is not None:
			param_count += 1
			update_fields.append(f"context_summary = ${param_count}")
			update_values.append(updates.context_summary)
			conversation.context_summary = updates.context_summary
		
		if update_fields:
			param_count += 1
			update_fields.append(f"updated_at = ${param_count}")
			update_values.append(datetime.now())
			conversation.updated_at = datetime.now()
			
			# Add WHERE clause parameters
			param_count += 1
			update_values.append(conversation_id)
			param_count += 1
			update_values.append(self.tenant_id)
			
			query = f"""
				UPDATE apg_rag_conversations 
				SET {', '.join(update_fields)}
				WHERE id = ${param_count-1} AND tenant_id = ${param_count}
			"""
			
			async with self.db_pool.acquire() as conn:
				await conn.execute(query, *update_values)
			
			# Update cache
			self.active_conversations[conversation_id] = conversation
		
		return conversation
	
	async def delete_conversation(self, conversation_id: str) -> bool:
		"""Delete conversation and all its turns"""
		
		try:
			async with self.db_pool.acquire() as conn:
				# Delete turns first (foreign key constraint)
				await conn.execute("""
					DELETE FROM apg_rag_conversation_turns 
					WHERE conversation_id = $1 AND tenant_id = $2
				""", conversation_id, self.tenant_id)
				
				# Delete conversation
				deleted_count = await conn.fetchval("""
					DELETE FROM apg_rag_conversations 
					WHERE id = $1 AND tenant_id = $2
					RETURNING 1
				""", conversation_id, self.tenant_id)
				
				if deleted_count:
					# Remove from cache
					self.active_conversations.pop(conversation_id, None)
					self.stats['active_conversations'] = max(0, self.stats['active_conversations'] - 1)
					return True
		
		except Exception as e:
			self.logger.error(f"Failed to delete conversation {conversation_id}: {str(e)}")
		
		return False
	
	async def list_conversations(self,
	                           user_id: Optional[str] = None,
	                           session_id: Optional[str] = None,
	                           status: Optional[ConversationStatus] = None,
	                           limit: int = 50,
	                           offset: int = 0) -> List[Conversation]:
		"""List conversations with optional filters"""
		
		# Build query with filters
		where_conditions = ["tenant_id = $1"]
		params = [self.tenant_id]
		param_count = 1
		
		if user_id:
			param_count += 1
			where_conditions.append(f"user_id = ${param_count}")
			params.append(user_id)
		
		if session_id:
			param_count += 1
			where_conditions.append(f"session_id = ${param_count}")
			params.append(session_id)
		
		if status:
			param_count += 1
			where_conditions.append(f"status = ${param_count}")
			params.append(status.value)
		
		where_clause = " AND ".join(where_conditions)
		
		param_count += 1
		params.append(limit)
		param_count += 1
		params.append(offset)
		
		query = f"""
			SELECT * FROM apg_rag_conversations 
			WHERE {where_clause}
			ORDER BY updated_at DESC
			LIMIT ${param_count-1} OFFSET ${param_count}
		"""
		
		async with self.db_pool.acquire() as conn:
			rows = await conn.fetch(query, *params)
			
			conversations = []
			for row in rows:
				conversation = Conversation(**dict(row))
				conversations.append(conversation)
			
			return conversations
	
	def _update_stats(self, processing_time_ms: float, had_retrieval: bool) -> None:
		"""Update conversation statistics"""
		self.stats['total_turns'] += 2  # User + assistant
		
		# Update average turns per conversation
		if self.stats['total_conversations'] > 0:
			self.stats['average_turns_per_conversation'] = self.stats['total_turns'] / self.stats['total_conversations']
		
		# Update average response time
		current_avg = self.stats['average_response_time_ms']
		total_conversations = self.stats['total_conversations']
		if total_conversations > 0:
			self.stats['average_response_time_ms'] = (
				(current_avg * (total_conversations - 1) + processing_time_ms) / total_conversations
			)
		
		# Update retrieval rate
		if had_retrieval:
			current_rate = self.stats['retrieval_rate']
			self.stats['retrieval_rate'] = (current_rate * (total_conversations - 1) + 1) / total_conversations
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive conversation statistics"""
		return {
			**self.stats,
			'active_conversations_count': len(self.active_conversations),
			'cached_conversations': len(self.active_conversations)
		}
	
	async def health_check(self) -> Dict[str, Any]:
		"""Comprehensive health check"""
		health_info = {
			'conversation_manager_healthy': True,
			'database_connection': False,
			'active_conversations': len(self.active_conversations),
			'timestamp': datetime.now().isoformat()
		}
		
		try:
			# Test database connection
			async with self.db_pool.acquire() as conn:
				await conn.fetchval("SELECT 1")
				health_info['database_connection'] = True
		
		except Exception as e:
			health_info['conversation_manager_healthy'] = False
			health_info['error'] = str(e)
		
		return health_info
	
	async def cleanup_inactive_conversations(self, inactive_hours: int = 24) -> int:
		"""Clean up inactive conversations from cache"""
		cutoff_time = datetime.now() - timedelta(hours=inactive_hours)
		
		inactive_ids = []
		for conv_id, conversation in self.active_conversations.items():
			if conversation.updated_at < cutoff_time:
				inactive_ids.append(conv_id)
		
		for conv_id in inactive_ids:
			del self.active_conversations[conv_id]
		
		self.logger.info(f"Cleaned up {len(inactive_ids)} inactive conversations from cache")
		return len(inactive_ids)

# Factory function for APG integration
async def create_conversation_manager(
	tenant_id: str,
	capability_id: str,
	db_pool: Pool,
	retrieval_engine: IntelligentRetrievalEngine,
	generation_engine: RAGGenerationEngine,
	config: ConversationConfig = None
) -> ConversationManager:
	"""Create conversation manager"""
	if config is None:
		config = ConversationConfig()
	
	manager = ConversationManager(
		config, db_pool, retrieval_engine, generation_engine, tenant_id, capability_id
	)
	
	return manager