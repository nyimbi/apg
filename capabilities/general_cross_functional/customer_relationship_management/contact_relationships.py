"""
APG Customer Relationship Management - Contact Relationships Module

Advanced contact relationship mapping and management system for tracking
connections, referrals, and business relationships between contacts.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ValidationError

from .models import CRMContact
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
	"""Types of relationships between contacts"""
	COLLEAGUE = "colleague"
	MANAGER = "manager"
	SUBORDINATE = "subordinate"
	PARTNER = "partner"
	VENDOR = "vendor"
	CUSTOMER = "customer"
	REFERRER = "referrer"
	REFERRED = "referred"
	FAMILY = "family"
	FRIEND = "friend"
	MENTOR = "mentor"
	MENTEE = "mentee"
	COMPETITOR = "competitor"
	COLLABORATOR = "collaborator"
	INFLUENCER = "influencer"
	DECISION_MAKER = "decision_maker"
	GATEKEEPER = "gatekeeper"
	CHAMPION = "champion"
	DETRACTOR = "detractor"
	NEUTRAL = "neutral"


class RelationshipStrength(str, Enum):
	"""Strength of the relationship"""
	WEAK = "weak"
	MODERATE = "moderate"
	STRONG = "strong"
	VERY_STRONG = "very_strong"


class RelationshipStatus(str, Enum):
	"""Status of the relationship"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	TERMINATED = "terminated"
	PENDING = "pending"


class ContactRelationship(BaseModel):
	"""Model for contact relationships"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	from_contact_id: str = Field(..., description="Source contact ID")
	to_contact_id: str = Field(..., description="Target contact ID")
	relationship_type: RelationshipType = Field(..., description="Type of relationship")
	relationship_strength: RelationshipStrength = Field(default=RelationshipStrength.MODERATE)
	relationship_status: RelationshipStatus = Field(default=RelationshipStatus.ACTIVE)
	is_mutual: bool = Field(default=False, description="Whether relationship is mutual")
	confidence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in relationship accuracy")
	notes: Optional[str] = Field(None, max_length=2000)
	tags: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Source information
	source: Optional[str] = Field(None, description="How relationship was discovered")
	source_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
	verified_at: Optional[datetime] = Field(None)
	verified_by: Optional[str] = Field(None)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(..., description="Creator user ID")
	updated_by: str = Field(..., description="Last updater user ID")
	version: int = Field(default=1)


class RelationshipError(Exception):
	"""Base exception for relationship operations"""
	pass


class ContactRelationshipManager:
	"""
	Advanced contact relationship management system with relationship
	discovery, mapping, and analytics capabilities.
	"""
	
	def __init__(self, database_manager: DatabaseManager):
		"""
		Initialize relationship manager
		
		Args:
			database_manager: Database manager instance
		"""
		self.db_manager = database_manager
		self.relationship_rules = self._load_relationship_rules()
	
	def _load_relationship_rules(self) -> Dict[str, Any]:
		"""Load relationship inference rules"""
		return {
			"email_domain_colleagues": {
				"enabled": True,
				"confidence": 0.7,
				"relationship_type": RelationshipType.COLLEAGUE
			},
			"company_name_colleagues": {
				"enabled": True,
				"confidence": 0.6,
				"relationship_type": RelationshipType.COLLEAGUE
			},
			"referral_tracking": {
				"enabled": True,
				"confidence": 0.9,
				"relationship_type": RelationshipType.REFERRER
			}
		}
	
	# ================================
	# Core Relationship Operations
	# ================================
	
	async def create_relationship(
		self,
		relationship_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> ContactRelationship:
		"""
		Create a new contact relationship
		
		Args:
			relationship_data: Relationship information
			tenant_id: Tenant identifier
			created_by: User creating the relationship
			
		Returns:
			Created relationship object
		"""
		try:
			# Add audit fields
			relationship_data.update({
				"tenant_id": tenant_id,
				"created_by": created_by,
				"updated_by": created_by
			})
			
			# Validate relationship
			relationship = ContactRelationship(**relationship_data)
			
			# Check for self-relationship
			if relationship.from_contact_id == relationship.to_contact_id:
				raise RelationshipError("Cannot create relationship with self")
			
			# Verify contacts exist
			from_contact = await self.db_manager.get_contact(relationship.from_contact_id, tenant_id)
			to_contact = await self.db_manager.get_contact(relationship.to_contact_id, tenant_id)
			
			if not from_contact or not to_contact:
				raise RelationshipError("One or both contacts not found")
			
			# Check for duplicate relationship
			existing = await self.get_relationship(
				relationship.from_contact_id,
				relationship.to_contact_id,
				tenant_id
			)
			
			if existing:
				raise RelationshipError("Relationship already exists")
			
			# Save to database
			saved_relationship = await self._save_relationship(relationship)
			
			# Create mutual relationship if specified
			if relationship.is_mutual:
				await self._create_mutual_relationship(relationship, created_by)
			
			logger.info(f"Created relationship: {saved_relationship.id}")
			return saved_relationship
			
		except ValidationError as e:
			raise RelationshipError(f"Invalid relationship data: {str(e)}")
		except Exception as e:
			logger.error(f"Failed to create relationship: {str(e)}", exc_info=True)
			raise RelationshipError(f"Relationship creation failed: {str(e)}")
	
	async def get_relationship(
		self,
		from_contact_id: str,
		to_contact_id: str,
		tenant_id: str
	) -> Optional[ContactRelationship]:
		"""
		Get relationship between two contacts
		
		Args:
			from_contact_id: Source contact ID
			to_contact_id: Target contact ID
			tenant_id: Tenant identifier
			
		Returns:
			Relationship object or None
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_contact_relationships 
					WHERE tenant_id = $1 
					AND from_contact_id = $2 
					AND to_contact_id = $3
				""", tenant_id, from_contact_id, to_contact_id)
				
				if row:
					return self._row_to_relationship(row)
				return None
				
		except Exception as e:
			logger.error(f"Failed to get relationship: {str(e)}", exc_info=True)
			raise RelationshipError(f"Get relationship failed: {str(e)}")
	
	async def get_contact_relationships(
		self,
		contact_id: str,
		tenant_id: str,
		relationship_type: Optional[RelationshipType] = None,
		include_incoming: bool = True,
		include_outgoing: bool = True
	) -> Dict[str, List[Dict[str, Any]]]:
		"""
		Get all relationships for a contact
		
		Args:
			contact_id: Contact identifier
			tenant_id: Tenant identifier
			relationship_type: Filter by relationship type
			include_incoming: Include relationships to this contact
			include_outgoing: Include relationships from this contact
			
		Returns:
			Dictionary with incoming and outgoing relationships
		"""
		try:
			relationships = {
				"incoming": [],
				"outgoing": [],
				"total": 0
			}
			
			async with self.db_manager.get_connection() as conn:
				# Get outgoing relationships
				if include_outgoing:
					query = """
						SELECT r.*, 
							   c.first_name as to_first_name,
							   c.last_name as to_last_name,
							   c.email as to_email,
							   c.company as to_company
						FROM crm_contact_relationships r
						JOIN crm_contacts c ON r.to_contact_id = c.id
						WHERE r.tenant_id = $1 AND r.from_contact_id = $2
					"""
					params = [tenant_id, contact_id]
					
					if relationship_type:
						query += " AND r.relationship_type = $3"
						params.append(relationship_type.value)
					
					query += " ORDER BY r.created_at DESC"
					
					rows = await conn.fetch(query, *params)
					
					for row in rows:
						rel_data = self._row_to_relationship(row).model_dump()
						rel_data['related_contact'] = {
							"id": row['to_contact_id'],
							"first_name": row['to_first_name'],
							"last_name": row['to_last_name'],
							"email": row['to_email'],
							"company": row['to_company']
						}
						relationships["outgoing"].append(rel_data)
				
				# Get incoming relationships
				if include_incoming:
					query = """
						SELECT r.*, 
							   c.first_name as from_first_name,
							   c.last_name as from_last_name,
							   c.email as from_email,
							   c.company as from_company
						FROM crm_contact_relationships r
						JOIN crm_contacts c ON r.from_contact_id = c.id
						WHERE r.tenant_id = $1 AND r.to_contact_id = $2
					"""
					params = [tenant_id, contact_id]
					
					if relationship_type:
						query += " AND r.relationship_type = $3"
						params.append(relationship_type.value)
					
					query += " ORDER BY r.created_at DESC"
					
					rows = await conn.fetch(query, *params)
					
					for row in rows:
						rel_data = self._row_to_relationship(row).model_dump()
						rel_data['related_contact'] = {
							"id": row['from_contact_id'],
							"first_name": row['from_first_name'],
							"last_name": row['from_last_name'],
							"email": row['from_email'],
							"company": row['from_company']
						}
						relationships["incoming"].append(rel_data)
			
			relationships["total"] = len(relationships["incoming"]) + len(relationships["outgoing"])
			
			return relationships
			
		except Exception as e:
			logger.error(f"Failed to get contact relationships: {str(e)}", exc_info=True)
			raise RelationshipError(f"Get contact relationships failed: {str(e)}")
	
	async def update_relationship(
		self,
		relationship_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> ContactRelationship:
		"""
		Update existing relationship
		
		Args:
			relationship_id: Relationship identifier
			update_data: Fields to update
			tenant_id: Tenant identifier
			updated_by: User updating the relationship
			
		Returns:
			Updated relationship object
		"""
		try:
			# Get existing relationship
			existing = await self._get_relationship_by_id(relationship_id, tenant_id)
			if not existing:
				raise RelationshipError(f"Relationship not found: {relationship_id}")
			
			# Update fields
			update_data.update({
				"updated_by": updated_by,
				"updated_at": datetime.utcnow(),
				"version": existing.version + 1
			})
			
			# Apply updates
			for field, value in update_data.items():
				if hasattr(existing, field):
					setattr(existing, field, value)
			
			# Save updated relationship
			updated_relationship = await self._save_relationship(existing)
			
			logger.info(f"Updated relationship: {relationship_id}")
			return updated_relationship
			
		except Exception as e:
			logger.error(f"Failed to update relationship: {str(e)}", exc_info=True)
			raise RelationshipError(f"Relationship update failed: {str(e)}")
	
	async def delete_relationship(
		self,
		relationship_id: str,
		tenant_id: str
	) -> bool:
		"""
		Delete a relationship
		
		Args:
			relationship_id: Relationship identifier
			tenant_id: Tenant identifier
			
		Returns:
			True if deleted successfully
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				result = await conn.execute("""
					DELETE FROM crm_contact_relationships 
					WHERE id = $1 AND tenant_id = $2
				""", relationship_id, tenant_id)
				
				deleted = result.split()[-1] == '1'
				
				if deleted:
					logger.info(f"Deleted relationship: {relationship_id}")
				
				return deleted
				
		except Exception as e:
			logger.error(f"Failed to delete relationship: {str(e)}", exc_info=True)
			raise RelationshipError(f"Relationship deletion failed: {str(e)}")
	
	# ================================
	# Relationship Discovery
	# ================================
	
	async def discover_relationships(
		self,
		tenant_id: str,
		auto_create: bool = False,
		min_confidence: float = 0.6
	) -> List[Dict[str, Any]]:
		"""
		Discover potential relationships between contacts
		
		Args:
			tenant_id: Tenant identifier
			auto_create: Whether to automatically create discovered relationships
			min_confidence: Minimum confidence threshold
			
		Returns:
			List of discovered potential relationships
		"""
		try:
			logger.info(f"🔍 Starting relationship discovery for tenant: {tenant_id}")
			
			discovered_relationships = []
			
			# Get all contacts for analysis
			contacts_result = await self.db_manager.list_contacts(tenant_id, limit=5000)
			contacts = contacts_result.get('items', [])
			
			if len(contacts) < 2:
				return discovered_relationships
			
			# Discover email domain colleagues
			email_relationships = await self._discover_email_domain_colleagues(contacts, tenant_id)
			discovered_relationships.extend(email_relationships)
			
			# Discover company colleagues
			company_relationships = await self._discover_company_colleagues(contacts, tenant_id)
			discovered_relationships.extend(company_relationships)
			
			# Filter by confidence threshold
			high_confidence_relationships = [
				rel for rel in discovered_relationships 
				if rel.get('confidence_score', 0) >= min_confidence
			]
			
			# Auto-create if requested
			if auto_create:
				created_count = 0
				for rel_data in high_confidence_relationships:
					try:
						await self.create_relationship(rel_data, tenant_id, "system")
						created_count += 1
					except RelationshipError:
						# Skip if relationship already exists or other error
						pass
				
				logger.info(f"Auto-created {created_count} relationships")
			
			logger.info(f"✅ Discovered {len(discovered_relationships)} potential relationships")
			return high_confidence_relationships
			
		except Exception as e:
			logger.error(f"Relationship discovery failed: {str(e)}", exc_info=True)
			raise RelationshipError(f"Relationship discovery failed: {str(e)}")
	
	async def _discover_email_domain_colleagues(
		self, 
		contacts: List[CRMContact], 
		tenant_id: str
	) -> List[Dict[str, Any]]:
		"""Discover colleagues based on email domains"""
		relationships = []
		
		# Group contacts by email domain
		domain_groups = {}
		for contact in contacts:
			if contact.email and '@' in contact.email:
				domain = contact.email.split('@')[1].lower()
				if domain not in domain_groups:
					domain_groups[domain] = []
				domain_groups[domain].append(contact)
		
		# Find colleagues within same domain
		for domain, domain_contacts in domain_groups.items():
			if len(domain_contacts) > 1:
				# Skip common email providers
				if domain in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']:
					continue
				
				# Create colleague relationships between all contacts in domain
				for i, contact1 in enumerate(domain_contacts):
					for contact2 in domain_contacts[i+1:]:
						# Check if relationship already exists
						existing = await self.get_relationship(contact1.id, contact2.id, tenant_id)
						if not existing:
							relationships.append({
								"from_contact_id": contact1.id,
								"to_contact_id": contact2.id,
								"relationship_type": RelationshipType.COLLEAGUE,
								"relationship_strength": RelationshipStrength.MODERATE,
								"confidence_score": 0.7,
								"source": "email_domain_analysis",
								"source_confidence": 0.7,
								"is_mutual": True,
								"notes": f"Discovered through shared email domain: {domain}"
							})
		
		return relationships
	
	async def _discover_company_colleagues(
		self, 
		contacts: List[CRMContact], 
		tenant_id: str
	) -> List[Dict[str, Any]]:
		"""Discover colleagues based on company names"""
		relationships = []
		
		# Group contacts by company
		company_groups = {}
		for contact in contacts:
			if contact.company and contact.company.strip():
				company = contact.company.strip().lower()
				if company not in company_groups:
					company_groups[company] = []
				company_groups[company].append(contact)
		
		# Find colleagues within same company
		for company, company_contacts in company_groups.items():
			if len(company_contacts) > 1:
				# Create colleague relationships
				for i, contact1 in enumerate(company_contacts):
					for contact2 in company_contacts[i+1:]:
						# Check if relationship already exists
						existing = await self.get_relationship(contact1.id, contact2.id, tenant_id)
						if not existing:
							relationships.append({
								"from_contact_id": contact1.id,
								"to_contact_id": contact2.id,
								"relationship_type": RelationshipType.COLLEAGUE,
								"relationship_strength": RelationshipStrength.MODERATE,
								"confidence_score": 0.6,
								"source": "company_name_analysis",
								"source_confidence": 0.6,
								"is_mutual": True,
								"notes": f"Discovered through shared company: {company}"
							})
		
		return relationships
	
	# ================================
	# Relationship Analytics
	# ================================
	
	async def get_relationship_analytics(
		self,
		tenant_id: str,
		contact_id: Optional[str] = None
	) -> Dict[str, Any]:
		"""
		Get relationship analytics
		
		Args:
			tenant_id: Tenant identifier
			contact_id: Specific contact ID for individual analytics
			
		Returns:
			Analytics data
		"""
		try:
			analytics = {
				"total_relationships": 0,
				"relationship_types": {},
				"relationship_strengths": {},
				"top_connected_contacts": [],
				"relationship_growth": []
			}
			
			async with self.db_manager.get_connection() as conn:
				# Get total relationships
				total_count = await conn.fetchval("""
					SELECT COUNT(*) FROM crm_contact_relationships 
					WHERE tenant_id = $1
				""", tenant_id)
				analytics["total_relationships"] = total_count
				
				# Get relationship type breakdown
				type_breakdown = await conn.fetch("""
					SELECT relationship_type, COUNT(*) as count
					FROM crm_contact_relationships 
					WHERE tenant_id = $1
					GROUP BY relationship_type
					ORDER BY count DESC
				""", tenant_id)
				
				analytics["relationship_types"] = {
					row['relationship_type']: row['count'] 
					for row in type_breakdown
				}
				
				# Get relationship strength breakdown
				strength_breakdown = await conn.fetch("""
					SELECT relationship_strength, COUNT(*) as count
					FROM crm_contact_relationships 
					WHERE tenant_id = $1
					GROUP BY relationship_strength
					ORDER BY count DESC
				""", tenant_id)
				
				analytics["relationship_strengths"] = {
					row['relationship_strength']: row['count'] 
					for row in strength_breakdown
				}
				
				# Get most connected contacts
				top_connected = await conn.fetch("""
					SELECT 
						c.id, c.first_name, c.last_name, c.email, c.company,
						(SELECT COUNT(*) FROM crm_contact_relationships r1 
						 WHERE r1.from_contact_id = c.id AND r1.tenant_id = $1) +
						(SELECT COUNT(*) FROM crm_contact_relationships r2 
						 WHERE r2.to_contact_id = c.id AND r2.tenant_id = $1) as connection_count
					FROM crm_contacts c
					WHERE c.tenant_id = $1
					ORDER BY connection_count DESC
					LIMIT 10
				""", tenant_id)
				
				analytics["top_connected_contacts"] = [
					{
						"contact_id": row['id'],
						"name": f"{row['first_name']} {row['last_name']}",
						"email": row['email'],
						"company": row['company'],
						"connection_count": row['connection_count']
					}
					for row in top_connected if row['connection_count'] > 0
				]
			
			return analytics
			
		except Exception as e:
			logger.error(f"Relationship analytics failed: {str(e)}", exc_info=True)
			raise RelationshipError(f"Relationship analytics failed: {str(e)}")
	
	# ================================
	# Relationship Graph Operations
	# ================================
	
	async def get_relationship_graph(
		self,
		contact_id: str,
		tenant_id: str,
		depth: int = 2,
		max_nodes: int = 50
	) -> Dict[str, Any]:
		"""
		Get relationship graph for a contact
		
		Args:
			contact_id: Center contact ID
			tenant_id: Tenant identifier
			depth: Maximum relationship depth
			max_nodes: Maximum number of nodes to return
			
		Returns:
			Graph data with nodes and edges
		"""
		try:
			nodes = {}
			edges = []
			visited = set()
			
			# Start with center contact
			center_contact = await self.db_manager.get_contact(contact_id, tenant_id)
			if not center_contact:
				raise RelationshipError(f"Contact not found: {contact_id}")
			
			nodes[contact_id] = {
				"id": contact_id,
				"name": f"{center_contact.first_name} {center_contact.last_name}",
				"email": center_contact.email,
				"company": center_contact.company,
				"type": "center",
				"depth": 0
			}
			
			# Build graph using BFS
			queue = [(contact_id, 0)]
			visited.add(contact_id)
			
			while queue and len(nodes) < max_nodes:
				current_id, current_depth = queue.pop(0)
				
				if current_depth >= depth:
					continue
				
				# Get relationships for current contact
				relationships = await self.get_contact_relationships(
					current_id, tenant_id, include_incoming=True, include_outgoing=True
				)
				
				# Process outgoing relationships
				for rel in relationships["outgoing"]:
					related_contact = rel["related_contact"]
					related_id = related_contact["id"]
					
					# Add node if not already added
					if related_id not in nodes:
						nodes[related_id] = {
							"id": related_id,
							"name": f"{related_contact['first_name']} {related_contact['last_name']}",
							"email": related_contact["email"],
							"company": related_contact["company"],
							"type": "connected",
							"depth": current_depth + 1
						}
					
					# Add edge
					edges.append({
						"id": rel["id"],
						"from": current_id,
						"to": related_id,
						"relationship_type": rel["relationship_type"],
						"relationship_strength": rel["relationship_strength"],
						"confidence_score": rel["confidence_score"]
					})
					
					# Add to queue for further exploration
					if related_id not in visited and current_depth + 1 < depth:
						queue.append((related_id, current_depth + 1))
						visited.add(related_id)
				
				# Process incoming relationships
				for rel in relationships["incoming"]:
					related_contact = rel["related_contact"]
					related_id = related_contact["id"]
					
					# Add node if not already added
					if related_id not in nodes:
						nodes[related_id] = {
							"id": related_id,
							"name": f"{related_contact['first_name']} {related_contact['last_name']}",
							"email": related_contact["email"],
							"company": related_contact["company"],
							"type": "connected",
							"depth": current_depth + 1
						}
					
					# Add edge
					edges.append({
						"id": rel["id"],
						"from": related_id,
						"to": current_id,
						"relationship_type": rel["relationship_type"],
						"relationship_strength": rel["relationship_strength"],
						"confidence_score": rel["confidence_score"]
					})
					
					# Add to queue for further exploration
					if related_id not in visited and current_depth + 1 < depth:
						queue.append((related_id, current_depth + 1))
						visited.add(related_id)
			
			return {
				"center_contact_id": contact_id,
				"nodes": list(nodes.values()),
				"edges": edges,
				"total_nodes": len(nodes),
				"total_edges": len(edges),
				"max_depth": depth
			}
			
		except Exception as e:
			logger.error(f"Relationship graph failed: {str(e)}", exc_info=True)
			raise RelationshipError(f"Relationship graph failed: {str(e)}")
	
	# ================================
	# Helper Methods
	# ================================
	
	async def _save_relationship(self, relationship: ContactRelationship) -> ContactRelationship:
		"""Save relationship to database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_contact_relationships (
						id, tenant_id, from_contact_id, to_contact_id,
						relationship_type, relationship_strength, relationship_status,
						is_mutual, confidence_score, notes, tags, metadata,
						source, source_confidence, verified_at, verified_by,
						created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
						$13, $14, $15, $16, $17, $18, $19, $20, $21
					)
					ON CONFLICT (id) DO UPDATE SET
						relationship_type = EXCLUDED.relationship_type,
						relationship_strength = EXCLUDED.relationship_strength,
						relationship_status = EXCLUDED.relationship_status,
						is_mutual = EXCLUDED.is_mutual,
						confidence_score = EXCLUDED.confidence_score,
						notes = EXCLUDED.notes,
						tags = EXCLUDED.tags,
						metadata = EXCLUDED.metadata,
						source = EXCLUDED.source,
						source_confidence = EXCLUDED.source_confidence,
						verified_at = EXCLUDED.verified_at,
						verified_by = EXCLUDED.verified_by,
						updated_at = EXCLUDED.updated_at,
						updated_by = EXCLUDED.updated_by,
						version = EXCLUDED.version
				""",
					relationship.id, relationship.tenant_id, relationship.from_contact_id,
					relationship.to_contact_id, relationship.relationship_type.value,
					relationship.relationship_strength.value, relationship.relationship_status.value,
					relationship.is_mutual, relationship.confidence_score, relationship.notes,
					relationship.tags, relationship.metadata, relationship.source,
					relationship.source_confidence, relationship.verified_at, relationship.verified_by,
					relationship.created_at, relationship.updated_at, relationship.created_by,
					relationship.updated_by, relationship.version
				)
			
			return relationship
			
		except Exception as e:
			logger.error(f"Failed to save relationship: {str(e)}", exc_info=True)
			raise RelationshipError(f"Save relationship failed: {str(e)}")
	
	async def _create_mutual_relationship(self, relationship: ContactRelationship, created_by: str):
		"""Create mutual relationship"""
		try:
			# Determine reverse relationship type
			reverse_type = self._get_reverse_relationship_type(relationship.relationship_type)
			
			mutual_relationship = ContactRelationship(
				tenant_id=relationship.tenant_id,
				from_contact_id=relationship.to_contact_id,
				to_contact_id=relationship.from_contact_id,
				relationship_type=reverse_type,
				relationship_strength=relationship.relationship_strength,
				relationship_status=relationship.relationship_status,
				is_mutual=True,
				confidence_score=relationship.confidence_score,
				source=relationship.source,
				source_confidence=relationship.source_confidence,
				notes=f"Mutual relationship for {relationship.id}",
				created_by=created_by,
				updated_by=created_by
			)
			
			await self._save_relationship(mutual_relationship)
			
		except Exception as e:
			logger.error(f"Failed to create mutual relationship: {str(e)}", exc_info=True)
			# Don't raise error for mutual relationship creation failure
	
	def _get_reverse_relationship_type(self, relationship_type: RelationshipType) -> RelationshipType:
		"""Get reverse relationship type for mutual relationships"""
		reverse_mappings = {
			RelationshipType.MANAGER: RelationshipType.SUBORDINATE,
			RelationshipType.SUBORDINATE: RelationshipType.MANAGER,
			RelationshipType.MENTOR: RelationshipType.MENTEE,
			RelationshipType.MENTEE: RelationshipType.MENTOR,
			RelationshipType.REFERRER: RelationshipType.REFERRED,
			RelationshipType.REFERRED: RelationshipType.REFERRER,
		}
		
		return reverse_mappings.get(relationship_type, relationship_type)
	
	async def _get_relationship_by_id(self, relationship_id: str, tenant_id: str) -> Optional[ContactRelationship]:
		"""Get relationship by ID"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_contact_relationships 
					WHERE id = $1 AND tenant_id = $2
				""", relationship_id, tenant_id)
				
				if row:
					return self._row_to_relationship(row)
				return None
				
		except Exception as e:
			logger.error(f"Failed to get relationship by ID: {str(e)}", exc_info=True)
			raise RelationshipError(f"Get relationship by ID failed: {str(e)}")
	
	def _row_to_relationship(self, row) -> ContactRelationship:
		"""Convert database row to ContactRelationship object"""
		return ContactRelationship(
			id=row['id'],
			tenant_id=row['tenant_id'],
			from_contact_id=row['from_contact_id'],
			to_contact_id=row['to_contact_id'],
			relationship_type=RelationshipType(row['relationship_type']),
			relationship_strength=RelationshipStrength(row['relationship_strength']),
			relationship_status=RelationshipStatus(row['relationship_status']),
			is_mutual=row['is_mutual'],
			confidence_score=row['confidence_score'],
			notes=row['notes'],
			tags=row['tags'] or [],
			metadata=row['metadata'] or {},
			source=row['source'],
			source_confidence=row['source_confidence'],
			verified_at=row['verified_at'],
			verified_by=row['verified_by'],
			created_at=row['created_at'],
			updated_at=row['updated_at'],
			created_by=row['created_by'],
			updated_by=row['updated_by'],
			version=row['version']
		)