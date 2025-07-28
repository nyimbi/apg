"""
APG Customer Relationship Management - Account Relationships Module

Advanced account relationship management system for complex business relationships,
partnerships, vendor connections, and multi-dimensional account interactions.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from enum import Enum
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ValidationError

from .models import CRMAccount, AccountType
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class AccountRelationshipType(str, Enum):
	"""Types of relationships between accounts"""
	CUSTOMER = "customer"
	VENDOR = "vendor"
	PARTNER = "partner"
	COMPETITOR = "competitor"
	SUBSIDIARY = "subsidiary"
	PARENT_COMPANY = "parent_company"
	JOINT_VENTURE = "joint_venture"
	STRATEGIC_ALLIANCE = "strategic_alliance"
	RESELLER = "reseller"
	DISTRIBUTOR = "distributor"
	SUPPLIER = "supplier"
	SERVICE_PROVIDER = "service_provider"
	INTEGRATION_PARTNER = "integration_partner"
	REFERRAL_SOURCE = "referral_source"
	ACQUISITION_TARGET = "acquisition_target"
	INVESTOR = "investor"
	BOARD_MEMBER = "board_member"
	CONSULTANT = "consultant"
	LEGAL_COUNSEL = "legal_counsel"
	OTHER = "other"


class RelationshipStrength(str, Enum):
	"""Strength of business relationship"""
	WEAK = "weak"
	MODERATE = "moderate"
	STRONG = "strong"
	STRATEGIC = "strategic"


class RelationshipStatus(str, Enum):
	"""Status of the relationship"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	PENDING = "pending"
	TERMINATED = "terminated"
	SUSPENDED = "suspended"


class RelationshipDirection(str, Enum):
	"""Direction of the relationship"""
	OUTBOUND = "outbound"  # From account A to account B
	INBOUND = "inbound"    # From account B to account A
	BIDIRECTIONAL = "bidirectional"  # Mutual relationship


class AccountRelationship(BaseModel):
	"""Model for account relationships"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	from_account_id: str = Field(..., description="Source account ID")
	to_account_id: str = Field(..., description="Target account ID")
	relationship_type: AccountRelationshipType = Field(..., description="Type of relationship")
	relationship_strength: RelationshipStrength = Field(default=RelationshipStrength.MODERATE)
	relationship_status: RelationshipStatus = Field(default=RelationshipStatus.ACTIVE)
	direction: RelationshipDirection = Field(default=RelationshipDirection.OUTBOUND)
	
	# Financial aspects
	annual_value: Optional[float] = Field(None, description="Annual relationship value")
	contract_start_date: Optional[datetime] = Field(None, description="Contract start date")
	contract_end_date: Optional[datetime] = Field(None, description="Contract end date")
	renewal_date: Optional[datetime] = Field(None, description="Next renewal date")
	
	# Relationship details
	key_contact_id: Optional[str] = Field(None, description="Key contact for this relationship")
	relationship_owner_id: str = Field(..., description="Owner of this relationship")
	
	# Risk and compliance
	risk_level: Optional[str] = Field(None, description="Risk assessment level")
	compliance_status: Optional[str] = Field(None, description="Compliance status")
	
	# Metadata
	description: Optional[str] = Field(None, max_length=2000, description="Relationship description")
	notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")
	tags: List[str] = Field(default_factory=list, description="Relationship tags")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	
	# Source tracking
	source: Optional[str] = Field(None, description="How relationship was discovered")
	verified_at: Optional[datetime] = Field(None, description="When relationship was verified")
	verified_by: Optional[str] = Field(None, description="Who verified the relationship")
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(..., description="Creator user ID")
	updated_by: str = Field(..., description="Last updater user ID")
	version: int = Field(default=1)


class RelationshipAnalytics(BaseModel):
	"""Analytics model for relationship insights"""
	total_relationships: int = 0
	relationship_types: Dict[str, int] = Field(default_factory=dict)
	relationship_strengths: Dict[str, int] = Field(default_factory=dict)
	total_annual_value: float = 0.0
	top_valued_relationships: List[Dict[str, Any]] = Field(default_factory=list)
	expiring_contracts: List[Dict[str, Any]] = Field(default_factory=list)
	relationship_growth: List[Dict[str, Any]] = Field(default_factory=list)


class RelationshipError(Exception):
	"""Base exception for relationship operations"""
	pass


class AccountRelationshipManager:
	"""
	Advanced account relationship management system for complex business
	relationships, partnerships, and multi-dimensional account interactions.
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
		"""Load relationship business rules and validations"""
		return {
			"mutual_relationships": {
				AccountRelationshipType.PARTNER: True,
				AccountRelationshipType.JOINT_VENTURE: True,
				AccountRelationshipType.STRATEGIC_ALLIANCE: True,
				AccountRelationshipType.COMPETITOR: True,
			},
			"incompatible_relationships": [
				(AccountRelationshipType.CUSTOMER, AccountRelationshipType.COMPETITOR),
				(AccountRelationshipType.VENDOR, AccountRelationshipType.COMPETITOR),
			],
			"auto_create_reverse": {
				AccountRelationshipType.CUSTOMER: AccountRelationshipType.VENDOR,
				AccountRelationshipType.VENDOR: AccountRelationshipType.CUSTOMER,
				AccountRelationshipType.PARENT_COMPANY: AccountRelationshipType.SUBSIDIARY,
				AccountRelationshipType.SUBSIDIARY: AccountRelationshipType.PARENT_COMPANY,
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
	) -> AccountRelationship:
		"""
		Create a new account relationship
		
		Args:
			relationship_data: Relationship information
			tenant_id: Tenant identifier
			created_by: User creating the relationship
			
		Returns:
			Created relationship object
		"""
		try:
			logger.info(f"🔗 Creating account relationship")
			
			# Add audit fields
			relationship_data.update({
				"tenant_id": tenant_id,
				"created_by": created_by,
				"updated_by": created_by
			})
			
			# Validate relationship
			relationship = AccountRelationship(**relationship_data)
			
			# Business rule validations
			await self._validate_relationship(relationship, tenant_id)
			
			# Save to database
			saved_relationship = await self._save_relationship(relationship)
			
			# Create reverse relationship if applicable
			if self._should_create_reverse_relationship(relationship):
				await self._create_reverse_relationship(relationship, created_by)
			
			# Update relationship analytics
			asyncio.create_task(self._update_relationship_analytics(tenant_id))
			
			logger.info(f"✅ Created relationship: {saved_relationship.id}")
			return saved_relationship
			
		except ValidationError as e:
			raise RelationshipError(f"Invalid relationship data: {str(e)}")
		except Exception as e:
			logger.error(f"Failed to create relationship: {str(e)}", exc_info=True)
			raise RelationshipError(f"Relationship creation failed: {str(e)}")
	
	async def get_relationship(
		self,
		relationship_id: str,
		tenant_id: str
	) -> Optional[AccountRelationship]:
		"""
		Get relationship by ID
		
		Args:
			relationship_id: Relationship identifier
			tenant_id: Tenant identifier
			
		Returns:
			Relationship object or None
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_account_relationships 
					WHERE id = $1 AND tenant_id = $2
				""", relationship_id, tenant_id)
				
				if row:
					return self._row_to_relationship(row)
				return None
				
		except Exception as e:
			logger.error(f"Failed to get relationship: {str(e)}", exc_info=True)
			raise RelationshipError(f"Get relationship failed: {str(e)}")
	
	async def get_account_relationships(
		self,
		account_id: str,
		tenant_id: str,
		relationship_type: Optional[AccountRelationshipType] = None,
		direction: Optional[RelationshipDirection] = None,
		status: Optional[RelationshipStatus] = None,
		include_details: bool = True
	) -> Dict[str, Any]:
		"""
		Get all relationships for an account
		
		Args:
			account_id: Account identifier
			tenant_id: Tenant identifier
			relationship_type: Filter by relationship type
			direction: Filter by direction
			status: Filter by status
			include_details: Include detailed account information
			
		Returns:
			Comprehensive relationship data
		"""
		try:
			logger.info(f"🔍 Getting relationships for account: {account_id}")
			
			relationships = {
				"outbound": [],
				"inbound": [],
				"bidirectional": [],
				"total": 0,
				"summary": {}
			}
			
			async with self.db_manager.get_connection() as conn:
				# Build base query
				base_query = """
					SELECT r.*, 
						   a_from.account_name as from_account_name,
						   a_from.account_type as from_account_type,
						   a_to.account_name as to_account_name,
						   a_to.account_type as to_account_type
					FROM crm_account_relationships r
					JOIN crm_accounts a_from ON r.from_account_id = a_from.id
					JOIN crm_accounts a_to ON r.to_account_id = a_to.id
					WHERE r.tenant_id = $1 
					AND (r.from_account_id = $2 OR r.to_account_id = $2)
				"""
				
				params = [tenant_id, account_id]
				param_counter = 3
				
				# Add filters
				if relationship_type:
					base_query += f" AND r.relationship_type = ${param_counter}"
					params.append(relationship_type.value)
					param_counter += 1
				
				if direction:
					base_query += f" AND r.direction = ${param_counter}"
					params.append(direction.value)
					param_counter += 1
				
				if status:
					base_query += f" AND r.relationship_status = ${param_counter}"
					params.append(status.value)
					param_counter += 1
				
				base_query += " ORDER BY r.created_at DESC"
				
				# Execute query
				rows = await conn.fetch(base_query, *params)
				
				# Process results
				for row in rows:
					rel = self._row_to_relationship(row)
					rel_data = rel.model_dump()
					
					# Add related account information
					if row['from_account_id'] == account_id:
						# Outbound relationship
						rel_data['related_account'] = {
							"id": row['to_account_id'],
							"name": row['to_account_name'],
							"type": row['to_account_type']
						}
						rel_data['relationship_direction'] = "outbound"
						relationships["outbound"].append(rel_data)
					else:
						# Inbound relationship
						rel_data['related_account'] = {
							"id": row['from_account_id'],
							"name": row['from_account_name'],
							"type": row['from_account_type']
						}
						rel_data['relationship_direction'] = "inbound"
						relationships["inbound"].append(rel_data)
					
					if rel.direction == RelationshipDirection.BIDIRECTIONAL:
						relationships["bidirectional"].append(rel_data)
				
				# Calculate summary statistics
				relationships["total"] = len(rows)
				relationships["summary"] = await self._calculate_relationship_summary(rows)
			
			logger.info(f"✅ Found {relationships['total']} relationships")
			return relationships
			
		except Exception as e:
			logger.error(f"Failed to get account relationships: {str(e)}", exc_info=True)
			raise RelationshipError(f"Get account relationships failed: {str(e)}")
	
	async def update_relationship(
		self,
		relationship_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> AccountRelationship:
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
			existing = await self.get_relationship(relationship_id, tenant_id)
			if not existing:
				raise RelationshipError(f"Relationship not found: {relationship_id}")
			
			# Update fields
			update_data.update({
				"updated_by": updated_by,
				"updated_at": datetime.utcnow(),
				"version": existing.version + 1
			})
			
			# Apply updates
			existing_dict = existing.model_dump()
			existing_dict.update(update_data)
			
			# Validate updated relationship
			updated_relationship = AccountRelationship(**existing_dict)
			
			# Business rule validations
			await self._validate_relationship(updated_relationship, tenant_id)
			
			# Save updated relationship
			saved_relationship = await self._save_relationship(updated_relationship)
			
			logger.info(f"✅ Updated relationship: {relationship_id}")
			return saved_relationship
			
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
					DELETE FROM crm_account_relationships 
					WHERE id = $1 AND tenant_id = $2
				""", relationship_id, tenant_id)
				
				deleted = result.split()[-1] == '1'
				
				if deleted:
					logger.info(f"✅ Deleted relationship: {relationship_id}")
				
				return deleted
				
		except Exception as e:
			logger.error(f"Failed to delete relationship: {str(e)}", exc_info=True)
			raise RelationshipError(f"Relationship deletion failed: {str(e)}")
	
	# ================================
	# Relationship Discovery & Analytics
	# ================================
	
	async def discover_potential_relationships(
		self,
		tenant_id: str,
		account_id: Optional[str] = None,
		min_confidence: float = 0.7
	) -> List[Dict[str, Any]]:
		"""
		Discover potential relationships between accounts
		
		Args:
			tenant_id: Tenant identifier
			account_id: Specific account to analyze
			min_confidence: Minimum confidence threshold
			
		Returns:
			List of potential relationships with confidence scores
		"""
		try:
			logger.info(f"🔍 Discovering potential relationships for tenant: {tenant_id}")
			
			potential_relationships = []
			
			# Get accounts for analysis
			if account_id:
				accounts_result = await self.db_manager.get_account(account_id, tenant_id)
				accounts = [accounts_result] if accounts_result else []
			else:
				accounts_result = await self.db_manager.list_accounts(tenant_id, limit=1000)
				accounts = accounts_result.get('items', [])
			
			if len(accounts) < 2:
				return potential_relationships
			
			# Analyze relationships based on various signals
			domain_relationships = await self._discover_domain_relationships(accounts, tenant_id)
			potential_relationships.extend(domain_relationships)
			
			industry_relationships = await self._discover_industry_relationships(accounts, tenant_id)
			potential_relationships.extend(industry_relationships)
			
			contact_relationships = await self._discover_contact_based_relationships(accounts, tenant_id)
			potential_relationships.extend(contact_relationships)
			
			# Filter by confidence threshold
			high_confidence = [
				rel for rel in potential_relationships 
				if rel.get('confidence_score', 0) >= min_confidence
			]
			
			logger.info(f"✅ Discovered {len(high_confidence)} potential relationships")
			return high_confidence
			
		except Exception as e:
			logger.error(f"Relationship discovery failed: {str(e)}", exc_info=True)
			raise RelationshipError(f"Relationship discovery failed: {str(e)}")
	
	async def get_relationship_analytics(
		self,
		tenant_id: str,
		account_id: Optional[str] = None,
		start_date: Optional[datetime] = None,
		end_date: Optional[datetime] = None
	) -> RelationshipAnalytics:
		"""
		Get comprehensive relationship analytics
		
		Args:
			tenant_id: Tenant identifier
			account_id: Specific account to analyze
			start_date: Analysis start date
			end_date: Analysis end date
			
		Returns:
			Comprehensive relationship analytics
		"""
		try:
			logger.info(f"📊 Generating relationship analytics for tenant: {tenant_id}")
			
			if not start_date:
				start_date = datetime.utcnow() - timedelta(days=365)
			if not end_date:
				end_date = datetime.utcnow()
			
			analytics = RelationshipAnalytics()
			
			async with self.db_manager.get_connection() as conn:
				# Base conditions
				base_conditions = "tenant_id = $1 AND created_at BETWEEN $2 AND $3"
				base_params = [tenant_id, start_date, end_date]
				
				if account_id:
					base_conditions += " AND (from_account_id = $4 OR to_account_id = $4)"
					base_params.append(account_id)
				
				# Total relationships
				total_count = await conn.fetchval(f"""
					SELECT COUNT(*) FROM crm_account_relationships 
					WHERE {base_conditions}
				""", *base_params)
				analytics.total_relationships = total_count
				
				# Relationship type distribution
				type_breakdown = await conn.fetch(f"""
					SELECT relationship_type, COUNT(*) as count
					FROM crm_account_relationships 
					WHERE {base_conditions}
					GROUP BY relationship_type
					ORDER BY count DESC
				""", *base_params)
				
				analytics.relationship_types = {
					row['relationship_type']: row['count'] 
					for row in type_breakdown
				}
				
				# Relationship strength distribution
				strength_breakdown = await conn.fetch(f"""
					SELECT relationship_strength, COUNT(*) as count
					FROM crm_account_relationships 
					WHERE {base_conditions}
					GROUP BY relationship_strength
					ORDER BY count DESC
				""", *base_params)
				
				analytics.relationship_strengths = {
					row['relationship_strength']: row['count'] 
					for row in strength_breakdown
				}
				
				# Total annual value
				total_value = await conn.fetchval(f"""
					SELECT COALESCE(SUM(annual_value), 0) 
					FROM crm_account_relationships 
					WHERE {base_conditions} AND annual_value IS NOT NULL
				""", *base_params)
				analytics.total_annual_value = float(total_value)
				
				# Top valued relationships
				top_valued = await conn.fetch(f"""
					SELECT r.*, a_from.account_name as from_name, a_to.account_name as to_name
					FROM crm_account_relationships r
					JOIN crm_accounts a_from ON r.from_account_id = a_from.id
					JOIN crm_accounts a_to ON r.to_account_id = a_to.id
					WHERE {base_conditions} AND r.annual_value IS NOT NULL
					ORDER BY r.annual_value DESC
					LIMIT 10
				""", *base_params)
				
				analytics.top_valued_relationships = [
					{
						"relationship_id": row['id'],
						"from_account": row['from_name'],
						"to_account": row['to_name'],
						"relationship_type": row['relationship_type'],
						"annual_value": float(row['annual_value']),
						"relationship_strength": row['relationship_strength']
					}
					for row in top_valued
				]
				
				# Expiring contracts
				expiring = await conn.fetch(f"""
					SELECT r.*, a_from.account_name as from_name, a_to.account_name as to_name
					FROM crm_account_relationships r
					JOIN crm_accounts a_from ON r.from_account_id = a_from.id
					JOIN crm_accounts a_to ON r.to_account_id = a_to.id
					WHERE {base_conditions} 
					AND r.contract_end_date IS NOT NULL
					AND r.contract_end_date BETWEEN NOW() AND NOW() + INTERVAL '90 days'
					ORDER BY r.contract_end_date ASC
				""", *base_params)
				
				analytics.expiring_contracts = [
					{
						"relationship_id": row['id'],
						"from_account": row['from_name'],
						"to_account": row['to_name'],
						"relationship_type": row['relationship_type'],
						"contract_end_date": row['contract_end_date'].isoformat(),
						"days_until_expiry": (row['contract_end_date'] - datetime.utcnow()).days
					}
					for row in expiring
				]
			
			logger.info(f"✅ Generated analytics for {analytics.total_relationships} relationships")
			return analytics
			
		except Exception as e:
			logger.error(f"Relationship analytics failed: {str(e)}", exc_info=True)
			raise RelationshipError(f"Relationship analytics failed: {str(e)}")
	
	# ================================
	# Helper Methods
	# ================================
	
	async def _validate_relationship(
		self,
		relationship: AccountRelationship,
		tenant_id: str
	):
		"""Validate relationship business rules"""
		# Check for self-relationship
		if relationship.from_account_id == relationship.to_account_id:
			raise RelationshipError("Account cannot have relationship with itself")
		
		# Verify accounts exist
		from_account = await self.db_manager.get_account(relationship.from_account_id, tenant_id)
		to_account = await self.db_manager.get_account(relationship.to_account_id, tenant_id)
		
		if not from_account or not to_account:
			raise RelationshipError("One or both accounts not found")
		
		# Check for duplicate relationships
		existing = await self._get_existing_relationship(
			relationship.from_account_id,
			relationship.to_account_id,
			relationship.relationship_type,
			tenant_id
		)
		
		if existing and existing.id != relationship.id:
			raise RelationshipError("Relationship already exists")
		
		# Check incompatible relationships
		await self._check_incompatible_relationships(relationship, tenant_id)
	
	async def _save_relationship(self, relationship: AccountRelationship) -> AccountRelationship:
		"""Save relationship to database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_account_relationships (
						id, tenant_id, from_account_id, to_account_id,
						relationship_type, relationship_strength, relationship_status, direction,
						annual_value, contract_start_date, contract_end_date, renewal_date,
						key_contact_id, relationship_owner_id, risk_level, compliance_status,
						description, notes, tags, metadata, source, verified_at, verified_by,
						created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
						$17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28
					)
					ON CONFLICT (id) DO UPDATE SET
						relationship_type = EXCLUDED.relationship_type,
						relationship_strength = EXCLUDED.relationship_strength,
						relationship_status = EXCLUDED.relationship_status,
						direction = EXCLUDED.direction,
						annual_value = EXCLUDED.annual_value,
						contract_start_date = EXCLUDED.contract_start_date,
						contract_end_date = EXCLUDED.contract_end_date,
						renewal_date = EXCLUDED.renewal_date,
						key_contact_id = EXCLUDED.key_contact_id,
						relationship_owner_id = EXCLUDED.relationship_owner_id,
						risk_level = EXCLUDED.risk_level,
						compliance_status = EXCLUDED.compliance_status,
						description = EXCLUDED.description,
						notes = EXCLUDED.notes,
						tags = EXCLUDED.tags,
						metadata = EXCLUDED.metadata,
						source = EXCLUDED.source,
						verified_at = EXCLUDED.verified_at,
						verified_by = EXCLUDED.verified_by,
						updated_at = EXCLUDED.updated_at,
						updated_by = EXCLUDED.updated_by,
						version = EXCLUDED.version
				""",
					relationship.id, relationship.tenant_id, relationship.from_account_id,
					relationship.to_account_id, relationship.relationship_type.value,
					relationship.relationship_strength.value, relationship.relationship_status.value,
					relationship.direction.value, relationship.annual_value,
					relationship.contract_start_date, relationship.contract_end_date,
					relationship.renewal_date, relationship.key_contact_id,
					relationship.relationship_owner_id, relationship.risk_level,
					relationship.compliance_status, relationship.description,
					relationship.notes, relationship.tags, relationship.metadata,
					relationship.source, relationship.verified_at, relationship.verified_by,
					relationship.created_at, relationship.updated_at,
					relationship.created_by, relationship.updated_by, relationship.version
				)
			
			return relationship
			
		except Exception as e:
			logger.error(f"Failed to save relationship: {str(e)}", exc_info=True)
			raise RelationshipError(f"Save relationship failed: {str(e)}")
	
	def _row_to_relationship(self, row) -> AccountRelationship:
		"""Convert database row to AccountRelationship object"""
		return AccountRelationship(
			id=row['id'],
			tenant_id=row['tenant_id'],
			from_account_id=row['from_account_id'],
			to_account_id=row['to_account_id'],
			relationship_type=AccountRelationshipType(row['relationship_type']),
			relationship_strength=RelationshipStrength(row['relationship_strength']),
			relationship_status=RelationshipStatus(row['relationship_status']),
			direction=RelationshipDirection(row['direction']),
			annual_value=row['annual_value'],
			contract_start_date=row['contract_start_date'],
			contract_end_date=row['contract_end_date'],
			renewal_date=row['renewal_date'],
			key_contact_id=row['key_contact_id'],
			relationship_owner_id=row['relationship_owner_id'],
			risk_level=row['risk_level'],
			compliance_status=row['compliance_status'],
			description=row['description'],
			notes=row['notes'],
			tags=row['tags'] or [],
			metadata=row['metadata'] or {},
			source=row['source'],
			verified_at=row['verified_at'],
			verified_by=row['verified_by'],
			created_at=row['created_at'],
			updated_at=row['updated_at'],
			created_by=row['created_by'],
			updated_by=row['updated_by'],
			version=row['version']
		)
	
	async def _get_existing_relationship(
		self,
		from_account_id: str,
		to_account_id: str,
		relationship_type: AccountRelationshipType,
		tenant_id: str
	) -> Optional[AccountRelationship]:
		"""Check if relationship already exists"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_account_relationships 
					WHERE tenant_id = $1 
					AND from_account_id = $2 
					AND to_account_id = $3
					AND relationship_type = $4
				""", tenant_id, from_account_id, to_account_id, relationship_type.value)
				
				if row:
					return self._row_to_relationship(row)
				return None
				
		except Exception:
			return None
	
	def _should_create_reverse_relationship(self, relationship: AccountRelationship) -> bool:
		"""Check if reverse relationship should be created"""
		return relationship.relationship_type in self.relationship_rules["auto_create_reverse"]
	
	async def _create_reverse_relationship(self, relationship: AccountRelationship, created_by: str):
		"""Create reverse relationship if applicable"""
		try:
			reverse_type = self.relationship_rules["auto_create_reverse"][relationship.relationship_type]
			
			reverse_relationship = AccountRelationship(
				tenant_id=relationship.tenant_id,
				from_account_id=relationship.to_account_id,
				to_account_id=relationship.from_account_id,
				relationship_type=AccountRelationshipType(reverse_type),
				relationship_strength=relationship.relationship_strength,
				relationship_status=relationship.relationship_status,
				direction=RelationshipDirection.INBOUND,
				annual_value=relationship.annual_value,
				contract_start_date=relationship.contract_start_date,
				contract_end_date=relationship.contract_end_date,
				relationship_owner_id=relationship.relationship_owner_id,
				description=f"Reverse relationship for {relationship.id}",
				source="auto_generated",
				created_by=created_by,
				updated_by=created_by
			)
			
			await self._save_relationship(reverse_relationship)
			
		except Exception as e:
			logger.error(f"Failed to create reverse relationship: {str(e)}", exc_info=True)
	
	async def _check_incompatible_relationships(self, relationship: AccountRelationship, tenant_id: str):
		"""Check for incompatible relationship combinations"""
		for incompatible_pair in self.relationship_rules["incompatible_relationships"]:
			type1, type2 = incompatible_pair
			
			if relationship.relationship_type == type1:
				# Check if type2 relationship exists
				existing = await self._get_existing_relationship(
					relationship.from_account_id,
					relationship.to_account_id,
					type2,
					tenant_id
				)
				if existing:
					raise RelationshipError(
						f"Cannot create {type1.value} relationship: "
						f"Incompatible {type2.value} relationship exists"
					)
	
	async def _calculate_relationship_summary(self, rows) -> Dict[str, Any]:
		"""Calculate relationship summary statistics"""
		summary = {
			"by_type": {},
			"by_strength": {},
			"by_status": {},
			"total_value": 0.0
		}
		
		for row in rows:
			# By type
			rel_type = row['relationship_type']
			summary["by_type"][rel_type] = summary["by_type"].get(rel_type, 0) + 1
			
			# By strength
			strength = row['relationship_strength']
			summary["by_strength"][strength] = summary["by_strength"].get(strength, 0) + 1
			
			# By status
			status = row['relationship_status']
			summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
			
			# Total value
			if row['annual_value']:
				summary["total_value"] += float(row['annual_value'])
		
		return summary
	
	async def _discover_domain_relationships(self, accounts: List, tenant_id: str) -> List[Dict[str, Any]]:
		"""Discover relationships based on email domains"""
		relationships = []
		# Implementation for domain-based relationship discovery
		return relationships
	
	async def _discover_industry_relationships(self, accounts: List, tenant_id: str) -> List[Dict[str, Any]]:
		"""Discover relationships based on industry connections"""
		relationships = []
		# Implementation for industry-based relationship discovery
		return relationships
	
	async def _discover_contact_based_relationships(self, accounts: List, tenant_id: str) -> List[Dict[str, Any]]:
		"""Discover relationships based on shared contacts"""
		relationships = []
		# Implementation for contact-based relationship discovery
		return relationships
	
	async def _update_relationship_analytics(self, tenant_id: str):
		"""Update cached relationship analytics"""
		try:
			# Implementation for updating analytics cache
			pass
		except Exception as e:
			logger.error(f"Failed to update relationship analytics: {str(e)}", exc_info=True)